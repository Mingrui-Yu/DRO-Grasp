#!/usr/bin/env python3
"""Compare three persisted DRO DGN2k grasps side by side in one Viser scene."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


GRASP_GENERATION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = GRASP_GENERATION_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_generation.experiments.bimanbodex_dro.contracts import STAGE_NAMES
from grasp_generation.experiments.bimanbodex_dro.visualizer import (
    MeshData,
    ShadowHandModel,
    ViewerRun,
)


PANEL_NAMES = ("very small", "medium", "very large")
STAGE_COLORS = {
    "pregrasp": (255, 158, 74),
    "grasp": (64, 174, 255),
    "squeeze": (238, 93, 152),
}


@dataclass(frozen=True)
class ComparisonPanel:
    """One validated scene selection plus its visualization-only placement."""

    panel_name: str
    scene_id: str
    scale_bucket: str
    actual_scale: float
    candidate_index: int
    display_offset: np.ndarray
    table_origin_world: np.ndarray
    table_normal_world: np.ndarray
    table_mesh: MeshData
    selection: object


@dataclass(frozen=True)
class PreparedComparison:
    """Three panels prepared without mutating persisted artifacts."""

    panels: tuple
    spacing: float
    table_size: float
    pose_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render exactly three completed scenes from one DRO output root in a "
            "metric-scale Viser comparison row. Display offsets are visualization-only."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--scene-root",
        type=Path,
        help="Current-machine DGN2k scene_cfg root; defaults to resolved_config.json.",
    )
    parser.add_argument(
        "--shadow-urdf",
        type=Path,
        help="Released extended Shadow URDF; defaults to resolved_config.json.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        required=True,
        help="Completed scene ID. Pass exactly three times in left-to-right order.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        type=int,
        help="Initial candidate. Pass once for all panels or three times; default is 0.",
    )
    parser.add_argument("--stage", choices=STAGE_NAMES, default="grasp")
    parser.add_argument(
        "--mode",
        choices=("single_stage", "three_poses"),
        default="single_stage",
        help="Initial stage visibility. Web controls remain independently toggleable.",
    )
    parser.add_argument(
        "--pose-source",
        choices=("exported", "raw"),
        default="exported",
        help="Raw is diagnostic controller state, not the Bench-facing artifact.",
    )
    parser.add_argument("--spacing", type=float, default=0.9)
    parser.add_argument("--table-size", type=float, default=0.7)
    parser.add_argument("--show-point-cloud", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate and CPU-prepare all three panels, print JSON, then exit.",
    )
    return parser.parse_args()


def comparison_offsets(count: int, spacing: float) -> tuple:
    """Return centered metric translations for a horizontal comparison row."""

    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("comparison count must be a positive integer")
    spacing = float(spacing)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("comparison spacing must be finite and positive")
    center = 0.5 * (count - 1)
    return tuple(
        np.array([(index - center) * spacing, 0.0, 0.0], dtype=np.float64)
        for index in range(count)
    )


def translate_vertices(vertices: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Apply a display-only translation to world-frame vertices."""

    vertices = np.asarray(vertices)
    offset = np.asarray(offset, dtype=np.float64).reshape(-1)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise ValueError("vertices must have shape [N,3]")
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("display offset must be a finite 3-vector")
    return (vertices + offset.astype(vertices.dtype, copy=False)).copy()


def translate_pose(pose_wxyz: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Translate only the xyz part of a world pose for display."""

    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(-1).copy()
    offset = np.asarray(offset, dtype=np.float64).reshape(-1)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("pose must be finite [xyz,wxyz]")
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("display offset must be a finite 3-vector")
    pose[:3] += offset
    return pose


def make_table_patch(origin: np.ndarray, normal: np.ndarray, size: float) -> MeshData:
    """Create a finite visualization patch for an authoritative infinite plane."""

    origin = np.asarray(origin, dtype=np.float64).reshape(-1)
    normal = np.asarray(normal, dtype=np.float64).reshape(-1)
    size = float(size)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("table origin must be a finite 3-vector")
    if normal.shape != (3,) or not np.isfinite(normal).all():
        raise ValueError("table normal must be a finite 3-vector")
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 0.0:
        raise ValueError("table normal must be non-zero")
    if not math.isfinite(size) or size <= 0.0:
        raise ValueError("table patch size must be finite and positive")
    normal = normal / normal_norm
    reference = (
        np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(normal[0])) < 0.9
        else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    tangent_u = np.cross(normal, reference)
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(normal, tangent_u)
    half = 0.5 * size
    vertices = np.stack(
        (
            origin - half * tangent_u - half * tangent_v,
            origin + half * tangent_u - half * tangent_v,
            origin + half * tangent_u + half * tangent_v,
            origin - half * tangent_u + half * tangent_v,
        )
    ).astype(np.float32)
    return MeshData(
        vertices=vertices,
        faces=np.array(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
        source_count=1,
    )


def _normalize_candidates(values, panel_count: int) -> tuple:
    if values is None:
        return tuple(0 for _ in range(panel_count))
    values = tuple(int(value) for value in values)
    if len(values) == 1:
        return values * panel_count
    if len(values) != panel_count:
        raise ValueError("pass --candidate once or exactly once per comparison scene")
    return values


def _scale_bucket(scene_id: str) -> str:
    name = Path(scene_id).name
    bucket = name.split("_pose", 1)[0]
    return bucket if bucket.startswith("scale") else "unknown"


def prepare_comparison(
    run: ViewerRun,
    hand_model: ShadowHandModel,
    scene_ids,
    *,
    candidate_indices=None,
    pose_source: str = "exported",
    spacing: float = 0.9,
    table_size: float = 0.7,
) -> PreparedComparison:
    """Strictly prepare three independent selections and display offsets."""

    scene_ids = tuple(scene_ids)
    if len(scene_ids) != 3:
        raise ValueError("comparison viewer requires exactly three scene IDs")
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("comparison scene IDs must be unique")
    candidates = _normalize_candidates(candidate_indices, len(scene_ids))
    offsets = comparison_offsets(len(scene_ids), spacing)
    panels = []
    for index, (scene_id, candidate_index, offset) in enumerate(
        zip(scene_ids, candidates, offsets)
    ):
        loaded = run.load_scene(scene_id)
        selection = run.prepare_selection(
            hand_model,
            scene_id,
            candidate_index=candidate_index,
            stage="grasp",
            mode="three_poses",
            pose_source=pose_source,
        )
        record = loaded.record
        panels.append(
            ComparisonPanel(
                panel_name=PANEL_NAMES[index],
                scene_id=scene_id,
                scale_bucket=_scale_bucket(scene_id),
                actual_scale=float(record.scale),
                candidate_index=int(candidate_index),
                display_offset=offset.copy(),
                table_origin_world=np.asarray(record.table_origin_world).copy(),
                table_normal_world=np.asarray(record.table_normal_world).copy(),
                table_mesh=make_table_patch(
                    record.table_origin_world,
                    record.table_normal_world,
                    table_size,
                ),
                selection=selection,
            )
        )
    return PreparedComparison(
        panels=tuple(panels),
        spacing=float(spacing),
        table_size=float(table_size),
        pose_source=pose_source,
    )


def comparison_summary(comparison: PreparedComparison, visible_stages) -> dict:
    """Return JSON-compatible read-only preparation evidence."""

    return {
        "status": "prepared_three_scene_comparison",
        "panel_count": len(comparison.panels),
        "spacing_m": comparison.spacing,
        "table_patch_size_m": comparison.table_size,
        "pose_source": comparison.pose_source,
        "visible_stages": list(visible_stages),
        "display_transform_contract": (
            "rendered_position = persisted_world_position + visualization_only_offset"
        ),
        "panels": [
            {
                "panel_name": panel.panel_name,
                "scene_id": panel.scene_id,
                "scale_bucket": panel.scale_bucket,
                "actual_scale": panel.actual_scale,
                "candidate_index": panel.candidate_index,
                "visualization_only_offset_m": panel.display_offset.tolist(),
                "persisted_object_pose_wxyz": (
                    panel.selection.object_pose_wxyz.tolist()
                ),
                "table_origin_world_m": panel.table_origin_world.tolist(),
                "table_normal_world": panel.table_normal_world.tolist(),
                "available_stages": list(panel.selection.stage_names),
                "artifact_pose_source": panel.selection.diagnostics[
                    "pose_source_label"
                ],
            }
            for panel in comparison.panels
        ],
    }


def comparison_markdown(comparison: PreparedComparison, visible_stages) -> str:
    lines = [
        "### Three-scale DRO comparison",
        f"- Pose source: `{comparison.pose_source}`",
        f"- Visible stages: `{', '.join(visible_stages) or 'none'}`",
        f"- Panel spacing: `{comparison.spacing:.3f} m`",
        "- Display offsets are visualization-only and are never written to artifacts.",
        "",
        "| panel | scale bucket | actual scale | candidate | display x |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for panel in comparison.panels:
        lines.append(
            f"| {panel.panel_name} | {panel.scale_bucket} | "
            f"{panel.actual_scale:.6g} | {panel.candidate_index} | "
            f"{panel.display_offset[0]:.3f} m |"
        )
    lines.extend(
        (
            "",
            "The table patch visualizes the authoritative infinite plane; its finite "
            "extent is not a collision boundary.",
        )
    )
    return "\n".join(lines)


class ComparisonViewerApp:
    """Interactive Viser UI for three independent persisted candidates."""

    def __init__(self, server, run, hand_model, args):
        self.server = server
        self.run = run
        self.hand_model = hand_model
        self.args = args
        self.scene_ids = tuple(args.scene)
        initial_candidates = _normalize_candidates(args.candidate, len(self.scene_ids))
        self.candidates = tuple(
            server.gui.add_slider(
                f"{PANEL_NAMES[index]} candidate",
                min=0,
                max=run.candidate_count - 1,
                step=1,
                initial_value=value,
            )
            for index, value in enumerate(initial_candidates)
        )
        initially_visible = (
            set(STAGE_NAMES) if args.mode == "three_poses" else {args.stage}
        )
        self.stage_controls = {
            stage: server.gui.add_checkbox(
                f"Show {stage}", initial_value=stage in initially_visible
            )
            for stage in STAGE_NAMES
        }
        self.pose_source = server.gui.add_dropdown(
            "Pose source", options=("exported", "raw"), initial_value=args.pose_source
        )
        self.table = server.gui.add_checkbox("Table plane", initial_value=True)
        self.point_cloud = server.gui.add_checkbox(
            "Input object point cloud", initial_value=args.show_point_cloud
        )
        self.world_axes = server.gui.add_checkbox("World axes", initial_value=False)
        self.object_axes = server.gui.add_checkbox("Object axes", initial_value=False)
        self.palm_axes = server.gui.add_checkbox("Palm axes", initial_value=False)
        self.diagnostics = server.gui.add_markdown("Preparing comparison...")
        self._handles = []
        self._rendering = False

        for control in (
            *self.candidates,
            *self.stage_controls.values(),
            self.pose_source,
            self.table,
            self.point_cloud,
            self.world_axes,
            self.object_axes,
            self.palm_axes,
        ):
            control.on_update(lambda _: self.render())
        self.render()

    def _clear(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def render(self):
        if self._rendering:
            return
        self._rendering = True
        try:
            comparison = prepare_comparison(
                self.run,
                self.hand_model,
                self.scene_ids,
                candidate_indices=tuple(int(item.value) for item in self.candidates),
                pose_source=self.pose_source.value,
                spacing=self.args.spacing,
                table_size=self.args.table_size,
            )
            visible_stages = tuple(
                stage
                for stage in STAGE_NAMES
                if bool(self.stage_controls[stage].value)
            )
            self._clear()
            for index, panel in enumerate(comparison.panels):
                prefix = f"/comparison/{index:02d}"
                offset = panel.display_offset
                selection = panel.selection
                if bool(self.table.value):
                    self._handles.append(
                        self.server.scene.add_mesh_simple(
                            f"{prefix}/table",
                            translate_vertices(panel.table_mesh.vertices, offset),
                            panel.table_mesh.faces,
                            color=(105, 112, 122),
                            opacity=0.28,
                            side="double",
                        )
                    )
                self._handles.append(
                    self.server.scene.add_mesh_simple(
                        f"{prefix}/object",
                        translate_vertices(selection.object_mesh.vertices, offset),
                        selection.object_mesh.faces,
                        color=(190, 190, 196),
                        opacity=0.78,
                        side="double",
                    )
                )
                self._handles.append(
                    self.server.scene.add_point_cloud(
                        f"{prefix}/object_points",
                        translate_vertices(selection.object_point_cloud_world, offset),
                        colors=(250, 80, 80),
                        point_size=0.0015,
                        point_shape="circle",
                        visible=bool(self.point_cloud.value),
                    )
                )
                max_z = float(selection.object_mesh.vertices[:, 2].max())
                for stage_name in visible_stages:
                    mesh = selection.hand_meshes[stage_name]
                    max_z = max(max_z, float(mesh.vertices[:, 2].max()))
                    self._handles.append(
                        self.server.scene.add_mesh_simple(
                            f"{prefix}/shadow/{stage_name}",
                            translate_vertices(mesh.vertices, offset),
                            mesh.faces,
                            color=STAGE_COLORS[stage_name],
                            opacity=0.88 if len(visible_stages) == 1 else 0.58,
                            side="double",
                        )
                    )
                    palm_pose = translate_pose(
                        selection.palm_poses_wxyz[stage_name], offset
                    )
                    self._handles.append(
                        self.server.scene.add_frame(
                            f"{prefix}/axes/palm/{stage_name}",
                            wxyz=palm_pose[3:],
                            position=palm_pose[:3],
                            axes_length=0.045,
                            axes_radius=0.001,
                            visible=bool(self.palm_axes.value),
                        )
                    )
                object_pose = translate_pose(selection.object_pose_wxyz, offset)
                self._handles.append(
                    self.server.scene.add_frame(
                        f"{prefix}/axes/object",
                        wxyz=object_pose[3:],
                        position=object_pose[:3],
                        axes_length=0.08,
                        axes_radius=0.0015,
                        visible=bool(self.object_axes.value),
                    )
                )
                self._handles.append(
                    self.server.scene.add_frame(
                        f"{prefix}/axes/world",
                        wxyz=(1.0, 0.0, 0.0, 0.0),
                        position=panel.table_origin_world + offset,
                        axes_length=0.12,
                        axes_radius=0.002,
                        visible=bool(self.world_axes.value),
                    )
                )
                label_position = panel.table_origin_world + offset
                label_position = label_position.copy()
                label_position[2] = max_z + 0.10
                self._handles.append(
                    self.server.scene.add_label(
                        f"{prefix}/label",
                        text=(
                            f"{panel.panel_name}\n{panel.scale_bucket} / "
                            f"actual {panel.actual_scale:.6g}"
                        ),
                        position=label_position,
                    )
                )
            self.diagnostics.content = comparison_markdown(
                comparison, visible_stages
            )
        except Exception as error:
            self._clear()
            self.diagnostics.content = (
                "### Comparison error\n\n"
                f"`{type(error).__name__}: {error}`\n\n"
                "No fallback geometry was rendered."
            )
        finally:
            self._rendering = False


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError(f"port must be in [1,65535], got {args.port}")
    candidates = _normalize_candidates(args.candidate, len(args.scene))
    run = ViewerRun(
        args.output_root,
        scene_root=args.scene_root,
        shadow_urdf=args.shadow_urdf,
    )
    hand_model = ShadowHandModel(run.shadow_urdf)
    comparison = prepare_comparison(
        run,
        hand_model,
        args.scene,
        candidate_indices=candidates,
        pose_source=args.pose_source,
        spacing=args.spacing,
        table_size=args.table_size,
    )
    visible_stages = STAGE_NAMES if args.mode == "three_poses" else (args.stage,)
    if args.prepare_only:
        print(
            json.dumps(
                comparison_summary(comparison, visible_stages),
                indent=2,
                sort_keys=True,
            )
        )
        return

    try:
        import viser
    except ImportError as error:
        raise RuntimeError(
            "viser is required to launch the Web UI; use the isolated dro environment"
        ) from error
    server = viser.ViserServer(host=args.host, port=args.port)

    @server.on_client_connect
    def _initialize_camera(client):
        client.camera.position = np.array([0.0, -2.3, 1.15], dtype=np.float64)
        client.camera.look_at = np.array([0.0, 0.0, 0.18], dtype=np.float64)

    ComparisonViewerApp(server, run, hand_model, args)
    actual_host = server.get_host()
    actual_port = server.get_port()
    print(
        "Read-only DRO three-scale comparison viewer started: "
        f"url=http://{actual_host}:{actual_port}",
        flush=True,
    )
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
