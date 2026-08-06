#!/usr/bin/env python3
"""Launch the read-only Viser viewer for exported BimanBODex DRO poses."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


GRASP_GENERATION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = GRASP_GENERATION_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_generation.experiments.bimanbodex_dro.contracts import STAGE_NAMES
from grasp_generation.experiments.bimanbodex_dro.visualizer import (
    BenchShadowHandModel,
    ShadowHandModel,
    ViewerRun,
    diagnostics_markdown,
)


STAGE_COLORS = {
    "pregrasp": (255, 158, 74),
    "grasp": (64, 174, 255),
    "squeeze": (238, 93, 152),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one #24 DRO output root and visualize the persisted exported "
            "pregrasp/grasp/squeeze poses without inference or file writes."
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
        "--bench-mjcf",
        type=Path,
        help="Bench palm-root Shadow MJCF; required for dro_bench_overlay mode.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--scene", help="Initial completed scene ID.")
    parser.add_argument("--scale", type=float, help="Initial isotropic object scale filter.")
    parser.add_argument("--candidate", type=int, default=0)
    parser.add_argument("--stage", choices=STAGE_NAMES, default="grasp")
    parser.add_argument(
        "--mode",
        choices=("single_stage", "three_poses", "dro_bench_overlay"),
        default="three_poses",
    )
    parser.add_argument(
        "--pose-source",
        choices=("exported", "raw"),
        default="exported",
        help="Raw is diagnostic controller state, not the Bench artifact.",
    )
    parser.add_argument(
        "--show-point-cloud",
        action="store_true",
        help="Show the persisted 512-point object-local input transformed to world.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate and CPU-prepare the initial selection, print JSON, then exit.",
    )
    return parser.parse_args()


def _initial_scene(run: ViewerRun, scene_id, scale):
    choices = run.scene_ids_for_scale(scale)
    if not choices:
        failures = []
        for entry in run.failed_scenes[:3]:
            failure = entry.failure or {}
            candidate_failures = failure.get("candidate_failures")
            message = failure.get("message")
            if not message and isinstance(candidate_failures, list) and candidate_failures:
                first = candidate_failures[0]
                message = first.get("message") if isinstance(first, dict) else None
            failures.append(
                f"{entry.scene_id}: {failure.get('stage', 'unknown')}"
                + (f" ({message})" if message else "")
            )
        raise ValueError(
            "output root has no completed scenes"
            + (f" at scale {scale:.9g}" if scale is not None else "")
            + (f"; failed scenes: {'; '.join(failures)}" if failures else "")
        )
    if scene_id is None:
        return choices[0]
    if scene_id not in choices:
        raise ValueError(
            f"initial scene {scene_id!r} is not a completed scene in the selected scale filter"
        )
    return scene_id


class ViewerApp:
    """Small Viser UI over the strict read-only preparation layer."""

    def __init__(
        self, server, run, hand_model, args, initial_scene, bench_model=None
    ):
        self.server = server
        self.run = run
        self.hand_model = hand_model
        self.bench_model = bench_model
        self._scene_handles = []
        self._updating = False

        scale_options = ["all"] + [format(value, ".9g") for value in run.scales]
        initial_scale = "all" if args.scale is None else format(args.scale, ".9g")
        self.scale = server.gui.add_dropdown(
            "Object scale", options=scale_options, initial_value=initial_scale
        )
        self.scene = server.gui.add_dropdown(
            "Scene ID",
            options=list(run.scene_ids_for_scale(args.scale)),
            initial_value=initial_scene,
        )
        self.candidate = server.gui.add_slider(
            "Candidate",
            min=0,
            max=run.candidate_count - 1,
            step=1,
            initial_value=args.candidate,
        )
        self.stage = server.gui.add_dropdown(
            "Exported stage", options=list(STAGE_NAMES), initial_value=args.stage
        )
        mode_options = ["single_stage", "three_poses"]
        if bench_model is not None:
            mode_options.append("dro_bench_overlay")
        if args.mode not in mode_options:
            raise ValueError("dro_bench_overlay mode requires --bench-mjcf")
        self.mode = server.gui.add_dropdown(
            "Display mode",
            options=mode_options,
            initial_value=args.mode,
        )
        self.pose_source = server.gui.add_dropdown(
            "Pose source",
            options=["exported", "raw"],
            initial_value=args.pose_source,
        )
        self.world_axes = server.gui.add_checkbox("World axes", initial_value=True)
        self.object_axes = server.gui.add_checkbox("Object axes", initial_value=True)
        self.palm_axes = server.gui.add_checkbox("Palm axes", initial_value=True)
        self.common_link_axes = server.gui.add_checkbox(
            "Common-link axes", initial_value=False
        )
        self.dro_palm_fingers = server.gui.add_checkbox(
            "DRO palm + fingers", initial_value=True
        )
        self.dro_forearm = server.gui.add_checkbox(
            "DRO forearm", initial_value=True
        )
        self.dro_wrist = server.gui.add_checkbox("DRO wrist", initial_value=True)
        self.bench_hand = server.gui.add_checkbox(
            "Bench palm + fingers", initial_value=True
        )
        self.point_cloud = server.gui.add_checkbox(
            "Input object point cloud", initial_value=args.show_point_cloud
        )
        self.diagnostics = server.gui.add_markdown("Loading selection...")

        self.scale.on_update(lambda _: self._on_scale())
        for handle in (
            self.scene,
            self.candidate,
            self.stage,
            self.mode,
            self.pose_source,
            self.world_axes,
            self.object_axes,
            self.palm_axes,
            self.common_link_axes,
            self.dro_palm_fingers,
            self.dro_forearm,
            self.dro_wrist,
            self.bench_hand,
            self.point_cloud,
        ):
            handle.on_update(lambda _: self.render())
        self.render()

    def _clear_scene(self):
        for handle in self._scene_handles:
            handle.remove()
        self._scene_handles = []

    def _on_scale(self):
        if self._updating:
            return
        scale = None if self.scale.value == "all" else float(self.scale.value)
        choices = list(self.run.scene_ids_for_scale(scale))
        if not choices:
            self.diagnostics.content = f"No completed scenes at scale `{self.scale.value}`."
            return
        self._updating = True
        try:
            self.scene.options = choices
            if self.scene.value not in choices:
                self.scene.value = choices[0]
        finally:
            self._updating = False
        self.render()

    def render(self):
        if self._updating:
            return
        try:
            prepared = self.run.prepare_selection(
                self.hand_model,
                self.scene.value,
                bench_model=self.bench_model,
                candidate_index=int(self.candidate.value),
                stage=self.stage.value,
                mode=self.mode.value,
                pose_source=self.pose_source.value,
            )
        except Exception as error:
            self.diagnostics.content = (
                "### Selection error\n"
                f"`{type(error).__name__}: {error}`\n\n"
                "No fallback pose or asset was rendered."
            )
            self._clear_scene()
            return

        self._clear_scene()
        self._scene_handles.append(
            self.server.scene.add_mesh_simple(
                "/dgn2k/object",
                prepared.object_mesh.vertices,
                prepared.object_mesh.faces,
                color=(190, 190, 196),
                opacity=0.72,
            )
        )
        self._scene_handles.append(
            self.server.scene.add_point_cloud(
                "/dgn2k/object_points",
                prepared.object_point_cloud_world,
                colors=(250, 80, 80),
                point_size=0.0015,
                point_shape="circle",
                visible=bool(self.point_cloud.value),
            )
        )
        self._scene_handles.append(
            self.server.scene.add_frame(
                "/axes/world",
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
                axes_length=0.12,
                axes_radius=0.002,
                visible=bool(self.world_axes.value),
            )
        )
        self._scene_handles.append(
            self.server.scene.add_frame(
                "/axes/object",
                wxyz=prepared.object_pose_wxyz[3:],
                position=prepared.object_pose_wxyz[:3],
                axes_length=0.08,
                axes_radius=0.0015,
                visible=bool(self.object_axes.value),
            )
        )
        is_dro_bench_overlay = bool(prepared.bench_hand_meshes)
        overlay = len(prepared.stage_names) == len(STAGE_NAMES)
        for stage_name in prepared.stage_names:
            if is_dro_bench_overlay:
                component_settings = (
                    ("palm_fingers", self.dro_palm_fingers, (52, 152, 255), 0.68),
                    ("forearm", self.dro_forearm, (46, 102, 190), 0.48),
                    ("wrist", self.dro_wrist, (45, 126, 210), 0.56),
                )
                for component, visibility, color, opacity in component_settings:
                    if not bool(visibility.value):
                        continue
                    mesh = prepared.dro_component_meshes[component][stage_name]
                    self._scene_handles.append(
                        self.server.scene.add_mesh_simple(
                            f"/overlay/dro/{component}/{stage_name}",
                            mesh.vertices,
                            mesh.faces,
                            color=color,
                            opacity=opacity,
                        )
                    )
                if bool(self.bench_hand.value):
                    mesh = prepared.bench_hand_meshes[stage_name]
                    self._scene_handles.append(
                        self.server.scene.add_mesh_simple(
                            f"/overlay/bench/{stage_name}",
                            mesh.vertices,
                            mesh.faces,
                            color=(244, 93, 178),
                            opacity=0.52,
                        )
                    )
                if bool(self.common_link_axes.value):
                    for link_name, sources in prepared.common_link_frames[
                        stage_name
                    ].items():
                        for source_name, pose in sources.items():
                            self._scene_handles.append(
                                self.server.scene.add_frame(
                                    f"/axes/common/{stage_name}/{link_name}/{source_name}",
                                    wxyz=pose[3:],
                                    position=pose[:3],
                                    axes_length=0.012,
                                    axes_radius=0.00035,
                                )
                            )
            else:
                mesh = prepared.hand_meshes[stage_name]
                opacity = (
                    0.48
                    if overlay and stage_name != prepared.selected_stage
                    else 0.88
                )
                self._scene_handles.append(
                    self.server.scene.add_mesh_simple(
                        f"/shadow/{stage_name}",
                        mesh.vertices,
                        mesh.faces,
                        color=STAGE_COLORS[stage_name],
                        opacity=opacity,
                    )
                )
            palm = prepared.palm_poses_wxyz[stage_name]
            self._scene_handles.append(
                self.server.scene.add_frame(
                    f"/axes/palm/{stage_name}",
                    wxyz=palm[3:],
                    position=palm[:3],
                    axes_length=0.045,
                    axes_radius=0.001,
                    visible=bool(self.palm_axes.value),
                )
            )
        self.diagnostics.content = diagnostics_markdown(prepared, self.run)


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError(f"port must be in [1,65535], got {args.port}")
    run = ViewerRun(
        args.output_root,
        scene_root=args.scene_root,
        shadow_urdf=args.shadow_urdf,
    )
    scene_id = _initial_scene(run, args.scene, args.scale)
    hand_model = ShadowHandModel(run.shadow_urdf)
    bench_model = (
        BenchShadowHandModel(args.bench_mjcf)
        if args.bench_mjcf is not None
        else None
    )
    prepared = run.prepare_selection(
        hand_model,
        scene_id,
        bench_model=bench_model,
        candidate_index=args.candidate,
        stage=args.stage,
        mode=args.mode,
        pose_source=args.pose_source,
    )
    if args.prepare_only:
        print(json.dumps(prepared.diagnostics, indent=2, sort_keys=True))
        return

    try:
        import viser
    except ImportError as error:
        raise RuntimeError(
            "viser is required to launch the Web UI; use the isolated dro environment "
            "from environment_dgn2k.yml"
        ) from error
    server = viser.ViserServer(host=args.host, port=args.port)
    ViewerApp(server, run, hand_model, args, scene_id, bench_model=bench_model)
    actual_host = server.get_host()
    actual_port = server.get_port()
    print(
        "Read-only DRO three-pose viewer started: "
        f"scene={scene_id} candidate={args.candidate} stage={args.stage} "
        f"mode={args.mode} url=http://{actual_host}:{actual_port}"
    )
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
