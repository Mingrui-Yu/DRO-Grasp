"""Contract tests for the three-scene DRO Viser comparison layer."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from grasp_generation.experiments.bimanbodex_dro.contracts import STAGE_NAMES
from grasp_generation.experiments.bimanbodex_dro.visualizer import (
    MeshData,
    PreparedSelection,
)
from grasp_generation.scripts.visualize_bimanbodex_dro_comparison import (
    ComparisonViewerApp,
    comparison_offsets,
    comparison_summary,
    make_table_patch,
    prepare_comparison,
    translate_pose,
    translate_vertices,
)


SCENE_IDS = (
    "small/tabletop_ur10e/scale002_pose000_0",
    "medium/tabletop_ur10e/scale011_pose000_0",
    "large/tabletop_ur10e/scale030_pose000_0",
)
SCALES = (0.02, 0.106, 0.30)


class _FakeHandle:
    def __init__(self, value=None):
        self.value = value
        self.content = value if isinstance(value, str) else ""
        self.callback = None
        self.removed = False

    def on_update(self, callback):
        self.callback = callback

    def remove(self):
        self.removed = True


class _FakeGui:
    def add_slider(self, _label, *, initial_value, **_kwargs):
        return _FakeHandle(initial_value)

    def add_checkbox(self, _label, *, initial_value):
        return _FakeHandle(initial_value)

    def add_dropdown(self, _label, *, initial_value, **_kwargs):
        return _FakeHandle(initial_value)

    def add_markdown(self, content):
        return _FakeHandle(content)


class _FakeScene:
    def __init__(self):
        self.calls = []

    def _add(self, kind, name, **kwargs):
        self.calls.append((kind, name, kwargs))
        return _FakeHandle()

    def add_mesh_simple(self, name, _vertices, _faces, **kwargs):
        return self._add("mesh", name, **kwargs)

    def add_point_cloud(self, name, _points, **kwargs):
        return self._add("point_cloud", name, **kwargs)

    def add_frame(self, name, **kwargs):
        return self._add("frame", name, **kwargs)

    def add_label(self, name, **kwargs):
        return self._add("label", name, **kwargs)


class _FakeServer:
    def __init__(self):
        self.gui = _FakeGui()
        self.scene = _FakeScene()


class _FakeRun:
    candidate_count = 20

    def __init__(self):
        self.calls = []

    def load_scene(self, scene_id):
        if scene_id not in SCENE_IDS:
            raise ValueError(f"unknown scene_id: {scene_id}")
        index = SCENE_IDS.index(scene_id)
        record = SimpleNamespace(
            scale=SCALES[index],
            table_origin_world=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            table_normal_world=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )
        return SimpleNamespace(record=record)

    def prepare_selection(
        self,
        _hand_model,
        scene_id,
        *,
        candidate_index,
        stage,
        mode,
        pose_source,
    ):
        self.calls.append(
            (scene_id, candidate_index, stage, mode, pose_source)
        )
        triangle = MeshData(
            vertices=np.array(
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.1]],
                dtype=np.float32,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        object_pose = np.array(
            [0.01, -0.02, 0.03, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        palm_pose = np.array(
            [0.02, 0.01, 0.08, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        return PreparedSelection(
            scene_id=scene_id,
            candidate_index=candidate_index,
            selected_stage="grasp",
            stage_names=STAGE_NAMES,
            pose_source=pose_source,
            object_mesh=triangle,
            object_point_cloud_world=np.zeros((512, 3), dtype=np.float32),
            hand_meshes={stage_name: triangle for stage_name in STAGE_NAMES},
            dro_component_meshes={},
            bench_hand_meshes={},
            common_link_frames={},
            object_pose_wxyz=object_pose,
            palm_poses_wxyz={
                stage_name: palm_pose.copy() for stage_name in STAGE_NAMES
            },
            diagnostics={
                "pose_source_label": "Bench artifact / export_stage_q"
            },
        )


class ComparisonVisualizerTests(unittest.TestCase):
    def test_offsets_and_display_transforms_preserve_metric_payloads(self):
        offsets = comparison_offsets(3, 0.9)
        np.testing.assert_allclose(
            np.stack(offsets),
            [[-0.9, 0.0, 0.0], [0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        )
        vertices = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        shifted = translate_vertices(vertices, offsets[0])
        np.testing.assert_allclose(shifted, [[0.1, 2.0, 3.0]], atol=1e-7)
        np.testing.assert_array_equal(vertices, [[1.0, 2.0, 3.0]])
        pose = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
        shifted_pose = translate_pose(pose, offsets[2])
        np.testing.assert_allclose(shifted_pose[:3], [1.9, 2.0, 3.0])
        np.testing.assert_array_equal(pose, [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            comparison_offsets(3, 0.0)

    def test_table_patch_uses_authoritative_plane_and_requested_metric_size(self):
        patch = make_table_patch(
            np.array([0.0, 0.0, 0.2]),
            np.array([0.0, 0.0, 1.0]),
            0.7,
        )
        self.assertEqual(patch.vertices.shape, (4, 3))
        self.assertEqual(patch.faces.shape, (2, 3))
        np.testing.assert_allclose(patch.vertices[:, 2], 0.2)
        extents = patch.vertices.max(axis=0) - patch.vertices.min(axis=0)
        np.testing.assert_allclose(sorted(extents[:2].tolist()), [0.7, 0.7])

    def test_prepare_comparison_keeps_artifacts_unshifted_and_candidates_independent(self):
        run = _FakeRun()
        comparison = prepare_comparison(
            run,
            hand_model=object(),
            scene_ids=SCENE_IDS,
            candidate_indices=(1, 7, 19),
            pose_source="exported",
            spacing=0.9,
            table_size=0.7,
        )
        self.assertEqual(len(comparison.panels), 3)
        self.assertEqual(
            [panel.candidate_index for panel in comparison.panels], [1, 7, 19]
        )
        self.assertEqual(
            [panel.actual_scale for panel in comparison.panels], list(SCALES)
        )
        np.testing.assert_allclose(
            comparison.panels[0].selection.object_mesh.vertices[0],
            [0.0, 0.0, 0.0],
        )
        summary = comparison_summary(comparison, ("grasp",))
        self.assertEqual(summary["panel_count"], 3)
        self.assertEqual(summary["visible_stages"], ["grasp"])
        self.assertEqual(
            [item["visualization_only_offset_m"][0] for item in summary["panels"]],
            [-0.9, 0.0, 0.9],
        )
        self.assertTrue(
            all(call[3] == "three_poses" for call in run.calls)
        )

    def test_prepare_comparison_rejects_non_triptych_and_duplicate_scenes(self):
        run = _FakeRun()
        with self.assertRaisesRegex(ValueError, "exactly three"):
            prepare_comparison(run, object(), SCENE_IDS[:2])
        with self.assertRaisesRegex(ValueError, "must be unique"):
            prepare_comparison(
                run,
                object(),
                (SCENE_IDS[0], SCENE_IDS[0], SCENE_IDS[2]),
            )
        with self.assertRaisesRegex(ValueError, "once or exactly once"):
            prepare_comparison(
                run,
                object(),
                SCENE_IDS,
                candidate_indices=(0, 1),
            )

    def test_viser_ui_creates_three_independent_candidate_controls(self):
        run = _FakeRun()
        server = _FakeServer()
        app = ComparisonViewerApp(
            server,
            run,
            hand_model=object(),
            args=SimpleNamespace(
                scene=list(SCENE_IDS),
                candidate=None,
                stage="grasp",
                mode="single_stage",
                pose_source="exported",
                spacing=0.9,
                table_size=0.7,
                show_point_cloud=False,
            ),
        )
        self.assertEqual(len(app.candidates), 3)
        self.assertEqual([item.value for item in app.candidates], [0, 0, 0])
        app.candidates[1].value = 5
        app.render()
        self.assertIn("actual scale", app.diagnostics.content)
        self.assertGreaterEqual(len(app._handles), 18)
        self.assertEqual(
            len([call for call in server.scene.calls if call[0] == "label"]),
            6,
        )


if __name__ == "__main__":
    unittest.main()
