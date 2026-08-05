"""CPU-only tests for the DRO/DGN2k/Bench boundary."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from grasp_generation.experiments.bimanbodex_dro.contracts import (
    BENCH_SHADOW_JOINT_NAMES,
    DRO_SHADOW_Q_NAMES,
    STAGE_NAMES,
    dro_q_to_bench_pose,
    dro_q_to_object_palm_transform,
    load_scene_record,
    make_bench_artifact,
    map_dro_shadow_fingers,
    sample_scaled_surface,
    sha256_array,
    validate_artifact,
)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.scene_root = self.root / "scene_cfg"
        self.scene_path = self.scene_root / "object_a" / "floating" / "scale013.npy"
        self.mesh_path = self.root / "processed_data" / "object_a" / "mesh" / "simplified.obj"
        self.mesh_path.parent.mkdir(parents=True)
        self.mesh_path.write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
            "f 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n",
            encoding="utf-8",
        )
        self.scene_path.parent.mkdir(parents=True)
        config = {
            "scene_id": "object_a/floating/scale013",
            "task": {"type": "force_closure", "obj_name": "object_a"},
            "scene": {
                "object_a": {
                    "type": "rigid_object",
                    "file_path": "../../../processed_data/object_a/mesh/simplified.obj",
                    "scale": np.array([0.133, 0.133, 0.133]),
                    "pose": np.array([0.2, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0]),
                }
            },
        }
        np.save(self.scene_path, config, allow_pickle=True)
        self.record = load_scene_record(self.scene_path, self.scene_root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_scene_loader_preserves_exact_asset_scale_and_pose(self):
        self.assertEqual(self.record.scene_id, "object_a/floating/scale013")
        self.assertEqual(self.record.mesh_path, self.mesh_path.resolve())
        self.assertAlmostEqual(self.record.scale, 0.133)
        np.testing.assert_allclose(
            self.record.object_pose_wxyz,
            [0.2, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0],
        )
        self.assertTrue(self.record.stored_scene_path.endswith("object_a/floating/scale013.npy"))

    def test_complete_surface_sampling_is_deterministic_scaled_and_not_recentered(self):
        first = sample_scaled_surface(self.mesh_path, 0.2, 512, 7)
        second = sample_scaled_surface(self.mesh_path, 0.2, 512, 7)
        third = sample_scaled_surface(self.mesh_path, 0.2, 512, 8)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (512, 3))
        self.assertEqual(first.dtype, np.float32)
        self.assertEqual(sha256_array(first), sha256_array(second))
        self.assertNotEqual(sha256_array(first), sha256_array(third))
        self.assertGreater(float(first.mean()), 0.0)
        self.assertLessEqual(float(first.max()), 0.2 + 1e-7)

    def test_root_and_wrist_are_resolved_to_the_palm_frame(self):
        q = np.zeros(len(DRO_SHADOW_Q_NAMES), dtype=np.float64)
        transform = dro_q_to_object_palm_transform(q)
        np.testing.assert_allclose(transform[:3, 3], [0.0, -0.010, 0.24701])
        np.testing.assert_allclose(transform[:3, :3], np.eye(3))
        q[DRO_SHADOW_Q_NAMES.index("virtual_joint_x")] = 0.4
        q[DRO_SHADOW_Q_NAMES.index("WRJ2")] = 0.2
        changed = dro_q_to_object_palm_transform(q)
        self.assertAlmostEqual(changed[0, 3], 0.4 + 0.034 * np.sin(0.2))

    def test_shadow_mapping_is_direct_by_name_and_reports_limit_excess(self):
        q = np.zeros(len(DRO_SHADOW_Q_NAMES), dtype=np.float64)
        q[DRO_SHADOW_Q_NAMES.index("FFJ4")] = 0.2
        q[DRO_SHADOW_Q_NAMES.index("THJ2")] = -0.4
        mapped, excess = map_dro_shadow_fingers(q)
        by_name = dict(zip(BENCH_SHADOW_JOINT_NAMES, mapped))
        self.assertAlmostEqual(by_name["rh_FFJ4"], 0.2)
        self.assertAlmostEqual(by_name["rh_THJ2"], -0.4)
        self.assertEqual(float(excess.max()), 0.0)
        q[DRO_SHADOW_Q_NAMES.index("FFJ2")] = -0.01
        _, excess = map_dro_shadow_fingers(q)
        self.assertAlmostEqual(excess[BENCH_SHADOW_JOINT_NAMES.index("rh_FFJ2")], 0.01)

    def test_three_official_controller_stages_export_to_unchanged_schema(self):
        stages = np.zeros((20, 3, len(DRO_SHADOW_Q_NAMES)), dtype=np.float32)
        stages[:, :, DRO_SHADOW_Q_NAMES.index("virtual_joint_z")] = 0.1
        stages[:, 0, DRO_SHADOW_Q_NAMES.index("FFJ1")] = 0.2
        stages[:, 1, DRO_SHADOW_Q_NAMES.index("FFJ1")] = 0.5
        stages[:, 2, DRO_SHADOW_Q_NAMES.index("FFJ1")] = 0.7
        artifact, excess = make_bench_artifact(
            stages, self.record.object_pose_wxyz, self.record.stored_scene_path
        )
        self.assertEqual(STAGE_NAMES, ("pregrasp", "grasp", "squeeze"))
        self.assertEqual(set(artifact), {"robot_pose", "joint_names", "scene_path"})
        self.assertEqual(artifact["robot_pose"].shape, (1, 20, 3, 29))
        self.assertEqual(artifact["robot_pose"].dtype, np.float32)
        self.assertEqual(excess.shape, (20, 3, 22))
        validate_artifact(artifact, self.record, stages)
        ffj1 = 7 + BENCH_SHADOW_JOINT_NAMES.index("rh_FFJ1")
        np.testing.assert_allclose(artifact["robot_pose"][0, 0, :, ffj1], [0.2, 0.5, 0.7])
        np.testing.assert_allclose(
            artifact["robot_pose"][0, 0, :, :7],
            np.repeat(artifact["robot_pose"][0, 0, 1:2, :7], 3, axis=0),
        )

    def test_world_export_uses_object_pose_composition(self):
        q = np.zeros(len(DRO_SHADOW_Q_NAMES), dtype=np.float64)
        pose, _ = dro_q_to_bench_pose(q, self.record.object_pose_wxyz)
        np.testing.assert_allclose(pose[:3], [0.2, -0.11, 0.54701], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
