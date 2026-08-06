"""CPU-only tests for the DRO/DGN2k/Bench boundary."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from grasp_generation.experiments.bimanbodex_dro.contracts import (
    BENCH_SHADOW_JOINT_NAMES,
    DRO_SHADOW_FINGER_JOINT_NAMES,
    DRO_SHADOW_Q_NAMES,
    STAGE_NAMES,
    build_dro_shadow_pk_chain,
    clamp_dro_shadow_export_stages,
    dro_stage_q_to_object_palm_transforms,
    dro_q_to_bench_pose,
    legacy_dro_q_to_object_palm_transform,
    legacy_dro_stage_q_to_object_palm_transforms,
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
                    "pose": np.array([0.2, -0.1, 0.3, 1.000001, 0.0, 0.0, 0.0]),
                },
                "table": {
                    "type": "plane",
                    "pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                    "size": np.array([0.0, 0.0, 1.0]),
                },
            },
        }
        np.save(self.scene_path, config, allow_pickle=True)
        self.record = load_scene_record(self.scene_path, self.scene_root)
        self.urdf_path = self.root / "shadow.urdf"
        self.urdf_path.write_text(self._shadow_urdf(), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _shadow_urdf() -> str:
        lines = ["<robot name=\"shadow\">", "<link name=\"world\"/>"]
        parent = "world"
        virtual = (
            ("virtual_joint_x", "prismatic", "1 0 0"),
            ("virtual_joint_y", "prismatic", "0 1 0"),
            ("virtual_joint_z", "prismatic", "0 0 1"),
            ("virtual_joint_roll", "revolute", "1 0 0"),
            ("virtual_joint_pitch", "revolute", "0 1 0"),
            ("virtual_joint_yaw", "revolute", "0 0 1"),
        )
        for name, joint_type, axis in virtual:
            child = name.replace("joint", "link")
            lines.extend(
                (
                    f'<link name="{child}"/>',
                    f'<joint name="{name}" type="{joint_type}">',
                    f'<parent link="{parent}"/><child link="{child}"/>',
                    '<origin xyz="0 0 0" rpy="0 0 0"/>',
                    f'<axis xyz="{axis}"/><limit lower="-10" upper="10" effort="1" velocity="1"/>',
                    "</joint>",
                )
            )
            parent = child
        lines.extend(
            (
                '<link name="forearm"/>',
                '<joint name="virtual_robot" type="fixed">',
                f'<parent link="{parent}"/><child link="forearm"/>',
                '<origin xyz="0 0 0" rpy="0 0 0"/></joint>',
                '<link name="wrist"/>',
                '<joint name="WRJ2" type="revolute"><parent link="forearm"/>',
                '<child link="wrist"/><origin xyz="0 -0.010 0.21301" rpy="0 0 0"/>',
                '<axis xyz="0 1 0"/><limit lower="-10" upper="10" effort="1" velocity="1"/></joint>',
                '<link name="palm"/>',
                '<joint name="WRJ1" type="revolute"><parent link="wrist"/>',
                '<child link="palm"/><origin xyz="0 0 0.034" rpy="0 0 0"/>',
                '<axis xyz="1 0 0"/><limit lower="-10" upper="10" effort="1" velocity="1"/></joint>',
            )
        )
        parent = "palm"
        for index, joint_name in enumerate(DRO_SHADOW_FINGER_JOINT_NAMES):
            child = f"finger_link_{index}"
            lines.extend(
                (
                    f'<link name="{child}"/>',
                    f'<joint name="{joint_name}" type="revolute">',
                    f'<parent link="{parent}"/><child link="{child}"/>',
                    '<origin xyz="0 0 0" rpy="0 0 0"/>',
                    '<axis xyz="1 0 0"/><limit lower="-10" upper="10" effort="1" velocity="1"/>',
                    "</joint>",
                )
            )
            parent = child
        lines.append("</robot>\n")
        return "\n".join(lines)

    def test_scene_loader_preserves_exact_asset_scale_and_pose(self):
        self.assertEqual(self.record.scene_id, "object_a/floating/scale013")
        self.assertEqual(self.record.mesh_path, self.mesh_path.resolve())
        self.assertAlmostEqual(self.record.scale, 0.133)
        np.testing.assert_allclose(
            self.record.object_pose_wxyz,
            [0.2, -0.1, 0.3, 1.000001, 0.0, 0.0, 0.0],
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(self.record.stored_scene_path.endswith("object_a/floating/scale013.npy"))
        np.testing.assert_array_equal(self.record.table_normal_world, [0.0, 0.0, 1.0])
        self.assertEqual(self.record.to_manifest()["table_type"], "plane")

    def test_scene_loader_rejects_missing_or_invalid_explicit_table(self):
        value = np.load(self.scene_path, allow_pickle=True).item()
        del value["scene"]["table"]
        np.save(self.scene_path, value, allow_pickle=True)
        with self.assertRaisesRegex(ValueError, "explicit plane"):
            load_scene_record(self.scene_path, self.scene_root)

        value["scene"]["table"] = {
            "type": "plane",
            "pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            "size": np.zeros(3),
        }
        np.save(self.scene_path, value, allow_pickle=True)
        with self.assertRaisesRegex(ValueError, "non-zero vector"):
            load_scene_record(self.scene_path, self.scene_root)

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

    def test_actual_pk_fk_resolves_root_and_wrist_to_the_palm_frame(self):
        stages = np.zeros((2, 3, len(DRO_SHADOW_Q_NAMES)), dtype=np.float32)
        chain = build_dro_shadow_pk_chain(self.urdf_path)
        transforms = dro_stage_q_to_object_palm_transforms(chain, stages)
        transform = transforms[0, 0]
        np.testing.assert_allclose(transform[:3, 3], [0.0, -0.010, 0.24701])
        np.testing.assert_allclose(transform[:3, :3], np.eye(3))
        stages[1, :, DRO_SHADOW_Q_NAMES.index("virtual_joint_x")] = 0.4
        stages[1, :, DRO_SHADOW_Q_NAMES.index("WRJ2")] = 0.2
        changed = dro_stage_q_to_object_palm_transforms(chain, stages)[1, 0]
        self.assertAlmostEqual(changed[0, 3], 0.4 + 0.034 * np.sin(0.2))
        np.testing.assert_allclose(
            changed,
            legacy_dro_q_to_object_palm_transform(stages[1, 0]),
            atol=5e-8,
        )

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
        palm_transforms = dro_stage_q_to_object_palm_transforms(
            build_dro_shadow_pk_chain(self.urdf_path),
            stages,
        )
        artifact, excess = make_bench_artifact(
            stages,
            self.record.object_pose_wxyz,
            self.record.stored_scene_path,
            palm_object_transforms=palm_transforms,
        )
        self.assertEqual(STAGE_NAMES, ("pregrasp", "grasp", "squeeze"))
        self.assertEqual(set(artifact), {"robot_pose", "joint_names", "scene_path"})
        self.assertEqual(artifact["robot_pose"].shape, (1, 20, 3, 29))
        self.assertEqual(artifact["robot_pose"].dtype, np.float32)
        self.assertEqual(excess.shape, (20, 3, 22))
        validate_artifact(
            artifact,
            self.record,
            stages,
            palm_object_transforms=palm_transforms,
        )
        ffj1 = 7 + BENCH_SHADOW_JOINT_NAMES.index("rh_FFJ1")
        np.testing.assert_allclose(artifact["robot_pose"][0, 0, :, ffj1], [0.2, 0.5, 0.7])
        np.testing.assert_allclose(
            artifact["robot_pose"][0, 0, :, :7],
            np.repeat(artifact["robot_pose"][0, 0, 1:2, :7], 3, axis=0),
        )

    def test_export_clamp_uses_limit_intersection_without_mutating_raw_q(self):
        raw = np.zeros((2, 3, len(DRO_SHADOW_Q_NAMES)), dtype=np.float32)
        raw[0, 1, DRO_SHADOW_Q_NAMES.index("THJ3")] = 0.5
        raw[1, 2, DRO_SHADOW_Q_NAMES.index("FFJ2")] = -0.5
        original = raw.copy()
        dro_limits = {
            name: (-10.0, 10.0) for name in DRO_SHADOW_FINGER_JOINT_NAMES
        }
        dro_limits["THJ3"] = (-0.1, 0.1)
        export, diagnostics = clamp_dro_shadow_export_stages(raw, dro_limits)

        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(raw[:, :, :8], export[:, :, :8])
        self.assertAlmostEqual(
            float(export[0, 1, DRO_SHADOW_Q_NAMES.index("THJ3")]), 0.1, places=6
        )
        self.assertAlmostEqual(
            float(export[1, 2, DRO_SHADOW_Q_NAMES.index("FFJ2")]), 0.0, places=6
        )
        self.assertEqual(len(diagnostics), 2)
        self.assertEqual(
            {(item["candidate_index"], item["stage_name"], item["bench_joint_name"])
             for item in diagnostics},
            {(0, "grasp", "rh_THJ3"), (1, "squeeze", "rh_FFJ2")},
        )
        self.assertAlmostEqual(diagnostics[0]["delta"], -0.4, places=6)

    def test_world_export_uses_object_pose_composition(self):
        q = np.zeros(len(DRO_SHADOW_Q_NAMES), dtype=np.float64)
        palm = legacy_dro_q_to_object_palm_transform(q)
        pose, _ = dro_q_to_bench_pose(
            q,
            self.record.object_pose_wxyz,
            palm_object_transform=palm,
        )
        np.testing.assert_allclose(pose[:3], [0.2, -0.11, 0.54701], atol=1e-8)

    def test_legacy_stage_fk_is_only_an_explicit_compatibility_path(self):
        stages = np.zeros((1, 3, len(DRO_SHADOW_Q_NAMES)), dtype=np.float32)
        legacy = legacy_dro_stage_q_to_object_palm_transforms(stages)
        self.assertEqual(legacy.shape, (1, 3, 4, 4))


if __name__ == "__main__":
    unittest.main()
