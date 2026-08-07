"""CPU contracts for final-pose tabletop collision filtering."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from grasp_generation.experiments.bimanbodex_dro.contracts import (
    DRO_SHADOW_FINGER_JOINT_NAMES,
    DRO_SHADOW_Q_NAMES,
    build_dro_shadow_pk_chain,
)
from grasp_generation.experiments.bimanbodex_dro.tabletop_filter import (
    TabletopCollisionModel,
)


class TabletopFilterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        mesh_path = self.root / "finger_collision.obj"
        mesh_path.write_text(
            "v -0.02 -0.02 0.00\n"
            "v 0.02 -0.02 0.00\n"
            "v 0.00 0.02 0.00\n"
            "v 0.00 0.00 0.04\n"
            "f 1 2 3\n"
            "f 1 2 4\n"
            "f 2 3 4\n"
            "f 3 1 4\n",
            encoding="utf-8",
        )
        self.urdf_path = self.root / "shadow.urdf"
        self.urdf_path.write_text(self._shadow_urdf(), encoding="utf-8")
        self.chain = build_dro_shadow_pk_chain(self.urdf_path)
        self.model = TabletopCollisionModel.from_urdf(self.urdf_path)
        self.record = SimpleNamespace(
            object_pose_wxyz=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            table_origin_world=np.array([0.0, 0.0, 0.0]),
            table_normal_world=np.array([0.0, 0.0, 1.0]),
        )

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _link(name: str, collision: str = "") -> str:
        return f'<link name="{name}">{collision}</link>'

    def _shadow_urdf(self) -> str:
        ancestor_box = (
            '<collision><origin xyz="0 0 -2" rpy="0 0 0"/>'
            '<geometry><box size="1 1 1"/></geometry></collision>'
        )
        palm_box = (
            '<collision><origin xyz="0 0 0.1" rpy="0 0 0"/>'
            '<geometry><box size="0.2 0.2 0.2"/></geometry></collision>'
        )
        finger_sphere = (
            '<collision><origin xyz="0 0 0.2" rpy="0 0 0"/>'
            '<geometry><sphere radius="0.05"/></geometry></collision>'
        )
        finger_cylinder = (
            '<collision><origin xyz="0 0 0.2" rpy="0 0 0"/>'
            '<geometry><cylinder radius="0.03" length="0.1"/></geometry></collision>'
        )
        finger_mesh = (
            '<collision><origin xyz="0 0 0.2" rpy="0 0 0"/>'
            '<geometry><mesh filename="finger_collision.obj" scale="1 1 1"/></geometry>'
            '</collision>'
        )

        lines = ['<robot name="shadow">', self._link("world")]
        parent = "world"
        virtual_joints = (
            ("virtual_joint_x", "prismatic", "1 0 0"),
            ("virtual_joint_y", "prismatic", "0 1 0"),
            ("virtual_joint_z", "prismatic", "0 0 1"),
            ("virtual_joint_roll", "revolute", "1 0 0"),
            ("virtual_joint_pitch", "revolute", "0 1 0"),
            ("virtual_joint_yaw", "revolute", "0 0 1"),
        )
        for index, (joint_name, joint_type, axis) in enumerate(virtual_joints):
            child = f"virtual_link_{index}"
            lines.extend(
                (
                    self._link(child),
                    f'<joint name="{joint_name}" type="{joint_type}">',
                    f'<parent link="{parent}"/><child link="{child}"/>',
                    '<origin xyz="0 0 0" rpy="0 0 0"/>',
                    f'<axis xyz="{axis}"/>',
                    '<limit lower="-10" upper="10" effort="1" velocity="1"/>',
                    '</joint>',
                )
            )
            parent = child

        lines.extend(
            (
                self._link("forearm", ancestor_box),
                '<joint name="virtual_robot" type="fixed">',
                f'<parent link="{parent}"/><child link="forearm"/>',
                '<origin xyz="0 0 0" rpy="0 0 0"/></joint>',
                self._link("wrist", ancestor_box),
                '<joint name="WRJ2" type="revolute"><parent link="forearm"/>',
                '<child link="wrist"/><origin xyz="0 0 0" rpy="0 0 0"/>',
                '<axis xyz="0 1 0"/><limit lower="-10" upper="10" effort="1" velocity="1"/></joint>',
                self._link("palm", palm_box),
                '<joint name="WRJ1" type="revolute"><parent link="wrist"/>',
                '<child link="palm"/><origin xyz="0 0 0" rpy="0 0 0"/>',
                '<axis xyz="1 0 0"/><limit lower="-10" upper="10" effort="1" velocity="1"/></joint>',
            )
        )
        parent = "palm"
        for index, joint_name in enumerate(DRO_SHADOW_FINGER_JOINT_NAMES):
            child = f"finger_link_{index}"
            collision = (
                finger_sphere
                if index == 0
                else finger_cylinder
                if index == 1
                else finger_mesh
                if index == 2
                else ""
            )
            lines.extend(
                (
                    self._link(child, collision),
                    f'<joint name="{joint_name}" type="revolute">',
                    f'<parent link="{parent}"/><child link="{child}"/>',
                    '<origin xyz="0 0 0" rpy="0 0 0"/>',
                    '<axis xyz="1 0 0"/><limit lower="-10" upper="10" effort="1" velocity="1"/>',
                    '</joint>',
                )
            )
            parent = child
        lines.append('</robot>\n')
        return "\n".join(lines)

    def test_scope_uses_palm_descendants_and_actual_collision_shapes(self):
        manifest = self.model.to_manifest()
        self.assertEqual(manifest["root_link"], "palm")
        self.assertNotIn("forearm", manifest["scoped_links"])
        self.assertNotIn("wrist", manifest["scoped_links"])
        self.assertEqual(
            manifest["geometry_counts"],
            {"box": 1, "cylinder": 1, "mesh": 1, "sphere": 1},
        )
        self.assertEqual(manifest["collision_geometry_count"], 4)

    def test_margin_zero_is_strict_for_above_contact_and_penetration(self):
        q = np.zeros((3, len(DRO_SHADOW_Q_NAMES)), dtype=np.float32)
        root_z = DRO_SHADOW_Q_NAMES.index("virtual_joint_z")
        q[:, root_z] = [0.01, 0.0, -0.01]
        passed, diagnostics = self.model.evaluate_final_grasps(
            self.chain, q, self.record, margin=0.0
        )
        np.testing.assert_array_equal(passed, [True, False, False])
        self.assertAlmostEqual(diagnostics[0]["min_z"], 0.01, places=7)
        self.assertAlmostEqual(diagnostics[1]["min_z"], 0.0, places=7)
        self.assertAlmostEqual(diagnostics[2]["min_z"], -0.01, places=7)
        self.assertEqual(diagnostics[0]["worst_link"], "palm")

    def test_signed_height_uses_explicit_rotated_table_frame(self):
        record = SimpleNamespace(
            object_pose_wxyz=self.record.object_pose_wxyz,
            table_origin_world=np.array([-0.1, 0.0, 0.0]),
            table_normal_world=np.array([1.0, 0.0, 0.0]),
        )
        q = np.zeros((2, len(DRO_SHADOW_Q_NAMES)), dtype=np.float32)
        root_x = DRO_SHADOW_Q_NAMES.index("virtual_joint_x")
        q[:, root_x] = [0.01, 0.0]
        passed, diagnostics = self.model.evaluate_final_grasps(
            self.chain, q, record, margin=0.0
        )
        np.testing.assert_array_equal(passed, [True, False])
        self.assertAlmostEqual(diagnostics[0]["min_z"], 0.01, places=7)
        self.assertAlmostEqual(diagnostics[1]["min_z"], 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
