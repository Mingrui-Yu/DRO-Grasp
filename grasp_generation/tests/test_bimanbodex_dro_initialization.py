"""CPU tests for tabletop proposal geometry and paired RNG invariants."""

from __future__ import annotations

import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from grasp_generation.experiments.bimanbodex_dro.contracts import (
    DRO_SHADOW_Q_NAMES,
    legacy_dro_q_to_object_palm_transform,
    quaternion_wxyz_to_matrix,
)
from grasp_generation.experiments.bimanbodex_dro.initialization import (
    PALM_APPROACH_AXIS_LOCAL,
    apply_initialization,
    default_initialization_config,
    resolve_initialization_config,
    torch_rng_state_digests,
)


class InitializationTests(unittest.TestCase):
    def setUp(self):
        angle = np.deg2rad(37.0)
        object_quaternion_wxyz = np.array(
            [np.cos(angle / 2.0), np.sin(angle / 2.0), 0.0, 0.0]
        )
        table_angle = np.deg2rad(23.0)
        table_quaternion_wxyz = np.array(
            [np.cos(table_angle / 2.0), 0.0, np.sin(table_angle / 2.0), 0.0]
        )
        table_normal_world = quaternion_wxyz_to_matrix(
            table_quaternion_wxyz
        ) @ np.array([0.0, 0.0, 1.0])
        self.record = SimpleNamespace(
            object_pose_wxyz=np.concatenate(
                (np.array([0.2, -0.1, 0.3]), object_quaternion_wxyz)
            ),
            table_pose_wxyz=np.concatenate((np.zeros(3), table_quaternion_wxyz)),
            table_normal_world=table_normal_world,
        )

    def test_default_sequence_has_fixed_6_8_6_midpoint_strata(self):
        config = default_initialization_config()
        config["mode"] = "tabletop_stratified"
        resolved = resolve_initialization_config(config, 20)
        sequence = resolved["candidate_sequence"]
        self.assertEqual(len(sequence), 20)
        self.assertEqual(
            [item["family"] for item in sequence],
            ["top_down"] * 6 + ["oblique"] * 8 + ["near_horizontal"] * 6,
        )
        for item in sequence:
            family_range = next(
                value["elevation_range_deg"]
                for value in resolved["candidate_ordering"]
                if value["family"] == item["family"]
            )
            self.assertGreater(item["elevation_deg"], family_range[0])
            self.assertLess(item["elevation_deg"], family_range[1])
            self.assertGreaterEqual(item["azimuth_deg"], 0.0)
            self.assertLess(item["azimuth_deg"], 360.0)
        roll_strata = [round(item["roll_deg"], 9) for item in sequence]
        self.assertEqual(len(set(roll_strata)), 20)

    def test_tabletop_q_preserves_translation_wrist_and_fingers(self):
        config = default_initialization_config()
        config["mode"] = "tabletop_stratified"
        resolved = resolve_initialization_config(config, 20)
        released = np.linspace(
            -0.3, 0.4, len(DRO_SHADOW_Q_NAMES), dtype=np.float32
        )
        released[:3] = [0.01, -0.02, 0.03]
        object_rotation_world = quaternion_wxyz_to_matrix(
            self.record.object_pose_wxyz[3:]
        )
        for candidate_index, proposal in enumerate(resolved["candidate_sequence"]):
            effective, metadata = apply_initialization(
                released, self.record, candidate_index, resolved
            )
            np.testing.assert_array_equal(effective[:3], released[:3])
            np.testing.assert_array_equal(effective[6:], released[6:])
            self.assertEqual(metadata["proposal_family"], proposal["family"])
            self.assertGreaterEqual(metadata["upper_hemisphere_dot"], 0.0)
            self.assertLess(metadata["target_alignment_error_rad"], 1e-5)
            palm_object = legacy_dro_q_to_object_palm_transform(effective)[:3, :3]
            direction_world = (
                object_rotation_world @ palm_object @ PALM_APPROACH_AXIS_LOCAL
            )
            np.testing.assert_allclose(
                direction_world,
                metadata["approach_direction_world"],
                rtol=0.0,
                atol=1e-6,
            )
            expected_dot = np.sin(np.deg2rad(proposal["elevation_deg"]))
            self.assertAlmostEqual(
                float(np.dot(direction_world, self.record.table_normal_world)),
                expected_dot,
                places=5,
            )

    def test_released_mode_is_identity_and_tabletop_sampler_consumes_no_torch_rng(self):
        def released_sample(seed: int):
            random.seed(seed)
            torch.manual_seed(seed)
            q = torch.zeros(len(DRO_SHADOW_Q_NAMES), dtype=torch.float32)
            q[3:6] = (torch.rand(3) * 2.0 - 1.0) * torch.pi
            q[5] /= 2.0
            portion = random.uniform(0.65, 0.85)
            q[6:] = portion
            return q.numpy(), torch_rng_state_digests(torch)

        released_config = resolve_initialization_config(None, 20)
        tabletop_value = default_initialization_config()
        tabletop_value["mode"] = "tabletop_stratified"
        tabletop_config = resolve_initialization_config(tabletop_value, 20)

        first_q, first_digest = released_sample(240825)
        released_effective, _ = apply_initialization(
            first_q, self.record, 7, released_config
        )
        after_released = torch_rng_state_digests(torch)
        second_q, second_digest = released_sample(240825)
        tabletop_effective, _ = apply_initialization(
            second_q, self.record, 7, tabletop_config
        )
        after_tabletop = torch_rng_state_digests(torch)

        np.testing.assert_array_equal(first_q, second_q)
        np.testing.assert_array_equal(first_q, released_effective)
        np.testing.assert_array_equal(first_q[:3], tabletop_effective[:3])
        np.testing.assert_array_equal(first_q[6:], tabletop_effective[6:])
        self.assertFalse(np.array_equal(first_q[3:6], tabletop_effective[3:6]))
        self.assertEqual(first_digest, second_digest)
        self.assertEqual(first_digest, after_released)
        self.assertEqual(first_digest, after_tabletop)

    def test_invalid_axis_allocation_and_ranges_are_rejected(self):
        config = default_initialization_config()
        config["palm_approach_axis_local"] = [0.0, -1.0, 0.0]
        with self.assertRaisesRegex(ValueError, r"local \+Y"):
            resolve_initialization_config(config, 20)
        config = default_initialization_config()
        config["candidate_ordering"][0]["count"] = 5
        with self.assertRaisesRegex(ValueError, "sum to 20"):
            resolve_initialization_config(config, 20)
        config = default_initialization_config()
        config["candidate_ordering"][2]["elevation_range_deg"] = [-1.0, 20.0]
        with self.assertRaisesRegex(ValueError, "within"):
            resolve_initialization_config(config, 20)


if __name__ == "__main__":
    unittest.main()
