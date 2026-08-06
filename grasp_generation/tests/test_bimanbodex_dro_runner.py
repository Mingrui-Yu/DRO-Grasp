"""CPU-only orchestration, failure accounting, and no-overwrite tests."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from grasp_generation.experiments.bimanbodex_dro.contracts import (
    DRO_SHADOW_FINGER_JOINT_NAMES,
    DRO_SHADOW_Q_NAMES,
    LEGACY_RAW_SCHEMA_VERSION,
    LEGACY_RUN_SCHEMA_VERSION,
    load_scene_record,
    scene_manifest_sha256,
)
from grasp_generation.experiments.bimanbodex_dro.initialization import (
    apply_initialization,
    default_initialization_config,
    resolve_initialization_config,
)
from grasp_generation.experiments.bimanbodex_dro.pairing import (
    validate_paired_outputs,
)
from grasp_generation.experiments.bimanbodex_dro.runner import (
    _batch_robot_point_cloud,
    _controller_stages_on_cpu,
    dry_run,
    run,
    validate_run_outputs,
)
from grasp_generation.experiments.bimanbodex_dro.visualizer import ViewerRun


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.scene_root = self.root / "scene_cfg"
        self.scene_path = self.scene_root / "object_a" / "floating" / "scale013.npy"
        mesh_path = self.root / "processed_data" / "object_a" / "mesh" / "simplified.obj"
        mesh_path.parent.mkdir(parents=True)
        mesh_path.write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
            "f 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n",
            encoding="utf-8",
        )
        self.scene_path.parent.mkdir(parents=True)
        np.save(
            self.scene_path,
            {
                "scene_id": "object_a/floating/scale013",
                "task": {"obj_name": "object_a"},
                "scene": {
                    "object_a": {
                        "type": "rigid_object",
                        "file_path": "../../../processed_data/object_a/mesh/simplified.obj",
                        "scale": np.array([0.133, 0.133, 0.133]),
                        "pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                    },
                    "table": {
                        "type": "plane",
                        "pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                        "size": np.array([0.0, 0.0, 1.0]),
                    },
                },
            },
            allow_pickle=True,
        )
        self.scene_list = self.root / "scenes.txt"
        self.scene_list.write_text("object_a/floating/scale013\n", encoding="utf-8")
        self.checkpoint = self.root / "model_3robots.pth"
        self.urdf = self.root / "shadow.urdf"
        self.robot_pc = self.root / "shadowhand.pt"
        self.checkpoint.write_bytes(self.checkpoint.name.encode("utf-8"))
        self.robot_pc.write_bytes(self.robot_pc.name.encode("utf-8"))
        self.urdf.write_text(self._shadow_urdf(), encoding="utf-8")
        self.repo_root = Path(__file__).resolve().parents[2]
        self.config = {
            "robot_name": "shadowhand",
            "scene_root": str(self.scene_root),
            "reference_grasp_roots": [],
            "scene_list": str(self.scene_list),
            "checkpoint": str(self.checkpoint),
            "shadow_urdf": str(self.urdf),
            "shadow_point_cloud": str(self.robot_pc),
            "point_cloud_mode": "complete_scaled_mesh",
            "frame_contract": "object_local_no_normalization",
            "network_center_robot_pc": True,
            "point_count": 512,
            "point_seed": 11,
            "candidate_count": 20,
            "inference_seed": 12,
            "n_iter": 2,
            "device": "cuda:0",
            "failure_policy": "scene_atomic_all_candidates_fail",
            "output_root": None,
            "max_scenes": None,
        }

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _shadow_urdf() -> str:
        lines = ["<robot name=\"shadow\">", "<link name=\"world\"/>"]
        parent = "world"
        moving_joints = (
            ("virtual_joint_x", "prismatic", "1 0 0"),
            ("virtual_joint_y", "prismatic", "0 1 0"),
            ("virtual_joint_z", "prismatic", "0 0 1"),
            ("virtual_joint_roll", "revolute", "1 0 0"),
            ("virtual_joint_pitch", "revolute", "0 1 0"),
            ("virtual_joint_yaw", "revolute", "0 0 1"),
            ("WRJ2", "revolute", "0 1 0"),
            ("WRJ1", "revolute", "1 0 0"),
        ) + tuple(
            (joint_name, "revolute", "1 0 0")
            for joint_name in DRO_SHADOW_FINGER_JOINT_NAMES
        )
        for index, (joint_name, joint_type, axis) in enumerate(moving_joints):
            child = "palm" if joint_name == "WRJ1" else f"link_{index}"
            origin = (
                "0 -0.010 0.21301"
                if joint_name == "WRJ2"
                else "0 0 0.034"
                if joint_name == "WRJ1"
                else "0 0 0"
            )
            lines.extend(
                (
                    f'<link name="{child}"/>',
                    f'<joint name="{joint_name}" type="{joint_type}">',
                    f'<parent link="{parent}"/><child link="{child}"/>',
                    f'<origin xyz="{origin}" rpy="0 0 0"/>',
                    f'<axis xyz="{axis}"/>',
                    '<limit lower="-10" upper="10" effort="1" velocity="1"/>',
                    "</joint>",
                )
            )
            parent = child
        lines.append("</robot>\n")
        return "\n".join(lines)

    def test_released_robot_point_cloud_gets_explicit_network_batch(self):
        released = np.zeros((512, 4), dtype=np.float32)
        released[:, 3] = 7.0
        batched = _batch_robot_point_cloud(released, 512)
        self.assertEqual(batched.shape, (1, 512, 3))
        self.assertEqual(batched.dtype, np.float32)
        with self.assertRaisesRegex(ValueError, "must have shape"):
            _batch_robot_point_cloud(released[None], 512)

    def test_released_controller_runs_on_cpu_and_preserves_stage_shapes(self):
        observed = {}

        def fake_controller(robot_name, q_grasp):
            observed["robot_name"] = robot_name
            observed["device"] = q_grasp.device.type
            return q_grasp - 1.0, q_grasp + 1.0

        q_grasp = torch.zeros((1, len(DRO_SHADOW_Q_NAMES)))
        q_outer, q_center, q_inner = _controller_stages_on_cpu(
            fake_controller, q_grasp
        )
        self.assertEqual(observed, {"robot_name": "shadowhand", "device": "cpu"})
        self.assertEqual(q_outer.shape, q_center.shape)
        self.assertEqual(q_inner.shape, q_center.shape)
        torch.testing.assert_close(q_outer, q_center - 1.0)
        torch.testing.assert_close(q_inner, q_center + 1.0)

    @staticmethod
    def successful_inference(record, points, candidate_seeds):
        count = len(candidate_seeds)
        released = np.zeros((count, len(DRO_SHADOW_Q_NAMES)), dtype=np.float32)
        initial = released.copy()
        resolved_initialization = resolve_initialization_config(None, count)
        metadata = []
        rng_digests = []
        for candidate_index, candidate_seed in enumerate(candidate_seeds):
            initial[candidate_index], item = apply_initialization(
                released[candidate_index],
                record,
                candidate_index,
                resolved_initialization,
            )
            item["candidate_seed"] = candidate_seed
            metadata.append(item)
            rng_digests.append({"cpu": format(candidate_index + 1, "064x")})
        stages = np.zeros((count, 3, len(DRO_SHADOW_Q_NAMES)), dtype=np.float32)
        stages[:, 0, DRO_SHADOW_Q_NAMES.index("FFJ1")] = 0.2
        stages[:, 1, DRO_SHADOW_Q_NAMES.index("FFJ1")] = 0.4
        stages[:, 2, DRO_SHADOW_Q_NAMES.index("FFJ1")] = 0.6
        stages[:, 1, DRO_SHADOW_Q_NAMES.index("THJ3")] = 0.5
        return {
            "released_initial_q": released,
            "initial_q": initial,
            "stage_q": stages,
            "timing_seconds": np.arange(count, dtype=np.float64) / 100.0,
            "initialization_metadata": metadata,
            "pre_network_rng_state_sha256": rng_digests,
            "failures": [],
        }

    @staticmethod
    def failed_inference(record, points, candidate_seeds):
        result = RunnerTests.successful_inference(record, points, candidate_seeds)
        result["stage_q"][3] = np.nan
        result["failures"] = [
            {
                "candidate_index": 3,
                "candidate_seed": candidate_seeds[3],
                "error_type": "RuntimeError",
                "message": "synthetic failure",
            }
        ]
        return result

    @staticmethod
    def malformed_inference(record, points, candidate_seeds):
        return {
            "initial_q": np.zeros((1, len(DRO_SHADOW_Q_NAMES)), dtype=np.float32),
            "stage_q": np.zeros((1, 3, len(DRO_SHADOW_Q_NAMES)), dtype=np.float32),
            "timing_seconds": np.zeros((1,), dtype=np.float64),
            "failures": [],
        }

    @staticmethod
    def inference_for_mode(mode):
        initialization = default_initialization_config()
        initialization["mode"] = mode
        resolved_initialization = resolve_initialization_config(initialization, 20)

        def inference(record, points, candidate_seeds):
            result = RunnerTests.successful_inference(record, points, candidate_seeds)
            released = result["released_initial_q"]
            metadata = []
            for candidate_index, candidate_seed in enumerate(candidate_seeds):
                result["initial_q"][candidate_index], item = apply_initialization(
                    released[candidate_index],
                    record,
                    candidate_index,
                    resolved_initialization,
                )
                item["candidate_seed"] = candidate_seed
                metadata.append(item)
            result["initialization_metadata"] = metadata
            return result

        return inference

    def test_dry_run_resolves_contract_without_creating_output(self):
        output_root = self.root / "dry-output"
        self.config["output_root"] = str(output_root)
        result = dry_run(self.repo_root, self.config)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["scene_count"], 1)
        self.assertEqual(result["source_scene_count"], 1)
        self.assertEqual(len(result["source_scene_manifest_sha256"]), 64)
        self.assertEqual(result["source_scale_histogram"], {"0.133": 1})
        self.assertEqual(result["first_scene_point_cloud_shape"], [512, 3])
        self.assertFalse(output_root.exists())

    def test_expected_scene_contract_rejects_count_mismatch(self):
        self.config["expected_scene_count"] = 996
        with self.assertRaisesRegex(ValueError, "authoritative scene count mismatch"):
            dry_run(self.repo_root, self.config)

    def test_success_writes_shape_1_20_3_29_and_exact_accounting(self):
        output_root = self.root / "success-output"
        self.config["output_root"] = str(output_root)
        manifest = run(self.repo_root, self.config, self.successful_inference)
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["candidate_count"], 20)
        self.assertEqual(manifest["completed_candidate_count"], 20)
        self.assertEqual(manifest["failed_candidate_count"], 0)
        self.assertEqual(manifest["source_scene_count"], 1)
        artifact_path = output_root / "graspdata" / "object_a" / "floating" / "scale013_grasp.npy"
        artifact = np.load(artifact_path, allow_pickle=True).item()
        self.assertEqual(artifact["robot_pose"].shape, (1, 20, 3, 29))
        self.assertTrue(np.isfinite(artifact["robot_pose"]).all())
        raw_path = output_root / "raw" / "object_a" / "floating" / "scale013.npy"
        raw = np.load(raw_path, allow_pickle=True).item()
        thj3 = DRO_SHADOW_Q_NAMES.index("THJ3")
        np.testing.assert_array_equal(raw["stage_q"][:, 1, thj3], 0.5)
        self.assertTrue(np.all(raw["export_stage_q"][:, 1, thj3] <= 0.20944))
        self.assertEqual(len(raw["export_clamp_diagnostics"]), 20)
        self.assertTrue(
            all(
                item["bench_joint_name"] == "rh_THJ3"
                for item in raw["export_clamp_diagnostics"]
            )
        )
        failure_manifest = json.loads((output_root / "failure_manifest.json").read_text())
        self.assertEqual(failure_manifest["failures"], [])
        validation = validate_run_outputs(output_root)
        self.assertEqual(validation["status"], "valid")
        self.assertEqual(validation["completed_candidate_count"], 20)
        with self.assertRaises(FileExistsError):
            run(self.repo_root, self.config, self.successful_inference)

    def test_candidate_failure_invalidates_scene_without_normal_artifact(self):
        output_root = self.root / "failure-output"
        self.config["output_root"] = str(output_root)
        manifest = run(self.repo_root, self.config, self.failed_inference)
        self.assertEqual(manifest["candidate_count"], 20)
        self.assertEqual(manifest["completed_candidate_count"], 0)
        self.assertEqual(manifest["failed_candidate_count"], 20)
        self.assertEqual(manifest["status"], "failed")
        self.assertFalse((output_root / "graspdata").exists())
        failed_raw = output_root / "failed_raw" / "object_a" / "floating" / "scale013.npy"
        self.assertTrue(failed_raw.is_file())
        failures = json.loads((output_root / "failure_manifest.json").read_text())["failures"]
        self.assertEqual(failures[0]["candidate_failures"][0]["candidate_index"], 3)
        validation = validate_run_outputs(output_root)
        self.assertEqual(validation["status"], "valid")
        self.assertEqual(validation["failed_candidate_count"], 20)

    def test_malformed_inference_is_accounted_as_scene_failure(self):
        output_root = self.root / "malformed-output"
        self.config["output_root"] = str(output_root)
        manifest = run(self.repo_root, self.config, self.malformed_inference)
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["failed_candidate_count"], 20)
        failures = json.loads(
            (output_root / "failure_manifest.json").read_text(encoding="utf-8")
        )["failures"]
        self.assertEqual(failures[0]["stage"], "inference_driver")
        validation = validate_run_outputs(output_root)
        self.assertEqual(validation["failed_candidate_count"], 20)

    def test_strict_paired_validator_checks_q_rng_and_candidate_identity(self):
        released_root = self.root / "paired-released"
        tabletop_root = self.root / "paired-tabletop"
        released_config = copy.deepcopy(self.config)
        released_config["output_root"] = str(released_root)
        released_config["initialization"] = default_initialization_config()
        tabletop_config = copy.deepcopy(released_config)
        tabletop_config["output_root"] = str(tabletop_root)
        tabletop_config["initialization"]["mode"] = "tabletop_stratified"
        run(
            self.repo_root,
            released_config,
            self.inference_for_mode("released_random"),
        )
        run(
            self.repo_root,
            tabletop_config,
            self.inference_for_mode("tabletop_stratified"),
        )
        result = validate_paired_outputs(released_root, tabletop_root)
        self.assertEqual(result["status"], "valid_paired_ablation")
        self.assertEqual(result["candidate_count_per_mode"], 20)
        self.assertEqual(result["changed_root_rotation_count"], 20)

        tabletop_raw_path = (
            tabletop_root / "raw" / "object_a" / "floating" / "scale013.npy"
        )
        tabletop_raw = np.load(tabletop_raw_path, allow_pickle=True).item()
        tabletop_raw["pre_network_rng_state_sha256"][0]["cpu"] = "0" * 64
        np.save(tabletop_raw_path, tabletop_raw, allow_pickle=True)
        with self.assertRaisesRegex(ValueError, "pre_network_rng_state_sha256"):
            validate_paired_outputs(released_root, tabletop_root)

    def test_v1_output_validator_and_viewer_remain_read_only_compatible(self):
        output_root = self.root / "legacy-v1-output"
        self.config["output_root"] = str(output_root)
        run(self.repo_root, self.config, self.successful_inference)
        record = load_scene_record(self.scene_path, self.scene_root)

        manifest_path = output_root / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = LEGACY_RUN_SCHEMA_VERSION
        manifest.pop("initialization_mode", None)
        manifest.pop("palm_fk", None)
        manifest["source_scene_manifest_sha256"] = scene_manifest_sha256(
            [record], include_table=False
        )
        table_keys = {
            "table_type",
            "table_pose_wxyz",
            "table_normal_local",
            "table_normal_world",
            "table_origin_world",
        }
        for key in table_keys:
            manifest["scenes"][0].pop(key, None)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        failure_path = output_root / "failure_manifest.json"
        failure_manifest = json.loads(failure_path.read_text(encoding="utf-8"))
        failure_manifest["schema_version"] = LEGACY_RUN_SCHEMA_VERSION
        failure_path.write_text(json.dumps(failure_manifest), encoding="utf-8")

        raw_path = output_root / "raw" / "object_a" / "floating" / "scale013.npy"
        raw = np.load(raw_path, allow_pickle=True).item()
        raw["schema_version"] = LEGACY_RAW_SCHEMA_VERSION
        raw["scene"] = record.to_manifest(include_table=False)
        for key in (
            "initialization_mode",
            "released_initial_q",
            "initialization_metadata",
            "pre_network_rng_state_sha256",
            "palm_fk",
        ):
            raw.pop(key, None)
        np.save(raw_path, raw, allow_pickle=True)

        self.assertEqual(validate_run_outputs(output_root)["status"], "valid")
        loaded = ViewerRun(output_root).load_scene(record.scene_id)
        self.assertEqual(loaded.raw["schema_version"], LEGACY_RAW_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
