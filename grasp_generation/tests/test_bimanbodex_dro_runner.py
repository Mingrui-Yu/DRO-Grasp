"""CPU-only orchestration, failure accounting, and no-overwrite tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from grasp_generation.experiments.bimanbodex_dro.contracts import DRO_SHADOW_Q_NAMES
from grasp_generation.experiments.bimanbodex_dro.runner import (
    dry_run,
    run,
    validate_run_outputs,
)


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
                    }
                },
            },
            allow_pickle=True,
        )
        self.scene_list = self.root / "scenes.txt"
        self.scene_list.write_text("object_a/floating/scale013\n", encoding="utf-8")
        self.checkpoint = self.root / "model_3robots.pth"
        self.urdf = self.root / "shadow.urdf"
        self.robot_pc = self.root / "shadowhand.pt"
        for path in (self.checkpoint, self.urdf, self.robot_pc):
            path.write_bytes(path.name.encode("utf-8"))
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
    def successful_inference(record, points, candidate_seeds):
        count = len(candidate_seeds)
        initial = np.zeros((count, len(DRO_SHADOW_Q_NAMES)), dtype=np.float32)
        stages = np.zeros((count, 3, len(DRO_SHADOW_Q_NAMES)), dtype=np.float32)
        stages[:, 0, DRO_SHADOW_Q_NAMES.index("FFJ1")] = 0.2
        stages[:, 1, DRO_SHADOW_Q_NAMES.index("FFJ1")] = 0.4
        stages[:, 2, DRO_SHADOW_Q_NAMES.index("FFJ1")] = 0.6
        return {
            "initial_q": initial,
            "stage_q": stages,
            "timing_seconds": np.arange(count, dtype=np.float64) / 100.0,
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


if __name__ == "__main__":
    unittest.main()
