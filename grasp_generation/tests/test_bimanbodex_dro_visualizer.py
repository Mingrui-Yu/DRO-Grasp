"""Read-only contract tests for the exported three-pose Viser preparation layer."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from grasp_generation.experiments.bimanbodex_dro.contracts import (
    DRO_SHADOW_FINGER_JOINT_NAMES,
    DRO_SHADOW_Q_NAMES,
    legacy_dro_q_to_object_palm_transform,
    map_dro_shadow_fingers,
    pose_wxyz_to_matrix,
)
from grasp_generation.experiments.bimanbodex_dro.initialization import (
    apply_initialization,
    resolve_initialization_config,
)
from grasp_generation.experiments.bimanbodex_dro.runner import run
from grasp_generation.experiments.bimanbodex_dro.visualizer import (
    BenchShadowHandModel,
    ShadowHandModel,
    ViewerRun,
    compare_dro_bench_links,
)
from grasp_generation.scripts.visualize_bimanbodex_dro import ViewerApp, _initial_scene


class VisualizerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.scene_root = self.root / "scene_cfg"
        self.scene_paths = []
        for object_id, scale in (("object_a", 0.133), ("object_b", 0.2)):
            mesh_path = (
                self.root / "processed_data" / object_id / "mesh" / "simplified.obj"
            )
            mesh_path.parent.mkdir(parents=True)
            mesh_path.write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
                "f 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n",
                encoding="utf-8",
            )
            scene_path = self.scene_root / object_id / "floating" / "scale013.npy"
            scene_path.parent.mkdir(parents=True)
            np.save(
                scene_path,
                {
                    "scene_id": f"{object_id}/floating/scale013",
                    "task": {"obj_name": object_id},
                    "scene": {
                        object_id: {
                            "type": "rigid_object",
                            "file_path": (
                                f"../../../processed_data/{object_id}/mesh/simplified.obj"
                            ),
                            "scale": np.array([scale, scale, scale]),
                            "pose": np.array(
                                [0.2, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0]
                            ),
                        },
                        "table": {
                            "type": "plane",
                            "pose": np.array(
                                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                            ),
                            "size": np.array([0.0, 0.0, 1.0]),
                        },
                    },
                },
                allow_pickle=True,
            )
            self.scene_paths.append(scene_path)

        self.scene_list = self.root / "scenes.txt"
        self.scene_list.write_text(
            "object_a/floating/scale013\nobject_b/floating/scale013\n",
            encoding="utf-8",
        )
        self.checkpoint = self.root / "model_3robots.pth"
        self.robot_pc = self.root / "shadowhand.pt"
        self.checkpoint.write_bytes(b"test-checkpoint")
        self.robot_pc.write_bytes(b"test-point-cloud")
        self.urdf = self.root / "shadow_hand_right_extended.urdf"
        self.urdf.write_text(self._shadow_urdf(), encoding="utf-8")
        self.output_root = self.root / "output"
        config = {
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
            "output_root": str(self.output_root),
            "max_scenes": None,
        }
        repo_root = Path(__file__).resolve().parents[2]
        run(repo_root, config, self._inference)

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
                '<link name="forearm"><visual><geometry><box size="0.02 0.02 0.1"/>'
                "</geometry></visual></link>",
                '<joint name="virtual_robot" type="fixed">',
                f'<parent link="{parent}"/><child link="forearm"/>',
                '<origin xyz="0 0 0" rpy="0 0 0"/></joint>',
                '<link name="wrist"><visual><geometry><box size="0.025 0.025 0.03"/>'
                "</geometry></visual></link>",
                '<joint name="WRJ2" type="revolute"><parent link="forearm"/>'
                '<child link="wrist"/><origin xyz="0 -0.010 0.21301" rpy="0 0 0"/>'
                '<axis xyz="0 1 0"/><limit lower="-10" upper="10" effort="1" velocity="1"/></joint>',
                '<link name="palm"><visual><geometry><box size="0.08 0.02 0.1"/>'
                "</geometry></visual></link>",
                '<joint name="WRJ1" type="revolute"><parent link="wrist"/>'
                '<child link="palm"/><origin xyz="0 0 0.034" rpy="0 0 0"/>'
                '<axis xyz="1 0 0"/><limit lower="-10" upper="10" effort="1" velocity="1"/></joint>',
            )
        )
        parent = "palm"
        for index, joint_name in enumerate(DRO_SHADOW_FINGER_JOINT_NAMES):
            child = f"finger_link_{index}"
            lines.extend(
                (
                    f'<link name="{child}"><visual><origin xyz="0 0 0.006"/>'
                    '<geometry><box size="0.008 0.008 0.012"/></geometry></visual></link>',
                    f'<joint name="{joint_name}" type="revolute">',
                    f'<parent link="{parent}"/><child link="{child}"/>',
                    '<origin xyz="0 0 0.012" rpy="0 0 0"/>',
                    '<axis xyz="1 0 0"/><limit lower="-10" upper="10" effort="1" velocity="1"/>',
                    "</joint>",
                )
            )
            parent = child
        lines.append("</robot>\n")
        return "\n".join(lines)

    @staticmethod
    def _inference(record, points, candidate_seeds):
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
        stages[:, :, DRO_SHADOW_Q_NAMES.index("virtual_joint_x")] = 0.05
        stages[:, :, DRO_SHADOW_Q_NAMES.index("WRJ2")] = 0.1
        stages[:, 0, DRO_SHADOW_Q_NAMES.index("FFJ1")] = 0.2
        stages[:, 1, DRO_SHADOW_Q_NAMES.index("FFJ1")] = 0.4
        stages[:, 2, DRO_SHADOW_Q_NAMES.index("FFJ1")] = 0.6
        stages[:, 1, DRO_SHADOW_Q_NAMES.index("THJ3")] = 0.5
        failures = []
        if record.object_id == "object_b":
            stages[3] = np.nan
            failures.append(
                {
                    "candidate_index": 3,
                    "candidate_seed": candidate_seeds[3],
                    "error_type": "RuntimeError",
                    "message": "synthetic failed scene",
                }
            )
        return {
            "released_initial_q": released,
            "initial_q": initial,
            "stage_q": stages,
            "timing_seconds": np.arange(count, dtype=np.float64) / 100.0,
            "initialization_metadata": metadata,
            "pre_network_rng_state_sha256": rng_digests,
            "failures": failures,
        }

    @staticmethod
    def _snapshot(root: Path) -> dict:
        result = {}
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            result[str(path.relative_to(root))] = (
                path.stat().st_size,
                path.stat().st_mtime_ns,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        return result

    @staticmethod
    def _workspace_file(repository: str, relative_path: str):
        for parent in Path(__file__).resolve().parents:
            candidate = parent / repository / relative_path
            if candidate.is_file():
                return candidate
        return None

    def test_index_prepare_three_exported_poses_and_preserve_input_root(self):
        before = self._snapshot(self.output_root)
        viewer = ViewerRun(self.output_root)
        self.assertEqual(
            viewer.completed_scene_ids, ("object_a/floating/scale013",)
        )
        self.assertEqual(len(viewer.failed_scenes), 1)
        self.assertEqual(
            viewer.failed_scenes[0].failure["candidate_failures"][0]["message"],
            "synthetic failed scene",
        )
        self.assertEqual(viewer.scene_ids_for_scale(0.133), viewer.completed_scene_ids)
        self.assertEqual(viewer.scene_ids_for_scale(0.2), ())

        hand = ShadowHandModel(self.urdf)
        prepared = viewer.prepare_selection(
            hand,
            viewer.completed_scene_ids[0],
            candidate_index=0,
            stage="grasp",
            mode="three_poses",
            pose_source="exported",
        )
        self.assertEqual(prepared.stage_names, ("pregrasp", "grasp", "squeeze"))
        self.assertEqual(set(prepared.hand_meshes), set(prepared.stage_names))
        self.assertTrue(all(mesh.source_count == 25 for mesh in prepared.hand_meshes.values()))
        palms = np.stack(list(prepared.palm_poses_wxyz.values()))
        np.testing.assert_allclose(
            palms, np.repeat(palms[1:2], 3, axis=0), rtol=0.0, atol=1e-6
        )
        np.testing.assert_allclose(
            prepared.object_mesh.vertices.min(axis=0), [0.2, -0.1, 0.3], atol=1e-7
        )
        np.testing.assert_allclose(
            prepared.object_mesh.vertices.max(axis=0),
            [0.333, 0.033, 0.433],
            atol=1e-7,
        )
        self.assertEqual(prepared.object_point_cloud_world.shape, (512, 3))
        self.assertEqual(prepared.diagnostics["clamp_count"], 1)
        self.assertGreater(prepared.diagnostics["clamp_max_abs_delta"], 0.0)
        self.assertEqual(before, self._snapshot(self.output_root))

    def test_raw_state_is_explicitly_diagnostic_and_single_stage(self):
        viewer = ViewerRun(self.output_root)
        prepared = viewer.prepare_selection(
            ShadowHandModel(self.urdf),
            viewer.completed_scene_ids[0],
            candidate_index=0,
            stage="grasp",
            mode="single_stage",
            pose_source="raw",
        )
        self.assertEqual(prepared.stage_names, ("grasp",))
        self.assertIn("not Bench artifact", prepared.diagnostics["pose_source_label"])
        self.assertEqual(prepared.diagnostics["clamp_count"], 1)

    def test_prepare_only_cli_loads_on_cpu_without_importing_viser(self):
        repo_root = Path(__file__).resolve().parents[2]
        before = self._snapshot(self.output_root)
        validation = subprocess.run(
            (
                sys.executable,
                "grasp_generation/scripts/validate_bimanbodex_dro_outputs.py",
                str(self.output_root),
            ),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(validation.returncode, 0, msg=validation.stderr)
        self.assertEqual(json.loads(validation.stdout)["status"], "valid")
        result = subprocess.run(
            (
                sys.executable,
                "grasp_generation/scripts/visualize_bimanbodex_dro.py",
                "--output-root",
                str(self.output_root),
                "--scene-root",
                str(self.scene_root),
                "--shadow-urdf",
                str(self.urdf),
                "--scene",
                "object_a/floating/scale013",
                "--candidate",
                "0",
                "--stage",
                "grasp",
                "--mode",
                "three_poses",
                "--prepare-only",
            ),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["scene_id"], "object_a/floating/scale013")
        self.assertEqual(summary["displayed_stages"], ["pregrasp", "grasp", "squeeze"])
        self.assertEqual(before, self._snapshot(self.output_root))

    def test_viser_server_ui_smoke_when_dependency_is_available(self):
        try:
            import viser
        except ImportError:
            self.skipTest("viser is not installed in this test environment")
        viewer = ViewerRun(self.output_root)
        hand = ShadowHandModel(self.urdf)
        server = viser.ViserServer(host="127.0.0.1", port=0)
        try:
            app = ViewerApp(
                server,
                viewer,
                hand,
                SimpleNamespace(
                    scale=None,
                    candidate=0,
                    stage="grasp",
                    mode="three_poses",
                    pose_source="exported",
                    show_point_cloud=True,
                ),
                viewer.completed_scene_ids[0],
            )
            self.assertEqual(app.scene.value, "object_a/floating/scale013")
            self.assertEqual(app.diagnostics.content.startswith("### Current selection"), True)
        finally:
            server.stop()

    def test_bounds_failed_scene_and_missing_artifact_errors_are_explicit(self):
        viewer = ViewerRun(self.output_root)
        hand = ShadowHandModel(self.urdf)
        with self.assertRaisesRegex(ValueError, "candidate index must be"):
            viewer.prepare_selection(
                hand, viewer.completed_scene_ids[0], candidate_index=20
            )
        with self.assertRaisesRegex(ValueError, "unknown stage"):
            viewer.prepare_selection(
                hand, viewer.completed_scene_ids[0], stage="contact"
            )
        with self.assertRaisesRegex(ValueError, "failed synthesis"):
            viewer.load_scene("object_b/floating/scale013")
        with self.assertRaisesRegex(ValueError, "synthetic failed scene"):
            _initial_scene(viewer, None, 0.2)

        raw_path = (
            self.output_root / "raw" / "object_a" / "floating" / "scale013.npy"
        )
        raw_path.unlink()
        with self.assertRaisesRegex(ValueError, "invalid raw artifact path"):
            ViewerRun(self.output_root)

    def test_malformed_joint_order_mesh_hash_and_nonfinite_arrays_are_rejected(self):
        raw_path = (
            self.output_root / "raw" / "object_a" / "floating" / "scale013.npy"
        )
        raw = np.load(raw_path, allow_pickle=True).item()
        raw["dro_q_names"] = list(reversed(raw["dro_q_names"]))
        np.save(raw_path, raw, allow_pickle=True)
        with self.assertRaisesRegex(ValueError, "DRO q order mismatch"):
            ViewerRun(self.output_root).load_scene("object_a/floating/scale013")

        raw["dro_q_names"] = list(DRO_SHADOW_Q_NAMES)
        raw["stage_q"][0, 0, 0] = np.nan
        np.save(raw_path, raw, allow_pickle=True)
        with self.assertRaisesRegex(ValueError, "stage_q contains non-finite"):
            ViewerRun(self.output_root).load_scene("object_a/floating/scale013")

        raw["stage_q"][0, 0, 0] = 0.05
        np.save(raw_path, raw, allow_pickle=True)
        manifest_path = self.output_root / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["scenes"][0]["mesh_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "mesh_sha256 mismatch"):
            ViewerRun(self.output_root).load_scene("object_a/floating/scale013")

        original_manifest = json.loads(
            (self.output_root / "run_manifest.json").read_text(encoding="utf-8")
        )
        original_manifest["scenes"][0]["mesh_sha256"] = raw["scene"]["mesh_sha256"]
        original_manifest["scenes"][0]["scale"] = 0.5
        manifest_path.write_text(json.dumps(original_manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "run manifest scale mismatch"):
            ViewerRun(self.output_root).load_scene("object_a/floating/scale013")

        original_manifest["scenes"][0]["scale"] = 0.133
        manifest_path.write_text(json.dumps(original_manifest), encoding="utf-8")
        raw["scene"]["object_pose_wxyz"][0] = 0.5
        np.save(raw_path, raw, allow_pickle=True)
        with self.assertRaisesRegex(ValueError, "raw scene object_pose_wxyz mismatch"):
            ViewerRun(self.output_root).load_scene("object_a/floating/scale013")

    def test_shadow_urdf_hash_and_resolved_config_are_strict(self):
        changed_urdf = self.root / "changed.urdf"
        changed_urdf.write_text(self._shadow_urdf() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "URDF hash"):
            ViewerRun(self.output_root, shadow_urdf=changed_urdf)

        resolved_path = self.output_root / "resolved_config.json"
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        resolved["candidate_count"] = 19
        resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match resolved_config"):
            ViewerRun(self.output_root)

    def test_release_shadow_urdf_builds_full_geometry_and_matches_palm_contract(self):
        release_urdf = self._workspace_file(
            "DRO-Grasp",
            "data/data_urdf/robot/shadowhand/shadow_hand_right_extended.urdf",
        )
        if release_urdf is None:
            self.skipTest("release Shadow URDF/assets are not installed")
        hand = ShadowHandModel(release_urdf)
        q = np.zeros(len(DRO_SHADOW_Q_NAMES), dtype=np.float32)
        transforms = hand.link_transforms(q)
        np.testing.assert_allclose(
            transforms["palm"],
            legacy_dro_q_to_object_palm_transform(q),
            atol=1e-12,
        )
        mesh = hand.mesh(q, np.eye(4))
        self.assertGreaterEqual(mesh.source_count, 20)
        self.assertGreater(len(mesh.vertices), 100000)
        self.assertGreater(len(mesh.faces), 50000)

    def test_release_dro_and_bench_models_overlay_within_half_millimetre(self):
        release_urdf = self._workspace_file(
            "DRO-Grasp",
            "data/data_urdf/robot/shadowhand/shadow_hand_right_extended.urdf",
        )
        bench_mjcf = self._workspace_file(
            "BimanDexGraspBench",
            "assets/hand/shadow/right_hand_v2.xml",
        )
        if release_urdf is None or bench_mjcf is None:
            self.skipTest("release DRO/Bench Shadow assets are not installed")

        dro = ShadowHandModel(release_urdf)
        bench = BenchShadowHandModel(bench_mjcf)
        q = np.zeros(len(DRO_SHADOW_Q_NAMES), dtype=np.float64)
        q[DRO_SHADOW_Q_NAMES.index("virtual_joint_x")] = 0.08
        q[DRO_SHADOW_Q_NAMES.index("virtual_joint_roll")] = 0.15
        q[DRO_SHADOW_Q_NAMES.index("WRJ2")] = -0.2
        q[DRO_SHADOW_Q_NAMES.index("WRJ1")] = 0.25
        q[DRO_SHADOW_Q_NAMES.index("FFJ3")] = 0.4
        q[DRO_SHADOW_Q_NAMES.index("LFJ5")] = 0.3
        q[DRO_SHADOW_Q_NAMES.index("THJ4")] = 0.5
        object_world = pose_wxyz_to_matrix(
            np.array([0.2, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0])
        )
        palm_world = dro.palm_world_transform(q, object_world)
        bench_joints, excess = map_dro_shadow_fingers(q)
        self.assertEqual(float(np.max(excess)), 0.0)
        diagnostics, frames = compare_dro_bench_links(
            dro,
            bench,
            q,
            bench_joints,
            object_world,
            palm_world,
        )
        self.assertLessEqual(diagnostics["max_link_position_error_m"], 0.0005)
        self.assertLessEqual(diagnostics["max_link_rotation_error_rad"], 0.003)
        self.assertEqual(len(frames), 1 + 22)

        bench_mesh = bench.mesh(bench_joints, palm_world)
        self.assertGreaterEqual(bench_mesh.source_count, 20)
        self.assertGreater(len(bench_mesh.vertices), 1000)
        full_dro_mesh = dro.mesh(q, object_world)
        forearm_mesh = dro.mesh(q, object_world, include_links={"forearm"})
        wrist_mesh = dro.mesh(q, object_world, include_links={"wrist"})
        self.assertGreater(forearm_mesh.source_count, 0)
        self.assertGreater(wrist_mesh.source_count, 0)
        self.assertLess(forearm_mesh.source_count, full_dro_mesh.source_count)
        self.assertLess(wrist_mesh.source_count, full_dro_mesh.source_count)


if __name__ == "__main__":
    unittest.main()
