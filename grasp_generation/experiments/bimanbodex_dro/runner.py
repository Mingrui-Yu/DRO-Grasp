"""Deterministic DRO inference orchestration and isolated artifact writing."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .contracts import (
    DRO_SHADOW_Q_NAMES,
    RAW_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    STAGE_NAMES,
    discover_scene_paths,
    load_scene_record,
    make_bench_artifact,
    sample_scaled_surface,
    scene_manifest_sha256,
    scene_scale_histogram,
    sha256_array,
    sha256_file,
    validate_artifact,
)


def utc_now() -> str:
    """Return an ISO 8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def git_state(repo_root: Path) -> dict:
    """Return the exact source state used by a run."""

    def command(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=str(repo_root), text=True
        ).strip()

    status = command("status", "--porcelain")
    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "dirty_paths": status.splitlines(),
    }


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_numpy(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, value, allow_pickle=True)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _stable_seed(base_seed: int, namespace: str) -> int:
    digest = sha256_array(np.frombuffer(namespace.encode("utf-8"), dtype=np.uint8))
    return (int(base_seed) + int(digest[:8], 16)) % (2**31 - 1)


def _resolve_path(repo_root: Path, value, *, required: bool = True):
    if value is None:
        if required:
            raise ValueError("required path is missing")
        return None
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve(strict=required)


def resolve_config(repo_root: Path, config: dict) -> dict:
    """Validate experiment controls without importing CUDA/network code."""

    resolved = dict(config)
    resolved["scene_root"] = str(_resolve_path(repo_root, config.get("scene_root")))
    resolved["checkpoint"] = str(_resolve_path(repo_root, config.get("checkpoint")))
    resolved["shadow_urdf"] = str(_resolve_path(repo_root, config.get("shadow_urdf")))
    resolved["shadow_point_cloud"] = str(
        _resolve_path(repo_root, config.get("shadow_point_cloud"))
    )
    resolved["reference_grasp_roots"] = [
        str(_resolve_path(repo_root, value))
        for value in config.get("reference_grasp_roots", [])
    ]
    scene_list = _resolve_path(repo_root, config.get("scene_list"), required=False)
    resolved["scene_list"] = str(scene_list) if scene_list is not None else None

    output_root = config.get("output_root")
    if output_root is not None:
        output_path = Path(output_root)
        if not output_path.is_absolute():
            output_path = repo_root / output_path
        resolved["output_root"] = str(output_path.resolve())

    expected_scene_count = config.get("expected_scene_count")
    if expected_scene_count is not None:
        if (
            not isinstance(expected_scene_count, int)
            or isinstance(expected_scene_count, bool)
            or expected_scene_count <= 0
        ):
            raise ValueError("expected_scene_count must be null or a positive integer")
    expected_manifest = config.get("expected_scene_manifest_sha256")
    if expected_manifest is not None:
        if not isinstance(expected_manifest, str) or len(expected_manifest) != 64:
            raise ValueError("expected_scene_manifest_sha256 must be a SHA256 hex string")
        try:
            int(expected_manifest, 16)
        except ValueError as error:
            raise ValueError("expected_scene_manifest_sha256 must be hexadecimal") from error
    expected_scale_range = config.get("expected_scale_range")
    if expected_scale_range is not None:
        expected_scale_range = np.asarray(expected_scale_range, dtype=np.float64).reshape(-1)
        if (
            expected_scale_range.shape != (2,)
            or not np.isfinite(expected_scale_range).all()
            or expected_scale_range[0] <= 0.0
            or expected_scale_range[0] > expected_scale_range[1]
        ):
            raise ValueError("expected_scale_range must be [positive_min, max]")
        resolved["expected_scale_range"] = expected_scale_range.tolist()

    integer_controls = ("candidate_count", "point_count", "point_seed", "inference_seed", "n_iter")
    for name in integer_controls:
        value = resolved.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    if resolved["candidate_count"] != 20:
        raise ValueError("candidate_count must remain the approved 20 candidates per scene")
    if resolved["point_count"] != 512:
        raise ValueError("point_count must remain the official 512-point contract")
    if resolved["n_iter"] <= 0:
        raise ValueError("n_iter must be positive")
    if resolved.get("robot_name") != "shadowhand":
        raise ValueError("this adapter is scoped to robot_name=shadowhand")
    if Path(resolved["checkpoint"]).name != "model_3robots.pth":
        raise ValueError("checkpoint must be the complete-point-cloud model_3robots.pth")
    if resolved.get("point_cloud_mode") != "complete_scaled_mesh":
        raise ValueError("point_cloud_mode must be complete_scaled_mesh")
    if resolved.get("frame_contract") != "object_local_no_normalization":
        raise ValueError("frame_contract must be object_local_no_normalization")
    if config.get("network_center_robot_pc", True) is not True:
        raise ValueError("the released checkpoint requires official robot-point-cloud centering")
    resolved["network_center_robot_pc"] = True
    if resolved.get("failure_policy") != "scene_atomic_all_candidates_fail":
        raise ValueError("unsupported failure_policy")
    device = resolved.get("device")
    if not isinstance(device, str) or not device.startswith("cuda:"):
        raise ValueError("official inference device must be an explicit cuda:<index>")
    return resolved


def resolve_scene_records(resolved_config: dict) -> tuple[list, list]:
    """Load and validate the authoritative scene set, then select a bounded subset."""

    scene_paths = discover_scene_paths(
        Path(resolved_config["scene_root"]),
        resolved_config["reference_grasp_roots"],
        resolved_config["scene_list"],
    )
    source_records = [
        load_scene_record(path, Path(resolved_config["scene_root"]))
        for path in scene_paths
    ]
    expected_count = resolved_config.get("expected_scene_count")
    if expected_count is not None and len(source_records) != expected_count:
        raise ValueError(
            "authoritative scene count mismatch: "
            f"expected {expected_count}, got {len(source_records)}"
        )
    expected_manifest = resolved_config.get("expected_scene_manifest_sha256")
    actual_manifest = scene_manifest_sha256(source_records)
    if expected_manifest is not None and actual_manifest != expected_manifest:
        raise ValueError(
            "authoritative scene manifest mismatch: "
            f"expected {expected_manifest}, got {actual_manifest}"
        )
    expected_scale_range = resolved_config.get("expected_scale_range")
    if expected_scale_range is not None:
        observed_range = (
            min(record.scale for record in source_records),
            max(record.scale for record in source_records),
        )
        expected_range = tuple(expected_scale_range)
        if not np.allclose(observed_range, expected_range, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"authoritative scale range mismatch: expected {expected_range}, got {observed_range}"
            )

    selected_records = source_records
    max_scenes = resolved_config.get("max_scenes")
    if max_scenes is not None:
        if not isinstance(max_scenes, int) or isinstance(max_scenes, bool) or max_scenes <= 0:
            raise ValueError("max_scenes must be null or a positive integer")
        selected_records = source_records[:max_scenes]
    return source_records, selected_records


def resolve_scenes(resolved_config: dict) -> list:
    """Load the exact reference scene set and optional bounded subset."""

    _, selected_records = resolve_scene_records(resolved_config)
    return selected_records


def dry_run(repo_root: Path, config: dict) -> dict:
    """Resolve assets/scenes and one representative point cloud without writing."""

    resolved = resolve_config(repo_root, config)
    source_records, records = resolve_scene_records(resolved)
    first = records[0]
    point_seed = _stable_seed(resolved["point_seed"], first.scene_id)
    points = sample_scaled_surface(
        first.mesh_path, first.scale, resolved["point_count"], point_seed
    )
    return {
        "status": "dry_run",
        "scene_count": len(records),
        "source_scene_count": len(source_records),
        "source_scene_manifest_sha256": scene_manifest_sha256(source_records),
        "source_scale_histogram": scene_scale_histogram(source_records),
        "first_scene": first.to_manifest(),
        "first_scene_point_seed": point_seed,
        "first_scene_point_cloud_shape": list(points.shape),
        "first_scene_point_cloud_sha256": sha256_array(points),
        "checkpoint_sha256": sha256_file(Path(resolved["checkpoint"])),
        "shadow_urdf_sha256": sha256_file(Path(resolved["shadow_urdf"])),
        "shadow_point_cloud_sha256": sha256_file(Path(resolved["shadow_point_cloud"])),
        "resolved_config": resolved,
    }


class OfficialDROInference:
    """Lazy wrapper around the released network, optimizer, and controller."""

    def __init__(self, repo_root: Path, resolved_config: dict):
        # Keep direct script execution independent of the caller's working directory.
        repo_root = Path(repo_root).resolve(strict=True)
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import torch

        from model.network import create_network
        from utils.hand_model import create_hand_model

        self.torch = torch
        self.config = resolved_config
        expected_checkpoint = repo_root / "ckpt/model/model_3robots.pth"
        expected_urdf = repo_root / "data/data_urdf/robot/shadowhand/shadow_hand_right_extended.urdf"
        expected_point_cloud = repo_root / "data/PointCloud/robot/shadowhand.pt"
        if Path(resolved_config["checkpoint"]) != expected_checkpoint.resolve():
            raise ValueError("checkpoint must be the released complete-point-cloud model")
        if Path(resolved_config["shadow_urdf"]) != expected_urdf.resolve():
            raise ValueError("shadow_urdf must be the released DRO Shadow URDF")
        if Path(resolved_config["shadow_point_cloud"]) != expected_point_cloud.resolve():
            raise ValueError("shadow_point_cloud must be the released DRO Shadow point cloud")
        self.device = torch.device(resolved_config["device"])
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; official DRO inference was not started")
        device_index = self.device.index if self.device.index is not None else 0
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"requested {self.device}, but only {torch.cuda.device_count()} CUDA devices are visible"
            )

        torch.manual_seed(resolved_config["inference_seed"])
        torch.cuda.manual_seed_all(resolved_config["inference_seed"])
        random.seed(resolved_config["inference_seed"])
        np.random.seed(resolved_config["inference_seed"])
        self.network = create_network(
            SimpleNamespace(
                emb_dim=512,
                latent_dim=64,
                pretrain=None,
                center_pc=self.config["network_center_robot_pc"],
                block_computing=True,
            ),
            mode="validate",
        ).to(self.device)
        state_dict = torch.load(
            resolved_config["checkpoint"], map_location=self.device, weights_only=True
        )
        self.network.load_state_dict(state_dict)
        self.network.eval()
        self.hand = create_hand_model("shadowhand", self.device)
        if tuple(self.hand.pk_chain.get_joint_parameter_names()) != DRO_SHADOW_Q_NAMES:
            raise ValueError("released Shadow URDF q order does not match the adapter contract")
        self.environment_manifest = {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(self.device),
            "device_name": torch.cuda.get_device_name(self.device),
            "candidate_batch_size": 1,
        }

    def __call__(
        self,
        record,
        object_points: np.ndarray,
        candidate_seeds: list[int],
    ) -> dict:
        """Generate candidates independently so failures keep exact denominator indices."""

        torch = self.torch
        from utils.controller import controller
        from utils.multilateration import multilateration
        from utils.optimization import create_problem, optimization, process_transform
        from utils.se3_transform import compute_link_pose

        candidate_count = len(candidate_seeds)
        initial_q = np.full((candidate_count, len(DRO_SHADOW_Q_NAMES)), np.nan, dtype=np.float32)
        stage_q = np.full((candidate_count, 3, len(DRO_SHADOW_Q_NAMES)), np.nan, dtype=np.float32)
        timings = np.full((candidate_count,), np.nan, dtype=np.float64)
        failures = []
        object_pc = torch.from_numpy(object_points).to(self.device).unsqueeze(0)

        for candidate_index, candidate_seed in enumerate(candidate_seeds):
            try:
                random.seed(candidate_seed)
                np.random.seed(candidate_seed)
                torch.manual_seed(candidate_seed)
                torch.cuda.manual_seed_all(candidate_seed)
                q_initial = self.hand.get_initial_q().unsqueeze(0).to(self.device)
                robot_pc = self.hand.get_transformed_links_pc(q_initial)[..., :3]
                torch.cuda.synchronize(self.device)
                started = time.perf_counter()
                with torch.no_grad():
                    dro = self.network(robot_pc, object_pc)["dro"].detach()
                mlat_pc = multilateration(dro, object_pc)
                transform, _ = compute_link_pose(self.hand.links_pc, mlat_pc, is_train=False)
                optim_transform = process_transform(self.hand.pk_chain, transform)
                layer = create_problem(self.hand.pk_chain, optim_transform.keys())
                q_grasp = optimization(
                    self.hand.pk_chain,
                    layer,
                    q_initial,
                    optim_transform,
                    n_iter=self.config["n_iter"],
                )
                q_outer, q_inner = controller("shadowhand", q_grasp)
                torch.cuda.synchronize(self.device)
                timings[candidate_index] = time.perf_counter() - started
                initial_q[candidate_index] = q_initial[0].detach().cpu().numpy()
                stage_q[candidate_index] = torch.stack(
                    (q_outer[0], q_grasp[0], q_inner[0]), dim=0
                ).detach().cpu().numpy()
            except Exception as error:  # Candidate failures are evidence, not silent drops.
                failures.append(
                    {
                        "candidate_index": candidate_index,
                        "candidate_seed": candidate_seed,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )
        return {
            "initial_q": initial_q,
            "stage_q": stage_q,
            "timing_seconds": timings,
            "failures": failures,
        }


def _artifact_paths(output_root: Path, scene_id: str) -> tuple[Path, Path, Path]:
    relative = Path(scene_id)
    stem = relative.name
    grasp_path = output_root / "graspdata" / relative.parent / f"{stem}_grasp.npy"
    raw_path = output_root / "raw" / relative.with_suffix(".npy")
    failed_raw_path = output_root / "failed_raw" / relative.with_suffix(".npy")
    return grasp_path, raw_path, failed_raw_path


def _write_scene_pair(grasp_path: Path, raw_path: Path, artifact: dict, raw: dict) -> None:
    try:
        _atomic_numpy(raw_path, raw)
        _atomic_numpy(grasp_path, artifact)
    except Exception:
        for path in (raw_path, grasp_path):
            if path.exists():
                path.unlink()
        raise


def _validate_inference_result(result: dict, candidate_count: int):
    """Validate candidate arrays and explicit failure indices from the inference layer."""

    if not isinstance(result, dict):
        raise ValueError("inference result must be a dictionary")
    stage_q = np.asarray(result["stage_q"], dtype=np.float32)
    initial_q = np.asarray(result["initial_q"], dtype=np.float32)
    timings = np.asarray(result["timing_seconds"], dtype=np.float64)
    candidate_failures = list(result.get("failures", []))
    failure_indices = []
    for failure in candidate_failures:
        if not isinstance(failure, dict) or not isinstance(
            failure.get("candidate_index"), int
        ):
            raise ValueError(
                "candidate failures must contain integer candidate_index values"
            )
        candidate_index = failure["candidate_index"]
        if candidate_index < 0 or candidate_index >= candidate_count:
            raise ValueError(f"candidate failure index out of range: {candidate_index}")
        if candidate_index in failure_indices:
            raise ValueError(f"duplicate candidate failure index: {candidate_index}")
        failure_indices.append(candidate_index)

    expected_q_shape = (candidate_count, len(DRO_SHADOW_Q_NAMES))
    expected_stage_shape = (
        candidate_count,
        len(STAGE_NAMES),
        len(DRO_SHADOW_Q_NAMES),
    )
    if (
        initial_q.shape != expected_q_shape
        or stage_q.shape != expected_stage_shape
        or timings.shape != (candidate_count,)
    ):
        raise ValueError(
            "inference result shape mismatch: "
            f"initial_q={initial_q.shape}, stage_q={stage_q.shape}, "
            f"timings={timings.shape}"
        )
    valid_indices = [
        index for index in range(candidate_count) if index not in failure_indices
    ]
    if valid_indices and (
        not np.isfinite(initial_q[valid_indices]).all()
        or not np.isfinite(stage_q[valid_indices]).all()
        or not np.isfinite(timings[valid_indices]).all()
    ):
        raise ValueError("successful candidate inference result contains non-finite values")
    return initial_q, stage_q, timings, candidate_failures, failure_indices


def run(repo_root: Path, config: dict, inference=None) -> dict:
    """Run isolated inference and write only scene-atomic valid Bench artifacts."""

    repo_root = Path(repo_root).resolve(strict=True)
    resolved = resolve_config(repo_root, config)
    source_records, records = resolve_scene_records(resolved)
    output_value = resolved.get("output_root")
    if not output_value:
        raise ValueError("output_root is required outside dry-run")
    output_root = Path(output_value)
    if output_root.exists():
        raise FileExistsError(f"output_root already exists: {output_root}")
    if inference is None:
        # Initialize before creating an output directory so a missing CUDA/runtime
        # dependency cannot leave a misleading partial run behind.
        inference = OfficialDROInference(repo_root, resolved)
    output_root.mkdir(parents=True)

    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "running",
        "started_at": utc_now(),
        "source": git_state(repo_root),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "checkpoint_sha256": sha256_file(Path(resolved["checkpoint"])),
        "shadow_urdf_sha256": sha256_file(Path(resolved["shadow_urdf"])),
        "shadow_point_cloud_sha256": sha256_file(Path(resolved["shadow_point_cloud"])),
        "resolved_config": resolved,
        "scene_count": len(records),
        "source_scene_count": len(source_records),
        "source_scene_manifest_sha256": scene_manifest_sha256(source_records),
        "source_scale_histogram": scene_scale_histogram(source_records),
        "candidate_count": len(records) * resolved["candidate_count"],
        "completed_candidate_count": 0,
        "failed_candidate_count": 0,
        "scenes": [],
    }
    failure_manifest = {"schema_version": RUN_SCHEMA_VERSION, "failures": []}
    _atomic_json(output_root / "resolved_config.json", resolved)
    _atomic_json(output_root / "run_manifest.json", manifest)
    _atomic_json(output_root / "failure_manifest.json", failure_manifest)

    manifest["inference_environment"] = getattr(
        inference, "environment_manifest", {"injected_test_inference": True}
    )
    _atomic_json(output_root / "run_manifest.json", manifest)

    try:
        for record in records:
            scene_point_seed = _stable_seed(resolved["point_seed"], record.scene_id)
            points = sample_scaled_surface(
                record.mesh_path,
                record.scale,
                resolved["point_count"],
                scene_point_seed,
            )
            candidate_seeds = [
                _stable_seed(resolved["inference_seed"], f"{record.scene_id}:{index}")
                for index in range(resolved["candidate_count"])
            ]
            grasp_path, raw_path, failed_raw_path = _artifact_paths(output_root, record.scene_id)
            scene_failure = None
            try:
                result = inference(record, points, candidate_seeds)
                (
                    initial_q,
                    stage_q,
                    timings,
                    candidate_failures,
                    failure_indices,
                ) = _validate_inference_result(
                    result, resolved["candidate_count"]
                )
            except Exception as error:
                initial_q = np.full(
                    (resolved["candidate_count"], len(DRO_SHADOW_Q_NAMES)),
                    np.nan,
                    dtype=np.float32,
                )
                stage_q = np.full(
                    (
                        resolved["candidate_count"],
                        len(STAGE_NAMES),
                        len(DRO_SHADOW_Q_NAMES),
                    ),
                    np.nan,
                    dtype=np.float32,
                )
                timings = np.full(
                    (resolved["candidate_count"],), np.nan, dtype=np.float64
                )
                candidate_failures = []
                failure_indices = list(range(resolved["candidate_count"]))
                scene_failure = {
                    "scene_id": record.scene_id,
                    "stage": "inference_driver",
                    "candidate_count": resolved["candidate_count"],
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "policy": resolved["failure_policy"],
                }
            raw = {
                "schema_version": RAW_SCHEMA_VERSION,
                "scene": record.to_manifest(),
                "stage_names": list(STAGE_NAMES),
                "dro_q_names": list(DRO_SHADOW_Q_NAMES),
                "candidate_seeds": candidate_seeds,
                "point_seed": scene_point_seed,
                "object_point_cloud": points,
                "object_point_cloud_sha256": sha256_array(points),
                "initial_q": initial_q,
                "stage_q": stage_q,
                "timing_seconds": timings,
                "failed_candidate_indices": failure_indices,
            }

            if scene_failure is None and candidate_failures:
                scene_failure = {
                    "scene_id": record.scene_id,
                    "stage": "inference",
                    "candidate_count": resolved["candidate_count"],
                    "candidate_failures": candidate_failures,
                    "policy": resolved["failure_policy"],
                }
            elif scene_failure is None:
                try:
                    export_started = time.perf_counter()
                    artifact, limit_excess = make_bench_artifact(
                        stage_q, record.object_pose_wxyz, record.stored_scene_path
                    )
                    validate_artifact(artifact, record, stage_q)
                    raw["bench_joint_limit_excess"] = limit_excess
                    raw["export_seconds"] = time.perf_counter() - export_started
                    _write_scene_pair(grasp_path, raw_path, artifact, raw)
                except Exception as error:
                    scene_failure = {
                        "scene_id": record.scene_id,
                        "stage": "export_validation",
                        "candidate_count": resolved["candidate_count"],
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "policy": resolved["failure_policy"],
                    }

            scene_record = record.to_manifest()
            scene_record.update(
                {
                    "point_seed": scene_point_seed,
                    "point_cloud_sha256": raw["object_point_cloud_sha256"],
                    "candidate_seeds": candidate_seeds,
                }
            )
            if scene_failure is None:
                manifest["completed_candidate_count"] += resolved["candidate_count"]
                scene_record["status"] = "completed"
                scene_record["grasp_artifact"] = str(grasp_path.relative_to(output_root))
                scene_record["raw_artifact"] = str(raw_path.relative_to(output_root))
            else:
                manifest["failed_candidate_count"] += resolved["candidate_count"]
                scene_record["status"] = "failed"
                failure_manifest["failures"].append(scene_failure)
                raw["scene_failure"] = scene_failure
                raw.setdefault("export_seconds", None)
                _atomic_numpy(failed_raw_path, raw)
                scene_record["failed_raw_artifact"] = str(failed_raw_path.relative_to(output_root))
            manifest["scenes"].append(scene_record)
            _atomic_json(output_root / "run_manifest.json", manifest)
            _atomic_json(output_root / "failure_manifest.json", failure_manifest)

        if manifest["candidate_count"] != (
            manifest["completed_candidate_count"] + manifest["failed_candidate_count"]
        ):
            raise RuntimeError("candidate denominator accounting is inconsistent")
        if manifest["completed_candidate_count"] == manifest["candidate_count"]:
            manifest["status"] = "completed"
        elif manifest["completed_candidate_count"] == 0:
            manifest["status"] = "failed"
        else:
            manifest["status"] = "completed_with_failures"
        manifest["completed_at"] = utc_now()
        manifest["output_bytes"] = sum(
            path.stat().st_size for path in output_root.rglob("*") if path.is_file()
        )
        _atomic_json(output_root / "run_manifest.json", manifest)
        return manifest
    except Exception:
        manifest["status"] = "failed"
        manifest["completed_at"] = utc_now()
        _atomic_json(output_root / "run_manifest.json", manifest)
        raise


def validate_run_outputs(output_root: Path) -> dict:
    """Independently revalidate persisted raw/artifact pairs and accounting."""

    output_root = Path(output_root).resolve(strict=True)
    manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
    failure_manifest = json.loads(
        (output_root / "failure_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("unsupported run manifest schema")
    if manifest["candidate_count"] != (
        manifest["completed_candidate_count"] + manifest["failed_candidate_count"]
    ):
        raise ValueError("candidate denominator accounting is inconsistent")
    if len(failure_manifest.get("failures", [])) != sum(
        scene.get("status") == "failed" for scene in manifest["scenes"]
    ):
        raise ValueError("failure manifest and scene statuses disagree")

    resolved = manifest["resolved_config"]
    source_records, selected_records = resolve_scene_records(resolved)
    if manifest.get("source_scene_count") != len(source_records):
        raise ValueError("persisted source scene count does not match current inputs")
    if manifest.get("source_scene_manifest_sha256") != scene_manifest_sha256(
        source_records
    ):
        raise ValueError("persisted source scene manifest does not match current inputs")
    if manifest.get("source_scale_histogram") != scene_scale_histogram(source_records):
        raise ValueError("persisted source scale histogram does not match current inputs")
    if manifest.get("scene_count") != len(selected_records):
        raise ValueError("persisted selected scene count does not match current inputs")
    if [scene["scene_id"] for scene in manifest["scenes"]] != [
        record.scene_id for record in selected_records
    ]:
        raise ValueError("persisted scene order does not match the resolved selection")

    scene_root = Path(resolved["scene_root"])
    candidate_count = resolved["candidate_count"]
    completed_scenes = 0
    failed_scenes = 0
    for scene in manifest["scenes"]:
        scene_path = scene_root / (scene["scene_id"] + ".npy")
        record = load_scene_record(scene_path, scene_root)
        if scene["status"] == "completed":
            raw_path = output_root / scene["raw_artifact"]
            grasp_path = output_root / scene["grasp_artifact"]
            raw = np.load(raw_path, allow_pickle=True).item()
            artifact = np.load(grasp_path, allow_pickle=True).item()
            if raw.get("schema_version") != RAW_SCHEMA_VERSION:
                raise ValueError(f"unsupported raw schema for {record.scene_id}")
            if raw.get("stage_names") != list(STAGE_NAMES):
                raise ValueError(f"stage order mismatch for {record.scene_id}")
            if raw.get("dro_q_names") != list(DRO_SHADOW_Q_NAMES):
                raise ValueError(f"DRO q order mismatch for {record.scene_id}")
            points = np.asarray(raw.get("object_point_cloud"))
            if points.shape != (512, 3) or points.dtype != np.float32:
                raise ValueError(f"point-cloud contract mismatch for {record.scene_id}")
            if sha256_array(points) != raw.get("object_point_cloud_sha256"):
                raise ValueError(f"point-cloud hash mismatch for {record.scene_id}")
            stage_q = np.asarray(raw.get("stage_q"))
            if stage_q.shape != (candidate_count, 3, len(DRO_SHADOW_Q_NAMES)):
                raise ValueError(f"stage q shape mismatch for {record.scene_id}")
            initial_q = np.asarray(raw.get("initial_q"))
            timings = np.asarray(raw.get("timing_seconds"))
            if initial_q.shape != (candidate_count, len(DRO_SHADOW_Q_NAMES)):
                raise ValueError(f"initial q shape mismatch for {record.scene_id}")
            if timings.shape != (candidate_count,) or not np.isfinite(timings).all():
                raise ValueError(f"timing contract mismatch for {record.scene_id}")
            if raw.get("failed_candidate_indices") != []:
                raise ValueError(f"completed scene records candidate failures: {record.scene_id}")
            export_seconds = raw.get("export_seconds")
            if not isinstance(export_seconds, float) or export_seconds < 0.0:
                raise ValueError(f"export timing contract mismatch for {record.scene_id}")
            validate_artifact(artifact, record, stage_q)
            completed_scenes += 1
        elif scene["status"] == "failed":
            failed_raw_path = output_root / scene["failed_raw_artifact"]
            raw = np.load(failed_raw_path, allow_pickle=True).item()
            if raw.get("schema_version") != RAW_SCHEMA_VERSION or "scene_failure" not in raw:
                raise ValueError(f"failed raw diagnostics are incomplete for {record.scene_id}")
            failed_scenes += 1
        else:
            raise ValueError(f"unknown scene status for {record.scene_id}: {scene['status']}")
    return {
        "status": "valid",
        "scene_count": len(manifest["scenes"]),
        "completed_scene_count": completed_scenes,
        "failed_scene_count": failed_scenes,
        "candidate_count": manifest["candidate_count"],
        "completed_candidate_count": manifest["completed_candidate_count"],
        "failed_candidate_count": manifest["failed_candidate_count"],
    }
