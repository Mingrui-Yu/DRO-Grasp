"""Strict paired-ablation validation for DRO initialization modes."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .contracts import (
    DRO_SHADOW_Q_NAMES,
    RAW_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    load_scene_record,
)
from .runner import _validate_v2_initialization_raw, validate_run_outputs


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _paired_config_projection(config: dict) -> dict:
    projected = copy.deepcopy(config)
    projected["output_root"] = None
    initialization = projected.get("initialization")
    if isinstance(initialization, dict):
        initialization["mode"] = "<paired-mode>"
    return projected


def _raw_for_scene(output_root: Path, scene: dict) -> dict:
    key = "raw_artifact" if scene.get("status") == "completed" else "failed_raw_artifact"
    relative = scene.get(key)
    if not isinstance(relative, str):
        raise ValueError(f"scene has no {key}: {scene.get('scene_id')}")
    raw = np.load(output_root / relative, allow_pickle=True).item()
    if not isinstance(raw, dict) or raw.get("schema_version") != RAW_SCHEMA_VERSION:
        raise ValueError(f"paired validation requires v2 raw: {scene.get('scene_id')}")
    return raw


def validate_paired_outputs(released_root: Path, tabletop_root: Path) -> dict:
    """Verify that the two runs differ intentionally only in root orientation."""

    released_root = Path(released_root).resolve(strict=True)
    tabletop_root = Path(tabletop_root).resolve(strict=True)
    released_validation = validate_run_outputs(released_root)
    tabletop_validation = validate_run_outputs(tabletop_root)
    released_manifest = _load_json(released_root / "run_manifest.json")
    tabletop_manifest = _load_json(tabletop_root / "run_manifest.json")
    for label, manifest, expected_mode in (
        ("released", released_manifest, "released_random"),
        ("tabletop", tabletop_manifest, "tabletop_stratified"),
    ):
        if manifest.get("schema_version") != RUN_SCHEMA_VERSION:
            raise ValueError(f"{label} run must use {RUN_SCHEMA_VERSION}")
        if manifest.get("initialization_mode") != expected_mode:
            raise ValueError(f"{label} run initialization mode is not {expected_mode}")

    released_config = released_manifest.get("resolved_config")
    tabletop_config = tabletop_manifest.get("resolved_config")
    if not isinstance(released_config, dict) or not isinstance(tabletop_config, dict):
        raise ValueError("paired run is missing resolved_config")
    if _paired_config_projection(released_config) != _paired_config_projection(
        tabletop_config
    ):
        raise ValueError("paired resolved configs differ beyond mode and output_root")
    for key in (
        "source",
        "checkpoint_sha256",
        "shadow_urdf_sha256",
        "shadow_point_cloud_sha256",
        "source_scene_count",
        "source_scene_manifest_sha256",
        "source_scale_histogram",
        "scene_count",
        "candidate_count",
    ):
        if released_manifest.get(key) != tabletop_manifest.get(key):
            raise ValueError(f"paired run manifest mismatch: {key}")

    released_scenes = released_manifest.get("scenes")
    tabletop_scenes = tabletop_manifest.get("scenes")
    if not isinstance(released_scenes, list) or not isinstance(tabletop_scenes, list):
        raise ValueError("paired run scenes must be lists")
    if [scene.get("scene_id") for scene in released_scenes] != [
        scene.get("scene_id") for scene in tabletop_scenes
    ]:
        raise ValueError("paired scene ordering differs")

    candidate_count = released_config["candidate_count"]
    compared_candidates = 0
    changed_root_rotations = 0
    scene_root = Path(released_config["scene_root"])

    for released_scene, tabletop_scene in zip(released_scenes, tabletop_scenes):
        scene_id = released_scene["scene_id"]
        record = load_scene_record(scene_root / f"{scene_id}.npy", scene_root)
        released_raw = _raw_for_scene(released_root, released_scene)
        tabletop_raw = _raw_for_scene(tabletop_root, tabletop_scene)
        _validate_v2_initialization_raw(
            released_raw, record, released_config, candidate_count, require_all=True
        )
        _validate_v2_initialization_raw(
            tabletop_raw, record, tabletop_config, candidate_count, require_all=True
        )
        for key in (
            "candidate_seeds",
            "point_seed",
            "object_point_cloud_sha256",
            "pre_network_rng_state_sha256",
        ):
            if released_raw.get(key) != tabletop_raw.get(key):
                raise ValueError(f"paired raw mismatch for {scene_id}: {key}")
        if not np.array_equal(
            np.asarray(released_raw["object_point_cloud"]),
            np.asarray(tabletop_raw["object_point_cloud"]),
        ):
            raise ValueError(f"paired object point clouds differ for {scene_id}")

        released_base = np.asarray(released_raw["released_initial_q"])
        tabletop_base = np.asarray(tabletop_raw["released_initial_q"])
        released_effective = np.asarray(released_raw["initial_q"])
        tabletop_effective = np.asarray(tabletop_raw["initial_q"])
        expected_shape = (candidate_count, len(DRO_SHADOW_Q_NAMES))
        if any(
            value.shape != expected_shape
            for value in (
                released_base,
                tabletop_base,
                released_effective,
                tabletop_effective,
            )
        ):
            raise ValueError(f"paired initial q shape mismatch for {scene_id}")
        if not np.array_equal(released_base, tabletop_base):
            raise ValueError(f"released sampler q differs between modes for {scene_id}")
        if not np.array_equal(released_base, released_effective):
            raise ValueError(f"released_random effective q changed for {scene_id}")
        if not (
            np.array_equal(released_effective[:, :3], tabletop_effective[:, :3])
            and np.array_equal(released_effective[:, 6:], tabletop_effective[:, 6:])
        ):
            raise ValueError(f"paired non-root-rotation q differs for {scene_id}")
        changed_root_rotations += int(
            np.count_nonzero(
                np.any(released_effective[:, 3:6] != tabletop_effective[:, 3:6], axis=1)
            )
        )
        compared_candidates += candidate_count

    if changed_root_rotations != compared_candidates:
        raise ValueError(
            "tabletop_stratified did not replace every paired root rotation: "
            f"{changed_root_rotations}/{compared_candidates}"
        )
    return {
        "status": "valid_paired_ablation",
        "scene_count": len(released_scenes),
        "candidate_count_per_mode": compared_candidates,
        "changed_root_rotation_count": changed_root_rotations,
        "released_validation": released_validation,
        "tabletop_validation": tabletop_validation,
    }
