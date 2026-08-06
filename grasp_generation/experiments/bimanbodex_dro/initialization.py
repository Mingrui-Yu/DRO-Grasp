"""Deterministic tabletop-oriented initialization for the DRO DGN2k adapter."""

from __future__ import annotations

import copy
import math
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import (
    DRO_SHADOW_Q_NAMES,
    dro_q_to_object_palm_transform,
    euler_xyz_to_matrix,
    quaternion_wxyz_to_matrix,
    sha256_array,
)

INITIALIZATION_MODES = ("released_random", "tabletop_stratified")
PALM_APPROACH_AXIS_LOCAL = np.array([0.0, 1.0, 0.0], dtype=np.float64)
PALM_APPROACH_AXIS_SIGN = "object_to_palm"


def default_initialization_config() -> dict:
    """Return the approved 20-candidate tabletop proposal configuration."""

    return {
        "mode": "released_random",
        "palm_approach_axis_local": PALM_APPROACH_AXIS_LOCAL.tolist(),
        "palm_approach_axis_sign": PALM_APPROACH_AXIS_SIGN,
        "candidate_ordering": [
            {
                "family": "top_down",
                "count": 6,
                "elevation_range_deg": [60.0, 90.0],
            },
            {
                "family": "oblique",
                "count": 8,
                "elevation_range_deg": [20.0, 60.0],
            },
            {
                "family": "near_horizontal",
                "count": 6,
                "elevation_range_deg": [0.0, 20.0],
            },
        ],
        "elevation_stratification": "family_midpoints",
        "azimuth": {
            "range_deg": [0.0, 360.0],
            "stratification": "family_midpoints",
        },
        "roll": {
            "range_deg": [0.0, 360.0],
            "stratification": "global_permuted_midpoints",
            "stride": 7,
            "offset": 0,
        },
    }


def _finite_range(value, label: str, *, lower: float, upper: float) -> list[float]:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if (
        result.shape != (2,)
        or not np.isfinite(result).all()
        or result[0] < lower
        or result[1] > upper
        or result[0] >= result[1]
    ):
        raise ValueError(f"{label} must be an increasing range within [{lower},{upper}]")
    return result.tolist()


def resolve_initialization_config(value: Optional[dict], candidate_count: int) -> dict:
    """Validate proposal parameters and materialize exact candidate ordering."""

    if not isinstance(candidate_count, int) or isinstance(candidate_count, bool):
        raise ValueError("candidate_count must be an integer")
    if value is not None and not isinstance(value, dict):
        raise ValueError("initialization must be an object")
    config = default_initialization_config()
    if value is not None:
        supplied = copy.deepcopy(value)
        for key, item in supplied.items():
            if key in {"azimuth", "roll"} and isinstance(item, dict):
                config[key].update(item)
            else:
                config[key] = item
    mode = config.get("mode")
    if mode not in INITIALIZATION_MODES:
        raise ValueError(f"initialization.mode must be one of {INITIALIZATION_MODES}")

    axis = np.asarray(config.get("palm_approach_axis_local"), dtype=np.float64).reshape(-1)
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError("palm_approach_axis_local must be a finite 3-vector")
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0.0:
        raise ValueError("palm_approach_axis_local must be non-zero")
    axis = axis / axis_norm
    if not np.allclose(axis, PALM_APPROACH_AXIS_LOCAL, rtol=0.0, atol=1e-12):
        raise ValueError("the validated DRO/Bench palm approach axis must remain local +Y")
    if config.get("palm_approach_axis_sign") != PALM_APPROACH_AXIS_SIGN:
        raise ValueError(
            f"palm_approach_axis_sign must be {PALM_APPROACH_AXIS_SIGN!r}"
        )

    ordering = config.get("candidate_ordering")
    if not isinstance(ordering, list) or not ordering:
        raise ValueError("initialization.candidate_ordering must be a non-empty list")
    expected_families = ("top_down", "oblique", "near_horizontal")
    if tuple(item.get("family") for item in ordering if isinstance(item, dict)) != expected_families:
        raise ValueError(
            "candidate_ordering families must be top_down, oblique, near_horizontal"
        )
    normalized_ordering = []
    total_count = 0
    for item in ordering:
        if not isinstance(item, dict):
            raise ValueError("candidate_ordering entries must be objects")
        count = item.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("candidate_ordering count must be a positive integer")
        elevation_range = _finite_range(
            item.get("elevation_range_deg"),
            f"{item.get('family')} elevation_range_deg",
            lower=0.0,
            upper=90.0,
        )
        normalized_ordering.append(
            {
                "family": item["family"],
                "count": count,
                "elevation_range_deg": elevation_range,
            }
        )
        total_count += count
    if total_count != candidate_count:
        raise ValueError(
            f"candidate_ordering counts must sum to {candidate_count}, got {total_count}"
        )
    family_ranges = {
        item["family"]: item["elevation_range_deg"] for item in normalized_ordering
    }
    if not (
        family_ranges["near_horizontal"][1] <= family_ranges["oblique"][0]
        and family_ranges["oblique"][1] <= family_ranges["top_down"][0]
    ):
        raise ValueError("elevation family ranges must be ordered and non-overlapping")
    if config.get("elevation_stratification") != "family_midpoints":
        raise ValueError("elevation_stratification must be family_midpoints")

    azimuth = config.get("azimuth")
    if not isinstance(azimuth, dict):
        raise ValueError("initialization.azimuth must be an object")
    azimuth_range = _finite_range(
        azimuth.get("range_deg"), "azimuth.range_deg", lower=0.0, upper=360.0
    )
    if azimuth.get("stratification") != "family_midpoints":
        raise ValueError("azimuth.stratification must be family_midpoints")

    roll = config.get("roll")
    if not isinstance(roll, dict):
        raise ValueError("initialization.roll must be an object")
    roll_range = _finite_range(
        roll.get("range_deg"), "roll.range_deg", lower=0.0, upper=360.0
    )
    if roll.get("stratification") != "global_permuted_midpoints":
        raise ValueError("roll.stratification must be global_permuted_midpoints")
    stride = roll.get("stride")
    offset = roll.get("offset")
    if (
        not isinstance(stride, int)
        or isinstance(stride, bool)
        or math.gcd(stride, candidate_count) != 1
    ):
        raise ValueError("roll.stride must be an integer coprime with candidate_count")
    if not isinstance(offset, int) or isinstance(offset, bool):
        raise ValueError("roll.offset must be an integer")

    sequence = []
    candidate_index = 0
    for family in normalized_ordering:
        count = family["count"]
        elevation_low, elevation_high = family["elevation_range_deg"]
        azimuth_low, azimuth_high = azimuth_range
        for family_index in range(count):
            elevation = elevation_low + (family_index + 0.5) * (
                elevation_high - elevation_low
            ) / count
            azimuth_value = azimuth_low + (family_index + 0.5) * (
                azimuth_high - azimuth_low
            ) / count
            roll_stratum = (candidate_index * stride + offset) % candidate_count
            roll_value = roll_range[0] + (roll_stratum + 0.5) * (
                roll_range[1] - roll_range[0]
            ) / candidate_count
            sequence.append(
                {
                    "candidate_index": candidate_index,
                    "family": family["family"],
                    "family_index": family_index,
                    "family_count": count,
                    "elevation_deg": elevation,
                    "azimuth_deg": azimuth_value,
                    "roll_deg": roll_value,
                }
            )
            candidate_index += 1

    return {
        "mode": mode,
        "palm_approach_axis_local": axis.tolist(),
        "palm_approach_axis_sign": PALM_APPROACH_AXIS_SIGN,
        "candidate_ordering": normalized_ordering,
        "elevation_stratification": "family_midpoints",
        "azimuth": {
            "range_deg": azimuth_range,
            "stratification": "family_midpoints",
        },
        "roll": {
            "range_deg": roll_range,
            "stratification": "global_permuted_midpoints",
            "stride": stride,
            "offset": offset,
        },
        "candidate_sequence": sequence,
    }


def _table_tangent_basis(normal_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(normal_world, dtype=np.float64).reshape(3)
    normal /= np.linalg.norm(normal)
    seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(seed, normal))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    tangent_u = seed - np.dot(seed, normal) * normal
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(normal, tangent_u)
    tangent_v /= np.linalg.norm(tangent_v)
    return tangent_u, tangent_v


def _proposal_direction_world(record, proposal: dict) -> np.ndarray:
    normal = np.asarray(record.table_normal_world, dtype=np.float64)
    tangent_u, tangent_v = _table_tangent_basis(normal)
    elevation = np.deg2rad(float(proposal["elevation_deg"]))
    azimuth = np.deg2rad(float(proposal["azimuth_deg"]))
    horizontal = np.cos(azimuth) * tangent_u + np.sin(azimuth) * tangent_v
    direction = np.cos(elevation) * horizontal + np.sin(elevation) * normal
    direction /= np.linalg.norm(direction)
    if float(np.dot(direction, normal)) < -1e-12:
        raise ValueError("tabletop proposal direction left the table upper hemisphere")
    return direction


def _target_palm_rotation_object(
    direction_object: np.ndarray,
    table_normal_object: np.ndarray,
    roll_deg: float,
) -> np.ndarray:
    approach = np.asarray(direction_object, dtype=np.float64).reshape(3)
    approach /= np.linalg.norm(approach)
    table_normal = np.asarray(table_normal_object, dtype=np.float64).reshape(3)
    table_normal /= np.linalg.norm(table_normal)
    local_x_object = np.cross(approach, table_normal)
    if np.linalg.norm(local_x_object) <= 1e-10:
        raise ValueError("tabletop proposal is degenerate with the table normal")
    local_x_object /= np.linalg.norm(local_x_object)
    local_z_object = np.cross(local_x_object, approach)
    local_z_object /= np.linalg.norm(local_z_object)
    base = np.column_stack((local_x_object, approach, local_z_object))
    roll = euler_xyz_to_matrix(np.array([0.0, np.deg2rad(roll_deg), 0.0]))
    target = base @ roll
    if np.linalg.det(target) < 0.999999:
        raise ValueError("tabletop proposal did not produce a proper palm rotation")
    return target


def apply_initialization(
    released_initial_q: np.ndarray,
    record,
    candidate_index: int,
    resolved_initialization: dict,
) -> tuple[np.ndarray, dict]:
    """Preserve released q except for the configured root-orientation override."""

    released = np.asarray(released_initial_q)
    if released.shape != (len(DRO_SHADOW_Q_NAMES),) or not np.isfinite(released).all():
        raise ValueError(
            f"released initial q must be finite shape {(len(DRO_SHADOW_Q_NAMES),)}"
        )
    sequence = resolved_initialization.get("candidate_sequence")
    if not isinstance(sequence, list) or not 0 <= candidate_index < len(sequence):
        raise ValueError(f"candidate_index is outside resolved proposal sequence: {candidate_index}")
    mode = resolved_initialization.get("mode")
    if mode not in INITIALIZATION_MODES:
        raise ValueError(f"unsupported initialization mode: {mode!r}")

    effective = released.copy()
    proposal = sequence[candidate_index]
    if proposal.get("candidate_index") != candidate_index:
        raise ValueError("resolved proposal sequence index mismatch")
    desired_direction_world = None
    if mode == "tabletop_stratified":
        desired_direction_world = _proposal_direction_world(record, proposal)
        object_rotation_world = quaternion_wxyz_to_matrix(record.object_pose_wxyz[3:])
        direction_object = object_rotation_world.T @ desired_direction_world
        table_normal_object = object_rotation_world.T @ record.table_normal_world
        target_palm_object = _target_palm_rotation_object(
            direction_object, table_normal_object, proposal["roll_deg"]
        )
        wrist_rotation = euler_xyz_to_matrix(
            np.array([0.0, float(released[6]), 0.0])
        ) @ euler_xyz_to_matrix(np.array([float(released[7]), 0.0, 0.0]))
        target_root_object = target_palm_object @ wrist_rotation.T
        effective[3:6] = Rotation.from_matrix(target_root_object).as_euler("XYZ")

    object_rotation_world = quaternion_wxyz_to_matrix(record.object_pose_wxyz[3:])
    palm_rotation_object = dro_q_to_object_palm_transform(effective)[:3, :3]
    actual_direction_world = (
        object_rotation_world @ palm_rotation_object @ PALM_APPROACH_AXIS_LOCAL
    )
    actual_direction_world /= np.linalg.norm(actual_direction_world)
    table_dot = float(np.dot(actual_direction_world, record.table_normal_world))
    if mode == "tabletop_stratified" and table_dot < -1e-6:
        raise ValueError("effective palm approach axis is outside the table upper hemisphere")
    alignment_error = None
    if desired_direction_world is not None:
        alignment_error = float(
            np.arccos(
                np.clip(
                    np.dot(actual_direction_world, desired_direction_world), -1.0, 1.0
                )
            )
        )
        if alignment_error > 1e-5:
            raise ValueError(
                f"effective palm approach axis alignment error is {alignment_error:.9g} rad"
            )

    metadata = {
        "candidate_index": candidate_index,
        "mode": mode,
        "proposal_family": (
            proposal["family"] if mode == "tabletop_stratified" else "released_random"
        ),
        "family_index": proposal["family_index"] if mode == "tabletop_stratified" else None,
        "family_count": proposal["family_count"] if mode == "tabletop_stratified" else None,
        "elevation_deg": proposal["elevation_deg"] if mode == "tabletop_stratified" else None,
        "azimuth_deg": proposal["azimuth_deg"] if mode == "tabletop_stratified" else None,
        "roll_deg": proposal["roll_deg"] if mode == "tabletop_stratified" else None,
        "palm_approach_axis_local": PALM_APPROACH_AXIS_LOCAL.tolist(),
        "palm_approach_axis_sign": PALM_APPROACH_AXIS_SIGN,
        "approach_direction_world": actual_direction_world.tolist(),
        "table_normal_world": np.asarray(record.table_normal_world).tolist(),
        "upper_hemisphere_dot": table_dot,
        "target_alignment_error_rad": alignment_error,
    }
    return effective, metadata


def torch_rng_state_digests(torch_module, device=None) -> dict:
    """Hash CPU and applicable CUDA Torch RNG states without consuming either."""

    result = {
        "cpu": sha256_array(torch_module.get_rng_state().detach().cpu().numpy())
    }
    if (
        device is not None
        and getattr(device, "type", None) == "cuda"
        and torch_module.cuda.is_available()
    ):
        result["cuda"] = sha256_array(
            torch_module.cuda.get_rng_state(device).detach().cpu().numpy()
        )
    return result
