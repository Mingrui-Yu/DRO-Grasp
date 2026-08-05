#!/usr/bin/env python3
"""Numerically compare the released DRO Shadow URDF with the Bench MJCF."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

GRASP_GENERATION_ROOT = Path(__file__).resolve().parents[1]
if str(GRASP_GENERATION_ROOT) not in sys.path:
    sys.path.insert(0, str(GRASP_GENERATION_ROOT))

from experiments.bimanbodex_dro.contracts import (  # noqa: E402
    BENCH_FROM_DRO,
    BENCH_SHADOW_JOINT_NAMES,
    DRO_SHADOW_Q_NAMES,
    dro_q_to_object_palm_transform,
    map_dro_shadow_fingers,
)


LINK_PAIRS = {
    "ffknuckle": "rh_ffknuckle",
    "ffproximal": "rh_ffproximal",
    "ffmiddle": "rh_ffmiddle",
    "ffdistal": "rh_ffdistal",
    "mfknuckle": "rh_mfknuckle",
    "mfproximal": "rh_mfproximal",
    "mfmiddle": "rh_mfmiddle",
    "mfdistal": "rh_mfdistal",
    "rfknuckle": "rh_rfknuckle",
    "rfproximal": "rh_rfproximal",
    "rfmiddle": "rh_rfmiddle",
    "rfdistal": "rh_rfdistal",
    "lfmetacarpal": "rh_lfmetacarpal",
    "lfknuckle": "rh_lfknuckle",
    "lfproximal": "rh_lfproximal",
    "lfmiddle": "rh_lfmiddle",
    "lfdistal": "rh_lfdistal",
    "thbase": "rh_thbase",
    "thproximal": "rh_thproximal",
    "thhub": "rh_thhub",
    "thmiddle": "rh_thmiddle",
    "thdistal": "rh_thdistal",
}


def _vector(value: Optional[str], default=(0.0, 0.0, 0.0)) -> np.ndarray:
    return np.fromstring(value, sep=" ", dtype=np.float64) if value else np.asarray(default, dtype=np.float64)


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _origin_transform(origin) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if origin is None:
        return transform
    roll, pitch, yaw = _vector(origin.get("rpy"))
    transform[:3, :3] = _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)
    transform[:3, 3] = _vector(origin.get("xyz"))
    return transform


def load_urdf(urdf_path: Path):
    """Parse the subset of URDF kinematics required by this validator."""

    root = ET.parse(urdf_path).getroot()
    joints = []
    limits = {}
    child_links = set()
    parent_links = set()
    for element in root.findall("joint"):
        parent = element.find("parent").get("link")
        child = element.find("child").get("link")
        joint_type = element.get("type")
        name = element.get("name")
        axis_element = element.find("axis")
        axis = _vector(axis_element.get("xyz") if axis_element is not None else None, (1, 0, 0))
        joints.append(
            {
                "name": name,
                "type": joint_type,
                "parent": parent,
                "child": child,
                "origin": _origin_transform(element.find("origin")),
                "axis": axis,
            }
        )
        child_links.add(child)
        parent_links.add(parent)
        limit = element.find("limit")
        if joint_type != "fixed":
            limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
    roots = sorted(parent_links - child_links)
    if roots != ["world"]:
        raise ValueError(f"unexpected URDF roots: {roots}")
    by_parent = {}
    for joint in joints:
        by_parent.setdefault(joint["parent"], []).append(joint)
    return by_parent, limits


def urdf_forward_kinematics(by_parent: dict, q_by_name: dict) -> dict:
    transforms = {"world": np.eye(4, dtype=np.float64)}
    pending = ["world"]
    while pending:
        parent = pending.pop()
        for joint in by_parent.get(parent, []):
            motion = np.eye(4, dtype=np.float64)
            value = q_by_name.get(joint["name"], 0.0)
            if joint["type"] == "revolute":
                motion[:3, :3] = _axis_rotation(joint["axis"], value)
            elif joint["type"] == "prismatic":
                motion[:3, 3] = joint["axis"] * value
            elif joint["type"] != "fixed":
                raise ValueError(f"unsupported URDF joint type: {joint['type']}")
            transforms[joint["child"]] = transforms[parent] @ joint["origin"] @ motion
            pending.append(joint["child"])
    return transforms


def _mujoco_body_transform(model, data, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"Bench body is missing: {name}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = data.xmat[body_id].reshape(3, 3)
    transform[:3, 3] = data.xpos[body_id]
    return transform


def _rotation_error(left: np.ndarray, right: np.ndarray) -> float:
    delta = left[:3, :3].T @ right[:3, :3]
    cosine = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def validate(urdf_path: Path, bench_mjcf: Path, samples: int, seed: int) -> dict:
    by_parent, urdf_limits = load_urdf(urdf_path)
    if tuple(urdf_limits) != DRO_SHADOW_Q_NAMES:
        raise ValueError(f"DRO q order mismatch: {tuple(urdf_limits)}")

    model = mujoco.MjModel.from_xml_path(str(bench_mjcf))
    data = mujoco.MjData(model)
    bench_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    )
    expected_bench_set = set(BENCH_SHADOW_JOINT_NAMES)
    if set(bench_names) != expected_bench_set or model.nq != len(BENCH_SHADOW_JOINT_NAMES):
        raise ValueError(f"Bench joint set mismatch: {bench_names}")

    rng = np.random.default_rng(seed)
    zero_q = {name: 0.0 for name in DRO_SHADOW_Q_NAMES}
    zero_urdf = urdf_forward_kinematics(by_parent, zero_q)
    mujoco.mj_forward(model, data)
    zero_urdf_palm_inverse = np.linalg.inv(zero_urdf["palm"])
    zero_bench_palm_inverse = np.linalg.inv(
        _mujoco_body_transform(model, data, "rh_palm")
    )
    frame_alignments = {}
    max_fixed_frame_offset = 0.0
    for urdf_link, bench_body in LINK_PAIRS.items():
        urdf_relative = zero_urdf_palm_inverse @ zero_urdf[urdf_link]
        bench_relative = zero_bench_palm_inverse @ _mujoco_body_transform(
            model, data, bench_body
        )
        alignment = urdf_relative[:3, :3].T @ bench_relative[:3, :3]
        frame_alignments[urdf_link] = alignment
        alignment_transform = np.eye(4, dtype=np.float64)
        alignment_transform[:3, :3] = alignment
        max_fixed_frame_offset = max(
            max_fixed_frame_offset,
            _rotation_error(np.eye(4, dtype=np.float64), alignment_transform),
        )
    max_root_position_error = 0.0
    max_root_rotation_error = 0.0
    max_link_position_error = 0.0
    max_link_rotation_error = 0.0
    max_joint_range_excess = 0.0
    worst_link = None

    for _ in range(samples):
        q = np.zeros(len(DRO_SHADOW_Q_NAMES), dtype=np.float64)
        for index, name in enumerate(DRO_SHADOW_Q_NAMES):
            lower, upper = urdf_limits[name]
            if name.startswith("virtual_joint_"):
                lower, upper = (-0.2, 0.2) if name in DRO_SHADOW_Q_NAMES[:3] else (-0.8, 0.8)
            q[index] = rng.uniform(lower, upper)
        q_by_name = dict(zip(DRO_SHADOW_Q_NAMES, q))
        urdf_status = urdf_forward_kinematics(by_parent, q_by_name)
        contract_palm = dro_q_to_object_palm_transform(q)
        urdf_palm = urdf_status["palm"]
        max_root_position_error = max(
            max_root_position_error,
            float(np.linalg.norm(contract_palm[:3, 3] - urdf_palm[:3, 3])),
        )
        max_root_rotation_error = max(
            max_root_rotation_error, _rotation_error(contract_palm, urdf_palm)
        )

        mapped, excess = map_dro_shadow_fingers(q)
        max_joint_range_excess = max(max_joint_range_excess, float(np.max(excess)))
        data.qpos[:] = 0.0
        for bench_name, value in zip(BENCH_SHADOW_JOINT_NAMES, mapped):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, bench_name)
            qpos_address = model.jnt_qposadr[joint_id]
            data.qpos[qpos_address] = value
            lower, upper = model.jnt_range[joint_id]
            max_joint_range_excess = max(
                max_joint_range_excess, max(lower - value, value - upper, 0.0)
            )
        mujoco.mj_forward(model, data)
        bench_palm = _mujoco_body_transform(model, data, "rh_palm")
        inverse_urdf_palm = np.linalg.inv(urdf_palm)
        inverse_bench_palm = np.linalg.inv(bench_palm)
        for urdf_link, bench_body in LINK_PAIRS.items():
            urdf_relative = inverse_urdf_palm @ urdf_status[urdf_link]
            bench_relative = inverse_bench_palm @ _mujoco_body_transform(model, data, bench_body)
            position_error = float(
                np.linalg.norm(urdf_relative[:3, 3] - bench_relative[:3, 3])
            )
            urdf_aligned = urdf_relative.copy()
            urdf_aligned[:3, :3] = (
                urdf_relative[:3, :3] @ frame_alignments[urdf_link]
            )
            rotation_error = _rotation_error(urdf_aligned, bench_relative)
            if position_error > max_link_position_error or rotation_error > max_link_rotation_error:
                worst_link = {"urdf_link": urdf_link, "bench_body": bench_body}
            max_link_position_error = max(max_link_position_error, position_error)
            max_link_rotation_error = max(max_link_rotation_error, rotation_error)

    return {
        "samples": samples,
        "seed": seed,
        "dro_q_names": list(DRO_SHADOW_Q_NAMES),
        "bench_joint_names": list(BENCH_SHADOW_JOINT_NAMES),
        "mapping": BENCH_FROM_DRO,
        "max_root_position_error_m": max_root_position_error,
        "max_root_rotation_error_rad": max_root_rotation_error,
        "max_link_position_error_m": max_link_position_error,
        "max_link_rotation_error_rad": max_link_rotation_error,
        "max_fixed_link_frame_offset_rad": max_fixed_frame_offset,
        "max_joint_range_excess_rad": max_joint_range_excess,
        "worst_link": worst_link,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dro-urdf", type=Path, required=True)
    parser.add_argument("--bench-mjcf", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=240826)
    parser.add_argument("--max-position-error", type=float, default=0.002)
    parser.add_argument("--max-rotation-error", type=float, default=0.003)
    args = parser.parse_args()
    result = validate(args.dro_urdf, args.bench_mjcf, args.samples, args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["max_root_position_error_m"] > 1e-9:
        raise SystemExit("DRO root-to-palm contract does not match the release URDF")
    if result["max_root_rotation_error_rad"] > 1e-7:
        raise SystemExit("DRO root-to-palm rotation does not match the release URDF")
    if result["max_link_position_error_m"] > args.max_position_error:
        raise SystemExit("DRO/Bench link position mismatch exceeds tolerance")
    if result["max_link_rotation_error_rad"] > args.max_rotation_error:
        raise SystemExit("DRO/Bench link rotation mismatch exceeds tolerance")
    if result["max_joint_range_excess_rad"] > 1e-6:
        raise SystemExit("DRO mapping exceeds Bench joint limits")


if __name__ == "__main__":
    main()
