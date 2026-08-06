"""CPU-testable contracts for DRO-Grasp on BimanBODex DGN2k scenes."""

from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

LEGACY_RAW_SCHEMA_VERSION = "drograsp.dgn2k.raw.v1"
LEGACY_RUN_SCHEMA_VERSION = "drograsp.dgn2k.run.v1"
RAW_SCHEMA_VERSION = "drograsp.dgn2k.raw.v2"
RUN_SCHEMA_VERSION = "drograsp.dgn2k.run.v2"
SUPPORTED_RAW_SCHEMA_VERSIONS = (LEGACY_RAW_SCHEMA_VERSION, RAW_SCHEMA_VERSION)
SUPPORTED_RUN_SCHEMA_VERSIONS = (LEGACY_RUN_SCHEMA_VERSION, RUN_SCHEMA_VERSION)
STORED_SCENE_PREFIX = "src/curobo/content/assets/object/DGN_2k/scene_cfg"
STAGE_NAMES = ("pregrasp", "grasp", "squeeze")

# Exact pytorch_kinematics parameter order in the released extended Shadow URDF.
DRO_SHADOW_Q_NAMES = (
    "virtual_joint_x", "virtual_joint_y", "virtual_joint_z",
    "virtual_joint_roll", "virtual_joint_pitch", "virtual_joint_yaw",
    "WRJ2", "WRJ1",
    "FFJ4", "FFJ3", "FFJ2", "FFJ1",
    "MFJ4", "MFJ3", "MFJ2", "MFJ1",
    "RFJ4", "RFJ3", "RFJ2", "RFJ1",
    "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1",
    "THJ5", "THJ4", "THJ3", "THJ2", "THJ1",
)
DRO_SHADOW_FINGER_JOINT_NAMES = DRO_SHADOW_Q_NAMES[8:]
BENCH_SHADOW_JOINT_NAMES = tuple(
    f"rh_{name}" for name in (
        "THJ5", "THJ4", "THJ3", "THJ2", "THJ1",
        "FFJ4", "FFJ3", "FFJ2", "FFJ1",
        "MFJ4", "MFJ3", "MFJ2", "MFJ1",
        "RFJ4", "RFJ3", "RFJ2", "RFJ1",
        "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1",
    )
)

# The released DRO URDF and Bench MJCF use the same right-hand joint semantics.
# Export therefore only strips/adds the ``rh_`` namespace and reorders by name.
BENCH_FROM_DRO = {
    bench_name: bench_name[3:]
    for bench_name in BENCH_SHADOW_JOINT_NAMES
}
BENCH_SHADOW_JOINT_LIMITS = {
    "rh_THJ5": (-1.0472, 1.0472),
    "rh_THJ4": (0.0, 1.22173),
    "rh_THJ3": (-0.20944, 0.20944),
    "rh_THJ2": (-0.698132, 0.698132),
    "rh_THJ1": (-0.261799, 1.5708),
    "rh_FFJ4": (-0.349066, 0.349066),
    "rh_FFJ3": (-0.261799, 1.5708),
    "rh_FFJ2": (0.0, 1.5708),
    "rh_FFJ1": (0.0, 1.5708),
    "rh_MFJ4": (-0.349066, 0.349066),
    "rh_MFJ3": (-0.261799, 1.5708),
    "rh_MFJ2": (0.0, 1.5708),
    "rh_MFJ1": (0.0, 1.5708),
    "rh_RFJ4": (-0.349066, 0.349066),
    "rh_RFJ3": (-0.261799, 1.5708),
    "rh_RFJ2": (0.0, 1.5708),
    "rh_RFJ1": (0.0, 1.5708),
    "rh_LFJ5": (0.0, 0.785398),
    "rh_LFJ4": (-0.349066, 0.349066),
    "rh_LFJ3": (-0.261799, 1.5708),
    "rh_LFJ2": (0.0, 1.5708),
    "rh_LFJ1": (0.0, 1.5708),
}


@dataclass(frozen=True)
class SceneRecord:
    """Validated inputs and provenance for one DGN2k scene."""

    scene_id: str
    scene_path: Path
    stored_scene_path: str
    object_id: str
    mesh_path: Path
    scale: float
    object_pose_wxyz: np.ndarray
    table_pose_wxyz: np.ndarray
    table_normal_local: np.ndarray
    table_normal_world: np.ndarray
    table_origin_world: np.ndarray
    scene_sha256: str
    mesh_sha256: str

    def to_manifest(self, *, include_table: bool = True) -> dict:
        """Return JSON-compatible scene provenance."""

        manifest = {
            "scene_id": self.scene_id,
            "stored_scene_path": self.stored_scene_path,
            "scene_sha256": self.scene_sha256,
            "object_id": self.object_id,
            "mesh_path_from_scene": os.path.relpath(self.mesh_path, self.scene_path.parent),
            "mesh_sha256": self.mesh_sha256,
            "scale": self.scale,
            "object_pose_wxyz": self.object_pose_wxyz.tolist(),
        }
        if include_table:
            manifest.update(
                {
                    "table_type": "plane",
                    "table_pose_wxyz": self.table_pose_wxyz.tolist(),
                    "table_normal_local": self.table_normal_local.tolist(),
                    "table_normal_world": self.table_normal_world.tolist(),
                    "table_origin_world": self.table_origin_world.tolist(),
                }
            )
        return manifest


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a file SHA256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    """Hash the exact dtype, shape, and contiguous bytes of an array."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def scene_manifest_sha256(
    records: list[SceneRecord], *, include_table: bool = True
) -> str:
    """Hash the ordered scene/asset provenance used by an experiment."""

    payload = json.dumps(
        [record.to_manifest(include_table=include_table) for record in records],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scene_scale_histogram(records: list[SceneRecord]) -> dict[str, int]:
    """Return a stable, JSON-compatible histogram of isotropic scene scales."""

    histogram: dict[str, int] = {}
    for record in records:
        key = format(record.scale, ".9g")
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items(), key=lambda item: float(item[0])))


def _relative_to(path: Path, parent: Path) -> Path:
    try:
        return path.relative_to(parent)
    except ValueError as error:
        raise ValueError(f"{path} is outside {parent}") from error


def load_scene_record(
    scene_path: Path,
    scene_root: Path,
    stored_prefix: str = STORED_SCENE_PREFIX,
) -> SceneRecord:
    """Load one exact BimanBODex DGN2k scene and asset contract."""

    scene_root = Path(scene_root).resolve(strict=True)
    scene_path = Path(scene_path).resolve(strict=True)
    relative = _relative_to(scene_path, scene_root)
    if scene_path.suffix != ".npy" or len(relative.parts) < 3:
        raise ValueError(f"invalid scene config path: {scene_path}")

    loaded = np.load(scene_path, allow_pickle=True)
    try:
        config = loaded.item()
    except ValueError as error:
        raise ValueError(f"scene config must contain one dictionary: {scene_path}") from error
    if not isinstance(config, dict):
        raise ValueError(f"scene config is not a dictionary: {scene_path}")

    scene_id = relative.with_suffix("").as_posix()
    if config.get("scene_id") != scene_id:
        raise ValueError(
            f"scene_id mismatch: expected {scene_id!r}, got {config.get('scene_id')!r}"
        )
    task = config.get("task")
    object_id = task.get("obj_name") if isinstance(task, dict) else None
    if object_id != relative.parts[0]:
        raise ValueError(f"task.obj_name does not match scene path: {scene_path}")
    objects = config.get("scene")
    object_config = objects.get(object_id) if isinstance(objects, dict) else None
    if not isinstance(object_config, dict) or object_config.get("type") != "rigid_object":
        raise ValueError(f"target object is not a rigid_object: {scene_path}")

    raw_mesh = object_config.get("file_path")
    if not isinstance(raw_mesh, str):
        raise ValueError(f"target object file_path is missing: {scene_path}")
    mesh_path = Path(raw_mesh) if os.path.isabs(raw_mesh) else scene_path.parent / raw_mesh
    mesh_path = mesh_path.resolve(strict=True)
    if mesh_path.name != "simplified.obj" or mesh_path.parent.name != "mesh":
        raise ValueError(f"target object must use exact mesh/simplified.obj: {mesh_path}")

    scale = np.asarray(object_config.get("scale"), dtype=np.float64).reshape(-1)
    if scale.shape != (3,) or not np.isfinite(scale).all() or scale[0] <= 0.0:
        raise ValueError(f"object scale must be a positive finite vector: {scene_path}")
    if not np.allclose(scale, scale[0], rtol=0.0, atol=1e-8):
        raise ValueError(f"object scale must be isotropic: {scene_path}")

    pose = np.asarray(object_config.get("pose"), dtype=np.float64).reshape(-1)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"object pose must be finite [xyz, qw, qx, qy, qz]: {scene_path}")
    quaternion_norm = np.linalg.norm(pose[3:])
    if not np.isclose(quaternion_norm, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError(f"object quaternion is not normalized: {scene_path}")

    table_config = objects.get("table") if isinstance(objects, dict) else None
    if not isinstance(table_config, dict) or table_config.get("type") != "plane":
        raise ValueError(f"scene table must be an explicit plane: {scene_path}")
    table_pose = np.asarray(table_config.get("pose"), dtype=np.float64).reshape(-1)
    if table_pose.shape != (7,) or not np.isfinite(table_pose).all():
        raise ValueError(
            f"table pose must be finite [xyz, qw, qx, qy, qz]: {scene_path}"
        )
    table_quaternion_norm = np.linalg.norm(table_pose[3:])
    if not np.isclose(table_quaternion_norm, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError(f"table quaternion is not normalized: {scene_path}")
    table_normal_local = np.asarray(
        table_config.get("size"), dtype=np.float64
    ).reshape(-1)
    if (
        table_normal_local.shape != (3,)
        or not np.isfinite(table_normal_local).all()
        or np.linalg.norm(table_normal_local) <= 0.0
    ):
        raise ValueError(f"table plane normal must be a finite non-zero vector: {scene_path}")
    table_normal_local = table_normal_local / np.linalg.norm(table_normal_local)
    table_normal_world = (
        quaternion_wxyz_to_matrix(table_pose[3:]) @ table_normal_local
    )
    table_normal_world /= np.linalg.norm(table_normal_world)

    return SceneRecord(
        scene_id=scene_id,
        scene_path=scene_path,
        stored_scene_path=f"{stored_prefix.rstrip('/')}/{relative.as_posix()}",
        object_id=object_id,
        mesh_path=mesh_path,
        scale=float(scale[0]),
        object_pose_wxyz=pose.copy(),
        table_pose_wxyz=table_pose.copy(),
        table_normal_local=table_normal_local.copy(),
        table_normal_world=table_normal_world.copy(),
        table_origin_world=table_pose[:3].copy(),
        scene_sha256=sha256_file(scene_path),
        mesh_sha256=sha256_file(mesh_path),
    )


def discover_scene_paths(
    scene_root: Path,
    reference_grasp_roots=(),
    scene_list: Optional[str] = None,
) -> list[Path]:
    """Resolve the same scene union used by reference 20-candidate outputs."""

    scene_root = Path(scene_root).resolve(strict=True)
    resolved: dict[str, Path] = {}
    for root_value in reference_grasp_roots:
        root = Path(root_value).resolve(strict=True)
        for grasp_path in sorted(root.rglob("*_grasp.npy")):
            relative = grasp_path.relative_to(root)
            scene_name = relative.name[: -len("_grasp.npy")] + ".npy"
            scene_path = (scene_root / relative.with_name(scene_name)).resolve()
            if not scene_path.is_file():
                raise ValueError(f"reference output has no matching scene: {grasp_path}")
            resolved[str(scene_path)] = scene_path

    if scene_list is not None:
        list_path = Path(scene_list).resolve(strict=True)
        for line_number, raw_line in enumerate(
            list_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            value = raw_line.split("#", 1)[0].strip()
            if not value:
                continue
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = scene_root / candidate
            if candidate.suffix != ".npy":
                candidate = candidate.with_suffix(".npy")
            candidate = candidate.resolve()
            if not candidate.is_file():
                raise ValueError(f"scene list line {line_number} does not exist: {candidate}")
            _relative_to(candidate, scene_root)
            resolved[str(candidate)] = candidate

    if not resolved:
        raise ValueError("no scenes found; pass --reference-grasp-root and/or --scene-list")
    return [resolved[key] for key in sorted(resolved)]


def sample_scaled_surface(
    mesh_path: Path,
    scale: float,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """Sample exact scaled mesh surfaces without normalization or recentering."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")

    import trimesh

    mesh = trimesh.load_mesh(str(mesh_path), process=False)
    if not hasattr(mesh, "faces") or not hasattr(mesh, "vertices"):
        raise ValueError(f"mesh is not a triangular mesh: {mesh_path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * float(scale)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = np.isfinite(areas) & (areas > 0.0)
    if not np.any(valid):
        raise ValueError(f"mesh has no non-degenerate triangles: {mesh_path}")
    triangles = triangles[valid]
    probabilities = areas[valid] / areas[valid].sum()

    rng = np.random.default_rng(int(seed))
    selected = triangles[rng.choice(len(triangles), size=sample_count, p=probabilities)]
    barycentric = rng.random((sample_count, 2))
    reflected = barycentric.sum(axis=1) > 1.0
    barycentric[reflected] = 1.0 - barycentric[reflected]
    points = (
        selected[:, 0]
        + barycentric[:, :1] * (selected[:, 1] - selected[:, 0])
        + barycentric[:, 1:] * (selected[:, 2] - selected[:, 0])
    ).astype(np.float32)
    if points.shape != (sample_count, 3) or not np.isfinite(points).all():
        raise ValueError(f"invalid sampled point cloud for {mesh_path}")
    return points


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def euler_xyz_to_matrix(euler: np.ndarray) -> np.ndarray:
    """Match SciPy ``Rotation.from_euler('XYZ', ...)`` used by DRO."""

    x, y, z = np.asarray(euler, dtype=np.float64).reshape(3)
    return _rotation_x(x) @ _rotation_y(y) @ _rotation_z(z)


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a normalized wxyz quaternion into a rotation matrix."""

    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if not np.isfinite(q).all() or norm == 0.0:
        raise ValueError("quaternion must be finite and non-zero")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix into canonicalized wxyz form."""

    rotation = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        root = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                0.25 * root,
                (rotation[2, 1] - rotation[1, 2]) / root,
                (rotation[0, 2] - rotation[2, 0]) / root,
                (rotation[1, 0] - rotation[0, 1]) / root,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            root = 2.0 * np.sqrt(max(1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 1e-12))
            quaternion = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / root,
                    0.25 * root,
                    (rotation[0, 1] + rotation[1, 0]) / root,
                    (rotation[0, 2] + rotation[2, 0]) / root,
                ]
            )
        elif index == 1:
            root = 2.0 * np.sqrt(max(1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 1e-12))
            quaternion = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / root,
                    (rotation[0, 1] + rotation[1, 0]) / root,
                    0.25 * root,
                    (rotation[1, 2] + rotation[2, 1]) / root,
                ]
            )
        else:
            root = 2.0 * np.sqrt(max(1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 1e-12))
            quaternion = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / root,
                    (rotation[0, 2] + rotation[2, 0]) / root,
                    (rotation[1, 2] + rotation[2, 1]) / root,
                    0.25 * root,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return quaternion


def pose_wxyz_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert ``[xyz, qw, qx, qy, qz]`` into a homogeneous transform."""

    value = np.asarray(pose, dtype=np.float64).reshape(7)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_wxyz_to_matrix(value[3:])
    transform[:3, 3] = value[:3]
    return transform


def matrix_to_pose_wxyz(transform: np.ndarray) -> np.ndarray:
    """Convert a homogeneous transform into ``[xyz, qw, qx, qy, qz]``."""

    value = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return np.concatenate((value[:3, 3], matrix_to_quaternion_wxyz(value[:3, :3])))


def dro_q_to_object_palm_transform(q_euler: np.ndarray) -> np.ndarray:
    """Resolve the released floating forearm and wrist chain to DRO's palm frame."""

    q = np.asarray(q_euler, dtype=np.float64).reshape(-1)
    if q.shape != (len(DRO_SHADOW_Q_NAMES),) or not np.isfinite(q).all():
        raise ValueError(f"DRO q must be finite shape {(len(DRO_SHADOW_Q_NAMES),)}, got {q.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = euler_xyz_to_matrix(q[3:6])
    transform[:3, 3] = q[:3]

    wrist_origin = np.eye(4, dtype=np.float64)
    wrist_origin[:3, 3] = (0.0, -0.010, 0.21301)
    wrist_rotation = np.eye(4, dtype=np.float64)
    wrist_rotation[:3, :3] = _rotation_y(q[6])
    palm_origin = np.eye(4, dtype=np.float64)
    palm_origin[:3, 3] = (0.0, 0.0, 0.034)
    palm_rotation = np.eye(4, dtype=np.float64)
    palm_rotation[:3, :3] = _rotation_x(q[7])
    return transform @ wrist_origin @ wrist_rotation @ palm_origin @ palm_rotation


def map_dro_shadow_fingers(q_euler: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorder released DRO finger joints into Bench order without relabeling signs."""

    q = np.asarray(q_euler, dtype=np.float64).reshape(-1)
    if q.shape != (len(DRO_SHADOW_Q_NAMES),) or not np.isfinite(q).all():
        raise ValueError(f"DRO q must be finite shape {(len(DRO_SHADOW_Q_NAMES),)}, got {q.shape}")
    by_name = dict(zip(DRO_SHADOW_Q_NAMES, q))
    mapped = np.array(
        [by_name[BENCH_FROM_DRO[name]] for name in BENCH_SHADOW_JOINT_NAMES],
        dtype=np.float64,
    )
    excess = np.zeros_like(mapped)
    for index, name in enumerate(BENCH_SHADOW_JOINT_NAMES):
        lower, upper = BENCH_SHADOW_JOINT_LIMITS[name]
        excess[index] = max(lower - mapped[index], mapped[index] - upper, 0.0)
    return mapped, excess


def load_dro_shadow_finger_joint_limits(
    urdf_path: Path,
) -> dict[str, tuple[float, float]]:
    """Load the released DRO Shadow finger limits from its URDF."""

    urdf_path = Path(urdf_path).resolve(strict=True)
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError as error:
        raise ValueError(f"invalid Shadow URDF XML: {urdf_path}") from error

    joint_elements = {
        joint.get("name"): joint
        for joint in root.findall("joint")
        if joint.get("name") is not None
    }
    limits: dict[str, tuple[float, float]] = {}
    for joint_name in DRO_SHADOW_FINGER_JOINT_NAMES:
        joint = joint_elements.get(joint_name)
        limit = joint.find("limit") if joint is not None else None
        if limit is None:
            raise ValueError(f"Shadow URDF joint has no limit: {joint_name}")
        try:
            lower = float(limit.attrib["lower"])
            upper = float(limit.attrib["upper"])
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"Shadow URDF joint has invalid limits: {joint_name}"
            ) from error
        if not np.isfinite((lower, upper)).all() or lower > upper:
            raise ValueError(f"Shadow URDF joint has invalid limits: {joint_name}")
        limits[joint_name] = (lower, upper)
    return limits


def _representable_interval_bound(value: float, dtype: np.dtype, *, lower: bool):
    """Return a floating-point bound rounded into the mathematical interval."""

    cast = np.asarray(value, dtype=dtype)[()]
    if lower and float(cast) < value:
        cast = np.nextafter(cast, np.asarray(np.inf, dtype=dtype)[()])
    elif not lower and float(cast) > value:
        cast = np.nextafter(cast, np.asarray(-np.inf, dtype=dtype)[()])
    return cast


def clamp_dro_shadow_export_stages(
    stage_q: np.ndarray,
    dro_joint_limits: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, list[dict]]:
    """Clamp only exported finger joints to the DRO/Bench limit intersection."""

    q = np.asarray(stage_q)
    if q.ndim != 3 or q.shape[1:] != (len(STAGE_NAMES), len(DRO_SHADOW_Q_NAMES)):
        raise ValueError(
            f"expected DRO stages [N,{len(STAGE_NAMES)},{len(DRO_SHADOW_Q_NAMES)}], "
            f"got {q.shape}"
        )
    if not np.issubdtype(q.dtype, np.floating) or not np.isfinite(q).all():
        raise ValueError("DRO stage q must contain finite floating-point values")

    export_q = q.copy()
    diagnostics = []
    for bench_joint_name in BENCH_SHADOW_JOINT_NAMES:
        dro_joint_name = BENCH_FROM_DRO[bench_joint_name]
        if dro_joint_name not in dro_joint_limits:
            raise ValueError(f"missing DRO joint limit: {dro_joint_name}")
        dro_lower, dro_upper = dro_joint_limits[dro_joint_name]
        bench_lower, bench_upper = BENCH_SHADOW_JOINT_LIMITS[bench_joint_name]
        export_lower = max(float(dro_lower), bench_lower)
        export_upper = min(float(dro_upper), bench_upper)
        if (
            not np.isfinite((export_lower, export_upper)).all()
            or export_lower > export_upper
        ):
            raise ValueError(
                f"DRO and Bench joint limits do not intersect: {dro_joint_name}"
            )

        dtype_lower = _representable_interval_bound(export_lower, q.dtype, lower=True)
        dtype_upper = _representable_interval_bound(export_upper, q.dtype, lower=False)
        q_index = DRO_SHADOW_Q_NAMES.index(dro_joint_name)
        raw_values = q[:, :, q_index]
        clamped_values = np.clip(raw_values, dtype_lower, dtype_upper)
        export_q[:, :, q_index] = clamped_values
        for candidate_index, stage_index in np.argwhere(raw_values != clamped_values):
            raw_value = float(raw_values[candidate_index, stage_index])
            clamped_value = float(clamped_values[candidate_index, stage_index])
            diagnostics.append(
                {
                    "candidate_index": int(candidate_index),
                    "stage_index": int(stage_index),
                    "stage_name": STAGE_NAMES[stage_index],
                    "dro_joint_name": dro_joint_name,
                    "bench_joint_name": bench_joint_name,
                    "raw_value": raw_value,
                    "clamped_value": clamped_value,
                    "delta": clamped_value - raw_value,
                    "dro_limit": [float(dro_lower), float(dro_upper)],
                    "bench_limit": [bench_lower, bench_upper],
                    "export_limit": [export_lower, export_upper],
                }
            )
    return export_q, diagnostics


def dro_q_to_bench_pose(q_euler: np.ndarray, object_pose_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Export one DRO q through ``T_WH = T_WO @ T_OPalm`` into Bench form."""

    object_world = pose_wxyz_to_matrix(object_pose_wxyz)
    palm_object = dro_q_to_object_palm_transform(q_euler)
    palm_world = matrix_to_pose_wxyz(object_world @ palm_object)
    joints, limit_excess = map_dro_shadow_fingers(q_euler)
    return np.concatenate((palm_world, joints)), limit_excess


def make_bench_artifact(
    stage_q: np.ndarray,
    object_pose_wxyz: np.ndarray,
    scene_path: str,
) -> tuple[dict, np.ndarray]:
    """Convert official ``q_outer/q/q_inner`` stages into the unchanged Bench schema."""

    q = np.asarray(stage_q, dtype=np.float64)
    if q.ndim != 3 or q.shape[1:] != (3, len(DRO_SHADOW_Q_NAMES)):
        raise ValueError(f"expected DRO stages [N,3,{len(DRO_SHADOW_Q_NAMES)}], got {q.shape}")
    if not np.isfinite(q).all():
        raise ValueError("DRO stage q contains non-finite values")
    if not isinstance(scene_path, str) or not scene_path:
        raise ValueError("scene_path must be a non-empty string")

    object_pose = np.asarray(object_pose_wxyz, dtype=np.float64).reshape(7)

    candidates = np.empty((q.shape[0], 3, 7 + len(BENCH_SHADOW_JOINT_NAMES)), dtype=np.float32)
    limit_excess = np.empty((q.shape[0], 3, len(BENCH_SHADOW_JOINT_NAMES)), dtype=np.float32)
    for candidate_index in range(q.shape[0]):
        for stage_index in range(3):
            pose, excess = dro_q_to_bench_pose(q[candidate_index, stage_index], object_pose)
            candidates[candidate_index, stage_index] = pose.astype(np.float32)
            limit_excess[candidate_index, stage_index] = excess.astype(np.float32)
    artifact = {
        "robot_pose": candidates[np.newaxis, ...],
        "joint_names": list(BENCH_SHADOW_JOINT_NAMES),
        "scene_path": [scene_path],
    }
    return artifact, limit_excess


def validate_artifact(
    artifact: dict,
    record: SceneRecord,
    stage_q: np.ndarray,
    *,
    joint_limit_tolerance: float = 1e-6,
    palm_tolerance: float = 1e-5,
) -> None:
    """Validate schema, stage semantics, frame composition, and joint limits."""

    q = np.asarray(stage_q, dtype=np.float64)
    expected_shape = (1, q.shape[0], 3, 7 + len(BENCH_SHADOW_JOINT_NAMES))
    if artifact.get("joint_names") != list(BENCH_SHADOW_JOINT_NAMES):
        raise ValueError("artifact joint_names do not match Bench order")
    if artifact.get("scene_path") != [record.stored_scene_path]:
        raise ValueError("artifact scene_path does not match the scene record")
    robot_pose = np.asarray(artifact.get("robot_pose"))
    if robot_pose.shape != expected_shape or robot_pose.dtype != np.float32:
        raise ValueError(f"artifact robot_pose must be float32 {expected_shape}, got {robot_pose.shape} {robot_pose.dtype}")
    if not np.isfinite(robot_pose).all():
        raise ValueError("artifact robot_pose contains non-finite values")

    expected, excess = make_bench_artifact(q, record.object_pose_wxyz, record.stored_scene_path)
    if not np.allclose(robot_pose, expected["robot_pose"], rtol=0.0, atol=1e-6):
        raise ValueError("artifact does not match DRO-to-Bench conversion")
    if float(np.max(excess, initial=0.0)) > joint_limit_tolerance:
        raise ValueError(f"artifact exceeds Bench joint limits by {float(np.max(excess)):.9g} rad")

    palms = robot_pose[0, :, :, :7]
    if not np.allclose(palms, palms[:, 1:2, :], rtol=0.0, atol=palm_tolerance):
        raise ValueError("official controller changed the exported palm pose across stages")
