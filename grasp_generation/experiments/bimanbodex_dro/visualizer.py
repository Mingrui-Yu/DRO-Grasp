"""Read-only loading and CPU scene preparation for exported DRO DGN2k poses."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .contracts import (
    BENCH_SHADOW_JOINT_NAMES,
    DRO_BENCH_LINK_PAIRS,
    DRO_SHADOW_Q_NAMES,
    FILTERED_RAW_SCHEMA_VERSION,
    FILTERED_RUN_SCHEMA_VERSION,
    LEGACY_RAW_SCHEMA_VERSION,
    LEGACY_RUN_SCHEMA_VERSION,
    RAW_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    STAGE_NAMES,
    SUPPORTED_RUN_SCHEMA_VERSIONS,
    build_dro_shadow_pk_chain,
    clamp_dro_shadow_export_stages,
    dro_q_to_bench_pose,
    dro_stage_q_to_object_palm_transforms,
    legacy_dro_stage_q_to_object_palm_transforms,
    load_dro_shadow_finger_joint_limits,
    load_scene_record,
    make_bench_artifact,
    map_dro_shadow_fingers,
    matrix_to_pose_wxyz,
    pose_wxyz_to_matrix,
    sha256_array,
    sha256_file,
    validate_artifact,
)


@dataclass(frozen=True)
class MeshData:
    """Triangle mesh ready for Viser in world coordinates."""

    vertices: np.ndarray
    faces: np.ndarray
    source_count: int = 1


@dataclass(frozen=True)
class OutputScene:
    """One scene entry from the persisted run manifest."""

    scene_id: str
    status: str
    scale: float
    manifest: dict
    failure: Optional[dict]


@dataclass(frozen=True)
class LoadedScene:
    """Strictly validated raw/artifact pair for a completed scene."""

    entry: OutputScene
    record: object
    raw: dict
    artifact: dict
    palm_object_transforms: np.ndarray
    raw_path: Path
    artifact_path: Path


@dataclass(frozen=True)
class PreparedSelection:
    """CPU-prepared geometry and diagnostics for one UI selection."""

    scene_id: str
    candidate_index: int
    selected_stage: str
    stage_names: tuple
    pose_source: str
    object_mesh: MeshData
    object_point_cloud_world: np.ndarray
    hand_meshes: dict
    dro_component_meshes: dict
    bench_hand_meshes: dict
    common_link_frames: dict
    object_pose_wxyz: np.ndarray
    palm_poses_wxyz: dict
    diagnostics: dict


@dataclass(frozen=True)
class _Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


@dataclass(frozen=True)
class _LinkMesh:
    link_name: str
    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class _BenchGeomMesh:
    geom_id: int
    vertices: np.ndarray
    faces: np.ndarray


def _load_json_dict(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing {label}: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {label} JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _load_numpy_dict(path: Path, label: str) -> dict:
    try:
        loaded = np.load(path, allow_pickle=True)
        value = loaded.item()
    except FileNotFoundError as error:
        raise ValueError(f"missing {label}: {path}") from error
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one dictionary: {path}")
    return value


def _resolve_inside(root: Path, relative_value, label: str) -> Path:
    if not isinstance(relative_value, str) or not relative_value:
        raise ValueError(f"missing {label} path")
    candidate = Path(relative_value)
    if candidate.is_absolute():
        raise ValueError(f"{label} path must be relative to output-root: {candidate}")
    try:
        resolved = (root / candidate).resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(f"invalid {label} path: {candidate}") from error
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    return resolved


def _as_float_scale(value, label: str) -> float:
    try:
        scale = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a positive finite scalar") from error
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{label} must be a positive finite scalar")
    return scale


def _failure_message(failure: Optional[dict]) -> Optional[str]:
    if not isinstance(failure, dict):
        return None
    message = failure.get("message")
    if isinstance(message, str) and message:
        return message
    candidate_failures = failure.get("candidate_failures")
    if isinstance(candidate_failures, list) and candidate_failures:
        first = candidate_failures[0]
        if isinstance(first, dict):
            candidate_message = first.get("message")
            if isinstance(candidate_message, str) and candidate_message:
                return candidate_message
    return None


def _parse_vector(value: Optional[str], size: int, default: tuple) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    parsed = np.fromstring(value, sep=" ", dtype=np.float64)
    if parsed.shape != (size,) or not np.isfinite(parsed).all():
        raise ValueError(f"invalid URDF vector {value!r}")
    return parsed


def _rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def _rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def _rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def _origin_transform(element: Optional[ET.Element]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if element is None:
        return transform
    xyz = _parse_vector(element.get("xyz"), 3, (0.0, 0.0, 0.0))
    roll, pitch, yaw = _parse_vector(
        element.get("rpy"), 3, (0.0, 0.0, 0.0)
    )
    transform[:3, :3] = (
        _rotation_z(float(yaw))
        @ _rotation_y(float(pitch))
        @ _rotation_x(float(roll))
    )
    transform[:3, 3] = xyz
    return transform


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("URDF revolute joint axis must be finite and non-zero")
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    one_minus_c = 1.0 - c
    return np.array(
        (
            (c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s),
            (y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s),
            (z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c),
        ),
        dtype=np.float64,
    )


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return (
        points @ transform[:3, :3].T + transform[:3, 3]
    ).astype(np.float32)


def _load_trimesh(path: Path):
    import trimesh

    geometry = trimesh.load_mesh(str(path), process=False)
    if isinstance(geometry, trimesh.Scene):
        dumped = geometry.dump()
        meshes = [item for item in dumped if isinstance(item, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"mesh scene contains no triangle meshes: {path}")
        geometry = trimesh.util.concatenate(meshes)
    if not isinstance(geometry, trimesh.Trimesh):
        raise ValueError(f"asset is not a triangle mesh: {path}")
    vertices = np.asarray(geometry.vertices, dtype=np.float64)
    faces = np.asarray(geometry.faces, dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or len(vertices) == 0
        or len(faces) == 0
        or not np.isfinite(vertices).all()
    ):
        raise ValueError(f"invalid triangle mesh geometry: {path}")
    return vertices, faces


class ShadowHandModel:
    """Minimal CPU URDF FK and visual-mesh loader for the released Shadow hand."""

    def __init__(self, urdf_path: Path):
        self.urdf_path = Path(urdf_path).resolve(strict=True)
        try:
            root = ET.parse(self.urdf_path).getroot()
        except ET.ParseError as error:
            raise ValueError(f"invalid Shadow URDF XML: {self.urdf_path}") from error
        if root.tag != "robot":
            raise ValueError(f"Shadow URDF root must be <robot>: {self.urdf_path}")

        links = [element.get("name") for element in root.findall("link")]
        if any(name is None for name in links) or len(set(links)) != len(links):
            raise ValueError("Shadow URDF has missing or duplicate link names")

        joints = []
        child_links = set()
        for element in root.findall("joint"):
            name = element.get("name")
            joint_type = element.get("type")
            parent = element.find("parent")
            child = element.find("child")
            if (
                name is None
                or joint_type not in {"fixed", "revolute", "continuous", "prismatic"}
                or parent is None
                or child is None
                or parent.get("link") not in links
                or child.get("link") not in links
            ):
                raise ValueError(f"invalid Shadow URDF joint: {name!r}")
            child_name = child.get("link")
            if child_name in child_links:
                raise ValueError(f"Shadow URDF link has multiple parents: {child_name}")
            child_links.add(child_name)
            axis = _parse_vector(
                element.find("axis").get("xyz") if element.find("axis") is not None else None,
                3,
                (1.0, 0.0, 0.0),
            )
            joints.append(
                _Joint(
                    name=name,
                    joint_type=joint_type,
                    parent=parent.get("link"),
                    child=child_name,
                    origin=_origin_transform(element.find("origin")),
                    axis=axis,
                )
            )

        moving_names = tuple(
            joint.name for joint in joints if joint.joint_type != "fixed"
        )
        if moving_names != DRO_SHADOW_Q_NAMES:
            raise ValueError(
                "released Shadow URDF movable joint order does not match "
                f"DRO_SHADOW_Q_NAMES: {moving_names}"
            )
        roots = set(links) - child_links
        if roots != {"world"}:
            raise ValueError(f"unexpected Shadow URDF roots: {sorted(roots)}")

        self._joints = tuple(joints)
        self._children = {}
        for joint in joints:
            self._children.setdefault(joint.parent, []).append(joint)
        self._link_meshes = self._load_link_meshes(root)
        if not self._link_meshes:
            raise ValueError("Shadow URDF contains no renderable visual geometry")

    def _load_link_meshes(self, root: ET.Element) -> tuple:
        import trimesh

        result = []
        mesh_cache = {}
        for link in root.findall("link"):
            link_name = link.get("name")
            for visual in link.findall("visual"):
                geometry = visual.find("geometry")
                if geometry is None or len(geometry) != 1:
                    raise ValueError(f"invalid visual geometry on Shadow link {link_name}")
                geometry_element = geometry[0]
                if geometry_element.tag == "mesh":
                    filename = geometry_element.get("filename")
                    if not filename or filename.startswith("package://"):
                        raise ValueError(
                            f"unsupported Shadow mesh URI on link {link_name}: {filename!r}"
                        )
                    mesh_path = (self.urdf_path.parent / filename).resolve(strict=True)
                    scale = _parse_vector(
                        geometry_element.get("scale"), 3, (1.0, 1.0, 1.0)
                    )
                    cache_key = (str(mesh_path), tuple(scale.tolist()))
                    if cache_key not in mesh_cache:
                        vertices, faces = _load_trimesh(mesh_path)
                        mesh_cache[cache_key] = (vertices * scale, faces)
                    vertices, faces = mesh_cache[cache_key]
                elif geometry_element.tag == "box":
                    size = _parse_vector(geometry_element.get("size"), 3, ())
                    primitive = trimesh.creation.box(extents=size)
                    vertices = np.asarray(primitive.vertices, dtype=np.float64)
                    faces = np.asarray(primitive.faces, dtype=np.int64)
                elif geometry_element.tag == "sphere":
                    radius = _as_float_scale(geometry_element.get("radius"), "sphere radius")
                    primitive = trimesh.creation.icosphere(subdivisions=2, radius=radius)
                    vertices = np.asarray(primitive.vertices, dtype=np.float64)
                    faces = np.asarray(primitive.faces, dtype=np.int64)
                elif geometry_element.tag == "cylinder":
                    radius = _as_float_scale(geometry_element.get("radius"), "cylinder radius")
                    length = _as_float_scale(geometry_element.get("length"), "cylinder length")
                    primitive = trimesh.creation.cylinder(radius=radius, height=length)
                    vertices = np.asarray(primitive.vertices, dtype=np.float64)
                    faces = np.asarray(primitive.faces, dtype=np.int64)
                else:
                    raise ValueError(
                        f"unsupported Shadow visual geometry {geometry_element.tag!r} "
                        f"on link {link_name}"
                    )
                visual_origin = _origin_transform(visual.find("origin"))
                result.append(
                    _LinkMesh(
                        link_name=link_name,
                        vertices=_transform_points(vertices, visual_origin).astype(np.float64),
                        faces=np.asarray(faces, dtype=np.int64).copy(),
                    )
                )
        return tuple(result)

    def link_transforms(self, q: np.ndarray) -> dict:
        """Return object-frame transforms for every URDF link."""

        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape != (len(DRO_SHADOW_Q_NAMES),) or not np.isfinite(q).all():
            raise ValueError(
                f"DRO q must be finite shape {(len(DRO_SHADOW_Q_NAMES),)}, got {q.shape}"
            )
        q_by_name = dict(zip(DRO_SHADOW_Q_NAMES, q))
        transforms = {"world": np.eye(4, dtype=np.float64)}
        pending = ["world"]
        while pending:
            parent = pending.pop()
            for joint in self._children.get(parent, ()):
                motion = np.eye(4, dtype=np.float64)
                value = 0.0 if joint.joint_type == "fixed" else float(q_by_name[joint.name])
                if joint.joint_type in {"revolute", "continuous"}:
                    motion[:3, :3] = _axis_rotation(joint.axis, value)
                elif joint.joint_type == "prismatic":
                    axis = joint.axis / np.linalg.norm(joint.axis)
                    motion[:3, 3] = axis * value
                transforms[joint.child] = transforms[parent] @ joint.origin @ motion
                pending.append(joint.child)
        expected_links = {"world"}
        expected_links.update(joint.child for joint in self._joints)
        if set(transforms) != expected_links:
            missing = sorted(expected_links - set(transforms))
            raise ValueError(f"Shadow URDF kinematic tree is disconnected: {missing}")
        return transforms

    def mesh(
        self,
        q: np.ndarray,
        object_world: np.ndarray,
        *,
        include_links=None,
        exclude_links=(),
    ) -> MeshData:
        """Build selected URDF visual links in the DGN2k world frame."""

        transforms = self.link_transforms(q)
        included = None if include_links is None else set(include_links)
        excluded = set(exclude_links)
        vertices = []
        faces = []
        vertex_offset = 0
        source_count = 0
        for link_mesh in self._link_meshes:
            if (
                (included is not None and link_mesh.link_name not in included)
                or link_mesh.link_name in excluded
            ):
                continue
            world = object_world @ transforms[link_mesh.link_name]
            transformed = _transform_points(link_mesh.vertices, world)
            vertices.append(transformed)
            faces.append(link_mesh.faces + vertex_offset)
            vertex_offset += len(transformed)
            source_count += 1
        if not vertices:
            raise ValueError("selected Shadow links contain no renderable visual geometry")
        return MeshData(
            vertices=np.concatenate(vertices, axis=0).astype(np.float32),
            faces=np.concatenate(faces, axis=0).astype(np.int32),
            source_count=source_count,
        )

    def palm_world_transform(self, q: np.ndarray, object_world: np.ndarray) -> np.ndarray:
        """Return ``T_WH`` from generic URDF FK."""

        return object_world @ self.link_transforms(q)["palm"]


class BenchShadowHandModel:
    """CPU MuJoCo FK and visual meshes for the palm-root Bench Shadow asset."""

    def __init__(self, mjcf_path: Path):
        try:
            import mujoco
        except ImportError as error:
            raise RuntimeError("mujoco is required for the DRO/Bench overlay") from error

        self._mujoco = mujoco
        self.mjcf_path = Path(mjcf_path).resolve(strict=True)
        self.model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self.data = mujoco.MjData(self.model)
        joint_names = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(self.model.njnt)
        )
        if (
            set(joint_names) != set(BENCH_SHADOW_JOINT_NAMES)
            or len(joint_names) != len(BENCH_SHADOW_JOINT_NAMES)
            or self.model.nq != len(BENCH_SHADOW_JOINT_NAMES)
        ):
            raise ValueError(f"Bench Shadow joint set mismatch: {joint_names}")
        if mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "rh_palm"
        ) < 0:
            raise ValueError("Bench Shadow MJCF has no rh_palm root body")

        self._joint_qpos_addresses = {}
        for name in BENCH_SHADOW_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            self._joint_qpos_addresses[name] = int(self.model.jnt_qposadr[joint_id])

        meshes = []
        for geom_id in range(self.model.ngeom):
            if int(self.model.geom_group[geom_id]) != 2:
                continue
            mesh_id = int(self.model.geom_dataid[geom_id])
            if mesh_id < 0:
                continue
            vertex_start = int(self.model.mesh_vertadr[mesh_id])
            vertex_count = int(self.model.mesh_vertnum[mesh_id])
            face_start = int(self.model.mesh_faceadr[mesh_id])
            face_count = int(self.model.mesh_facenum[mesh_id])
            meshes.append(
                _BenchGeomMesh(
                    geom_id=geom_id,
                    vertices=np.asarray(
                        self.model.mesh_vert[
                            vertex_start : vertex_start + vertex_count
                        ],
                        dtype=np.float64,
                    ).copy(),
                    faces=np.asarray(
                        self.model.mesh_face[face_start : face_start + face_count],
                        dtype=np.int64,
                    ).copy(),
                )
            )
        if not meshes:
            raise ValueError("Bench Shadow MJCF contains no group-2 visual mesh geoms")
        self._geom_meshes = tuple(meshes)

        required_bodies = {"rh_palm", *DRO_BENCH_LINK_PAIRS.values()}
        missing = sorted(
            name
            for name in required_bodies
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) < 0
        )
        if missing:
            raise ValueError(f"Bench Shadow MJCF is missing common bodies: {missing}")

    def _forward(self, joint_values: np.ndarray) -> None:
        values = np.asarray(joint_values, dtype=np.float64).reshape(-1)
        if values.shape != (len(BENCH_SHADOW_JOINT_NAMES),) or not np.isfinite(
            values
        ).all():
            raise ValueError(
                "Bench Shadow joint values must be finite shape "
                f"{(len(BENCH_SHADOW_JOINT_NAMES),)}"
            )
        self.data.qpos[:] = 0.0
        for name, value in zip(BENCH_SHADOW_JOINT_NAMES, values):
            self.data.qpos[self._joint_qpos_addresses[name]] = value
        self._mujoco.mj_kinematics(self.model, self.data)

    def _body_transform(self, name: str) -> np.ndarray:
        body_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_BODY, name
        )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.data.xmat[body_id].reshape(3, 3)
        transform[:3, 3] = self.data.xpos[body_id]
        return transform

    def local_body_transforms(self, joint_values: np.ndarray) -> dict:
        """Return standalone-MJCF transforms before applying the palm world pose."""

        self._forward(joint_values)
        names = {"rh_palm", *DRO_BENCH_LINK_PAIRS.values()}
        return {name: self._body_transform(name) for name in names}

    def world_body_transforms(
        self, joint_values: np.ndarray, palm_world: np.ndarray
    ) -> dict:
        """Place standalone Bench bodies under the exported ``rh_palm`` pose."""

        local = self.local_body_transforms(joint_values)
        root_world = np.asarray(palm_world, dtype=np.float64).reshape(4, 4)
        model_to_world = root_world @ np.linalg.inv(local["rh_palm"])
        return {name: model_to_world @ transform for name, transform in local.items()}

    def mesh(self, joint_values: np.ndarray, palm_world: np.ndarray) -> MeshData:
        """Build the Bench group-2 visual mesh under the exported palm pose."""

        self._forward(joint_values)
        local_palm = self._body_transform("rh_palm")
        model_to_world = (
            np.asarray(palm_world, dtype=np.float64).reshape(4, 4)
            @ np.linalg.inv(local_palm)
        )
        vertices = []
        faces = []
        vertex_offset = 0
        for item in self._geom_meshes:
            geom_transform = np.eye(4, dtype=np.float64)
            geom_transform[:3, :3] = self.data.geom_xmat[item.geom_id].reshape(3, 3)
            geom_transform[:3, 3] = self.data.geom_xpos[item.geom_id]
            transformed = _transform_points(
                item.vertices, model_to_world @ geom_transform
            )
            vertices.append(transformed)
            faces.append(item.faces + vertex_offset)
            vertex_offset += len(transformed)
        return MeshData(
            vertices=np.concatenate(vertices, axis=0).astype(np.float32),
            faces=np.concatenate(faces, axis=0).astype(np.int32),
            source_count=len(self._geom_meshes),
        )


def _rotation_error(left: np.ndarray, right: np.ndarray) -> float:
    def project(rotation):
        u, _, vt = np.linalg.svd(rotation)
        projected = u @ vt
        if np.linalg.det(projected) < 0.0:
            u[:, -1] *= -1.0
            projected = u @ vt
        return projected

    delta = project(left[:3, :3]).T @ project(right[:3, :3])
    cosine = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def compare_dro_bench_links(
    dro_model: ShadowHandModel,
    bench_model: BenchShadowHandModel,
    q: np.ndarray,
    bench_joint_values: np.ndarray,
    object_world: np.ndarray,
    palm_world: np.ndarray,
) -> tuple:
    """Compare palm/common-link origins with zero-pose frame alignment."""

    pairs = {"palm": "rh_palm", **DRO_BENCH_LINK_PAIRS}
    zero_dro = dro_model.link_transforms(np.zeros(len(DRO_SHADOW_Q_NAMES)))
    zero_bench = bench_model.local_body_transforms(
        np.zeros(len(BENCH_SHADOW_JOINT_NAMES))
    )
    zero_dro_palm_inverse = np.linalg.inv(zero_dro["palm"])
    zero_bench_palm_inverse = np.linalg.inv(zero_bench["rh_palm"])

    dro_object = dro_model.link_transforms(q)
    dro_world = {
        name: object_world @ transform for name, transform in dro_object.items()
    }
    bench_world = bench_model.world_body_transforms(
        bench_joint_values, palm_world
    )
    details = {}
    frames = {}
    max_position_error = -1.0
    max_rotation_error = -1.0
    worst_position_link = None
    worst_rotation_link = None
    for dro_link, bench_body in pairs.items():
        dro_zero_relative = zero_dro_palm_inverse @ zero_dro[dro_link]
        bench_zero_relative = zero_bench_palm_inverse @ zero_bench[bench_body]
        frame_alignment = (
            dro_zero_relative[:3, :3].T @ bench_zero_relative[:3, :3]
        )
        dro_aligned = dro_world[dro_link].copy()
        dro_aligned[:3, :3] = dro_aligned[:3, :3] @ frame_alignment
        bench_transform = bench_world[bench_body]
        position_error = float(
            np.linalg.norm(dro_aligned[:3, 3] - bench_transform[:3, 3])
        )
        rotation_error = _rotation_error(dro_aligned, bench_transform)
        details[dro_link] = {
            "bench_body": bench_body,
            "position_error_m": position_error,
            "rotation_error_rad": rotation_error,
        }
        frames[dro_link] = {
            "dro": matrix_to_pose_wxyz(dro_aligned),
            "bench": matrix_to_pose_wxyz(bench_transform),
        }
        if position_error > max_position_error:
            max_position_error = position_error
            worst_position_link = dro_link
        if rotation_error > max_rotation_error:
            max_rotation_error = rotation_error
            worst_rotation_link = dro_link
    return (
        {
            "max_link_position_error_m": max_position_error,
            "max_link_rotation_error_rad": max_rotation_error,
            "worst_link": worst_position_link,
            "worst_rotation_link": worst_rotation_link,
            "link_errors": details,
        },
        frames,
    )


class ViewerRun:
    """Read-only index and validator for one DRO DGN2k synthesis output root."""

    def __init__(
        self,
        output_root: Path,
        *,
        scene_root: Optional[Path] = None,
        shadow_urdf: Optional[Path] = None,
    ):
        self.output_root = Path(output_root).resolve(strict=True)
        if not self.output_root.is_dir():
            raise ValueError(f"output-root is not a directory: {self.output_root}")
        self.manifest_path = self.output_root / "run_manifest.json"
        self.resolved_config_path = self.output_root / "resolved_config.json"
        self.failure_manifest_path = self.output_root / "failure_manifest.json"
        self.manifest = _load_json_dict(self.manifest_path, "run manifest")
        self.resolved_config = _load_json_dict(
            self.resolved_config_path, "resolved config"
        )
        failure_manifest = _load_json_dict(
            self.failure_manifest_path, "failure manifest"
        )

        self.run_schema_version = self.manifest.get("schema_version")
        if self.run_schema_version not in SUPPORTED_RUN_SCHEMA_VERSIONS:
            raise ValueError("unsupported run manifest schema")
        if failure_manifest.get("schema_version") != self.run_schema_version:
            raise ValueError("failure manifest schema does not match run manifest")
        if self.manifest.get("status") not in {
            "completed",
            "completed_with_failures",
            "failed",
        }:
            raise ValueError("run manifest is not in a terminal state")
        if self.manifest.get("resolved_config") != self.resolved_config:
            raise ValueError("run_manifest resolved_config does not match resolved_config.json")
        if self.run_schema_version in (RUN_SCHEMA_VERSION, FILTERED_RUN_SCHEMA_VERSION):
            initialization = self.resolved_config.get("initialization")
            if (
                not isinstance(initialization, dict)
                or self.manifest.get("initialization_mode")
                != initialization.get("mode")
            ):
                raise ValueError("run manifest initialization mode does not match config")
        failures = failure_manifest.get("failures")
        if not isinstance(failures, list) or not all(
            isinstance(item, dict) for item in failures
        ):
            raise ValueError("failure manifest failures must be a list of objects")
        failure_by_scene = {}
        for failure in failures:
            scene_id = failure.get("scene_id")
            if not isinstance(scene_id, str) or scene_id in failure_by_scene:
                raise ValueError("failure manifest has missing or duplicate scene_id")
            failure_by_scene[scene_id] = failure

        scene_root_value = scene_root or self.resolved_config.get("scene_root")
        if scene_root_value is None:
            raise ValueError("scene-root is required because resolved config has no scene_root")
        try:
            self.scene_root = Path(scene_root_value).resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(
                "scene-root does not exist; pass --scene-root for the current machine: "
                f"{scene_root_value}"
            ) from error
        if not self.scene_root.is_dir():
            raise ValueError(f"scene-root is not a directory: {self.scene_root}")

        shadow_urdf_value = shadow_urdf or self.resolved_config.get("shadow_urdf")
        if shadow_urdf_value is None:
            raise ValueError(
                "shadow-urdf is required because resolved config has no shadow_urdf"
            )
        try:
            self.shadow_urdf = Path(shadow_urdf_value).resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(
                "Shadow URDF does not exist; pass --shadow-urdf for the current machine: "
                f"{shadow_urdf_value}"
            ) from error
        expected_urdf_hash = self.manifest.get("shadow_urdf_sha256")
        if not isinstance(expected_urdf_hash, str) or len(expected_urdf_hash) != 64:
            raise ValueError("run manifest is missing shadow_urdf_sha256")
        if sha256_file(self.shadow_urdf) != expected_urdf_hash:
            raise ValueError("Shadow URDF hash does not match the persisted run manifest")
        self.palm_fk_metadata = self.manifest.get("palm_fk")
        if self.palm_fk_metadata is None:
            self.palm_fk_chain = None
        else:
            from .runner import _validate_palm_fk_manifest

            _validate_palm_fk_manifest(self.palm_fk_metadata, self.shadow_urdf)
            self.palm_fk_chain = build_dro_shadow_pk_chain(
                self.shadow_urdf, device="cpu"
            )

        candidate_count = self.resolved_config.get("candidate_count")
        if (
            not isinstance(candidate_count, int)
            or isinstance(candidate_count, bool)
            or candidate_count <= 0
        ):
            raise ValueError("resolved candidate_count must be a positive integer")
        self.candidate_count = candidate_count

        scene_values = self.manifest.get("scenes")
        if not isinstance(scene_values, list):
            raise ValueError("run manifest scenes must be a list")
        scenes = {}
        for value in scene_values:
            if not isinstance(value, dict):
                raise ValueError("run manifest scene entries must be objects")
            scene_id = value.get("scene_id")
            status = value.get("status")
            if not isinstance(scene_id, str) or not scene_id or scene_id in scenes:
                raise ValueError("run manifest has missing or duplicate scene_id")
            if status not in {"completed", "failed"}:
                raise ValueError(f"unknown scene status for {scene_id}: {status!r}")
            scale = _as_float_scale(value.get("scale"), f"scene {scene_id} scale")
            failure = failure_by_scene.get(scene_id)
            if status == "completed":
                if failure is not None:
                    raise ValueError(
                        f"completed scene appears in failure manifest: {scene_id}"
                    )
                _resolve_inside(self.output_root, value.get("raw_artifact"), "raw artifact")
                _resolve_inside(
                    self.output_root, value.get("grasp_artifact"), "grasp artifact"
                )
            else:
                if failure is None:
                    raise ValueError(f"failed scene has no failure reason: {scene_id}")
                _resolve_inside(
                    self.output_root,
                    value.get("failed_raw_artifact"),
                    "failed raw artifact",
                )
            scenes[scene_id] = OutputScene(
                scene_id=scene_id,
                status=status,
                scale=scale,
                manifest=value,
                failure=failure,
            )
        if set(failure_by_scene) != {
            scene.scene_id for scene in scenes.values() if scene.status == "failed"
        }:
            raise ValueError("failure manifest and failed scene entries disagree")
        if self.manifest.get("scene_count") != len(scenes):
            raise ValueError("run manifest scene_count does not match scene entries")
        completed_count = sum(scene.status == "completed" for scene in scenes.values())
        failed_count = len(scenes) - completed_count
        expected_candidate_count = len(scenes) * self.candidate_count
        if self.manifest.get("candidate_count") != expected_candidate_count:
            raise ValueError("run manifest candidate_count accounting mismatch")
        if self.manifest.get("completed_candidate_count") != (
            completed_count * self.candidate_count
        ):
            raise ValueError("run manifest completed candidate accounting mismatch")
        if self.manifest.get("failed_candidate_count") != (
            failed_count * self.candidate_count
        ):
            raise ValueError("run manifest failed candidate accounting mismatch")
        for key in (
            "checkpoint_sha256",
            "source_scene_manifest_sha256",
        ):
            value = self.manifest.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"run manifest is missing {key}")
        self.scenes = scenes

    @property
    def completed_scene_ids(self) -> tuple:
        return tuple(
            scene_id
            for scene_id, scene in self.scenes.items()
            if scene.status == "completed"
        )

    @property
    def failed_scenes(self) -> tuple:
        return tuple(scene for scene in self.scenes.values() if scene.status == "failed")

    @property
    def scales(self) -> tuple:
        return tuple(sorted({self.scenes[scene_id].scale for scene_id in self.completed_scene_ids}))

    def scene_ids_for_scale(self, scale: Optional[float]) -> tuple:
        if scale is None:
            return self.completed_scene_ids
        requested = _as_float_scale(scale, "scale filter")
        return tuple(
            scene_id
            for scene_id in self.completed_scene_ids
            if math.isclose(
                self.scenes[scene_id].scale, requested, rel_tol=0.0, abs_tol=1e-12
            )
        )

    def _palm_object_transforms(self, stage_q: np.ndarray) -> np.ndarray:
        if self.palm_fk_chain is None:
            return legacy_dro_stage_q_to_object_palm_transforms(stage_q)
        return dro_stage_q_to_object_palm_transforms(
            self.palm_fk_chain,
            stage_q,
        )

    def _validate_scene_provenance(self, entry: OutputScene, record, raw: dict) -> None:
        expected = record.to_manifest()
        manifest_scene = entry.manifest
        raw_scene = raw.get("scene")
        if not isinstance(raw_scene, dict):
            raise ValueError(f"raw scene metadata is missing for {entry.scene_id}")
        for key in (
            "scene_id",
            "stored_scene_path",
            "scene_sha256",
            "object_id",
            "mesh_path_from_scene",
            "mesh_sha256",
        ):
            if manifest_scene.get(key) != expected[key]:
                raise ValueError(
                    f"run manifest {key} mismatch for {entry.scene_id}: "
                    f"expected {expected[key]!r}, got {manifest_scene.get(key)!r}"
                )
            if raw_scene.get(key) != expected[key]:
                raise ValueError(f"raw scene {key} mismatch for {entry.scene_id}")
        for source_name, source in (("run manifest", manifest_scene), ("raw scene", raw_scene)):
            if not math.isclose(
                _as_float_scale(source.get("scale"), f"{source_name} scale"),
                record.scale,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{source_name} scale mismatch for {entry.scene_id}")
            pose = np.asarray(source.get("object_pose_wxyz"), dtype=np.float64)
            if pose.shape != (7,) or not np.isfinite(pose).all():
                raise ValueError(
                    f"{source_name} object_pose_wxyz must be finite [xyz,wxyz] "
                    f"for {entry.scene_id}"
                )
            if not np.array_equal(pose, record.object_pose_wxyz):
                raise ValueError(
                    f"{source_name} object_pose_wxyz mismatch for {entry.scene_id}"
                )
        if self.run_schema_version in (RUN_SCHEMA_VERSION, FILTERED_RUN_SCHEMA_VERSION):
            for source_name, source in (
                ("run manifest", manifest_scene),
                ("raw scene", raw_scene),
            ):
                for key in (
                    "table_type",
                    "table_pose_wxyz",
                    "table_normal_local",
                    "table_normal_world",
                    "table_origin_world",
                ):
                    if source.get(key) != expected[key]:
                        raise ValueError(
                            f"{source_name} {key} mismatch for {entry.scene_id}"
                        )

    def load_scene(self, scene_id: str) -> LoadedScene:
        """Load and independently validate one completed scene without writing."""

        entry = self.scenes.get(scene_id)
        if entry is None:
            raise ValueError(f"unknown scene_id: {scene_id}")
        if entry.status != "completed":
            message = _failure_message(entry.failure)
            raise ValueError(
                f"scene {scene_id} failed synthesis and has no renderable poses"
                + (f": {message}" if message else "")
            )

        scene_path = self.scene_root / (scene_id + ".npy")
        try:
            record = load_scene_record(scene_path, self.scene_root)
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(f"invalid exact scene asset for {scene_id}: {error}") from error
        raw_path = _resolve_inside(
            self.output_root, entry.manifest.get("raw_artifact"), "raw artifact"
        )
        artifact_path = _resolve_inside(
            self.output_root, entry.manifest.get("grasp_artifact"), "grasp artifact"
        )
        raw = _load_numpy_dict(raw_path, "raw artifact")
        artifact = _load_numpy_dict(artifact_path, "grasp artifact")
        expected_raw_schema = (
            LEGACY_RAW_SCHEMA_VERSION
            if self.run_schema_version == LEGACY_RUN_SCHEMA_VERSION
            else FILTERED_RAW_SCHEMA_VERSION
            if self.run_schema_version == FILTERED_RUN_SCHEMA_VERSION
            else RAW_SCHEMA_VERSION
        )
        if raw.get("schema_version") != expected_raw_schema:
            raise ValueError(f"unsupported raw schema for {scene_id}")
        if raw.get("stage_names") != list(STAGE_NAMES):
            raise ValueError(f"stage order mismatch for {scene_id}")
        if raw.get("dro_q_names") != list(DRO_SHADOW_Q_NAMES):
            raise ValueError(f"DRO q order mismatch for {scene_id}")
        if raw.get("palm_fk") != self.palm_fk_metadata:
            raise ValueError(f"palm FK provenance mismatch for {scene_id}")
        self._validate_scene_provenance(entry, record, raw)
        if self.run_schema_version in (RUN_SCHEMA_VERSION, FILTERED_RUN_SCHEMA_VERSION):
            from .runner import _validate_v2_initialization_raw

            _validate_v2_initialization_raw(
                raw,
                record,
                self.resolved_config,
                self.candidate_count,
                require_all=True,
                proposal_indices=(
                    raw.get("selected_proposal_indices")
                    if self.run_schema_version == FILTERED_RUN_SCHEMA_VERSION
                    else None
                ),
            )

        points = np.asarray(raw.get("object_point_cloud"))
        if points.shape != (512, 3) or points.dtype != np.float32:
            raise ValueError(
                f"object point cloud must be float32 [512,3] for {scene_id}, "
                f"got {points.shape} {points.dtype}"
            )
        if not np.isfinite(points).all():
            raise ValueError(f"object point cloud contains non-finite values for {scene_id}")
        if sha256_array(points) != raw.get("object_point_cloud_sha256"):
            raise ValueError(f"object point-cloud hash mismatch for {scene_id}")

        expected_stage_shape = (
            self.candidate_count,
            len(STAGE_NAMES),
            len(DRO_SHADOW_Q_NAMES),
        )
        stage_q = np.asarray(raw.get("stage_q"))
        export_stage_q = np.asarray(raw.get("export_stage_q"))
        for label, value in (("stage_q", stage_q), ("export_stage_q", export_stage_q)):
            if value.shape != expected_stage_shape or value.dtype != np.float32:
                raise ValueError(
                    f"{label} must be float32 {expected_stage_shape} for {scene_id}, "
                    f"got {value.shape} {value.dtype}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"{label} contains non-finite values for {scene_id}")

        dro_limits = load_dro_shadow_finger_joint_limits(self.shadow_urdf)
        expected_export_q, expected_clamps = clamp_dro_shadow_export_stages(
            stage_q, dro_limits
        )
        if not np.array_equal(export_stage_q, expected_export_q):
            raise ValueError(
                f"export_stage_q does not match the approved clamp for {scene_id}"
            )
        if raw.get("export_clamp_diagnostics") != expected_clamps:
            raise ValueError(f"export clamp diagnostics mismatch for {scene_id}")

        palm_object_transforms = self._palm_object_transforms(export_stage_q)

        if set(artifact) != {"robot_pose", "joint_names", "scene_path"}:
            raise ValueError(f"grasp artifact keys do not match the Bench contract for {scene_id}")
        if artifact.get("joint_names") != list(BENCH_SHADOW_JOINT_NAMES):
            raise ValueError(f"Bench joint order mismatch for {scene_id}")
        expected_artifact, expected_excess = make_bench_artifact(
            export_stage_q,
            record.object_pose_wxyz,
            record.stored_scene_path,
            palm_object_transforms=palm_object_transforms,
        )
        robot_pose = np.asarray(artifact.get("robot_pose"))
        if robot_pose.shape != (
            1,
            self.candidate_count,
            len(STAGE_NAMES),
            7 + len(BENCH_SHADOW_JOINT_NAMES),
        ) or robot_pose.dtype != np.float32:
            raise ValueError(
                f"robot_pose must be float32 [1,{self.candidate_count},3,29] "
                f"for {scene_id}, got {robot_pose.shape} {robot_pose.dtype}"
            )
        if not np.isfinite(robot_pose).all():
            raise ValueError(f"robot_pose contains non-finite values for {scene_id}")
        if not np.array_equal(robot_pose, expected_artifact["robot_pose"]):
            raise ValueError(f"persisted robot_pose round-trip mismatch for {scene_id}")
        validate_artifact(
            artifact,
            record,
            export_stage_q,
            palm_object_transforms=palm_object_transforms,
        )
        persisted_excess = np.asarray(raw.get("bench_joint_limit_excess"))
        if (
            persisted_excess.shape != expected_excess.shape
            or persisted_excess.dtype != np.float32
            or not np.array_equal(persisted_excess, expected_excess)
        ):
            raise ValueError(f"Bench joint-limit diagnostics mismatch for {scene_id}")
        return LoadedScene(
            entry=entry,
            record=record,
            raw=raw,
            artifact=artifact,
            palm_object_transforms=palm_object_transforms,
            raw_path=raw_path,
            artifact_path=artifact_path,
        )

    def prepare_selection(
        self,
        hand_model: ShadowHandModel,
        scene_id: str,
        *,
        bench_model: Optional[BenchShadowHandModel] = None,
        candidate_index: int = 0,
        stage: str = "grasp",
        mode: str = "three_poses",
        pose_source: str = "exported",
    ) -> PreparedSelection:
        """Prepare one selection for Viser without mutating any input file."""

        if (
            not isinstance(candidate_index, int)
            or isinstance(candidate_index, bool)
            or not 0 <= candidate_index < self.candidate_count
        ):
            raise ValueError(
                f"candidate index must be in [0,{self.candidate_count - 1}], "
                f"got {candidate_index}"
            )
        if stage not in STAGE_NAMES:
            raise ValueError(f"unknown stage {stage!r}; expected one of {STAGE_NAMES}")
        if mode not in {"single_stage", "three_poses", "dro_bench_overlay"}:
            raise ValueError(
                "mode must be single_stage, three_poses, or dro_bench_overlay"
            )
        if mode == "dro_bench_overlay" and bench_model is None:
            raise ValueError("dro_bench_overlay mode requires --bench-mjcf")
        if pose_source not in {"exported", "raw"}:
            raise ValueError("pose_source must be exported or raw")
        loaded = self.load_scene(scene_id)
        record = loaded.record
        object_world = pose_wxyz_to_matrix(record.object_pose_wxyz)

        object_vertices, object_faces = _load_trimesh(record.mesh_path)
        object_vertices = object_vertices * record.scale
        object_mesh = MeshData(
            vertices=_transform_points(object_vertices, object_world),
            faces=object_faces.astype(np.int32),
            source_count=1,
        )
        points_world = _transform_points(
            np.asarray(loaded.raw["object_point_cloud"]), object_world
        )

        q_key = "export_stage_q" if pose_source == "exported" else "stage_q"
        q_values = np.asarray(loaded.raw[q_key])
        stage_names = STAGE_NAMES if mode == "three_poses" else (stage,)
        hand_meshes = {}
        dro_component_meshes = {}
        bench_hand_meshes = {}
        common_link_frames = {}
        palm_poses = {}
        link_diagnostics = []
        stage_diagnostics = []
        for stage_name in stage_names:
            stage_index = STAGE_NAMES.index(stage_name)
            q = q_values[candidate_index, stage_index]
            hand_meshes[stage_name] = hand_model.mesh(q, object_world)
            # The approved clamp changes finger joints only, so raw and exported
            # controller states share the same palm FK for a given stage.
            palm_object = loaded.palm_object_transforms[
                candidate_index, stage_index
            ]
            palm_world = object_world @ palm_object
            expected_pose, _ = dro_q_to_bench_pose(
                q,
                record.object_pose_wxyz,
                palm_object_transform=palm_object,
            )
            if not np.allclose(
                hand_model.palm_world_transform(q, object_world),
                pose_wxyz_to_matrix(expected_pose[:7]),
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(
                    f"Shadow URDF palm FK does not match T_WH for {scene_id} "
                    f"candidate {candidate_index} stage {stage_name}"
                )
            palm_pose = matrix_to_pose_wxyz(palm_world)
            if pose_source == "exported":
                persisted = loaded.artifact["robot_pose"][
                    0, candidate_index, stage_index, :7
                ]
                if not np.allclose(palm_pose, persisted, rtol=0.0, atol=1e-6):
                    raise ValueError(
                        f"rendered palm pose does not match persisted robot_pose for "
                        f"{scene_id} candidate {candidate_index} stage {stage_name}"
                    )
            palm_poses[stage_name] = palm_pose

            if mode == "dro_bench_overlay":
                dro_component_meshes.setdefault("forearm", {})[stage_name] = (
                    hand_model.mesh(q, object_world, include_links={"forearm"})
                )
                dro_component_meshes.setdefault("wrist", {})[stage_name] = (
                    hand_model.mesh(q, object_world, include_links={"wrist"})
                )
                dro_component_meshes.setdefault("palm_fingers", {})[stage_name] = (
                    hand_model.mesh(
                        q,
                        object_world,
                        exclude_links={"forearm", "wrist"},
                    )
                )
                if pose_source == "exported":
                    bench_joint_values = np.asarray(loaded.artifact["robot_pose"])[
                        0, candidate_index, stage_index, 7:
                    ]
                    bench_palm_world = pose_wxyz_to_matrix(
                        np.asarray(loaded.artifact["robot_pose"])[
                            0, candidate_index, stage_index, :7
                        ]
                    )
                else:
                    bench_joint_values, _ = map_dro_shadow_fingers(q)
                    bench_palm_world = palm_world
                bench_hand_meshes[stage_name] = bench_model.mesh(
                    bench_joint_values,
                    bench_palm_world,
                )
                alignment, frames = compare_dro_bench_links(
                    hand_model,
                    bench_model,
                    q,
                    bench_joint_values,
                    object_world,
                    bench_palm_world,
                )
                alignment["stage_name"] = stage_name
                link_diagnostics.append(alignment)
                common_link_frames[stage_name] = frames

        if not np.allclose(
            np.stack(list(palm_poses.values())),
            next(iter(palm_poses.values())),
            rtol=0.0,
            atol=1e-5,
        ):
            raise ValueError(
                f"selected {pose_source} palm pose changes across controller stages"
            )

        selected_stage_indices = {STAGE_NAMES.index(name) for name in stage_names}
        for item in loaded.raw["export_clamp_diagnostics"]:
            if (
                item.get("candidate_index") == candidate_index
                and item.get("stage_index") in selected_stage_indices
            ):
                stage_diagnostics.append(item)
        max_delta = max(
            (abs(float(item["delta"])) for item in stage_diagnostics), default=0.0
        )
        max_link_position_error = max(
            (
                float(item["max_link_position_error_m"])
                for item in link_diagnostics
            ),
            default=None,
        )
        max_link_rotation_error = max(
            (
                float(item["max_link_rotation_error_rad"])
                for item in link_diagnostics
            ),
            default=None,
        )
        worst_link = None
        if link_diagnostics:
            worst_item = max(
                link_diagnostics,
                key=lambda item: float(item["max_link_position_error_m"]),
            )
            worst_link = worst_item["worst_link"]
        diagnostics = {
            "scene_id": scene_id,
            "object_id": record.object_id,
            "scale": record.scale,
            "candidate_index": candidate_index,
            "selected_stage": stage,
            "displayed_stages": list(stage_names),
            "pose_source": pose_source,
            "pose_source_label": (
                "Bench artifact / export_stage_q"
                if pose_source == "exported"
                else "raw controller diagnostic (not Bench artifact)"
            ),
            "frame_contract": "T_WH = T_WO @ T_OPalm; xyz metres; quaternion wxyz",
            "stage_contract": "q_outer -> pregrasp, q -> grasp, q_inner -> squeeze",
            "mesh_path": str(record.mesh_path),
            "mesh_sha256": record.mesh_sha256,
            "scene_path": str(record.scene_path),
            "scene_sha256": record.scene_sha256,
            "raw_artifact": str(loaded.raw_path),
            "grasp_artifact": str(loaded.artifact_path),
            "clamp_count": len(stage_diagnostics),
            "clamp_max_abs_delta": max_delta,
            "clamps": stage_diagnostics,
            "palm_fk": self.palm_fk_metadata or {
                "backend": "legacy_hand_written_compatibility"
            },
            "bench_mjcf": (
                str(bench_model.mjcf_path) if bench_model is not None else None
            ),
            "max_link_position_error_m": max_link_position_error,
            "max_link_rotation_error_rad": max_link_rotation_error,
            "worst_link": worst_link,
            "link_alignment": link_diagnostics,
            "adapter_source": self.manifest.get("source"),
            "checkpoint_sha256": self.manifest.get("checkpoint_sha256"),
            "source_scene_manifest_sha256": self.manifest.get(
                "source_scene_manifest_sha256"
            ),
            "resolved_config": str(self.resolved_config_path),
            "failed_scene_count": len(self.failed_scenes),
        }
        return PreparedSelection(
            scene_id=scene_id,
            candidate_index=candidate_index,
            selected_stage=stage,
            stage_names=tuple(stage_names),
            pose_source=pose_source,
            object_mesh=object_mesh,
            object_point_cloud_world=points_world,
            hand_meshes=hand_meshes,
            dro_component_meshes=dro_component_meshes,
            bench_hand_meshes=bench_hand_meshes,
            common_link_frames=common_link_frames,
            object_pose_wxyz=record.object_pose_wxyz.copy(),
            palm_poses_wxyz=palm_poses,
            diagnostics=diagnostics,
        )


def diagnostics_markdown(prepared: PreparedSelection, run: ViewerRun) -> str:
    """Format the current read-only selection for the Viser sidebar."""

    value = prepared.diagnostics
    source = value.get("adapter_source") or {}
    commit = source.get("commit", "unknown") if isinstance(source, dict) else "unknown"
    failed_lines = []
    for scene in run.failed_scenes[:5]:
        failure = scene.failure or {}
        failed_lines.append(
            f"- `{scene.scene_id}`: {failure.get('stage', 'unknown')} — "
            f"{_failure_message(failure) or 'no message'}"
        )
    failed_text = "\n".join(failed_lines) if failed_lines else "- none"
    if value["max_link_position_error_m"] is None:
        alignment_text = "- DRO/Bench overlay: not selected\n"
    else:
        alignment_text = (
            "- DRO/Bench max common-link position / rotation error: "
            f"`{value['max_link_position_error_m']:.9g} m` / "
            f"`{value['max_link_rotation_error_rad']:.9g} rad`\n"
            f"- Worst position link: `{value['worst_link']}`\n"
        )
    return (
        "### Current selection\n"
        f"- Scene: `{value['scene_id']}`\n"
        f"- Object / scale: `{value['object_id']}` / `{value['scale']:.9g}`\n"
        f"- Candidate / stage: `{value['candidate_index']}` / `{value['selected_stage']}`\n"
        f"- Displayed: `{', '.join(value['displayed_stages'])}`\n"
        f"- Pose source: **{value['pose_source_label']}**\n"
        f"- Export clamp count / max |delta|: `{value['clamp_count']}` / "
        f"`{value['clamp_max_abs_delta']:.9g} rad`\n"
        f"- Mesh: `{value['mesh_path']}`\n"
        f"- Raw: `{value['raw_artifact']}`\n"
        f"- Bench artifact: `{value['grasp_artifact']}`\n"
        f"- Adapter commit: `{commit}`\n"
        f"- Frame: `{value['frame_contract']}`\n"
        f"{alignment_text}"
        "\n### Failed scenes (not renderable)\n"
        f"{failed_text}"
    )
