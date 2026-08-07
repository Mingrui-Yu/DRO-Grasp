"""Final-pose tabletop filtering from the released Shadow URDF collisions."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import (
    DRO_SHADOW_Q_NAMES,
    pose_wxyz_to_matrix,
    sha256_file,
)


def _parse_vector(value: str | None, size: int, *, default, label: str) -> np.ndarray:
    """Parse one finite fixed-size URDF vector."""

    text = value if value is not None else default
    try:
        result = np.asarray([float(item) for item in text.split()], dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"{label} must contain finite numbers") from error
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {size} finite numbers")
    return result


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _urdf_origin_transform(collision: ET.Element, *, label: str) -> np.ndarray:
    """Return the URDF collision origin as a homogeneous link transform."""

    origin = collision.find("origin")
    xyz = _parse_vector(
        None if origin is None else origin.get("xyz"),
        3,
        default="0 0 0",
        label=f"{label} origin xyz",
    )
    rpy = _parse_vector(
        None if origin is None else origin.get("rpy"),
        3,
        default="0 0 0",
        label=f"{label} origin rpy",
    )
    transform = np.eye(4, dtype=np.float64)
    # URDF rpy uses fixed-axis roll, pitch, yaw: Rz(yaw) Ry(pitch) Rx(roll).
    transform[:3, :3] = _rotation_z(rpy[2]) @ _rotation_y(rpy[1]) @ _rotation_x(rpy[0])
    transform[:3, 3] = xyz
    return transform


def _load_mesh_vertices(mesh_path: Path) -> np.ndarray:
    """Load all mesh vertices with scene-node transforms applied."""

    try:
        import trimesh
    except ImportError as error:
        raise RuntimeError("trimesh is required for URDF collision meshes") from error

    loaded = trimesh.load(str(mesh_path), force="scene", process=False)
    meshes = []
    for node_name in loaded.graph.nodes_geometry:
        transform, geometry_name = loaded.graph[node_name]
        geometry = loaded.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        copy = geometry.copy()
        copy.apply_transform(transform)
        meshes.append(copy)
    if not meshes:
        raise ValueError(f"collision mesh contains no triangle geometry: {mesh_path}")
    vertices = np.concatenate(
        [np.asarray(mesh.vertices, dtype=np.float64) for mesh in meshes], axis=0
    )
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f"collision mesh vertices are invalid: {mesh_path}")
    return vertices


@dataclass(frozen=True)
class CollisionGeometry:
    """One exact URDF collision shape attached to a palm-scope link."""

    link_name: str
    collision_index: int
    geometry_type: str
    link_from_collision: np.ndarray
    half_extents: np.ndarray | None = None
    radius: float | None = None
    half_length: float | None = None
    vertices: np.ndarray | None = None
    provenance: dict | None = None

    def minimum_signed_height(
        self,
        world_from_link: np.ndarray,
        table_origin_world: np.ndarray,
        table_normal_world: np.ndarray,
    ) -> np.ndarray:
        """Return the exact minimum plane height for each batched link pose."""

        link_transforms = np.asarray(world_from_link, dtype=np.float64)
        if link_transforms.ndim != 3 or link_transforms.shape[1:] != (4, 4):
            raise ValueError("world_from_link must have shape [N,4,4]")
        world_from_collision = link_transforms @ self.link_from_collision
        rotations = world_from_collision[:, :3, :3]
        centers = world_from_collision[:, :3, 3]
        normal = np.asarray(table_normal_world, dtype=np.float64).reshape(3)
        origin = np.asarray(table_origin_world, dtype=np.float64).reshape(3)
        normal_collision = np.einsum("nji,j->ni", rotations, normal)
        center_height = (centers - origin) @ normal

        if self.geometry_type == "box":
            support = np.abs(normal_collision) @ self.half_extents
            heights = center_height - support
        elif self.geometry_type == "sphere":
            heights = center_height - float(self.radius)
        elif self.geometry_type == "cylinder":
            radial = float(self.radius) * np.linalg.norm(normal_collision[:, :2], axis=1)
            axial = float(self.half_length) * np.abs(normal_collision[:, 2])
            heights = center_height - radial - axial
        elif self.geometry_type == "mesh":
            offsets = normal_collision @ self.vertices.T
            heights = center_height + offsets.min(axis=1)
        else:
            raise ValueError(f"unsupported collision geometry: {self.geometry_type}")
        if not np.isfinite(heights).all():
            raise ValueError("collision signed heights contain non-finite values")
        return heights


class TabletopCollisionModel:
    """Palm-and-descendants collision geometry for final-pose table filtering."""

    def __init__(
        self,
        *,
        urdf_path: Path,
        root_link: str,
        scoped_links: tuple[str, ...],
        geometries: tuple[CollisionGeometry, ...],
    ):
        self.urdf_path = Path(urdf_path).resolve(strict=True)
        self.root_link = root_link
        self.scoped_links = scoped_links
        self.geometries = geometries
        if not self.geometries:
            raise ValueError(f"{root_link} subtree has no URDF collision geometry")

    @classmethod
    def from_urdf(cls, urdf_path: Path, *, root_link: str = "palm"):
        """Parse collision geometry for root_link and all kinematic descendants."""

        urdf_path = Path(urdf_path).resolve(strict=True)
        root = ET.parse(urdf_path).getroot()
        link_elements = root.findall("link")
        link_names = [link.get("name") for link in link_elements]
        if any(not name for name in link_names) or len(set(link_names)) != len(link_names):
            raise ValueError("URDF link names must be present and unique")
        if root_link not in link_names:
            raise ValueError(f"URDF has no requested filter root link: {root_link}")

        children = {name: [] for name in link_names}
        for joint in root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            parent_name = None if parent is None else parent.get("link")
            child_name = None if child is None else child.get("link")
            if parent_name not in children or child_name not in children:
                raise ValueError("URDF joint references an unknown parent or child link")
            children[parent_name].append(child_name)

        scoped = set()
        pending = [root_link]
        while pending:
            link_name = pending.pop()
            if link_name in scoped:
                raise ValueError("URDF link hierarchy contains a cycle")
            scoped.add(link_name)
            pending.extend(children[link_name])
        scoped_links = tuple(name for name in link_names if name in scoped)

        geometries = []
        for link in link_elements:
            link_name = link.get("name")
            if link_name not in scoped:
                continue
            for collision_index, collision in enumerate(link.findall("collision")):
                label = f"{link_name} collision {collision_index}"
                geometry_parent = collision.find("geometry")
                geometry_children = [] if geometry_parent is None else list(geometry_parent)
                if len(geometry_children) != 1:
                    raise ValueError(f"{label} must contain exactly one geometry")
                geometry = geometry_children[0]
                origin = _urdf_origin_transform(collision, label=label)
                geometry_type = geometry.tag.rsplit("}", 1)[-1]
                kwargs = {
                    "link_name": link_name,
                    "collision_index": collision_index,
                    "geometry_type": geometry_type,
                    "link_from_collision": origin,
                }
                provenance = {
                    "link_name": link_name,
                    "collision_index": collision_index,
                    "geometry_type": geometry_type,
                    "origin": origin.tolist(),
                }
                if geometry_type == "box":
                    size = _parse_vector(
                        geometry.get("size"), 3, default="", label=f"{label} box size"
                    )
                    if np.any(size <= 0.0):
                        raise ValueError(f"{label} box size must be positive")
                    kwargs["half_extents"] = size / 2.0
                    provenance["size"] = size.tolist()
                elif geometry_type == "sphere":
                    radius = float(geometry.get("radius", "nan"))
                    if not np.isfinite(radius) or radius <= 0.0:
                        raise ValueError(f"{label} sphere radius must be positive")
                    kwargs["radius"] = radius
                    provenance["radius"] = radius
                elif geometry_type == "cylinder":
                    radius = float(geometry.get("radius", "nan"))
                    length = float(geometry.get("length", "nan"))
                    if (
                        not np.isfinite(radius)
                        or not np.isfinite(length)
                        or radius <= 0.0
                        or length <= 0.0
                    ):
                        raise ValueError(f"{label} cylinder dimensions must be positive")
                    kwargs["radius"] = radius
                    kwargs["half_length"] = length / 2.0
                    provenance.update({"radius": radius, "length": length})
                elif geometry_type == "mesh":
                    filename = geometry.get("filename")
                    if not isinstance(filename, str) or not filename:
                        raise ValueError(f"{label} mesh filename is missing")
                    if filename.startswith("package://") or filename.startswith("file://"):
                        raise ValueError(f"{label} mesh URI must be repository-relative")
                    mesh_path = (urdf_path.parent / filename).resolve(strict=True)
                    scale = _parse_vector(
                        geometry.get("scale"), 3, default="1 1 1", label=f"{label} mesh scale"
                    )
                    if np.any(scale <= 0.0):
                        raise ValueError(f"{label} mesh scale must be positive")
                    vertices = _load_mesh_vertices(mesh_path) * scale
                    kwargs["vertices"] = vertices
                    provenance.update(
                        {
                            "filename": filename,
                            "mesh_sha256": sha256_file(mesh_path),
                            "scale": scale.tolist(),
                            "vertex_count": int(vertices.shape[0]),
                        }
                    )
                else:
                    raise ValueError(f"{label} uses unsupported geometry {geometry_type}")
                kwargs["provenance"] = provenance
                geometries.append(CollisionGeometry(**kwargs))

        return cls(
            urdf_path=urdf_path,
            root_link=root_link,
            scoped_links=scoped_links,
            geometries=tuple(geometries),
        )

    def to_manifest(self) -> dict:
        """Return stable, JSON-compatible collision-scope provenance."""

        geometry_counts = {}
        for geometry in self.geometries:
            geometry_counts[geometry.geometry_type] = (
                geometry_counts.get(geometry.geometry_type, 0) + 1
            )
        return {
            "backend": "urdf_collision_geometry_signed_plane_height",
            "urdf_sha256": sha256_file(self.urdf_path),
            "root_link": self.root_link,
            "scoped_links": list(self.scoped_links),
            "collision_links": list(dict.fromkeys(item.link_name for item in self.geometries)),
            "collision_geometry_count": len(self.geometries),
            "geometry_counts": dict(sorted(geometry_counts.items())),
            "geometries": [item.provenance for item in self.geometries],
        }

    def evaluate_final_grasps(
        self,
        pk_chain,
        grasp_q: np.ndarray,
        record,
        *,
        margin: float = 0.0,
    ) -> tuple[np.ndarray, list[dict]]:
        """Check final grasp q against the explicit scene table plane."""

        try:
            import torch
        except ImportError as error:
            raise RuntimeError("torch is required for DRO tabletop collision FK") from error

        q = np.asarray(grasp_q)
        if q.ndim != 2 or q.shape[1] != len(DRO_SHADOW_Q_NAMES):
            raise ValueError(
                f"grasp_q must have shape [N,{len(DRO_SHADOW_Q_NAMES)}], got {q.shape}"
            )
        if not np.issubdtype(q.dtype, np.floating) or not np.isfinite(q).all():
            raise ValueError("grasp_q must contain finite floating-point values")
        if not isinstance(margin, (int, float)) or isinstance(margin, bool):
            raise ValueError("table margin must be numeric")
        margin = float(margin)
        if not np.isfinite(margin):
            raise ValueError("table margin must be finite")
        if tuple(pk_chain.get_joint_parameter_names()) != DRO_SHADOW_Q_NAMES:
            raise ValueError("DRO PK chain q order does not match the table filter contract")
        chain_links = set(pk_chain.get_link_names())
        missing_links = [name for name in self.scoped_links if name not in chain_links]
        if missing_links:
            raise ValueError(f"DRO PK chain is missing table-filter links: {missing_links}")

        tensor_q = torch.as_tensor(q, dtype=pk_chain.dtype, device=pk_chain.device)
        with torch.no_grad():
            frames = pk_chain.forward_kinematics(tensor_q)
        world_from_object = pose_wxyz_to_matrix(record.object_pose_wxyz)
        table_origin = np.asarray(record.table_origin_world, dtype=np.float64).reshape(3)
        table_normal = np.asarray(record.table_normal_world, dtype=np.float64).reshape(3)
        normal_norm = np.linalg.norm(table_normal)
        if not np.isfinite(table_normal).all() or not np.isclose(
            normal_norm, 1.0, rtol=0.0, atol=1e-7
        ):
            raise ValueError("table normal must be finite and normalized")

        minimum = np.full((q.shape[0],), np.inf, dtype=np.float64)
        worst_link = np.full((q.shape[0],), "", dtype=object)
        worst_collision_index = np.full((q.shape[0],), -1, dtype=np.int64)
        worst_geometry_type = np.full((q.shape[0],), "", dtype=object)
        for geometry in self.geometries:
            object_from_link = (
                frames[geometry.link_name].get_matrix().detach().cpu().numpy().astype(np.float64)
            )
            world_from_link = world_from_object[None, :, :] @ object_from_link
            heights = geometry.minimum_signed_height(
                world_from_link, table_origin, table_normal
            )
            changed = heights < minimum
            minimum[changed] = heights[changed]
            worst_link[changed] = geometry.link_name
            worst_collision_index[changed] = geometry.collision_index
            worst_geometry_type[changed] = geometry.geometry_type

        passed = minimum > margin
        diagnostics = [
            {
                "candidate_index": candidate_index,
                "status": "table_collision_free" if passed[candidate_index] else "table_rejected",
                "passed": bool(passed[candidate_index]),
                "min_z": float(minimum[candidate_index]),
                "margin": margin,
                "worst_link": str(worst_link[candidate_index]),
                "worst_collision_index": int(worst_collision_index[candidate_index]),
                "worst_geometry_type": str(worst_geometry_type[candidate_index]),
            }
            for candidate_index in range(q.shape[0])
        ]
        return passed, diagnostics
