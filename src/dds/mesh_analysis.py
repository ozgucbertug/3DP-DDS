"""Headless manufacturability and geometry metrics for TriangleMesh objects."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
import numpy.typing as npt

from .geometry.mesh import TriangleMesh
from .utils import EPSILON, ensure_finite_triplet


def _load_trimesh() -> Any:
    try:
        return import_module("trimesh")
    except ImportError as exc:
        raise ImportError("trimesh is required for mesh analysis. Install `3dp-dds`.") from exc


def _normalized_build_direction(
    build_direction: tuple[float, float, float] | npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    vector = np.asarray(ensure_finite_triplet(build_direction, "build_direction"), dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= EPSILON:
        raise ValueError("build_direction must not be the zero vector.")
    return vector / norm


def _oriented_mesh(mesh: TriangleMesh) -> TriangleMesh:
    """Return a mesh with consistent outward winding when possible."""

    if mesh.is_empty:
        return mesh
    tri_mesh = mesh.to_trimesh()
    trimesh = _load_trimesh()
    trimesh.repair.fix_normals(tri_mesh)
    if tri_mesh.is_watertight and not tri_mesh.is_volume:
        tri_mesh.invert()
        trimesh.repair.fix_normals(tri_mesh)
    return TriangleMesh.from_trimesh(tri_mesh, metadata=mesh.metadata)


def face_normals(mesh: TriangleMesh) -> npt.NDArray[np.float64]:
    """Return one outward-facing normal per face."""

    oriented = _oriented_mesh(mesh)
    if oriented.is_empty:
        return np.empty((0, 3), dtype=float)
    vertices = oriented.vertices
    triangles = vertices[oriented.faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > EPSILON
    normals[valid] /= lengths[valid, np.newaxis]
    normals[~valid] = 0.0
    return normals


def vertex_normals(mesh: TriangleMesh) -> npt.NDArray[np.float64]:
    """Return area-weighted vertex normals."""

    oriented = _oriented_mesh(mesh)
    if oriented.is_empty:
        return np.empty((0, 3), dtype=float)
    normals = np.zeros_like(oriented.vertices, dtype=float)
    face_normal_values = face_normals(oriented)
    face_area_values = face_areas(oriented)
    for face_index, face in enumerate(oriented.faces):
        normals[face] += face_normal_values[face_index] * face_area_values[face_index]
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > EPSILON
    normals[valid] /= lengths[valid, np.newaxis]
    normals[~valid] = 0.0
    return normals


def face_centroids(mesh: TriangleMesh) -> npt.NDArray[np.float64]:
    """Return one centroid per face."""

    if mesh.is_empty:
        return np.empty((0, 3), dtype=float)
    triangles = mesh.vertices[mesh.faces]
    return np.mean(triangles, axis=1)


def face_areas(mesh: TriangleMesh) -> npt.NDArray[np.float64]:
    """Return one triangle area per face."""

    if mesh.is_empty:
        return np.empty((0,), dtype=float)
    triangles = mesh.vertices[mesh.faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def overhang_angles(
    mesh: TriangleMesh,
    *,
    build_direction: tuple[float, float, float] | npt.ArrayLike = (0.0, 0.0, 1.0),
) -> npt.NDArray[np.float64]:
    """Measure face overhang angle relative to the downward build direction."""

    normals = face_normals(mesh)
    if normals.size == 0:
        return np.empty((0,), dtype=float)
    downward = -_normalized_build_direction(build_direction)
    cosine = np.clip(normals @ downward, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def downfacing_mask(
    mesh: TriangleMesh,
    *,
    build_direction: tuple[float, float, float] | npt.ArrayLike = (0.0, 0.0, 1.0),
) -> npt.NDArray[np.bool_]:
    """Return faces whose normals have a downward component."""

    angles = overhang_angles(mesh, build_direction=build_direction)
    return angles < 90.0


def support_risk_mask(
    mesh: TriangleMesh,
    *,
    build_direction: tuple[float, float, float] | npt.ArrayLike = (0.0, 0.0, 1.0),
    critical_angle_deg: float = 45.0,
) -> npt.NDArray[np.bool_]:
    """Return faces below the chosen overhang critical angle."""

    if critical_angle_deg < 0.0:
        raise ValueError("critical_angle_deg must be non-negative.")
    angles = overhang_angles(mesh, build_direction=build_direction)
    return angles <= float(critical_angle_deg)


def normal_rgb_from_normals(normals: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """Map `x, y, z` normal components from `[-1, 1]` to `R, G, B` bytes."""

    normal_array = np.asarray(normals, dtype=float)
    if normal_array.ndim != 2 or normal_array.shape[1] != 3:
        raise ValueError("normals must have shape `(n, 3)`.")
    return np.rint((np.clip(normal_array, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)


def mesh_bounds_stats(mesh: TriangleMesh) -> dict[str, float]:
    """Return basic axis-aligned mesh bounds statistics."""

    if mesh.is_empty:
        return {
            "xmin": 0.0,
            "xmax": 0.0,
            "ymin": 0.0,
            "ymax": 0.0,
            "zmin": 0.0,
            "zmax": 0.0,
            "dx": 0.0,
            "dy": 0.0,
            "dz": 0.0,
        }
    lower, upper = mesh.bounds
    return {
        "xmin": float(lower[0]),
        "xmax": float(upper[0]),
        "ymin": float(lower[1]),
        "ymax": float(upper[1]),
        "zmin": float(lower[2]),
        "zmax": float(upper[2]),
        "dx": float(upper[0] - lower[0]),
        "dy": float(upper[1] - lower[1]),
        "dz": float(upper[2] - lower[2]),
    }


def mesh_surface_area(mesh: TriangleMesh) -> float:
    """Return the total triangle area of a mesh."""

    return float(np.sum(face_areas(mesh)))


def mesh_volume_estimate(mesh: TriangleMesh) -> float | None:
    """Return the enclosed volume when the mesh is watertight."""

    if mesh.is_empty:
        return 0.0
    tri_mesh = mesh.to_trimesh()
    if not tri_mesh.is_watertight:
        return None
    if not tri_mesh.is_volume:
        tri_mesh.invert()
    return float(abs(tri_mesh.volume))
