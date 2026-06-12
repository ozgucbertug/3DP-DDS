"""Headless manufacturability and geometry metrics for TriangleMesh objects.

All public functions are also accessible via ``dds.geometry`` (the canonical
user-facing namespace), which re-exports them from ``geometry/__init__.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .geometry._utils import load_trimesh
from .geometry.mesh import TriangleMesh
from .utils import normalize_axis

# ---------------------------------------------------------------------------
# Public batch data-holder and factory
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FaceData:
    """Pre-computed per-face geometry for a consistently oriented mesh.

    Build once with :func:`compute_face_data` and pass as ``precomputed=`` to
    any per-face metric function to avoid re-orienting the mesh multiple times.
    """

    mesh: TriangleMesh
    normals: npt.NDArray[np.float64]
    areas: npt.NDArray[np.float64]
    centroids: npt.NDArray[np.float64]


# Private aliases for modules that import the old names directly.
_OrientedFaceData = FaceData


def _oriented_mesh(mesh: TriangleMesh) -> TriangleMesh:
    """Return a mesh with consistent outward winding when possible."""

    if mesh.is_empty:
        return mesh
    tri_mesh = mesh.to_trimesh()
    trimesh = load_trimesh()
    trimesh.repair.fix_normals(tri_mesh)
    if tri_mesh.is_watertight and not tri_mesh.is_volume:
        tri_mesh.invert()
        trimesh.repair.fix_normals(tri_mesh)
    return TriangleMesh.from_trimesh(tri_mesh)


def compute_face_data(mesh: TriangleMesh) -> FaceData:
    """Orient *mesh* once and return reusable per-face normals, areas, and centroids.

    Pass the returned :class:`FaceData` as ``precomputed=`` to any per-face
    metric function to skip the (potentially expensive) mesh-orientation step.
    """

    oriented = _oriented_mesh(mesh)
    if oriented.is_empty:
        return FaceData(
            mesh=oriented,
            normals=np.empty((0, 3), dtype=float),
            areas=np.empty((0,), dtype=float),
            centroids=np.empty((0, 3), dtype=float),
        )

    tri_mesh = oriented.to_trimesh()
    return FaceData(
        mesh=oriented,
        normals=np.asarray(tri_mesh.face_normals, dtype=float),
        areas=np.asarray(tri_mesh.area_faces, dtype=float),
        centroids=np.asarray(tri_mesh.triangles_center, dtype=float),
    )


# Private alias used by internal modules.
_oriented_face_data = compute_face_data


def _overhang_angles_from_normals(
    normals: npt.NDArray[np.float64],
    build_direction: tuple[float, float, float] | npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    if normals.size == 0:
        return np.empty((0,), dtype=float)
    downward = -np.asarray(normalize_axis(build_direction, "build_direction"), dtype=float)
    cosine = np.clip(normals @ downward, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


# ---------------------------------------------------------------------------
# Public per-face metric functions — all accept an optional precomputed=
# ---------------------------------------------------------------------------

def face_normals(
    mesh: TriangleMesh,
    *,
    precomputed: FaceData | None = None,
) -> npt.NDArray[np.float64]:
    """Return one outward-facing normal per face."""

    data = precomputed if precomputed is not None else compute_face_data(mesh)
    return data.normals


def vertex_normals(
    mesh: TriangleMesh,
    *,
    precomputed: FaceData | None = None,
) -> npt.NDArray[np.float64]:
    """Return vertex normals computed by trimesh."""

    data = precomputed if precomputed is not None else compute_face_data(mesh)
    if data.mesh.is_empty:
        return np.empty((0, 3), dtype=float)
    return np.asarray(data.mesh.to_trimesh().vertex_normals, dtype=float)


def face_centroids(
    mesh: TriangleMesh,
    *,
    precomputed: FaceData | None = None,
) -> npt.NDArray[np.float64]:
    """Return one centroid per face."""

    if precomputed is not None:
        return precomputed.centroids
    if mesh.is_empty:
        return np.empty((0, 3), dtype=float)
    return np.asarray(mesh.to_trimesh().triangles_center, dtype=float)


def face_areas(
    mesh: TriangleMesh,
    *,
    precomputed: FaceData | None = None,
) -> npt.NDArray[np.float64]:
    """Return one triangle area per face."""

    if precomputed is not None:
        return precomputed.areas
    if mesh.is_empty:
        return np.empty((0,), dtype=float)
    return np.asarray(mesh.to_trimesh().area_faces, dtype=float)


def overhang_angles(
    mesh: TriangleMesh,
    *,
    build_direction: tuple[float, float, float] | npt.ArrayLike = (0.0, 0.0, 1.0),
    precomputed: FaceData | None = None,
) -> npt.NDArray[np.float64]:
    """Measure face overhang angle relative to the downward build direction."""

    data = precomputed if precomputed is not None else compute_face_data(mesh)
    return _overhang_angles_from_normals(data.normals, build_direction)


def downfacing_mask(
    mesh: TriangleMesh,
    *,
    build_direction: tuple[float, float, float] | npt.ArrayLike = (0.0, 0.0, 1.0),
    precomputed: FaceData | None = None,
) -> npt.NDArray[np.bool_]:
    """Return faces whose normals have a downward component."""

    angles = overhang_angles(mesh, build_direction=build_direction, precomputed=precomputed)
    return angles < 90.0


def support_risk_mask(
    mesh: TriangleMesh,
    *,
    build_direction: tuple[float, float, float] | npt.ArrayLike = (0.0, 0.0, 1.0),
    critical_angle_deg: float = 45.0,
    precomputed: FaceData | None = None,
) -> npt.NDArray[np.bool_]:
    """Return faces below the chosen overhang critical angle."""

    if critical_angle_deg < 0.0:
        raise ValueError("critical_angle_deg must be non-negative.")
    angles = overhang_angles(mesh, build_direction=build_direction, precomputed=precomputed)
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
    lower, upper = np.asarray(mesh.to_trimesh().bounds, dtype=float)
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

    if mesh.is_empty:
        return 0.0
    return float(mesh.to_trimesh().area)


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
