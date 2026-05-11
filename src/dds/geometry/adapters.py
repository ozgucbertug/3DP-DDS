"""Mesh IO and conversions between meshes, dense fields, and SDF wrappers."""

from __future__ import annotations

import warnings
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ..domain import Domain
from .mesh import TriangleMesh
from .sdf import GridSDF3, SDF3


def _load_meshio() -> Any:
    try:
        return import_module("meshio")
    except ImportError as exc:
        raise ImportError("meshio is required for mesh file IO. Install `3dp-dds`.") from exc


def _load_trimesh() -> Any:
    try:
        return import_module("trimesh")
    except ImportError as exc:
        raise ImportError("trimesh is required for mesh/SDF conversion. Install `3dp-dds`.") from exc


def _load_scipy_ndimage() -> Any:
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise ImportError("SciPy is required for occupancy-to-SDF conversion. Install `3dp-dds`.") from exc
    return ndimage


def _validate_field_shape(
    domain: Domain,
    values: npt.ArrayLike,
    *,
    field_name: str,
) -> npt.NDArray[np.float64]:
    array = np.asarray(values)
    if array.shape != domain.grid_shape:
        raise ValueError(f"{field_name} shape {array.shape} does not match domain grid shape {domain.grid_shape}.")
    return array


def _sample_points(domain: Domain) -> npt.NDArray[np.float64]:
    xs, ys, zs = domain.grid_centers()
    return np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)


def _ensure_watertight(mesh: Any, *, require_watertight: bool, context: str) -> None:
    if mesh.is_watertight:
        return
    message = (
        f"Mesh is not watertight; {context} is unreliable for open surfaces. "
        "Provide a watertight mesh or pass require_watertight=False to continue anyway."
    )
    if require_watertight:
        raise ValueError(message)
    warnings.warn(message, RuntimeWarning)


def read_mesh(path: str | Path) -> TriangleMesh:
    """Read a triangle mesh from disk using meshio."""

    meshio = _load_meshio()
    source = Path(path)
    raw = meshio.read(source)
    vertices = np.asarray(raw.points[:, :3], dtype=float)
    triangle_blocks = [np.asarray(block.data, dtype=np.int64) for block in raw.cells if block.type == "triangle"]
    if not triangle_blocks:
        raise ValueError(f"Mesh file {source} does not contain triangle cells.")
    faces = np.concatenate(triangle_blocks, axis=0)
    return TriangleMesh(vertices=vertices, faces=faces, metadata={"path": str(source)})


def write_mesh(path: str | Path, mesh: TriangleMesh) -> Path:
    """Write a triangle mesh to disk using meshio."""

    meshio = _load_meshio()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = meshio.Mesh(points=mesh.vertices, cells=[("triangle", mesh.faces)])
    raw.write(target)
    return target


class MeshSDF3(SDF3):
    """A mesh-backed signed-distance wrapper built on trimesh proximity queries."""

    def __init__(
        self,
        mesh: TriangleMesh,
        *,
        require_watertight: bool = True,
        name: str = "mesh_sdf",
    ) -> None:
        self.mesh = mesh
        self._trimesh = mesh.to_trimesh()
        # trimesh signed-distance semantics assume a positive-volume orientation.
        if self._trimesh.is_watertight and not self._trimesh.is_volume:
            self._trimesh.invert()
        _ensure_watertight(self._trimesh, require_watertight=require_watertight, context="signed-distance queries")
        super().__init__(self._evaluate_mesh, name=name)

    def _evaluate_mesh(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        trimesh = _load_trimesh()
        distances = trimesh.proximity.signed_distance(self._trimesh, points)
        return -np.asarray(distances, dtype=float)


def mesh_to_sdf_field(
    domain: Domain,
    mesh: TriangleMesh,
    *,
    require_watertight: bool = True,
) -> npt.NDArray[np.float64]:
    """Sample a watertight mesh into a signed-distance field on the domain grid."""

    return MeshSDF3(mesh, require_watertight=require_watertight).sample(domain)


def mesh_to_occupancy(
    domain: Domain,
    mesh: TriangleMesh,
    *,
    require_watertight: bool = True,
) -> npt.NDArray[np.bool_]:
    """Sample mesh occupancy at voxel centers using trimesh containment tests."""

    tri_mesh = mesh.to_trimesh()
    _ensure_watertight(tri_mesh, require_watertight=require_watertight, context="occupancy queries")
    points = _sample_points(domain)
    contained = np.asarray(tri_mesh.contains(points), dtype=bool)
    return contained.reshape(domain.grid_shape)


def occupancy_to_sdf_field(domain: Domain, occupancy: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert a dense occupancy grid into a sampled signed-distance field."""

    ndimage = _load_scipy_ndimage()
    occupancy_array = _validate_field_shape(domain, occupancy, field_name="occupancy").astype(bool)
    inside_distance = ndimage.distance_transform_edt(occupancy_array, sampling=domain.voxel_size)
    outside_distance = ndimage.distance_transform_edt(~occupancy_array, sampling=domain.voxel_size)
    return outside_distance - inside_distance


def density_to_sdf_field(
    domain: Domain,
    density: npt.ArrayLike,
    *,
    threshold: float = 0.5,
) -> npt.NDArray[np.float64]:
    """Threshold a density field into occupancy and convert it to a signed-distance field."""

    density_array = _validate_field_shape(domain, density, field_name="density")
    occupancy = density_array >= threshold
    return occupancy_to_sdf_field(domain, occupancy)


def occupancy_to_sdf(domain: Domain, occupancy: npt.ArrayLike) -> GridSDF3:
    """Wrap an occupancy-derived signed-distance field as an interpolated GridSDF3."""

    values = occupancy_to_sdf_field(domain, occupancy)
    return GridSDF3(domain, values, name="occupancy_sdf")


def density_to_sdf(
    domain: Domain,
    density: npt.ArrayLike,
    *,
    threshold: float = 0.5,
) -> GridSDF3:
    """Wrap a thresholded density field as an interpolated GridSDF3."""

    values = density_to_sdf_field(domain, density, threshold=threshold)
    return GridSDF3(domain, values, name="density_sdf")
