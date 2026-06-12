"""Mesh IO and conversions between meshes, dense fields, and SDF wrappers."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ..domain import Domain
from .mesh import TriangleMesh, _load_trimesh, _validate_field_shape
from .point_cloud import PointCloud
from .sdf import SDF3, GridSDF3


def _load_scipy_ndimage() -> Any:
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise ImportError("SciPy is required for occupancy-to-SDF conversion. Install `3dp-dds`.") from exc
    return ndimage


def _ensure_watertight(mesh: Any, *, require_watertight: bool, context: str) -> None:
    if mesh.is_watertight:
        return
    message = (
        f"Mesh is not watertight; {context} is unreliable for open surfaces. "
        "Provide a watertight mesh or pass require_watertight=False to continue anyway."
    )
    if require_watertight:
        raise ValueError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def read_mesh(path: str | Path) -> TriangleMesh:
    """Read a triangle mesh from disk using trimesh."""

    trimesh = _load_trimesh()
    source = Path(path)
    loaded = trimesh.load(source, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = loaded.dump()
        if not geometries or not all(
            isinstance(geometry, trimesh.Trimesh) for geometry in geometries
        ):
            raise ValueError(f"File {source} does not contain only triangle meshes.")
        loaded = loaded.to_mesh()
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"File {source} does not contain a triangle mesh.")
    return TriangleMesh.from_trimesh(loaded, metadata={"path": str(source)})


def write_mesh(path: str | Path, mesh: TriangleMesh) -> Path:
    """Write a triangle mesh to disk using trimesh."""

    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mesh.to_trimesh().export(target)
    return target


def read_point_cloud(path: str | Path) -> PointCloud:
    """Read a point cloud from disk using trimesh."""

    trimesh = _load_trimesh()
    source = Path(path)
    loaded = trimesh.load(source)
    if not isinstance(loaded, trimesh.points.PointCloud):
        raise ValueError(f"File {source} does not contain a point cloud.")
    return PointCloud.from_trimesh(loaded, metadata={"path": str(source)})


def write_point_cloud(path: str | Path, cloud: PointCloud) -> Path:
    """Write a point cloud to disk using trimesh."""

    if not isinstance(cloud, PointCloud):
        raise TypeError("cloud must be a PointCloud")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cloud.to_trimesh().export(target)
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
    xs, ys, zs = domain.grid_centers()
    points = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)
    contained = np.asarray(tri_mesh.contains(points), dtype=bool)
    return contained.reshape(domain.grid_shape)


def occupancy_to_sdf_field(domain: Domain, occupancy: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert a dense occupancy grid into a sampled signed-distance field."""

    ndimage = _load_scipy_ndimage()
    occupancy_array = _validate_field_shape(domain, occupancy, field_name="occupancy").astype(bool)
    inside_distance = ndimage.distance_transform_edt(occupancy_array, sampling=domain.voxel_size)
    outside_distance = ndimage.distance_transform_edt(~occupancy_array, sampling=domain.voxel_size)
    return outside_distance - inside_distance


def implicit_field_to_sdf_values(
    domain: Domain,
    implicit_field: npt.ArrayLike,
    *,
    threshold: float = 0.5,
) -> npt.NDArray[np.float64]:
    """Convert a thresholded implicit field to signed-distance values."""

    values = _validate_field_shape(
        domain,
        implicit_field,
        field_name="implicit_field",
    )
    occupancy = values >= threshold
    return occupancy_to_sdf_field(domain, occupancy)


def occupancy_to_sdf(domain: Domain, occupancy: npt.ArrayLike) -> GridSDF3:
    """Wrap an occupancy-derived signed-distance field as an interpolated GridSDF3."""

    values = occupancy_to_sdf_field(domain, occupancy)
    return GridSDF3(domain, values, name="occupancy_sdf")


def implicit_field_to_sdf(
    domain: Domain,
    implicit_field: npt.ArrayLike,
    *,
    threshold: float = 0.5,
) -> GridSDF3:
    """Wrap a thresholded implicit field as an interpolated GridSDF3."""

    values = implicit_field_to_sdf_values(
        domain,
        implicit_field,
        threshold=threshold,
    )
    return GridSDF3(domain, values, name="implicit_field_sdf")
