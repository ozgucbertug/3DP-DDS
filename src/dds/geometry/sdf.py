"""Continuous signed-distance abstractions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from ..domain import Domain
from ._utils import validate_field_shape
from .mesh import TriangleMesh, _ensure_watertight

SDFCallable = Callable[[npt.NDArray[np.float64]], npt.ArrayLike]


def _coerce_points(points: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], bool]:
    """Normalize input points to an `(n, 3)` float array."""

    array = np.asarray(points, dtype=float)
    if array.ndim == 1:
        if array.shape != (3,):
            raise ValueError("Single-point evaluation expects exactly three coordinates.")
        if not np.all(np.isfinite(array)):
            raise ValueError("Point coordinates must be finite.")
        return array.reshape(1, 3), True
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Point evaluation expects shape `(3,)` or `(n, 3)`.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Point coordinates must be finite.")
    return array, False


def as_sdf3(value: "SDF3 | SDFCallable") -> "SDF3":
    """Coerce an SDF-like value into an SDF3 wrapper."""

    if isinstance(value, SDF3):
        return value
    if not callable(value):
        raise TypeError("Expected an SDF3 instance or a callable SDF.")
    return SDF3(value, name=getattr(value, "__name__", "callable_sdf"))


class SDF3:
    """Continuous signed-distance function in 3D.

    The sign convention is `negative inside, positive outside, zero on the surface`.
    """

    def __init__(self, func: SDFCallable, *, name: str | None = None) -> None:
        self._func = func
        self.name = name or getattr(func, "__name__", "sdf3")

    def _evaluate(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        values = np.asarray(self._func(points), dtype=float).reshape(points.shape[0])
        if not np.all(np.isfinite(values)):
            raise ValueError(f"SDF {self.name!r} returned non-finite values.")
        return values

    def __call__(self, points: npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        array, single = _coerce_points(points)
        values = self._evaluate(array)
        return float(values[0]) if single else values

    def __or__(self, other: "SDF3 | SDFCallable") -> "SDF3":
        from .ops import union

        return union(self, other)

    def __and__(self, other: "SDF3 | SDFCallable") -> "SDF3":
        from .ops import intersection

        return intersection(self, other)

    def __sub__(self, other: "SDF3 | SDFCallable") -> "SDF3":
        from .ops import difference

        return difference(self, other)

    def sample(self, domain: Domain) -> npt.NDArray[np.float64]:
        """Sample the SDF on the voxel-center lattice of a domain."""

        xs, ys, zs = domain.grid_centers()
        points = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)
        values = self._evaluate(points)
        return values.reshape(domain.grid_shape)

    def to_mesh(self, domain: Domain, *, level: float = 0.0, step_size: int = 1) -> Any:
        """Sample the SDF on a domain and extract an isosurface mesh."""

        from .mesh import sdf_to_mesh

        return sdf_to_mesh(domain, self.sample(domain), level=level, step_size=step_size)

    def translate(self, offset: Any) -> "SDF3":
        from .transforms import translate

        return translate(self, offset)

    def scale(self, factor: float | Sequence[float]) -> "SDF3":
        from .transforms import scale

        return scale(self, factor)

    def rotate(self, angle: float, axis: Any = (0.0, 0.0, 1.0)) -> "SDF3":
        from .transforms import rotate

        return rotate(self, angle, axis)

    def orient(
        self,
        axis: Any,
        *,
        source_axis: Any = (0.0, 0.0, 1.0),
    ) -> "SDF3":
        from .transforms import orient

        return orient(self, axis, source_axis=source_axis)

    def dilate(self, radius: float) -> "SDF3":
        from .ops import dilate

        return dilate(self, radius)

    def erode(self, radius: float) -> "SDF3":
        from .ops import erode

        return erode(self, radius)

    def shell(self, thickness: float, *, type: str = "center") -> "SDF3":
        from .ops import shell

        return shell(self, thickness, type=type)

    @classmethod
    def from_grid(
        cls,
        domain: Domain,
        values: npt.ArrayLike,
        *,
        fill_value: float | None = None,
        name: str = "grid_sdf",
    ) -> "GridSDF3":
        """Construct a sampled-grid SDF wrapper."""

        return GridSDF3(domain, values, fill_value=fill_value, name=name)


class GridSDF3(SDF3):
    """An SDF backed by a sampled dense grid and trilinear interpolation."""

    def __init__(
        self,
        domain: Domain,
        values: npt.ArrayLike,
        *,
        fill_value: float | None = None,
        name: str = "grid_sdf",
    ) -> None:
        self._domain = domain
        self._values = np.array(values, dtype=float, copy=True)
        if self._values.shape != domain.grid_shape:
            raise ValueError(
                f"GridSDF3 values shape {self._values.shape} does not match domain shape {domain.grid_shape}."
            )
        self._values.setflags(write=False)
        self._fill_value = fill_value if fill_value is not None else self._default_fill_value()
        self._interpolator: Any | None = None
        super().__init__(self._evaluate_grid, name=name)

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def values(self) -> npt.NDArray[np.float64]:
        return self._values

    @property
    def fill_value(self) -> float:
        return float(self._fill_value)

    def _default_fill_value(self) -> float:
        if self.values.size == 0:
            return 1.0
        diagonal = float(np.linalg.norm(np.asarray(self.domain.grid_shape) * np.asarray(self.domain.voxel_size)))
        max_magnitude = float(np.max(np.abs(self.values)))
        return max(diagonal, max_magnitude, 1.0)

    def _get_interpolator(self) -> Any:
        if self._interpolator is None:
            try:
                from scipy.interpolate import RegularGridInterpolator
            except ImportError as exc:
                raise ImportError("SciPy is required for sampled-grid SDF interpolation. Install `3dp-dds`.") from exc

            axes = tuple(self.domain.axis_centers(axis) for axis in range(3))
            self._interpolator = RegularGridInterpolator(
                axes,
                self.values,
                method="linear",
                bounds_error=False,
                fill_value=self.fill_value,
            )
        return self._interpolator

    def _evaluate_grid(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(self._get_interpolator()(points), dtype=float)

    def sample(self, domain: Domain) -> npt.NDArray[np.float64]:
        """Reuse the stored grid when sampling the same domain."""

        if domain == self.domain:
            return self.values.copy()
        return super().sample(domain)


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
        _ensure_watertight(
            self._trimesh,
            require_watertight=require_watertight,
            context="signed-distance queries",
        )
        super().__init__(self._evaluate_mesh, name=name)

    def _evaluate_mesh(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        distances = self._trimesh.nearest.signed_distance(points)
        return -np.asarray(distances, dtype=float)


def mesh_to_sdf_field(
    domain: Domain,
    mesh: TriangleMesh,
    *,
    require_watertight: bool = True,
) -> npt.NDArray[np.float64]:
    """Sample a watertight mesh into a signed-distance field on the domain grid."""

    return MeshSDF3(mesh, require_watertight=require_watertight).sample(domain)


def occupancy_to_sdf_field(domain: Domain, occupancy: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert a dense occupancy grid into a sampled signed-distance field."""

    occupancy_array = validate_field_shape(domain, occupancy, field_name="occupancy").astype(bool)
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

    values = validate_field_shape(
        domain,
        implicit_field,
        field_name="implicit_field",
    )
    return occupancy_to_sdf_field(domain, values >= threshold)


def occupancy_to_sdf(domain: Domain, occupancy: npt.ArrayLike) -> GridSDF3:
    """Wrap an occupancy-derived signed-distance field as an interpolated GridSDF3."""

    return GridSDF3(
        domain,
        occupancy_to_sdf_field(domain, occupancy),
        name="occupancy_sdf",
    )


def implicit_field_to_sdf(
    domain: Domain,
    implicit_field: npt.ArrayLike,
    *,
    threshold: float = 0.5,
) -> GridSDF3:
    """Wrap a thresholded implicit field as an interpolated GridSDF3."""

    return GridSDF3(
        domain,
        implicit_field_to_sdf_values(
            domain,
            implicit_field,
            threshold=threshold,
        ),
        name="implicit_field_sdf",
    )
