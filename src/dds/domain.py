"""Simulation domain and dense-grid coordinate transforms."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .primitives import DepositInput, Point3D, iter_deposits
from .utils import ensure_finite_triplet, ensure_positive_triplet

IndexBounds = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


@dataclass(frozen=True, slots=True)
class Domain:
    """Axis-aligned simulation workspace sampled on a dense 3D grid."""

    min_corner: tuple[float, float, float]
    max_corner: tuple[float, float, float]
    voxel_size: tuple[float, float, float]
    grid_shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        minimum = ensure_finite_triplet(self.min_corner, "min_corner")
        maximum = ensure_finite_triplet(self.max_corner, "max_corner")
        spacing = ensure_positive_triplet(self.voxel_size, "voxel_size")
        if len(self.grid_shape) != 3:
            raise ValueError("grid_shape must contain exactly three integer values.")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in self.grid_shape
        ):
            raise TypeError("grid_shape must contain exactly three integer values.")
        shape = tuple(int(value) for value in self.grid_shape)
        if any(value <= 0 for value in shape):
            raise ValueError("grid_shape values must all be positive.")
        if any(lower >= upper for lower, upper in zip(minimum, maximum)):
            raise ValueError("Domain bounds must be strictly increasing on every axis.")

        expected_maximum = tuple(
            lower + count * step
            for lower, count, step in zip(minimum, shape, spacing)
        )
        if not np.allclose(maximum, expected_maximum, rtol=1e-12, atol=1e-12):
            raise ValueError(
                "max_corner must equal min_corner + grid_shape * voxel_size. "
                "Use Domain.from_bounds() to align arbitrary requested bounds."
            )

        object.__setattr__(self, "min_corner", minimum)
        object.__setattr__(self, "max_corner", expected_maximum)
        object.__setattr__(self, "voxel_size", spacing)
        object.__setattr__(self, "grid_shape", shape)

    @classmethod
    def from_bounds(
        cls,
        *,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        zmin: float,
        zmax: float,
        voxel_size: float | Sequence[float],
    ) -> "Domain":
        """Create a domain from scalar bounds and isotropic or anisotropic voxel size."""

        minimum = ensure_finite_triplet((xmin, ymin, zmin), "minimum bounds")
        maximum = ensure_finite_triplet((xmax, ymax, zmax), "maximum bounds")
        if isinstance(voxel_size, (int, float)):
            voxel_triplet = ensure_positive_triplet((voxel_size, voxel_size, voxel_size), "voxel_size")
        else:
            voxel_triplet = ensure_positive_triplet(voxel_size, "voxel_size")

        if any(lower >= upper for lower, upper in zip(minimum, maximum)):
            raise ValueError("Domain bounds must be strictly increasing on every axis.")

        shape = tuple(
            int(math.ceil((upper - lower) / step))
            for lower, upper, step in zip(minimum, maximum, voxel_triplet)
        )
        aligned_maximum = tuple(
            lower + count * step
            for lower, count, step in zip(minimum, shape, voxel_triplet)
        )
        return cls(
            min_corner=minimum,
            max_corner=aligned_maximum,
            voxel_size=voxel_triplet,
            grid_shape=shape,
        )

    @classmethod
    def from_deposits(
        cls,
        deposits: Iterable[DepositInput] | DepositInput,
        *,
        voxel_size: float | Sequence[float],
        padding: float | str = "auto",
    ) -> "Domain":
        """Create a padded domain that encloses the support bounds of deposits."""

        if isinstance(voxel_size, (int, float)):
            voxel_triplet = ensure_positive_triplet((voxel_size, voxel_size, voxel_size), "voxel_size")
        else:
            voxel_triplet = ensure_positive_triplet(voxel_size, "voxel_size")

        if padding == "auto":
            resolved_padding = float(max(voxel_triplet))
        elif isinstance(padding, (int, float)):
            resolved_padding = float(padding)
            if resolved_padding < 0.0:
                raise ValueError("padding must be non-negative.")
        else:
            raise ValueError("padding must be a non-negative float or 'auto'.")

        minima: list[tuple[float, float, float]] = []
        maxima: list[tuple[float, float, float]] = []
        for deposit in iter_deposits(deposits):
            minimum, maximum = deposit.support_bounds(padding=resolved_padding)
            minima.append(minimum.to_tuple())
            maxima.append(maximum.to_tuple())

        if not minima:
            raise ValueError("from_deposits requires at least one deposit.")

        lower = np.asarray(minima, dtype=float).min(axis=0)
        upper = np.asarray(maxima, dtype=float).max(axis=0)
        return cls.from_bounds(
            xmin=float(lower[0]),
            xmax=float(upper[0]),
            ymin=float(lower[1]),
            ymax=float(upper[1]),
            zmin=float(lower[2]),
            zmax=float(upper[2]),
            voxel_size=voxel_triplet,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a dictionary representation for export."""

        return {
            "min_corner": self.min_corner,
            "max_corner": self.max_corner,
            "voxel_size": self.voxel_size,
            "grid_shape": self.grid_shape,
        }

    def contains_point(self, point: Point3D | Sequence[float]) -> bool:
        """Return True when a point lies inside the half-open domain bounds."""

        x, y, z = ensure_finite_triplet(point, "point")
        return (
            self.min_corner[0] <= x < self.max_corner[0]
            and self.min_corner[1] <= y < self.max_corner[1]
            and self.min_corner[2] <= z < self.max_corner[2]
        )

    def world_to_index(
        self,
        point: Point3D | Sequence[float],
        *,
        clip: bool = False,
    ) -> tuple[int, int, int]:
        """Convert world coordinates to integer voxel indices."""

        coordinates = np.asarray(ensure_finite_triplet(point, "point"), dtype=float)
        origin = np.asarray(self.min_corner, dtype=float)
        spacing = np.asarray(self.voxel_size, dtype=float)
        raw = np.floor((coordinates - origin) / spacing).astype(int)
        shape = np.asarray(self.grid_shape, dtype=int)
        if clip:
            clipped = np.clip(raw, 0, shape - 1)
            return tuple(int(value) for value in clipped)
        if np.any(raw < 0) or np.any(raw >= shape):
            raise ValueError("Point lies outside the domain.")
        return tuple(int(value) for value in raw)

    def index_to_world(self, index: Sequence[int]) -> Point3D:
        """Convert voxel indices to the voxel-center world coordinate."""

        if len(index) != 3:
            raise ValueError("index must contain exactly three integer values.")
        ix, iy, iz = (int(value) for value in index)
        shape = self.grid_shape
        if not (0 <= ix < shape[0] and 0 <= iy < shape[1] and 0 <= iz < shape[2]):
            raise ValueError("Index lies outside the domain.")

        coordinate = tuple(
            self.min_corner[axis] + (index_value + 0.5) * self.voxel_size[axis]
            for axis, index_value in enumerate((ix, iy, iz))
        )
        return Point3D.from_value(coordinate)

    def axis_centers(self, axis: int, start: int = 0, stop: int | None = None) -> npt.NDArray[np.float64]:
        """Return voxel-center coordinates along one axis."""

        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2.")
        if isinstance(start, bool) or not isinstance(start, (int, np.integer)):
            raise TypeError("start must be an integer.")
        end = self.grid_shape[axis] if stop is None else stop
        if isinstance(end, bool) or not isinstance(end, (int, np.integer)):
            raise TypeError("stop must be an integer or None.")
        if not 0 <= int(start) <= int(end) <= self.grid_shape[axis]:
            raise ValueError("axis-center bounds must satisfy 0 <= start <= stop <= axis size.")
        indices = np.arange(start, end, dtype=float)
        return self.min_corner[axis] + (indices + 0.5) * self.voxel_size[axis]

    def grid_centers(
        self,
        index_bounds: IndexBounds | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return dense voxel-center coordinate arrays using `(x, y, z)` indexing."""

        if index_bounds is None:
            x_bounds = (0, self.grid_shape[0])
            y_bounds = (0, self.grid_shape[1])
            z_bounds = (0, self.grid_shape[2])
        else:
            x_bounds, y_bounds, z_bounds = index_bounds

        xs = self.axis_centers(0, *x_bounds)
        ys = self.axis_centers(1, *y_bounds)
        zs = self.axis_centers(2, *z_bounds)
        return np.meshgrid(xs, ys, zs, indexing="ij")

    def index_bounds_for_aabb(
        self,
        minimum: Point3D | Sequence[float],
        maximum: Point3D | Sequence[float],
    ) -> IndexBounds | None:
        """Return half-open index bounds for voxel centers inside an AABB."""

        minimum_array = np.asarray(ensure_finite_triplet(minimum, "minimum"), dtype=float)
        maximum_array = np.asarray(ensure_finite_triplet(maximum, "maximum"), dtype=float)
        if np.any(minimum_array > maximum_array):
            raise ValueError("minimum must not exceed maximum.")

        origin = np.asarray(self.min_corner, dtype=float)
        spacing = np.asarray(self.voxel_size, dtype=float)
        shape = np.asarray(self.grid_shape, dtype=int)

        start = np.ceil((minimum_array - origin) / spacing - 0.5).astype(int)
        stop = np.floor((maximum_array - origin) / spacing - 0.5).astype(int) + 1

        start = np.clip(start, 0, shape)
        stop = np.clip(stop, 0, shape)
        if np.any(stop <= start):
            return None

        return (
            (int(start[0]), int(stop[0])),
            (int(start[1]), int(stop[1])),
            (int(start[2]), int(stop[2])),
        )
