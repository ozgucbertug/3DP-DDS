"""Material contribution kernels for deposition primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .attributes import DepositionAttributes
from .domain import Domain
from .primitives import LineDeposit, Point3D, PointDeposit
from .utils import expand_aabb, point_to_segment_distances


@dataclass(frozen=True, slots=True)
class SampledKernel:
    """A local kernel sample and its target array window."""

    slices: tuple[slice, slice, slice]
    values: npt.NDArray[np.float64]


def resolve_bead_radii(
    attributes: DepositionAttributes,
    domain: Domain,
) -> tuple[float, float, float]:
    """Resolve cross-section radii from deposit metadata and domain defaults."""

    default_width = min(domain.voxel_size)
    width = attributes.width if attributes.width is not None else default_width
    height = attributes.height if attributes.height is not None else width
    if width <= 0.0 or height <= 0.0:
        raise ValueError("Resolved bead width and height must both be positive.")
    radius_xy = width / 2.0
    radius_z = height / 2.0
    return radius_xy, radius_xy, radius_z


def compact_quadratic_profile(r_squared: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Evaluate the smooth compact profile `max(0, 1 - r^2)`."""

    return np.clip(1.0 - r_squared, 0.0, None)


def sample_point_kernel(domain: Domain, deposit: PointDeposit) -> SampledKernel | None:
    """Sample an anisotropic ellipsoidal point kernel on the local grid window."""

    rx, ry, rz = resolve_bead_radii(deposit.attributes, domain)
    center = deposit.point.to_array()
    support_min, support_max = expand_aabb(center, center, (rx, ry, rz))
    index_bounds = domain.index_bounds_for_aabb(support_min, support_max)
    if index_bounds is None:
        return None

    x_bounds, y_bounds, z_bounds = index_bounds
    xs, ys, zs = domain.grid_centers(index_bounds)
    r_squared = ((xs - center[0]) / rx) ** 2 + ((ys - center[1]) / ry) ** 2 + ((zs - center[2]) / rz) ** 2
    values = compact_quadratic_profile(r_squared)
    return SampledKernel(
        slices=(
            slice(*x_bounds),
            slice(*y_bounds),
            slice(*z_bounds),
        ),
        values=values.astype(float, copy=False),
    )


def sample_line_kernel(domain: Domain, deposit: LineDeposit) -> SampledKernel | None:
    """Sample a swept-sphere line kernel using a z-scaled closest-distance model."""

    rx, _, rz = resolve_bead_radii(deposit.attributes, domain)
    start = deposit.start.to_array()
    end = deposit.end.to_array()
    if np.allclose(start, end):
        point_deposit = PointDeposit(
            x=float(start[0]),
            y=float(start[1]),
            z=float(start[2]),
            attributes=deposit.attributes,
        )
        return sample_point_kernel(domain, point_deposit)

    support_min, support_max = expand_aabb(np.minimum(start, end), np.maximum(start, end), (rx, rx, rz))
    index_bounds = domain.index_bounds_for_aabb(support_min, support_max)
    if index_bounds is None:
        return None

    x_bounds, y_bounds, z_bounds = index_bounds
    xs, ys, zs = domain.grid_centers(index_bounds)
    z_scale = rx / rz
    sample_points = np.stack((xs, ys, zs * z_scale), axis=-1)
    start_scaled = start.copy()
    end_scaled = end.copy()
    start_scaled[2] *= z_scale
    end_scaled[2] *= z_scale
    distances = point_to_segment_distances(sample_points, start_scaled, end_scaled)
    r_squared = (distances / rx) ** 2
    values = compact_quadratic_profile(r_squared)
    return SampledKernel(
        slices=(
            slice(*x_bounds),
            slice(*y_bounds),
            slice(*z_bounds),
        ),
        values=values.astype(float, copy=False),
    )


def sample_deposit_kernel(
    domain: Domain,
    deposit: PointDeposit | LineDeposit,
) -> SampledKernel | None:
    """Dispatch kernel sampling based on deposit type."""

    if isinstance(deposit, PointDeposit):
        return sample_point_kernel(domain, deposit)
    if isinstance(deposit, LineDeposit):
        return sample_line_kernel(domain, deposit)
    raise TypeError("Unsupported deposit type.")
