"""Material contribution kernels for deposition primitives."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .attributes import BeadProfile
from .domain import Domain
from .primitives import (
    LineDeposit,
    PointDeposit,
    _point_target_support_bounds,
)
from .utils import closest_point_parameters, slerp_unit_vectors


@dataclass(frozen=True, slots=True)
class SampledKernel:
    """A local kernel sample and its target array window."""

    slices: tuple[slice, slice, slice]
    values: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ResolvedBeadProfile:
    """Resolved bead geometry and density transition settings."""

    width: float
    height: float
    radius: float
    half_height: float
    rounding_radius: float
    transition_width: float
    support_padding: float


def resolve_bead_profile(
    profile: BeadProfile | None,
    domain: Domain,
) -> ResolvedBeadProfile:
    """Resolve bead geometry and transition width from metadata and domain defaults."""

    default_width = min(domain.voxel_size)
    if profile is None:
        warnings.warn(
            "No BeadProfile supplied; falling back to a default profile derived from the domain voxel size. "
            "This may produce unexpected geometry. Use BeadProfile.default(voxel_size) to create an explicit profile.",
            UserWarning,
            stacklevel=3,
        )
    width = profile.width if profile is not None else default_width
    height = profile.height if profile is not None else width
    if width <= 0.0 or height <= 0.0:
        raise ValueError("Resolved bead width and height must both be positive.")
    rounding_radius = min(width, height) / 2.0
    transition_width = max(min(domain.voxel_size), rounding_radius)
    return ResolvedBeadProfile(
        width=float(width),
        height=float(height),
        radius=float(width / 2.0),
        half_height=float(height / 2.0),
        rounding_radius=float(rounding_radius),
        transition_width=float(transition_width),
        support_padding=float(transition_width / 2.0),
    )


def density_from_signed_distance(
    signed_distance: npt.NDArray[np.float64],
    transition_width: float,
) -> npt.NDArray[np.float64]:
    """Map signed distance to a dense accumulation value with a 0.5 isosurface."""

    if transition_width <= 0.0:
        raise ValueError("transition_width must be positive.")
    return np.clip(0.5 - signed_distance / transition_width, 0.0, 1.0)


def rounded_cylinder_signed_distance(
    points: npt.NDArray[np.float64],
    *,
    target: npt.NDArray[np.float64],
    axis: npt.NDArray[np.float64],
    profile: ResolvedBeadProfile,
) -> npt.NDArray[np.float64]:
    """Evaluate the signed distance to a top-referenced rounded cylinder."""

    center = target - axis * profile.half_height
    relative = points - center
    axial = np.sum(relative * axis, axis=-1)
    radial_vectors = relative - axial[..., np.newaxis] * axis
    radial = np.linalg.norm(radial_vectors, axis=-1)
    bounds = np.stack(
        (
            radial - profile.radius + profile.rounding_radius,
            np.abs(axial) - profile.half_height + profile.rounding_radius,
        ),
        axis=-1,
    )
    return (
        np.minimum(np.maximum(bounds[..., 0], bounds[..., 1]), 0.0)
        + np.linalg.norm(np.maximum(bounds, 0.0), axis=-1)
        - profile.rounding_radius
    )


def sample_point_kernel(domain: Domain, deposit: PointDeposit) -> SampledKernel | None:
    """Sample a rounded-cylinder point kernel on the local grid window."""

    profile = resolve_bead_profile(deposit.profile, domain)
    target = deposit.target.to_array()
    axis = deposit.axis.to_array()
    support_min, support_max = _point_target_support_bounds(
        target,
        axis,
        width=profile.width,
        height=profile.height,
        padding=profile.support_padding,
    )
    index_bounds = domain.index_bounds_for_aabb(support_min, support_max)
    if index_bounds is None:
        return None

    x_bounds, y_bounds, z_bounds = index_bounds
    xs, ys, zs = domain.grid_centers(index_bounds)
    points = np.stack((xs, ys, zs), axis=-1)
    signed_distance = rounded_cylinder_signed_distance(
        points,
        target=target,
        axis=axis,
        profile=profile,
    )
    values = density_from_signed_distance(signed_distance, profile.transition_width)
    return SampledKernel(
        slices=(
            slice(*x_bounds),
            slice(*y_bounds),
            slice(*z_bounds),
        ),
        values=values.astype(float, copy=False),
    )


def sample_line_kernel(domain: Domain, deposit: LineDeposit) -> SampledKernel | None:
    """Sample a swept rounded-bead line kernel."""

    profile = resolve_bead_profile(deposit.profile, domain)
    start = deposit.start.to_array()
    end = deposit.end.to_array()
    if np.allclose(start, end):
        point_deposit = PointDeposit(
            x=float(start[0]),
            y=float(start[1]),
            z=float(start[2]),
            profile=deposit.profile,
            metadata=deposit.metadata,
            z_axis=deposit.start_axis,
        )
        return sample_point_kernel(domain, point_deposit)

    start_axis = deposit.start_axis.to_array()
    end_axis = deposit.end_axis.to_array()
    support_min, support_max = deposit.support_bounds(
        padding=profile.support_padding,
    )
    index_bounds = domain.index_bounds_for_aabb(support_min, support_max)
    if index_bounds is None:
        return None

    x_bounds, y_bounds, z_bounds = index_bounds
    xs, ys, zs = domain.grid_centers(index_bounds)
    points = np.stack((xs, ys, zs), axis=-1)
    flat_points = points.reshape(-1, 3)
    parameters = closest_point_parameters(flat_points, start, end)
    closest_targets = start + parameters[:, np.newaxis] * (end - start)
    axes = slerp_unit_vectors(start_axis, end_axis, parameters)
    signed_distance = rounded_cylinder_signed_distance(
        flat_points,
        target=closest_targets,
        axis=axes,
        profile=profile,
    ).reshape(xs.shape)
    values = density_from_signed_distance(signed_distance, profile.transition_width)
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
