"""Material contribution kernels for deposition primitives."""

from __future__ import annotations

import heapq
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import count

import numpy as np
import numpy.typing as npt

from .attributes import BeadProfile
from .domain import Domain
from .primitives import (
    LineDeposit,
    PointDeposit,
    PolylineDeposit,
    _point_target_support_bounds,
)
from .utils import closest_point_parameters, slerp_unit_vectors

TileShape = tuple[int, int, int]
_LINE_SWEEP_SEARCH_SAMPLES = 9
_LINE_SWEEP_SEARCH_ITERATIONS = 24


@dataclass(frozen=True, slots=True)
class _SampledKernel:
    """Private local kernel sample and its target array window."""

    slices: tuple[slice, slice, slice]
    values: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _ResolvedBeadProfile:
    """Private resolved bead geometry and transition settings."""

    width: float
    height: float
    radius: float
    half_height: float
    rounding_radius: float
    transition_width: float
    support_padding: float


def _resolve_bead_profile(
    profile: BeadProfile,
    domain: Domain,
) -> _ResolvedBeadProfile:
    """Resolve transition settings for an explicit bead profile."""

    width = profile.width
    height = profile.height
    if width <= 0.0 or height <= 0.0:
        raise ValueError("Resolved bead width and height must both be positive.")
    rounding_radius = min(width, height) / 2.0
    transition_width = max(min(domain.voxel_size), rounding_radius)
    return _ResolvedBeadProfile(
        width=float(width),
        height=float(height),
        radius=float(width / 2.0),
        half_height=float(height / 2.0),
        rounding_radius=float(rounding_radius),
        transition_width=float(transition_width),
        support_padding=float(transition_width / 2.0),
    )


def _implicit_values_from_signed_distance(
    signed_distance: npt.NDArray[np.float64],
    transition_width: float,
) -> npt.NDArray[np.float64]:
    """Map signed distance to bounded implicit values with a 0.5 isosurface."""

    if transition_width <= 0.0:
        raise ValueError("transition_width must be positive.")
    return np.clip(0.5 - signed_distance / transition_width, 0.0, 1.0)


def rounded_cylinder_signed_distance(
    points: npt.NDArray[np.float64],
    *,
    target: npt.NDArray[np.float64],
    axis: npt.NDArray[np.float64],
    profile: _ResolvedBeadProfile,
) -> npt.NDArray[np.float64]:
    """Evaluate the signed distance to a top-referenced rounded cylinder."""

    center = target - axis * profile.half_height
    relative = points - center
    axial = np.sum(relative * axis, axis=-1)
    radial_vectors = relative - axial[..., np.newaxis] * axis
    radial = np.sqrt(np.sum(radial_vectors * radial_vectors, axis=-1))
    radial_bound = radial - profile.radius + profile.rounding_radius
    axial_bound = (
        np.abs(axial) - profile.half_height + profile.rounding_radius
    )
    return (
        np.minimum(np.maximum(radial_bound, axial_bound), 0.0)
        + np.sqrt(
            np.maximum(radial_bound, 0.0) ** 2
            + np.maximum(axial_bound, 0.0) ** 2
        )
        - profile.rounding_radius
    )


def _line_sweep_signed_distance_at_parameters(
    points: npt.NDArray[np.float64],
    *,
    start: npt.NDArray[np.float64],
    end: npt.NDArray[np.float64],
    start_axis: npt.NDArray[np.float64],
    end_axis: npt.NDArray[np.float64],
    parameters: npt.NDArray[np.float64],
    profile: _ResolvedBeadProfile,
) -> npt.NDArray[np.float64]:
    targets = start + parameters[:, np.newaxis] * (end - start)
    axes = (
        start_axis
        if np.allclose(start_axis, end_axis, rtol=0.0, atol=1e-12)
        else slerp_unit_vectors(start_axis, end_axis, parameters)
    )
    return rounded_cylinder_signed_distance(
        points,
        target=targets,
        axis=axes,
        profile=profile,
    )


def _line_sweep_needs_parameter_search(
    start: npt.NDArray[np.float64],
    end: npt.NDArray[np.float64],
    start_axis: npt.NDArray[np.float64],
    end_axis: npt.NDArray[np.float64],
) -> bool:
    if not np.allclose(start_axis, end_axis, rtol=0.0, atol=1e-12):
        return True

    segment = end - start
    length = float(np.linalg.norm(segment))
    if length <= 1e-12:
        return False
    tangent = segment / length
    return abs(float(np.dot(tangent, start_axis))) > 1e-12


def _minimized_line_sweep_signed_distance(
    points: npt.NDArray[np.float64],
    *,
    start: npt.NDArray[np.float64],
    end: npt.NDArray[np.float64],
    start_axis: npt.NDArray[np.float64],
    end_axis: npt.NDArray[np.float64],
    profile: _ResolvedBeadProfile,
) -> npt.NDArray[np.float64]:
    sample_parameters = np.linspace(0.0, 1.0, _LINE_SWEEP_SEARCH_SAMPLES)
    best_distance = np.full(points.shape[0], np.inf, dtype=float)
    best_index = np.zeros(points.shape[0], dtype=np.intp)
    for index, parameter in enumerate(sample_parameters):
        parameters = np.full(points.shape[0], parameter, dtype=float)
        distance = _line_sweep_signed_distance_at_parameters(
            points,
            start=start,
            end=end,
            start_axis=start_axis,
            end_axis=end_axis,
            parameters=parameters,
            profile=profile,
        )
        is_better = distance < best_distance
        best_distance[is_better] = distance[is_better]
        best_index[is_better] = index

    step = 1.0 / (_LINE_SWEEP_SEARCH_SAMPLES - 1)
    centers = sample_parameters[best_index]
    lower = np.clip(centers - step, 0.0, 1.0)
    upper = np.clip(centers + step, 0.0, 1.0)
    golden = (np.sqrt(5.0) - 1.0) / 2.0

    for _ in range(_LINE_SWEEP_SEARCH_ITERATIONS):
        left = upper - golden * (upper - lower)
        right = lower + golden * (upper - lower)
        left_distance = _line_sweep_signed_distance_at_parameters(
            points,
            start=start,
            end=end,
            start_axis=start_axis,
            end_axis=end_axis,
            parameters=left,
            profile=profile,
        )
        right_distance = _line_sweep_signed_distance_at_parameters(
            points,
            start=start,
            end=end,
            start_axis=start_axis,
            end_axis=end_axis,
            parameters=right,
            profile=profile,
        )
        choose_left = left_distance < right_distance
        upper = np.where(choose_left, right, upper)
        lower = np.where(choose_left, lower, left)

    refined_parameters = (lower + upper) / 2.0
    refined_distance = _line_sweep_signed_distance_at_parameters(
        points,
        start=start,
        end=end,
        start_axis=start_axis,
        end_axis=end_axis,
        parameters=refined_parameters,
        profile=profile,
    )
    return np.minimum(best_distance, refined_distance)


def validate_tile_shape(tile_shape: Sequence[int]) -> TileShape:
    if len(tile_shape) != 3:
        raise ValueError("tile_shape must contain exactly three integers.")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in tile_shape
    ):
        raise TypeError("tile_shape must contain exactly three integers.")
    resolved = tuple(int(value) for value in tile_shape)
    if any(value <= 0 for value in resolved):
        raise ValueError("tile_shape values must all be positive.")
    return (resolved[0], resolved[1], resolved[2])


def _iter_index_tiles(
    index_bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    tile_shape: TileShape,
    grid_shape: tuple[int, int, int],
) -> Iterator[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    x_bounds, y_bounds, z_bounds = index_bounds
    first = tuple(
        (bounds[0] // tile_shape[axis]) * tile_shape[axis]
        for axis, bounds in enumerate(index_bounds)
    )
    for x_start in range(first[0], x_bounds[1], tile_shape[0]):
        for y_start in range(first[1], y_bounds[1], tile_shape[1]):
            for z_start in range(first[2], z_bounds[1], tile_shape[2]):
                yield (
                    (x_start, min(x_start + tile_shape[0], grid_shape[0])),
                    (y_start, min(y_start + tile_shape[1], grid_shape[1])),
                    (z_start, min(z_start + tile_shape[2], grid_shape[2])),
                )


def _grid_points_on_bounds(
    domain: Domain,
    index_bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> npt.NDArray[np.float64]:
    shape = tuple(stop - start for start, stop in index_bounds)
    points = np.empty((shape[0], shape[1], shape[2], 3), dtype=float)
    points[..., 0] = domain.axis_centers(0, *index_bounds[0])[:, np.newaxis, np.newaxis]
    points[..., 1] = domain.axis_centers(1, *index_bounds[1])[np.newaxis, :, np.newaxis]
    points[..., 2] = domain.axis_centers(2, *index_bounds[2])[np.newaxis, np.newaxis, :]
    return points


def _sample_point_on_bounds(
    domain: Domain,
    deposit: PointDeposit,
    profile: _ResolvedBeadProfile,
    index_bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> _SampledKernel:
    points = _grid_points_on_bounds(domain, index_bounds)
    signed_distance = rounded_cylinder_signed_distance(
        points,
        target=deposit.target.position.to_array(),
        axis=deposit.target.normal.to_array(),
        profile=profile,
    )
    values = _implicit_values_from_signed_distance(signed_distance, profile.transition_width)
    return _SampledKernel(
        slices=(
            slice(*index_bounds[0]),
            slice(*index_bounds[1]),
            slice(*index_bounds[2]),
        ),
        values=values.astype(float, copy=False),
    )


def _sample_line_on_bounds(
    domain: Domain,
    deposit: LineDeposit,
    profile: _ResolvedBeadProfile,
    index_bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> _SampledKernel:
    start = deposit.start.position.to_array()
    end = deposit.end.position.to_array()
    points = _grid_points_on_bounds(domain, index_bounds)
    flat_points = points.reshape(-1, 3)
    start_axis = deposit.start.normal.to_array()
    end_axis = deposit.end.normal.to_array()
    if _line_sweep_needs_parameter_search(start, end, start_axis, end_axis):
        signed_distance = _minimized_line_sweep_signed_distance(
            flat_points,
            start=start,
            end=end,
            start_axis=start_axis,
            end_axis=end_axis,
            profile=profile,
        )
    else:
        parameters = closest_point_parameters(flat_points, start, end)
        signed_distance = _line_sweep_signed_distance_at_parameters(
            flat_points,
            start=start,
            end=end,
            start_axis=start_axis,
            end_axis=end_axis,
            parameters=parameters,
            profile=profile,
        )
    signed_distance = signed_distance.reshape(points.shape[:-1])
    values = _implicit_values_from_signed_distance(signed_distance, profile.transition_width)
    return _SampledKernel(
        slices=(
            slice(*index_bounds[0]),
            slice(*index_bounds[1]),
            slice(*index_bounds[2]),
        ),
        values=values.astype(float, copy=False),
    )


def _iter_point_kernels(
    domain: Domain,
    deposit: PointDeposit,
    tile_shape: TileShape,
) -> Iterator[_SampledKernel]:
    profile = _resolve_bead_profile(deposit.profile, domain)
    support_min, support_max = _point_target_support_bounds(
        deposit.target.position,
        deposit.target.normal,
        width=profile.width,
        height=profile.height,
        padding=profile.support_padding,
    )
    index_bounds = domain.index_bounds_for_aabb(
        support_min.tolist(),
        support_max.tolist(),
    )
    if index_bounds is None:
        return
    for tile_bounds in _iter_index_tiles(index_bounds, tile_shape, domain.grid_shape):
        sampled = _sample_point_on_bounds(domain, deposit, profile, tile_bounds)
        if np.any(sampled.values > 0.0):
            yield sampled


def _iter_line_kernels(
    domain: Domain,
    deposit: LineDeposit,
    tile_shape: TileShape,
) -> Iterator[_SampledKernel]:
    if np.allclose(
        deposit.start.position.to_array(),
        deposit.end.position.to_array(),
    ):
        yield from _iter_point_kernels(
            domain,
            PointDeposit(
                target=deposit.start,
                profile=deposit.profile,
            ),
            tile_shape,
        )
        return

    profile = _resolve_bead_profile(deposit.profile, domain)
    support_min, support_max = deposit.support_bounds(
        padding=profile.support_padding,
    )
    index_bounds = domain.index_bounds_for_aabb(support_min, support_max)
    if index_bounds is None:
        return
    for tile_bounds in _iter_index_tiles(index_bounds, tile_shape, domain.grid_shape):
        sampled = _sample_line_on_bounds(domain, deposit, profile, tile_bounds)
        if np.any(sampled.values > 0.0):
            yield sampled


def _iter_polyline_kernels(
    domain: Domain,
    deposit: PolylineDeposit,
    tile_shape: TileShape,
) -> Iterator[_SampledKernel]:
    serial = count()
    heap: list[tuple[tuple[tuple[int, int], ...], int, _SampledKernel, Iterator[_SampledKernel]]] = []
    for segment in deposit.segments():
        iterator = _iter_line_kernels(domain, segment, tile_shape)
        sampled = next(iterator, None)
        if sampled is not None:
            bounds = tuple((int(s.start), int(s.stop)) for s in sampled.slices)
            heapq.heappush(heap, (bounds, next(serial), sampled, iterator))

    while heap:
        bounds, _, sampled, iterator = heapq.heappop(heap)
        merged = sampled.values.copy()
        next_sample = next(iterator, None)
        if next_sample is not None:
            next_bounds = tuple((int(s.start), int(s.stop)) for s in next_sample.slices)
            heapq.heappush(heap, (next_bounds, next(serial), next_sample, iterator))

        while heap and heap[0][0] == bounds:
            _, _, overlapping, overlapping_iterator = heapq.heappop(heap)
            np.maximum(merged, overlapping.values, out=merged)
            next_overlapping = next(overlapping_iterator, None)
            if next_overlapping is not None:
                next_bounds = tuple((int(s.start), int(s.stop)) for s in next_overlapping.slices)
                heapq.heappush(
                    heap,
                    (next_bounds, next(serial), next_overlapping, overlapping_iterator),
                )

        yield _SampledKernel(
            slices=(
                slice(*bounds[0]),
                slice(*bounds[1]),
                slice(*bounds[2]),
            ),
            values=merged,
        )


def iter_deposit_kernels(
    domain: Domain,
    deposit: PointDeposit | LineDeposit | PolylineDeposit,
    *,
    tile_shape: Sequence[int] = (32, 32, 32),
) -> Iterator[_SampledKernel]:
    """Yield nonempty, bounded kernel tiles for one deposition event."""

    resolved_tile_shape = validate_tile_shape(tile_shape)
    if isinstance(deposit, PointDeposit):
        yield from _iter_point_kernels(domain, deposit, resolved_tile_shape)
        return
    if isinstance(deposit, LineDeposit):
        yield from _iter_line_kernels(domain, deposit, resolved_tile_shape)
        return
    if isinstance(deposit, PolylineDeposit):
        yield from _iter_polyline_kernels(domain, deposit, resolved_tile_shape)
        return
    raise TypeError("Unsupported deposit type.")
