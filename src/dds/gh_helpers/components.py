"""Component-facing helpers for Grasshopper Python 3 scripts."""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import MutableMapping as MutableMappingType
from typing import Optional, Sequence, Union, cast

import dds

from .convert import (
    box_to_domain,
    coerce_deposits,
    coerce_target,
    domain_to_rhino_box,
    plane_to_target,
    point3d_to_tuple,
    point_to_deposit,
    result_to_rhino_mesh,
    stable_signature,
    targets_to_line_deposit,
    targets_to_polyline_deposit,
)

SIM_CACHE_KEY = "dds.gh_helpers.simulation_cache"


def make_domain_from_box(
    box: object,
    voxel_size: Union[float, Sequence[float]],
) -> tuple[dds.Domain, object]:
    """Create a DDS domain and Rhino preview box from a Rhino box."""

    domain = box_to_domain(box, voxel_size=voxel_size, length_unit="mm")
    try:
        preview = domain_to_rhino_box(domain)
    except RuntimeError:
        preview = None
    return domain, preview


def make_bead_profile(width: float, height: float) -> dds.BeadProfile:
    """Create a DDS bead profile."""

    return dds.BeadProfile(width=width, height=height)


def make_target_from_plane(
    plane: object,
) -> dds.DepositionTarget:
    """Create a DDS target from a plane."""

    return plane_to_target(plane)


def make_target(
    target: object,
    *,
    normal: Optional[object] = None,
) -> dds.DepositionTarget:
    """Create a DDS target from a point, plane/frame, or existing target."""

    return coerce_target(target, normal=normal)


def make_point_deposit(
    target: object,
    profile: dds.BeadProfile,
) -> dds.PointDeposit:
    """Create a point deposit from a target-like value and bead profile."""

    return point_to_deposit(target, profile)


def make_line_deposit(
    start_target: object,
    end_target: object,
    profile: dds.BeadProfile,
    *,
    sweep_resolution: Optional[float] = None,
) -> dds.LineDeposit:
    """Create a line deposit from two target-like values and a bead profile."""

    return targets_to_line_deposit(
        start_target,
        end_target,
        profile,
        sweep_resolution=sweep_resolution,
    )


def make_polyline_deposit(
    targets: object,
    profile: dds.BeadProfile,
    *,
    sweep_resolution: Optional[float] = None,
) -> dds.PolylineDeposit:
    """Create a polyline deposit from target-like values and a bead profile."""

    return targets_to_polyline_deposit(
        targets,
        profile,
        sweep_resolution=sweep_resolution,
    )


def run_simulation(
    domain: dds.Domain,
    deposits: object,
    *,
    run: bool = True,
    reset: bool = False,
    include_coverage: bool = False,
    threshold: float = 0.5,
    use_cache: bool = True,
) -> Optional[dds.SimulationResult]:
    """Run or retrieve a DDS simulation result for a GH component."""

    deposit_tuple = coerce_deposits(deposits)
    if not run:
        return None

    signature = stable_signature(domain, deposit_tuple, include_coverage, threshold)
    cache = _simulation_cache()
    if reset and cache is not None:
        cache.clear()
    if use_cache and cache is not None and signature in cache:
        return cache[signature]

    result = dds.simulate(domain, deposit_tuple, include_coverage=include_coverage, threshold=threshold)
    if use_cache and cache is not None:
        cache[signature] = result
    return result


def make_mesh(
    result: dds.SimulationResult,
    *,
    threshold: Optional[float] = None,
    step_size: int = 1,
) -> object:
    """Extract a Rhino mesh from a DDS result."""

    return result_to_rhino_mesh(result, threshold=threshold, step_size=step_size)


def inspect_points(
    result: dds.SimulationResult,
    points: object,
    *,
    threshold: Optional[float] = None,
) -> tuple[tuple[float, ...], tuple[bool, ...], tuple[tuple[int, int, int], ...]]:
    """Sample result values at GH point inputs."""

    point_tuple = _coerce_points(points)
    threshold_value = result.default_threshold if threshold is None else float(threshold)
    values = tuple(result.analysis.sample_implicit_value(point3d_to_tuple(point)) for point in point_tuple)
    occupied = tuple(value >= threshold_value for value in values)
    indices = tuple(result.domain.world_to_index(point3d_to_tuple(point), clip=True) for point in point_tuple)
    return values, occupied, indices


def _coerce_points(points: object) -> tuple[object, ...]:
    if points is None:
        return ()
    if hasattr(points, "Branches"):
        flat: list[object] = []
        for branch in points.Branches:
            flat.extend(_coerce_points(branch))
        return tuple(flat)
    if all(hasattr(points, attr) for attr in ("X", "Y", "Z")):
        return (points,)
    try:
        return tuple(cast(Iterable[object], points))
    except TypeError:
        return (points,)


def _simulation_cache() -> Optional[MutableMappingType[str, dds.SimulationResult]]:
    try:
        import scriptcontext
    except ImportError:
        return None
    sticky = getattr(scriptcontext, "sticky", None)
    if not isinstance(sticky, MutableMapping):
        return None
    cache = sticky.get(SIM_CACHE_KEY)
    if cache is None:
        cache = {}
        sticky[SIM_CACHE_KEY] = cache
    return cache
