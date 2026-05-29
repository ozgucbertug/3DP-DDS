"""Typed support and overhang analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ..mesh_analysis import downfacing_mask, face_areas, face_centroids, overhang_angles, support_risk_mask
from .models import SupportAnalysis


def _resolve_result(source: Any, *, threshold: float) -> Any:
    from ..results import simulation_result

    return simulation_result(source, threshold=threshold)


def _axis_aligned_projection_axis(
    build_direction: tuple[float, float, float] | npt.ArrayLike,
) -> tuple[int, int]:
    vector = np.asarray(build_direction, dtype=float).reshape(3)
    axis = int(np.argmax(np.abs(vector)))
    if not np.isclose(abs(vector[axis]), 1.0, atol=1e-6) or not np.allclose(np.delete(vector, axis), 0.0, atol=1e-6):
        raise ValueError("support shadow is currently defined only for axis-aligned build directions.")
    return axis, int(np.sign(vector[axis]) or 1)


def _support_shadow_field(
    occupancy: npt.NDArray[np.bool_],
    centroids: npt.NDArray[np.float64],
    *,
    domain: Any,
    build_direction: tuple[float, float, float],
) -> tuple[npt.NDArray[np.float64], float]:
    shadow = np.zeros(occupancy.shape, dtype=float)
    max_span = 0.0
    axis, direction_sign = _axis_aligned_projection_axis(build_direction)
    support_step = -direction_sign
    voxel_step = float(domain.voxel_size[axis])
    shape_axis = occupancy.shape[axis]

    for centroid in centroids:
        try:
            base_index = list(domain.world_to_index(tuple(float(value) for value in centroid), clip=True))
        except ValueError:
            continue
        probe = base_index.copy()
        probe[axis] += support_step
        span = 0
        while 0 <= probe[axis] < shape_axis:
            index_tuple = tuple(int(value) for value in probe)
            if occupancy[index_tuple]:
                break
            shadow[index_tuple] = 1.0
            span += 1
            probe[axis] += support_step
        if span > 0:
            max_span = max(max_span, span * voxel_step)

    return shadow, max_span


def support(
    source: Any,
    *,
    build_direction: tuple[float, float, float] | npt.ArrayLike = (0.0, 0.0, 1.0),
    critical_angle_deg: float = 45.0,
    threshold: float = 0.5,
) -> SupportAnalysis:
    """Return typed support and overhang metrics for the max-based geometry."""

    result = _resolve_result(source, threshold=threshold)
    build_dir = tuple(float(value) for value in np.asarray(build_direction, dtype=float).reshape(3))
    mesh = result.surface_mesh(threshold=threshold)
    angles = np.asarray(overhang_angles(mesh, build_direction=build_dir), dtype=float)
    downfacing = np.asarray(downfacing_mask(mesh, build_direction=build_dir), dtype=bool)
    risk = np.asarray(
        support_risk_mask(mesh, build_direction=build_dir, critical_angle_deg=critical_angle_deg),
        dtype=bool,
    )
    areas = np.asarray(face_areas(mesh), dtype=float)
    centroids = np.asarray(face_centroids(mesh), dtype=float)
    occupancy = result.occupancy(threshold=threshold)
    risky_centroids = centroids[risk] if centroids.size else np.empty((0, 3), dtype=float)
    shadow_field, max_span = _support_shadow_field(
        occupancy,
        risky_centroids,
        domain=result.domain,
        build_direction=build_dir,
    )
    shadow_voxel_count = int(np.count_nonzero(shadow_field))
    shadow_volume = shadow_voxel_count * float(np.prod(result.domain.voxel_size))
    return SupportAnalysis(
        mesh=mesh,
        build_direction=build_dir,
        overhang_angles=angles,
        downfacing_mask=downfacing,
        support_risk_mask=risk,
        face_areas=areas,
        downfacing_area=float(np.sum(areas[downfacing])) if areas.size else 0.0,
        risk_area=float(np.sum(areas[risk])) if areas.size else 0.0,
        support_shadow_field=shadow_field,
        shadow_voxel_count=shadow_voxel_count,
        shadow_volume=shadow_volume,
        max_unsupported_span=float(max_span),
    )
