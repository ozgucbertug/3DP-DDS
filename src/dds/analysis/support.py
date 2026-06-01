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
    if centroids.size == 0:
        return shadow, 0.0

    axis, direction_sign = _axis_aligned_projection_axis(build_direction)
    voxel_step = float(domain.voxel_size[axis])

    # Vectorised centroid → voxel-index conversion.
    origin = np.asarray(domain.min_corner, dtype=float)
    spacing = np.asarray(domain.voxel_size, dtype=float)
    shape_array = np.asarray(occupancy.shape, dtype=np.intp)
    raw = np.floor((centroids - origin) / spacing).astype(np.intp)
    in_bounds = np.all((raw >= 0) & (raw < shape_array), axis=1)
    valid_indices = np.clip(raw[in_bounds], 0, shape_array - 1)
    if valid_indices.size == 0:
        return shadow, 0.0

    # Mark voxels that contain a risky face centroid.
    centroid_grid = np.zeros(occupancy.shape, dtype=bool)
    centroid_grid[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = True

    # Bring the build axis to the last position for a uniform scan loop.
    occ = np.moveaxis(occupancy, axis, -1)   # (..., depth)
    cen = np.moveaxis(centroid_grid, axis, -1)
    shd = np.zeros(occ.shape, dtype=bool)

    # Propagate shadow one step at a time opposite to the build direction.
    # Shadow starts immediately below (in shadow direction) a risky centroid
    # and continues through empty voxels until blocked by occupied material.
    depth = occ.shape[-1]
    if direction_sign == 1:
        # Build goes +axis → shadow goes toward lower indices.
        for k in range(depth - 2, -1, -1):
            above = cen[..., k + 1] | shd[..., k + 1]
            shd[..., k] = above & ~occ[..., k]
    else:
        # Build goes -axis → shadow goes toward higher indices.
        for k in range(1, depth):
            below = cen[..., k - 1] | shd[..., k - 1]
            shd[..., k] = below & ~occ[..., k]

    shadow = np.moveaxis(shd, -1, axis).astype(float)

    # max_span: maximum per-column shadow depth.
    shadow_bool = shadow.astype(bool)
    column_counts = shadow_bool.sum(axis=axis)
    max_span = float(column_counts.max()) * voxel_step if shadow_bool.any() else 0.0

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
