"""Typed support and overhang analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from ..mesh_analysis import _oriented_face_data, _overhang_angles_from_normals
from .models import SupportAnalysis

if TYPE_CHECKING:
    from .simulation import SimulationAnalysis

BuildDirection = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
BUILD_DIRECTION_VECTORS: dict[BuildDirection, tuple[float, float, float]] = {
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
}
_BUILD_DIRECTIONS = BUILD_DIRECTION_VECTORS  # internal alias


def _axis_aligned_projection_axis(
    build_direction: BuildDirection,
) -> tuple[int, int]:
    vector = np.asarray(_BUILD_DIRECTIONS[build_direction], dtype=float)
    axis = int(np.argmax(np.abs(vector)))
    return axis, int(np.sign(vector[axis]) or 1)


def _longest_true_run(values: npt.NDArray[np.bool_]) -> int:
    padded = np.pad(values, ((0, 0), (1, 1)), constant_values=False)
    transitions = np.diff(padded.astype(np.int8), axis=1)
    longest = 0
    for row in transitions:
        starts = np.flatnonzero(row == 1)
        stops = np.flatnonzero(row == -1)
        if starts.size:
            longest = max(longest, int(np.max(stops - starts)))
    return longest


def _support_shadow_field(
    occupancy: npt.NDArray[np.bool_],
    centroids: npt.NDArray[np.float64],
    *,
    domain: Any,
    build_direction: BuildDirection,
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
    shadow_bool = np.moveaxis(shadow.astype(bool), axis, -1)
    columns = shadow_bool.reshape(-1, shadow_bool.shape[-1])
    max_span = float(_longest_true_run(columns)) * voxel_step

    return shadow, max_span


def support(
    source: SimulationAnalysis,
    *,
    build_direction: BuildDirection = "+Z",
    critical_angle_deg: float = 45.0,
    threshold: float = 0.5,
) -> SupportAnalysis:
    """Return typed support and overhang metrics for the max-based geometry."""

    if build_direction not in _BUILD_DIRECTIONS:
        raise ValueError(f"build_direction must be one of {sorted(_BUILD_DIRECTIONS)}.")
    build_dir = _BUILD_DIRECTIONS[build_direction]
    mesh = source.surface_mesh(threshold=threshold)
    if critical_angle_deg < 0.0:
        raise ValueError("critical_angle_deg must be non-negative.")
    face_data = _oriented_face_data(mesh)
    angles = _overhang_angles_from_normals(face_data.normals, build_dir)
    downfacing = angles < 90.0
    risk = angles <= float(critical_angle_deg)
    areas = face_data.areas
    centroids = face_data.centroids
    occupancy = source.occupancy(threshold=threshold)
    risky_centroids = centroids[risk] if centroids.size else np.empty((0, 3), dtype=float)
    shadow_field, max_span = _support_shadow_field(
        occupancy,
        risky_centroids,
        domain=source.domain,
        build_direction=build_direction,
    )
    shadow_voxel_count = int(np.count_nonzero(shadow_field))
    shadow_volume = shadow_voxel_count * float(np.prod(source.domain.voxel_size))
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
