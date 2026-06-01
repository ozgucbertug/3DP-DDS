"""Interface contact and overlap analysis for layered or ordered deposition."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .models import InterfaceAnalysis, InterfacePairSummary, StratumFieldSet
from .strata import strata as build_strata

StrataMode = Literal["auto", "layer", "order"]


def _contact_for_pair(
    previous: np.ndarray,
    next_field: np.ndarray,
    *,
    voxel_size: tuple[float, float, float],
) -> tuple[np.ndarray, int, float]:
    contact = previous & next_field
    face_count = 0
    contact_area = 0.0
    face_areas = (
        float(voxel_size[1] * voxel_size[2]),
        float(voxel_size[0] * voxel_size[2]),
        float(voxel_size[0] * voxel_size[1]),
    )

    for axis, face_area in enumerate(face_areas):
        prev_forward = [slice(None)] * 3
        prev_backward = [slice(None)] * 3
        next_forward = [slice(None)] * 3
        next_backward = [slice(None)] * 3
        prev_forward[axis] = slice(1, None)
        prev_backward[axis] = slice(None, -1)
        next_forward[axis] = slice(1, None)
        next_backward[axis] = slice(None, -1)

        touches_forward = previous[tuple(prev_forward)] & next_field[tuple(next_backward)]
        touches_backward = previous[tuple(prev_backward)] & next_field[tuple(next_forward)]

        if np.any(touches_forward):
            face_count += int(np.count_nonzero(touches_forward))
            contact_area += face_area * float(np.count_nonzero(touches_forward))
            contribution = np.zeros_like(next_field, dtype=bool)
            contribution[tuple(next_backward)] = touches_forward
            contact |= contribution
        if np.any(touches_backward):
            face_count += int(np.count_nonzero(touches_backward))
            contact_area += face_area * float(np.count_nonzero(touches_backward))
            contribution = np.zeros_like(next_field, dtype=bool)
            contribution[tuple(next_forward)] = touches_backward
            contact |= contribution

    return contact, face_count, contact_area


def interface(
    source: Any,
    *,
    mode: StrataMode = "auto",
    threshold: float = 0.5,
) -> InterfaceAnalysis:
    """Compute aggregate contact and overlap metrics across consecutive strata."""

    field_set = build_strata(source, mode=mode, threshold=threshold)
    occupancy_fields = field_set.occupancy_fields
    stratum_ids = field_set.stratum_ids
    shape = field_set.domain.grid_shape

    contact_mask = np.zeros(shape, dtype=bool)
    overlap_mask = np.zeros(shape, dtype=bool)
    unsupported_next_mask = np.zeros(shape, dtype=bool)
    summaries: list[InterfacePairSummary] = []
    total_contact_faces = 0
    total_contact_area = 0.0
    total_overlap_voxels = 0
    total_next_voxels = 0
    voxel_size = field_set.domain.voxel_size

    for index in range(len(stratum_ids) - 1):
        previous_id = stratum_ids[index]
        next_id = stratum_ids[index + 1]
        previous = occupancy_fields[index]
        next_field = occupancy_fields[index + 1]

        overlap = previous & next_field
        pair_contact_mask, face_count, contact_area = _contact_for_pair(previous, next_field, voxel_size=voxel_size)
        pair_unsupported_next = next_field & ~pair_contact_mask

        overlap_count = int(np.count_nonzero(overlap))
        next_count = int(np.count_nonzero(next_field))
        overlap_fraction = float(overlap_count / next_count) if next_count > 0 else 0.0

        contact_mask |= pair_contact_mask
        overlap_mask |= overlap
        unsupported_next_mask |= pair_unsupported_next
        total_contact_faces += face_count
        total_contact_area += contact_area
        total_overlap_voxels += overlap_count
        total_next_voxels += next_count
        summaries.append(
            InterfacePairSummary(
                previous_id=previous_id,
                next_id=next_id,
                contact_face_count=face_count,
                contact_area=float(contact_area),
                overlap_voxel_count=overlap_count,
                overlap_fraction=overlap_fraction,
            )
        )

    aggregate_overlap_fraction = float(total_overlap_voxels / total_next_voxels) if total_next_voxels > 0 else 0.0
    return InterfaceAnalysis(
        stratification_mode=field_set.mode,
        stratum_ids=stratum_ids,
        contact_mask=contact_mask,
        overlap_mask=overlap_mask,
        unsupported_next_mask=unsupported_next_mask,
        contact_face_count=total_contact_faces,
        contact_area=float(total_contact_area),
        overlap_voxel_count=total_overlap_voxels,
        overlap_fraction=aggregate_overlap_fraction,
        pair_summaries=tuple(summaries),
    )
