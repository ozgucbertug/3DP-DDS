"""Typed analysis result models for dense simulation workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..domain import Domain
from ..geometry.mesh import TriangleMesh
from ..utils import readonly_array


@dataclass(slots=True, frozen=True)
class StratumFieldSet:
    """Implicit and occupancy fields partitioned by deposit order."""

    domain: Domain
    threshold: float
    stratum_ids: tuple[int, ...]
    implicit_fields: tuple[npt.NDArray[np.float64], ...]
    occupancy_fields: tuple[npt.NDArray[np.bool_], ...]
    label_field: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "implicit_fields",
            tuple(readonly_array(field, dtype=float) for field in self.implicit_fields),
        )
        object.__setattr__(
            self,
            "occupancy_fields",
            tuple(readonly_array(field, dtype=bool) for field in self.occupancy_fields),
        )
        object.__setattr__(self, "label_field", readonly_array(self.label_field, dtype=float))
        if len(self.stratum_ids) != len(self.implicit_fields) or len(self.stratum_ids) != len(self.occupancy_fields):
            raise ValueError("stratum_ids, implicit_fields, and occupancy_fields must have the same length.")
        expected_shape = self.domain.grid_shape
        for implicit_field in self.implicit_fields:
            if implicit_field.shape != expected_shape:
                raise ValueError("implicit_fields must match the domain grid shape.")
        for occupancy in self.occupancy_fields:
            if occupancy.shape != expected_shape:
                raise ValueError("occupancy_fields must match the domain grid shape.")
        if self.label_field.shape != expected_shape:
            raise ValueError("label_field must match the domain grid shape.")

    def stratum_index(self, stratum_id: int) -> int:
        try:
            return self.stratum_ids.index(int(stratum_id))
        except ValueError as exc:
            raise KeyError(f"Unknown stratum_id {stratum_id!r}.") from exc

    def implicit_field(self, stratum_id: int) -> npt.NDArray[np.float64]:
        return self.implicit_fields[self.stratum_index(stratum_id)]

    def occupancy(self, stratum_id: int) -> npt.NDArray[np.bool_]:
        return self.occupancy_fields[self.stratum_index(stratum_id)]


@dataclass(slots=True, frozen=True)
class InterfacePairSummary:
    """Summary metrics for one adjacent ordered-deposit pair."""

    previous_id: int
    next_id: int
    contact_face_count: int
    contact_area: float
    overlap_voxel_count: int
    overlap_fraction: float


@dataclass(slots=True, frozen=True)
class InterfaceAnalysis:
    """Typed interface/contact analysis result."""

    stratum_ids: tuple[int, ...]
    contact_mask: npt.NDArray[np.bool_]
    overlap_mask: npt.NDArray[np.bool_]
    unsupported_next_mask: npt.NDArray[np.bool_]
    contact_face_count: int
    contact_area: float
    overlap_voxel_count: int
    overlap_fraction: float
    pair_summaries: tuple[InterfacePairSummary, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "contact_mask", readonly_array(self.contact_mask, dtype=bool))
        object.__setattr__(self, "overlap_mask", readonly_array(self.overlap_mask, dtype=bool))
        object.__setattr__(
            self,
            "unsupported_next_mask",
            readonly_array(self.unsupported_next_mask, dtype=bool),
        )
        object.__setattr__(self, "pair_summaries", tuple(self.pair_summaries))


@dataclass(slots=True, frozen=True)
class SupportAnalysis:
    """Typed support and overhang analysis result."""

    mesh: TriangleMesh
    build_direction: tuple[float, float, float]
    overhang_angles: npt.NDArray[np.float64]
    downfacing_mask: npt.NDArray[np.bool_]
    support_risk_mask: npt.NDArray[np.bool_]
    face_areas: npt.NDArray[np.float64]
    downfacing_area: float
    risk_area: float
    support_shadow_field: npt.NDArray[np.float64]
    shadow_voxel_count: int
    shadow_volume: float
    max_unsupported_span: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "overhang_angles", readonly_array(self.overhang_angles, dtype=float))
        object.__setattr__(self, "downfacing_mask", readonly_array(self.downfacing_mask, dtype=bool))
        object.__setattr__(
            self,
            "support_risk_mask",
            readonly_array(self.support_risk_mask, dtype=bool),
        )
        object.__setattr__(self, "face_areas", readonly_array(self.face_areas, dtype=float))
        object.__setattr__(
            self,
            "support_shadow_field",
            readonly_array(self.support_shadow_field, dtype=float),
        )
