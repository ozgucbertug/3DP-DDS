"""Typed analysis result models for dense simulation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from ..domain import Domain
from ..geometry.mesh import TriangleMesh

StratificationMode = Literal["layer", "order"]


@dataclass(slots=True)
class StratumFieldSet:
    """Max-density and occupancy fields partitioned by layer or deposit order."""

    domain: Domain
    mode: StratificationMode
    threshold: float
    stratum_ids: tuple[int, ...]
    density_max_fields: tuple[npt.NDArray[np.float64], ...]
    occupancy_fields: tuple[npt.NDArray[np.bool_], ...]
    label_field: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if len(self.stratum_ids) != len(self.density_max_fields) or len(self.stratum_ids) != len(self.occupancy_fields):
            raise ValueError("stratum_ids, density_max_fields, and occupancy_fields must have the same length.")
        expected_shape = self.domain.grid_shape
        for density in self.density_max_fields:
            if density.shape != expected_shape:
                raise ValueError("density_max_fields must match the domain grid shape.")
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

    def density(self, stratum_id: int) -> npt.NDArray[np.float64]:
        return self.density_max_fields[self.stratum_index(stratum_id)]

    def occupancy(self, stratum_id: int) -> npt.NDArray[np.bool_]:
        return self.occupancy_fields[self.stratum_index(stratum_id)]


@dataclass(slots=True)
class InterfacePairSummary:
    """Summary metrics for one adjacent layer or ordered-deposit pair."""

    previous_id: int
    next_id: int
    contact_face_count: int
    contact_area: float
    overlap_voxel_count: int
    overlap_fraction: float


@dataclass(slots=True)
class InterfaceAnalysis:
    """Typed interface/contact analysis result."""

    stratification_mode: StratificationMode
    stratum_ids: tuple[int, ...]
    contact_mask: npt.NDArray[np.bool_]
    overlap_mask: npt.NDArray[np.bool_]
    unsupported_next_mask: npt.NDArray[np.bool_]
    contact_face_count: int
    contact_area: float
    overlap_voxel_count: int
    overlap_fraction: float
    pair_summaries: tuple[InterfacePairSummary, ...]


@dataclass(slots=True)
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

