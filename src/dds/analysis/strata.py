"""Dense-field access partitioned by deposition order."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..fields import accumulate_fields
from ..occupancy import occupancy_from_implicit_field
from ..primitives import iter_deposits
from .models import StratumFieldSet

if TYPE_CHECKING:
    from .simulation import SimulationAnalysis

def strata(
    source: SimulationAnalysis,
    *,
    threshold: float = 0.5,
) -> StratumFieldSet:
    """Build implicit and occupancy fields for each ordered deposit."""

    deposit_tuple = tuple(iter_deposits(source.deposits))
    label_field = np.zeros(source.domain.grid_shape, dtype=float)
    implicit_fields: list[np.ndarray] = []
    occupancy_fields: list[np.ndarray] = []
    stratum_ids: list[int] = []

    for stratum_id, deposit in enumerate(deposit_tuple):
        implicit_field = accumulate_fields(
            source.domain,
            (deposit,),
        )["implicit"]
        occupancy = occupancy_from_implicit_field(
            implicit_field,
            threshold=threshold,
        )
        implicit_fields.append(implicit_field)
        occupancy_fields.append(occupancy)
        label_field[occupancy] = float(stratum_id + 1)
        stratum_ids.append(stratum_id)

    return StratumFieldSet(
        domain=source.domain,
        threshold=float(threshold),
        stratum_ids=tuple(stratum_ids),
        implicit_fields=tuple(implicit_fields),
        occupancy_fields=tuple(occupancy_fields),
        label_field=label_field,
    )
