"""Analysis helpers for derived deposition maps and summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from .primitives import LineDeposit, PointDeposit, iter_deposits


def normalize_field(field: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalize a field by its maximum value when possible."""

    result = field.astype(float, copy=True)
    if result.size == 0:
        return result
    maximum = float(np.max(result))
    if maximum > 0.0:
        result /= maximum
    return result


def deposition_index_from_density(
    density: npt.NDArray[np.float64],
    *,
    normalize: bool = False,
) -> npt.NDArray[np.float64]:
    """Return the v0 deposition index field from a density-like field."""

    return normalize_field(density) if normalize else density.astype(float, copy=True)


def summarize_layers(
    deposits: Iterable[PointDeposit | LineDeposit],
) -> dict[int | None, dict[str, float | int]]:
    """Return a lightweight per-layer deposit summary."""

    summary: dict[int | None, dict[str, float | int]] = defaultdict(
        lambda: {
            "deposit_count": 0,
            "point_deposits": 0,
            "line_deposits": 0,
            "total_line_length": 0.0,
        }
    )
    for deposit in iter_deposits(deposits):
        layer_id = deposit.metadata.layer_id
        summary[layer_id]["deposit_count"] += 1
        if isinstance(deposit, PointDeposit):
            summary[layer_id]["point_deposits"] += 1
        else:
            summary[layer_id]["line_deposits"] += 1
            summary[layer_id]["total_line_length"] += deposit.segment.length
    return dict(summary)
