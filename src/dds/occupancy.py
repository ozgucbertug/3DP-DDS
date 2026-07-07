"""Helpers for turning scalar accumulation fields into occupancy maps."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def occupancy_from_implicit_field(
    implicit_field: npt.NDArray[np.float64],
    threshold: float = 0.5,
) -> npt.NDArray[np.bool_]:
    """Threshold an implicit field into a binary occupancy grid.

    Parameters
    ----------
    implicit_field
        Nonnegative scalar field produced by simulation.
    threshold
        Values greater than or equal to this threshold are occupied.
    """

    if threshold < 0.0:
        raise ValueError("threshold must be non-negative.")
    return implicit_field >= threshold


def occupancy_fraction(occupancy: npt.NDArray[np.bool_]) -> float:
    """Return the occupied fraction of a boolean grid."""

    if occupancy.size == 0:
        return 0.0
    return float(np.count_nonzero(occupancy) / occupancy.size)
