"""Dense scalar-field sampling helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import numpy.typing as npt

from .analysis import deposition_index_from_density, normalize_field
from .domain import Domain
from .kernels import sample_deposit_kernel
from .occupancy import occupancy_from_density
from .primitives import DepositInput, LineDeposit, PointDeposit, iter_deposits

FieldName = Literal["density", "occupancy", "deposition_index"]


def accumulate_density(
    domain: Domain,
    deposits: Iterable[DepositInput] | DepositInput,
) -> npt.NDArray[np.float64]:
    """Accumulate weighted deposit contributions on a dense grid."""

    field = np.zeros(domain.grid_shape, dtype=float)
    for deposit in iter_deposits(deposits):
        sampled = sample_deposit_kernel(domain, deposit)
        if sampled is None:
            continue
        field[sampled.slices] += sampled.values
    return field


def sample_field(
    domain: Domain,
    deposits: Iterable[DepositInput] | DepositInput,
    *,
    field: FieldName = "density",
    threshold: float = 0.5,
    normalize: bool = False,
) -> npt.NDArray[np.float64] | npt.NDArray[np.bool_]:
    """Sample a dense field from deposition events."""

    density = accumulate_density(domain, deposits)
    if field == "density":
        return normalize_field(density) if normalize else density
    if field == "deposition_index":
        return deposition_index_from_density(density, normalize=normalize)
    if field == "occupancy":
        base = normalize_field(density) if normalize else density
        return occupancy_from_density(base, threshold=threshold)
    raise ValueError("field must be 'density', 'occupancy', or 'deposition_index'.")
