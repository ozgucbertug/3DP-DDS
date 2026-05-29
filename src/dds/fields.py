"""Dense scalar-field sampling helpers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from .analysis import normalize_field
from .domain import Domain
from .kernels import sample_deposit_kernel
from .occupancy import occupancy_from_density
from .primitives import DepositInput, LineDeposit, PointDeposit, iter_deposits
from .types import DensityComposition, FieldName


def accumulate_density_fields(
    domain: Domain,
    deposits: Iterable[DepositInput] | DepositInput,
    *,
    compositions: tuple[DensityComposition, ...] = ("sum",),
) -> dict[DensityComposition, npt.NDArray[np.float64]]:
    """Accumulate one or more density compositions on a dense grid."""

    requested = tuple(dict.fromkeys(compositions))
    if not requested:
        raise ValueError("compositions must contain at least one value.")
    invalid = [composition for composition in requested if composition not in {"max", "sum"}]
    if invalid:
        raise ValueError("compositions must contain only 'max' and/or 'sum'.")

    fields: dict[DensityComposition, npt.NDArray[np.float64]] = {
        composition: np.zeros(domain.grid_shape, dtype=float)
        for composition in requested
    }
    for deposit in iter_deposits(deposits):
        sampled = sample_deposit_kernel(domain, deposit)
        if sampled is None:
            continue
        if "max" in fields:
            fields["max"][sampled.slices] = np.maximum(fields["max"][sampled.slices], sampled.values)
        if "sum" in fields:
            fields["sum"][sampled.slices] += sampled.values
    return fields


def accumulate_density(
    domain: Domain,
    deposits: Iterable[DepositInput] | DepositInput,
    *,
    composition: DensityComposition = "sum",
) -> npt.NDArray[np.float64]:
    """Accumulate one density composition on a dense grid."""

    return accumulate_density_fields(domain, deposits, compositions=(composition,))[composition]


def accumulate_deposition_index(
    domain: Domain,
    deposits: Iterable[DepositInput] | DepositInput,
) -> npt.NDArray[np.intp]:
    """Return a grid recording the 0-based index of the last deposit touching each voxel.

    Voxels untouched by any deposit are set to -1.
    Last-writer wins: where multiple deposits overlap, the later one's index is stored.
    """

    index_field = np.full(domain.grid_shape, -1, dtype=np.intp)
    for deposit_index, deposit in enumerate(iter_deposits(deposits)):
        sampled = sample_deposit_kernel(domain, deposit)
        if sampled is None:
            continue
        touched = sampled.values > 0.0
        index_field[sampled.slices][touched] = deposit_index
    return index_field


def sample_field(
    domain: Domain,
    deposits: Iterable[DepositInput] | DepositInput,
    *,
    field: FieldName = "density",
    threshold: float = 0.5,
    normalize: bool = False,
) -> npt.NDArray[np.float64] | npt.NDArray[np.bool_]:
    """Sample a dense field from deposition events."""

    density = accumulate_density(domain, deposits, composition="sum")
    if field == "density":
        return normalize_field(density) if normalize else density
    if field == "deposition_index":
        return accumulate_deposition_index(domain, deposits).astype(float, copy=False)
    if field == "occupancy":
        base = normalize_field(density) if normalize else density
        return occupancy_from_density(base, threshold=threshold)
    raise ValueError("field must be 'density', 'occupancy', or 'deposition_index'.")
