"""Dense scalar-field sampling helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .domain import Domain
from .kernels import SampledKernel, iter_deposit_kernels
from .primitives import Deposit, DepositInput, iter_deposits
from .types import FieldComposition

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Private kernel application helpers — shared by all accumulation functions.
# ---------------------------------------------------------------------------

def _apply_kernel_to_field(
    fields: dict[FieldComposition, npt.NDArray[np.float64]],
    sampled: SampledKernel,
) -> None:
    """Apply one sampled kernel tile to every requested composition in *fields*."""
    if "max" in fields:
        np.maximum(
            fields["max"][sampled.slices],
            sampled.values,
            out=fields["max"][sampled.slices],
        )
    if "coverage" in fields:
        fields["coverage"][sampled.slices] += sampled.values


def _apply_kernel_to_index_field(
    index_field: npt.NDArray[np.intp],
    sampled: SampledKernel,
    deposit_index: int,
) -> None:
    """Write *deposit_index* to every voxel in *index_field* touched by *sampled*."""
    touched = sampled.values > 0.0
    index_field[sampled.slices][touched] = deposit_index


# ---------------------------------------------------------------------------
# Public accumulation API
# ---------------------------------------------------------------------------

def accumulate_fields(
    domain: Domain,
    deposits: Iterable[DepositInput] | DepositInput,
    *,
    compositions: tuple[FieldComposition, ...] = ("max",),
) -> dict[FieldComposition, npt.NDArray[np.float64]]:
    """Accumulate max-envelope and/or nonphysical coverage fields."""

    requested = tuple(dict.fromkeys(compositions))
    if not requested:
        raise ValueError("compositions must contain at least one value.")
    invalid = [composition for composition in requested if composition not in {"max", "coverage"}]
    if invalid:
        raise ValueError("compositions must contain only 'max' and/or 'coverage'.")

    fields: dict[FieldComposition, npt.NDArray[np.float64]] = {
        composition: np.zeros(domain.grid_shape, dtype=float)
        for composition in requested
    }
    for deposit in iter_deposits(deposits):
        for sampled in iter_deposit_kernels(domain, deposit):
            _apply_kernel_to_field(fields, sampled)
    return fields


def accumulate_field(
    domain: Domain,
    deposits: Iterable[DepositInput] | DepositInput,
    *,
    composition: FieldComposition = "max",
) -> npt.NDArray[np.float64]:
    """Accumulate one max-envelope or nonphysical coverage field."""

    return accumulate_fields(domain, deposits, compositions=(composition,))[composition]


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
        for sampled in iter_deposit_kernels(domain, deposit):
            _apply_kernel_to_index_field(index_field, sampled, deposit_index)
    return index_field


def apply_deposit_to_field(
    domain: Domain,
    grid: npt.NDArray[np.float64],
    deposit: Deposit,
    *,
    composition: FieldComposition = "max",
) -> bool:
    """Apply one deposit kernel to *grid* in-place.

    Returns ``True`` when the kernel overlapped the domain and was applied,
    ``False`` when the deposit falls entirely outside the domain.

    Parameters
    ----------
    domain:
        Simulation domain.
    grid:
        Dense float64 array of shape ``domain.grid_shape``.  Modified in-place.
    deposit:
        A single point, line, or polyline deposition event.
    composition:
        ``"max"`` takes the geometric envelope. ``"coverage"`` adds kernel
        samples as a nonphysical overlap diagnostic whose values depend on
        voxel resolution and path segmentation.
    """

    if composition not in {"max", "coverage"}:
        raise ValueError("composition must be 'max' or 'coverage'.")
    # Wrap the caller's grid in the shared dict form so _apply_kernel_to_field can be reused.
    fields: dict[FieldComposition, npt.NDArray[np.float64]] = {composition: grid}
    hit = False
    for sampled in iter_deposit_kernels(domain, deposit):
        hit = True
        _apply_kernel_to_field(fields, sampled)
    return hit


def apply_deposit_to_index_field(
    domain: Domain,
    index_field: npt.NDArray[np.intp],
    deposit: Deposit,
    deposit_index: int,
) -> bool:
    """Update *index_field* in-place for one deposit using last-writer-wins semantics.

    Returns ``True`` when the kernel overlapped the domain, ``False`` otherwise.

    Parameters
    ----------
    domain:
        Simulation domain.
    index_field:
        Dense ``np.intp`` array of shape ``domain.grid_shape`` initialised to -1.
    deposit:
        A single point, line, or polyline deposition event.
    deposit_index:
        0-based index of this deposit; written to every voxel whose kernel value
        is strictly positive.
    """

    hit = False
    for sampled in iter_deposit_kernels(domain, deposit):
        hit = True
        _apply_kernel_to_index_field(index_field, sampled, deposit_index)
    return hit


# Re-exported for backward compatibility — implementation lives in chunked.py.
from .chunked import accumulate_chunked_field as accumulate_chunked_field  # noqa: E402
