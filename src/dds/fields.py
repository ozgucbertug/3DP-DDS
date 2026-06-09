"""Dense scalar-field sampling helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .analysis import normalize_field
from .domain import Domain
from .kernels import iter_deposit_kernels
from .occupancy import occupancy_from_density
from .primitives import Deposit, DepositInput, iter_deposits
from .types import FieldComposition, FieldName

if TYPE_CHECKING:
    from .chunked import ChunkedField


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
            if "max" in fields:
                np.maximum(
                    fields["max"][sampled.slices],
                    sampled.values,
                    out=fields["max"][sampled.slices],
                )
            if "coverage" in fields:
                fields["coverage"][sampled.slices] += sampled.values
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
            touched = sampled.values > 0.0
            index_field[sampled.slices][touched] = deposit_index
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
    hit = False
    for sampled in iter_deposit_kernels(domain, deposit):
        hit = True
        if composition == "coverage":
            grid[sampled.slices] += sampled.values
        elif composition == "max":
            np.maximum(grid[sampled.slices], sampled.values, out=grid[sampled.slices])
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
        touched = sampled.values > 0.0
        index_field[sampled.slices][touched] = deposit_index
    return hit


def sample_field(
    domain: Domain,
    deposits: Iterable[DepositInput] | DepositInput,
    *,
    field: FieldName = "density",
    threshold: float = 0.5,
    normalize: bool = False,
) -> npt.NDArray[np.float64] | npt.NDArray[np.bool_]:
    """Sample a dense field from deposition events."""

    density = accumulate_field(domain, deposits, composition="max")
    if field == "density":
        return normalize_field(density) if normalize else density
    if field == "coverage":
        if normalize:
            raise ValueError("coverage cannot be normalized as a physical density.")
        return accumulate_field(domain, deposits, composition="coverage")
    if field == "deposition_index":
        return accumulate_deposition_index(domain, deposits).astype(float, copy=False)
    if field == "occupancy":
        base = normalize_field(density) if normalize else density
        return occupancy_from_density(base, threshold=threshold)
    raise ValueError("field must be 'density', 'coverage', 'occupancy', or 'deposition_index'.")


def accumulate_chunked_field(
    domain: Domain,
    deposits: Iterable[DepositInput] | DepositInput,
    *,
    chunk_shape: Sequence[int] = (32, 32, 32),
) -> "ChunkedField":
    """Build a chunked field without allocating full-domain dense arrays.

    Parameters
    ----------
    domain:
        Simulation domain.
    deposits:
        One or more deposit primitives or sequences thereof.
    """

    from .chunked import ChunkedField

    chunked = ChunkedField(domain, chunk_shape=tuple(chunk_shape))
    for deposit in iter_deposits(deposits):
        hit = False
        for sampled in iter_deposit_kernels(
            domain,
            deposit,
            tile_shape=chunked.chunk_shape,
        ):
            hit = chunked.add_kernel(sampled) or hit
        if hit:
            chunked.record_event()
    return chunked
