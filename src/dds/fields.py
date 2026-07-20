"""Dense scalar-field sampling helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Union

import numpy as np
import numpy.typing as npt

from .domain import Domain
from .kernels import _SampledKernel, iter_deposit_kernels
from .primitives import Deposit, DepositInput, iter_deposits
from .utils import ensure_finite_scalar

FieldName = Literal["implicit", "coverage"]


def _apply_kernel_to_field(
    fields: dict[FieldName, npt.NDArray[np.float64]],
    sampled: _SampledKernel,
) -> None:
    if "implicit" in fields:
        np.maximum(
            fields["implicit"][sampled.slices],
            sampled.values,
            out=fields["implicit"][sampled.slices],
        )
    if "coverage" in fields:
        fields["coverage"][sampled.slices] += sampled.values


def _apply_kernel_to_index_field(
    index_field: npt.NDArray[np.intp],
    sampled: _SampledKernel,
    deposit_index: int,
) -> None:
    touched = sampled.values > 0.0
    index_field[sampled.slices][touched] = deposit_index


def accumulate_fields(
    domain: Domain,
    deposits: Union[Iterable[DepositInput], DepositInput],
    *,
    include_coverage: bool = False,
) -> dict[FieldName, npt.NDArray[np.float64]]:
    """Accumulate the implicit field and optional additive coverage.

    The implicit field uses maximum composition and represents the fabricated
    geometry envelope. Coverage uses additive composition and is an overlap
    diagnostic, not a physical density.
    """

    fields: dict[FieldName, npt.NDArray[np.float64]] = {
        "implicit": np.zeros(domain.grid_shape, dtype=float)
    }
    if include_coverage:
        fields["coverage"] = np.zeros(domain.grid_shape, dtype=float)
    for deposit in iter_deposits(deposits):
        for sampled in iter_deposit_kernels(domain, deposit):
            _apply_kernel_to_field(fields, sampled)
    return fields


def accumulate_field(
    domain: Domain,
    deposits: Union[Iterable[DepositInput], DepositInput],
    *,
    field: FieldName = "implicit",
) -> npt.NDArray[np.float64]:
    """Accumulate one implicit or additive coverage field."""

    if field not in {"implicit", "coverage"}:
        raise ValueError("field must be 'implicit' or 'coverage'.")
    return accumulate_fields(
        domain,
        deposits,
        include_coverage=field == "coverage",
    )[field]


def accumulate_deposition_index(
    domain: Domain,
    deposits: Union[Iterable[DepositInput], DepositInput],
) -> npt.NDArray[np.intp]:
    """Record the zero-based index of the last deposit touching each voxel.

    Untouched voxels contain ``-1``.
    """

    index_field = np.full(domain.grid_shape, -1, dtype=np.intp)
    for deposit_index, deposit in enumerate(iter_deposits(deposits)):
        for sampled in iter_deposit_kernels(domain, deposit):
            _apply_kernel_to_index_field(index_field, sampled, deposit_index)
    return index_field


def accumulate_deposition_order(
    domain: Domain,
    deposits: Union[Iterable[DepositInput], DepositInput],
    *,
    threshold: float = 0.5,
) -> npt.NDArray[np.intp]:
    """Record the one-based order of the last deposit reaching a threshold.

    Untouched voxels contain ``0``.
    """

    threshold_value = ensure_finite_scalar(threshold, "threshold")
    if threshold_value < 0.0:
        raise ValueError("threshold must be non-negative.")

    order_field = np.zeros(domain.grid_shape, dtype=np.intp)
    for deposit_order, deposit in enumerate(iter_deposits(deposits), start=1):
        for sampled in iter_deposit_kernels(domain, deposit):
            occupied = (
                sampled.values > 0.0
                if threshold_value == 0.0
                else sampled.values >= threshold_value
            )
            order_field[sampled.slices][occupied] = deposit_order
    return order_field


def apply_deposit_to_field(
    domain: Domain,
    grid: npt.NDArray[np.float64],
    deposit: Deposit,
    *,
    field: FieldName = "implicit",
) -> bool:
    """Apply one deposit to an implicit or coverage grid in place.

    Returns
    -------
    bool
        ``True`` if at least one sampled kernel tile intersected the domain.
    """

    if field not in {"implicit", "coverage"}:
        raise ValueError("field must be 'implicit' or 'coverage'.")
    fields: dict[FieldName, npt.NDArray[np.float64]] = {field: grid}
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
    """Update an index field for one deposit using last-writer-wins semantics."""

    hit = False
    for sampled in iter_deposit_kernels(domain, deposit):
        hit = True
        _apply_kernel_to_index_field(index_field, sampled, deposit_index)
    return hit


from .chunked import accumulate_chunked_field as accumulate_chunked_field  # noqa: E402
