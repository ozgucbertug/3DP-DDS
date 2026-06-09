"""Stratified dense-field access for layered or ordered deposition workflows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np

from ..fields import accumulate_fields
from ..occupancy import occupancy_from_density
from ..primitives import Deposit, iter_deposits
from .models import StratumFieldSet, StratificationMode

StrataMode = Literal["auto", "layer", "order"]


def _resolve_result(source: Any, *, threshold: float) -> Any:
    from ..results import simulation_result

    return simulation_result(source, threshold=threshold)


def _real_layer_ids(deposits: Iterable[Deposit]) -> tuple[int, ...]:
    values = sorted({deposit.metadata.layer_id for deposit in deposits if deposit.metadata.layer_id is not None})
    return tuple(int(value) for value in values)


def _resolve_mode(result: Any, mode: StrataMode) -> StratificationMode:
    if mode == "auto":
        return "layer" if len(_real_layer_ids(result.deposits)) >= 2 else "order"
    if mode not in {"layer", "order"}:
        raise ValueError("mode must be 'auto', 'layer', or 'order'.")
    return mode


def _group_deposits(
    deposits: tuple[Deposit, ...],
    *,
    mode: StratificationMode,
) -> tuple[tuple[int, tuple[Deposit, ...]], ...]:
    if mode == "order":
        return tuple((index, (deposit,)) for index, deposit in enumerate(deposits))

    grouped: dict[int, list[Deposit]] = defaultdict(list)
    for deposit in deposits:
        if deposit.metadata.layer_id is None:
            continue
        grouped[int(deposit.metadata.layer_id)].append(deposit)
    if not grouped:
        raise ValueError("Requested layer stratification but deposits do not define at least one layer_id.")
    return tuple((layer_id, tuple(grouped[layer_id])) for layer_id in sorted(grouped))


def strata(
    source: Any,
    *,
    mode: StrataMode = "auto",
    threshold: float = 0.5,
) -> StratumFieldSet:
    """Build max-density and occupancy fields for each layer or ordered deposit stratum."""

    result = _resolve_result(source, threshold=threshold)
    resolved_mode = _resolve_mode(result, mode)
    deposit_tuple = tuple(iter_deposits(result.deposits))
    groups = _group_deposits(deposit_tuple, mode=resolved_mode)
    label_field = np.zeros(result.domain.grid_shape, dtype=float)
    density_fields: list[np.ndarray] = []
    occupancy_fields: list[np.ndarray] = []
    stratum_ids: list[int] = []

    for position, (stratum_id, grouped_deposits) in enumerate(groups, start=1):
        density = accumulate_fields(
            result.domain,
            grouped_deposits,
            compositions=("max",),
        )["max"]
        occupancy = occupancy_from_density(density, threshold=threshold)
        density_fields.append(density)
        occupancy_fields.append(occupancy)
        label_field[occupancy] = float(position)
        stratum_ids.append(int(stratum_id))

    return StratumFieldSet(
        domain=result.domain,
        mode=resolved_mode,
        threshold=float(threshold),
        stratum_ids=tuple(stratum_ids),
        density_max_fields=tuple(density_fields),
        occupancy_fields=tuple(occupancy_fields),
        label_field=label_field,
    )
