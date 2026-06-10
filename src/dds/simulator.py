"""Main simulation orchestration and user-facing helper API."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .domain import Domain
from .fields import accumulate_chunked_field, accumulate_deposition_index, accumulate_field
from .kernels import iter_deposit_kernels
from .primitives import Deposit, DepositInput, iter_deposits
from .results import SimulationResult
from .types import FieldComposition

if TYPE_CHECKING:
    from .chunked import ChunkedField


class Simulator:
    """Stateful simulator with cached dense fields and headless analysis queries."""

    def __init__(
        self,
        domain: Domain,
        deposits: Iterable[DepositInput] | DepositInput | None = None,
    ) -> None:
        self.domain = domain
        self._deposits: list[Deposit] = []
        self._coverage_cache: npt.NDArray[np.float64] | None = None
        self._density_max_cache: npt.NDArray[np.float64] | None = None
        self._deposition_index_cache: npt.NDArray[np.intp] | None = None
        self._chunked_cache: ChunkedField | None = None
        if deposits is not None:
            self.add_deposits(deposits)

    @property
    def deposits(self) -> tuple[Deposit, ...]:
        """Return the current deposit list."""

        return tuple(self._deposits)

    def _invalidate_cache(self) -> None:
        self._coverage_cache = None
        self._density_max_cache = None
        self._deposition_index_cache = None
        self._chunked_cache = None

    def _apply_incremental(self, deposit: Deposit, index: int) -> None:
        """Apply one deposit kernel to every live dense cache in a single sample pass."""

        if not (
            self._coverage_cache is not None
            or self._density_max_cache is not None
            or self._deposition_index_cache is not None
            or self._chunked_cache is not None
        ):
            # No base caches are warm yet; skip sampling.
            return

        tile_shape = (
            self._chunked_cache.chunk_shape
            if self._chunked_cache is not None
            else (32, 32, 32)
        )
        hit = False
        for sampled in iter_deposit_kernels(
            self.domain,
            deposit,
            tile_shape=tile_shape,
        ):
            hit = True
            if self._coverage_cache is not None:
                self._coverage_cache[sampled.slices] += sampled.values
            if self._density_max_cache is not None:
                np.maximum(
                    self._density_max_cache[sampled.slices],
                    sampled.values,
                    out=self._density_max_cache[sampled.slices],
                )
            if self._deposition_index_cache is not None:
                touched = sampled.values > 0.0
                self._deposition_index_cache[sampled.slices][touched] = index
            if self._chunked_cache is not None:
                self._chunked_cache.add_kernel(sampled)
        if hit and self._chunked_cache is not None:
            self._chunked_cache.record_event()

    def _density_field(self) -> npt.NDArray[np.float64]:
        if self._density_max_cache is None:
            self._density_max_cache = accumulate_field(
                self.domain,
                self._deposits,
                composition="max",
            )
        return self._density_max_cache

    def _density_max_field(self) -> npt.NDArray[np.float64]:
        return self._density_field()

    def _coverage_field(self) -> npt.NDArray[np.float64]:
        if self._coverage_cache is None:
            self._coverage_cache = accumulate_field(
                self.domain,
                self._deposits,
                composition="coverage",
            )
        return self._coverage_cache

    def _deposition_index_field(self) -> npt.NDArray[np.intp]:
        if self._deposition_index_cache is None:
            self._deposition_index_cache = accumulate_deposition_index(self.domain, self._deposits)
        return self._deposition_index_cache

    def add_deposit(self, deposit: DepositInput) -> None:
        """Add one deposit or toolpath sequence, updating caches incrementally."""

        for leaf in iter_deposits(deposit):
            new_index = len(self._deposits)
            self._deposits.append(leaf)
            self._apply_incremental(leaf, new_index)

    def add_deposits(self, deposits: Iterable[DepositInput] | DepositInput) -> None:
        """Add multiple deposits or sequences, updating caches incrementally."""

        for leaf in iter_deposits(deposits):
            new_index = len(self._deposits)
            self._deposits.append(leaf)
            self._apply_incremental(leaf, new_index)

    def clear_deposits(self) -> None:
        """Remove all deposits, reusing grid allocations where possible."""

        self._deposits.clear()
        if self._coverage_cache is not None:
            self._coverage_cache.fill(0.0)
        if self._density_max_cache is not None:
            self._density_max_cache.fill(0.0)
        if self._deposition_index_cache is not None:
            self._deposition_index_cache.fill(-1)
        if self._chunked_cache is not None:
            self._chunked_cache.clear()

    def chunked_field(
        self,
        *,
        chunk_shape: Sequence[int] | None = None,
    ) -> ChunkedField:
        """Return a :class:`~dds.chunked.ChunkedField` kept in sync with deposits.

        The chunked field is built lazily on first access and updated
        incrementally as new deposits are added, sharing the same kernel
        sample that updates the dense caches (no extra SDF evaluation).
        """

        if self._chunked_cache is None:
            self._chunked_cache = accumulate_chunked_field(
                self.domain,
                self._deposits,
                chunk_shape=(32, 32, 32) if chunk_shape is None else chunk_shape,
            )
        elif (
            chunk_shape is not None
            and tuple(chunk_shape) != self._chunked_cache.chunk_shape
        ):
            raise ValueError(
                "chunk_shape cannot change after the chunked cache is created."
            )
        return self._chunked_cache

    def result(
        self,
        *,
        compositions: tuple[FieldComposition, ...] = ("max",),
        threshold: float = 0.5,
    ) -> SimulationResult:
        """Return a reusable SimulationResult built from cached density fields."""

        requested = tuple(dict.fromkeys(compositions))
        if not requested:
            raise ValueError("compositions must contain at least one density composition.")
        if any(composition not in {"max", "coverage"} for composition in requested):
            raise ValueError("compositions must contain only 'max' and/or 'coverage'.")
        coverage = self._coverage_field().copy() if "coverage" in requested else None
        return SimulationResult(
            domain=self.domain,
            deposits=tuple(self._deposits),
            density_max=self._density_max_field().copy(),
            coverage=coverage,
            default_threshold=threshold,
        )
