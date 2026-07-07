"""Mutable deposition construction and dense simulation snapshots."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from .domain import Domain
from .fields import accumulate_field, accumulate_fields
from .kernels import iter_deposit_kernels
from .primitives import Deposit, DepositInput, iter_deposits
from .results import SimulationResult


class Simulator:
    """Stateful deposit collection with incrementally updated dense caches.

    Parameters
    ----------
    domain
        Simulation domain used for all deposits and result snapshots.
    deposits
        Optional initial deposit or iterable of deposits.

    Notes
    -----
    ``Simulator`` is the mutable construction API. Calls to :meth:`result`
    return immutable snapshots; later calls to :meth:`add_deposit`,
    :meth:`add_deposits`, or :meth:`clear_deposits` do not mutate snapshots
    that have already been returned.
    """

    def __init__(
        self,
        domain: Domain,
        deposits: Iterable[DepositInput] | DepositInput | None = None,
    ) -> None:
        self.domain = domain
        self._deposits: list[Deposit] = []
        self._coverage_cache: npt.NDArray[np.float64] | None = None
        self._implicit_field_cache: npt.NDArray[np.float64] | None = None
        if deposits is not None:
            self.add_deposits(deposits)

    @property
    def deposits(self) -> tuple[Deposit, ...]:
        """Deposits currently stored by the simulator in simulation order."""

        return tuple(self._deposits)

    def _apply_incremental(self, deposit: Deposit) -> None:
        if (
            self._coverage_cache is None
            and self._implicit_field_cache is None
        ):
            return
        for sampled in iter_deposit_kernels(self.domain, deposit):
            if self._coverage_cache is not None:
                self._coverage_cache[sampled.slices] += sampled.values
            if self._implicit_field_cache is not None:
                np.maximum(
                    self._implicit_field_cache[sampled.slices],
                    sampled.values,
                    out=self._implicit_field_cache[sampled.slices],
                )

    def _implicit_field(self) -> npt.NDArray[np.float64]:
        if self._implicit_field_cache is None:
            self._implicit_field_cache = accumulate_field(
                self.domain,
                self._deposits,
                field="implicit",
            )
        return self._implicit_field_cache

    def _coverage_field(self) -> npt.NDArray[np.float64]:
        if self._coverage_cache is None:
            self._coverage_cache = accumulate_field(
                self.domain,
                self._deposits,
                field="coverage",
            )
        return self._coverage_cache

    def _warm_result_fields(self, *, include_coverage: bool) -> None:
        if (
            include_coverage
            and self._implicit_field_cache is None
            and self._coverage_cache is None
        ):
            fields = accumulate_fields(
                self.domain,
                self._deposits,
                include_coverage=True,
            )
            self._implicit_field_cache = fields["implicit"]
            self._coverage_cache = fields["coverage"]
            return
        self._implicit_field()
        if include_coverage:
            self._coverage_field()

    def add_deposit(self, deposit: DepositInput) -> None:
        """Add one deposit, updating any warm dense caches.

        Parameters
        ----------
        deposit
            Point, line, or polyline deposit to append.
        """

        for leaf in iter_deposits(deposit):
            self._deposits.append(leaf)
            self._apply_incremental(leaf)

    def add_deposits(self, deposits: Iterable[DepositInput] | DepositInput) -> None:
        """Add one or more deposits, updating any warm dense caches."""

        for leaf in iter_deposits(deposits):
            self._deposits.append(leaf)
            self._apply_incremental(leaf)

    def clear_deposits(self) -> None:
        """Remove all deposits while retaining allocated dense arrays."""

        self._deposits.clear()
        if self._coverage_cache is not None:
            self._coverage_cache.fill(0.0)
        if self._implicit_field_cache is not None:
            self._implicit_field_cache.fill(0.0)

    def result(
        self,
        *,
        include_coverage: bool = False,
        threshold: float = 0.5,
    ) -> SimulationResult:
        """Return an immutable snapshot of the current simulation.

        Parameters
        ----------
        include_coverage
            Include additive coverage diagnostics in the result.
        threshold
            Default threshold stored on the result for analysis queries.

        Returns
        -------
        SimulationResult
            Snapshot containing read-only copies of the computed fields.
        """

        self._warm_result_fields(include_coverage=include_coverage)
        coverage = self._coverage_cache if include_coverage else None
        assert self._implicit_field_cache is not None
        return SimulationResult(
            domain=self.domain,
            deposits=tuple(self._deposits),
            implicit_field=self._implicit_field_cache,
            coverage=coverage,
            default_threshold=threshold,
        )
