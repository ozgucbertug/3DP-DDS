"""Main simulation orchestration and user-facing helper API."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from .analysis import deposition_index_from_density, normalize_field
from .domain import Domain
from .fields import accumulate_density, sample_field as sample_dense_field
from .occupancy import occupancy_from_density
from .primitives import Deposit, DepositInput, iter_deposits


def sample_field(
    domain: Domain,
    deposits: Iterable[DepositInput] | DepositInput,
    *,
    field: str = "density",
    threshold: float = 0.5,
    normalize: bool = False,
) -> npt.NDArray[np.float64] | npt.NDArray[np.bool_]:
    """Public convenience wrapper for dense field sampling."""

    return sample_dense_field(
        domain,
        deposits,
        field=field,
        threshold=threshold,
        normalize=normalize,
    )


def simulate_occupancy(
    domain: Domain,
    deposits: Iterable[DepositInput] | DepositInput,
    *,
    threshold: float = 0.5,
    normalize: bool = False,
) -> npt.NDArray[np.bool_]:
    """Return a dense occupancy grid."""

    result = sample_dense_field(
        domain,
        deposits,
        field="occupancy",
        threshold=threshold,
        normalize=normalize,
    )
    return result.astype(bool, copy=False)


def simulate_deposition_index(
    domain: Domain,
    deposits: Iterable[DepositInput] | DepositInput,
    *,
    normalize: bool = False,
) -> npt.NDArray[np.float64]:
    """Return the v0 deposition index field."""

    result = sample_dense_field(
        domain,
        deposits,
        field="deposition_index",
        normalize=normalize,
    )
    return result.astype(float, copy=False)


class Simulator:
    """Stateful simulator with cached dense fields and simple point queries."""

    def __init__(
        self,
        domain: Domain,
        deposits: Iterable[DepositInput] | DepositInput | None = None,
    ) -> None:
        self.domain = domain
        self._deposits: list[Deposit] = []
        self._density_cache: npt.NDArray[np.float64] | None = None
        self._normalized_density_cache: npt.NDArray[np.float64] | None = None
        # TODO: replace full-field invalidation with partial updates for streamed toolpaths.
        if deposits is not None:
            self.add_deposits(deposits)

    @property
    def deposits(self) -> tuple[Deposit, ...]:
        """Return the current deposit list."""

        return tuple(self._deposits)

    def _invalidate_cache(self) -> None:
        self._density_cache = None
        self._normalized_density_cache = None

    def _density_field(self, *, normalize: bool = False) -> npt.NDArray[np.float64]:
        if self._density_cache is None:
            self._density_cache = accumulate_density(self.domain, self._deposits)
        if not normalize:
            return self._density_cache
        if self._normalized_density_cache is None:
            self._normalized_density_cache = normalize_field(self._density_cache)
        return self._normalized_density_cache

    def add_deposit(self, deposit: DepositInput) -> None:
        """Add one deposit or toolpath sequence."""

        self._deposits.extend(iter_deposits(deposit))
        self._invalidate_cache()

    def add_deposits(self, deposits: Iterable[DepositInput] | DepositInput) -> None:
        """Add multiple deposits or sequences."""

        self._deposits.extend(iter_deposits(deposits))
        self._invalidate_cache()

    def clear_deposits(self) -> None:
        """Remove all deposits and reset caches."""

        self._deposits.clear()
        self._invalidate_cache()

    def sample_field(
        self,
        *,
        field: str = "density",
        threshold: float = 0.5,
        normalize: bool = False,
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.bool_]:
        """Sample the requested field using cached density when possible."""

        density = self._density_field(normalize=normalize)
        if field == "density":
            return density.copy()
        if field == "deposition_index":
            return deposition_index_from_density(density, normalize=False)
        if field == "occupancy":
            return occupancy_from_density(density, threshold=threshold)
        raise ValueError("field must be 'density', 'occupancy', or 'deposition_index'.")

    def simulate_occupancy(
        self,
        *,
        threshold: float = 0.5,
        normalize: bool = False,
    ) -> npt.NDArray[np.bool_]:
        """Return a dense occupancy grid for current deposits."""

        return self.sample_field(field="occupancy", threshold=threshold, normalize=normalize)

    def simulate_deposition_index(
        self,
        *,
        normalize: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Return a dense deposition index grid for current deposits."""

        return self.sample_field(field="deposition_index", normalize=normalize)

    def is_occupied(
        self,
        point: tuple[float, float, float],
        *,
        threshold: float = 0.5,
        normalize: bool = False,
    ) -> bool:
        """Query occupancy at a point using nearest dense-grid lookup."""

        if not self.domain.contains_point(point):
            return False
        density = self._density_field(normalize=normalize)
        index = self.domain.world_to_index(point, clip=True)
        return bool(occupancy_from_density(density[index], threshold=threshold))

    def query_deposition_index(
        self,
        point: tuple[float, float, float],
        *,
        normalize: bool = False,
    ) -> float:
        """Query the deposition index at a point using nearest-grid lookup."""
        if not self.domain.contains_point(point):
            return 0.0
        density = self._density_field(normalize=normalize)
        index = self.domain.world_to_index(point, clip=True)
        return float(deposition_index_from_density(np.asarray(density[index]), normalize=False))
