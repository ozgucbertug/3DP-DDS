"""Main simulation orchestration and user-facing helper API."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt

from .analysis import AnalysisBundle, deposition_index_from_density, normalize_field
from .domain import Domain
from .fields import accumulate_density, sample_field as sample_dense_field
from .occupancy import occupancy_from_density
from .primitives import Deposit, DepositInput, iter_deposits
from .results import DensityComposition, SimulationResult, simulate


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
    """Stateful simulator with cached dense fields and headless analysis queries."""

    def __init__(
        self,
        domain: Domain,
        deposits: Iterable[DepositInput] | DepositInput | None = None,
    ) -> None:
        self.domain = domain
        self._deposits: list[Deposit] = []
        self._density_cache: npt.NDArray[np.float64] | None = None
        self._normalized_density_cache: npt.NDArray[np.float64] | None = None
        self._analysis_bundle_cache: AnalysisBundle | None = None
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
        self._analysis_bundle_cache = None

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

    def analysis_bundle(self) -> AnalysisBundle:
        """Return a cached AnalysisBundle for the current density field."""

        if self._analysis_bundle_cache is None:
            self._analysis_bundle_cache = AnalysisBundle(self.domain, self._density_field(normalize=False))
        return self._analysis_bundle_cache

    def result(
        self,
        *,
        compositions: tuple[DensityComposition, ...] = ("max",),
        threshold: float = 0.5,
    ) -> SimulationResult:
        """Return a reusable SimulationResult for the current deposits."""

        return simulate(
            self.domain,
            self.deposits,
            compositions=compositions,
            threshold=threshold,
        )

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

        return self.contains_point(point, representation="occupancy", threshold=threshold, normalize=normalize)

    def query_deposition_index(
        self,
        point: tuple[float, float, float],
        *,
        normalize: bool = False,
    ) -> float:
        """Query the deposition index at a point using nearest-grid lookup."""

        return self.sample_deposition_index_at(point, normalize=normalize)

    def contains_point(
        self,
        point: tuple[float, float, float],
        *,
        representation: str = "occupancy",
        threshold: float = 0.5,
        interpolation: str = "nearest",
        normalize: bool = False,
        step_size: int = 1,
    ) -> bool:
        """Query whether a point lies inside a chosen representation."""

        return self.analysis_bundle().contains_point(
            point,
            representation=representation,
            threshold=threshold,
            interpolation=interpolation,
            normalize=normalize,
            step_size=step_size,
        )

    def sample_density_at(
        self,
        point: tuple[float, float, float],
        *,
        interpolation: str = "nearest",
        normalize: bool = False,
    ) -> float:
        """Sample density at a world-space point."""

        return self.analysis_bundle().sample_density_at(
            point,
            interpolation=interpolation,
            normalize=normalize,
        )

    def sample_deposition_index_at(
        self,
        point: tuple[float, float, float],
        *,
        interpolation: str = "nearest",
        normalize: bool = False,
    ) -> float:
        """Sample deposition index at a world-space point."""

        return self.analysis_bundle().sample_deposition_index_at(
            point,
            interpolation=interpolation,
            normalize=normalize,
        )

    def signed_distance_at(
        self,
        point: tuple[float, float, float],
        *,
        threshold: float = 0.5,
        normalize: bool = False,
        source: str = "surface_sdf",
        step_size: int = 1,
    ) -> float:
        """Sample signed distance relative to the current analysis surface."""

        return self.analysis_bundle().signed_distance_at(
            point,
            threshold=threshold,
            normalize=normalize,
            source=source,
            step_size=step_size,
        )

    def surface_normal_at(
        self,
        point: tuple[float, float, float],
        *,
        threshold: float = 0.5,
        normalize: bool = False,
        source: str = "surface_sdf",
        step_size: int = 1,
    ) -> tuple[float, float, float]:
        """Estimate a surface normal from the sampled signed-distance gradient."""

        return self.analysis_bundle().surface_normal_at(
            point,
            threshold=threshold,
            normalize=normalize,
            source=source,
            step_size=step_size,
        )

    def sample_points(
        self,
        points: npt.ArrayLike,
        *,
        fields: tuple[str, ...] = ("density", "occupancy", "deposition_index", "signed_distance"),
        threshold: float = 0.5,
        interpolation: str = "nearest",
        normalize: bool = False,
    ) -> dict[str, npt.NDArray[np.generic]]:
        """Sample one or more derived fields at many world-space points."""

        return self.analysis_bundle().sample_points(
            points,
            fields=fields,
            threshold=threshold,
            interpolation=interpolation,
            normalize=normalize,
        )

    def subvolume_stats(
        self,
        bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
        *,
        threshold: float = 0.5,
        normalize: bool = False,
        step_size: int = 1,
    ) -> dict[str, float]:
        """Return summary statistics for a bounded region of interest."""

        return self.analysis_bundle().subvolume_stats(
            bounds,
            threshold=threshold,
            normalize=normalize,
            step_size=step_size,
        )

    def mesh_analysis(
        self,
        *,
        build_direction: tuple[float, float, float] | npt.ArrayLike = (0.0, 0.0, 1.0),
        critical_angle_deg: float = 45.0,
        threshold: float = 0.5,
        normalize: bool = False,
        step_size: int = 1,
    ) -> dict[str, Any]:
        """Return cached headless mesh-analysis metrics for the current surface."""

        return self.analysis_bundle().mesh_analysis(
            build_direction=build_direction,
            critical_angle_deg=critical_angle_deg,
            threshold=threshold,
            normalize=normalize,
            step_size=step_size,
        )
