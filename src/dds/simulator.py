"""Main simulation orchestration and user-facing helper API."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .analysis import AnalysisBundle, normalize_field
from .domain import Domain
from .fields import accumulate_chunked_field, accumulate_deposition_index, accumulate_field
from .kernels import iter_deposit_kernels
from .occupancy import occupancy_from_density
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
        self._normalized_density_cache: npt.NDArray[np.float64] | None = None
        self._deposition_index_cache: npt.NDArray[np.intp] | None = None
        self._analysis_bundle_cache: AnalysisBundle | None = None
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
        self._normalized_density_cache = None
        self._deposition_index_cache = None
        self._analysis_bundle_cache = None
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
            self._normalized_density_cache = None
            self._analysis_bundle_cache = None
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
        self._normalized_density_cache = None
        self._analysis_bundle_cache = None

    def _density_field(self, *, normalize: bool = False) -> npt.NDArray[np.float64]:
        if self._density_max_cache is None:
            self._density_max_cache = accumulate_field(
                self.domain,
                self._deposits,
                composition="max",
            )
        if not normalize:
            return self._density_max_cache
        if self._normalized_density_cache is None:
            self._normalized_density_cache = normalize_field(self._density_max_cache)
        return self._normalized_density_cache

    def _density_max_field(self) -> npt.NDArray[np.float64]:
        return self._density_field(normalize=False)

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
        self._normalized_density_cache = None
        self._analysis_bundle_cache = None

    def analysis_bundle(self) -> AnalysisBundle:
        """Return a cached AnalysisBundle for the current density field."""

        if self._analysis_bundle_cache is None:
            self._analysis_bundle_cache = AnalysisBundle(
                self.domain,
                self._density_field(normalize=False).copy(),
                deposition_index=self._deposition_index_field().copy(),
            )
        return self._analysis_bundle_cache

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

    def sample_field(
        self,
        *,
        field: str = "density",
        threshold: float = 0.5,
        normalize: bool = False,
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.bool_]:
        """Sample the requested field using cached density when possible."""

        if field == "density":
            return self._density_field(normalize=normalize).copy()
        if field == "coverage":
            if normalize:
                raise ValueError("coverage cannot be normalized as a physical density.")
            return self._coverage_field().copy()
        if field == "deposition_index":
            return self._deposition_index_field().astype(float, copy=False)
        if field == "occupancy":
            return occupancy_from_density(
                self._density_field(normalize=normalize),
                threshold=threshold,
            )
        raise ValueError("field must be 'density', 'coverage', 'occupancy', or 'deposition_index'.")

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
    ) -> npt.NDArray[np.intp]:
        """Return the per-voxel last-deposit-index grid (0-based; -1 = untouched)."""

        return self._deposition_index_field().copy()

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
    ) -> float:
        """Query the per-voxel last-deposit index at a point (0-based; -1 = untouched)."""

        return self.sample_deposition_index_at(point)

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
    ) -> float:
        """Sample deposition index at a world-space point (0-based; -1 = untouched)."""

        return self.analysis_bundle().sample_deposition_index_at(
            point,
            interpolation=interpolation,
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
