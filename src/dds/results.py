"""Result containers and high-level simulation entry points."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import numpy.typing as npt

from .analysis import (
    AnalysisBundle,
    InterfaceAnalysis,
    StratumFieldSet,
    SupportAnalysis,
    interface as build_interface,
    strata as build_strata,
    support as build_support,
)
from .fields import accumulate_density_fields
from .io import save_array, save_simulation_bundle
from .primitives import Deposit, DepositInput, iter_deposits
from .domain import Domain
from .types import DensityComposition

ViewMode = Literal["surface", "occupancy", "density"]
ViewColorMode = Literal["plain", "normals", "overhang"]
ViewScalarField = Literal["occupancy", "density", "accumulation", "deposition_order"]


@dataclass(slots=True, frozen=True)
class WorkbenchViewConfig:
    """Initial viewer state for SimulationWorkbench / SimulationResult.show()."""

    view_mode: ViewMode = "surface"
    scalar_field: ViewScalarField | None = None
    color_mode: ViewColorMode | None = None
    build_direction: str | tuple[float, float, float] = "+Z"
    _VALID_DIRECTION_STRINGS: ClassVar[frozenset[str]] = frozenset(
        {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}
    )

    def __post_init__(self) -> None:
        if isinstance(self.build_direction, str) and self.build_direction not in self._VALID_DIRECTION_STRINGS:
            raise ValueError(
                f"build_direction string {self.build_direction!r} is not valid. "
                f"Must be one of {sorted(self._VALID_DIRECTION_STRINGS)}."
            )


@dataclass(slots=True)
class SimulationResult:
    """Reusable simulation outputs and derived geometry/query helpers."""

    domain: Domain
    deposits: tuple[Deposit, ...]
    density_max: npt.NDArray[np.float64]
    density_sum: npt.NDArray[np.float64] | None = None
    default_threshold: float = 0.5
    _analysis_bundle_cache: AnalysisBundle | None = field(default=None, init=False, repr=False)
    _deposition_index_cache: npt.NDArray[np.intp] | None = field(default=None, init=False, repr=False)
    _strata_cache: dict[tuple[str, float], StratumFieldSet] = field(default_factory=dict, init=False, repr=False)
    _interface_cache: dict[tuple[str, float], InterfaceAnalysis] = field(default_factory=dict, init=False, repr=False)
    _support_cache: dict[tuple[tuple[float, float, float], float, float], SupportAnalysis] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.deposits = tuple(self.deposits)
        self.density_max = np.asarray(self.density_max, dtype=float)
        if self.density_max.shape != self.domain.grid_shape:
            raise ValueError(
                f"density_max shape {self.density_max.shape} does not match domain grid shape {self.domain.grid_shape}."
            )
        if self.density_sum is not None:
            self.density_sum = np.asarray(self.density_sum, dtype=float)
            if self.density_sum.shape != self.domain.grid_shape:
                raise ValueError(
                    f"density_sum shape {self.density_sum.shape} does not match domain grid shape {self.domain.grid_shape}."
                )
        self.default_threshold = float(self.default_threshold)

    def _deposition_index_field(self) -> npt.NDArray[np.intp]:
        if self._deposition_index_cache is None:
            from .fields import accumulate_deposition_index
            self._deposition_index_cache = accumulate_deposition_index(self.domain, self.deposits)
        return self._deposition_index_cache

    def analysis_bundle(self) -> AnalysisBundle:
        """Return the canonical analysis bundle derived from density_max."""

        if self._analysis_bundle_cache is None:
            self._analysis_bundle_cache = AnalysisBundle(
                self.domain,
                self.density_max,
                deposition_index=self._deposition_index_field(),
            )
        return self._analysis_bundle_cache

    def layer_ids(self) -> tuple[int, ...]:
        """Return sorted explicit layer IDs when deposits provide them."""

        values = sorted({deposit.metadata.layer_id for deposit in self.deposits if deposit.metadata.layer_id is not None})
        return tuple(int(value) for value in values)

    def strata(
        self,
        *,
        mode: Literal["auto", "layer", "order"] = "auto",
        threshold: float | None = None,
    ) -> StratumFieldSet:
        """Return max-density and occupancy fields partitioned by layer or deposit order."""

        threshold_value = self.default_threshold if threshold is None else float(threshold)
        key = (str(mode), threshold_value)
        cached = self._strata_cache.get(key)
        if cached is None:
            cached = build_strata(self, mode=mode, threshold=threshold_value)
            self._strata_cache[key] = cached
        return cached

    def layer_density(self, layer_id: int, *, threshold: float | None = None) -> npt.NDArray[np.float64]:
        """Return the max-density field for one explicit layer."""

        field_set = self.strata(mode="layer", threshold=threshold)
        return field_set.density(layer_id)

    def layer_occupancy(self, layer_id: int, *, threshold: float | None = None) -> npt.NDArray[np.bool_]:
        """Return the occupancy field for one explicit layer."""

        field_set = self.strata(mode="layer", threshold=threshold)
        return field_set.occupancy(layer_id)

    def interface(
        self,
        *,
        mode: Literal["auto", "layer", "order"] = "auto",
        threshold: float | None = None,
    ) -> InterfaceAnalysis:
        """Return aggregate contact and overlap metrics across consecutive strata."""

        threshold_value = self.default_threshold if threshold is None else float(threshold)
        key = (str(mode), threshold_value)
        cached = self._interface_cache.get(key)
        if cached is None:
            cached = build_interface(self, mode=mode, threshold=threshold_value)
            self._interface_cache[key] = cached
        return cached

    def support(
        self,
        *,
        build_direction: tuple[float, float, float] | npt.ArrayLike = (0.0, 0.0, 1.0),
        critical_angle_deg: float = 45.0,
        threshold: float | None = None,
    ) -> SupportAnalysis:
        """Return mesh-first support and overhang metrics for the max-based geometry."""

        threshold_value = self.default_threshold if threshold is None else float(threshold)
        build_dir = tuple(float(value) for value in np.asarray(build_direction, dtype=float).reshape(3))
        key = (build_dir, float(critical_angle_deg), threshold_value)
        cached = self._support_cache.get(key)
        if cached is None:
            cached = build_support(
                self,
                build_direction=build_dir,
                critical_angle_deg=critical_angle_deg,
                threshold=threshold_value,
            )
            self._support_cache[key] = cached
        return cached

    def density(self, composition: DensityComposition = "max") -> npt.NDArray[np.float64]:
        """Return the requested density composition field."""

        if composition == "max":
            return self.density_max
        if composition == "sum":
            if self.density_sum is None:
                raise ValueError("density_sum is not available on this SimulationResult.")
            return self.density_sum
        raise ValueError("composition must be 'max' or 'sum'.")

    def occupancy(self, *, threshold: float | None = None, normalize: bool = False) -> npt.NDArray[np.bool_]:
        """Return a max-based occupancy field."""

        return self.analysis_bundle().occupancy_field(
            threshold=self.default_threshold if threshold is None else threshold,
            normalize=normalize,
        )

    def surface_mesh(
        self,
        *,
        threshold: float | None = None,
        normalize: bool = False,
        step_size: int = 1,
    ) -> object:
        """Return a max-based surface mesh."""

        return self.analysis_bundle().surface_mesh(
            threshold=self.default_threshold if threshold is None else threshold,
            normalize=normalize,
            step_size=step_size,
        )

    def save(self, directory: str | Path, *, metadata: dict[str, object] | None = None) -> dict[str, Path]:
        """Write a standard simulation bundle and any extra density compositions."""

        threshold = self.default_threshold
        written = save_simulation_bundle(
            directory,
            domain=self.domain,
            occupancy=self.occupancy(threshold=threshold),
            deposition_index=self.analysis_bundle().deposition_index_field(),
            density=self.density_max,
            metadata=metadata,
        )
        if self.density_sum is not None:
            written["density_sum"] = save_array(Path(directory) / "density_sum.npy", self.density_sum)
        return written


def simulation_result(
    source: SimulationResult | AnalysisBundle | object,
    *,
    threshold: float = 0.5,
) -> SimulationResult:
    """Resolve a supported source into a SimulationResult."""

    if isinstance(source, SimulationResult):
        return source
    if isinstance(source, AnalysisBundle):
        result = SimulationResult(
            domain=source.domain,
            deposits=(),
            density_max=source.density_field(normalize=False),
            density_sum=None,
            default_threshold=threshold,
        )
        result._analysis_bundle_cache = source
        return result
    if hasattr(source, "result") and callable(source.result):
        if "compositions" in inspect.signature(source.result).parameters:
            return source.result(threshold=threshold, compositions=("max", "sum"))
        return source.result(threshold=threshold)
    if hasattr(source, "analysis_bundle") and callable(source.analysis_bundle):
        bundle = source.analysis_bundle()
        return simulation_result(bundle, threshold=threshold)
    raise TypeError("Expected a SimulationResult, AnalysisBundle, or an object exposing result()/analysis_bundle().")


def simulate(
    domain: Domain,
    deposits: Iterable[DepositInput] | DepositInput,
    *,
    compositions: Sequence[DensityComposition] = ("max",),
    threshold: float = 0.5,
) -> SimulationResult:
    """Run a high-level simulation and return a reusable SimulationResult."""

    requested = tuple(dict.fromkeys(compositions))
    if not requested:
        raise ValueError("compositions must contain at least one density composition.")
    if any(v not in {"max", "sum"} for v in requested):
        raise ValueError("compositions must contain only 'max' and/or 'sum'.")
    deposit_tuple = tuple(iter_deposits(deposits))
    needed = ("max",) if "sum" not in requested else ("max", "sum")
    fields = accumulate_density_fields(domain, deposit_tuple, compositions=needed)
    return SimulationResult(
        domain=domain,
        deposits=deposit_tuple,
        density_max=fields["max"],
        density_sum=fields.get("sum"),
        default_threshold=threshold,
    )
