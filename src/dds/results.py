"""Result containers and high-level simulation entry points."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from .analysis import AnalysisBundle
from .io import save_array, save_simulation_bundle
from .kernels import sample_deposit_kernel
from .primitives import Deposit, DepositInput, iter_deposits
from .domain import Domain

DensityComposition = Literal["max", "sum"]


def _normalize_compositions(compositions: Sequence[DensityComposition]) -> tuple[DensityComposition, ...]:
    values = tuple(dict.fromkeys(compositions))
    if not values:
        raise ValueError("compositions must contain at least one density composition.")
    invalid = [value for value in values if value not in {"max", "sum"}]
    if invalid:
        raise ValueError("compositions must contain only 'max' and/or 'sum'.")
    return values


def _accumulate_density_max(
    domain: Domain,
    deposits: tuple[Deposit, ...],
) -> npt.NDArray[np.float64]:
    field = np.zeros(domain.grid_shape, dtype=float)
    for deposit in deposits:
        sampled = sample_deposit_kernel(domain, deposit)
        if sampled is None:
            continue
        field[sampled.slices] = np.maximum(field[sampled.slices], sampled.values)
    return field


@dataclass(slots=True)
class SimulationResult:
    """Reusable simulation outputs and derived geometry/query helpers."""

    domain: Domain
    deposits: tuple[Deposit, ...]
    density_max: npt.NDArray[np.float64]
    density_sum: npt.NDArray[np.float64] | None = None
    default_threshold: float = 0.5
    _analysis_bundle_cache: AnalysisBundle | None = field(default=None, init=False, repr=False)

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

    def analysis_bundle(self) -> AnalysisBundle:
        """Return the canonical analysis bundle derived from density_max."""

        if self._analysis_bundle_cache is None:
            self._analysis_bundle_cache = AnalysisBundle(self.domain, self.density_max)
        return self._analysis_bundle_cache

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

    def show(
        self,
        *,
        view_mode: Literal["surface", "occupancy", "density"] = "surface",
        off_screen: bool = False,
    ) -> object:
        """Open the interactive workbench for this result."""

        from .workbench import SimulationWorkbench

        workbench = SimulationWorkbench(self, threshold=self.default_threshold, off_screen=off_screen)
        workbench.set_representation(view_mode)
        workbench.show()
        return workbench


def simulate(
    domain: Domain,
    deposits: Iterable[DepositInput] | DepositInput,
    *,
    compositions: Sequence[DensityComposition] = ("max",),
    threshold: float = 0.5,
) -> SimulationResult:
    """Run a high-level simulation and return a reusable SimulationResult."""

    requested = _normalize_compositions(compositions)
    if requested != ("max",):
        raise ValueError("Only the 'max' density composition is supported until the max/sum simulation stage lands.")

    deposit_tuple = tuple(iter_deposits(deposits))
    density_max = _accumulate_density_max(domain, deposit_tuple)
    return SimulationResult(
        domain=domain,
        deposits=deposit_tuple,
        density_max=density_max,
        density_sum=None,
        default_threshold=threshold,
    )
