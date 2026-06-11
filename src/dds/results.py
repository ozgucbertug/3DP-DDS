"""Result containers and high-level simulation entry points."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .analysis import SimulationAnalysis
from .domain import Domain
from .fields import accumulate_fields
from .io import save_array, save_simulation_bundle
from .primitives import Deposit, DepositInput, iter_deposits
from .utils import ensure_finite_scalar, readonly_array


@dataclass(slots=True, frozen=True)
class SimulationResult:
    """Immutable implicit geometry and optional coverage diagnostic."""

    domain: Domain
    deposits: tuple[Deposit, ...]
    implicit_field: npt.NDArray[np.float64]
    coverage: npt.NDArray[np.float64] | None = None
    default_threshold: float = 0.5
    _analysis_cache: SimulationAnalysis | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "deposits", tuple(self.deposits))
        object.__setattr__(
            self,
            "implicit_field",
            readonly_array(self.implicit_field, dtype=float),
        )
        if self.implicit_field.shape != self.domain.grid_shape:
            raise ValueError(
                "implicit_field shape "
                f"{self.implicit_field.shape} does not match domain grid shape "
                f"{self.domain.grid_shape}."
            )
        if self.coverage is not None:
            object.__setattr__(
                self,
                "coverage",
                readonly_array(self.coverage, dtype=float),
            )
            if self.coverage.shape != self.domain.grid_shape:
                raise ValueError(
                    f"coverage shape {self.coverage.shape} does not match domain "
                    f"grid shape {self.domain.grid_shape}."
                )
        if not np.all(np.isfinite(self.implicit_field)) or np.any(
            self.implicit_field < 0.0
        ):
            raise ValueError(
                "implicit_field must contain only finite, non-negative values."
            )
        if self.coverage is not None and (
            not np.all(np.isfinite(self.coverage)) or np.any(self.coverage < 0.0)
        ):
            raise ValueError("coverage must contain only finite, non-negative values.")
        object.__setattr__(
            self,
            "default_threshold",
            ensure_finite_scalar(self.default_threshold, "default_threshold"),
        )
        if not 0.0 <= self.default_threshold <= 1.0:
            raise ValueError("default_threshold must be between 0 and 1.")

    @property
    def analysis(self) -> SimulationAnalysis:
        """Return cached derived queries for this result snapshot."""

        if self._analysis_cache is None:
            object.__setattr__(
                self,
                "_analysis_cache",
                SimulationAnalysis(
                    self.domain,
                    self.implicit_field,
                    self.deposits,
                    self.default_threshold,
                    _copy_implicit_field=False,
                ),
            )
        assert self._analysis_cache is not None
        return self._analysis_cache

    def save(
        self,
        directory: str | Path,
        *,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, Path]:
        """Write occupancy, deposition index, implicit field, and metadata."""

        threshold = self.default_threshold
        written = save_simulation_bundle(
            directory,
            domain=self.domain,
            occupancy=self.analysis.occupancy(threshold=threshold),
            deposition_index=self.analysis.deposition_index_field(),
            implicit_field=self.implicit_field,
            metadata=metadata,
        )
        if self.coverage is not None:
            written["coverage"] = save_array(
                Path(directory) / "coverage.npy",
                self.coverage,
            )
        return written

    def checkpoint(self, path: str | Path) -> Path:
        """Save this result as a typed round-trip checkpoint."""

        from .io import save_checkpoint

        return save_checkpoint(path, self)

    @classmethod
    def load(cls, path: str | Path) -> SimulationResult:
        """Restore a result from a typed checkpoint."""

        from .io import load_checkpoint

        return load_checkpoint(path)


def simulate(
    domain: Domain,
    deposits: Iterable[DepositInput] | DepositInput,
    *,
    include_coverage: bool = False,
    threshold: float = 0.5,
) -> SimulationResult:
    """Run a simulation and return an immutable result."""

    deposit_tuple = tuple(iter_deposits(deposits))
    fields = accumulate_fields(
        domain,
        deposit_tuple,
        include_coverage=include_coverage,
    )
    return SimulationResult(
        domain=domain,
        deposits=deposit_tuple,
        implicit_field=fields["implicit"],
        coverage=fields.get("coverage"),
        default_threshold=threshold,
    )
