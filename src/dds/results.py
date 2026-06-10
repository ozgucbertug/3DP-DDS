"""Result containers and high-level simulation entry points."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import numpy.typing as npt

from .analysis import SimulationAnalysis
from .domain import Domain
from .fields import accumulate_fields
from .io import save_array, save_simulation_bundle
from .primitives import Deposit, DepositInput, iter_deposits
from .types import FieldComposition
from .utils import ensure_finite_scalar, readonly_array

ViewMode = Literal["surface", "occupancy", "density"]
ViewColorMode = Literal["plain", "normals", "overhang"]
ViewScalarField = Literal["occupancy", "density", "coverage", "deposition_order"]


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


@dataclass(slots=True, frozen=True)
class SimulationResult:
    """Reusable simulation outputs and derived geometry/query helpers."""

    domain: Domain
    deposits: tuple[Deposit, ...]
    density_max: npt.NDArray[np.float64]
    coverage: npt.NDArray[np.float64] | None = None
    default_threshold: float = 0.5
    _analysis_cache: SimulationAnalysis | None = field(default=None, init=False, repr=False)
    _deposition_index_cache: npt.NDArray[np.intp] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "deposits", tuple(self.deposits))
        object.__setattr__(self, "density_max", readonly_array(self.density_max, dtype=float))
        if self.density_max.shape != self.domain.grid_shape:
            raise ValueError(
                f"density_max shape {self.density_max.shape} does not match domain grid shape {self.domain.grid_shape}."
            )
        if self.coverage is not None:
            object.__setattr__(self, "coverage", readonly_array(self.coverage, dtype=float))
            if self.coverage.shape != self.domain.grid_shape:
                raise ValueError(
                    f"coverage shape {self.coverage.shape} does not match domain grid shape {self.domain.grid_shape}."
                )
        if not np.all(np.isfinite(self.density_max)) or np.any(self.density_max < 0.0):
            raise ValueError("density_max must contain only finite, non-negative values.")
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

    def _deposition_index_field(self) -> npt.NDArray[np.intp]:
        if self._deposition_index_cache is None:
            from .fields import accumulate_deposition_index
            object.__setattr__(
                self,
                "_deposition_index_cache",
                readonly_array(
                    accumulate_deposition_index(self.domain, self.deposits),
                    dtype=np.intp,
                ),
            )
        return self._deposition_index_cache

    @property
    def analysis(self) -> SimulationAnalysis:
        """Return cached derived-field and geometry queries for this snapshot."""

        if self._analysis_cache is None:
            object.__setattr__(
                self,
                "_analysis_cache",
                SimulationAnalysis(
                    self.domain,
                    self.density_max,
                    self._deposition_index_field(),
                    self.deposits,
                    self.default_threshold,
                ),
            )
        return self._analysis_cache

    def field(self, composition: FieldComposition = "max") -> npt.NDArray[np.float64]:
        """Return the geometric envelope or nonphysical coverage field."""

        if composition == "max":
            return self.density_max
        if composition == "coverage":
            if self.coverage is None:
                raise ValueError("coverage is not available on this SimulationResult.")
            return self.coverage
        raise ValueError("composition must be 'max' or 'coverage'.")

    def save(self, directory: str | Path, *, metadata: dict[str, object] | None = None) -> dict[str, Path]:
        """Write a standard simulation bundle and any extra density compositions."""

        threshold = self.default_threshold
        written = save_simulation_bundle(
            directory,
            domain=self.domain,
            occupancy=self.analysis.occupancy(threshold=threshold),
            deposition_index=self.analysis.deposition_index_field(),
            density=self.density_max,
            metadata=metadata,
        )
        if self.coverage is not None:
            written["coverage"] = save_array(Path(directory) / "coverage.npy", self.coverage)
        return written

    def checkpoint(self, path: str | Path) -> Path:
        """Save this result as a typed round-trip checkpoint.

        Stores density arrays and the full deposit list in a single compressed
        ``npz`` file.  The result can be restored with :meth:`load`.

        Parameters
        ----------
        path:
            Destination file path.  A ``.npz`` extension is appended if absent.

        Returns
        -------
        Path
            Absolute path of the written checkpoint file.
        """

        from .io import save_checkpoint

        return save_checkpoint(path, self)

    @classmethod
    def load(cls, path: str | Path) -> "SimulationResult":
        """Restore a result from a checkpoint written by :meth:`checkpoint`.

        Parameters
        ----------
        path:
            Path to the ``.npz`` checkpoint file.

        Returns
        -------
        SimulationResult
            A fully reconstructed result with density arrays and deposits.
        """

        from .io import load_checkpoint

        return load_checkpoint(path)

def simulate(
    domain: Domain,
    deposits: Iterable[DepositInput] | DepositInput,
    *,
    compositions: Sequence[FieldComposition] = ("max",),
    threshold: float = 0.5,
) -> SimulationResult:
    """Run a high-level simulation and return a reusable SimulationResult."""

    requested = tuple(dict.fromkeys(compositions))
    if not requested:
        raise ValueError("compositions must contain at least one density composition.")
    if any(v not in {"max", "coverage"} for v in requested):
        raise ValueError("compositions must contain only 'max' and/or 'coverage'.")
    deposit_tuple = tuple(iter_deposits(deposits))
    needed = ("max",) if "coverage" not in requested else ("max", "coverage")
    fields = accumulate_fields(domain, deposit_tuple, compositions=needed)
    return SimulationResult(
        domain=domain,
        deposits=deposit_tuple,
        density_max=fields["max"],
        coverage=fields.get("coverage"),
        default_threshold=threshold,
    )
