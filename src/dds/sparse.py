"""Lazy, contribution-list density accumulation for memory-constrained domains."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .domain import Domain
from .kernels import SampledKernel
from .types import DensityComposition

if TYPE_CHECKING:
    pass


@dataclass(slots=True)
class SparseDensityField:
    """Memory-efficient density accumulation stored as a list of kernel contributions.

    Instead of allocating a full dense grid up-front, each deposit kernel is
    stored as a compact sub-array together with its target slice window.  The
    dense grid is materialised on demand via :meth:`to_dense` or
    :meth:`to_dense_all`.

    For a 1000³ float64 domain (~8 GB dense), a toolpath that touches 5 % of
    voxels stores only ~400 MB in contributions—a 20x memory saving before any
    analysis is needed.

    Parameters
    ----------
    domain:
        Simulation domain that defines the target grid shape.
    """

    domain: Domain
    _contributions: list[tuple[tuple[slice, slice, slice], npt.NDArray[np.float64]]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def add_contribution(self, sampled: SampledKernel) -> None:
        """Record one kernel contribution without touching a dense grid.

        The contribution array is copied so that the caller may safely
        discard or reuse *sampled*.
        """

        self._contributions.append((sampled.slices, sampled.values.copy()))

    def to_dense(self, composition: DensityComposition = "max") -> npt.NDArray[np.float64]:
        """Materialise all contributions into a dense grid.

        Parameters
        ----------
        composition:
            ``"max"`` keeps the element-wise maximum across contributions
            (suitable for geometric occupancy queries).  ``"sum"`` accumulates
            additively (suitable for total-material queries).

        Returns
        -------
        npt.NDArray[np.float64]
            A new float64 array of shape ``domain.grid_shape``.
        """

        grid = np.zeros(self.domain.grid_shape, dtype=float)
        if composition == "sum":
            for slices, values in self._contributions:
                grid[slices] += values
        else:
            for slices, values in self._contributions:
                np.maximum(grid[slices], values, out=grid[slices])
        return grid

    def to_dense_all(
        self,
        *compositions: DensityComposition,
    ) -> dict[DensityComposition, npt.NDArray[np.float64]]:
        """Materialise several compositions in a single pass over contributions.

        More efficient than calling :meth:`to_dense` multiple times when both
        ``"max"`` and ``"sum"`` are needed, because each contribution array is
        read only once.

        Parameters
        ----------
        *compositions:
            One or more :class:`~dds.types.DensityComposition` strings.
            Duplicates are silently deduplicated.

        Returns
        -------
        dict[DensityComposition, npt.NDArray[np.float64]]
            A dense grid for each requested composition.

        Raises
        ------
        ValueError
            When no compositions are supplied.
        """

        requested: tuple[DensityComposition, ...] = tuple(dict.fromkeys(compositions))
        if not requested:
            raise ValueError("At least one composition must be requested.")
        grids: dict[DensityComposition, npt.NDArray[np.float64]] = {
            c: np.zeros(self.domain.grid_shape, dtype=float) for c in requested
        }
        for slices, values in self._contributions:
            if "sum" in grids:
                grids["sum"][slices] += values
            if "max" in grids:
                np.maximum(grids["max"][slices], values, out=grids["max"][slices])
        return grids

    def clear(self) -> None:
        """Remove all stored contributions, resetting to an empty field."""

        self._contributions.clear()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def nbytes(self) -> int:
        """Total bytes occupied by stored contribution arrays."""

        return sum(v.nbytes for _, v in self._contributions)

    @property
    def dense_nbytes(self) -> int:
        """Bytes the equivalent dense float64 grid would require."""

        return math.prod(self.domain.grid_shape) * 8

    @property
    def sparsity(self) -> float:
        """Fraction of dense memory currently used (0.0 = empty, 1.0 = fully dense).

        Values well below 1.0 indicate meaningful memory savings over a dense
        grid.  Note that overlapping contributions are counted separately, so
        the value can exceed 1.0 for heavily overlapping toolpaths.
        """

        dense = self.dense_nbytes
        return self.nbytes / dense if dense > 0 else 0.0

    @property
    def contribution_count(self) -> int:
        """Number of deposit kernels currently stored."""

        return len(self._contributions)
