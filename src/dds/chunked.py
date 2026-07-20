"""Chunked deposition-field accumulation for large sparse workspaces."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
import numpy.typing as npt

from .domain import Domain, IndexBounds
from .kernels import TileShape, _SampledKernel, iter_deposit_kernels, validate_tile_shape
from .primitives import DepositInput, iter_deposits

if TYPE_CHECKING:
    from .results import SimulationResult

ChunkIndex = tuple[int, int, int]
ChunkFieldName = Literal["implicit", "coverage"]


@dataclass
class _Chunk:
    implicit: npt.NDArray[np.float64]
    coverage: Optional[npt.NDArray[np.float64]]


@dataclass
class ChunkedField:
    """Sparse field backed by fixed-size dense chunks.

    Every chunk stores the implicit field. Coverage is allocated only when
    requested. Empty chunks are never stored.

    Parameters
    ----------
    domain
        Simulation domain for the chunked storage.
    chunk_shape
        Dense chunk dimensions in ``(x, y, z)`` index order.
    include_coverage
        Allocate additive coverage chunks in addition to implicit chunks.
    """

    domain: Domain
    chunk_shape: TileShape = (32, 32, 32)
    include_coverage: bool = False
    _chunks: dict[ChunkIndex, _Chunk] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _event_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.chunk_shape = validate_tile_shape(self.chunk_shape)

    def _chunk_bounds(self, index: ChunkIndex) -> IndexBounds:
        lower = tuple(index[axis] * self.chunk_shape[axis] for axis in range(3))
        upper = tuple(
            min(lower[axis] + self.chunk_shape[axis], self.domain.grid_shape[axis])
            for axis in range(3)
        )
        return (
            (lower[0], upper[0]),
            (lower[1], upper[1]),
            (lower[2], upper[2]),
        )

    def _chunk_array_shape(self, index: ChunkIndex) -> TileShape:
        bounds = self._chunk_bounds(index)
        return (
            bounds[0][1] - bounds[0][0],
            bounds[1][1] - bounds[1][0],
            bounds[2][1] - bounds[2][0],
        )

    def _get_or_create_chunk(self, index: ChunkIndex) -> _Chunk:
        chunk = self._chunks.get(index)
        if chunk is None:
            shape = self._chunk_array_shape(index)
            chunk = _Chunk(
                implicit=np.zeros(shape, dtype=float),
                coverage=(
                    np.zeros(shape, dtype=float) if self.include_coverage else None
                ),
            )
            self._chunks[index] = chunk
        return chunk

    def add_kernel(self, sampled: _SampledKernel) -> bool:
        """Accumulate one sampled kernel tile and return whether it was nonempty."""

        starts = tuple(int(axis_slice.start) for axis_slice in sampled.slices)
        stops = tuple(int(axis_slice.stop) for axis_slice in sampled.slices)
        expected_shape = tuple(stops[axis] - starts[axis] for axis in range(3))
        if any(stop <= start for start, stop in zip(starts, stops)):
            raise ValueError("Kernel slices must be nonempty.")
        if sampled.values.shape != expected_shape:
            raise ValueError(
                f"Kernel shape {sampled.values.shape} does not match slices {expected_shape}."
            )
        if any(starts[axis] < 0 or stops[axis] > self.domain.grid_shape[axis] for axis in range(3)):
            raise ValueError("Kernel slices must lie inside the domain.")
        if not np.all(np.isfinite(sampled.values)) or np.any(sampled.values < 0.0):
            raise ValueError("Kernel values must be finite and non-negative.")
        if not np.any(sampled.values > 0.0):
            return False

        first_chunk = tuple(starts[axis] // self.chunk_shape[axis] for axis in range(3))
        last_chunk = tuple((stops[axis] - 1) // self.chunk_shape[axis] for axis in range(3))
        hit = False
        for ix in range(first_chunk[0], last_chunk[0] + 1):
            for iy in range(first_chunk[1], last_chunk[1] + 1):
                for iz in range(first_chunk[2], last_chunk[2] + 1):
                    chunk_index = (ix, iy, iz)
                    chunk_bounds = self._chunk_bounds(chunk_index)
                    intersection = tuple(
                        (
                            max(starts[axis], chunk_bounds[axis][0]),
                            min(stops[axis], chunk_bounds[axis][1]),
                        )
                        for axis in range(3)
                    )
                    if any(stop <= start for start, stop in intersection):
                        continue

                    sampled_slices = tuple(
                        slice(
                            intersection[axis][0] - starts[axis],
                            intersection[axis][1] - starts[axis],
                        )
                        for axis in range(3)
                    )
                    values = sampled.values[sampled_slices]
                    if not np.any(values > 0.0):
                        continue

                    chunk = self._get_or_create_chunk(chunk_index)
                    chunk_slices = tuple(
                        slice(
                            intersection[axis][0] - chunk_bounds[axis][0],
                            intersection[axis][1] - chunk_bounds[axis][0],
                        )
                        for axis in range(3)
                    )
                    np.maximum(
                        chunk.implicit[chunk_slices],
                        values,
                        out=chunk.implicit[chunk_slices],
                    )
                    if chunk.coverage is not None:
                        chunk.coverage[chunk_slices] += values
                    hit = True
        return hit

    def record_event(self) -> None:
        """Record one deposition event that contributed at least one kernel tile."""

        self._event_count += 1

    def materialize(
        self,
        field: ChunkFieldName = "implicit",
        *,
        index_bounds: Optional[IndexBounds] = None,
    ) -> npt.NDArray[np.float64]:
        """Materialize the full field or an index-space region of interest.

        Parameters
        ----------
        field
            ``"implicit"`` or ``"coverage"``.
        index_bounds
            Optional half-open bounds in domain index coordinates.
        """

        if field not in {"implicit", "coverage"}:
            raise ValueError("field must be 'implicit' or 'coverage'.")
        if field == "coverage" and not self.include_coverage:
            raise ValueError("coverage was not requested for this ChunkedField.")
        bounds = self._validate_index_bounds(index_bounds)
        output_shape = tuple(stop - start for start, stop in bounds)
        output = np.zeros(output_shape, dtype=float)

        for chunk_index, chunk in self._chunks.items():
            chunk_bounds = self._chunk_bounds(chunk_index)
            intersection = tuple(
                (
                    max(bounds[axis][0], chunk_bounds[axis][0]),
                    min(bounds[axis][1], chunk_bounds[axis][1]),
                )
                for axis in range(3)
            )
            if any(stop <= start for start, stop in intersection):
                continue

            output_slices = tuple(
                slice(
                    intersection[axis][0] - bounds[axis][0],
                    intersection[axis][1] - bounds[axis][0],
                )
                for axis in range(3)
            )
            chunk_slices = tuple(
                slice(
                    intersection[axis][0] - chunk_bounds[axis][0],
                    intersection[axis][1] - chunk_bounds[axis][0],
                )
                for axis in range(3)
            )
            values = chunk.implicit if field == "implicit" else chunk.coverage
            assert values is not None
            output[output_slices] = values[chunk_slices]
        return output

    def _validate_index_bounds(self, index_bounds: Optional[IndexBounds]) -> IndexBounds:
        if index_bounds is None:
            return (
                (0, self.domain.grid_shape[0]),
                (0, self.domain.grid_shape[1]),
                (0, self.domain.grid_shape[2]),
            )
        if len(index_bounds) != 3:
            raise ValueError("index_bounds must contain bounds for three axes.")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for bounds in index_bounds
            for value in bounds
        ):
            raise TypeError("index_bounds values must be integers.")
        resolved = tuple((int(start), int(stop)) for start, stop in index_bounds)
        for axis, (start, stop) in enumerate(resolved):
            if not 0 <= start < stop <= self.domain.grid_shape[axis]:
                raise ValueError(
                    "index_bounds must satisfy 0 <= start < stop <= axis size."
                )
        return (resolved[0], resolved[1], resolved[2])

    def to_dense(
        self,
        field: ChunkFieldName = "implicit",
    ) -> npt.NDArray[np.float64]:
        """Materialize a full-domain dense field."""

        return self.materialize(field)

    def to_dense_all(self) -> dict[ChunkFieldName, npt.NDArray[np.float64]]:
        """Materialize all fields configured on this storage object."""

        fields: dict[ChunkFieldName, npt.NDArray[np.float64]] = {
            "implicit": self.materialize("implicit")
        }
        if self.include_coverage:
            fields["coverage"] = self.materialize("coverage")
        return fields

    def clear(self) -> None:
        """Remove all allocated chunks and event diagnostics."""

        self._chunks.clear()
        self._event_count = 0

    @property
    def chunk_count(self) -> int:
        """Number of allocated chunks."""

        return len(self._chunks)

    @property
    def event_count(self) -> int:
        """Number of in-domain deposition events accumulated."""

        return self._event_count

    @property
    def active_voxel_count(self) -> int:
        """Number of voxels with a positive geometric envelope."""

        return sum(
            int(np.count_nonzero(chunk.implicit > 0.0))
            for chunk in self._chunks.values()
        )

    @property
    def allocated_voxel_count(self) -> int:
        """Number of voxel slots allocated across all chunks."""

        return sum(int(chunk.implicit.size) for chunk in self._chunks.values())

    @property
    def nbytes(self) -> int:
        """Bytes occupied by the implicit field and optional coverage."""

        return sum(
            sum(array.nbytes for array in (chunk.implicit, chunk.coverage) if array is not None)
            for chunk in self._chunks.values()
        )

    @property
    def dense_field_nbytes(self) -> int:
        """Bytes required by one equivalent dense float64 field."""

        return math.prod(self.domain.grid_shape) * np.dtype(float).itemsize

    @property
    def dense_nbytes(self) -> int:
        """Bytes required by equivalent configured dense fields."""

        return (2 if self.include_coverage else 1) * self.dense_field_nbytes

    @property
    def active_fraction(self) -> float:
        """Fraction of domain voxels with positive geometric support."""

        voxel_count = math.prod(self.domain.grid_shape)
        return self.active_voxel_count / voxel_count if voxel_count else 0.0

    @property
    def allocation_fraction(self) -> float:
        """Fraction of domain voxel slots allocated in chunks."""

        voxel_count = math.prod(self.domain.grid_shape)
        return self.allocated_voxel_count / voxel_count if voxel_count else 0.0

    @property
    def memory_ratio(self) -> float:
        """Stored bytes divided by equivalent requested dense fields."""

        return self.nbytes / self.dense_nbytes if self.dense_nbytes else 0.0

    def to_result(
        self,
        deposits: Union[Iterable[DepositInput], DepositInput],
        *,
        threshold: float = 0.5,
    ) -> "SimulationResult":
        """Construct a :class:`~dds.results.SimulationResult` from this chunked field.

        Materializes the dense arrays and wraps them in a ``SimulationResult``
        without re-running the simulation.  Useful when the chunked field was
        built via :func:`accumulate_chunked_field` and the caller wants access
        to the full analysis interface.

        Parameters
        ----------
        deposits:
            The same deposit sequence that was used to build this field.
        threshold:
            Occupancy threshold forwarded to the result's ``default_threshold``.

        Returns
        -------
        SimulationResult
        """

        # Lazy import to avoid a circular dependency (results imports chunked
        # only under TYPE_CHECKING).
        from .results import SimulationResult

        implicit_field = self.materialize("implicit")
        coverage = self.materialize("coverage") if self.include_coverage else None
        deposit_tuple = tuple(iter_deposits(deposits))
        return SimulationResult(
            domain=self.domain,
            deposits=deposit_tuple,
            implicit_field=implicit_field,
            coverage=coverage,
            default_threshold=threshold,
        )


# ---------------------------------------------------------------------------
# Module-level factory — previously lived in fields.py
# ---------------------------------------------------------------------------

def accumulate_chunked_field(
    domain: Domain,
    deposits: Union[Iterable[DepositInput], DepositInput],
    *,
    chunk_shape: Sequence[int] = (32, 32, 32),
    include_coverage: bool = False,
) -> ChunkedField:
    """Build a chunked field without allocating full-domain dense arrays.

    Parameters
    ----------
    domain
        Simulation domain.
    deposits
        One or more deposit primitives or sequences thereof.
    chunk_shape
        Dense chunk shape in ``(x, y, z)`` index order.
    include_coverage
        Whether to store additive coverage in each active chunk.

    Returns
    -------
    ChunkedField
        Sparse chunk-backed accumulation result.
    """

    chunked = ChunkedField(
        domain,
        chunk_shape=validate_tile_shape(chunk_shape),
        include_coverage=include_coverage,
    )
    for deposit in iter_deposits(deposits):
        hit = False
        for sampled in iter_deposit_kernels(
            domain,
            deposit,
            tile_shape=chunked.chunk_shape,
        ):
            hit = chunked.add_kernel(sampled) or hit
        if hit:
            chunked.record_event()
    return chunked
