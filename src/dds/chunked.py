"""Chunked deposition-field accumulation for large sparse workspaces."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .domain import Domain, IndexBounds
from .kernels import SampledKernel, TileShape, validate_tile_shape
from .types import FieldComposition

ChunkIndex = tuple[int, int, int]


@dataclass(slots=True)
class _Chunk:
    maximum: npt.NDArray[np.float64]
    coverage: npt.NDArray[np.float64]


@dataclass(slots=True)
class ChunkedField:
    """Sparse field backed by fixed-size dense chunks.

    Each allocated chunk stores both the geometric max envelope and the
    additive, nonphysical coverage diagnostic. Empty chunks are never stored.
    """

    domain: Domain
    chunk_shape: TileShape = (32, 32, 32)
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
        return tuple((lower[axis], upper[axis]) for axis in range(3))

    def _chunk_array_shape(self, index: ChunkIndex) -> TileShape:
        bounds = self._chunk_bounds(index)
        return tuple(stop - start for start, stop in bounds)

    def _get_or_create_chunk(self, index: ChunkIndex) -> _Chunk:
        chunk = self._chunks.get(index)
        if chunk is None:
            shape = self._chunk_array_shape(index)
            chunk = _Chunk(
                maximum=np.zeros(shape, dtype=float),
                coverage=np.zeros(shape, dtype=float),
            )
            self._chunks[index] = chunk
        return chunk

    def add_kernel(self, sampled: SampledKernel) -> bool:
        """Accumulate one sampled kernel tile and return whether it was nonempty."""

        starts = tuple(int(axis_slice.start) for axis_slice in sampled.slices)
        stops = tuple(int(axis_slice.stop) for axis_slice in sampled.slices)
        expected_shape = tuple(stops[axis] - starts[axis] for axis in range(3))
        if any(stop <= start for start, stop in zip(starts, stops, strict=True)):
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
                        chunk.maximum[chunk_slices],
                        values,
                        out=chunk.maximum[chunk_slices],
                    )
                    chunk.coverage[chunk_slices] += values
                    hit = True
        return hit

    def record_event(self) -> None:
        """Record one deposition event that contributed at least one kernel tile."""

        self._event_count += 1

    def materialize(
        self,
        composition: FieldComposition = "max",
        *,
        index_bounds: IndexBounds | None = None,
    ) -> npt.NDArray[np.float64]:
        """Materialize the full field or an index-space region of interest."""

        if composition not in {"max", "coverage"}:
            raise ValueError("composition must be 'max' or 'coverage'.")
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
            values = chunk.maximum if composition == "max" else chunk.coverage
            output[output_slices] = values[chunk_slices]
        return output

    def _validate_index_bounds(self, index_bounds: IndexBounds | None) -> IndexBounds:
        if index_bounds is None:
            return tuple((0, size) for size in self.domain.grid_shape)
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
        return resolved

    def to_dense(
        self,
        composition: FieldComposition = "max",
    ) -> npt.NDArray[np.float64]:
        """Materialize a full-domain dense field."""

        return self.materialize(composition)

    def to_dense_all(
        self,
        *compositions: FieldComposition,
    ) -> dict[FieldComposition, npt.NDArray[np.float64]]:
        """Materialize multiple field compositions."""

        requested = tuple(dict.fromkeys(compositions))
        if not requested:
            raise ValueError("At least one composition must be requested.")
        return {
            composition: self.materialize(composition)
            for composition in requested
        }

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

        return sum(int(np.count_nonzero(chunk.maximum > 0.0)) for chunk in self._chunks.values())

    @property
    def allocated_voxel_count(self) -> int:
        """Number of voxel slots allocated across all chunks."""

        return sum(int(chunk.maximum.size) for chunk in self._chunks.values())

    @property
    def nbytes(self) -> int:
        """Bytes occupied by both stored field compositions."""

        return sum(
            chunk.maximum.nbytes + chunk.coverage.nbytes
            for chunk in self._chunks.values()
        )

    @property
    def dense_field_nbytes(self) -> int:
        """Bytes required by one equivalent dense float64 field."""

        return math.prod(self.domain.grid_shape) * np.dtype(float).itemsize

    @property
    def dense_nbytes(self) -> int:
        """Bytes required by both equivalent dense float64 fields."""

        return 2 * self.dense_field_nbytes

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
        """Stored bytes divided by equivalent two-field dense bytes."""

        return self.nbytes / self.dense_nbytes if self.dense_nbytes else 0.0
