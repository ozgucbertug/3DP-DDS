# Architecture

3DP-DDS separates the deposition event model from field accumulation,
analysis, serialization, and optional visualization.

## Data flow

1. `Pose3D`, `PointDeposit`, `LineDeposit`, and `PolylineDeposit` describe
   ordered fabrication events.
2. `Domain` maps world coordinates to an aligned voxel grid and records units.
3. Kernel iterators sample compact bead support in bounded tiles.
4. Dense helpers, `Simulator`, or `ChunkedField` accumulate the geometric
   envelope and optional coverage diagnostic.
5. `SimulationResult` provides immutable field snapshots and creates cached
   analysis products.
6. Checkpoints preserve the domain, events, process state, and computed fields.
7. `dds.viz` lazily loads the optional interactive workbench.

## Module boundaries

- `attributes.py`: bead geometry, units, process state, and provenance.
- `primitives.py`: geometric points, poses, and deposition events.
- `domain.py`: aligned grid geometry and coordinate transforms.
- `kernels.py`: tiled field sampling for each event type.
- `fields.py` and `chunked.py`: storage-specific accumulation.
- `simulator.py`: mutable orchestration and incremental cache updates.
- `results.py`: isolated result snapshots and high-level queries.
- `analysis/`: derived fields, interfaces, support, and point queries.
- `geometry/`: analytic SDFs, mesh conversion, and mesh metrics.
- `formats/`: optional external format adapters.
- `io.py`: array bundles and typed checkpoint round trips.

## Extension rules

New deposition primitives should define conservative support bounds and tiled
kernel iteration before being exposed through `Deposit`. New storage backends
should consume `SampledKernel` tiles so numerical behavior remains consistent.
Analysis should depend on `SimulationResult` or `AnalysisBundle`, not mutable
simulator internals.
