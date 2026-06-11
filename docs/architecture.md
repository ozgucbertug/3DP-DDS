# Architecture

3DP-DDS separates deposition events, numerical accumulation, immutable
results, derived analysis, persistence, and optional integrations.

## Data flow

1. `Pose3D` represents a complete rigid transform using SciPy `Rotation`.
2. `DepositionTarget` reduces a pose to the top position and normal consumed
   by the rotationally symmetric bead kernels.
3. Point, line, and polyline deposits normalize triplets, points, poses, or
   targets into concrete `DepositionTarget` fields and combine them with an explicit
   `BeadProfile` and immutable `DepositionMetadata`.
4. `Domain` maps world coordinates to an aligned voxel grid and records a
   `length_unit`.
5. Private kernel iterators sample bounded, globally aligned tiles.
6. Dense accumulation or standalone `ChunkedField` storage composes those
   tiles into a max envelope and optional coverage diagnostic.
7. `SimulationResult` freezes the deposits and computed arrays.
8. `SimulationAnalysis`, reached through `result.analysis`, caches derived
   occupancy, index, SDF, mesh, strata, interface, and support products.

## Module boundaries

- `attributes.py`: bead profile and immutable metadata.
- `primitives.py`: points, vectors, rigid poses, deposition targets, geometric
  wrappers, and deposition events.
- `domain.py`: aligned grid geometry and coordinate transforms.
- `kernels.py`: private tiled sampling implementation.
- `fields.py`: dense, chunked, and low-level in-place accumulation.
- `chunked.py`: standalone sparse chunk storage.
- `simulator.py`: mutable deposit collection and dense incremental caches.
- `results.py`: immutable result snapshots and `simulate`.
- `analysis/`: typed derived queries and result models.
- `geometry/`: supported analytic SDF, CAD, mesh, and metric APIs.
- `targets.py` and `formats/`: external path/format adapters.
- `io.py`: array bundles and typed checkpoint round trips.
- `viz.py` and `workbench.py`: optional visualization entry points.

The root `dds` namespace contains only core deposition and simulation types.
Specialized capabilities remain in their owning namespace so importing the
core library does not load visualization, mesh, format, or CLI dependencies.

## Ownership rules

`Simulator` owns mutable construction. `SimulationResult` owns raw immutable
fields and persistence entry points. `SimulationAnalysis` owns derived
queries. Storage backends consume the same sampled kernel tiles so dense and
chunked numerical behavior remains consistent.

New deposition primitives must provide conservative world-space support bounds
and a tiled kernel iterator. New analysis must consume an immutable result
snapshot rather than mutable simulator internals.

## Checkpoint schema

The current schema stores `length_unit` and omits inactive process/material
state. Schema compatibility is strict: unsupported versions raise an error and
there is no pre-release migration layer.
