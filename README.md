# 3DP-DDS
`3DP-DDS` is a lightweight Python library for sampling deposited material on a dense 3D grid from point and line-based deposition events. The current scope is the core dense simulator: domain setup, deposition primitives, kernel sampling, occupancy extraction, and a small stateful `Simulator` API.

## Current Scope

- Point and line deposition primitives with metadata-rich attributes
- Dense 3D simulation domains with world/index coordinate transforms
- Smooth compact deposition kernels for point and line deposits
- Dense scalar accumulation, occupancy extraction, and deposition index sampling
- A small stateful `Simulator` API for repeated dense-field queries
- Optional SDF-based geometry primitives, booleans, and transforms
- Tyro-backed typed CLI handling for repo scripts and examples

## Installation

```bash
python -m pip install -e .
```

For tests and development extras:

```bash
python -m pip install -e ".[dev]"
```

Repository CLI interfaces use `tyro` as the common parser and configuration layer. The included example scripts therefore expose typed `--help` output through dataclass-based configs rather than handwritten `argparse` parsers.

## Minimal Example

```python
from dds import (
    DepositionAttributes,
    Domain,
    LineDeposit,
    PointDeposit,
    simulate_deposition_index,
    simulate_occupancy,
)

domain = Domain.from_bounds(
    xmin=0.0,
    xmax=100.0,
    ymin=0.0,
    ymax=100.0,
    zmin=0.0,
    zmax=20.0,
    voxel_size=0.5,
)

attrs = DepositionAttributes(width=1.2, height=0.4, layer_id=0)
deposits = [
    PointDeposit(x=10.25, y=10.25, z=0.25, attributes=attrs),
    LineDeposit(start=(10.25, 10.25, 0.25), end=(50.25, 10.25, 0.25), attributes=attrs),
]

occupancy = simulate_occupancy(domain, deposits, threshold=0.5)
deposition_index = simulate_deposition_index(domain, deposits)
```

## Geometry and SDFs

`dds.geometry` adds a small analytic SDF layer on top of the dense simulator. This stage includes the core SDF abstraction, boolean composition, spatial transforms, and a stable primitive subset.

Key conventions:

- SDF sign convention is `negative inside`, `positive outside`, `zero on the surface`
- Boolean operations are continuous SDF booleans, not exact mesh booleans
- `GridSDF3` uses SciPy-backed trilinear interpolation for sampled fields

```python
from dds import Domain
from dds.geometry import box, cylinder, sphere, union

domain = Domain.from_bounds(
    xmin=-8.0,
    xmax=8.0,
    ymin=-8.0,
    ymax=8.0,
    zmin=-8.0,
    zmax=8.0,
    voxel_size=0.25,
)

shape = union(
    sphere(radius=3.0),
    box(size=(4.0, 4.0, 6.0)),
)
cut = cylinder(radius=0.8, height=8.0)
combined = shape - cut
sampled = combined.sample(domain)
```

Base geometry API:

- `SDF3`, `GridSDF3`, `as_sdf3`
- `sphere`, `box`, `cylinder`, `capsule`, `plane`, `slab`, `ellipsoid`, `torus`
- `union`, `intersection`, `difference`, `dilate`, `erode`, `shell`
- `translate`, `scale`, `rotate`, `orient`, `rotation_matrix`

## Design Assumptions

- The import package is `dds`; the repository/distribution branding remains `3DP-DDS` / `3dp-dds`.
- Dense array indexing follows `(x, y, z)` ordering via NumPy `indexing="ij"`.
- Geometry SDF sign convention is `negative inside`, `positive outside`, `zero on the surface`.
- `width` is the full bead width in the XY plane, `height` is the full bead height in Z.
- Point deposits use an ellipsoidal compact kernel.
- Line deposits use a capsule-like closest-distance model with Z scaling to support anisotropic bead height.
- The v0 deposition index is the weighted sum of deposit contributions per voxel.
- Occupancy is derived by thresholding the accumulated scalar field. The default threshold `0.5` is a practical confidence-style threshold, not a strict geometric boundary.
- Deposits fully outside the domain are skipped; partial overlaps are clipped to the intersecting sampling window.

## Package Layout

```text
src/dds/
  __init__.py
  cli.py
  attributes.py
  primitives.py
  domain.py
  kernels.py
  fields.py
  simulator.py
  occupancy.py
  analysis.py
  utils.py
  geometry/
    __init__.py
    sdf.py
    ops.py
    primitives.py
    transforms.py
```

## Example Script

Run the included example from the repository root:

```bash
python examples/basic_simulation.py
```

Inspect the typed CLI:

```bash
python examples/basic_simulation.py --help
```

Adjust the occupancy threshold used by the example:

```bash
python examples/basic_simulation.py --threshold 0.35
```

The example creates a simple deposition scene, prints summary metrics, and exercises the dense simulator without geometry or visualization extras.
