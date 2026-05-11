# 3DP-DDS
`3DP-DDS` is a Python library for dense deposition simulation on a 3D voxel grid. The import package is `dds`. The current library includes the dense simulator, analytic SDF geometry, and mesh extraction and conversion helpers in the same install.

## Current Scope

- Point and line deposition primitives with metadata-rich attributes
- Dense 3D simulation domains with world/index coordinate transforms
- Smooth compact deposition kernels for point and line deposits
- Dense scalar accumulation, occupancy extraction, and deposition index sampling
- A small stateful `Simulator` API for repeated dense-field queries
- Analytic SDF primitives, booleans, and spatial transforms in `dds.geometry`
- Mesh extraction, mesh IO, and dense-field or mesh-to-SDF conversions
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

`dds.geometry` adds an analytic SDF layer on top of the dense simulator.

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

Geometry API:

- `SDF3`, `GridSDF3`, `MeshSDF3`, `as_sdf3`
- `sphere`, `box`, `cylinder`, `capsule`, `plane`, `slab`, `ellipsoid`, `torus`
- `rounded_box`, `capped_cylinder`, `rounded_cylinder`, `capped_cone`, `cone`, `rounded_cone`, `capsule_chain`
- `union`, `intersection`, `difference`, `dilate`, `erode`, `shell`
- `translate`, `scale`, `rotate`, `orient`, `rotation_matrix`

## Mesh Extraction and Conversion

The mesh layer converts dense fields and SDFs into triangle meshes, reads and writes triangle mesh files, and samples watertight meshes back into dense fields.

```python
from dds import Domain
from dds.geometry import (
    density_to_mesh,
    density_to_sdf,
    mesh_to_sdf_field,
    occupancy_to_mesh,
    read_mesh,
    sphere,
    write_mesh,
)

domain = Domain.from_bounds(
    xmin=-6.0,
    xmax=6.0,
    ymin=-6.0,
    ymax=6.0,
    zmin=-6.0,
    zmax=6.0,
    voxel_size=0.25,
)

density = (-sphere(radius=2.0).sample(domain)).clip(min=0.0)
surface = density_to_mesh(domain, density, threshold=0.5)
write_mesh("outputs/sphere.ply", surface)

reloaded = read_mesh("outputs/sphere.ply")
sampled_sdf = mesh_to_sdf_field(domain, reloaded)
wrapped = density_to_sdf(domain, density, threshold=0.5)
occupancy_surface = occupancy_to_mesh(domain, wrapped.sample(domain) <= 0.0)
```

Mesh API:

- `TriangleMesh`, `read_mesh`, `write_mesh`
- `extract_mesh_from_field`, `occupancy_to_mesh`, `density_to_mesh`, `sdf_to_mesh`
- `mesh_to_occupancy`, `mesh_to_sdf_field`
- `occupancy_to_sdf_field`, `density_to_sdf_field`, `occupancy_to_sdf`, `density_to_sdf`

Mesh conversions assume triangle meshes. Signed-distance and containment queries are intended for watertight meshes.

## Design Assumptions

- The import package is `dds`; the repository and distribution branding are `3DP-DDS` and `3dp-dds`.
- Dense array indexing follows `(x, y, z)` ordering via NumPy `indexing="ij"`.
- `width` is the full bead width in the XY plane, and `height` is the full bead height in Z.
- Point deposits use an ellipsoidal compact kernel.
- Line deposits use a capsule-like closest-distance model with Z scaling to support anisotropic bead height.
- The v0 deposition index is the weighted sum of deposit contributions per voxel.
- Occupancy is derived by thresholding the accumulated scalar field.

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
    mesh.py
    adapters.py
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

The example creates a simple deposition scene, prints summary metrics, and exercises the dense simulator without visualization features.
