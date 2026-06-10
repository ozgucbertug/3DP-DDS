# 3DP-DDS

3DP-DDS is a geometry-first deposition simulator for robotic additive
manufacturing. It represents fabrication paths as point, line, and polyline
events and samples their bead geometry into a reproducible voxel field that can
serve as a lightweight digital twin.

The current scope is deposited geometry. Material flow, thermal history,
curing, robot dynamics, and controller behavior are intentionally outside the
model.

## Core workflow

```python
from dds import (
    BeadProfile,
    DepositionMetadata,
    Domain,
    PointDeposit,
    Pose3D,
    simulate,
)

domain = Domain.from_bounds(
    xmin=0.0,
    xmax=10.0,
    ymin=0.0,
    ymax=10.0,
    zmin=0.0,
    zmax=5.0,
    voxel_size=0.5,
    length_unit="mm",
)

profile = BeadProfile(width=1.2, height=0.5)
deposit = PointDeposit(
    target=Pose3D(
        position=(1.0, 2.0, 3.0),
        axis=(0.0, 0.0, 1.0),
    ),
    profile=profile,
    metadata=DepositionMetadata(
        layer_id=0,
        user_data={"material_id": "clay"},
    ),
)

result = simulate(domain, [deposit])
occupancy = result.analysis.occupancy(threshold=0.5)
index = result.analysis.sample_deposition_index((1.0, 2.0, 3.0))
support = result.analysis.support(build_direction="+Z")
```

Every deposit requires an explicit `BeadProfile`. `Domain.length_unit` records
whether world coordinates are expressed in millimeters or meters; it does not
perform unit conversion.

## Deposition primitives

- `Point3D` represents a position; `Vector3D` represents a direction or displacement.
- `Pose3D` combines a target point and normalized axis defining its target plane.
- `Line3D` and `Polyline3D` describe finite path geometry without deposition data.
- `PointDeposit` samples a compact bead at one `Pose3D`.
- `LineDeposit` sweeps a bead between two poses and interpolates its axis.
- `PolylineDeposit` represents one ordered, multi-segment fabrication event.
- `DepositionMetadata` stores an optional `layer_id` and immutable JSON-like
  `user_data` for research provenance.

`Domain.from_deposits(...)` can infer aligned bounds from explicit bead support:

```python
from dds import BeadProfile, Domain, LineDeposit

deposit = LineDeposit(
    start=(0.0, 0.0, 1.0),
    end=(20.0, 0.0, 1.0),
    profile=BeadProfile(width=1.2, height=0.5),
)
domain = Domain.from_deposits(deposit, voxel_size=0.25, padding="auto")
```

## Results and analysis

`SimulationResult` is an immutable snapshot. Its NumPy arrays are copied and
read-only. Derived queries live on the cached `result.analysis` object:

```python
analysis = result.analysis

density = analysis.density_field()
occupancy = analysis.occupancy(threshold=0.5)
deposition_index = analysis.deposition_index_field()
surface = analysis.surface_mesh(threshold=0.5)
sdf = analysis.surface_sdf(threshold=0.5)
layers = analysis.strata(mode="layer")
interfaces = analysis.interface(mode="layer")
support = analysis.support(build_direction="+Z")
```

The `"max"` composition is the canonical fabricated geometry. Optional
`"coverage"` adds kernel contributions and is only an overlap diagnostic:

```python
result = simulate(
    domain,
    [deposit],
    compositions=("max", "coverage"),
)
coverage = result.field("coverage")
```

## Incremental and sparse workflows

Use `Simulator` when deposits arrive incrementally:

```python
from dds import Simulator

simulator = Simulator(domain)
simulator.add_deposit(deposit)
result = simulator.result()
```

For a live view, keep one workbench open and refresh it after each batch:

```python
import dds.viz

workbench = dds.viz.show(simulator)
simulator.add_deposits(next_batch)
workbench.refresh(simulator)
workbench.app.exec()
```

See `examples/live_simulation.py` for a timer-driven example.

Chunked storage is a separate workflow under `dds.fields`. It allocates only
requested compositions and defaults to max-only storage:

```python
from dds.fields import accumulate_chunked_field

chunked = accumulate_chunked_field(
    domain,
    [deposit],
    chunk_shape=(32, 32, 32),
    compositions=("max",),
)
dense = chunked.to_dense("max")
```

Low-level in-place accumulation helpers also live in `dds.fields`.

## Persistence

Result bundles write interoperable arrays and JSON metadata:

```python
result.save("outputs/run_01")
```

Typed checkpoints preserve the domain, `length_unit`, profiles, metadata,
deposits, threshold, and computed fields:

```python
from dds.io import load_checkpoint, save_checkpoint

path = save_checkpoint("outputs/run_01.npz", result)
restored = load_checkpoint(path)
```

The checkpoint schema is intentionally pre-release and has no migration layer.
Loading an older or newer schema raises a clear `ValueError`.

## Geometry and mesh API

Analytic SDF shapes, Boolean operations, transforms, mesh adapters, and mesh
metrics are available from `dds.geometry`. Mesh-dependent operations require
the `mesh` optional dependencies.

```python
from dds import Domain
from dds.geometry import mesh_surface_area, sphere

shape = sphere(radius=2.0)
mesh_domain = Domain.from_bounds(
    xmin=-3.0,
    xmax=3.0,
    ymin=-3.0,
    ymax=3.0,
    zmin=-3.0,
    zmax=3.0,
    voxel_size=0.25,
)
mesh = shape.to_mesh(mesh_domain)
area = mesh_surface_area(mesh)
```

## Formats and targets

External format adapters are isolated from the root package:

```python
from dds.formats.yaml import load_targets
from dds.targets import point_deposits_from_targets

targets = load_targets("example_wall.yaml")
deposits = point_deposits_from_targets(targets, profile=profile)
```

## Visualization

Visualization is optional and is not imported by `import dds`:

```python
import dds.viz
from dds.viz import ViewConfig

dds.viz.show(
    result,
    initial_view=ViewConfig(
        view_mode="surface",
        color_mode="overhang",
        build_direction="+Z",
    ),
)
```

## Examples

Examples do not write files unless an output directory is supplied:

```bash
python examples/basic_simulation.py --help
python examples/yaml_simulation.py --help
```

## Development

```bash
pytest
ruff check src tests examples
mypy
```

The optional workbench test is gated by `DDS_RUN_VIZ_TESTS=1`.

## Design documentation

- [Architecture](docs/architecture.md)
- [Modeling assumptions](docs/modeling-assumptions.md)

The project is pre-release. No package release or stable checkpoint/API
compatibility guarantee has been declared.
