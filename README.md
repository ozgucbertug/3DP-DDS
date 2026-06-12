# 3DP-DDS

`3DP-DDS` is a Python library for geometry-first deposition simulation on a
3D voxel grid. The import package is `dds`.

It represents robotic additive-manufacturing paths as point, line, and
polyline deposition events, samples their bead geometry into reproducible
dense or chunked fields, and provides headless analysis, mesh conversion,
persistence, and optional interactive visualization.

The current scope is deposited geometry. Material flow, thermal history,
curing, robot dynamics, controller behavior, and bead deformation are
intentionally outside the model.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Simulation domain](#simulation-domain)
- [Geometry and deposit primitives](#geometry-and-deposit-primitives)
- [Simulation workflows](#simulation-workflows)
- [Chunked fields](#chunked-fields)
- [Results and analysis](#results-and-analysis)
- [Persistence](#persistence)
- [Target workflows and YAML](#target-workflows-and-yaml)
- [Geometry and mesh API](#geometry-and-mesh-api)
- [Visualization](#visualization)
- [Example scripts](#example-scripts)
- [Package layout](#package-layout)
- [Design conventions](#design-conventions)

## Installation

3DP-DDS requires Python 3.11 or newer. From a local clone, install the core
library in editable mode:

```bash
python -m pip install -e .
```

The core dependencies are NumPy, SciPy, and Tyro.

### Optional extras

| Extra | Adds | Install |
| --- | --- | --- |
| `formats` | YAML target loading through PyYAML | `python -m pip install -e ".[formats]"` |
| `mesh` | Trimesh-backed mesh and point-cloud I/O, extraction, containment, and signed-distance operations | `python -m pip install -e ".[mesh]"` |
| `viz` | PyVistaQt interactive workbench and mesh dependencies | `python -m pip install -e ".[viz]"` |
| `all` | All `formats`, `mesh`, and `viz` capabilities | `python -m pip install -e ".[all]"` |

Install several selected extras or all optional runtime capabilities:

```bash
python -m pip install -e ".[formats,mesh]"
python -m pip install -e ".[all]"
```

## Quick start

```python
from dds import (
    BeadProfile,
    DepositionTarget,
    Domain,
    LineDeposit,
    PointDeposit,
    simulate,
)

domain = Domain.from_bounds(
    xmin=0.0,
    xmax=20.0,
    ymin=0.0,
    ymax=20.0,
    zmin=-1.0,
    zmax=5.0,
    voxel_size=0.25,
    length_unit="mm",
)

profile = BeadProfile(width=1.2, height=0.6)

deposits = [
    PointDeposit(
        target=DepositionTarget(
            position=(2.0, 2.0, 0.6),
            normal=(0.0, 0.0, 1.0),
        ),
        profile=profile,
    ),
    LineDeposit(
        start=(2.0, 2.0, 0.6),
        end=(10.0, 2.0, 0.6),
        profile=profile,
    ),
]

result = simulate(domain, deposits, threshold=0.5)

implicit = result.implicit_field
occupancy = result.analysis.occupancy()
deposition_index = result.analysis.deposition_index_field()

print(domain.grid_shape)
print(int(occupancy.sum()))
```

Every deposit requires an explicit `BeadProfile`. Deposit targets are
top-referenced: a target is the nozzle position or top surface of the bead,
not its center.

## Simulation domain

`Domain` defines an axis-aligned workspace and the voxel-center lattice used
for sampling.

```python
from dds import Domain

# Isotropic voxels.
domain = Domain.from_bounds(
    xmin=0.0,
    xmax=50.0,
    ymin=0.0,
    ymax=50.0,
    zmin=0.0,
    zmax=10.0,
    voxel_size=0.5,
    length_unit="mm",
)

# Anisotropic voxels.
anisotropic = Domain.from_bounds(
    xmin=0.0,
    xmax=50.0,
    ymin=0.0,
    ymax=50.0,
    zmin=0.0,
    zmax=10.0,
    voxel_size=(0.5, 0.5, 0.25),
)

print(domain.grid_shape)
print(domain.voxel_size)
print(domain.min_corner)
print(domain.max_corner)

point = domain.index_to_world((5, 10, 2))
index = domain.world_to_index(point)
```

`Domain.from_bounds()` aligns the upper bounds to whole voxels. If a requested
extent is not divisible by the voxel size, the resulting `max_corner` is
expanded to the next voxel boundary.

A domain can also be fitted around bead support:

```python
from dds import Domain

fitted_domain = Domain.from_deposits(
    deposits,
    voxel_size=0.25,
    padding="auto",
    length_unit="mm",
)
```

Array indexing follows `(x, y, z)` and NumPy `indexing="ij"` conventions.
`length_unit` records whether world-space values use millimeters or meters;
the library does not perform unit conversion.

## Geometry and deposit primitives

The core geometry types separate positions, directions, poses, and paths:

- `Point3D`: a Cartesian position.
- `Vector3D`: a direction or displacement.
- `Pose3D`: a complete rigid transform composed of a position and SciPy
  `Rotation`.
- `DepositionTarget`: the top position and normalized deposition normal
  consumed by bead kernels.
- `Line3D`: a finite line between two points.
- `Polyline3D`: a connected sequence of points.

Deposition types combine that geometry with bead dimensions:

- `PointDeposit`: one compact bead at a target.
- `LineDeposit`: a bead swept between two targets.
- `PolylineDeposit`: one ordered, multi-segment deposition event.

```python
from dds import (
    BeadProfile,
    DepositionTarget,
    LineDeposit,
    PointDeposit,
    PolylineDeposit,
)

profile = BeadProfile(width=1.5, height=0.6)

# A coordinate triplet is interpreted as a world-+Z target.
point = PointDeposit(
    target=(5.0, 5.0, 0.6),
    profile=profile,
)

# Explicit targets support non-vertical deposition normals.
line = LineDeposit(
    start=DepositionTarget(
        position=(5.0, 5.0, 0.6),
        normal=(0.0, 0.0, 1.0),
    ),
    end=DepositionTarget(
        position=(15.0, 5.0, 1.6),
        normal=(0.1, 0.0, 0.995),
    ),
    profile=profile,
)

polyline = PolylineDeposit(
    targets=(
        (5.0, 5.0, 0.6),
        (15.0, 5.0, 0.6),
        (15.0, 15.0, 0.6),
    ),
    profile=profile,
)
```

Normals are normalized automatically. Antiparallel normals on consecutive line
endpoints are rejected because their interpolation is ambiguous.

Use a full pose when robot orientation is available:

```python
from scipy.spatial.transform import Rotation

from dds import DepositionTarget, Pose3D

pose = Pose3D(
    position=(10.0, 20.0, 30.0),
    orientation=Rotation.from_euler("xyz", [0.0, 45.0, 90.0], degrees=True),
)
target = DepositionTarget.from_pose(pose)
```

By default, conversion transforms tool-local `+Z` into the world deposition
normal. Use `DepositionTarget.from_pose(pose, local_axis=...)` for another
tool-local deposition axis. Roll about that selected axis is intentionally
discarded because the current bead model is rotationally symmetric.

## Simulation workflows

### One-shot simulation

Use `simulate()` when all deposits are already available:

```python
from dds import simulate

result = simulate(
    domain,
    deposits,
    include_coverage=True,
    threshold=0.5,
)

geometry = result.implicit_field
coverage = result.coverage
```

`implicit_field` is the canonical union-like fabricated geometry used for
occupancy, surfaces, SDFs, and support analysis.

It is not itself an SDF: values are clipped to `[0, 1]`, remain nonnegative,
and define the nominal surface at `0.5`. Use
`result.analysis.surface_sdf()` when signed metric distance is required.

Optional `coverage` adds kernel contributions. It is useful for locating path
overlap, but it is not physical density, mass, volume fraction, or material
flow.

### Stateful and incremental simulation

Use `Simulator` when deposits arrive over time:

```python
from dds import Simulator

simulator = Simulator(domain)

for deposit in deposits:
    simulator.add_deposit(deposit)

result = simulator.result(
    include_coverage=True,
    threshold=0.5,
)
```

Several events can be added at once:

```python
simulator.add_deposits(next_batch)
print(len(simulator.deposits))
```

`Simulator` maintains lazily created dense caches. Once a cache is warm, new
deposits update it incrementally. `clear_deposits()` resets the simulation
while reusing existing dense allocations where possible.

Each call to `result()` creates an immutable snapshot. Adding deposits later
does not mutate snapshots that have already been returned.

### Low-level accumulation

For custom loops that own their arrays, use the helpers in `dds.fields`:

```python
import numpy as np

from dds.fields import apply_deposit_to_field, apply_deposit_to_index_field

implicit_grid = np.zeros(domain.grid_shape, dtype=float)
index_grid = np.full(domain.grid_shape, -1, dtype=np.intp)

for deposit_index, deposit in enumerate(deposits):
    apply_deposit_to_field(
        domain,
        implicit_grid,
        deposit,
        field="implicit",
    )
    apply_deposit_to_index_field(
        domain,
        index_grid,
        deposit,
        deposit_index,
    )
```

## Chunked fields

`ChunkedField` allocates fixed-size dense chunks only where deposition touches
the domain. This is useful for large workspaces with spatially localized
toolpaths.

```python
from dds.fields import accumulate_chunked_field

chunked = accumulate_chunked_field(
    domain,
    deposits,
    chunk_shape=(32, 32, 32),
    include_coverage=True,
)

dense_implicit = chunked.to_dense("implicit")
dense_coverage = chunked.to_dense("coverage")

roi = chunked.materialize(
    "implicit",
    index_bounds=((0, 32), (0, 32), (0, 16)),
)

print(chunked.chunk_count)
print(chunked.event_count)
print(chunked.active_voxel_count)
print(chunked.allocation_fraction)
print(chunked.memory_ratio)
```

A chunked field can be converted to a normal immutable result without
re-running deposition:

```python
result = chunked.to_result(deposits, threshold=0.5)
occupancy = result.analysis.occupancy()
```

Chunked storage is a standalone workflow; `Simulator` owns dense incremental
caches.

## Results and analysis

`SimulationResult` stores immutable copies of the computed fields and deposit
sequence. Derived operations are cached on `result.analysis`.

```python
analysis = result.analysis

implicit = result.implicit_field
occupancy = analysis.occupancy(threshold=0.5)
deposition_index = analysis.deposition_index_field()
deposition_order = analysis.deposition_order_field(threshold=0.5)

implicit_at_point = analysis.sample_implicit_value(
    (5.0, 5.0, 0.3),
    interpolation="trilinear",
)
deposit_at_point = analysis.sample_deposition_index((5.0, 5.0, 0.3))
inside = analysis.contains_point(
    (5.0, 5.0, 0.3),
    representation="occupancy",
)

samples = analysis.sample_points(
    [(5.0, 5.0, 0.3), (10.0, 10.0, 1.0)],
    fields=("implicit", "occupancy", "deposition_index", "signed_distance"),
    interpolation="trilinear",
)
```

Grid-derived SDF queries are available from the core installation:

```python
sdf = analysis.surface_sdf(threshold=0.5)
distance = analysis.signed_distance_at((5.0, 5.0, 0.3))
normal = analysis.surface_normal_at((5.0, 5.0, 0.3))

stats = analysis.subvolume_stats(
    ((0.0, 0.0, -1.0), (10.0, 10.0, 2.0)),
    threshold=0.5,
)
```

Surface extraction and mesh-backed signed-distance queries require the `mesh`
extra:

```python
surface = analysis.surface_mesh(threshold=0.5)
mesh_sdf = analysis.mesh_sdf(threshold=0.5)
inside_mesh = analysis.contains_point(
    (5.0, 5.0, 0.3),
    representation="mesh",
)
```

Deposit-order interface and support analysis are also available:

```python
strata = analysis.strata(threshold=0.5)
interfaces = analysis.interface(threshold=0.5)
support = analysis.support(
    build_direction="+Z",
    critical_angle_deg=45.0,
    threshold=0.5,
)

print(strata.stratum_ids)
print(interfaces.contact_area)
print(support.risk_area)
```

Strata and interfaces follow deposit order. For visualization or per-voxel
order queries without materializing every stratum, use
`deposition_order_field()`. It returns a cached, read-only integer field with
one-based deposit order and zero for untouched voxels.

## Persistence

### Result bundle

`SimulationResult.save()` writes interoperable arrays and JSON metadata:

```python
written = result.save(
    "outputs/run_01",
    metadata={"experiment": "wall_01"},
)

print(written)
```

The bundle includes occupancy, deposition index, the implicit field, metadata, and
coverage when it was requested.

### Typed checkpoint

A checkpoint is a compressed `.npz` round trip containing the domain, deposit
sequence, bead profiles, threshold, and computed fields:

```python
from dds import SimulationResult

path = result.checkpoint("outputs/run_01.npz")
restored = SimulationResult.load(path)

print(restored.domain.grid_shape)
print(len(restored.deposits))
```

The same operations are available from `dds.io`:

```python
from dds.io import load_checkpoint, save_checkpoint

path = save_checkpoint("outputs/run_02.npz", result)
restored = load_checkpoint(path)
```

The checkpoint schema is pre-release and strict. Unsupported schema versions
raise `ValueError`; there is currently no migration layer.

## Target workflows and YAML

`dds.targets` converts ordered deposition targets into point, line, or
polyline deposits. `dds.formats.yaml` loads targets from YAML and requires the `formats`
extra.

```python
from dds import BeadProfile, Domain, simulate
from dds.formats.yaml import load_targets
from dds.targets import (
    line_deposits_from_targets,
    point_deposits_from_targets,
    toolpath_from_targets,
)

targets = load_targets("example_wall.yaml")
profile = BeadProfile(width=18.0, height=12.0)

point_deposits = point_deposits_from_targets(
    targets,
    profile=profile,
    origin_reference="top",
)
line_deposits = line_deposits_from_targets(
    targets,
    profile=profile,
    origin_reference="top",
)
toolpath = toolpath_from_targets(
    targets,
    profile=profile,
    origin_reference="top",
)

domain = Domain.from_deposits(
    point_deposits,
    voxel_size=1.0,
    padding="auto",
)
result = simulate(domain, point_deposits)
```

Minimal YAML format:

```yaml
targets:
  - index: 0
    origin: [10.0, 10.0, 0.6]
    axis: [0.0, 0.0, 1.0]
  - index: 1
    origin: [20.0, 10.0, 0.6]
    axis: [0.0, 0.0, 1.0]
```

Targets may also use compact plane strings such as
`O(10,10,0.6) Z(0,0,1)`.

## Geometry and mesh API

`dds.geometry` provides analytic signed-distance shapes, Boolean operations,
transforms, mesh conversion, mesh I/O, and mesh metrics. Analytic SDF creation
and sampling use the core dependencies. SDFs use the convention negative
inside, positive outside, and zero on the surface.

```python
from dds import Domain
from dds.geometry import box, difference, sphere

mesh_domain = Domain.from_bounds(
    xmin=-5.0,
    xmax=5.0,
    ymin=-5.0,
    ymax=5.0,
    zmin=-5.0,
    zmax=5.0,
    voxel_size=0.25,
)

outer = sphere(radius=3.0)
hole = box(size=(2.0, 2.0, 8.0))
shape = difference(outer, hole)

sdf_values = shape.sample(mesh_domain)
mesh = shape.to_mesh(mesh_domain)
```

Available shape families include spheres, boxes, cylinders, capsules,
ellipsoids, toruses, cones, rounded primitives, slabs, planes, and capsule
chains. Shapes support:

- Boolean operations: `union`, `intersection`, `difference`.
- Morphology: `dilate`, `erode`, `shell`.
- Transforms: `translate`, `scale`, `rotate`, `orient`.

Mesh conversion and I/O require the `mesh` extra:

```python
from dds.geometry import (
    implicit_field_to_mesh,
    implicit_field_to_sdf,
    mesh_surface_area,
    mesh_to_sdf_field,
    occupancy_to_mesh,
    read_mesh,
    write_mesh,
)

surface = implicit_field_to_mesh(domain, result.implicit_field, threshold=0.5)
write_mesh("outputs/surface.ply", surface)

loaded = read_mesh("outputs/surface.ply")
sdf_field = mesh_to_sdf_field(domain, loaded)
surface_area = mesh_surface_area(loaded)

occupancy_surface = occupancy_to_mesh(
    domain,
    result.analysis.occupancy(),
)
implicit_sdf = implicit_field_to_sdf(
    domain,
    result.implicit_field,
    threshold=0.5,
)
```

Signed-distance and containment operations on triangle meshes generally
require watertight input.

## Visualization

Install the visualization dependencies:

```bash
python -m pip install -e ".[viz]"
```

Visualization is optional and is not imported by `import dds`.

### General geometry viewer

`dds.viz.Viewer` provides a retained scene for DDS geometry. Additions return
named handles that can be updated, hidden, restyled, or removed without
rebuilding unrelated visuals.

```python
from dds import DepositionTarget, Line3D, Point3D, Pose3D
from dds.geometry import read_mesh, read_point_cloud
from dds.viz import FrameStyle, LineStyle, MeshStyle, Viewer

viewer = Viewer()
viewer.add_mesh(
    read_mesh("part.stl"),
    name="part",
    style=MeshStyle(color="#93aec7", opacity=0.7),
)
viewer.add_point_cloud(read_point_cloud("scan.ply"), name="scan")
path = viewer.add_line(
    Line3D(Point3D(0.0, 0.0, 0.0), Point3D(20.0, 0.0, 0.0)),
    name="path",
    style=LineStyle(color="#355c9a", width=4.0),
)
viewer.add_pose(Pose3D((0.0, 0.0, 0.0)), style=FrameStyle(scale=5.0))
viewer.add_target(DepositionTarget((20.0, 0.0, 0.0)))

path.set_visible(False)
viewer.run()
```

The viewer accepts the DDS types `TriangleMesh`, `PointCloud`, `Point3D`,
`Line3D`, `Polyline3D`, `Vector3D`, `Pose3D`, `DepositionTarget`, and
deposition events. Point clouds preserve per-point RGB or RGBA colors by
default, and triangle meshes preserve vertex or face colors. A pose renders a
complete RGB XYZ frame. A deposition target renders only its point and normal
because it does not retain tool roll.

Use `viewer.batch()` when adding or updating several visuals so the viewport
renders once at the end of the operation.

### Simulation workbench

```python
import dds.viz
from dds.viz import ViewConfig

workbench = dds.viz.show(
    result,
    initial_view=ViewConfig(
        view_mode="surface",
        color_mode="overhang",
        build_direction="+Z",
        show_toolpath=True,
        show_targets=True,
        show_world_axes=True,
    ),
)
workbench.app.exec()
```

Available view modes are `"surface"`, `"occupancy"`, and `"implicit"`.

### Live incremental view

Keep one workbench open and refresh it after adding each batch:

```python
import dds.viz
from dds import Simulator

simulator = Simulator(domain)
workbench = dds.viz.show(simulator, threshold=0.5)

simulator.add_deposits(next_batch)
workbench.refresh(simulator)

workbench.app.exec()
```

See `examples/live_simulation.py` for a YAML-driven example. Press `Space` to
advance the deposition sequence and `R` to reset it without reopening the
workbench.

## Example scripts

The examples use typed Tyro command-line arguments where applicable:

```bash
# Basic point and line simulation.
python examples/basic_simulation.py
python examples/basic_simulation.py --help
python examples/basic_simulation.py --view
python examples/basic_simulation.py --output-dir outputs/basic

# YAML target workflow.
python examples/yaml_simulation.py --help
python examples/yaml_simulation.py --include-coverage
python examples/yaml_simulation.py --view

# Live YAML-driven visualization.
python examples/live_simulation.py --help
python examples/live_simulation.py
python examples/live_simulation.py --step-size 5
```

Examples do not write files unless an output directory is supplied.

## Package layout

```text
src/dds/
├── analysis/
│   ├── interface.py    Inter-stratum contact and overlap analysis
│   ├── models.py       Typed analysis result models
│   ├── simulation.py   Cached SimulationAnalysis query API
│   ├── strata.py       Deposit-order field partitioning
│   └── support.py      Overhang and support-shadow analysis
├── formats/
│   └── yaml.py         YAML target adapter (optional: formats)
├── geometry/
│   ├── _utils.py       Shared optional imports and validation
│   ├── mesh.py         Triangle meshes, I/O, and extraction
│   ├── ops.py          SDF Boolean and morphological operations
│   ├── point_cloud.py  Point clouds, I/O, and conversion
│   ├── sdf.py          SDF types and dense-field conversion
│   ├── shapes.py       Analytic SDF primitives
│   └── transforms.py   SDF spatial transforms
├── __init__.py         Core public API
├── attributes.py       BeadProfile
├── chunked.py          ChunkedField sparse storage
├── cli.py              Tyro-backed CLI helper
├── domain.py           Grid definition and coordinate transforms
├── fields.py           Dense and low-level accumulation helpers
├── io.py               Array bundles and typed checkpoints
├── kernels.py          Private tiled bead-kernel sampling
├── mesh_analysis.py    Triangle-mesh metrics
├── occupancy.py        Implicit threshold helpers
├── primitives.py       Geometry wrappers and deposition events
├── results.py          SimulationResult and simulate()
├── simulator.py        Stateful incremental dense simulation
├── targets.py          Ordered-pose conversion helpers
├── types.py            Shared public type aliases
├── viz/                Lazy viewer API, styles, and PyVista converters
└── workbench.py        PyVistaQt workbench (optional: viz)
```

The root `dds` namespace contains the core deposition and simulation types.
Specialized capabilities live in `dds.analysis`, `dds.geometry`, `dds.fields`,
`dds.formats`, `dds.io`, `dds.targets`, and `dds.viz`.

## Design conventions

- **Names**: distribution `3dp-dds`, import package `dds`, repository
  `3DP-DDS`.
- **Axis order**: arrays use `(x, y, z)` and NumPy `indexing="ij"`.
- **Top-referenced targets**: a target marks the bead top along its normal.
- **Bead dimensions**: width is the full transverse width; height is the full
  distance along the local axis.
- **Units**: world coordinates, bead dimensions, and voxel size use the
  domain's recorded `length_unit`; no conversion is performed.
- **Implicit field**: `implicit_field` is the canonical fabricated geometry.
- **Coverage**: `coverage` is a nonphysical overlap diagnostic that may
  change with voxel resolution and path segmentation.
- **SDF sign**: negative inside, positive outside, zero on the surface.
- **Deposition index**: the 0-based index of the last deposit touching each
  voxel; untouched voxels contain `-1`.
- **Snapshot isolation**: results and analysis arrays are copied and read-only.
- **Optional boundaries**: importing `dds` does not import visualization,
  format, or mesh dependencies.

For quantitative work, report the domain bounds, `length_unit`, voxel size,
threshold, bead profile, and path definition. Convergence checks across voxel
sizes are recommended.

Further documentation:

- [Architecture](docs/architecture.md)
- [Modeling assumptions](docs/modeling-assumptions.md)
- [Contributing and development setup](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

The project is pre-release. No package release or stable API/checkpoint
compatibility guarantee has been declared.
