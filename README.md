# 3DP-DDS

`3DP-DDS` is a Python library for dense deposition simulation on a 3D voxel grid. The import package is `dds`. It covers the full pipeline from raw deposit primitives through density accumulation, occupancy and surface extraction, headless analysis, and result serialization — all without requiring a display or visualization stack.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Simulation domain](#simulation-domain)
- [Deposit primitives](#deposit-primitives)
- [Stateless simulation](#stateless-simulation)
- [Stateful Simulator](#stateful-simulator)
  - [Incremental accumulation](#incremental-accumulation)
  - [Step-by-step / iterative use](#step-by-step--iterative-use)
  - [Low-level field helpers](#low-level-field-helpers)
- [Sparse density field](#sparse-density-field)
- [Checkpoints](#checkpoints)
- [Analysis](#analysis)
- [Target workflows and YAML](#target-workflows-and-yaml)
- [Geometry and SDFs](#geometry-and-sdfs)
- [Mesh extraction and conversion](#mesh-extraction-and-conversion)
- [Headless mesh analysis](#headless-mesh-analysis)
- [Exporting results](#exporting-results)
- [Interactive workbench](#interactive-workbench)
- [CLI example scripts](#cli-example-scripts)
- [Package layout](#package-layout)
- [Design conventions](#design-conventions)

---

## Installation

### Core (required)

```bash
pip install -e .
```

Core dependencies: `numpy`, `scipy`, `trimesh`, `tyro`.

### Optional extras

| Extra | Adds | Install |
|-------|------|---------|
| `formats` | YAML target file support via PyYAML | `pip install -e ".[formats]"` |
| `mesh` | Mesh file I/O and marching-cubes extraction via meshio + scikit-image | `pip install -e ".[mesh]"` |
| `viz` | Interactive PyVistaQt workbench | `pip install -e ".[viz]"` |
| `dev` | pytest and pytest-qt for running the test suite | `pip install -e ".[dev]"` |

Install multiple extras together:

```bash
pip install -e ".[formats,mesh]"
pip install -e ".[formats,mesh,viz,dev]"
```

---

## Quick start

```python
from dds import BeadProfile, DepositionMetadata, Domain, LineDeposit, PointDeposit, simulate

domain = Domain.from_bounds(
    xmin=0.0, xmax=100.0,
    ymin=0.0, ymax=100.0,
    zmin=0.0, zmax=20.0,
    voxel_size=0.5,
)

profile = BeadProfile(width=1.2, height=0.4)
metadata = DepositionMetadata(layer_id=0)

deposits = [
    PointDeposit(x=10.25, y=10.25, z=0.25, profile=profile, metadata=metadata),
    LineDeposit(
        start=(10.25, 10.25, 0.25),
        end=(50.25, 10.25, 0.25),
        profile=profile,
        metadata=metadata,
    ),
]

result = simulate(domain, deposits, threshold=0.5)
occupancy = result.occupancy()          # dense bool grid
surface   = result.surface_mesh()       # triangle mesh (requires [mesh])
```

---

## Simulation domain

`Domain` defines the axis-aligned workspace and its dense voxel grid.

```python
from dds import Domain

# Explicit scalar bounds with isotropic voxel size
domain = Domain.from_bounds(
    xmin=0.0, xmax=50.0,
    ymin=0.0, ymax=50.0,
    zmin=0.0, zmax=10.0,
    voxel_size=0.5,
)

# Anisotropic voxels
domain = Domain.from_bounds(
    xmin=0.0, xmax=50.0,
    ymin=0.0, ymax=50.0,
    zmin=0.0, zmax=10.0,
    voxel_size=(0.5, 0.5, 0.25),
)

# Fit a domain tightly around a deposit list
domain = Domain.from_deposits(deposits, voxel_size=0.5, padding="auto")

print(domain.grid_shape)           # (nx, ny, nz)
print(domain.voxel_size)           # (dx, dy, dz)
print(domain.min_corner)           # world origin
world = domain.index_to_world((5, 10, 2))
idx   = domain.world_to_index(world)
```

Array indexing follows `(x, y, z)` / NumPy `indexing="ij"` convention throughout.

---

## Deposit primitives

```python
from dds import BeadProfile, DepositionMetadata, LineDeposit, PointDeposit

profile  = BeadProfile(width=1.5, height=0.6)
metadata = DepositionMetadata(
    layer_id=0,
    material_id="PLA",
    tool_id="T0",
    feedrate=2400.0,
    temperature=215.0,
    timestamp=0.0,
    user_data={"pass": "contour"},
)

# Point deposit — nozzle-tip / top-of-bead target at (x, y, z)
pt = PointDeposit(x=5.0, y=5.0, z=0.3, profile=profile, metadata=metadata)

# Or from a sequence or Point3D
pt = PointDeposit.from_point((5.0, 5.0, 0.3), profile=profile, metadata=metadata)

# Line deposit — swept bead between two nozzle-tip targets
ln = LineDeposit(
    start=(5.0, 5.0, 0.3),
    end=(45.0, 5.0, 0.3),
    profile=profile,
    metadata=metadata,
)

# Non-vertical print axis (e.g. for tilted deposition heads)
ln_tilted = LineDeposit(
    start=(5.0, 5.0, 0.3),
    end=(45.0, 5.0, 0.3),
    profile=profile,
    start_z_axis=(0.0, 0.0, 1.0),
    end_z_axis=(0.1, 0.0, 0.995),   # axis interpolated along segment
)
```

Coordinates are top-referenced: the target represents the nozzle tip or the top surface of the bead, not the bead centre.

---

## Stateless simulation

`simulate()` is the one-shot entry point. It is stateless and returns a rich `SimulationResult`.

```python
from dds import simulate

# Default: max-envelope density only
result = simulate(domain, deposits, threshold=0.5)

# Request the geometry field and additive coverage diagnostic
result = simulate(domain, deposits, compositions=("max", "coverage"), threshold=0.5)

# Result queries
density_max  = result.field("max")             # ndarray (nx, ny, nz)
coverage     = result.field("coverage")        # ndarray or raises if not computed
occupancy    = result.occupancy()              # bool ndarray
surface      = result.surface_mesh()          # TriangleMesh (requires [mesh])
bundle       = result.analysis_bundle()       # AnalysisBundle for further queries

# Layer/strata breakdown (needs metadata.layer_id set on deposits)
strata       = result.strata(mode="layer")
layer0_occ   = result.layer_occupancy(layer_id=0)
layer0_dens  = result.layer_density(layer_id=0)
iface        = result.interface()             # contact and overlap metrics
```

---

## Stateful Simulator

`Simulator` maintains an internal deposit list and lazily cached dense fields. All caches are updated incrementally when new deposits are added — no full recompute on each append.

```python
from dds import Domain, Simulator

sim = Simulator(domain)                        # empty
sim = Simulator(domain, deposits)             # pre-populated
```

### Querying the live state

```python
# Dense field snapshots (backed by independent arrays — safe to hold)
result  = sim.result(compositions=("max",), threshold=0.5)
bundle  = sim.analysis_bundle()

# Convenience field accessors
density = sim.sample_field(field="density")       # max-envelope geometry
coverage = sim.sample_field(field="coverage")     # additive diagnostic
occ     = sim.simulate_occupancy(threshold=0.5)
idx     = sim.simulate_deposition_index()

# Point queries
sim.is_occupied((5.0, 5.0, 0.3), threshold=0.5)
sim.query_deposition_index((5.0, 5.0, 0.3))
sim.sample_density_at((5.0, 5.0, 0.3), interpolation="trilinear")

# Inspect current deposits
print(len(sim.deposits))
```

### Incremental accumulation

Adding deposits updates only the warm caches rather than recomputing the full grid. The kernel is sampled once and written to every live dense cache simultaneously.

```python
sim = Simulator(domain)

for deposit in toolpath:
    sim.add_deposit(deposit)               # O(kernel window) per deposit

# add_deposits accepts a list or any DepositInput
sim.add_deposits([dep_a, dep_b, dep_c])

# clear_deposits reuses the allocated grids (fill to zero) instead of freeing them
sim.clear_deposits()
sim.add_deposits(new_toolpath)
```

Snapshot objects (`result()`, `analysis_bundle()`) copy their backing arrays at creation time, so an existing snapshot is never mutated by a later `add_deposit`.

### Step-by-step / iterative use

Because `add_deposit` and field queries are both cheap after the first warm-up, you can query mid-toolpath without triggering a full recompute:

```python
sim = Simulator(domain)
probe = (25.0, 25.0, 1.0)

for i, deposit in enumerate(toolpath):
    sim.add_deposit(deposit)

    # query every 100 deposits
    if i % 100 == 0:
        print(f"deposit {i}: occupied={sim.is_occupied(probe)}")

    # snapshot per layer boundary
    if deposit.metadata.layer_id != getattr(previous, "metadata.layer_id", None):
        layer_result = sim.result()
        layer_result.checkpoint(f"checkpoints/layer_{deposit.metadata.layer_id}.npz")
```

### Low-level field helpers

For custom accumulation loops where you manage the grid yourself:

```python
import numpy as np
from dds import apply_deposit_to_field, apply_deposit_to_index_field, Domain

grid       = np.zeros(domain.grid_shape, dtype=float)
index_grid = np.full(domain.grid_shape, -1, dtype=np.intp)

for i, deposit in enumerate(deposits):
    apply_deposit_to_field(domain, grid, deposit, composition="coverage")  # returns bool
    apply_deposit_to_index_field(domain, index_grid, deposit, i)
```

---

## Sparse density field

`SparseDensityField` stores deposit kernels as compact sub-arrays instead of a full dense grid. The full grid is materialised on demand. This is useful when the domain is large but the toolpath only covers a small fraction of it.

### Standalone

```python
from dds import accumulate_density_sparse

sparse = accumulate_density_sparse(domain, deposits)

# Materialise when needed
dense_max = sparse.to_dense(composition="max")
dense_coverage = sparse.to_dense(composition="coverage")

# Both compositions in a single pass
grids = sparse.to_dense_all("max", "coverage")

# Memory diagnostics
print(f"{sparse.contribution_count} kernel sub-arrays")
print(f"{sparse.nbytes / 1e6:.1f} MB stored  vs  {sparse.dense_nbytes / 1e6:.1f} MB dense")
print(f"sparsity ratio: {sparse.sparsity:.3f}")
```

### Via Simulator

`Simulator.sparse_field()` is built lazily and kept in sync with the deposit list. The same kernel sample that updates the dense caches also updates the sparse cache — no extra SDF evaluation.

```python
sim = Simulator(domain, deposits[:10])

sparse = sim.sparse_field()          # built from current deposits
print(sparse.contribution_count)     # 10

sim.add_deposit(deposits[10])
print(sim.sparse_field().contribution_count)  # 11 — incremental update

sim.clear_deposits()
print(sim.sparse_field().contribution_count)  # 0 — cleared, object reused
```

---

## Checkpoints

A checkpoint is a single compressed `.npz` file containing the density arrays and a JSON blob with the full deposit list, domain geometry, and threshold. No re-simulation is needed to restore a result.

```python
from dds import simulate, save_checkpoint, load_checkpoint, SimulationResult

result = simulate(domain, deposits, compositions=("max", "coverage"), threshold=0.5)

# Save
path = save_checkpoint("outputs/wall_layer3.npz", result)

# Load
restored = load_checkpoint("outputs/wall_layer3.npz")

# Convenience wrappers on SimulationResult
result.checkpoint("outputs/wall_layer3")        # .npz appended automatically
restored = SimulationResult.load("outputs/wall_layer3.npz")

# The restored result is fully functional
print(restored.domain.grid_shape)
print(len(restored.deposits))
occ = restored.occupancy()
```

The checkpoint format is versioned. An unsupported version raises `ValueError`.

---

## Analysis

`AnalysisBundle` is the main headless analysis object. It caches derived fields and surfaces until inputs change.

```python
from dds import Simulator
from dds.analysis import sample_points, signed_distance_at

sim    = Simulator(domain, deposits)
bundle = sim.analysis_bundle()           # or result.analysis_bundle()

# Density and field sampling
density = bundle.density_field()
occ     = bundle.occupancy_field(threshold=0.5)
d_idx   = bundle.deposition_index_field()

# Point queries
inside_density = bundle.sample_density_at((10.0, 10.0, 0.5), interpolation="trilinear")
inside_mesh    = bundle.contains_point((10.0, 10.0, 0.5), representation="mesh", threshold=0.5)
sdf_dist       = signed_distance_at(bundle, (12.0, 10.0, 0.5), threshold=0.5)

# Batch point sampling
samples = sample_points(
    bundle,
    [(10.0, 10.0, 0.5), (20.0, 20.0, 1.0)],
    fields=("density", "occupancy", "deposition_index", "signed_distance"),
    threshold=0.5,
    interpolation="trilinear",
)

# Strata (requires deposits to carry layer_id metadata)
strata = bundle.strata(mode="layer", threshold=0.5)

# Interface metrics between consecutive strata
iface = bundle.interface(mode="layer", threshold=0.5)

# Support analysis
support = bundle.support(build_direction=(0.0, 0.0, 1.0), threshold=0.5)
```

Analysis API summary:

- `AnalysisBundle`, `analysis_bundle(...)`
- `contains_point(...)`, `sample_density_at(...)`, `sample_deposition_index_at(...)`
- `signed_distance_at(...)`, `surface_normal_at(...)`
- `sample_points(...)`

---

## Target workflows and YAML

`dds.targets` converts ordered lists of nozzle-tip target points into deposit primitives. `dds.formats.yaml` loads targets from YAML files (requires the `formats` extra).

```python
from dds import BeadProfile, Domain, simulate
from dds.formats.yaml import load_targets
from dds.targets import point_deposits_from_targets, line_deposits_from_targets

# Load from a YAML target file
targets  = load_targets("example_wall.yaml")
profile  = BeadProfile(width=18.0, height=12.0)

# Each target becomes a PointDeposit
point_deposits = point_deposits_from_targets(
    targets,
    profile=profile,
    origin_reference="top",    # "top" or "center"
)

# Consecutive pairs become LineDeposits
line_deposits = line_deposits_from_targets(
    targets,
    profile=profile,
    origin_reference="top",
)

domain = Domain.from_deposits(point_deposits, voxel_size=1.0, padding="auto")
result = simulate(domain, point_deposits, compositions=("max", "coverage"), threshold=0.5)
```

YAML target file structure:

```yaml
targets:
  - index: 0
    origin: [10.0, 10.0, 0.6]
    z_axis: [0.0, 0.0, 1.0]
  - index: 1
    origin: [20.0, 10.0, 0.6]
```

---

## Geometry and SDFs

`dds.geometry` adds an analytic SDF layer. SDFs follow the **negative inside** convention.

```python
from dds import Domain
from dds.geometry import box, cylinder, sphere, union, intersection, difference

domain = Domain.from_bounds(
    xmin=-8.0, xmax=8.0,
    ymin=-8.0, ymax=8.0,
    zmin=-8.0, zmax=8.0,
    voxel_size=0.25,
)

# Compose shapes
outer = union(sphere(radius=3.0), box(size=(4.0, 4.0, 6.0)))
hole  = cylinder(radius=0.8, height=8.0)
shape = difference(outer, hole)          # outer - hole

# Sample onto a dense grid
density = shape.sample(domain)           # ndarray; values ≤ 0 are inside
occupancy = density <= 0.0
```

Available primitives: `sphere`, `box`, `cylinder`, `capsule`, `plane`, `slab`, `ellipsoid`, `torus`, `rounded_box`, `capped_cylinder`, `rounded_cylinder`, `capped_cone`, `cone`, `rounded_cone`, `capsule_chain`

Boolean / field operations: `union`, `intersection`, `difference`, `dilate`, `erode`, `shell`

Transforms: `translate`, `scale`, `rotate`, `orient`, `rotation_matrix`

SDF types: `SDF3`, `GridSDF3`, `MeshSDF3`, `as_sdf3`

---

## Mesh extraction and conversion

Requires the `mesh` extra (`pip install -e ".[mesh]"`).

```python
from dds.geometry import (
    density_to_mesh,
    density_to_sdf,
    mesh_to_sdf_field,
    occupancy_to_mesh,
    read_mesh,
    write_mesh,
)

# Extract a surface mesh from a density field
surface = density_to_mesh(domain, density, threshold=0.5)
write_mesh("outputs/surface.ply", surface)

# Load a mesh and convert it back to a sampled SDF
mesh    = read_mesh("outputs/surface.ply")
sdf_arr = mesh_to_sdf_field(domain, mesh)    # float64 grid

# Derive an SDF from density and wrap it as a sampled SDF object
sdf_obj = density_to_sdf(domain, density, threshold=0.5)

# Occupancy → mesh (marching cubes on a bool grid)
occ_surface = occupancy_to_mesh(domain, density <= 0.5)
```

Mesh API:

- `TriangleMesh`, `read_mesh`, `write_mesh`
- `extract_mesh_from_field`, `occupancy_to_mesh`, `density_to_mesh`, `sdf_to_mesh`
- `mesh_to_occupancy`, `mesh_to_sdf_field`
- `occupancy_to_sdf_field`, `density_to_sdf_field`, `occupancy_to_sdf`, `density_to_sdf`

Signed-distance and containment queries assume watertight triangle meshes.

---

## Headless mesh analysis

```python
from dds.geometry import mesh_surface_area, overhang_angles

bundle   = sim.analysis_bundle()
analysis = bundle.mesh_analysis(build_direction=(0.0, 0.0, 1.0), critical_angle_deg=45.0)
stats    = bundle.subvolume_stats(
    ((0.0, 0.0, 0.0), (20.0, 20.0, 2.0)),
    threshold=0.5,
)

mesh    = analysis["mesh"]
area    = mesh_surface_area(mesh)
angles  = overhang_angles(mesh, build_direction=(0.0, 0.0, 1.0))
```

Mesh analysis API:

- `face_normals(...)`, `vertex_normals(...)`
- `face_centroids(...)`, `face_areas(...)`
- `overhang_angles(...)`, `downfacing_mask(...)`, `support_risk_mask(...)`
- `normal_rgb_from_normals(...)`
- `mesh_bounds_stats(...)`, `mesh_surface_area(...)`, `mesh_volume_estimate(...)`
- `AnalysisBundle.mesh_analysis(...)`, `AnalysisBundle.subvolume_stats(...)`

---

## Exporting results

### Simulation bundle (directory of arrays)

```python
from dds.io import save_array, save_simulation_bundle

save_simulation_bundle(
    "outputs/run_01",
    domain=domain,
    occupancy=result.occupancy(),
    deposition_index=result.analysis_bundle().deposition_index_field(),
    density=result.field("max"),
    metadata={"run": "01", "threshold": 0.5},
)

save_array("outputs/run_01/coverage.npy", result.field("coverage"))
```

Files written: `occupancy.npy`, `deposition_index.npy`, `density.npy`, `metadata.json` (plus any arrays you write separately).

### Checkpoint (single-file round-trip)

```python
from dds import save_checkpoint, load_checkpoint

save_checkpoint("outputs/run_01.npz", result)
restored = load_checkpoint("outputs/run_01.npz")
```

A checkpoint stores the full deposit list alongside the density arrays, so the restored `SimulationResult` can rerun analysis, strata, interface, and support queries without re-simulating.

---

## Interactive workbench

Requires the `viz` extra (`pip install -e ".[viz]"`).

```python
from dds import Simulator, WorkbenchViewConfig

sim = Simulator(domain, deposits)

sim.show(
    WorkbenchViewConfig(
        view_mode="surface",        # "surface" | "occupancy" | "density"
        build_direction="+Z",
        threshold=0.5,
    )
)

# Or from a SimulationResult
result.show()
```

`SimulationWorkbench` is loaded lazily; an `ImportError` with a clear install hint is raised if the `viz` dependencies are missing.

---

## CLI example scripts

```bash
# Run the basic simulation example
python examples/basic_simulation.py

# Typed --help via tyro
python examples/basic_simulation.py --help

# Adjust threshold
python examples/basic_simulation.py --threshold 0.35

# Write outputs to disk
python examples/basic_simulation.py --output-dir outputs/basic

# YAML-driven example
python examples/yaml_simulation.py
```

All example scripts expose typed `--help` output through dataclass-backed configs using `tyro`.

---

## Package layout

```text
src/dds/
├── analysis/
│   ├── __init__.py
│   ├── bundle.py       AnalysisBundle and cached query methods
│   ├── fields.py       Strata and layer-field helpers
│   ├── interface.py    Inter-layer contact and overlap metrics
│   ├── models.py       Result dataclasses for analysis outputs
│   ├── strata.py       Layer partitioning logic
│   └── support.py      Overhang and support-risk analysis
├── formats/
│   ├── __init__.py
│   └── yaml.py         YAML target file loader (optional: [formats])
├── geometry/
│   ├── __init__.py
│   ├── adapters.py     Dense-field ↔ SDF / mesh conversions
│   ├── mesh.py         Mesh I/O, extraction, and sampling (optional: [mesh])
│   ├── ops.py          SDF boolean and morphological operations
│   ├── sdf.py          SDF3, GridSDF3, MeshSDF3
│   ├── shapes.py       Analytic SDF primitives
│   └── transforms.py   Spatial transforms for SDFs
├── __init__.py         Public API surface
├── attributes.py       BeadProfile, DepositionMetadata
├── cli.py              Tyro-backed CLI entry point
├── domain.py           Domain definition and coordinate transforms
├── fields.py           Dense accumulation, apply_deposit_to_field, SparseDensityField helpers
├── io.py               save_array, save_simulation_bundle, save/load_checkpoint
├── kernels.py          SDF-kernel sampling (SampledKernel, sample_deposit_kernel)
├── mesh_analysis.py    Headless triangle-mesh metrics
├── occupancy.py        Threshold helpers
├── primitives.py       PointDeposit, LineDeposit, ToolpathDepositSequence, Point3D
├── results.py          SimulationResult, simulate(), WorkbenchViewConfig
├── simulator.py        Simulator (stateful, incremental caches)
├── sparse.py           SparseDensityField
├── targets.py          TargetPoint, point/line_deposits_from_targets
├── types.py            FieldComposition, FieldName type aliases
├── utils.py            Shared math helpers
├── viz.py              Lazy viz imports
└── workbench.py        SimulationWorkbench (optional: [viz])
```

---

## Design conventions

- **Import name**: `dds`; distribution name: `3dp-dds`; repository name: `3DP-DDS`
- **Array axis order**: `(x, y, z)` / NumPy `indexing="ij"` throughout
- **Top-referenced deposits**: the target point is the nozzle tip or the top surface of the bead, not the bead centre
- **Bead dimensions**: `width` is the full transverse bead width; `height` is the full bead height along the local axis
- **SDF sign convention**: negative inside, positive outside, zero on the surface
- **Field composition**: `"max"` is the canonical geometry field used for occupancy and surfaces
- **Coverage diagnostic**: `"coverage"` adds anti-aliased kernel samples for overlap inspection; it is not mass, volume, or physical density and changes with voxel resolution and path segmentation
- **Deposition index**: 0-based index of the last deposit touching each voxel; −1 for untouched voxels
- **Snapshot isolation**: `Simulator.result()` and `Simulator.analysis_bundle()` copy their backing arrays at creation time — holding an old snapshot is safe after further `add_deposit` calls
- **Cache sharing**: when multiple caches are warm, `add_deposit` samples the kernel once and fans the result out to all of them

