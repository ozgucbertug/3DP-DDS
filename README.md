# 3DP-DDS
`3DP-DDS` is a Python library for dense deposition simulation on a 3D voxel grid. The import package is `dds`. The current library includes the dense simulator, rich simulation result objects, headless analysis, target-driven workflows, analytic SDF geometry, mesh extraction and conversion helpers, and an optional PyVistaQt workbench.

## Current Scope

- Explicit bead profiles plus per-deposit process metadata
- Dense 3D simulation domains with world/index coordinate transforms
- Smooth compact deposition kernels for point and line deposits
- Dense scalar accumulation with `max` and optional `sum` composition support
- A small stateful `Simulator` API plus reusable `SimulationResult` objects
- Cached headless `AnalysisBundle` queries over max-based dense fields, derived surfaces, and SDFs
- Generic target workflows in `dds.targets` plus YAML adapters in `dds.formats.yaml`
- Analytic SDF primitives, booleans, and spatial transforms in `dds.geometry`
- Mesh extraction, mesh IO, and dense-field or mesh-to-SDF conversions
- Headless triangle-mesh metrics, overhang analysis, and ROI summaries
- Dense result export helpers for arrays and simulation bundles
- Optional PyVistaQt workbench for interactive inspection
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

For the interactive workbench:

```bash
python -m pip install -e ".[viz]"
```

## Minimal Example

```python
from dds import (
    BeadProfile,
    DepositionMetadata,
    Domain,
    LineDeposit,
    PointDeposit,
    simulate,
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

profile = BeadProfile(width=1.2, height=0.4)
metadata = DepositionMetadata(layer_id=0)
deposits = [
    PointDeposit(x=10.25, y=10.25, z=0.25, profile=profile, metadata=metadata),
    LineDeposit(start=(10.25, 10.25, 0.25), end=(50.25, 10.25, 0.25), profile=profile, metadata=metadata),
]

result = simulate(domain, deposits, threshold=0.5)
occupancy = result.occupancy()
surface = result.surface_mesh()
```

## Target Workflows

`dds.targets` and `dds.formats.yaml` support target-driven workflows where each target represents the nozzle-tip or top-of-bead target.

```python
from dds import BeadProfile, Domain, simulate
from dds.formats.yaml import load_targets
from dds.targets import point_deposits_from_targets

targets = load_targets("example_wall.yaml")
profile = BeadProfile(width=18.0, height=12.0)
deposits = point_deposits_from_targets(targets, profile=profile, origin_reference="top")
domain = Domain.from_deposits(deposits, voxel_size=1.0, padding="auto")
result = simulate(domain, deposits, compositions=("max", "sum"), threshold=0.5)
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

## Headless Analysis Queries

`dds.analysis` provides cached, headless analysis over dense fields and derived surfaces. `AnalysisBundle` is the main analysis object, and `SimulationResult.analysis_bundle()` or `Simulator.analysis_bundle()` reuse it until inputs change.

```python
from dds import Simulator
from dds.analysis import sample_points, signed_distance_at

simulator = Simulator(domain, deposits)
bundle = simulator.analysis_bundle()

inside_density = bundle.sample_density_at((10.25, 10.25, 0.25), interpolation="trilinear")
inside_mesh = bundle.contains_point((10.25, 10.25, 0.25), representation="mesh", threshold=0.5)
surface_distance = signed_distance_at(bundle, (12.0, 10.25, 0.25), threshold=0.5)

samples = sample_points(
    bundle,
    [(10.25, 10.25, 0.25), (20.0, 20.0, 1.0)],
    fields=("density", "occupancy", "deposition_index", "signed_distance"),
    threshold=0.5,
    interpolation="trilinear",
)
```

Headless analysis API:

- `AnalysisBundle`, `analysis_bundle(...)`
- `contains_point(...)`
- `sample_density_at(...)`
- `sample_deposition_index_at(...)`
- `signed_distance_at(...)`
- `surface_normal_at(...)`
- `sample_points(...)`

The analysis layer intentionally returns NumPy values and geometry objects only. It does not create visualization datasets in this branch stage.

## Headless Mesh Analysis

`dds.mesh_analysis` and the matching `dds.geometry` re-exports provide pure triangle-mesh metrics without any viewer dependency.

```python
from dds.geometry import mesh_surface_area, overhang_angles

bundle = simulator.analysis_bundle()
analysis = bundle.mesh_analysis(build_direction=(0.0, 0.0, 1.0), critical_angle_deg=45.0)
stats = bundle.subvolume_stats(((0.0, 0.0, 0.0), (20.0, 20.0, 2.0)), threshold=0.5)

surface_area = mesh_surface_area(analysis["mesh"])
angles = overhang_angles(analysis["mesh"], build_direction=(0.0, 0.0, 1.0))
```

Headless mesh-analysis API:

- `face_normals(...)`, `vertex_normals(...)`
- `face_centroids(...)`, `face_areas(...)`
- `overhang_angles(...)`, `downfacing_mask(...)`, `support_risk_mask(...)`
- `normal_rgb_from_normals(...)`
- `mesh_bounds_stats(...)`, `mesh_surface_area(...)`, `mesh_volume_estimate(...)`
- `AnalysisBundle.mesh_analysis(...)`
- `AnalysisBundle.subvolume_stats(...)`

## Exporting Results

`dds.io` provides simple helpers for writing dense outputs to disk.

```python
from dds import Domain
from dds.io import save_array, save_simulation_bundle

domain = Domain.from_bounds(
    xmin=0.0,
    xmax=4.0,
    ymin=0.0,
    ymax=4.0,
    zmin=0.0,
    zmax=2.0,
    voxel_size=0.5,
)

written = save_simulation_bundle(
    "outputs/basic",
    domain=domain,
    occupancy=occupancy,
    deposition_index=deposition_index,
    metadata={"example": "basic_simulation"},
)

save_array("outputs/basic/raw_density.npy", deposition_index)
```

Bundle outputs:

- `occupancy.npy` when an occupancy field is provided
- `deposition_index.npy` when a deposition index field is provided
- `density.npy` when a density-like field is provided
- `density_sum.npy` when a result also carries accumulation density
- `metadata.json` containing serialized domain metadata and caller metadata

## Design Assumptions

- The import package is `dds`; the repository and distribution branding are `3DP-DDS` and `3dp-dds`.
- Dense array indexing follows `(x, y, z)` ordering via NumPy `indexing="ij"`.
- `width` is the full bead width in the local transverse plane, and `height` is the full bead height along the local bead axis.
- Point and line deposits are top-referenced: the target point represents the nozzle-tip or top-of-bead target, not the bead center.
- Point deposits use a rounded-bead kernel, and line deposits sweep the same rounded profile along the target path.
- `SimulationResult` always treats `density_max` as the canonical geometry field for occupancy, surface extraction, and signed-distance analysis.
- `density_sum` is optional and is intended for accumulation or overlap inspection, not geometry reconstruction.
- The deposition index is the per-voxel last-deposit index (0-based; −1 for untouched voxels).
- Occupancy is derived by thresholding the accumulated scalar field.

## Package Layout

```text
src/dds/
├── analysis/
│   ├── __init__.py
│   ├── bundle.py
│   ├── fields.py
│   ├── interface.py
│   ├── models.py
│   ├── strata.py
│   └── support.py
├── formats/
│   ├── __init__.py
│   └── yaml.py
├── geometry/
│   ├── __init__.py
│   ├── adapters.py
│   ├── mesh.py
│   ├── ops.py
│   ├── shapes.py
│   ├── sdf.py
│   └── transforms.py
├── __init__.py
├── attributes.py
├── cli.py
├── domain.py
├── fields.py
├── io.py
├── kernels.py
├── mesh_analysis.py
├── occupancy.py
├── primitives.py
├── results.py
├── simulator.py
├── targets.py
├── types.py
├── utils.py
├── viz.py
└── workbench.py
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

Write the example outputs to disk:

```bash
python examples/basic_simulation.py --output-dir outputs/basic
```

The example creates a simple deposition scene, prints summary metrics, and can write `occupancy.npy`, `deposition_index.npy`, and `metadata.json` without pulling in visualization features.
