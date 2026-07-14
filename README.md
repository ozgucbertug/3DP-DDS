# 3DP-DDS

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://ozgucbertug.github.io/3DP-DDS/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)](pyproject.toml)
[![License](https://img.shields.io/github/license/ozgucbertug/3DP-DDS)](LICENSE)
[![Typing](https://img.shields.io/badge/typing-py.typed-blue)](src/dds/py.typed)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange)](pyproject.toml)

`3DP-DDS` is a Python library for geometry-first deposition simulation on a
3D voxel grid. The import package is `dds`.

It represents robotic additive-manufacturing paths as point, line, and
polyline deposition events, samples bead geometry into dense or chunked
fields, and provides analysis, persistence, mesh conversion, YAML workflows,
and optional interactive visualization.

The current scope is deposited geometry. Material flow, thermal history,
curing, robot dynamics, controller behavior, and bead deformation are
intentionally outside the model.

**Documentation:** [ozgucbertug.github.io/3DP-DDS](https://ozgucbertug.github.io/3DP-DDS/)

## Installation

3DP-DDS requires Python 3.11 or newer. From a local clone, install the core
library in editable mode:

```bash
python -m pip install -e .
```

The core dependencies are NumPy, SciPy, and Tyro. Optional capabilities are
installed through extras:

| Extra | Adds | Install |
| --- | --- | --- |
| `formats` | YAML target loading through PyYAML | `python -m pip install -e ".[formats]"` |
| `mesh` | Trimesh-backed mesh and point-cloud I/O, extraction, containment, and signed-distance operations | `python -m pip install -e ".[mesh]"` |
| `viz` | PyVistaQt interactive workbench and mesh dependencies | `python -m pip install -e ".[viz]"` |
| `all` | All `formats`, `mesh`, and `viz` capabilities | `python -m pip install -e ".[all]"` |

## Quick Start

```python
from dds import BeadProfile, Domain, LineDeposit, PointDeposit, simulate

profile = BeadProfile(width=1.2, height=0.6)
deposits = [
    PointDeposit(target=(2.0, 2.0, 0.6), profile=profile),
    LineDeposit(
        start=(2.0, 2.0, 0.6),
        end=(10.0, 2.0, 0.6),
        profile=profile,
    ),
]

domain = Domain.from_deposits(deposits, voxel_size=0.25, padding="auto")
result = simulate(domain, deposits, threshold=0.5)

occupancy = result.analysis.occupancy()
deposition_index = result.analysis.deposition_index_field()

print(domain.grid_shape)
print(int(occupancy.sum()))
print(float(deposition_index.max()))
```

Deposit targets are top-referenced: a target is the nozzle position or bead
top surface, not the bead center. Arrays use `(x, y, z)` index order.

## What Is Included

- Domain and bead-profile modeling.
- Point, line, and polyline deposition primitives.
- Dense one-shot and stateful incremental simulation.
- Chunked accumulation for sparse large workspaces.
- Occupancy, deposition order, SDF, strata, interface, and support analysis.
- Result persistence and checkpointing.
- Optional YAML target loading, mesh/SDF utilities, and interactive
  visualization.

## Examples

Runnable examples live in [examples](examples/):

- [basic_simulation.py](examples/basic_simulation.py): core domain, deposits,
  simulation, analysis summary, and optional output.
- [yaml_simulation.py](examples/yaml_simulation.py): target-driven YAML
  workflow with optional analysis, mesh output, and visualization.
- [live_simulation.py](examples/live_simulation.py): interactive step-through
  deposition workflow for YAML targets.

Detailed tutorials and generated API pages are available in the
[documentation](https://ozgucbertug.github.io/3DP-DDS/).

## Project Status

3DP-DDS is pre-alpha research software. APIs, checkpoint schemas, and
documentation structure may change before a stable release.

## Citation

If you use 3DP-DDS in research, cite the repository and the associated
publication when available. Citation metadata is provided in [CITATION.cff](CITATION.cff).

## License

3DP-DDS is distributed under the [MIT License](LICENSE).
