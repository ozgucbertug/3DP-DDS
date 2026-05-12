from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dds import BeadProfile, DepositionMetadata, Domain, SimulationResult, run_cli, simulate
from dds.analysis import occupancy_fraction
from dds.formats.yaml import load_targets
from dds.targets import point_deposits_from_targets

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class YamlSimulationConfig:
    """Configuration for running a target-driven YAML deposition simulation."""

    yaml_path: Path = ROOT / "lamine_curvedwall.yaml"
    output_dir: Path = ROOT / "outputs" / "yaml_lamine_curvedwall"
    voxel_size: float = 1.0
    bead_width: float = 18.0
    bead_height: float = 12.0
    threshold: float = 0.5
    padding: float | None = None
    origin_reference: Literal["top", "center"] = "top"
    density_composition: Literal["max", "sum"] = "max"
    write_mesh_output: bool = True
    mesh_step_size: int = 1
    view: bool = False
    view_mode: Literal["surface", "occupancy", "density"] = "surface"


def run_simulation(config: YamlSimulationConfig | None = None) -> SimulationResult:
    """Run the YAML target workflow through the public dds API."""

    config = config or YamlSimulationConfig()
    profile = BeadProfile(width=config.bead_width, height=config.bead_height)
    metadata = DepositionMetadata()
    targets = load_targets(config.yaml_path)
    deposits = point_deposits_from_targets(
        targets,
        profile=profile,
        metadata=metadata,
        origin_reference=config.origin_reference,
    )
    domain = Domain.from_deposits(
        deposits,
        voxel_size=config.voxel_size,
        padding="auto" if config.padding is None else config.padding,
    )
    result = simulate(
        domain,
        deposits,
        compositions=("max", "sum"),
        threshold=config.threshold,
    )

    occupancy = result.occupancy(threshold=config.threshold)
    deposition_index = result.analysis_bundle().deposition_index_field()

    print(f"Loaded targets: {len(targets)}")
    print(f"Created point deposits: {len(deposits)}")
    print(f"Bead width: {config.bead_width}")
    print(f"Bead height: {config.bead_height}")
    print(f"Domain min/max: {domain.min_corner} -> {domain.max_corner}")
    print(f"Grid shape: {domain.grid_shape}")
    print(f"Occupied voxels: {int(occupancy.sum())}")
    print(f"Occupancy fraction: {occupancy_fraction(occupancy):.4f}")
    print(f"Max deposition index: {float(deposition_index.max()):.4f}")
    print(f"Max envelope density: {float(result.density_max.max()):.4f}")
    if result.density_sum is not None:
        print(f"Max accumulation density: {float(result.density_sum.max()):.4f}")

    # written = result.save(
    #     config.output_dir,
    #     metadata={
    #         "example": "yaml_simulation",
    #         "yaml_path": config.yaml_path,
    #         "target_count": len(targets),
    #         "deposit_count": len(deposits),
    #         "bead_width": config.bead_width,
    #         "bead_height": config.bead_height,
    #         "origin_reference": config.origin_reference,
    #         "threshold": config.threshold,
    #         "density_composition": config.density_composition,
    #     },
    # )
    # if config.write_mesh_output:
    #     mesh = result.surface_mesh(threshold=config.threshold, step_size=config.mesh_step_size)
    #     write_mesh(config.output_dir / "surface_mesh.ply", mesh)

    if config.view:
        workbench = result.show(view_mode=config.view_mode, off_screen=False)
        if config.view_mode == "density":
            workbench.set_density_composition(config.density_composition)
        workbench.app.exec()

    return result


def main(config: YamlSimulationConfig) -> None:
    run_simulation(config)


if __name__ == "__main__":
    run_cli(YamlSimulationConfig, main)
