from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dds import (
    BeadProfile,
    DepositionMetadata,
    Domain,
    SimulationResult,
    Simulator,
    WorkbenchViewConfig,
    run_cli,
)
import dds.viz
from dds.analysis import occupancy_fraction
from dds.formats.yaml import load_targets
from dds.targets import point_deposits_from_targets

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class YamlSimulationConfig:
    """Configuration for running a target-driven YAML deposition simulation."""

    yaml_path: Path = ROOT / "example_wall.yaml"
    output_dir: Path = ROOT / "outputs" / "yaml_example_wall"
    voxel_size: float = 1.0
    bead_width: float = 18.0
    bead_height: float = 12.0
    threshold: float = 0.5
    padding: float | None = None
    origin_reference: Literal["top", "center"] = "top"
    field_composition: Literal["max", "coverage"] = "max"
    analysis: Literal["none", "interface", "support", "all"] = "none"
    stratification: Literal["auto", "layer", "order"] = "auto"
    build_direction: Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"] = "+Z"
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
    simulator = Simulator(domain, deposits)
    result = simulator.result(compositions=("max", "coverage"), threshold=config.threshold)

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
    if result.coverage is not None:
        print(f"Max nonphysical coverage: {float(result.coverage.max()):.4f}")

    if config.analysis in {"interface", "all"}:
        interface_analysis = result.interface(mode=config.stratification, threshold=config.threshold)
        print(f"Interface stratification: {interface_analysis.stratification_mode}")
        print(f"Interface strata: {interface_analysis.stratum_ids}")
        print(f"Contact area: {interface_analysis.contact_area:.4f}")
        print(f"Contact face count: {interface_analysis.contact_face_count}")
        print(f"Overlap voxels: {interface_analysis.overlap_voxel_count}")
        print(f"Overlap fraction: {interface_analysis.overlap_fraction:.4f}")
        print(f"Unsupported next voxels: {int(interface_analysis.unsupported_next_mask.sum())}")

    if config.analysis in {"support", "all"}:
        build_direction = {
            "+X": (1.0, 0.0, 0.0),
            "-X": (-1.0, 0.0, 0.0),
            "+Y": (0.0, 1.0, 0.0),
            "-Y": (0.0, -1.0, 0.0),
            "+Z": (0.0, 0.0, 1.0),
            "-Z": (0.0, 0.0, -1.0),
        }[config.build_direction]
        support_analysis = result.support(build_direction=build_direction, threshold=config.threshold)
        print(f"Support build direction: {config.build_direction}")
        print(f"Downfacing area: {support_analysis.downfacing_area:.4f}")
        print(f"Risk area: {support_analysis.risk_area:.4f}")
        print(f"Shadow voxels: {support_analysis.shadow_voxel_count}")
        print(f"Shadow volume: {support_analysis.shadow_volume:.4f}")
        print(f"Max unsupported span: {support_analysis.max_unsupported_span:.4f}")

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
    #         "field_composition": config.field_composition,
    #     },
    # )
    # if config.write_mesh_output:
    #     mesh = result.surface_mesh(threshold=config.threshold, step_size=config.mesh_step_size)
    #     write_mesh(config.output_dir / "surface_mesh.ply", mesh)

    if config.view:
        initial_scalar_field = None
        initial_color_mode = None
        if config.view_mode == "density":
            if config.analysis in {"interface", "all"} or config.field_composition == "coverage":
                initial_scalar_field = "coverage"
            else:
                initial_scalar_field = "density"
        elif config.view_mode == "occupancy":
            initial_scalar_field = "occupancy"
        elif config.view_mode == "surface" and config.analysis in {"support", "all"}:
            initial_color_mode = "overhang"

        workbench = dds.viz.show(
            result,
            initial_view=WorkbenchViewConfig(
                view_mode=config.view_mode,
                scalar_field=initial_scalar_field,
                color_mode=initial_color_mode,
                build_direction=config.build_direction,
            ),
            off_screen=False,
        )
        workbench.app.exec()

    return result


def main(config: YamlSimulationConfig) -> None:
    run_simulation(config)


if __name__ == "__main__":
    run_cli(YamlSimulationConfig, main)
