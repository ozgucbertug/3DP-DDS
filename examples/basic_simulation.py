from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import dds.viz
from dds import (
    BeadProfile,
    DepositionMetadata,
    Domain,
    LineDeposit,
    PointDeposit,
    Simulator,
    run_cli,
)
from dds.analysis import occupancy_fraction, summarize_layers


def build_example_domain() -> Domain:
    return Domain.from_bounds(
        xmin=0.0,
        xmax=20.0,
        ymin=0.0,
        ymax=20.0,
        zmin=-1.0,
        zmax=6.0,
        voxel_size=0.1,
    )


def build_example_deposits() -> list[PointDeposit | LineDeposit]:
    profile = BeadProfile(width=1.2, height=0.6)
    metadata = DepositionMetadata(layer_id=0)
    return [
        PointDeposit(
            x=2.25,
            y=2.25,
            z=0.55,
            profile=profile,
            metadata=metadata,
        ),
        LineDeposit(
            start=(2.25, 2.25, 0.55),
            end=(10.25, 2.25, 0.55),
            profile=profile,
            metadata=metadata,
        ),
        LineDeposit(
            start=(10.25, 2.25, 0.55),
            end=(10.25, 8.25, 0.55),
            profile=profile,
            metadata=metadata,
        ),
    ]


@dataclass
class Args:
    """Run a basic dds simulation. Optionally save dense outputs or inspect them interactively."""

    threshold: float = 0.5
    output_dir: Path | None = None
    view: bool = False
    view_mode: Literal["surface", "occupancy", "density"] = "surface"


def main(args: Args) -> None:
    domain = build_example_domain()
    deposits = build_example_deposits()
    simulator = Simulator(domain, deposits)
    result = simulator.result(compositions=("max", "coverage"), threshold=args.threshold)

    occupancy = result.occupancy(threshold=args.threshold)
    deposition_index = result.analysis_bundle().deposition_index_field()

    print(f"Grid shape: {occupancy.shape}")
    print(f"Occupied voxels: {int(occupancy.sum())}")
    print(f"Occupancy fraction: {occupancy_fraction(occupancy):.4f}")
    print(f"Max deposition index: {float(deposition_index.max()):.4f}")
    print(f"Layer summary: {summarize_layers(deposits)}")

    if args.output_dir is not None:
        written = result.save(
            args.output_dir,
            metadata={"example": "basic_simulation", "threshold": args.threshold},
        )
        for label, path in written.items():
            print(f"Saved {label}: {path}")

    if args.view:
        dds.viz.show(
            result,
            view_mode=args.view_mode,
            off_screen=False,
        ).app.exec()


if __name__ == "__main__":
    run_cli(Args, main)
