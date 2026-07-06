from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import dds.viz
from dds import (
    BeadProfile,
    DepositionTarget,
    Domain,
    LineDeposit,
    PointDeposit,
    PolylineDeposit,
    Simulator,
)
from dds.analysis import occupancy_fraction
from dds.cli import run_cli


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


def build_example_deposits() -> list[PointDeposit | LineDeposit | PolylineDeposit]:
    profile = BeadProfile(width=1.2, height=0.6)
    vertical_normal = (0.0, 0.0, 1.0)
    tilted_x_normal = (0.4, 0.0, 0.916515)
    tilted_y_normal = (-0.25, 0.35, 0.902)
    return [
        PointDeposit(
            target=DepositionTarget((2.25, 2.25, 0.55), vertical_normal),
            profile=profile,
        ),
        LineDeposit(
            start=DepositionTarget((4.25, 2.25, 0.55), vertical_normal),
            end=DepositionTarget((4.25, 2.25, 5.55), vertical_normal),
            profile=profile,
        ),
        LineDeposit(
            start=DepositionTarget((6.25, 2.25, 0.55), vertical_normal),
            end=DepositionTarget((14.25, 2.25, 0.55), vertical_normal),
            profile=profile,
        ),
        LineDeposit(
            start=DepositionTarget((6.25, 5.25, 0.75), vertical_normal),
            end=DepositionTarget((14.25, 5.25, 2.75), tilted_x_normal),
            profile=profile,
        ),
        PolylineDeposit(
            targets=(
                DepositionTarget((2.25, 8.25, 0.55), vertical_normal),
                DepositionTarget((2.25, 8.25, 4.55), vertical_normal),
                DepositionTarget((10.25, 8.25, 4.55), vertical_normal),
                DepositionTarget((10.25, 12.25, 4.55), vertical_normal),
            ),
            profile=profile,
        ),
        PolylineDeposit(
            targets=(
                DepositionTarget((13.25, 9.25, 0.75), vertical_normal),
                DepositionTarget((13.25, 9.25, 4.75), tilted_x_normal),
                DepositionTarget((17.25, 13.25, 4.25), tilted_y_normal),
            ),
            profile=profile,
        ),
    ]


@dataclass
class Args:
    """Run a basic dds simulation. Optionally save dense outputs or inspect them interactively."""

    threshold: float = 0.5
    output_dir: Path | None = None
    view: bool = False
    view_mode: Literal["surface", "occupancy", "implicit"] = "surface"


def main(args: Args) -> None:
    domain = build_example_domain()
    deposits = build_example_deposits()
    simulator = Simulator(domain, deposits)
    result = simulator.result(include_coverage=True, threshold=args.threshold)

    occupancy = result.analysis.occupancy(threshold=args.threshold)
    deposition_index = result.analysis.deposition_index_field()

    print(f"Grid shape: {occupancy.shape}")
    print(f"Occupied voxels: {int(occupancy.sum())}")
    print(f"Occupancy fraction: {occupancy_fraction(occupancy):.4f}")
    print(f"Max deposition index: {float(deposition_index.max()):.4f}")

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
