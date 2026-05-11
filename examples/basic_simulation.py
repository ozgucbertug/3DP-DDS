from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dds import (  # noqa: E402
    DepositionAttributes,
    Domain,
    LineDeposit,
    PointDeposit,
    Simulator,
)
from dds.analysis import summarize_layers  # noqa: E402
from dds.cli import run_cli  # noqa: E402
from dds.occupancy import occupancy_fraction  # noqa: E402


def build_example_domain() -> Domain:
    return Domain.from_bounds(
        xmin=0.0,
        xmax=20.0,
        ymin=0.0,
        ymax=20.0,
        zmin=0.0,
        zmax=6.0,
        voxel_size=0.1,
    )


def build_example_deposits() -> list[PointDeposit | LineDeposit]:
    attrs = DepositionAttributes(width=1.2, height=0.6, layer_id=0, material_id="mat0", tool_id="tool0")
    return [
        PointDeposit(x=2.25, y=2.25, z=0.25, attributes=attrs),
        LineDeposit(start=(2.25, 2.25, 0.25), end=(10.25, 2.25, 0.25), attributes=attrs),
        LineDeposit(start=(10.25, 2.25, 0.25), end=(10.25, 8.25, 0.25), attributes=attrs),
    ]


@dataclass
class Args:
    """Run a basic dds simulation."""

    threshold: float = 0.5


def main(args: Args) -> None:

    domain = build_example_domain()
    deposits = build_example_deposits()
    simulator = Simulator(domain, deposits)

    occupancy = simulator.simulate_occupancy(threshold=args.threshold)
    density = simulator.simulate_deposition_index()

    print(f"Grid shape: {occupancy.shape}")
    print(f"Occupied voxels: {int(occupancy.sum())}")
    print(f"Occupancy fraction: {occupancy_fraction(occupancy):.4f}")
    print(f"Max deposition index: {float(density.max()):.4f}")
    print(f"Layer summary: {summarize_layers(deposits)}")

if __name__ == "__main__":
    run_cli(Args, main)
