from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dds import BeadProfile, Domain, PointDeposit, Simulator
from dds.cli import run_cli
from dds.formats.yaml import load_targets
from dds.targets import point_deposits_from_targets

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class LiveSimulationConfig:
    """Interactively advance a YAML target sequence in the visualization workbench."""

    yaml_path: Path = ROOT / "example_wall.yaml"
    voxel_size: float = 1.0
    bead_width: float = 18.0
    bead_height: float = 12.0
    threshold: float = 0.5
    padding: float | None = None
    origin_reference: Literal["top", "center"] = "top"
    step_size: int = 1
    advance_key: str = "space"
    reset_key: str = "r"
    view_mode: Literal["surface", "occupancy", "implicit"] = "surface"


@dataclass
class DepositionStepper:
    """Advance a fixed deposit sequence into a mutable simulator."""

    simulator: Simulator
    deposits: tuple[PointDeposit, ...]
    step_size: int = 1
    next_index: int = 0

    def __post_init__(self) -> None:
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")

    @property
    def complete(self) -> bool:
        return self.next_index >= len(self.deposits)

    def advance(self) -> int:
        """Add the next batch and return the number of deposits added."""

        batch = self.deposits[
            self.next_index : self.next_index + self.step_size
        ]
        if not batch:
            return 0
        self.simulator.add_deposits(batch)
        self.next_index += len(batch)
        return len(batch)

    def reset(self) -> None:
        """Clear the simulator and rewind the target sequence."""

        self.simulator.clear_deposits()
        self.next_index = 0


def build_live_simulation(
    config: LiveSimulationConfig,
) -> tuple[Domain, DepositionStepper]:
    """Load YAML targets and construct an empty simulator plus stepper."""

    profile = BeadProfile(
        width=config.bead_width,
        height=config.bead_height,
    )
    targets = load_targets(config.yaml_path)
    deposits = point_deposits_from_targets(
        targets,
        profile=profile,
        origin_reference=config.origin_reference,
    )
    domain = Domain.from_deposits(
        deposits,
        voxel_size=config.voxel_size,
        padding="auto" if config.padding is None else config.padding,
    )
    return domain, DepositionStepper(
        simulator=Simulator(domain),
        deposits=deposits,
        step_size=config.step_size,
    )


def run_live_simulation(config: LiveSimulationConfig) -> None:
    """Open the workbench and bind deposition advancement to keyboard input."""

    import dds.viz
    from dds.viz import ViewConfig

    domain, stepper = build_live_simulation(config)
    workbench = dds.viz.show(
        stepper.simulator,
        threshold=config.threshold,
        initial_view=ViewConfig(view_mode=config.view_mode),
    )

    def refresh() -> None:
        workbench.refresh(stepper.simulator)

    def advance() -> None:
        added = stepper.advance()
        if added == 0:
            print("Deposition sequence is complete.")
            return
        refresh()
        print(
            f"Displayed {stepper.next_index}/{len(stepper.deposits)} "
            "YAML targets."
        )

    def reset() -> None:
        stepper.reset()
        refresh()
        print("Deposition sequence reset.")

    workbench.plotter.add_key_event(config.advance_key, advance)
    workbench.plotter.add_key_event(config.reset_key, reset)

    print(f"Loaded {len(stepper.deposits)} targets from {config.yaml_path}")
    print(f"Domain: {domain.min_corner} -> {domain.max_corner}")
    print(
        f"Press {config.advance_key!r} to add {config.step_size} target(s); "
        f"press {config.reset_key!r} to reset."
    )
    workbench.app.exec()


def main(config: LiveSimulationConfig) -> None:
    run_live_simulation(config)


if __name__ == "__main__":
    run_cli(LiveSimulationConfig, main)
