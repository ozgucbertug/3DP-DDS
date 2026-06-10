from __future__ import annotations

from PySide6 import QtCore

import dds.viz
from dds import BeadProfile, Domain, PointDeposit, Simulator
from dds.viz import ViewConfig


def main() -> None:
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
    simulator = Simulator(domain)

    positions = [
        (2.0, 2.0, 0.6),
        (4.0, 2.0, 0.6),
        (6.0, 2.0, 0.6),
        (8.0, 2.0, 0.6),
        (2.0, 4.0, 0.6),
        (4.0, 4.0, 0.6),
        (6.0, 4.0, 0.6),
        (8.0, 4.0, 0.6),
    ]
    batch_size = 2
    next_position = 0

    workbench = dds.viz.show(
        simulator,
        threshold=0.5,
        initial_view=ViewConfig(view_mode="surface"),
    )

    timer = QtCore.QTimer(workbench)

    def add_next_batch() -> None:
        nonlocal next_position

        batch_positions = positions[next_position : next_position + batch_size]
        if not batch_positions:
            timer.stop()
            return

        simulator.add_deposits(
            PointDeposit(target=position, profile=profile)
            for position in batch_positions
        )
        next_position += len(batch_positions)
        workbench.refresh(simulator)
        print(f"Displayed {len(simulator.deposits)} beads")

    timer.timeout.connect(add_next_batch)
    timer.start(750)
    workbench.app.exec()


if __name__ == "__main__":
    main()
