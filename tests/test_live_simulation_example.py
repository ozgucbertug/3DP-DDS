from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

import dds.viz

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT / "examples" / "live_simulation.py"


def source_environment() -> dict[str, str]:
    env = os.environ.copy()
    source_path = str(ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_path, env.get("PYTHONPATH")) if value
    )
    return env


def load_example_module() -> object:
    spec = importlib.util.spec_from_file_location(
        "live_simulation_example",
        EXAMPLE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load live_simulation example module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_live_simulation_example_exposes_typed_help() -> None:
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH), "--help"],
        cwd=ROOT,
        env=source_environment(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--yaml-path" in result.stdout
    assert "--step-size" in result.stdout
    assert "--advance-key" in result.stdout
    assert "--reset-key" in result.stdout
    assert "--view-mode {surface,occupancy,implicit}" in result.stdout


def test_live_stepper_advances_yaml_deposits_and_resets(tmp_path: Path) -> None:
    example = load_example_module()
    yaml_path = tmp_path / "targets.yaml"
    yaml_path.write_text(
        """
targets:
  - index: 0
    origin: [0, 0, 1]
  - index: 1
    origin: [1, 0, 1]
  - index: 2
    origin: [2, 0, 1]
""".strip(),
        encoding="utf-8",
    )
    config = example.LiveSimulationConfig(
        yaml_path=yaml_path,
        voxel_size=0.5,
        bead_width=1.0,
        bead_height=0.5,
        step_size=2,
    )

    domain, stepper = example.build_live_simulation(config)

    assert domain.grid_shape[0] > 0
    assert len(stepper.deposits) == 3
    assert stepper.simulator.deposits == ()
    assert stepper.advance() == 2
    assert len(stepper.simulator.deposits) == 2
    assert stepper.advance() == 1
    assert stepper.complete
    assert stepper.advance() == 0

    stepper.reset()
    assert not stepper.complete
    assert stepper.next_index == 0
    assert stepper.simulator.deposits == ()


def test_live_stepper_rejects_nonpositive_step_size(tmp_path: Path) -> None:
    example = load_example_module()
    with pytest.raises(ValueError, match="positive"):
        example.DepositionStepper(
            simulator=example.Simulator(
                example.Domain.from_bounds(
                    xmin=0.0,
                    xmax=1.0,
                    ymin=0.0,
                    ymax=1.0,
                    zmin=0.0,
                    zmax=1.0,
                    voxel_size=1.0,
                )
            ),
            deposits=(),
            step_size=0,
        )


def test_live_run_registers_advance_and_reset_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = load_example_module()
    yaml_path = tmp_path / "targets.yaml"
    yaml_path.write_text(
        """
targets:
  - index: 0
    origin: [0, 0, 1]
  - index: 1
    origin: [1, 0, 1]
""".strip(),
        encoding="utf-8",
    )
    callbacks: dict[str, object] = {}
    refresh_counts: list[int] = []

    class FakePlotter:
        def add_key_event(self, key: str, callback: object) -> None:
            callbacks[key] = callback

    class FakeApp:
        def exec(self) -> None:
            return None

    class FakeWorkbench:
        plotter = FakePlotter()
        app = FakeApp()

        def refresh(self, simulator: object) -> None:
            refresh_counts.append(len(simulator.deposits))  # type: ignore[attr-defined]

    monkeypatch.setattr(
        dds.viz,
        "show",
        lambda *args, **kwargs: FakeWorkbench(),
    )
    config = example.LiveSimulationConfig(
        yaml_path=yaml_path,
        voxel_size=0.5,
        bead_width=1.0,
        bead_height=0.5,
        advance_key="space",
        reset_key="r",
    )

    example.run_live_simulation(config)

    assert set(callbacks) == {"space", "r"}
    callbacks["space"]()  # type: ignore[operator]
    callbacks["r"]()  # type: ignore[operator]
    assert refresh_counts == [1, 0]
