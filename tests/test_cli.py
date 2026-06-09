from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from dds.cli import parse_cli, run_cli

ROOT = Path(__file__).resolve().parents[1]


def source_environment() -> dict[str, str]:
    env = os.environ.copy()
    source_path = str(ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_path, env.get("PYTHONPATH")) if value
    )
    return env


@dataclass
class ExampleArgs:
    count: int = 1
    label: str = "default"
    enabled: bool = False


def test_parse_cli_returns_typed_config() -> None:
    args = parse_cli(ExampleArgs, argv=["--count", "3", "--label", "demo", "--enabled"])

    assert args == ExampleArgs(count=3, label="demo", enabled=True)


def test_run_cli_dispatches_handler() -> None:
    captured: list[ExampleArgs] = []

    def handler(args: ExampleArgs) -> None:
        captured.append(args)

    run_cli(ExampleArgs, handler, argv=["--count", "2"])

    assert captured == [ExampleArgs(count=2)]


def test_basic_simulation_example_exposes_tyro_help() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "basic_simulation.py"), "--help"],
        cwd=ROOT,
        env=source_environment(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--threshold" in result.stdout
    assert "--view" in result.stdout
    assert "--view-mode {surface,occupancy,density}" in result.stdout
    assert "slice" not in result.stdout
    assert "Run a basic dds simulation." in result.stdout
