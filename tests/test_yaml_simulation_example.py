from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dds import SimulationResult

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT / "examples" / "yaml_simulation.py"


def source_environment() -> dict[str, str]:
    env = os.environ.copy()
    source_path = str(ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_path, env.get("PYTHONPATH")) if value
    )
    return env


def load_example_module() -> object:
    spec = importlib.util.spec_from_file_location("yaml_simulation_example", EXAMPLE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load yaml_simulation example module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_yaml_simulation_example_exposes_tyro_help() -> None:
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
    assert "--origin-reference" in result.stdout
    assert "--field-composition {max,coverage}" in result.stdout
    assert "--analysis {none,interface,support,all}" in result.stdout
    assert "--stratification {auto,layer,order}" in result.stdout
    assert "--build-direction {+X,-X,+Y,-Y,+Z,-Z}" in result.stdout
    assert "--view" in result.stdout
    assert "--view-mode {surface,occupancy,density}" in result.stdout
    assert "slice" not in result.stdout


def test_yaml_simulation_config_is_ide_friendly() -> None:
    example = load_example_module()
    config = example.YamlSimulationConfig(view=False)

    assert config.yaml_path.name == "example_wall.yaml"
    assert config.origin_reference == "top"
    assert config.field_composition == "max"
    assert config.analysis == "none"
    assert config.stratification == "auto"
    assert config.build_direction == "+Z"
    assert config.bead_width == pytest.approx(18.0)
    assert config.bead_height == pytest.approx(12.0)
    assert config.output_dir is None
    assert config.write_mesh_output is False
    assert config.view is False


def test_yaml_example_runs_through_public_api_and_returns_result(tmp_path: Path) -> None:
    example = load_example_module()
    yaml_path = tmp_path / "targets.yaml"
    yaml_path.write_text(
        """
targets:
  - index: 0
    plane: "O(0,0,6) Z(0,0,1)"
""".strip(),
        encoding="utf-8",
    )

    result = example.run_simulation(
        example.YamlSimulationConfig(
            yaml_path=yaml_path,
            output_dir=tmp_path / "out",
            voxel_size=1.0,
            bead_width=18.0,
            bead_height=12.0,
            analysis="interface",
            build_direction="+Z",
            field_composition="coverage",
            write_mesh_output=False,
            view=False,
        )
    )

    assert isinstance(result, SimulationResult)
    assert result.analysis.occupancy(threshold=0.5).any()
    assert result.analysis.deposition_index_field().max() >= 0  # at least one deposit has index ≥ 0
    assert result.density_max.max() >= 0.5
    assert result.coverage is not None
    assert result.coverage.shape == result.domain.grid_shape
    assert np.all(result.coverage >= result.density_max)
    assert (tmp_path / "out" / "density.npy").exists()
    assert (tmp_path / "out" / "coverage.npy").exists()
    assert (tmp_path / "out" / "metadata.json").exists()


def test_yaml_example_keeps_coverage_distinct_from_max_for_overlap(tmp_path: Path) -> None:
    example = load_example_module()
    yaml_path = tmp_path / "targets.yaml"
    yaml_path.write_text(
        """
targets:
  - index: 0
    plane: "O(0,0,6) Z(0,0,1)"
  - index: 1
    plane: "O(0,0,6) Z(0,0,1)"
""".strip(),
        encoding="utf-8",
    )

    result = example.run_simulation(
        example.YamlSimulationConfig(
            yaml_path=yaml_path,
            output_dir=tmp_path / "out",
            voxel_size=1.0,
            bead_width=18.0,
            bead_height=12.0,
            analysis="interface",
            build_direction="+Z",
            field_composition="coverage",
            write_mesh_output=False,
            view=False,
        )
    )

    assert result.coverage is not None
    assert np.any(result.coverage > result.density_max)


def test_yaml_example_support_analysis_runs(tmp_path: Path) -> None:
    example = load_example_module()
    yaml_path = tmp_path / "targets.yaml"
    yaml_path.write_text(
        """
targets:
  - index: 0
    plane: "O(0,0,6) Z(0,0,1)"
  - index: 1
    plane: "O(4,0,6) Z(0,0,1)"
""".strip(),
        encoding="utf-8",
    )

    result = example.run_simulation(
        example.YamlSimulationConfig(
            yaml_path=yaml_path,
            output_dir=tmp_path / "out",
            voxel_size=1.0,
            bead_width=18.0,
            bead_height=12.0,
            analysis="support",
            build_direction="+Z",
            write_mesh_output=False,
            view=False,
        )
    )

    support = result.analysis.support(build_direction="+Z", threshold=0.5)
    assert support.support_shadow_field.shape == result.domain.grid_shape
