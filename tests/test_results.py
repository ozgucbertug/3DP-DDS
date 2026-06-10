from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from dds import BeadProfile, DepositionMetadata, Domain, PointDeposit, SimulationResult, Simulator, simulate


def test_root_import_does_not_load_optional_visualization_modules() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import dds; "
                "assert 'dds.viz' not in sys.modules; "
                "assert 'dds.workbench' not in sys.modules; "
                "assert 'pyvista' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
    )

    assert completed.returncode == 0, completed.stderr


def make_domain() -> Domain:
    return Domain.from_bounds(
        xmin=0.0,
        xmax=12.0,
        ymin=0.0,
        ymax=12.0,
        zmin=0.0,
        zmax=8.0,
        voxel_size=1.0,
    )


def make_profile() -> BeadProfile:
    return BeadProfile(width=2.0, height=2.0)


def test_simulation_result_strata_prefers_real_layers_in_auto_mode() -> None:
    domain = make_domain()
    profile = make_profile()
    deposits = [
        PointDeposit(x=2.5, y=2.5, z=2.5, profile=profile, metadata=DepositionMetadata(layer_id=0)),
        PointDeposit(x=4.5, y=2.5, z=2.5, profile=profile, metadata=DepositionMetadata(layer_id=0)),
        PointDeposit(x=2.5, y=2.5, z=4.5, profile=profile, metadata=DepositionMetadata(layer_id=1)),
    ]
    result = simulate(domain, deposits, threshold=0.5)

    field_set = result.analysis.strata(mode="auto", threshold=0.5)

    assert field_set.mode == "layer"
    assert field_set.stratum_ids == (0, 1)
    assert field_set.density(0).shape == domain.grid_shape
    assert field_set.occupancy(1).dtype == np.bool_
    assert np.max(field_set.label_field) == pytest.approx(2.0)


def test_simulation_result_strata_falls_back_to_order_without_real_layers() -> None:
    domain = make_domain()
    profile = make_profile()
    deposits = [
        PointDeposit(x=2.5, y=2.5, z=2.5, profile=profile, metadata=DepositionMetadata()),
        PointDeposit(x=4.5, y=2.5, z=2.5, profile=profile, metadata=DepositionMetadata()),
    ]
    result = simulate(domain, deposits, threshold=0.5)

    field_set = result.analysis.strata(mode="auto", threshold=0.5)

    assert field_set.mode == "order"
    assert field_set.stratum_ids == (0, 1)
    assert np.max(field_set.label_field) == pytest.approx(2.0)


def test_layer_density_and_occupancy_require_real_layer_ids() -> None:
    domain = make_domain()
    profile = make_profile()
    deposits = [
        PointDeposit(x=2.5, y=2.5, z=2.5, profile=profile, metadata=DepositionMetadata(layer_id=3)),
    ]
    result = simulate(domain, deposits, threshold=0.5)

    field_set = result.analysis.strata(mode="layer", threshold=0.5)
    density = field_set.density(3)
    occupancy = field_set.occupancy(3)

    assert density.shape == domain.grid_shape
    assert occupancy.dtype == np.bool_

    missing_layer_result = simulate(
        domain,
        [PointDeposit(x=2.5, y=2.5, z=2.5, profile=profile, metadata=DepositionMetadata())],
        threshold=0.5,
    )
    with pytest.raises(ValueError):
        missing_layer_result.analysis.strata(mode="layer", threshold=0.5)


def test_simulator_result_requests_coverage() -> None:
    domain = make_domain()
    profile = make_profile()
    simulator = Simulator(
        domain,
        [
            PointDeposit(x=2.5, y=2.5, z=2.5, profile=profile, metadata=DepositionMetadata()),
            PointDeposit(x=3.0, y=2.5, z=2.5, profile=profile, metadata=DepositionMetadata()),
        ],
    )

    result = simulator.result(threshold=0.5, compositions=("max", "coverage"))

    assert result.coverage is not None
    assert result.coverage.shape == domain.grid_shape
    assert np.all(result.coverage >= result.density_max)


def test_simulation_result_is_an_immutable_snapshot() -> None:
    density = np.zeros(make_domain().grid_shape, dtype=float)
    result = SimulationResult(
        domain=make_domain(),
        deposits=(),
        density_max=density,
    )
    analysis = result.analysis
    density.fill(1.0)

    assert analysis is result.analysis
    assert float(result.density_max.max()) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        result.density_max[0, 0, 0] = 1.0
    with pytest.raises(FrozenInstanceError):
        result.density_max = np.ones(result.domain.grid_shape)
