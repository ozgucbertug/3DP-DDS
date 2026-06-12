from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from dds import (
    BeadProfile,
    Domain,
    LineDeposit,
    PointDeposit,
    PolylineDeposit,
    SimulationResult,
    Simulator,
    simulate,
)


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


def test_simulation_result_strata_uses_deposit_order() -> None:
    domain = make_domain()
    profile = make_profile()
    deposits = [
        PointDeposit(target=(2.5, 2.5, 2.5), profile=profile),
        PointDeposit(target=(4.5, 2.5, 2.5), profile=profile),
        PointDeposit(target=(2.5, 2.5, 4.5), profile=profile),
    ]
    result = simulate(domain, deposits, threshold=0.5)

    field_set = result.analysis.strata(threshold=0.5)

    assert field_set.stratum_ids == (0, 1, 2)
    assert field_set.implicit_field(0).shape == domain.grid_shape
    assert field_set.occupancy(1).dtype == np.bool_
    assert np.max(field_set.label_field) == pytest.approx(3.0)


def test_simulation_result_strata_labels_ordered_deposits() -> None:
    domain = make_domain()
    profile = make_profile()
    deposits = [
        PointDeposit(target=(2.5, 2.5, 2.5), profile=profile),
        PointDeposit(target=(4.5, 2.5, 2.5), profile=profile),
    ]
    result = simulate(domain, deposits, threshold=0.5)

    field_set = result.analysis.strata(threshold=0.5)

    assert field_set.stratum_ids == (0, 1)
    assert np.max(field_set.label_field) == pytest.approx(2.0)


@pytest.mark.parametrize("threshold", [0.25, 0.5, 0.8])
def test_deposition_order_field_matches_order_strata(threshold: float) -> None:
    domain = make_domain()
    profile = make_profile()
    deposits = [
        PointDeposit(target=(2.5, 2.5, 2.5), profile=profile),
        LineDeposit(
            start=(2.5, 2.5, 2.5),
            end=(8.5, 2.5, 2.5),
            profile=profile,
        ),
        PolylineDeposit(
            targets=(
                (5.5, 2.5, 3.5),
                (5.5, 7.5, 3.5),
                (8.5, 7.5, 3.5),
            ),
            profile=profile,
        ),
    ]
    analysis = simulate(domain, deposits, threshold=threshold).analysis

    order_field = analysis.deposition_order_field()
    strata_labels = analysis.strata().label_field

    assert order_field.dtype == np.intp
    np.testing.assert_array_equal(order_field, strata_labels)


def test_zero_threshold_deposition_order_keeps_untouched_voxels_zero() -> None:
    result = simulate(
        make_domain(),
        [PointDeposit(target=(2.5, 2.5, 2.5), profile=make_profile())],
        threshold=0.0,
    )

    order_field = result.analysis.deposition_order_field()

    assert np.any(order_field == 0)
    assert np.any(order_field == 1)


def test_deposition_order_field_is_cached_per_threshold_and_read_only() -> None:
    result = simulate(
        make_domain(),
        [
            PointDeposit(target=(2.5, 2.5, 2.5), profile=make_profile()),
            PointDeposit(target=(2.5, 2.5, 2.5), profile=make_profile()),
        ],
    )

    default_field = result.analysis.deposition_order_field()
    explicit_default = result.analysis.deposition_order_field(threshold=0.5)
    lower_threshold = result.analysis.deposition_order_field(threshold=0.25)

    assert default_field is explicit_default
    assert default_field is not lower_threshold
    assert not default_field.flags.writeable
    assert np.max(default_field) == 2
    with pytest.raises(ValueError):
        default_field[0, 0, 0] = 1


def test_stratum_density_and_occupancy_use_deposit_index() -> None:
    domain = make_domain()
    profile = make_profile()
    deposits = [
        PointDeposit(target=(2.5, 2.5, 2.5), profile=profile),
    ]
    result = simulate(domain, deposits, threshold=0.5)

    field_set = result.analysis.strata(threshold=0.5)
    density = field_set.implicit_field(0)
    occupancy = field_set.occupancy(0)

    assert density.shape == domain.grid_shape
    assert occupancy.dtype == np.bool_

def test_simulator_result_requests_coverage() -> None:
    domain = make_domain()
    profile = make_profile()
    simulator = Simulator(
        domain,
        [
            PointDeposit(target=(2.5, 2.5, 2.5), profile=profile),
            PointDeposit(target=(3.0, 2.5, 2.5), profile=profile),
        ],
    )

    result = simulator.result(threshold=0.5, include_coverage=True)

    assert result.coverage is not None
    assert result.coverage.shape == domain.grid_shape
    assert np.all(result.coverage >= result.implicit_field)


def test_simulation_result_is_an_immutable_snapshot() -> None:
    density = np.zeros(make_domain().grid_shape, dtype=float)
    result = SimulationResult(
        domain=make_domain(),
        deposits=(),
        implicit_field=density,
    )
    analysis = result.analysis
    density.fill(1.0)

    assert analysis is result.analysis
    assert float(result.implicit_field.max()) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        result.implicit_field[0, 0, 0] = 1.0
    with pytest.raises(FrozenInstanceError):
        result.implicit_field = np.ones(result.domain.grid_shape)
