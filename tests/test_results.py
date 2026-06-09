from __future__ import annotations

import numpy as np
import pytest

from dds import BeadProfile, DepositionMetadata, Domain, PointDeposit, Simulator, simulate
from dds.results import simulation_result


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

    field_set = result.strata(mode="auto", threshold=0.5)

    assert result.layer_ids() == (0, 1)
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

    field_set = result.strata(mode="auto", threshold=0.5)

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

    density = result.layer_density(3, threshold=0.5)
    occupancy = result.layer_occupancy(3, threshold=0.5)

    assert density.shape == domain.grid_shape
    assert occupancy.dtype == np.bool_

    missing_layer_result = simulate(
        domain,
        [PointDeposit(x=2.5, y=2.5, z=2.5, profile=profile, metadata=DepositionMetadata())],
        threshold=0.5,
    )
    with pytest.raises(ValueError):
        missing_layer_result.layer_density(0, threshold=0.5)


def test_simulation_result_from_simulator_requests_coverage() -> None:
    domain = make_domain()
    profile = make_profile()
    simulator = Simulator(
        domain,
        [
            PointDeposit(x=2.5, y=2.5, z=2.5, profile=profile, metadata=DepositionMetadata()),
            PointDeposit(x=3.0, y=2.5, z=2.5, profile=profile, metadata=DepositionMetadata()),
        ],
    )

    result = simulation_result(simulator, threshold=0.5)

    assert result.coverage is not None
    assert result.coverage.shape == domain.grid_shape
    assert np.all(result.coverage >= result.density_max)
