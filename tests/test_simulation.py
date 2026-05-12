from __future__ import annotations

import numpy as np
import pytest

from dds import (
    BeadProfile,
    DepositionMetadata,
    Domain,
    LineDeposit,
    PointDeposit,
    Simulator,
    sample_field,
    simulate_deposition_index,
    simulate_occupancy,
)


def make_domain() -> Domain:
    return Domain.from_bounds(
        xmin=0.0,
        xmax=10.0,
        ymin=0.0,
        ymax=10.0,
        zmin=0.0,
        zmax=10.0,
        voxel_size=1.0,
    )


def make_profile(width: float = 1.2, height: float = 1.2) -> BeadProfile:
    return BeadProfile(width=width, height=height)


def make_metadata() -> DepositionMetadata:
    return DepositionMetadata(layer_id=0)


def test_domain_shape_matches_bounds() -> None:
    domain = make_domain()
    assert domain.grid_shape == (10, 10, 10)


def test_coordinate_conversion_round_trip_for_voxel_centers() -> None:
    domain = make_domain()
    point = domain.index_to_world((2, 3, 4))
    assert domain.world_to_index(point) == (2, 3, 4)


def test_point_deposit_contributes_locally() -> None:
    domain = make_domain()
    deposit = PointDeposit(x=2.5, y=2.5, z=3.5, profile=make_profile(width=2.0, height=2.0), metadata=make_metadata())
    density = sample_field(domain, [deposit], field="density")

    assert density[2, 2, 2] > 0.0
    assert density[0, 0, 0] == pytest.approx(0.0)


def test_point_deposit_target_marks_the_top_of_the_bead() -> None:
    domain = make_domain()
    deposit = PointDeposit(x=2.5, y=2.5, z=3.5, profile=make_profile(width=2.0, height=2.0), metadata=make_metadata())
    density = sample_field(domain, [deposit], field="density")

    assert density[2, 2, 2] > density[2, 2, 3]
    assert density[2, 2, 3] == pytest.approx(0.5)


def test_point_deposit_uses_rounded_bead_geometry_not_ellipsoidal_falloff() -> None:
    domain = make_domain()
    deposit = PointDeposit(x=2.5, y=2.5, z=3.5, profile=make_profile(width=4.0, height=2.0), metadata=make_metadata())
    occupancy = simulate_occupancy(domain, [deposit], threshold=0.5)

    assert occupancy[2, 2, 2]
    assert occupancy[3, 2, 3]
    assert not occupancy[4, 2, 3]


def test_line_deposit_produces_continuous_occupied_region() -> None:
    domain = make_domain()
    deposit = LineDeposit(
        start=(1.5, 2.5, 3.5),
        end=(6.5, 2.5, 3.5),
        profile=make_profile(),
        metadata=make_metadata(),
    )
    occupancy = simulate_occupancy(domain, [deposit], threshold=0.25)

    assert all(bool(occupancy[x, 2, 2]) for x in range(1, 7))


def test_deposition_index_accumulates_for_overlapping_deposits() -> None:
    domain = make_domain()
    deposit = PointDeposit(x=2.5, y=2.5, z=3.5, profile=make_profile(width=2.0, height=2.0), metadata=make_metadata())
    single = simulate_deposition_index(domain, [deposit])
    overlap = simulate_deposition_index(domain, [deposit, deposit])

    assert overlap.max() == pytest.approx(single.max() * 2.0)


def test_thresholding_changes_occupied_voxel_count() -> None:
    domain = make_domain()
    deposit = PointDeposit(x=2.5, y=2.5, z=4.0, profile=make_profile(width=3.0, height=3.0), metadata=make_metadata())

    low_threshold = simulate_occupancy(domain, [deposit], threshold=0.1)
    high_threshold = simulate_occupancy(domain, [deposit], threshold=0.8)

    assert int(low_threshold.sum()) >= int(high_threshold.sum())


def test_deposits_outside_bounds_are_skipped_and_partial_overlap_is_kept() -> None:
    domain = make_domain()
    outside = PointDeposit(x=-5.0, y=-5.0, z=-5.0, profile=make_profile(), metadata=make_metadata())
    partial = LineDeposit(
        start=(-0.5, 2.5, 3.5),
        end=(2.5, 2.5, 3.5),
        profile=make_profile(),
        metadata=make_metadata(),
    )

    outside_density = simulate_deposition_index(domain, [outside])
    partial_occupancy = simulate_occupancy(domain, [partial], threshold=0.25)

    assert np.count_nonzero(outside_density) == 0
    assert partial_occupancy[0, 2, 2]
    assert partial_occupancy[2, 2, 2]


def test_zero_length_line_matches_point_deposit() -> None:
    domain = make_domain()
    profile = make_profile(width=2.0, height=2.0)
    metadata = make_metadata()
    point = PointDeposit(x=2.5, y=2.5, z=3.5, profile=profile, metadata=metadata)
    zero_length_line = LineDeposit(start=(2.5, 2.5, 3.5), end=(2.5, 2.5, 3.5), profile=profile, metadata=metadata)

    point_field = simulate_deposition_index(domain, [point])
    line_field = simulate_deposition_index(domain, [zero_length_line])

    np.testing.assert_allclose(point_field, line_field)


def test_simulator_queries_use_nearest_grid_samples_and_safe_defaults() -> None:
    domain = make_domain()
    deposit = PointDeposit(x=2.5, y=2.5, z=3.5, profile=make_profile(width=2.0, height=2.0), metadata=make_metadata())
    simulator = Simulator(domain, [deposit])

    assert simulator.is_occupied((2.5, 2.5, 2.5), threshold=0.5)
    assert simulator.query_deposition_index((2.5, 2.5, 2.5)) > 0.0
    assert simulator.is_occupied((-1.0, -1.0, -1.0)) is False
    assert simulator.query_deposition_index((-1.0, -1.0, -1.0)) == pytest.approx(0.0)
