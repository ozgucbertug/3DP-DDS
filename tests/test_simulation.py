from __future__ import annotations

import numpy as np
import pytest

from dds import (
    BeadProfile,
    DepositionMetadata,
    Domain,
    LineDeposit,
    PointDeposit,
    SimulationResult,
    Simulator,
    WorkbenchViewConfig,
    simulate,
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
    density = Simulator(domain, [deposit]).sample_field(field="density")

    assert density[2, 2, 2] > 0.0
    assert density[0, 0, 0] == pytest.approx(0.0)


def test_point_deposit_target_marks_the_top_of_the_bead() -> None:
    domain = make_domain()
    deposit = PointDeposit(x=2.5, y=2.5, z=3.5, profile=make_profile(width=2.0, height=2.0), metadata=make_metadata())
    density = Simulator(domain, [deposit]).sample_field(field="density")

    assert density[2, 2, 2] > density[2, 2, 3]
    assert density[2, 2, 3] == pytest.approx(0.5)


def test_point_deposit_uses_rounded_bead_geometry_not_ellipsoidal_falloff() -> None:
    domain = make_domain()
    deposit = PointDeposit(x=2.5, y=2.5, z=3.5, profile=make_profile(width=4.0, height=2.0), metadata=make_metadata())
    occupancy = Simulator(domain, [deposit]).simulate_occupancy(threshold=0.5)

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
    occupancy = Simulator(domain, [deposit]).simulate_occupancy(threshold=0.25)

    assert all(bool(occupancy[x, 2, 2]) for x in range(1, 7))


def test_deposition_index_accumulates_for_overlapping_deposits() -> None:
    domain = make_domain()
    deposit = PointDeposit(x=2.5, y=2.5, z=3.5, profile=make_profile(width=2.0, height=2.0), metadata=make_metadata())
    single = Simulator(domain, [deposit]).simulate_deposition_index()
    overlap = Simulator(domain, [deposit, deposit]).simulate_deposition_index()

    # single deposit → index 0 for touched voxels, -1 for untouched
    assert int(single.max()) == 0
    # two deposits (same position) → last index is 1 for touched voxels
    assert int(overlap.max()) == 1


def test_thresholding_changes_occupied_voxel_count() -> None:
    domain = make_domain()
    deposit = PointDeposit(x=2.5, y=2.5, z=4.0, profile=make_profile(width=3.0, height=3.0), metadata=make_metadata())

    low_threshold = Simulator(domain, [deposit]).simulate_occupancy(threshold=0.1)
    high_threshold = Simulator(domain, [deposit]).simulate_occupancy(threshold=0.8)

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

    outside_density = Simulator(domain, [outside]).simulate_deposition_index()
    partial_occupancy = Simulator(domain, [partial]).simulate_occupancy(threshold=0.25)

    assert np.all(outside_density == -1), "deposit outside bounds should leave all voxels untouched (-1)"
    assert partial_occupancy[0, 2, 2]
    assert partial_occupancy[2, 2, 2]


def test_zero_length_line_matches_point_deposit() -> None:
    domain = make_domain()
    profile = make_profile(width=2.0, height=2.0)
    metadata = make_metadata()
    point = PointDeposit(x=2.5, y=2.5, z=3.5, profile=profile, metadata=metadata)
    zero_length_line = LineDeposit(start=(2.5, 2.5, 3.5), end=(2.5, 2.5, 3.5), profile=profile, metadata=metadata)

    point_field = Simulator(domain, [point]).simulate_deposition_index()
    line_field = Simulator(domain, [zero_length_line]).simulate_deposition_index()

    np.testing.assert_allclose(point_field, line_field)


def test_simulator_queries_use_nearest_grid_samples_and_safe_defaults() -> None:
    domain = make_domain()
    deposit = PointDeposit(x=2.5, y=2.5, z=3.5, profile=make_profile(width=2.0, height=2.0), metadata=make_metadata())
    simulator = Simulator(domain, [deposit])

    assert simulator.is_occupied((2.5, 2.5, 2.5), threshold=0.5)
    assert simulator.query_deposition_index((2.5, 2.5, 2.5)) == pytest.approx(0.0)  # first deposit → index 0
    assert simulator.is_occupied((-1.0, -1.0, -1.0)) is False
    assert simulator.query_deposition_index((-1.0, -1.0, -1.0)) == pytest.approx(-1.0)  # outside domain → -1


def test_simulate_returns_rich_result_with_max_based_geometry() -> None:
    domain = make_domain()
    deposits = [
        PointDeposit(x=2.5, y=2.5, z=3.5, profile=make_profile(width=2.0, height=2.0), metadata=make_metadata()),
        PointDeposit(x=2.5, y=2.5, z=3.5, profile=make_profile(width=2.0, height=2.0), metadata=make_metadata()),
    ]

    result = simulate(domain, deposits, threshold=0.5)

    assert isinstance(result, SimulationResult)
    assert result.density("max").shape == domain.grid_shape
    assert result.density_max.max() <= 1.0
    assert result.density_sum is None
    assert result.occupancy(threshold=0.5)[2, 2, 2]
    assert result.analysis_bundle().contains_point((2.5, 2.5, 2.5), representation="occupancy", threshold=0.5)


def test_simulator_result_matches_stateless_simulate() -> None:
    domain = make_domain()
    deposits = [
        LineDeposit(
            start=(1.5, 2.5, 3.5),
            end=(6.5, 2.5, 3.5),
            profile=make_profile(),
            metadata=make_metadata(),
        )
    ]
    simulator = Simulator(domain, deposits)

    direct = simulate(domain, deposits, threshold=0.5)
    cached = simulator.result(threshold=0.5)

    np.testing.assert_allclose(cached.density("max"), direct.density("max"))
    assert cached.default_threshold == pytest.approx(0.5)


def test_simulate_can_produce_max_and_sum_density_fields() -> None:
    domain = make_domain()
    profile = make_profile(width=2.0, height=2.0)
    metadata = make_metadata()
    deposits = [
        PointDeposit(x=2.5, y=2.5, z=3.5, profile=profile, metadata=metadata),
        PointDeposit(x=2.5, y=2.5, z=3.5, profile=profile, metadata=metadata),
    ]

    result = simulate(domain, deposits, compositions=("max", "sum"), threshold=0.5)

    assert result.density_sum is not None
    assert np.all(result.density_sum >= result.density_max)
    assert float(result.density_sum.max()) > float(result.density_max.max())


def test_domain_from_deposits_anisotropic_voxel_size() -> None:
    profile = make_profile(width=2.0, height=1.0)
    deposits = [
        PointDeposit(x=2.0, y=3.0, z=1.0, profile=profile),
        PointDeposit(x=6.0, y=7.0, z=3.0, profile=profile),
    ]
    domain = Domain.from_deposits(deposits, voxel_size=(0.5, 0.25, 1.0))

    assert domain.voxel_size == (0.5, 0.25, 1.0)
    assert domain.grid_shape[0] > 0
    assert domain.grid_shape[1] > 0
    assert domain.grid_shape[2] > 0
    # Domain must enclose all deposit support-bound centres.
    for deposit in deposits:
        minimum, maximum = deposit.support_bounds()
        assert domain.contains_point((
            (minimum.x + maximum.x) / 2,
            (minimum.y + maximum.y) / 2,
            (minimum.z + maximum.z) / 2,
        ))


def test_workbench_view_config_rejects_non_canonical_build_direction_string() -> None:
    import pytest
    with pytest.raises(ValueError, match="build_direction"):
        WorkbenchViewConfig(build_direction="Z+")  # canonical form is "+Z"

    # Valid canonical strings must not raise.
    for direction in ("+X", "-X", "+Y", "-Y", "+Z", "-Z"):
        cfg = WorkbenchViewConfig(build_direction=direction)
        assert cfg.build_direction == direction

    # Tuple form must also be accepted without error.
    cfg = WorkbenchViewConfig(build_direction=(0.0, 0.0, 1.0))
    assert cfg.build_direction == (0.0, 0.0, 1.0)
