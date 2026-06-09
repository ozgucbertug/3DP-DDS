from __future__ import annotations

import numpy as np
import pytest

from dds import (
    BeadProfile,
    DepositionMetadata,
    Domain,
    LineDeposit,
    PointDeposit,
    PolylineDeposit,
    Pose3D,
    ProcessState,
    SimulationResult,
    Simulator,
    UnitSystem,
    WorkbenchViewConfig,
    apply_deposit_to_field,
    apply_deposit_to_index_field,
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


def test_domain_aligns_nonintegral_bounds_to_voxel_grid() -> None:
    domain = Domain.from_bounds(
        xmin=0.0,
        xmax=1.1,
        ymin=0.0,
        ymax=1.1,
        zmin=0.0,
        zmax=1.1,
        voxel_size=1.0,
    )

    assert domain.grid_shape == (2, 2, 2)
    assert domain.max_corner == pytest.approx((2.0, 2.0, 2.0))
    assert domain.contains_point(domain.index_to_world((1, 1, 1)))


def test_domain_records_explicit_unit_system() -> None:
    units = UnitSystem(length="m", time="s", temperature="K")
    domain = Domain.from_bounds(
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        zmin=0.0,
        zmax=1.0,
        voxel_size=0.1,
        unit_system=units,
    )

    assert domain.unit_system == units
    assert domain.to_dict()["unit_system"] == {
        "length": "m",
        "time": "s",
        "temperature": "K",
    }
    with pytest.raises(ValueError):
        UnitSystem(length="inch")


def test_domain_rejects_inconsistent_direct_construction() -> None:
    with pytest.raises(ValueError, match="max_corner"):
        Domain(
            min_corner=(0.0, 0.0, 0.0),
            max_corner=(1.1, 1.1, 1.1),
            voxel_size=(1.0, 1.0, 1.0),
            grid_shape=(2, 2, 2),
        )


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


def test_line_deposit_with_varying_axes_is_not_clipped_by_endpoint_bounds() -> None:
    domain = Domain.from_bounds(
        xmin=-3.0,
        xmax=5.0,
        ymin=-3.0,
        ymax=3.0,
        zmin=-4.0,
        zmax=4.0,
        voxel_size=0.25,
    )
    deposit = LineDeposit(
        start=(0.0, 0.0, 1.0),
        end=(2.0, 0.0, 1.0),
        profile=BeadProfile(width=1.0, height=2.0),
        start_z_axis=(-0.58861627, -0.45646528, -0.66721087),
        end_z_axis=(0.68238459, -0.07779286, -0.72684217),
    )

    density = Simulator(domain, [deposit]).sample_field(field="density")

    assert density[domain.world_to_index((0.875, 0.625, 3.125))] > 0.0


def test_line_deposit_rejects_antiparallel_endpoint_axes() -> None:
    with pytest.raises(ValueError, match="antiparallel"):
        LineDeposit(
            start=(0.0, 0.0, 0.0),
            end=(1.0, 0.0, 0.0),
            start_z_axis=(0.0, 0.0, 1.0),
            end_z_axis=(0.0, 0.0, -1.0),
        )


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


def test_simulator_deposition_index_is_a_snapshot() -> None:
    domain = make_domain()
    deposit = PointDeposit(
        x=2.5,
        y=2.5,
        z=3.5,
        profile=make_profile(width=2.0, height=2.0),
    )
    simulator = Simulator(domain, [deposit])

    first = simulator.simulate_deposition_index()
    first.fill(99)
    second = simulator.simulate_deposition_index()

    assert int(second.max()) == 0


def test_simulate_returns_rich_result_with_max_based_geometry() -> None:
    domain = make_domain()
    deposits = [
        PointDeposit(x=2.5, y=2.5, z=3.5, profile=make_profile(width=2.0, height=2.0), metadata=make_metadata()),
        PointDeposit(x=2.5, y=2.5, z=3.5, profile=make_profile(width=2.0, height=2.0), metadata=make_metadata()),
    ]

    result = simulate(domain, deposits, threshold=0.5)

    assert isinstance(result, SimulationResult)
    assert result.field("max").shape == domain.grid_shape
    assert result.density_max.max() <= 1.0
    assert result.coverage is None
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

    np.testing.assert_allclose(cached.field("max"), direct.field("max"))
    assert cached.default_threshold == pytest.approx(0.5)


def test_simulate_can_produce_max_and_coverage_fields() -> None:
    domain = make_domain()
    profile = make_profile(width=2.0, height=2.0)
    metadata = make_metadata()
    deposits = [
        PointDeposit(x=2.5, y=2.5, z=3.5, profile=profile, metadata=metadata),
        PointDeposit(x=2.5, y=2.5, z=3.5, profile=profile, metadata=metadata),
    ]

    result = simulate(domain, deposits, compositions=("max", "coverage"), threshold=0.5)

    assert result.coverage is not None
    assert np.all(result.coverage >= result.density_max)
    assert float(result.coverage.max()) > float(result.density_max.max())


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


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (float("nan"), 1.0),
        (1.0, float("inf")),
        (0.0, 1.0),
        (1.0, -1.0),
    ],
)
def test_bead_profile_rejects_invalid_dimensions(width: float, height: float) -> None:
    with pytest.raises(ValueError):
        BeadProfile(width=width, height=height)


def test_metadata_and_process_state_are_validated() -> None:
    source = {"labels": ["calibration"]}
    metadata = DepositionMetadata(layer_id=1, user_data=source)
    process = ProcessState(feedrate=10.0, extrusion_rate=2.0)
    source["labels"].append("mutated")

    assert metadata.to_dict()["user_data"] == {"labels": ["calibration"]}
    with pytest.raises(TypeError):
        metadata.user_data["new"] = "value"  # type: ignore[index]

    with pytest.raises(ValueError):
        ProcessState(feedrate=float("nan"))
    with pytest.raises(ValueError):
        ProcessState(extrusion_rate=-1.0)
    with pytest.raises(TypeError):
        DepositionMetadata(user_data={"bad": object()})
    assert process.feedrate == pytest.approx(10.0)


def test_pose_based_deposits_and_polyline_event() -> None:
    profile = make_profile(width=2.0, height=1.0)
    process = ProcessState(material_id="test", feedrate=5.0)
    start = Pose3D((1.5, 2.5, 3.5))
    corner = Pose3D((4.5, 2.5, 3.5), (0.0, 1.0, 1.0))
    end = Pose3D((4.5, 5.5, 3.5))

    point = PointDeposit.from_pose(start, profile=profile, process=process)
    line = LineDeposit.from_poses(start, corner, profile=profile, process=process)
    polyline = PolylineDeposit(
        poses=(start, corner, end),
        profile=profile,
        process=process,
    )

    assert point.pose == start
    assert line.start_pose == start
    assert line.end_pose.position == corner.position
    assert line.end_pose.axis.to_tuple() == pytest.approx(corner.axis.to_tuple())
    assert len(polyline.segments()) == 2

    result = simulate(
        make_domain(),
        [polyline],
        compositions=("max", "coverage"),
    )
    assert len(result.deposits) == 1
    assert result.coverage is not None
    np.testing.assert_allclose(result.coverage, result.density_max)


# ---------------------------------------------------------------------------
# Incremental accumulation tests
# ---------------------------------------------------------------------------


def _two_deposits() -> list[PointDeposit]:
    profile = make_profile(width=2.0, height=2.0)
    return [
        PointDeposit(x=2.5, y=2.5, z=3.5, profile=profile, metadata=make_metadata()),
        PointDeposit(x=5.5, y=5.5, z=3.5, profile=profile, metadata=DepositionMetadata(layer_id=1)),
    ]


def test_incremental_add_deposit_matches_batch_simulate() -> None:
    """Density from sequential add_deposit must equal simulate() over the same list."""
    domain = make_domain()
    deposits = _two_deposits()

    expected = simulate(domain, deposits)

    sim = Simulator(domain)
    for dep in deposits:
        sim.add_deposit(dep)

    np.testing.assert_allclose(sim.sample_field(field="density"), expected.field("max"))


def test_incremental_add_deposits_matches_batch_simulate() -> None:
    """add_deposits in one call produces the same result as add_deposit one at a time."""
    domain = make_domain()
    deposits = _two_deposits()

    sim_single = Simulator(domain)
    for dep in deposits:
        sim_single.add_deposit(dep)

    sim_bulk = Simulator(domain)
    sim_bulk.add_deposits(deposits)

    np.testing.assert_allclose(
        sim_single.sample_field(field="density"),
        sim_bulk.sample_field(field="density"),
    )


def test_clear_and_readd_reproduces_original_density() -> None:
    """clear_deposits() followed by re-adding the same deposits gives identical fields."""
    domain = make_domain()
    deposits = _two_deposits()

    sim = Simulator(domain, deposits)
    before = sim.sample_field(field="density").copy()

    sim.clear_deposits()
    assert sim.sample_field(field="density").sum() == pytest.approx(0.0)

    sim.add_deposits(deposits)
    np.testing.assert_allclose(sim.sample_field(field="density"), before)


def test_incremental_deposition_index_matches_batch() -> None:
    """Incrementally built deposition index matches the fully recomputed one."""
    domain = make_domain()
    deposits = _two_deposits()

    expected_idx = Simulator(domain, deposits).sample_field(field="deposition_index")

    sim = Simulator(domain)
    for dep in deposits:
        sim.add_deposit(dep)

    np.testing.assert_allclose(sim.sample_field(field="deposition_index"), expected_idx)


def test_apply_deposit_to_field_accumulates_in_place() -> None:
    """apply_deposit_to_field applies one kernel to a pre-allocated grid."""
    domain = make_domain()
    deposit = PointDeposit(x=5.0, y=5.0, z=5.0, profile=make_profile(width=2.0, height=2.0))
    grid = np.zeros(domain.grid_shape, dtype=float)

    hit = apply_deposit_to_field(domain, grid, deposit, composition="coverage")

    assert hit is True
    assert grid.sum() > 0.0


def test_density_is_geometric_envelope_and_coverage_is_additive() -> None:
    domain = make_domain()
    deposit = PointDeposit(
        x=5.0,
        y=5.0,
        z=5.0,
        profile=make_profile(width=2.0, height=2.0),
    )
    simulator = Simulator(domain, [deposit, deposit])

    density = simulator.sample_field(field="density")
    coverage = simulator.sample_field(field="coverage")

    assert float(density.max()) <= 1.0
    assert float(coverage.max()) > float(density.max())
    with pytest.raises(ValueError, match="cannot be normalized"):
        simulator.sample_field(field="coverage", normalize=True)


def test_apply_deposit_to_field_returns_false_for_out_of_bounds_deposit() -> None:
    domain = make_domain()
    deposit = PointDeposit(x=-50.0, y=-50.0, z=-50.0, profile=make_profile())
    grid = np.zeros(domain.grid_shape, dtype=float)

    hit = apply_deposit_to_field(domain, grid, deposit)

    assert hit is False
    assert grid.sum() == pytest.approx(0.0)


def test_apply_deposit_to_field_rejects_invalid_composition_before_sampling() -> None:
    domain = make_domain()
    deposit = PointDeposit(
        x=-50.0,
        y=-50.0,
        z=-50.0,
        profile=make_profile(),
    )
    grid = np.zeros(domain.grid_shape, dtype=float)

    with pytest.raises(ValueError, match="composition"):
        apply_deposit_to_field(
            domain,
            grid,
            deposit,
            composition="invalid",  # type: ignore[arg-type]
        )


def test_apply_deposit_to_index_field_marks_touched_voxels() -> None:
    domain = make_domain()
    deposit = PointDeposit(x=5.0, y=5.0, z=5.0, profile=make_profile(width=2.0, height=2.0))
    index_field = np.full(domain.grid_shape, -1, dtype=np.intp)

    hit = apply_deposit_to_index_field(domain, index_field, deposit, deposit_index=7)

    assert hit is True
    assert int(index_field.max()) == 7
    assert int(index_field[4, 4, 4]) == 7
