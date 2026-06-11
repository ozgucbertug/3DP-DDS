from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dds import (
    BeadProfile,
    DepositionMetadata,
    DepositionTarget,
    Domain,
    Line3D,
    LineDeposit,
    Point3D,
    PointDeposit,
    PolylineDeposit,
    Pose3D,
    SimulationResult,
    Simulator,
    Vector3D,
    simulate,
)
from dds.analysis import summarize_layers
from dds.fields import apply_deposit_to_field, apply_deposit_to_index_field
from dds.viz import ViewConfig


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


def test_geometry_primitives_have_distinct_roles() -> None:
    point = Point3D(1.0, 2.0, 3.0)
    pose = Pose3D(point, Rotation.from_euler("z", 90.0, degrees=True))
    target = DepositionTarget(point, Vector3D(0.0, 0.0, 2.0))
    line = Line3D(point, Point3D(4.0, 6.0, 3.0))

    assert pose.position == point
    assert target.normal == Vector3D(0.0, 0.0, 1.0)
    assert pose.transform_vector((1.0, 0.0, 0.0)).to_tuple() == pytest.approx(
        (0.0, 1.0, 0.0)
    )
    assert line.direction == Vector3D(3.0, 4.0, 0.0)
    assert line.length == pytest.approx(5.0)


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


def test_domain_records_explicit_length_unit() -> None:
    domain = Domain.from_bounds(
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        zmin=0.0,
        zmax=1.0,
        voxel_size=0.1,
        length_unit="m",
    )

    assert domain.length_unit == "m"
    assert domain.to_dict()["length_unit"] == "m"
    with pytest.raises(ValueError, match="length_unit"):
        Domain.from_bounds(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            voxel_size=0.1,
            length_unit="inch",  # type: ignore[arg-type]
        )


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
    deposit = PointDeposit(target=(2.5, 2.5, 3.5), profile=make_profile(width=2.0, height=2.0), metadata=make_metadata())
    density = Simulator(domain, [deposit]).result().implicit_field

    assert density[2, 2, 2] > 0.0
    assert density[0, 0, 0] == pytest.approx(0.0)


def test_point_deposit_target_marks_the_top_of_the_bead() -> None:
    domain = make_domain()
    deposit = PointDeposit(target=(2.5, 2.5, 3.5), profile=make_profile(width=2.0, height=2.0), metadata=make_metadata())
    density = Simulator(domain, [deposit]).result().implicit_field

    assert density[2, 2, 2] > density[2, 2, 3]
    assert density[2, 2, 3] == pytest.approx(0.5)


def test_point_deposit_uses_rounded_bead_geometry_not_ellipsoidal_falloff() -> None:
    domain = make_domain()
    deposit = PointDeposit(target=(2.5, 2.5, 3.5), profile=make_profile(width=4.0, height=2.0), metadata=make_metadata())
    occupancy = Simulator(domain, [deposit]).result().analysis.occupancy(threshold=0.5)

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
    occupancy = Simulator(domain, [deposit]).result().analysis.occupancy(threshold=0.25)

    assert all(bool(occupancy[x, 2, 2]) for x in range(1, 7))


def test_line_deposit_with_equal_endpoint_axes_has_stable_field_values() -> None:
    domain = make_domain()
    axis = (0.0, 0.0, 1.0)
    deposit = LineDeposit(
        start=DepositionTarget((1.5, 2.5, 3.5), axis),
        end=DepositionTarget((6.5, 2.5, 3.5), axis),
        profile=make_profile(width=2.0, height=2.0),
    )

    field = simulate(domain, [deposit]).implicit_field

    np.testing.assert_allclose(field[1:7, 2, 2], 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(field[1:7, 2, 3], 0.5, rtol=0.0, atol=0.0)
    assert field[0, 0, 0] == pytest.approx(0.0)


def test_line_deposit_with_equal_endpoint_axes_skips_spherical_interpolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dds.kernels

    def unexpected_interpolation(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("equal endpoint axes should not use spherical interpolation")

    monkeypatch.setattr(
        dds.kernels,
        "slerp_unit_vectors",
        unexpected_interpolation,
    )
    deposit = LineDeposit(
        start=DepositionTarget((1.5, 2.5, 3.5), (0.0, 0.0, 1.0)),
        end=DepositionTarget((6.5, 2.5, 3.5), (0.0, 0.0, 1.0)),
        profile=make_profile(width=2.0, height=2.0),
    )

    assert simulate(make_domain(), [deposit]).implicit_field.max() == pytest.approx(
        1.0
    )


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
        start=DepositionTarget(
            (0.0, 0.0, 1.0),
            (-0.58861627, -0.45646528, -0.66721087),
        ),
        end=DepositionTarget(
            (2.0, 0.0, 1.0),
            (0.68238459, -0.07779286, -0.72684217),
        ),
        profile=BeadProfile(width=1.0, height=2.0),
    )

    density = Simulator(domain, [deposit]).result().implicit_field

    assert density[domain.world_to_index((0.875, 0.625, 3.125))] > 0.0
    assert float(density.max()) == pytest.approx(1.0)
    assert float(density.sum()) == pytest.approx(
        163.88229860627575,
        rel=0.0,
        abs=1e-12,
    )


def test_line_deposit_remains_continuous_across_kernel_tile_boundaries() -> None:
    domain = Domain.from_bounds(
        xmin=0.0,
        xmax=80.0,
        ymin=0.0,
        ymax=8.0,
        zmin=0.0,
        zmax=8.0,
        voxel_size=1.0,
    )
    deposit = LineDeposit(
        start=(4.5, 3.5, 4.5),
        end=(68.5, 3.5, 4.5),
        profile=make_profile(width=2.0, height=2.0),
    )

    field = simulate(domain, [deposit]).implicit_field

    np.testing.assert_allclose(field[4:69, 3, 3], 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(field[4:69, 3, 4], 0.5, rtol=0.0, atol=0.0)
    assert field[31, 3, 3] == field[32, 3, 3] == pytest.approx(1.0)


def test_line_deposit_rejects_antiparallel_endpoint_axes() -> None:
    with pytest.raises(ValueError, match="antiparallel"):
            LineDeposit(
                start=DepositionTarget((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                end=DepositionTarget((1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
                profile=make_profile(),
            )


def test_explicit_profile_geometry_is_independent_of_voxel_size() -> None:
    profile = BeadProfile(width=2.0, height=1.0)
    deposit = PointDeposit(target=(4.0, 5.0, 6.0), profile=profile)

    coarse = Domain.from_deposits(deposit, voxel_size=1.0, padding=0.0)
    fine = Domain.from_deposits(deposit, voxel_size=0.25, padding=0.0)

    minimum, maximum = deposit.support_bounds()
    assert minimum.to_tuple() == pytest.approx((3.0, 4.0, 5.0))
    assert maximum.to_tuple() == pytest.approx((5.0, 6.0, 6.0))
    assert coarse.min_corner == fine.min_corner
    assert coarse.max_corner == fine.max_corner


def test_deposition_index_accumulates_for_overlapping_deposits() -> None:
    domain = make_domain()
    deposit = PointDeposit(target=(2.5, 2.5, 3.5), profile=make_profile(width=2.0, height=2.0), metadata=make_metadata())
    single = Simulator(domain, [deposit]).result().analysis.deposition_index_field()
    overlap = Simulator(domain, [deposit, deposit]).result().analysis.deposition_index_field()

    # single deposit → index 0 for touched voxels, -1 for untouched
    assert int(single.max()) == 0
    # two deposits (same position) → last index is 1 for touched voxels
    assert int(overlap.max()) == 1


def test_thresholding_changes_occupied_voxel_count() -> None:
    domain = make_domain()
    deposit = PointDeposit(target=(2.5, 2.5, 4.0), profile=make_profile(width=3.0, height=3.0), metadata=make_metadata())
    result = Simulator(domain, [deposit]).result()

    low_threshold = result.analysis.occupancy(threshold=0.1)
    high_threshold = result.analysis.occupancy(threshold=0.8)

    np.testing.assert_array_equal(low_threshold, result.implicit_field >= 0.1)
    np.testing.assert_array_equal(high_threshold, result.implicit_field >= 0.8)
    assert int(low_threshold.sum()) > int(high_threshold.sum())
    assert np.any(low_threshold & ~high_threshold)


def test_deposits_outside_bounds_are_skipped_and_partial_overlap_is_kept() -> None:
    domain = make_domain()
    outside = PointDeposit(target=(-5.0, -5.0, -5.0), profile=make_profile(), metadata=make_metadata())
    partial = LineDeposit(
        start=(-0.5, 2.5, 3.5),
        end=(2.5, 2.5, 3.5),
        profile=make_profile(),
        metadata=make_metadata(),
    )

    outside_density = Simulator(domain, [outside]).result().analysis.deposition_index_field()
    partial_occupancy = Simulator(domain, [partial]).result().analysis.occupancy(threshold=0.25)

    assert np.all(outside_density == -1), "deposit outside bounds should leave all voxels untouched (-1)"
    assert partial_occupancy[0, 2, 2]
    assert partial_occupancy[2, 2, 2]


def test_zero_length_line_matches_point_deposit() -> None:
    domain = make_domain()
    profile = make_profile(width=2.0, height=2.0)
    metadata = make_metadata()
    point = PointDeposit(target=(2.5, 2.5, 3.5), profile=profile, metadata=metadata)
    zero_length_line = LineDeposit(start=(2.5, 2.5, 3.5), end=(2.5, 2.5, 3.5), profile=profile, metadata=metadata)

    point_result = Simulator(domain, [point]).result(include_coverage=True)
    line_result = Simulator(domain, [zero_length_line]).result(include_coverage=True)

    np.testing.assert_allclose(
        point_result.implicit_field,
        line_result.implicit_field,
        rtol=0.0,
        atol=0.0,
    )
    assert point_result.coverage is not None
    assert line_result.coverage is not None
    np.testing.assert_allclose(
        point_result.coverage,
        line_result.coverage,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        point_result.analysis.occupancy(threshold=0.5),
        line_result.analysis.occupancy(threshold=0.5),
    )


def test_zero_length_line_uses_point_kernel_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dds.kernels

    def unexpected_line_sampling(*args: object, **kwargs: object) -> object:
        raise AssertionError("zero-length lines should use point-kernel sampling")

    monkeypatch.setattr(
        dds.kernels,
        "_sample_line_on_bounds",
        unexpected_line_sampling,
    )
    deposit = LineDeposit(
        start=(2.5, 2.5, 3.5),
        end=(2.5, 2.5, 3.5),
        profile=make_profile(width=2.0, height=2.0),
    )

    assert simulate(make_domain(), [deposit]).implicit_field.max() == pytest.approx(
        1.0
    )


def test_simulator_queries_use_nearest_grid_samples_and_safe_defaults() -> None:
    domain = make_domain()
    deposit = PointDeposit(target=(2.5, 2.5, 3.5), profile=make_profile(width=2.0, height=2.0), metadata=make_metadata())
    simulator = Simulator(domain, [deposit])

    analysis = simulator.result().analysis
    assert analysis.contains_point((2.5, 2.5, 2.5), threshold=0.5)
    assert analysis.sample_deposition_index((2.5, 2.5, 2.5)) == 0
    assert analysis.contains_point((-1.0, -1.0, -1.0)) is False
    assert analysis.sample_deposition_index((-1.0, -1.0, -1.0)) == -1


def test_simulator_deposition_index_is_a_snapshot() -> None:
    domain = make_domain()
    deposit = PointDeposit(
        target=(2.5, 2.5, 3.5),
        profile=make_profile(width=2.0, height=2.0),
    )
    simulator = Simulator(domain, [deposit])

    first = simulator.result().analysis.deposition_index_field()
    with pytest.raises(ValueError):
        first.fill(99)
    second = simulator.result().analysis.deposition_index_field()

    assert int(second.max()) == 0


def test_simulate_returns_rich_result_with_implicit_geometry() -> None:
    domain = make_domain()
    deposits = [
        PointDeposit(target=(2.5, 2.5, 3.5), profile=make_profile(width=2.0, height=2.0), metadata=make_metadata()),
        PointDeposit(target=(2.5, 2.5, 3.5), profile=make_profile(width=2.0, height=2.0), metadata=make_metadata()),
    ]

    result = simulate(domain, deposits, threshold=0.5)

    assert isinstance(result, SimulationResult)
    assert result.implicit_field.shape == domain.grid_shape
    assert result.implicit_field.max() <= 1.0
    assert result.coverage is None
    assert result.analysis.occupancy(threshold=0.5)[2, 2, 2]
    assert result.analysis.contains_point((2.5, 2.5, 2.5), representation="occupancy", threshold=0.5)


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

    np.testing.assert_allclose(cached.implicit_field, direct.implicit_field)
    assert cached.default_threshold == pytest.approx(0.5)


def test_simulator_cold_dual_field_result_samples_each_deposit_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dds.fields

    domain = make_domain()
    deposits = _two_deposits()
    calls = 0
    original = dds.fields.iter_deposit_kernels

    def tracked(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        yield from original(*args, **kwargs)

    monkeypatch.setattr(dds.fields, "iter_deposit_kernels", tracked)

    result = Simulator(domain, deposits).result(include_coverage=True)

    assert result.coverage is not None
    assert calls == len(deposits)


def test_simulate_can_include_coverage() -> None:
    domain = make_domain()
    profile = make_profile(width=2.0, height=2.0)
    metadata = make_metadata()
    deposits = [
        PointDeposit(target=(2.5, 2.5, 3.5), profile=profile, metadata=metadata),
        PointDeposit(target=(2.5, 2.5, 3.5), profile=profile, metadata=metadata),
    ]

    result = simulate(domain, deposits, include_coverage=True, threshold=0.5)

    assert result.coverage is not None
    assert np.all(result.coverage >= result.implicit_field)
    assert float(result.coverage.max()) > float(result.implicit_field.max())


def test_domain_from_deposits_anisotropic_voxel_size() -> None:
    profile = make_profile(width=2.0, height=1.0)
    deposits = [
        PointDeposit(target=(2.0, 3.0, 1.0), profile=profile),
        PointDeposit(target=(6.0, 7.0, 3.0), profile=profile),
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


def test_view_config_rejects_non_canonical_build_direction_string() -> None:
    import pytest
    with pytest.raises(ValueError, match="build_direction"):
        ViewConfig(build_direction="Z+")  # canonical form is "+Z"

    # Valid canonical strings must not raise.
    for direction in ("+X", "-X", "+Y", "-Y", "+Z", "-Z"):
        cfg = ViewConfig(build_direction=direction)
        assert cfg.build_direction == direction

    # Tuple form must also be accepted without error.
    cfg = ViewConfig(build_direction=(0.0, 0.0, 1.0))
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


def test_metadata_is_immutable_and_validated() -> None:
    source = {"labels": ["calibration"]}
    metadata = DepositionMetadata(layer_id=1, user_data=source)
    source["labels"].append("mutated")

    assert metadata.to_dict()["user_data"] == {"labels": ["calibration"]}
    with pytest.raises(TypeError):
        metadata.user_data["new"] = "value"  # type: ignore[index]

    with pytest.raises(TypeError):
        DepositionMetadata(user_data={"bad": object()})


def test_pose_based_deposits_and_polyline_event() -> None:
    profile = make_profile(width=2.0, height=1.0)
    start = Pose3D((1.5, 2.5, 3.5))
    corner = DepositionTarget((4.5, 2.5, 3.5), (0.0, 1.0, 1.0))
    end = Pose3D((4.5, 5.5, 3.5))

    point = PointDeposit(target=start, profile=profile)
    line = LineDeposit(start=start, end=corner, profile=profile)
    polyline = PolylineDeposit(
        targets=(start, corner, end),
        profile=profile,
    )

    assert point.target == DepositionTarget.from_pose(start)
    assert line.start == DepositionTarget.from_pose(start)
    assert line.end == corner
    assert len(polyline.segments()) == 2

    result = simulate(
        make_domain(),
        [polyline],
        include_coverage=True,
    )
    assert len(result.deposits) == 1
    assert result.coverage is not None
    np.testing.assert_allclose(result.coverage, result.implicit_field)


# ---------------------------------------------------------------------------
# Incremental accumulation tests
# ---------------------------------------------------------------------------


def _two_deposits() -> list[PointDeposit]:
    profile = make_profile(width=2.0, height=2.0)
    return [
        PointDeposit(target=(2.5, 2.5, 3.5), profile=profile, metadata=make_metadata()),
        PointDeposit(target=(5.5, 5.5, 3.5), profile=profile, metadata=DepositionMetadata(layer_id=1)),
    ]


def test_incremental_add_deposit_matches_batch_simulate() -> None:
    """Density from sequential add_deposit must equal simulate() over the same list."""
    domain = make_domain()
    deposits = _two_deposits()

    expected = simulate(domain, deposits)

    sim = Simulator(domain)
    for dep in deposits:
        sim.add_deposit(dep)

    np.testing.assert_allclose(sim.result().implicit_field, expected.implicit_field)


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
        sim_single.result().implicit_field,
        sim_bulk.result().implicit_field,
    )


def test_clear_and_readd_reproduces_original_implicit_field() -> None:
    """clear_deposits() followed by re-adding the same deposits gives identical fields."""
    domain = make_domain()
    deposits = _two_deposits()

    sim = Simulator(domain, deposits)
    before = sim.result().implicit_field.copy()

    sim.clear_deposits()
    assert sim.result().implicit_field.sum() == pytest.approx(0.0)

    sim.add_deposits(deposits)
    np.testing.assert_allclose(sim.result().implicit_field, before)


def test_incremental_deposition_index_matches_batch() -> None:
    """Incrementally built deposition index matches the fully recomputed one."""
    domain = make_domain()
    deposits = _two_deposits()

    expected_idx = Simulator(domain, deposits).result().analysis.deposition_index_field()

    sim = Simulator(domain)
    for dep in deposits:
        sim.add_deposit(dep)

    np.testing.assert_allclose(sim.result().analysis.deposition_index_field(), expected_idx)


def test_apply_deposit_to_field_accumulates_in_place() -> None:
    """apply_deposit_to_field applies one kernel to a pre-allocated grid."""
    domain = make_domain()
    deposit = PointDeposit(target=(5.0, 5.0, 5.0), profile=make_profile(width=2.0, height=2.0))
    grid = np.zeros(domain.grid_shape, dtype=float)

    hit = apply_deposit_to_field(domain, grid, deposit, field="coverage")

    assert hit is True
    assert grid.sum() > 0.0


def test_implicit_field_is_geometric_envelope_and_coverage_is_additive() -> None:
    domain = make_domain()
    deposit = PointDeposit(
        target=(5.0, 5.0, 5.0),
        profile=make_profile(width=2.0, height=2.0),
    )
    simulator = Simulator(domain, [deposit, deposit])

    result = simulator.result(include_coverage=True)
    implicit_field = result.implicit_field
    coverage = result.coverage

    assert float(implicit_field.max()) <= 1.0
    assert float(coverage.max()) > float(implicit_field.max())
    assert coverage is not None


def test_apply_deposit_to_field_returns_false_for_out_of_bounds_deposit() -> None:
    domain = make_domain()
    deposit = PointDeposit(target=(-50.0, -50.0, -50.0), profile=make_profile())
    grid = np.zeros(domain.grid_shape, dtype=float)

    hit = apply_deposit_to_field(domain, grid, deposit)

    assert hit is False
    assert grid.sum() == pytest.approx(0.0)


def test_apply_deposit_to_field_rejects_invalid_field_before_sampling() -> None:
    domain = make_domain()
    deposit = PointDeposit(
        target=(-50.0, -50.0, -50.0),
        profile=make_profile(),
    )
    grid = np.zeros(domain.grid_shape, dtype=float)

    with pytest.raises(ValueError, match="field"):
        apply_deposit_to_field(
            domain,
            grid,
            deposit,
            field="invalid",  # type: ignore[arg-type]
        )


def test_apply_deposit_to_index_field_marks_touched_voxels() -> None:
    domain = make_domain()
    deposit = PointDeposit(target=(5.0, 5.0, 5.0), profile=make_profile(width=2.0, height=2.0))
    index_field = np.full(domain.grid_shape, -1, dtype=np.intp)

    hit = apply_deposit_to_index_field(domain, index_field, deposit, deposit_index=7)

    assert hit is True
    assert int(index_field.max()) == 7
    assert int(index_field[4, 4, 4]) == 7


def test_summarize_layers_handles_polyline_deposits() -> None:
    deposit = PolylineDeposit(
        targets=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
        ),
        profile=make_profile(),
        metadata=DepositionMetadata(layer_id=2),
    )

    summary = summarize_layers([deposit])[2]

    assert summary["deposit_count"] == 1
    assert summary["polyline_deposits"] == 1
    assert summary["total_line_length"] == pytest.approx(2.0)
