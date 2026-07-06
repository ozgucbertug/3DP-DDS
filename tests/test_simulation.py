from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dds import (
    BeadProfile,
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


def brute_force_line_sweep_field(
    domain: Domain,
    deposit: LineDeposit,
    *,
    samples: int = 801,
) -> np.ndarray:
    from dds.kernels import (
        _implicit_values_from_signed_distance,
        _line_sweep_signed_distance_at_parameters,
        _resolve_bead_profile,
    )

    profile = _resolve_bead_profile(deposit.profile, domain)
    xs, ys, zs = domain.grid_centers()
    points = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)
    start = deposit.start.position.to_array()
    end = deposit.end.position.to_array()
    start_axis = deposit.start.normal.to_array()
    end_axis = deposit.end.normal.to_array()
    best_distance = np.full(points.shape[0], np.inf, dtype=float)
    for parameter in np.linspace(0.0, 1.0, samples):
        parameters = np.full(points.shape[0], parameter, dtype=float)
        distance = _line_sweep_signed_distance_at_parameters(
            points,
            start=start,
            end=end,
            start_axis=start_axis,
            end_axis=end_axis,
            parameters=parameters,
            profile=profile,
        )
        best_distance = np.minimum(best_distance, distance)
    return _implicit_values_from_signed_distance(
        best_distance,
        profile.transition_width,
    ).reshape(domain.grid_shape)


def assert_no_threshold_underfill(
    field: np.ndarray,
    reference: np.ndarray,
    *,
    threshold: float = 0.5,
) -> None:
    underfilled = (reference >= threshold) & (field < threshold)
    assert not np.any(underfilled)


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
    deposit = PointDeposit(target=(2.5, 2.5, 3.5), profile=make_profile(width=2.0, height=2.0))
    density = Simulator(domain, [deposit]).result().implicit_field

    assert density[2, 2, 2] > 0.0
    assert density[0, 0, 0] == pytest.approx(0.0)


def test_point_deposit_target_marks_the_top_of_the_bead() -> None:
    domain = make_domain()
    deposit = PointDeposit(target=(2.5, 2.5, 3.5), profile=make_profile(width=2.0, height=2.0))
    density = Simulator(domain, [deposit]).result().implicit_field

    assert density[2, 2, 2] > density[2, 2, 3]
    assert density[2, 2, 3] == pytest.approx(0.5)


def test_point_deposit_uses_rounded_bead_geometry_not_ellipsoidal_falloff() -> None:
    domain = make_domain()
    deposit = PointDeposit(target=(2.5, 2.5, 3.5), profile=make_profile(width=4.0, height=2.0))
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


def test_line_deposit_parallel_to_normal_fills_centerline() -> None:
    domain = make_domain()
    axis = (0.0, 0.0, 1.0)
    deposit = LineDeposit(
        start=DepositionTarget((2.5, 2.5, 2.5), axis),
        end=DepositionTarget((2.5, 2.5, 7.5), axis),
        profile=make_profile(width=2.0, height=2.0),
    )

    field = simulate(domain, [deposit]).implicit_field

    np.testing.assert_allclose(field[2, 2, 1:7], 1.0, rtol=0.0, atol=0.0)
    assert field[2, 2, 7] == pytest.approx(0.5)


def test_line_deposit_parallel_to_normal_matches_long_rounded_cylinder() -> None:
    from dds.kernels import (
        _implicit_values_from_signed_distance,
        _resolve_bead_profile,
        top_referenced_rounded_cylinder_signed_distance,
    )

    domain = Domain.from_bounds(
        xmin=-1.5,
        xmax=1.5,
        ymin=-1.5,
        ymax=1.5,
        zmin=-1.0,
        zmax=4.0,
        voxel_size=0.5,
    )
    axis = (0.0, 0.0, 1.0)
    deposit = LineDeposit(
        start=DepositionTarget((0.0, 0.0, 1.0), axis),
        end=DepositionTarget((0.0, 0.0, 3.0), axis),
        profile=make_profile(width=1.0, height=1.0),
    )
    profile = _resolve_bead_profile(deposit.profile, domain)
    xs, ys, zs = domain.grid_centers()
    points = np.stack((xs, ys, zs), axis=-1)

    field = simulate(domain, [deposit]).implicit_field
    signed_distance = top_referenced_rounded_cylinder_signed_distance(
        points.reshape(-1, 3),
        target=deposit.end.position.to_array(),
        axis=deposit.end.normal.to_array(),
        height=profile.height
        + np.linalg.norm(
            deposit.end.position.to_array() - deposit.start.position.to_array()
        ),
        profile=profile,
    ).reshape(domain.grid_shape)
    expected = _implicit_values_from_signed_distance(
        signed_distance,
        profile.transition_width,
    )

    np.testing.assert_allclose(field, expected, rtol=0.0, atol=0.0)


def test_polyline_deposit_parallel_to_normal_segment_fills_centerline() -> None:
    domain = make_domain()
    axis = (0.0, 0.0, 1.0)
    deposit = PolylineDeposit(
        targets=(
            DepositionTarget((2.5, 2.5, 2.5), axis),
            DepositionTarget((2.5, 2.5, 7.5), axis),
            DepositionTarget((6.5, 2.5, 7.5), axis),
        ),
        profile=make_profile(width=2.0, height=2.0),
    )

    field = simulate(domain, [deposit]).implicit_field

    np.testing.assert_allclose(field[2, 2, 1:7], 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(field[2:7, 2, 6], 1.0, rtol=0.0, atol=0.0)


def test_polyline_deposit_with_one_varying_normal_segment_matches_line() -> None:
    domain = Domain.from_bounds(
        xmin=-2.0,
        xmax=3.0,
        ymin=-2.0,
        ymax=2.0,
        zmin=-1.0,
        zmax=6.0,
        voxel_size=0.25,
    )
    profile = BeadProfile(width=1.0, height=2.0)
    start = DepositionTarget((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
    end = DepositionTarget((0.0, 0.0, 4.0), (0.4, 0.0, 0.916515))
    line = LineDeposit(start=start, end=end, profile=profile)
    polyline = PolylineDeposit(targets=(start, end), profile=profile)

    line_field = simulate(domain, [line]).implicit_field
    polyline_field = simulate(domain, [polyline]).implicit_field

    np.testing.assert_allclose(polyline_field, line_field, rtol=0.0, atol=0.0)


def test_line_deposit_oblique_to_normal_fills_swept_interior() -> None:
    domain = Domain.from_bounds(
        xmin=-2.0,
        xmax=3.0,
        ymin=-2.0,
        ymax=2.0,
        zmin=-1.0,
        zmax=6.0,
        voxel_size=0.25,
    )
    axis = (0.0, 0.0, 1.0)
    deposit = LineDeposit(
        start=DepositionTarget((0.0, 0.0, 1.0), axis),
        end=DepositionTarget((3.0, 0.0, 3.0), axis),
        profile=BeadProfile(width=1.0, height=2.0),
    )

    field = simulate(domain, [deposit]).implicit_field
    reference = brute_force_line_sweep_field(domain, deposit)

    assert field[domain.world_to_index((0.375, -0.125, 1.125))] > 0.8
    assert_no_threshold_underfill(field, reference)


def test_line_deposit_with_varying_axes_fills_swept_interior() -> None:
    domain = Domain.from_bounds(
        xmin=-2.0,
        xmax=3.0,
        ymin=-2.0,
        ymax=2.0,
        zmin=-1.0,
        zmax=6.0,
        voxel_size=0.25,
    )
    deposit = LineDeposit(
        start=DepositionTarget((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        end=DepositionTarget((0.0, 0.0, 4.0), (0.4, 0.0, 0.916515)),
        profile=BeadProfile(width=1.0, height=2.0),
    )

    field = simulate(domain, [deposit]).implicit_field
    reference = brute_force_line_sweep_field(domain, deposit)

    assert field[domain.world_to_index((-0.625, -0.125, 1.625))] == pytest.approx(
        1.0
    )
    assert_no_threshold_underfill(field, reference)


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


def test_line_and_polyline_deposits_expose_sweep_resolution_control() -> None:
    profile = make_profile()
    line = LineDeposit(
        start=(0.0, 0.0, 0.0),
        end=(1.0, 0.0, 1.0),
        profile=profile,
        sweep_resolution=0.25,
    )
    polyline = PolylineDeposit(
        targets=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (2.0, 0.0, 1.0),
        ),
        profile=profile,
        sweep_resolution=0.5,
    )

    assert LineDeposit(
        start=(0.0, 0.0, 0.0),
        end=(1.0, 0.0, 1.0),
        profile=profile,
    ).sweep_resolution is None
    assert line.sweep_resolution == pytest.approx(0.25)
    assert polyline.sweep_resolution == pytest.approx(0.5)
    assert [
        segment.sweep_resolution for segment in polyline.segments()
    ] == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize("sweep_resolution", [0, -1, 0.0])
def test_deposit_sweep_resolution_rejects_non_positive_values(
    sweep_resolution: float,
) -> None:
    with pytest.raises(ValueError, match="sweep_resolution"):
        LineDeposit(
            start=(0.0, 0.0, 0.0),
            end=(1.0, 0.0, 1.0),
            profile=make_profile(),
            sweep_resolution=sweep_resolution,
        )


@pytest.mark.parametrize("sweep_resolution", [float("nan"), float("inf")])
def test_deposit_sweep_resolution_rejects_non_finite_values(
    sweep_resolution: float,
) -> None:
    with pytest.raises(ValueError, match="sweep_resolution"):
        LineDeposit(
            start=(0.0, 0.0, 0.0),
            end=(1.0, 0.0, 1.0),
            profile=make_profile(),
            sweep_resolution=sweep_resolution,
        )


@pytest.mark.parametrize("sweep_resolution", [True, "auto", "manual"])
def test_deposit_sweep_resolution_rejects_invalid_values(
    sweep_resolution: object,
) -> None:
    with pytest.raises(TypeError, match="sweep_resolution"):
        PolylineDeposit(
            targets=((0.0, 0.0, 0.0), (1.0, 0.0, 1.0)),
            profile=make_profile(),
            sweep_resolution=sweep_resolution,  # type: ignore[arg-type]
        )


def test_line_deposit_explicit_sweep_resolution_overrides_auto_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dds.kernels

    observed_counts: list[int] = []
    original_subdivide = dds.kernels._subdivide_line_deposit

    def tracked_subdivide(deposit: LineDeposit, count: int) -> tuple[LineDeposit, ...]:
        observed_counts.append(count)
        return original_subdivide(deposit, count)

    def unexpected_auto_count(*args: object, **kwargs: object) -> int:
        raise AssertionError("explicit resolution should bypass auto counting")

    monkeypatch.setattr(dds.kernels, "_subdivide_line_deposit", tracked_subdivide)
    monkeypatch.setattr(
        dds.kernels,
        "_line_sweep_subdivision_count",
        unexpected_auto_count,
    )
    domain = Domain.from_bounds(
        xmin=-1.0,
        xmax=4.0,
        ymin=-1.0,
        ymax=1.0,
        zmin=-1.0,
        zmax=4.0,
        voxel_size=0.5,
    )
    deposit = LineDeposit(
        start=DepositionTarget((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        end=DepositionTarget((3.0, 0.0, 3.0), (0.0, 0.0, 1.0)),
        profile=BeadProfile(width=1.0, height=2.0),
        sweep_resolution=1.0,
    )

    field = simulate(domain, [deposit]).implicit_field

    assert observed_counts == [4]
    assert float(field.max()) > 0.0


def test_line_deposit_auto_sweep_count_increases_with_finer_voxels() -> None:
    import dds.kernels

    profile = BeadProfile(width=4.0, height=4.0)
    deposit = LineDeposit(
        start=DepositionTarget((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        end=DepositionTarget((3.0, 0.0, 2.0), (0.0, 0.0, 1.0)),
        profile=profile,
    )
    coarse_domain = Domain.from_bounds(
        xmin=-1.0,
        xmax=4.0,
        ymin=-1.0,
        ymax=1.0,
        zmin=-1.0,
        zmax=3.0,
        voxel_size=1.0,
    )
    fine_domain = Domain.from_bounds(
        xmin=-1.0,
        xmax=4.0,
        ymin=-1.0,
        ymax=1.0,
        zmin=-1.0,
        zmax=3.0,
        voxel_size=0.25,
    )

    coarse_count = dds.kernels._line_sweep_subdivision_count(
        coarse_domain,
        deposit.start.position.to_array(),
        deposit.end.position.to_array(),
        deposit.start.normal.to_array(),
        deposit.end.normal.to_array(),
        dds.kernels._resolve_bead_profile(profile, coarse_domain),
    )
    fine_count = dds.kernels._line_sweep_subdivision_count(
        fine_domain,
        deposit.start.position.to_array(),
        deposit.end.position.to_array(),
        deposit.start.normal.to_array(),
        deposit.end.normal.to_array(),
        dds.kernels._resolve_bead_profile(profile, fine_domain),
    )

    assert fine_count > coarse_count


def test_line_deposit_auto_sweep_count_increases_with_smaller_beads() -> None:
    import dds.kernels

    domain = Domain.from_bounds(
        xmin=-1.0,
        xmax=4.0,
        ymin=-1.0,
        ymax=1.0,
        zmin=-1.0,
        zmax=3.0,
        voxel_size=1.0,
    )
    large_profile = BeadProfile(width=4.0, height=4.0)
    small_profile = BeadProfile(width=0.4, height=4.0)
    deposit = LineDeposit(
        start=DepositionTarget((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        end=DepositionTarget((3.0, 0.0, 2.0), (0.0, 0.0, 1.0)),
        profile=large_profile,
    )

    large_count = dds.kernels._line_sweep_subdivision_count(
        domain,
        deposit.start.position.to_array(),
        deposit.end.position.to_array(),
        deposit.start.normal.to_array(),
        deposit.end.normal.to_array(),
        dds.kernels._resolve_bead_profile(large_profile, domain),
    )
    small_count = dds.kernels._line_sweep_subdivision_count(
        domain,
        deposit.start.position.to_array(),
        deposit.end.position.to_array(),
        deposit.start.normal.to_array(),
        deposit.end.normal.to_array(),
        dds.kernels._resolve_bead_profile(small_profile, domain),
    )

    assert small_count > large_count


def test_line_deposit_falls_back_to_minimization_for_pathological_subdivision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dds.kernels

    calls = 0
    original = dds.kernels._minimized_line_sweep_signed_distance

    def tracked_minimization(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(dds.kernels, "_LINE_SWEEP_MAX_SUBSEGMENTS", 1)
    monkeypatch.setattr(
        dds.kernels,
        "_minimized_line_sweep_signed_distance",
        tracked_minimization,
    )
    domain = Domain.from_bounds(
        xmin=-1.0,
        xmax=4.0,
        ymin=-1.0,
        ymax=1.0,
        zmin=-1.0,
        zmax=4.0,
        voxel_size=0.5,
    )
    deposit = LineDeposit(
        start=DepositionTarget((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        end=DepositionTarget((3.0, 0.0, 3.0), (0.0, 0.0, 1.0)),
        profile=BeadProfile(width=1.0, height=2.0),
    )

    field = simulate(domain, [deposit]).implicit_field

    assert calls > 0
    assert float(field.max()) > 0.0


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
    reference = brute_force_line_sweep_field(domain, deposit)

    assert density[domain.world_to_index((0.875, 0.625, 3.125))] > 0.0
    assert float(density.max()) == pytest.approx(1.0)
    assert_no_threshold_underfill(density, reference)


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
    deposit = PointDeposit(target=(2.5, 2.5, 3.5), profile=make_profile(width=2.0, height=2.0))
    single = Simulator(domain, [deposit]).result().analysis.deposition_index_field()
    overlap = Simulator(domain, [deposit, deposit]).result().analysis.deposition_index_field()

    # single deposit → index 0 for touched voxels, -1 for untouched
    assert int(single.max()) == 0
    # two deposits (same position) → last index is 1 for touched voxels
    assert int(overlap.max()) == 1


def test_thresholding_changes_occupied_voxel_count() -> None:
    domain = make_domain()
    deposit = PointDeposit(target=(2.5, 2.5, 4.0), profile=make_profile(width=3.0, height=3.0))
    result = Simulator(domain, [deposit]).result()

    low_threshold = result.analysis.occupancy(threshold=0.1)
    high_threshold = result.analysis.occupancy(threshold=0.8)

    np.testing.assert_array_equal(low_threshold, result.implicit_field >= 0.1)
    np.testing.assert_array_equal(high_threshold, result.implicit_field >= 0.8)
    assert int(low_threshold.sum()) > int(high_threshold.sum())
    assert np.any(low_threshold & ~high_threshold)


def test_deposits_outside_bounds_are_skipped_and_partial_overlap_is_kept() -> None:
    domain = make_domain()
    outside = PointDeposit(target=(-5.0, -5.0, -5.0), profile=make_profile())
    partial = LineDeposit(
        start=(-0.5, 2.5, 3.5),
        end=(2.5, 2.5, 3.5),
        profile=make_profile(),
    )

    outside_density = Simulator(domain, [outside]).result().analysis.deposition_index_field()
    partial_occupancy = Simulator(domain, [partial]).result().analysis.occupancy(threshold=0.25)

    assert np.all(outside_density == -1), "deposit outside bounds should leave all voxels untouched (-1)"
    assert partial_occupancy[0, 2, 2]
    assert partial_occupancy[2, 2, 2]


def test_zero_length_line_matches_point_deposit() -> None:
    domain = make_domain()
    profile = make_profile(width=2.0, height=2.0)
    point = PointDeposit(target=(2.5, 2.5, 3.5), profile=profile)
    zero_length_line = LineDeposit(
        start=(2.5, 2.5, 3.5),
        end=(2.5, 2.5, 3.5),
        profile=profile,
    )

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
    deposit = PointDeposit(target=(2.5, 2.5, 3.5), profile=make_profile(width=2.0, height=2.0))
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
        PointDeposit(target=(2.5, 2.5, 3.5), profile=make_profile(width=2.0, height=2.0)),
        PointDeposit(target=(2.5, 2.5, 3.5), profile=make_profile(width=2.0, height=2.0)),
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
    deposits = [
        PointDeposit(target=(2.5, 2.5, 3.5), profile=profile),
        PointDeposit(target=(2.5, 2.5, 3.5), profile=profile),
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
        PointDeposit(target=(2.5, 2.5, 3.5), profile=profile),
        PointDeposit(target=(5.5, 5.5, 3.5), profile=profile),
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
