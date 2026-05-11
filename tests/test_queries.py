from __future__ import annotations

import numpy as np
import pytest

from dds import AnalysisBundle, DepositionAttributes, Domain, LineDeposit, PointDeposit, Simulator, analysis_bundle
from dds.queries import (
    contains_point,
    sample_density_at,
    sample_deposition_index_at,
    sample_points,
    signed_distance_at,
    surface_normal_at,
)


def make_domain() -> Domain:
    return Domain.from_bounds(
        xmin=0.0,
        xmax=10.0,
        ymin=0.0,
        ymax=10.0,
        zmin=0.0,
        zmax=4.0,
        voxel_size=0.5,
    )


def make_simulator() -> Simulator:
    attrs = DepositionAttributes(width=1.2, height=0.8, layer_id=0)
    deposits = [
        PointDeposit(x=2.25, y=2.25, z=0.25, attributes=attrs),
        LineDeposit(start=(2.25, 2.25, 0.25), end=(6.25, 2.25, 0.25), attributes=attrs),
    ]
    return Simulator(make_domain(), deposits)


def test_analysis_bundle_queries_cover_density_occupancy_sdf_and_points() -> None:
    simulator = make_simulator()
    bundle = simulator.analysis_bundle()

    inside = (2.25, 2.25, 0.25)
    outside = (9.0, 9.0, 3.5)

    assert isinstance(bundle, AnalysisBundle)
    assert analysis_bundle(simulator) is bundle
    assert contains_point(simulator, inside, representation="occupancy", threshold=0.5)
    assert contains_point(bundle, inside, representation="density", threshold=0.5, interpolation="trilinear")
    assert contains_point(bundle, inside, representation="sdf", threshold=0.5)
    assert contains_point(bundle, inside, representation="mesh", threshold=0.5)
    assert not contains_point(bundle, outside, representation="occupancy", threshold=0.5)
    assert signed_distance_at(bundle, inside, threshold=0.5) <= 0.0
    assert signed_distance_at(bundle, outside, threshold=0.5) > 0.0
    assert sample_density_at(bundle, inside, interpolation="nearest") >= sample_density_at(
        bundle,
        outside,
        interpolation="nearest",
    )
    assert sample_deposition_index_at(bundle, inside, interpolation="trilinear") > 0.0

    sampled = sample_points(
        bundle,
        np.asarray([inside, outside], dtype=float),
        fields=("density", "occupancy", "deposition_index", "signed_distance"),
        threshold=0.5,
        interpolation="trilinear",
    )
    assert set(sampled) == {"density", "occupancy", "deposition_index", "signed_distance"}
    assert sampled["density"].shape == (2,)
    assert bool(sampled["occupancy"][0]) is True
    assert bool(sampled["occupancy"][1]) is False

    normal = surface_normal_at(bundle, (6.75, 2.25, 0.25), threshold=0.5)
    assert np.isclose(np.linalg.norm(normal), 1.0, atol=0.2)
    assert normal[0] > 0.0


def test_analysis_bundle_cache_invalidates_after_deposit_changes() -> None:
    simulator = make_simulator()
    before = simulator.analysis_bundle()

    assert simulator.analysis_bundle() is before

    simulator.add_deposit(
        PointDeposit(
            x=4.25,
            y=4.25,
            z=0.25,
            attributes=DepositionAttributes(width=1.2, height=0.8, layer_id=1),
        )
    )

    after = simulator.analysis_bundle()
    assert after is not before
    assert float(after.density_field().sum()) > float(before.density_field().sum())


def test_trilinear_sampling_differs_from_nearest_near_gradient_regions() -> None:
    simulator = make_simulator()
    point = (2.6, 2.25, 0.25)

    nearest = simulator.sample_density_at(point, interpolation="nearest")
    trilinear = simulator.sample_density_at(point, interpolation="trilinear")

    assert trilinear != pytest.approx(nearest)
