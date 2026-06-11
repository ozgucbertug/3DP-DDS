from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from dds import BeadProfile, DepositionMetadata, Domain, LineDeposit, PointDeposit, Simulator
from dds.analysis import SimulationAnalysis


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


def make_result():
    profile = BeadProfile(width=1.2, height=0.8)
    metadata = DepositionMetadata(layer_id=0)
    deposits = [
        PointDeposit(target=(2.25, 2.25, 0.65), profile=profile, metadata=metadata),
        LineDeposit(
            start=(2.25, 2.25, 0.65),
            end=(6.25, 2.25, 0.65),
            profile=profile,
            metadata=metadata,
        ),
    ]
    return Simulator(make_domain(), deposits).result()


def test_result_analysis_queries_implicit_occupancy_sdf_and_points() -> None:
    analysis = make_result().analysis
    inside = (2.25, 2.25, 0.25)
    outside = (9.0, 9.0, 3.5)

    assert isinstance(analysis, SimulationAnalysis)
    assert analysis.contains_point(inside, representation="occupancy")
    assert analysis.contains_point(inside, representation="implicit", interpolation="trilinear")
    assert analysis.contains_point(inside, representation="sdf")
    with pytest.raises(ValueError, match="not watertight"):
        analysis.contains_point(inside, representation="mesh")
    assert not analysis.contains_point(outside)
    assert analysis.signed_distance_at(inside) <= 0.0
    assert analysis.signed_distance_at(outside) > 0.0
    assert analysis.sample_implicit_value(inside) >= analysis.sample_implicit_value(outside)
    assert analysis.sample_deposition_index(inside) == 1

    sampled = analysis.sample_points(
        np.asarray([inside, outside], dtype=float),
        fields=("implicit", "occupancy", "deposition_index", "signed_distance"),
        interpolation="trilinear",
    )
    assert set(sampled) == {"implicit", "occupancy", "deposition_index", "signed_distance"}
    assert bool(sampled["occupancy"][0]) is True
    assert bool(sampled["occupancy"][1]) is False


def test_deposition_index_sampling_is_nearest_only_and_integer() -> None:
    domain = Domain.from_bounds(
        xmin=0.0, xmax=4.0, ymin=0.0, ymax=4.0, zmin=0.0, zmax=4.0, voxel_size=1.0
    )
    profile = BeadProfile(width=1.0, height=1.0)
    deposits = (
        PointDeposit(target=(1.5, 1.5, 1.5), profile=profile),
        PointDeposit(target=(2.5, 1.5, 1.5), profile=profile),
    )
    analysis = SimulationAnalysis(
        domain,
        np.zeros(domain.grid_shape),
        deposits,
    )

    value = analysis.sample_deposition_index((2.0, 1.5, 1.5))

    assert value == 1
    assert isinstance(value, int)


def test_analysis_owns_read_only_snapshots() -> None:
    domain = make_domain()
    implicit_field = np.zeros(domain.grid_shape, dtype=float)
    analysis = SimulationAnalysis(domain, implicit_field, ())
    implicit_field.fill(1.0)

    assert float(analysis.implicit_field.max()) == pytest.approx(0.0)
    assert int(analysis.deposition_index_field().max()) == -1
    with pytest.raises(ValueError):
        analysis.implicit_field[0, 0, 0] = 1.0
    with pytest.raises((FrozenInstanceError, TypeError)):
        analysis.implicit_field = np.ones(domain.grid_shape)


def test_occupancy_does_not_construct_deposition_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dds.fields

    calls = 0
    original = dds.fields.accumulate_deposition_index

    def tracked(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(dds.fields, "accumulate_deposition_index", tracked)
    analysis = make_result().analysis

    assert analysis.occupancy().any()
    assert calls == 0
    assert analysis.deposition_index_field().max() >= 0
    assert calls == 1
    analysis.deposition_index_field()
    assert calls == 1


def test_result_analysis_is_cached_and_simulator_results_are_isolated() -> None:
    simulator = Simulator(make_domain())
    simulator.add_deposit(
        PointDeposit(target=(2.25, 2.25, 0.65), profile=BeadProfile(width=1.2, height=0.8))
    )
    before = simulator.result()
    assert before.analysis is before.analysis
    assert np.shares_memory(
        before.implicit_field,
        before.analysis.implicit_field,
    )

    simulator.add_deposit(
        PointDeposit(target=(4.25, 4.25, 0.65), profile=BeadProfile(width=1.2, height=0.8))
    )
    after = simulator.result()

    assert float(after.implicit_field.sum()) > float(before.implicit_field.sum())
