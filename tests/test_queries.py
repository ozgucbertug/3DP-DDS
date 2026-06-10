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
        PointDeposit(x=2.25, y=2.25, z=0.65, profile=profile, metadata=metadata),
        LineDeposit(
            start=(2.25, 2.25, 0.65),
            end=(6.25, 2.25, 0.65),
            profile=profile,
            metadata=metadata,
        ),
    ]
    return Simulator(make_domain(), deposits).result()


def test_result_analysis_queries_density_occupancy_sdf_and_points() -> None:
    analysis = make_result().analysis
    inside = (2.25, 2.25, 0.25)
    outside = (9.0, 9.0, 3.5)

    assert isinstance(analysis, SimulationAnalysis)
    assert analysis.contains_point(inside, representation="occupancy")
    assert analysis.contains_point(inside, representation="density", interpolation="trilinear")
    assert analysis.contains_point(inside, representation="sdf")
    with pytest.raises(ValueError, match="not watertight"):
        analysis.contains_point(inside, representation="mesh")
    assert not analysis.contains_point(outside)
    assert analysis.signed_distance_at(inside) <= 0.0
    assert analysis.signed_distance_at(outside) > 0.0
    assert analysis.sample_density_at(inside) >= analysis.sample_density_at(outside)
    assert analysis.sample_deposition_index(inside) == 1

    sampled = analysis.sample_points(
        np.asarray([inside, outside], dtype=float),
        fields=("density", "occupancy", "deposition_index", "signed_distance"),
        interpolation="trilinear",
    )
    assert set(sampled) == {"density", "occupancy", "deposition_index", "signed_distance"}
    assert bool(sampled["occupancy"][0]) is True
    assert bool(sampled["occupancy"][1]) is False


def test_deposition_index_sampling_is_nearest_only_and_integer() -> None:
    domain = Domain.from_bounds(
        xmin=0.0, xmax=4.0, ymin=0.0, ymax=4.0, zmin=0.0, zmax=4.0, voxel_size=1.0
    )
    deposition_index = np.full(domain.grid_shape, -1, dtype=np.intp)
    deposition_index[1, 1, 1] = 0
    deposition_index[2, 1, 1] = 1
    analysis = SimulationAnalysis(
        domain,
        np.zeros(domain.grid_shape),
        deposition_index,
        (),
    )

    value = analysis.sample_deposition_index((2.0, 1.5, 1.5))

    assert value == 1
    assert isinstance(value, int)


def test_analysis_owns_read_only_snapshots() -> None:
    domain = make_domain()
    density = np.zeros(domain.grid_shape, dtype=float)
    deposition_index = np.full(domain.grid_shape, -1, dtype=np.intp)
    analysis = SimulationAnalysis(domain, density, deposition_index, ())
    density.fill(1.0)
    deposition_index.fill(5)

    assert float(analysis.density.max()) == pytest.approx(0.0)
    assert int(analysis.deposition_index.max()) == -1
    with pytest.raises(ValueError):
        analysis.density[0, 0, 0] = 1.0
    with pytest.raises(FrozenInstanceError):
        analysis.density = np.ones(domain.grid_shape)


def test_result_analysis_is_cached_and_simulator_results_are_isolated() -> None:
    simulator = Simulator(make_domain())
    simulator.add_deposit(
        PointDeposit(x=2.25, y=2.25, z=0.65, profile=BeadProfile(width=1.2, height=0.8))
    )
    before = simulator.result()
    assert before.analysis is before.analysis

    simulator.add_deposit(
        PointDeposit(x=4.25, y=4.25, z=0.65, profile=BeadProfile(width=1.2, height=0.8))
    )
    after = simulator.result()

    assert float(after.density_max.sum()) > float(before.density_max.sum())
