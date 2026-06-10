"""Tests for ChunkedField and accumulate_chunked_field."""

from __future__ import annotations

import numpy as np
import pytest

from dds import (
    BeadProfile,
    ChunkedField,
    Domain,
    LineDeposit,
    PointDeposit,
    PolylineDeposit,
    Pose3D,
    simulate,
)
from dds.fields import accumulate_chunked_field


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


def make_profile(width: float = 2.0, height: float = 2.0) -> BeadProfile:
    return BeadProfile(width=width, height=height)


def make_deposits() -> list[PointDeposit]:
    profile = make_profile()
    return [
        PointDeposit(target=(2.5, 2.5, 3.5), profile=profile),
        PointDeposit(target=(7.5, 7.5, 3.5), profile=profile),
    ]


def test_chunked_to_dense_max_matches_simulate_max() -> None:
    domain = make_domain()
    deposits = make_deposits()

    chunked = accumulate_chunked_field(domain, deposits)

    np.testing.assert_allclose(
        chunked.to_dense("max"),
        simulate(domain, deposits).field("max"),
    )


def test_chunked_to_dense_coverage_matches_dense_accumulation() -> None:
    domain = make_domain()
    deposits = make_deposits()

    chunked = accumulate_chunked_field(domain, deposits, compositions=("max", "coverage"))
    expected = simulate(
        domain,
        deposits,
        compositions=("max", "coverage"),
    )

    assert expected.coverage is not None
    np.testing.assert_allclose(chunked.to_dense("coverage"), expected.coverage)


def test_chunked_to_dense_all_matches_individual_calls() -> None:
    chunked = accumulate_chunked_field(
        make_domain(),
        make_deposits(),
        compositions=("max", "coverage"),
    )

    all_fields = chunked.to_dense_all("max", "coverage")

    np.testing.assert_allclose(all_fields["max"], chunked.to_dense("max"))
    np.testing.assert_allclose(
        all_fields["coverage"],
        chunked.to_dense("coverage"),
    )


def test_chunked_to_dense_all_raises_on_empty_compositions() -> None:
    chunked = accumulate_chunked_field(make_domain(), make_deposits())

    with pytest.raises(ValueError, match="composition"):
        chunked.to_dense_all()


def test_chunked_empty_and_clear_behavior() -> None:
    domain = make_domain()
    chunked = ChunkedField(domain)
    np.testing.assert_array_equal(
        chunked.to_dense(),
        np.zeros(domain.grid_shape),
    )

    populated = accumulate_chunked_field(domain, make_deposits())
    assert populated.chunk_count > 0
    populated.clear()

    assert populated.chunk_count == 0
    assert populated.event_count == 0
    assert populated.to_dense().sum() == pytest.approx(0.0)


def test_chunked_memory_and_activity_diagnostics() -> None:
    domain = make_domain()
    chunked = accumulate_chunked_field(domain, make_deposits())

    assert chunked.nbytes > 0
    assert chunked.dense_field_nbytes == int(np.prod(domain.grid_shape)) * 8
    assert chunked.dense_nbytes == int(np.prod(domain.grid_shape)) * 8
    assert chunked.event_count == 2
    assert 0 < chunked.active_voxel_count <= chunked.allocated_voxel_count
    assert 0.0 < chunked.active_fraction <= chunked.allocation_fraction <= 1.0
    assert 0.0 < chunked.memory_ratio <= 1.0


def test_chunked_out_of_bounds_deposit_is_skipped() -> None:
    domain = make_domain()
    deposit = PointDeposit(
        target=(-50.0, -50.0, -50.0),
        profile=make_profile(),
    )

    chunked = accumulate_chunked_field(domain, [deposit])

    assert chunked.event_count == 0
    assert chunked.chunk_count == 0


def test_chunked_roi_materialization_matches_dense_slice() -> None:
    chunked = accumulate_chunked_field(make_domain(), make_deposits())
    bounds = ((1, 5), (1, 6), (1, 7))

    roi = chunked.materialize("max", index_bounds=bounds)
    dense = chunked.to_dense("max")

    np.testing.assert_allclose(roi, dense[1:5, 1:6, 1:7])


def test_chunked_diagonal_path_uses_subset_of_domain_memory() -> None:
    domain = Domain.from_bounds(
        xmin=0.0,
        xmax=64.0,
        ymin=0.0,
        ymax=64.0,
        zmin=0.0,
        zmax=64.0,
        voxel_size=1.0,
    )
    deposit = LineDeposit(
        start=(2.5, 2.5, 2.5),
        end=(61.5, 61.5, 61.5),
        profile=make_profile(),
    )
    chunked = accumulate_chunked_field(
        domain,
        [deposit],
        chunk_shape=(8, 8, 8),
    )

    assert chunked.chunk_count < 8**3
    assert chunked.memory_ratio < 0.2
    np.testing.assert_allclose(
        chunked.to_dense("max"),
        simulate(domain, [deposit]).density_max,
    )


def test_polyline_event_merges_internal_segments_across_chunk_boundaries() -> None:
    domain = make_domain()
    deposit = PolylineDeposit(
        poses=(
            Pose3D((1.5, 1.5, 3.5)),
            Pose3D((5.5, 1.5, 3.5)),
            Pose3D((5.5, 6.5, 3.5)),
        ),
        profile=make_profile(),
    )
    chunked = accumulate_chunked_field(
        domain,
        [deposit],
        chunk_shape=(2, 2, 2),
        compositions=("max", "coverage"),
    )
    result = simulate(
        domain,
        [deposit],
        compositions=("max", "coverage"),
    )

    assert result.coverage is not None
    np.testing.assert_allclose(chunked.to_dense("max"), result.density_max)
    np.testing.assert_allclose(chunked.to_dense("coverage"), result.coverage)
    np.testing.assert_allclose(
        chunked.to_dense("coverage"),
        chunked.to_dense("max"),
    )


def test_max_only_chunked_field_uses_half_the_dual_composition_memory() -> None:
    domain = make_domain()
    deposits = make_deposits()
    max_only = accumulate_chunked_field(domain, deposits)
    both = accumulate_chunked_field(
        domain,
        deposits,
        compositions=("max", "coverage"),
    )

    assert both.nbytes == 2 * max_only.nbytes
    np.testing.assert_allclose(max_only.to_dense("max"), both.to_dense("max"))
