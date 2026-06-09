"""Tests for SparseDensityField and accumulate_density_sparse."""

from __future__ import annotations

import numpy as np
import pytest

from dds import (
    BeadProfile,
    Domain,
    PointDeposit,
    Simulator,
    SparseDensityField,
    accumulate_density_sparse,
    simulate,
)


def make_domain() -> Domain:
    return Domain.from_bounds(
        xmin=0.0, xmax=10.0, ymin=0.0, ymax=10.0, zmin=0.0, zmax=10.0, voxel_size=1.0
    )


def make_profile(width: float = 2.0, height: float = 2.0) -> BeadProfile:
    return BeadProfile(width=width, height=height)


def make_deposits() -> list[PointDeposit]:
    profile = make_profile()
    return [
        PointDeposit(x=2.5, y=2.5, z=3.5, profile=profile),
        PointDeposit(x=7.5, y=7.5, z=3.5, profile=profile),
    ]


# ---------------------------------------------------------------------------
# accumulate_density_sparse: standalone function
# ---------------------------------------------------------------------------


def test_sparse_to_dense_max_matches_simulate_max() -> None:
    """to_dense('max') must reproduce the max-composition dense grid."""
    domain = make_domain()
    deposits = make_deposits()

    sparse = accumulate_density_sparse(domain, deposits)
    dense_max = sparse.to_dense(composition="max")

    expected = simulate(domain, deposits).field("max")
    np.testing.assert_allclose(dense_max, expected)


def test_sparse_to_dense_coverage_matches_accumulate_field() -> None:
    """to_dense('coverage') must reproduce the additive coverage grid."""
    from dds.fields import accumulate_field

    domain = make_domain()
    deposits = make_deposits()

    sparse = accumulate_density_sparse(domain, deposits)
    dense_coverage = sparse.to_dense(composition="coverage")

    expected = accumulate_field(domain, deposits, composition="coverage")
    np.testing.assert_allclose(dense_coverage, expected)


def test_sparse_to_dense_all_matches_individual_calls() -> None:
    """to_dense_all produces the same grids as separate to_dense calls."""
    domain = make_domain()
    deposits = make_deposits()

    sparse = accumulate_density_sparse(domain, deposits)
    all_grids = sparse.to_dense_all("max", "coverage")

    np.testing.assert_allclose(all_grids["max"], sparse.to_dense("max"))
    np.testing.assert_allclose(all_grids["coverage"], sparse.to_dense("coverage"))


def test_sparse_to_dense_all_raises_on_empty_compositions() -> None:
    domain = make_domain()
    sparse = accumulate_density_sparse(domain, make_deposits())
    with pytest.raises(ValueError, match="composition"):
        sparse.to_dense_all()


def test_sparse_empty_field_is_zero() -> None:
    """An empty SparseDensityField materialises as an all-zeros grid."""
    domain = make_domain()
    sparse = SparseDensityField(domain)

    np.testing.assert_array_equal(sparse.to_dense(), np.zeros(domain.grid_shape))


def test_sparse_clear_resets_to_empty() -> None:
    domain = make_domain()
    deposits = make_deposits()

    sparse = accumulate_density_sparse(domain, deposits)
    assert sparse.contribution_count > 0

    sparse.clear()
    assert sparse.contribution_count == 0
    assert sparse.to_dense().sum() == pytest.approx(0.0)


def test_sparse_nbytes_and_sparsity_diagnostics() -> None:
    domain = make_domain()
    deposits = make_deposits()

    sparse = accumulate_density_sparse(domain, deposits)

    assert sparse.nbytes > 0
    assert sparse.dense_nbytes == int(np.prod(domain.grid_shape)) * 8
    assert sparse.contribution_count == len(deposits)
    assert 0.0 < sparse.sparsity <= 1.0


def test_sparse_out_of_bounds_deposit_is_skipped() -> None:
    """A deposit entirely outside the domain contributes zero kernels."""
    domain = make_domain()
    deposit = PointDeposit(x=-50.0, y=-50.0, z=-50.0, profile=make_profile())

    sparse = accumulate_density_sparse(domain, [deposit])
    assert sparse.contribution_count == 0
    assert sparse.to_dense().sum() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Simulator.sparse_field(): lazy + incremental
# ---------------------------------------------------------------------------


def test_simulator_sparse_field_matches_accumulate_density_sparse() -> None:
    """Simulator.sparse_field().to_dense() must match the standalone function."""
    domain = make_domain()
    deposits = make_deposits()

    sim = Simulator(domain, deposits)
    dense_via_simulator = sim.sparse_field().to_dense(composition="max")
    dense_standalone = accumulate_density_sparse(domain, deposits).to_dense(composition="max")

    np.testing.assert_allclose(dense_via_simulator, dense_standalone)


def test_simulator_sparse_field_is_cached() -> None:
    """sparse_field() returns the same object on repeated calls."""
    domain = make_domain()
    sim = Simulator(domain, make_deposits())

    assert sim.sparse_field() is sim.sparse_field()


def test_simulator_sparse_field_updated_incrementally() -> None:
    """Adding a deposit after sparse_field() is warm appends one contribution."""
    domain = make_domain()
    deposits = make_deposits()

    sim = Simulator(domain, deposits[:1])
    _ = sim.sparse_field()  # warm up
    count_before = sim.sparse_field().contribution_count

    sim.add_deposit(deposits[1])
    count_after = sim.sparse_field().contribution_count

    assert count_after == count_before + 1


def test_simulator_sparse_field_incremental_matches_batch() -> None:
    """Dense grid from incrementally built sparse cache equals the batch result."""
    domain = make_domain()
    deposits = make_deposits()

    sim = Simulator(domain, deposits[:1])
    _ = sim.sparse_field()  # warm up
    sim.add_deposit(deposits[1])

    expected = accumulate_density_sparse(domain, deposits).to_dense("max")
    np.testing.assert_allclose(sim.sparse_field().to_dense("max"), expected)


def test_simulator_clear_resets_sparse_field() -> None:
    """clear_deposits() empties the sparse cache contributions."""
    domain = make_domain()
    deposits = make_deposits()

    sim = Simulator(domain, deposits)
    _ = sim.sparse_field()  # warm up

    sim.clear_deposits()
    assert sim.sparse_field().contribution_count == 0
    assert sim.sparse_field().to_dense().sum() == pytest.approx(0.0)


def test_simulator_sparse_and_dense_agree_after_mixed_operations() -> None:
    """Sparse max field agrees with dense max field after clear + partial re-add."""
    domain = make_domain()
    deposits = make_deposits()

    sim = Simulator(domain, deposits)
    _ = sim.sparse_field()  # warm up sparse

    sim.clear_deposits()
    sim.add_deposit(deposits[0])

    dense_max = sim.sample_field(field="density")
    sparse_max = sim.sparse_field().to_dense("max")
    np.testing.assert_allclose(sparse_max, dense_max)
