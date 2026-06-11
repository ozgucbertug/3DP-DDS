from __future__ import annotations

import numpy as np

from dds import BeadProfile, DepositionMetadata, Domain, PointDeposit, simulate
from dds.analysis.interface import _contact_for_pair


def make_domain() -> Domain:
    return Domain.from_bounds(
        xmin=0.0,
        xmax=10.0,
        ymin=0.0,
        ymax=10.0,
        zmin=0.0,
        zmax=8.0,
        voxel_size=1.0,
    )


def make_profile() -> BeadProfile:
    return BeadProfile(width=2.0, height=2.0)


def test_interface_analysis_uses_real_layers_when_available() -> None:
    domain = make_domain()
    profile = make_profile()
    deposits = [
        PointDeposit(target=(2.5, 2.5, 3.5), profile=profile, metadata=DepositionMetadata(layer_id=0)),
        PointDeposit(target=(2.5, 2.5, 3.5), profile=profile, metadata=DepositionMetadata(layer_id=1)),
    ]
    result = simulate(domain, deposits, threshold=0.5)

    analysis = result.analysis.interface(mode="auto", threshold=0.5)

    assert analysis.stratification_mode == "layer"
    assert analysis.stratum_ids == (0, 1)
    assert analysis.contact_mask.any()
    assert analysis.overlap_voxel_count > 0
    assert len(analysis.pair_summaries) == 1


def test_interface_analysis_falls_back_to_deposit_order() -> None:
    domain = make_domain()
    profile = make_profile()
    deposits = [
        PointDeposit(target=(2.5, 2.5, 3.5), profile=profile, metadata=DepositionMetadata()),
        PointDeposit(target=(2.5, 2.5, 3.5), profile=profile, metadata=DepositionMetadata()),
        PointDeposit(target=(4.5, 2.5, 3.5), profile=profile, metadata=DepositionMetadata()),
    ]
    result = simulate(domain, deposits, threshold=0.5)

    analysis = result.analysis.interface(mode="auto", threshold=0.5)

    assert analysis.stratification_mode == "order"
    assert analysis.stratum_ids == (0, 1, 2)
    assert len(analysis.pair_summaries) == 2
    assert analysis.contact_mask.dtype == np.bool_
    assert analysis.overlap_mask.dtype == np.bool_
    assert analysis.unsupported_next_mask.dtype == np.bool_
    assert analysis.contact_area >= 0.0
    assert 0.0 <= analysis.overlap_fraction <= 1.0


def test_contact_for_pair_reports_exact_adjacent_face_geometry() -> None:
    previous = np.zeros((3, 3, 3), dtype=bool)
    next_field = np.zeros_like(previous)
    previous[1, 1, 1] = True
    next_field[2, 1, 1] = True

    contact, face_count, contact_area = _contact_for_pair(
        previous,
        next_field,
        voxel_size=(2.0, 3.0, 4.0),
    )

    expected = np.zeros_like(previous)
    expected[2, 1, 1] = True
    np.testing.assert_array_equal(contact, expected)
    assert face_count == 1
    assert contact_area == 12.0
