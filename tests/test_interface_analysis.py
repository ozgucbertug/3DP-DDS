from __future__ import annotations

import numpy as np

from dds import BeadProfile, DepositionMetadata, Domain, PointDeposit, simulate
from dds.analysis import interface


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
        PointDeposit(x=2.5, y=2.5, z=3.5, profile=profile, metadata=DepositionMetadata(layer_id=0)),
        PointDeposit(x=2.5, y=2.5, z=3.5, profile=profile, metadata=DepositionMetadata(layer_id=1)),
    ]
    result = simulate(domain, deposits, threshold=0.5)

    analysis = result.interface(mode="auto", threshold=0.5)

    assert analysis.stratification_mode == "layer"
    assert analysis.stratum_ids == (0, 1)
    assert analysis.contact_mask.any()
    assert analysis.overlap_voxel_count > 0
    assert len(analysis.pair_summaries) == 1


def test_interface_analysis_falls_back_to_deposit_order() -> None:
    domain = make_domain()
    profile = make_profile()
    deposits = [
        PointDeposit(x=2.5, y=2.5, z=3.5, profile=profile, metadata=DepositionMetadata()),
        PointDeposit(x=2.5, y=2.5, z=3.5, profile=profile, metadata=DepositionMetadata()),
        PointDeposit(x=4.5, y=2.5, z=3.5, profile=profile, metadata=DepositionMetadata()),
    ]
    result = simulate(domain, deposits, threshold=0.5)

    analysis = interface(result, mode="auto", threshold=0.5)

    assert analysis.stratification_mode == "order"
    assert analysis.stratum_ids == (0, 1, 2)
    assert len(analysis.pair_summaries) == 2
    assert analysis.contact_mask.dtype == np.bool_
    assert analysis.overlap_mask.dtype == np.bool_
    assert analysis.unsupported_next_mask.dtype == np.bool_
    assert analysis.contact_area >= 0.0
    assert 0.0 <= analysis.overlap_fraction <= 1.0
