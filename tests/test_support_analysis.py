from __future__ import annotations

import numpy as np
import pytest

import dds.mesh_analysis as mesh_analysis
from dds import BeadProfile, DepositionMetadata, Domain, LineDeposit, PointDeposit, simulate
from dds.analysis.support import _support_shadow_field


def _cantilever_result():
    profile = BeadProfile(width=1.2, height=0.8)
    deposits = [
        PointDeposit(target=(2.0, 2.0, 1.0), profile=profile, metadata=DepositionMetadata(layer_id=0)),
        LineDeposit(
            start=(2.0, 2.0, 1.0),
            end=(6.0, 2.0, 1.0),
            profile=profile,
            metadata=DepositionMetadata(layer_id=0),
        ),
    ]
    domain = Domain.from_bounds(
        xmin=0.0,
        xmax=8.0,
        ymin=0.0,
        ymax=4.0,
        zmin=-1.0,
        zmax=3.0,
        voxel_size=0.25,
    )
    return simulate(domain, deposits, threshold=0.5)


def test_support_analysis_builds_shadow_for_axis_aligned_direction() -> None:
    result = _cantilever_result()
    analysis = result.analysis.support(build_direction="+Z", threshold=0.5)

    assert analysis.overhang_angles.shape == analysis.face_areas.shape
    assert analysis.downfacing_mask.shape == analysis.face_areas.shape
    assert analysis.support_risk_mask.shape == analysis.face_areas.shape
    assert analysis.shadow_voxel_count > 0
    assert analysis.shadow_volume > 0.0
    assert analysis.max_unsupported_span > 0.0
    assert np.count_nonzero(analysis.support_shadow_field) == analysis.shadow_voxel_count


def test_support_analysis_rejects_non_axis_aligned_direction() -> None:
    result = _cantilever_result()

    with pytest.raises(ValueError, match="build_direction"):
        result.analysis.support(build_direction="diagonal", threshold=0.5)  # type: ignore[arg-type]


def test_support_analysis_orients_surface_once(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _cantilever_result()
    calls = 0
    original = mesh_analysis._oriented_mesh

    def counted(mesh):
        nonlocal calls
        calls += 1
        return original(mesh)

    monkeypatch.setattr(mesh_analysis, "_oriented_mesh", counted)

    result.analysis.support(build_direction="+Z", threshold=0.5)

    assert calls == 1


def test_support_span_uses_longest_contiguous_shadow_run() -> None:
    domain = Domain.from_bounds(
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        zmin=0.0,
        zmax=8.0,
        voxel_size=1.0,
    )
    occupancy = np.zeros(domain.grid_shape, dtype=bool)
    occupancy[0, 0, 3] = True
    centroids = np.asarray([[0.5, 0.5, 2.5], [0.5, 0.5, 7.5]])

    shadow, max_span = _support_shadow_field(
        occupancy,
        centroids,
        domain=domain,
        build_direction="+Z",
    )

    np.testing.assert_array_equal(np.flatnonzero(shadow[0, 0]), np.asarray([0, 1, 4, 5, 6]))
    assert max_span == pytest.approx(3.0)
