from __future__ import annotations

import numpy as np
import pytest

import dds.mesh_analysis as mesh_analysis
from dds import BeadProfile, DepositionMetadata, Domain, LineDeposit, PointDeposit, simulate
from dds.analysis.support import _longest_true_run, _support_shadow_field


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


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.zeros((3, 7), dtype=bool), 0),
        (np.ones((2, 5), dtype=bool), 5),
        (
            np.asarray(
                [
                    [False, True, True, False, True, True, True],
                    [True, True, False, True, False, False, False],
                ],
                dtype=bool,
            ),
            3,
        ),
    ],
)
def test_longest_true_run_handles_edge_cases(
    values: np.ndarray,
    expected: int,
) -> None:
    assert _longest_true_run(values) == expected


def test_longest_true_run_matches_scalar_reference_for_random_rows() -> None:
    values = np.random.default_rng(42).random((128, 37)) < 0.35

    expected = 0
    for row in values:
        current = 0
        for value in row:
            current = current + 1 if value else 0
            expected = max(expected, current)

    assert _longest_true_run(values) == expected


@pytest.mark.parametrize(
    ("build_direction", "axis", "expected_indices"),
    [
        ("+X", 0, [0, 1]),
        ("-X", 0, [3, 4]),
        ("+Y", 1, [0, 1]),
        ("-Y", 1, [3, 4]),
        ("+Z", 2, [0, 1]),
        ("-Z", 2, [3, 4]),
    ],
)
def test_support_shadow_handles_all_axis_aligned_build_directions(
    build_direction: str,
    axis: int,
    expected_indices: list[int],
) -> None:
    domain = Domain.from_bounds(
        xmin=0.0,
        xmax=5.0,
        ymin=0.0,
        ymax=5.0,
        zmin=0.0,
        zmax=5.0,
        voxel_size=1.0,
    )
    occupancy = np.zeros(domain.grid_shape, dtype=bool)
    centroids = np.asarray([[2.5, 2.5, 2.5]])

    shadow, max_span = _support_shadow_field(
        occupancy,
        centroids,
        domain=domain,
        build_direction=build_direction,  # type: ignore[arg-type]
    )

    projection_axes = tuple(index for index in range(3) if index != axis)
    projected = np.any(shadow > 0.0, axis=projection_axes)
    np.testing.assert_array_equal(
        np.flatnonzero(projected),
        np.asarray(expected_indices),
    )
    assert max_span == pytest.approx(2.0)
