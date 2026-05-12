from __future__ import annotations

import numpy as np
import pytest

from dds import BeadProfile, DepositionMetadata, Domain, LineDeposit, PointDeposit, Simulator
from dds.geometry import (
    TriangleMesh,
    downfacing_mask,
    face_areas,
    face_centroids,
    face_normals,
    mesh_bounds_stats,
    mesh_surface_area,
    mesh_volume_estimate,
    normal_rgb_from_normals,
    overhang_angles,
    support_risk_mask,
    vertex_normals,
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
    profile = BeadProfile(width=1.2, height=0.8)
    metadata = DepositionMetadata(layer_id=0)
    deposits = [
        PointDeposit(x=2.25, y=2.25, z=0.65, profile=profile, metadata=metadata),
        LineDeposit(start=(2.25, 2.25, 0.65), end=(6.25, 2.25, 0.65), profile=profile, metadata=metadata),
    ]
    return Simulator(make_domain(), deposits)


def make_box_mesh() -> TriangleMesh:
    vertices = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    faces = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return TriangleMesh(vertices=vertices, faces=faces)


def test_mesh_analysis_metrics_and_overhang_conventions() -> None:
    mesh = make_box_mesh()

    normals = face_normals(mesh)
    v_normals = vertex_normals(mesh)
    centroids = face_centroids(mesh)
    areas = face_areas(mesh)
    angles = overhang_angles(mesh, build_direction=(0.0, 0.0, 1.0))
    risk = support_risk_mask(mesh, build_direction=(0.0, 0.0, 1.0), critical_angle_deg=45.0)
    down = downfacing_mask(mesh, build_direction=(0.0, 0.0, 1.0))
    bounds = mesh_bounds_stats(mesh)

    assert normals.shape == (12, 3)
    assert v_normals.shape == (8, 3)
    assert centroids.shape == (12, 3)
    np.testing.assert_allclose(np.mean(centroids, axis=0), np.zeros(3), atol=1e-6)
    np.testing.assert_array_equal(
        normal_rgb_from_normals(np.asarray([[-1.0, 0.0, 1.0]], dtype=float)),
        np.asarray([[0, 128, 255]], dtype=np.uint8),
    )
    np.testing.assert_allclose(areas, np.ones(12) * 2.0)
    assert np.sum(np.isclose(angles, 0.0)) == 2
    assert np.sum(np.isclose(angles, 90.0)) == 8
    assert np.sum(np.isclose(angles, 180.0)) == 2
    assert int(np.count_nonzero(risk)) == 2
    assert int(np.count_nonzero(down)) == 2
    assert mesh_surface_area(mesh) == pytest.approx(24.0)
    assert mesh_volume_estimate(mesh) == pytest.approx(8.0)
    assert bounds["dx"] == pytest.approx(2.0)
    assert bounds["dy"] == pytest.approx(2.0)
    assert bounds["dz"] == pytest.approx(2.0)


def test_mesh_volume_estimate_returns_none_for_non_watertight_mesh() -> None:
    mesh = TriangleMesh(
        vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
    )

    assert mesh_volume_estimate(mesh) is None


def test_analysis_bundle_subvolume_stats_and_mesh_analysis_are_headless() -> None:
    simulator = make_simulator()
    bundle = simulator.analysis_bundle()

    stats = bundle.subvolume_stats(((0.0, 0.0, 0.0), (5.0, 5.0, 2.0)), threshold=0.5)
    analysis = bundle.mesh_analysis(build_direction=(0.0, 0.0, 1.0), critical_angle_deg=45.0)

    assert stats["voxel_count"] > 0.0
    assert stats["occupied_voxel_count"] > 0.0
    assert stats["mesh_area"] > 0.0
    assert analysis["mesh"].n_faces > 0
    assert analysis["face_normals"].shape[1] == 3
    assert analysis["face_centroids"].shape[1] == 3
    assert analysis["face_areas"].ndim == 1
    assert analysis["support_risk_mask"].dtype == np.bool_
    assert simulator.mesh_analysis(build_direction=(0.0, 0.0, 1.0), critical_angle_deg=45.0) is analysis
    assert simulator.subvolume_stats(((0.0, 0.0, 0.0), (5.0, 5.0, 2.0)), threshold=0.5) == stats
