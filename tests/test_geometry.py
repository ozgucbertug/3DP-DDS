from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dds import BeadProfile, DepositionMetadata, Domain, LineDeposit, PointDeposit, Simulator
from dds.geometry import (
    GridSDF3,
    MeshSDF3,
    TriangleMesh,
    box,
    capped_cone,
    capped_cylinder,
    capsule,
    capsule_chain,
    cone,
    cylinder,
    difference,
    ellipsoid,
    implicit_field_to_mesh,
    implicit_field_to_sdf_values,
    intersection,
    mesh_to_sdf_field,
    occupancy_to_mesh,
    occupancy_to_sdf,
    occupancy_to_sdf_field,
    orient,
    read_mesh,
    rotate,
    rotation_matrix,
    rounded_box,
    rounded_cone,
    rounded_cylinder,
    sdf_to_mesh,
    slab,
    sphere,
    torus,
    union,
    write_mesh,
)


def make_domain() -> Domain:
    return Domain.from_bounds(
        xmin=-6.0,
        xmax=6.0,
        ymin=-6.0,
        ymax=6.0,
        zmin=-6.0,
        zmax=6.0,
        voxel_size=0.5,
    )


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
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int64,
    )
    return TriangleMesh(vertices=vertices, faces=faces)


def test_primitives_follow_signed_distance_convention() -> None:
    assert sphere(radius=2.0)([0.0, 0.0, 0.0]) < 0.0
    assert sphere(radius=2.0)([3.0, 0.0, 0.0]) > 0.0
    assert abs(sphere(radius=2.0)([2.0, 0.0, 0.0])) < 1e-6

    assert box(size=(2.0, 4.0, 6.0))([0.0, 0.0, 0.0]) < 0.0
    assert cylinder(radius=1.0, height=2.0)([0.0, 0.0, 0.0]) < 0.0
    assert capsule((0.0, 0.0, -1.0), (0.0, 0.0, 1.0), radius=0.5)([0.0, 0.0, 0.0]) < 0.0
    assert ellipsoid(size=(2.0, 1.0, 1.0))([3.0, 0.0, 0.0]) > 0.0
    assert torus(major_radius=2.0, minor_radius=0.5)([2.0, 0.0, 0.0]) < 0.0
    assert slab(dx=2.0, dy=2.0, dz=2.0)([0.0, 0.0, 0.0]) < 0.0


def test_extended_primitives_follow_signed_distance_convention() -> None:
    rounded = rounded_box(size=(4.0, 4.0, 4.0), radius=0.5)
    assert rounded([0.0, 0.0, 0.0]) < 0.0
    assert abs(rounded([2.0, 0.0, 0.0])) < 1e-6
    assert rounded([3.0, 0.0, 0.0]) > 0.0

    capped = capped_cylinder((0.0, 0.0, -1.0), (0.0, 0.0, 1.0), radius=0.5)
    assert capped([0.0, 0.0, 0.0]) < 0.0
    assert abs(capped([0.5, 0.0, 0.0])) < 1e-6
    assert capped([0.0, 0.0, 1.5]) > 0.0

    rounded_cyl = rounded_cylinder(radius=1.0, height=3.0, rounding_radius=0.25)
    assert rounded_cyl([0.0, 0.0, 0.0]) < 0.0
    assert abs(rounded_cyl([1.0, 0.0, 0.0])) < 1e-6
    assert abs(rounded_cyl([0.0, 0.0, 1.5])) < 1e-6
    assert rounded_cyl([1.5, 0.0, 0.0]) > 0.0

    frustum = capped_cone(
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
        radius_a=1.0,
        radius_b=0.25,
    )
    assert frustum([0.0, 0.0, 0.0]) < 0.0
    assert abs(frustum([1.0, 0.0, -1.0])) < 1e-6
    assert frustum([1.5, 0.0, -1.0]) > 0.0

    centered_cone = cone(height=2.0, radius_bottom=1.0, radius_top=0.25)
    assert centered_cone([0.0, 0.0, 0.0]) < 0.0
    assert centered_cone([1.5, 0.0, -1.0]) > 0.0

    rounded_frustum = rounded_cone(
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
        radius_a=0.5,
        radius_b=0.25,
    )
    assert rounded_frustum([0.0, 0.0, -1.0]) < 0.0
    assert abs(rounded_frustum([0.5, 0.0, -1.0])) < 1e-6
    assert rounded_frustum([1.0, 0.0, -1.0]) > 0.0

    chain = capsule_chain([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0)], radius=0.25)
    assert chain([1.0, 0.0, 0.0]) < 0.0
    assert chain([2.0, 1.0, 0.0]) < 0.0
    assert chain([4.0, 4.0, 0.0]) > 0.0

    tapered_chain = capsule_chain(
        [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0)],
        radii=(0.25, 0.5, 0.25),
    )
    assert tapered_chain([0.0, 0.0, 0.0]) < 0.0
    assert tapered_chain([2.0, 0.0, 0.0]) < 0.0


def test_boolean_ops_return_expected_signs() -> None:
    left = sphere(radius=1.0, center=(-1.0, 0.0, 0.0))
    right = sphere(radius=1.0, center=(1.0, 0.0, 0.0))

    merged = union(left, right)
    clipped = intersection(left, box(size=(3.0, 2.0, 2.0), center=(-0.5, 0.0, 0.0)))
    carved = difference(box(size=4.0), sphere(radius=1.0))

    assert merged([-1.0, 0.0, 0.0]) < 0.0
    assert merged([1.0, 0.0, 0.0]) < 0.0
    assert merged([4.0, 0.0, 0.0]) > 0.0

    assert clipped([-0.5, 0.0, 0.0]) < 0.0
    assert clipped([2.0, 0.0, 0.0]) > 0.0

    assert carved([0.0, 0.0, 0.0]) > 0.0
    assert carved([1.8, 0.0, 0.0]) < 0.0


def test_smooth_and_chamfer_booleans_modify_the_seam() -> None:
    left = sphere(radius=1.2, center=(-0.8, 0.0, 0.0))
    right = sphere(radius=1.2, center=(0.8, 0.0, 0.0))

    hard = union(left, right)
    smooth = union(left, right, radius=0.4)
    chamfered = union(left, right, chamfer=0.2)

    sample = np.asarray([[0.0, 0.0, 0.0]])
    hard_value = hard(sample)[0]
    smooth_value = smooth(sample)[0]
    chamfer_value = chamfered(sample)[0]

    assert smooth_value != pytest.approx(hard_value)
    assert chamfer_value != pytest.approx(hard_value)
    assert smooth([-3.0, 0.0, 0.0]) == pytest.approx(hard([-3.0, 0.0, 0.0]))


def test_transforms_preserve_expected_distances() -> None:
    translated = sphere(radius=1.0).translate((2.0, 0.0, 0.0))
    scaled = sphere(radius=1.0).scale((2.0, 1.0, 1.0))
    rotated = rotate(capsule((0.0, 0.0, -1.0), (0.0, 0.0, 1.0), radius=0.5), np.pi / 2.0, (0.0, 1.0, 0.0))
    oriented = orient(capsule((0.0, 0.0, -1.0), (0.0, 0.0, 1.0), radius=0.5), (1.0, 0.0, 0.0))

    assert translated([2.0, 0.0, 0.0]) < 0.0
    assert scaled([1.5, 0.0, 0.0]) < 0.0
    assert rotated([1.0, 0.0, 0.0]) < 0.0
    assert oriented([1.0, 0.0, 0.0]) < 0.0


def test_rotation_matrix_rotates_axes_as_expected() -> None:
    matrix = rotation_matrix(np.pi / 2.0, (0.0, 0.0, 1.0))
    rotated = np.array([1.0, 0.0, 0.0]) @ matrix.T
    np.testing.assert_allclose(rotated, np.array([0.0, 1.0, 0.0]), atol=1e-6)


def test_extract_mesh_from_sdf_and_density_respects_domain() -> None:
    domain = make_domain()
    sampled = sphere(radius=2.0).sample(domain)
    mesh = sdf_to_mesh(domain, sampled)

    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    lower, upper = mesh.bounds
    assert lower[0] < -1.0 and upper[0] > 1.0

    density = np.maximum(1.5 - sampled, 0.0)
    density_mesh = implicit_field_to_mesh(domain, density, threshold=0.5)
    assert density_mesh.n_faces > 0


def test_occupancy_to_mesh_returns_empty_for_no_level_crossing() -> None:
    domain = make_domain()
    empty_mesh = occupancy_to_mesh(domain, np.zeros(domain.grid_shape, dtype=bool))
    assert empty_mesh.is_empty


def test_mesh_io_roundtrip_preserves_vertex_and_face_counts(tmp_path: Path) -> None:
    mesh = make_box_mesh()
    path = tmp_path / "cube.ply"
    write_mesh(path, mesh)
    loaded = read_mesh(path)

    assert loaded.n_vertices == mesh.n_vertices
    assert loaded.n_faces == mesh.n_faces


def test_mesh_to_sdf_field_inverts_trimesh_sign_convention() -> None:
    domain = make_domain()
    mesh = make_box_mesh()
    sdf_values = mesh_to_sdf_field(domain, mesh)

    center_index = domain.world_to_index((0.0, 0.0, 0.0), clip=True)
    outside_index = domain.world_to_index((3.0, 0.0, 0.0), clip=True)

    assert sdf_values[center_index] < 0.0
    assert sdf_values[outside_index] > 0.0


def test_non_watertight_mesh_rejected_for_signed_distance_queries() -> None:
    mesh = TriangleMesh(
        vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
    )

    with pytest.raises(ValueError):
        MeshSDF3(mesh)


def test_grid_sdf_wrapper_allows_csg_with_sampled_occupancy() -> None:
    domain = make_domain()
    occupancy = sphere(radius=2.0).sample(domain) <= 0.0
    sampled = occupancy_to_sdf(domain, occupancy)
    carved = sampled - sphere(radius=0.75)

    assert isinstance(sampled, GridSDF3)
    assert carved([0.0, 0.0, 0.0]) > 0.0
    assert carved([2.5, 0.0, 0.0]) > 0.0


def test_deposition_occupancy_to_sdf_and_mesh_is_nonempty() -> None:
    domain = Domain.from_bounds(
        xmin=0.0,
        xmax=10.0,
        ymin=0.0,
        ymax=10.0,
        zmin=0.0,
        zmax=4.0,
        voxel_size=0.5,
    )
    profile = BeadProfile(width=1.2, height=0.8)
    metadata = DepositionMetadata(layer_id=0)
    deposits = [
        PointDeposit(target=(2.25, 2.25, 0.65), profile=profile, metadata=metadata),
        LineDeposit(start=(2.25, 2.25, 0.65), end=(6.25, 2.25, 0.65), profile=profile, metadata=metadata),
    ]

    occupancy = Simulator(domain, deposits).result().analysis.occupancy(threshold=0.5)
    sdf_values = occupancy_to_sdf_field(domain, occupancy)
    mesh = sdf_to_mesh(domain, sdf_values)

    assert mesh.n_faces > 0


def test_implicit_field_to_sdf_values_uses_threshold() -> None:
    domain = make_domain()
    density = np.zeros(domain.grid_shape, dtype=float)
    density[domain.world_to_index((0.0, 0.0, 0.0), clip=True)] = 1.0
    sdf_values = implicit_field_to_sdf_values(domain, density, threshold=0.5)
    center = domain.world_to_index((0.0, 0.0, 0.0), clip=True)
    assert sdf_values[center] <= 0.0


def test_grid_sdf3_sample_different_domain_falls_back_to_interpolation() -> None:
    domain_a = Domain.from_bounds(xmin=-2.0, xmax=2.0, ymin=-2.0, ymax=2.0, zmin=-2.0, zmax=2.0, voxel_size=0.5)
    domain_b = Domain.from_bounds(xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0, zmin=-1.0, zmax=1.0, voxel_size=1.0)
    sdf_values = sphere(radius=1.0).sample(domain_a)
    grid_sdf = GridSDF3(domain_a, sdf_values)

    # Sampling on the same domain should return the stored values.
    same = grid_sdf.sample(domain_a)
    np.testing.assert_array_equal(same, sdf_values)

    # Sampling on a different domain should interpolate and return a finite result.
    different = grid_sdf.sample(domain_b)
    assert different.shape == domain_b.grid_shape
    assert np.all(np.isfinite(different))


def test_triangle_mesh_and_grid_sdf_own_read_only_arrays() -> None:
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.asarray([[0, 1, 2]])
    mesh = TriangleMesh(vertices=vertices, faces=faces, metadata={"source": "test"})
    vertices[0, 0] = 9.0

    assert mesh.vertices[0, 0] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        mesh.vertices[0, 0] = 2.0
    with pytest.raises(TypeError):
        mesh.metadata["source"] = "changed"  # type: ignore[index]

    domain = make_domain()
    values = np.zeros(domain.grid_shape)
    sdf = GridSDF3(domain, values)
    values.fill(1.0)

    assert float(sdf.values.max()) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        sdf.values[0, 0, 0] = 1.0
    with pytest.raises(AttributeError):
        sdf.values = np.ones(domain.grid_shape)  # type: ignore[misc]


def test_occupancy_to_sdf_field_is_negative_inside_positive_outside() -> None:
    domain = Domain.from_bounds(xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0, zmin=-5.0, zmax=5.0, voxel_size=0.5)
    occupancy = np.zeros(domain.grid_shape, dtype=bool)
    cx, cy, cz = (s // 2 for s in domain.grid_shape)
    r = 3
    occupancy[cx - r : cx + r, cy - r : cy + r, cz - r : cz + r] = True
    sdf = occupancy_to_sdf_field(domain, occupancy)

    assert sdf.shape == domain.grid_shape
    # Inside the occupied region: SDF must be negative.
    assert sdf[cx, cy, cz] < 0.0
    # Outside, well away from the surface: SDF must be positive.
    assert sdf[0, 0, 0] > 0.0


def test_implicit_field_to_sdf_values_matches_occupancy_to_sdf_field_at_threshold() -> None:
    from dds import PointDeposit, simulate

    domain = Domain.from_bounds(xmin=0.0, xmax=10.0, ymin=0.0, ymax=10.0, zmin=0.0, zmax=10.0, voxel_size=0.5)
    profile = BeadProfile(width=2.0, height=2.0)
    result = simulate(domain, [PointDeposit(target=(5.0, 5.0, 5.0), profile=profile)], threshold=0.5)
    density = result.implicit_field
    threshold = 0.5
    sdf_from_density = implicit_field_to_sdf_values(domain, density, threshold=threshold)
    occupancy = density >= threshold
    sdf_from_occupancy = occupancy_to_sdf_field(domain, occupancy)
    np.testing.assert_allclose(sdf_from_density, sdf_from_occupancy, atol=1e-10)
