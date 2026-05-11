from __future__ import annotations

import numpy as np
import pytest

from dds.geometry import (
    box,
    capped_cone,
    capped_cylinder,
    capsule,
    capsule_chain,
    cone,
    cylinder,
    difference,
    ellipsoid,
    intersection,
    orient,
    rotate,
    rotation_matrix,
    rounded_box,
    rounded_cone,
    rounded_cylinder,
    slab,
    sphere,
    torus,
    union,
)


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
