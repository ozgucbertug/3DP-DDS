from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dds import DepositionTarget, Pose3D


def test_pose_accepts_scipy_rotation_representations() -> None:
    euler = Rotation.from_euler("xyz", [20.0, -30.0, 45.0], degrees=True)
    poses = (
        Pose3D((1.0, 2.0, 3.0), euler),
        Pose3D((1.0, 2.0, 3.0), Rotation.from_quat(euler.as_quat())),
        Pose3D((1.0, 2.0, 3.0), Rotation.from_matrix(euler.as_matrix())),
        Pose3D((1.0, 2.0, 3.0), Rotation.from_rotvec(euler.as_rotvec())),
    )

    reference = DepositionTarget.from_pose(poses[0])
    for pose in poses[1:]:
        assert pose.is_close(poses[0])
        assert DepositionTarget.from_pose(pose).normal.to_tuple() == pytest.approx(
            reference.normal.to_tuple()
        )


def test_pose_transforms_points_vectors_and_composes() -> None:
    world = Pose3D(
        (10.0, 0.0, 0.0),
        Rotation.from_euler("z", 90.0, degrees=True),
    )
    local = Pose3D(
        (2.0, 0.0, 0.0),
        Rotation.from_euler("x", 90.0, degrees=True),
    )
    composed = world.compose(local)

    assert world.transform_point((1.0, 0.0, 0.0)).to_tuple() == pytest.approx(
        (10.0, 1.0, 0.0)
    )
    assert world.transform_vector((1.0, 0.0, 0.0)).to_tuple() == pytest.approx(
        (0.0, 1.0, 0.0)
    )
    assert composed.transform_point((0.0, 0.0, 0.0)).to_tuple() == pytest.approx(
        world.transform_point(local.position).to_tuple()
    )
    assert composed.transform_point((1.0, 2.0, 3.0)).to_tuple() == pytest.approx(
        world.transform_point(local.transform_point((1.0, 2.0, 3.0))).to_tuple()
    )


def test_pose_inverse_and_matrix_round_trip() -> None:
    pose = Pose3D(
        (1.0, -2.0, 3.0),
        Rotation.from_euler("zyx", [15.0, 25.0, -35.0], degrees=True),
    )
    identity = pose.compose(pose.inverse())
    restored = Pose3D.from_matrix(pose.as_matrix())

    assert identity.is_close(Pose3D((0.0, 0.0, 0.0)), angle_atol=1e-8)
    assert restored.is_close(pose)


def test_pose_serializes_canonical_scalar_last_quaternion() -> None:
    pose = Pose3D((1.0, 2.0, 3.0), Rotation.from_quat([0.0, 0.0, -1.0, 0.0]))
    payload = pose.to_dict()

    assert payload["position"] == [1.0, 2.0, 3.0]
    assert payload["orientation_xyzw"] == pytest.approx([0.0, 0.0, 1.0, 0.0])


def test_pose_rejects_stacked_rotations_and_malformed_matrices() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Pose3D((0.0, 0.0, 0.0), Rotation.from_euler("z", [0.0, 1.0]))
    with pytest.raises(ValueError, match="shape"):
        Pose3D.from_matrix(np.eye(3))
    malformed = np.eye(4)
    malformed[0, 0] = 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        Pose3D.from_matrix(malformed)


def test_target_conversion_defaults_to_world_and_tool_local_positive_z() -> None:
    coordinate_target = DepositionTarget.from_value((1.0, 2.0, 3.0))
    rotated_pose = Pose3D(
        (1.0, 2.0, 3.0),
        Rotation.from_euler("x", 90.0, degrees=True),
    )
    pose_target = DepositionTarget.from_pose(rotated_pose)
    custom_axis_target = DepositionTarget.from_pose(
        rotated_pose,
        local_axis=(0.0, 1.0, 0.0),
    )

    assert coordinate_target.normal.to_tuple() == (0.0, 0.0, 1.0)
    assert pose_target.normal.to_tuple() == pytest.approx((0.0, -1.0, 0.0))
    assert custom_axis_target.normal.to_tuple() == pytest.approx((0.0, 0.0, 1.0))


def test_roll_about_local_deposition_axis_does_not_change_target() -> None:
    position = (1.0, 2.0, 3.0)
    unrolled = Pose3D(position, Rotation.identity())
    rolled = Pose3D(
        position,
        Rotation.from_euler("z", 137.0, degrees=True),
    )

    assert DepositionTarget.from_pose(rolled) == DepositionTarget.from_pose(unrolled)


def test_target_rejects_zero_and_nonfinite_normals() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        DepositionTarget((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        DepositionTarget((0.0, 0.0, 0.0), (0.0, np.nan, 1.0))
