"""Helpers for converting ordered poses into deposition inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from .attributes import BeadProfile, DepositionMetadata
from .primitives import LineDeposit, Point3D, PointDeposit, PolylineDeposit, Pose3D

OriginReference = Literal["top", "center"]


def target_pose_from_origin(
    pose: Pose3D,
    *,
    profile: BeadProfile,
    origin_reference: OriginReference = "top",
) -> Pose3D:
    """Convert a top- or center-referenced pose to a top-referenced target."""

    if origin_reference == "top":
        return pose
    if origin_reference != "center":
        raise ValueError("origin_reference must be 'top' or 'center'")

    position = pose.position.to_array() + (profile.height / 2.0) * pose.axis.to_array()
    return Pose3D(position=Point3D.from_value(position), axis=pose.axis)


def point_deposits_from_targets(
    targets: Sequence[Pose3D],
    *,
    profile: BeadProfile,
    metadata: DepositionMetadata | None = None,
    origin_reference: OriginReference = "top",
) -> tuple[PointDeposit, ...]:
    """Convert ordered target poses into point deposits."""

    metadata_value = metadata or DepositionMetadata()
    return tuple(
        PointDeposit(
            target=target_pose_from_origin(
                target,
                profile=profile,
                origin_reference=origin_reference,
            ),
            profile=profile,
            metadata=metadata_value,
        )
        for target in targets
    )


def line_deposits_from_targets(
    targets: Sequence[Pose3D],
    *,
    profile: BeadProfile,
    metadata: DepositionMetadata | None = None,
    origin_reference: OriginReference = "top",
) -> tuple[LineDeposit, ...]:
    """Convert ordered target poses into consecutive line deposits."""

    if len(targets) < 2:
        raise ValueError("line_deposits_from_targets requires at least two targets")
    metadata_value = metadata or DepositionMetadata()
    poses = tuple(
        target_pose_from_origin(
            target,
            profile=profile,
            origin_reference=origin_reference,
        )
        for target in targets
    )
    return tuple(
        LineDeposit(
            start=start,
            end=end,
            profile=profile,
            metadata=metadata_value,
        )
        for start, end in zip(poses[:-1], poses[1:], strict=True)
    )


def toolpath_from_targets(
    targets: Sequence[Pose3D],
    *,
    profile: BeadProfile,
    metadata: DepositionMetadata | None = None,
    origin_reference: OriginReference = "top",
) -> PolylineDeposit:
    """Convert ordered target poses into one polyline deposit."""

    if len(targets) < 2:
        raise ValueError("toolpath_from_targets requires at least two targets")
    return PolylineDeposit(
        poses=tuple(
            target_pose_from_origin(
                target,
                profile=profile,
                origin_reference=origin_reference,
            )
            for target in targets
        ),
        profile=profile,
        metadata=metadata or DepositionMetadata(),
    )
