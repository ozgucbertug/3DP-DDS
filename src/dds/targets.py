"""Helpers for converting ordered deposition targets into deposits."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from .attributes import BeadProfile
from .primitives import DepositionTarget, LineDeposit, PointDeposit, PolylineDeposit

OriginReference = Literal["top", "center"]


def _target_from_origin(
    target: DepositionTarget,
    *,
    profile: BeadProfile,
    origin_reference: OriginReference = "top",
) -> DepositionTarget:
    """Convert a top- or center-referenced target to a top reference."""

    if origin_reference == "top":
        return target
    if origin_reference != "center":
        raise ValueError("origin_reference must be 'top' or 'center'")

    position = (
        target.position.to_array()
        + (profile.height / 2.0) * target.normal.to_array()
    )
    return DepositionTarget(position=position, normal=target.normal)


def point_deposits_from_targets(
    targets: Sequence[DepositionTarget],
    *,
    profile: BeadProfile,
    origin_reference: OriginReference = "top",
) -> tuple[PointDeposit, ...]:
    """Convert ordered targets into point deposits."""

    return tuple(
        PointDeposit(
            target=_target_from_origin(
                target,
                profile=profile,
                origin_reference=origin_reference,
            ),
            profile=profile,
        )
        for target in targets
    )


def line_deposits_from_targets(
    targets: Sequence[DepositionTarget],
    *,
    profile: BeadProfile,
    origin_reference: OriginReference = "top",
) -> tuple[LineDeposit, ...]:
    """Convert ordered targets into consecutive line deposits."""

    if len(targets) < 2:
        raise ValueError("line_deposits_from_targets requires at least two targets")
    normalized_targets = tuple(
        _target_from_origin(
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
        )
        for start, end in zip(
            normalized_targets[:-1],
            normalized_targets[1:],
        )
    )


def toolpath_from_targets(
    targets: Sequence[DepositionTarget],
    *,
    profile: BeadProfile,
    origin_reference: OriginReference = "top",
) -> PolylineDeposit:
    """Convert ordered targets into one polyline deposit."""

    if len(targets) < 2:
        raise ValueError("toolpath_from_targets requires at least two targets")
    return PolylineDeposit(
        targets=tuple(
            _target_from_origin(
                target,
                profile=profile,
                origin_reference=origin_reference,
            )
            for target in targets
        ),
        profile=profile,
    )
