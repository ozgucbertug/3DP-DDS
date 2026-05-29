"""Generic target-driven deposition workflow helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .attributes import BeadProfile, DepositionMetadata
from .primitives import DEFAULT_Z_AXIS, LineDeposit, PointDeposit, ToolpathDepositSequence
from .utils import ensure_finite_triplet, normalize_axis

OriginReference = Literal["top", "center"]


@dataclass(frozen=True, slots=True)
class TargetPoint:
    """One ordered nozzle target with an optional local bead axis."""

    index: int
    origin: tuple[float, float, float]
    z_axis: tuple[float, float, float] = DEFAULT_Z_AXIS

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", ensure_finite_triplet(self.origin, "TargetPoint.origin"))
        object.__setattr__(self, "z_axis", normalize_axis(self.z_axis, name="TargetPoint.z_axis"))


def target_point_from_origin(
    target: TargetPoint,
    *,
    profile: BeadProfile,
    origin_reference: OriginReference = "top",
) -> tuple[float, float, float]:
    """Return the PointDeposit target implied by the stored origin reference."""

    origin = np.asarray(target.origin, dtype=float)
    if origin_reference == "top":
        return tuple(float(value) for value in origin)
    if origin_reference != "center":
        raise ValueError("origin_reference must be 'top' or 'center'.")

    axis = np.asarray(target.z_axis, dtype=float)
    target_point = origin + (profile.height / 2.0) * axis
    return tuple(float(value) for value in target_point)


def point_deposits_from_targets(
    targets: Sequence[TargetPoint],
    *,
    profile: BeadProfile,
    metadata: DepositionMetadata | None = None,
    origin_reference: OriginReference = "top",
) -> tuple[PointDeposit, ...]:
    """Convert ordered targets into top-referenced point deposits."""

    metadata_value = metadata or DepositionMetadata()
    deposits = []
    for target in targets:
        pt = target_point_from_origin(target, profile=profile, origin_reference=origin_reference)
        deposits.append(
            PointDeposit(
                x=pt[0],
                y=pt[1],
                z=pt[2],
                profile=profile,
                metadata=metadata_value,
                z_axis=target.z_axis,
            )
        )
    return tuple(deposits)


def line_deposits_from_targets(
    targets: Sequence[TargetPoint],
    *,
    profile: BeadProfile,
    metadata: DepositionMetadata | None = None,
    origin_reference: OriginReference = "top",
) -> tuple[LineDeposit, ...]:
    """Convert ordered targets into top-referenced line deposits."""

    if len(targets) < 2:
        raise ValueError("line_deposits_from_targets requires at least two targets.")
    metadata_value = metadata or DepositionMetadata()
    return tuple(
        LineDeposit(
            start=target_point_from_origin(targets[index], profile=profile, origin_reference=origin_reference),
            end=target_point_from_origin(targets[index + 1], profile=profile, origin_reference=origin_reference),
            profile=profile,
            metadata=metadata_value,
            start_z_axis=targets[index].z_axis,
            end_z_axis=targets[index + 1].z_axis,
        )
        for index in range(len(targets) - 1)
    )


def toolpath_from_targets(
    targets: Sequence[TargetPoint],
    *,
    profile: BeadProfile,
    metadata: DepositionMetadata | None = None,
    origin_reference: OriginReference = "top",
) -> ToolpathDepositSequence:
    """Convert ordered targets into a toolpath deposit sequence."""

    return ToolpathDepositSequence(
        deposits=line_deposits_from_targets(
            targets,
            profile=profile,
            metadata=metadata,
            origin_reference=origin_reference,
        )
    )
