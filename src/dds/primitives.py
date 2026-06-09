"""Geometric primitives and deposition event containers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np

from .attributes import BeadProfile, DepositionMetadata, ProcessState
from .utils import (
    bounding_box_from_points,
    ensure_finite_scalar,
    ensure_finite_triplet,
    normalize_axis,
)

DEFAULT_Z_AXIS = (0.0, 0.0, 1.0)


def _validate_deposit_attributes(
    profile: BeadProfile | None,
    metadata: DepositionMetadata,
    process: ProcessState,
) -> None:
    if profile is not None and not isinstance(profile, BeadProfile):
        raise TypeError("profile must be a BeadProfile or None.")
    if not isinstance(metadata, DepositionMetadata):
        raise TypeError("metadata must be DepositionMetadata.")
    if not isinstance(process, ProcessState):
        raise TypeError("process must be ProcessState.")


def _resolve_explicit_bead_dimensions(
    profile: BeadProfile | None,
) -> tuple[float, float] | None:
    if profile is None:
        return None
    return float(profile.width), float(profile.height)


def _bead_half_extents(
    axis: "Point3D | Sequence[float]",
    *,
    width: float,
    height: float,
    padding: float = 0.0,
) -> np.ndarray:
    axis_array = np.abs(np.asarray(ensure_finite_triplet(axis, "axis"), dtype=float))
    radius = width / 2.0
    half_height = height / 2.0
    radial_extent = radius * np.sqrt(np.maximum(0.0, 1.0 - axis_array * axis_array))
    axial_extent = half_height * axis_array
    return radial_extent + axial_extent + float(padding)


def _point_target_support_bounds(
    target: "Point3D | Sequence[float]",
    axis: "Point3D | Sequence[float]",
    *,
    width: float,
    height: float,
    padding: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    padding = ensure_finite_scalar(padding, "padding")
    if padding < 0.0:
        raise ValueError("padding must be non-negative.")
    target_array = np.asarray(ensure_finite_triplet(target, "target"), dtype=float)
    axis_array = np.asarray(ensure_finite_triplet(axis, "axis"), dtype=float)
    center = target_array - axis_array * (height / 2.0)
    extent = _bead_half_extents(axis_array, width=width, height=height, padding=padding)
    return center - extent, center + extent


@dataclass(frozen=True, slots=True)
class Point3D:
    """A 3D Cartesian point."""

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        ensure_finite_triplet((self.x, self.y, self.z), "Point3D")

    @classmethod
    def from_value(cls, value: "Point3D | Sequence[float]") -> "Point3D":
        """Coerce a point-like value into a Point3D instance."""

        if isinstance(value, cls):
            return value
        x, y, z = ensure_finite_triplet(value, "point")
        return cls(x=x, y=y, z=z)

    def to_tuple(self) -> tuple[float, float, float]:
        """Return the point as a tuple."""

        return (self.x, self.y, self.z)

    def to_array(self) -> np.ndarray:
        """Return the point as a NumPy array."""

        return np.asarray(self.to_tuple(), dtype=float)


@dataclass(frozen=True, slots=True)
class Pose3D:
    """A nozzle position and local bead-axis orientation."""

    position: Point3D | Sequence[float]
    z_axis: Point3D | Sequence[float] = DEFAULT_Z_AXIS

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", Point3D.from_value(self.position))
        object.__setattr__(
            self,
            "z_axis",
            Point3D.from_value(normalize_axis(self.z_axis, name="Pose3D.z_axis")),
        )

    @property
    def axis(self) -> Point3D:
        """Return the normalized local bead axis."""

        return Point3D.from_value(self.z_axis)

    def to_dict(self) -> dict[str, list[float]]:
        """Return a JSON-compatible representation."""

        return {
            "position": list(self.position.to_tuple()),
            "z_axis": list(self.axis.to_tuple()),
        }


@dataclass(frozen=True, slots=True)
class LineSegment3D:
    """A line segment between two 3D points."""

    start: Point3D
    end: Point3D

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", Point3D.from_value(self.start))
        object.__setattr__(self, "end", Point3D.from_value(self.end))

    @property
    def length(self) -> float:
        """Return the segment length."""

        return float(np.linalg.norm(self.end.to_array() - self.start.to_array()))

    def bounds(self) -> tuple[Point3D, Point3D]:
        """Return geometric bounds for the segment."""

        minimum, maximum = bounding_box_from_points((self.start.to_tuple(), self.end.to_tuple()))
        return Point3D.from_value(minimum), Point3D.from_value(maximum)


@dataclass(frozen=True, slots=True)
class Polyline3D:
    """An ordered list of 3D points."""

    points: tuple[Point3D, ...]

    def __post_init__(self) -> None:
        coerced = tuple(Point3D.from_value(point) for point in self.points)
        if len(coerced) < 2:
            raise ValueError("Polyline3D requires at least two points.")
        object.__setattr__(self, "points", coerced)

    def segments(self) -> tuple[LineSegment3D, ...]:
        """Return line segments connecting consecutive points."""

        return tuple(
            LineSegment3D(start=start, end=end)
            for start, end in zip(self.points[:-1], self.points[1:])
        )

    def bounds(self) -> tuple[Point3D, Point3D]:
        """Return the axis-aligned bounds of the polyline."""

        minimum, maximum = bounding_box_from_points(point.to_tuple() for point in self.points)
        return Point3D.from_value(minimum), Point3D.from_value(maximum)


@runtime_checkable
class DepositionPrimitive(Protocol):
    """Protocol shared by deposition event primitives."""

    profile: BeadProfile | None
    metadata: DepositionMetadata
    process: ProcessState

    def bounds(self) -> tuple[Point3D, Point3D]:
        """Return geometric bounds for the primitive."""


@dataclass(frozen=True, slots=True)
class PointDeposit:
    """A material deposition target whose point lies at the top of the bead."""

    x: float
    y: float
    z: float
    profile: BeadProfile | None = None
    metadata: DepositionMetadata = field(default_factory=DepositionMetadata)
    process: ProcessState = field(default_factory=ProcessState)
    z_axis: Point3D | Sequence[float] = DEFAULT_Z_AXIS

    def __post_init__(self) -> None:
        _validate_deposit_attributes(self.profile, self.metadata, self.process)
        ensure_finite_triplet((self.x, self.y, self.z), "PointDeposit coordinates")
        object.__setattr__(self, "z_axis", Point3D.from_value(normalize_axis(self.z_axis, name="PointDeposit.z_axis")))

    @property
    def point(self) -> Point3D:
        """Return the nozzle-tip target point as a Point3D."""

        return Point3D(x=self.x, y=self.y, z=self.z)

    @property
    def target(self) -> Point3D:
        """Alias for the nozzle-tip target point."""

        return self.point

    @property
    def pose(self) -> Pose3D:
        """Return the target position and local bead axis as a pose."""

        return Pose3D(position=self.point, z_axis=self.axis)

    @property
    def axis(self) -> Point3D:
        """Return the normalized local bead axis."""

        return Point3D.from_value(self.z_axis)

    def support_bounds(self, *, padding: float = 0.0) -> tuple[Point3D, Point3D]:
        """Return explicit bead bounds when dimensions are available."""

        dimensions = _resolve_explicit_bead_dimensions(self.profile)
        if dimensions is None:
            return self.point, self.point
        width, height = dimensions
        minimum, maximum = _point_target_support_bounds(
            self.point,
            self.axis,
            width=width,
            height=height,
            padding=padding,
        )
        return Point3D.from_value(minimum), Point3D.from_value(maximum)

    def bounds(self) -> tuple[Point3D, Point3D]:
        """Return geometric bounds for the point deposit."""

        return self.support_bounds()

    @classmethod
    def from_point(
        cls,
        point: "Point3D | Sequence[float]",
        *,
        profile: BeadProfile | None = None,
        metadata: DepositionMetadata | None = None,
        process: ProcessState | None = None,
        z_axis: "Point3D | Sequence[float]" = DEFAULT_Z_AXIS,
    ) -> "PointDeposit":
        """Create a PointDeposit from a point-like value.

        Parameters
        ----------
        point:
            The nozzle-tip target position. Accepts a :class:`Point3D` or any
            three-element sequence ``(x, y, z)``.
        profile:
            Bead dimensions for this deposit.
        metadata:
            Optional deposition metadata. Defaults to an empty
            :class:`DepositionMetadata`.
        z_axis:
            Local bead axis direction. Defaults to ``(0, 0, 1)``.
        """

        p = Point3D.from_value(point)
        return cls(
            x=p.x,
            y=p.y,
            z=p.z,
            profile=profile,
            metadata=metadata if metadata is not None else DepositionMetadata(),
            process=process if process is not None else ProcessState(),
            z_axis=z_axis,
        )

    @classmethod
    def from_pose(
        cls,
        pose: Pose3D,
        *,
        profile: BeadProfile | None = None,
        metadata: DepositionMetadata | None = None,
        process: ProcessState | None = None,
    ) -> "PointDeposit":
        """Create a point deposit from a nozzle pose."""

        return cls.from_point(
            pose.position,
            profile=profile,
            metadata=metadata,
            process=process,
            z_axis=pose.axis,
        )


@dataclass(frozen=True, slots=True)
class LineDeposit:
    """A material deposition path whose endpoints are nozzle-tip targets."""

    start: Point3D | Sequence[float]
    end: Point3D | Sequence[float]
    profile: BeadProfile | None = None
    metadata: DepositionMetadata = field(default_factory=DepositionMetadata)
    process: ProcessState = field(default_factory=ProcessState)
    start_z_axis: Point3D | Sequence[float] = DEFAULT_Z_AXIS
    end_z_axis: Point3D | Sequence[float] | None = None

    def __post_init__(self) -> None:
        _validate_deposit_attributes(self.profile, self.metadata, self.process)
        object.__setattr__(self, "start", Point3D.from_value(self.start))
        object.__setattr__(self, "end", Point3D.from_value(self.end))
        start_z_axis = normalize_axis(self.start_z_axis, name="LineDeposit.start_z_axis")
        end_z_axis = (
            start_z_axis
            if self.end_z_axis is None
            else normalize_axis(self.end_z_axis, name="LineDeposit.end_z_axis")
        )
        object.__setattr__(self, "start_z_axis", Point3D.from_value(start_z_axis))
        object.__setattr__(self, "end_z_axis", Point3D.from_value(end_z_axis))
        if float(np.dot(self.start_axis.to_array(), self.end_axis.to_array())) <= -1.0 + 1e-8:
            raise ValueError(
                "LineDeposit endpoint axes must not be antiparallel. "
                "Subdivide the path with an intermediate orientation."
            )

    @property
    def segment(self) -> LineSegment3D:
        """Return the deposited segment."""

        return LineSegment3D(start=self.start, end=self.end)

    @property
    def start_axis(self) -> Point3D:
        """Return the normalized local bead axis at the segment start."""

        return Point3D.from_value(self.start_z_axis)

    @property
    def end_axis(self) -> Point3D:
        """Return the normalized local bead axis at the segment end.

        Always a normalized :class:`Point3D` after construction.
        Passing ``end_z_axis=None`` at construction time is shorthand for
        inheriting ``start_z_axis`` at the end of the segment.
        """

        return Point3D.from_value(self.end_z_axis)

    @property
    def start_pose(self) -> Pose3D:
        """Return the segment-start nozzle pose."""

        return Pose3D(position=self.start, z_axis=self.start_axis)

    @property
    def end_pose(self) -> Pose3D:
        """Return the segment-end nozzle pose."""

        return Pose3D(position=self.end, z_axis=self.end_axis)

    @classmethod
    def from_poses(
        cls,
        start_pose: Pose3D,
        end_pose: Pose3D,
        *,
        profile: BeadProfile | None = None,
        metadata: DepositionMetadata | None = None,
        process: ProcessState | None = None,
    ) -> "LineDeposit":
        """Create a line deposit from start and end nozzle poses."""

        return cls(
            start=start_pose.position,
            end=end_pose.position,
            profile=profile,
            metadata=metadata or DepositionMetadata(),
            process=process or ProcessState(),
            start_z_axis=start_pose.axis,
            end_z_axis=end_pose.axis,
        )

    def support_bounds(self, *, padding: float = 0.0) -> tuple[Point3D, Point3D]:
        """Return explicit swept-bead bounds when dimensions are available."""

        dimensions = _resolve_explicit_bead_dimensions(self.profile)
        if dimensions is None:
            return self.segment.bounds()
        width, height = dimensions
        padding_value = ensure_finite_scalar(padding, "padding")
        if padding_value < 0.0:
            raise ValueError("padding must be non-negative.")
        support_radius = math.sqrt((width / 2.0) ** 2 + height**2) + padding_value
        endpoints = np.stack((self.start.to_array(), self.end.to_array()), axis=0)
        minimum = endpoints.min(axis=0) - support_radius
        maximum = endpoints.max(axis=0) + support_radius
        return Point3D.from_value(minimum), Point3D.from_value(maximum)

    def bounds(self) -> tuple[Point3D, Point3D]:
        """Return geometric bounds for the line deposit."""

        return self.support_bounds()


@dataclass(frozen=True, slots=True)
class PolylineDeposit:
    """One deposition event swept through an ordered sequence of nozzle poses."""

    poses: tuple[Pose3D, ...]
    profile: BeadProfile | None = None
    metadata: DepositionMetadata = field(default_factory=DepositionMetadata)
    process: ProcessState = field(default_factory=ProcessState)

    def __post_init__(self) -> None:
        _validate_deposit_attributes(self.profile, self.metadata, self.process)
        poses = tuple(
            pose if isinstance(pose, Pose3D) else Pose3D(pose)
            for pose in self.poses
        )
        if len(poses) < 2:
            raise ValueError("PolylineDeposit requires at least two poses.")
        for start, end in zip(poses[:-1], poses[1:]):
            if float(np.dot(start.axis.to_array(), end.axis.to_array())) <= -1.0 + 1e-8:
                raise ValueError(
                    "Consecutive PolylineDeposit axes must not be antiparallel."
                )
        object.__setattr__(self, "poses", poses)

    @classmethod
    def from_polyline(
        cls,
        polyline: Polyline3D,
        profile: BeadProfile | None = None,
        metadata: DepositionMetadata | None = None,
        process: ProcessState | None = None,
        target_z_axes: Sequence[Point3D | Sequence[float]] | None = None,
    ) -> "PolylineDeposit":
        """Create a polyline deposit from geometric points and target axes."""

        if target_z_axes is None:
            axes = (DEFAULT_Z_AXIS,) * len(polyline.points)
        else:
            axes = tuple(target_z_axes)
            if len(axes) != len(polyline.points):
                raise ValueError("target_z_axes must match the number of polyline points.")
        return cls(
            poses=tuple(
                Pose3D(position=point, z_axis=axis)
                for point, axis in zip(polyline.points, axes)
            ),
            profile=profile,
            metadata=metadata or DepositionMetadata(),
            process=process or ProcessState(),
        )

    def segments(self) -> tuple[LineDeposit, ...]:
        """Return line deposits used to evaluate the polyline envelope."""

        return tuple(
            LineDeposit.from_poses(
                start,
                end,
                profile=self.profile,
                metadata=self.metadata,
                process=self.process,
            )
            for start, end in zip(self.poses[:-1], self.poses[1:])
        )

    def support_bounds(self, *, padding: float = 0.0) -> tuple[Point3D, Point3D]:
        """Return aggregate swept-bead bounds for the polyline."""

        points: list[tuple[float, float, float]] = []
        for deposit in self.segments():
            minimum, maximum = deposit.support_bounds(padding=padding)
            points.extend((minimum.to_tuple(), maximum.to_tuple()))
        minimum, maximum = bounding_box_from_points(points)
        return Point3D.from_value(minimum), Point3D.from_value(maximum)

    def bounds(self) -> tuple[Point3D, Point3D]:
        """Return aggregate bounds for the polyline deposit."""

        return self.support_bounds()


Deposit: TypeAlias = PointDeposit | LineDeposit | PolylineDeposit

DepositInput: TypeAlias = Deposit


def iter_deposits(deposits: Iterable[DepositInput] | DepositInput) -> Iterator[Deposit]:
    """Yield validated deposition events without changing event boundaries."""

    if isinstance(deposits, (PointDeposit, LineDeposit, PolylineDeposit)):
        yield deposits
        return

    for item in deposits:
        if isinstance(item, (PointDeposit, LineDeposit, PolylineDeposit)):
            yield item
        else:
            raise TypeError(
                "Deposits must be PointDeposit, LineDeposit, or PolylineDeposit instances."
            )
