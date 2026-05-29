"""Geometric primitives and deposition event containers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np

from .attributes import BeadProfile, DepositionMetadata
from .utils import bounding_box_from_points, ensure_finite_triplet, normalize_axis

DEFAULT_Z_AXIS = (0.0, 0.0, 1.0)


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
    """Protocol shared by point and line deposits."""

    profile: BeadProfile | None
    metadata: DepositionMetadata

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
    z_axis: Point3D | Sequence[float] = DEFAULT_Z_AXIS

    def __post_init__(self) -> None:
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
            z_axis=z_axis,
        )


@dataclass(frozen=True, slots=True)
class LineDeposit:
    """A material deposition path whose endpoints are nozzle-tip targets."""

    start: Point3D | Sequence[float]
    end: Point3D | Sequence[float]
    profile: BeadProfile | None = None
    metadata: DepositionMetadata = field(default_factory=DepositionMetadata)
    start_z_axis: Point3D | Sequence[float] = DEFAULT_Z_AXIS
    end_z_axis: Point3D | Sequence[float] | None = None

    def __post_init__(self) -> None:
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
        """Return the normalized local bead axis at the segment end."""

        return Point3D.from_value(self.end_z_axis)

    def support_bounds(self, *, padding: float = 0.0) -> tuple[Point3D, Point3D]:
        """Return explicit swept-bead bounds when dimensions are available."""

        dimensions = _resolve_explicit_bead_dimensions(self.profile)
        if dimensions is None:
            return self.segment.bounds()
        width, height = dimensions
        start_min, start_max = _point_target_support_bounds(
            self.start,
            self.start_axis,
            width=width,
            height=height,
            padding=padding,
        )
        end_min, end_max = _point_target_support_bounds(
            self.end,
            self.end_axis,
            width=width,
            height=height,
            padding=padding,
        )
        minimum = np.minimum(start_min, end_min)
        maximum = np.maximum(start_max, end_max)
        return Point3D.from_value(minimum), Point3D.from_value(maximum)

    def bounds(self) -> tuple[Point3D, Point3D]:
        """Return geometric bounds for the line deposit."""

        return self.support_bounds()

_LeafDeposit: TypeAlias = PointDeposit | LineDeposit


@dataclass(frozen=True, slots=True)
class ToolpathDepositSequence:
    """An ordered sequence of deposition events from a toolpath."""

    deposits: tuple[_LeafDeposit, ...]

    def __post_init__(self) -> None:
        if not self.deposits:
            raise ValueError("ToolpathDepositSequence requires at least one deposit.")
        object.__setattr__(self, "deposits", tuple(self.deposits))

    @classmethod
    def from_polyline(
        cls,
        polyline: Polyline3D,
        profile: BeadProfile | None = None,
        metadata: DepositionMetadata | None = None,
        target_z_axes: Sequence[Point3D | Sequence[float]] | None = None,
    ) -> "ToolpathDepositSequence":
        """Expand a polyline into an ordered line-deposit sequence."""

        profile_value = profile
        metadata_value = metadata or DepositionMetadata()
        if target_z_axes is None:
            axes = (DEFAULT_Z_AXIS,) * len(polyline.points)
        else:
            axes = tuple(target_z_axes)
            if len(axes) != len(polyline.points):
                raise ValueError("target_z_axes must match the number of polyline points.")
        deposits = tuple(
            LineDeposit(
                start=segment.start,
                end=segment.end,
                profile=profile_value,
                metadata=metadata_value,
                start_z_axis=axes[index],
                end_z_axis=axes[index + 1],
            )
            for index, segment in enumerate(polyline.segments())
        )
        return cls(deposits=deposits)

    def bounds(self) -> tuple[Point3D, Point3D]:
        """Return aggregate bounds for the sequence."""

        points: list[tuple[float, float, float]] = []
        for deposit in self.deposits:
            minimum, maximum = deposit.bounds()
            points.extend((minimum.to_tuple(), maximum.to_tuple()))
        minimum, maximum = bounding_box_from_points(points)
        return Point3D.from_value(minimum), Point3D.from_value(maximum)

    def __iter__(self) -> Iterator[_LeafDeposit]:
        return iter(self.deposits)


Deposit: TypeAlias = PointDeposit | LineDeposit

DepositInput: TypeAlias = Deposit | ToolpathDepositSequence


def iter_deposits(deposits: Iterable[DepositInput] | DepositInput) -> Iterator[Deposit]:
    """Yield point and line deposits, flattening toolpath sequences."""

    if isinstance(deposits, ToolpathDepositSequence):
        yield from deposits.deposits
        return
    if isinstance(deposits, (PointDeposit, LineDeposit)):
        yield deposits
        return

    for item in deposits:
        if isinstance(item, ToolpathDepositSequence):
            yield from item.deposits
        elif isinstance(item, (PointDeposit, LineDeposit)):
            yield item
        else:
            raise TypeError(
                "Deposits must be PointDeposit, LineDeposit, or ToolpathDepositSequence instances."
            )
