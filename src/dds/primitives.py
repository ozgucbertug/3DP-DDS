"""Geometric primitives and deposition event containers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np

from .attributes import DepositionAttributes
from .utils import bounding_box_from_points, ensure_finite_triplet


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

    attributes: DepositionAttributes

    def bounds(self) -> tuple[Point3D, Point3D]:
        """Return geometric bounds for the primitive."""


@dataclass(frozen=True, slots=True)
class PointDeposit:
    """A material deposition event centered at a point."""

    x: float
    y: float
    z: float
    attributes: DepositionAttributes = field(default_factory=DepositionAttributes)

    def __post_init__(self) -> None:
        ensure_finite_triplet((self.x, self.y, self.z), "PointDeposit coordinates")

    @property
    def point(self) -> Point3D:
        """Return the deposited point as a Point3D."""

        return Point3D(x=self.x, y=self.y, z=self.z)

    def bounds(self) -> tuple[Point3D, Point3D]:
        """Return geometric bounds for the point deposit."""

        point = self.point
        return point, point


@dataclass(frozen=True, slots=True)
class LineDeposit:
    """A material deposition event along a line segment."""

    start: Point3D | Sequence[float]
    end: Point3D | Sequence[float]
    attributes: DepositionAttributes = field(default_factory=DepositionAttributes)

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", Point3D.from_value(self.start))
        object.__setattr__(self, "end", Point3D.from_value(self.end))

    @property
    def segment(self) -> LineSegment3D:
        """Return the deposited segment."""

        return LineSegment3D(start=self.start, end=self.end)

    def bounds(self) -> tuple[Point3D, Point3D]:
        """Return geometric bounds for the line deposit."""

        return self.segment.bounds()

Deposit: TypeAlias = PointDeposit | LineDeposit


@dataclass(frozen=True, slots=True)
class ToolpathDepositSequence:
    """An ordered sequence of deposition events from a toolpath."""

    deposits: tuple[Deposit, ...]

    def __post_init__(self) -> None:
        if not self.deposits:
            raise ValueError("ToolpathDepositSequence requires at least one deposit.")
        object.__setattr__(self, "deposits", tuple(self.deposits))

    @classmethod
    def from_polyline(
        cls,
        polyline: Polyline3D,
        attributes: DepositionAttributes | None = None,
    ) -> "ToolpathDepositSequence":
        """Expand a polyline into an ordered line-deposit sequence."""

        attrs = attributes or DepositionAttributes()
        deposits = tuple(
            LineDeposit(start=segment.start, end=segment.end, attributes=attrs)
            for segment in polyline.segments()
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

    def __iter__(self) -> Iterator[Deposit]:
        return iter(self.deposits)


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
