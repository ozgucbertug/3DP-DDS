"""Geometry primitives and deposition inputs."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias, cast

import numpy as np

from .attributes import BeadProfile, DepositionMetadata
from .utils import ensure_finite_scalar


Coordinate3D: TypeAlias = Sequence[float]
PointLike: TypeAlias = "Point3D | Coordinate3D"
VectorLike: TypeAlias = "Vector3D | Coordinate3D"
PoseLike: TypeAlias = "Pose3D | Point3D | Coordinate3D"

DEFAULT_AXIS = (0.0, 0.0, 1.0)


def _coerce_xyz(value: object, *, name: str) -> tuple[float, float, float]:
    if isinstance(value, (Point3D, Vector3D)):
        xyz = value.to_tuple()
    else:
        try:
            xyz = tuple(float(component) for component in value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain exactly three numeric values") from exc
    if len(xyz) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    if not np.all(np.isfinite(xyz)):
        raise ValueError(f"{name} values must be finite")
    return xyz


@dataclass(frozen=True, slots=True)
class Point3D:
    """A position in three-dimensional Cartesian space."""

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        values = (float(self.x), float(self.y), float(self.z))
        if not np.all(np.isfinite(values)):
            raise ValueError("point coordinates must be finite")
        object.__setattr__(self, "x", values[0])
        object.__setattr__(self, "y", values[1])
        object.__setattr__(self, "z", values[2])

    @classmethod
    def from_value(cls, value: PointLike) -> Point3D:
        if isinstance(value, cls):
            return value
        return cls(*_coerce_xyz(value, name="point"))

    def to_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    def to_array(self) -> np.ndarray:
        return np.asarray(self.to_tuple(), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Vector3D:
    """A direction or displacement in three-dimensional Cartesian space."""

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        values = (float(self.x), float(self.y), float(self.z))
        if not np.all(np.isfinite(values)):
            raise ValueError("vector components must be finite")
        object.__setattr__(self, "x", values[0])
        object.__setattr__(self, "y", values[1])
        object.__setattr__(self, "z", values[2])

    @classmethod
    def from_value(cls, value: VectorLike) -> Vector3D:
        if isinstance(value, cls):
            return value
        return cls(*_coerce_xyz(value, name="vector"))

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.to_array()))

    def normalized(self) -> Vector3D:
        length = self.length
        if length == 0.0:
            raise ValueError("vector must be non-zero")
        return Vector3D(self.x / length, self.y / length, self.z / length)

    def to_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    def to_array(self) -> np.ndarray:
        return np.asarray(self.to_tuple(), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Pose3D:
    """A target point and normalized axis defining its target plane."""

    position: PointLike
    axis: VectorLike = DEFAULT_AXIS

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", Point3D.from_value(self.position))
        object.__setattr__(
            self,
            "axis",
            Vector3D.from_value(self.axis).normalized(),
        )

    @classmethod
    def from_value(cls, value: PoseLike) -> Pose3D:
        if isinstance(value, cls):
            return value
        return cls(position=Point3D.from_value(value))

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "position": list(self.position.to_tuple()),  # type: ignore[union-attr]
            "axis": list(self.axis.to_tuple()),  # type: ignore[union-attr]
        }


@dataclass(frozen=True, slots=True)
class Line3D:
    """A finite line defined by start and end points."""

    start: PointLike
    end: PointLike

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", Point3D.from_value(self.start))
        object.__setattr__(self, "end", Point3D.from_value(self.end))

    @property
    def direction(self) -> Vector3D:
        start = self.start.to_array()  # type: ignore[union-attr]
        end = self.end.to_array()  # type: ignore[union-attr]
        return Vector3D.from_value(end - start)

    @property
    def length(self) -> float:
        start = self.start.to_array()  # type: ignore[union-attr]
        end = self.end.to_array()  # type: ignore[union-attr]
        return float(np.linalg.norm(end - start))

    @property
    def bounds(self) -> tuple[Point3D, Point3D]:
        start = self.start.to_array()  # type: ignore[union-attr]
        end = self.end.to_array()  # type: ignore[union-attr]
        return Point3D.from_value(np.minimum(start, end)), Point3D.from_value(
            np.maximum(start, end)
        )


@dataclass(frozen=True, slots=True)
class Polyline3D:
    """A connected sequence of points."""

    points: tuple[PointLike, ...]

    def __post_init__(self) -> None:
        points = tuple(Point3D.from_value(point) for point in self.points)
        if len(points) < 2:
            raise ValueError("polyline requires at least two points")
        object.__setattr__(self, "points", points)

    @property
    def segments(self) -> tuple[Line3D, ...]:
        return tuple(
            Line3D(start, end)
            for start, end in zip(self.points[:-1], self.points[1:], strict=True)
        )

    @property
    def length(self) -> float:
        return sum(segment.length for segment in self.segments)

    @property
    def bounds(self) -> tuple[Point3D, Point3D]:
        coordinates = np.asarray(
            [point.to_tuple() for point in self.points],
            dtype=np.float64,
        )
        return Point3D.from_value(coordinates.min(axis=0)), Point3D.from_value(
            coordinates.max(axis=0)
        )


def _point_target_support_bounds(
    target: PointLike,
    axis: VectorLike,
    *,
    width: float,
    height: float,
    padding: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conservative bounds for a top-referenced oriented bead."""

    padding_value = ensure_finite_scalar(padding, "padding")
    if padding_value < 0.0:
        raise ValueError("padding must be non-negative")
    target_array = Point3D.from_value(target).to_array()
    axis_array = Vector3D.from_value(axis).normalized().to_array()
    center = target_array - axis_array * (height / 2.0)
    radius = width / 2.0
    radial_extent = radius * np.sqrt(np.maximum(0.0, 1.0 - axis_array**2))
    axial_extent = (height / 2.0) * np.abs(axis_array)
    extent = radial_extent + axial_extent + padding_value
    return center - extent, center + extent


@dataclass(frozen=True, slots=True)
class PointDeposit:
    """A bead deposited at one pose."""

    target: PoseLike
    profile: BeadProfile
    metadata: DepositionMetadata = field(default_factory=DepositionMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", Pose3D.from_value(self.target))
        if not isinstance(self.profile, BeadProfile):
            raise TypeError("profile must be a BeadProfile")
        if not isinstance(self.metadata, DepositionMetadata):
            raise TypeError("metadata must be DepositionMetadata")

    def support_bounds(self, *, padding: float = 0.0) -> tuple[Point3D, Point3D]:
        minimum, maximum = _point_target_support_bounds(
            self.target.position,
            self.target.axis,
            width=self.profile.width,
            height=self.profile.height,
            padding=padding,
        )
        return Point3D.from_value(minimum), Point3D.from_value(maximum)


@dataclass(frozen=True, slots=True)
class LineDeposit:
    """A bead swept between two poses."""

    start: PoseLike
    end: PoseLike
    profile: BeadProfile
    metadata: DepositionMetadata = field(default_factory=DepositionMetadata)

    def __post_init__(self) -> None:
        start = Pose3D.from_value(self.start)
        end = Pose3D.from_value(self.end)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        if float(np.dot(start.axis.to_array(), end.axis.to_array())) <= -1.0 + 1e-12:
            raise ValueError("line endpoint axes cannot be antiparallel")
        if not isinstance(self.profile, BeadProfile):
            raise TypeError("profile must be a BeadProfile")
        if not isinstance(self.metadata, DepositionMetadata):
            raise TypeError("metadata must be DepositionMetadata")

    @property
    def line(self) -> Line3D:
        return Line3D(self.start.position, self.end.position)  # type: ignore[union-attr]

    def support_bounds(self, *, padding: float = 0.0) -> tuple[Point3D, Point3D]:
        padding_value = ensure_finite_scalar(padding, "padding")
        if padding_value < 0.0:
            raise ValueError("padding must be non-negative")
        support_radius = math.sqrt(
            (self.profile.width / 2.0) ** 2 + self.profile.height**2
        ) + padding_value
        endpoints = np.stack(
            (self.start.position.to_array(), self.end.position.to_array()),
            axis=0,
        )
        return Point3D.from_value(
            endpoints.min(axis=0) - support_radius
        ), Point3D.from_value(endpoints.max(axis=0) + support_radius)


@dataclass(frozen=True, slots=True)
class PolylineDeposit:
    """A bead swept through a connected sequence of poses."""

    poses: tuple[PoseLike, ...]
    profile: BeadProfile
    metadata: DepositionMetadata = field(default_factory=DepositionMetadata)

    def __post_init__(self) -> None:
        poses = tuple(Pose3D.from_value(pose) for pose in self.poses)
        if len(poses) < 2:
            raise ValueError("polyline deposit requires at least two poses")
        for start, end in zip(poses[:-1], poses[1:], strict=True):
            if (
                float(np.dot(start.axis.to_array(), end.axis.to_array()))
                <= -1.0 + 1e-12
            ):
                raise ValueError("consecutive polyline axes cannot be antiparallel")
        object.__setattr__(self, "poses", poses)
        if not isinstance(self.profile, BeadProfile):
            raise TypeError("profile must be a BeadProfile")
        if not isinstance(self.metadata, DepositionMetadata):
            raise TypeError("metadata must be DepositionMetadata")

    @property
    def polyline(self) -> Polyline3D:
        return Polyline3D(
            tuple(pose.position for pose in self.poses)  # type: ignore[union-attr]
        )

    def segments(self) -> tuple[LineDeposit, ...]:
        return tuple(
            LineDeposit(
                start=start,
                end=end,
                profile=self.profile,
                metadata=self.metadata,
            )
            for start, end in zip(self.poses[:-1], self.poses[1:], strict=True)
        )

    def support_bounds(self, *, padding: float = 0.0) -> tuple[Point3D, Point3D]:
        segment_bounds = [
            segment.support_bounds(padding=padding) for segment in self.segments()
        ]
        lower = np.min([bound[0].to_array() for bound in segment_bounds], axis=0)
        upper = np.max([bound[1].to_array() for bound in segment_bounds], axis=0)
        return Point3D.from_value(lower), Point3D.from_value(upper)


Deposit: TypeAlias = PointDeposit | LineDeposit | PolylineDeposit
DepositInput: TypeAlias = Deposit


def iter_deposits(deposits: DepositInput | Iterable[DepositInput]) -> Iterator[Deposit]:
    """Yield deposition events from one deposit or an iterable of deposits."""

    if isinstance(deposits, (PointDeposit, LineDeposit, PolylineDeposit)):
        yield deposits
        return
    if isinstance(deposits, (str, bytes)):
        raise TypeError("deposits must contain deposition primitives")
    try:
        iterator = iter(deposits)
    except TypeError as exc:
        raise TypeError("deposits must be a deposit or iterable of deposits") from exc
    for deposit in iterator:
        if not isinstance(deposit, (PointDeposit, LineDeposit, PolylineDeposit)):
            raise TypeError(
                "deposits must contain PointDeposit, LineDeposit, or PolylineDeposit"
            )
        yield cast(Deposit, deposit)
