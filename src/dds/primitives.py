"""Geometry primitives and deposition inputs."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from scipy.spatial.transform import Rotation

from .attributes import BeadProfile, DepositionMetadata
from .utils import ensure_finite_scalar


Coordinate3D: TypeAlias = Sequence[float]
PointLike: TypeAlias = "Point3D | Coordinate3D"
VectorLike: TypeAlias = "Vector3D | Coordinate3D"
TargetLike: TypeAlias = "DepositionTarget | Pose3D | Point3D | Coordinate3D"

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


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Pose3D:
    """An active local-to-parent rigid transform."""

    position: Point3D
    orientation: Rotation

    def __init__(
        self,
        position: PointLike,
        orientation: Rotation | None = None,
    ) -> None:
        resolved_orientation = (
            Rotation.identity() if orientation is None else orientation
        )
        if not isinstance(resolved_orientation, Rotation):
            raise TypeError("orientation must be a scipy Rotation")
        if not resolved_orientation.single:
            raise ValueError("orientation must contain exactly one rotation")
        quaternion = np.asarray(
            resolved_orientation.as_quat(canonical=True),
            dtype=np.float64,
        )
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("orientation must be a finite single rotation")
        object.__setattr__(self, "position", Point3D.from_value(position))
        object.__setattr__(
            self,
            "orientation",
            Rotation.from_quat(quaternion),
        )

    def transform_point(self, point: PointLike) -> Point3D:
        transformed = (
            self.orientation.apply(Point3D.from_value(point).to_array())
            + self.position.to_array()
        )
        return Point3D.from_value(transformed)

    def transform_vector(self, vector: VectorLike) -> Vector3D:
        transformed = self.orientation.apply(Vector3D.from_value(vector).to_array())
        return Vector3D.from_value(transformed)

    def inverse(self) -> Pose3D:
        orientation = self.orientation.inv()
        position = -orientation.apply(self.position.to_array())
        return Pose3D(position=position, orientation=orientation)

    def compose(self, local_pose: Pose3D) -> Pose3D:
        """Apply ``local_pose`` and then this pose."""

        if not isinstance(local_pose, Pose3D):
            raise TypeError("local_pose must be a Pose3D")
        return Pose3D(
            position=self.transform_point(local_pose.position),
            orientation=self.orientation * local_pose.orientation,
        )

    def as_matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.orientation.as_matrix()
        matrix[:3, 3] = self.position.to_array()
        return matrix

    @classmethod
    def from_matrix(cls, matrix: object) -> Pose3D:
        values = np.asarray(matrix, dtype=np.float64)
        if values.shape != (4, 4):
            raise ValueError("pose matrix must have shape (4, 4)")
        if not np.all(np.isfinite(values)):
            raise ValueError("pose matrix values must be finite")
        if not np.allclose(values[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12):
            raise ValueError("pose matrix must have homogeneous final row [0, 0, 0, 1]")
        rotation_matrix = values[:3, :3]
        if not np.allclose(
            rotation_matrix.T @ rotation_matrix,
            np.eye(3),
            atol=1e-10,
        ) or not np.isclose(np.linalg.det(rotation_matrix), 1.0, atol=1e-10):
            raise ValueError("pose matrix must contain a proper orthonormal rotation")
        return cls(
            position=values[:3, 3],
            orientation=Rotation.from_matrix(rotation_matrix),
        )

    def is_close(
        self,
        other: object,
        *,
        position_atol: float = 1e-9,
        angle_atol: float = 1e-9,
    ) -> bool:
        if not isinstance(other, Pose3D):
            return False
        if position_atol < 0.0 or angle_atol < 0.0:
            raise ValueError("comparison tolerances must be non-negative")
        position_close = np.allclose(
            self.position.to_array(),
            other.position.to_array(),
            rtol=0.0,
            atol=position_atol,
        )
        relative_angle = float((self.orientation.inv() * other.orientation).magnitude())
        return bool(position_close and relative_angle <= angle_atol)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "position": list(self.position.to_tuple()),
            "orientation_xyzw": self.orientation.as_quat(canonical=True).tolist(),
        }


@dataclass(frozen=True, slots=True, init=False)
class DepositionTarget:
    """A top-referenced position and deposition normal."""

    position: Point3D
    normal: Vector3D

    def __init__(
        self,
        position: PointLike,
        normal: VectorLike = DEFAULT_AXIS,
    ) -> None:
        object.__setattr__(self, "position", Point3D.from_value(position))
        object.__setattr__(
            self,
            "normal",
            Vector3D.from_value(normal).normalized(),
        )

    @classmethod
    def from_pose(
        cls,
        pose: Pose3D,
        *,
        local_axis: VectorLike = DEFAULT_AXIS,
    ) -> DepositionTarget:
        if not isinstance(pose, Pose3D):
            raise TypeError("pose must be a Pose3D")
        axis = Vector3D.from_value(local_axis).normalized()
        return cls(
            position=pose.position,
            normal=pose.transform_vector(axis),
        )

    @classmethod
    def from_value(cls, value: TargetLike) -> DepositionTarget:
        if isinstance(value, cls):
            return value
        if isinstance(value, Pose3D):
            return cls.from_pose(value)
        return cls(position=Point3D.from_value(value))

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "position": list(self.position.to_tuple()),
            "normal": list(self.normal.to_tuple()),
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
        return Vector3D.from_value(self.end.to_array() - self.start.to_array())

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end.to_array() - self.start.to_array()))

    @property
    def bounds(self) -> tuple[Point3D, Point3D]:
        start = self.start.to_array()
        end = self.end.to_array()
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
    normal: VectorLike,
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
    normal_array = Vector3D.from_value(normal).normalized().to_array()
    center = target_array - normal_array * (height / 2.0)
    radius = width / 2.0
    radial_extent = radius * np.sqrt(np.maximum(0.0, 1.0 - normal_array**2))
    axial_extent = (height / 2.0) * np.abs(normal_array)
    extent = radial_extent + axial_extent + padding_value
    return center - extent, center + extent


def _validate_deposit_attributes(
    profile: BeadProfile,
    metadata: DepositionMetadata,
) -> None:
    if not isinstance(profile, BeadProfile):
        raise TypeError("profile must be a BeadProfile")
    if not isinstance(metadata, DepositionMetadata):
        raise TypeError("metadata must be DepositionMetadata")


@dataclass(frozen=True, slots=True, init=False)
class PointDeposit:
    """A bead deposited at one target."""

    target: DepositionTarget
    profile: BeadProfile
    metadata: DepositionMetadata = field(default_factory=DepositionMetadata)

    def __init__(
        self,
        target: TargetLike,
        profile: BeadProfile,
        metadata: DepositionMetadata | None = None,
    ) -> None:
        resolved_metadata = DepositionMetadata() if metadata is None else metadata
        _validate_deposit_attributes(profile, resolved_metadata)
        object.__setattr__(self, "target", DepositionTarget.from_value(target))
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "metadata", resolved_metadata)

    def support_bounds(self, *, padding: float = 0.0) -> tuple[Point3D, Point3D]:
        minimum, maximum = _point_target_support_bounds(
            self.target.position,
            self.target.normal,
            width=self.profile.width,
            height=self.profile.height,
            padding=padding,
        )
        return Point3D.from_value(minimum), Point3D.from_value(maximum)


@dataclass(frozen=True, slots=True, init=False)
class LineDeposit:
    """A bead swept between two deposition targets."""

    start: DepositionTarget
    end: DepositionTarget
    profile: BeadProfile
    metadata: DepositionMetadata = field(default_factory=DepositionMetadata)

    def __init__(
        self,
        start: TargetLike,
        end: TargetLike,
        profile: BeadProfile,
        metadata: DepositionMetadata | None = None,
    ) -> None:
        resolved_start = DepositionTarget.from_value(start)
        resolved_end = DepositionTarget.from_value(end)
        resolved_metadata = DepositionMetadata() if metadata is None else metadata
        _validate_deposit_attributes(profile, resolved_metadata)
        if (
            float(
                np.dot(
                    resolved_start.normal.to_array(),
                    resolved_end.normal.to_array(),
                )
            )
            <= -1.0 + 1e-12
        ):
            raise ValueError("line endpoint normals cannot be antiparallel")
        object.__setattr__(self, "start", resolved_start)
        object.__setattr__(self, "end", resolved_end)
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "metadata", resolved_metadata)

    @property
    def line(self) -> Line3D:
        return Line3D(self.start.position, self.end.position)

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


@dataclass(frozen=True, slots=True, init=False)
class PolylineDeposit:
    """A bead swept through a connected sequence of deposition targets."""

    targets: tuple[DepositionTarget, ...]
    profile: BeadProfile
    metadata: DepositionMetadata = field(default_factory=DepositionMetadata)

    def __init__(
        self,
        targets: Iterable[TargetLike],
        profile: BeadProfile,
        metadata: DepositionMetadata | None = None,
    ) -> None:
        resolved_targets = tuple(
            DepositionTarget.from_value(target) for target in targets
        )
        if len(resolved_targets) < 2:
            raise ValueError("polyline deposit requires at least two targets")
        for start, end in zip(
            resolved_targets[:-1],
            resolved_targets[1:],
            strict=True,
        ):
            if (
                float(np.dot(start.normal.to_array(), end.normal.to_array()))
                <= -1.0 + 1e-12
            ):
                raise ValueError(
                    "consecutive polyline target normals cannot be antiparallel"
                )
        resolved_metadata = DepositionMetadata() if metadata is None else metadata
        _validate_deposit_attributes(profile, resolved_metadata)
        object.__setattr__(self, "targets", resolved_targets)
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "metadata", resolved_metadata)

    @property
    def polyline(self) -> Polyline3D:
        return Polyline3D(tuple(target.position for target in self.targets))

    def segments(self) -> tuple[LineDeposit, ...]:
        return tuple(
            LineDeposit(
                start=start,
                end=end,
                profile=self.profile,
                metadata=self.metadata,
            )
            for start, end in zip(
                self.targets[:-1],
                self.targets[1:],
                strict=True,
            )
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
        yield deposit
