"""Public API for the dds package."""

from .attributes import BeadProfile, DepositionMetadata
from .chunked import ChunkedField
from .domain import Domain
from .primitives import (
    Deposit,
    DepositInput,
    LineDeposit,
    LineSegment3D,
    Point3D,
    PointDeposit,
    Polyline3D,
    PolylineDeposit,
    Pose3D,
)
from .results import SimulationResult, simulate
from .simulator import Simulator
from .types import FieldComposition

__all__ = [
    "BeadProfile",
    "Deposit",
    "DepositInput",
    "DepositionMetadata",
    "Domain",
    "FieldComposition",
    "LineDeposit",
    "LineSegment3D",
    "Point3D",
    "PointDeposit",
    "Polyline3D",
    "PolylineDeposit",
    "Pose3D",
    "SimulationResult",
    "Simulator",
    "ChunkedField",
    "simulate",
]
