"""Public API for the dds package."""

from .attributes import BeadProfile, DepositionMetadata
from .chunked import ChunkedField
from .domain import Domain
from .primitives import (
    Deposit,
    DepositInput,
    Line3D,
    LineDeposit,
    Point3D,
    PointDeposit,
    Polyline3D,
    PolylineDeposit,
    Pose3D,
    Vector3D,
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
    "Line3D",
    "LineDeposit",
    "Point3D",
    "PointDeposit",
    "Polyline3D",
    "PolylineDeposit",
    "Pose3D",
    "Vector3D",
    "SimulationResult",
    "Simulator",
    "ChunkedField",
    "simulate",
]
