"""Public API for the dds package."""

from .attributes import BeadProfile, DepositionMetadata
from .chunked import ChunkedField
from .domain import Domain
from .primitives import (
    Deposit,
    DepositInput,
    DepositionTarget,
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

__all__ = [
    "BeadProfile",
    "Deposit",
    "DepositInput",
    "DepositionMetadata",
    "DepositionTarget",
    "Domain",
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
