"""Public API for the dds package."""

from . import geometry
from .attributes import DepositionAttributes
from .domain import Domain
from .primitives import (
    LineDeposit,
    LineSegment3D,
    Point3D,
    PointDeposit,
    Polyline3D,
    ToolpathDepositSequence,
)
from .simulator import Simulator, sample_field, simulate_deposition_index, simulate_occupancy

__all__ = [
    "DepositionAttributes",
    "Domain",
    "LineDeposit",
    "LineSegment3D",
    "Point3D",
    "PointDeposit",
    "Polyline3D",
    "Simulator",
    "ToolpathDepositSequence",
    "geometry",
    "sample_field",
    "simulate_deposition_index",
    "simulate_occupancy",
]

__version__ = "0.1.0"
