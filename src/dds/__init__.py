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
from .queries import AnalysisBundle, analysis_bundle
from .simulator import Simulator, sample_field, simulate_deposition_index, simulate_occupancy

__all__ = [
    "AnalysisBundle",
    "DepositionAttributes",
    "Domain",
    "LineDeposit",
    "LineSegment3D",
    "Point3D",
    "PointDeposit",
    "Polyline3D",
    "Simulator",
    "ToolpathDepositSequence",
    "analysis_bundle",
    "geometry",
    "sample_field",
    "simulate_deposition_index",
    "simulate_occupancy",
]

__version__ = "0.1.0"
