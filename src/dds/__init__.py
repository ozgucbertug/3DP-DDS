"""Public API for the dds package."""

from typing import TYPE_CHECKING, Any

from . import geometry
from .attributes import BeadProfile, DepositionMetadata
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

if TYPE_CHECKING:
    from .workbench import SimulationWorkbench

__all__ = [
    "AnalysisBundle",
    "BeadProfile",
    "DepositionMetadata",
    "Domain",
    "LineDeposit",
    "LineSegment3D",
    "Point3D",
    "PointDeposit",
    "Polyline3D",
    "SimulationWorkbench",
    "Simulator",
    "ToolpathDepositSequence",
    "analysis_bundle",
    "geometry",
    "sample_field",
    "simulate_deposition_index",
    "simulate_occupancy",
]

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name == "SimulationWorkbench":
        try:
            from .workbench import SimulationWorkbench
        except ImportError as exc:
            raise ImportError(
                'SimulationWorkbench requires optional visualization dependencies. '
                'Install them with `pip install -e ".[viz]"`.'
            ) from exc
        return SimulationWorkbench
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
