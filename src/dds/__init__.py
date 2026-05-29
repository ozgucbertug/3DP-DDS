"""Public API for the dds package."""

from typing import TYPE_CHECKING, Any

from . import formats, geometry, targets, viz
from .attributes import BeadProfile, DepositionMetadata
from .cli import run_cli
from .domain import Domain
from .primitives import (
    LineDeposit,
    LineSegment3D,
    Point3D,
    PointDeposit,
    Polyline3D,
    SDFDeposit,
    ToolpathDepositSequence,
)
from .results import SimulationResult, WorkbenchViewConfig, simulate
from .analysis import AnalysisBundle, analysis_bundle
from .simulator import Simulator

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
    "SDFDeposit",
    "SimulationWorkbench",
    "SimulationResult",
    "Simulator",
    "ToolpathDepositSequence",
    "WorkbenchViewConfig",
    "analysis_bundle",
    "formats",
    "geometry",
    "run_cli",
    "simulate",
    "targets",
    "viz",
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
