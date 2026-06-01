"""Public API for the dds package."""

from typing import TYPE_CHECKING, Any

from . import formats, geometry, targets, viz
from .attributes import BeadProfile, DepositionMetadata
from .cli import run_cli
from .domain import Domain
from .primitives import (
    Deposit,
    DepositInput,
    LineDeposit,
    LineSegment3D,
    Point3D,
    PointDeposit,
    Polyline3D,
    ToolpathDepositSequence,
)
from .types import DensityComposition, FieldName
from .results import SimulationResult, WorkbenchViewConfig, simulate
from .analysis import AnalysisBundle, analysis_bundle
from .fields import apply_deposit_to_field, apply_deposit_to_index_field, accumulate_density_sparse
from .simulator import Simulator
from .sparse import SparseDensityField

if TYPE_CHECKING:
    from .workbench import SimulationWorkbench

__all__ = [
    "AnalysisBundle",
    "BeadProfile",
    "DensityComposition",
    "Deposit",
    "DepositInput",
    "DepositionMetadata",
    "Domain",
    "FieldName",
    "LineDeposit",
    "LineSegment3D",
    "Point3D",
    "PointDeposit",
    "Polyline3D",
    "SimulationWorkbench",
    "SimulationResult",
    "Simulator",
    "SparseDensityField",
    "ToolpathDepositSequence",
    "WorkbenchViewConfig",
    "accumulate_density_sparse",
    "analysis_bundle",
    "apply_deposit_to_field",
    "apply_deposit_to_index_field",
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
