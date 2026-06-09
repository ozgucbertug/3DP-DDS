"""Public API for the dds package."""

from typing import TYPE_CHECKING, Any

from . import formats, geometry, targets, viz
from .analysis import AnalysisBundle, analysis_bundle
from .attributes import BeadProfile, DepositionMetadata, ProcessState, UnitSystem
from .chunked import ChunkedField
from .cli import run_cli
from .domain import Domain
from .fields import accumulate_chunked_field, apply_deposit_to_field, apply_deposit_to_index_field
from .io import load_checkpoint, save_checkpoint
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
from .results import SimulationResult, WorkbenchViewConfig, simulate
from .simulator import Simulator
from .types import FieldComposition, FieldName

if TYPE_CHECKING:
    from .workbench import SimulationWorkbench

__all__ = [
    "AnalysisBundle",
    "BeadProfile",
    "Deposit",
    "DepositInput",
    "DepositionMetadata",
    "Domain",
    "FieldName",
    "FieldComposition",
    "LineDeposit",
    "LineSegment3D",
    "Point3D",
    "PointDeposit",
    "Polyline3D",
    "PolylineDeposit",
    "Pose3D",
    "ProcessState",
    "SimulationWorkbench",
    "SimulationResult",
    "Simulator",
    "ChunkedField",
    "UnitSystem",
    "WorkbenchViewConfig",
    "accumulate_chunked_field",
    "analysis_bundle",
    "apply_deposit_to_field",
    "apply_deposit_to_index_field",
    "formats",
    "geometry",
    "load_checkpoint",
    "run_cli",
    "save_checkpoint",
    "simulate",
    "targets",
    "viz",
]

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
