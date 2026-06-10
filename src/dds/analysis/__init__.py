"""Public analysis namespace for dense simulation results."""

from ..occupancy import occupancy_fraction
from .bundle import SimulationAnalysis
from .fields import normalize_field, summarize_layers
from .models import InterfaceAnalysis, InterfacePairSummary, StratumFieldSet, SupportAnalysis

__all__ = [
    "InterfaceAnalysis",
    "InterfacePairSummary",
    "SimulationAnalysis",
    "StratumFieldSet",
    "SupportAnalysis",
    "normalize_field",
    "occupancy_fraction",
    "summarize_layers",
]
