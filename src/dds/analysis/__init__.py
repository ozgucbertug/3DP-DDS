"""Public analysis namespace for dense simulation results."""

from ..occupancy import occupancy_fraction
from .models import InterfaceAnalysis, InterfacePairSummary, StratumFieldSet, SupportAnalysis
from .simulation import SimulationAnalysis

__all__ = [
    "InterfaceAnalysis",
    "InterfacePairSummary",
    "SimulationAnalysis",
    "StratumFieldSet",
    "SupportAnalysis",
    "occupancy_fraction",
]
