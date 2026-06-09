"""Public analysis namespace for dense simulation results."""

from ..occupancy import occupancy_fraction
from .bundle import (
    AnalysisBundle,
    analysis_bundle,
    contains_point,
    sample_density_at,
    sample_deposition_index_at,
    sample_points,
    signed_distance_at,
    subvolume_stats,
    surface_normal_at,
)
from .fields import normalize_field, summarize_layers
from .interface import interface
from .models import InterfaceAnalysis, InterfacePairSummary, StratumFieldSet, SupportAnalysis
from .strata import strata
from .support import support

__all__ = [
    "AnalysisBundle",
    "InterfaceAnalysis",
    "InterfacePairSummary",
    "StratumFieldSet",
    "SupportAnalysis",
    "analysis_bundle",
    "contains_point",
    "interface",
    "normalize_field",
    "occupancy_fraction",
    "sample_density_at",
    "sample_deposition_index_at",
    "sample_points",
    "signed_distance_at",
    "strata",
    "subvolume_stats",
    "support",
    "summarize_layers",
    "surface_normal_at",
]
