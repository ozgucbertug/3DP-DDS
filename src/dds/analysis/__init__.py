"""Public analysis namespace for dense simulation results."""

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
from .fields import deposition_index_from_density, normalize_field, summarize_layers
from ..occupancy import occupancy_fraction

__all__ = [
    "AnalysisBundle",
    "analysis_bundle",
    "contains_point",
    "deposition_index_from_density",
    "normalize_field",
    "occupancy_fraction",
    "sample_density_at",
    "sample_deposition_index_at",
    "sample_points",
    "signed_distance_at",
    "subvolume_stats",
    "summarize_layers",
    "surface_normal_at",
]
