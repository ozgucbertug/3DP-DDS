"""Shared type aliases for the dds package."""

from __future__ import annotations

from typing import Literal

DensityComposition = Literal["max", "sum"]
FieldName = Literal["density", "occupancy", "deposition_index"]
