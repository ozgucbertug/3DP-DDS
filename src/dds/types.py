"""Shared type aliases for the dds package."""

from __future__ import annotations

from typing import Literal

FieldComposition = Literal["max", "coverage"]
FieldName = Literal["density", "coverage", "occupancy", "deposition_index"]
