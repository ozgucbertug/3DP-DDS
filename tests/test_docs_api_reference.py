from __future__ import annotations

import inspect
from pathlib import Path

import dds
from dds import fields
from dds.analysis import (
    InterfaceAnalysis,
    InterfacePairSummary,
    SimulationAnalysis,
    StratumFieldSet,
    SupportAnalysis,
)
from dds.occupancy import occupancy_fraction, occupancy_from_implicit_field
from dds.primitives import iter_deposits

ROOT = Path(__file__).resolve().parents[1]

DOC_API_SYMBOLS = {
    "docs/source/api/core.rst": [
        "BeadProfile",
        "Domain",
        "Point3D",
        "Vector3D",
        "Pose3D",
        "DepositionTarget",
        "Line3D",
        "Polyline3D",
        "PointDeposit",
        "LineDeposit",
        "PolylineDeposit",
        "SimulationResult",
        "Simulator",
        "ChunkedField",
        "simulate",
    ],
    "docs/source/api/primitives.rst": ["iter_deposits"],
    "docs/source/api/fields.rst": [
        "accumulate_fields",
        "accumulate_field",
        "accumulate_deposition_index",
        "accumulate_deposition_order",
        "apply_deposit_to_field",
        "apply_deposit_to_index_field",
        "accumulate_chunked_field",
    ],
    "docs/source/api/analysis.rst": [
        "SimulationAnalysis",
        "StratumFieldSet",
        "InterfacePairSummary",
        "InterfaceAnalysis",
        "SupportAnalysis",
    ],
    "docs/source/api/occupancy.rst": [
        "occupancy_from_implicit_field",
        "occupancy_fraction",
    ],
}

DOCUMENTED_OBJECTS = [
    dds.BeadProfile,
    dds.Domain,
    dds.Point3D,
    dds.Vector3D,
    dds.Pose3D,
    dds.DepositionTarget,
    dds.Line3D,
    dds.Polyline3D,
    dds.PointDeposit,
    dds.LineDeposit,
    dds.PolylineDeposit,
    iter_deposits,
    dds.SimulationResult,
    dds.Simulator,
    dds.ChunkedField,
    dds.simulate,
    fields.accumulate_fields,
    fields.accumulate_field,
    fields.accumulate_deposition_index,
    fields.accumulate_deposition_order,
    fields.apply_deposit_to_field,
    fields.apply_deposit_to_index_field,
    fields.accumulate_chunked_field,
    SimulationAnalysis,
    StratumFieldSet,
    InterfacePairSummary,
    InterfaceAnalysis,
    SupportAnalysis,
    occupancy_from_implicit_field,
    occupancy_fraction,
]


def test_core_api_symbols_are_listed_in_reference_pages() -> None:
    for path, symbols in DOC_API_SYMBOLS.items():
        text = (ROOT / path).read_text(encoding="utf-8")
        for symbol in symbols:
            assert f"   {symbol}" in text


def test_documented_core_objects_have_docstrings() -> None:
    for obj in DOCUMENTED_OBJECTS:
        doc = inspect.getdoc(obj)
        assert doc, f"{obj!r} is missing a docstring"
        assert len(doc.split()) >= 5, f"{obj!r} docstring is too thin"
