from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import dds
from dds.gh_helpers import components, convert

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FakePoint:
    X: float
    Y: float
    Z: float


@dataclass(frozen=True)
class FakeVector:
    X: float
    Y: float
    Z: float


@dataclass(frozen=True)
class FakePlane:
    Origin: FakePoint
    Normal: FakeVector


@dataclass(frozen=True)
class FakeLine:
    From: FakePoint
    To: FakePoint


class FakePolyline:
    def __init__(self, points: list[FakePoint]) -> None:
        self._points = points

    @property
    def Count(self) -> int:
        return len(self._points)

    def __getitem__(self, index: int) -> FakePoint:
        return self._points[index]


class FakeBoundingBox:
    def __init__(self, minimum: FakePoint, maximum: FakePoint) -> None:
        self.Min = minimum
        self.Max = maximum


class FakeBox:
    def __init__(self, minimum: FakePoint, maximum: FakePoint) -> None:
        self.BoundingBox = FakeBoundingBox(minimum, maximum)


class FakeCurve:
    def DivideByCount(self, count: int, include_ends: bool) -> list[float]:
        assert include_ends
        return [index / count for index in range(count + 1)]

    def PointAt(self, parameter: float) -> FakePoint:
        return FakePoint(parameter, 0.0, 1.0)


def test_core_import_does_not_import_gh_or_viz_modules() -> None:
    code = """
import sys
before = set(sys.modules)
import dds
import dds.geometry
added = set(sys.modules) - before
bad = sorted(
    name for name in added
    if name in {"Rhino", "scriptcontext", "dds.viz", "dds.gh_helpers"}
    or name.startswith(("pyvista", "PySide6", "rhinoscriptsyntax"))
)
if bad:
    raise SystemExit("\\n".join(bad))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_gh_helpers_import_does_not_import_rhino_or_viz_modules() -> None:
    code = """
import sys
before = set(sys.modules)
import dds.gh_helpers
added = set(sys.modules) - before
bad = sorted(
    name for name in added
    if name in {"Rhino", "scriptcontext", "dds.viz"}
    or name.startswith(("pyvista", "PySide6", "rhinoscriptsyntax"))
)
if bad:
    raise SystemExit("\\n".join(bad))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_box_to_domain_uses_bounding_box_and_aligned_voxels() -> None:
    domain = convert.box_to_domain(
        FakeBox(FakePoint(0, 0, 0), FakePoint(1.1, 2.0, 3.0)),
        voxel_size=0.5,
    )

    assert domain.min_corner == (0.0, 0.0, 0.0)
    assert domain.max_corner == (1.5, 2.0, 3.0)
    assert domain.grid_shape == (3, 4, 6)


def test_plane_to_target_center_reference_offsets_to_top() -> None:
    profile = dds.BeadProfile(width=2.0, height=0.4)
    target = convert.plane_to_target(
        FakePlane(FakePoint(1, 2, 3), FakeVector(0, 0, 1)),
        profile=profile,
        origin_reference="center",
    )

    assert target.position.to_tuple() == pytest.approx((1.0, 2.0, 3.2))
    assert target.normal.to_tuple() == pytest.approx((0.0, 0.0, 1.0))


def test_line_and_polyline_deposit_conversion() -> None:
    profile = dds.BeadProfile(width=1.0, height=0.2)
    line = convert.line_to_deposit(
        FakeLine(FakePoint(0, 0, 1), FakePoint(2, 0, 1)),
        profile,
        sweep_resolution=0.25,
    )
    polyline = convert.polyline_to_deposit(
        FakePolyline([FakePoint(0, 0, 1), FakePoint(1, 0, 1), FakePoint(1, 1, 1)]),
        profile,
    )

    assert line.line.length == pytest.approx(2.0)
    assert line.sweep_resolution == pytest.approx(0.25)
    assert len(polyline.targets) == 3
    assert polyline.polyline.length == pytest.approx(2.0)


def test_deposit_helpers_accept_normals_and_targets() -> None:
    profile = dds.BeadProfile(width=1.0, height=0.2)
    normal = FakeVector(0, 1, 0)
    start = convert.target_from_point(FakePoint(0, 0, 1), normal=normal)
    end = convert.target_from_point(FakePoint(2, 0, 1), normal=normal)

    point = convert.point_to_deposit(FakePoint(0, 0, 1), profile, normal=normal)
    line = convert.line_to_deposit(None, profile, start_target=start, end_target=end)
    polyline = convert.polyline_to_deposit(None, profile, targets=[start, end])

    assert point.target.normal.to_tuple() == pytest.approx((0.0, 1.0, 0.0))
    assert line.start.normal.to_tuple() == pytest.approx((0.0, 1.0, 0.0))
    assert line.end.normal.to_tuple() == pytest.approx((0.0, 1.0, 0.0))
    assert [target.normal.to_tuple() for target in polyline.targets] == pytest.approx(
        [(0.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    )


def test_target_coercion_accepts_plane_or_point_with_default_z() -> None:
    plane = FakePlane(FakePoint(1, 2, 3), FakeVector(0, 1, 0))

    from_plane = convert.coerce_target(plane)
    from_point = convert.coerce_target(FakePoint(1, 2, 3))

    assert from_plane.position.to_tuple() == pytest.approx((1.0, 2.0, 3.0))
    assert from_plane.normal.to_tuple() == pytest.approx((0.0, 1.0, 0.0))
    assert from_point.normal.to_tuple() == pytest.approx((0.0, 0.0, 1.0))


def test_target_first_deposit_helpers_accept_planes_and_points() -> None:
    profile = dds.BeadProfile(width=1.0, height=0.2)
    start_plane = FakePlane(FakePoint(0, 0, 1), FakeVector(0, 1, 0))
    end_plane = FakePlane(FakePoint(2, 0, 1), FakeVector(0, 1, 0))

    point = convert.point_to_deposit(start_plane, profile)
    line = convert.targets_to_line_deposit(start_plane, end_plane, profile)
    polyline = convert.targets_to_polyline_deposit(
        [start_plane, FakePoint(1, 0, 1), end_plane],
        profile,
    )

    assert point.target.normal.to_tuple() == pytest.approx((0.0, 1.0, 0.0))
    assert line.start.normal.to_tuple() == pytest.approx((0.0, 1.0, 0.0))
    assert line.end.normal.to_tuple() == pytest.approx((0.0, 1.0, 0.0))
    assert polyline.targets[1].normal.to_tuple() == pytest.approx((0.0, 0.0, 1.0))


def test_component_line_deposit_supports_target_first_call() -> None:
    profile = dds.BeadProfile(width=1.0, height=0.2)
    start = FakePlane(FakePoint(0, 0, 1), FakeVector(0, 1, 0))
    end = FakePlane(FakePoint(2, 0, 1), FakeVector(0, 1, 0))

    deposit = components.make_line_deposit(start, end, profile)

    assert deposit.line.length == pytest.approx(2.0)


def test_component_helpers_return_objects_without_info_payloads() -> None:
    profile = components.make_bead_profile(1.0, 0.2)
    target = components.make_target(FakePoint(0, 0, 1))
    point = components.make_point_deposit(target, profile)
    line = components.make_line_deposit(FakePoint(0, 0, 1), FakePoint(1, 0, 1), profile)
    polyline = components.make_polyline_deposit([FakePoint(0, 0, 1), FakePoint(1, 0, 1)], profile)

    assert isinstance(profile, dds.BeadProfile)
    assert isinstance(target, dds.DepositionTarget)
    assert isinstance(point, dds.PointDeposit)
    assert isinstance(line, dds.LineDeposit)
    assert isinstance(polyline, dds.PolylineDeposit)


def test_line_helper_uses_shared_normal_when_targets_are_not_supplied() -> None:
    profile = dds.BeadProfile(width=1.0, height=0.2)
    deposit = convert.line_to_deposit(
        FakeLine(FakePoint(0, 0, 1), FakePoint(2, 0, 1)),
        profile,
        normal=FakeVector(1, 0, 0),
    )

    assert deposit.start.normal.to_tuple() == pytest.approx((1.0, 0.0, 0.0))
    assert deposit.end.normal.to_tuple() == pytest.approx((1.0, 0.0, 0.0))


def test_curve_to_deposit_samples_by_count() -> None:
    profile = dds.BeadProfile(width=1.0, height=0.2)
    deposit = convert.curve_to_deposit(FakeCurve(), profile, count=4)

    assert len(deposit.targets) == 5
    assert deposit.targets[-1].position.to_tuple() == pytest.approx((1.0, 0.0, 1.0))


def test_coerce_deposits_flattens_nested_inputs() -> None:
    profile = dds.BeadProfile(width=1.0, height=0.2)
    point = dds.PointDeposit((0, 0, 1), profile)
    line = dds.LineDeposit((0, 0, 1), (1, 0, 1), profile)

    assert convert.coerce_deposits([[point], (line,)]) == (point, line)


def test_result_summary_and_inspect_points_use_current_analysis_api() -> None:
    profile = dds.BeadProfile(width=1.0, height=0.5)
    deposit = dds.PointDeposit((0.5, 0.5, 1.0), profile)
    domain = dds.Domain.from_bounds(
        xmin=0,
        xmax=1,
        ymin=0,
        ymax=1,
        zmin=0,
        zmax=1,
        voxel_size=0.25,
    )
    result = dds.simulate(domain, deposit)

    assert "occupied voxels" in convert.summarize_result(result)
    values, occupied, indices = components.inspect_points(result, [FakePoint(0.5, 0.5, 0.75)])
    assert len(values) == len(occupied) == len(indices) == 1


def test_triangle_mesh_to_rhino_is_guarded_outside_rhino() -> None:
    mesh = type(
        "Mesh",
        (),
        {
            "vertices": np.empty((0, 3), dtype=float),
            "faces": np.empty((0, 3), dtype=np.int64),
            "vertex_colors": None,
        },
    )()

    with pytest.raises(RuntimeError, match="requires RhinoCommon"):
        convert.triangle_mesh_to_rhino(mesh)
