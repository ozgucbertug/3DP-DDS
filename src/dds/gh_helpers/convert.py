"""Conversion helpers between RhinoCommon-style geometry and DDS objects."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from typing import Any, Literal, Optional, Sequence, Union, cast

import numpy as np

import dds
from dds.targets import _target_from_origin

OriginReference = Literal["top", "center"]
DEFAULT_NORMAL = (0.0, 0.0, 1.0)


def point3d_to_tuple(point: object) -> tuple[float, float, float]:
    """Convert a RhinoCommon-style point or numeric triplet to ``(x, y, z)``."""

    return _xyz_tuple(point, name="point")


def vector3d_to_tuple(vector: object) -> tuple[float, float, float]:
    """Convert a RhinoCommon-style vector or numeric triplet to ``(x, y, z)``."""

    return _xyz_tuple(vector, name="vector")


def target_from_point(
    point: object,
    *,
    normal: Optional[object] = None,
) -> dds.DepositionTarget:
    """Create a top-referenced DDS target from a point and optional normal."""

    resolved_normal = DEFAULT_NORMAL if normal is None else vector3d_to_tuple(normal)
    return dds.DepositionTarget(position=point3d_to_tuple(point), normal=resolved_normal)


def target_from_plane(plane: object) -> dds.DepositionTarget:
    """Create a top-referenced DDS target from a Rhino Plane-like object."""

    origin = getattr(plane, "Origin", None)
    if origin is None:
        raise TypeError("plane must expose an Origin point")
    normal = getattr(plane, "Normal", getattr(plane, "ZAxis", None))
    if normal is None:
        raise TypeError("plane must expose a Normal or ZAxis vector")
    return target_from_point(origin, normal=normal)


def convert_target_origin(
    target: dds.DepositionTarget,
    *,
    profile: dds.BeadProfile,
    origin_reference: OriginReference = "top",
) -> dds.DepositionTarget:
    """Convert a top- or center-referenced target to DDS' top reference."""

    return _target_from_origin(target, profile=profile, origin_reference=origin_reference)


def coerce_target(
    value: object,
    *,
    normal: Optional[object] = None,
    profile: Optional[dds.BeadProfile] = None,
    origin_reference: OriginReference = "top",
) -> dds.DepositionTarget:
    """Coerce a DDS target, plane/frame, or point into a deposition target."""

    if isinstance(value, dds.DepositionTarget):
        target = value
    elif _is_plane_like(value):
        target = target_from_plane(value)
    else:
        target = target_from_point(value, normal=normal)
    if origin_reference == "top":
        return target
    if profile is None:
        raise ValueError("profile is required when origin_reference is 'center'")
    return convert_target_origin(target, profile=profile, origin_reference=origin_reference)


def coerce_targets(
    values: object,
    *,
    normal: Optional[object] = None,
    profile: Optional[dds.BeadProfile] = None,
    origin_reference: OriginReference = "top",
) -> tuple[dds.DepositionTarget, ...]:
    """Coerce one or more target-like values into DDS deposition targets."""

    if values is None:
        return ()
    if _is_single_target_like(values):
        return (
            coerce_target(
                values,
                normal=normal,
                profile=profile,
                origin_reference=origin_reference,
            ),
        )
    if hasattr(values, "Branches"):
        targets: list[dds.DepositionTarget] = []
        for branch in values.Branches:
            targets.extend(
                coerce_targets(
                    branch,
                    normal=normal,
                    profile=profile,
                    origin_reference=origin_reference,
                )
            )
        return tuple(targets)
    try:
        iterator = iter(cast(Iterable[object], values))
    except TypeError as exc:
        raise TypeError("targets must be a target-like value or iterable of target-like values") from exc
    return tuple(
        coerce_target(
            value,
            normal=normal,
            profile=profile,
            origin_reference=origin_reference,
        )
        for value in iterator
    )


def bbox_to_domain(
    bounds: object,
    *,
    voxel_size: Union[float, Sequence[float]],
    length_unit: Literal["mm", "m"] = "mm",
) -> dds.Domain:
    """Create a DDS domain from a Rhino BoundingBox or explicit bounds."""

    minimum, maximum = _bounds_min_max(bounds)
    return dds.Domain.from_bounds(
        xmin=minimum[0],
        xmax=maximum[0],
        ymin=minimum[1],
        ymax=maximum[1],
        zmin=minimum[2],
        zmax=maximum[2],
        voxel_size=voxel_size,
        length_unit=length_unit,
    )


def box_to_domain(
    box: object,
    *,
    voxel_size: Union[float, Sequence[float]],
    length_unit: Literal["mm", "m"] = "mm",
) -> dds.Domain:
    """Create a DDS domain from a Rhino Box or box-like object."""

    if hasattr(box, "BoundingBox"):
        return bbox_to_domain(box.BoundingBox, voxel_size=voxel_size, length_unit=length_unit)
    if hasattr(box, "GetBoundingBox"):
        try:
            return bbox_to_domain(box.GetBoundingBox(True), voxel_size=voxel_size, length_unit=length_unit)
        except TypeError:
            return bbox_to_domain(box.GetBoundingBox(), voxel_size=voxel_size, length_unit=length_unit)
    return bbox_to_domain(box, voxel_size=voxel_size, length_unit=length_unit)


def plane_to_target(
    plane: object,
    *,
    profile: Optional[dds.BeadProfile] = None,
    origin_reference: OriginReference = "top",
) -> dds.DepositionTarget:
    """Convert a Rhino Plane-like object into a DDS deposition target."""

    target = target_from_plane(plane)
    if profile is None:
        if origin_reference != "top":
            raise ValueError("profile is required when origin_reference is 'center'")
        return target
    return convert_target_origin(target, profile=profile, origin_reference=origin_reference)


def point_to_deposit(
    point_or_target: object,
    profile: dds.BeadProfile,
    *,
    normal: Optional[object] = None,
    origin_reference: OriginReference = "top",
) -> dds.PointDeposit:
    """Create a DDS point deposit from a point or target."""

    target = coerce_target(
        point_or_target,
        normal=normal,
        profile=profile,
        origin_reference=origin_reference,
    )
    return dds.PointDeposit(target=target, profile=profile)


def targets_to_line_deposit(
    start_target: object,
    end_target: object,
    profile: dds.BeadProfile,
    *,
    sweep_resolution: Optional[float] = None,
    origin_reference: OriginReference = "top",
) -> dds.LineDeposit:
    """Create a DDS line deposit from two target-like values."""

    start = coerce_target(start_target, profile=profile, origin_reference=origin_reference)
    end = coerce_target(end_target, profile=profile, origin_reference=origin_reference)
    return dds.LineDeposit(start, end, profile=profile, sweep_resolution=sweep_resolution)


def targets_to_polyline_deposit(
    targets: object,
    profile: dds.BeadProfile,
    *,
    sweep_resolution: Optional[float] = None,
    origin_reference: OriginReference = "top",
) -> dds.PolylineDeposit:
    """Create a DDS polyline deposit from target-like values."""

    resolved_targets = coerce_targets(targets, profile=profile, origin_reference=origin_reference)
    return dds.PolylineDeposit(resolved_targets, profile=profile, sweep_resolution=sweep_resolution)


def line_to_deposit(
    line: Optional[object],
    profile: dds.BeadProfile,
    *,
    normal: Optional[object] = None,
    start_target: Optional[object] = None,
    end_target: Optional[object] = None,
    sweep_resolution: Optional[float] = None,
    origin_reference: OriginReference = "top",
) -> dds.LineDeposit:
    """Convert a Rhino Line-like object into a DDS line deposit."""

    if start_target is not None and end_target is not None:
        start_obj = start_target
        end_obj = end_target
    else:
        if line is None:
            raise ValueError("line is required unless start_target and end_target are supplied")
        start_obj, end_obj = _line_endpoints(line)
    start = coerce_target(
        start_obj,
        normal=normal,
        profile=profile,
        origin_reference=origin_reference,
    )
    end = coerce_target(
        end_obj,
        normal=normal,
        profile=profile,
        origin_reference=origin_reference,
    )
    return dds.LineDeposit(start, end, profile=profile, sweep_resolution=sweep_resolution)


def polyline_to_deposit(
    polyline: Optional[object],
    profile: dds.BeadProfile,
    *,
    targets: Optional[object] = None,
    normal: Optional[object] = None,
    sweep_resolution: Optional[float] = None,
    origin_reference: OriginReference = "top",
) -> dds.PolylineDeposit:
    """Convert a Rhino Polyline-like object into a DDS polyline deposit."""

    if targets is None:
        if polyline is None:
            raise ValueError("polyline is required unless targets are supplied")
        target_values = tuple(_iter_polyline_points(polyline))
    else:
        target_values = tuple(_iter_target_values(targets))
    resolved_targets = tuple(
        coerce_target(
            target_value,
            normal=normal,
            profile=profile,
            origin_reference=origin_reference,
        )
        for target_value in target_values
    )
    return dds.PolylineDeposit(resolved_targets, profile=profile, sweep_resolution=sweep_resolution)


def curve_to_deposit(
    curve: object,
    profile: dds.BeadProfile,
    *,
    spacing: Optional[float] = None,
    count: Optional[int] = None,
    normal: Optional[object] = None,
    sweep_resolution: Optional[float] = None,
    origin_reference: OriginReference = "top",
) -> dds.PolylineDeposit:
    """Sample a Rhino Curve-like object into a DDS polyline deposit."""

    polyline_points = _try_curve_polyline_points(curve)
    if polyline_points is not None and spacing is None and count is None:
        return polyline_to_deposit(
            polyline_points,
            profile,
            normal=normal,
            sweep_resolution=sweep_resolution,
            origin_reference=origin_reference,
        )

    if spacing is not None and count is not None:
        raise ValueError("provide spacing or count, not both")
    if spacing is None and count is None:
        raise ValueError("curve_to_deposit requires spacing or count unless the curve is already a polyline")
    if spacing is not None:
        if spacing <= 0:
            raise ValueError("spacing must be positive")
        parameters = _curve_divide_by_length(curve, spacing)
    else:
        assert count is not None
        if count < 1:
            raise ValueError("count must be at least 1")
        parameters = _curve_divide_by_count(curve, count)
    points = tuple(_curve_point_at(curve, parameter) for parameter in parameters)
    return polyline_to_deposit(
        points,
        profile,
        normal=normal,
        sweep_resolution=sweep_resolution,
        origin_reference=origin_reference,
    )


def triangle_mesh_to_rhino(mesh: object) -> object:
    """Convert a DDS ``TriangleMesh`` to ``Rhino.Geometry.Mesh``."""

    rhino = _require_rhino()
    rhino_mesh = rhino.Geometry.Mesh()
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    for x, y, z in vertices:
        rhino_mesh.Vertices.Add(float(x), float(y), float(z))
    for face in faces:
        rhino_mesh.Faces.AddFace(int(face[0]), int(face[1]), int(face[2]))
    _apply_vertex_colors(rhino_mesh, getattr(mesh, "vertex_colors", None))
    rhino_mesh.Normals.ComputeNormals()
    rhino_mesh.Compact()
    return rhino_mesh


def result_to_rhino_mesh(
    result: dds.SimulationResult,
    *,
    threshold: Optional[float] = None,
    step_size: int = 1,
) -> object:
    """Extract a DDS result surface and convert it to a Rhino mesh."""

    return triangle_mesh_to_rhino(result.analysis.surface_mesh(threshold=threshold, step_size=step_size))


def domain_to_rhino_box(domain: dds.Domain) -> object:
    """Convert an axis-aligned DDS domain to a Rhino ``Box``."""

    rhino = _require_rhino()
    return rhino.Geometry.Box(
        rhino.Geometry.Plane.WorldXY,
        rhino.Geometry.Interval(domain.min_corner[0], domain.max_corner[0]),
        rhino.Geometry.Interval(domain.min_corner[1], domain.max_corner[1]),
        rhino.Geometry.Interval(domain.min_corner[2], domain.max_corner[2]),
    )


def coerce_deposits(value: object) -> tuple[dds.Deposit, ...]:
    """Flatten a GH-style item, list, tuple, tree, or iterable into deposits."""

    deposits: list[dds.Deposit] = []
    _collect_deposits(value, deposits)
    return tuple(deposits)


def summarize_domain(domain: dds.Domain) -> str:
    """Return a compact human-readable domain summary for GH panels."""

    nx, ny, nz = domain.grid_shape
    vx, vy, vz = domain.voxel_size
    memory = estimate_field_memory(domain)
    return (
        f"Domain {nx} x {ny} x {nz} voxels, voxel {vx:g}, {vy:g}, {vz:g} {domain.length_unit}, "
        f"field {memory / 1024**2:.1f} MiB"
    )


def summarize_result(result: dds.SimulationResult) -> str:
    """Return a compact human-readable simulation summary for GH panels."""

    occupancy = result.analysis.occupancy()
    occupied = int(np.count_nonzero(occupancy))
    fraction = occupied / occupancy.size if occupancy.size else 0.0
    coverage = "with coverage" if result.coverage is not None else "implicit only"
    return f"Result {len(result.deposits)} deposits, {occupied} occupied voxels, {fraction:.3%} occupied, {coverage}"


def estimate_field_memory(domain: dds.Domain, fields: int = 1) -> int:
    """Estimate dense ``float64`` field memory in bytes."""

    if fields < 1:
        raise ValueError("fields must be at least 1")
    return int(math.prod(domain.grid_shape) * np.dtype(np.float64).itemsize * fields)


def stable_signature(*values: object) -> str:
    """Return a deterministic hash for simple DDS/GH cache keys."""

    digest = hashlib.sha256()
    for value in values:
        digest.update(repr(_signature_value(value)).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _xyz_tuple(value: object, *, name: str) -> tuple[float, float, float]:
    if all(hasattr(value, attr) for attr in ("X", "Y", "Z")):
        xyz = (value.X, value.Y, value.Z)
    elif all(hasattr(value, attr) for attr in ("x", "y", "z")):
        xyz = (value.x, value.y, value.z)
    else:
        try:
            xyz = tuple(cast(Iterable[object], value))
        except TypeError as exc:
            raise TypeError(f"{name} must be a point/vector object or a three-value sequence") from exc
    if len(xyz) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    floats = tuple(float(component) for component in xyz)
    if not np.all(np.isfinite(floats)):
        raise ValueError(f"{name} values must be finite")
    return floats[0], floats[1], floats[2]


def _is_plane_like(value: object) -> bool:
    return hasattr(value, "Origin") and (hasattr(value, "Normal") or hasattr(value, "ZAxis"))


def _is_point_like(value: object) -> bool:
    if all(hasattr(value, attr) for attr in ("X", "Y", "Z")):
        return True
    if all(hasattr(value, attr) for attr in ("x", "y", "z")):
        return True
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3:
        try:
            tuple(float(component) for component in value)
        except (TypeError, ValueError):
            return False
        return True
    return False


def _is_single_target_like(value: object) -> bool:
    return isinstance(value, dds.DepositionTarget) or _is_plane_like(value) or _is_point_like(value)


def _bounds_min_max(bounds: object) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if hasattr(bounds, "Min") and hasattr(bounds, "Max"):
        return point3d_to_tuple(bounds.Min), point3d_to_tuple(bounds.Max)
    if isinstance(bounds, Sequence) and len(bounds) == 6:
        xmin, xmax, ymin, ymax, zmin, zmax = (float(value) for value in bounds)
        return (xmin, ymin, zmin), (xmax, ymax, zmax)
    if isinstance(bounds, Sequence) and len(bounds) == 2:
        return point3d_to_tuple(bounds[0]), point3d_to_tuple(bounds[1])
    if hasattr(bounds, "GetCorners"):
        corners = tuple(bounds.GetCorners())
        coords = np.asarray([point3d_to_tuple(point) for point in corners], dtype=float)
        minimum = tuple(float(value) for value in coords.min(axis=0).tolist())
        maximum = tuple(float(value) for value in coords.max(axis=0).tolist())
        return minimum, maximum
    raise TypeError("bounds must be a BoundingBox, two points, six bounds, or expose GetCorners()")


def _iter_polyline_points(polyline: object) -> Iterable[object]:
    if hasattr(polyline, "ToArray"):
        yield from polyline.ToArray()
        return
    if hasattr(polyline, "Count") and hasattr(polyline, "__getitem__"):
        for index in range(int(polyline.Count)):
            yield polyline[index]
        return
    try:
        yield from cast(Iterable[object], polyline)
    except TypeError as exc:
        raise TypeError("polyline must be iterable or expose ToArray()/Count") from exc


def _iter_target_values(targets: object) -> Iterable[object]:
    if _is_single_target_like(targets):
        yield targets
        return
    if hasattr(targets, "Branches"):
        for branch in targets.Branches:
            yield from _iter_target_values(branch)
        return
    try:
        yield from cast(Iterable[object], targets)
    except TypeError as exc:
        raise TypeError("targets must be a DDS target or iterable of targets") from exc


def _line_endpoints(line: object) -> tuple[object, object]:
    start_obj = getattr(line, "From", getattr(line, "FromPoint", None))
    end_obj = getattr(line, "To", getattr(line, "ToPoint", None))
    if start_obj is not None and end_obj is not None:
        return start_obj, end_obj
    try:
        start_obj, end_obj = cast(Sequence[object], line)
    except (TypeError, ValueError) as exc:
        raise TypeError("line must expose From/To points or contain two points") from exc
    return start_obj, end_obj


def _try_curve_polyline_points(curve: object) -> Optional[tuple[object, ...]]:
    if hasattr(curve, "TryGetPolyline"):
        result = curve.TryGetPolyline()
        if isinstance(result, tuple) and len(result) == 2 and result[0]:
            return tuple(_iter_polyline_points(result[1]))
        if result:
            return tuple(_iter_polyline_points(result))
    return None


def _curve_divide_by_length(curve: object, spacing: float) -> tuple[float, ...]:
    if not hasattr(curve, "DivideByLength"):
        raise TypeError("curve must expose DivideByLength() when spacing is used")
    parameters = curve.DivideByLength(float(spacing), True)
    if parameters is None:
        raise ValueError("curve division by length produced no parameters")
    return tuple(float(parameter) for parameter in parameters)


def _curve_divide_by_count(curve: object, count: int) -> tuple[float, ...]:
    if not hasattr(curve, "DivideByCount"):
        raise TypeError("curve must expose DivideByCount() when count is used")
    parameters = curve.DivideByCount(int(count), True)
    if parameters is None:
        raise ValueError("curve division by count produced no parameters")
    return tuple(float(parameter) for parameter in parameters)


def _curve_point_at(curve: object, parameter: float) -> object:
    if not hasattr(curve, "PointAt"):
        raise TypeError("curve must expose PointAt()")
    return curve.PointAt(parameter)


def _collect_deposits(value: object, deposits: list[dds.Deposit]) -> None:
    if value is None:
        return
    if isinstance(value, (dds.PointDeposit, dds.LineDeposit, dds.PolylineDeposit)):
        deposits.append(value)
        return
    if isinstance(value, (str, bytes)):
        raise TypeError("deposits cannot be strings")
    if hasattr(value, "Branches"):
        for branch in value.Branches:
            _collect_deposits(branch, deposits)
        return
    try:
        iterator = iter(cast(Iterable[object], value))
    except TypeError as exc:
        raise TypeError("expected a DDS deposit or an iterable of deposits") from exc
    for item in iterator:
        _collect_deposits(item, deposits)


def _apply_vertex_colors(rhino_mesh: object, vertex_colors: object) -> None:
    if vertex_colors is None:
        return
    try:
        from System.Drawing import Color  # type: ignore[import-not-found]
    except ImportError:
        return
    for color in np.asarray(vertex_colors, dtype=np.uint8):
        if len(color) == 3:
            rhino_mesh.VertexColors.Add(Color.FromArgb(int(color[0]), int(color[1]), int(color[2])))
        else:
            rhino_mesh.VertexColors.Add(Color.FromArgb(int(color[3]), int(color[0]), int(color[1]), int(color[2])))


def _require_rhino() -> Any:
    try:
        import Rhino  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("This function requires RhinoCommon and must run inside Rhino/Grasshopper.") from exc
    return Rhino


def _signature_value(value: object) -> object:
    if isinstance(value, dds.Domain):
        return value.to_dict()
    if isinstance(value, dds.BeadProfile):
        return value.to_dict()
    if isinstance(value, dds.DepositionTarget):
        return value.to_dict()
    if isinstance(value, dds.PointDeposit):
        return ("PointDeposit", _signature_value(value.target), _signature_value(value.profile))
    if isinstance(value, dds.LineDeposit):
        return (
            "LineDeposit",
            _signature_value(value.start),
            _signature_value(value.end),
            _signature_value(value.profile),
            value.sweep_resolution,
        )
    if isinstance(value, dds.PolylineDeposit):
        return (
            "PolylineDeposit",
            tuple(_signature_value(target) for target in value.targets),
            _signature_value(value.profile),
            value.sweep_resolution,
        )
    if isinstance(value, tuple):
        return tuple(_signature_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_signature_value(item) for item in value)
    return value
