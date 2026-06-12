"""Conversions from DDS geometry types to PyVista datasets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from ..geometry import PointCloud, TriangleMesh
from ..primitives import Line3D, Point3D, Polyline3D


def triangle_mesh_to_polydata(mesh: TriangleMesh, pv: Any) -> Any:
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    if mesh.is_empty:
        return pv.PolyData()
    faces = np.hstack(
        [
            np.full((mesh.n_faces, 1), 3, dtype=np.int64),
            mesh.faces.astype(np.int64, copy=False),
        ]
    ).ravel()
    return pv.PolyData(np.asarray(mesh.vertices, dtype=float), faces)


def points_to_polydata(points: Sequence[Point3D], pv: Any) -> Any:
    resolved = tuple(points)
    if not resolved:
        raise ValueError("points must not be empty")
    if not all(isinstance(point, Point3D) for point in resolved):
        raise TypeError("points must contain Point3D values")
    values = np.asarray([point.to_tuple() for point in resolved], dtype=float)
    return pv.PolyData(values)


def point_cloud_to_polydata(cloud: PointCloud, pv: Any) -> Any:
    if not isinstance(cloud, PointCloud):
        raise TypeError("cloud must be a PointCloud")
    dataset = pv.PolyData(np.asarray(cloud.points, dtype=float))
    if cloud.colors is not None:
        dataset.point_data["point_colors"] = np.asarray(cloud.colors, dtype=np.uint8)
    return dataset


def line_to_polydata(line: Line3D, pv: Any) -> Any:
    if not isinstance(line, Line3D):
        raise TypeError("line must be a Line3D")
    return pv.Line(
        cast(Point3D, line.start).to_tuple(),
        cast(Point3D, line.end).to_tuple(),
    )


def polyline_to_polydata(polyline: Polyline3D, pv: Any) -> Any:
    if not isinstance(polyline, Polyline3D):
        raise TypeError("polyline must be a Polyline3D")
    points = np.asarray(
        [cast(Point3D, point).to_tuple() for point in polyline.points],
        dtype=float,
    )
    return pv.lines_from_points(points, close=False)
