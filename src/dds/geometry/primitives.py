"""Analytic SDF primitives inspired by sdfCAD's core subset."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..utils import ensure_finite_triplet, ensure_positive_triplet
from .ops import intersection
from .sdf import SDF3

ORIGIN = np.asarray((0.0, 0.0, 0.0), dtype=float)
X = np.asarray((1.0, 0.0, 0.0), dtype=float)
Y = np.asarray((0.0, 1.0, 0.0), dtype=float)
Z = np.asarray((0.0, 0.0, 1.0), dtype=float)


def _vector(value: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    return np.asarray(ensure_finite_triplet(value, name), dtype=float)


def _size_vector(value: float | Sequence[float], *, name: str) -> np.ndarray:
    if isinstance(value, (int, float)):
        scalar = float(value)
        if scalar <= 0.0:
            raise ValueError(f"{name} must be positive.")
        return np.asarray((scalar, scalar, scalar), dtype=float)
    return np.asarray(ensure_positive_triplet(value, name), dtype=float)


def _radius_from_inputs(
    *,
    radius: float | None = None,
    diameter: float | None = None,
    name: str = "radius",
) -> float:
    if (radius is None) == (diameter is None):
        raise ValueError(f"Specify exactly one of {name}=... or diameter=....")
    result = float(radius) if radius is not None else float(diameter) / 2.0
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _normalize_axis(axis: Sequence[float] | np.ndarray) -> np.ndarray:
    axis_array = _vector(axis, name="axis")
    length = float(np.linalg.norm(axis_array))
    if length == 0.0:
        raise ValueError("axis must be non-zero.")
    return axis_array / length


def sphere(
    *,
    radius: float | None = None,
    diameter: float | None = None,
    center: Sequence[float] | np.ndarray = ORIGIN,
) -> SDF3:
    """Return a sphere SDF."""

    resolved_radius = _radius_from_inputs(radius=radius, diameter=diameter, name="radius")
    center_array = _vector(center, name="center")
    return SDF3(lambda points: np.linalg.norm(points - center_array, axis=1) - resolved_radius, name="sphere")


def plane(
    normal: Sequence[float] | np.ndarray = Z,
    point: Sequence[float] | np.ndarray = ORIGIN,
) -> SDF3:
    """Return a plane SDF with negative values in the half-space pointed to by `normal`."""

    point_array = _vector(point, name="point")
    normal_array = _normalize_axis(normal)
    return SDF3(lambda points: (point_array - points) @ normal_array, name="plane")


def slab(
    *,
    x0: float | None = None,
    y0: float | None = None,
    z0: float | None = None,
    x1: float | None = None,
    y1: float | None = None,
    z1: float | None = None,
    dx: float | None = None,
    dy: float | None = None,
    dz: float | None = None,
) -> SDF3:
    """Return an axis-aligned slab or bounded intersection of half-spaces."""

    if dx is not None:
        if dx <= 0.0:
            raise ValueError("dx must be positive.")
        x0 = -dx / 2.0 if x0 is None else x0
        x1 = dx / 2.0 if x1 is None else x1
    if dy is not None:
        if dy <= 0.0:
            raise ValueError("dy must be positive.")
        y0 = -dy / 2.0 if y0 is None else y0
        y1 = dy / 2.0 if y1 is None else y1
    if dz is not None:
        if dz <= 0.0:
            raise ValueError("dz must be positive.")
        z0 = -dz / 2.0 if z0 is None else z0
        z1 = dz / 2.0 if z1 is None else z1

    parts: list[SDF3] = []
    if x0 is not None:
        parts.append(plane(X, (x0, 0.0, 0.0)))
    if x1 is not None:
        parts.append(plane(-X, (x1, 0.0, 0.0)))
    if y0 is not None:
        parts.append(plane(Y, (0.0, y0, 0.0)))
    if y1 is not None:
        parts.append(plane(-Y, (0.0, y1, 0.0)))
    if z0 is not None:
        parts.append(plane(Z, (0.0, 0.0, z0)))
    if z1 is not None:
        parts.append(plane(-Z, (0.0, 0.0, z1)))
    if not parts:
        raise ValueError("slab requires at least one bound or extent.")
    return intersection(*parts)


def box(
    *,
    size: float | Sequence[float] = 1.0,
    center: Sequence[float] | np.ndarray = ORIGIN,
    a: Sequence[float] | np.ndarray | None = None,
    b: Sequence[float] | np.ndarray | None = None,
) -> SDF3:
    """Return an axis-aligned box SDF."""

    if a is not None and b is not None:
        minimum = _vector(a, name="a")
        maximum = _vector(b, name="b")
        extents = maximum - minimum
        if np.any(extents <= 0.0):
            raise ValueError("Box corner `b` must exceed `a` on every axis.")
        return box(size=extents, center=minimum + extents / 2.0)

    size_array = _size_vector(size, name="size")
    center_array = _vector(center, name="center")

    def evaluate(points: np.ndarray) -> np.ndarray:
        q = np.abs(points - center_array) - size_array / 2.0
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)

    return SDF3(evaluate, name="box")


def cylinder(
    *,
    radius: float | None = None,
    diameter: float | None = None,
    height: float | None = None,
    center: Sequence[float] | np.ndarray = ORIGIN,
    axis: Sequence[float] | np.ndarray = Z,
) -> SDF3:
    """Return an infinite or capped cylinder SDF."""

    resolved_radius = _radius_from_inputs(radius=radius, diameter=diameter, name="radius")
    if height is not None and height <= 0.0:
        raise ValueError("height must be positive when provided.")
    center_array = _vector(center, name="center")
    axis_array = _normalize_axis(axis)

    def evaluate(points: np.ndarray) -> np.ndarray:
        relative = points - center_array
        axial = relative @ axis_array
        radial_vectors = relative - axial[:, np.newaxis] * axis_array
        radial_distance = np.linalg.norm(radial_vectors, axis=1) - resolved_radius
        if height is None:
            return radial_distance
        bounds = np.stack((radial_distance, np.abs(axial) - height / 2.0), axis=-1)
        return np.minimum(np.maximum(bounds[:, 0], bounds[:, 1]), 0.0) + np.linalg.norm(
            np.maximum(bounds, 0.0),
            axis=1,
        )

    return SDF3(evaluate, name="cylinder")


def capsule(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    radius: float | None = None,
    diameter: float | None = None,
) -> SDF3:
    """Return a capsule SDF between two endpoints."""

    resolved_radius = _radius_from_inputs(radius=radius, diameter=diameter, name="radius")
    start = _vector(a, name="a")
    end = _vector(b, name="b")
    axis = end - start
    axis_length_sq = float(np.dot(axis, axis))
    if axis_length_sq == 0.0:
        return sphere(radius=resolved_radius, center=start)

    def evaluate(points: np.ndarray) -> np.ndarray:
        relative = points - start
        projection = np.clip((relative @ axis) / axis_length_sq, 0.0, 1.0)
        closest = start + projection[:, np.newaxis] * axis
        return np.linalg.norm(points - closest, axis=1) - resolved_radius

    return SDF3(evaluate, name="capsule")


def ellipsoid(
    *,
    size: float | Sequence[float],
    center: Sequence[float] | np.ndarray = ORIGIN,
) -> SDF3:
    """Return an ellipsoid SDF using the standard IQ approximation."""

    size_array = _size_vector(size, name="size")
    center_array = _vector(center, name="center")

    def evaluate(points: np.ndarray) -> np.ndarray:
        relative = points - center_array
        k0 = np.linalg.norm(relative / size_array, axis=1)
        k1 = np.linalg.norm(relative / (size_array * size_array), axis=1)
        safe_k1 = np.where(k1 == 0.0, 1.0, k1)
        values = k0 * (k0 - 1.0) / safe_k1
        values[k1 == 0.0] = -np.min(size_array)
        return values

    return SDF3(evaluate, name="ellipsoid")


def torus(
    *,
    major_radius: float | None = None,
    minor_radius: float | None = None,
    r1: float | None = None,
    r2: float | None = None,
    center: Sequence[float] | np.ndarray = ORIGIN,
    axis: Sequence[float] | np.ndarray = Z,
) -> SDF3:
    """Return a torus SDF around the given axis."""

    major = major_radius if major_radius is not None else r1
    minor = minor_radius if minor_radius is not None else r2
    if major is None or minor is None:
        raise ValueError("Specify major_radius/r1 and minor_radius/r2.")
    major = float(major)
    minor = float(minor)
    if major <= 0.0 or minor <= 0.0:
        raise ValueError("Torus radii must be positive.")
    center_array = _vector(center, name="center")
    axis_array = _normalize_axis(axis)

    def evaluate(points: np.ndarray) -> np.ndarray:
        relative = points - center_array
        axial = relative @ axis_array
        radial_vectors = relative - axial[:, np.newaxis] * axis_array
        ring = np.linalg.norm(radial_vectors, axis=1) - major
        return np.sqrt(ring * ring + axial * axial) - minor

    return SDF3(evaluate, name="torus")
