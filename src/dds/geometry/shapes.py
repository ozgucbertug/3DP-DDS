"""Analytic SDF primitives inspired by sdfCAD's core subset."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..utils import ensure_finite_triplet, ensure_positive_triplet
from .ops import intersection, union
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
    result = _radius_value(radius=radius, diameter=diameter, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _radius_value(
    *,
    radius: float | None = None,
    diameter: float | None = None,
    name: str = "radius",
    allow_zero: bool = False,
) -> float:
    if (radius is None) == (diameter is None):
        raise ValueError(f"Specify exactly one of {name}=... or diameter=....")
    result = float(radius) if radius is not None else float(diameter) / 2.0
    if allow_zero:
        if result < 0.0:
            raise ValueError(f"{name} must be non-negative.")
    elif result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _normalize_axis(axis: Sequence[float] | np.ndarray) -> np.ndarray:
    axis_array = _vector(axis, name="axis")
    length = float(np.linalg.norm(axis_array))
    if length == 0.0:
        raise ValueError("axis must be non-zero.")
    return axis_array / length


def _axis_frame(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    start = _vector(a, name="a")
    end = _vector(b, name="b")
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length == 0.0:
        raise ValueError("Endpoints must not be coincident.")
    return start, end, axis / length, length


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


def rounded_box(
    *,
    size: float | Sequence[float] = 1.0,
    radius: float,
    center: Sequence[float] | np.ndarray = ORIGIN,
) -> SDF3:
    """Return an axis-aligned box with rounded edges."""

    size_array = _size_vector(size, name="size")
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if radius > 0.5 * float(np.min(size_array)):
        raise ValueError("radius must not exceed half of the smallest box dimension.")
    center_array = _vector(center, name="center")

    def evaluate(points: np.ndarray) -> np.ndarray:
        q = np.abs(points - center_array) - size_array / 2.0 + radius
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0) - radius

    return SDF3(evaluate, name="rounded_box")


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


def capped_cylinder(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    radius: float | None = None,
    diameter: float | None = None,
) -> SDF3:
    """Return a finite cylinder between two cap centers."""

    resolved_radius = _radius_from_inputs(radius=radius, diameter=diameter, name="radius")
    start, end, axis_array, length = _axis_frame(a, b)
    center_array = (start + end) / 2.0

    def evaluate(points: np.ndarray) -> np.ndarray:
        relative = points - center_array
        axial = relative @ axis_array
        radial_vectors = relative - axial[:, np.newaxis] * axis_array
        radial_distance = np.linalg.norm(radial_vectors, axis=1) - resolved_radius
        bounds = np.stack((radial_distance, np.abs(axial) - length / 2.0), axis=-1)
        return np.minimum(np.maximum(bounds[:, 0], bounds[:, 1]), 0.0) + np.linalg.norm(
            np.maximum(bounds, 0.0),
            axis=1,
        )

    return SDF3(evaluate, name="capped_cylinder")


def rounded_cylinder(
    *,
    radius: float | None = None,
    diameter: float | None = None,
    height: float,
    rounding_radius: float,
    center: Sequence[float] | np.ndarray = ORIGIN,
    axis: Sequence[float] | np.ndarray = Z,
) -> SDF3:
    """Return a finite cylinder with rounded cap edges."""

    resolved_radius = _radius_from_inputs(radius=radius, diameter=diameter, name="radius")
    height = float(height)
    rounding_radius = float(rounding_radius)
    if height <= 0.0:
        raise ValueError("height must be positive.")
    if rounding_radius <= 0.0:
        raise ValueError("rounding_radius must be positive.")
    if rounding_radius > resolved_radius or rounding_radius > height / 2.0:
        raise ValueError("rounding_radius must fit within the cylinder radius and half height.")
    center_array = _vector(center, name="center")
    axis_array = _normalize_axis(axis)

    def evaluate(points: np.ndarray) -> np.ndarray:
        relative = points - center_array
        axial = relative @ axis_array
        radial_vectors = relative - axial[:, np.newaxis] * axis_array
        bounds = np.stack(
            (
                np.linalg.norm(radial_vectors, axis=1) - resolved_radius + rounding_radius,
                np.abs(axial) - height / 2.0 + rounding_radius,
            ),
            axis=-1,
        )
        return np.minimum(np.maximum(bounds[:, 0], bounds[:, 1]), 0.0) + np.linalg.norm(
            np.maximum(bounds, 0.0),
            axis=1,
        ) - rounding_radius

    return SDF3(evaluate, name="rounded_cylinder")


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


def capped_cone(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    radius_a: float | None = None,
    radius_b: float | None = None,
    diameter_a: float | None = None,
    diameter_b: float | None = None,
) -> SDF3:
    """Return a flat-capped cone or frustum between two cap centers."""

    start, end, axis_array, length = _axis_frame(a, b)
    ra = _radius_value(radius=radius_a, diameter=diameter_a, name="radius_a", allow_zero=True)
    rb = _radius_value(radius=radius_b, diameter=diameter_b, name="radius_b", allow_zero=True)
    if ra == 0.0 and rb == 0.0:
        raise ValueError("At least one cone radius must be positive.")

    def evaluate(points: np.ndarray) -> np.ndarray:
        relative = points - start
        axial = relative @ axis_array
        radial_vectors = relative - axial[:, np.newaxis] * axis_array
        x = np.linalg.norm(radial_vectors, axis=1)
        paba = axial / length
        rba = rb - ra
        length_sq = length * length
        cax = np.maximum(0.0, x - np.where(paba < 0.5, ra, rb))
        cay = np.abs(paba - 0.5) - 0.5
        k = rba * rba + length_sq
        f = np.clip((rba * (x - ra) + paba * length_sq) / k, 0.0, 1.0)
        cbx = x - ra - f * rba
        cby = paba - f
        sign = np.where((cbx < 0.0) & (cay < 0.0), -1.0, 1.0)
        distance_sq = np.minimum(cax * cax + cay * cay * length_sq, cbx * cbx + cby * cby * length_sq)
        return sign * np.sqrt(np.maximum(distance_sq, 0.0))

    return SDF3(evaluate, name="capped_cone")


def cone(
    *,
    height: float,
    radius_bottom: float | None = None,
    radius_top: float | None = None,
    diameter_bottom: float | None = None,
    diameter_top: float | None = None,
    center: Sequence[float] | np.ndarray = ORIGIN,
    axis: Sequence[float] | np.ndarray = Z,
) -> SDF3:
    """Return a centered flat-capped cone or frustum."""

    height = float(height)
    if height <= 0.0:
        raise ValueError("height must be positive.")
    if radius_top is None and diameter_top is None:
        radius_top = 0.0
    center_array = _vector(center, name="center")
    axis_array = _normalize_axis(axis)
    half_axis = 0.5 * height * axis_array
    return capped_cone(
        center_array - half_axis,
        center_array + half_axis,
        radius_a=radius_bottom,
        radius_b=radius_top,
        diameter_a=diameter_bottom,
        diameter_b=diameter_top,
    )


def rounded_cone(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    radius_a: float | None = None,
    radius_b: float | None = None,
    diameter_a: float | None = None,
    diameter_b: float | None = None,
) -> SDF3:
    """Return a cone-like swept shape with spherical end caps."""

    start, end, axis_array, length = _axis_frame(a, b)
    ra = _radius_from_inputs(radius=radius_a, diameter=diameter_a, name="radius_a")
    rb = _radius_from_inputs(radius=radius_b, diameter=diameter_b, name="radius_b")
    if abs(ra - rb) >= length:
        raise ValueError("Endpoint distance must exceed the absolute radius difference.")
    slope = (ra - rb) / length
    tangent = float(np.sqrt(max(0.0, 1.0 - slope * slope)))

    def evaluate(points: np.ndarray) -> np.ndarray:
        relative = points - start
        axial = relative @ axis_array
        radial_vectors = relative - axial[:, np.newaxis] * axis_array
        radial = np.linalg.norm(radial_vectors, axis=1)
        q = np.stack((radial, axial), axis=-1)
        k = q @ np.asarray((-slope, tangent), dtype=float)
        c1 = np.linalg.norm(q, axis=1) - ra
        c2 = np.linalg.norm(q - np.asarray((0.0, length), dtype=float), axis=1) - rb
        c3 = q @ np.asarray((tangent, slope), dtype=float) - ra
        return np.where(k < 0.0, c1, np.where(k > tangent * length, c2, c3))

    return SDF3(evaluate, name="rounded_cone")


def capsule_chain(
    points: Sequence[Sequence[float] | np.ndarray],
    *,
    radius: float | None = None,
    diameter: float | None = None,
    radii: Sequence[float] | np.ndarray | None = None,
    union_radius: float = 0.0,
    chamfer: float = 0.0,
) -> SDF3:
    """Return a union of capsule or rounded-cone segments through a point chain."""

    point_array = np.asarray([_vector(point, name="point") for point in points], dtype=float)
    if point_array.ndim != 2 or point_array.shape[1] != 3 or point_array.shape[0] < 2:
        raise ValueError("capsule_chain requires at least two 3D points.")

    if radii is None:
        resolved_radius = _radius_from_inputs(radius=radius, diameter=diameter, name="radius")
        parts = [
            capsule(start, end, radius=resolved_radius)
            for start, end in zip(point_array[:-1], point_array[1:], strict=True)
        ]
    else:
        if radius is not None or diameter is not None:
            raise ValueError("Specify either radii=... or radius/diameter, not both.")
        radius_array = np.asarray(radii, dtype=float)
        if radius_array.shape != (point_array.shape[0],):
            raise ValueError("radii must contain one positive radius per chain point.")
        if np.any(radius_array <= 0.0):
            raise ValueError("radii entries must be positive.")
        parts = []
        for start, end, start_radius, end_radius in zip(
            point_array[:-1],
            point_array[1:],
            radius_array[:-1],
            radius_array[1:], strict=True,
        ):
            if np.isclose(start_radius, end_radius):
                parts.append(capsule(start, end, radius=float(start_radius)))
            else:
                parts.append(rounded_cone(start, end, radius_a=float(start_radius), radius_b=float(end_radius)))

    return parts[0] if len(parts) == 1 else union(*parts, radius=union_radius, chamfer=chamfer)


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
