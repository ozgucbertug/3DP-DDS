"""Boolean and morphological operations for SDF3 objects."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .sdf import SDF3, SDFCallable, as_sdf3

SQRT_HALF = np.sqrt(0.5)


def minimum(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], radius: float = 0.0) -> npt.NDArray[np.float64]:
    """Smooth or hard minimum adapted from sdfCAD's `dn.py`."""

    if radius > 0.0:
        delta = b - a
        blend = np.clip(0.5 + 0.5 * delta / radius, 0.0, 1.0)
        return b - delta * blend - radius * blend * (1.0 - blend)
    return np.minimum(a, b)


def maximum(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], radius: float = 0.0) -> npt.NDArray[np.float64]:
    """Smooth or hard maximum adapted from sdfCAD's `dn.py`."""

    if radius > 0.0:
        delta = b - a
        blend = np.clip(0.5 - 0.5 * delta / radius, 0.0, 1.0)
        return b - delta * blend + radius * blend * (1.0 - blend)
    return np.maximum(a, b)


def union(*sdfs: SDF3 | SDFCallable, radius: float = 0.0, chamfer: float = 0.0) -> SDF3:
    """Return the hard or smooth union of multiple SDFs."""

    wrapped = [as_sdf3(sdf) for sdf in sdfs]
    if not wrapped:
        raise ValueError("union requires at least one SDF.")
    radius = max(0.0, float(radius))
    chamfer = max(0.0, float(chamfer))

    def evaluate(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        distance = wrapped[0]._evaluate(points)
        for sdf in wrapped[1:]:
            candidate = sdf._evaluate(points)
            parts = (distance, candidate)
            if chamfer > 0.0:
                parts = (minimum(distance, candidate), (distance + candidate - chamfer) * SQRT_HALF)
            distance = minimum(parts[0], parts[1], radius)
        return distance

    return SDF3(evaluate, name="union")


def intersection(*sdfs: SDF3 | SDFCallable, radius: float = 0.0, chamfer: float = 0.0) -> SDF3:
    """Return the hard or smooth intersection of multiple SDFs."""

    wrapped = [as_sdf3(sdf) for sdf in sdfs]
    if not wrapped:
        raise ValueError("intersection requires at least one SDF.")
    radius = max(0.0, float(radius))
    chamfer = max(0.0, float(chamfer))

    def evaluate(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        distance = wrapped[0]._evaluate(points)
        for sdf in wrapped[1:]:
            candidate = sdf._evaluate(points)
            parts = (distance, candidate)
            if chamfer > 0.0:
                parts = (maximum(distance, candidate), (distance + candidate + chamfer) * SQRT_HALF)
            distance = maximum(parts[0], parts[1], radius)
        return distance

    return SDF3(evaluate, name="intersection")


def difference(
    a: SDF3 | SDFCallable,
    *bs: SDF3 | SDFCallable,
    radius: float = 0.0,
    chamfer: float = 0.0,
) -> SDF3:
    """Return the hard or smooth difference of multiple SDFs."""

    base = as_sdf3(a)
    wrapped = [as_sdf3(sdf) for sdf in bs]
    radius = max(0.0, float(radius))
    chamfer = max(0.0, float(chamfer))
    if not wrapped:
        return base

    def evaluate(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        distance = base._evaluate(points)
        for sdf in wrapped:
            candidate = sdf._evaluate(points)
            parts = (distance, -candidate)
            if chamfer > 0.0:
                parts = (maximum(distance, -candidate), (distance - candidate + chamfer) * SQRT_HALF)
            distance = maximum(parts[0], parts[1], radius)
        return distance

    return SDF3(evaluate, name="difference")


def dilate(other: SDF3 | SDFCallable, radius: float) -> SDF3:
    """Expand an SDF by subtracting a scalar radius."""

    sdf = as_sdf3(other)
    radius = float(radius)
    if radius < 0.0:
        raise ValueError("dilate radius must be non-negative.")
    return SDF3(lambda points: sdf._evaluate(points) - radius, name="dilate")


def erode(other: SDF3 | SDFCallable, radius: float) -> SDF3:
    """Shrink an SDF by adding a scalar radius."""

    sdf = as_sdf3(other)
    radius = float(radius)
    if radius < 0.0:
        raise ValueError("erode radius must be non-negative.")
    return SDF3(lambda points: sdf._evaluate(points) + radius, name="erode")


def shell(other: SDF3 | SDFCallable, thickness: float = 1.0, *, type: str = "center") -> SDF3:
    """Keep only a shell around the SDF boundary."""

    sdf = as_sdf3(other)
    thickness = float(thickness)
    if thickness <= 0.0:
        raise ValueError("shell thickness must be positive.")
    if type == "center":
        return SDF3(lambda points: np.abs(sdf._evaluate(points)) - thickness / 2.0, name="shell_center")
    if type == "inner":
        return difference(sdf, erode(sdf, thickness))
    if type == "outer":
        return difference(dilate(sdf, thickness), sdf)
    raise ValueError("shell type must be 'center', 'inner', or 'outer'.")
