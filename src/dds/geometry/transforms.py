"""Geometric transforms for SDF3 objects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np

from ..utils import ensure_finite_triplet, ensure_positive_triplet, normalize_axis
from .sdf import SDF3, SDFCallable, as_sdf3


def _positive_scale(factor: Union[float, Sequence[float]]) -> np.ndarray:
    if isinstance(factor, (int, float)):
        value = float(factor)
        if value <= 0.0:
            raise ValueError("scale factor must be positive.")
        return np.asarray((value, value, value), dtype=float)
    return np.asarray(ensure_positive_triplet(factor, "scale factor"), dtype=float)


def translate(other: Union[SDF3, SDFCallable], offset: Union[Sequence[float], np.ndarray]) -> SDF3:
    """Translate an SDF in world coordinates."""

    sdf = as_sdf3(other)
    vector = np.asarray(ensure_finite_triplet(offset, "offset"), dtype=float)
    return SDF3(lambda points: sdf._evaluate(points - vector), name="translate")


def scale(other: Union[SDF3, SDFCallable], factor: Union[float, Sequence[float]]) -> SDF3:
    """Scale an SDF by a scalar or per-axis factor."""

    sdf = as_sdf3(other)
    scale_vector = _positive_scale(factor)
    metric_scale = float(np.min(scale_vector))
    return SDF3(lambda points: sdf._evaluate(points / scale_vector) * metric_scale, name="scale")


def rotation_matrix(angle: float, axis: Union[Sequence[float], np.ndarray] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Return a Rodrigues rotation matrix for a column-vector rotation."""

    x, y, z = normalize_axis(axis, name="axis")
    angle = float(angle)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    outer = np.outer((x, y, z), (x, y, z))
    skew = np.asarray(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )
    return cosine * np.eye(3) + (1.0 - cosine) * outer + sine * skew


def rotate(other: Union[SDF3, SDFCallable], angle: float, axis: Union[Sequence[float], np.ndarray] = (0.0, 0.0, 1.0)) -> SDF3:
    """Rotate an SDF around an axis through the origin."""

    sdf = as_sdf3(other)
    matrix = rotation_matrix(angle, axis)
    return SDF3(lambda points: sdf._evaluate(points @ matrix), name="rotate")


def _perpendicular(vector: np.ndarray) -> np.ndarray:
    if np.allclose(vector, (0.0, 0.0, 0.0)):
        raise ValueError("Cannot determine a perpendicular for the zero vector.")
    if np.allclose(vector[1:], (0.0, 0.0)):
        return np.cross(vector, np.array((0.0, 1.0, 0.0), dtype=float))
    return np.cross(vector, np.array((1.0, 0.0, 0.0), dtype=float))


def orient(
    other: Union[SDF3, SDFCallable],
    axis: Union[Sequence[float], np.ndarray],
    *,
    source_axis: Union[Sequence[float], np.ndarray] = (0.0, 0.0, 1.0),
) -> SDF3:
    """Rotate an SDF so that `source_axis` aligns with `axis`."""

    sdf = as_sdf3(other)
    source = np.asarray(normalize_axis(source_axis, name="source_axis"), dtype=float)
    target = np.asarray(normalize_axis(axis, name="axis"), dtype=float)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if np.isclose(dot, 1.0):
        return sdf
    if np.isclose(dot, -1.0):
        return rotate(sdf, np.pi, _perpendicular(source))
    rotation_axis = np.cross(source, target)
    angle = float(np.arccos(dot))
    return rotate(sdf, angle, rotation_axis)
