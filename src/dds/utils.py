"""Numerical helper utilities used across the package."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

EPSILON = 1e-12


def ensure_finite_triplet(values: Any, name: str) -> tuple[float, float, float]:
    """Convert an input to a finite 3-tuple of floats."""

    if isinstance(values, np.ndarray):
        array = values.astype(float, copy=False).reshape(-1)
        if array.size != 3:
            raise ValueError(f"{name} must contain exactly three values.")
        result = tuple(float(item) for item in array)
    elif all(hasattr(values, axis) for axis in ("x", "y", "z")):
        result = (float(values.x), float(values.y), float(values.z))
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        if len(values) != 3:
            raise ValueError(f"{name} must contain exactly three values.")
        result = tuple(float(item) for item in values)
    else:
        raise TypeError(f"{name} must be a sequence of three numeric values.")

    if not all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def ensure_positive_triplet(values: Any, name: str) -> tuple[float, float, float]:
    """Convert an input to a positive 3-tuple of floats."""

    triplet = ensure_finite_triplet(values, name)
    if any(value <= 0.0 for value in triplet):
        raise ValueError(f"{name} values must all be positive.")
    return triplet


def bounding_box_from_points(
    points: Iterable[Sequence[float]],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return min/max bounds for a sequence of 3D points."""

    array = np.asarray(list(points), dtype=float)
    if array.size == 0:
        raise ValueError("At least one point is required to build a bounding box.")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Each point must have exactly three coordinates.")
    if not np.all(np.isfinite(array)):
        raise ValueError("All point coordinates must be finite.")
    return array.min(axis=0), array.max(axis=0)


def expand_aabb(
    minimum: Sequence[float],
    maximum: Sequence[float],
    padding: Sequence[float],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Expand an axis-aligned bounding box by the given padding."""

    minimum_array = np.asarray(ensure_finite_triplet(minimum, "minimum"), dtype=float)
    maximum_array = np.asarray(ensure_finite_triplet(maximum, "maximum"), dtype=float)
    padding_array = np.asarray(ensure_positive_triplet(padding, "padding"), dtype=float)
    return minimum_array - padding_array, maximum_array + padding_array


def closest_point_parameters(
    points: npt.NDArray[np.float64],
    start: npt.NDArray[np.float64],
    end: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return clamped closest-point parameters for a segment."""

    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= EPSILON:
        return np.zeros(points.shape[:-1], dtype=float)
    projection = np.sum((points - start) * segment, axis=-1) / denominator
    return np.clip(projection, 0.0, 1.0)


def normalize_axis(value: Any, name: str) -> tuple[float, float, float]:
    """Normalize a 3-vector to unit length and return it as a float 3-tuple."""

    array = np.asarray(ensure_finite_triplet(value, name), dtype=float)
    norm = float(np.linalg.norm(array))
    if norm <= EPSILON:
        raise ValueError(f"{name} must not be the zero vector.")
    return tuple(float(component) for component in array / norm)  # type: ignore[return-value]


def point_to_segment_distances(
    points: npt.NDArray[np.float64],
    start: npt.NDArray[np.float64],
    end: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute Euclidean distances from points to a segment."""

    t = closest_point_parameters(points, start, end)[..., np.newaxis]
    closest_points = start + t * (end - start)
    return np.linalg.norm(points - closest_points, axis=-1)
