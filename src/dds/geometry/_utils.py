"""Shared implementation details for geometry containers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
import numpy.typing as npt

from ..domain import Domain


def load_trimesh() -> Any:
    """Import the optional trimesh dependency on first use."""

    try:
        return import_module("trimesh")
    except ImportError as exc:
        raise ImportError("trimesh is required. Install it with `pip install trimesh`.") from exc


def validate_colors(
    values: npt.ArrayLike | None,
    *,
    count: int,
    name: str,
) -> npt.NDArray[np.uint8] | None:
    """Return an immutable RGB or RGBA color array."""

    if values is None:
        return None
    colors = np.asarray(values)
    if colors.ndim != 2 or colors.shape[0] != count or colors.shape[1] not in {3, 4}:
        raise ValueError(f"{name} must have shape `(n, 3)` or `(n, 4)`.")
    if not np.issubdtype(colors.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values.")
    if (
        not np.all(np.isfinite(colors))
        or np.any(colors < 0)
        or np.any(colors > 255)
        or not np.all(colors == np.floor(colors))
    ):
        raise ValueError(f"{name} must contain integer values from 0 to 255.")
    result = np.array(colors, dtype=np.uint8, copy=True)
    result.setflags(write=False)
    return result


def validate_field_shape(
    domain: Domain,
    values: npt.ArrayLike,
    *,
    field_name: str,
) -> npt.NDArray[np.float64]:
    """Coerce a dense field and validate it against a domain grid."""

    array = np.asarray(values, dtype=float)
    if array.shape != domain.grid_shape:
        raise ValueError(f"{field_name} shape {array.shape} does not match domain grid shape {domain.grid_shape}.")
    return array
