"""SDF-oriented geometry helpers for dds."""

from .ops import difference, dilate, erode, intersection, shell, union
from .primitives import ORIGIN, X, Y, Z, box, capsule, cylinder, ellipsoid, plane, slab, sphere, torus
from .sdf import GridSDF3, SDF3, as_sdf3
from .transforms import orient, rotate, rotation_matrix, scale, translate

__all__ = [
    "GridSDF3",
    "ORIGIN",
    "SDF3",
    "X",
    "Y",
    "Z",
    "as_sdf3",
    "box",
    "capsule",
    "cylinder",
    "difference",
    "dilate",
    "ellipsoid",
    "erode",
    "intersection",
    "orient",
    "plane",
    "rotate",
    "rotation_matrix",
    "scale",
    "shell",
    "slab",
    "sphere",
    "torus",
    "translate",
    "union",
]
