"""Point-cloud geometry backed by immutable NumPy arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

from ._utils import load_trimesh, validate_colors


@dataclass(frozen=True)
class PointCloud:
    """A collection of 3D points with optional per-point RGB or RGBA colors."""

    points: npt.NDArray[np.float64]
    colors: Optional[npt.NDArray[np.uint8]] = None

    def __post_init__(self) -> None:
        points = np.array(self.points, dtype=np.float64, copy=True)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("PointCloud.points must have shape `(n, 3)`.")
        if not np.all(np.isfinite(points)):
            raise ValueError("PointCloud.points must contain finite values.")

        colors = validate_colors(
            self.colors,
            count=len(points),
            name="PointCloud.colors",
        )

        points.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "colors", colors)

    @classmethod
    def empty(cls) -> PointCloud:
        """Return an empty point cloud."""

        return cls(points=np.empty((0, 3), dtype=np.float64))

    @property
    def is_empty(self) -> bool:
        return self.points.shape[0] == 0

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def has_colors(self) -> bool:
        return self.colors is not None

    @property
    def bounds(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if self.is_empty:
            raise ValueError("Empty point clouds do not have bounds.")
        return self.points.min(axis=0), self.points.max(axis=0)

    def to_trimesh(self) -> Any:
        """Convert to a trimesh.points.PointCloud instance."""

        trimesh = load_trimesh()
        return trimesh.points.PointCloud(
            vertices=self.points.copy(),
            colors=None if self.colors is None else self.colors.copy(),
        )

    @classmethod
    def from_trimesh(cls, cloud: Any) -> PointCloud:
        """Build a PointCloud from a trimesh.points.PointCloud instance."""

        trimesh = load_trimesh()
        if not isinstance(cloud, trimesh.points.PointCloud):
            raise TypeError("cloud must be a trimesh.points.PointCloud")
        colors = np.asarray(cloud.colors)
        resolved_colors = (
            colors
            if colors.ndim == 2 and colors.shape[0] == len(cloud.vertices) and colors.shape[1] in {3, 4}
            else None
        )
        return cls(
            points=np.asarray(cloud.vertices, dtype=np.float64),
            colors=resolved_colors,
        )


def read_point_cloud(path: Union[str, Path]) -> PointCloud:
    """Read a point cloud from disk using trimesh."""

    trimesh = load_trimesh()
    source = Path(path)
    loaded = trimesh.load(source)
    if not isinstance(loaded, trimesh.points.PointCloud):
        raise ValueError(f"File {source} does not contain a point cloud.")
    return PointCloud.from_trimesh(loaded)


def write_point_cloud(path: Union[str, Path], cloud: PointCloud) -> Path:
    """Write a point cloud to disk using trimesh."""

    if not isinstance(cloud, PointCloud):
        raise TypeError("cloud must be a PointCloud")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cloud.to_trimesh().export(target)
    return target
