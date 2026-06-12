"""Point-cloud geometry backed by immutable NumPy arrays."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from ._utils import load_trimesh, validate_colors


@dataclass(frozen=True, slots=True)
class PointCloud:
    """A collection of 3D points with optional per-point RGB or RGBA colors."""

    points: npt.NDArray[np.float64]
    colors: npt.NDArray[np.uint8] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

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
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def empty(cls, *, metadata: Mapping[str, Any] | None = None) -> PointCloud:
        """Return an empty point cloud."""

        return cls(
            points=np.empty((0, 3), dtype=np.float64),
            metadata=metadata or {},
        )

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
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_trimesh(
        cls,
        cloud: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> PointCloud:
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
        resolved_metadata = dict(cloud.metadata)
        if metadata is not None:
            resolved_metadata.update(metadata)
        return cls(
            points=np.asarray(cloud.vertices, dtype=np.float64),
            colors=resolved_colors,
            metadata=resolved_metadata,
        )


def read_point_cloud(path: str | Path) -> PointCloud:
    """Read a point cloud from disk using trimesh."""

    trimesh = load_trimesh()
    source = Path(path)
    loaded = trimesh.load(source)
    if not isinstance(loaded, trimesh.points.PointCloud):
        raise ValueError(f"File {source} does not contain a point cloud.")
    return PointCloud.from_trimesh(loaded, metadata={"path": str(source)})


def write_point_cloud(path: str | Path, cloud: PointCloud) -> Path:
    """Write a point cloud to disk using trimesh."""

    if not isinstance(cloud, PointCloud):
        raise TypeError("cloud must be a PointCloud")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cloud.to_trimesh().export(target)
    return target
