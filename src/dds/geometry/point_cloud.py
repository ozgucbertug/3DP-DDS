"""Point-cloud geometry backed by immutable NumPy arrays."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from .mesh import _load_trimesh


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

        colors: npt.NDArray[np.uint8] | None = None
        if self.colors is not None:
            raw_colors = np.asarray(self.colors)
            if (
                raw_colors.ndim != 2
                or raw_colors.shape[0] != points.shape[0]
                or raw_colors.shape[1] not in {3, 4}
            ):
                raise ValueError(
                    "PointCloud.colors must have shape `(n, 3)` or `(n, 4)`."
                )
            if not np.issubdtype(raw_colors.dtype, np.number):
                raise TypeError("PointCloud.colors must contain numeric values.")
            if (
                not np.all(np.isfinite(raw_colors))
                or np.any(raw_colors < 0)
                or np.any(raw_colors > 255)
                or not np.all(raw_colors == np.floor(raw_colors))
            ):
                raise ValueError(
                    "PointCloud.colors must contain integer values from 0 to 255."
                )
            colors = np.array(raw_colors, dtype=np.uint8, copy=True)
            colors.setflags(write=False)

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

        trimesh = _load_trimesh()
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

        trimesh = _load_trimesh()
        if not isinstance(cloud, trimesh.points.PointCloud):
            raise TypeError("cloud must be a trimesh.points.PointCloud")
        colors = np.asarray(cloud.colors)
        resolved_colors = (
            colors
            if colors.ndim == 2
            and colors.shape[0] == len(cloud.vertices)
            and colors.shape[1] in {3, 4}
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
