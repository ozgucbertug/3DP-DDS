"""Triangle-mesh containers and field-to-mesh extraction helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from ..domain import Domain


def _load_skimage_measure() -> Any:
    try:
        return import_module("skimage.measure")
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for mesh extraction. "
            'Install it with `pip install -e ".[mesh]"`.',
        ) from exc


def _load_trimesh() -> Any:
    try:
        return import_module("trimesh")
    except ImportError as exc:
        raise ImportError("trimesh is required. Install it with `pip install trimesh`.") from exc


def _validate_field_shape(
    domain: Domain,
    values: npt.ArrayLike,
    *,
    field_name: str,
) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.shape != domain.grid_shape:
        raise ValueError(f"{field_name} shape {array.shape} does not match domain grid shape {domain.grid_shape}.")
    return array


@dataclass(frozen=True, slots=True)
class TriangleMesh:
    """A simple triangle mesh container."""

    vertices: npt.ArrayLike
    faces: npt.ArrayLike
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        vertices = np.array(self.vertices, dtype=float, copy=True)
        faces = np.array(self.faces, dtype=np.int64, copy=True)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("TriangleMesh.vertices must have shape `(n, 3)`.")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("TriangleMesh.faces must have shape `(m, 3)`.")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("TriangleMesh.vertices must contain finite values.")
        if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("TriangleMesh.faces contain invalid vertex indices.")
        vertices.setflags(write=False)
        faces.setflags(write=False)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def empty(cls, *, metadata: dict[str, Any] | None = None) -> "TriangleMesh":
        """Return an empty triangle mesh."""

        return cls(vertices=np.empty((0, 3), dtype=float), faces=np.empty((0, 3), dtype=np.int64), metadata=metadata or {})

    @property
    def is_empty(self) -> bool:
        """Return True when the mesh has no triangles."""

        return self.faces.size == 0 or self.vertices.size == 0

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def bounds(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return axis-aligned bounds."""

        if self.is_empty:
            raise ValueError("Empty meshes do not have bounds.")
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def to_trimesh(self) -> Any:
        """Convert to a trimesh.Trimesh instance."""

        trimesh = _load_trimesh()
        return trimesh.Trimesh(vertices=self.vertices.copy(), faces=self.faces.copy(), process=False)

    @classmethod
    def from_trimesh(cls, mesh: Any, *, metadata: dict[str, Any] | None = None) -> "TriangleMesh":
        """Build a TriangleMesh from a trimesh.Trimesh object."""

        return cls(vertices=np.asarray(mesh.vertices, dtype=float), faces=np.asarray(mesh.faces, dtype=np.int64), metadata=metadata or {})


def extract_mesh_from_field(
    domain: Domain,
    values: npt.ArrayLike,
    *,
    level: float = 0.0,
    gradient_direction: str = "ascent",
    step_size: int = 1,
) -> TriangleMesh:
    """Extract a triangle mesh from a dense scalar field sampled on voxel centers."""

    if gradient_direction not in {"ascent", "descent"}:
        raise ValueError("gradient_direction must be 'ascent' or 'descent'.")
    if step_size < 1:
        raise ValueError("step_size must be at least 1.")

    field = _validate_field_shape(domain, values, field_name="values")
    if field.size == 0:
        return TriangleMesh.empty(metadata={"level": level, "gradient_direction": gradient_direction})
    if float(np.min(field)) > level or float(np.max(field)) < level:
        return TriangleMesh.empty(metadata={"level": level, "gradient_direction": gradient_direction})

    measure = _load_skimage_measure()
    try:
        vertices, faces, _normals, _samples = measure.marching_cubes(
            field.astype(np.float32, copy=False),
            level=level,
            spacing=domain.voxel_size,
            gradient_direction=gradient_direction,
            step_size=step_size,
            allow_degenerate=False,
        )
    except ValueError:
        return TriangleMesh.empty(metadata={"level": level, "gradient_direction": gradient_direction})

    origin = np.asarray(domain.min_corner, dtype=float) + 0.5 * np.asarray(domain.voxel_size, dtype=float)
    vertices = np.asarray(vertices, dtype=float) + origin
    return TriangleMesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        metadata={"level": level, "gradient_direction": gradient_direction, "step_size": step_size},
    )


def occupancy_to_mesh(
    domain: Domain,
    occupancy: npt.ArrayLike,
    *,
    level: float = 0.5,
    step_size: int = 1,
) -> TriangleMesh:
    """Extract a mesh from a dense occupancy field."""

    return extract_mesh_from_field(
        domain,
        np.asarray(occupancy, dtype=float),
        level=level,
        gradient_direction="descent",
        step_size=step_size,
    )


def density_to_mesh(
    domain: Domain,
    density: npt.ArrayLike,
    *,
    threshold: float = 0.5,
    step_size: int = 1,
) -> TriangleMesh:
    """Extract a mesh from a density-like field using an isovalue threshold."""

    return extract_mesh_from_field(
        domain,
        density,
        level=threshold,
        gradient_direction="descent",
        step_size=step_size,
    )


def sdf_to_mesh(
    domain: Domain,
    sdf_values: npt.ArrayLike,
    *,
    level: float = 0.0,
    step_size: int = 1,
) -> TriangleMesh:
    """Extract a mesh from a signed-distance field."""

    return extract_mesh_from_field(
        domain,
        sdf_values,
        level=level,
        gradient_direction="ascent",
        step_size=step_size,
    )
