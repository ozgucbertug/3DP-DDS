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

    vertices: npt.NDArray[np.float64]
    faces: npt.NDArray[np.int64]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    vertex_colors: npt.NDArray[np.uint8] | None = None
    face_colors: npt.NDArray[np.uint8] | None = None

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
        if self.vertex_colors is not None and self.face_colors is not None:
            raise ValueError(
                "TriangleMesh accepts vertex_colors or face_colors, not both."
            )
        vertex_colors = _validate_colors(
            self.vertex_colors,
            count=len(vertices),
            name="TriangleMesh.vertex_colors",
        )
        face_colors = _validate_colors(
            self.face_colors,
            count=len(faces),
            name="TriangleMesh.face_colors",
        )
        vertices.setflags(write=False)
        faces.setflags(write=False)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "vertex_colors", vertex_colors)
        object.__setattr__(self, "face_colors", face_colors)

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
        return trimesh.Trimesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            vertex_colors=(
                None if self.vertex_colors is None else self.vertex_colors.copy()
            ),
            face_colors=None if self.face_colors is None else self.face_colors.copy(),
            metadata=dict(self.metadata),
            process=False,
        )

    @classmethod
    def from_trimesh(
        cls,
        mesh: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "TriangleMesh":
        """Build a TriangleMesh from a trimesh.Trimesh object."""

        trimesh = _load_trimesh()
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("mesh must be a trimesh.Trimesh")
        resolved_metadata = dict(mesh.metadata)
        if metadata is not None:
            resolved_metadata.update(metadata)
        vertex_colors = None
        face_colors = None
        if mesh.visual.defined and mesh.visual.kind == "vertex":
            vertex_colors = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)
        elif mesh.visual.defined and mesh.visual.kind == "face":
            face_colors = np.asarray(mesh.visual.face_colors, dtype=np.uint8)
        return cls(
            vertices=np.asarray(mesh.vertices, dtype=float),
            faces=np.asarray(mesh.faces, dtype=np.int64),
            metadata=resolved_metadata,
            vertex_colors=vertex_colors,
            face_colors=face_colors,
        )


def _validate_colors(
    values: npt.ArrayLike | None,
    *,
    count: int,
    name: str,
) -> npt.NDArray[np.uint8] | None:
    if values is None:
        return None
    colors = np.asarray(values)
    if (
        colors.ndim != 2
        or colors.shape[0] != count
        or colors.shape[1] not in {3, 4}
    ):
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


def implicit_field_to_mesh(
    domain: Domain,
    implicit_field: npt.ArrayLike,
    *,
    threshold: float = 0.5,
    step_size: int = 1,
) -> TriangleMesh:
    """Extract a mesh from an implicit field using an isovalue threshold."""

    return extract_mesh_from_field(
        domain,
        implicit_field,
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
