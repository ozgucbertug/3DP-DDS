"""Triangle-mesh containers, I/O, and dense-field conversions."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

from ..domain import Domain
from ._utils import load_trimesh, validate_colors, validate_field_shape


def _load_skimage_measure() -> Any:
    try:
        return import_module("skimage.measure")
    except ImportError as exc:
        raise ImportError(
            'scikit-image is required for mesh extraction. Install it with `pip install -e ".[mesh]"`.',
        ) from exc


@dataclass(frozen=True)
class TriangleMesh:
    """A simple triangle mesh container."""

    vertices: npt.NDArray[np.float64]
    faces: npt.NDArray[np.int64]
    vertex_colors: Optional[npt.NDArray[np.uint8]] = None
    face_colors: Optional[npt.NDArray[np.uint8]] = None

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
            raise ValueError("TriangleMesh accepts vertex_colors or face_colors, not both.")
        vertex_colors = validate_colors(
            self.vertex_colors,
            count=len(vertices),
            name="TriangleMesh.vertex_colors",
        )
        face_colors = validate_colors(
            self.face_colors,
            count=len(faces),
            name="TriangleMesh.face_colors",
        )
        vertices.setflags(write=False)
        faces.setflags(write=False)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "vertex_colors", vertex_colors)
        object.__setattr__(self, "face_colors", face_colors)

    @classmethod
    def empty(cls) -> "TriangleMesh":
        """Return an empty triangle mesh."""

        return cls(
            vertices=np.empty((0, 3), dtype=float),
            faces=np.empty((0, 3), dtype=np.int64),
        )

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

        trimesh = load_trimesh()
        return trimesh.Trimesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            vertex_colors=(None if self.vertex_colors is None else self.vertex_colors.copy()),
            face_colors=None if self.face_colors is None else self.face_colors.copy(),
            process=False,
        )

    @classmethod
    def from_trimesh(cls, mesh: Any) -> "TriangleMesh":
        """Build a TriangleMesh from a trimesh.Trimesh object."""

        trimesh = load_trimesh()
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("mesh must be a trimesh.Trimesh")
        vertex_colors = None
        face_colors = None
        if mesh.visual.defined and mesh.visual.kind == "vertex":
            vertex_colors = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)
        elif mesh.visual.defined and mesh.visual.kind == "face":
            face_colors = np.asarray(mesh.visual.face_colors, dtype=np.uint8)
        return cls(
            vertices=np.asarray(mesh.vertices, dtype=float),
            faces=np.asarray(mesh.faces, dtype=np.int64),
            vertex_colors=vertex_colors,
            face_colors=face_colors,
        )


def _ensure_watertight(mesh: Any, *, require_watertight: bool, context: str) -> None:
    if mesh.is_watertight:
        return
    message = (
        f"Mesh is not watertight; {context} is unreliable for open surfaces. "
        "Provide a watertight mesh or pass require_watertight=False to continue anyway."
    )
    if require_watertight:
        raise ValueError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def read_mesh(path: Union[str, Path]) -> TriangleMesh:
    """Read a triangle mesh from disk using trimesh."""

    trimesh = load_trimesh()
    source = Path(path)
    loaded = trimesh.load(source, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = loaded.dump()
        if not geometries or not all(isinstance(geometry, trimesh.Trimesh) for geometry in geometries):
            raise ValueError(f"File {source} does not contain only triangle meshes.")
        loaded = loaded.to_mesh()
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"File {source} does not contain a triangle mesh.")
    return TriangleMesh.from_trimesh(loaded)


def write_mesh(path: Union[str, Path], mesh: TriangleMesh) -> Path:
    """Write a triangle mesh to disk using trimesh."""

    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mesh.to_trimesh().export(target)
    return target


def mesh_to_occupancy(
    domain: Domain,
    mesh: TriangleMesh,
    *,
    require_watertight: bool = True,
) -> npt.NDArray[np.bool_]:
    """Sample mesh occupancy at voxel centers using trimesh containment tests."""

    tri_mesh = mesh.to_trimesh()
    _ensure_watertight(
        tri_mesh,
        require_watertight=require_watertight,
        context="occupancy queries",
    )
    xs, ys, zs = domain.grid_centers()
    points = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)
    contained = np.asarray(tri_mesh.contains(points), dtype=bool)
    return contained.reshape(domain.grid_shape)


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

    field = validate_field_shape(domain, values, field_name="values")
    if field.size == 0:
        return TriangleMesh.empty()
    if float(np.min(field)) > level or float(np.max(field)) < level:
        return TriangleMesh.empty()

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
        return TriangleMesh.empty()

    origin = np.asarray(domain.min_corner, dtype=float) + 0.5 * np.asarray(domain.voxel_size, dtype=float)
    vertices = np.asarray(vertices, dtype=float) + origin
    return TriangleMesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
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
