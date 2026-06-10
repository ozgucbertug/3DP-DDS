"""Reusable headless query and analysis helpers for dense simulation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from ..domain import Domain
from ..geometry.sdf import _coerce_points
from ..occupancy import occupancy_from_density
from ..utils import EPSILON, ensure_finite_triplet, normalize_axis, readonly_array
from .fields import normalize_field

InterpolationMode = Literal["nearest", "trilinear"]
RepresentationMode = Literal["occupancy", "density", "sdf", "mesh"]
SampleFieldName = Literal["density", "occupancy", "deposition_index", "signed_distance"]



def _surface_cache_key(
    threshold: float,
    *,
    normalize: bool,
    step_size: int = 1,
) -> tuple[float, bool, int]:
    return (float(threshold), bool(normalize), int(step_size))


def _sample_nearest(
    domain: Domain,
    values: npt.NDArray[np.float64],
    points: npt.NDArray[np.float64],
    *,
    fill_value: float,
) -> npt.NDArray[np.float64]:
    result = np.full(points.shape[0], fill_value, dtype=float)
    origin = np.asarray(domain.min_corner, dtype=float)
    spacing = np.asarray(domain.voxel_size, dtype=float)
    shape_array = np.asarray(domain.grid_shape, dtype=np.intp)
    raw = np.floor((points - origin) / spacing).astype(np.intp)
    inside = np.all((raw >= 0) & (raw < shape_array), axis=1)
    if inside.any():
        clipped = np.clip(raw[inside], 0, shape_array - 1)
        result[inside] = values[clipped[:, 0], clipped[:, 1], clipped[:, 2]]
    return result


def _sample_trilinear(
    domain: Domain,
    values: npt.NDArray[np.float64],
    points: npt.NDArray[np.float64],
    *,
    fill_value: float,
) -> npt.NDArray[np.float64]:
    from scipy.ndimage import map_coordinates

    min_corner = np.asarray(domain.min_corner, dtype=float)
    spacing = np.asarray(domain.voxel_size, dtype=float)
    first_center = min_corner + 0.5 * spacing

    # Compute fractional voxel coordinates for all points at once.
    fractional = (points - first_center) / spacing  # (n, 3)

    # Points outside the domain get fill_value via mode='constant'.
    # map_coordinates expects coordinates as (ndim, n).
    coords = fractional.T  # (3, n)
    sampled = map_coordinates(values, coords, order=1, mode="constant", cval=fill_value)
    return sampled.astype(float)


def _sample_scalar_field(
    domain: Domain,
    values: npt.NDArray[np.float64],
    points: npt.NDArray[np.float64],
    *,
    interpolation: InterpolationMode,
    fill_value: float = 0.0,
) -> npt.NDArray[np.float64]:
    if interpolation == "nearest":
        return _sample_nearest(domain, values, points, fill_value=fill_value)
    if interpolation == "trilinear":
        return _sample_trilinear(domain, values, points, fill_value=fill_value)
    raise ValueError("interpolation must be 'nearest' or 'trilinear'.")


def _resolve_bundle(source: Any) -> "AnalysisBundle":
    if isinstance(source, AnalysisBundle):
        return source
    if hasattr(source, "analysis_bundle") and callable(source.analysis_bundle):
        return source.analysis_bundle()
    raise TypeError("Expected an AnalysisBundle or an object exposing analysis_bundle().")


@dataclass(slots=True, frozen=True)
class AnalysisBundle:
    """Cached query state derived from a dense density field."""

    domain: Domain
    density: npt.NDArray[np.float64]
    deposition_index: npt.NDArray[np.intp] | None = None
    _normalized_density: npt.NDArray[np.float64] | None = None
    _occupancy_cache: dict[tuple[float, bool], npt.NDArray[np.bool_]] = field(default_factory=dict)
    _surface_mesh_cache: dict[tuple[float, bool, int], Any] = field(default_factory=dict)
    _surface_sdf_cache: dict[tuple[float, bool], Any] = field(default_factory=dict)
    _mesh_sdf_cache: dict[tuple[float, bool, int], Any | None] = field(default_factory=dict)
    _mesh_analysis_cache: dict[tuple[float, bool, int, tuple[float, float, float], float], dict[str, Any]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "density", readonly_array(self.density, dtype=float))
        if self.density.shape != self.domain.grid_shape:
            raise ValueError(
                f"density shape {self.density.shape} does not match domain grid shape {self.domain.grid_shape}."
            )

        if not np.all(np.isfinite(self.density)) or np.any(self.density < 0.0):
            raise ValueError("density must contain only finite, non-negative values.")
        if self.deposition_index is not None:
            deposition_index = readonly_array(self.deposition_index, dtype=np.intp)
            if deposition_index.shape != self.domain.grid_shape:
                raise ValueError(
                    "deposition_index shape "
                    f"{deposition_index.shape} does not match domain grid shape {self.domain.grid_shape}."
                )
            object.__setattr__(self, "deposition_index", deposition_index)

    def density_field(self, *, normalize: bool = False) -> npt.NDArray[np.float64]:
        if not normalize:
            return self.density
        if self._normalized_density is None:
            object.__setattr__(
                self,
                "_normalized_density",
                readonly_array(
                    normalize_field(self.density),
                    dtype=float,
                ),
            )
        return self._normalized_density

    def deposition_index_field(self) -> npt.NDArray[np.intp]:
        """Return the per-voxel last-deposit-index grid (0-based; -1 = untouched).

        Requires the bundle to have been created with a ``deposition_index`` array.
        Raises ``ValueError`` when no index is available.
        """
        if self.deposition_index is None:
            raise ValueError(
                "No deposition index available on this AnalysisBundle. "
                "Create the bundle via Simulator.analysis_bundle() or "
                "SimulationResult.analysis_bundle() to populate it."
            )
        return self.deposition_index

    def occupancy_field(
        self,
        *,
        threshold: float = 0.5,
        normalize: bool = False,
    ) -> npt.NDArray[np.bool_]:
        key = (float(threshold), bool(normalize))
        if key not in self._occupancy_cache:
            self._occupancy_cache[key] = readonly_array(
                occupancy_from_density(
                    self.density_field(normalize=normalize),
                    threshold=threshold,
                ),
                dtype=bool,
            )
        return self._occupancy_cache[key]

    def surface_mesh(
        self,
        *,
        threshold: float = 0.5,
        normalize: bool = False,
        step_size: int = 1,
    ) -> Any:
        key = _surface_cache_key(threshold, normalize=normalize, step_size=step_size)
        if key not in self._surface_mesh_cache:
            from ..geometry import density_to_mesh

            self._surface_mesh_cache[key] = density_to_mesh(
                self.domain,
                self.density_field(normalize=normalize),
                threshold=threshold,
                step_size=step_size,
            )
        return self._surface_mesh_cache[key]

    def surface_sdf(
        self,
        *,
        threshold: float = 0.5,
        normalize: bool = False,
    ) -> Any:
        key = (float(threshold), bool(normalize))
        if key not in self._surface_sdf_cache:
            from ..geometry import density_to_sdf

            self._surface_sdf_cache[key] = density_to_sdf(
                self.domain,
                self.density_field(normalize=normalize),
                threshold=threshold,
            )
        return self._surface_sdf_cache[key]

    def mesh_sdf(
        self,
        *,
        threshold: float = 0.5,
        normalize: bool = False,
        step_size: int = 1,
    ) -> Any | None:
        key = _surface_cache_key(threshold, normalize=normalize, step_size=step_size)
        if key not in self._mesh_sdf_cache:
            from ..geometry import MeshSDF3

            mesh = self.surface_mesh(threshold=threshold, normalize=normalize, step_size=step_size)
            if mesh.is_empty:
                raise ValueError("Cannot construct a mesh SDF from an empty analysis surface.")
            self._mesh_sdf_cache[key] = MeshSDF3(
                mesh,
                require_watertight=True,
                name="analysis_surface_mesh",
            )
        return self._mesh_sdf_cache[key]

    def sample_density_at(
        self,
        point: tuple[float, float, float] | npt.ArrayLike,
        *,
        interpolation: InterpolationMode = "nearest",
        normalize: bool = False,
    ) -> float:
        points, _single = _coerce_points(point)
        values = _sample_scalar_field(
            self.domain,
            self.density_field(normalize=normalize),
            points,
            interpolation=interpolation,
            fill_value=0.0,
        )
        return float(values[0])

    def sample_deposition_index_at(
        self,
        point: tuple[float, float, float] | npt.ArrayLike,
    ) -> int:
        points, _single = _coerce_points(point)
        values = _sample_nearest(
            self.domain,
            self.deposition_index_field(),
            points,
            fill_value=-1,
        )
        return int(values[0])

    def signed_distance_at(
        self,
        point: tuple[float, float, float] | npt.ArrayLike,
        *,
        threshold: float = 0.5,
        normalize: bool = False,
        source: str = "surface_sdf",
        step_size: int = 1,
    ) -> float:
        points, _single = _coerce_points(point)
        if source == "surface_sdf":
            sdf = self.surface_sdf(threshold=threshold, normalize=normalize)
        elif source == "mesh":
            sdf = self.mesh_sdf(threshold=threshold, normalize=normalize, step_size=step_size)
        else:
            raise ValueError("source must be 'surface_sdf' or 'mesh'.")
        return float(sdf(points)[0])

    def surface_normal_at(
        self,
        point: tuple[float, float, float] | npt.ArrayLike,
        *,
        threshold: float = 0.5,
        normalize: bool = False,
        source: str = "surface_sdf",
        step_size: int = 1,
    ) -> tuple[float, float, float]:
        base_point = np.asarray(ensure_finite_triplet(point, "point"), dtype=float)
        sdf = self.surface_sdf(threshold=threshold, normalize=normalize) if source == "surface_sdf" else None
        if source == "mesh":
            sdf = self.mesh_sdf(threshold=threshold, normalize=normalize, step_size=step_size)
        if sdf is None:
            raise ValueError("source must be 'surface_sdf' or 'mesh'.")

        steps = 0.5 * np.asarray(self.domain.voxel_size, dtype=float)
        gradient = np.empty(3, dtype=float)
        for axis in range(3):
            offset = np.zeros(3, dtype=float)
            offset[axis] = steps[axis]
            gradient[axis] = float(sdf(base_point + offset) - sdf(base_point - offset)) / (2.0 * steps[axis])
        norm = float(np.linalg.norm(gradient))
        if norm <= EPSILON:
            return (0.0, 0.0, 0.0)
        return tuple(float(value) for value in (gradient / norm))

    def contains_point(
        self,
        point: tuple[float, float, float] | npt.ArrayLike,
        *,
        representation: RepresentationMode = "occupancy",
        threshold: float = 0.5,
        interpolation: InterpolationMode = "nearest",
        normalize: bool = False,
        step_size: int = 1,
    ) -> bool:
        coordinates = ensure_finite_triplet(point, "point")
        if representation == "occupancy":
            if interpolation == "nearest":
                if not self.domain.contains_point(coordinates):
                    return False
                return bool(
                    self.occupancy_field(threshold=threshold, normalize=normalize)[
                        self.domain.world_to_index(coordinates, clip=True)
                    ]
                )
            return self.sample_density_at(coordinates, interpolation=interpolation, normalize=normalize) >= threshold
        if representation == "density":
            return self.sample_density_at(coordinates, interpolation=interpolation, normalize=normalize) >= threshold
        if representation == "sdf":
            return self.signed_distance_at(coordinates, threshold=threshold, normalize=normalize) <= 0.0
        if representation == "mesh":
            return (
                self.signed_distance_at(
                    coordinates,
                    threshold=threshold,
                    normalize=normalize,
                    source="mesh",
                    step_size=step_size,
                )
                <= 0.0
            )
        raise ValueError("representation must be 'occupancy', 'density', 'sdf', or 'mesh'.")

    def sample_points(
        self,
        points: npt.ArrayLike,
        *,
        fields: tuple[SampleFieldName, ...] = ("density", "occupancy", "deposition_index", "signed_distance"),
        threshold: float = 0.5,
        interpolation: InterpolationMode = "nearest",
        normalize: bool = False,
    ) -> dict[str, npt.NDArray[np.generic]]:
        samples, _single = _coerce_points(points)
        result: dict[str, npt.NDArray[np.generic]] = {}
        for field_name in fields:
            if field_name == "density":
                result[field_name] = _sample_scalar_field(
                    self.domain,
                    self.density_field(normalize=normalize),
                    samples,
                    interpolation=interpolation,
                    fill_value=0.0,
                )
            elif field_name == "deposition_index":
                result[field_name] = _sample_nearest(
                    self.domain,
                    self.deposition_index_field(),
                    samples,
                    fill_value=-1,
                )
            elif field_name == "occupancy":
                density_samples = _sample_scalar_field(
                    self.domain,
                    self.density_field(normalize=normalize),
                    samples,
                    interpolation=interpolation,
                    fill_value=0.0,
                )
                result[field_name] = density_samples >= threshold
            elif field_name == "signed_distance":
                sdf = self.surface_sdf(threshold=threshold, normalize=normalize)
                result[field_name] = np.asarray(sdf(samples), dtype=float)
            else:
                raise ValueError(
                    "fields entries must be 'density', 'occupancy', 'deposition_index', or 'signed_distance'."
                )
        return result

    def subvolume_stats(
        self,
        bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
        *,
        threshold: float = 0.5,
        normalize: bool = False,
        step_size: int = 1,
    ) -> dict[str, float]:
        minimum, maximum = bounds
        index_bounds = self.domain.index_bounds_for_aabb(minimum, maximum)
        if index_bounds is None:
            return {
                "voxel_count": 0.0,
                "occupied_voxel_count": 0.0,
                "occupied_fraction": 0.0,
                "density_mean": 0.0,
                "density_max": 0.0,
                "deposition_index_mean": 0.0,
                "deposition_index_max": 0.0,
                "mesh_area": 0.0,
            }

        slices = tuple(slice(start, stop) for start, stop in index_bounds)
        density = self.density_field(normalize=normalize)[slices]
        try:
            deposition_index = self.deposition_index_field().astype(float, copy=False)[slices]
        except ValueError:
            deposition_index = np.zeros_like(density)
        occupancy = self.occupancy_field(threshold=threshold, normalize=normalize)[slices]
        voxel_count = float(np.prod(density.shape))
        occupied_voxel_count = float(np.count_nonzero(occupancy))

        mesh_area = 0.0
        mesh = self.surface_mesh(threshold=threshold, normalize=normalize, step_size=step_size)
        if not mesh.is_empty:
            from ..mesh_analysis import face_areas, face_centroids

            centroids = face_centroids(mesh)
            areas = face_areas(mesh)
            min_array = np.asarray(ensure_finite_triplet(minimum, "minimum"), dtype=float)
            max_array = np.asarray(ensure_finite_triplet(maximum, "maximum"), dtype=float)
            inside = np.all((centroids >= min_array) & (centroids <= max_array), axis=1)
            mesh_area = float(np.sum(areas[inside]))

        return {
            "voxel_count": voxel_count,
            "occupied_voxel_count": occupied_voxel_count,
            "occupied_fraction": occupied_voxel_count / voxel_count if voxel_count else 0.0,
            "density_mean": float(np.mean(density)) if density.size else 0.0,
            "density_max": float(np.max(density)) if density.size else 0.0,
            "deposition_index_mean": float(np.mean(deposition_index)) if deposition_index.size else 0.0,
            "deposition_index_max": float(np.max(deposition_index)) if deposition_index.size else 0.0,
            "mesh_area": mesh_area,
        }

    def mesh_analysis(
        self,
        *,
        build_direction: tuple[float, float, float] | npt.ArrayLike = (0.0, 0.0, 1.0),
        critical_angle_deg: float = 45.0,
        threshold: float = 0.5,
        normalize: bool = False,
        step_size: int = 1,
    ) -> dict[str, Any]:
        build_dir = normalize_axis(build_direction, "build_direction")
        key = _surface_cache_key(threshold, normalize=normalize, step_size=step_size) + (
            build_dir,
            float(critical_angle_deg),
        )
        if key not in self._mesh_analysis_cache:
            from ..mesh_analysis import (
                downfacing_mask,
                face_areas,
                face_centroids,
                face_normals,
                overhang_angles,
                support_risk_mask,
            )

            mesh = self.surface_mesh(threshold=threshold, normalize=normalize, step_size=step_size)
            self._mesh_analysis_cache[key] = {
                "mesh": mesh,
                "face_normals": face_normals(mesh),
                "face_centroids": face_centroids(mesh),
                "face_areas": face_areas(mesh),
                "overhang_angles": overhang_angles(mesh, build_direction=build_dir),
                "downfacing_mask": downfacing_mask(mesh, build_direction=build_dir),
                "support_risk_mask": support_risk_mask(
                    mesh,
                    build_direction=build_dir,
                    critical_angle_deg=critical_angle_deg,
                ),
            }
        return self._mesh_analysis_cache[key]


def analysis_bundle(source: Any) -> AnalysisBundle:
    return _resolve_bundle(source)


def contains_point(source: Any, point: tuple[float, float, float] | npt.ArrayLike, **kwargs: Any) -> bool:
    return _resolve_bundle(source).contains_point(point, **kwargs)


def sample_density_at(source: Any, point: tuple[float, float, float] | npt.ArrayLike, **kwargs: Any) -> float:
    return _resolve_bundle(source).sample_density_at(point, **kwargs)


def sample_deposition_index_at(
    source: Any,
    point: tuple[float, float, float] | npt.ArrayLike,
    **kwargs: Any,
) -> int:
    return _resolve_bundle(source).sample_deposition_index_at(point, **kwargs)


def signed_distance_at(source: Any, point: tuple[float, float, float] | npt.ArrayLike, **kwargs: Any) -> float:
    return _resolve_bundle(source).signed_distance_at(point, **kwargs)


def surface_normal_at(
    source: Any,
    point: tuple[float, float, float] | npt.ArrayLike,
    **kwargs: Any,
) -> tuple[float, float, float]:
    return _resolve_bundle(source).surface_normal_at(point, **kwargs)


def sample_points(source: Any, points: npt.ArrayLike, **kwargs: Any) -> dict[str, npt.NDArray[np.generic]]:
    return _resolve_bundle(source).sample_points(points, **kwargs)


def subvolume_stats(
    source: Any,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    **kwargs: Any,
) -> dict[str, float]:
    return _resolve_bundle(source).subvolume_stats(bounds, **kwargs)
