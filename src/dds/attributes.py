"""Profile and metadata containers for deposition primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .utils import ensure_finite_scalar

_SUPPORTED_LENGTH_UNITS = frozenset({"mm", "m"})
_SUPPORTED_TIME_UNITS = frozenset({"s"})
_SUPPORTED_TEMPERATURE_UNITS = frozenset({"degC", "K"})


def _freeze_json_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return ensure_finite_scalar(value, path)
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings.")
            frozen[key] = _freeze_json_value(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__!r}.")


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class BeadProfile:
    """Nominal bead geometry used by deposition kernels."""

    width: float
    height: float

    def __post_init__(self) -> None:
        width = ensure_finite_scalar(self.width, "BeadProfile.width")
        height = ensure_finite_scalar(self.height, "BeadProfile.height")
        if width <= 0.0:
            raise ValueError("BeadProfile.width must be positive.")
        if height <= 0.0:
            raise ValueError("BeadProfile.height must be positive.")
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)

    @classmethod
    def default(cls, voxel_size: float = 1.0) -> "BeadProfile":
        """Return a sensible default bead profile sized to the given voxel resolution.

        The default bead is a 2-voxel wide, 1-voxel tall rounded cylinder.
        This is a convenience for quick experimentation — production toolpaths
        should always supply explicit profiles.

        Parameters
        ----------
        voxel_size:
            Edge length of a single voxel in world units (default 1.0).
            Width and height are set to ``2 * voxel_size`` and ``voxel_size``.
        """

        voxel = ensure_finite_scalar(voxel_size, "voxel_size")
        if voxel <= 0.0:
            raise ValueError("voxel_size must be positive.")
        return cls(width=2.0 * voxel, height=voxel)

    def to_dict(self) -> dict[str, Any]:
        """Return an export-friendly dictionary representation."""

        return {"width": self.width, "height": self.height}


@dataclass(frozen=True, slots=True)
class UnitSystem:
    """Units used by domain geometry and process-state values."""

    length: str = "mm"
    time: str = "s"
    temperature: str = "degC"

    def __post_init__(self) -> None:
        if self.length not in _SUPPORTED_LENGTH_UNITS:
            raise ValueError(f"length must be one of {sorted(_SUPPORTED_LENGTH_UNITS)}.")
        if self.time not in _SUPPORTED_TIME_UNITS:
            raise ValueError(f"time must be one of {sorted(_SUPPORTED_TIME_UNITS)}.")
        if self.temperature not in _SUPPORTED_TEMPERATURE_UNITS:
            raise ValueError(
                f"temperature must be one of {sorted(_SUPPORTED_TEMPERATURE_UNITS)}."
            )

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible representation."""

        return {
            "length": self.length,
            "time": self.time,
            "temperature": self.temperature,
        }


@dataclass(frozen=True, slots=True)
class ProcessState:
    """Robot and material process state associated with a deposition event."""

    material_id: str | None = None
    tool_id: str | None = None
    timestamp: float | None = None
    feedrate: float | None = None
    extrusion_rate: float | None = None
    temperature: float | None = None

    def __post_init__(self) -> None:
        for name in ("material_id", "tool_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"ProcessState.{name} must be a non-empty string or None.")
        for name in ("timestamp", "feedrate", "extrusion_rate", "temperature"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    ensure_finite_scalar(value, f"ProcessState.{name}"),
                )
        for name in ("feedrate", "extrusion_rate"):
            value = getattr(self, name)
            if value is not None and value < 0.0:
                raise ValueError(f"ProcessState.{name} must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "material_id": self.material_id,
            "tool_id": self.tool_id,
            "timestamp": self.timestamp,
            "feedrate": self.feedrate,
            "extrusion_rate": self.extrusion_rate,
            "temperature": self.temperature,
        }


@dataclass(frozen=True, slots=True)
class DepositionMetadata:
    """Provenance and user annotations describing a deposition event."""

    layer_id: int | None = None
    user_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.layer_id is not None:
            if isinstance(self.layer_id, bool) or not isinstance(self.layer_id, int):
                raise TypeError("DepositionMetadata.layer_id must be an integer or None.")
            if self.layer_id < 0:
                raise ValueError("DepositionMetadata.layer_id must be non-negative.")
        object.__setattr__(
            self,
            "user_data",
            _freeze_json_value(self.user_data, path="DepositionMetadata.user_data"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return an export-friendly dictionary representation."""

        return {
            "layer_id": self.layer_id,
            "user_data": _thaw_json_value(self.user_data),
        }
