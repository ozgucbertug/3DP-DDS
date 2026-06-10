"""Profile and metadata containers for deposition primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .utils import ensure_finite_scalar


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

    def to_dict(self) -> dict[str, Any]:
        """Return an export-friendly dictionary representation."""

        return {"width": self.width, "height": self.height}


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
