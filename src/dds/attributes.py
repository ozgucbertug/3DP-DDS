"""Profile and metadata containers for deposition primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class BeadProfile:
    """Nominal bead geometry used by deposition kernels."""

    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0.0:
            raise ValueError("BeadProfile.width must be positive.")
        if self.height <= 0.0:
            raise ValueError("BeadProfile.height must be positive.")

    def to_dict(self) -> dict[str, Any]:
        """Return an export-friendly dictionary representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class DepositionMetadata:
    """Process or provenance metadata describing a deposition event."""

    layer_id: int | None = None
    material_id: str | None = None
    tool_id: str | None = None
    timestamp: float | None = None
    feedrate: float | None = None
    temperature: float | None = None
    user_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_data", dict(self.user_data))

    def to_dict(self) -> dict[str, Any]:
        """Return an export-friendly dictionary representation."""

        return asdict(self)
