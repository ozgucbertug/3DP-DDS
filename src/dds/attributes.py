"""Metadata containers for deposition primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DepositionAttributes:
    """Metadata describing nominal bead and process parameters."""

    width: float | None = None
    height: float | None = None
    layer_id: int | None = None
    material_id: str | None = None
    tool_id: str | None = None
    timestamp: float | None = None
    feedrate: float | None = None
    temperature: float | None = None
    user_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.width is not None and self.width <= 0.0:
            raise ValueError("DepositionAttributes.width must be positive when provided.")
        if self.height is not None and self.height <= 0.0:
            raise ValueError("DepositionAttributes.height must be positive when provided.")
        object.__setattr__(self, "user_data", dict(self.user_data))

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation that is easy to export."""

        return asdict(self)
