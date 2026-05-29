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

        voxel = float(voxel_size)
        if voxel <= 0.0:
            raise ValueError("voxel_size must be positive.")
        return cls(width=2.0 * voxel, height=voxel)

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
