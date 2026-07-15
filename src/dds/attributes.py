"""Physical attributes for deposition primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .utils import ensure_finite_scalar


@dataclass(frozen=True)
class BeadProfile:
    """Nominal bead geometry used by deposition kernels.

    Parameters
    ----------
    width
        Full bead width transverse to the deposition normal.
    height
        Full bead height measured along the deposition normal.

    Notes
    -----
    Width and height are world-space values in the same unit as the simulation
    :class:`dds.Domain`. Every deposit requires an explicit ``BeadProfile``.
    """

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
