"""Optional visualization configuration and lazy public entry points."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
import numpy.typing as npt

from .styles import (
    DepositStyle,
    FrameStyle,
    LineStyle,
    MeshStyle,
    PointCloudStyle,
    PointStyle,
    TargetStyle,
)

if TYPE_CHECKING:
    from ..results import SimulationResult
    from ..simulator import Simulator
    from ..workbench import SimulationWorkbench
    from .viewer import Viewer, VisualHandle

ViewMode = Literal["surface", "occupancy", "implicit"]
ViewColorMode = Literal["plain", "normals", "overhang"]
ViewScalarField = Literal["occupancy", "implicit", "coverage", "deposition_order"]

__all__ = [
    "DepositStyle",
    "FrameStyle",
    "LineStyle",
    "MeshStyle",
    "PointCloudStyle",
    "PointStyle",
    "TargetStyle",
    "ViewColorMode",
    "ViewConfig",
    "ViewMode",
    "ViewScalarField",
    "Viewer",
    "VisualHandle",
    "show",
]


def __getattr__(name: str) -> object:
    if name in {"Viewer", "VisualHandle"}:
        module = import_module(".viewer", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _occupied_index_bounds(
    occupancy: npt.ArrayLike,
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    """Return inclusive occupied index bounds without materializing coordinates."""

    values = np.asarray(occupancy, dtype=bool)
    if values.ndim != 3:
        raise ValueError("occupancy must be a three-dimensional array.")

    bounds: list[tuple[int, int]] = []
    for axis in range(3):
        projection = np.any(
            values,
            axis=tuple(other_axis for other_axis in range(3) if other_axis != axis),
        )
        occupied_indices = np.flatnonzero(projection)
        if occupied_indices.size == 0:
            return None
        bounds.append((int(occupied_indices[0]), int(occupied_indices[-1])))

    return (
        (bounds[0][0], bounds[1][0], bounds[2][0]),
        (bounds[0][1], bounds[1][1], bounds[2][1]),
    )


@dataclass(slots=True, frozen=True)
class ViewConfig:
    """Initial state for the optional interactive workbench."""

    view_mode: ViewMode = "surface"
    scalar_field: ViewScalarField | None = None
    color_mode: ViewColorMode | None = None
    build_direction: str | tuple[float, float, float] = "+Z"
    show_toolpath: bool = False
    show_targets: bool = False
    show_world_axes: bool = False
    _VALID_DIRECTION_STRINGS: ClassVar[frozenset[str]] = frozenset(
        {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.build_direction, str)
            and self.build_direction not in self._VALID_DIRECTION_STRINGS
        ):
            raise ValueError(
                f"build_direction string {self.build_direction!r} is not valid. "
                f"Must be one of {sorted(self._VALID_DIRECTION_STRINGS)}."
            )


def show(
    simulator_or_result: Simulator | SimulationResult,
    *,
    view_mode: ViewMode = "surface",
    initial_view: ViewConfig | None = None,
    threshold: float | None = None,
    off_screen: bool = False,
) -> SimulationWorkbench:
    """Open the interactive workbench for a simulator or result snapshot."""

    from ..results import SimulationResult
    from ..workbench import SimulationWorkbench

    resolved_threshold = (
        simulator_or_result.default_threshold
        if threshold is None and isinstance(simulator_or_result, SimulationResult)
        else 0.5 if threshold is None else float(threshold)
    )
    workbench = SimulationWorkbench(
        simulator_or_result,
        threshold=resolved_threshold,
        off_screen=off_screen,
        initial_view=initial_view or ViewConfig(view_mode=view_mode),
    )
    workbench.show()
    return workbench
