"""Configuration and lazy entry points for optional visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from .results import SimulationResult
    from .simulator import Simulator
    from .workbench import SimulationWorkbench

ViewMode = Literal["surface", "occupancy", "implicit"]
ViewColorMode = Literal["plain", "normals", "overhang"]
ViewScalarField = Literal["occupancy", "implicit", "coverage", "deposition_order"]


@dataclass(slots=True, frozen=True)
class ViewConfig:
    """Initial state for the optional interactive workbench."""

    view_mode: ViewMode = "surface"
    scalar_field: ViewScalarField | None = None
    color_mode: ViewColorMode | None = None
    build_direction: str | tuple[float, float, float] = "+Z"
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
    simulator_or_result: "Simulator | SimulationResult",
    *,
    view_mode: "ViewMode" = "surface",
    initial_view: ViewConfig | None = None,
    threshold: float | None = None,
    off_screen: bool = False,
) -> "SimulationWorkbench":
    """Open the interactive workbench for a simulator or result snapshot.

    Requires ``pip install -e "[viz]"``.
    """

    from .results import SimulationResult
    from .workbench import SimulationWorkbench

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
