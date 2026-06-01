"""Optional visualization helpers.

Importing from this module requires the [viz] extras:
    pip install -e ".[viz]"
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .results import SimulationResult, ViewMode
    from .workbench import SimulationWorkbench


def show(
    result: "SimulationResult",
    *,
    view_mode: "ViewMode" = "surface",
    off_screen: bool = False,
) -> "SimulationWorkbench":
    """Open the interactive workbench for a SimulationResult.

    Requires ``pip install -e "[viz]"``.
    """

    from .results import WorkbenchViewConfig
    from .workbench import SimulationWorkbench

    workbench = SimulationWorkbench(
        result,
        threshold=result.default_threshold,
        off_screen=off_screen,
        initial_view=WorkbenchViewConfig(view_mode=view_mode),
    )
    workbench.show()
    return workbench
