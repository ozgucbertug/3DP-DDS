"""Validated immutable styles for visualization primitives."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Color = str | tuple[float, float, float]


def _validate_color(value: Color, name: str) -> None:
    if isinstance(value, str):
        if not value:
            raise ValueError(f"{name} must not be empty")
        return
    if (
        len(value) != 3
        or not np.all(np.isfinite(value))
        or not np.all((0.0 <= np.asarray(value)) & (np.asarray(value) <= 1.0))
    ):
        raise ValueError(
            f"{name} must be a color string or three values between 0 and 1"
        )


def _positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")


def _opacity(value: float) -> None:
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("opacity must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class MeshStyle:
    color: Color | None = None
    opacity: float = 1.0
    show_edges: bool = False
    smooth_shading: bool = True

    def __post_init__(self) -> None:
        if self.color is not None:
            _validate_color(self.color, "color")
        _opacity(self.opacity)


@dataclass(frozen=True, slots=True)
class PointStyle:
    color: Color = "#d64292"
    size: float = 10.0
    render_as_spheres: bool = True
    opacity: float = 1.0

    def __post_init__(self) -> None:
        _validate_color(self.color, "color")
        _positive(self.size, "size")
        _opacity(self.opacity)


@dataclass(frozen=True, slots=True)
class PointCloudStyle:
    color: Color | None = None
    size: float = 3.0
    render_as_spheres: bool = False
    opacity: float = 1.0

    def __post_init__(self) -> None:
        if self.color is not None:
            _validate_color(self.color, "color")
        _positive(self.size, "size")
        _opacity(self.opacity)


@dataclass(frozen=True, slots=True)
class LineStyle:
    color: Color = "#34495e"
    width: float = 3.0
    opacity: float = 1.0
    render_as_tubes: bool = False

    def __post_init__(self) -> None:
        _validate_color(self.color, "color")
        _positive(self.width, "width")
        _opacity(self.opacity)


@dataclass(frozen=True, slots=True)
class FrameStyle:
    scale: float = 1.0
    line_width: float = 3.0
    show_origin: bool = True
    origin_style: PointStyle = field(
        default_factory=lambda: PointStyle(color="#252525", size=8.0)
    )

    def __post_init__(self) -> None:
        _positive(self.scale, "scale")
        _positive(self.line_width, "line_width")


@dataclass(frozen=True, slots=True)
class TargetStyle:
    scale: float = 1.0
    point_style: PointStyle = field(default_factory=PointStyle)
    normal_color: Color = "#f39c12"
    normal_width: float = 3.0

    def __post_init__(self) -> None:
        _positive(self.scale, "scale")
        _validate_color(self.normal_color, "normal_color")
        _positive(self.normal_width, "normal_width")


@dataclass(frozen=True, slots=True)
class DepositStyle:
    line_style: LineStyle = field(
        default_factory=lambda: LineStyle(color="#355c9a", width=4.0)
    )
    target_style: TargetStyle = field(default_factory=TargetStyle)
    show_path: bool = True
    show_targets: bool = True
    show_normals: bool = True
