"""Retained-mode PyVistaQt viewer for DDS geometry."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, cast

try:
    import pyvista as pv
    from PySide6 import QtWidgets
    from pyvistaqt import QtInteractor
except ImportError as exc:
    raise ImportError(
        'Viewer requires optional visualization dependencies. '
        'Install them with `pip install -e ".[viz]".'
    ) from exc

from ..geometry import PointCloud, TriangleMesh
from ..primitives import (
    Deposit,
    DepositionTarget,
    Line3D,
    LineDeposit,
    Point3D,
    PointDeposit,
    Polyline3D,
    PolylineDeposit,
    Pose3D,
    Vector3D,
)
from .converters import (
    line_to_polydata,
    point_cloud_to_polydata,
    points_to_polydata,
    polyline_to_polydata,
    triangle_mesh_to_polydata,
)
from .styles import (
    DepositStyle,
    FrameStyle,
    LineStyle,
    MeshStyle,
    PointCloudStyle,
    PointStyle,
    TargetStyle,
)

VisualKind = Literal[
    "mesh",
    "point_cloud",
    "points",
    "line",
    "polyline",
    "vector",
    "pose",
    "target",
    "deposits",
]
VisualSource = (
    TriangleMesh
    | PointCloud
    | tuple[Point3D, ...]
    | Line3D
    | Polyline3D
    | tuple[Point3D, Vector3D]
    | Pose3D
    | DepositionTarget
    | tuple[Deposit, ...]
)
VisualStyle = (
    MeshStyle
    | PointCloudStyle
    | PointStyle
    | LineStyle
    | FrameStyle
    | TargetStyle
    | DepositStyle
)


@dataclass(slots=True)
class _VisualRecord:
    kind: VisualKind
    source: VisualSource
    style: VisualStyle
    actors: list[Any]
    visible: bool = True


class VisualHandle:
    """Stable reference to one named visual, including composite visuals."""

    __slots__ = ("_viewer", "name")

    def __init__(self, viewer: Viewer, name: str) -> None:
        self._viewer = viewer
        self.name = name

    @property
    def visible(self) -> bool:
        return self._viewer._record(self.name).visible

    @property
    def source(self) -> VisualSource:
        return self._viewer._record(self.name).source

    @property
    def style(self) -> VisualStyle:
        return self._viewer._record(self.name).style

    def update(
        self,
        source: object | None = None,
        *,
        style: VisualStyle | None = None,
    ) -> VisualHandle:
        self._viewer._update(self.name, source=source, style=style)
        return self

    def set_visible(self, visible: bool) -> VisualHandle:
        self._viewer._set_visible(self.name, visible)
        return self

    def set_style(self, style: VisualStyle) -> VisualHandle:
        self._viewer._update(self.name, style=style)
        return self

    def remove(self) -> None:
        self._viewer.remove(self.name)


class Viewer:
    """Minimal retained-mode desktop viewer for DDS geometry."""

    def __init__(
        self,
        *,
        title: str = "3DP-DDS Viewer",
        off_screen: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.window: QtWidgets.QMainWindow | None = QtWidgets.QMainWindow(parent)
        assert self.window is not None
        self.window.setWindowTitle(title)
        self.window.resize(1100, 800)
        self.plotter = QtInteractor(self.window, off_screen=off_screen)
        self.window.setCentralWidget(getattr(self.plotter, "interactor", self.plotter))
        self._initialize_scene_state()
        self.plotter.set_background("#f5f6f8")

    @classmethod
    def _attach(cls, plotter: Any) -> Viewer:
        """Attach retained scene management to an existing workbench plotter."""

        viewer = cls.__new__(cls)
        viewer.plotter = plotter
        viewer.app = cast(Any, QtWidgets.QApplication.instance())
        viewer.window = None
        viewer._initialize_scene_state()
        return viewer

    def _initialize_scene_state(self) -> None:
        self._records: dict[str, _VisualRecord] = {}
        self._name_counters: dict[str, int] = {}
        self._batch_depth = 0
        self._render_pending = False

    def _record(self, name: str) -> _VisualRecord:
        try:
            return self._records[name]
        except KeyError as exc:
            raise KeyError(f"Unknown visual {name!r}") from exc

    def _next_name(self, kind: str) -> str:
        index = self._name_counters.get(kind, 0) + 1
        self._name_counters[kind] = index
        return f"{kind}_{index}"

    def _resolve_name(self, kind: str, name: str | None) -> str:
        resolved = self._next_name(kind) if name is None else name
        if not resolved:
            raise ValueError("name must not be empty")
        if resolved in self._records:
            raise ValueError(f"A visual named {resolved!r} already exists")
        return resolved

    def _capture_camera(self) -> tuple[Any, Any, Any, Any] | None:
        camera = getattr(self.plotter, "camera", None)
        if camera is None:
            return None
        return (
            tuple(camera.position),
            tuple(camera.focal_point),
            tuple(camera.up),
            float(camera.parallel_scale),
        )

    def _restore_camera(self, state: tuple[Any, Any, Any, Any] | None) -> None:
        if state is None:
            return
        camera = self.plotter.camera
        camera.position, camera.focal_point, camera.up, camera.parallel_scale = state
        self.plotter.reset_camera_clipping_range()

    def _request_render(self) -> None:
        if self._batch_depth > 0:
            self._render_pending = True
            return
        self.plotter.render()

    @contextmanager
    def batch(self) -> Iterator[Viewer]:
        """Defer rendering until the outermost batch completes."""

        self._batch_depth += 1
        try:
            yield self
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0 and self._render_pending:
                self._render_pending = False
                self.plotter.render()

    def _add_record(
        self,
        kind: VisualKind,
        source: VisualSource,
        style: VisualStyle,
        *,
        name: str | None,
    ) -> VisualHandle:
        resolved_name = self._resolve_name(kind, name)
        actors = self._build_actors(kind, source, style, resolved_name)
        self._records[resolved_name] = _VisualRecord(kind, source, style, actors)
        self._request_render()
        return VisualHandle(self, resolved_name)

    def add_mesh(
        self,
        mesh: TriangleMesh,
        *,
        style: MeshStyle | None = None,
        name: str | None = None,
    ) -> VisualHandle:
        if not isinstance(mesh, TriangleMesh):
            raise TypeError("mesh must be a TriangleMesh")
        return self._add_record("mesh", mesh, style or MeshStyle(), name=name)

    def add_point(
        self,
        point: Point3D,
        *,
        style: PointStyle | None = None,
        name: str | None = None,
    ) -> VisualHandle:
        if not isinstance(point, Point3D):
            raise TypeError("point must be a Point3D")
        return self.add_points((point,), style=style, name=name)

    def add_point_cloud(
        self,
        cloud: PointCloud,
        *,
        style: PointCloudStyle | None = None,
        name: str | None = None,
    ) -> VisualHandle:
        if not isinstance(cloud, PointCloud):
            raise TypeError("cloud must be a PointCloud")
        return self._add_record(
            "point_cloud",
            cloud,
            style or PointCloudStyle(),
            name=name,
        )

    def add_points(
        self,
        points: Sequence[Point3D],
        *,
        style: PointStyle | None = None,
        name: str | None = None,
    ) -> VisualHandle:
        resolved = tuple(points)
        if not resolved:
            raise ValueError("points must not be empty")
        if not all(isinstance(point, Point3D) for point in resolved):
            raise TypeError("points must contain Point3D values")
        return self._add_record("points", resolved, style or PointStyle(), name=name)

    def add_line(
        self,
        line: Line3D,
        *,
        style: LineStyle | None = None,
        name: str | None = None,
    ) -> VisualHandle:
        if not isinstance(line, Line3D):
            raise TypeError("line must be a Line3D")
        return self._add_record("line", line, style or LineStyle(), name=name)

    def add_polyline(
        self,
        polyline: Polyline3D,
        *,
        style: LineStyle | None = None,
        name: str | None = None,
    ) -> VisualHandle:
        if not isinstance(polyline, Polyline3D):
            raise TypeError("polyline must be a Polyline3D")
        return self._add_record("polyline", polyline, style or LineStyle(), name=name)

    def add_vector(
        self,
        origin: Point3D,
        vector: Vector3D,
        *,
        style: LineStyle | None = None,
        name: str | None = None,
    ) -> VisualHandle:
        if not isinstance(origin, Point3D) or not isinstance(vector, Vector3D):
            raise TypeError("origin and vector must be Point3D and Vector3D")
        if vector.length == 0.0:
            raise ValueError("vector must be non-zero")
        return self._add_record("vector", (origin, vector), style or LineStyle(), name=name)

    def add_pose(
        self,
        pose: Pose3D,
        *,
        style: FrameStyle | None = None,
        name: str | None = None,
    ) -> VisualHandle:
        if not isinstance(pose, Pose3D):
            raise TypeError("pose must be a Pose3D")
        return self._add_record("pose", pose, style or FrameStyle(), name=name)

    def add_target(
        self,
        target: DepositionTarget,
        *,
        style: TargetStyle | None = None,
        name: str | None = None,
    ) -> VisualHandle:
        if not isinstance(target, DepositionTarget):
            raise TypeError("target must be a DepositionTarget")
        return self._add_record("target", target, style or TargetStyle(), name=name)

    def add_deposit(
        self,
        deposit: Deposit,
        *,
        style: DepositStyle | None = None,
        name: str | None = None,
    ) -> VisualHandle:
        return self.add_deposits((deposit,), style=style, name=name)

    def add_deposits(
        self,
        deposits: Iterable[Deposit],
        *,
        style: DepositStyle | None = None,
        name: str | None = None,
    ) -> VisualHandle:
        resolved = tuple(deposits)
        if not resolved:
            raise ValueError("deposits must not be empty")
        if not all(
            isinstance(deposit, (PointDeposit, LineDeposit, PolylineDeposit))
            for deposit in resolved
        ):
            raise TypeError("deposits must contain deposition primitives")
        return self._add_record("deposits", resolved, style or DepositStyle(), name=name)

    def get(self, name: str) -> VisualHandle:
        self._record(name)
        return VisualHandle(self, name)

    def remove(self, visual: str | VisualHandle) -> None:
        name = visual.name if isinstance(visual, VisualHandle) else visual
        record = self._record(name)
        for actor in record.actors:
            self.plotter.remove_actor(actor, render=False)
        del self._records[name]
        self._request_render()

    def clear(self) -> None:
        with self.batch():
            for name in tuple(self._records):
                self.remove(name)

    def _set_visible(self, name: str, visible: bool) -> None:
        record = self._record(name)
        record.visible = bool(visible)
        for actor in record.actors:
            actor.SetVisibility(record.visible)
        self._request_render()

    def _update(
        self,
        name: str,
        *,
        source: object | None = None,
        style: VisualStyle | None = None,
    ) -> None:
        record = self._record(name)
        next_source = (
            record.source
            if source is None
            else self._normalize_source(record.kind, source)
        )
        next_style = record.style if style is None else style
        self._validate_record_types(record.kind, next_source, next_style)
        camera = self._capture_camera()
        for actor in record.actors:
            self.plotter.remove_actor(actor, render=False)
        actors = self._build_actors(record.kind, next_source, next_style, name)
        record.source = next_source
        record.style = next_style
        record.actors = actors
        for actor in actors:
            actor.SetVisibility(record.visible)
        self._restore_camera(camera)
        self._request_render()

    def _normalize_source(self, kind: VisualKind, source: object) -> VisualSource:
        if kind == "points":
            if isinstance(source, Point3D):
                return (source,)
            try:
                return tuple(cast(Iterable[Point3D], source))
            except TypeError as exc:
                raise TypeError(
                    "point visuals require a Point3D or iterable of Point3D"
                ) from exc
        if kind == "deposits":
            if isinstance(source, (PointDeposit, LineDeposit, PolylineDeposit)):
                return (source,)
            try:
                return tuple(cast(Iterable[Deposit], source))
            except TypeError as exc:
                raise TypeError(
                    "deposit visuals require a deposit or iterable of deposits"
                ) from exc
        return cast(VisualSource, source)

    def _validate_record_types(
        self,
        kind: VisualKind,
        source: VisualSource,
        style: VisualStyle,
    ) -> None:
        expected_styles: dict[VisualKind, type[VisualStyle]] = {
            "mesh": MeshStyle,
            "point_cloud": PointCloudStyle,
            "points": PointStyle,
            "line": LineStyle,
            "polyline": LineStyle,
            "vector": LineStyle,
            "pose": FrameStyle,
            "target": TargetStyle,
            "deposits": DepositStyle,
        }
        if not isinstance(style, expected_styles[kind]):
            raise TypeError(f"{kind} visuals require {expected_styles[kind].__name__}")
        valid_source = {
            "mesh": isinstance(source, TriangleMesh),
            "point_cloud": isinstance(source, PointCloud),
            "points": isinstance(source, tuple)
            and bool(source)
            and all(isinstance(value, Point3D) for value in source),
            "line": isinstance(source, Line3D),
            "polyline": isinstance(source, Polyline3D),
            "vector": isinstance(source, tuple)
            and len(source) == 2
            and isinstance(source[0], Point3D)
            and isinstance(source[1], Vector3D),
            "pose": isinstance(source, Pose3D),
            "target": isinstance(source, DepositionTarget),
            "deposits": isinstance(source, tuple)
            and bool(source)
            and all(
                isinstance(value, (PointDeposit, LineDeposit, PolylineDeposit))
                for value in source
            ),
        }[kind]
        if not valid_source:
            raise TypeError(f"source is not valid for {kind} visuals")

    def _add_dataset(
        self,
        dataset: Any,
        name: str,
        *,
        color: Any,
        opacity: float,
        line_width: float | None = None,
        point_size: float | None = None,
        render_points_as_spheres: bool = False,
        render_lines_as_tubes: bool = False,
        show_edges: bool = False,
        smooth_shading: bool = False,
        scalars: str | None = None,
        rgb: bool = False,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "name": name,
            "opacity": opacity,
            "render": False,
            "reset_camera": False,
            "show_scalar_bar": False,
            "show_edges": show_edges,
            "smooth_shading": smooth_shading,
        }
        if color is not None:
            kwargs["color"] = color
        if scalars is not None:
            kwargs["scalars"] = scalars
            kwargs["rgb"] = rgb
        if line_width is not None:
            kwargs["line_width"] = line_width
            kwargs["render_lines_as_tubes"] = render_lines_as_tubes
        if point_size is not None:
            kwargs["point_size"] = point_size
            kwargs["render_points_as_spheres"] = render_points_as_spheres
        return self.plotter.add_mesh(dataset, **kwargs)

    def _arrow_actor(
        self,
        origin: Point3D,
        vector: Vector3D,
        style: LineStyle,
        name: str,
    ) -> Any:
        arrow = pv.Arrow(
            start=origin.to_tuple(),
            direction=vector.to_tuple(),
            scale=vector.length,
            shaft_radius=0.01 * style.width,
            tip_radius=0.02 * style.width,
        )
        return self._add_dataset(
            arrow,
            name,
            color=style.color,
            opacity=style.opacity,
            smooth_shading=True,
        )

    def _target_actors(
        self,
        target: DepositionTarget,
        style: TargetStyle,
        name: str,
        *,
        show_point: bool = True,
        show_normal: bool = True,
    ) -> list[Any]:
        actors: list[Any] = []
        if show_point:
            point = points_to_polydata((target.position,), pv)
            actors.append(
                self._add_dataset(
                    point,
                    f"{name}:point",
                    color=style.point_style.color,
                    opacity=style.point_style.opacity,
                    point_size=style.point_style.size,
                    render_points_as_spheres=style.point_style.render_as_spheres,
                )
            )
        if show_normal:
            normal = Vector3D.from_value(
                (target.normal.to_array() * style.scale).tolist()
            )
            actors.append(
                self._arrow_actor(
                    target.position,
                    normal,
                    LineStyle(color=style.normal_color, width=style.normal_width),
                    f"{name}:normal",
                )
            )
        return actors

    def _deposit_targets(self, deposit: Deposit) -> tuple[DepositionTarget, ...]:
        if isinstance(deposit, PointDeposit):
            return (deposit.target,)
        if isinstance(deposit, LineDeposit):
            return (deposit.start, deposit.end)
        return deposit.targets

    def _build_actors(
        self,
        kind: VisualKind,
        source: VisualSource,
        style: VisualStyle,
        name: str,
    ) -> list[Any]:
        self._validate_record_types(kind, source, style)
        if kind == "mesh":
            mesh_style = cast(MeshStyle, style)
            dataset = triangle_mesh_to_polydata(cast(TriangleMesh, source), pv)
            if dataset.n_cells == 0:
                return []
            return [
                self._add_dataset(
                    dataset,
                    name,
                    color=mesh_style.color,
                    opacity=mesh_style.opacity,
                    show_edges=mesh_style.show_edges,
                    smooth_shading=mesh_style.smooth_shading,
                )
            ]
        if kind == "point_cloud":
            cloud = cast(PointCloud, source)
            cloud_style = cast(PointCloudStyle, style)
            dataset = point_cloud_to_polydata(cloud, pv)
            if dataset.n_points == 0:
                return []
            use_embedded_colors = (
                cloud_style.color is None and cloud.colors is not None
            )
            return [
                self._add_dataset(
                    dataset,
                    name,
                    color=None if use_embedded_colors else cloud_style.color or "#d64292",
                    opacity=cloud_style.opacity,
                    point_size=cloud_style.size,
                    render_points_as_spheres=cloud_style.render_as_spheres,
                    scalars="point_colors" if use_embedded_colors else None,
                    rgb=use_embedded_colors,
                )
            ]
        if kind == "points":
            point_style = cast(PointStyle, style)
            dataset = points_to_polydata(cast(tuple[Point3D, ...], source), pv)
            return [
                self._add_dataset(
                    dataset,
                    name,
                    color=point_style.color,
                    opacity=point_style.opacity,
                    point_size=point_style.size,
                    render_points_as_spheres=point_style.render_as_spheres,
                )
            ]
        if kind in {"line", "polyline"}:
            line_style = cast(LineStyle, style)
            dataset = (
                line_to_polydata(cast(Line3D, source), pv)
                if kind == "line"
                else polyline_to_polydata(cast(Polyline3D, source), pv)
            )
            return [
                self._add_dataset(
                    dataset,
                    name,
                    color=line_style.color,
                    opacity=line_style.opacity,
                    line_width=line_style.width,
                    render_lines_as_tubes=line_style.render_as_tubes,
                )
            ]
        if kind == "vector":
            origin, vector = cast(tuple[Point3D, Vector3D], source)
            return [self._arrow_actor(origin, vector, cast(LineStyle, style), name)]
        if kind == "pose":
            pose = cast(Pose3D, source)
            frame_style = cast(FrameStyle, style)
            actors: list[Any] = []
            for axis, color, label in (
                ((1.0, 0.0, 0.0), "#e74c3c", "x"),
                ((0.0, 1.0, 0.0), "#27ae60", "y"),
                ((0.0, 0.0, 1.0), "#2980b9", "z"),
            ):
                vector = pose.transform_vector(axis)
                scaled = Vector3D.from_value(
                    (vector.to_array() * frame_style.scale).tolist()
                )
                actors.append(
                    self._arrow_actor(
                        pose.position,
                        scaled,
                        LineStyle(color=color, width=frame_style.line_width),
                        f"{name}:{label}",
                    )
                )
            if frame_style.show_origin:
                origin = points_to_polydata((pose.position,), pv)
                point_style = frame_style.origin_style
                actors.append(
                    self._add_dataset(
                        origin,
                        f"{name}:origin",
                        color=point_style.color,
                        opacity=point_style.opacity,
                        point_size=point_style.size,
                        render_points_as_spheres=point_style.render_as_spheres,
                    )
                )
            return actors
        if kind == "target":
            return self._target_actors(
                cast(DepositionTarget, source),
                cast(TargetStyle, style),
                name,
            )

        deposit_style = cast(DepositStyle, style)
        actors = []
        for index, deposit in enumerate(cast(tuple[Deposit, ...], source)):
            targets = self._deposit_targets(deposit)
            child_name = f"{name}:{index}"
            if deposit_style.show_path and len(targets) >= 2:
                polyline = Polyline3D(tuple(target.position for target in targets))
                dataset = polyline_to_polydata(polyline, pv)
                line_style = deposit_style.line_style
                actors.append(
                    self._add_dataset(
                        dataset,
                        f"{child_name}:path",
                        color=line_style.color,
                        opacity=line_style.opacity,
                        line_width=line_style.width,
                        render_lines_as_tubes=line_style.render_as_tubes,
                    )
                )
            for target_index, target in enumerate(targets):
                actors.extend(
                    self._target_actors(
                        target,
                        deposit_style.target_style,
                        f"{child_name}:target:{target_index}",
                        show_point=deposit_style.show_targets,
                        show_normal=deposit_style.show_normals,
                    )
                )
        return actors

    def apply_camera_preset(
        self,
        preset: Literal["perspective", "top", "front", "left"],
    ) -> None:
        if preset == "perspective":
            self.plotter.view_isometric()
        elif preset == "top":
            self.plotter.view_xy()
        elif preset == "front":
            self.plotter.view_xz()
        elif preset == "left":
            self.plotter.view_yz()
        else:
            raise ValueError("Unknown camera preset")
        self.reset_camera()

    def reset_camera(self) -> None:
        self.plotter.reset_camera(render=False)
        self.plotter.reset_camera_clipping_range()
        self._request_render()

    def show(self) -> None:
        if self.window is None:
            raise RuntimeError("Attached viewers do not own a window")
        self.window.show()
        self.plotter.render()

    def run(self) -> int:
        self.show()
        return int(self.app.exec())

    def close(self) -> None:
        self.clear()
        self.plotter.close()
        if self.window is not None:
            self.window.close()
