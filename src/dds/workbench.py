"""Optional PyVistaQt workbench for interactive dense-field inspection."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt

try:
    from PySide6 import QtCore, QtWidgets
    import pyvista as pv
    from pyvistaqt import QtInteractor
except ImportError as exc:
    raise ImportError(
        'SimulationWorkbench requires optional visualization dependencies. '
        'Install them with `pip install -e ".[viz]"`.'
    ) from exc

from .domain import Domain
from .mesh_analysis import normal_rgb_from_normals, overhang_angles
from .analysis import AnalysisBundle
from .results import SimulationResult, WorkbenchViewConfig, simulation_result

Representation = Literal["surface", "occupancy", "density"]
ColorMode = Literal["plain", "normals", "overhang"]
ScalarFieldName = Literal["occupancy", "density", "accumulation", "deposition_order"]


def _field_to_image_data(
    domain: Domain,
    values: npt.ArrayLike,
    *,
    field_name: str,
    association: Literal["cell", "point"],
) -> Any:
    array = np.asarray(values)
    if array.shape != domain.grid_shape:
        raise ValueError(f"{field_name} shape {array.shape} does not match domain grid shape {domain.grid_shape}.")

    grid = pv.ImageData()
    spacing = np.asarray(domain.voxel_size, dtype=float)
    if association == "cell":
        grid.dimensions = np.asarray(domain.grid_shape, dtype=int) + 1
        grid.origin = domain.min_corner
        grid.spacing = domain.voxel_size
        grid.cell_data[field_name] = np.ascontiguousarray(array).ravel(order="F")
    else:
        grid.dimensions = np.asarray(domain.grid_shape, dtype=int)
        grid.origin = tuple(
            float(domain.min_corner[axis] + 0.5 * spacing[axis])
            for axis in range(3)
        )
        grid.spacing = domain.voxel_size
        grid.point_data[field_name] = np.ascontiguousarray(array).ravel(order="F")
    grid.set_active_scalars(field_name)
    return grid


def _triangle_mesh_to_polydata(mesh: Any) -> Any:
    if mesh.is_empty:
        return pv.PolyData()
    faces = np.hstack(
        [
            np.full((mesh.n_faces, 1), 3, dtype=np.int64),
            mesh.faces.astype(np.int64, copy=False),
        ]
    ).ravel()
    return pv.PolyData(np.asarray(mesh.vertices, dtype=float), faces)


class SimulationWorkbench(QtWidgets.QMainWindow):
    """Minimal Qt workbench for dense-field inspection and mesh analysis."""

    _BUILD_DIRECTIONS: dict[str, tuple[float, float, float]] = {
        "+X": (1.0, 0.0, 0.0),
        "-X": (-1.0, 0.0, 0.0),
        "+Y": (0.0, 1.0, 0.0),
        "-Y": (0.0, -1.0, 0.0),
        "+Z": (0.0, 0.0, 1.0),
        "-Z": (0.0, 0.0, -1.0),
    }
    _SCALAR_BAR_TITLES = ("Density", "Overhang (deg)")

    def __init__(
        self,
        simulator_or_bundle: SimulationResult | AnalysisBundle | Any,
        *,
        threshold: float = 0.5,
        build_direction: str | tuple[float, float, float] = "+Z",
        initial_view: WorkbenchViewConfig | None = None,
        off_screen: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        super().__init__(parent)

        self.result = simulation_result(simulator_or_bundle, threshold=threshold)
        self.bundle = self.result.analysis_bundle()
        self.threshold = float(threshold)
        self.representation: Representation = "surface"
        self.view_opacity: dict[Representation, float] = {
            "surface": 1.0,
            "occupancy": 1.0,
            "density": 1.0,
        }
        self.occupancy_field_name = "occupancy"
        self.density_field_name = "density"
        self.color_mode: ColorMode = "plain"
        self.build_direction = self._coerce_build_direction(build_direction)
        self.off_screen = bool(off_screen)
        self.clip_enabled = False
        self.point_picking_enabled = False
        self.roi_enabled = False

        self._surface_actor: Any | None = None
        self._occupancy_actor: Any | None = None
        self._density_actor: Any | None = None
        self._clip_actor: Any | None = None
        self._clip_widget: Any | None = None
        self._roi_widget: Any | None = None
        self._pick_marker_actor: Any | None = None
        self._roi_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
        self._last_pick_payload: dict[str, Any] | None = None
        self._last_roi_stats: dict[str, float] | None = None
        self._surface_polydata_cache: dict[float, Any] = {}
        self._occupied_bounds_cache: dict[float, tuple[float, float, float, float, float, float]] = {}
        self._density_sum = self.result.density_sum
        self._surface_coloring_registry: dict[str, Any] = {
            "plain": self._apply_plain_surface_coloring,
            "normals": self._apply_normal_surface_coloring,
            "overhang": self._apply_overhang_surface_coloring,
        }
        self._scalar_field_registry: dict[str, dict[str, Any]] = {
            "occupancy": {
                "occupancy": self._occupancy_scalar_field,
                "deposition_order": self._deposition_order_scalar_field,
            },
            "density": {
                "density": self._density_scalar_field,
                "accumulation": self._accumulation_scalar_field,
                "deposition_order": self._deposition_order_scalar_field,
            },
        }
        self._scalar_field_labels: dict[str, dict[str, str]] = {
            "occupancy": {
                "occupancy": "Occupancy",
                "deposition_order": "Deposition Order",
            },
            "density": {
                "density": "Density",
                "accumulation": "Accumulation",
                "deposition_order": "Deposition Order",
            },
        }
        self._overlay_registry: dict[str, dict[str, Any]] = {
            "clip": {"activate": self._activate_clip_widget, "clear": self._clear_clip_state},
            "point_picking": {"activate": self._install_point_picking, "clear": self.clear_pick},
            "roi": {"refresh": self._refresh_roi_stats},
        }
        self._initial_view = self._resolve_initial_view_config(initial_view, build_direction)

        self._build_ui()
        self._apply_window_style()
        self._rebuild_scene()
        self._apply_initial_view_config()
        self._initialize_camera()
        self.set_point_picking_enabled(False)
        self._sync_threshold_controls()
        self._sync_opacity_controls()
        self._sync_scalar_field_options()
        self._sync_surface_controls()
        self._sync_status_controls()

    def _resolve_initial_view_config(
        self,
        initial_view: WorkbenchViewConfig | None,
        build_direction: str | tuple[float, float, float],
    ) -> WorkbenchViewConfig:
        config = initial_view or WorkbenchViewConfig(build_direction=build_direction)
        view_mode = config.view_mode

        if view_mode == "surface":
            color_mode = config.color_mode or "plain"
            scalar_field = None
        elif view_mode == "occupancy":
            labels = self._available_scalar_field_labels("occupancy")
            scalar_field = config.scalar_field if config.scalar_field in labels else next(iter(labels))
            color_mode = config.color_mode or "plain"
        else:
            labels = self._available_scalar_field_labels("density")
            scalar_field = config.scalar_field if config.scalar_field in labels else next(iter(labels))
            color_mode = config.color_mode or "plain"

        return WorkbenchViewConfig(
            view_mode=view_mode,
            scalar_field=scalar_field,
            color_mode=color_mode,
            build_direction=config.build_direction,
        )

    def _apply_initial_view_config(self) -> None:
        config = self._initial_view
        self.representation = config.view_mode
        self.color_mode = config.color_mode or "plain"
        self.build_direction = self._coerce_build_direction(config.build_direction)
        if self.representation == "occupancy" and config.scalar_field is not None:
            self.occupancy_field_name = config.scalar_field
        if self.representation == "density":
            if config.scalar_field is not None:
                self.density_field_name = config.scalar_field

        self._set_combo_current_data(self.view_mode_combo, self.representation)
        self._set_combo_current_data(self.color_mode_combo, self.color_mode)
        self.build_direction_combo.setCurrentText(self._build_direction_label())
        self._sync_scalar_field_options()
        self._sync_surface_controls()
        self._sync_status_controls()
        self._sync_opacity_controls()
        self._rebuild_scene()

    def _build_ui(self) -> None:
        self.setWindowTitle("3DP-DDS Workbench")
        self.resize(1440, 920)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        viewport_frame = QtWidgets.QFrame(central)
        viewport_frame.setObjectName("viewportFrame")
        viewport_layout = QtWidgets.QVBoxLayout(viewport_frame)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(viewport_frame, off_screen=self.off_screen)
        viewport_layout.addWidget(getattr(self.plotter, "interactor", self.plotter))
        layout.addWidget(viewport_frame, stretch=1)

        sidebar_scroll = QtWidgets.QScrollArea(central)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        sidebar_scroll.setFixedWidth(380)
        layout.addWidget(sidebar_scroll)

        sidebar = QtWidgets.QWidget(sidebar_scroll)
        sidebar_scroll.setWidget(sidebar)
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(10)

        view_box, view_layout = self._add_section(sidebar_layout, "View")
        view_mode_row = QtWidgets.QWidget(view_box)
        view_mode_row_layout = QtWidgets.QVBoxLayout(view_mode_row)
        view_mode_row_layout.setContentsMargins(0, 0, 0, 0)
        view_mode_row_layout.setSpacing(6)
        self.view_mode_label = QtWidgets.QLabel("Mode", view_mode_row)
        view_mode_row_layout.addWidget(self.view_mode_label)
        self.view_mode_combo = QtWidgets.QComboBox(view_mode_row)
        self._configure_combo_box(self.view_mode_combo, min_width=170)
        self.view_mode_combo.addItem("Surface", "surface")
        self.view_mode_combo.addItem("Occupancy", "occupancy")
        self.view_mode_combo.addItem("Density", "density")
        self.view_mode_combo.currentIndexChanged.connect(
            lambda _index: self.set_representation(self.view_mode_combo.currentData())
        )
        view_mode_row_layout.addWidget(self.view_mode_combo)
        view_layout.addRow(view_mode_row)

        preset_widget = QtWidgets.QWidget(view_box)
        preset_layout = QtWidgets.QHBoxLayout(preset_widget)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(6)
        for label in ("Perspective", "Top", "Front", "Left"):
            button = QtWidgets.QPushButton(label, preset_widget)
            button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
            selected = label.lower()
            button.clicked.connect(lambda _checked=False, selected=selected: self.apply_camera_preset(selected))
            preset_layout.addWidget(button)
        view_layout.addRow("Camera", preset_widget)

        display_box, display_layout = self._add_section(sidebar_layout, "Display")
        self.display_box = display_box
        self.display_layout = display_layout
        threshold_row = QtWidgets.QWidget(display_box)
        threshold_row_layout = QtWidgets.QVBoxLayout(threshold_row)
        threshold_row_layout.setContentsMargins(0, 0, 0, 0)
        threshold_row_layout.setSpacing(6)
        threshold_label = QtWidgets.QLabel("Threshold", threshold_row)
        threshold_row_layout.addWidget(threshold_label)
        threshold_widget = QtWidgets.QWidget(threshold_row)
        threshold_layout = QtWidgets.QHBoxLayout(threshold_widget)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.setSpacing(8)
        self.threshold_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, threshold_widget)
        self.threshold_slider.setRange(0, 1000)
        self.threshold_slider.setSingleStep(10)
        self.threshold_slider.setPageStep(50)
        self.threshold_slider.valueChanged.connect(self._on_threshold_slider_changed)
        threshold_layout.addWidget(self.threshold_slider, stretch=1)
        self.threshold_spin = QtWidgets.QDoubleSpinBox(threshold_widget)
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setRange(0.0, 1_000_000.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(self.threshold)
        self.threshold_spin.setFixedWidth(84)
        self.threshold_spin.valueChanged.connect(self._on_threshold_spin_changed)
        threshold_layout.addWidget(self.threshold_spin)
        threshold_row_layout.addWidget(threshold_widget)
        display_layout.addRow(threshold_row)

        opacity_row = QtWidgets.QWidget(display_box)
        opacity_row_layout = QtWidgets.QVBoxLayout(opacity_row)
        opacity_row_layout.setContentsMargins(0, 0, 0, 0)
        opacity_row_layout.setSpacing(6)
        opacity_label = QtWidgets.QLabel("Opacity", opacity_row)
        opacity_row_layout.addWidget(opacity_label)
        opacity_widget = QtWidgets.QWidget(opacity_row)
        opacity_layout = QtWidgets.QHBoxLayout(opacity_widget)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        opacity_layout.setSpacing(8)
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, opacity_widget)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setSingleStep(5)
        self.opacity_slider.setPageStep(10)
        self.opacity_slider.valueChanged.connect(self._on_opacity_slider_changed)
        opacity_layout.addWidget(self.opacity_slider, stretch=1)
        self.opacity_spin = QtWidgets.QDoubleSpinBox(opacity_widget)
        self.opacity_spin.setDecimals(2)
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setFixedWidth(72)
        self.opacity_spin.valueChanged.connect(self._on_opacity_spin_changed)
        opacity_layout.addWidget(self.opacity_spin)
        opacity_row_layout.addWidget(opacity_widget)
        display_layout.addRow(opacity_row)

        self.scalar_field_row = QtWidgets.QWidget(display_box)
        scalar_field_row_layout = QtWidgets.QVBoxLayout(self.scalar_field_row)
        scalar_field_row_layout.setContentsMargins(0, 0, 0, 0)
        scalar_field_row_layout.setSpacing(6)
        self.scalar_field_label = QtWidgets.QLabel("Field", self.scalar_field_row)
        scalar_field_row_layout.addWidget(self.scalar_field_label)
        self.scalar_field_combo = QtWidgets.QComboBox(self.scalar_field_row)
        self._configure_combo_box(self.scalar_field_combo, min_width=170)
        self.scalar_field_combo.currentIndexChanged.connect(
            lambda _index: self.set_scalar_field(self.scalar_field_combo.currentData())
        )
        scalar_field_row_layout.addWidget(self.scalar_field_combo)
        display_layout.addRow(self.scalar_field_row)

        self.color_mode_row = QtWidgets.QWidget(display_box)
        color_mode_row_layout = QtWidgets.QVBoxLayout(self.color_mode_row)
        color_mode_row_layout.setContentsMargins(0, 0, 0, 0)
        color_mode_row_layout.setSpacing(6)
        self._surface_coloring_label = QtWidgets.QLabel("Coloring", self.color_mode_row)
        color_mode_row_layout.addWidget(self._surface_coloring_label)
        self.color_mode_combo = QtWidgets.QComboBox(self.color_mode_row)
        self._configure_combo_box(self.color_mode_combo, min_width=170)
        self.color_mode_combo.addItem("Plain", "plain")
        self.color_mode_combo.addItem("Normal", "normals")
        self.color_mode_combo.addItem("Overhang", "overhang")
        self.color_mode_combo.currentIndexChanged.connect(
            lambda _index: self.set_color_mode(self.color_mode_combo.currentData())
        )
        color_mode_row_layout.addWidget(self.color_mode_combo)
        display_layout.addRow(self.color_mode_row)

        surface_box, surface_layout = self._add_section(sidebar_layout, "Surface")
        self.surface_box = surface_box
        self.surface_layout = surface_layout

        self.build_direction_row = QtWidgets.QWidget(surface_box)
        build_direction_row_layout = QtWidgets.QVBoxLayout(self.build_direction_row)
        build_direction_row_layout.setContentsMargins(0, 0, 0, 0)
        build_direction_row_layout.setSpacing(6)
        self._build_direction_label_widget = QtWidgets.QLabel("Build Dir.", self.build_direction_row)
        build_direction_row_layout.addWidget(self._build_direction_label_widget)
        self.build_direction_combo = QtWidgets.QComboBox(self.build_direction_row)
        self._configure_combo_box(self.build_direction_combo, min_width=120)
        for label in self._BUILD_DIRECTIONS:
            self.build_direction_combo.addItem(label, label)
        self.build_direction_combo.setCurrentText(self._build_direction_label())
        self.build_direction_combo.currentIndexChanged.connect(
            lambda _index: self.set_build_direction(self.build_direction_combo.currentData())
        )
        build_direction_row_layout.addWidget(self.build_direction_combo)
        surface_layout.addRow(self.build_direction_row)

        tools_box, tools_layout = self._add_section(sidebar_layout, "Tools")
        self.clip_checkbox = QtWidgets.QCheckBox("Clip Plane", tools_box)
        self.clip_checkbox.toggled.connect(self.set_clip_enabled)
        tools_layout.addRow(self.clip_checkbox)

        status_box, status_layout = self._add_section(sidebar_layout, "Stats")
        self.point_picking_checkbox = QtWidgets.QCheckBox("Point", status_box)
        self.point_picking_checkbox.setChecked(False)
        self.point_picking_checkbox.toggled.connect(self.set_point_picking_enabled)
        status_layout.addRow(self.point_picking_checkbox)

        self.clear_pick_button = QtWidgets.QPushButton("Clear Pick", status_box)
        self.clear_pick_button.clicked.connect(self.clear_pick)
        status_layout.addRow(self.clear_pick_button)

        self.pick_status = QtWidgets.QPlainTextEdit(status_box)
        self.pick_status.setReadOnly(True)
        self.pick_status.setPlaceholderText("Pick a point in the viewport to inspect field values.")
        self.pick_status.setMaximumBlockCount(12)
        self.pick_status.setMinimumHeight(178)
        status_layout.addRow("Picked Point", self.pick_status)
        self._pick_status_label_widget = status_layout.labelForField(self.pick_status)

        self.roi_checkbox = QtWidgets.QCheckBox("Box", status_box)
        self.roi_checkbox.toggled.connect(self.set_roi_enabled)
        status_layout.addRow(self.roi_checkbox)

        self.reset_roi_button = QtWidgets.QPushButton("Reset ROI", status_box)
        self.reset_roi_button.clicked.connect(self.reset_roi)
        status_layout.addRow(self.reset_roi_button)

        self.roi_status = QtWidgets.QPlainTextEdit(status_box)
        self.roi_status.setReadOnly(True)
        self.roi_status.setPlaceholderText("Enable the ROI box to inspect subvolume statistics.")
        self.roi_status.setMaximumBlockCount(8)
        self.roi_status.setMinimumHeight(126)
        status_layout.addRow("ROI", self.roi_status)
        self._roi_status_label_widget = status_layout.labelForField(self.roi_status)

        self.support_status = QtWidgets.QPlainTextEdit(status_box)
        self.support_status.setReadOnly(True)
        self.support_status.setPlaceholderText("Activate overhang or support-shadow analysis to inspect support metrics.")
        self.support_status.setMaximumBlockCount(8)
        self.support_status.setMinimumHeight(118)
        status_layout.addRow("Support", self.support_status)
        self._support_status_label_widget = status_layout.labelForField(self.support_status)

        sidebar_layout.addStretch(1)

        self.plotter.set_background("#f5f6f8")
        self.plotter.add_axes(line_width=2)

    def _apply_window_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f5f6f8;
                color: #1d2733;
            }
            QFrame#viewportFrame {
                border: 1px solid #d7dde5;
                border-radius: 10px;
                background: #ffffff;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d7dde5;
                border-radius: 10px;
                font-weight: 600;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QComboBox,
            QDoubleSpinBox,
            QPushButton,
            QPlainTextEdit {
                background: #ffffff;
                border: 1px solid #cdd5df;
                border-radius: 6px;
                padding: 6px 8px;
            }
            QPushButton {
                font-weight: 500;
            }
            QLabel {
                background: transparent;
            }
            """
        )

    def _configure_combo_box(self, combo: QtWidgets.QComboBox, *, min_width: int) -> None:
        combo.setMinimumWidth(min_width)
        combo.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        view = combo.view()
        if view is not None:
            view.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
            view.setMinimumWidth(max(min_width, 150))

    def _set_combo_current_data(self, combo: QtWidgets.QComboBox, value: Any) -> None:
        index = combo.findData(value)
        if index < 0:
            return
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _add_section(
        self,
        parent_layout: QtWidgets.QVBoxLayout,
        title: str,
    ) -> tuple[QtWidgets.QGroupBox, QtWidgets.QFormLayout]:
        box = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QFormLayout(box)
        layout.setContentsMargins(12, 14, 12, 12)
        layout.setSpacing(8)
        layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        parent_layout.addWidget(box)
        return box, layout

    def _coerce_build_direction(
        self,
        axis_or_vector: str | tuple[float, float, float] | npt.ArrayLike,
    ) -> tuple[float, float, float]:
        if isinstance(axis_or_vector, str):
            if axis_or_vector not in self._BUILD_DIRECTIONS:
                raise ValueError(f"build_direction must be one of {tuple(self._BUILD_DIRECTIONS)}.")
            return self._BUILD_DIRECTIONS[axis_or_vector]
        vector = np.asarray(axis_or_vector, dtype=float)
        if vector.shape != (3,):
            raise ValueError("build_direction must contain exactly three coordinates.")
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            raise ValueError("build_direction must not be the zero vector.")
        return tuple(float(value) for value in (vector / norm))

    def _build_direction_label(self) -> str:
        for label, vector in self._BUILD_DIRECTIONS.items():
            if np.allclose(vector, self.build_direction):
                return label
        return "+Z"

    def _domain_bounds(self) -> tuple[float, float, float, float, float, float]:
        return (
            float(self.bundle.domain.min_corner[0]),
            float(self.bundle.domain.max_corner[0]),
            float(self.bundle.domain.min_corner[1]),
            float(self.bundle.domain.max_corner[1]),
            float(self.bundle.domain.min_corner[2]),
            float(self.bundle.domain.max_corner[2]),
        )

    def _sdf_vertex_normals(self, mesh: Any) -> npt.NDArray[np.float64]:
        if mesh.is_empty:
            return np.empty((0, 3), dtype=float)
        sdf = self.bundle.surface_sdf(threshold=self.threshold)
        vertices = np.asarray(mesh.vertices, dtype=float)
        steps = 0.5 * np.asarray(self.bundle.domain.voxel_size, dtype=float)
        gradients = np.empty_like(vertices, dtype=float)
        for axis in range(3):
            offset = np.zeros(3, dtype=float)
            offset[axis] = steps[axis]
            gradients[:, axis] = (
                np.asarray(sdf(vertices + offset), dtype=float) - np.asarray(sdf(vertices - offset), dtype=float)
            ) / (2.0 * steps[axis])
        lengths = np.linalg.norm(gradients, axis=1)
        valid = lengths > 1e-12
        gradients[valid] /= lengths[valid, np.newaxis]
        gradients[~valid] = 0.0
        return gradients

    def _base_surface_dataset(self) -> Any:
        key = float(self.threshold)
        cached = self._surface_polydata_cache.get(key)
        if cached is not None:
            return cached
        mesh = self.bundle.surface_mesh(threshold=self.threshold)
        dataset = _triangle_mesh_to_polydata(mesh)
        self._surface_polydata_cache[key] = dataset
        return dataset

    def _apply_plain_surface_coloring(self, mesh: Any, dataset: Any) -> Any:
        return dataset

    def _apply_normal_surface_coloring(self, mesh: Any, dataset: Any) -> Any:
        normal_dataset = dataset.compute_normals(
            cell_normals=False,
            point_normals=True,
            split_vertices=False,
            consistent_normals=True,
            auto_orient_normals=True,
            inplace=False,
        )
        mesh_normals = np.asarray(normal_dataset.point_data["Normals"], dtype=float)
        sdf_normals = self._sdf_vertex_normals(mesh)
        if mesh_normals.shape == sdf_normals.shape and mesh_normals.size > 0:
            dots = np.sum(mesh_normals * sdf_normals, axis=1)
            if float(np.mean(dots)) < 0.0:
                mesh_normals = -mesh_normals
        normal_dataset.point_data["normal_rgb"] = normal_rgb_from_normals(mesh_normals)
        normal_dataset.set_active_scalars("normal_rgb")
        return normal_dataset

    def _apply_overhang_surface_coloring(self, mesh: Any, dataset: Any) -> Any:
        dataset.cell_data["overhang_angle_deg"] = np.asarray(
            self.result.support(build_direction=self.build_direction, threshold=self.threshold).overhang_angles,
            dtype=float,
        )
        dataset.set_active_scalars("overhang_angle_deg")
        return dataset

    def _surface_dataset(self) -> Any:
        mesh = self.bundle.surface_mesh(threshold=self.threshold)
        dataset = self._base_surface_dataset().copy(deep=True)
        if dataset.n_cells == 0:
            return dataset
        return self._surface_coloring_registry[self.color_mode](mesh, dataset)

    def _surface_color_kwargs(self, dataset: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "show_scalar_bar": False,
            "show_edges": False,
            "smooth_shading": self.color_mode != "normals",
        }
        if dataset.n_cells == 0:
            kwargs["color"] = "#93aec7"
            return kwargs
        if self.color_mode == "normals":
            kwargs.update(
                {
                    "scalars": "normal_rgb",
                    "rgb": True,
                    "preference": "point",
                    "lighting": False,
                }
            )
            return kwargs
        if self.color_mode == "overhang":
            kwargs.update(
                {
                    "scalars": "overhang_angle_deg",
                    "cmap": "viridis",
                    "clim": (0.0, 180.0),
                    "preference": "cell",
                }
            )
            return kwargs
        kwargs["color"] = "#93aec7"
        return kwargs

    def _occupancy_scalar_field(self) -> npt.NDArray[np.uint8]:
        return self.bundle.occupancy_field(threshold=self.threshold).astype(np.uint8, copy=False)

    def _density_scalar_field(self) -> npt.NDArray[np.float64]:
        density = np.asarray(self._active_density_field(), dtype=float)
        view_values = density.copy()
        maximum = float(np.max(view_values)) if view_values.size else 0.0
        if maximum > 0.0:
            floor = max(0.05 * maximum, 0.05 * self.threshold)
            view_values[view_values < floor] = 0.0
        return view_values

    def _accumulation_scalar_field(self) -> npt.NDArray[np.float64]:
        if self._density_sum is None:
            return self._density_scalar_field()
        density = np.asarray(self._density_sum, dtype=float)
        view_values = density.copy()
        maximum = float(np.max(view_values)) if view_values.size else 0.0
        if maximum > 0.0:
            floor = max(0.05 * maximum, 0.05 * self.threshold)
            view_values[view_values < floor] = 0.0
        return view_values

    def _deposition_order_scalar_field(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.result.strata(mode="order", threshold=self.threshold).label_field, dtype=float)

    def _scalar_field(self, representation: Literal["occupancy", "density"], field_name: str) -> npt.NDArray[Any]:
        try:
            producer = self._scalar_field_registry[representation][field_name]
        except KeyError as exc:
            raise ValueError(f"Unknown {representation} scalar field {field_name!r}.") from exc
        return producer()

    def _active_scalar_field_name(self, representation: Literal["occupancy", "density"]) -> ScalarFieldName:
        return self.occupancy_field_name if representation == "occupancy" else self.density_field_name

    def _set_active_scalar_field_name(self, representation: Literal["occupancy", "density"], field_name: ScalarFieldName) -> None:
        if representation == "occupancy":
            self.occupancy_field_name = field_name
        else:
            self.density_field_name = field_name

    def _available_scalar_field_labels(self, representation: Literal["occupancy", "density"]) -> dict[str, str]:
        labels = dict(self._scalar_field_labels[representation])
        if representation == "density" and self._density_sum is None:
            labels.pop("accumulation", None)
        return labels

    def _sync_scalar_field_options(self) -> None:
        is_scalar_representation = self.representation in {"occupancy", "density"}
        self.scalar_field_row.setVisible(is_scalar_representation)
        if not is_scalar_representation:
            return

        representation = self.representation
        labels = self._available_scalar_field_labels(representation)
        current = self._active_scalar_field_name(representation)
        if current not in labels:
            current = next(iter(labels))
            self._set_active_scalar_field_name(representation, current)

        self.scalar_field_combo.blockSignals(True)
        self.scalar_field_combo.clear()
        for field_name, label in labels.items():
            self.scalar_field_combo.addItem(label, field_name)
        index = self.scalar_field_combo.findData(current)
        if index >= 0:
            self.scalar_field_combo.setCurrentIndex(index)
        self.scalar_field_combo.blockSignals(False)

    def _occupancy_dataset(self) -> Any:
        occupancy = self._scalar_field("occupancy", self.occupancy_field_name)
        grid = _field_to_image_data(
            self.bundle.domain,
            occupancy,
            field_name=self.occupancy_field_name,
            association="cell",
        )
        return grid.threshold(value=0.5, scalars=self.occupancy_field_name, preference="cell")

    def _active_density_field(self) -> npt.NDArray[np.float64]:
        if self.density_field_name == "accumulation" and self._density_sum is not None:
            return self._density_sum
        return self.bundle.density_field()

    def _density_grid(self) -> Any:
        view_values = self._scalar_field("density", self.density_field_name)
        return _field_to_image_data(
            self.bundle.domain,
            view_values,
            field_name=self.density_field_name,
            association="point",
        )

    def _density_clim(self) -> tuple[float, float]:
        if self.density_field_name == "deposition_order":
            maximum = float(np.max(self._deposition_order_scalar_field()))
            return (0.0, maximum if maximum > 0.0 else 1.0)
        maximum = float(np.max(self._active_density_field()))
        upper = maximum if maximum > 0.0 else 1.0
        lower = max(self.threshold * 0.2, upper * 0.05 if upper > 0.0 else 0.0)
        return (lower, upper)

    def _density_cmap(self) -> str:
        if self.density_field_name == "deposition_order":
            return "turbo"
        return "viridis"

    def _density_scalar_bar_title(self) -> str:
        return self._available_scalar_field_labels("density")[self.density_field_name]

    def _volume_opacity(self) -> list[float]:
        alpha = float(np.clip(self.view_opacity["density"], 0.0, 1.0))
        if self.density_field_name == "deposition_order":
            return [0.0] + [alpha] * 255
        base = np.asarray([0.0, 0.0, 0.05, 0.12, 0.28, 0.55, 1.0], dtype=float)
        return np.clip(base * alpha, 0.0, 1.0).tolist()

    def _occupancy_color_kwargs(self, dataset: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "opacity": self.view_opacity["occupancy"],
            "show_edges": False,
            "show_scalar_bar": False,
            "render": False,
            "name": "occupancy_actor",
            "reset_camera": False,
        }
        if self.occupancy_field_name == "deposition_order":
            maximum = float(np.max(self._deposition_order_scalar_field()))
            kwargs.update(
                {
                    "scalars": self.occupancy_field_name,
                    "cmap": "turbo",
                    "clim": (0.0, maximum if maximum > 0.0 else 1.0),
                    "preference": "cell",
                }
            )
        else:
            kwargs["color"] = "#de6b48"
        return kwargs

    def _occupancy_scalar_bar_title(self) -> str:
        return self._available_scalar_field_labels("occupancy")[self.occupancy_field_name]

    def _threshold_slider_upper(self) -> float:
        density_max = float(np.max(self._active_density_field()))
        return max(1.0, density_max * 1.1, self.threshold)

    def _threshold_to_slider(self, value: float) -> int:
        upper = self._threshold_slider_upper()
        if upper <= 0.0:
            return 0
        return int(round(np.clip(value / upper, 0.0, 1.0) * 1000.0))

    def _slider_to_threshold(self, slider_value: int) -> float:
        return float(slider_value) / 1000.0 * self._threshold_slider_upper()

    def _sync_threshold_controls(self) -> None:
        slider_value = self._threshold_to_slider(self.threshold)
        if self.threshold_slider.value() != slider_value:
            self.threshold_slider.blockSignals(True)
            self.threshold_slider.setValue(slider_value)
            self.threshold_slider.blockSignals(False)
        if abs(self.threshold_spin.value() - self.threshold) > 1e-9:
            self.threshold_spin.blockSignals(True)
            self.threshold_spin.setValue(self.threshold)
            self.threshold_spin.blockSignals(False)

    def _current_opacity(self) -> float:
        return float(self.view_opacity[self.representation])

    def _sync_opacity_controls(self) -> None:
        value = self._current_opacity()
        slider_value = int(round(value * 100.0))
        if self.opacity_slider.value() != slider_value:
            self.opacity_slider.blockSignals(True)
            self.opacity_slider.setValue(slider_value)
            self.opacity_slider.blockSignals(False)
        if abs(self.opacity_spin.value() - value) > 1e-9:
            self.opacity_spin.blockSignals(True)
            self.opacity_spin.setValue(value)
            self.opacity_spin.blockSignals(False)

    def _remove_actor(self, actor: Any | None) -> None:
        if actor is not None:
            self.plotter.remove_actor(actor, render=False)

    def _capture_camera_state(self) -> dict[str, Any] | None:
        if self._surface_actor is None and self._occupancy_actor is None and self._density_actor is None:
            return None
        camera = self.plotter.camera
        return {
            "position": tuple(float(value) for value in camera.position),
            "focal_point": tuple(float(value) for value in camera.focal_point),
            "viewup": tuple(float(value) for value in camera.up),
            "parallel_scale": float(camera.parallel_scale),
            "parallel_projection": bool(camera.parallel_projection),
            "clipping_range": tuple(float(value) for value in camera.clipping_range),
        }

    def _restore_camera_state(self, state: dict[str, Any] | None) -> None:
        if state is None:
            return
        camera = self.plotter.camera
        camera.position = state["position"]
        camera.focal_point = state["focal_point"]
        camera.up = state["viewup"]
        camera.parallel_scale = state["parallel_scale"]
        camera.parallel_projection = state["parallel_projection"]
        camera.clipping_range = state["clipping_range"]

    def _domain_bounds_for_camera(self) -> tuple[float, float, float, float, float, float]:
        return (
            float(self.bundle.domain.min_corner[0]),
            float(self.bundle.domain.max_corner[0]),
            float(self.bundle.domain.min_corner[1]),
            float(self.bundle.domain.max_corner[1]),
            float(self.bundle.domain.min_corner[2]),
            float(self.bundle.domain.max_corner[2]),
        )

    def _occupied_bounds_for_camera(self) -> tuple[float, float, float, float, float, float]:
        key = float(self.threshold)
        cached = self._occupied_bounds_cache.get(key)
        if cached is not None:
            return cached

        occupancy = self.bundle.occupancy_field(threshold=self.threshold)
        indices = np.argwhere(occupancy)
        if indices.size == 0:
            bounds = self._domain_bounds_for_camera()
            self._occupied_bounds_cache[key] = bounds
            return bounds

        lower_index = tuple(int(value) for value in indices.min(axis=0))
        upper_index = tuple(int(value) for value in indices.max(axis=0))
        lower_point = self.bundle.domain.index_to_world(lower_index)
        upper_point = self.bundle.domain.index_to_world(upper_index)
        spacing = np.asarray(self.bundle.domain.voxel_size, dtype=float)
        bounds = (
            float(lower_point.x - 0.5 * spacing[0]),
            float(upper_point.x + 0.5 * spacing[0]),
            float(lower_point.y - 0.5 * spacing[1]),
            float(upper_point.y + 0.5 * spacing[1]),
            float(lower_point.z - 0.5 * spacing[2]),
            float(upper_point.z + 0.5 * spacing[2]),
        )
        self._occupied_bounds_cache[key] = bounds
        return bounds

    def _current_view_bounds(self) -> tuple[float, float, float, float, float, float]:
        return self._occupied_bounds_for_camera()

    def _clear_managed_scalar_bars(self) -> None:
        scalar_bars = getattr(self.plotter, "scalar_bars", None)
        if scalar_bars is None:
            return
        titles = list(scalar_bars.keys()) if hasattr(scalar_bars, "keys") else list(scalar_bars)
        for title in titles:
            try:
                self.plotter.remove_scalar_bar(title=title, render=False)
            except (AttributeError, KeyError, ValueError):
                continue

    def _mapper_from_actor(self, actor: Any | None) -> Any | None:
        if actor is None:
            return None
        mapper = getattr(actor, "mapper", None)
        if mapper is not None:
            return mapper
        if hasattr(actor, "GetMapper"):
            return actor.GetMapper()
        return None

    def _sync_scalar_bars(self) -> None:
        self._clear_managed_scalar_bars()
        if self.representation == "density" and self.density_field_name != "deposition_order":
            mapper = self._mapper_from_actor(self._density_actor)
            if mapper is not None:
                self.plotter.add_scalar_bar(
                    title=self._density_scalar_bar_title(),
                    mapper=mapper,
                    title_font_size=12,
                    label_font_size=10,
                    width=0.22,
                    height=0.1,
                    position_x=0.73,
                    position_y=0.03,
                    render=False,
                )
            return
        if self.representation == "surface" and self.color_mode == "overhang":
            actor = self._clip_actor if self.clip_enabled else self._surface_actor
            mapper = self._mapper_from_actor(actor)
            if mapper is not None:
                self.plotter.add_scalar_bar(
                    title="Overhang (deg)",
                    mapper=mapper,
                    title_font_size=12,
                    label_font_size=10,
                    width=0.22,
                    height=0.1,
                    position_x=0.73,
                    position_y=0.03,
                    render=False,
                )

    def _rebuild_surface_actor(self) -> None:
        self._remove_actor(self._surface_actor)
        dataset = self._surface_dataset()
        if dataset.n_cells == 0:
            self._surface_actor = None
            return
        kwargs = self._surface_color_kwargs(dataset)
        kwargs.update({"render": False, "name": "surface_actor", "reset_camera": False, "opacity": self.view_opacity["surface"]})
        self._surface_actor = self.plotter.add_mesh(dataset, **kwargs)

    def _rebuild_occupancy_actor(self) -> None:
        self._remove_actor(self._occupancy_actor)
        dataset = self._occupancy_dataset()
        if dataset.n_cells == 0:
            self._occupancy_actor = None
            return
        self._occupancy_actor = self.plotter.add_mesh(dataset, **self._occupancy_color_kwargs(dataset))

    def _rebuild_density_actor(self) -> None:
        self._remove_actor(self._density_actor)
        volume_kwargs: dict[str, Any] = {
            "scalars": self.density_field_name,
            "cmap": self._density_cmap(),
            "clim": self._density_clim(),
            "show_scalar_bar": False,
            "render": False,
            "name": "density_actor",
            "reset_camera": False,
            "shade": False,
        }
        if self.density_field_name == "deposition_order":
            volume_kwargs.update(
                {
                    "opacity": "foreground",
                    "categories": True,
                }
            )
        else:
            volume_kwargs["opacity"] = self._volume_opacity()
        self._density_actor = self.plotter.add_volume(
            self._density_grid(),
            **volume_kwargs,
        )

    def _activate_overlay(self, name: str) -> None:
        handler = self._overlay_registry.get(name, {}).get("activate")
        if callable(handler):
            handler()

    def _clear_overlay(self, name: str) -> None:
        handler = self._overlay_registry.get(name, {}).get("clear")
        if callable(handler):
            handler()

    def _clear_clip_state(self) -> None:
        self.plotter.clear_plane_widgets()
        if self._clip_actor is not None:
            self.plotter.remove_actor(self._clip_actor, render=False)
            self._clip_actor = None
        self._clip_widget = None

    def _apply_visibility(self) -> None:
        if self._surface_actor is not None:
            self._surface_actor.SetVisibility(self.representation == "surface" and not self.clip_enabled)
        if self._occupancy_actor is not None:
            self._occupancy_actor.SetVisibility(self.representation == "occupancy" and not self.clip_enabled)
        if self._density_actor is not None:
            self._density_actor.SetVisibility(self.representation == "density")
        if self._clip_actor is not None:
            self._clip_actor.SetVisibility(self.clip_enabled and self.representation in {"surface", "occupancy"})

    def _activate_clip_widget(self) -> None:
        self._clear_clip_state()
        if self.representation == "surface":
            dataset = self._surface_dataset()
            if dataset.n_cells > 0:
                kwargs = self._surface_color_kwargs(dataset)
                clipped = dataset.clip(
                    normal=(1.0, 0.0, 0.0),
                    origin=dataset.center,
                    invert=False,
                )
                kwargs["show_scalar_bar"] = False
                kwargs.update({"render": False, "name": "surface_clip_actor", "reset_camera": False})
                self._clip_actor = self.plotter.add_mesh(clipped, **kwargs)
                self._clip_widget = self.plotter.add_plane_widget(
                    callback=lambda normal, origin: self._update_surface_clip(dataset, normal, origin),
                    normal="x",
                    origin=dataset.center,
                    interaction_event="always",
                )
        elif self.representation == "occupancy":
            dataset = self._occupancy_dataset()
            if dataset.n_cells > 0:
                self._clip_actor = self.plotter.add_mesh_clip_plane(
                    dataset,
                    interaction_event="end",
                    color="#de6b48",
                    opacity=self.view_opacity["occupancy"],
                    show_scalar_bar=False,
                    reset_camera=False,
                )
        elif self.representation == "density" and self._density_actor is not None:
            self._clip_widget = self.plotter.add_volume_clip_plane(
                self._density_actor,
                interaction_event="always",
            )
        self._apply_visibility()
        self._sync_scalar_bars()

    def _update_surface_clip(
        self,
        dataset: Any,
        normal: tuple[float, float, float] | npt.ArrayLike,
        origin: tuple[float, float, float] | npt.ArrayLike,
    ) -> None:
        if self._clip_actor is None:
            return
        clipped = dataset.clip(normal=normal, origin=origin, invert=False)
        mapper = self._mapper_from_actor(self._clip_actor)
        if mapper is not None:
            mapper.SetInputData(clipped)
        self._sync_scalar_bars()
        self.plotter.render()

    def _refresh_roi_stats(self) -> None:
        if not self.roi_enabled or self._roi_bounds is None:
            return
        stats = self.bundle.subvolume_stats(self._roi_bounds, threshold=self.threshold)
        self._last_roi_stats = stats
        self.roi_status.setPlainText(self._format_roi_text(stats))

    def _rebuild_scene(self) -> None:
        camera_state = self._capture_camera_state()
        self._clear_clip_state()
        self._rebuild_surface_actor()
        self._rebuild_occupancy_actor()
        self._rebuild_density_actor()
        if self.clip_enabled:
            self._activate_clip_widget()
        else:
            self._apply_visibility()
            self._sync_scalar_bars()
        self._sync_status_controls()
        self._restore_camera_state(camera_state)
        self.plotter.render()

    def _sync_surface_controls(self) -> None:
        surface_mode = self.representation == "surface"
        overhang_mode = surface_mode and self.color_mode == "overhang"
        scalar_mode = self.representation in {"occupancy", "density"}
        self.color_mode_row.setVisible(surface_mode)
        self.surface_box.setVisible(overhang_mode)
        self.build_direction_row.setVisible(overhang_mode)
        self.color_mode_combo.setEnabled(surface_mode)
        self.build_direction_combo.setEnabled(overhang_mode)
        self.scalar_field_row.setVisible(scalar_mode)

    def _set_row_visible(self, label_widget: QtWidgets.QWidget | None, field_widget: QtWidgets.QWidget, visible: bool) -> None:
        if label_widget is not None:
            label_widget.setVisible(visible)
        field_widget.setVisible(visible)

    def _on_opacity_slider_changed(self, value: int) -> None:
        opacity_value = value / 100.0
        if abs(self.opacity_spin.value() - opacity_value) > 1e-9:
            self.opacity_spin.blockSignals(True)
            self.opacity_spin.setValue(opacity_value)
            self.opacity_spin.blockSignals(False)
        self.set_opacity(opacity_value)

    def _on_opacity_spin_changed(self, value: float) -> None:
        slider_value = int(round(float(value) * 100.0))
        if self.opacity_slider.value() != slider_value:
            self.opacity_slider.blockSignals(True)
            self.opacity_slider.setValue(slider_value)
            self.opacity_slider.blockSignals(False)
        self.set_opacity(float(value))

    def _on_threshold_slider_changed(self, value: int) -> None:
        threshold_value = self._slider_to_threshold(value)
        if abs(self.threshold_spin.value() - threshold_value) > 1e-9:
            self.threshold_spin.blockSignals(True)
            self.threshold_spin.setValue(threshold_value)
            self.threshold_spin.blockSignals(False)
        self.set_threshold(threshold_value)

    def _on_threshold_spin_changed(self, value: float) -> None:
        slider_value = self._threshold_to_slider(float(value))
        if self.threshold_slider.value() != slider_value:
            self.threshold_slider.blockSignals(True)
            self.threshold_slider.setValue(slider_value)
            self.threshold_slider.blockSignals(False)
        self.set_threshold(float(value))

    def _sync_status_controls(self) -> None:
        pick_visible = self.point_picking_enabled
        roi_visible = self.roi_enabled
        support_visible = self._support_is_active()
        self.clear_pick_button.setVisible(pick_visible)
        self._set_row_visible(self._pick_status_label_widget, self.pick_status, pick_visible)
        self.reset_roi_button.setVisible(roi_visible)
        self._set_row_visible(self._roi_status_label_widget, self.roi_status, roi_visible)
        self._set_row_visible(self._support_status_label_widget, self.support_status, support_visible)
        if support_visible:
            self._refresh_support_status()
        else:
            self.support_status.clear()

    def _initialize_camera(self) -> None:
        self.apply_camera_preset("perspective")

    def apply_camera_preset(self, preset: Literal["perspective", "top", "front", "left"]) -> None:
        bounds = self._current_view_bounds()
        center = np.asarray(
            [
                0.5 * (bounds[0] + bounds[1]),
                0.5 * (bounds[2] + bounds[3]),
                0.5 * (bounds[4] + bounds[5]),
            ],
            dtype=float,
        )
        if preset == "perspective":
            direction = np.asarray((-1.0, -1.0, 1.0), dtype=float)
            direction /= np.linalg.norm(direction)
            up = np.asarray((0.0, 0.0, 1.0), dtype=float)
            position = center + direction
        elif preset == "top":
            up = np.asarray((0.0, 1.0, 0.0), dtype=float)
            position = center + np.asarray((0.0, 0.0, 1.0), dtype=float)
        elif preset == "front":
            up = np.asarray((0.0, 0.0, 1.0), dtype=float)
            position = center - np.asarray((0.0, 1.0, 0.0), dtype=float)
        elif preset == "left":
            up = np.asarray((0.0, 0.0, 1.0), dtype=float)
            position = center - np.asarray((1.0, 0.0, 0.0), dtype=float)
        else:
            raise ValueError("preset must be 'perspective', 'top', 'front', or 'left'.")
        camera = self.plotter.camera
        camera.position = tuple(float(value) for value in position)
        camera.focal_point = tuple(float(value) for value in center)
        camera.up = tuple(float(value) for value in up)
        self.plotter.reset_camera(bounds=bounds, render=False)
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def _format_pick_text(self, payload: dict[str, Any]) -> str:
        point = payload["point"]
        normal = payload["surface_normal"]
        lines = [
            f"Point: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})\n"
            f"Representation: {payload['representation']}\n",
        ]
        lines.extend(
            [
                f"Voxel Index: {payload['voxel_index']}\n",
                f"Occupied: {payload['occupied']}\n",
                f"Density: {payload['density']:.3f}\n",
                f"Dep. Index: {payload['deposition_index']:.3f}\n",
                f"Signed Distance: {payload['signed_distance']:.3f}\n",
                f"Normal: ({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})",
            ]
        )
        return "".join(lines)

    def _format_roi_text(self, stats: dict[str, float]) -> str:
        return (
            f"Occupied Fraction: {stats['occupied_fraction']:.3f}\n"
            f"Density Max / Mean: {stats['density_max']:.3f} / {stats['density_mean']:.3f}\n"
            f"Dep. Index Max / Mean: {stats['deposition_index_max']:.3f} / {stats['deposition_index_mean']:.3f}\n"
            f"Mesh Area: {stats['mesh_area']:.3f}\n"
            f"Voxel Count: {int(stats['voxel_count'])}"
        )

    def _support_is_active(self) -> bool:
        return self.representation == "surface" and self.color_mode == "overhang"

    def _format_support_text(self, stats: Any) -> str:
        return (
            f"Build Dir.: {self._build_direction_label()}\n"
            f"Downfacing Area: {stats.downfacing_area:.3f}\n"
            f"Risk Area: {stats.risk_area:.3f}\n"
            f"Shadow Volume: {stats.shadow_volume:.3f}\n"
            f"Shadow Voxels: {stats.shadow_voxel_count}\n"
            f"Max Unsupported Span: {stats.max_unsupported_span:.3f}"
        )

    def _refresh_support_status(self) -> None:
        stats = self.result.support(build_direction=self.build_direction, threshold=self.threshold)
        self.support_status.setPlainText(self._format_support_text(stats))

    def _marker_radius(self) -> float:
        return 0.5 * min(self.bundle.domain.voxel_size)

    def _handle_picked_point(self, point: tuple[float, float, float] | npt.ArrayLike) -> dict[str, Any]:
        coordinates = tuple(float(value) for value in np.asarray(point, dtype=float).reshape(3))
        domain = self.bundle.domain
        payload = {
            "point": coordinates,
            "representation": self.representation,
            "voxel_index": domain.world_to_index(coordinates, clip=True) if domain.contains_point(coordinates) else None,
            "occupied": self.bundle.contains_point(coordinates, representation="occupancy", threshold=self.threshold),
            "density": self.bundle.sample_density_at(coordinates, interpolation="trilinear"),
            "deposition_index": self.bundle.sample_deposition_index_at(coordinates, interpolation="trilinear"),
            "signed_distance": self.bundle.signed_distance_at(coordinates, threshold=self.threshold),
            "surface_normal": self.bundle.surface_normal_at(coordinates, threshold=self.threshold),
        }
        self._last_pick_payload = payload
        self.pick_status.setPlainText(self._format_pick_text(payload))
        camera_state = self._capture_camera_state()
        if self._pick_marker_actor is not None:
            self.plotter.remove_actor(self._pick_marker_actor, render=False)
        marker = pv.Sphere(radius=self._marker_radius(), center=coordinates)
        self._pick_marker_actor = self.plotter.add_mesh(
            marker,
            color="#d64292",
            show_scalar_bar=False,
            render=False,
            name="picked_point",
            reset_camera=False,
        )
        self._restore_camera_state(camera_state)
        self.plotter.render()
        return payload

    def _handle_roi_box(self, box: Any) -> dict[str, float]:
        bounds = getattr(box, "bounds", box)
        minimum = (float(bounds[0]), float(bounds[2]), float(bounds[4]))
        maximum = (float(bounds[1]), float(bounds[3]), float(bounds[5]))
        self._roi_bounds = (minimum, maximum)
        stats = self.bundle.subvolume_stats(self._roi_bounds, threshold=self.threshold)
        self._last_roi_stats = stats
        self.roi_status.setPlainText(self._format_roi_text(stats))
        self.plotter.render()
        return stats

    def set_representation(self, representation: Representation) -> None:
        if representation not in {"surface", "occupancy", "density"}:
            raise ValueError("representation must be 'surface', 'occupancy', or 'density'.")
        camera_state = self._capture_camera_state()
        self.representation = representation
        self._set_combo_current_data(self.view_mode_combo, representation)
        self._sync_scalar_field_options()
        self._sync_opacity_controls()
        self._sync_surface_controls()
        self._sync_status_controls()
        if self.clip_enabled:
            self._activate_clip_widget()
        else:
            self._apply_visibility()
            self._sync_scalar_bars()
        if self.point_picking_enabled:
            self._install_point_picking()
        self._restore_camera_state(camera_state)
        self.plotter.render()

    def set_threshold(self, value: float) -> None:
        value = float(value)
        if abs(self.threshold - value) <= 1e-9:
            self._sync_threshold_controls()
            return
        previous_threshold = float(self.threshold)
        self.threshold = value
        self._surface_polydata_cache.pop(previous_threshold, None)
        self._occupied_bounds_cache.pop(previous_threshold, None)
        self._rebuild_scene()
        self._refresh_roi_stats()
        self._sync_threshold_controls()

    def set_opacity(self, value: float) -> None:
        value = float(np.clip(value, 0.0, 1.0))
        if abs(self.view_opacity[self.representation] - value) <= 1e-9:
            self._sync_opacity_controls()
            return
        self.view_opacity[self.representation] = value
        self._rebuild_scene()
        self._sync_opacity_controls()

    def set_scalar_field(self, field_name: ScalarFieldName | None) -> None:
        if self.representation not in {"occupancy", "density"} or field_name is None:
            return
        if field_name not in self._available_scalar_field_labels(self.representation):
            raise ValueError(f"Unknown {self.representation} scalar field {field_name!r}.")
        current = self._active_scalar_field_name(self.representation)
        if current == field_name:
            self._sync_scalar_field_options()
            self._sync_surface_controls()
            self._sync_status_controls()
            return
        self._set_active_scalar_field_name(self.representation, field_name)
        self._sync_scalar_field_options()
        self._sync_surface_controls()
        self._sync_status_controls()
        self._rebuild_scene()

    def set_color_mode(self, value: ColorMode) -> None:
        if value not in {"plain", "normals", "overhang"}:
            raise ValueError("color_mode must be 'plain', 'normals', or 'overhang'.")
        camera_state = self._capture_camera_state()
        self.color_mode = value
        self._set_combo_current_data(self.color_mode_combo, value)
        self._sync_surface_controls()
        self._sync_status_controls()
        self._rebuild_surface_actor()
        if self.clip_enabled and self.representation == "surface":
            self._activate_clip_widget()
        else:
            self._apply_visibility()
            self._sync_scalar_bars()
        self._restore_camera_state(camera_state)
        self.plotter.render()

    def set_build_direction(self, axis_or_vector: str | tuple[float, float, float] | npt.ArrayLike) -> None:
        camera_state = self._capture_camera_state()
        self.build_direction = self._coerce_build_direction(axis_or_vector)
        self.build_direction_combo.setCurrentText(self._build_direction_label())
        self._sync_status_controls()
        self._rebuild_scene()
        self._restore_camera_state(camera_state)
        self.plotter.render()

    def clear_pick(self) -> None:
        self._last_pick_payload = None
        self.pick_status.clear()
        if self._pick_marker_actor is not None:
            camera_state = self._capture_camera_state()
            self.plotter.remove_actor(self._pick_marker_actor, render=False)
            self._pick_marker_actor = None
            self._restore_camera_state(camera_state)
            self.plotter.render()

    def _install_point_picking(self) -> None:
        disable_picking = getattr(self.plotter, "disable_picking", None)
        if callable(disable_picking):
            disable_picking()
        if self.representation == "surface":
            self.plotter.enable_surface_point_picking(
                callback=self._handle_picked_point,
                left_clicking=True,
                show_message=False,
                show_point=False,
            )
            return

        picker = "cell" if self.representation == "occupancy" else "volume"
        self.plotter.enable_point_picking(
            callback=self._handle_non_surface_picked_point,
            left_clicking=True,
            show_message=False,
            show_point=False,
            use_picker=True,
            picker=picker,
            pickable_window=self.representation == "density",
        )

    def _handle_non_surface_picked_point(self, point: npt.ArrayLike, picker: Any) -> dict[str, Any] | None:
        coordinates = tuple(float(value) for value in np.asarray(point, dtype=float).reshape(3))
        if self.representation == "density" and not self.bundle.domain.contains_point(coordinates):
            self.clear_pick()
            return None
        if self.representation == "occupancy" and hasattr(picker, "GetDataSet") and picker.GetDataSet() is None:
            self.clear_pick()
            return None
        return self._handle_picked_point(coordinates)

    def set_point_picking_enabled(self, value: bool) -> None:
        self.point_picking_enabled = bool(value)
        if self.point_picking_enabled:
            self._activate_overlay("point_picking")
        else:
            disable_picking = getattr(self.plotter, "disable_picking", None)
            if callable(disable_picking):
                disable_picking()
            self._clear_overlay("point_picking")
        self._sync_status_controls()
        self.plotter.render()

    def set_clip_enabled(self, value: bool) -> None:
        self.clip_enabled = bool(value)
        if not self.clip_enabled:
            self._rebuild_scene()
            return
        self._activate_overlay("clip")
        self.plotter.render()

    def set_roi_enabled(self, value: bool) -> None:
        self.roi_enabled = bool(value)
        self.plotter.clear_box_widgets()
        self._roi_widget = None
        if self.roi_enabled:
            self._roi_widget = self.plotter.add_box_widget(
                callback=self._handle_roi_box,
                bounds=self._domain_bounds(),
                rotation_enabled=False,
                interaction_event="end",
            )
        else:
            self._roi_bounds = None
            self._last_roi_stats = None
            self.roi_status.clear()
        self._sync_status_controls()
        self.plotter.render()

    def reset_roi(self) -> None:
        if not self.roi_enabled:
            return
        self.plotter.clear_box_widgets()
        self._roi_widget = self.plotter.add_box_widget(
            callback=self._handle_roi_box,
            bounds=self._domain_bounds(),
            rotation_enabled=False,
            interaction_event="end",
        )
        self._roi_bounds = None
        self._last_roi_stats = None
        self.roi_status.clear()
        self._sync_status_controls()
        self.plotter.render()

    def show(self) -> None:
        super().show()
        self.plotter.render()

    def closeEvent(self, event: Any) -> None:
        self._clear_clip_state()
        self.plotter.clear_box_widgets()
        self.plotter.close()
        super().closeEvent(event)
