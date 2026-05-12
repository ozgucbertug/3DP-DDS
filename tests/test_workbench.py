from __future__ import annotations

import os
import numpy as np
import pytest

if os.environ.get("DDS_RUN_VIZ_TESTS") != "1":
    pytest.skip("Set DDS_RUN_VIZ_TESTS=1 to run PyVistaQt workbench tests.", allow_module_level=True)

pyvistaqt = pytest.importorskip("pyvistaqt")
pytest.importorskip("PySide6")

from dds import BeadProfile, DepositionMetadata, Domain, LineDeposit, PointDeposit, Simulator  # noqa: E402
from dds.workbench import SimulationWorkbench  # noqa: E402


def make_domain() -> Domain:
    return Domain.from_bounds(
        xmin=0.0,
        xmax=10.0,
        ymin=0.0,
        ymax=10.0,
        zmin=-1.0,
        zmax=4.0,
        voxel_size=0.5,
    )


def make_simulator() -> Simulator:
    profile = BeadProfile(width=1.2, height=0.8)
    metadata = DepositionMetadata(layer_id=0)
    deposits = [
        PointDeposit(x=2.25, y=2.25, z=0.65, profile=profile, metadata=metadata),
        LineDeposit(start=(2.25, 2.25, 0.65), end=(6.25, 2.25, 0.65), profile=profile, metadata=metadata),
        LineDeposit(start=(6.25, 2.25, 0.65), end=(6.25, 6.25, 0.65), profile=profile, metadata=metadata),
    ]
    return Simulator(make_domain(), deposits)


def camera_signature(workbench: SimulationWorkbench) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    camera = workbench.plotter.camera
    return (
        tuple(float(value) for value in camera.position),
        tuple(float(value) for value in camera.focal_point),
        tuple(float(value) for value in camera.up),
    )


def test_surface_modes_share_same_geometry_counts(qtbot: object) -> None:
    workbench = SimulationWorkbench(make_simulator(), off_screen=True)
    qtbot.addWidget(workbench)

    workbench.set_color_mode("plain")
    plain_dataset = workbench._surface_dataset()
    workbench.set_color_mode("normals")
    normals_dataset = workbench._surface_dataset()
    workbench.set_color_mode("overhang")
    overhang_dataset = workbench._surface_dataset()

    assert plain_dataset.n_points == normals_dataset.n_points == overhang_dataset.n_points
    assert plain_dataset.n_cells == normals_dataset.n_cells == overhang_dataset.n_cells
    assert "normal_rgb" in normals_dataset.point_data
    assert "overhang_angle_deg" in overhang_dataset.cell_data

    workbench.close()


def test_normal_rgb_uses_blue_for_upward_faces(qtbot: object) -> None:
    workbench = SimulationWorkbench(make_simulator(), off_screen=True)
    qtbot.addWidget(workbench)

    workbench.apply_camera_preset("top")
    workbench.set_color_mode("normals")
    dataset = workbench._surface_dataset()

    top_index = int(np.argmax(np.asarray(dataset.points)[:, 2]))
    rgb = np.asarray(dataset.point_data["normal_rgb"][top_index], dtype=np.uint8)
    assert int(rgb[2]) >= int(rgb[0])
    assert int(rgb[2]) >= int(rgb[1])

    workbench.close()


def test_overhang_legend_uses_viridis_and_camera_is_stable(qtbot: object) -> None:
    workbench = SimulationWorkbench(make_simulator(), off_screen=True)
    qtbot.addWidget(workbench)

    workbench.set_color_mode("overhang")
    before = camera_signature(workbench)
    workbench.set_opacity(0.3)
    after_opacity = camera_signature(workbench)
    workbench._handle_picked_point((2.25, 2.25, 0.25))
    after_pick = camera_signature(workbench)

    assert before == after_opacity == after_pick
    assert "Overhang (deg)" in workbench.plotter.scalar_bars

    workbench.close()


def test_camera_presets_are_deterministic(qtbot: object) -> None:
    workbench = SimulationWorkbench(make_simulator(), off_screen=True)
    qtbot.addWidget(workbench)

    workbench.apply_camera_preset("perspective")
    perspective = camera_signature(workbench)
    workbench.apply_camera_preset("left")
    left = camera_signature(workbench)
    workbench.plotter.camera.position = tuple(4.0 * np.asarray(workbench.plotter.camera.position, dtype=float))
    workbench.apply_camera_preset("left")
    left_again = camera_signature(workbench)
    workbench.apply_camera_preset("front")
    front = camera_signature(workbench)

    assert perspective != left
    assert front != left
    assert left_again == left

    workbench.close()


def test_mode_specific_controls_and_pick_payload(qtbot: object) -> None:
    workbench = SimulationWorkbench(make_simulator(), off_screen=True)
    qtbot.addWidget(workbench)

    workbench.set_color_mode("normals")
    assert workbench.surface_box.isVisible()
    assert not workbench.build_direction_row.isVisible()

    workbench.set_color_mode("overhang")
    assert workbench.build_direction_combo.isVisible()

    workbench.set_representation("density")
    assert not workbench.surface_box.isVisible()
    workbench.set_point_picking_enabled(True)
    density_payload = workbench._handle_non_surface_picked_point((2.3, 2.3, 0.3), picker=object())
    assert density_payload is not None
    assert density_payload["representation"] == "density"

    workbench.set_representation("surface")
    payload = workbench._handle_picked_point((2.3, 2.3, 0.3))

    assert payload["representation"] == "surface"
    assert payload["voxel_index"] is not None

    workbench.clear_pick()
    assert workbench._last_pick_payload is None
    assert workbench.pick_status.toPlainText() == ""

    workbench.close()


def test_density_composition_control_is_density_only(qtbot: object) -> None:
    simulator = make_simulator()
    result = simulator.result(compositions=("max", "sum"))
    workbench = SimulationWorkbench(result, off_screen=True)
    qtbot.addWidget(workbench)

    workbench.set_representation("surface")
    assert not workbench.density_composition_row.isVisible()

    workbench.set_representation("occupancy")
    assert not workbench.density_composition_row.isVisible()

    workbench.set_representation("density")
    assert workbench.density_composition_row.isVisible()

    workbench.close()


def test_density_composition_switch_changes_active_density_field(qtbot: object) -> None:
    simulator = make_simulator()
    result = simulator.result(compositions=("max", "sum"))
    density_max = result.density_max
    density_sum = result.density_sum
    assert density_sum is not None
    workbench = SimulationWorkbench(result, off_screen=True)
    qtbot.addWidget(workbench)

    workbench.set_representation("density")
    workbench.set_density_composition("max")
    max_clim = workbench._density_clim()
    workbench.set_density_composition("sum")
    sum_clim = workbench._density_clim()

    np.testing.assert_allclose(workbench._active_density_field(), density_sum)
    assert sum_clim[1] > max_clim[1]

    workbench.close()
