from __future__ import annotations

import os

import numpy as np
import pytest

if os.environ.get("DDS_RUN_VIZ_TESTS") != "1":
    pytest.skip("Set DDS_RUN_VIZ_TESTS=1 to run PyVistaQt workbench tests.", allow_module_level=True)

pyvistaqt = pytest.importorskip("pyvistaqt")
pytest.importorskip("PySide6")

from dds import (  # noqa: E402
    BeadProfile,
    Domain,
    LineDeposit,
    PointDeposit,
    Simulator,
)
from dds.viz import ViewConfig  # noqa: E402
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
    deposits = [
        PointDeposit(target=(2.25, 2.25, 0.65), profile=profile),
        LineDeposit(start=(2.25, 2.25, 0.65), end=(6.25, 2.25, 0.65), profile=profile),
        LineDeposit(start=(6.25, 2.25, 0.65), end=(6.25, 6.25, 0.65), profile=profile),
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

    workbench.set_representation("implicit")
    assert not workbench.surface_box.isVisible()
    workbench.set_point_picking_enabled(True)
    implicit_payload = workbench._handle_non_surface_picked_point((2.3, 2.3, 0.3), picker=object())
    assert implicit_payload is not None
    assert implicit_payload["representation"] == "implicit"

    workbench.set_representation("surface")
    payload = workbench._handle_picked_point((2.3, 2.3, 0.3))

    assert payload["representation"] == "surface"
    assert payload["voxel_index"] is not None

    workbench.clear_pick()
    assert workbench._last_pick_payload is None
    assert workbench.pick_status.toPlainText() == ""

    workbench.close()


def test_scalar_field_options_match_representation(qtbot: object) -> None:
    simulator = make_simulator()
    result = simulator.result(include_coverage=True)
    workbench = SimulationWorkbench(result, off_screen=True)
    qtbot.addWidget(workbench)

    workbench.set_representation("occupancy")
    occupancy_labels = [workbench.scalar_field_combo.itemText(i) for i in range(workbench.scalar_field_combo.count())]
    assert occupancy_labels == ["Occupancy", "Deposition Order"]

    workbench.set_representation("implicit")
    implicit_labels = [workbench.scalar_field_combo.itemText(i) for i in range(workbench.scalar_field_combo.count())]
    assert implicit_labels == ["Implicit", "Coverage (nonphysical)", "Deposition Order"]

    workbench.close()


def test_implicit_field_switch_changes_active_implicit_field(qtbot: object) -> None:
    simulator = make_simulator()
    result = simulator.result(include_coverage=True)
    coverage = result.coverage
    assert coverage is not None
    workbench = SimulationWorkbench(result, off_screen=True)
    qtbot.addWidget(workbench)

    workbench.set_representation("implicit")
    workbench.set_scalar_field("implicit")
    max_clim = workbench._implicit_clim()
    workbench.set_scalar_field("coverage")
    coverage_clim = workbench._implicit_clim()
    np.testing.assert_allclose(workbench._active_implicit_field(), coverage)
    assert coverage_clim[1] > max_clim[1]

    workbench.close()


def test_deposition_order_view_uses_cached_order_field(
    qtbot: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = make_simulator().result(include_coverage=True)
    workbench = SimulationWorkbench(result, off_screen=True)
    qtbot.addWidget(workbench)
    expected = result.analysis.deposition_order_field(threshold=workbench.threshold)

    def fail_strata(*args: object, **kwargs: object) -> None:
        raise AssertionError("deposition-order visualization must not build strata")

    monkeypatch.setattr(type(result.analysis), "strata", fail_strata)

    np.testing.assert_array_equal(workbench._deposition_order_scalar_field(), expected)
    workbench.close()


def test_initial_view_config_applies_without_example_side_mutation(qtbot: object) -> None:
    simulator = make_simulator()
    result = simulator.result(include_coverage=True)
    workbench = SimulationWorkbench(
        result,
        initial_view=ViewConfig(
            view_mode="implicit",
            scalar_field="coverage",
            build_direction="+Y",
            show_toolpath=True,
            show_targets=True,
            show_world_axes=True,
        ),
        off_screen=True,
    )
    qtbot.addWidget(workbench)

    assert workbench.representation == "implicit"
    assert workbench.implicit_field_name == "coverage"
    assert workbench.view_mode_combo.currentData() == "implicit"
    assert workbench.scalar_field_combo.currentData() == "coverage"
    assert workbench.build_direction_combo.currentData() == "+Y"
    assert workbench.view_opacity["implicit"] == pytest.approx(1.0)
    assert workbench._implicit_actor is not None
    assert workbench._surface_actor is None
    assert workbench._occupancy_actor is None
    assert workbench.viewer.get("workbench_toolpath").visible
    assert workbench.viewer.get("workbench_targets").visible
    assert workbench.viewer.get("workbench_world_axes").visible

    workbench.close()


def test_dirty_views_rebuild_only_when_active(
    qtbot: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workbench = SimulationWorkbench(make_simulator(), off_screen=True)
    qtbot.addWidget(workbench)
    rebuild_counts = {"surface": 0, "occupancy": 0, "implicit": 0}

    for representation in rebuild_counts:
        method_name = f"_rebuild_{representation}_actor"
        original = getattr(workbench, method_name)

        def tracked(
            original: object = original,
            representation: str = representation,
        ) -> None:
            rebuild_counts[representation] += 1
            original()

        monkeypatch.setattr(workbench, method_name, tracked)

    workbench.set_threshold(0.6)

    assert rebuild_counts == {"surface": 1, "occupancy": 0, "implicit": 0}
    assert workbench._dirty_representations == {"occupancy", "implicit"}

    workbench.set_representation("occupancy")
    assert rebuild_counts == {"surface": 1, "occupancy": 1, "implicit": 0}

    workbench.set_representation("surface")
    assert rebuild_counts == {"surface": 1, "occupancy": 1, "implicit": 0}

    workbench.set_color_mode("normals")
    assert rebuild_counts == {"surface": 2, "occupancy": 1, "implicit": 0}

    workbench.set_representation("implicit")
    assert rebuild_counts == {"surface": 2, "occupancy": 1, "implicit": 1}

    workbench.close()


def test_inactive_surface_changes_remain_lazy(
    qtbot: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workbench = SimulationWorkbench(make_simulator(), off_screen=True)
    qtbot.addWidget(workbench)
    workbench.set_representation("occupancy")
    rebuild_count = 0
    original = workbench._rebuild_surface_actor

    def tracked() -> None:
        nonlocal rebuild_count
        rebuild_count += 1
        original()

    monkeypatch.setattr(workbench, "_rebuild_surface_actor", tracked)

    workbench.set_color_mode("overhang")
    workbench.set_build_direction("+Y")

    assert rebuild_count == 0
    assert "surface" in workbench._dirty_representations

    workbench.set_representation("surface")
    assert rebuild_count == 1

    workbench.close()


def test_refresh_updates_live_simulator_state(
    qtbot: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulator = Simulator(make_domain())
    workbench = SimulationWorkbench(simulator, off_screen=True)
    qtbot.addWidget(workbench)

    def fail_inactive_rebuild() -> None:
        raise AssertionError("refresh must not rebuild inactive actors")

    monkeypatch.setattr(
        workbench,
        "_rebuild_occupancy_actor",
        fail_inactive_rebuild,
    )
    monkeypatch.setattr(
        workbench,
        "_rebuild_implicit_actor",
        fail_inactive_rebuild,
    )
    simulator.add_deposit(
        PointDeposit(
            target=(2.25, 2.25, 0.65),
            profile=BeadProfile(width=1.2, height=0.8),
        )
    )
    workbench.refresh(simulator)

    assert len(workbench.result.deposits) == 1
    assert int(workbench.bundle.occupancy(threshold=0.5).sum()) > 0
    assert workbench._dirty_representations == {"occupancy", "implicit"}

    workbench.close()


def test_overlay_controls_refresh_named_visuals(qtbot: object) -> None:
    simulator = make_simulator()
    workbench = SimulationWorkbench(simulator, off_screen=True)
    qtbot.addWidget(workbench)

    workbench.set_toolpath_visible(True)
    workbench.set_targets_visible(True)
    workbench.set_world_axes_visible(True)

    assert workbench.viewer.get("workbench_toolpath").source == tuple(
        simulator.deposits
    )
    assert workbench.viewer.get("workbench_targets").source == tuple(
        simulator.deposits
    )
    assert workbench.viewer.get("workbench_world_axes").visible

    simulator.add_deposit(
        PointDeposit(
            target=(8.0, 8.0, 0.65),
            profile=BeadProfile(width=1.2, height=0.8),
        )
    )
    workbench.refresh(simulator)
    assert len(workbench.viewer.get("workbench_toolpath").source) == len(
        simulator.deposits
    )

    workbench.set_targets_visible(False)
    with pytest.raises(KeyError):
        workbench.viewer.get("workbench_targets")

    workbench.close()


def test_workbench_screenshot_smoke(qtbot: object) -> None:
    workbench = SimulationWorkbench(make_simulator(), off_screen=True)
    qtbot.addWidget(workbench)

    image = workbench.plotter.screenshot(return_img=True)

    assert image is not None
    assert image.ndim == 3
    assert image.shape[2] in {3, 4}
    workbench.close()
