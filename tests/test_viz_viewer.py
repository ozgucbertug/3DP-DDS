from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")
pytest.importorskip("PySide6")

import pyvista as pv  # noqa: E402

from dds import (  # noqa: E402
    BeadProfile,
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
from dds.geometry import PointCloud, TriangleMesh  # noqa: E402
from dds.viz import (  # noqa: E402
    DepositStyle,
    FrameStyle,
    LineStyle,
    MeshStyle,
    PointCloudStyle,
    PointStyle,
    TargetStyle,
)
from dds.viz.converters import (  # noqa: E402
    line_to_polydata,
    point_cloud_to_polydata,
    points_to_polydata,
    polyline_to_polydata,
    triangle_mesh_to_polydata,
)
from dds.viz.viewer import Viewer  # noqa: E402


class FakeActor:
    def __init__(self, dataset: Any, kwargs: dict[str, Any]) -> None:
        self.dataset = dataset
        self.kwargs = kwargs
        self.visible = True

    def SetVisibility(self, value: bool) -> None:
        self.visible = bool(value)


class FakeCamera:
    position = (5.0, 4.0, 3.0)
    focal_point = (0.0, 0.0, 0.0)
    up = (0.0, 0.0, 1.0)
    parallel_scale = 2.0


class FakePlotter:
    def __init__(self) -> None:
        self.camera = FakeCamera()
        self.actors: list[FakeActor] = []
        self.render_count = 0

    def add_mesh(self, dataset: Any, **kwargs: Any) -> FakeActor:
        actor = FakeActor(dataset, kwargs)
        self.actors.append(actor)
        return actor

    def remove_actor(self, actor: FakeActor, **_kwargs: Any) -> None:
        self.actors.remove(actor)

    def render(self) -> None:
        self.render_count += 1

    def reset_camera_clipping_range(self) -> None:
        pass


def make_viewer() -> tuple[Viewer, FakePlotter]:
    plotter = FakePlotter()
    return Viewer._attach(plotter), plotter


def make_mesh() -> TriangleMesh:
    return TriangleMesh(
        vertices=np.asarray(
            [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ]
        ),
        faces=np.asarray([(0, 1, 2)]),
    )


def test_converters_preserve_geometry_and_connectivity() -> None:
    colored_mesh = TriangleMesh(
        make_mesh().vertices,
        make_mesh().faces,
        face_colors=np.asarray([(255, 0, 0)], dtype=np.uint8),
    )
    mesh_data = triangle_mesh_to_polydata(colored_mesh, pv)
    cloud_data = point_cloud_to_polydata(
        PointCloud(
            np.asarray([(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)]),
            np.asarray([(255, 0, 0), (0, 255, 0)], dtype=np.uint8),
        ),
        pv,
    )
    points_data = points_to_polydata(
        (Point3D(0.0, 0.0, 0.0), Point3D(1.0, 2.0, 3.0)),
        pv,
    )
    line_data = line_to_polydata(
        Line3D(Point3D(0.0, 0.0, 0.0), Point3D(1.0, 0.0, 0.0)),
        pv,
    )
    polyline_data = polyline_to_polydata(
        Polyline3D(
            (
                Point3D(0.0, 0.0, 0.0),
                Point3D(1.0, 0.0, 0.0),
                Point3D(1.0, 1.0, 0.0),
            )
        ),
        pv,
    )

    assert mesh_data.n_points == 3
    assert mesh_data.n_cells == 1
    assert "face_colors" in mesh_data.cell_data
    assert cloud_data.n_points == 2
    assert "point_colors" in cloud_data.point_data
    assert points_data.n_points == 2
    assert line_data.n_points == 2
    assert line_data.n_lines == 1
    assert polyline_data.n_points == 3
    assert polyline_data.n_lines == 2


def test_style_validation_and_immutability() -> None:
    with pytest.raises(ValueError, match="positive"):
        PointStyle(size=0.0)
    with pytest.raises(ValueError, match="positive"):
        PointCloudStyle(size=0.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        MeshStyle(opacity=1.1)
    with pytest.raises(ValueError, match="positive"):
        FrameStyle(scale=-1.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        LineStyle(color=(2.0, 0.0, 0.0))

    style = LineStyle()
    with pytest.raises(AttributeError):
        style.width = 8.0  # type: ignore[misc]


def test_named_lifecycle_updates_only_one_visual_and_preserves_camera() -> None:
    viewer, plotter = make_viewer()
    first = viewer.add_point(Point3D(0.0, 0.0, 0.0), name="point")
    second = viewer.add_line(
        Line3D(Point3D(0.0, 0.0, 0.0), Point3D(1.0, 0.0, 0.0)),
    )
    original_second_actor = plotter.actors[1]
    camera_before = (
        plotter.camera.position,
        plotter.camera.focal_point,
        plotter.camera.up,
        plotter.camera.parallel_scale,
    )

    with pytest.raises(ValueError, match="already exists"):
        viewer.add_point(Point3D(1.0, 1.0, 1.0), name="point")

    first.update(Point3D(2.0, 2.0, 2.0))
    assert original_second_actor in plotter.actors
    assert viewer.get(second.name).name == "line_1"
    assert camera_before == (
        plotter.camera.position,
        plotter.camera.focal_point,
        plotter.camera.up,
        plotter.camera.parallel_scale,
    )

    first.set_visible(False)
    assert not viewer._record("point").actors[0].visible
    first.set_style(replace(PointStyle(), color="#ffffff"))
    assert not viewer._record("point").actors[0].visible
    first.remove()
    with pytest.raises(KeyError):
        viewer.get("point")


def test_point_cloud_uses_embedded_colors_and_supports_uniform_override() -> None:
    viewer, _plotter = make_viewer()
    cloud = PointCloud(
        np.asarray([(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)]),
        np.asarray([(255, 0, 0), (0, 255, 0)], dtype=np.uint8),
    )
    handle = viewer.add_point_cloud(cloud, name="scan")
    actor = viewer._record(handle.name).actors[0]

    assert actor.kwargs["scalars"] == "point_colors"
    assert actor.kwargs["rgb"] is True
    assert "color" not in actor.kwargs

    handle.set_style(PointCloudStyle(color="#ffffff", size=5.0))
    actor = viewer._record(handle.name).actors[0]
    assert actor.kwargs["color"] == "#ffffff"
    assert "scalars" not in actor.kwargs

    updated = PointCloud(np.asarray([(4.0, 5.0, 6.0)]))
    handle.update(updated)
    assert handle.source is updated
    assert viewer._record(handle.name).actors[0].dataset.n_points == 1


def test_mesh_uses_embedded_colors_and_supports_uniform_override() -> None:
    viewer, _plotter = make_viewer()
    base = make_mesh()
    mesh = TriangleMesh(
        base.vertices,
        base.faces,
        face_colors=np.asarray([(255, 0, 0)], dtype=np.uint8),
    )
    handle = viewer.add_mesh(mesh, name="colored_mesh")
    actor = viewer._record(handle.name).actors[0]

    assert actor.kwargs["scalars"] == "face_colors"
    assert actor.kwargs["preference"] == "cell"
    assert actor.kwargs["rgb"] is True
    assert "color" not in actor.kwargs

    handle.set_style(MeshStyle(color="#ffffff"))
    actor = viewer._record(handle.name).actors[0]
    assert actor.kwargs["color"] == "#ffffff"
    assert "scalars" not in actor.kwargs


def test_nested_batches_render_once() -> None:
    viewer, plotter = make_viewer()
    with viewer.batch():
        viewer.add_point(Point3D(0.0, 0.0, 0.0))
        with viewer.batch():
            viewer.add_point(Point3D(1.0, 0.0, 0.0))
            viewer.add_vector(
                Point3D(0.0, 0.0, 0.0),
                Vector3D(0.0, 0.0, 1.0),
            )

    assert plotter.render_count == 1
    viewer.clear()
    assert plotter.render_count == 2
    assert not plotter.actors


def test_pose_is_full_frame_and_target_is_point_plus_normal() -> None:
    viewer, _plotter = make_viewer()
    pose = viewer.add_pose(
        Pose3D(
            Point3D(1.0, 2.0, 3.0),
            Rotation.from_euler("z", 90.0, degrees=True),
        ),
        style=FrameStyle(scale=2.0),
    )
    target = viewer.add_target(
        DepositionTarget(Point3D(0.0, 0.0, 0.0), Vector3D(0.0, 1.0, 0.0)),
        style=TargetStyle(scale=3.0),
    )

    assert len(viewer._record(pose.name).actors) == 4
    assert len(viewer._record(target.name).actors) == 2


def test_deposit_group_covers_all_deposit_variants() -> None:
    viewer, plotter = make_viewer()
    profile = BeadProfile(width=1.0, height=0.5)
    deposits = (
        PointDeposit(Point3D(0.0, 0.0, 0.0), profile),
        LineDeposit(
            Point3D(1.0, 0.0, 0.0),
            Point3D(2.0, 0.0, 0.0),
            profile,
        ),
        PolylineDeposit(
            (
                Point3D(2.0, 0.0, 0.0),
                Point3D(2.0, 1.0, 0.0),
                Point3D(3.0, 1.0, 0.0),
            ),
            profile,
        ),
    )
    handle = viewer.add_deposits(deposits, style=DepositStyle(), name="deposits")

    record = viewer._record(handle.name)
    assert len(record.actors) == 14
    handle.update(deposits[0])
    assert len(viewer._record(handle.name).actors) == 2
    handle.remove()
    assert not plotter.actors


def test_invalid_and_empty_inputs_are_rejected() -> None:
    viewer, _plotter = make_viewer()

    with pytest.raises(ValueError, match="must not be empty"):
        viewer.add_points(())
    with pytest.raises(ValueError, match="must not be empty"):
        viewer.add_deposits(())
    with pytest.raises(TypeError, match="PointCloud"):
        viewer.add_point_cloud(np.zeros((2, 3)))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Point3D"):
        viewer.add_point((0.0, 0.0, 0.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-zero"):
        viewer.add_vector(Point3D(0.0, 0.0, 0.0), Vector3D(0.0, 0.0, 0.0))
