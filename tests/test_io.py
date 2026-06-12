from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dds import Domain
from dds.io import save_array, save_simulation_bundle


def make_domain() -> Domain:
    return Domain.from_bounds(
        xmin=0.0,
        xmax=2.0,
        ymin=0.0,
        ymax=2.0,
        zmin=0.0,
        zmax=2.0,
        voxel_size=1.0,
    )


def test_save_array_writes_npy_file(tmp_path: Path) -> None:
    array = np.arange(8, dtype=float).reshape(2, 2, 2)
    path = save_array(tmp_path / "field.npy", array)

    assert path.exists()
    np.testing.assert_array_equal(np.load(path), array)


def test_save_array_returns_actual_path_when_extension_is_omitted(tmp_path: Path) -> None:
    array = np.arange(4, dtype=float)
    path = save_array(tmp_path / "field", array)

    assert path == (tmp_path / "field.npy").resolve()
    assert path.exists()


def test_save_simulation_bundle_writes_expected_outputs(tmp_path: Path) -> None:
    domain = make_domain()
    occupancy = np.zeros(domain.grid_shape, dtype=bool)
    occupancy[0, 0, 0] = True
    deposition_index = np.ones(domain.grid_shape, dtype=float)

    written = save_simulation_bundle(
        tmp_path / "bundle",
        domain=domain,
        occupancy=occupancy,
        deposition_index=deposition_index,
        metadata={"example": "io_test"},
    )

    assert set(written) == {"occupancy", "deposition_index", "metadata"}
    np.testing.assert_array_equal(np.load(written["occupancy"]), occupancy)
    np.testing.assert_array_equal(np.load(written["deposition_index"]), deposition_index)

    payload = json.loads(written["metadata"].read_text(encoding="utf-8"))
    assert payload["metadata"] == {"example": "io_test"}
    assert payload["domain"]["grid_shape"] == [2, 2, 2]


def test_save_simulation_bundle_rejects_non_json_metadata(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="not JSON serializable"):
        save_simulation_bundle(
            tmp_path / "bundle",
            domain=make_domain(),
            metadata={"unsupported": object()},
        )


def test_write_read_mesh_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("trimesh")
    pytest.importorskip("skimage")

    from dds import BeadProfile, PointDeposit, simulate
    from dds.geometry import read_mesh, write_mesh

    domain = Domain.from_bounds(
        xmin=0.0, xmax=4.0, ymin=0.0, ymax=4.0, zmin=0.0, zmax=4.0, voxel_size=0.25
    )
    result = simulate(
        domain,
        [PointDeposit(target=(2.0, 2.0, 2.0), profile=BeadProfile(width=1.5, height=1.5))],
        threshold=0.5,
    )
    mesh = result.analysis.surface_mesh()
    assert not mesh.is_empty

    path = write_mesh(tmp_path / "mesh.stl", mesh)
    loaded = read_mesh(path)
    # STL doesn't preserve vertex ordering, so only check structural properties.
    assert loaded.n_faces == mesh.n_faces
    assert loaded.n_vertices > 0
