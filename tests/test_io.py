from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
