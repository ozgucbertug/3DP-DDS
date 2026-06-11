from __future__ import annotations

import numpy as np
import pytest

from dds.viz import _occupied_index_bounds


def test_occupied_index_bounds_uses_inclusive_axis_extents() -> None:
    occupancy = np.zeros((9, 7, 5), dtype=bool)
    occupancy[1, 5, 2] = True
    occupancy[7, 2, 4] = True

    assert _occupied_index_bounds(occupancy) == ((1, 2, 2), (7, 5, 4))


def test_occupied_index_bounds_handles_empty_and_dense_fields() -> None:
    occupancy = np.zeros((3, 4, 5), dtype=bool)

    assert _occupied_index_bounds(occupancy) is None
    assert _occupied_index_bounds(np.ones_like(occupancy)) == ((0, 0, 0), (2, 3, 4))


def test_occupied_index_bounds_rejects_non_volume_inputs() -> None:
    with pytest.raises(ValueError, match="three-dimensional"):
        _occupied_index_bounds(np.zeros((3, 4), dtype=bool))
