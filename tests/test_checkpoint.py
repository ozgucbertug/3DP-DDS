"""Tests for typed checkpoint serialization: save_checkpoint / load_checkpoint."""

from __future__ import annotations

import numpy as np
import pytest

from dds import (
    BeadProfile,
    DepositionTarget,
    DepositionMetadata,
    Domain,
    LineDeposit,
    PointDeposit,
    PolylineDeposit,
    SimulationResult,
    simulate,
)
from dds.io import _deposit_from_dict, _deposit_to_dict, load_checkpoint, save_checkpoint


def make_domain() -> Domain:
    return Domain.from_bounds(
        xmin=0.0, xmax=10.0, ymin=0.0, ymax=10.0, zmin=0.0, zmax=10.0, voxel_size=1.0
    )


def make_result(*, include_coverage: bool = False) -> SimulationResult:
    domain = make_domain()
    deposits = [
        PointDeposit(
            target=(2.5, 2.5, 3.5),
            profile=BeadProfile(width=2.0, height=2.0),
            metadata=DepositionMetadata(layer_id=0, user_data={"material_id": "PLA"}),
        ),
        LineDeposit(
            start=(1.5, 5.0, 3.5),
            end=(8.5, 5.0, 3.5),
            profile=BeadProfile(width=1.5, height=1.0),
            metadata=DepositionMetadata(layer_id=1),
        ),
    ]
    compositions = ("max", "coverage") if include_coverage else ("max",)
    return simulate(domain, deposits, compositions=compositions, threshold=0.3)


# ---------------------------------------------------------------------------
# _deposit_to_dict / _deposit_from_dict round-trips
# ---------------------------------------------------------------------------


def test_point_deposit_serialization_round_trip() -> None:
    original = PointDeposit(
        target=DepositionTarget((1.0, 2.0, 3.0), (0.0, 0.0, 1.0)),
        profile=BeadProfile(width=1.5, height=0.8),
        metadata=DepositionMetadata(layer_id=5, user_data={"tag": "A"}),
    )
    restored = _deposit_from_dict(_deposit_to_dict(original))
    assert isinstance(restored, PointDeposit)
    assert restored.target.position == original.target.position
    assert restored.target.normal == original.target.normal
    assert restored.profile == original.profile
    assert restored.metadata.layer_id == original.metadata.layer_id
    assert restored.metadata.user_data == {"tag": "A"}


def test_line_deposit_serialization_round_trip() -> None:
    original = LineDeposit(
        start=DepositionTarget((1.0, 2.0, 3.0), (0.0, 0.0, 1.0)),
        end=DepositionTarget((4.0, 5.0, 6.0), (0.0, 1.0, 1.0)),
        profile=BeadProfile(width=2.0, height=1.0),
    )
    restored = _deposit_from_dict(_deposit_to_dict(original))
    assert isinstance(restored, LineDeposit)
    assert restored.start == original.start
    assert restored.end.position == original.end.position
    assert restored.end.normal.to_tuple() == pytest.approx(
        original.end.normal.to_tuple()
    )


def test_polyline_deposit_serialization_round_trip() -> None:
    original = PolylineDeposit(
        targets=(
            DepositionTarget((0.0, 0.0, 1.0)),
            DepositionTarget((1.0, 0.0, 1.0), (0.0, 1.0, 1.0)),
            DepositionTarget((1.0, 1.0, 1.0)),
        ),
        profile=BeadProfile(width=1.0, height=0.5),
        metadata=DepositionMetadata(
            layer_id=2,
            user_data={"material_id": "clay", "feedrate": 20.0},
        ),
    )

    restored = _deposit_from_dict(_deposit_to_dict(original))

    assert isinstance(restored, PolylineDeposit)
    assert len(restored.targets) == 3
    assert restored.targets[1].normal.to_tuple() == pytest.approx(
        original.targets[1].normal.to_tuple()
    )
    assert restored.metadata.user_data["material_id"] == "clay"


def test_line_deposit_default_poses_round_trip() -> None:
    original = LineDeposit(
        start=(0.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        profile=BeadProfile(width=1.0, height=0.5),
    )
    restored = _deposit_from_dict(_deposit_to_dict(original))
    assert restored.start == original.start
    assert restored.end == original.end


def test_deposit_to_dict_raises_for_unknown_type() -> None:
    class FakeDeposit:
        pass

    with pytest.raises(TypeError, match="Cannot serialise"):
        _deposit_to_dict(FakeDeposit())  # type: ignore[arg-type]


def test_deposit_from_dict_raises_for_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown deposit type"):
        _deposit_from_dict(
            {
                    "type": "GhostDeposit",
                    "profile": {"width": 1.0, "height": 1.0},
                    "metadata": {},
                }
            )


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint: round-trip
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip_density_max(tmp_path) -> None:
    result = make_result()
    path = tmp_path / "sim.npz"
    written = save_checkpoint(path, result)
    assert written == path

    loaded = load_checkpoint(path)
    np.testing.assert_allclose(loaded.density_max, result.density_max)
    assert not loaded.density_max.flags.writeable


def test_checkpoint_round_trip_with_coverage(tmp_path) -> None:
    result = make_result(include_coverage=True)
    assert result.coverage is not None

    path = save_checkpoint(tmp_path / "sim", result)
    loaded = load_checkpoint(path)

    assert loaded.coverage is not None
    np.testing.assert_allclose(loaded.coverage, result.coverage)


def test_checkpoint_round_trip_no_coverage(tmp_path) -> None:
    result = make_result(include_coverage=False)
    assert result.coverage is None

    loaded = load_checkpoint(save_checkpoint(tmp_path / "sim", result))
    assert loaded.coverage is None


def test_checkpoint_round_trip_domain(tmp_path) -> None:
    result = make_result()
    loaded = load_checkpoint(save_checkpoint(tmp_path / "sim", result))

    assert loaded.domain.min_corner == pytest.approx(result.domain.min_corner)
    assert loaded.domain.max_corner == pytest.approx(result.domain.max_corner)
    assert loaded.domain.voxel_size == pytest.approx(result.domain.voxel_size)
    assert loaded.domain.grid_shape == result.domain.grid_shape


def test_checkpoint_round_trip_threshold(tmp_path) -> None:
    result = make_result()
    loaded = load_checkpoint(save_checkpoint(tmp_path / "sim", result))
    assert loaded.default_threshold == pytest.approx(result.default_threshold)


def test_checkpoint_round_trip_deposits(tmp_path) -> None:
    result = make_result()
    loaded = load_checkpoint(save_checkpoint(tmp_path / "sim", result))

    assert len(loaded.deposits) == len(result.deposits)
    point_orig = result.deposits[0]
    point_loaded = loaded.deposits[0]
    assert isinstance(point_loaded, PointDeposit)
    assert point_loaded.target == point_orig.target
    assert point_loaded.profile == point_orig.profile
    assert point_loaded.metadata.layer_id == point_orig.metadata.layer_id
    assert point_loaded.metadata.user_data == point_orig.metadata.user_data


def test_checkpoint_extension_appended_automatically(tmp_path) -> None:
    result = make_result()
    path_no_ext = tmp_path / "sim_no_ext"
    written = save_checkpoint(path_no_ext, result)
    assert written.suffix == ".npz"
    assert written.exists()


def test_checkpoint_load_accepts_path_without_extension(tmp_path) -> None:
    result = make_result()
    save_checkpoint(tmp_path / "sim", result)
    loaded = load_checkpoint(tmp_path / "sim")  # no .npz
    np.testing.assert_allclose(loaded.density_max, result.density_max)


def test_checkpoint_raises_on_unsupported_version(tmp_path) -> None:
    import json

    import numpy as np_inner

    # Build a minimal npz with wrong version number.
    meta = {"version": 999, "threshold": 0.5, "domain": {}, "deposits": []}
    meta_bytes = json.dumps(meta).encode("utf-8")
    domain = make_domain()
    density = np_inner.zeros(domain.grid_shape)
    path = tmp_path / "bad.npz"
    np_inner.savez_compressed(
        path,
        density_max=density,
        _meta=np_inner.frombuffer(meta_bytes, dtype=np_inner.uint8),
    )
    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        load_checkpoint(path)


# ---------------------------------------------------------------------------
# SimulationResult.checkpoint() / SimulationResult.load() convenience API
# ---------------------------------------------------------------------------


def test_simulation_result_checkpoint_method(tmp_path) -> None:
    result = make_result()
    path = result.checkpoint(tmp_path / "via_method")
    assert path.exists()

    loaded = SimulationResult.load(path)
    np.testing.assert_allclose(loaded.density_max, result.density_max)
    assert len(loaded.deposits) == len(result.deposits)
