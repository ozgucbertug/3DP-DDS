from __future__ import annotations

from pathlib import Path

import pytest

from dds import BeadProfile, DepositionMetadata, Domain, simulate
from dds.formats.yaml import load_targets, parse_plane_string
from dds.targets import (
    TargetPoint,
    line_deposits_from_targets,
    point_deposits_from_targets,
    toolpath_from_targets,
)


def test_parse_plane_string_extracts_origin_and_z_axis() -> None:
    parsed = parse_plane_string("O(1,2,3) Z(0,0,1)")
    assert parsed["O"] == (1.0, 2.0, 3.0)
    assert parsed["Z"] == (0.0, 0.0, 1.0)


def test_load_targets_reads_origin_and_plane_entries(tmp_path: Path) -> None:
    yaml_path = tmp_path / "targets.yaml"
    yaml_path.write_text(
        """
targets:
  - index: 2
    origin: [1, 2, 3]
  - index: 1
    plane: O(4,5,6) Z(0,1,0)
""".strip(),
        encoding="utf-8",
    )

    targets = load_targets(yaml_path)

    assert [target.index for target in targets] == [1, 2]
    assert targets[0].origin == (4.0, 5.0, 6.0)
    assert targets[0].z_axis == (0.0, 1.0, 0.0)
    assert targets[1].origin == (1.0, 2.0, 3.0)


def test_load_targets_reads_z_axis_with_origin_form(tmp_path: Path) -> None:
    yaml_path = tmp_path / "targets.yaml"
    yaml_path.write_text(
        """
targets:
  - index: 0
    origin: [1, 2, 3]
    z_axis: [0, 1, 0]
""".strip(),
        encoding="utf-8",
    )

    targets = load_targets(yaml_path)

    assert targets[0].z_axis == (0.0, 1.0, 0.0)


def test_load_targets_rejects_duplicate_indices(tmp_path: Path) -> None:
    yaml_path = tmp_path / "targets.yaml"
    yaml_path.write_text(
        """
targets:
  - index: 1
    origin: [1, 2, 3]
  - index: 1
    origin: [4, 5, 6]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique"):
        load_targets(yaml_path)


def test_point_target_workflow_creates_top_referenced_deposits() -> None:
    targets = (TargetPoint(index=0, origin=(1.0, 2.0, 3.0)),)
    profile = BeadProfile(width=4.0, height=2.0)
    metadata = DepositionMetadata(layer_id=7)

    deposits = point_deposits_from_targets(targets, profile=profile, metadata=metadata)

    assert len(deposits) == 1
    assert deposits[0].target.to_tuple() == (1.0, 2.0, 3.0)
    assert deposits[0].profile == profile
    assert deposits[0].metadata.layer_id == 7


def test_line_and_toolpath_workflows_follow_target_order() -> None:
    targets = (
        TargetPoint(index=0, origin=(0.0, 0.0, 1.0)),
        TargetPoint(index=1, origin=(2.0, 0.0, 1.0)),
        TargetPoint(index=2, origin=(2.0, 2.0, 1.0)),
    )
    profile = BeadProfile(width=2.0, height=1.0)
    line_deposits = line_deposits_from_targets(targets, profile=profile)
    toolpath = toolpath_from_targets(targets, profile=profile)

    assert len(line_deposits) == 2
    assert line_deposits[0].start.to_tuple() == (0.0, 0.0, 1.0)
    assert line_deposits[0].end.to_tuple() == (2.0, 0.0, 1.0)
    assert len(toolpath.poses) == 3
    assert len(toolpath.segments()) == 2

    domain = Domain.from_deposits(toolpath, voxel_size=0.5)
    result = simulate(domain, toolpath, compositions=("max", "coverage"))
    assert len(result.deposits) == 1
    assert result.coverage is not None


def test_domain_from_deposits_infers_padded_bounds() -> None:
    targets = (
        TargetPoint(index=0, origin=(1.0, 1.0, 2.0)),
        TargetPoint(index=1, origin=(3.0, 1.0, 2.0)),
    )
    profile = BeadProfile(width=2.0, height=2.0)
    deposits = point_deposits_from_targets(targets, profile=profile)

    domain = Domain.from_deposits(deposits, voxel_size=0.5, padding="auto")

    assert domain.grid_shape[0] > 0
    assert domain.min_corner[0] < 1.0
    assert domain.max_corner[0] > 3.0


def test_line_workflow_requires_at_least_two_targets() -> None:
    with pytest.raises(ValueError):
        line_deposits_from_targets((TargetPoint(index=0, origin=(0.0, 0.0, 0.0)),), profile=BeadProfile(width=1.0, height=1.0))
