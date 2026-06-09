"""YAML format adapter for ordered target-point workflows."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..targets import TargetPoint

PLANE_COMPONENT_RE = re.compile(r"([A-Za-z])\(([^)]*)\)")


def _parse_vector(text: str, *, name: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{name} must contain exactly three comma-separated numbers.")
    values = tuple(float(part) for part in parts)
    if not all(float("-inf") < value < float("inf") for value in values):
        raise ValueError(f"{name} must contain finite values.")
    return values


def parse_plane_string(plane: str) -> dict[str, tuple[float, float, float]]:
    """Parse compact plane strings like `O(x,y,z) Z(nx,ny,nz)`."""

    result: dict[str, tuple[float, float, float]] = {}
    for label, payload in PLANE_COMPONENT_RE.findall(plane):
        result[label.upper()] = _parse_vector(payload, name=f"plane component {label}")
    if "O" not in result:
        raise ValueError(f"Plane string does not contain an origin component: {plane!r}")
    return result


def _parse_origin_value(value: Any, *, name: str) -> tuple[float, float, float]:
    if isinstance(value, str):
        return _parse_vector(value, name=name)
    return _parse_vector(",".join(str(component) for component in value), name=name)


def load_targets(path: str | Path) -> tuple[TargetPoint, ...]:
    """Load ordered target points from a YAML file."""

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            'pyyaml is required for YAML target loading. Install it with `pip install -e ".[formats]"`.',
        ) from exc

    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict) or not isinstance(payload.get("targets"), list):
        raise ValueError("YAML file must contain a top-level `targets` list.")

    targets: list[TargetPoint] = []
    for ordinal, item in enumerate(payload["targets"]):
        if not isinstance(item, dict):
            raise ValueError(f"Target entry {ordinal} must be a mapping.")
        index = int(item.get("index", ordinal))

        if "origin" in item:
            origin = _parse_origin_value(item["origin"], name=f"target {index} origin")
            z_axis = _parse_origin_value(
                item.get("z_axis", (0.0, 0.0, 1.0)),
                name=f"target {index} z_axis",
            )
        elif "plane" in item:
            components = parse_plane_string(str(item["plane"]))
            origin = components["O"]
            z_axis = components.get("Z", (0.0, 0.0, 1.0))
        else:
            raise ValueError(f"Target {index} must contain either `plane` or `origin`.")

        targets.append(TargetPoint(index=index, origin=origin, z_axis=z_axis))

    indices = [target.index for target in targets]
    if len(indices) != len(set(indices)):
        raise ValueError("Target indices must be unique.")
    return tuple(sorted(targets, key=lambda target: target.index))
