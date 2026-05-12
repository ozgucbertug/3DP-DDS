"""YAML format adapter for ordered target-point workflows."""

from __future__ import annotations

import re
from ast import literal_eval
from pathlib import Path
from typing import Any

from ..targets import TargetPoint

PLANE_COMPONENT_RE = re.compile(r"([A-Za-z])\(([^)]*)\)")


def _strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_double:
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def _parse_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        raise ValueError("Nested YAML values are not supported by the fallback parser.")
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if value[0] in {"'", '"', "["}:
        try:
            return literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Could not parse YAML scalar: {value!r}") from exc
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_yaml_assignment(line: str) -> tuple[str, Any]:
    if ":" not in line:
        raise ValueError(f"Expected a YAML key/value assignment, got: {line!r}")
    key, value = line.split(":", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Expected a YAML key before ':', got: {line!r}")
    return key, _parse_yaml_scalar(value)


def _load_yaml_targets_subset(path: Path) -> dict[str, list[dict[str, Any]]]:
    targets: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_targets = False

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = _strip_yaml_comment(raw_line).rstrip()
        if not line.strip():
            continue

        stripped = line.strip()
        if stripped == "targets:":
            in_targets = True
            continue
        if not in_targets:
            raise ValueError(
                f"Fallback YAML parser only supports a top-level `targets` list. Unexpected line {line_number}: {stripped!r}"
            )

        if stripped.startswith("- "):
            if current is not None:
                targets.append(current)
            current = {}
            rest = stripped[2:].strip()
            if rest:
                key, value = _parse_yaml_assignment(rest)
                current[key] = value
            continue

        if current is None:
            raise ValueError(f"Expected a target list item before line {line_number}: {stripped!r}")
        key, value = _parse_yaml_assignment(stripped)
        current[key] = value

    if current is not None:
        targets.append(current)
    if not in_targets:
        raise ValueError("YAML file must contain a top-level `targets` list.")
    return {"targets": targets}


def _load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError:
        return _load_yaml_targets_subset(path)

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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

    yaml_path = Path(path)
    payload = _load_yaml(yaml_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("targets"), list):
        raise ValueError("YAML file must contain a top-level `targets` list.")

    targets: list[TargetPoint] = []
    for ordinal, item in enumerate(payload["targets"]):
        if not isinstance(item, dict):
            raise ValueError(f"Target entry {ordinal} must be a mapping.")
        index = int(item.get("index", ordinal))

        if "origin" in item:
            origin = _parse_origin_value(item["origin"], name=f"target {index} origin")
            z_axis = (0.0, 0.0, 1.0)
        elif "plane" in item:
            components = parse_plane_string(str(item["plane"]))
            origin = components["O"]
            z_axis = components.get("Z", (0.0, 0.0, 1.0))
        else:
            raise ValueError(f"Target {index} must contain either `plane` or `origin`.")

        targets.append(TargetPoint(index=index, origin=origin, z_axis=z_axis))

    return tuple(sorted(targets, key=lambda target: target.index))
