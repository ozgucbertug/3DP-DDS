"""Simple array and metadata export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .domain import Domain


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def save_array(path: str | Path, array: npt.NDArray[np.generic]) -> Path:
    """Save an array to disk using NumPy's native `.npy` format."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, array)
    return target


def save_simulation_bundle(
    directory: str | Path,
    *,
    domain: Domain,
    occupancy: npt.NDArray[np.bool_] | None = None,
    deposition_index: npt.NDArray[np.float64] | None = None,
    density: npt.NDArray[np.float64] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Save simulation outputs and metadata into a directory."""

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    if occupancy is not None:
        written["occupancy"] = save_array(output_dir / "occupancy.npy", occupancy)
    if deposition_index is not None:
        written["deposition_index"] = save_array(output_dir / "deposition_index.npy", deposition_index)
    if density is not None:
        written["density"] = save_array(output_dir / "density.npy", density)

    payload = {
        "domain": domain.to_dict(),
        "metadata": metadata or {},
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    written["metadata"] = metadata_path
    return written
