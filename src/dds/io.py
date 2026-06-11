"""Simple array and metadata export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .domain import Domain

if TYPE_CHECKING:
    from .primitives import Deposit
    from .results import SimulationResult

_CHECKPOINT_VERSION = 7


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def save_array(path: str | Path, array: npt.NDArray[np.generic]) -> Path:
    """Save an array to disk using NumPy's native `.npy` format."""

    target = Path(path)
    if target.suffix != ".npy":
        target = Path(f"{target}.npy")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, array)
    return target.resolve()


def save_simulation_bundle(
    directory: str | Path,
    *,
    domain: Domain,
    occupancy: npt.NDArray[np.bool_] | None = None,
    deposition_index: npt.NDArray[np.integer[Any]] | npt.NDArray[np.float64] | None = None,
    implicit_field: npt.NDArray[np.float64] | None = None,
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
    if implicit_field is not None:
        written["implicit_field"] = save_array(
            output_dir / "implicit_field.npy",
            implicit_field,
        )

    payload = {
        "domain": domain.to_dict(),
        "metadata": metadata or {},
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    written["metadata"] = metadata_path.resolve()
    return written


# ---------------------------------------------------------------------------
# Typed round-trip checkpoint
# ---------------------------------------------------------------------------


def _deposit_to_dict(deposit: Deposit) -> dict[str, Any]:
    """Serialise one leaf deposit to a plain JSON-compatible dict."""

    from .primitives import LineDeposit, PointDeposit, PolylineDeposit

    if isinstance(deposit, PointDeposit):
        return {
            "type": "PointDeposit",
            "target": deposit.target.to_dict(),
            "profile": deposit.profile.to_dict(),
            "metadata": deposit.metadata.to_dict(),
        }
    if isinstance(deposit, LineDeposit):
        return {
            "type": "LineDeposit",
            "start": deposit.start.to_dict(),
            "end": deposit.end.to_dict(),
            "profile": deposit.profile.to_dict(),
            "metadata": deposit.metadata.to_dict(),
        }
    if isinstance(deposit, PolylineDeposit):
        return {
            "type": "PolylineDeposit",
            "targets": [target.to_dict() for target in deposit.targets],
            "profile": deposit.profile.to_dict(),
            "metadata": deposit.metadata.to_dict(),
        }
    raise TypeError(f"Cannot serialise deposit of type {type(deposit).__name__!r}.")


def _deposit_from_dict(d: dict[str, Any]) -> Deposit:
    """Reconstruct a leaf deposit from a plain dict produced by :func:`_deposit_to_dict`."""

    from .attributes import BeadProfile, DepositionMetadata
    from .primitives import DepositionTarget, LineDeposit, PointDeposit, PolylineDeposit

    deposit_type = d.get("type")
    if deposit_type not in {"PointDeposit", "LineDeposit", "PolylineDeposit"}:
        raise ValueError(f"Unknown deposit type {deposit_type!r}.")

    profile = BeadProfile(**d["profile"])
    metadata = DepositionMetadata(**d["metadata"])

    if deposit_type == "PointDeposit":
        return PointDeposit(
            target=DepositionTarget(
                position=tuple(d["target"]["position"]),
                normal=tuple(d["target"]["normal"]),
            ),
            profile=profile,
            metadata=metadata,
        )
    if deposit_type == "LineDeposit":
        return LineDeposit(
            start=DepositionTarget(
                position=tuple(d["start"]["position"]),
                normal=tuple(d["start"]["normal"]),
            ),
            end=DepositionTarget(
                position=tuple(d["end"]["position"]),
                normal=tuple(d["end"]["normal"]),
            ),
            profile=profile,
            metadata=metadata,
        )
    if deposit_type == "PolylineDeposit":
        return PolylineDeposit(
            targets=tuple(
                DepositionTarget(
                    position=tuple(target["position"]),
                    normal=tuple(target["normal"]),
                )
                for target in d["targets"]
            ),
            profile=profile,
            metadata=metadata,
        )
    raise AssertionError("unreachable")


def save_checkpoint(path: str | Path, result: SimulationResult) -> Path:
    """Save a :class:`~dds.results.SimulationResult` as a typed checkpoint.

    The checkpoint is a single compressed ``npz`` file containing the implicit
    field and a JSON blob with the domain geometry, deposit list, and
    threshold.  Use :func:`load_checkpoint` to restore the result.

    Parameters
    ----------
    path:
        Destination file path.  A ``.npz`` extension is appended if absent.
    result:
        The simulation result to checkpoint.

    Returns
    -------
    Path
        Absolute path of the written file (with ``.npz`` extension).
    """

    target = Path(path)
    if target.suffix != ".npz":
        target = target.with_suffix(".npz")
    target.parent.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "version": _CHECKPOINT_VERSION,
        "threshold": result.default_threshold,
        "domain": result.domain.to_dict(),
        "deposits": [_deposit_to_dict(d) for d in result.deposits],
    }
    meta_bytes = json.dumps(meta, default=_json_default).encode("utf-8")

    arrays: dict[str, Any] = {
        "implicit_field": result.implicit_field,
        "_meta": np.frombuffer(meta_bytes, dtype=np.uint8),
    }
    if result.coverage is not None:
        arrays["coverage"] = result.coverage

    np.savez_compressed(target, **arrays)
    return target.resolve()


def load_checkpoint(path: str | Path) -> SimulationResult:
    """Restore a :class:`~dds.results.SimulationResult` from a typed checkpoint.

    Parameters
    ----------
    path:
        Path to the ``.npz`` checkpoint file.

    Returns
    -------
    SimulationResult
        A fully reconstructed result with implicit geometry and deposits.

    Raises
    ------
    ValueError
        When the file format version is not supported.
    """

    from .results import SimulationResult

    target = Path(path)
    if target.suffix != ".npz":
        target = target.with_suffix(".npz")

    data = np.load(target, allow_pickle=False)
    meta = json.loads(data["_meta"].tobytes().decode("utf-8"))

    version = meta.get("version")
    if version != _CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {version!r}. Expected {_CHECKPOINT_VERSION}."
        )

    d = meta["domain"]
    domain = Domain(
        min_corner=tuple(d["min_corner"]),
        max_corner=tuple(d["max_corner"]),
        voxel_size=tuple(d["voxel_size"]),
        grid_shape=tuple(d["grid_shape"]),
        length_unit=d["length_unit"],
    )

    deposits = tuple(_deposit_from_dict(dep) for dep in meta["deposits"])
    coverage = data["coverage"] if "coverage" in data else None

    return SimulationResult(
        domain=domain,
        deposits=deposits,
        implicit_field=data["implicit_field"],
        coverage=coverage,
        default_threshold=float(meta["threshold"]),
    )
