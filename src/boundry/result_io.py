"""Shared result serialization and output-path helpers."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

from boundry.interface_position_energetics import write_position_csv

if False:  # pragma: no cover
    from boundry.operations import InterfaceAnalysisResult, Structure


PathLike = Union[str, Path]


def write_structure_output(structure: "Structure", output_path: PathLike) -> Path:
    """Write a structure result to disk and return the normalized path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    structure.write(path)
    return path


def interface_result_to_dict(result: "InterfaceAnalysisResult") -> dict[str, Any]:
    """Convert InterfaceAnalysisResult into JSON-serializable data."""
    return {
        "interface_info": _to_jsonable(result.interface_info),
        "binding_energy": _to_jsonable(result.binding_energy),
        "sasa": _to_jsonable(result.sasa),
        "shape_complementarity": _to_jsonable(result.shape_complementarity),
        "per_position": _to_jsonable(result.per_position),
    }


def write_interface_json(
    result: "InterfaceAnalysisResult",
    output_path: PathLike,
) -> Path:
    """Write interface analysis summary JSON and return the path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(interface_result_to_dict(result), f, indent=2, default=str)
    return path


def write_interface_csv(
    result: "InterfaceAnalysisResult",
    output_path: PathLike,
) -> Optional[Path]:
    """Write per-position CSV when available and return written path."""
    if result.per_position is None:
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_position_csv(result.per_position, path)
    return path


def resolve_interface_output_paths(
    output_path: PathLike,
    *,
    include_position_csv: bool,
    position_csv: Optional[PathLike] = None,
) -> Tuple[Path, Optional[Path]]:
    """Resolve JSON summary path and optional position CSV path.

    Rules:
    - Directory output (no suffix): summary is ``interface_analysis.json``
      and default CSV is ``interface_positions.csv``.
    - File output: suffix must be ``.json`` and default CSV is
      ``<stem>_positions.csv`` in the same directory.
    - Explicit ``position_csv`` overrides default CSV path.
    """
    base = Path(output_path)
    is_directory = base.suffix == ""

    if is_directory:
        summary_path = base / "interface_analysis.json"
        default_csv = base / "interface_positions.csv"
    else:
        if base.suffix.lower() != ".json":
            raise ValueError(
                "analyze-interface output must be a .json file or directory"
            )
        summary_path = base
        default_csv = base.with_name(f"{base.stem}_positions.csv")

    csv_path = Path(position_csv) if position_csv is not None else None
    if include_position_csv and csv_path is None:
        csv_path = default_csv

    return summary_path, csv_path


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return {
            field.name: _to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value
