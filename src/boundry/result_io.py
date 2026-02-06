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
        "alanine_scan": _to_jsonable(result.alanine_scan),
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
    per_position_path: Optional[PathLike] = None,
    alanine_scan_path: Optional[PathLike] = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Write per-position and/or alanine scan CSVs.

    Returns a tuple of ``(per_position_path, alanine_scan_path)``
    where each element is the written path or ``None``.
    """
    pp_written: Optional[Path] = None
    ala_written: Optional[Path] = None

    if result.per_position is not None and per_position_path is not None:
        path = Path(per_position_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_position_csv(result.per_position, path)
        pp_written = path

    if result.alanine_scan is not None and alanine_scan_path is not None:
        path = Path(alanine_scan_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_position_csv(result.alanine_scan, path)
        ala_written = path

    return pp_written, ala_written


def resolve_interface_output_paths(
    output_path: PathLike,
    *,
    include_per_position_csv: bool = False,
    include_alanine_scan_csv: bool = False,
    per_position_csv: Optional[PathLike] = None,
    alanine_scan_csv: Optional[PathLike] = None,
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """Resolve JSON summary path and optional CSV paths.

    Returns ``(summary_json, per_position_csv, alanine_scan_csv)``.

    Rules:
    - Directory output (no suffix): summary is ``interface_analysis.json``
      and default CSVs are ``interface_per_position.csv`` and
      ``interface_alanine_scan.csv``.
    - File output: suffix must be ``.json`` and default CSVs are
      ``<stem>_per_position.csv`` and ``<stem>_alanine_scan.csv`` in
      the same directory.
    - Explicit ``per_position_csv``/``alanine_scan_csv`` overrides
      default CSV paths.
    """
    base = Path(output_path)
    is_directory = base.suffix == ""

    if is_directory:
        summary_path = base / "interface_analysis.json"
        default_pp_csv = base / "interface_per_position.csv"
        default_ala_csv = base / "interface_alanine_scan.csv"
    else:
        if base.suffix.lower() != ".json":
            raise ValueError(
                "analyze-interface output must be a .json file or directory"
            )
        summary_path = base
        default_pp_csv = base.with_name(f"{base.stem}_per_position.csv")
        default_ala_csv = base.with_name(f"{base.stem}_alanine_scan.csv")

    pp_path = Path(per_position_csv) if per_position_csv is not None else None
    if include_per_position_csv and pp_path is None:
        pp_path = default_pp_csv

    ala_path = (
        Path(alanine_scan_csv) if alanine_scan_csv is not None else None
    )
    if include_alanine_scan_csv and ala_path is None:
        ala_path = default_ala_csv

    return summary_path, pp_path, ala_path


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
