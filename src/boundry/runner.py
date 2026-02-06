"""Shared operation runners with invocation-aware output handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from boundry.invocation import InvocationMode, OutputPolicy, PathLike
from boundry.result_io import (
    resolve_interface_output_paths,
    write_interface_csv,
    write_interface_json,
    write_structure_output,
)

if False:  # pragma: no cover
    from boundry.operations import InterfaceAnalysisResult, Structure, StructureInput


@dataclass(frozen=True)
class InterfaceOutputs:
    """Materialized analyze-interface output paths."""

    summary_json: Optional[Path] = None
    per_position_csv: Optional[Path] = None
    alanine_scan_csv: Optional[Path] = None


def run_structure_operation(
    *,
    name: str,
    operation: Callable[..., "Structure"],
    structure: "StructureInput",
    output: Optional[PathLike] = None,
    mode: InvocationMode = InvocationMode.API,
    output_policy: OutputPolicy = OutputPolicy(),
    **operation_kwargs: Any,
) -> "Structure":
    """Execute a structure-producing operation and materialize output."""
    output_policy.validate(output, operation=name, mode=mode)
    result = operation(structure, **operation_kwargs)
    if output is not None:
        write_structure_output(result, output)
    return result


def run_interface_operation(
    *,
    operation: Callable[..., "InterfaceAnalysisResult"],
    structure: "StructureInput",
    output: Optional[PathLike] = None,
    per_position_csv: Optional[PathLike] = None,
    alanine_scan_csv: Optional[PathLike] = None,
    include_per_position_csv: bool = False,
    include_alanine_scan_csv: bool = False,
    **operation_kwargs: Any,
) -> Tuple["InterfaceAnalysisResult", InterfaceOutputs]:
    """Execute analyze-interface and optionally write JSON/CSV artifacts."""
    result = operation(structure, **operation_kwargs)
    outputs = InterfaceOutputs()

    if output is not None:
        summary_path, pp_csv_path, ala_csv_path = (
            resolve_interface_output_paths(
                output,
                include_per_position_csv=include_per_position_csv,
                include_alanine_scan_csv=include_alanine_scan_csv,
                per_position_csv=per_position_csv,
                alanine_scan_csv=alanine_scan_csv,
            )
        )
        pp_written, ala_written = write_interface_csv(
            result,
            per_position_path=pp_csv_path,
            alanine_scan_path=ala_csv_path,
        )
        outputs = InterfaceOutputs(
            summary_json=write_interface_json(result, summary_path),
            per_position_csv=pp_written,
            alanine_scan_csv=ala_written,
        )
    else:
        # Handle explicit CSV paths without output directory
        if per_position_csv is not None or alanine_scan_csv is not None:
            pp_written, ala_written = write_interface_csv(
                result,
                per_position_path=per_position_csv,
                alanine_scan_path=alanine_scan_csv,
            )
            outputs = InterfaceOutputs(
                per_position_csv=pp_written,
                alanine_scan_csv=ala_written,
            )

    return result, outputs
