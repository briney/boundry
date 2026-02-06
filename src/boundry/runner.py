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
    position_csv: Optional[Path] = None


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
    position_csv: Optional[PathLike] = None,
    include_position_csv: bool = False,
    **operation_kwargs: Any,
) -> Tuple["InterfaceAnalysisResult", InterfaceOutputs]:
    """Execute analyze-interface and optionally write JSON/CSV artifacts."""
    result = operation(structure, **operation_kwargs)
    outputs = InterfaceOutputs()

    if output is not None:
        summary_path, csv_path = resolve_interface_output_paths(
            output,
            include_position_csv=include_position_csv,
            position_csv=position_csv,
        )
        outputs = InterfaceOutputs(
            summary_json=write_interface_json(result, summary_path),
            position_csv=write_interface_csv(result, csv_path)
            if csv_path is not None
            else None,
        )
    elif position_csv is not None:
        outputs = InterfaceOutputs(
            position_csv=write_interface_csv(result, position_csv)
        )

    return result, outputs
