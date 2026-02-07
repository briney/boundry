"""Process-level parallelism for beam and step-level execution.

Uses ``ProcessPoolExecutor`` with the ``spawn`` start method to avoid
CUDA fork hazards and GIL contention during CPU-bound OpenMM
minimization.

Each worker imports modules inside the function body, creates fresh
Designer/Relaxer instances, and operates on pickle-safe ``Structure``
data — full process isolation with no shared state.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Serializable task / result types
# ------------------------------------------------------------------


@dataclass(frozen=True)
class BranchTask:
    """Serializable inputs for one beam branch expansion.

    All fields are pickle-safe (strings, dicts, tuples of primitives).
    """

    candidate_pdb_string: str
    candidate_metadata: Dict[str, Any]
    candidate_source_path: Optional[str]
    steps: List[Tuple[str, Dict[str, Any]]]
    branch_seed: Optional[int]


@dataclass
class SnapshotData:
    """Serializable snapshot of a single step result."""

    operation: str
    step_index: int
    result_metadata: Dict[str, Any]
    pdb_string: str
    merged_pdb_string: str
    merged_metadata: Dict[str, Any]
    merged_source_path: Optional[str]


@dataclass
class BranchResult:
    """Serializable outputs from one beam branch expansion."""

    pdb_string: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None
    snapshots: List[SnapshotData] = field(default_factory=list)
    error: Optional[str] = None


@dataclass(frozen=True)
class StepTask:
    """Serializable inputs for one population member's operation."""

    pdb_string: str
    metadata: Dict[str, Any]
    source_path: Optional[str]
    operation: str
    params: Dict[str, Any]


@dataclass
class StepResult:
    """Serializable outputs from one population member's operation."""

    pdb_string: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None
    error: Optional[str] = None


# ------------------------------------------------------------------
# Worker functions (top-level, pickle-safe targets)
# ------------------------------------------------------------------


def _execute_branch_worker(task: BranchTask) -> BranchResult:
    """Execute a full beam branch (sequence of steps) in a worker.

    This is the top-level function submitted to the process pool.
    It imports all dependencies inside the function body to work
    correctly with the ``spawn`` start method.
    """
    os.environ["BOUNDRY_IN_WORKER_PROCESS"] = "1"
    try:
        from boundry.operations import Structure
        from boundry.workflow import Workflow, _compose_seed
        from boundry.workflow_metadata import merge_metadata

        structure = Structure(
            pdb_string=task.candidate_pdb_string,
            metadata=dict(task.candidate_metadata),
            source_path=task.candidate_source_path,
        )

        snapshots: List[SnapshotData] = []

        for step_index, (operation, step_params) in enumerate(
            task.steps
        ):
            params = Workflow._with_seed(
                operation,
                dict(step_params),
                task.branch_seed,
                0,
            )
            result = Workflow._run_operation(
                operation, structure, params
            )

            merged = merge_metadata(
                structure.metadata,
                result.metadata,
                operation=operation,
            )
            merged_struct = Structure(
                pdb_string=result.pdb_string,
                metadata=merged,
                source_path=(
                    result.source_path or structure.source_path
                ),
            )

            snapshots.append(
                SnapshotData(
                    operation=operation,
                    step_index=step_index,
                    result_metadata=dict(result.metadata),
                    pdb_string=result.pdb_string,
                    merged_pdb_string=merged_struct.pdb_string,
                    merged_metadata=dict(merged_struct.metadata),
                    merged_source_path=merged_struct.source_path,
                )
            )

            structure = merged_struct

        return BranchResult(
            pdb_string=structure.pdb_string,
            metadata=dict(structure.metadata),
            source_path=structure.source_path,
            snapshots=snapshots,
        )

    except Exception as exc:
        return BranchResult(error=f"{type(exc).__name__}: {exc}")


def _execute_step_worker(task: StepTask) -> StepResult:
    """Execute a single operation on one population member in a worker.

    This is the top-level function submitted to the process pool for
    step-level parallelism (multi-member populations).
    """
    os.environ["BOUNDRY_IN_WORKER_PROCESS"] = "1"
    try:
        from boundry.operations import Structure
        from boundry.workflow import Workflow

        structure = Structure(
            pdb_string=task.pdb_string,
            metadata=dict(task.metadata),
            source_path=task.source_path,
        )

        result = Workflow._run_operation(
            task.operation, structure, dict(task.params)
        )

        return StepResult(
            pdb_string=result.pdb_string,
            metadata=dict(result.metadata),
            source_path=result.source_path,
        )

    except Exception as exc:
        return StepResult(error=f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------------
# Scan parallelism (per-position / alanine scan)
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ScanTask:
    """Serializable inputs for one per-position scan computation."""

    scan_type: str  # "alanine_scan" or "per_position"
    pdb_string: str
    chain_id: str
    residue_number: int
    insertion_code: str
    residue_name: str
    chain_pairs: List[Tuple[str, str]]
    distance_cutoff: float
    relax_separated: bool
    position_relax: str
    dG_wt: float
    quiet: bool


@dataclass
class ScanResult:
    """Serializable outputs from one per-position scan computation."""

    scan_type: str
    chain_id: str
    residue_number: int
    insertion_code: str
    residue_name: str
    dG: Optional[float] = None
    ddG: Optional[float] = None
    error: Optional[str] = None


# Module-level globals for worker-process reuse
_scan_relaxer = None
_scan_designer = None


def _init_scan_worker(
    relax_config_dict: Dict[str, Any],
    design_config_dict: Optional[Dict[str, Any]],
) -> None:
    """Initializer for scan worker processes.

    Creates ``Relaxer`` (and optionally ``Designer``) once per worker,
    stored in module globals for reuse across tasks.
    """
    global _scan_relaxer, _scan_designer  # noqa: PLW0603
    os.environ["BOUNDRY_IN_WORKER_PROCESS"] = "1"

    from boundry.config import DesignConfig, RelaxConfig
    from boundry.relaxer import Relaxer

    _scan_relaxer = Relaxer(RelaxConfig(**relax_config_dict))

    if design_config_dict is not None:
        from boundry.designer import Designer

        _scan_designer = Designer(DesignConfig(**design_config_dict))


def _execute_scan_worker(task: ScanTask) -> ScanResult:
    """Execute a single scan task in a worker process."""
    try:
        import contextlib

        from boundry.interface_position_energetics import (
            _compute_rosetta_dG,
            mutate_to_alanine,
            remove_residue,
        )
        from boundry.utils import suppress_stderr as _suppress_stderr

        if task.scan_type == "alanine_scan":
            modified_pdb = mutate_to_alanine(
                task.pdb_string,
                task.chain_id,
                task.residue_number,
                task.insertion_code,
            )
        else:
            modified_pdb = remove_residue(
                task.pdb_string,
                task.chain_id,
                task.residue_number,
                task.insertion_code,
            )

        relax_sep = task.position_relax in ("both", "unbound")
        relax_designer = _scan_designer if relax_sep else None

        ctx = _suppress_stderr() if task.quiet else contextlib.nullcontext()
        with ctx:
            dG = _compute_rosetta_dG(
                modified_pdb,
                _scan_relaxer,
                chain_pairs=task.chain_pairs,
                distance_cutoff=task.distance_cutoff,
                relax_separated=relax_sep or task.relax_separated,
                designer=relax_designer,
            )

        ddG = dG - task.dG_wt

        return ScanResult(
            scan_type=task.scan_type,
            chain_id=task.chain_id,
            residue_number=task.residue_number,
            insertion_code=task.insertion_code,
            residue_name=task.residue_name,
            dG=dG,
            ddG=ddG,
        )

    except Exception as exc:
        return ScanResult(
            scan_type=task.scan_type,
            chain_id=task.chain_id,
            residue_number=task.residue_number,
            insertion_code=task.insertion_code,
            residue_name=task.residue_name,
            error=f"{type(exc).__name__}: {exc}",
        )


# ------------------------------------------------------------------
# Pool factory
# ------------------------------------------------------------------


def get_pool(max_workers: int) -> ProcessPoolExecutor:
    """Create a ``ProcessPoolExecutor`` using the ``spawn`` context.

    The ``spawn`` start method avoids CUDA fork hazards on Linux and
    ensures each worker gets a clean Python interpreter.
    """
    ctx = multiprocessing.get_context("spawn")
    return ProcessPoolExecutor(
        max_workers=max_workers, mp_context=ctx
    )
