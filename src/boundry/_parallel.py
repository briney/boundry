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
