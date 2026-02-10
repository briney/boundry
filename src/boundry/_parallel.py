"""Process-level parallelism for workflow execution.

Provides a shared ``WorkPool`` context manager wrapping a
``ProcessPoolExecutor`` with the ``spawn`` start method.  The pool is
created once at workflow start and torn down at the end, avoiding
repeated heavy-import overhead in worker processes.

Operations are the unit of parallelism — the main process submits
batches of ``OperationTask`` objects and waits at a barrier for all
results.  Scan tasks (per-position interface energetics) are also
submitted to the same shared pool.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ------------------------------------------------------------------
# WorkPool — shared process pool
# ------------------------------------------------------------------


class WorkPool:
    """Shared process pool for workflow parallelism.

    Context manager wrapping ``ProcessPoolExecutor(spawn)``.  Created
    once at ``Workflow.run()`` start, torn down at end.  Provides a
    ``map(worker_fn, tasks)`` method that submits a batch, collects
    results in original order, and raises on worker errors.

    When ``max_workers <= 1``, no pool is created and ``map()`` falls
    through to sequential execution.
    """

    def __init__(self, max_workers: int) -> None:
        self._max_workers = max_workers
        self._pool: Optional[ProcessPoolExecutor] = None

    @property
    def active(self) -> bool:
        """True if the pool is available for parallel dispatch."""
        return self._pool is not None

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def map(
        self,
        fn: Callable[[T], Any],
        tasks: List[T],
    ) -> List[Any]:
        """Submit all *tasks*, wait for completion, return ordered results.

        On any ``Future`` exception, cancels pending futures and raises
        a ``WorkflowError`` with the task index and exception context.
        """
        if self._pool is None:
            return [fn(task) for task in tasks]

        from boundry.workflow import WorkflowError

        total = len(tasks)
        results: List[Any] = [None] * total

        future_to_idx = {
            self._pool.submit(fn, task): idx
            for idx, task in enumerate(tasks)
        }

        try:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    # Cancel remaining futures
                    for f in future_to_idx:
                        f.cancel()
                    raise WorkflowError(
                        f"Parallel task {idx + 1}/{total} "
                        f"failed: {type(exc).__name__}: {exc}"
                    ) from exc
        except WorkflowError:
            raise
        except Exception as exc:
            raise WorkflowError(
                f"Parallel execution failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        return results

    def submit(
        self,
        fn: Callable,
        *args: Any,
    ):
        """Submit a single task to the pool. Returns a Future."""
        if self._pool is None:
            raise RuntimeError(
                "WorkPool is not active (max_workers <= 1)"
            )
        return self._pool.submit(fn, *args)

    def __enter__(self) -> "WorkPool":
        if self._max_workers > 1:
            ctx = multiprocessing.get_context("spawn")
            self._pool = ProcessPoolExecutor(
                max_workers=self._max_workers, mp_context=ctx
            )
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


# ------------------------------------------------------------------
# Serializable task / result types
# ------------------------------------------------------------------


@dataclass(frozen=True)
class OperationTask:
    """Serializable inputs for one operation on one structure.

    Replaces both ``StepTask`` and ``BranchTask``.  One task = one
    operation on one structure.
    """

    pdb_string: str
    metadata: Dict[str, Any]
    source_path: Optional[str]
    operation: str
    params: Dict[str, Any]


@dataclass
class OperationResult:
    """Serializable outputs from one operation execution."""

    pdb_string: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None
    error: Optional[str] = None


# ------------------------------------------------------------------
# Worker warning suppression
# ------------------------------------------------------------------


def _suppress_worker_warnings() -> None:
    """Suppress noisy dependency warnings in spawned worker processes.

    Worker processes start fresh without the warning filters configured
    by ``cli._setup_logging``.  This applies the same filters so that
    dependency warnings (simtk deprecation, torch tensor indexing, etc.)
    are silenced in workers too.
    """
    import warnings

    warnings.filterwarnings(
        "ignore",
        module=r"(openmm|pdbfixer|Bio|freesasa|torch|absl|openfold|simtk)",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*simtk.*",
        category=DeprecationWarning,
    )

    # Suppress ProDy logger (uses '.prody' name with dot prefix).
    # Setting propagate=False here persists through ProDy's PackageLogger
    # __init__, which resets the logger level but not propagation.
    prody_logger = logging.getLogger(".prody")
    prody_logger.propagate = False


# ------------------------------------------------------------------
# Worker functions (top-level, pickle-safe targets)
# ------------------------------------------------------------------


def _execute_operation_worker(task: OperationTask) -> OperationResult:
    """Execute a single operation on one structure in a worker.

    This is the top-level function submitted to the process pool.
    It imports all dependencies inside the function body to work
    correctly with the ``spawn`` start method.
    """
    _suppress_worker_warnings()
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

        return OperationResult(
            pdb_string=result.pdb_string,
            metadata=dict(result.metadata),
            source_path=result.source_path,
        )

    except Exception as exc:
        return OperationResult(error=f"{type(exc).__name__}: {exc}")


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
    relax_config_dict: Dict[str, Any] = field(default_factory=dict)
    design_config_dict: Optional[Dict[str, Any]] = None


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


# Module-level cache for worker-process reuse (keyed by config fingerprint)
_worker_cache: Dict[str, Any] = {}


def _config_fingerprint(d: Dict[str, Any]) -> str:
    """Deterministic string hash of a config dict for cache invalidation."""
    items = []
    for k in sorted(d.keys()):
        items.append(f"{k}={d[k]!r}")
    return "|".join(items)


def _execute_scan_worker(task: ScanTask) -> ScanResult:
    """Execute a single scan task in a worker process.

    Uses config-fingerprint-keyed lazy caching of Relaxer/Designer
    instances so they are only created once per unique config per
    worker.
    """
    _suppress_worker_warnings()
    try:
        import contextlib

        from boundry.interface_position_energetics import (
            _compute_rosetta_dG,
            mutate_to_alanine,
            remove_residue,
        )
        from boundry.utils import suppress_stderr as _suppress_stderr

        # Lazy-init Relaxer
        relax_key = _config_fingerprint(task.relax_config_dict)
        if _worker_cache.get("relax_key") != relax_key:
            from boundry.config import RelaxConfig
            from boundry.relaxer import Relaxer

            _worker_cache["relaxer"] = Relaxer(
                RelaxConfig(**task.relax_config_dict)
            )
            _worker_cache["relax_key"] = relax_key

        # Lazy-init Designer (if needed)
        design_key = (
            _config_fingerprint(task.design_config_dict)
            if task.design_config_dict
            else None
        )
        if (
            design_key
            and _worker_cache.get("design_key") != design_key
        ):
            from boundry.config import DesignConfig
            from boundry.designer import Designer

            _worker_cache["designer"] = Designer(
                DesignConfig(**task.design_config_dict)
            )
            _worker_cache["design_key"] = design_key

        relaxer = _worker_cache["relaxer"]
        designer = _worker_cache.get("designer")

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
        relax_designer = designer if relax_sep else None

        ctx = _suppress_stderr() if task.quiet else contextlib.nullcontext()
        with ctx:
            dG = _compute_rosetta_dG(
                modified_pdb,
                relaxer,
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
