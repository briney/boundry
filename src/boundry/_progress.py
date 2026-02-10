"""Rich-based progress monitoring for workflow execution."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Any, Optional

_VAR_RE = re.compile(r"\{([^}]+)\}")


def _extract_metric_names(condition: Optional[str]) -> list[str]:
    """Extract ``{var}`` references from a condition string."""
    if condition is None:
        return []
    return _VAR_RE.findall(condition)


@dataclass
class _BlockState:
    """Per-block state pushed onto the block stack."""

    task_id: Any
    metric_name: str = ""
    total: Optional[int] = None


class WorkflowProgress:
    """Context manager wrapping ``rich.progress.Progress``.

    Provides a three-level hierarchy:

    1. **Workflow** — top-level step counter
    2. **Block** — iterate cycle or beam round (stacked for nesting)
    3. **Inner** — branch expansion or single-step spinner (stacked)

    All public methods are safe to call unconditionally; when
    ``enabled=False`` (or non-TTY stderr) every method is a no-op.
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled and sys.stderr.isatty()
        self._progress = None
        self._workflow_task = None
        self._block_tasks: list[_BlockState] = []
        self._inner_tasks: list[Any] = []

    # -- context manager --------------------------------------------------

    def __enter__(self) -> "WorkflowProgress":
        if not self._enabled:
            return self

        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=35),
            MofNCompleteColumn(),
            TextColumn("{task.fields[status]}"),
            TimeElapsedColumn(),
            transient=False,
            console=self._make_console(),
        )
        self._progress.start()
        return self

    def __exit__(self, *args) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

    @staticmethod
    def _make_console():
        from rich.console import Console

        return Console(stderr=True)

    # -- workflow level ---------------------------------------------------

    def start_workflow(self, total_steps: int) -> None:
        if self._progress is None:
            return
        self._workflow_task = self._progress.add_task(
            "Workflow",
            total=total_steps,
            status="",
        )

    def advance_workflow(self, description: str) -> None:
        if self._progress is None or self._workflow_task is None:
            return
        self._progress.update(
            self._workflow_task,
            advance=1,
            status=description,
        )

    def update_workflow_status(self, description: str) -> None:
        if self._progress is None or self._workflow_task is None:
            return
        self._progress.update(self._workflow_task, status=description)

    def finish_workflow(self) -> None:
        if self._progress is None or self._workflow_task is None:
            return
        self._progress.update(
            self._workflow_task,
            status="done",
        )
        self._workflow_task = None

    # -- iterate block level ----------------------------------------------

    def start_iterate(
        self,
        total: int,
        convergence: bool,
        metric_name: str = "",
    ) -> None:
        if self._progress is None:
            return
        if convergence:
            task_id = self._progress.add_task(
                "  Cycle",
                total=None,
                status=f"0 / {total}",
            )
        else:
            task_id = self._progress.add_task(
                "  Cycle",
                total=total,
                status="",
            )
        self._block_tasks.append(
            _BlockState(
                task_id=task_id,
                metric_name=metric_name,
                total=total,
            )
        )

    def advance_iterate(
        self,
        cycle: int,
        metric_value: Optional[float] = None,
    ) -> None:
        if self._progress is None or not self._block_tasks:
            return

        state = self._block_tasks[-1]

        metric_text = ""
        if metric_value is not None and state.metric_name:
            metric_text = (
                f"  {state.metric_name}={metric_value:.4g}"
            )

        if (
            state.total is not None
            and self._progress._tasks[state.task_id].total is None
        ):
            # Convergence mode: spinner with "cycle N / max_n"
            self._progress.update(
                state.task_id,
                status=f"cycle {cycle} / {state.total}{metric_text}",
            )
        else:
            self._progress.update(
                state.task_id,
                advance=1,
                status=metric_text,
            )

    def finish_iterate(self) -> None:
        if self._progress is None or not self._block_tasks:
            return
        state = self._block_tasks.pop()
        self._progress.remove_task(state.task_id)

    # -- beam block level -------------------------------------------------

    def start_beam(self, total_rounds: int) -> None:
        if self._progress is None:
            return
        task_id = self._progress.add_task(
            "  Round",
            total=total_rounds,
            status="",
        )
        self._block_tasks.append(_BlockState(task_id=task_id))

    def advance_beam_round(
        self,
        round_num: int,
        best_metric: Optional[float] = None,
        metric_name: str = "",
    ) -> None:
        if self._progress is None or not self._block_tasks:
            return
        state = self._block_tasks[-1]
        status = ""
        if best_metric is not None and metric_name:
            status = f"best {metric_name}={best_metric:.4g}"
        self._progress.update(
            state.task_id,
            advance=1,
            status=status,
        )

    def finish_beam(self) -> None:
        if self._progress is None or not self._block_tasks:
            return
        state = self._block_tasks.pop()
        self._progress.remove_task(state.task_id)

    # -- inner level (branches / single-step spinner) ---------------------

    def start_branches(self, total: int) -> None:
        if self._progress is None:
            return
        task_id = self._progress.add_task(
            "    Branch",
            total=total,
            status="",
        )
        self._inner_tasks.append(task_id)

    def advance_branch(self) -> None:
        if self._progress is None or not self._inner_tasks:
            return
        self._progress.update(self._inner_tasks[-1], advance=1)

    def finish_branches(self) -> None:
        if self._progress is None or not self._inner_tasks:
            return
        task_id = self._inner_tasks.pop()
        self._progress.remove_task(task_id)

    def start_inner_step(self, description: str) -> None:
        if self._progress is None:
            return
        task_id = self._progress.add_task(
            f"    Step",
            total=None,
            status=description,
        )
        self._inner_tasks.append(task_id)

    def finish_inner_step(self) -> None:
        if self._progress is None or not self._inner_tasks:
            return
        task_id = self._inner_tasks.pop()
        self._progress.remove_task(task_id)
