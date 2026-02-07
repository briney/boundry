"""Rich-based progress monitoring for workflow execution."""

from __future__ import annotations

import re
import sys
from typing import Optional

_VAR_RE = re.compile(r"\{([^}]+)\}")


def _extract_metric_names(condition: Optional[str]) -> list[str]:
    """Extract ``{var}`` references from a condition string."""
    if condition is None:
        return []
    return _VAR_RE.findall(condition)


class WorkflowProgress:
    """Context manager wrapping ``rich.progress.Progress``.

    Provides a three-level hierarchy:

    1. **Workflow** — top-level step counter
    2. **Block** — iterate cycle or beam round
    3. **Inner** — branch expansion or single-step spinner

    All public methods are safe to call unconditionally; when
    ``enabled=False`` (or non-TTY stderr) every method is a no-op.
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled and sys.stderr.isatty()
        self._progress = None
        self._workflow_task = None
        self._block_task = None
        self._inner_task = None

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
            # Indeterminate: show spinner + "cycle N / max_n"
            self._block_task = self._progress.add_task(
                "  Cycle",
                total=None,
                status=f"0 / {total}",
            )
        else:
            self._block_task = self._progress.add_task(
                "  Cycle",
                total=total,
                status="",
            )
        self._block_metric_name = metric_name
        self._block_total = total

    def advance_iterate(
        self,
        cycle: int,
        metric_value: Optional[float] = None,
    ) -> None:
        if self._progress is None or self._block_task is None:
            return

        metric_text = ""
        if (
            metric_value is not None
            and hasattr(self, "_block_metric_name")
            and self._block_metric_name
        ):
            metric_text = (
                f"  {self._block_metric_name}={metric_value:.4g}"
            )

        total = getattr(self, "_block_total", None)
        if total is not None and self._progress._tasks[
            self._block_task
        ].total is None:
            # Convergence mode: spinner with "cycle N / max_n"
            self._progress.update(
                self._block_task,
                status=f"cycle {cycle} / {total}{metric_text}",
            )
        else:
            self._progress.update(
                self._block_task,
                advance=1,
                status=metric_text,
            )

    def finish_iterate(self) -> None:
        if self._progress is None or self._block_task is None:
            return
        self._progress.remove_task(self._block_task)
        self._block_task = None

    # -- beam block level -------------------------------------------------

    def start_beam(self, total_rounds: int) -> None:
        if self._progress is None:
            return
        self._block_task = self._progress.add_task(
            "  Round",
            total=total_rounds,
            status="",
        )

    def advance_beam_round(
        self,
        round_num: int,
        best_metric: Optional[float] = None,
        metric_name: str = "",
    ) -> None:
        if self._progress is None or self._block_task is None:
            return
        status = ""
        if best_metric is not None and metric_name:
            status = f"best {metric_name}={best_metric:.4g}"
        self._progress.update(
            self._block_task,
            advance=1,
            status=status,
        )

    def finish_beam(self) -> None:
        if self._progress is None or self._block_task is None:
            return
        self._progress.remove_task(self._block_task)
        self._block_task = None

    # -- inner level (branches / single-step spinner) ---------------------

    def start_branches(self, total: int) -> None:
        if self._progress is None:
            return
        self._inner_task = self._progress.add_task(
            "    Branch",
            total=total,
            status="",
        )

    def advance_branch(self) -> None:
        if self._progress is None or self._inner_task is None:
            return
        self._progress.update(self._inner_task, advance=1)

    def finish_branches(self) -> None:
        if self._progress is None or self._inner_task is None:
            return
        self._progress.remove_task(self._inner_task)
        self._inner_task = None

    def start_inner_step(self, description: str) -> None:
        if self._progress is None:
            return
        self._inner_task = self._progress.add_task(
            f"    Step",
            total=None,
            status=description,
        )

    def finish_inner_step(self) -> None:
        if self._progress is None or self._inner_task is None:
            return
        self._progress.remove_task(self._inner_task)
        self._inner_task = None
