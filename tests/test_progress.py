"""Tests for boundry._progress.WorkflowProgress."""

import sys
from unittest.mock import patch

import pytest

from boundry._progress import WorkflowProgress, _extract_metric_names


# ------------------------------------------------------------------
# _extract_metric_names
# ------------------------------------------------------------------


class TestExtractMetricNames:
    def test_simple_variable(self):
        assert _extract_metric_names("{dG}") == ["dG"]

    def test_multiple_variables(self):
        assert _extract_metric_names("delta({dG}) < {threshold}") == [
            "dG",
            "threshold",
        ]

    def test_dotted_path(self):
        assert _extract_metric_names("{energy.total}") == [
            "energy.total"
        ]

    def test_no_variables(self):
        assert _extract_metric_names("42 < 100") == []

    def test_none_condition(self):
        assert _extract_metric_names(None) == []


# ------------------------------------------------------------------
# WorkflowProgress disabled (no-op) behaviour
# ------------------------------------------------------------------


class TestWorkflowProgressDisabled:
    """All methods must be safe to call when disabled."""

    def test_disabled_explicitly(self):
        p = WorkflowProgress(enabled=False)
        with p:
            p.start_workflow(3)
            p.advance_workflow("idealize")
            p.finish_workflow()

    def test_disabled_all_iterate_methods(self):
        p = WorkflowProgress(enabled=False)
        with p:
            p.start_iterate(10, convergence=True, metric_name="dG")
            p.advance_iterate(1, metric_value=-5.0)
            p.finish_iterate()

    def test_disabled_all_beam_methods(self):
        p = WorkflowProgress(enabled=False)
        with p:
            p.start_beam(5)
            p.advance_beam_round(1, best_metric=-12.0, metric_name="dG")
            p.finish_beam()

    def test_disabled_all_branch_methods(self):
        p = WorkflowProgress(enabled=False)
        with p:
            p.start_branches(10)
            p.advance_branch()
            p.finish_branches()

    def test_disabled_inner_step(self):
        p = WorkflowProgress(enabled=False)
        with p:
            p.start_inner_step("minimizing")
            p.finish_inner_step()

    def test_disabled_update_workflow_status(self):
        p = WorkflowProgress(enabled=False)
        with p:
            p.start_workflow(3)
            p.update_workflow_status("idealize")
            p.finish_workflow()

    def test_methods_safe_without_context_manager(self):
        """Methods should not raise even outside a context manager."""
        p = WorkflowProgress(enabled=False)
        p.start_workflow(3)
        p.update_workflow_status("test")
        p.advance_workflow("test")
        p.finish_workflow()
        p.start_iterate(5, False)
        p.advance_iterate(1)
        p.finish_iterate()
        p.start_beam(3)
        p.advance_beam_round(1)
        p.finish_beam()
        p.start_branches(5)
        p.advance_branch()
        p.finish_branches()
        p.start_inner_step("x")
        p.finish_inner_step()


# ------------------------------------------------------------------
# WorkflowProgress enabled (with mocked TTY)
# ------------------------------------------------------------------


class TestWorkflowProgressEnabled:
    """Test with a fake TTY to exercise the rich.progress path."""

    @pytest.fixture()
    def progress(self):
        """Create an enabled progress with mocked TTY."""
        with patch.object(
            sys, "stderr", wraps=sys.stderr
        ) as mock_stderr:
            mock_stderr.isatty = lambda: True
            p = WorkflowProgress(enabled=True)
            with p:
                yield p

    def test_workflow_lifecycle(self, progress):
        progress.start_workflow(3)
        assert progress._workflow_task is not None

        # Mirrors real usage: set status before execution, advance after
        progress.update_workflow_status("idealize")
        progress.advance_workflow("idealize")
        progress.update_workflow_status("iterate")
        progress.advance_workflow("iterate")
        progress.update_workflow_status("beam")
        progress.advance_workflow("beam")
        progress.finish_workflow()
        assert progress._workflow_task is None

    def test_update_workflow_status_does_not_advance(self, progress):
        progress.start_workflow(3)
        task = progress._progress._tasks[progress._workflow_task]
        assert task.completed == 0

        progress.update_workflow_status("idealize")

        # Counter should remain at 0 — only status text changes
        task = progress._progress._tasks[progress._workflow_task]
        assert task.completed == 0
        assert task.fields["status"] == "idealize"
        progress.finish_workflow()

    def test_iterate_fixed_n(self, progress):
        progress.start_iterate(5, convergence=False)
        assert progress._block_task is not None
        for i in range(1, 6):
            progress.advance_iterate(i)
        progress.finish_iterate()
        assert progress._block_task is None

    def test_iterate_convergence(self, progress):
        progress.start_iterate(
            100, convergence=True, metric_name="dG"
        )
        assert progress._block_task is not None
        progress.advance_iterate(1, metric_value=-8.0)
        progress.advance_iterate(2, metric_value=-9.5)
        progress.finish_iterate()
        assert progress._block_task is None

    def test_beam_lifecycle(self, progress):
        progress.start_beam(5)
        assert progress._block_task is not None
        for r in range(1, 6):
            progress.advance_beam_round(
                r, best_metric=-10.0 - r, metric_name="dG"
            )
        progress.finish_beam()
        assert progress._block_task is None

    def test_branch_lifecycle(self, progress):
        progress.start_branches(10)
        assert progress._inner_task is not None
        for _ in range(10):
            progress.advance_branch()
        progress.finish_branches()
        assert progress._inner_task is None

    def test_inner_step_lifecycle(self, progress):
        progress.start_inner_step("minimizing")
        assert progress._inner_task is not None
        progress.finish_inner_step()
        assert progress._inner_task is None

    def test_advance_beam_round_no_metric(self, progress):
        """advance_beam_round without metric should not raise."""
        progress.start_beam(3)
        progress.advance_beam_round(1)
        progress.advance_beam_round(2, best_metric=None, metric_name="")
        progress.finish_beam()

    def test_advance_iterate_no_metric(self, progress):
        """advance_iterate without metric should not raise."""
        progress.start_iterate(
            5, convergence=False, metric_name=""
        )
        progress.advance_iterate(1, metric_value=None)
        progress.finish_iterate()


# ------------------------------------------------------------------
# TTY detection
# ------------------------------------------------------------------


class TestTTYDetection:
    def test_non_tty_disables_progress(self):
        """When stderr is not a TTY, progress should be disabled."""
        with patch.object(
            sys, "stderr", wraps=sys.stderr
        ) as mock_stderr:
            mock_stderr.isatty = lambda: False
            p = WorkflowProgress(enabled=True)
            # _enabled should be False because isatty() is False
            assert not p._enabled

    def test_tty_enables_progress(self):
        """When stderr is a TTY, progress should be enabled."""
        with patch.object(
            sys, "stderr", wraps=sys.stderr
        ) as mock_stderr:
            mock_stderr.isatty = lambda: True
            p = WorkflowProgress(enabled=True)
            assert p._enabled
