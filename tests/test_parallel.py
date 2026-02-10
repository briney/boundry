"""Tests for boundry._parallel module."""

from unittest.mock import MagicMock, patch

import pytest

from boundry._parallel import (
    OperationResult,
    OperationTask,
    WorkPool,
    _execute_operation_worker,
    _suppress_worker_warnings,
)


def _double(x):
    """Top-level function for pickling in spawn workers."""
    return x * 2


# ------------------------------------------------------------------
# Task / Result dataclasses
# ------------------------------------------------------------------


class TestOperationTask:
    """Tests for OperationTask dataclass."""

    def test_frozen(self):
        task = OperationTask(
            pdb_string="ATOM\nEND\n",
            metadata={"dG": -5.0},
            source_path=None,
            operation="design",
            params={"temperature": 0.1},
        )
        with pytest.raises(AttributeError):
            task.operation = "relax"

    def test_fields(self):
        task = OperationTask(
            pdb_string="ATOM\nEND\n",
            metadata={"dG": -5.0},
            source_path="/tmp/test.pdb",
            operation="minimize",
            params={"constrained": True},
        )
        assert task.pdb_string == "ATOM\nEND\n"
        assert task.metadata == {"dG": -5.0}
        assert task.source_path == "/tmp/test.pdb"
        assert task.operation == "minimize"
        assert task.params == {"constrained": True}


class TestOperationResult:
    """Tests for OperationResult dataclass."""

    def test_defaults(self):
        result = OperationResult()
        assert result.pdb_string == ""
        assert result.metadata == {}
        assert result.source_path is None
        assert result.error is None

    def test_error_result(self):
        result = OperationResult(error="ValueError: bad input")
        assert result.error == "ValueError: bad input"


# ------------------------------------------------------------------
# WorkPool
# ------------------------------------------------------------------


class TestWorkPool:
    """Tests for WorkPool context manager."""

    def test_sequential_when_workers_one(self):
        """WorkPool with max_workers=1 has no active pool."""
        with WorkPool(1) as pool:
            assert not pool.active
            assert pool.max_workers == 1

    def test_map_sequential_fallback(self):
        """map() runs sequentially when pool is not active."""
        with WorkPool(1) as pool:
            results = pool.map(
                lambda x: x * 2, [1, 2, 3]
            )
            assert results == [2, 4, 6]

    def test_active_when_workers_gt_1(self):
        """WorkPool with max_workers>1 creates a pool."""
        with WorkPool(2) as pool:
            assert pool.active

    def test_map_preserves_order(self):
        """map() returns results in input order."""
        with WorkPool(2) as pool:
            results = pool.map(_double, [1, 2, 3, 4])
            assert results == [2, 4, 6, 8]

    def test_pool_cleaned_up_on_exit(self):
        """Pool is shut down after context exit."""
        pool = WorkPool(2)
        pool.__enter__()
        assert pool.active
        pool.__exit__(None, None, None)
        assert not pool.active

    def test_submit_raises_when_no_pool(self):
        """submit() raises when pool is not active."""
        with WorkPool(1) as pool:
            with pytest.raises(RuntimeError, match="not active"):
                pool.submit(lambda: None)


# ------------------------------------------------------------------
# Worker functions
# ------------------------------------------------------------------


class TestExecuteOperationWorker:
    """Tests for _execute_operation_worker."""

    @patch("boundry.workflow.Workflow._run_operation")
    def test_success(self, mock_op):
        from boundry.operations import Structure

        mock_op.return_value = Structure(
            pdb_string="ATOM minimized\nEND\n",
            metadata={"final_energy": -50.0},
        )

        task = OperationTask(
            pdb_string="ATOM\nEND\n",
            metadata={},
            source_path=None,
            operation="minimize",
            params={"constrained": True},
        )
        result = _execute_operation_worker(task)

        assert result.error is None
        assert result.pdb_string == "ATOM minimized\nEND\n"
        assert result.metadata["final_energy"] == -50.0

    @patch("boundry.workflow.Workflow._run_operation")
    def test_error_captured(self, mock_op):
        mock_op.side_effect = RuntimeError("openmm crash")

        task = OperationTask(
            pdb_string="ATOM\nEND\n",
            metadata={},
            source_path=None,
            operation="minimize",
            params={},
        )
        result = _execute_operation_worker(task)

        assert result.error is not None
        assert "RuntimeError" in result.error

    @patch("boundry._parallel._suppress_worker_warnings")
    @patch("boundry.workflow.Workflow._run_operation")
    def test_worker_calls_suppress(self, mock_op, mock_suppress):
        from boundry.operations import Structure

        mock_op.return_value = Structure(
            pdb_string="ATOM\nEND\n", metadata={}
        )
        task = OperationTask(
            pdb_string="ATOM\nEND\n",
            metadata={},
            source_path=None,
            operation="minimize",
            params={},
        )
        _execute_operation_worker(task)
        mock_suppress.assert_called_once()


# ------------------------------------------------------------------
# Worker warning suppression
# ------------------------------------------------------------------


class TestSuppressWorkerWarnings:
    """Tests for _suppress_worker_warnings."""

    def test_sets_warning_filters(self):
        import warnings

        original_filters = warnings.filters[:]
        try:
            _suppress_worker_warnings()
            # Should have added at least 2 filters
            assert len(warnings.filters) >= len(original_filters) + 2
        finally:
            warnings.filters[:] = original_filters
