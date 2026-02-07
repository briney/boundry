"""Tests for boundry._parallel module."""

from unittest.mock import MagicMock, patch

import pytest

from boundry._parallel import (
    BranchResult,
    BranchTask,
    SnapshotData,
    StepResult,
    StepTask,
    _execute_branch_worker,
    _execute_step_worker,
    _init_scan_worker,
    _suppress_worker_warnings,
    get_pool,
)


# ------------------------------------------------------------------
# Task / Result dataclasses
# ------------------------------------------------------------------


class TestBranchTask:
    """Tests for BranchTask dataclass."""

    def test_frozen(self):
        task = BranchTask(
            candidate_pdb_string="ATOM\nEND\n",
            candidate_metadata={"dG": -5.0},
            candidate_source_path=None,
            steps=[("design", {"temperature": 0.1})],
            branch_seed=42,
        )
        with pytest.raises(AttributeError):
            task.branch_seed = 99

    def test_fields(self):
        task = BranchTask(
            candidate_pdb_string="ATOM\nEND\n",
            candidate_metadata={"dG": -5.0},
            candidate_source_path="/tmp/test.pdb",
            steps=[
                ("design", {"temperature": 0.1}),
                ("minimize", {}),
            ],
            branch_seed=42,
        )
        assert task.candidate_pdb_string == "ATOM\nEND\n"
        assert task.candidate_metadata == {"dG": -5.0}
        assert task.candidate_source_path == "/tmp/test.pdb"
        assert len(task.steps) == 2
        assert task.branch_seed == 42


class TestBranchResult:
    """Tests for BranchResult dataclass."""

    def test_defaults(self):
        result = BranchResult()
        assert result.pdb_string == ""
        assert result.metadata == {}
        assert result.source_path is None
        assert result.snapshots == []
        assert result.error is None

    def test_error_result(self):
        result = BranchResult(error="ValueError: bad input")
        assert result.error == "ValueError: bad input"


class TestStepTask:
    """Tests for StepTask dataclass."""

    def test_frozen(self):
        task = StepTask(
            pdb_string="ATOM\nEND\n",
            metadata={},
            source_path=None,
            operation="minimize",
            params={"constrained": True},
        )
        with pytest.raises(AttributeError):
            task.operation = "relax"


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_defaults(self):
        result = StepResult()
        assert result.pdb_string == ""
        assert result.metadata == {}
        assert result.source_path is None
        assert result.error is None


# ------------------------------------------------------------------
# Worker functions
# ------------------------------------------------------------------


class TestExecuteBranchWorker:
    """Tests for _execute_branch_worker."""

    @patch("boundry.workflow.Workflow._run_operation")
    def test_single_step_branch(self, mock_op):
        from boundry.operations import Structure

        mock_op.return_value = Structure(
            pdb_string="ATOM designed\nEND\n",
            metadata={"dG": -10.0},
        )

        task = BranchTask(
            candidate_pdb_string="ATOM\nEND\n",
            candidate_metadata={},
            candidate_source_path=None,
            steps=[("design", {"temperature": 0.1})],
            branch_seed=42,
        )
        result = _execute_branch_worker(task)

        assert result.error is None
        assert result.pdb_string == "ATOM designed\nEND\n"
        assert result.metadata.get("dG") == -10.0
        assert len(result.snapshots) == 1
        assert result.snapshots[0].operation == "design"

    @patch("boundry.workflow.Workflow._run_operation")
    def test_multi_step_branch(self, mock_op):
        from boundry.operations import Structure

        call_count = {"n": 0}

        def _dispatch(name, structure, params):
            call_count["n"] += 1
            return Structure(
                pdb_string=f"ATOM step {call_count['n']}\nEND\n",
                metadata={"step": call_count["n"]},
            )

        mock_op.side_effect = _dispatch

        task = BranchTask(
            candidate_pdb_string="ATOM\nEND\n",
            candidate_metadata={},
            candidate_source_path=None,
            steps=[
                ("design", {}),
                ("analyze_interface", {}),
            ],
            branch_seed=42,
        )
        result = _execute_branch_worker(task)

        assert result.error is None
        assert call_count["n"] == 2
        assert len(result.snapshots) == 2
        assert result.snapshots[0].operation == "design"
        assert result.snapshots[1].operation == "analyze_interface"

    @patch("boundry.workflow.Workflow._run_operation")
    def test_error_captured(self, mock_op):
        mock_op.side_effect = ValueError("test error")

        task = BranchTask(
            candidate_pdb_string="ATOM\nEND\n",
            candidate_metadata={},
            candidate_source_path=None,
            steps=[("design", {})],
            branch_seed=42,
        )
        result = _execute_branch_worker(task)

        assert result.error is not None
        assert "ValueError" in result.error
        assert "test error" in result.error


class TestExecuteStepWorker:
    """Tests for _execute_step_worker."""

    @patch("boundry.workflow.Workflow._run_operation")
    def test_success(self, mock_op):
        from boundry.operations import Structure

        mock_op.return_value = Structure(
            pdb_string="ATOM minimized\nEND\n",
            metadata={"final_energy": -50.0},
        )

        task = StepTask(
            pdb_string="ATOM\nEND\n",
            metadata={},
            source_path=None,
            operation="minimize",
            params={"constrained": True},
        )
        result = _execute_step_worker(task)

        assert result.error is None
        assert result.pdb_string == "ATOM minimized\nEND\n"
        assert result.metadata["final_energy"] == -50.0

    @patch("boundry.workflow.Workflow._run_operation")
    def test_error_captured(self, mock_op):
        mock_op.side_effect = RuntimeError("openmm crash")

        task = StepTask(
            pdb_string="ATOM\nEND\n",
            metadata={},
            source_path=None,
            operation="minimize",
            params={},
        )
        result = _execute_step_worker(task)

        assert result.error is not None
        assert "RuntimeError" in result.error


# ------------------------------------------------------------------
# Pool factory
# ------------------------------------------------------------------


class TestGetPool:
    """Tests for get_pool factory."""

    def test_creates_pool(self):
        pool = get_pool(2)
        try:
            assert pool._max_workers == 2
        finally:
            pool.shutdown(wait=False)

    def test_uses_spawn_context(self):
        pool = get_pool(1)
        try:
            ctx = pool._mp_context
            assert ctx.get_start_method() == "spawn"
        finally:
            pool.shutdown(wait=False)


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

    @patch("boundry._parallel._suppress_worker_warnings")
    @patch("boundry.workflow.Workflow._run_operation")
    def test_branch_worker_calls_suppress(self, mock_op, mock_suppress):
        from boundry.operations import Structure

        mock_op.return_value = Structure(
            pdb_string="ATOM\nEND\n", metadata={}
        )
        task = BranchTask(
            candidate_pdb_string="ATOM\nEND\n",
            candidate_metadata={},
            candidate_source_path=None,
            steps=[("minimize", {})],
            branch_seed=None,
        )
        _execute_branch_worker(task)
        mock_suppress.assert_called_once()

    @patch("boundry._parallel._suppress_worker_warnings")
    @patch("boundry.workflow.Workflow._run_operation")
    def test_step_worker_calls_suppress(self, mock_op, mock_suppress):
        from boundry.operations import Structure

        mock_op.return_value = Structure(
            pdb_string="ATOM\nEND\n", metadata={}
        )
        task = StepTask(
            pdb_string="ATOM\nEND\n",
            metadata={},
            source_path=None,
            operation="minimize",
            params={},
        )
        _execute_step_worker(task)
        mock_suppress.assert_called_once()

    @patch("boundry._parallel._suppress_worker_warnings")
    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.config.RelaxConfig")
    def test_init_scan_worker_calls_suppress(
        self, mock_config, mock_relaxer, mock_suppress
    ):
        _init_scan_worker(
            relax_config_dict={},
            design_config_dict=None,
        )
        mock_suppress.assert_called_once()
