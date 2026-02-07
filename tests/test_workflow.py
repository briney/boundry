"""Tests for boundry.workflow module."""

import json
from unittest.mock import MagicMock, patch

import pytest
import yaml

from boundry.config import (
    BeamBlock,
    IterateBlock,
    WorkflowConfig,
    WorkflowStep,
)
from boundry.workflow import (
    VALID_OPERATIONS,
    OutputPathContext,
    Workflow,
    WorkflowError,
    _NATIVE_METRIC_KEYS,
    _OPERATION_OUTPUT_SPECS,
)


# ------------------------------------------------------------------
# YAML parsing
# ------------------------------------------------------------------


class TestFromYaml:
    """Tests for Workflow.from_yaml()."""

    def test_minimal_workflow(self, tmp_path):
        """Test loading a minimal valid workflow."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": "input.pdb",
                    "steps": [{"operation": "idealize"}],
                }
            )
        )
        wf = Workflow.from_yaml(wf_file)
        assert wf.config.input == "input.pdb"
        assert wf.config.project_path is None
        assert len(wf.config.steps) == 1
        assert wf.config.steps[0].operation == "idealize"
        assert wf.config.steps[0].params == {}

    def test_full_workflow(self, tmp_path):
        """Test loading a workflow with project_path and params."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": "input.pdb",
                    "project_path": "results",
                    "steps": [
                        {
                            "operation": "idealize",
                            "params": {"fix_cis_omega": True},
                        },
                        {
                            "operation": "minimize",
                            "params": {
                                "constrained": False,
                                "max_iterations": 1000,
                            },
                        },
                    ],
                }
            )
        )
        wf = Workflow.from_yaml(wf_file)
        assert wf.config.input == "input.pdb"
        assert wf.config.project_path == "results"
        assert len(wf.config.steps) == 2

        step1 = wf.config.steps[0]
        assert step1.operation == "idealize"
        assert step1.params == {"fix_cis_omega": True}

        step2 = wf.config.steps[1]
        assert step2.operation == "minimize"
        assert step2.params["constrained"] is False
        assert step2.params["max_iterations"] == 1000

    def test_null_params_treated_as_empty(self, tmp_path):
        """Test that null params in YAML become an empty dict."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "steps:\n"
            "  - operation: idealize\n"
            "    params:\n"
        )
        wf = Workflow.from_yaml(wf_file)
        assert wf.config.steps[0].params == {}

    def test_all_valid_operations(self, tmp_path):
        """Test that all valid operations can be loaded."""
        steps = [{"operation": op} for op in sorted(VALID_OPERATIONS)]
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump({"input": "input.pdb", "steps": steps})
        )
        wf = Workflow.from_yaml(wf_file)
        assert len(wf.config.steps) == len(VALID_OPERATIONS)

    def test_deprecated_output_key_treated_as_project_path(
        self, tmp_path
    ):
        """Old 'output' key is treated as project_path with warning."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "output: results\n"
            "steps:\n"
            "  - operation: idealize\n"
        )
        with pytest.warns(DeprecationWarning, match="output.*deprecated"):
            wf = Workflow.from_yaml(wf_file)
        assert wf.config.project_path == "results"

    def test_step_output_key_rejected(self, tmp_path):
        """Per-step output key is no longer allowed."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": "input.pdb",
                    "steps": [
                        {
                            "operation": "idealize",
                            "output": "idealized.pdb",
                        }
                    ],
                }
            )
        )
        with pytest.raises(WorkflowError, match="unknown fields"):
            Workflow.from_yaml(wf_file)


# ------------------------------------------------------------------
# Validation errors
# ------------------------------------------------------------------


class TestValidationErrors:
    """Tests for YAML parsing and workflow validation errors."""

    def test_non_mapping_yaml(self, tmp_path):
        """Test error for non-mapping YAML."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text("- just a list\n")
        with pytest.raises(WorkflowError, match="mapping"):
            Workflow.from_yaml(wf_file)

    def test_missing_input(self, tmp_path):
        """Test error when 'input' field is missing."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump({"steps": [{"operation": "idealize"}]})
        )
        with pytest.raises(WorkflowError, match="input"):
            Workflow.from_yaml(wf_file)

    def test_missing_steps(self, tmp_path):
        """Test error when 'steps' field is missing."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(yaml.dump({"input": "in.pdb"}))
        with pytest.raises(WorkflowError, match="steps"):
            Workflow.from_yaml(wf_file)

    def test_empty_steps(self, tmp_path):
        """Test error when 'steps' list is empty."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump({"input": "in.pdb", "steps": []})
        )
        with pytest.raises(WorkflowError, match="steps"):
            Workflow.from_yaml(wf_file)

    def test_step_not_mapping(self, tmp_path):
        """Test error when a step is not a mapping."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump({"input": "in.pdb", "steps": ["idealize"]})
        )
        with pytest.raises(WorkflowError, match="Step 1.*mapping"):
            Workflow.from_yaml(wf_file)

    def test_step_missing_operation(self, tmp_path):
        """Test error when a step lacks a valid step key."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": "in.pdb",
                    "steps": [{"params": {"foo": 1}}],
                }
            )
        )
        with pytest.raises(
            WorkflowError, match="Step 1.*expected one of"
        ):
            Workflow.from_yaml(wf_file)

    def test_unknown_operation(self, tmp_path):
        """Test error for an unknown operation name."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": "in.pdb",
                    "steps": [{"operation": "foobar"}],
                }
            )
        )
        with pytest.raises(WorkflowError, match="unknown.*foobar"):
            Workflow.from_yaml(wf_file)


# ------------------------------------------------------------------
# Direct construction
# ------------------------------------------------------------------


class TestDirectConstruction:
    """Tests for constructing Workflow from WorkflowConfig."""

    def test_construct_from_config(self):
        """Test creating a Workflow from a WorkflowConfig directly."""
        config = WorkflowConfig(
            input="input.pdb",
            project_path="results",
            steps=[
                WorkflowStep(operation="idealize"),
                WorkflowStep(
                    operation="minimize",
                    params={"constrained": True},
                ),
            ],
        )
        wf = Workflow(config)
        assert wf.config is config

    def test_construct_with_bad_operation_raises(self):
        """Test that validation catches unknown operations."""
        config = WorkflowConfig(
            input="input.pdb",
            steps=[WorkflowStep(operation="bogus")],
        )
        with pytest.raises(WorkflowError, match="unknown.*bogus"):
            Workflow(config)


class TestBlockParsing:
    """Tests for iterate and beam parsing."""

    def test_parse_iterate_block(self, tmp_path):
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": "input.pdb",
                    "steps": [
                        {
                            "iterate": {
                                "n": 3,
                                "steps": [{"operation": "relax"}],
                            }
                        }
                    ],
                }
            )
        )
        wf = Workflow.from_yaml(wf_file)
        assert isinstance(wf.config.steps[0], IterateBlock)
        block = wf.config.steps[0]
        assert block.n == 3
        assert len(block.steps) == 1

    def test_parse_beam_block(self, tmp_path):
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": "input.pdb",
                    "steps": [
                        {
                            "beam": {
                                "width": 2,
                                "rounds": 3,
                                "metric": "dG",
                                "direction": "min",
                                "steps": [
                                    {"operation": "analyze_interface"}
                                ],
                            }
                        }
                    ],
                }
            )
        )
        wf = Workflow.from_yaml(wf_file)
        assert isinstance(wf.config.steps[0], BeamBlock)
        block = wf.config.steps[0]
        assert block.width == 2
        assert block.rounds == 3

    def test_unknown_block_key_raises(self, tmp_path):
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": "input.pdb",
                    "steps": [{"unknown": {"steps": []}}],
                }
            )
        )
        with pytest.raises(WorkflowError, match="expected one of"):
            Workflow.from_yaml(wf_file)

    def test_iterate_output_key_rejected(self, tmp_path):
        """Per-block output key is no longer allowed."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": "input.pdb",
                    "steps": [
                        {
                            "iterate": {
                                "n": 3,
                                "output": "results/",
                                "steps": [{"operation": "relax"}],
                            }
                        }
                    ],
                }
            )
        )
        with pytest.raises(WorkflowError, match="unknown fields"):
            Workflow.from_yaml(wf_file)

    def test_beam_output_key_rejected(self, tmp_path):
        """Per-block output key is no longer allowed."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": "input.pdb",
                    "steps": [
                        {
                            "beam": {
                                "width": 2,
                                "rounds": 1,
                                "metric": "dG",
                                "output": "results/",
                                "steps": [
                                    {"operation": "analyze_interface"}
                                ],
                            }
                        }
                    ],
                }
            )
        )
        with pytest.raises(WorkflowError, match="unknown fields"):
            Workflow.from_yaml(wf_file)


# ------------------------------------------------------------------
# Workflow execution
# ------------------------------------------------------------------


class TestWorkflowRun:
    """Tests for Workflow.run() with mocked operations."""

    def _make_workflow(self, tmp_path, steps, project_path=None):
        """Helper to create a workflow YAML and return Workflow."""
        wf_file = tmp_path / "wf.yaml"
        data = {
            "input": str(tmp_path / "input.pdb"),
            "steps": steps,
        }
        if project_path is not None:
            data["project_path"] = str(tmp_path / project_path)
        wf_file.write_text(yaml.dump(data))
        return Workflow.from_yaml(wf_file)

    def _make_input(self, tmp_path):
        """Write a dummy PDB file and return path."""
        pdb = tmp_path / "input.pdb"
        pdb.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000"
            "  1.00  0.00           N\nEND\n"
        )
        return pdb

    @patch("boundry.workflow.Workflow._run_operation")
    def test_single_step(self, mock_op, tmp_path):
        """Test running a single-step workflow."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        mock_op.return_value = Structure(
            pdb_string="ATOM idealized\nEND\n"
        )

        wf = self._make_workflow(
            tmp_path, [{"operation": "idealize"}]
        )
        result = wf.run()

        mock_op.assert_called_once()
        assert result.pdb_string == "ATOM idealized\nEND\n"

    @patch("boundry.workflow.Workflow._run_operation")
    def test_multi_step_chaining(self, mock_op, tmp_path):
        """Test that output of step 1 feeds into step 2."""
        from boundry.operations import Structure

        self._make_input(tmp_path)

        struct_after_ideal = Structure(
            pdb_string="ATOM idealized\nEND\n"
        )
        struct_after_min = Structure(
            pdb_string="ATOM minimized\nEND\n"
        )

        def _dispatch(name, structure, params):
            if name == "idealize":
                return struct_after_ideal
            return struct_after_min

        mock_op.side_effect = _dispatch

        wf = self._make_workflow(
            tmp_path,
            [
                {"operation": "idealize"},
                {"operation": "minimize"},
            ],
        )
        result = wf.run()

        # Step 2 should receive output of step 1
        min_call = [
            c
            for c in mock_op.call_args_list
            if c[0][0] == "minimize"
        ][0]
        assert (
            min_call[0][1].pdb_string
            == struct_after_ideal.pdb_string
        )
        assert result.pdb_string == struct_after_min.pdb_string

    @patch("boundry.workflow.Workflow._run_operation")
    def test_output_written_to_project_path(
        self, mock_op, tmp_path
    ):
        """Test that step output is written to project_path."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        mock_op.return_value = Structure(
            pdb_string=(
                "ATOM      1  N   ALA A   1       "
                "1.000   1.000   1.000  1.00  0.00"
                "           N\nEND\n"
            )
        )

        wf = self._make_workflow(
            tmp_path,
            [{"operation": "idealize"}],
            project_path="results",
        )
        wf.run()

        # Check that 0.idealize/ directory was created
        output_dir = tmp_path / "results" / "0.idealize"
        assert output_dir.exists()
        pdb_files = list(output_dir.glob("*.pdb"))
        assert len(pdb_files) == 1
        assert pdb_files[0].name == "idealized.pdb"

    @patch("boundry.workflow.Workflow._run_operation")
    def test_default_project_path_is_cwd(
        self, mock_op, tmp_path, monkeypatch
    ):
        """Workflow without project_path writes to cwd."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        monkeypatch.chdir(tmp_path)
        mock_op.return_value = Structure(
            pdb_string="ATOM\nEND\n"
        )

        wf = self._make_workflow(
            tmp_path, [{"operation": "idealize"}]
        )
        wf.run()

        # Should have created 0.idealize/ in cwd
        output_dir = tmp_path / "0.idealize"
        assert output_dir.exists()

    def test_nonexistent_input_raises(self, tmp_path):
        """Test that a missing input file raises WorkflowError."""
        wf = self._make_workflow(
            tmp_path, [{"operation": "idealize"}]
        )
        # input.pdb was never created
        with pytest.raises(WorkflowError, match="not found"):
            wf.run()

    @patch("boundry.workflow.Workflow._run_operation")
    def test_params_passed_to_handler(self, mock_op, tmp_path):
        """Test that step params are forwarded to the handler."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        mock_op.return_value = Structure(
            pdb_string="ATOM\nEND\n"
        )

        wf = self._make_workflow(
            tmp_path,
            [
                {
                    "operation": "idealize",
                    "params": {"fix_cis_omega": False},
                }
            ],
        )
        wf.run()

        call_args = mock_op.call_args
        params = call_args[0][2]  # third positional arg
        assert params == {"fix_cis_omega": False}


class TestCompoundExecution:
    """Tests for iterate/beam workflow execution."""

    def _make_workflow(self, tmp_path, steps, project_path=None):
        wf_file = tmp_path / "wf.yaml"
        data = {
            "input": str(tmp_path / "input.pdb"),
            "steps": steps,
        }
        if project_path is not None:
            data["project_path"] = str(tmp_path / project_path)
        wf_file.write_text(yaml.dump(data))
        return Workflow.from_yaml(wf_file)

    def _make_workflow_with_seed(
        self, tmp_path, steps, seed, project_path=None
    ):
        wf_file = tmp_path / "wf.yaml"
        data = {
            "input": str(tmp_path / "input.pdb"),
            "seed": seed,
            "steps": steps,
        }
        if project_path is not None:
            data["project_path"] = str(tmp_path / project_path)
        wf_file.write_text(yaml.dump(data))
        return Workflow.from_yaml(wf_file)

    def _make_input(self, tmp_path):
        pdb = tmp_path / "input.pdb"
        pdb.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000"
            "  1.00  0.00           N\nEND\n"
        )
        return pdb

    @patch("boundry.workflow.Workflow._run_operation")
    def test_iterate_fixed_count_runs_n_times(
        self, mock_op, tmp_path
    ):
        from boundry.operations import Structure

        self._make_input(tmp_path)
        call_count = {"n": 0}

        def _side_effect(name, structure, params):
            call_count["n"] += 1
            return Structure(
                pdb_string=f"ATOM iter {call_count['n']}\nEND\n",
                metadata={"final_energy": -1.0 * call_count["n"]},
            )

        mock_op.side_effect = _side_effect
        wf = self._make_workflow(
            tmp_path,
            [
                {
                    "iterate": {
                        "n": 3,
                        "steps": [{"operation": "relax"}],
                    }
                }
            ],
        )
        result = wf.run()
        assert call_count["n"] == 3
        assert "iter 3" in result.pdb_string

    @patch("boundry.workflow.Workflow._run_operation")
    def test_iterate_convergence_stops_early(
        self, mock_op, tmp_path
    ):
        from boundry.operations import Structure

        self._make_input(tmp_path)
        dgs = [-1.0, -2.0, -4.0, -6.0]
        calls = {"n": 0}

        def _side_effect(name, structure, params):
            idx = calls["n"]
            calls["n"] += 1
            return Structure(
                pdb_string=f"ATOM dG {idx}\nEND\n",
                metadata={"dG": dgs[idx]},
            )

        mock_op.side_effect = _side_effect
        wf = self._make_workflow(
            tmp_path,
            [
                {
                    "iterate": {
                        "until": "{dG} < -3.0",
                        "max_n": 10,
                        "steps": [
                            {"operation": "analyze_interface"}
                        ],
                    }
                }
            ],
        )
        wf.run()
        assert calls["n"] == 3

    @patch("boundry.workflow.Workflow._run_operation")
    def test_iterate_seed_injected_with_workflow_seed(
        self, mock_op, tmp_path
    ):
        from boundry.operations import Structure
        from boundry.workflow import _compose_seed

        self._make_input(tmp_path)
        seen_seeds = []

        def _side_effect(name, structure, params):
            seen_seeds.append(params.get("seed"))
            return Structure(
                pdb_string="ATOM\nEND\n",
                metadata={"final_energy": -1.0},
            )

        mock_op.side_effect = _side_effect
        wf = self._make_workflow_with_seed(
            tmp_path,
            [
                {
                    "iterate": {
                        "n": 3,
                        "steps": [{"operation": "relax"}],
                    }
                }
            ],
            seed=42,
        )
        wf.run()
        expected = [_compose_seed(42, c) for c in range(1, 4)]
        assert seen_seeds == expected

    @patch("boundry.workflow.Workflow._run_operation")
    def test_no_seed_means_no_injection(self, mock_op, tmp_path):
        from boundry.operations import Structure

        self._make_input(tmp_path)
        seen_seeds = []

        def _side_effect(name, structure, params):
            seen_seeds.append(params.get("seed"))
            return Structure(
                pdb_string="ATOM\nEND\n",
                metadata={"final_energy": -1.0},
            )

        mock_op.side_effect = _side_effect
        wf = self._make_workflow(
            tmp_path,
            [
                {
                    "iterate": {
                        "n": 2,
                        "steps": [{"operation": "relax"}],
                    }
                }
            ],
        )
        wf.run()
        assert seen_seeds == [None, None]

    @patch("boundry.workflow.Workflow._run_operation")
    def test_step_seed_wins_over_workflow_seed(
        self, mock_op, tmp_path
    ):
        """Explicit step-level seed takes precedence."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        seen_seeds = []

        def _side_effect(name, structure, params):
            seen_seeds.append(params.get("seed"))
            return Structure(
                pdb_string="ATOM\nEND\n",
                metadata={"final_energy": -1.0},
            )

        mock_op.side_effect = _side_effect
        wf = self._make_workflow_with_seed(
            tmp_path,
            [
                {
                    "operation": "relax",
                    "params": {"seed": 999},
                }
            ],
            seed=42,
        )
        wf.run()
        assert seen_seeds == [999]

    @patch("boundry.workflow.Workflow._run_operation")
    def test_step_seed_wins_inside_iterate(
        self, mock_op, tmp_path
    ):
        """Explicit step-level seed inside iterate block is preserved."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        seen_seeds = []

        def _side_effect(name, structure, params):
            seen_seeds.append(params.get("seed"))
            return Structure(
                pdb_string="ATOM\nEND\n",
                metadata={"final_energy": -1.0},
            )

        mock_op.side_effect = _side_effect
        wf = self._make_workflow_with_seed(
            tmp_path,
            [
                {
                    "iterate": {
                        "n": 2,
                        "steps": [
                            {
                                "operation": "relax",
                                "params": {"seed": 999},
                            }
                        ],
                    }
                }
            ],
            seed=42,
        )
        wf.run()
        assert seen_seeds == [999, 999]

    @patch("boundry.workflow.Workflow._run_operation")
    def test_beam_top_k_continues_to_next_step(
        self, mock_op, tmp_path
    ):
        from boundry.operations import Structure

        self._make_input(tmp_path)
        scores = [5.0, 1.0, 3.0, 2.0]
        call_idx = {"i": 0}

        def _dispatch(name, structure, params):
            if name == "analyze_interface":
                i = call_idx["i"]
                call_idx["i"] += 1
                return Structure(
                    pdb_string=f"ATOM beam {i}\nEND\n",
                    metadata={"dG": scores[i]},
                )
            return Structure(
                pdb_string=structure.pdb_string,
                metadata={"final_energy": -10.0},
            )

        mock_op.side_effect = _dispatch

        wf = self._make_workflow(
            tmp_path,
            [
                {
                    "beam": {
                        "width": 2,
                        "rounds": 1,
                        "expand": 4,
                        "metric": "dG",
                        "direction": "min",
                        "steps": [
                            {"operation": "analyze_interface"}
                        ],
                    }
                },
                {"operation": "minimize"},
            ],
        )
        population = wf.run_population()
        assert len(population) == 2
        minimize_calls = [
            c
            for c in mock_op.call_args_list
            if c[0][0] == "minimize"
        ]
        assert len(minimize_calls) == 2

    @patch("boundry.workflow.Workflow._run_operation")
    def test_beam_missing_metric_raises(self, mock_op, tmp_path):
        from boundry.operations import Structure

        self._make_input(tmp_path)
        mock_op.return_value = Structure(
            pdb_string="ATOM\nEND\n",
            metadata={"final_energy": -10.0},
        )
        wf = self._make_workflow(
            tmp_path,
            [
                {
                    "beam": {
                        "width": 2,
                        "rounds": 1,
                        "metric": "dG",
                        "steps": [{"operation": "idealize"}],
                    }
                }
            ],
        )
        with pytest.raises(
            WorkflowError, match="Missing metric|missing metric"
        ):
            wf.run()


# ------------------------------------------------------------------
# Operation runner unit tests
# ------------------------------------------------------------------


class TestRunIdealize:
    """Tests for _run_idealize param building."""

    @patch("boundry.operations.idealize")
    def test_builds_config(self, mock_op):
        from boundry.operations import Structure

        struct = Structure(pdb_string="ATOM\nEND\n")
        mock_op.return_value = struct

        Workflow._run_operation(
            "idealize", struct, {"fix_cis_omega": False}
        )

        _, kwargs = mock_op.call_args
        config = kwargs["config"]
        assert config.enabled is True
        assert config.fix_cis_omega is False


class TestRunMinimize:
    """Tests for _run_minimize param building."""

    @patch("boundry.operations.minimize")
    def test_builds_config(self, mock_op):
        from boundry.operations import Structure

        struct = Structure(pdb_string="ATOM\nEND\n")
        mock_op.return_value = struct

        Workflow._run_operation(
            "minimize",
            struct,
            {"constrained": True, "max_iterations": 500},
        )

        _, kwargs = mock_op.call_args
        config = kwargs["config"]
        assert config.constrained is True
        assert config.max_iterations == 500

    @patch("boundry.operations.minimize")
    def test_pre_idealize_extracted(self, mock_op):
        from boundry.operations import Structure

        struct = Structure(pdb_string="ATOM\nEND\n")
        mock_op.return_value = struct

        Workflow._run_operation(
            "minimize", struct, {"pre_idealize": True}
        )

        _, kwargs = mock_op.call_args
        assert kwargs["pre_idealize"] is True


class TestRunRelax:
    """Tests for _run_relax param splitting."""

    @patch("boundry.weights.ensure_weights")
    @patch("boundry.operations.relax")
    def test_splits_design_and_relax_params(
        self, mock_op, mock_weights
    ):
        from boundry.operations import Structure

        struct = Structure(pdb_string="ATOM\nEND\n")
        mock_op.return_value = struct

        Workflow._run_operation(
            "relax",
            struct,
            {
                "temperature": 0.2,
                "constrained": True,
                "n_iterations": 3,
            },
        )

        _, kwargs = mock_op.call_args
        config = kwargs["config"]
        assert config.design.temperature == 0.2
        assert config.relax.constrained is True
        assert kwargs["n_iterations"] == 3

    @patch("boundry.weights.ensure_weights")
    @patch("boundry.operations.relax")
    def test_resfile_extracted(self, mock_op, mock_weights):
        from boundry.operations import Structure

        struct = Structure(pdb_string="ATOM\nEND\n")
        mock_op.return_value = struct

        Workflow._run_operation(
            "relax", struct, {"resfile": "design.resfile"}
        )

        _, kwargs = mock_op.call_args
        assert kwargs["resfile"] == "design.resfile"


class TestRunDesign:
    """Tests for _run_design param splitting."""

    @patch("boundry.weights.ensure_weights")
    @patch("boundry.operations.design")
    def test_splits_params(self, mock_op, mock_weights):
        from boundry.operations import Structure

        struct = Structure(pdb_string="ATOM\nEND\n")
        mock_op.return_value = struct

        Workflow._run_operation(
            "design",
            struct,
            {
                "model_type": "protein_mpnn",
                "stiffness": 5.0,
                "n_iterations": 2,
            },
        )

        _, kwargs = mock_op.call_args
        config = kwargs["config"]
        assert config.design.model_type == "protein_mpnn"
        assert config.relax.stiffness == 5.0
        assert kwargs["n_iterations"] == 2


class TestRunAnalyzeInterface:
    """Tests for _run_analyze_interface."""

    @patch("boundry.operations.analyze_interface")
    def test_string_chain_pairs(self, mock_op):
        from boundry.operations import (
            InterfaceAnalysisResult,
            Structure,
        )

        struct = Structure(pdb_string="ATOM\nEND\n")
        mock_op.return_value = InterfaceAnalysisResult()

        result = Workflow._run_analyze_interface(
            struct,
            {
                "chain_pairs": "H:A,L:A",
                "calculate_binding_energy": False,
            },
        )

        _, kwargs = mock_op.call_args
        config = kwargs["config"]
        assert config.chain_pairs == [("H", "A"), ("L", "A")]
        assert result.pdb_string == struct.pdb_string

    @patch("boundry.operations.analyze_interface")
    def test_list_chain_pairs(self, mock_op):
        from boundry.operations import (
            InterfaceAnalysisResult,
            Structure,
        )

        struct = Structure(pdb_string="ATOM\nEND\n")
        mock_op.return_value = InterfaceAnalysisResult()

        Workflow._run_analyze_interface(
            struct,
            {
                "chain_pairs": [["H", "A"]],
                "calculate_binding_energy": False,
            },
        )

        _, kwargs = mock_op.call_args
        config = kwargs["config"]
        assert config.chain_pairs == [("H", "A")]

    @patch("boundry.operations.analyze_interface")
    def test_returns_original_structure(self, mock_op):
        from boundry.operations import (
            InterfaceAnalysisResult,
            Structure,
        )

        struct = Structure(pdb_string="ATOM\nEND\n")
        mock_op.return_value = InterfaceAnalysisResult()

        result = Workflow._run_analyze_interface(
            struct, {"calculate_binding_energy": False}
        )
        assert result.pdb_string == struct.pdb_string


# ------------------------------------------------------------------
# extract_fields helper
# ------------------------------------------------------------------


class TestExtractFields:
    """Tests for _extract_fields helper."""

    def test_extracts_matching_keys(self):
        from boundry.config import RelaxConfig
        from boundry.workflow import _extract_fields

        params = {
            "constrained": True,
            "max_iterations": 500,
            "unrelated": "foo",
        }
        extracted = _extract_fields(params, RelaxConfig)

        assert extracted == {
            "constrained": True,
            "max_iterations": 500,
        }
        assert params == {"unrelated": "foo"}

    def test_empty_params(self):
        from boundry.config import RelaxConfig
        from boundry.workflow import _extract_fields

        params = {}
        extracted = _extract_fields(params, RelaxConfig)
        assert extracted == {}


# ------------------------------------------------------------------
# VALID_OPERATIONS constant
# ------------------------------------------------------------------


class TestValidOperations:
    """Tests for the VALID_OPERATIONS constant."""

    def test_contains_all_expected(self):
        expected = {
            "idealize",
            "minimize",
            "repack",
            "relax",
            "mpnn",
            "design",
            "renumber",
            "select_positions",
            "analyze_interface",
        }
        assert VALID_OPERATIONS == expected

    def test_renumber_in_valid_operations(self):
        """Renumber is a valid workflow operation."""
        assert "renumber" in VALID_OPERATIONS

    def test_select_positions_in_valid_operations(self):
        """select_positions is a valid workflow operation."""
        assert "select_positions" in VALID_OPERATIONS


# ------------------------------------------------------------------
# OutputPathContext
# ------------------------------------------------------------------


class TestOutputPathContext:
    """Tests for OutputPathContext."""

    def test_resolve_base_path(self, tmp_path):
        ctx = OutputPathContext(base_path=tmp_path)
        assert ctx.resolve() == tmp_path

    def test_child(self, tmp_path):
        ctx = OutputPathContext(base_path=tmp_path)
        child = ctx.child("subdir")
        assert child.resolve() == tmp_path / "subdir"

    def test_step_dir(self, tmp_path):
        ctx = OutputPathContext(base_path=tmp_path)
        step = ctx.step_dir(0, "idealize")
        assert step.resolve() == tmp_path / "0.idealize"

    def test_step_dir_index(self, tmp_path):
        ctx = OutputPathContext(base_path=tmp_path)
        step = ctx.step_dir(2, "relax")
        assert step.resolve() == tmp_path / "2.relax"

    def test_cycle_dir(self, tmp_path):
        ctx = OutputPathContext(base_path=tmp_path)
        cycle = ctx.cycle_dir(1)
        assert cycle.resolve() == tmp_path / "cycle_1"

    def test_round_dir(self, tmp_path):
        ctx = OutputPathContext(base_path=tmp_path)
        rnd = ctx.round_dir(3)
        assert rnd.resolve() == tmp_path / "round_3"

    def test_rank_dir(self, tmp_path):
        ctx = OutputPathContext(base_path=tmp_path)
        rank = ctx.rank_dir(1)
        assert rank.resolve() == tmp_path / "rank_1"

    def test_others_rank_dir(self, tmp_path):
        ctx = OutputPathContext(base_path=tmp_path)
        others = ctx.others_rank_dir(3)
        assert others.resolve() == tmp_path / "others" / "rank_3"

    def test_chained_context(self, tmp_path):
        """Test full chained context for iterate block."""
        ctx = OutputPathContext(base_path=tmp_path)
        result = (
            ctx.step_dir(1, "iterate")
            .cycle_dir(2)
            .step_dir(0, "relax")
        )
        expected = (
            tmp_path / "1.iterate" / "cycle_2" / "0.relax"
        )
        assert result.resolve() == expected

    def test_beam_chained_context(self, tmp_path):
        """Test full chained context for beam block."""
        ctx = OutputPathContext(base_path=tmp_path)
        result = (
            ctx.step_dir(0, "beam")
            .round_dir(1)
            .rank_dir(2)
            .step_dir(0, "design")
        )
        expected = (
            tmp_path
            / "0.beam"
            / "round_1"
            / "rank_2"
            / "0.design"
        )
        assert result.resolve() == expected

    def test_beam_others_chained_context(self, tmp_path):
        """Test full chained context for non-selected beam candidate."""
        ctx = OutputPathContext(base_path=tmp_path)
        result = (
            ctx.step_dir(0, "beam")
            .round_dir(1)
            .others_rank_dir(5)
            .step_dir(0, "design")
        )
        expected = (
            tmp_path
            / "0.beam"
            / "round_1"
            / "others"
            / "rank_5"
            / "0.design"
        )
        assert result.resolve() == expected

    def test_frozen(self, tmp_path):
        """OutputPathContext is immutable."""
        ctx = OutputPathContext(base_path=tmp_path)
        with pytest.raises(AttributeError):
            ctx.base_path = tmp_path / "new"


# ------------------------------------------------------------------
# Automatic directory structure
# ------------------------------------------------------------------


class TestAutomaticDirectoryStructure:
    """Tests for automatic directory output writing."""

    def _make_workflow(self, tmp_path, steps, project_path=None):
        wf_file = tmp_path / "wf.yaml"
        data = {
            "input": str(tmp_path / "input.pdb"),
            "steps": steps,
        }
        if project_path is not None:
            data["project_path"] = str(tmp_path / project_path)
        wf_file.write_text(yaml.dump(data))
        return Workflow.from_yaml(wf_file)

    def _make_input(self, tmp_path):
        pdb = tmp_path / "input.pdb"
        pdb.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000"
            "  1.00  0.00           N\nEND\n"
        )
        return pdb

    @patch("boundry.workflow.Workflow._run_operation")
    def test_single_step_creates_step_dir(
        self, mock_op, tmp_path
    ):
        """Single step creates 0.{operation}/ directory."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        mock_op.return_value = Structure(
            pdb_string="ATOM\nEND\n"
        )

        wf = self._make_workflow(
            tmp_path,
            [{"operation": "idealize"}],
            project_path="results",
        )
        wf.run()

        step_dir = tmp_path / "results" / "0.idealize"
        assert step_dir.exists()
        assert (step_dir / "idealized.pdb").exists()

    @patch("boundry.workflow.Workflow._run_operation")
    def test_multi_step_creates_indexed_dirs(
        self, mock_op, tmp_path
    ):
        """Multiple steps create 0-indexed directories."""
        from boundry.operations import Structure

        self._make_input(tmp_path)

        def _dispatch(name, structure, params):
            return Structure(
                pdb_string="ATOM\nEND\n",
                metadata={"final_energy": -1.0},
            )

        mock_op.side_effect = _dispatch

        wf = self._make_workflow(
            tmp_path,
            [
                {"operation": "idealize"},
                {"operation": "minimize"},
            ],
            project_path="results",
        )
        wf.run()

        assert (
            tmp_path / "results" / "0.idealize"
        ).exists()
        assert (
            tmp_path / "results" / "1.minimize"
        ).exists()

    @patch("boundry.workflow.Workflow._run_operation")
    def test_iterate_creates_cycle_dirs(
        self, mock_op, tmp_path
    ):
        """Iterate block creates cycle_N subdirectories."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        call_count = {"n": 0}

        def _side_effect(name, structure, params):
            call_count["n"] += 1
            return Structure(
                pdb_string="ATOM\nEND\n",
                metadata={"final_energy": -1.0 * call_count["n"]},
            )

        mock_op.side_effect = _side_effect

        wf = self._make_workflow(
            tmp_path,
            [
                {
                    "iterate": {
                        "n": 3,
                        "steps": [{"operation": "relax"}],
                    }
                }
            ],
            project_path="results",
        )
        wf.run()

        base = tmp_path / "results" / "0.iterate"
        assert base.exists()
        for c in range(1, 4):
            cycle_dir = base / f"cycle_{c}" / "0.relax"
            assert cycle_dir.exists(), (
                f"cycle_{c}/0.relax missing"
            )
            assert (cycle_dir / "relaxed.pdb").exists()

    @patch("boundry.workflow.Workflow._run_operation")
    def test_beam_creates_round_rank_dirs(
        self, mock_op, tmp_path
    ):
        """Beam block creates round_N/rank_N structure."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        scores = [5.0, 1.0, 3.0, 2.0]
        call_idx = {"i": 0}

        def _dispatch(name, structure, params):
            i = call_idx["i"]
            call_idx["i"] += 1
            return Structure(
                pdb_string=f"ATOM beam {i}\nEND\n",
                metadata={"dG": scores[i % len(scores)]},
            )

        mock_op.side_effect = _dispatch

        wf = self._make_workflow(
            tmp_path,
            [
                {
                    "beam": {
                        "width": 2,
                        "rounds": 1,
                        "expand": 4,
                        "metric": "dG",
                        "direction": "min",
                        "steps": [
                            {"operation": "analyze_interface"}
                        ],
                    }
                }
            ],
            project_path="results",
        )
        wf.run_population()

        beam_dir = tmp_path / "results" / "0.beam"
        assert beam_dir.exists()
        round_dir = beam_dir / "round_1"
        assert round_dir.exists()
        # Selected candidates
        assert (round_dir / "rank_1").exists()
        assert (round_dir / "rank_2").exists()
        # Non-selected candidates
        assert (round_dir / "others").exists()

    @patch("boundry.workflow.Workflow._run_operation")
    def test_step_before_iterate(self, mock_op, tmp_path):
        """Step before iterate block gets correct index."""
        from boundry.operations import Structure

        self._make_input(tmp_path)

        def _dispatch(name, structure, params):
            return Structure(
                pdb_string="ATOM\nEND\n",
                metadata={"final_energy": -1.0},
            )

        mock_op.side_effect = _dispatch

        wf = self._make_workflow(
            tmp_path,
            [
                {"operation": "idealize"},
                {
                    "iterate": {
                        "n": 2,
                        "steps": [{"operation": "relax"}],
                    }
                },
            ],
            project_path="results",
        )
        wf.run()

        assert (
            tmp_path / "results" / "0.idealize"
        ).exists()
        assert (
            tmp_path / "results" / "1.iterate"
        ).exists()
        assert (
            tmp_path
            / "results"
            / "1.iterate"
            / "cycle_1"
            / "0.relax"
        ).exists()


# ------------------------------------------------------------------
# Operation-aware output
# ------------------------------------------------------------------


class TestOperationAwareOutput:
    """Tests that each operation writes only its native artifacts."""

    def _make_workflow(self, tmp_path, steps, project_path=None):
        wf_file = tmp_path / "wf.yaml"
        data = {
            "input": str(tmp_path / "input.pdb"),
            "steps": steps,
        }
        if project_path is not None:
            data["project_path"] = str(tmp_path / project_path)
        wf_file.write_text(yaml.dump(data))
        return Workflow.from_yaml(wf_file)

    def _make_input(self, tmp_path):
        pdb = tmp_path / "input.pdb"
        pdb.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000"
            "  1.00  0.00           N\nEND\n"
        )
        return pdb

    @patch("boundry.workflow.Workflow._run_operation")
    def test_analyze_interface_writes_no_pdb(
        self, mock_op, tmp_path
    ):
        """analyze_interface should not write a PDB file."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        mock_op.return_value = Structure(
            pdb_string="ATOM\nEND\n",
            metadata={
                "dG": -5.0,
                "complex_energy": -100.0,
                "buried_sasa": 1200.0,
                "sc_score": 0.65,
                "n_interface_residues": 20,
            },
        )

        wf = self._make_workflow(
            tmp_path,
            [{"operation": "analyze_interface"}],
            project_path="results",
        )
        wf.run()

        step_dir = tmp_path / "results" / "0.analyze_interface"
        assert step_dir.exists()
        # No PDB files should be written
        pdb_files = list(step_dir.glob("*.pdb"))
        assert len(pdb_files) == 0
        # Interface JSON should be written
        assert (step_dir / "interface.json").exists()

    @patch("boundry.workflow.Workflow._run_operation")
    def test_relax_writes_pdb_and_energy(
        self, mock_op, tmp_path
    ):
        """relax should write PDB and energy.json."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        mock_op.return_value = Structure(
            pdb_string=(
                "ATOM      1  N   ALA A   1       "
                "1.000   1.000   1.000  1.00  0.00"
                "           N\nEND\n"
            ),
            metadata={
                "final_energy": -50.0,
                "energy_breakdown": {"total": -50.0},
            },
        )

        wf = self._make_workflow(
            tmp_path,
            [{"operation": "relax"}],
            project_path="results",
        )
        wf.run()

        step_dir = tmp_path / "results" / "0.relax"
        assert (step_dir / "relaxed.pdb").exists()
        assert (step_dir / "energy.json").exists()

    @patch("boundry.workflow.Workflow._run_operation")
    def test_idealize_writes_pdb_only(
        self, mock_op, tmp_path
    ):
        """idealize writes PDB and metrics, no energy.json."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        mock_op.return_value = Structure(
            pdb_string="ATOM\nEND\n",
            metadata={"chain_gaps": 0},
        )

        wf = self._make_workflow(
            tmp_path,
            [{"operation": "idealize"}],
            project_path="results",
        )
        wf.run()

        step_dir = tmp_path / "results" / "0.idealize"
        assert (step_dir / "idealized.pdb").exists()
        # No energy.json for idealize
        assert not (step_dir / "energy.json").exists()
        # Should have metrics.json with chain_gaps
        assert (step_dir / "metrics.json").exists()
        metrics = json.loads(
            (step_dir / "metrics.json").read_text()
        )
        assert metrics["chain_gaps"] == 0

    @patch("boundry.workflow.Workflow._run_operation")
    def test_metrics_from_prior_steps_dont_leak(
        self, mock_op, tmp_path
    ):
        """Metrics from step 1 should not appear in step 2's output."""
        from boundry.operations import Structure

        self._make_input(tmp_path)

        def _dispatch(name, structure, params):
            if name == "idealize":
                return Structure(
                    pdb_string="ATOM\nEND\n",
                    metadata={"chain_gaps": 2},
                )
            else:
                return Structure(
                    pdb_string="ATOM\nEND\n",
                    metadata={
                        "final_energy": -50.0,
                        "energy_breakdown": {"total": -50.0},
                    },
                )

        mock_op.side_effect = _dispatch

        wf = self._make_workflow(
            tmp_path,
            [
                {"operation": "idealize"},
                {"operation": "relax"},
            ],
            project_path="results",
        )
        wf.run()

        # Relax metrics should only contain relax-native keys
        relax_dir = tmp_path / "results" / "1.relax"
        if (relax_dir / "metrics.json").exists():
            metrics = json.loads(
                (relax_dir / "metrics.json").read_text()
            )
            # chain_gaps is from idealize, should not appear
            assert "chain_gaps" not in metrics

    def test_operation_output_specs_complete(self):
        """All valid operations have an output spec."""
        for op in VALID_OPERATIONS:
            assert op in _OPERATION_OUTPUT_SPECS, (
                f"Missing output spec for {op}"
            )

    def test_native_metric_keys_complete(self):
        """All valid operations have native metric keys defined."""
        for op in VALID_OPERATIONS:
            assert op in _NATIVE_METRIC_KEYS, (
                f"Missing native metric keys for {op}"
            )

    def test_structural_ops_write_pdb(self):
        """Operations that modify structure should write PDB."""
        structural_ops = {
            "idealize",
            "minimize",
            "repack",
            "mpnn",
            "relax",
            "design",
            "renumber",
        }
        for op in structural_ops:
            spec = _OPERATION_OUTPUT_SPECS[op]
            assert spec.writes_pdb, (
                f"{op} should write PDB"
            )

    def test_non_structural_ops_skip_pdb(self):
        """Operations that don't modify structure skip PDB."""
        non_structural_ops = {
            "select_positions",
            "analyze_interface",
        }
        for op in non_structural_ops:
            spec = _OPERATION_OUTPUT_SPECS[op]
            assert not spec.writes_pdb, (
                f"{op} should not write PDB"
            )


# ------------------------------------------------------------------
# Project path
# ------------------------------------------------------------------


class TestProjectPath:
    """Tests for project_path handling."""

    def test_project_path_from_yaml(self, tmp_path):
        """project_path is parsed from YAML."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": "input.pdb",
                    "project_path": "results",
                    "steps": [{"operation": "idealize"}],
                }
            )
        )
        wf = Workflow.from_yaml(wf_file)
        assert wf.config.project_path == "results"

    def test_project_path_none_when_omitted(self, tmp_path):
        """project_path is None when not specified."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": "input.pdb",
                    "steps": [{"operation": "idealize"}],
                }
            )
        )
        wf = Workflow.from_yaml(wf_file)
        assert wf.config.project_path is None

    def test_cli_override_project_path(self, tmp_path):
        """project_path can be overridden via CLI."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "project_path: original\n"
            "steps:\n"
            "  - operation: idealize\n"
        )
        wf = Workflow.from_yaml(
            wf_file, overrides=["project_path=overridden"]
        )
        assert wf.config.project_path == "overridden"


# ------------------------------------------------------------------
# Safe param handling in _run_* methods
# ------------------------------------------------------------------


class TestSafeParamHandling:
    """Tests that _run_* methods gracefully handle unknown params."""

    @patch("boundry.operations.idealize")
    def test_idealize_ignores_unknown_params(self, mock_op):
        from boundry.operations import Structure

        struct = Structure(pdb_string="ATOM\nEND\n")
        mock_op.return_value = struct

        # Should not raise, even with unknown 'bogus' param
        Workflow._run_operation(
            "idealize",
            struct,
            {"fix_cis_omega": False, "bogus": 42},
        )

        _, kwargs = mock_op.call_args
        config = kwargs["config"]
        assert config.fix_cis_omega is False

    @patch("boundry.operations.minimize")
    def test_minimize_ignores_unknown_params(self, mock_op):
        from boundry.operations import Structure

        struct = Structure(pdb_string="ATOM\nEND\n")
        mock_op.return_value = struct

        Workflow._run_operation(
            "minimize",
            struct,
            {"constrained": True, "bogus": 42},
        )

        _, kwargs = mock_op.call_args
        config = kwargs["config"]
        assert config.constrained is True

    @patch("boundry.weights.ensure_weights")
    @patch("boundry.operations.repack")
    def test_repack_ignores_unknown_params(
        self, mock_op, mock_weights
    ):
        from boundry.operations import Structure

        struct = Structure(pdb_string="ATOM\nEND\n")
        mock_op.return_value = struct

        Workflow._run_operation(
            "repack",
            struct,
            {"temperature": 0.2, "bogus": 42},
        )

        _, kwargs = mock_op.call_args
        config = kwargs["config"]
        assert config.temperature == 0.2

    @patch("boundry.weights.ensure_weights")
    @patch("boundry.operations.mpnn")
    def test_mpnn_ignores_unknown_params(
        self, mock_op, mock_weights
    ):
        from boundry.operations import Structure

        struct = Structure(pdb_string="ATOM\nEND\n")
        mock_op.return_value = struct

        Workflow._run_operation(
            "mpnn",
            struct,
            {"temperature": 0.2, "bogus": 42},
        )

        _, kwargs = mock_op.call_args
        config = kwargs["config"]
        assert config.temperature == 0.2


# ------------------------------------------------------------------
# Workflow-level seed parsing and validation
# ------------------------------------------------------------------


class TestWorkflowSeedParsing:
    """Tests for workflow-level seed YAML/CLI parsing."""

    def test_yaml_seed_parsed(self, tmp_path):
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": "input.pdb",
                    "seed": 42,
                    "steps": [{"operation": "idealize"}],
                }
            )
        )
        wf = Workflow.from_yaml(wf_file)
        assert wf.config.seed == 42

    def test_yaml_seed_null(self, tmp_path):
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "seed:\n"
            "steps:\n"
            "  - operation: idealize\n"
        )
        wf = Workflow.from_yaml(wf_file)
        assert wf.config.seed is None

    def test_cli_seed_overrides_yaml(self, tmp_path):
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": "input.pdb",
                    "seed": 42,
                    "steps": [{"operation": "idealize"}],
                }
            )
        )
        wf = Workflow.from_yaml(wf_file, seed=99)
        assert wf.config.seed == 99

    def test_cli_seed_when_yaml_has_none(self, tmp_path):
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": "input.pdb",
                    "steps": [{"operation": "idealize"}],
                }
            )
        )
        wf = Workflow.from_yaml(wf_file, seed=7)
        assert wf.config.seed == 7

    def test_invalid_seed_type_raises(self, tmp_path):
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": "input.pdb",
                    "seed": "not_an_int",
                    "steps": [{"operation": "idealize"}],
                }
            )
        )
        with pytest.raises(
            WorkflowError, match="seed.*must be an integer"
        ):
            Workflow.from_yaml(wf_file)

    def test_bool_seed_rejected(self, tmp_path):
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": "input.pdb",
                    "seed": True,
                    "steps": [{"operation": "idealize"}],
                }
            )
        )
        with pytest.raises(
            WorkflowError, match="seed.*must be an integer"
        ):
            Workflow.from_yaml(wf_file)

    def test_old_iterate_seed_key_rejected(self, tmp_path):
        """The old iterate-level seed: true key should be rejected."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": "input.pdb",
                    "steps": [
                        {
                            "iterate": {
                                "n": 3,
                                "seed": True,
                                "steps": [
                                    {"operation": "relax"}
                                ],
                            }
                        }
                    ],
                }
            )
        )
        with pytest.raises(WorkflowError, match="unknown fields"):
            Workflow.from_yaml(wf_file)


# ------------------------------------------------------------------
# _with_seed unit tests
# ------------------------------------------------------------------


class TestWithSeed:
    """Tests for Workflow._with_seed static method."""

    def test_no_seed_base_returns_params_unchanged(self):
        params = {"temperature": 0.1}
        result = Workflow._with_seed("relax", params, None, 0)
        assert result == {"temperature": 0.1}
        assert "seed" not in result

    def test_seed_injected_for_supported_operation(self):
        params = {"temperature": 0.1}
        result = Workflow._with_seed("relax", params, 42, 0)
        assert result["seed"] == 42

    def test_seed_with_candidate_offset(self):
        params = {}
        result = Workflow._with_seed("mpnn", params, 100, 3)
        assert result["seed"] == 103

    def test_seed_not_injected_for_unsupported_operation(self):
        params = {}
        result = Workflow._with_seed("idealize", params, 42, 0)
        assert "seed" not in result

    def test_explicit_step_seed_preserved(self):
        """Step-level seed takes precedence over workflow seed."""
        params = {"seed": 999}
        result = Workflow._with_seed("relax", params, 42, 0)
        assert result["seed"] == 999


# ------------------------------------------------------------------
# Variable interpolation
# ------------------------------------------------------------------


class TestVarInterpolation:
    """Tests for ${key} variable interpolation in workflow YAML."""

    def test_input_referenceable(self, tmp_path):
        """${input} resolves to the top-level input value."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: structures/my_protein.pdb\n"
            "project_path: ${input}_results\n"
            "steps:\n"
            "  - operation: idealize\n"
        )
        wf = Workflow.from_yaml(wf_file)
        assert (
            wf.config.project_path
            == "structures/my_protein.pdb_results"
        )

    def test_user_var_in_params(self, tmp_path):
        """Custom key referenced in params."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "resfile_path: design.resfile\n"
            "steps:\n"
            "  - operation: relax\n"
            "    params:\n"
            "      resfile: ${resfile_path}\n"
        )
        wf = Workflow.from_yaml(wf_file)
        assert (
            wf.config.steps[0].params["resfile"]
            == "design.resfile"
        )

    def test_cross_reference(self, tmp_path):
        """User var can reference another variable."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "project_path: results\n"
            "run_dir: ${project_path}/run_1\n"
            "steps:\n"
            "  - operation: idealize\n"
        )
        wf = Workflow.from_yaml(wf_file)
        assert wf.config.vars["run_dir"] == "results/run_1"

    def test_no_var_references(self, tmp_path):
        """Existing workflows without ${...} work unchanged."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": "input.pdb",
                    "project_path": "results",
                    "steps": [{"operation": "idealize"}],
                }
            )
        )
        wf = Workflow.from_yaml(wf_file)
        assert wf.config.input == "input.pdb"
        assert wf.config.project_path == "results"

    def test_numeric_var_coerced(self, tmp_path):
        """Numeric user var is coerced to string in namespace."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "run_id: 42\n"
            "project_path: run_${run_id}\n"
            "steps:\n"
            "  - operation: idealize\n"
        )
        wf = Workflow.from_yaml(wf_file)
        assert wf.config.project_path == "run_42"

    def test_numeric_params_not_affected(self, tmp_path):
        """Int/float params pass through unchanged."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "steps:\n"
            "  - operation: relax\n"
            "    params:\n"
            "      n_iterations: 5\n"
            "      temperature: 0.1\n"
        )
        wf = Workflow.from_yaml(wf_file)
        params = wf.config.steps[0].params
        assert params["n_iterations"] == 5
        assert params["temperature"] == 0.1

    def test_user_vars_stored_on_config(self, tmp_path):
        """User-defined vars are accessible via workflow.config.vars."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "project: my_project\n"
            "run_id: 7\n"
            "steps:\n"
            "  - operation: idealize\n"
        )
        wf = Workflow.from_yaml(wf_file)
        assert wf.config.vars == {
            "project": "my_project",
            "run_id": "7",
        }

    def test_undefined_var_raises(self, tmp_path):
        """${nonexistent} raises WorkflowError."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "steps:\n"
            "  - operation: idealize\n"
            "    params:\n"
            "      resfile: ${nonexistent}/out.resfile\n"
        )
        with pytest.raises(
            WorkflowError, match="resolution failed"
        ):
            Workflow.from_yaml(wf_file)

    def test_non_scalar_user_var_raises(self, tmp_path):
        """List/dict user var values are rejected."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "my_list:\n"
            "  - a\n"
            "  - b\n"
            "steps:\n"
            "  - operation: idealize\n"
        )
        with pytest.raises(WorkflowError, match="scalar"):
            Workflow.from_yaml(wf_file)

    def test_invalid_var_name_raises(self, tmp_path):
        """Hyphens in user var names are rejected."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "my-var: value\n"
            "steps:\n"
            "  - operation: idealize\n"
        )
        with pytest.raises(
            WorkflowError, match="Invalid variable name"
        ):
            Workflow.from_yaml(wf_file)

    def test_circular_reference_raises(self, tmp_path):
        """Circular cross-references are detected."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "a: ${b}\n"
            "b: ${a}\n"
            "steps:\n"
            "  - operation: idealize\n"
        )
        with pytest.raises(WorkflowError, match="Circular"):
            Workflow.from_yaml(wf_file)

    def test_steps_not_referenceable(self, tmp_path):
        """${steps} is non-scalar, so referencing it causes an error."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "steps:\n"
            "  - operation: idealize\n"
            "    params:\n"
            "      resfile: ${steps}\n"
        )
        with pytest.raises(
            WorkflowError, match="resolution failed"
        ):
            Workflow.from_yaml(wf_file)

    def test_bool_user_var_rejected(self, tmp_path):
        """Boolean user var values are rejected."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "flag: true\n"
            "steps:\n"
            "  - operation: idealize\n"
        )
        with pytest.raises(WorkflowError, match="scalar"):
            Workflow.from_yaml(wf_file)

    def test_seed_in_namespace(self, tmp_path):
        """${seed} resolves to the YAML seed value."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "seed: 42\n"
            "project_path: run_seed_${seed}\n"
            "steps:\n"
            "  - operation: idealize\n"
        )
        wf = Workflow.from_yaml(wf_file)
        assert wf.config.project_path == "run_seed_42"


# ------------------------------------------------------------------
# CLI overrides
# ------------------------------------------------------------------


class TestCliOverrides:
    """Tests for the overrides parameter in from_yaml."""

    def test_override_project_path(self, tmp_path):
        """Override project_path via dotlist."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "project_path: original\n"
            "steps:\n"
            "  - operation: idealize\n"
        )
        wf = Workflow.from_yaml(
            wf_file, overrides=["project_path=overridden"]
        )
        assert wf.config.project_path == "overridden"

    def test_override_user_var(self, tmp_path):
        """Override a user-defined variable."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "project: default_proj\n"
            "project_path: ${project}\n"
            "steps:\n"
            "  - operation: idealize\n"
        )
        wf = Workflow.from_yaml(
            wf_file, overrides=["project=custom_proj"]
        )
        assert wf.config.project_path == "custom_proj"

    def test_override_seed(self, tmp_path):
        """Override seed via dotlist."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "seed: 1\n"
            "steps:\n"
            "  - operation: idealize\n"
        )
        wf = Workflow.from_yaml(wf_file, overrides=["seed=99"])
        assert wf.config.seed == 99

    def test_cli_seed_wins_over_override(self, tmp_path):
        """CLI --seed takes precedence over override."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "seed: 1\n"
            "steps:\n"
            "  - operation: idealize\n"
        )
        wf = Workflow.from_yaml(
            wf_file, seed=777, overrides=["seed=99"]
        )
        assert wf.config.seed == 777

    def test_invalid_override_raises(self, tmp_path):
        """Bad override syntax raises WorkflowError."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "steps:\n"
            "  - operation: idealize\n"
        )
        # "=bad" creates an empty-string key that fails identifier
        # validation; any WorkflowError is acceptable here
        with pytest.raises(WorkflowError):
            Workflow.from_yaml(wf_file, overrides=["=bad"])

    def test_no_overrides_is_noop(self, tmp_path):
        """None overrides behaves like no overrides."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "project_path: results\n"
            "steps:\n"
            "  - operation: idealize\n"
        )
        wf = Workflow.from_yaml(wf_file, overrides=None)
        assert wf.config.project_path == "results"


# ------------------------------------------------------------------
# Built-in workflow lookup
# ------------------------------------------------------------------


class TestBuiltinWorkflowLookup:
    """Tests for _resolve_workflow."""

    def test_existing_file_returned(self, tmp_path):
        """Explicit file path is returned as-is."""
        from boundry.cli import _resolve_workflow

        wf_file = tmp_path / "my.yaml"
        wf_file.write_text(
            "input: x\nsteps:\n  - operation: idealize\n"
        )
        result = _resolve_workflow(str(wf_file))
        assert result == wf_file

    def test_builtin_name_resolved(self):
        """Built-in workflow name resolves to package directory."""
        from boundry.cli import _resolve_workflow

        result = _resolve_workflow("simple_relax")
        assert result.exists()
        assert result.name == "simple_relax.yaml"

    def test_unknown_name_raises(self):
        """Unknown name raises BadParameter."""
        import typer

        from boundry.cli import _resolve_workflow

        with pytest.raises(typer.BadParameter, match="not found"):
            _resolve_workflow("nonexistent_workflow_xyz")

    def test_user_dir_checked(self, tmp_path, monkeypatch):
        """User ~/.boundry/workflows/ is checked."""
        from pathlib import Path as _Path

        from boundry.cli import _resolve_workflow

        user_wf_dir = tmp_path / ".boundry" / "workflows"
        user_wf_dir.mkdir(parents=True)
        wf_file = user_wf_dir / "custom.yaml"
        wf_file.write_text(
            "input: x\nsteps:\n  - operation: idealize\n"
        )
        monkeypatch.setattr(_Path, "home", lambda: tmp_path)
        result = _resolve_workflow("custom")
        assert result == wf_file
