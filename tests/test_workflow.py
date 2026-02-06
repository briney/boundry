"""Tests for boundry.workflow module."""

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
    Workflow,
    WorkflowError,
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
        assert wf.config.output is None
        assert len(wf.config.steps) == 1
        assert wf.config.steps[0].operation == "idealize"
        assert wf.config.steps[0].params == {}

    def test_full_workflow(self, tmp_path):
        """Test loading a workflow with output and params."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": "input.pdb",
                    "output": "final.pdb",
                    "steps": [
                        {
                            "operation": "idealize",
                            "params": {"fix_cis_omega": True},
                            "output": "idealized.pdb",
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
        assert wf.config.output == "final.pdb"
        assert len(wf.config.steps) == 2

        step1 = wf.config.steps[0]
        assert step1.operation == "idealize"
        assert step1.params == {"fix_cis_omega": True}
        assert step1.output == "idealized.pdb"

        step2 = wf.config.steps[1]
        assert step2.operation == "minimize"
        assert step2.params["constrained"] is False
        assert step2.params["max_iterations"] == 1000
        assert step2.output is None

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
            output="output.pdb",
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
                                "steps": [{"operation": "analyze_interface"}],
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


# ------------------------------------------------------------------
# Workflow execution
# ------------------------------------------------------------------


class TestWorkflowRun:
    """Tests for Workflow.run() with mocked operations."""

    def _make_workflow(self, tmp_path, steps, output=None):
        """Helper to create a workflow YAML and return Workflow."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": str(tmp_path / "input.pdb"),
                    "output": (
                        str(tmp_path / output) if output else None
                    ),
                    "steps": steps,
                }
            )
        )
        return Workflow.from_yaml(wf_file, require_output=False)

    def _make_input(self, tmp_path):
        """Write a dummy PDB file and return path."""
        pdb = tmp_path / "input.pdb"
        pdb.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000"
            "  1.00  0.00           N\nEND\n"
        )
        return pdb

    @patch("boundry.workflow.Workflow._run_idealize")
    def test_run_requires_output_by_default(
        self, mock_idealize, tmp_path
    ):
        """Workflow.run() requires output paths unless opted out."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        mock_idealize.return_value = Structure(pdb_string="ATOM\nEND\n")

        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": str(tmp_path / "input.pdb"),
                    "steps": [{"operation": "idealize"}],
                }
            )
        )
        wf = Workflow.from_yaml(wf_file)
        with pytest.raises(WorkflowError, match="output is required"):
            wf.run()

    @patch("boundry.workflow.Workflow._run_idealize")
    def test_run_can_opt_out_of_output_requirement(
        self, mock_idealize, tmp_path
    ):
        """Programmatic workflows can still run fully in-memory."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        mock_idealize.return_value = Structure(
            pdb_string="ATOM idealized\nEND\n"
        )

        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": str(tmp_path / "input.pdb"),
                    "steps": [{"operation": "idealize"}],
                }
            )
        )
        wf = Workflow.from_yaml(wf_file, require_output=False)
        result = wf.run()
        assert "idealized" in result.pdb_string

    @patch("boundry.workflow.Workflow._run_idealize")
    def test_single_step(self, mock_idealize, tmp_path):
        """Test running a single-step workflow."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        mock_idealize.return_value = Structure(
            pdb_string="ATOM idealized\nEND\n"
        )

        wf = self._make_workflow(
            tmp_path, [{"operation": "idealize"}]
        )
        result = wf.run()

        mock_idealize.assert_called_once()
        assert result.pdb_string == "ATOM idealized\nEND\n"

    @patch("boundry.workflow.Workflow._run_minimize")
    @patch("boundry.workflow.Workflow._run_idealize")
    def test_multi_step_chaining(
        self, mock_idealize, mock_minimize, tmp_path
    ):
        """Test that output of step 1 feeds into step 2."""
        from boundry.operations import Structure

        self._make_input(tmp_path)

        struct_after_ideal = Structure(
            pdb_string="ATOM idealized\nEND\n"
        )
        struct_after_min = Structure(
            pdb_string="ATOM minimized\nEND\n"
        )
        mock_idealize.return_value = struct_after_ideal
        mock_minimize.return_value = struct_after_min

        wf = self._make_workflow(
            tmp_path,
            [
                {"operation": "idealize"},
                {"operation": "minimize"},
            ],
        )
        result = wf.run()

        # Step 2 should receive output of step 1
        call_args = mock_minimize.call_args
        assert call_args[0][0].pdb_string == struct_after_ideal.pdb_string
        assert result.pdb_string == struct_after_min.pdb_string

    @patch("boundry.workflow.Workflow._run_idealize")
    def test_final_output_written(self, mock_idealize, tmp_path):
        """Test that final output is written to disk."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        mock_idealize.return_value = Structure(
            pdb_string=(
                "ATOM      1  N   ALA A   1       "
                "1.000   1.000   1.000  1.00  0.00"
                "           N\nEND\n"
            )
        )

        output_path = tmp_path / "final.pdb"
        wf = self._make_workflow(
            tmp_path,
            [{"operation": "idealize"}],
            output="final.pdb",
        )
        wf.run()

        assert output_path.exists()
        assert "ATOM" in output_path.read_text()

    @patch("boundry.workflow.Workflow._run_minimize")
    @patch("boundry.workflow.Workflow._run_idealize")
    def test_intermediate_output_written(
        self, mock_idealize, mock_minimize, tmp_path
    ):
        """Test that intermediate output is written."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        intermediate_path = tmp_path / "idealized.pdb"

        intermediate_struct = Structure(
            pdb_string=(
                "ATOM      1  N   ALA A   1       "
                "1.000   1.000   1.000  1.00  0.00"
                "           N\nEND\n"
            )
        )
        mock_idealize.return_value = intermediate_struct
        mock_minimize.return_value = Structure(
            pdb_string="ATOM minimized\nEND\n"
        )

        wf = self._make_workflow(
            tmp_path,
            [
                {
                    "operation": "idealize",
                    "output": str(intermediate_path),
                },
                {"operation": "minimize"},
            ],
        )
        wf.run()

        assert intermediate_path.exists()
        assert "ATOM" in intermediate_path.read_text()

    def test_nonexistent_input_raises(self, tmp_path):
        """Test that a missing input file raises WorkflowError."""
        wf = self._make_workflow(
            tmp_path, [{"operation": "idealize"}]
        )
        # input.pdb was never created
        with pytest.raises(WorkflowError, match="not found"):
            wf.run()

    @patch("boundry.workflow.Workflow._run_idealize")
    def test_params_passed_to_handler(self, mock_idealize, tmp_path):
        """Test that step params are forwarded to the handler."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        mock_idealize.return_value = Structure(
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

        call_args = mock_idealize.call_args
        params = call_args[0][1]  # second positional arg
        assert params == {"fix_cis_omega": False}


class TestCompoundExecution:
    """Tests for iterate/beam workflow execution."""

    def _make_workflow(self, tmp_path, steps, output=None):
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": str(tmp_path / "input.pdb"),
                    "output": (
                        str(tmp_path / output) if output else None
                    ),
                    "steps": steps,
                }
            )
        )
        return Workflow.from_yaml(wf_file, require_output=False)

    def _make_workflow_with_seed(
        self, tmp_path, steps, seed, output=None
    ):
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": str(tmp_path / "input.pdb"),
                    "output": (
                        str(tmp_path / output) if output else None
                    ),
                    "seed": seed,
                    "steps": steps,
                }
            )
        )
        return Workflow.from_yaml(wf_file, require_output=False)

    def _make_input(self, tmp_path):
        pdb = tmp_path / "input.pdb"
        pdb.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000"
            "  1.00  0.00           N\nEND\n"
        )
        return pdb

    @patch("boundry.workflow.Workflow._run_relax")
    def test_iterate_fixed_count_runs_n_times(self, mock_relax, tmp_path):
        from boundry.operations import Structure

        self._make_input(tmp_path)
        call_count = {"n": 0}

        def _side_effect(structure, params):
            call_count["n"] += 1
            return Structure(
                pdb_string=f"ATOM iter {call_count['n']}\nEND\n",
                metadata={"final_energy": -1.0 * call_count["n"]},
            )

        mock_relax.side_effect = _side_effect
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

    @patch("boundry.workflow.Workflow._run_analyze_interface")
    def test_iterate_convergence_stops_early(
        self, mock_analyze, tmp_path
    ):
        from boundry.operations import Structure

        self._make_input(tmp_path)
        dgs = [-1.0, -2.0, -4.0, -6.0]
        calls = {"n": 0}

        def _side_effect(structure, params):
            idx = calls["n"]
            calls["n"] += 1
            return Structure(
                pdb_string=f"ATOM dG {idx}\nEND\n",
                metadata={"dG": dgs[idx]},
            )

        mock_analyze.side_effect = _side_effect
        wf = self._make_workflow(
            tmp_path,
            [
                {
                    "iterate": {
                        "until": "{dG} < -3.0",
                        "max_n": 10,
                        "steps": [{"operation": "analyze_interface"}],
                    }
                }
            ],
        )
        wf.run()
        assert calls["n"] == 3

    @patch("boundry.workflow.Workflow._run_relax")
    def test_iterate_seed_injected_with_workflow_seed(
        self, mock_relax, tmp_path
    ):
        from boundry.workflow import _compose_seed
        from boundry.operations import Structure

        self._make_input(tmp_path)
        seen_seeds = []

        def _side_effect(structure, params):
            seen_seeds.append(params.get("seed"))
            return Structure(
                pdb_string="ATOM\nEND\n",
                metadata={"final_energy": -1.0},
            )

        mock_relax.side_effect = _side_effect
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

    @patch("boundry.workflow.Workflow._run_relax")
    def test_no_seed_means_no_injection(self, mock_relax, tmp_path):
        from boundry.operations import Structure

        self._make_input(tmp_path)
        seen_seeds = []

        def _side_effect(structure, params):
            seen_seeds.append(params.get("seed"))
            return Structure(
                pdb_string="ATOM\nEND\n",
                metadata={"final_energy": -1.0},
            )

        mock_relax.side_effect = _side_effect
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

    @patch("boundry.workflow.Workflow._run_relax")
    def test_step_seed_wins_over_workflow_seed(
        self, mock_relax, tmp_path
    ):
        """Explicit step-level seed takes precedence."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        seen_seeds = []

        def _side_effect(structure, params):
            seen_seeds.append(params.get("seed"))
            return Structure(
                pdb_string="ATOM\nEND\n",
                metadata={"final_energy": -1.0},
            )

        mock_relax.side_effect = _side_effect
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

    @patch("boundry.workflow.Workflow._run_relax")
    def test_step_seed_wins_inside_iterate(
        self, mock_relax, tmp_path
    ):
        """Explicit step-level seed inside iterate block is preserved."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        seen_seeds = []

        def _side_effect(structure, params):
            seen_seeds.append(params.get("seed"))
            return Structure(
                pdb_string="ATOM\nEND\n",
                metadata={"final_energy": -1.0},
            )

        mock_relax.side_effect = _side_effect
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

    @patch("boundry.workflow.Workflow._run_minimize")
    @patch("boundry.workflow.Workflow._run_analyze_interface")
    def test_beam_top_k_continues_to_next_step(
        self, mock_analyze, mock_minimize, tmp_path
    ):
        from boundry.operations import Structure

        self._make_input(tmp_path)
        scores = [5.0, 1.0, 3.0, 2.0]
        call_idx = {"i": 0}

        def _analyze(structure, params):
            i = call_idx["i"]
            call_idx["i"] += 1
            return Structure(
                pdb_string=f"ATOM beam {i}\nEND\n",
                metadata={"dG": scores[i]},
            )

        def _minimize(structure, params):
            return Structure(
                pdb_string=structure.pdb_string,
                metadata={"final_energy": -10.0},
            )

        mock_analyze.side_effect = _analyze
        mock_minimize.side_effect = _minimize

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
                        "steps": [{"operation": "analyze_interface"}],
                    }
                },
                {"operation": "minimize"},
            ],
        )
        population = wf.run_population()
        assert len(population) == 2
        assert mock_minimize.call_count == 2

    @patch("boundry.workflow.Workflow._run_idealize")
    def test_beam_missing_metric_raises(self, mock_idealize, tmp_path):
        from boundry.operations import Structure

        self._make_input(tmp_path)
        mock_idealize.return_value = Structure(
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
        with pytest.raises(WorkflowError, match="Missing metric|missing metric"):
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

        Workflow._run_idealize(
            struct, {"fix_cis_omega": False}
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

        Workflow._run_minimize(
            struct, {"constrained": True, "max_iterations": 500}
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

        Workflow._run_minimize(struct, {"pre_idealize": True})

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

        Workflow._run_relax(
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

        Workflow._run_relax(
            struct, {"resfile": "design.resfile"}
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

        Workflow._run_design(
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
            "analyze_interface",
        }
        assert VALID_OPERATIONS == expected

    def test_renumber_in_valid_operations(self):
        """Renumber is a valid workflow operation."""
        assert "renumber" in VALID_OPERATIONS


# ------------------------------------------------------------------
# Directory output detection
# ------------------------------------------------------------------


class TestIsDirectoryOutput:
    """Tests for _is_directory_output helper."""

    def test_trailing_slash(self):
        from boundry.workflow import _is_directory_output

        assert _is_directory_output("results/") is True

    def test_pdb_extension(self):
        from boundry.workflow import _is_directory_output

        assert _is_directory_output("output.pdb") is False

    def test_cif_extension(self):
        from boundry.workflow import _is_directory_output

        assert _is_directory_output("output.cif") is False

    def test_mmcif_extension(self):
        from boundry.workflow import _is_directory_output

        assert _is_directory_output("output.mmcif") is False

    def test_no_extension(self):
        from boundry.workflow import _is_directory_output

        assert _is_directory_output("results") is True

    def test_template_with_pdb(self):
        from boundry.workflow import _is_directory_output

        assert _is_directory_output("output_{cycle}.pdb") is False

    def test_template_dir_with_slash(self):
        from boundry.workflow import _is_directory_output

        assert _is_directory_output("results/cycle_{cycle}/") is True

    def test_non_structure_extension(self):
        from boundry.workflow import _is_directory_output

        assert _is_directory_output("output.json") is True


# ------------------------------------------------------------------
# Directory output writing
# ------------------------------------------------------------------


class TestDirectoryOutput:
    """Tests for directory output writing."""

    def _make_workflow(self, tmp_path, steps, output=None):
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": str(tmp_path / "input.pdb"),
                    "output": (
                        str(tmp_path / output) if output else None
                    ),
                    "steps": steps,
                }
            )
        )
        return Workflow.from_yaml(wf_file, require_output=False)

    def _make_input(self, tmp_path):
        pdb = tmp_path / "input.pdb"
        pdb.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000"
            "  1.00  0.00           N\nEND\n"
        )
        return pdb

    @patch("boundry.workflow.Workflow._run_idealize")
    def test_directory_output_creates_pdb(self, mock_idealize, tmp_path):
        """Test that directory output creates a PDB file."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        mock_idealize.return_value = Structure(
            pdb_string=(
                "ATOM      1  N   ALA A   1       "
                "1.000   1.000   1.000  1.00  0.00"
                "           N\nEND\n"
            )
        )

        output_dir = tmp_path / "results"
        wf = self._make_workflow(
            tmp_path,
            [{"operation": "idealize"}],
            output="results/",
        )
        wf.run()

        assert output_dir.exists()
        pdb_files = list(output_dir.glob("*.pdb"))
        assert len(pdb_files) == 1
        assert "idealized" in pdb_files[0].stem

    @patch("boundry.workflow.Workflow._run_idealize")
    def test_directory_output_energy_json(self, mock_idealize, tmp_path):
        """Test that energy breakdown is written as JSON."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        mock_idealize.return_value = Structure(
            pdb_string=(
                "ATOM      1  N   ALA A   1       "
                "1.000   1.000   1.000  1.00  0.00"
                "           N\nEND\n"
            ),
            metadata={"energy_breakdown": {"total_energy": -100.0}},
        )

        wf = self._make_workflow(
            tmp_path,
            [{"operation": "idealize"}],
            output="results/",
        )
        wf.run()

        output_dir = tmp_path / "results"
        json_files = list(output_dir.glob("*_energy.json"))
        assert len(json_files) == 1

    @patch("boundry.workflow.Workflow._run_relax")
    def test_directory_output_with_tokens(self, mock_relax, tmp_path):
        """Test that tokens are incorporated into filenames."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        call_count = {"n": 0}

        def _side_effect(structure, params):
            call_count["n"] += 1
            return Structure(
                pdb_string=(
                    "ATOM      1  N   ALA A   1       "
                    "1.000   1.000   1.000  1.00  0.00"
                    "           N\nEND\n"
                ),
                metadata={"final_energy": -1.0 * call_count["n"]},
            )

        mock_relax.side_effect = _side_effect

        output_dir = tmp_path / "results"
        wf = self._make_workflow(
            tmp_path,
            [
                {
                    "iterate": {
                        "n": 2,
                        "output": str(output_dir) + "/",
                        "steps": [{"operation": "relax"}],
                    }
                }
            ],
        )
        wf.run()

        assert output_dir.exists()
        pdb_files = list(output_dir.glob("*.pdb"))
        assert len(pdb_files) == 2
        # Token should be in filename
        filenames = sorted(f.stem for f in pdb_files)
        assert any("cycle_1" in f for f in filenames)
        assert any("cycle_2" in f for f in filenames)

    @patch("boundry.workflow.Workflow._run_idealize")
    def test_directory_output_multiple_population(
        self, mock_idealize, tmp_path
    ):
        """Test that multiple structures get rank suffixes."""
        from boundry.operations import Structure

        self._make_input(tmp_path)
        mock_idealize.return_value = Structure(
            pdb_string=(
                "ATOM      1  N   ALA A   1       "
                "1.000   1.000   1.000  1.00  0.00"
                "           N\nEND\n"
            ),
        )

        wf = self._make_workflow(
            tmp_path,
            [{"operation": "idealize"}],
            output="results/",
        )
        # Manually set up a multi-structure population for _write_population
        from boundry.operations import Structure as S

        population = [
            S(pdb_string=(
                "ATOM      1  N   ALA A   1       "
                "1.000   1.000   1.000  1.00  0.00"
                "           N\nEND\n"
            )),
            S(pdb_string=(
                "ATOM      1  N   ALA A   1       "
                "2.000   2.000   2.000  1.00  0.00"
                "           N\nEND\n"
            )),
        ]

        output_dir = tmp_path / "results"
        wf._write_population(population, str(output_dir) + "/", operation="relax")

        pdb_files = list(output_dir.glob("*.pdb"))
        assert len(pdb_files) == 2
        filenames = sorted(f.stem for f in pdb_files)
        assert any("rank_1" in f for f in filenames)
        assert any("rank_2" in f for f in filenames)


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
        Workflow._run_idealize(struct, {"fix_cis_omega": False, "bogus": 42})

        _, kwargs = mock_op.call_args
        config = kwargs["config"]
        assert config.fix_cis_omega is False

    @patch("boundry.operations.minimize")
    def test_minimize_ignores_unknown_params(self, mock_op):
        from boundry.operations import Structure

        struct = Structure(pdb_string="ATOM\nEND\n")
        mock_op.return_value = struct

        Workflow._run_minimize(
            struct, {"constrained": True, "bogus": 42}
        )

        _, kwargs = mock_op.call_args
        config = kwargs["config"]
        assert config.constrained is True

    @patch("boundry.weights.ensure_weights")
    @patch("boundry.operations.repack")
    def test_repack_ignores_unknown_params(self, mock_op, mock_weights):
        from boundry.operations import Structure

        struct = Structure(pdb_string="ATOM\nEND\n")
        mock_op.return_value = struct

        Workflow._run_repack(
            struct, {"temperature": 0.2, "bogus": 42}
        )

        _, kwargs = mock_op.call_args
        config = kwargs["config"]
        assert config.temperature == 0.2

    @patch("boundry.weights.ensure_weights")
    @patch("boundry.operations.mpnn")
    def test_mpnn_ignores_unknown_params(self, mock_op, mock_weights):
        from boundry.operations import Structure

        struct = Structure(pdb_string="ATOM\nEND\n")
        mock_op.return_value = struct

        Workflow._run_mpnn(
            struct, {"temperature": 0.2, "bogus": 42}
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
        with pytest.raises(WorkflowError, match="seed.*must be an integer"):
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
        with pytest.raises(WorkflowError, match="seed.*must be an integer"):
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
                                "steps": [{"operation": "relax"}],
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

    def test_output_referenced_in_step(self, tmp_path):
        """${output} in a step output resolves to top-level output."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "output: results\n"
            "steps:\n"
            "  - operation: idealize\n"
            "    output: ${output}/idealized.pdb\n"
        )
        wf = Workflow.from_yaml(wf_file)
        assert wf.config.steps[0].output == "results/idealized.pdb"

    def test_output_referenced_in_iterate(self, tmp_path):
        """${output} in iterate output resolves correctly."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "output: results\n"
            "steps:\n"
            "  - iterate:\n"
            "      n: 3\n"
            "      output: ${output}/cycle_{cycle}/\n"
            "      steps:\n"
            "        - operation: relax\n"
        )
        wf = Workflow.from_yaml(wf_file)
        block = wf.config.steps[0]
        assert block.output == "results/cycle_{cycle}/"

    def test_input_referenceable(self, tmp_path):
        """${input} resolves to the top-level input value."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: structures/my_protein.pdb\n"
            "steps:\n"
            "  - operation: idealize\n"
            "    output: ${input}_idealized.pdb\n"
        )
        wf = Workflow.from_yaml(wf_file)
        assert (
            wf.config.steps[0].output
            == "structures/my_protein.pdb_idealized.pdb"
        )

    def test_user_var_in_step_output(self, tmp_path):
        """Custom user-defined key in step output."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "project: my_project\n"
            "steps:\n"
            "  - operation: idealize\n"
            "    output: ${project}/idealized.pdb\n"
        )
        wf = Workflow.from_yaml(wf_file)
        assert wf.config.steps[0].output == "my_project/idealized.pdb"

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
        assert wf.config.steps[0].params["resfile"] == "design.resfile"

    def test_cross_reference(self, tmp_path):
        """User var can reference another variable."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "output: results\n"
            "run_dir: ${output}/run_1\n"
            "steps:\n"
            "  - operation: idealize\n"
            "    output: ${run_dir}/idealized.pdb\n"
        )
        wf = Workflow.from_yaml(wf_file)
        assert (
            wf.config.steps[0].output
            == "results/run_1/idealized.pdb"
        )

    def test_multiple_refs_in_one_string(self, tmp_path):
        """Multiple ${...} in a single string."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "output: results\n"
            "project: myproj\n"
            "steps:\n"
            "  - operation: idealize\n"
            "    output: ${output}/${project}/final.pdb\n"
        )
        wf = Workflow.from_yaml(wf_file)
        assert (
            wf.config.steps[0].output
            == "results/myproj/final.pdb"
        )

    def test_vars_coexist_with_runtime_tokens(self, tmp_path):
        """${var} resolved at load, {cycle} preserved for runtime."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "output: results\n"
            "steps:\n"
            "  - iterate:\n"
            "      n: 2\n"
            "      output: ${output}/cycle_{cycle}/\n"
            "      steps:\n"
            "        - operation: relax\n"
        )
        wf = Workflow.from_yaml(wf_file)
        block = wf.config.steps[0]
        assert block.output == "results/cycle_{cycle}/"
        assert "${" not in block.output
        assert "{cycle}" in block.output

    def test_no_var_references(self, tmp_path):
        """Existing workflows without ${...} work unchanged."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            yaml.dump(
                {
                    "input": "input.pdb",
                    "output": "output.pdb",
                    "steps": [{"operation": "idealize"}],
                }
            )
        )
        wf = Workflow.from_yaml(wf_file)
        assert wf.config.input == "input.pdb"
        assert wf.config.output == "output.pdb"

    def test_numeric_var_coerced(self, tmp_path):
        """Numeric user var is coerced to string in namespace."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "run_id: 42\n"
            "steps:\n"
            "  - operation: idealize\n"
            "    output: run_${run_id}/idealized.pdb\n"
        )
        wf = Workflow.from_yaml(wf_file)
        assert wf.config.steps[0].output == "run_42/idealized.pdb"

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
            "    output: ${nonexistent}/out.pdb\n"
        )
        with pytest.raises(WorkflowError, match="resolution failed"):
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
        with pytest.raises(WorkflowError, match="Invalid variable name"):
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
        """${steps} is undefined because steps is non-scalar."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "steps:\n"
            "  - operation: idealize\n"
            "    output: ${steps}/out.pdb\n"
        )
        with pytest.raises(WorkflowError, match="resolution failed"):
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

    def test_partial_dollar_brace_raises(self, tmp_path):
        """Partial ${ without } raises a resolution error."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            'input: input.pdb\n'
            'steps:\n'
            '  - operation: idealize\n'
            '    output: "some_${_incomplete"\n'
        )
        with pytest.raises(WorkflowError, match="resolution failed"):
            Workflow.from_yaml(wf_file)

    def test_seed_in_namespace(self, tmp_path):
        """${seed} resolves to the YAML seed value."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "seed: 42\n"
            "steps:\n"
            "  - operation: idealize\n"
            "    output: run_seed_${seed}.pdb\n"
        )
        wf = Workflow.from_yaml(wf_file)
        assert wf.config.steps[0].output == "run_seed_42.pdb"


# ------------------------------------------------------------------
# CLI overrides
# ------------------------------------------------------------------


class TestCliOverrides:
    """Tests for the overrides parameter in from_yaml."""

    def test_override_output(self, tmp_path):
        """Override output via dotlist."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "output: original.pdb\n"
            "steps:\n"
            "  - operation: idealize\n"
        )
        wf = Workflow.from_yaml(
            wf_file, overrides=["output=overridden.pdb"]
        )
        assert wf.config.output == "overridden.pdb"

    def test_override_user_var(self, tmp_path):
        """Override a user-defined variable."""
        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(
            "input: input.pdb\n"
            "project: default_proj\n"
            "steps:\n"
            "  - operation: idealize\n"
            "    output: ${project}/out.pdb\n"
        )
        wf = Workflow.from_yaml(
            wf_file, overrides=["project=custom_proj"]
        )
        assert wf.config.steps[0].output == "custom_proj/out.pdb"

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
            "output: out.pdb\n"
            "steps:\n"
            "  - operation: idealize\n"
        )
        wf = Workflow.from_yaml(wf_file, overrides=None)
        assert wf.config.output == "out.pdb"


# ------------------------------------------------------------------
# Built-in workflow lookup
# ------------------------------------------------------------------


class TestBuiltinWorkflowLookup:
    """Tests for _resolve_workflow."""

    def test_existing_file_returned(self, tmp_path):
        """Explicit file path is returned as-is."""
        from boundry.cli import _resolve_workflow

        wf_file = tmp_path / "my.yaml"
        wf_file.write_text("input: x\nsteps:\n  - operation: idealize\n")
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
        wf_file.write_text("input: x\nsteps:\n  - operation: idealize\n")
        monkeypatch.setattr(_Path, "home", lambda: tmp_path)
        result = _resolve_workflow("custom")
        assert result == wf_file
