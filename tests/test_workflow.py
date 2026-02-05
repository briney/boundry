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
from boundry.workflow import VALID_OPERATIONS, Workflow, WorkflowError


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
        return Workflow.from_yaml(wf_file)

    def _make_input(self, tmp_path):
        """Write a dummy PDB file and return path."""
        pdb = tmp_path / "input.pdb"
        pdb.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000"
            "  1.00  0.00           N\nEND\n"
        )
        return pdb

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
        return Workflow.from_yaml(wf_file)

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
    def test_iterate_seed_injected(self, mock_relax, tmp_path):
        from boundry.operations import Structure

        self._make_input(tmp_path)
        seen_seeds = []

        def _side_effect(structure, params):
            seen_seeds.append(params["seed"])
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
                        "n": 3,
                        "seed": True,
                        "steps": [{"operation": "relax"}],
                    }
                }
            ],
        )
        wf.run()
        assert seen_seeds == [1, 2, 3]

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
        return Workflow.from_yaml(wf_file)

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
