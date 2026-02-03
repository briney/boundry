"""Tests for boundry.cli module (Typer-based CLI)."""

import pytest
from typer.testing import CliRunner

from boundry.cli import app

runner = CliRunner()


class TestAppStructure:
    """Tests for the CLI app structure and subcommands."""

    def test_no_args_shows_help(self):
        """Test that running with no args shows help text."""
        result = runner.invoke(app, [])
        # Typer shows help and exits (exit code may be 0 or 2)
        assert "idealize" in result.output or "Usage" in result.output

    def test_help_flag(self):
        """Test --help flag on the main app."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "idealize" in result.output
        assert "minimize" in result.output
        assert "repack" in result.output
        assert "relax" in result.output
        assert "mpnn" in result.output
        assert "design" in result.output
        assert "analyze-interface" in result.output
        assert "run" in result.output


class TestIdealize:
    """Tests for the idealize subcommand."""

    def test_help(self):
        """Test idealize --help."""
        result = runner.invoke(app, ["idealize", "--help"])
        assert result.exit_code == 0
        assert "backbone geometry" in result.output.lower()

    def test_missing_args(self):
        """Test idealize with no arguments fails."""
        result = runner.invoke(app, ["idealize"])
        assert result.exit_code != 0

    def test_missing_input_file(self, tmp_path):
        """Test idealize with nonexistent input file."""
        result = runner.invoke(
            app,
            [
                "idealize",
                str(tmp_path / "nonexistent.pdb"),
                str(tmp_path / "out.pdb"),
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_unsupported_format(self, tmp_path):
        """Test idealize with unsupported file format."""
        bad_file = tmp_path / "input.xyz"
        bad_file.write_text("dummy")
        result = runner.invoke(
            app,
            ["idealize", str(bad_file), str(tmp_path / "out.pdb")],
        )
        assert result.exit_code != 0
        assert "unsupported" in result.output.lower()


class TestMinimize:
    """Tests for the minimize subcommand."""

    def test_help(self):
        """Test minimize --help."""
        result = runner.invoke(app, ["minimize", "--help"])
        assert result.exit_code == 0
        assert "energy minimization" in result.output.lower()

    def test_missing_args(self):
        """Test minimize with no arguments fails."""
        result = runner.invoke(app, ["minimize"])
        assert result.exit_code != 0

    def test_has_constrained_option(self):
        """Test that --constrained option is available."""
        result = runner.invoke(app, ["minimize", "--help"])
        assert "--constrained" in result.output

    def test_has_pre_idealize_option(self):
        """Test that --pre-idealize option is available."""
        result = runner.invoke(app, ["minimize", "--help"])
        assert "--pre-idealize" in result.output


class TestRepack:
    """Tests for the repack subcommand."""

    def test_help(self):
        """Test repack --help."""
        result = runner.invoke(app, ["repack", "--help"])
        assert result.exit_code == 0
        assert "repack" in result.output.lower()

    def test_has_resfile_option(self):
        """Test that --resfile option is available."""
        result = runner.invoke(app, ["repack", "--help"])
        assert "--resfile" in result.output

    def test_has_temperature_option(self):
        """Test that --temperature option is available."""
        result = runner.invoke(app, ["repack", "--help"])
        assert "--temperature" in result.output

    def test_has_model_type_option(self):
        """Test that --model-type option is available."""
        result = runner.invoke(app, ["repack", "--help"])
        assert "--model-type" in result.output


class TestRelax:
    """Tests for the relax subcommand."""

    def test_help(self):
        """Test relax --help."""
        result = runner.invoke(app, ["relax", "--help"])
        assert result.exit_code == 0
        assert "repacking" in result.output.lower()

    def test_has_n_iter_option(self):
        """Test that --n-iter option is available."""
        result = runner.invoke(app, ["relax", "--help"])
        assert "--n-iter" in result.output

    def test_has_design_and_relax_options(self):
        """Test that both design and relaxation options are available."""
        result = runner.invoke(app, ["relax", "--help"])
        assert "--temperature" in result.output
        assert "--constrained" in result.output
        assert "--stiffness" in result.output


class TestMpnn:
    """Tests for the mpnn subcommand."""

    def test_help(self):
        """Test mpnn --help."""
        result = runner.invoke(app, ["mpnn", "--help"])
        assert result.exit_code == 0
        assert "sequence design" in result.output.lower()

    def test_has_design_options(self):
        """Test that design options are available."""
        result = runner.invoke(app, ["mpnn", "--help"])
        assert "--resfile" in result.output
        assert "--temperature" in result.output
        assert "--model-type" in result.output
        assert "--seed" in result.output


class TestDesign:
    """Tests for the design subcommand."""

    def test_help(self):
        """Test design --help."""
        result = runner.invoke(app, ["design", "--help"])
        assert result.exit_code == 0
        assert "design" in result.output.lower()

    def test_has_n_iter_option(self):
        """Test that --n-iter option is available."""
        result = runner.invoke(app, ["design", "--help"])
        assert "--n-iter" in result.output

    def test_has_design_and_relax_options(self):
        """Test that both design and relaxation options are available."""
        result = runner.invoke(app, ["design", "--help"])
        assert "--temperature" in result.output
        assert "--constrained" in result.output
        assert "--model-type" in result.output


class TestAnalyzeInterface:
    """Tests for the analyze-interface subcommand."""

    def test_help(self):
        """Test analyze-interface --help."""
        result = runner.invoke(app, ["analyze-interface", "--help"])
        assert result.exit_code == 0
        assert "interface" in result.output.lower()

    def test_has_chains_option(self):
        """Test that --chains option is available."""
        result = runner.invoke(app, ["analyze-interface", "--help"])
        assert "--chains" in result.output

    def test_has_analysis_options(self):
        """Test that analysis options are available."""
        result = runner.invoke(app, ["analyze-interface", "--help"])
        assert "--distance-cutoff" in result.output
        assert "--no-binding-energy" in result.output
        # Rich may truncate long option names; match prefix
        assert "--shape-complement" in result.output
        assert "--pack-separated" in result.output

    def test_no_output_file_argument(self):
        """Test that analyze-interface takes only one positional arg."""
        result = runner.invoke(app, ["analyze-interface", "--help"])
        # Should only have INPUT, not OUTPUT
        assert "INPUT" in result.output


class TestRun:
    """Tests for the run subcommand."""

    def test_help(self):
        """Test run --help."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "workflow" in result.output.lower()

    def test_missing_workflow_file(self, tmp_path):
        """Test run with nonexistent workflow file."""
        result = runner.invoke(
            app, ["run", str(tmp_path / "nonexistent.yaml")]
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()


class TestHelpers:
    """Tests for CLI helper functions."""

    def test_parse_chain_pairs(self):
        """Test chain pair parsing."""
        from boundry.cli import _parse_chain_pairs

        assert _parse_chain_pairs("H:L") == [("H", "L")]
        assert _parse_chain_pairs("H:L,H:A") == [("H", "L"), ("H", "A")]
        assert _parse_chain_pairs("A:B, C:D") == [("A", "B"), ("C", "D")]

    def test_parse_chain_pairs_empty(self):
        """Test chain pair parsing with invalid input."""
        from boundry.cli import _parse_chain_pairs

        assert _parse_chain_pairs("") == []
        assert _parse_chain_pairs("ABC") == []

    def test_verbose_option_on_all_commands(self):
        """Test that --verbose / -v is available on all commands."""
        commands = [
            "idealize",
            "minimize",
            "repack",
            "relax",
            "mpnn",
            "design",
            "analyze-interface",
            "run",
        ]
        for cmd in commands:
            result = runner.invoke(app, [cmd, "--help"])
            assert "--verbose" in result.output, (
                f"--verbose missing from {cmd}"
            )
            assert "-v" in result.output, f"-v missing from {cmd}"
