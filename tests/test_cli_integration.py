"""CLI integration tests for boundry.

These tests verify the CLI interface works correctly end-to-end.
"""

import subprocess
import sys

import pytest

from boundry.weights import weights_exist as weights_available

# Skip entire module if OpenMM not available
pytest.importorskip("openmm")


class TestCLIHelp:
    """Tests for CLI help and version output."""

    def test_help_output(self):
        """Test that --help produces output and exits successfully."""
        result = subprocess.run(
            [sys.executable, "-m", "boundry.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "boundry" in result.stdout.lower()

    def test_help_shows_subcommands(self):
        """Test that help shows all available subcommands."""
        result = subprocess.run(
            [sys.executable, "-m", "boundry.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert "idealize" in result.stdout
        assert "minimize" in result.stdout
        assert "repack" in result.stdout
        assert "relax" in result.stdout
        assert "mpnn" in result.stdout
        assert "design" in result.stdout
        assert "analyze-interface" in result.stdout
        assert "run" in result.stdout

    def test_subcommand_help_shows_positional_args(self):
        """Test that subcommand help shows positional INPUT/OUTPUT args."""
        result = subprocess.run(
            [sys.executable, "-m", "boundry.cli", "minimize", "--help"],
            capture_output=True,
            text=True,
        )
        assert "INPUT" in result.stdout
        assert "OUTPUT" in result.stdout


class TestCLIArgumentValidation:
    """Tests for CLI argument validation."""

    def test_missing_args_fails(self):
        """Test that calling a subcommand with no args fails."""
        result = subprocess.run(
            [sys.executable, "-m", "boundry.cli", "minimize"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_nonexistent_input_fails(self, tmp_path):
        """Test that nonexistent input file fails with error."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "boundry.cli",
                "minimize",
                str(tmp_path / "nonexistent.pdb"),
                str(tmp_path / "output.pdb"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        combined = result.stderr + result.stdout
        assert "not found" in combined.lower()

    def test_unsupported_format_fails(self, tmp_path):
        """Test that unsupported file format fails with error."""
        bad_file = tmp_path / "input.xyz"
        bad_file.write_text("dummy")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "boundry.cli",
                "idealize",
                str(bad_file),
                str(tmp_path / "output.pdb"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        combined = result.stderr + result.stdout
        assert "unsupported" in combined.lower()


@pytest.mark.integration
class TestCLIMinimize:
    """Integration tests for CLI minimize subcommand."""

    def test_minimize_completes(
        self, small_peptide_pdb, tmp_path, weights_available
    ):
        """Test that minimize completes successfully."""
        output_pdb = tmp_path / "minimized.pdb"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "boundry.cli",
                "minimize",
                str(small_peptide_pdb),
                str(output_pdb),
                "--max-iterations",
                "50",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"CLI failed with: {result.stderr}"
        assert output_pdb.exists()

    def test_minimize_creates_valid_output(
        self, small_peptide_pdb, tmp_path, weights_available
    ):
        """Test that minimize creates valid PDB output."""
        output_pdb = tmp_path / "minimized.pdb"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "boundry.cli",
                "minimize",
                str(small_peptide_pdb),
                str(output_pdb),
                "--max-iterations",
                "50",
            ],
            capture_output=True,
            text=True,
        )

        content = output_pdb.read_text()
        assert "ATOM" in content


@pytest.mark.integration
@pytest.mark.skipif(
    not weights_available(),
    reason="LigandMPNN weights not downloaded",
)
class TestCLIDesign:
    """Integration tests for CLI design subcommand (requires weights)."""

    def test_design_completes(self, small_peptide_pdb, tmp_path):
        """Test that design subcommand completes successfully."""
        output_pdb = tmp_path / "designed.pdb"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "boundry.cli",
                "design",
                str(small_peptide_pdb),
                str(output_pdb),
                "--n-iter",
                "1",
                "--max-iterations",
                "50",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, f"CLI failed with: {result.stderr}"
        assert output_pdb.exists()

    def test_design_with_resfile(self, small_peptide_pdb, tmp_path):
        """Test design with resfile constraints."""
        resfile = tmp_path / "design.resfile"
        resfile.write_text(
            """NATAA
START
1 A ALLAA
2 A NATRO
3 A ALLAA
4 A NATRO
5 A ALLAA
"""
        )

        output_pdb = tmp_path / "designed.pdb"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "boundry.cli",
                "design",
                str(small_peptide_pdb),
                str(output_pdb),
                "--resfile",
                str(resfile),
                "--n-iter",
                "1",
                "--max-iterations",
                "50",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0
        assert output_pdb.exists()


@pytest.mark.integration
@pytest.mark.skipif(
    not weights_available(),
    reason="LigandMPNN weights not downloaded",
)
class TestCLIRelax:
    """Integration tests for CLI relax subcommand (requires weights)."""

    def test_relax_completes(self, small_peptide_pdb, tmp_path):
        """Test that relax subcommand completes successfully."""
        output_pdb = tmp_path / "relaxed.pdb"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "boundry.cli",
                "relax",
                str(small_peptide_pdb),
                str(output_pdb),
                "--n-iter",
                "1",
                "--max-iterations",
                "50",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, f"CLI failed with: {result.stderr}"
        assert output_pdb.exists()


class TestCLIVerbosity:
    """Tests for CLI verbosity options."""

    @pytest.mark.integration
    def test_verbose_output(
        self, small_peptide_pdb, tmp_path, weights_available
    ):
        """Test that -v produces more output."""
        output_pdb = tmp_path / "minimized.pdb"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "boundry.cli",
                "minimize",
                str(small_peptide_pdb),
                str(output_pdb),
                "--max-iterations",
                "50",
                "-v",
            ],
            capture_output=True,
            text=True,
        )

        # Verbose mode should produce more detailed logging
        assert result.returncode == 0
        # Check for some logging output (stderr or stdout depending on config)
        output = result.stderr + result.stdout
        assert len(output) > 0
