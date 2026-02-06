"""Tests for boundry.operations module.

Tests the core Python API: Structure data class, input resolution
helpers, and each standalone operation function. Heavy dependencies
(Designer, Relaxer, OpenMM, PyTorch) are mocked in unit tests.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from boundry.operations import (
    InterfaceAnalysisResult,
    Structure,
    _resolve_input,
    _write_temp_pdb,
)


# ------------------------------------------------------------------
# Test PDB strings
# ------------------------------------------------------------------

SINGLE_CHAIN_PDB = (
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000"
    "  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1       1.458   0.000   0.000"
    "  1.00  0.00           C\n"
    "ATOM      3  C   ALA A   1       2.009   1.420   0.000"
    "  1.00  0.00           C\n"
    "ATOM      4  O   ALA A   1       1.246   2.390   0.000"
    "  1.00  0.00           O\n"
    "ATOM      5  CB  ALA A   1       1.986  -0.760  -1.216"
    "  1.00  0.00           C\n"
    "END\n"
)

TWO_CHAIN_PDB = (
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000"
    "  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1       1.458   0.000   0.000"
    "  1.00  0.00           C\n"
    "TER\n"
    "ATOM      3  N   ALA B   1       2.000   0.000   3.500"
    "  1.00  0.00           N\n"
    "ATOM      4  CA  ALA B   1       3.458   0.000   3.500"
    "  1.00  0.00           C\n"
    "TER\n"
    "END\n"
)


# ------------------------------------------------------------------
# Structure data class
# ------------------------------------------------------------------


class TestStructure:
    """Tests for the Structure data class."""

    def test_construction_defaults(self):
        """Test constructing a Structure with defaults."""
        s = Structure(pdb_string="ATOM\nEND\n")
        assert s.pdb_string == "ATOM\nEND\n"
        assert s.metadata == {}
        assert s.source_path is None

    def test_construction_with_metadata(self):
        """Test constructing a Structure with metadata."""
        s = Structure(
            pdb_string="ATOM\nEND\n",
            metadata={"energy": -100.0, "operation": "minimize"},
            source_path="/path/to/input.pdb",
        )
        assert s.metadata["energy"] == -100.0
        assert s.metadata["operation"] == "minimize"
        assert s.source_path == "/path/to/input.pdb"

    def test_from_file_pdb(self, tmp_path):
        """Test loading a Structure from a PDB file."""
        pdb_path = tmp_path / "test.pdb"
        pdb_path.write_text(SINGLE_CHAIN_PDB)

        s = Structure.from_file(pdb_path)
        assert "ATOM" in s.pdb_string
        assert "ALA" in s.pdb_string
        assert s.source_path == str(pdb_path)

    def test_from_file_str_path(self, tmp_path):
        """Test loading a Structure from a string path."""
        pdb_path = tmp_path / "test.pdb"
        pdb_path.write_text(SINGLE_CHAIN_PDB)

        s = Structure.from_file(str(pdb_path))
        assert "ATOM" in s.pdb_string
        assert s.source_path == str(pdb_path)

    def test_from_file_cif(self, small_peptide_cif):
        """Test loading a Structure from a CIF file."""
        s = Structure.from_file(small_peptide_cif)
        # CIF is converted to PDB internally
        assert "ATOM" in s.pdb_string
        assert s.source_path == str(small_peptide_cif)

    def test_write_pdb(self, tmp_path):
        """Test writing a Structure to a PDB file."""
        s = Structure(pdb_string=SINGLE_CHAIN_PDB)
        out_path = tmp_path / "output.pdb"
        s.write(out_path)

        assert out_path.exists()
        content = out_path.read_text()
        assert "ATOM" in content
        assert "ALA" in content

    def test_write_cif(self, tmp_path):
        """Test writing a Structure to a CIF file."""
        s = Structure(pdb_string=SINGLE_CHAIN_PDB)
        out_path = tmp_path / "output.cif"
        s.write(out_path)

        assert out_path.exists()
        content = out_path.read_text()
        # CIF files contain _atom_site entries
        assert "_atom_site" in content

    def test_write_string_path(self, tmp_path):
        """Test writing with a string path."""
        s = Structure(pdb_string=SINGLE_CHAIN_PDB)
        out_path = str(tmp_path / "output.pdb")
        s.write(out_path)

        assert Path(out_path).exists()

    def test_metadata_is_independent(self):
        """Test that metadata dicts are independent across instances."""
        s1 = Structure(pdb_string="A")
        s2 = Structure(pdb_string="B")
        s1.metadata["key"] = "value"
        assert "key" not in s2.metadata


# ------------------------------------------------------------------
# InterfaceAnalysisResult
# ------------------------------------------------------------------


class TestInterfaceAnalysisResult:
    """Tests for the InterfaceAnalysisResult data class."""

    def test_defaults_are_none(self):
        """Test that all fields default to None."""
        r = InterfaceAnalysisResult()
        assert r.interface_info is None
        assert r.binding_energy is None
        assert r.sasa is None
        assert r.shape_complementarity is None

    def test_to_metadata_empty(self):
        """Test to_metadata with no results populated."""
        r = InterfaceAnalysisResult()
        meta = r.to_metadata()
        assert meta["operation"] == "analyze_interface"
        assert meta["metrics"]["interface"] == {}

    def test_to_metadata_binding_energy(self):
        """Test to_metadata with binding energy populated."""
        be = MagicMock()
        be.binding_energy = -5.0
        be.complex_energy = -100.0
        r = InterfaceAnalysisResult(binding_energy=be)
        meta = r.to_metadata()
        assert meta["dG"] == -5.0
        assert meta["complex_energy"] == -100.0
        assert meta["metrics"]["interface"]["dG"] == -5.0
        assert meta["metrics"]["interface"]["complex_energy"] == -100.0

    def test_to_metadata_sasa(self):
        """Test to_metadata with SASA populated."""
        sasa = MagicMock()
        sasa.buried_sasa = 500.0
        r = InterfaceAnalysisResult(sasa=sasa)
        meta = r.to_metadata()
        assert meta["buried_sasa"] == 500.0
        assert meta["metrics"]["interface"]["buried_sasa"] == 500.0

    def test_to_metadata_shape_complementarity(self):
        """Test to_metadata with shape complementarity populated."""
        sc = MagicMock()
        sc.sc_score = 0.75
        r = InterfaceAnalysisResult(shape_complementarity=sc)
        meta = r.to_metadata()
        assert meta["sc_score"] == 0.75
        assert meta["metrics"]["interface"]["sc_score"] == 0.75

    def test_to_metadata_interface_info(self):
        """Test to_metadata with interface info populated."""
        info = MagicMock()
        info.n_interface_residues = 42
        r = InterfaceAnalysisResult(interface_info=info)
        meta = r.to_metadata()
        assert meta["n_interface_residues"] == 42
        assert meta["metrics"]["interface"]["n_interface_residues"] == 42

    def test_to_metadata_per_position(self):
        """Test to_metadata with per-position results."""
        pp = MagicMock()
        r = InterfaceAnalysisResult(per_position=pp)
        meta = r.to_metadata()
        assert meta["per_position"] is pp

    def test_to_structure(self):
        """Test to_structure merges metadata and returns Structure."""
        be = MagicMock()
        be.binding_energy = -5.0
        be.complex_energy = -100.0
        r = InterfaceAnalysisResult(binding_energy=be)
        s = r.to_structure(
            "ATOM\nEND\n",
            source_path="/test.pdb",
            base_metadata={"existing": True},
        )
        assert isinstance(s, Structure)
        assert s.pdb_string == "ATOM\nEND\n"
        assert s.source_path == "/test.pdb"
        assert s.metadata["existing"] is True
        assert s.metadata["dG"] == -5.0
        assert s.metadata["operation"] == "analyze_interface"

    def test_to_structure_no_base_metadata(self):
        """Test to_structure without base_metadata."""
        r = InterfaceAnalysisResult()
        s = r.to_structure("ATOM\nEND\n")
        assert s.metadata["operation"] == "analyze_interface"


# ------------------------------------------------------------------
# _resolve_input helper
# ------------------------------------------------------------------


class TestResolveInput:
    """Tests for the _resolve_input helper function."""

    def test_structure_object(self):
        """Test resolving a Structure object."""
        s = Structure(
            pdb_string="ATOM\nEND\n",
            source_path="/input.pdb",
        )
        pdb_string, source_path = _resolve_input(s)
        assert pdb_string == "ATOM\nEND\n"
        assert source_path == "/input.pdb"

    def test_path_object(self, tmp_path):
        """Test resolving a Path to an existing file."""
        pdb_path = tmp_path / "test.pdb"
        pdb_path.write_text(SINGLE_CHAIN_PDB)

        pdb_string, source_path = _resolve_input(pdb_path)
        assert "ATOM" in pdb_string
        assert source_path == str(pdb_path)

    def test_str_file_path(self, tmp_path):
        """Test resolving a string file path."""
        pdb_path = tmp_path / "test.pdb"
        pdb_path.write_text(SINGLE_CHAIN_PDB)

        pdb_string, source_path = _resolve_input(str(pdb_path))
        assert "ATOM" in pdb_string
        assert source_path == str(pdb_path)

    def test_str_pdb_string(self):
        """Test resolving a PDB string (no file on disk)."""
        pdb_string, source_path = _resolve_input(SINGLE_CHAIN_PDB)
        assert pdb_string == SINGLE_CHAIN_PDB
        assert source_path is None

    def test_invalid_type_raises(self):
        """Test that invalid types raise TypeError."""
        with pytest.raises(TypeError, match="Expected"):
            _resolve_input(42)


# ------------------------------------------------------------------
# _write_temp_pdb helper
# ------------------------------------------------------------------


class TestWriteTempPdb:
    """Tests for the _write_temp_pdb helper function."""

    def test_creates_file(self):
        """Test that a temporary file is created with content."""
        path = _write_temp_pdb(SINGLE_CHAIN_PDB)
        try:
            assert path.exists()
            assert path.suffix == ".pdb"
            assert path.read_text() == SINGLE_CHAIN_PDB
        finally:
            path.unlink(missing_ok=True)


# ------------------------------------------------------------------
# idealize operation
# ------------------------------------------------------------------


class TestIdealize:
    """Tests for the idealize() operation."""

    @patch("boundry.idealize.idealize_structure")
    def test_returns_structure(self, mock_idealize):
        """Test that idealize returns a Structure."""
        mock_idealize.return_value = ("IDEALIZED\nEND\n", [])

        from boundry.operations import idealize

        result = idealize(SINGLE_CHAIN_PDB)

        assert isinstance(result, Structure)
        assert result.pdb_string == "IDEALIZED\nEND\n"

    @patch("boundry.idealize.idealize_structure")
    def test_default_config(self, mock_idealize):
        """Test that default config has enabled=True."""
        mock_idealize.return_value = ("IDEALIZED\nEND\n", [])

        from boundry.operations import idealize

        idealize(SINGLE_CHAIN_PDB)

        config_arg = mock_idealize.call_args[0][1]
        assert config_arg.enabled is True

    @patch("boundry.idealize.idealize_structure")
    def test_custom_config(self, mock_idealize):
        """Test passing a custom IdealizeConfig."""
        from boundry.config import IdealizeConfig
        from boundry.operations import idealize

        mock_idealize.return_value = ("IDEALIZED\nEND\n", [])
        cfg = IdealizeConfig(
            enabled=True,
            fix_cis_omega=False,
        )

        idealize(SINGLE_CHAIN_PDB, config=cfg)

        config_arg = mock_idealize.call_args[0][1]
        assert config_arg.fix_cis_omega is False

    @patch("boundry.idealize.idealize_structure")
    def test_metadata(self, mock_idealize):
        """Test that metadata includes chain_gaps and operation."""
        mock_idealize.return_value = ("IDEALIZED\nEND\n", ["gap1"])

        from boundry.operations import idealize

        result = idealize(SINGLE_CHAIN_PDB)

        assert result.metadata["chain_gaps"] == 1
        assert result.metadata["operation"] == "idealize"

    @patch("boundry.idealize.idealize_structure")
    def test_source_path_from_file(self, mock_idealize, tmp_path):
        """Test that source_path is set when input is a file."""
        mock_idealize.return_value = ("IDEALIZED\nEND\n", [])

        from boundry.operations import idealize

        pdb_path = tmp_path / "input.pdb"
        pdb_path.write_text(SINGLE_CHAIN_PDB)

        result = idealize(pdb_path)
        assert result.source_path == str(pdb_path)

    @patch("boundry.idealize.idealize_structure")
    def test_source_path_from_string(self, mock_idealize):
        """Test that source_path is None when input is a PDB string."""
        mock_idealize.return_value = ("IDEALIZED\nEND\n", [])

        from boundry.operations import idealize

        result = idealize(SINGLE_CHAIN_PDB)
        assert result.source_path is None

    @patch("boundry.idealize.idealize_structure")
    def test_accepts_structure_input(self, mock_idealize):
        """Test that idealize accepts a Structure object."""
        mock_idealize.return_value = ("IDEALIZED\nEND\n", [])

        from boundry.operations import idealize

        s = Structure(pdb_string=SINGLE_CHAIN_PDB, source_path="/in.pdb")
        result = idealize(s)
        assert result.source_path == "/in.pdb"


# ------------------------------------------------------------------
# minimize operation
# ------------------------------------------------------------------


class TestMinimize:
    """Tests for the minimize() operation."""

    @patch("boundry.relaxer.Relaxer")
    def test_returns_structure(self, MockRelaxer):
        """Test that minimize returns a Structure with energy metadata."""
        mock_relaxer = MockRelaxer.return_value
        mock_relaxer.relax.return_value = (
            "RELAXED\nEND\n",
            {
                "initial_energy": -50.0,
                "final_energy": -100.0,
                "rmsd": 0.5,
            },
            [],
        )

        from boundry.operations import minimize

        result = minimize(SINGLE_CHAIN_PDB)

        assert isinstance(result, Structure)
        assert result.pdb_string == "RELAXED\nEND\n"
        assert result.metadata["initial_energy"] == -50.0
        assert result.metadata["final_energy"] == -100.0
        assert result.metadata["rmsd"] == 0.5
        assert result.metadata["operation"] == "minimize"

    @patch("boundry.relaxer.Relaxer")
    def test_default_config(self, MockRelaxer):
        """Test that a default RelaxConfig is created."""
        mock_relaxer = MockRelaxer.return_value
        mock_relaxer.relax.return_value = (
            "RELAXED\nEND\n",
            {"initial_energy": 0, "final_energy": 0, "rmsd": 0},
            [],
        )

        from boundry.operations import minimize

        minimize(SINGLE_CHAIN_PDB)

        config_arg = MockRelaxer.call_args[0][0]
        assert config_arg.constrained is False

    @patch("boundry.relaxer.Relaxer")
    def test_custom_config(self, MockRelaxer):
        """Test passing a custom RelaxConfig."""
        from boundry.config import RelaxConfig
        from boundry.operations import minimize

        mock_relaxer = MockRelaxer.return_value
        mock_relaxer.relax.return_value = (
            "RELAXED\nEND\n",
            {"initial_energy": 0, "final_energy": 0, "rmsd": 0},
            [],
        )

        cfg = RelaxConfig(constrained=True, max_iterations=500)
        minimize(SINGLE_CHAIN_PDB, config=cfg)

        config_arg = MockRelaxer.call_args[0][0]
        assert config_arg.constrained is True
        assert config_arg.max_iterations == 500

    @patch("boundry.idealize.idealize_structure")
    @patch("boundry.relaxer.Relaxer")
    def test_pre_idealize(self, MockRelaxer, mock_idealize):
        """Test that pre_idealize runs idealization first."""
        mock_idealize.return_value = ("IDEALIZED\nEND\n", [])
        mock_relaxer = MockRelaxer.return_value
        mock_relaxer.relax.return_value = (
            "RELAXED\nEND\n",
            {"initial_energy": 0, "final_energy": 0, "rmsd": 0},
            [],
        )

        from boundry.operations import minimize

        minimize(SINGLE_CHAIN_PDB, pre_idealize=True)

        mock_idealize.assert_called_once()
        # Relaxer.relax should receive the idealized PDB
        relaxed_input = mock_relaxer.relax.call_args[0][0]
        assert relaxed_input == "IDEALIZED\nEND\n"


# ------------------------------------------------------------------
# repack operation
# ------------------------------------------------------------------


class TestRepack:
    """Tests for the repack() operation."""

    @patch("boundry.designer.Designer")
    def test_returns_structure(self, MockDesigner):
        """Test that repack returns a Structure with sequence metadata."""
        mock_designer = MockDesigner.return_value
        mock_designer.repack.return_value = {
            "sequence": "AAAAA",
            "native_sequence": "AAAAA",
            "loss": [0.5],
        }
        mock_designer.result_to_pdb_string.return_value = (
            "REPACKED\nEND\n"
        )

        from boundry.operations import repack

        result = repack(SINGLE_CHAIN_PDB)

        assert isinstance(result, Structure)
        assert result.pdb_string == "REPACKED\nEND\n"
        assert result.metadata["sequence"] == "AAAAA"
        assert result.metadata["native_sequence"] == "AAAAA"
        assert result.metadata["ligandmpnn_loss"] == 0.5
        assert result.metadata["operation"] == "repack"

    @patch("boundry.designer.Designer")
    def test_default_config(self, MockDesigner):
        """Test that a default DesignConfig is created."""
        mock_designer = MockDesigner.return_value
        mock_designer.repack.return_value = {
            "sequence": "A",
            "native_sequence": "A",
            "loss": [0.1],
        }
        mock_designer.result_to_pdb_string.return_value = "PDB\n"

        from boundry.operations import repack

        repack(SINGLE_CHAIN_PDB)

        config_arg = MockDesigner.call_args[0][0]
        assert config_arg.model_type == "ligand_mpnn"

    @patch("boundry.designer.Designer")
    def test_with_resfile(self, MockDesigner, tmp_path):
        """Test that a resfile is parsed and passed to designer."""
        mock_designer = MockDesigner.return_value
        mock_designer.repack.return_value = {
            "sequence": "A",
            "native_sequence": "A",
            "loss": [0.1],
        }
        mock_designer.result_to_pdb_string.return_value = "PDB\n"

        resfile = tmp_path / "test.resfile"
        resfile.write_text("NATAA\nSTART\n10 A ALLAA\n")

        from boundry.operations import repack

        repack(SINGLE_CHAIN_PDB, resfile=resfile)

        call_kwargs = mock_designer.repack.call_args[1]
        assert call_kwargs["design_spec"] is not None

    @patch("boundry.idealize.idealize_structure")
    @patch("boundry.designer.Designer")
    def test_pre_idealize(self, MockDesigner, mock_idealize):
        """Test that pre_idealize runs idealization first."""
        mock_idealize.return_value = ("IDEALIZED\nEND\n", [])
        mock_designer = MockDesigner.return_value
        mock_designer.repack.return_value = {
            "sequence": "A",
            "native_sequence": "A",
            "loss": [0.1],
        }
        mock_designer.result_to_pdb_string.return_value = "PDB\n"

        from boundry.operations import repack

        repack(SINGLE_CHAIN_PDB, pre_idealize=True)

        mock_idealize.assert_called_once()

    @patch("boundry.designer.Designer")
    def test_temp_file_cleanup(self, MockDesigner):
        """Test that temporary PDB files are cleaned up."""
        mock_designer = MockDesigner.return_value
        mock_designer.repack.return_value = {
            "sequence": "A",
            "native_sequence": "A",
            "loss": [0.1],
        }
        mock_designer.result_to_pdb_string.return_value = "PDB\n"

        from boundry.operations import repack

        repack(SINGLE_CHAIN_PDB)

        # The temp file path was passed to designer.repack
        pdb_path = mock_designer.repack.call_args[0][0]
        assert not pdb_path.exists()


# ------------------------------------------------------------------
# relax operation
# ------------------------------------------------------------------


class TestRelax:
    """Tests for the relax() operation (repack + minimize loop)."""

    def _setup_mocks(self, MockDesigner, MockRelaxer):
        """Set up standard mocks for relax tests."""
        mock_designer = MockDesigner.return_value
        mock_designer.repack.return_value = {
            "sequence": "AAAAA",
            "native_sequence": "AAAAA",
            "loss": [0.3],
        }
        mock_designer.result_to_pdb_string.return_value = (
            "REPACKED\nEND\n"
        )

        mock_relaxer = MockRelaxer.return_value
        mock_relaxer.relax.return_value = (
            "RELAXED\nEND\n",
            {
                "initial_energy": -50.0,
                "final_energy": -100.0,
                "rmsd": 0.5,
            },
            [],
        )
        mock_relaxer.get_energy_breakdown.return_value = {
            "total_energy": -100.0,
            "bond": -10.0,
        }

        return mock_designer, mock_relaxer

    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.designer.Designer")
    def test_returns_structure(self, MockDesigner, MockRelaxer):
        """Test that relax returns a Structure with iteration metadata."""
        self._setup_mocks(MockDesigner, MockRelaxer)

        from boundry.operations import relax

        result = relax(SINGLE_CHAIN_PDB, n_iterations=2)

        assert isinstance(result, Structure)
        assert result.metadata["operation"] == "relax"
        assert result.metadata["final_energy"] == -100.0
        assert result.metadata["sequence"] == "AAAAA"
        assert len(result.metadata["iterations"]) == 2

    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.designer.Designer")
    def test_iteration_count(self, MockDesigner, MockRelaxer):
        """Test that the correct number of iterations are run."""
        mock_designer, mock_relaxer = self._setup_mocks(
            MockDesigner, MockRelaxer
        )

        from boundry.operations import relax

        relax(SINGLE_CHAIN_PDB, n_iterations=3)

        assert mock_designer.repack.call_count == 3
        assert mock_relaxer.relax.call_count == 3

    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.designer.Designer")
    def test_energy_breakdown(self, MockDesigner, MockRelaxer):
        """Test that energy breakdown is included."""
        self._setup_mocks(MockDesigner, MockRelaxer)

        from boundry.operations import relax

        result = relax(SINGLE_CHAIN_PDB, n_iterations=1)

        assert "energy_breakdown" in result.metadata
        assert result.metadata["energy_breakdown"]["total_energy"] == -100.0

    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.designer.Designer")
    def test_iteration_metadata(self, MockDesigner, MockRelaxer):
        """Test that each iteration records energy and sequence."""
        self._setup_mocks(MockDesigner, MockRelaxer)

        from boundry.operations import relax

        result = relax(SINGLE_CHAIN_PDB, n_iterations=2)

        for i, it in enumerate(result.metadata["iterations"], 1):
            assert it["iteration"] == i
            assert "initial_energy" in it
            assert "final_energy" in it
            assert "rmsd" in it
            assert "sequence" in it

    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.designer.Designer")
    def test_default_config(self, MockDesigner, MockRelaxer):
        """Test that a default PipelineConfig is created."""
        self._setup_mocks(MockDesigner, MockRelaxer)

        from boundry.operations import relax

        relax(SINGLE_CHAIN_PDB, n_iterations=1)

        design_config = MockDesigner.call_args[0][0]
        relax_config = MockRelaxer.call_args[0][0]
        assert design_config.model_type == "ligand_mpnn"
        assert relax_config.constrained is False

    @patch("boundry.idealize.idealize_structure")
    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.designer.Designer")
    def test_pre_idealize(
        self, MockDesigner, MockRelaxer, mock_idealize
    ):
        """Test that pre_idealize runs idealization first."""
        mock_idealize.return_value = ("IDEALIZED\nEND\n", [])
        self._setup_mocks(MockDesigner, MockRelaxer)

        from boundry.operations import relax

        relax(SINGLE_CHAIN_PDB, pre_idealize=True, n_iterations=1)

        mock_idealize.assert_called_once()

    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.designer.Designer")
    def test_with_resfile(self, MockDesigner, MockRelaxer, tmp_path):
        """Test that a resfile is parsed and passed through."""
        self._setup_mocks(MockDesigner, MockRelaxer)

        resfile = tmp_path / "test.resfile"
        resfile.write_text("NATAA\nSTART\n10 A ALLAA\n")

        from boundry.operations import relax

        relax(SINGLE_CHAIN_PDB, resfile=resfile, n_iterations=1)

        call_kwargs = MockDesigner.return_value.repack.call_args[1]
        assert call_kwargs["design_spec"] is not None


# ------------------------------------------------------------------
# mpnn operation
# ------------------------------------------------------------------


class TestMpnn:
    """Tests for the mpnn() operation."""

    @patch("boundry.utils.format_sequence_alignment")
    @patch("boundry.designer.Designer")
    def test_returns_structure(self, MockDesigner, mock_align):
        """Test that mpnn returns a Structure with design metadata."""
        mock_designer = MockDesigner.return_value
        mock_designer.design.return_value = {
            "sequence": "MKTLV",
            "native_sequence": "AAAAA",
            "loss": [0.8],
        }
        mock_designer.result_to_pdb_string.return_value = (
            "DESIGNED\nEND\n"
        )
        mock_align.return_value = "alignment"

        from boundry.operations import mpnn

        result = mpnn(SINGLE_CHAIN_PDB)

        assert isinstance(result, Structure)
        assert result.pdb_string == "DESIGNED\nEND\n"
        assert result.metadata["sequence"] == "MKTLV"
        assert result.metadata["native_sequence"] == "AAAAA"
        assert result.metadata["ligandmpnn_loss"] == 0.8
        assert result.metadata["operation"] == "mpnn"

    @patch("boundry.utils.format_sequence_alignment")
    @patch("boundry.designer.Designer")
    def test_design_all_by_default(self, MockDesigner, mock_align):
        """Test that design_all=True when no resfile is given."""
        mock_designer = MockDesigner.return_value
        mock_designer.design.return_value = {
            "sequence": "M",
            "native_sequence": "A",
            "loss": [0.1],
        }
        mock_designer.result_to_pdb_string.return_value = "PDB\n"
        mock_align.return_value = ""

        from boundry.operations import mpnn

        mpnn(SINGLE_CHAIN_PDB)

        call_kwargs = mock_designer.design.call_args[1]
        assert call_kwargs["design_all"] is True

    @patch("boundry.utils.format_sequence_alignment")
    @patch("boundry.designer.Designer")
    def test_resfile_disables_design_all(
        self, MockDesigner, mock_align, tmp_path
    ):
        """Test that providing a resfile sets design_all=False."""
        mock_designer = MockDesigner.return_value
        mock_designer.design.return_value = {
            "sequence": "M",
            "native_sequence": "A",
            "loss": [0.1],
        }
        mock_designer.result_to_pdb_string.return_value = "PDB\n"
        mock_align.return_value = ""

        resfile = tmp_path / "test.resfile"
        resfile.write_text("NATAA\nSTART\n10 A ALLAA\n")

        from boundry.operations import mpnn

        mpnn(SINGLE_CHAIN_PDB, resfile=resfile)

        call_kwargs = mock_designer.design.call_args[1]
        assert call_kwargs["design_all"] is False
        assert call_kwargs["design_spec"] is not None

    @patch("boundry.idealize.idealize_structure")
    @patch("boundry.utils.format_sequence_alignment")
    @patch("boundry.designer.Designer")
    def test_pre_idealize(
        self, MockDesigner, mock_align, mock_idealize
    ):
        """Test that pre_idealize runs idealization first."""
        mock_idealize.return_value = ("IDEALIZED\nEND\n", [])
        mock_designer = MockDesigner.return_value
        mock_designer.design.return_value = {
            "sequence": "M",
            "native_sequence": "A",
            "loss": [0.1],
        }
        mock_designer.result_to_pdb_string.return_value = "PDB\n"
        mock_align.return_value = ""

        from boundry.operations import mpnn

        mpnn(SINGLE_CHAIN_PDB, pre_idealize=True)

        mock_idealize.assert_called_once()

    @patch("boundry.utils.format_sequence_alignment")
    @patch("boundry.designer.Designer")
    def test_temp_file_cleanup(self, MockDesigner, mock_align):
        """Test that temporary PDB files are cleaned up."""
        mock_designer = MockDesigner.return_value
        mock_designer.design.return_value = {
            "sequence": "M",
            "native_sequence": "A",
            "loss": [0.1],
        }
        mock_designer.result_to_pdb_string.return_value = "PDB\n"
        mock_align.return_value = ""

        from boundry.operations import mpnn

        mpnn(SINGLE_CHAIN_PDB)

        pdb_path = mock_designer.design.call_args[0][0]
        assert not pdb_path.exists()


# ------------------------------------------------------------------
# design operation
# ------------------------------------------------------------------


class TestDesign:
    """Tests for the design() operation (mpnn + minimize loop)."""

    def _setup_mocks(self, MockDesigner, MockRelaxer, mock_align):
        """Set up standard mocks for design tests."""
        mock_designer = MockDesigner.return_value
        mock_designer.design.return_value = {
            "sequence": "MKTLV",
            "native_sequence": "AAAAA",
            "loss": [0.8],
        }
        mock_designer.result_to_pdb_string.return_value = (
            "DESIGNED\nEND\n"
        )
        mock_align.return_value = "alignment"

        mock_relaxer = MockRelaxer.return_value
        mock_relaxer.relax.return_value = (
            "RELAXED\nEND\n",
            {
                "initial_energy": -50.0,
                "final_energy": -100.0,
                "rmsd": 0.5,
            },
            [],
        )
        mock_relaxer.get_energy_breakdown.return_value = {
            "total_energy": -100.0,
        }

        return mock_designer, mock_relaxer

    @patch("boundry.utils.format_sequence_alignment")
    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.designer.Designer")
    def test_returns_structure(
        self, MockDesigner, MockRelaxer, mock_align
    ):
        """Test that design returns a Structure with full metadata."""
        self._setup_mocks(MockDesigner, MockRelaxer, mock_align)

        from boundry.operations import design

        result = design(SINGLE_CHAIN_PDB, n_iterations=2)

        assert isinstance(result, Structure)
        assert result.metadata["operation"] == "design"
        assert result.metadata["final_energy"] == -100.0
        assert result.metadata["sequence"] == "MKTLV"
        assert result.metadata["native_sequence"] == "AAAAA"
        assert result.metadata["ligandmpnn_loss"] == 0.8
        assert len(result.metadata["iterations"]) == 2

    @patch("boundry.utils.format_sequence_alignment")
    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.designer.Designer")
    def test_iteration_count(
        self, MockDesigner, MockRelaxer, mock_align
    ):
        """Test that the correct number of iterations are run."""
        mock_designer, mock_relaxer = self._setup_mocks(
            MockDesigner, MockRelaxer, mock_align
        )

        from boundry.operations import design

        design(SINGLE_CHAIN_PDB, n_iterations=3)

        assert mock_designer.design.call_count == 3
        assert mock_relaxer.relax.call_count == 3

    @patch("boundry.utils.format_sequence_alignment")
    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.designer.Designer")
    def test_iteration_metadata(
        self, MockDesigner, MockRelaxer, mock_align
    ):
        """Test that each iteration records all fields."""
        self._setup_mocks(MockDesigner, MockRelaxer, mock_align)

        from boundry.operations import design

        result = design(SINGLE_CHAIN_PDB, n_iterations=2)

        for i, it in enumerate(result.metadata["iterations"], 1):
            assert it["iteration"] == i
            assert "initial_energy" in it
            assert "final_energy" in it
            assert "rmsd" in it
            assert "sequence" in it
            assert "native_sequence" in it
            assert "ligandmpnn_loss" in it

    @patch("boundry.utils.format_sequence_alignment")
    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.designer.Designer")
    def test_energy_breakdown(
        self, MockDesigner, MockRelaxer, mock_align
    ):
        """Test that energy breakdown is included."""
        self._setup_mocks(MockDesigner, MockRelaxer, mock_align)

        from boundry.operations import design

        result = design(SINGLE_CHAIN_PDB, n_iterations=1)

        assert "energy_breakdown" in result.metadata

    @patch("boundry.idealize.idealize_structure")
    @patch("boundry.utils.format_sequence_alignment")
    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.designer.Designer")
    def test_pre_idealize(
        self,
        MockDesigner,
        MockRelaxer,
        mock_align,
        mock_idealize,
    ):
        """Test that pre_idealize runs idealization first."""
        mock_idealize.return_value = ("IDEALIZED\nEND\n", [])
        self._setup_mocks(MockDesigner, MockRelaxer, mock_align)

        from boundry.operations import design

        design(SINGLE_CHAIN_PDB, pre_idealize=True, n_iterations=1)

        mock_idealize.assert_called_once()


# ------------------------------------------------------------------
# analyze_interface operation
# ------------------------------------------------------------------


class TestAnalyzeInterface:
    """Tests for the analyze_interface() operation."""

    @patch("boundry.interface.identify_interface_residues")
    def test_returns_result(self, mock_identify):
        """Test basic return type with interface residues found."""
        mock_info = MagicMock()
        mock_info.interface_residues = [MagicMock()]
        mock_info.chain_pairs = [("A", "B")]
        mock_info.summary = "1 residue"
        mock_identify.return_value = mock_info

        from boundry.config import InterfaceConfig
        from boundry.operations import analyze_interface

        config = InterfaceConfig(
            enabled=True,
            calculate_binding_energy=False,
            calculate_sasa=False,
            calculate_shape_complementarity=False,
        )
        result = analyze_interface(TWO_CHAIN_PDB, config=config)

        assert isinstance(result, InterfaceAnalysisResult)
        assert result.interface_info is mock_info

    @patch("boundry.interface.identify_interface_residues")
    def test_empty_interface(self, mock_identify):
        """Test early return when no interface residues found."""
        mock_info = MagicMock()
        mock_info.interface_residues = []
        mock_info.chain_pairs = []
        mock_info.summary = "0 residues"
        mock_identify.return_value = mock_info

        from boundry.operations import analyze_interface

        result = analyze_interface(TWO_CHAIN_PDB)

        assert result.interface_info is mock_info
        assert result.binding_energy is None
        assert result.sasa is None
        assert result.shape_complementarity is None

    @patch("boundry.surface_area.calculate_surface_area")
    @patch("boundry.interface.identify_interface_residues")
    def test_sasa_calculation(self, mock_identify, mock_sasa):
        """Test that SASA is calculated when enabled."""
        mock_info = MagicMock()
        mock_info.interface_residues = [MagicMock()]
        mock_info.chain_pairs = [("A", "B")]
        mock_info.summary = "1 residue"
        mock_identify.return_value = mock_info

        mock_sasa_result = MagicMock()
        mock_sasa_result.buried_sasa = 500.0
        mock_sasa.return_value = mock_sasa_result

        from boundry.config import InterfaceConfig
        from boundry.operations import analyze_interface

        config = InterfaceConfig(
            enabled=True,
            calculate_sasa=True,
            calculate_binding_energy=False,
            calculate_shape_complementarity=False,
        )
        result = analyze_interface(TWO_CHAIN_PDB, config=config)

        mock_sasa.assert_called_once()
        assert result.sasa is mock_sasa_result

    @patch("boundry.binding_energy.calculate_binding_energy")
    @patch("boundry.interface.identify_interface_residues")
    def test_binding_energy_requires_relaxer(
        self, mock_identify, mock_be
    ):
        """Test that binding energy is skipped without a relaxer."""
        mock_info = MagicMock()
        mock_info.interface_residues = [MagicMock()]
        mock_info.chain_pairs = [("A", "B")]
        mock_info.summary = "1 residue"
        mock_identify.return_value = mock_info

        from boundry.config import InterfaceConfig
        from boundry.operations import analyze_interface

        config = InterfaceConfig(
            enabled=True,
            calculate_binding_energy=True,
            calculate_sasa=False,
            calculate_shape_complementarity=False,
        )
        result = analyze_interface(TWO_CHAIN_PDB, config=config)

        mock_be.assert_not_called()
        assert result.binding_energy is None

    @patch("boundry.binding_energy.calculate_binding_energy")
    @patch("boundry.interface.identify_interface_residues")
    def test_binding_energy_with_relaxer(
        self, mock_identify, mock_be
    ):
        """Test binding energy is calculated when relaxer provided."""
        mock_info = MagicMock()
        mock_info.interface_residues = [MagicMock()]
        mock_info.chain_pairs = [("A", "B")]
        mock_info.summary = "1 residue"
        mock_identify.return_value = mock_info

        mock_be_result = MagicMock()
        mock_be_result.binding_energy = 42.0
        mock_be.return_value = mock_be_result

        from boundry.config import InterfaceConfig
        from boundry.operations import analyze_interface

        config = InterfaceConfig(
            enabled=True,
            calculate_binding_energy=True,
            calculate_sasa=False,
            calculate_shape_complementarity=False,
        )
        mock_relaxer = MagicMock()
        result = analyze_interface(
            TWO_CHAIN_PDB, config=config, relaxer=mock_relaxer
        )

        mock_be.assert_called_once()
        assert result.binding_energy is mock_be_result

    @patch(
        "boundry.surface_area.calculate_shape_complementarity"
    )
    @patch("boundry.interface.identify_interface_residues")
    def test_shape_complementarity(self, mock_identify, mock_sc):
        """Test shape complementarity is calculated when enabled."""
        mock_info = MagicMock()
        mock_info.interface_residues = [MagicMock()]
        mock_info.chain_pairs = [("A", "B")]
        mock_info.summary = "1 residue"
        mock_identify.return_value = mock_info

        mock_sc_result = MagicMock()
        mock_sc_result.sc_score = 0.75
        mock_sc.return_value = mock_sc_result

        from boundry.config import InterfaceConfig
        from boundry.operations import analyze_interface

        config = InterfaceConfig(
            enabled=True,
            calculate_sasa=False,
            calculate_binding_energy=False,
            calculate_shape_complementarity=True,
        )
        result = analyze_interface(TWO_CHAIN_PDB, config=config)

        mock_sc.assert_called_once()
        assert result.shape_complementarity is mock_sc_result

    @patch("boundry.interface.identify_interface_residues")
    def test_default_config(self, mock_identify):
        """Test that default config has enabled=True."""
        mock_info = MagicMock()
        mock_info.interface_residues = []
        mock_info.chain_pairs = []
        mock_info.summary = "0 residues"
        mock_identify.return_value = mock_info

        from boundry.operations import analyze_interface

        analyze_interface(TWO_CHAIN_PDB)

        call_kwargs = mock_identify.call_args[1]
        assert call_kwargs["distance_cutoff"] == 8.0

    @patch("boundry.interface.identify_interface_residues")
    def test_custom_distance_cutoff(self, mock_identify):
        """Test custom distance cutoff is passed through."""
        mock_info = MagicMock()
        mock_info.interface_residues = []
        mock_info.chain_pairs = []
        mock_info.summary = "0 residues"
        mock_identify.return_value = mock_info

        from boundry.config import InterfaceConfig
        from boundry.operations import analyze_interface

        config = InterfaceConfig(
            enabled=True,
            distance_cutoff=5.0,
            calculate_binding_energy=False,
            calculate_sasa=False,
        )
        analyze_interface(TWO_CHAIN_PDB, config=config)

        call_kwargs = mock_identify.call_args[1]
        assert call_kwargs["distance_cutoff"] == 5.0

    @patch("boundry.interface.identify_interface_residues")
    def test_accepts_structure_input(self, mock_identify):
        """Test that analyze_interface accepts a Structure object."""
        mock_info = MagicMock()
        mock_info.interface_residues = []
        mock_info.chain_pairs = []
        mock_info.summary = "0 residues"
        mock_identify.return_value = mock_info

        from boundry.operations import analyze_interface

        s = Structure(pdb_string=TWO_CHAIN_PDB)
        analyze_interface(s)

        call_args = mock_identify.call_args[0]
        assert call_args[0] == TWO_CHAIN_PDB


# ------------------------------------------------------------------
# Top-level import tests
# ------------------------------------------------------------------


class TestTopLevelImports:
    """Test that operations are accessible from the boundry package."""

    def test_operations_importable(self):
        """Test that all operations are importable.

        The ``idealize`` function is imported from ``boundry.operations``
        because the ``boundry.idealize`` sub-module shadows the function
        name once it has been imported by other tests.
        """
        from boundry import (
            analyze_interface,
            design,
            minimize,
            mpnn,
            repack,
            relax,
        )
        from boundry.operations import idealize

        assert callable(idealize)
        assert callable(minimize)
        assert callable(repack)
        assert callable(relax)
        assert callable(mpnn)
        assert callable(design)
        assert callable(analyze_interface)

    def test_structure_importable(self):
        """Test that Structure is importable from boundry."""
        from boundry import Structure

        s = Structure(pdb_string="ATOM\nEND\n")
        assert s.pdb_string == "ATOM\nEND\n"

    def test_interface_analysis_result_importable(self):
        """Test that InterfaceAnalysisResult is importable."""
        from boundry import InterfaceAnalysisResult

        r = InterfaceAnalysisResult()
        assert r.interface_info is None

    def test_workflow_importable(self):
        """Test that Workflow is importable from boundry."""
        from boundry import Workflow

        assert hasattr(Workflow, "from_yaml")
        assert hasattr(Workflow, "run")

    def test_config_classes_importable(self):
        """Test that all config classes are importable."""
        from boundry import (
            DesignConfig,
            IdealizeConfig,
            InterfaceConfig,
            PipelineConfig,
            RelaxConfig,
            WorkflowConfig,
            WorkflowStep,
        )

        assert DesignConfig().model_type == "ligand_mpnn"
        assert RelaxConfig().constrained is False
        assert IdealizeConfig().enabled is False
        assert InterfaceConfig().enabled is False
        assert PipelineConfig().n_iterations == 5
        assert WorkflowConfig(input="in.pdb").input == "in.pdb"
        assert WorkflowStep(operation="idealize").operation == "idealize"

    def test_resfile_classes_importable(self):
        """Test that resfile classes are importable."""
        from boundry import (
            DesignSpec,
            ResfileParser,
            ResidueMode,
            ResidueSpec,
        )

        assert ResidueMode.ALLAA.value == "ALLAA"
        assert callable(ResfileParser().parse)

    def test_renumber_importable(self):
        """Test that renumber is importable from boundry.

        The ``renumber`` function is imported from
        ``boundry.operations`` because the ``boundry.renumber``
        sub-module shadows the function name once it has been
        imported by other tests.
        """
        from boundry.operations import renumber

        assert callable(renumber)

    def test_pipeline_removed(self):
        """Test that Pipeline and PipelineMode are no longer exposed."""
        import boundry

        assert not hasattr(boundry, "Pipeline")
        assert not hasattr(boundry, "PipelineMode")


# ------------------------------------------------------------------
# Auto-renumber in operations
# ------------------------------------------------------------------


# PDB with Kabat insertion codes for auto-renumber tests
ICODE_PDB = (
    "ATOM      1  N   ALA H  99       0.000   0.000   0.000"
    "  1.00  0.00           N\n"
    "ATOM      2  CA  ALA H  99       1.458   0.000   0.000"
    "  1.00  0.00           C\n"
    "ATOM      3  N   GLY H 100       3.326   1.540   0.000"
    "  1.00  0.00           N\n"
    "ATOM      4  CA  GLY H 100       3.941   2.861   0.000"
    "  1.00  0.00           C\n"
    "ATOM      5  N   ALA H 100A      5.000   3.000   0.000"
    "  1.00  0.00           N\n"
    "ATOM      6  CA  ALA H 100A      6.000   4.000   0.000"
    "  1.00  0.00           C\n"
    "ATOM      7  N   TYR H 101       9.000   7.000   0.000"
    "  1.00  0.00           N\n"
    "ATOM      8  CA  TYR H 101      10.000   8.000   0.000"
    "  1.00  0.00           C\n"
    "END\n"
)


class TestAutoRenumber:
    """Tests for auto-renumber in affected operations."""

    @patch("boundry.relaxer.Relaxer")
    def test_minimize_auto_renumber(self, MockRelaxer):
        """Minimize transparently handles insertion codes."""
        mock_relaxer = MockRelaxer.return_value
        # The relaxer receives renumbered PDB (no icodes) and
        # returns it as-is for simplicity
        mock_relaxer.relax.return_value = (
            "ATOM      1  N   ALA H   1       0.000   0.000   0.000"
            "  1.00  0.00           N\n"
            "ATOM      2  CA  ALA H   1       1.458   0.000   0.000"
            "  1.00  0.00           C\n"
            "ATOM      3  N   GLY H   2       3.326   1.540   0.000"
            "  1.00  0.00           N\n"
            "ATOM      4  CA  GLY H   2       3.941   2.861   0.000"
            "  1.00  0.00           C\n"
            "ATOM      5  N   ALA H   3       5.000   3.000   0.000"
            "  1.00  0.00           N\n"
            "ATOM      6  CA  ALA H   3       6.000   4.000   0.000"
            "  1.00  0.00           C\n"
            "ATOM      7  N   TYR H   4       9.000   7.000   0.000"
            "  1.00  0.00           N\n"
            "ATOM      8  CA  TYR H   4      10.000   8.000   0.000"
            "  1.00  0.00           C\n"
            "END\n",
            {
                "initial_energy": -50.0,
                "final_energy": -100.0,
                "rmsd": 0.5,
            },
            [],
        )

        from boundry.operations import minimize

        result = minimize(ICODE_PDB)

        # Output should have original numbering restored
        assert "renumber_mapping" in result.metadata
        # Check that insertion codes are restored in output
        lines = result.pdb_string.splitlines()
        atom_lines = [l for l in lines if l.startswith("ATOM")]
        # Residue 100A should be present in output
        icodes = [l[26] for l in atom_lines]
        assert "A" in icodes

    @patch("boundry.relaxer.Relaxer")
    def test_auto_renumber_noop(self, MockRelaxer):
        """No insertion codes = no renumbering applied."""
        mock_relaxer = MockRelaxer.return_value
        mock_relaxer.relax.return_value = (
            "RELAXED\nEND\n",
            {
                "initial_energy": -50.0,
                "final_energy": -100.0,
                "rmsd": 0.5,
            },
            [],
        )

        from boundry.operations import minimize

        result = minimize(SINGLE_CHAIN_PDB)

        # No renumber_mapping when no insertion codes
        assert "renumber_mapping" not in result.metadata


# ------------------------------------------------------------------
# filter_protein_only utility
# ------------------------------------------------------------------


class TestFilterProteinOnly:
    """Tests for the filter_protein_only utility function."""

    def test_strips_hetatm(self):
        """Test that HETATM lines (waters, ligands) are removed."""
        from boundry.utils import filter_protein_only

        pdb = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000"
            "  1.00  0.00           N\n"
            "HETATM    2  O   HOH A   2       1.000   1.000   1.000"
            "  1.00  0.00           O\n"
            "HETATM    3  C1  NAG M   1      10.000  10.000  10.000"
            "  1.00  0.00           C\n"
            "END\n"
        )
        result = filter_protein_only(pdb)
        assert "HETATM" not in result
        assert "HOH" not in result
        assert "NAG" not in result

    def test_preserves_atom_records(self):
        """Test that ATOM records are preserved."""
        from boundry.utils import filter_protein_only

        pdb = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000"
            "  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       1.458   0.000   0.000"
            "  1.00  0.00           C\n"
            "END\n"
        )
        result = filter_protein_only(pdb)
        atom_lines = [
            l for l in result.splitlines() if l.startswith("ATOM")
        ]
        assert len(atom_lines) == 2

    def test_adds_ter_between_chains(self):
        """Test that TER records are added between chains."""
        from boundry.utils import filter_protein_only

        result = filter_protein_only(TWO_CHAIN_PDB)
        lines = result.strip().splitlines()
        ter_count = sum(1 for l in lines if l.startswith("TER"))
        assert ter_count == 2  # One between A/B, one after B

    def test_ends_with_end(self):
        """Test that output ends with END record."""
        from boundry.utils import filter_protein_only

        result = filter_protein_only(SINGLE_CHAIN_PDB)
        lines = result.strip().splitlines()
        assert lines[-1] == "END"

    def test_protein_only_input_unchanged(self):
        """Test that protein-only PDB retains all ATOM lines."""
        from boundry.utils import filter_protein_only

        result = filter_protein_only(SINGLE_CHAIN_PDB)
        original_atoms = [
            l
            for l in SINGLE_CHAIN_PDB.splitlines()
            if l.startswith("ATOM")
        ]
        result_atoms = [
            l for l in result.splitlines() if l.startswith("ATOM")
        ]
        assert len(result_atoms) == len(original_atoms)


# ------------------------------------------------------------------
# None energy handling
# ------------------------------------------------------------------


class TestNoneEnergyHandling:
    """Tests for None energy propagation through the pipeline."""

    @patch("boundry.binding_energy.calculate_binding_energy")
    @patch("boundry.interface.identify_interface_residues")
    def test_analyze_interface_none_binding_energy(
        self, mock_identify, mock_be
    ):
        """Test that None binding_energy is handled gracefully."""
        from boundry.binding_energy import BindingEnergyResult
        from boundry.config import InterfaceConfig
        from boundry.operations import analyze_interface

        mock_info = MagicMock()
        mock_info.interface_residues = [MagicMock()]
        mock_info.chain_pairs = [("A", "B")]
        mock_info.summary = "1 residue"
        mock_identify.return_value = mock_info

        mock_be.return_value = BindingEnergyResult(
            complex_energy=None,
            binding_energy=None,
        )

        config = InterfaceConfig(
            enabled=True,
            calculate_binding_energy=True,
            calculate_sasa=False,
            calculate_shape_complementarity=False,
        )
        mock_relaxer = MagicMock()
        result = analyze_interface(
            TWO_CHAIN_PDB, config=config, relaxer=mock_relaxer
        )

        assert result.binding_energy is not None
        assert result.binding_energy.binding_energy is None

    def test_binding_energy_result_optional_fields(self):
        """Test BindingEnergyResult accepts None for energy fields."""
        from boundry.binding_energy import BindingEnergyResult

        result = BindingEnergyResult(
            complex_energy=None,
            binding_energy=None,
        )
        assert result.complex_energy is None
        assert result.binding_energy is None

    def test_compute_rosetta_dG_raises_on_none(self):
        """Test _compute_rosetta_dG raises RuntimeError on None."""
        from boundry.binding_energy import BindingEnergyResult
        from boundry.interface_position_energetics import (
            _compute_rosetta_dG,
        )

        mock_relaxer = MagicMock()
        mock_relaxer.get_energy_breakdown.return_value = {
            "total_energy": None,
        }

        with patch(
            "boundry.interface_position_energetics"
            ".calculate_binding_energy"
        ) as mock_be:
            mock_be.return_value = BindingEnergyResult(
                binding_energy=None,
            )
            with pytest.raises(RuntimeError, match="failed"):
                _compute_rosetta_dG(
                    SINGLE_CHAIN_PDB,
                    mock_relaxer,
                    chain_pairs=[("A", "B")],
                )
