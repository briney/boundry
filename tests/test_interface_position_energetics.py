"""Tests for boundry.interface_position_energetics module.

Tests the PDB mutation/removal helpers, sign conventions, alanine
scan math, per-position dG math, CSV output, and hotspot table
formatting.  Heavy dependencies (Relaxer, Designer, OpenMM) are
mocked.
"""

import csv
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from boundry.interface import InterfaceResidue
from boundry.interface_position_energetics import (
    PositionEnergeticsResult,
    PositionResult,
    PositionRow,
    ResidueKey,
    _ALANINE_SCAN_SKIP,
    _select_scan_sites,
    compute_alanine_scan,
    compute_per_position_dG,
    compute_position_energetics,
    format_hotspot_table,
    format_position_table,
    mutate_to_alanine,
    remove_residue,
    write_position_csv,
)

# ------------------------------------------------------------------
# Test PDB strings
# ------------------------------------------------------------------

# A minimal two-chain PDB with LEU on chain A and VAL on chain B
TWO_CHAIN_PDB = (
    "ATOM      1  N   LEU A   1       0.000   0.000   0.000"
    "  1.00  0.00           N\n"
    "ATOM      2  CA  LEU A   1       1.458   0.000   0.000"
    "  1.00  0.00           C\n"
    "ATOM      3  C   LEU A   1       2.009   1.420   0.000"
    "  1.00  0.00           C\n"
    "ATOM      4  O   LEU A   1       1.246   2.390   0.000"
    "  1.00  0.00           O\n"
    "ATOM      5  CB  LEU A   1       1.986  -0.760  -1.216"
    "  1.00  0.00           C\n"
    "ATOM      6  CG  LEU A   1       1.500  -2.200  -1.216"
    "  1.00  0.00           C\n"
    "ATOM      7  CD1 LEU A   1       2.028  -2.960  -2.432"
    "  1.00  0.00           C\n"
    "ATOM      8  CD2 LEU A   1       1.928  -2.860   0.100"
    "  1.00  0.00           C\n"
    "TER\n"
    "ATOM      9  N   VAL B   1       5.000   0.000   0.000"
    "  1.00  0.00           N\n"
    "ATOM     10  CA  VAL B   1       6.458   0.000   0.000"
    "  1.00  0.00           C\n"
    "ATOM     11  C   VAL B   1       7.009   1.420   0.000"
    "  1.00  0.00           C\n"
    "ATOM     12  O   VAL B   1       6.246   2.390   0.000"
    "  1.00  0.00           O\n"
    "ATOM     13  CB  VAL B   1       6.986  -0.760  -1.216"
    "  1.00  0.00           C\n"
    "ATOM     14  CG1 VAL B   1       6.500  -2.200  -1.216"
    "  1.00  0.00           C\n"
    "ATOM     15  CG2 VAL B   1       8.500  -0.760  -1.216"
    "  1.00  0.00           C\n"
    "TER\n"
    "END\n"
)

# PDB with GLY, PRO, ALA residues (scan-skip test)
SKIP_RESIDUES_PDB = (
    "ATOM      1  N   GLY A   1       0.000   0.000   0.000"
    "  1.00  0.00           N\n"
    "ATOM      2  CA  GLY A   1       1.458   0.000   0.000"
    "  1.00  0.00           C\n"
    "ATOM      3  C   GLY A   1       2.009   1.420   0.000"
    "  1.00  0.00           C\n"
    "ATOM      4  O   GLY A   1       1.246   2.390   0.000"
    "  1.00  0.00           O\n"
    "ATOM      5  N   PRO A   2       3.326   1.540   0.000"
    "  1.00  0.00           N\n"
    "ATOM      6  CA  PRO A   2       3.941   2.861   0.000"
    "  1.00  0.00           C\n"
    "ATOM      7  C   PRO A   2       5.459   2.789   0.000"
    "  1.00  0.00           C\n"
    "ATOM      8  O   PRO A   2       6.065   1.719   0.000"
    "  1.00  0.00           O\n"
    "ATOM      9  CB  PRO A   2       3.473   3.699   1.186"
    "  1.00  0.00           C\n"
    "ATOM     10  CG  PRO A   2       2.000   3.699   1.186"
    "  1.00  0.00           C\n"
    "ATOM     11  CD  PRO A   2       1.500   2.500   0.400"
    "  1.00  0.00           C\n"
    "ATOM     12  N   ALA A   3       6.063   3.970   0.000"
    "  1.00  0.00           N\n"
    "ATOM     13  CA  ALA A   3       7.510   4.096   0.000"
    "  1.00  0.00           C\n"
    "ATOM     14  C   ALA A   3       8.061   5.516   0.000"
    "  1.00  0.00           C\n"
    "ATOM     15  O   ALA A   3       7.298   6.486   0.000"
    "  1.00  0.00           O\n"
    "ATOM     16  CB  ALA A   3       8.038   3.336  -1.216"
    "  1.00  0.00           C\n"
    "END\n"
)


def _make_ir(
    chain_id="A",
    residue_number=1,
    residue_name="LEU",
    insertion_code="",
    partner_chain="B",
    min_distance=3.5,
    num_contacts=5,
):
    return InterfaceResidue(
        chain_id=chain_id,
        residue_number=residue_number,
        residue_name=residue_name,
        insertion_code=insertion_code,
        partner_chain=partner_chain,
        min_distance=min_distance,
        num_contacts=num_contacts,
    )


# ------------------------------------------------------------------
# ResidueKey
# ------------------------------------------------------------------


class TestResidueKey:
    def test_str_representation(self):
        key = ResidueKey("A", 42, "")
        assert str(key) == "A42"

    def test_str_with_insertion_code(self):
        key = ResidueKey("B", 10, "A")
        assert str(key) == "B10A"

    def test_frozen(self):
        key = ResidueKey("A", 1)
        with pytest.raises(AttributeError):
            key.chain_id = "B"

    def test_hashable(self):
        k1 = ResidueKey("A", 1, "")
        k2 = ResidueKey("A", 1, "")
        assert k1 == k2
        assert hash(k1) == hash(k2)
        assert len({k1, k2}) == 1


# ------------------------------------------------------------------
# mutate_to_alanine
# ------------------------------------------------------------------


class TestMutateToAlanine:
    def test_renames_residue(self):
        result = mutate_to_alanine(TWO_CHAIN_PDB, "A", 1)
        # All chain A, resnum 1 lines should say ALA
        for line in result.splitlines():
            if line.startswith("ATOM") and line[21] == "A":
                assert line[17:20] == "ALA"

    def test_removes_side_chain_atoms(self):
        result = mutate_to_alanine(TWO_CHAIN_PDB, "A", 1)
        atom_names = []
        for line in result.splitlines():
            if line.startswith("ATOM") and line[21] == "A":
                atom_names.append(line[12:16].strip())
        # LEU has CG, CD1, CD2 which should be removed
        assert "CG" not in atom_names
        assert "CD1" not in atom_names
        assert "CD2" not in atom_names
        # Backbone + CB should remain
        assert "N" in atom_names
        assert "CA" in atom_names
        assert "C" in atom_names
        assert "O" in atom_names
        assert "CB" in atom_names

    def test_preserves_other_chains(self):
        result = mutate_to_alanine(TWO_CHAIN_PDB, "A", 1)
        chain_b_lines = [
            l for l in result.splitlines()
            if l.startswith("ATOM") and l[21] == "B"
        ]
        # Chain B should be untouched (VAL with CG1, CG2)
        assert len(chain_b_lines) == 7  # N, CA, C, O, CB, CG1, CG2

    def test_preserves_ter_and_end(self):
        result = mutate_to_alanine(TWO_CHAIN_PDB, "A", 1)
        assert "TER" in result
        assert "END" in result

    def test_already_alanine(self):
        """Mutating ALA to ALA should be a no-op."""
        pdb = (
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
        result = mutate_to_alanine(pdb, "A", 1)
        atom_count = sum(
            1 for l in result.splitlines() if l.startswith("ATOM")
        )
        assert atom_count == 5  # All atoms preserved


# ------------------------------------------------------------------
# remove_residue
# ------------------------------------------------------------------


class TestRemoveResidue:
    def test_removes_target_atoms(self):
        result = remove_residue(TWO_CHAIN_PDB, "A", 1)
        chain_a_lines = [
            l for l in result.splitlines()
            if l.startswith("ATOM") and l[21] == "A"
        ]
        assert len(chain_a_lines) == 0

    def test_preserves_other_chains(self):
        result = remove_residue(TWO_CHAIN_PDB, "A", 1)
        chain_b_lines = [
            l for l in result.splitlines()
            if l.startswith("ATOM") and l[21] == "B"
        ]
        assert len(chain_b_lines) == 7

    def test_preserves_end_record(self):
        result = remove_residue(TWO_CHAIN_PDB, "A", 1)
        assert result.rstrip().endswith("END")

    def test_remove_nonexistent(self):
        """Removing a residue that doesn't exist returns PDB unchanged."""
        result = remove_residue(TWO_CHAIN_PDB, "C", 99)
        assert result == TWO_CHAIN_PDB


# ------------------------------------------------------------------
# Scan-site selection
# ------------------------------------------------------------------


class TestSelectScanSites:
    def test_deduplicates(self):
        """Duplicate interface residues (same chain/resnum/icode) are merged."""
        residues = [
            _make_ir(chain_id="A", residue_number=1, partner_chain="B"),
            _make_ir(chain_id="A", residue_number=1, partner_chain="C"),
        ]
        sites = _select_scan_sites(residues)
        assert len(sites) == 1

    def test_filters_by_scan_chains(self):
        residues = [
            _make_ir(chain_id="A", residue_number=1),
            _make_ir(chain_id="B", residue_number=1, residue_name="VAL"),
        ]
        sites = _select_scan_sites(residues, scan_chains=["A"])
        assert len(sites) == 1
        assert sites[0].chain_id == "A"

    def test_max_scan_sites(self):
        residues = [
            _make_ir(chain_id="A", residue_number=i)
            for i in range(1, 11)
        ]
        sites = _select_scan_sites(residues, max_scan_sites=3)
        assert len(sites) == 3

    def test_sorted_by_chain_resnum(self):
        residues = [
            _make_ir(chain_id="B", residue_number=5),
            _make_ir(chain_id="A", residue_number=10),
            _make_ir(chain_id="A", residue_number=2),
        ]
        sites = _select_scan_sites(residues)
        keys = [(s.chain_id, s.residue_number) for s in sites]
        assert keys == [("A", 2), ("A", 10), ("B", 5)]


# ------------------------------------------------------------------
# Sign convention tests (mocked)
# ------------------------------------------------------------------


class TestSignConvention:
    """Verify that dG is computed as E_bound - E_unbound (negative = favorable)."""

    @patch("boundry.interface_position_energetics.calculate_binding_energy")
    def test_dG_wt_passthrough(self, mock_calc):
        """dG_wt should match binding_energy directly (no sign inversion)."""
        from boundry.interface_position_energetics import _compute_rosetta_dG

        mock_result = MagicMock()
        mock_result.binding_energy = -10.0  # negative = favorable
        mock_calc.return_value = mock_result

        relaxer = MagicMock()
        dG = _compute_rosetta_dG(
            "PDB", relaxer, chain_pairs=[("A", "B")]
        )
        # dG should be passed through unchanged
        assert dG == -10.0


# ------------------------------------------------------------------
# Alanine scan math (mocked)
# ------------------------------------------------------------------


class TestAlanineScanMath:
    """Test the alanine scan ddG computation with mocked energies."""

    @patch("boundry.interface_position_energetics._compute_rosetta_dG")
    def test_ddG_computation(self, mock_dG):
        """ddG = dG_ala - dG_wt."""
        # dG_ala for the mutant
        mock_dG.return_value = -5.0

        relaxer = MagicMock()
        ir = _make_ir(residue_name="LEU")

        results = compute_alanine_scan(
            TWO_CHAIN_PDB,
            [ir],
            chain_pairs=[("A", "B")],
            relaxer=relaxer,
            dG_wt=-10.0,
            position_relax="none",
        )

        key = ResidueKey("A", 1, "")
        dG_ala, ddG = results[key]
        assert dG_ala == -5.0
        # ddG = -5.0 - (-10.0) = 5.0 (positive = destabilising)
        assert ddG == 5.0

    @patch("boundry.interface_position_energetics._compute_rosetta_dG")
    def test_skips_gly_pro_ala(self, mock_dG):
        """GLY, PRO, ALA should be skipped with (None, None) result."""
        relaxer = MagicMock()
        residues = [
            _make_ir(residue_name="GLY", residue_number=1),
            _make_ir(residue_name="PRO", residue_number=2),
            _make_ir(residue_name="ALA", residue_number=3),
            _make_ir(residue_name="LEU", residue_number=4),
        ]

        mock_dG.return_value = -5.0

        results = compute_alanine_scan(
            TWO_CHAIN_PDB,
            residues,
            chain_pairs=[("A", "B")],
            relaxer=relaxer,
            dG_wt=-10.0,
            position_relax="none",
        )

        # GLY, PRO, ALA skipped
        assert results[ResidueKey("A", 1, "")] == (None, None)
        assert results[ResidueKey("A", 2, "")] == (None, None)
        assert results[ResidueKey("A", 3, "")] == (None, None)
        # LEU scanned
        assert results[ResidueKey("A", 4, "")][0] is not None

        # _compute_rosetta_dG should only be called once (for LEU)
        assert mock_dG.call_count == 1


# ------------------------------------------------------------------
# Per-position dG (residue removal, mocked)
# ------------------------------------------------------------------


class TestPerPositionDG:
    """Test the residue-removal dG computation."""

    @patch("boundry.interface_position_energetics._compute_rosetta_dG")
    def test_returns_dG_without_i(self, mock_dG):
        """compute_per_position_dG returns dG_without_i directly."""
        # dG_without_i for the removed-residue complex
        mock_dG.return_value = -6.0

        relaxer = MagicMock()
        ir = _make_ir(residue_name="LEU")

        results = compute_per_position_dG(
            TWO_CHAIN_PDB,
            [ir],
            chain_pairs=[("A", "B")],
            relaxer=relaxer,
            dG_wt=-10.0,
            position_relax="none",
        )

        key = ResidueKey("A", 1, "")
        # Returns dG_without_i directly (not dG_i = dG_total - dG_without_i)
        assert results[key] == pytest.approx(-6.0)

    @patch("boundry.interface_position_energetics._compute_rosetta_dG")
    def test_handles_failure(self, mock_dG):
        """Failed energy evaluation should yield None."""
        mock_dG.side_effect = RuntimeError("OpenMM crash")
        relaxer = MagicMock()
        ir = _make_ir()

        results = compute_per_position_dG(
            TWO_CHAIN_PDB,
            [ir],
            chain_pairs=[("A", "B")],
            relaxer=relaxer,
            dG_wt=-10.0,
            position_relax="none",
        )

        key = ResidueKey("A", 1, "")
        assert results[key] is None


# ------------------------------------------------------------------
# Full pipeline (mocked)
# ------------------------------------------------------------------


class TestComputePositionEnergetics:
    """Test the orchestration function."""

    @patch("boundry.interface_position_energetics._compute_rosetta_dG")
    def test_two_sided_requirement(self, mock_dG):
        """Raises ValueError if chain groups != 2."""
        relaxer = MagicMock()
        # 3 individual chains -> 3 groups -> error
        with pytest.raises(ValueError, match="two-sided"):
            compute_position_energetics(
                TWO_CHAIN_PDB,
                [_make_ir()],
                # pairs that create >2 groups (A-B, B-C -> sides overlap)
                chain_pairs=[("A", "B"), ("B", "C")],
                relaxer=relaxer,
                run_alanine_scan=True,
            )

    @patch("boundry.interface_position_energetics._compute_rosetta_dG")
    def test_alanine_scan_only(self, mock_dG):
        """Running with alanine_scan=True, per_position=False."""
        mock_dG.side_effect = [-10.0, -5.0]  # dG_wt, dG_ala
        relaxer = MagicMock()
        ir = _make_ir(residue_name="LEU")

        energetics = compute_position_energetics(
            TWO_CHAIN_PDB,
            [ir],
            chain_pairs=[("A", "B")],
            relaxer=relaxer,
            run_alanine_scan=True,
            run_per_position=False,
            position_relax="none",
        )

        assert isinstance(energetics, PositionEnergeticsResult)
        assert energetics.per_position is None
        assert energetics.alanine_scan is not None
        assert energetics.alanine_scan.dG_wt == -10.0
        assert len(energetics.alanine_scan.rows) == 1
        row = energetics.alanine_scan.rows[0]
        assert row.dG_wt == -10.0
        assert row.dG == -5.0
        assert row.ddG == pytest.approx(5.0)

    @patch("boundry.interface_position_energetics._compute_rosetta_dG")
    def test_per_position_only(self, mock_dG):
        """Running with per_position=True, alanine_scan=False."""
        mock_dG.side_effect = [-10.0, -6.0]  # dG_wt, dG_without_i
        relaxer = MagicMock()
        ir = _make_ir(residue_name="LEU")

        energetics = compute_position_energetics(
            TWO_CHAIN_PDB,
            [ir],
            chain_pairs=[("A", "B")],
            relaxer=relaxer,
            run_alanine_scan=False,
            run_per_position=True,
            position_relax="none",
        )

        assert energetics.alanine_scan is None
        assert energetics.per_position is not None
        assert energetics.per_position.dG_wt == -10.0
        row = energetics.per_position.rows[0]
        assert row.dG == pytest.approx(-6.0)  # dG_without_i
        # ddG = -6.0 - (-10.0) = 4.0 (positive = hotspot)
        assert row.ddG == pytest.approx(4.0)

    @patch("boundry.interface_position_energetics._compute_rosetta_dG")
    def test_skipped_residues_marked(self, mock_dG):
        """GLY/PRO/ALA rows should have scan_skipped=True."""
        mock_dG.return_value = -10.0  # dG_wt
        relaxer = MagicMock()
        ir = _make_ir(residue_name="GLY", residue_number=1)

        energetics = compute_position_energetics(
            TWO_CHAIN_PDB,
            [ir],
            chain_pairs=[("A", "B")],
            relaxer=relaxer,
            run_alanine_scan=True,
            position_relax="none",
        )

        row = energetics.alanine_scan.rows[0]
        assert row.scan_skipped is True
        assert row.dG is None
        assert row.ddG is None

    @patch("boundry.interface_position_energetics._compute_rosetta_dG")
    def test_sasa_delta_attached(self, mock_dG):
        """SASA delta values are attached to rows."""
        mock_dG.return_value = -10.0
        relaxer = MagicMock()
        ir = _make_ir(residue_name="LEU", residue_number=1)

        energetics = compute_position_energetics(
            TWO_CHAIN_PDB,
            [ir],
            chain_pairs=[("A", "B")],
            relaxer=relaxer,
            run_alanine_scan=True,
            run_per_position=False,
            position_relax="none",
            sasa_delta={"A1": 42.5},
        )

        row = energetics.alanine_scan.rows[0]
        assert row.delta_sasa == 42.5

    @patch("boundry.interface_position_energetics._compute_rosetta_dG")
    def test_both_analyses(self, mock_dG):
        """Both per_position and alanine_scan produce results."""
        # dG_wt, dG_ala (alanine scan), dG_without_i (per-position)
        mock_dG.side_effect = [-10.0, -5.0, -6.0]
        relaxer = MagicMock()
        ir = _make_ir(residue_name="LEU")

        energetics = compute_position_energetics(
            TWO_CHAIN_PDB,
            [ir],
            chain_pairs=[("A", "B")],
            relaxer=relaxer,
            run_alanine_scan=True,
            run_per_position=True,
            position_relax="none",
        )

        assert energetics.per_position is not None
        assert energetics.alanine_scan is not None

        pp_row = energetics.per_position.rows[0]
        assert pp_row.dG == pytest.approx(-6.0)
        assert pp_row.ddG == pytest.approx(4.0)

        ala_row = energetics.alanine_scan.rows[0]
        assert ala_row.dG == pytest.approx(-5.0)
        assert ala_row.ddG == pytest.approx(5.0)


# ------------------------------------------------------------------
# CSV output
# ------------------------------------------------------------------


class TestWritePositionCSV:
    def test_writes_csv(self, tmp_path):
        """CSV file has header, metadata comments, and correct columns."""
        row = PositionRow(
            chain_id="A",
            residue_number=10,
            insertion_code="",
            wt_resname="LEU",
            partner_chain="B",
            min_distance=3.5,
            num_contacts=5,
            delta_sasa=25.0,
            dG_wt=-10.0,
            dG=-5.0,
            ddG=5.0,
        )
        result = PositionResult(
            rows=[row],
            dG_wt=-10.0,
            distance_cutoff=8.0,
            chain_pairs=[("A", "B")],
            position_relax="both",
        )

        csv_path = tmp_path / "test.csv"
        write_position_csv(result, csv_path)

        content = csv_path.read_text()
        # Metadata comments
        assert "# distance_cutoff=8.0" in content
        assert "# chain_pairs=A:B" in content
        assert "# position_relax=both" in content
        assert "# dG_wt=-10.0000" in content

        # Parse data rows (skip comments)
        data_lines = [
            l for l in content.splitlines() if not l.startswith("#")
        ]
        reader = csv.DictReader(io.StringIO("\n".join(data_lines)))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["chain_id"] == "A"
        assert rows[0]["residue_number"] == "10"
        assert rows[0]["wt_resname"] == "LEU"
        assert float(rows[0]["dG_wt"]) == pytest.approx(-10.0)
        assert float(rows[0]["dG"]) == pytest.approx(-5.0)
        assert float(rows[0]["ddG"]) == pytest.approx(5.0)

    def test_none_values_empty(self, tmp_path):
        """None values are written as empty strings."""
        row = PositionRow(
            chain_id="A",
            residue_number=1,
            insertion_code="",
            wt_resname="GLY",
            partner_chain="B",
            min_distance=4.0,
            num_contacts=3,
            dG_wt=-10.0,
            scan_skipped=True,
            skip_reason="Skipped: GLY",
        )
        result = PositionResult(
            rows=[row],
            dG_wt=-10.0,
            chain_pairs=[("A", "B")],
        )

        csv_path = tmp_path / "test.csv"
        write_position_csv(result, csv_path)

        content = csv_path.read_text()
        data_lines = [
            l for l in content.splitlines() if not l.startswith("#")
        ]
        reader = csv.DictReader(io.StringIO("\n".join(data_lines)))
        rows = list(reader)
        assert rows[0]["dG"] == ""
        assert rows[0]["ddG"] == ""


# ------------------------------------------------------------------
# Position table formatting
# ------------------------------------------------------------------


class TestFormatPositionTable:
    def test_top_n(self):
        rows = [
            PositionRow(
                chain_id="A",
                residue_number=i,
                insertion_code="",
                wt_resname="LEU",
                partner_chain="B",
                min_distance=3.0,
                num_contacts=5,
                dG_wt=-10.0,
                dG=-10.0 + i,
                ddG=float(i),
            )
            for i in range(1, 6)
        ]
        result = PositionResult(rows=rows, dG_wt=-10.0)

        table = format_position_table(result, top_n=3)
        assert "Top 3" in table
        # Highest ddG first
        lines = table.splitlines()
        data_lines = [l for l in lines if "LEU" in l]
        assert len(data_lines) == 3

    def test_empty_when_no_ddg(self):
        row = PositionRow(
            chain_id="A",
            residue_number=1,
            insertion_code="",
            wt_resname="GLY",
            partner_chain="B",
            min_distance=3.0,
            num_contacts=5,
            dG_wt=-10.0,
            scan_skipped=True,
        )
        result = PositionResult(rows=[row], dG_wt=-10.0)
        assert format_position_table(result) == ""

    def test_deprecated_alias(self):
        """format_hotspot_table still works as deprecated alias."""
        rows = [
            PositionRow(
                chain_id="A",
                residue_number=1,
                insertion_code="",
                wt_resname="LEU",
                partner_chain="B",
                min_distance=3.0,
                num_contacts=5,
                dG_wt=-10.0,
                dG=-5.0,
                ddG=5.0,
            )
        ]
        result = PositionResult(rows=rows, dG_wt=-10.0)
        table = format_hotspot_table(result)
        assert "AlaScan hotspots" in table


# ------------------------------------------------------------------
# CLI integration (help text only)
# ------------------------------------------------------------------


class TestCLINewOptions:
    """Verify new CLI flags appear in analyze-interface --help."""

    def test_per_position_option(self):
        from typer.testing import CliRunner

        from boundry.cli import app

        result = CliRunner().invoke(app, ["analyze-interface", "--help"])
        assert "--per-position" in result.output

    def test_alanine_scan_option(self):
        from typer.testing import CliRunner

        from boundry.cli import app

        result = CliRunner().invoke(app, ["analyze-interface", "--help"])
        assert "--alanine-scan" in result.output

    def test_scan_chains_option(self):
        from typer.testing import CliRunner

        from boundry.cli import app

        result = CliRunner().invoke(app, ["analyze-interface", "--help"])
        assert "--scan-chains" in result.output

    def test_position_relax_option(self):
        from typer.testing import CliRunner

        from boundry.cli import app

        result = CliRunner().invoke(app, ["analyze-interface", "--help"])
        assert "--position-relax" in result.output

    def test_per_position_csv_option(self):
        from typer.testing import CliRunner

        from boundry.cli import app

        result = CliRunner().invoke(app, ["analyze-interface", "--help"])
        assert "--per-position-csv" in result.output

    def test_alanine_scan_csv_option(self):
        from typer.testing import CliRunner

        from boundry.cli import app

        result = CliRunner().invoke(app, ["analyze-interface", "--help"])
        assert "--alanine-scan-csv" in result.output

    def test_max_scan_sites_option(self):
        from typer.testing import CliRunner

        from boundry.cli import app

        result = CliRunner().invoke(app, ["analyze-interface", "--help"])
        assert "--max-scan-sites" in result.output

    def test_sign_convention_in_help(self):
        """Help text mentions sign convention (negative = favorable)."""
        from typer.testing import CliRunner

        from boundry.cli import app

        result = CliRunner().invoke(app, ["analyze-interface", "--help"])
        assert "negative" in result.output.lower()

    def test_workers_option(self):
        """--workers / -j option appears in help."""
        from typer.testing import CliRunner

        from boundry.cli import app

        result = CliRunner().invoke(app, ["analyze-interface", "--help"])
        assert "--workers" in result.output


# ------------------------------------------------------------------
# Parallel dispatch tests
# ------------------------------------------------------------------


class TestParallelDispatch:
    """Test parallel scan dispatch produces same results as sequential."""

    @patch("boundry.interface_position_energetics._compute_rosetta_dG")
    def test_workers_1_uses_sequential(self, mock_dG):
        """workers=1 should use the sequential path (no pool)."""
        mock_dG.side_effect = [-10.0, -5.0]  # dG_wt, dG_ala
        relaxer = MagicMock()
        ir = _make_ir(residue_name="LEU")

        energetics = compute_position_energetics(
            TWO_CHAIN_PDB,
            [ir],
            chain_pairs=[("A", "B")],
            relaxer=relaxer,
            run_alanine_scan=True,
            position_relax="none",
            workers=1,
        )

        assert energetics.alanine_scan is not None
        row = energetics.alanine_scan.rows[0]
        assert row.dG == -5.0
        assert row.ddG == pytest.approx(5.0)

    @patch(
        "boundry.interface_position_energetics."
        "_run_scans_parallel"
    )
    @patch("boundry.interface_position_energetics._compute_rosetta_dG")
    def test_workers_gt1_calls_parallel(
        self, mock_dG, mock_parallel
    ):
        """workers > 1 should call _run_scans_parallel."""
        mock_dG.return_value = -10.0  # dG_wt
        mock_parallel.return_value = (
            {ResidueKey("A", 1, ""): (-5.0, 5.0)},
            {},
        )
        relaxer = MagicMock()
        ir = _make_ir(residue_name="LEU")

        energetics = compute_position_energetics(
            TWO_CHAIN_PDB,
            [ir],
            chain_pairs=[("A", "B")],
            relaxer=relaxer,
            run_alanine_scan=True,
            position_relax="none",
            workers=4,
        )

        mock_parallel.assert_called_once()
        assert energetics.alanine_scan is not None

    @patch.dict(
        "os.environ", {"BOUNDRY_IN_WORKER_PROCESS": "1"}
    )
    @patch("boundry.interface_position_energetics._compute_rosetta_dG")
    def test_nested_guard_forces_sequential(self, mock_dG):
        """workers > 1 inside a worker process falls back to sequential."""
        mock_dG.side_effect = [-10.0, -5.0]  # dG_wt, dG_ala
        relaxer = MagicMock()
        ir = _make_ir(residue_name="LEU")

        energetics = compute_position_energetics(
            TWO_CHAIN_PDB,
            [ir],
            chain_pairs=[("A", "B")],
            relaxer=relaxer,
            run_alanine_scan=True,
            position_relax="none",
            workers=4,  # should be forced to 1
        )

        # Should still produce correct results via sequential path
        assert energetics.alanine_scan is not None
        row = energetics.alanine_scan.rows[0]
        assert row.dG == -5.0
        assert row.ddG == pytest.approx(5.0)


class TestScanTaskResult:
    """Test ScanTask and ScanResult dataclasses."""

    def test_scan_task_frozen(self):
        from boundry._parallel import ScanTask

        task = ScanTask(
            scan_type="alanine_scan",
            pdb_string="PDB",
            chain_id="A",
            residue_number=1,
            insertion_code="",
            residue_name="LEU",
            chain_pairs=[("A", "B")],
            distance_cutoff=8.0,
            relax_separated=False,
            position_relax="none",
            dG_wt=-10.0,
            quiet=False,
        )
        with pytest.raises(AttributeError):
            task.scan_type = "per_position"

    def test_scan_result_fields(self):
        from boundry._parallel import ScanResult

        result = ScanResult(
            scan_type="alanine_scan",
            chain_id="A",
            residue_number=1,
            insertion_code="",
            residue_name="LEU",
            dG=-5.0,
            ddG=5.0,
        )
        assert result.dG == -5.0
        assert result.ddG == 5.0
        assert result.error is None
