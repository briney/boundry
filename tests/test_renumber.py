"""Tests for boundry.renumber module."""

import pytest

from boundry.renumber import (
    RenumberMapping,
    has_insertion_codes,
    renumber_pdb,
    restore_numbering,
)


# ------------------------------------------------------------------
# Test PDB strings
# ------------------------------------------------------------------

# Simple PDB without insertion codes
NORMAL_PDB = (
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000"
    "  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1       1.458   0.000   0.000"
    "  1.00  0.00           C\n"
    "ATOM      3  N   ALA A   2       3.326   1.540   0.000"
    "  1.00  0.00           N\n"
    "ATOM      4  CA  ALA A   2       3.941   2.861   0.000"
    "  1.00  0.00           C\n"
    "END\n"
)

# PDB with Kabat-style insertion codes on chain H
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
    "ATOM      7  N   SER H 100B      7.000   5.000   0.000"
    "  1.00  0.00           N\n"
    "ATOM      8  CA  SER H 100B      8.000   6.000   0.000"
    "  1.00  0.00           C\n"
    "ATOM      9  N   TYR H 101       9.000   7.000   0.000"
    "  1.00  0.00           N\n"
    "ATOM     10  CA  TYR H 101      10.000   8.000   0.000"
    "  1.00  0.00           C\n"
    "END\n"
)


# ------------------------------------------------------------------
# has_insertion_codes
# ------------------------------------------------------------------


class TestHasInsertionCodes:
    """Tests for has_insertion_codes()."""

    def test_returns_false_for_normal_pdb(self):
        """No insertion codes in a standard PDB."""
        assert has_insertion_codes(NORMAL_PDB) is False

    def test_returns_true_for_icode_pdb(self):
        """Insertion codes present in Kabat-numbered PDB."""
        assert has_insertion_codes(ICODE_PDB) is True

    def test_returns_false_for_empty_string(self):
        """Empty string has no insertion codes."""
        assert has_insertion_codes("") is False

    def test_returns_false_for_non_atom_lines(self):
        """Non-ATOM/HETATM lines are ignored."""
        pdb = "REMARK   This has an A at col 27\nEND\n"
        assert has_insertion_codes(pdb) is False


# ------------------------------------------------------------------
# renumber_pdb
# ------------------------------------------------------------------


class TestRenumberPdb:
    """Tests for renumber_pdb()."""

    def test_renumber_simple(self):
        """Single chain with insertion codes gets sequential numbering."""
        renumbered, mapping = renumber_pdb(ICODE_PDB)

        # Verify no insertion codes remain
        assert has_insertion_codes(renumbered) is False

        # Verify sequential numbering: 5 residues -> 1,2,3,4,5
        assert mapping.forward[("H", 1)] == ("H", 99, " ")
        assert mapping.forward[("H", 2)] == ("H", 100, " ")
        assert mapping.forward[("H", 3)] == ("H", 100, "A")
        assert mapping.forward[("H", 4)] == ("H", 100, "B")
        assert mapping.forward[("H", 5)] == ("H", 101, " ")

    def test_renumber_multi_chain(self, kabat_numbered_pdb_string):
        """Multi-chain PDB: each chain renumbered independently."""
        renumbered, mapping = renumber_pdb(kabat_numbered_pdb_string)

        assert has_insertion_codes(renumbered) is False

        # Chain H: 98, 99, 100, 100A, 100B, 101 -> 1..6
        assert mapping.forward[("H", 1)] == ("H", 98, " ")
        assert mapping.forward[("H", 6)] == ("H", 101, " ")

        # Chain L: 1, 2 -> 1, 2
        assert mapping.forward[("L", 1)] == ("L", 1, " ")
        assert mapping.forward[("L", 2)] == ("L", 2, " ")

    def test_renumber_preserves_coordinates(self):
        """Coordinates are unchanged after renumbering."""
        renumbered, _ = renumber_pdb(ICODE_PDB)
        # Check that coordinate columns are preserved
        for orig_line, new_line in zip(
            ICODE_PDB.splitlines(), renumbered.splitlines()
        ):
            if orig_line.startswith("ATOM"):
                # Columns 30-54 are x, y, z coordinates
                assert orig_line[30:54] == new_line[30:54]

    def test_renumber_normal_pdb_is_idempotent(self):
        """Renumbering a PDB without insertion codes still works."""
        renumbered, mapping = renumber_pdb(NORMAL_PDB)
        # Mapping should exist but be trivial
        assert mapping.forward[("A", 1)] == ("A", 1, " ")
        assert mapping.forward[("A", 2)] == ("A", 2, " ")

    def test_renumber_hetatm_and_ter_records(self):
        """HETATM and TER records are also renumbered."""
        pdb = (
            "ATOM      1  N   ALA H 100       0.0   0.0   0.0"
            "  1.00  0.00           N\n"
            "HETATM    2  FE  HEM H 100A      1.0   1.0   1.0"
            "  1.00  0.00          FE\n"
            "TER       3      HEM H 100A\n"
            "END\n"
        )
        renumbered, mapping = renumber_pdb(pdb)
        assert has_insertion_codes(renumbered) is False

        # HETATM residue 100A should be renumbered to 2
        assert mapping.forward[("H", 1)] == ("H", 100, " ")
        assert mapping.forward[("H", 2)] == ("H", 100, "A")

        # Verify HETATM line was rewritten
        lines = renumbered.splitlines()
        hetatm_line = [l for l in lines if l.startswith("HETATM")][0]
        assert hetatm_line[22:27] == "   2 "

        # Verify TER line was rewritten (TER may be shorter than 27 chars)
        ter_line = [l for l in lines if l.startswith("TER")][0]
        assert ter_line[22:26] == "   2"


# ------------------------------------------------------------------
# restore_numbering
# ------------------------------------------------------------------


class TestRestoreNumbering:
    """Tests for restore_numbering()."""

    def test_roundtrip(self):
        """renumber then restore should produce the original."""
        renumbered, mapping = renumber_pdb(ICODE_PDB)
        restored = restore_numbering(renumbered, mapping)

        # Compare line by line (ignoring trailing whitespace differences)
        orig_lines = [l.rstrip() for l in ICODE_PDB.splitlines() if l.strip()]
        rest_lines = [l.rstrip() for l in restored.splitlines() if l.strip()]

        assert len(orig_lines) == len(rest_lines)
        for orig, rest in zip(orig_lines, rest_lines):
            if orig.startswith(("ATOM", "HETATM", "TER")):
                # Check residue number + insertion code columns
                assert orig[22:27] == rest[22:27]

    def test_roundtrip_multi_chain(self, kabat_numbered_pdb_string):
        """Roundtrip with multi-chain Kabat-numbered PDB."""
        renumbered, mapping = renumber_pdb(kabat_numbered_pdb_string)
        restored = restore_numbering(renumbered, mapping)

        orig_lines = [
            l.rstrip()
            for l in kabat_numbered_pdb_string.splitlines()
            if l.strip()
        ]
        rest_lines = [l.rstrip() for l in restored.splitlines() if l.strip()]

        assert len(orig_lines) == len(rest_lines)
        for orig, rest in zip(orig_lines, rest_lines):
            if orig.startswith(("ATOM", "HETATM", "TER")):
                assert orig[22:27] == rest[22:27]


# ------------------------------------------------------------------
# RenumberMapping
# ------------------------------------------------------------------


class TestRenumberMapping:
    """Tests for the RenumberMapping dataclass."""

    def test_default_empty(self):
        """Default mapping has empty dicts."""
        m = RenumberMapping()
        assert m.forward == {}
        assert m.reverse == {}

    def test_bidirectional_consistency(self):
        """Forward and reverse mappings should be consistent."""
        _, mapping = renumber_pdb(ICODE_PDB)

        for (chain, seq_num), (c, orig_num, icode) in mapping.forward.items():
            reverse_key = (c, orig_num, icode)
            assert reverse_key in mapping.reverse
            assert mapping.reverse[reverse_key] == (chain, seq_num)


# ------------------------------------------------------------------
# renumber operation
# ------------------------------------------------------------------


class TestRenumberOperation:
    """Tests for the renumber() standalone operation."""

    def test_returns_structure(self, kabat_numbered_pdb_string):
        """renumber() returns a Structure with mapping in metadata."""
        from boundry.operations import Structure, renumber

        result = renumber(kabat_numbered_pdb_string)

        assert isinstance(result, Structure)
        assert result.metadata["operation"] == "renumber"
        assert "renumber_mapping" in result.metadata
        assert isinstance(
            result.metadata["renumber_mapping"], RenumberMapping
        )

    def test_removes_insertion_codes(self, kabat_numbered_pdb_string):
        """renumber() output has no insertion codes."""
        from boundry.operations import renumber

        result = renumber(kabat_numbered_pdb_string)
        assert has_insertion_codes(result.pdb_string) is False

    def test_accepts_structure_input(self, kabat_numbered_pdb_string):
        """renumber() accepts a Structure object."""
        from boundry.operations import Structure, renumber

        s = Structure(
            pdb_string=kabat_numbered_pdb_string,
            source_path="/test.pdb",
        )
        result = renumber(s)
        assert result.source_path == "/test.pdb"
        assert has_insertion_codes(result.pdb_string) is False
