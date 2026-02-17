"""Tests for boundry.ddg module — mutation specification, parsing,
and validation utilities.
"""

import pytest

from boundry.ddg import (
    MutationSpec,
    _normalize_aa,
    parse_mutation_dict,
    parse_mutation_string,
    parse_mutations,
    validate_mutations,
)


# ------------------------------------------------------------------
# Test PDB strings
# ------------------------------------------------------------------

# Minimal two-chain PDB: LEU A:1, ALA A:2, VAL B:1
TWO_CHAIN_PDB = (
    "ATOM      1  N   LEU A   1       0.000   0.000   0.000"
    "  1.00  0.00           N\n"
    "ATOM      2  CA  LEU A   1       1.458   0.000   0.000"
    "  1.00  0.00           C\n"
    "ATOM      3  N   ALA A   2       3.000   0.000   0.000"
    "  1.00  0.00           N\n"
    "ATOM      4  CA  ALA A   2       4.458   0.000   0.000"
    "  1.00  0.00           C\n"
    "TER\n"
    "ATOM      5  N   VAL B   1       8.000   0.000   0.000"
    "  1.00  0.00           N\n"
    "ATOM      6  CA  VAL B   1       9.458   0.000   0.000"
    "  1.00  0.00           C\n"
    "TER\n"
    "END\n"
)

# PDB with insertion code: SER H:100a
ICODE_PDB = (
    "ATOM      1  N   SER H 100A      0.000   0.000   0.000"
    "  1.00  0.00           N\n"
    "ATOM      2  CA  SER H 100A      1.458   0.000   0.000"
    "  1.00  0.00           C\n"
    "END\n"
)


# ------------------------------------------------------------------
# MutationSpec
# ------------------------------------------------------------------


class TestMutationSpec:
    def test_basic_fields(self):
        spec = MutationSpec("A", 5, "LEU", "ALA")
        assert spec.chain_id == "A"
        assert spec.residue_number == 5
        assert spec.wild_type == "LEU"
        assert spec.mutant == "ALA"
        assert spec.insertion_code == ""

    def test_with_insertion_code(self):
        spec = MutationSpec("H", 100, "SER", "ALA", "A")
        assert spec.insertion_code == "A"

    def test_frozen(self):
        spec = MutationSpec("A", 5, "LEU", "ALA")
        with pytest.raises(AttributeError):
            spec.chain_id = "B"

    def test_str_format(self):
        spec = MutationSpec("A", 5, "LEU", "ALA")
        assert str(spec) == "A:L5A"

    def test_str_format_with_icode(self):
        spec = MutationSpec("H", 100, "SER", "ALA", "a")
        assert str(spec) == "H:S100aA"

    def test_equality(self):
        s1 = MutationSpec("A", 5, "LEU", "ALA")
        s2 = MutationSpec("A", 5, "LEU", "ALA")
        assert s1 == s2
        assert hash(s1) == hash(s2)

    def test_inequality(self):
        s1 = MutationSpec("A", 5, "LEU", "ALA")
        s2 = MutationSpec("A", 5, "LEU", "GLY")
        assert s1 != s2


# ------------------------------------------------------------------
# _normalize_aa
# ------------------------------------------------------------------


class TestNormalizeAA:
    def test_one_letter(self):
        assert _normalize_aa("L") == "LEU"
        assert _normalize_aa("A") == "ALA"
        assert _normalize_aa("G") == "GLY"

    def test_three_letter(self):
        assert _normalize_aa("LEU") == "LEU"
        assert _normalize_aa("ALA") == "ALA"

    def test_case_insensitive(self):
        assert _normalize_aa("l") == "LEU"
        assert _normalize_aa("leu") == "LEU"
        assert _normalize_aa("Ala") == "ALA"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unrecognized amino acid"):
            _normalize_aa("X")
        with pytest.raises(ValueError, match="Unrecognized amino acid"):
            _normalize_aa("FOO")


# ------------------------------------------------------------------
# parse_mutation_string
# ------------------------------------------------------------------


class TestParseMutationString:
    def test_single_mutation(self):
        specs = parse_mutation_string("A:L5A")
        assert len(specs) == 1
        assert specs[0] == MutationSpec("A", 5, "LEU", "ALA")

    def test_multiple_mutations(self):
        specs = parse_mutation_string("A:L5A,B:W10G")
        assert len(specs) == 2
        assert specs[0] == MutationSpec("A", 5, "LEU", "ALA")
        assert specs[1] == MutationSpec("B", 10, "TRP", "GLY")

    def test_with_insertion_code(self):
        specs = parse_mutation_string("H:S100aA")
        assert len(specs) == 1
        assert specs[0] == MutationSpec("H", 100, "SER", "ALA", "a")

    def test_whitespace_tolerance(self):
        specs = parse_mutation_string(" A:L5A , B:V10G ")
        assert len(specs) == 2

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="No mutations found"):
            parse_mutation_string("")

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Cannot parse mutation"):
            parse_mutation_string("bad_format")

    def test_missing_chain_raises(self):
        with pytest.raises(ValueError, match="Cannot parse mutation"):
            parse_mutation_string("L5A")

    def test_invalid_aa_code_raises(self):
        with pytest.raises(ValueError, match="Unrecognized amino acid"):
            parse_mutation_string("A:X5A")


# ------------------------------------------------------------------
# parse_mutation_dict
# ------------------------------------------------------------------


class TestParseMutationDict:
    def test_basic_dict(self):
        spec = parse_mutation_dict(
            {"chain": "A", "resnum": "5", "wt": "L", "mut": "A"}
        )
        assert spec == MutationSpec("A", 5, "LEU", "ALA")

    def test_three_letter_codes(self):
        spec = parse_mutation_dict(
            {"chain": "A", "resnum": "5", "wt": "LEU", "mut": "ALA"}
        )
        assert spec == MutationSpec("A", 5, "LEU", "ALA")

    def test_alternative_keys(self):
        spec = parse_mutation_dict(
            {
                "chain": "A",
                "residue_number": "5",
                "wild_type": "LEU",
                "mutant": "ALA",
                "insertion_code": "B",
            }
        )
        assert spec == MutationSpec("A", 5, "LEU", "ALA", "B")

    def test_case_insensitive_keys(self):
        spec = parse_mutation_dict(
            {"Chain": "A", "ResNum": "5", "WT": "L", "MUT": "A"}
        )
        assert spec == MutationSpec("A", 5, "LEU", "ALA")

    def test_missing_chain_raises(self):
        with pytest.raises(ValueError, match="Missing 'chain'"):
            parse_mutation_dict({"resnum": "5", "wt": "L", "mut": "A"})

    def test_missing_resnum_raises(self):
        with pytest.raises(ValueError, match="Missing 'resnum'"):
            parse_mutation_dict({"chain": "A", "wt": "L", "mut": "A"})

    def test_missing_wt_raises(self):
        with pytest.raises(ValueError, match="Missing 'wt'"):
            parse_mutation_dict({"chain": "A", "resnum": "5", "mut": "A"})

    def test_missing_mut_raises(self):
        with pytest.raises(ValueError, match="Missing 'mut'"):
            parse_mutation_dict({"chain": "A", "resnum": "5", "wt": "L"})


# ------------------------------------------------------------------
# parse_mutations (unified)
# ------------------------------------------------------------------


class TestParseMutations:
    def test_from_string(self):
        specs = parse_mutations(mutation_string="A:L5A")
        assert len(specs) == 1
        assert specs[0].wild_type == "LEU"

    def test_from_dicts(self):
        specs = parse_mutations(
            mutations=[{"chain": "A", "resnum": "5", "wt": "L", "mut": "A"}]
        )
        assert len(specs) == 1
        assert specs[0].wild_type == "LEU"

    def test_both_raises(self):
        with pytest.raises(ValueError, match="not both"):
            parse_mutations(
                mutations=[{"chain": "A", "resnum": "5", "wt": "L", "mut": "A"}],
                mutation_string="A:L5A",
            )

    def test_neither_raises(self):
        with pytest.raises(ValueError, match="Must provide"):
            parse_mutations()


# ------------------------------------------------------------------
# validate_mutations
# ------------------------------------------------------------------


class TestValidateMutations:
    def test_valid_mutation(self):
        spec = MutationSpec("A", 1, "LEU", "ALA")
        validate_mutations(TWO_CHAIN_PDB, [spec])  # should not raise

    def test_valid_chain_b(self):
        spec = MutationSpec("B", 1, "VAL", "ALA")
        validate_mutations(TWO_CHAIN_PDB, [spec])  # should not raise

    def test_wrong_wt_raises(self):
        spec = MutationSpec("A", 1, "ALA", "GLY")  # actual is LEU
        with pytest.raises(ValueError, match="expected ALA.*found LEU"):
            validate_mutations(TWO_CHAIN_PDB, [spec])

    def test_missing_position_raises(self):
        spec = MutationSpec("A", 99, "LEU", "ALA")
        with pytest.raises(ValueError, match="not found in structure"):
            validate_mutations(TWO_CHAIN_PDB, [spec])

    def test_missing_chain_raises(self):
        spec = MutationSpec("Z", 1, "LEU", "ALA")
        with pytest.raises(ValueError, match="not found in structure"):
            validate_mutations(TWO_CHAIN_PDB, [spec])

    def test_multiple_mutations_validated(self):
        specs = [
            MutationSpec("A", 1, "LEU", "ALA"),
            MutationSpec("A", 2, "ALA", "GLY"),
        ]
        validate_mutations(TWO_CHAIN_PDB, specs)  # should not raise

    def test_multiple_errors_reported(self):
        specs = [
            MutationSpec("A", 1, "ALA", "GLY"),  # wrong WT
            MutationSpec("A", 99, "LEU", "ALA"),  # missing
        ]
        with pytest.raises(ValueError) as exc_info:
            validate_mutations(TWO_CHAIN_PDB, specs)
        msg = str(exc_info.value)
        assert "expected ALA" in msg
        assert "not found" in msg

    def test_insertion_code_validation(self):
        spec = MutationSpec("H", 100, "SER", "ALA", "A")
        validate_mutations(ICODE_PDB, [spec])  # should not raise

    def test_wrong_insertion_code_raises(self):
        spec = MutationSpec("H", 100, "SER", "ALA", "B")  # wrong icode
        with pytest.raises(ValueError, match="not found in structure"):
            validate_mutations(ICODE_PDB, [spec])
