"""Tests for binding energy calculation."""

import pytest

from boundry.binding_energy import (
    BindingEnergyResult,
    _get_interface_chain_groups,
    extract_chain,
)

# Two-chain peptide for testing chain extraction
TWO_CHAIN_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.986  -0.760  -1.216  1.00  0.00           C
TER
ATOM      6  N   ALA B   1       2.000   0.000   3.500  1.00  0.00           N
ATOM      7  CA  ALA B   1       3.458   0.000   3.500  1.00  0.00           C
ATOM      8  C   ALA B   1       4.009   1.420   3.500  1.00  0.00           C
ATOM      9  O   ALA B   1       3.246   2.390   3.500  1.00  0.00           O
ATOM     10  CB  ALA B   1       3.986  -0.760   2.284  1.00  0.00           C
TER
END
"""


class TestExtractChain:
    def test_extract_single_chain(self):
        """Test extracting a single chain from multi-chain PDB."""
        result = extract_chain(TWO_CHAIN_PDB, ["A"])

        assert "ALA A" in result
        assert "ALA B" not in result

    def test_extract_other_chain(self):
        """Test extracting chain B."""
        result = extract_chain(TWO_CHAIN_PDB, ["B"])

        assert "ALA B" in result
        assert "ALA A" not in result

    def test_extract_multiple_chains(self):
        """Test extracting multiple chains."""
        result = extract_chain(TWO_CHAIN_PDB, ["A", "B"])

        assert "ALA A" in result
        assert "ALA B" in result

    def test_extract_nonexistent_chain(self):
        """Test extracting a chain that doesn't exist returns empty PDB."""
        result = extract_chain(TWO_CHAIN_PDB, ["X"])

        # Should still return valid PDB, just no ATOM records
        assert "ATOM" not in result or "ALA" not in result


class TestGetInterfaceChainGroups:
    def test_single_pair(self):
        """Test grouping with a single chain pair."""
        groups = _get_interface_chain_groups([("A", "B")])
        assert len(groups) == 2
        assert ["A"] in groups
        assert ["B"] in groups

    def test_antibody_antigen_pairs(self):
        """Test typical antibody-antigen chain grouping."""
        groups = _get_interface_chain_groups([("H", "A"), ("L", "A")])
        assert len(groups) == 2
        # H and L should be grouped together, A separate
        assert sorted(groups[0]) == ["H", "L"] or sorted(groups[1]) == [
            "H",
            "L",
        ]

    def test_overlapping_sides_fallback(self):
        """Test that overlapping chain sides fall back to individual chains."""
        # A appears on both sides
        groups = _get_interface_chain_groups([("A", "B"), ("B", "C")])
        # Should fall back to individual chains since B is on both sides
        assert len(groups) == 3


class TestBindingEnergyResult:
    def test_default_values(self):
        """Test default BindingEnergyResult values."""
        result = BindingEnergyResult()
        assert result.complex_energy == 0.0
        assert result.binding_energy == 0.0
        assert result.separated_energies == {}
        assert result.interface_residues == []
        assert result.interface_energy is None
