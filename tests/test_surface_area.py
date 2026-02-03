"""Tests for surface area and shape complementarity calculations."""

import pytest

from boundry.interface import InterfaceResidue, identify_interface_residues
from boundry.surface_area import (
    ShapeComplementarityResult,
    SurfaceAreaResult,
    calculate_shape_complementarity,
    calculate_surface_area,
)

# Two-chain peptide dimer with interface
TWO_CHAIN_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.986  -0.760  -1.216  1.00  0.00           C
ATOM      6  N   ALA A   2       3.326   1.540   0.000  1.00  0.00           N
ATOM      7  CA  ALA A   2       3.941   2.861   0.000  1.00  0.00           C
ATOM      8  C   ALA A   2       5.459   2.789   0.000  1.00  0.00           C
ATOM      9  O   ALA A   2       6.065   1.719   0.000  1.00  0.00           O
ATOM     10  CB  ALA A   2       3.473   3.699   1.186  1.00  0.00           C
TER
ATOM     11  N   ALA B   1       2.000   0.000   3.500  1.00  0.00           N
ATOM     12  CA  ALA B   1       3.458   0.000   3.500  1.00  0.00           C
ATOM     13  C   ALA B   1       4.009   1.420   3.500  1.00  0.00           C
ATOM     14  O   ALA B   1       3.246   2.390   3.500  1.00  0.00           O
ATOM     15  CB  ALA B   1       3.986  -0.760   2.284  1.00  0.00           C
ATOM     16  N   ALA B   2       5.326   1.540   3.500  1.00  0.00           N
ATOM     17  CA  ALA B   2       5.941   2.861   3.500  1.00  0.00           C
ATOM     18  C   ALA B   2       7.459   2.789   3.500  1.00  0.00           C
ATOM     19  O   ALA B   2       8.065   1.719   3.500  1.00  0.00           O
ATOM     20  CB  ALA B   2       5.473   3.699   4.686  1.00  0.00           C
TER
END
"""


@pytest.fixture
def interface_residues():
    """Get interface residues from the two-chain PDB."""
    info = identify_interface_residues(TWO_CHAIN_PDB, distance_cutoff=8.0)
    return info.interface_residues


class TestCalculateSurfaceArea:
    def test_basic_sasa_calculation(self, interface_residues):
        """Test that SASA calculation returns valid results."""
        result = calculate_surface_area(
            TWO_CHAIN_PDB,
            interface_residues,
        )

        assert isinstance(result, SurfaceAreaResult)
        assert result.complex_sasa > 0
        assert len(result.chain_sasa) == 2  # Two chains
        assert "A" in result.chain_sasa
        assert "B" in result.chain_sasa

    def test_buried_sasa_non_negative(self, interface_residues):
        """Test that buried SASA is non-negative."""
        result = calculate_surface_area(
            TWO_CHAIN_PDB,
            interface_residues,
        )

        # Buried SASA should be >= 0
        # (sum of individual chain SASA >= complex SASA)
        assert result.buried_sasa >= 0

    def test_chain_sasa_positive(self, interface_residues):
        """Test that individual chain SASAs are positive."""
        result = calculate_surface_area(
            TWO_CHAIN_PDB,
            interface_residues,
        )

        for chain_id, sasa in result.chain_sasa.items():
            assert sasa > 0, f"Chain {chain_id} SASA should be positive"

    def test_custom_probe_radius(self, interface_residues):
        """Test SASA with different probe radii."""
        result_small = calculate_surface_area(
            TWO_CHAIN_PDB,
            interface_residues,
            probe_radius=1.0,
        )
        result_large = calculate_surface_area(
            TWO_CHAIN_PDB,
            interface_residues,
            probe_radius=2.0,
        )

        # Larger probe = generally different SASA
        # (not necessarily larger or smaller, depends on geometry)
        assert result_small.complex_sasa != result_large.complex_sasa


class TestCalculateShapeComplementarity:
    def test_basic_shape_complementarity(self, interface_residues):
        """Test simplified shape complementarity calculation."""
        info = identify_interface_residues(TWO_CHAIN_PDB, distance_cutoff=8.0)

        result = calculate_shape_complementarity(
            TWO_CHAIN_PDB,
            info.chain_pairs,
            info.interface_residues,
        )

        assert isinstance(result, ShapeComplementarityResult)
        assert 0.0 <= result.sc_score <= 1.0
        assert result.interface_area >= 0
        assert result.interface_gap_volume >= 0

    def test_empty_interface(self):
        """Test shape complementarity with no interface residues."""
        result = calculate_shape_complementarity(
            TWO_CHAIN_PDB,
            [],
            [],
        )

        assert result.sc_score == 0.0

    def test_default_result_values(self):
        """Test default ShapeComplementarityResult."""
        result = ShapeComplementarityResult()
        assert result.sc_score == 0.0
        assert result.interface_area == 0.0
        assert result.interface_gap_volume == 0.0
