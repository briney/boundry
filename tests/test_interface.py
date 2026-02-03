"""Tests for interface residue identification."""

from boundry.interface import (
    InterfaceInfo,
    identify_interface_residues,
)

# Two-chain peptide dimer (chains A and B, close enough to form interface)
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
ATOM     11  N   ALA A   3       6.063   3.970   0.000  1.00  0.00           N
ATOM     12  CA  ALA A   3       7.510   4.096   0.000  1.00  0.00           C
ATOM     13  C   ALA A   3       8.061   5.516   0.000  1.00  0.00           C
ATOM     14  O   ALA A   3       7.298   6.486   0.000  1.00  0.00           O
ATOM     15  CB  ALA A   3       8.038   3.336  -1.216  1.00  0.00           C
TER
ATOM     16  N   ALA B   1       2.000   0.000   3.500  1.00  0.00           N
ATOM     17  CA  ALA B   1       3.458   0.000   3.500  1.00  0.00           C
ATOM     18  C   ALA B   1       4.009   1.420   3.500  1.00  0.00           C
ATOM     19  O   ALA B   1       3.246   2.390   3.500  1.00  0.00           O
ATOM     20  CB  ALA B   1       3.986  -0.760   2.284  1.00  0.00           C
ATOM     21  N   ALA B   2       5.326   1.540   3.500  1.00  0.00           N
ATOM     22  CA  ALA B   2       5.941   2.861   3.500  1.00  0.00           C
ATOM     23  C   ALA B   2       7.459   2.789   3.500  1.00  0.00           C
ATOM     24  O   ALA B   2       8.065   1.719   3.500  1.00  0.00           O
ATOM     25  CB  ALA B   2       5.473   3.699   4.686  1.00  0.00           C
ATOM     26  N   ALA B   3       8.063   3.970   3.500  1.00  0.00           N
ATOM     27  CA  ALA B   3       9.510   4.096   3.500  1.00  0.00           C
ATOM     28  C   ALA B   3      10.061   5.516   3.500  1.00  0.00           C
ATOM     29  O   ALA B   3       9.298   6.486   3.500  1.00  0.00           O
ATOM     30  CB  ALA B   3      10.038   3.336   2.284  1.00  0.00           C
TER
END
"""

# Two chains far apart (no interface)
NO_INTERFACE_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.986  -0.760  -1.216  1.00  0.00           C
TER
ATOM      6  N   ALA B   1      50.000  50.000  50.000  1.00  0.00           N
ATOM      7  CA  ALA B   1      51.458  50.000  50.000  1.00  0.00           C
ATOM      8  C   ALA B   1      52.009  51.420  50.000  1.00  0.00           C
ATOM      9  O   ALA B   1      51.246  52.390  50.000  1.00  0.00           O
ATOM     10  CB  ALA B   1      51.986  49.240  48.784  1.00  0.00           C
TER
END
"""

# PDB with a HETATM ligand
LIGAND_PDB = """\
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
HETATM   11  C1  LIG C   1       1.500   0.500   1.750  1.00  0.00           C
HETATM   12  C2  LIG C   1       2.500   0.500   1.750  1.00  0.00           C
END
"""


class TestIdentifyInterfaceResidues:
    def test_basic_interface_detection(self):
        """Test that interface residues are identified between two chains."""
        result = identify_interface_residues(TWO_CHAIN_PDB, distance_cutoff=8.0)

        assert isinstance(result, InterfaceInfo)
        assert result.n_interface_residues > 0
        assert len(result.chain_pairs) > 0

        # Should find contacts between chains A and B
        chain_ids = {ir.chain_id for ir in result.interface_residues}
        assert "A" in chain_ids
        assert "B" in chain_ids

    def test_no_interface_when_far_apart(self):
        """Test that no interface is found when chains are far apart."""
        result = identify_interface_residues(
            NO_INTERFACE_PDB, distance_cutoff=8.0
        )

        assert result.n_interface_residues == 0
        assert len(result.chain_pairs) == 0

    def test_distance_cutoff_affects_results(self):
        """Test that distance cutoff affects number of interface residues."""
        result_tight = identify_interface_residues(
            TWO_CHAIN_PDB, distance_cutoff=4.0
        )
        result_loose = identify_interface_residues(
            TWO_CHAIN_PDB, distance_cutoff=10.0
        )

        # Loose cutoff should find at least as many residues
        assert (
            result_loose.n_interface_residues
            >= result_tight.n_interface_residues
        )

    def test_specific_chain_pairs(self):
        """Test specifying chain pairs to analyze."""
        result = identify_interface_residues(
            TWO_CHAIN_PDB,
            distance_cutoff=8.0,
            chain_pairs=[("A", "B")],
        )

        assert result.n_interface_residues > 0
        assert ("A", "B") in result.chain_pairs

    def test_invalid_chain_pairs_skipped(self):
        """Test that invalid chain pairs are skipped with warning."""
        result = identify_interface_residues(
            TWO_CHAIN_PDB,
            distance_cutoff=8.0,
            chain_pairs=[("X", "Y")],
        )

        assert result.n_interface_residues == 0

    def test_single_chain_returns_empty(self):
        """Test that single-chain structure returns empty result."""
        single_chain_pdb = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
END
"""
        result = identify_interface_residues(single_chain_pdb)

        assert result.n_interface_residues == 0
        assert len(result.chain_pairs) == 0

    def test_exclude_ligands(self):
        """Test that HETATM ligands are excluded by default."""
        result = identify_interface_residues(
            LIGAND_PDB, distance_cutoff=8.0, exclude_ligands=True
        )

        # Should only find protein chains A and B, not ligand chain C
        chain_ids = {ir.chain_id for ir in result.interface_residues}
        assert "C" not in chain_ids

    def test_interface_residue_attributes(self):
        """Test that InterfaceResidue has expected attributes."""
        result = identify_interface_residues(TWO_CHAIN_PDB, distance_cutoff=8.0)

        if result.interface_residues:
            ir = result.interface_residues[0]
            assert isinstance(ir.chain_id, str)
            assert isinstance(ir.residue_number, int)
            assert isinstance(ir.residue_name, str)
            assert isinstance(ir.insertion_code, str)
            assert isinstance(ir.partner_chain, str)
            assert isinstance(ir.min_distance, float)
            assert isinstance(ir.num_contacts, int)
            assert ir.min_distance > 0
            assert ir.num_contacts > 0

    def test_summary_property(self):
        """Test InterfaceInfo summary string."""
        result = identify_interface_residues(TWO_CHAIN_PDB, distance_cutoff=8.0)
        summary = result.summary
        assert isinstance(summary, str)
        assert "interface residues" in summary

    def test_empty_interface_info(self):
        """Test default InterfaceInfo properties."""
        info = InterfaceInfo()
        assert info.n_interface_residues == 0
        assert info.summary == "0 interface residues across "
