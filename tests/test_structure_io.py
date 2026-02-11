"""Tests for boundry.structure_io module."""

import io

import pytest
from Bio.PDB import MMCIFIO
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure as BioStructure

from boundry.structure_io import (
    ChainIdMapping,
    StructureFormat,
    _needs_chain_remapping,
    _remap_chain_ids,
    convert_cif_to_pdb,
    convert_pdb_to_cif,
    convert_to_format,
    detect_format,
    ensure_pdb_format,
    get_output_format,
    read_structure,
    restore_cif_chain_ids,
    write_structure,
)


def _build_multi_chain_cif() -> str:
    """Build a synthetic CIF string with chains A, B, AA, AB."""
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Atom import Atom

    structure = BioStructure("test")
    model = Model(0)
    structure.add(model)

    for cid in ["A", "B", "AA", "AB"]:
        chain = Chain(cid)
        model.add(chain)
        res = Residue((" ", 1, " "), "ALA", " ")
        chain.add(res)
        atom = Atom(
            "CA",
            [0.0, 0.0, 0.0],
            1.0,
            1.0,
            " ",
            "CA",
            1,
            element="C",
        )
        res.add(atom)

    cif_io = MMCIFIO()
    cif_io.set_structure(structure)
    output = io.StringIO()
    cif_io.save(output)
    return output.getvalue()


class TestDetectFormat:
    """Tests for detect_format function."""

    def test_detect_pdb_format(self, tmp_path):
        """Test detection of PDB format."""
        path = tmp_path / "test.pdb"
        assert detect_format(path) == StructureFormat.PDB

    def test_detect_cif_format(self, tmp_path):
        """Test detection of CIF format."""
        path = tmp_path / "test.cif"
        assert detect_format(path) == StructureFormat.CIF

    def test_detect_mmcif_format(self, tmp_path):
        """Test detection of mmCIF format."""
        path = tmp_path / "test.mmcif"
        assert detect_format(path) == StructureFormat.CIF

    def test_detect_uppercase_extension(self, tmp_path):
        """Test detection handles uppercase extensions."""
        path = tmp_path / "test.PDB"
        assert detect_format(path) == StructureFormat.PDB

        path = tmp_path / "test.CIF"
        assert detect_format(path) == StructureFormat.CIF

    def test_unknown_format_raises(self, tmp_path):
        """Test that unknown extension raises ValueError."""
        path = tmp_path / "test.xyz"
        with pytest.raises(ValueError, match="Unknown structure format"):
            detect_format(path)

    def test_no_extension_raises(self, tmp_path):
        """Test that missing extension raises ValueError."""
        path = tmp_path / "testfile"
        with pytest.raises(ValueError, match="Unknown structure format"):
            detect_format(path)


class TestReadStructure:
    """Tests for read_structure function."""

    def test_read_pdb_file(self, tmp_path, small_peptide_pdb_string):
        """Test reading PDB file."""
        path = tmp_path / "test.pdb"
        path.write_text(small_peptide_pdb_string)

        content = read_structure(path)
        assert content == small_peptide_pdb_string

    def test_read_preserves_newlines(self, tmp_path):
        """Test that newlines are preserved."""
        content = "ATOM      1\nATOM      2\n"
        path = tmp_path / "test.pdb"
        path.write_text(content)

        result = read_structure(path)
        assert result == content


class TestWriteStructure:
    """Tests for write_structure function."""

    def test_write_pdb_format(self, tmp_path):
        """Test writing PDB format."""
        content = "ATOM      1  N   ALA A   1"
        path = tmp_path / "output.pdb"

        write_structure(content, path, StructureFormat.PDB)

        assert path.exists()
        assert path.read_text() == content

    def test_write_creates_parent_dirs(self, tmp_path):
        """Test that parent directories are created."""
        content = "ATOM      1  N   ALA A   1"
        path = tmp_path / "subdir" / "nested" / "output.pdb"

        write_structure(content, path)

        assert path.exists()
        assert path.read_text() == content

    def test_write_auto_detect_format(self, tmp_path):
        """Test format auto-detection from path."""
        content = "ATOM      1  N   ALA A   1"
        path = tmp_path / "output.pdb"

        # Should not raise even without explicit format
        write_structure(content, path)
        assert path.exists()


class TestConversion:
    """Tests for format conversion functions."""

    def test_pdb_to_cif_conversion(self, small_peptide_pdb_string):
        """Test converting PDB to CIF format."""
        cif_string = convert_pdb_to_cif(small_peptide_pdb_string)

        # CIF files start with data_ block
        assert cif_string.startswith("data_")
        # Should contain atom_site loop
        assert "_atom_site" in cif_string

    def test_cif_to_pdb_conversion(self, small_peptide_pdb_string):
        """Test converting CIF to PDB format."""
        # First convert to CIF
        cif_string = convert_pdb_to_cif(small_peptide_pdb_string)

        # Then convert back to PDB
        pdb_string, mapping = convert_cif_to_pdb(cif_string)

        # Should have ATOM records
        assert "ATOM" in pdb_string
        # Single-char chains — no remapping needed
        assert mapping == {}

    def test_roundtrip_preserves_atoms(self, small_peptide_pdb_string):
        """Test that roundtrip conversion preserves atom count."""
        # Count atoms in original
        original_atoms = len(
            [
                line
                for line in small_peptide_pdb_string.splitlines()
                if line.startswith("ATOM")
            ]
        )

        # Roundtrip: PDB -> CIF -> PDB
        cif_string = convert_pdb_to_cif(small_peptide_pdb_string)
        pdb_string, _ = convert_cif_to_pdb(cif_string)

        # Count atoms after roundtrip
        final_atoms = len(
            [
                line
                for line in pdb_string.splitlines()
                if line.startswith("ATOM")
            ]
        )

        assert final_atoms == original_atoms

    def test_convert_to_format_pdb(self, small_peptide_pdb_string):
        """Test convert_to_format with PDB target."""
        result = convert_to_format(
            small_peptide_pdb_string, StructureFormat.PDB
        )
        # Should return unchanged
        assert result == small_peptide_pdb_string

    def test_convert_to_format_cif(self, small_peptide_pdb_string):
        """Test convert_to_format with CIF target."""
        result = convert_to_format(
            small_peptide_pdb_string, StructureFormat.CIF
        )
        assert result.startswith("data_")


class TestEnsurePdbFormat:
    """Tests for ensure_pdb_format function."""

    def test_pdb_input_unchanged(self, tmp_path, small_peptide_pdb_string):
        """Test that PDB input is returned unchanged with empty mapping."""
        path = tmp_path / "test.pdb"
        path.write_text(small_peptide_pdb_string)

        pdb_string, mapping = ensure_pdb_format(
            small_peptide_pdb_string, path
        )
        assert pdb_string == small_peptide_pdb_string
        assert mapping == {}

    def test_cif_input_converted(self, tmp_path, small_peptide_pdb_string):
        """Test that CIF input is converted to PDB."""
        # Create CIF content
        cif_string = convert_pdb_to_cif(small_peptide_pdb_string)
        path = tmp_path / "test.cif"
        path.write_text(cif_string)

        pdb_string, mapping = ensure_pdb_format(cif_string, path)
        # Should be PDB format now
        assert "ATOM" in pdb_string
        assert not pdb_string.startswith("data_")
        # Single-char chains — no remapping needed
        assert mapping == {}


class TestGetOutputFormat:
    """Tests for get_output_format function."""

    def test_uses_output_path_extension(self, tmp_path):
        """Test that output path extension determines format."""
        input_path = tmp_path / "input.pdb"
        output_path = tmp_path / "output.cif"

        result = get_output_format(input_path, output_path)
        assert result == StructureFormat.CIF

    def test_pdb_output_extension(self, tmp_path):
        """Test PDB output extension."""
        input_path = tmp_path / "input.cif"
        output_path = tmp_path / "output.pdb"

        result = get_output_format(input_path, output_path)
        assert result == StructureFormat.PDB

    def test_falls_back_to_input_format(self, tmp_path):
        """Test fallback to input format when output has no extension."""
        input_path = tmp_path / "input.cif"
        output_path = tmp_path / "output"  # No extension

        result = get_output_format(input_path, output_path)
        assert result == StructureFormat.CIF

    def test_same_format_preserved(self, tmp_path):
        """Test same format is preserved."""
        input_path = tmp_path / "input.pdb"
        output_path = tmp_path / "output.pdb"

        result = get_output_format(input_path, output_path)
        assert result == StructureFormat.PDB


class TestChainIdRemapping:
    """Tests for multi-character CIF chain ID remapping."""

    def test_cif_single_char_no_remapping(self, small_peptide_pdb_string):
        """CIF with only single-char chains produces empty mapping."""
        cif_string = convert_pdb_to_cif(small_peptide_pdb_string)
        pdb_string, mapping = convert_cif_to_pdb(cif_string)
        assert mapping == {}
        assert "ATOM" in pdb_string

    def test_cif_multi_char_remapped(self):
        """Multi-char chain IDs are remapped to single chars."""
        cif_string = _build_multi_chain_cif()
        pdb_string, mapping = convert_cif_to_pdb(cif_string)

        # mapping should contain entries for the multi-char chains
        assert len(mapping) == 2
        # Original CIF IDs should be in the values
        assert "AA" in mapping.values()
        assert "AB" in mapping.values()
        # All keys should be single characters
        assert all(len(k) == 1 for k in mapping)
        # Result should be valid PDB
        assert "ATOM" in pdb_string

    def test_cif_mixed_chains_preserve_single(self):
        """Single-char chains keep their IDs; only multi-char are remapped."""
        cif_string = _build_multi_chain_cif()
        pdb_string, mapping = convert_cif_to_pdb(cif_string)

        # A and B should NOT appear as mapping keys (they kept their IDs)
        for new_id in mapping:
            assert new_id not in ("A", "B")

        # PDB should have chains A and B with original IDs
        pdb_chains = set()
        for line in pdb_string.splitlines():
            if line.startswith("ATOM"):
                pdb_chains.add(line[21])
        assert "A" in pdb_chains
        assert "B" in pdb_chains

    def test_roundtrip_preserves_chain_ids(self):
        """CIF -> PDB -> CIF restores original multi-char chain IDs."""
        cif_string = _build_multi_chain_cif()
        pdb_string, mapping = convert_cif_to_pdb(cif_string)

        # Convert back to CIF with mapping
        restored_cif = convert_pdb_to_cif(
            pdb_string, chain_id_mapping=mapping
        )

        # The restored CIF should contain the original chain IDs
        assert "AA" in restored_cif
        assert "AB" in restored_cif

    def test_too_many_chains_raises(self):
        """More than 62 chains raises ValueError."""
        from Bio.PDB.Chain import Chain
        from Bio.PDB.Residue import Residue
        from Bio.PDB.Atom import Atom

        structure = BioStructure("test")
        model = Model(0)
        structure.add(model)

        # Create 63 chains with multi-char IDs to exhaust pool
        for i in range(63):
            cid = f"C{i:02d}"
            chain = Chain(cid)
            model.add(chain)
            res = Residue((" ", 1, " "), "ALA", " ")
            chain.add(res)
            atom = Atom(
                "CA",
                [float(i), 0.0, 0.0],
                1.0,
                1.0,
                " ",
                "CA",
                1,
                element="C",
            )
            res.add(atom)

        assert _needs_chain_remapping(structure)
        with pytest.raises(ValueError, match="supports at most"):
            _remap_chain_ids(structure)

    def test_convert_pdb_to_cif_with_mapping(
        self, small_peptide_pdb_string
    ):
        """convert_pdb_to_cif restores original IDs when mapping provided."""
        mapping: ChainIdMapping = {"A": "XY"}
        cif_string = convert_pdb_to_cif(
            small_peptide_pdb_string, chain_id_mapping=mapping
        )
        # The CIF should reference chain "XY" instead of "A"
        assert "XY" in cif_string

    def test_ensure_pdb_format_returns_tuple_pdb(
        self, tmp_path, small_peptide_pdb_string
    ):
        """ensure_pdb_format returns (content, {}) for PDB input."""
        path = tmp_path / "test.pdb"
        path.write_text(small_peptide_pdb_string)
        result = ensure_pdb_format(small_peptide_pdb_string, path)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[1] == {}

    def test_ensure_pdb_format_returns_tuple_cif(
        self, tmp_path, small_peptide_pdb_string
    ):
        """ensure_pdb_format returns (pdb_string, mapping) for CIF input."""
        cif_string = convert_pdb_to_cif(small_peptide_pdb_string)
        path = tmp_path / "test.cif"
        path.write_text(cif_string)
        result = ensure_pdb_format(cif_string, path)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert "ATOM" in result[0]
        # Single-char chains, so mapping is empty
        assert result[1] == {}

    def test_needs_chain_remapping_false(self):
        """_needs_chain_remapping returns False for single-char chains."""
        structure = BioStructure("test")
        model = Model(0)
        structure.add(model)

        from Bio.PDB.Chain import Chain

        chain = Chain("A")
        model.add(chain)
        assert not _needs_chain_remapping(structure)

    def test_needs_chain_remapping_true(self):
        """_needs_chain_remapping returns True for multi-char chains."""
        structure = BioStructure("test")
        model = Model(0)
        structure.add(model)

        from Bio.PDB.Chain import Chain

        chain = Chain("AA")
        model.add(chain)
        assert _needs_chain_remapping(structure)

    def test_restore_cif_chain_ids(self):
        """restore_cif_chain_ids restores original chain IDs."""
        structure = BioStructure("test")
        model = Model(0)
        structure.add(model)

        from Bio.PDB.Chain import Chain

        chain_c = Chain("C")
        chain_d = Chain("D")
        model.add(chain_c)
        model.add(chain_d)

        mapping = {"C": "AA", "D": "AB"}
        restore_cif_chain_ids(structure, mapping)

        chain_ids = [c.id for c in structure.get_chains()]
        assert "AA" in chain_ids
        assert "AB" in chain_ids
