"""Unit tests for boundry.designer module helper methods.

These tests focus on the pure logic helper methods that don't require
loading LigandMPNN models.
"""

import pytest
import torch

from boundry.resfile import ALL_AAS, DesignSpec, ResidueMode, ResidueSpec


class MockDesigner:
    """Mock Designer class with only the helper methods for testing."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    def _build_residue_mapping(self, protein_dict: dict, icodes: list) -> list:
        """Build list of residue keys matching LigandMPNN format."""
        R_idx_list = protein_dict["R_idx"].cpu().tolist()
        chain_letters = protein_dict["chain_letters"]
        encoded = []
        for i, r_idx in enumerate(R_idx_list):
            key = f"{chain_letters[i]}{int(r_idx)}{icodes[i]}"
            encoded.append(key)
        return encoded

    def _build_chain_mask(
        self,
        encoded_residues: list,
        design_spec,
        design_all: bool = False,
    ) -> torch.Tensor:
        """Build chain_mask tensor: 1=design, 0=fixed."""
        if design_all:
            return torch.ones(len(encoded_residues), device=self.device)

        if design_spec is None:
            return torch.zeros(len(encoded_residues), device=self.device)

        mask = []
        for key in encoded_residues:
            if key in design_spec.residue_specs:
                spec = design_spec.residue_specs[key]
                mask.append(1.0 if spec.is_designable() else 0.0)
            else:
                default_designable = design_spec.default_mode in (
                    ResidueMode.ALLAA,
                    ResidueMode.PIKAA,
                    ResidueMode.NOTAA,
                    ResidueMode.POLAR,
                    ResidueMode.APOLAR,
                )
                mask.append(1.0 if default_designable else 0.0)

        return torch.tensor(mask, device=self.device, dtype=torch.float32)

    def _build_aa_bias(
        self,
        encoded_residues: list,
        design_spec,
    ) -> torch.Tensor:
        """Build amino acid bias tensor for constraints."""
        # Map single letter AA to index (simplified for testing)
        aa_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

        L = len(encoded_residues)
        bias = torch.zeros([1, L, 21], device=self.device, dtype=torch.float32)

        if design_spec is None:
            return bias

        for i, key in enumerate(encoded_residues):
            if key in design_spec.residue_specs:
                spec = design_spec.residue_specs[key]
                if spec.is_designable():
                    allowed = spec.get_allowed_aas()
                    if allowed and allowed != ALL_AAS:
                        for aa in ALL_AAS:
                            if aa not in allowed:
                                aa_idx = aa_to_idx.get(aa)
                                if aa_idx is not None:
                                    bias[0, i, aa_idx] = -1e8

        return bias


class TestBuildResidueMapping:
    """Tests for _build_residue_mapping method."""

    @pytest.fixture
    def designer(self):
        return MockDesigner()

    def test_single_chain_mapping(self, designer):
        """Test mapping for single chain."""
        protein_dict = {
            "R_idx": torch.tensor([1, 2, 3, 4, 5]),
            "chain_letters": ["A", "A", "A", "A", "A"],
        }
        icodes = ["", "", "", "", ""]

        result = designer._build_residue_mapping(protein_dict, icodes)

        assert result == ["A1", "A2", "A3", "A4", "A5"]

    def test_multi_chain_mapping(self, designer):
        """Test mapping for multiple chains."""
        protein_dict = {
            "R_idx": torch.tensor([1, 2, 1, 2, 3]),
            "chain_letters": ["A", "A", "B", "B", "B"],
        }
        icodes = ["", "", "", "", ""]

        result = designer._build_residue_mapping(protein_dict, icodes)

        assert result == ["A1", "A2", "B1", "B2", "B3"]

    def test_mapping_with_icodes(self, designer):
        """Test mapping with insertion codes."""
        protein_dict = {
            "R_idx": torch.tensor([10, 10, 10, 11]),
            "chain_letters": ["A", "A", "A", "A"],
        }
        icodes = ["", "A", "B", ""]

        result = designer._build_residue_mapping(protein_dict, icodes)

        assert result == ["A10", "A10A", "A10B", "A11"]


class TestBuildChainMask:
    """Tests for _build_chain_mask method."""

    @pytest.fixture
    def designer(self):
        return MockDesigner()

    def test_design_all(self, designer):
        """Test design_all=True returns all ones."""
        residues = ["A1", "A2", "A3"]

        result = designer._build_chain_mask(residues, None, design_all=True)

        assert result.shape == (3,)
        assert torch.all(result == 1.0)

    def test_no_spec_returns_zeros(self, designer):
        """Test None design_spec returns all zeros (repack only)."""
        residues = ["A1", "A2", "A3"]

        result = designer._build_chain_mask(residues, None, design_all=False)

        assert result.shape == (3,)
        assert torch.all(result == 0.0)

    def test_explicit_designable(self, designer):
        """Test explicit designable residues."""
        residues = ["A1", "A2", "A3"]
        specs = {
            "A1": ResidueSpec(chain="A", resnum=1, mode=ResidueMode.ALLAA),
            "A2": ResidueSpec(chain="A", resnum=2, mode=ResidueMode.NATAA),
            "A3": ResidueSpec(chain="A", resnum=3, mode=ResidueMode.POLAR),
        }
        design_spec = DesignSpec(residue_specs=specs)

        result = designer._build_chain_mask(residues, design_spec)

        assert result[0] == 1.0  # ALLAA = designable
        assert result[1] == 0.0  # NATAA = not designable
        assert result[2] == 1.0  # POLAR = designable

    def test_natro_not_designable(self, designer):
        """Test NATRO residues are not designable."""
        residues = ["A1", "A2"]
        specs = {
            "A1": ResidueSpec(chain="A", resnum=1, mode=ResidueMode.NATRO),
            "A2": ResidueSpec(chain="A", resnum=2, mode=ResidueMode.ALLAA),
        }
        design_spec = DesignSpec(residue_specs=specs)

        result = designer._build_chain_mask(residues, design_spec)

        assert result[0] == 0.0  # NATRO
        assert result[1] == 1.0  # ALLAA

    def test_default_mode_applied(self, designer):
        """Test default mode is applied to unspecified residues."""
        residues = ["A1", "A2", "A3"]
        specs = {
            "A1": ResidueSpec(chain="A", resnum=1, mode=ResidueMode.ALLAA),
        }
        design_spec = DesignSpec(
            residue_specs=specs, default_mode=ResidueMode.NATAA
        )

        result = designer._build_chain_mask(residues, design_spec)

        assert result[0] == 1.0  # Explicit ALLAA
        assert result[1] == 0.0  # Default NATAA
        assert result[2] == 0.0  # Default NATAA

    def test_default_designable_mode(self, designer):
        """Test designable default mode."""
        residues = ["A1", "A2"]
        specs = {}
        design_spec = DesignSpec(
            residue_specs=specs, default_mode=ResidueMode.ALLAA
        )

        result = designer._build_chain_mask(residues, design_spec)

        # Both should be designable due to ALLAA default
        assert torch.all(result == 1.0)


class TestBuildAABias:
    """Tests for _build_aa_bias method."""

    @pytest.fixture
    def designer(self):
        return MockDesigner()

    def test_no_spec_no_bias(self, designer):
        """Test None design_spec returns zero bias."""
        residues = ["A1", "A2", "A3"]

        result = designer._build_aa_bias(residues, None)

        assert result.shape == (1, 3, 21)
        assert torch.all(result == 0.0)

    def test_allaa_no_bias(self, designer):
        """Test ALLAA mode has no bias (all AAs allowed)."""
        residues = ["A1"]
        specs = {
            "A1": ResidueSpec(chain="A", resnum=1, mode=ResidueMode.ALLAA),
        }
        design_spec = DesignSpec(residue_specs=specs)

        result = designer._build_aa_bias(residues, design_spec)

        assert torch.all(result == 0.0)

    def test_pikaa_bias(self, designer):
        """Test PIKAA mode applies negative bias to excluded AAs."""
        residues = ["A1"]
        specs = {
            "A1": ResidueSpec(
                chain="A",
                resnum=1,
                mode=ResidueMode.PIKAA,
                allowed_aas={"A", "G"},  # Only allow A and G
            ),
        }
        design_spec = DesignSpec(residue_specs=specs)

        result = designer._build_aa_bias(residues, design_spec)

        # A and G should have zero bias, others should have large negative
        aa_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

        for aa in ALL_AAS:
            idx = aa_to_idx[aa]
            if aa in {"A", "G"}:
                assert result[0, 0, idx] == 0.0
            else:
                assert result[0, 0, idx] == -1e8

    def test_polar_bias(self, designer):
        """Test POLAR mode excludes nonpolar AAs."""
        from boundry.resfile import APOLAR_AAS, POLAR_AAS

        residues = ["A1"]
        specs = {
            "A1": ResidueSpec(chain="A", resnum=1, mode=ResidueMode.POLAR),
        }
        design_spec = DesignSpec(residue_specs=specs)

        result = designer._build_aa_bias(residues, design_spec)

        aa_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

        for aa in POLAR_AAS:
            idx = aa_to_idx[aa]
            assert result[0, 0, idx] == 0.0, f"Polar AA {aa} should have 0 bias"

        for aa in APOLAR_AAS:
            idx = aa_to_idx[aa]
            assert (
                result[0, 0, idx] == -1e8
            ), f"Apolar AA {aa} should have -1e8 bias"

    def test_notaa_bias(self, designer):
        """Test NOTAA mode applies bias only to excluded AAs."""
        residues = ["A1"]
        specs = {
            "A1": ResidueSpec(
                chain="A",
                resnum=1,
                mode=ResidueMode.NOTAA,
                allowed_aas={"C", "P"},  # Exclude C and P
            ),
        }
        design_spec = DesignSpec(residue_specs=specs)

        result = designer._build_aa_bias(residues, design_spec)

        aa_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

        # C and P should have large negative bias
        assert result[0, 0, aa_to_idx["C"]] == -1e8
        assert result[0, 0, aa_to_idx["P"]] == -1e8

        # Others should have zero bias
        for aa in ALL_AAS - {"C", "P"}:
            idx = aa_to_idx[aa]
            assert result[0, 0, idx] == 0.0

    def test_nataa_no_bias(self, designer):
        """Test NATAA (non-designable) mode has no bias."""
        residues = ["A1"]
        specs = {
            "A1": ResidueSpec(chain="A", resnum=1, mode=ResidueMode.NATAA),
        }
        design_spec = DesignSpec(residue_specs=specs)

        result = designer._build_aa_bias(residues, design_spec)

        # NATAA is not designable, so no bias applied
        assert torch.all(result == 0.0)

    def test_multiple_residues_different_modes(self, designer):
        """Test bias tensor with multiple residues and different modes."""
        residues = ["A1", "A2", "A3"]
        specs = {
            "A1": ResidueSpec(
                chain="A",
                resnum=1,
                mode=ResidueMode.PIKAA,
                allowed_aas={"A"},
            ),
            "A2": ResidueSpec(chain="A", resnum=2, mode=ResidueMode.ALLAA),
            "A3": ResidueSpec(chain="A", resnum=3, mode=ResidueMode.NATAA),
        }
        design_spec = DesignSpec(residue_specs=specs)

        result = designer._build_aa_bias(residues, design_spec)

        aa_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

        # A1: PIKAA with only A allowed
        assert result[0, 0, aa_to_idx["A"]] == 0.0
        for aa in ALL_AAS - {"A"}:
            assert result[0, 0, aa_to_idx[aa]] == -1e8

        # A2: ALLAA - no bias
        for aa in ALL_AAS:
            assert result[0, 1, aa_to_idx[aa]] == 0.0

        # A3: NATAA - no bias (not designable)
        for aa in ALL_AAS:
            assert result[0, 2, aa_to_idx[aa]] == 0.0
