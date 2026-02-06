"""Tests for the select_positions operation."""

import pytest

from boundry.config import SelectPositionsConfig
from boundry.interface_position_energetics import PositionResult, PositionRow
from boundry.operations import Structure, select_positions
from boundry.resfile import DesignSpec, ResidueMode, ResidueSpec


_PDB = (
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000"
    "  1.00  0.00           N\nEND\n"
)


def _make_rows():
    """Build a list of PositionRows for testing."""
    return [
        PositionRow(
            chain_id="A",
            residue_number=10,
            insertion_code="",
            wt_resname="TYR",
            partner_chain="B",
            min_distance=3.2,
            num_contacts=5,
            dG_wt=-10.0,
            dG=-6.0,
            ddG=4.0,
        ),
        PositionRow(
            chain_id="A",
            residue_number=20,
            insertion_code="",
            wt_resname="ALA",
            partner_chain="B",
            min_distance=4.5,
            num_contacts=3,
            dG_wt=-10.0,
            dG=-9.5,
            ddG=0.5,
        ),
        PositionRow(
            chain_id="B",
            residue_number=15,
            insertion_code="",
            wt_resname="TRP",
            partner_chain="A",
            min_distance=3.0,
            num_contacts=8,
            dG_wt=-10.0,
            dG=-7.5,
            ddG=2.5,
        ),
    ]


def _make_structure(source="alanine_scan", rows=None):
    """Build a Structure with PositionResult in metadata."""
    if rows is None:
        rows = _make_rows()
    pr = PositionResult(rows=rows, dG_wt=-10.0)
    return Structure(
        pdb_string=_PDB,
        metadata={source: pr, "operation": "analyze_interface"},
    )


class TestSelectPositions:
    def test_basic_selection_above(self):
        """Positions with ddG > threshold are selected."""
        struct = _make_structure()
        result = select_positions(
            struct,
            config=SelectPositionsConfig(
                source="alanine_scan",
                metric="ddG",
                threshold=1.0,
                direction="above",
                mode="ALLAA",
            ),
        )
        spec = result.metadata["design_spec"]
        assert isinstance(spec, DesignSpec)
        # ddG > 1.0: A10 (4.0) and B15 (2.5), NOT A20 (0.5)
        assert len(spec.residue_specs) == 2
        assert "A10" in spec.residue_specs
        assert "B15" in spec.residue_specs
        assert "A20" not in spec.residue_specs
        assert spec.residue_specs["A10"].mode == ResidueMode.ALLAA

    def test_basic_selection_below(self):
        """direction='below' selects positions with metric < threshold."""
        struct = _make_structure()
        result = select_positions(
            struct,
            config=SelectPositionsConfig(
                metric="ddG",
                threshold=1.0,
                direction="below",
            ),
        )
        spec = result.metadata["design_spec"]
        # ddG < 1.0: only A20 (0.5)
        assert len(spec.residue_specs) == 1
        assert "A20" in spec.residue_specs

    def test_source_per_position(self):
        """Source 'per_position' reads from metadata['per_position']."""
        struct = _make_structure(source="per_position")
        result = select_positions(
            struct,
            config=SelectPositionsConfig(
                source="per_position",
                threshold=1.0,
            ),
        )
        assert result.metadata["selected_positions"] == 2
        assert result.metadata["selection_source"] == "per_position"

    def test_source_alanine_scan(self):
        """Source 'alanine_scan' reads from metadata['alanine_scan']."""
        struct = _make_structure(source="alanine_scan")
        result = select_positions(
            struct,
            config=SelectPositionsConfig(
                source="alanine_scan",
                threshold=1.0,
            ),
        )
        assert result.metadata["selected_positions"] == 2

    def test_missing_source_raises(self):
        """Raises ValueError when source key not in metadata."""
        struct = Structure(pdb_string=_PDB, metadata={})
        with pytest.raises(ValueError, match="not found"):
            select_positions(
                struct,
                config=SelectPositionsConfig(source="alanine_scan"),
            )

    def test_skipped_rows_excluded(self):
        """Rows with scan_skipped=True are not selected."""
        rows = [
            PositionRow(
                chain_id="A",
                residue_number=10,
                insertion_code="",
                wt_resname="GLY",
                partner_chain="B",
                min_distance=3.2,
                num_contacts=5,
                ddG=5.0,
                scan_skipped=True,
                skip_reason="GLY",
            ),
        ]
        struct = _make_structure(rows=rows)
        result = select_positions(struct)
        assert result.metadata["selected_positions"] == 0

    def test_none_metric_excluded(self):
        """Rows where the metric is None are skipped."""
        rows = [
            PositionRow(
                chain_id="A",
                residue_number=10,
                insertion_code="",
                wt_resname="TYR",
                partner_chain="B",
                min_distance=3.2,
                num_contacts=5,
                ddG=None,
            ),
        ]
        struct = _make_structure(rows=rows)
        result = select_positions(struct)
        assert result.metadata["selected_positions"] == 0

    def test_mode_allaa(self):
        """Selected positions get ALLAA mode."""
        struct = _make_structure()
        result = select_positions(
            struct,
            config=SelectPositionsConfig(
                threshold=1.0, mode="ALLAA"
            ),
        )
        spec = result.metadata["design_spec"]
        for rs in spec.residue_specs.values():
            assert rs.mode == ResidueMode.ALLAA

    def test_mode_pikaa_with_allowed_aas(self):
        """PIKAA mode assigns the provided allowed_aas."""
        struct = _make_structure()
        result = select_positions(
            struct,
            config=SelectPositionsConfig(
                threshold=1.0,
                mode="PIKAA",
                allowed_aas="ACF",
            ),
        )
        spec = result.metadata["design_spec"]
        for rs in spec.residue_specs.values():
            assert rs.mode == ResidueMode.PIKAA
            assert rs.allowed_aas == {"A", "C", "F"}

    def test_default_mode_applied(self):
        """Non-selected positions get default_mode in DesignSpec."""
        struct = _make_structure()
        result = select_positions(
            struct,
            config=SelectPositionsConfig(
                threshold=1.0,
                default_mode="NATRO",
            ),
        )
        spec = result.metadata["design_spec"]
        assert spec.default_mode == ResidueMode.NATRO

    def test_invalid_mode_raises(self):
        """Invalid mode string raises ValueError."""
        struct = _make_structure()
        with pytest.raises(ValueError, match="Unknown mode"):
            select_positions(
                struct,
                config=SelectPositionsConfig(mode="INVALID"),
            )

    def test_invalid_default_mode_raises(self):
        """Invalid default_mode string raises ValueError."""
        struct = _make_structure()
        with pytest.raises(ValueError, match="Unknown default_mode"):
            select_positions(
                struct,
                config=SelectPositionsConfig(default_mode="INVALID"),
            )

    def test_metadata_provenance(self):
        """Metadata records source, metric, threshold, direction."""
        struct = _make_structure()
        result = select_positions(
            struct,
            config=SelectPositionsConfig(
                source="alanine_scan",
                metric="ddG",
                threshold=2.0,
                direction="above",
                mode="ALLAA",
            ),
        )
        meta = result.metadata
        assert meta["operation"] == "select_positions"
        assert meta["selection_source"] == "alanine_scan"
        assert meta["selection_metric"] == "ddG"
        assert meta["selection_threshold"] == 2.0
        assert meta["selection_direction"] == "above"
        assert meta["selection_mode"] == "ALLAA"

    def test_pdb_string_unchanged(self):
        """PDB string is passed through without modification."""
        struct = _make_structure()
        result = select_positions(struct)
        assert result.pdb_string == _PDB

    def test_zero_selections(self):
        """Zero positions selected returns valid DesignSpec."""
        struct = _make_structure()
        result = select_positions(
            struct,
            config=SelectPositionsConfig(threshold=100.0),
        )
        spec = result.metadata["design_spec"]
        assert isinstance(spec, DesignSpec)
        assert len(spec.residue_specs) == 0
        assert result.metadata["selected_positions"] == 0

    def test_selected_positions_count(self):
        """metadata['selected_positions'] is correct count."""
        struct = _make_structure()
        result = select_positions(
            struct,
            config=SelectPositionsConfig(threshold=1.0),
        )
        assert result.metadata["selected_positions"] == 2

    def test_insertion_codes(self):
        """Insertion codes are preserved in ResidueSpec keys."""
        rows = [
            PositionRow(
                chain_id="A",
                residue_number=10,
                insertion_code="A",
                wt_resname="TYR",
                partner_chain="B",
                min_distance=3.2,
                num_contacts=5,
                ddG=5.0,
            ),
        ]
        struct = _make_structure(rows=rows)
        result = select_positions(struct)
        spec = result.metadata["design_spec"]
        assert "A10A" in spec.residue_specs
        rs = spec.residue_specs["A10A"]
        assert rs.icode == "A"

    def test_returns_structure(self):
        """Return type is Structure."""
        struct = _make_structure()
        result = select_positions(struct)
        assert isinstance(result, Structure)

    def test_metric_dG(self):
        """Can filter on dG instead of ddG."""
        struct = _make_structure()
        result = select_positions(
            struct,
            config=SelectPositionsConfig(
                metric="dG",
                threshold=-8.0,
                direction="below",
            ),
        )
        spec = result.metadata["design_spec"]
        # dG < -8.0: A20 (-9.5)
        assert len(spec.residue_specs) == 1
        assert "A20" in spec.residue_specs
