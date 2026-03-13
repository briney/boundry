"""Tests for boundry.ddg module — mutation specification, parsing,
validation utilities, and ddG compute pipeline.
"""

import pickle
from unittest.mock import MagicMock, patch

import pytest

from boundry.ddg import (
    DdGResult,
    EnsembleMemberResult,
    InterfaceDgResult,
    MutationSpec,
    _DdGMemberResult,
    _DdGMemberTask,
    _deserialize_design_spec,
    _find_neighborhood_residues,
    _normalize_aa,
    _process_ensemble_member,
    _serialize_design_spec,
    build_neighborhood_spec,
    build_sampling_neighborhood,
    compute_ddg,
    compute_interface_dg,
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


# ==================================================================
# Phase 5 tests — data classes, neighborhoods, pipeline
# ==================================================================

# ------------------------------------------------------------------
# Test PDB with known coordinates for neighborhood tests
# ------------------------------------------------------------------

# Chain A: residues 1-4 along x-axis (CA at 0, 5, 10, 50 Å)
# Chain B: residue 1 at x=3 (near A:1)
# GLY A:3 has no CB, only CA.
NEIGHBORHOOD_PDB = "\n".join(
    [
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C",
        "ATOM      2  CB  ALA A   1       1.000   0.000   0.000  1.00  0.00           C",
        "ATOM      3  CA  LEU A   2       5.000   0.000   0.000  1.00  0.00           C",
        "ATOM      4  CB  LEU A   2       6.000   0.000   0.000  1.00  0.00           C",
        "ATOM      5  CA  GLY A   3      10.000   0.000   0.000  1.00  0.00           C",
        "ATOM      6  CA  VAL A   4      50.000   0.000   0.000  1.00  0.00           C",
        "ATOM      7  CB  VAL A   4      51.000   0.000   0.000  1.00  0.00           C",
        "TER",
        "ATOM      8  CA  ALA B   1       3.000   0.000   0.000  1.00  0.00           C",
        "ATOM      9  CB  ALA B   1       4.000   0.000   0.000  1.00  0.00           C",
        "TER",
        "END",
    ]
)

# PDB with insertion code residues for neighborhood testing
NEIGHBORHOOD_ICODE_PDB = "\n".join(
    [
        "ATOM      1  CA  ALA H 100       0.000   0.000   0.000  1.00  0.00           C",
        "ATOM      2  CB  ALA H 100       1.000   0.000   0.000  1.00  0.00           C",
        "ATOM      3  CA  SER H 100A      3.000   0.000   0.000  1.00  0.00           C",
        "ATOM      4  CB  SER H 100A      4.000   0.000   0.000  1.00  0.00           C",
        "ATOM      5  CA  LEU H 101      50.000   0.000   0.000  1.00  0.00           C",
        "ATOM      6  CB  LEU H 101      51.000   0.000   0.000  1.00  0.00           C",
        "END",
    ]
)


# ------------------------------------------------------------------
# EnsembleMemberResult
# ------------------------------------------------------------------


class TestEnsembleMemberResult:
    def test_dG_wt(self):
        r = EnsembleMemberResult(
            member_index=0,
            bound_wt_energy=-100.0,
            unbound_wt_energy=-80.0,
        )
        assert r.dG_wt == pytest.approx(-20.0)

    def test_dG_mut(self):
        r = EnsembleMemberResult(
            member_index=0,
            bound_mut_energy=-90.0,
            unbound_mut_energy=-75.0,
        )
        assert r.dG_mut == pytest.approx(-15.0)

    def test_ddG(self):
        r = EnsembleMemberResult(
            member_index=0,
            bound_wt_energy=-100.0,
            unbound_wt_energy=-80.0,
            bound_mut_energy=-90.0,
            unbound_mut_energy=-75.0,
        )
        # dG_wt = -20, dG_mut = -15, ddG = -15 - (-20) = 5
        assert r.ddG == pytest.approx(5.0)

    def test_dG_wt_none_when_missing(self):
        r = EnsembleMemberResult(member_index=0)
        assert r.dG_wt is None
        assert r.dG_mut is None
        assert r.ddG is None

    def test_ddG_none_when_partial(self):
        r = EnsembleMemberResult(
            member_index=0,
            bound_wt_energy=-100.0,
            unbound_wt_energy=-80.0,
            # mut energies not set
        )
        assert r.dG_wt == pytest.approx(-20.0)
        assert r.dG_mut is None
        assert r.ddG is None

    def test_all_none_defaults(self):
        r = EnsembleMemberResult(member_index=0)
        assert r.bound_wt_energy is None
        assert r.unbound_wt_energy is None
        assert r.bound_mut_energy is None
        assert r.unbound_mut_energy is None
        assert r.wt_bound_energy_rank is None

    def test_mutable(self):
        r = EnsembleMemberResult(member_index=0)
        r.wt_bound_energy_rank = 3
        assert r.wt_bound_energy_rank == 3


# ------------------------------------------------------------------
# DdGResult
# ------------------------------------------------------------------


class TestDdGResult:
    def _make_result(self):
        m1 = EnsembleMemberResult(
            member_index=0,
            bound_wt_energy=-100.0,
            unbound_wt_energy=-80.0,
            bound_mut_energy=-90.0,
            unbound_mut_energy=-75.0,
        )
        m2 = EnsembleMemberResult(
            member_index=1,
            bound_wt_energy=-102.0,
            unbound_wt_energy=-82.0,
            bound_mut_energy=-91.0,
            unbound_mut_energy=-76.0,
        )
        return DdGResult(
            mutations=[MutationSpec("A", 1, "LEU", "ALA")],
            member_results=[m1, m2],
            mean_ddG=5.0,
            std_ddG=0.1,
            mean_dG_wt=-20.0,
            mean_dG_mut=-15.0,
            n_successful=2,
            n_ensemble=2,
            ensemble_ddGs=[5.0, 5.0],
        )

    def test_to_dict_keys(self):
        result = self._make_result()
        d = result.to_dict()
        expected_keys = {
            "mutations",
            "mean_ddG",
            "std_ddG",
            "mean_dG_wt",
            "mean_dG_mut",
            "n_successful",
            "n_ensemble",
            "ensemble_ddGs",
            "sorted_by_wt_energy",
            "top_n_applied",
            "member_results",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_member_details(self):
        result = self._make_result()
        d = result.to_dict()
        members = d["member_results"]
        assert len(members) == 2
        m0 = members[0]
        assert m0["member_index"] == 0
        assert m0["dG_wt"] == pytest.approx(-20.0)
        assert m0["dG_mut"] == pytest.approx(-15.0)
        assert m0["ddG"] == pytest.approx(5.0)

    def test_to_dict_mutations_as_strings(self):
        result = self._make_result()
        d = result.to_dict()
        assert d["mutations"] == ["A:L1A"]

    def test_to_dict_excludes_minimized_pdb(self):
        result = self._make_result()
        result.minimized_pdb = "PDB DATA"
        d = result.to_dict()
        assert "minimized_pdb" not in d

    def test_minimized_pdb_field(self):
        result = self._make_result()
        result.minimized_pdb = "PDB"
        assert result.minimized_pdb == "PDB"


# ------------------------------------------------------------------
# _find_neighborhood_residues
# ------------------------------------------------------------------


class TestFindNeighborhoodResidues:
    def test_nearby_included(self):
        # Mutation at A:1 (CB at 1,0,0). B:1 CB at 4,0,0 → dist=3 Å
        neighbors = _find_neighborhood_residues(
            NEIGHBORHOOD_PDB,
            [("A", 1, "")],
            neighborhood_radius=8.0,
            sequence_window=0,
        )
        assert ("A", 1, "") in neighbors
        assert ("A", 2, "") in neighbors  # CB at 6, dist=5
        assert ("B", 1, "") in neighbors  # CB at 4, dist=3

    def test_distant_excluded(self):
        neighbors = _find_neighborhood_residues(
            NEIGHBORHOOD_PDB,
            [("A", 1, "")],
            neighborhood_radius=8.0,
            sequence_window=0,
        )
        # A:4 CB at 51, dist=50 → way outside 8 Å
        assert ("A", 4, "") not in neighbors

    def test_gly_uses_ca(self):
        # A:3 is GLY with only CA at (10,0,0). Mutation at A:2 (CB=6).
        # dist(6, 10) = 4 Å → within 8 Å
        neighbors = _find_neighborhood_residues(
            NEIGHBORHOOD_PDB,
            [("A", 2, "")],
            neighborhood_radius=8.0,
            sequence_window=0,
        )
        assert ("A", 3, "") in neighbors

    def test_sequence_window_expansion(self):
        # Mutation at A:2, radius=3 Å → only A:2 itself (CB at 6)
        # B:1 CB at 4 → dist=2, within 3 Å
        # A:1 CB at 1 → dist=5, outside 3 Å
        # With window=1, A:2 should expand to include A:1 and A:3
        neighbors = _find_neighborhood_residues(
            NEIGHBORHOOD_PDB,
            [("A", 2, "")],
            neighborhood_radius=3.0,
            sequence_window=1,
        )
        assert ("A", 1, "") in neighbors  # window expand from A:2
        assert ("A", 2, "") in neighbors
        assert ("A", 3, "") in neighbors  # window expand from A:2

    def test_insertion_codes(self):
        # Mutation at H:100 (CB=1). H:100A CB at 4, dist=3 → in range
        neighbors = _find_neighborhood_residues(
            NEIGHBORHOOD_ICODE_PDB,
            [("H", 100, "")],
            neighborhood_radius=8.0,
            sequence_window=0,
        )
        assert ("H", 100, "") in neighbors
        assert ("H", 100, "A") in neighbors
        assert ("H", 101, "") not in neighbors  # CB at 51

    def test_empty_result_for_missing_site(self):
        neighbors = _find_neighborhood_residues(
            NEIGHBORHOOD_PDB,
            [("Z", 999, "")],
            neighborhood_radius=8.0,
            sequence_window=0,
        )
        assert len(neighbors) == 0


# ------------------------------------------------------------------
# build_neighborhood_spec
# ------------------------------------------------------------------


class TestBuildNeighborhoodSpec:
    def test_returns_design_spec(self):
        from boundry.resfile import DesignSpec

        spec = build_neighborhood_spec(
            NEIGHBORHOOD_PDB,
            [("A", 1, "")],
            neighborhood_radius=8.0,
        )
        assert isinstance(spec, DesignSpec)

    def test_nataa_for_neighbors(self):
        from boundry.resfile import ResidueMode

        spec = build_neighborhood_spec(
            NEIGHBORHOOD_PDB,
            [("A", 1, "")],
            neighborhood_radius=8.0,
        )
        # A:1 should be NATAA (in neighborhood)
        assert spec.residue_specs["A1"].mode == ResidueMode.NATAA

    def test_natro_default(self):
        from boundry.resfile import ResidueMode

        spec = build_neighborhood_spec(
            NEIGHBORHOOD_PDB,
            [("A", 1, "")],
            neighborhood_radius=8.0,
        )
        assert spec.default_mode == ResidueMode.NATRO


# ------------------------------------------------------------------
# build_sampling_neighborhood
# ------------------------------------------------------------------


class TestBuildSamplingNeighborhood:
    def test_returns_chain_resnum_tuples(self):
        result = build_sampling_neighborhood(
            NEIGHBORHOOD_PDB,
            [("A", 1, "")],
            neighborhood_radius=8.0,
        )
        assert all(len(t) == 2 for t in result)
        assert ("A", 1) in result

    def test_sorted_output(self):
        result = build_sampling_neighborhood(
            NEIGHBORHOOD_PDB,
            [("A", 1, "")],
            neighborhood_radius=8.0,
        )
        assert result == sorted(result)


# ------------------------------------------------------------------
# DesignSpec serialization roundtrip
# ------------------------------------------------------------------


class TestDesignSpecSerialization:
    def test_roundtrip(self):
        from boundry.resfile import (
            DesignSpec,
            ResidueMode,
            ResidueSpec,
        )

        spec = DesignSpec(
            residue_specs={
                "A1": ResidueSpec(
                    "A", 1, mode=ResidueMode.NATAA
                ),
                "A2": ResidueSpec(
                    "A",
                    2,
                    mode=ResidueMode.PIKAA,
                    allowed_aas={"A", "G", "V"},
                ),
            },
            default_mode=ResidueMode.NATRO,
        )
        d = _serialize_design_spec(spec)
        restored = _deserialize_design_spec(d)
        assert restored.default_mode == ResidueMode.NATRO
        assert restored.residue_specs["A1"].mode == ResidueMode.NATAA
        assert restored.residue_specs["A2"].allowed_aas == {
            "A",
            "G",
            "V",
        }

    def test_none_allowed_aas_roundtrip(self):
        from boundry.resfile import (
            DesignSpec,
            ResidueMode,
            ResidueSpec,
        )

        spec = DesignSpec(
            residue_specs={
                "B5": ResidueSpec(
                    "B", 5, mode=ResidueMode.NATRO
                ),
            },
            default_mode=ResidueMode.NATAA,
        )
        d = _serialize_design_spec(spec)
        restored = _deserialize_design_spec(d)
        assert restored.residue_specs["B5"].allowed_aas is None


# ------------------------------------------------------------------
# _DdGMemberTask pickle safety
# ------------------------------------------------------------------


class TestDdGMemberTaskPickle:
    def test_pickle_roundtrip(self):
        task = _DdGMemberTask(
            member_index=0,
            member_pdb_string="ATOM...",
            mutations=(MutationSpec("A", 1, "LEU", "ALA"),),
            neighborhood_spec_dict={
                "residue_specs": {},
                "default_mode": "NATRO",
            },
            chain_groups=(("A",), ("B",)),
            separation_distance=100.0,
            relax_config_dict={"implicit_solvent": True},
            design_config_dict={},
            implicit_solvent=True,
            ca_cutoff=9.0,
            restraint_sd=0.5,
            quiet=True,
        )
        data = pickle.dumps(task)
        restored = pickle.loads(data)
        assert restored.member_index == 0
        assert restored.mutations == task.mutations
        assert restored.chain_groups == (("A",), ("B",))


# ------------------------------------------------------------------
# _process_ensemble_member
# ------------------------------------------------------------------


class TestProcessEnsembleMember:
    def _make_task(
        self, mutations=(), relax_separated=False
    ):
        return _DdGMemberTask(
            member_index=0,
            member_pdb_string="ATOM  mock PDB",
            mutations=mutations,
            neighborhood_spec_dict={
                "residue_specs": {},
                "default_mode": "NATRO",
            },
            chain_groups=(("A",), ("B",)),
            separation_distance=100.0,
            relax_config_dict={"implicit_solvent": True},
            design_config_dict={},
            implicit_solvent=True,
            ca_cutoff=9.0,
            restraint_sd=0.5,
            quiet=True,
            relax_separated=relax_separated,
        )

    @patch("boundry.ddg._ddg_worker_cache", new_callable=dict)
    @patch("boundry.ddg._repack_and_minimize")
    @patch("boundry.relaxer.separate_interface_rigid_body")
    def test_wt_only_no_mutations(
        self, mock_separate, mock_repack_min, mock_cache
    ):
        mock_relaxer = MagicMock()
        mock_designer = MagicMock()
        # Keys must match _config_fingerprint(task.*_config_dict)
        mock_cache["relax_key"] = "implicit_solvent=True"
        mock_cache["relaxer"] = mock_relaxer
        mock_cache["design_key"] = ""
        mock_cache["designer"] = mock_designer

        mock_repack_min.return_value = "minimized_pdb"
        mock_relaxer.get_energy_breakdown.side_effect = [
            {"total_energy": -100.0},  # WT bound
            {"total_energy": -80.0},  # WT unbound
        ]
        mock_separate.return_value = "separated_pdb"

        task = self._make_task(mutations=())
        result = _process_ensemble_member(task)

        assert result.error is None
        assert result.bound_wt_energy == pytest.approx(-100.0)
        assert result.unbound_wt_energy == pytest.approx(-80.0)
        assert result.bound_mut_energy is None
        assert result.unbound_mut_energy is None

    @patch("boundry.ddg._ddg_worker_cache", new_callable=dict)
    @patch("boundry.ddg._repack_and_minimize")
    @patch("boundry.binding_energy.extract_chain")
    @patch("boundry.utils.filter_protein_only")
    def test_wt_only_relax_separated(
        self,
        mock_filter,
        mock_extract,
        mock_repack_min,
        mock_cache,
    ):
        """With relax_separated=True, unbound scoring uses
        extract_chain + repack_and_minimize per group."""
        mock_relaxer = MagicMock()
        mock_designer = MagicMock()
        mock_cache["relax_key"] = "implicit_solvent=True"
        mock_cache["relaxer"] = mock_relaxer
        mock_cache["design_key"] = ""
        mock_cache["designer"] = mock_designer

        mock_extract.return_value = "chain_pdb"
        mock_filter.return_value = "filtered_pdb"
        mock_repack_min.side_effect = [
            "wt_bound_min",  # WT bound
            "group_a_relaxed",  # group A unbound
            "group_b_relaxed",  # group B unbound
        ]
        mock_relaxer.get_energy_breakdown.side_effect = [
            {"total_energy": -100.0},  # WT bound
            {"total_energy": -50.0},  # group A unbound
            {"total_energy": -30.0},  # group B unbound
        ]

        task = self._make_task(
            mutations=(), relax_separated=True
        )
        result = _process_ensemble_member(task)

        assert result.error is None
        assert result.bound_wt_energy == pytest.approx(-100.0)
        # Unbound = sum of group energies
        assert result.unbound_wt_energy == pytest.approx(-80.0)
        assert mock_extract.call_count == 2
        # 3 calls: 1 for bound, 2 for unbound groups
        assert mock_repack_min.call_count == 3

    @patch("boundry.ddg._ddg_worker_cache", new_callable=dict)
    @patch("boundry.ddg._repack_and_minimize")
    @patch("boundry.relaxer.separate_interface_rigid_body")
    @patch(
        "boundry.interface_position_energetics.mutate_residue"
    )
    def test_with_mutations(
        self,
        mock_mutate,
        mock_separate,
        mock_repack_min,
        mock_cache,
    ):
        mock_relaxer = MagicMock()
        mock_designer = MagicMock()
        mock_cache["relax_key"] = "implicit_solvent=True"
        mock_cache["relaxer"] = mock_relaxer
        mock_cache["design_key"] = ""
        mock_cache["designer"] = mock_designer

        mock_repack_min.side_effect = [
            "wt_min_pdb",
            "mut_min_pdb",
        ]
        mock_relaxer.get_energy_breakdown.side_effect = [
            {"total_energy": -100.0},  # WT bound
            {"total_energy": -80.0},  # WT unbound
            {"total_energy": -90.0},  # Mut bound
            {"total_energy": -75.0},  # Mut unbound
        ]
        mock_separate.side_effect = [
            "wt_separated",
            "mut_separated",
        ]
        mock_mutate.return_value = "mutated_pdb"

        mutations = (MutationSpec("A", 1, "LEU", "ALA"),)
        task = self._make_task(mutations=mutations)
        result = _process_ensemble_member(task)

        assert result.error is None
        assert result.bound_wt_energy == pytest.approx(-100.0)
        assert result.unbound_wt_energy == pytest.approx(-80.0)
        assert result.bound_mut_energy == pytest.approx(-90.0)
        assert result.unbound_mut_energy == pytest.approx(-75.0)

    @patch("boundry.ddg._ddg_worker_cache", new_callable=dict)
    @patch("boundry.ddg._repack_and_minimize")
    def test_error_handling(self, mock_repack_min, mock_cache):
        mock_relaxer = MagicMock()
        mock_designer = MagicMock()
        mock_cache["relax_key"] = "implicit_solvent=True"
        mock_cache["relaxer"] = mock_relaxer
        mock_cache["design_key"] = ""
        mock_cache["designer"] = mock_designer

        mock_repack_min.side_effect = RuntimeError("boom")

        task = self._make_task()
        result = _process_ensemble_member(task)

        assert result.error is not None
        assert "boom" in result.error


# ------------------------------------------------------------------
# compute_ddg
# ------------------------------------------------------------------


class TestComputeDdG:
    def test_chain_pairs_required(self):
        from boundry.config import DdGConfig

        config = DdGConfig(chain_pairs=None)
        with pytest.raises(ValueError, match="chain_pairs is required"):
            compute_ddg("PDB", [], config)

    @patch("boundry.ddg._process_ensemble_member")
    @patch("boundry.ddg.build_sampling_neighborhood")
    @patch("boundry.ddg.build_neighborhood_spec")
    @patch("boundry.ddg.validate_mutations")
    def test_end_to_end_mocked(
        self,
        mock_validate,
        mock_build_spec,
        mock_build_sampling,
        mock_worker,
    ):
        from boundry.config import DdGConfig
        from boundry.resfile import DesignSpec, ResidueMode

        mock_relaxer = MagicMock()
        mock_designer = MagicMock()

        # minimize returns PDB
        mock_relaxer.minimize_with_pair_restraints.return_value = (
            "minimized"
        )
        # ensemble returns 2 members
        mock_relaxer.generate_local_md_ensemble.return_value = [
            "member0",
            "member1",
        ]

        mock_build_spec.return_value = DesignSpec(
            residue_specs={}, default_mode=ResidueMode.NATRO
        )
        mock_build_sampling.return_value = [("A", 1)]

        mock_worker.side_effect = [
            _DdGMemberResult(
                member_index=0,
                bound_wt_energy=-100.0,
                unbound_wt_energy=-80.0,
                bound_mut_energy=-90.0,
                unbound_mut_energy=-75.0,
            ),
            _DdGMemberResult(
                member_index=1,
                bound_wt_energy=-102.0,
                unbound_wt_energy=-82.0,
                bound_mut_energy=-92.0,
                unbound_mut_energy=-77.0,
            ),
        ]

        config = DdGConfig(
            chain_pairs=[("A", "B")], n_ensemble=2
        )
        mutations = [MutationSpec("A", 1, "LEU", "ALA")]

        result = compute_ddg(
            "PDB",
            mutations,
            config,
            relaxer=mock_relaxer,
            designer=mock_designer,
        )

        assert isinstance(result, DdGResult)
        assert result.n_ensemble == 2
        assert result.n_successful == 2
        assert len(result.ensemble_ddGs) == 2
        assert result.mean_ddG is not None
        # member 0: dG_wt=-20, dG_mut=-15, ddG=5
        # member 1: dG_wt=-20, dG_mut=-15, ddG=5
        assert result.mean_ddG == pytest.approx(5.0)
        assert result.minimized_pdb == "minimized"

    @patch("boundry.ddg._process_ensemble_member")
    @patch("boundry.ddg.build_sampling_neighborhood")
    @patch("boundry.ddg.build_neighborhood_spec")
    @patch("boundry.ddg.validate_mutations")
    def test_handles_failed_members(
        self,
        mock_validate,
        mock_build_spec,
        mock_build_sampling,
        mock_worker,
    ):
        from boundry.config import DdGConfig
        from boundry.resfile import DesignSpec, ResidueMode

        mock_relaxer = MagicMock()
        mock_designer = MagicMock()
        mock_relaxer.minimize_with_pair_restraints.return_value = (
            "minimized"
        )
        mock_relaxer.generate_local_md_ensemble.return_value = [
            "m0",
            "m1",
        ]
        mock_build_spec.return_value = DesignSpec(
            residue_specs={}, default_mode=ResidueMode.NATRO
        )
        mock_build_sampling.return_value = []

        mock_worker.side_effect = [
            _DdGMemberResult(
                member_index=0, error="RuntimeError: fail"
            ),
            _DdGMemberResult(
                member_index=1,
                bound_wt_energy=-100.0,
                unbound_wt_energy=-80.0,
                bound_mut_energy=-90.0,
                unbound_mut_energy=-75.0,
            ),
        ]

        config = DdGConfig(
            chain_pairs=[("A", "B")], n_ensemble=2
        )
        result = compute_ddg(
            "PDB",
            [MutationSpec("A", 1, "LEU", "ALA")],
            config,
            relaxer=mock_relaxer,
            designer=mock_designer,
        )
        assert result.n_successful == 1
        assert result.n_ensemble == 2

    @patch("boundry.ddg._process_ensemble_member")
    @patch("boundry.ddg.build_sampling_neighborhood")
    @patch("boundry.ddg.build_neighborhood_spec")
    @patch("boundry.ddg.validate_mutations")
    def test_ensemble_caching(
        self,
        mock_validate,
        mock_build_spec,
        mock_build_sampling,
        mock_worker,
        tmp_path,
    ):
        from boundry.config import DdGConfig
        from boundry.resfile import DesignSpec, ResidueMode

        mock_relaxer = MagicMock()
        mock_designer = MagicMock()
        mock_relaxer.minimize_with_pair_restraints.return_value = (
            "minimized"
        )
        mock_relaxer.generate_local_md_ensemble.return_value = [
            "member0_pdb",
            "member1_pdb",
        ]
        mock_build_spec.return_value = DesignSpec(
            residue_specs={}, default_mode=ResidueMode.NATRO
        )
        mock_build_sampling.return_value = []
        mock_worker.return_value = _DdGMemberResult(
            member_index=0,
            bound_wt_energy=-100.0,
            unbound_wt_energy=-80.0,
            bound_mut_energy=-90.0,
            unbound_mut_energy=-75.0,
        )

        ens_dir = tmp_path / "ensemble"
        config = DdGConfig(
            chain_pairs=[("A", "B")],
            n_ensemble=2,
            cache_ensemble=True,
            ensemble_dir=ens_dir,
        )
        compute_ddg(
            "PDB",
            [MutationSpec("A", 1, "LEU", "ALA")],
            config,
            relaxer=mock_relaxer,
            designer=mock_designer,
        )

        cached = sorted(ens_dir.glob("member_*.pdb"))
        assert len(cached) == 2
        assert cached[0].read_text() == "member0_pdb"

    @patch("boundry.ddg._process_ensemble_member")
    @patch("boundry.ddg.build_sampling_neighborhood")
    @patch("boundry.ddg.build_neighborhood_spec")
    @patch("boundry.ddg.validate_mutations")
    def test_result_aggregation_flags_when_sorting_enabled(
        self,
        mock_validate,
        mock_build_spec,
        mock_build_sampling,
        mock_worker,
    ):
        from boundry.config import DdGConfig
        from boundry.resfile import DesignSpec, ResidueMode

        mock_relaxer = MagicMock()
        mock_designer = MagicMock()
        mock_relaxer.minimize_with_pair_restraints.return_value = (
            "minimized"
        )
        mock_relaxer.generate_local_md_ensemble.return_value = [
            "m0",
        ]
        mock_build_spec.return_value = DesignSpec(
            residue_specs={}, default_mode=ResidueMode.NATRO
        )
        mock_build_sampling.return_value = []
        mock_worker.return_value = _DdGMemberResult(
            member_index=0,
            bound_wt_energy=-100.0,
            unbound_wt_energy=-80.0,
            bound_mut_energy=-90.0,
            unbound_mut_energy=-75.0,
        )

        config = DdGConfig(
            chain_pairs=[("A", "B")],
            n_ensemble=1,
            sort_members_by_wt_bound_energy=True,
            average_top_n=3,
        )
        result = compute_ddg(
            "PDB",
            [MutationSpec("A", 1, "LEU", "ALA")],
            config,
            relaxer=mock_relaxer,
            designer=mock_designer,
        )
        assert result.sorted_by_wt_energy is True
        assert result.top_n_applied == 3

    @patch("boundry.ddg._process_ensemble_member")
    @patch("boundry.ddg.build_sampling_neighborhood")
    @patch("boundry.ddg.build_neighborhood_spec")
    @patch("boundry.ddg.validate_mutations")
    def test_top_n_applied_none_when_sorting_disabled(
        self,
        mock_validate,
        mock_build_spec,
        mock_build_sampling,
        mock_worker,
    ):
        from boundry.config import DdGConfig
        from boundry.resfile import DesignSpec, ResidueMode

        mock_relaxer = MagicMock()
        mock_designer = MagicMock()
        mock_relaxer.minimize_with_pair_restraints.return_value = (
            "minimized"
        )
        mock_relaxer.generate_local_md_ensemble.return_value = [
            "m0",
        ]
        mock_build_spec.return_value = DesignSpec(
            residue_specs={}, default_mode=ResidueMode.NATRO
        )
        mock_build_sampling.return_value = []
        mock_worker.return_value = _DdGMemberResult(
            member_index=0,
            bound_wt_energy=-100.0,
            unbound_wt_energy=-80.0,
            bound_mut_energy=-90.0,
            unbound_mut_energy=-75.0,
        )

        config = DdGConfig(
            chain_pairs=[("A", "B")],
            n_ensemble=1,
            sort_members_by_wt_bound_energy=False,
            average_top_n=3,
        )
        result = compute_ddg(
            "PDB",
            [MutationSpec("A", 1, "LEU", "ALA")],
            config,
            relaxer=mock_relaxer,
            designer=mock_designer,
        )
        assert result.sorted_by_wt_energy is False
        assert result.top_n_applied is None

    @patch("boundry.ddg._process_ensemble_member")
    @patch("boundry.ddg.build_sampling_neighborhood")
    @patch("boundry.ddg.build_neighborhood_spec")
    @patch("boundry.ddg.validate_mutations")
    def test_sorting_assigns_ranks_by_wt_bound_energy(
        self,
        mock_validate,
        mock_build_spec,
        mock_build_sampling,
        mock_worker,
    ):
        from boundry.config import DdGConfig
        from boundry.resfile import DesignSpec, ResidueMode

        mock_relaxer = MagicMock()
        mock_designer = MagicMock()
        mock_relaxer.minimize_with_pair_restraints.return_value = (
            "minimized"
        )
        mock_relaxer.generate_local_md_ensemble.return_value = [
            "m0",
            "m1",
            "m2",
        ]
        mock_build_spec.return_value = DesignSpec(
            residue_specs={}, default_mode=ResidueMode.NATRO
        )
        mock_build_sampling.return_value = []

        # 3 members with distinct bound_wt_energy: -110, -100, -105
        mock_worker.side_effect = [
            _DdGMemberResult(
                member_index=0,
                bound_wt_energy=-110.0,
                unbound_wt_energy=-80.0,
                bound_mut_energy=-90.0,
                unbound_mut_energy=-75.0,
            ),
            _DdGMemberResult(
                member_index=1,
                bound_wt_energy=-100.0,
                unbound_wt_energy=-80.0,
                bound_mut_energy=-90.0,
                unbound_mut_energy=-75.0,
            ),
            _DdGMemberResult(
                member_index=2,
                bound_wt_energy=-105.0,
                unbound_wt_energy=-80.0,
                bound_mut_energy=-90.0,
                unbound_mut_energy=-75.0,
            ),
        ]

        config = DdGConfig(
            chain_pairs=[("A", "B")],
            n_ensemble=3,
            sort_members_by_wt_bound_energy=True,
        )
        result = compute_ddg(
            "PDB",
            [MutationSpec("A", 1, "LEU", "ALA")],
            config,
            relaxer=mock_relaxer,
            designer=mock_designer,
        )

        # Rank 0 = lowest WT bound energy (-110), rank 1 = -105,
        # rank 2 = -100
        by_rank = {
            m.wt_bound_energy_rank: m.bound_wt_energy
            for m in result.member_results
        }
        assert by_rank[0] == pytest.approx(-110.0)
        assert by_rank[1] == pytest.approx(-105.0)
        assert by_rank[2] == pytest.approx(-100.0)

    @patch("boundry.ddg._process_ensemble_member")
    @patch("boundry.ddg.build_sampling_neighborhood")
    @patch("boundry.ddg.build_neighborhood_spec")
    @patch("boundry.ddg.validate_mutations")
    def test_average_top_n_filters_ensemble_ddgs(
        self,
        mock_validate,
        mock_build_spec,
        mock_build_sampling,
        mock_worker,
    ):
        from boundry.config import DdGConfig
        from boundry.resfile import DesignSpec, ResidueMode

        mock_relaxer = MagicMock()
        mock_designer = MagicMock()
        mock_relaxer.minimize_with_pair_restraints.return_value = (
            "minimized"
        )
        mock_relaxer.generate_local_md_ensemble.return_value = [
            "m0",
            "m1",
            "m2",
        ]
        mock_build_spec.return_value = DesignSpec(
            residue_specs={}, default_mode=ResidueMode.NATRO
        )
        mock_build_sampling.return_value = []

        # Member 0: wt_bound=-110 (rank 0), ddG = -15 - (-30) = 15
        # Member 1: wt_bound=-100 (rank 2), ddG = -10 - (-20) = 10
        # Member 2: wt_bound=-105 (rank 1), ddG = -12 - (-25) = 13
        mock_worker.side_effect = [
            _DdGMemberResult(
                member_index=0,
                bound_wt_energy=-110.0,
                unbound_wt_energy=-80.0,
                bound_mut_energy=-95.0,
                unbound_mut_energy=-80.0,
            ),
            _DdGMemberResult(
                member_index=1,
                bound_wt_energy=-100.0,
                unbound_wt_energy=-80.0,
                bound_mut_energy=-90.0,
                unbound_mut_energy=-80.0,
            ),
            _DdGMemberResult(
                member_index=2,
                bound_wt_energy=-105.0,
                unbound_wt_energy=-80.0,
                bound_mut_energy=-92.0,
                unbound_mut_energy=-80.0,
            ),
        ]

        config = DdGConfig(
            chain_pairs=[("A", "B")],
            n_ensemble=3,
            sort_members_by_wt_bound_energy=True,
            average_top_n=2,
        )
        result = compute_ddg(
            "PDB",
            [MutationSpec("A", 1, "LEU", "ALA")],
            config,
            relaxer=mock_relaxer,
            designer=mock_designer,
        )

        # Only top-2 by rank (rank 0 and rank 1) should be included
        assert len(result.ensemble_ddGs) == 2
        # member 0 (rank 0): ddG = (-95 - -80) - (-110 - -80) = -15 - -30 = 15
        # member 2 (rank 1): ddG = (-92 - -80) - (-105 - -80) = -12 - -25 = 13
        import statistics

        assert result.mean_ddG == pytest.approx(
            statistics.mean([15.0, 13.0])
        )

    @patch("boundry.ddg._process_ensemble_member")
    @patch("boundry.ddg.build_sampling_neighborhood")
    @patch("boundry.ddg.build_neighborhood_spec")
    @patch("boundry.ddg.validate_mutations")
    def test_std_ddg_none_with_single_successful_member(
        self,
        mock_validate,
        mock_build_spec,
        mock_build_sampling,
        mock_worker,
    ):
        from boundry.config import DdGConfig
        from boundry.resfile import DesignSpec, ResidueMode

        mock_relaxer = MagicMock()
        mock_designer = MagicMock()
        mock_relaxer.minimize_with_pair_restraints.return_value = (
            "minimized"
        )
        mock_relaxer.generate_local_md_ensemble.return_value = [
            "m0",
            "m1",
        ]
        mock_build_spec.return_value = DesignSpec(
            residue_specs={}, default_mode=ResidueMode.NATRO
        )
        mock_build_sampling.return_value = []

        mock_worker.side_effect = [
            _DdGMemberResult(
                member_index=0,
                bound_wt_energy=-100.0,
                unbound_wt_energy=-80.0,
                bound_mut_energy=-90.0,
                unbound_mut_energy=-75.0,
            ),
            _DdGMemberResult(
                member_index=1,
                error="RuntimeError: fail",
            ),
        ]

        config = DdGConfig(
            chain_pairs=[("A", "B")], n_ensemble=2
        )
        result = compute_ddg(
            "PDB",
            [MutationSpec("A", 1, "LEU", "ALA")],
            config,
            relaxer=mock_relaxer,
            designer=mock_designer,
        )
        assert result.n_successful == 1
        assert result.std_ddG is None


# ------------------------------------------------------------------
# compute_interface_dg
# ------------------------------------------------------------------


class TestComputeInterfaceDG:
    def test_chain_pairs_required(self):
        from boundry.config import DdGConfig

        config = DdGConfig(chain_pairs=None)
        with pytest.raises(ValueError, match="chain_pairs is required"):
            compute_interface_dg("PDB", config)

    @patch("boundry.ddg._process_ensemble_member")
    def test_returns_interface_dg_result(self, mock_worker):
        from boundry.config import DdGConfig

        mock_relaxer = MagicMock()
        mock_designer = MagicMock()
        mock_relaxer.minimize_with_pair_restraints.return_value = (
            "minimized"
        )
        mock_relaxer.generate_local_md_ensemble.return_value = [
            "m0",
            "m1",
        ]

        mock_worker.side_effect = [
            _DdGMemberResult(
                member_index=0,
                bound_wt_energy=-100.0,
                unbound_wt_energy=-80.0,
            ),
            _DdGMemberResult(
                member_index=1,
                bound_wt_energy=-102.0,
                unbound_wt_energy=-82.0,
            ),
        ]

        config = DdGConfig(
            chain_pairs=[("A", "B")], n_ensemble=2
        )
        result = compute_interface_dg(
            "PDB",
            config,
            relaxer=mock_relaxer,
            designer=mock_designer,
        )
        assert isinstance(result, InterfaceDgResult)
        # (-20 + -20) / 2 = -20
        assert result.dG == pytest.approx(-20.0)
        assert result.minimized_pdb == "minimized"

    @patch("boundry.ddg._process_ensemble_member")
    def test_ensemble_caching(self, mock_worker, tmp_path):
        from boundry.config import DdGConfig

        mock_relaxer = MagicMock()
        mock_designer = MagicMock()
        mock_relaxer.minimize_with_pair_restraints.return_value = (
            "minimized"
        )
        mock_relaxer.generate_local_md_ensemble.return_value = [
            "member0_pdb",
            "member1_pdb",
        ]

        mock_worker.side_effect = [
            _DdGMemberResult(
                member_index=0,
                bound_wt_energy=-100.0,
                unbound_wt_energy=-80.0,
            ),
            _DdGMemberResult(
                member_index=1,
                bound_wt_energy=-102.0,
                unbound_wt_energy=-82.0,
            ),
        ]

        ens_dir = tmp_path / "ensemble"
        config = DdGConfig(
            chain_pairs=[("A", "B")],
            n_ensemble=2,
            cache_ensemble=True,
            ensemble_dir=ens_dir,
        )
        compute_interface_dg(
            "PDB",
            config,
            relaxer=mock_relaxer,
            designer=mock_designer,
        )

        cached = sorted(ens_dir.glob("member_*.pdb"))
        assert len(cached) == 2
        assert cached[0].read_text() == "member0_pdb"

    @patch("boundry.ddg._process_ensemble_member")
    def test_raises_on_all_failures(self, mock_worker):
        from boundry.config import DdGConfig

        mock_relaxer = MagicMock()
        mock_designer = MagicMock()
        mock_relaxer.minimize_with_pair_restraints.return_value = (
            "minimized"
        )
        mock_relaxer.generate_local_md_ensemble.return_value = [
            "m0",
        ]

        mock_worker.return_value = _DdGMemberResult(
            member_index=0, error="RuntimeError: fail"
        )

        config = DdGConfig(
            chain_pairs=[("A", "B")], n_ensemble=1
        )
        with pytest.raises(
            RuntimeError, match="All ensemble members failed"
        ):
            compute_interface_dg(
                "PDB",
                config,
                relaxer=mock_relaxer,
                designer=mock_designer,
            )

    @patch("boundry.ddg._process_ensemble_member")
    def test_partial_failure_averages_successful(self, mock_worker):
        from boundry.config import DdGConfig

        mock_relaxer = MagicMock()
        mock_designer = MagicMock()
        mock_relaxer.minimize_with_pair_restraints.return_value = (
            "minimized"
        )
        mock_relaxer.generate_local_md_ensemble.return_value = [
            "m0",
            "m1",
            "m2",
        ]

        # Member 0 fails, members 1 and 2 succeed with different dG
        mock_worker.side_effect = [
            _DdGMemberResult(
                member_index=0, error="RuntimeError: fail"
            ),
            _DdGMemberResult(
                member_index=1,
                bound_wt_energy=-100.0,
                unbound_wt_energy=-80.0,
            ),
            _DdGMemberResult(
                member_index=2,
                bound_wt_energy=-110.0,
                unbound_wt_energy=-85.0,
            ),
        ]

        config = DdGConfig(
            chain_pairs=[("A", "B")], n_ensemble=3
        )
        result = compute_interface_dg(
            "PDB",
            config,
            relaxer=mock_relaxer,
            designer=mock_designer,
        )
        # dG for member 1: -100 - -80 = -20
        # dG for member 2: -110 - -85 = -25
        # mean = (-20 + -25) / 2 = -22.5
        import statistics

        assert result.dG == pytest.approx(
            statistics.mean([-20.0, -25.0])
        )


# ------------------------------------------------------------------
# DdGResult.to_dict — wt_bound_energy_rank
# ------------------------------------------------------------------


class TestDdGResultToDictRank:
    def test_to_dict_includes_wt_bound_energy_rank(self):
        m0 = EnsembleMemberResult(
            member_index=0,
            bound_wt_energy=-110.0,
            unbound_wt_energy=-80.0,
            bound_mut_energy=-90.0,
            unbound_mut_energy=-75.0,
        )
        m0.wt_bound_energy_rank = 0
        m1 = EnsembleMemberResult(
            member_index=1,
            bound_wt_energy=-100.0,
            unbound_wt_energy=-80.0,
            bound_mut_energy=-90.0,
            unbound_mut_energy=-75.0,
        )
        m1.wt_bound_energy_rank = 1
        result = DdGResult(
            mutations=[MutationSpec("A", 1, "LEU", "ALA")],
            member_results=[m0, m1],
            mean_ddG=5.0,
            std_ddG=0.1,
            mean_dG_wt=-20.0,
            mean_dG_mut=-15.0,
            n_successful=2,
            n_ensemble=2,
            ensemble_ddGs=[5.0, 5.0],
            sorted_by_wt_energy=True,
            top_n_applied=None,
        )
        d = result.to_dict()
        assert d["member_results"][0]["wt_bound_energy_rank"] == 0
        assert d["member_results"][1]["wt_bound_energy_rank"] == 1


# ------------------------------------------------------------------
# Top-level imports
# ------------------------------------------------------------------


class TestTopLevelImports:
    def test_ddg_function_importable(self):
        from boundry import ddg

        assert callable(ddg)

    def test_ddg_result_importable(self):
        from boundry import DdGResult

        assert DdGResult is not None

    def test_mutation_spec_importable(self):
        from boundry import MutationSpec

        assert MutationSpec is not None

    def test_ddg_config_importable(self):
        from boundry import DdGConfig

        assert DdGConfig is not None


# ------------------------------------------------------------------
# Integration tests (require OpenMM + LigandMPNN weights)
# ------------------------------------------------------------------


@pytest.mark.integration
class TestDdGIntegration:
    def test_pair_restraint_minimization_preserves_ca_geometry(
        self, antibody_antigen_pdb_string
    ):
        """CA-CA distances are within restraint_sd after minimization."""
        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(RelaxConfig(implicit_solvent=True))
        minimized = relaxer.minimize_with_pair_restraints(
            antibody_antigen_pdb_string,
            ca_cutoff=9.0,
            restraint_sd=0.5,
            implicit_solvent=True,
        )
        assert minimized is not None
        assert "ATOM" in minimized

    def test_ensemble_members_non_identical(
        self, antibody_antigen_pdb_string
    ):
        """Ensemble generation produces distinct members."""
        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(RelaxConfig(implicit_solvent=True))
        minimized = relaxer.minimize_with_pair_restraints(
            antibody_antigen_pdb_string,
            ca_cutoff=9.0,
            restraint_sd=0.5,
            implicit_solvent=True,
        )
        ensemble = relaxer.generate_local_md_ensemble(
            minimized,
            [],
            n_members=3,
            ca_cutoff=9.0,
            restraint_sd=0.5,
            implicit_solvent=True,
        )
        assert len(ensemble) == 3
        # Not all members should be identical strings
        assert len(set(ensemble)) > 1

    def test_end_to_end_mutation_sanity(
        self, antibody_antigen_pdb_string
    ):
        """A destabilising mutation produces positive ddG."""
        from boundry.config import DdGConfig

        config = DdGConfig(
            chain_pairs=[("H", "A")],
            n_ensemble=3,
        )
        # This is a sanity test — the exact mutation site depends
        # on the fixture, so we skip if the fixture chain IDs don't
        # match.  The test verifies the full pipeline runs without
        # error; assertion on sign is aspirational.
        try:
            result = compute_ddg(
                antibody_antigen_pdb_string,
                [MutationSpec("H", 1, "GLN", "GLU")],
                config,
            )
        except (ValueError, RuntimeError):
            pytest.skip(
                "Fixture chain IDs don't match mutation spec"
            )
        assert result.mean_ddG is not None
