"""Tests for boundry.workflow_metadata module."""

from __future__ import annotations

import copy

import pytest

from boundry.workflow_metadata import (
    WORKFLOW_NAMESPACE,
    _residue_map_to_sequences,
    extract_residue_map,
    merge_metadata,
    update_design_history,
)


# ------------------------------------------------------------------
# PDB helpers
# ------------------------------------------------------------------


def _atom_line(
    serial, name, resname, chain, resnum, x, y, z, icode=" "
):
    """Build a PDB ATOM line."""
    return (
        f"ATOM  {serial:5d} {name:<4s} {resname:>3s} "
        f"{chain}{resnum:4d}{icode}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
        f"           {name[0]}"
    )


def _make_pdb(residues, chain="A"):
    """Build a PDB string from a list of (resnum, resname3) tuples.

    Each residue gets N, CA, C atoms.
    """
    lines = []
    serial = 1
    for resnum, resname in residues:
        for atom in ("N", "CA", "C"):
            lines.append(
                _atom_line(
                    serial,
                    atom,
                    resname,
                    chain,
                    resnum,
                    float(serial),
                    0.0,
                    0.0,
                )
            )
            serial += 1
    lines.append("END")
    return "\n".join(lines)


def _make_multichain_pdb(chains_residues):
    """Build a multi-chain PDB.

    chains_residues: list of (chain_id, [(resnum, resname3), ...])
    """
    lines = []
    serial = 1
    for chain, residues in chains_residues:
        for resnum, resname in residues:
            for atom in ("N", "CA", "C"):
                lines.append(
                    _atom_line(
                        serial,
                        atom,
                        resname,
                        chain,
                        resnum,
                        float(serial),
                        0.0,
                        0.0,
                    )
                )
                serial += 1
    lines.append("END")
    return "\n".join(lines)


def _make_history(pdb_string, mutations=None):
    """Build a metadata dict with initialized design_history."""
    rmap = extract_residue_map(pdb_string)
    seqs = _residue_map_to_sequences(rmap)
    return {
        WORKFLOW_NAMESPACE: {
            "design_history": {
                "original_sequences": dict(seqs),
                "original_residue_map": rmap,
                "mutations": list(mutations or []),
                "current_sequences": dict(seqs),
                "total_mutations": len(mutations or []),
                "sequence_recovery": (
                    1.0
                    if not mutations
                    else round(
                        1.0 - len(mutations) / len(rmap), 6
                    )
                ),
            }
        }
    }


# ------------------------------------------------------------------
# extract_residue_map tests
# ------------------------------------------------------------------


class TestExtractResidueMap:
    def test_single_chain(self):
        pdb = _make_pdb(
            [(1, "ALA"), (2, "GLY"), (3, "VAL")], chain="A"
        )
        result = extract_residue_map(pdb)
        assert len(result) == 3
        assert result[0] == ["A", 1, " ", "A"]
        assert result[1] == ["A", 2, " ", "G"]
        assert result[2] == ["A", 3, " ", "V"]

    def test_multi_chain(self):
        pdb = _make_multichain_pdb(
            [
                ("A", [(1, "MET"), (2, "LYS")]),
                ("B", [(1, "GLY"), (2, "PRO")]),
            ]
        )
        result = extract_residue_map(pdb)
        assert len(result) == 4
        assert result[0] == ["A", 1, " ", "M"]
        assert result[1] == ["A", 2, " ", "K"]
        assert result[2] == ["B", 1, " ", "G"]
        assert result[3] == ["B", 2, " ", "P"]

    def test_with_insertion_codes(self):
        """Insertion codes should be preserved."""
        lines = [
            _atom_line(1, "CA", "ALA", "A", 1, 0, 0, 0, " "),
            _atom_line(2, "CA", "GLY", "A", 1, 1, 0, 0, "A"),
            _atom_line(3, "CA", "VAL", "A", 2, 2, 0, 0, " "),
            "END",
        ]
        pdb = "\n".join(lines)
        result = extract_residue_map(pdb)
        assert len(result) == 3
        assert result[0] == ["A", 1, " ", "A"]
        assert result[1] == ["A", 1, "A", "G"]
        assert result[2] == ["A", 2, " ", "V"]

    def test_skips_hetatm(self):
        """Only ATOM records should be included."""
        pdb = (
            _atom_line(1, "CA", "ALA", "A", 1, 0, 0, 0)
            + "\n"
            + "HETATM    2  CA  UNK A   2       "
            "1.000   0.000   0.000  1.00  0.00"
            "           C"
            + "\nEND"
        )
        result = extract_residue_map(pdb)
        assert len(result) == 1
        assert result[0][3] == "A"

    def test_unknown_residue(self):
        """Noncanonical residue names should map to 'X'."""
        pdb = _make_pdb([(1, "ALA"), (2, "UNK")], chain="A")
        result = extract_residue_map(pdb)
        assert result[0][3] == "A"
        assert result[1][3] == "X"

    def test_deduplicates_ca(self):
        """Multiple CA atoms for the same residue keep first."""
        lines = [
            _atom_line(1, "CA", "ALA", "A", 1, 0, 0, 0),
            _atom_line(2, "CA", "GLY", "A", 1, 1, 0, 0),
            "END",
        ]
        pdb = "\n".join(lines)
        result = extract_residue_map(pdb)
        assert len(result) == 1
        assert result[0][3] == "A"  # first CA wins


class TestResidueMapToSequences:
    def test_groups_by_chain(self):
        rmap = [
            ["A", 1, " ", "M"],
            ["A", 2, " ", "K"],
            ["B", 1, " ", "G"],
        ]
        result = _residue_map_to_sequences(rmap)
        assert result == {"A": "MK", "B": "G"}


# ------------------------------------------------------------------
# update_design_history tests
# ------------------------------------------------------------------


class TestUpdateDesignHistory:
    def test_no_mutations(self):
        """Identical sequence yields empty mutations list."""
        pdb = _make_pdb(
            [(1, "ALA"), (2, "GLY"), (3, "VAL")], chain="A"
        )
        meta = _make_history(pdb)
        result_meta = {"sequence": "AGV"}
        updated = update_design_history(
            meta, result_meta, "design", cycle=1
        )
        history = updated[WORKFLOW_NAMESPACE]["design_history"]
        assert history["mutations"] == []
        assert history["total_mutations"] == 0
        assert history["sequence_recovery"] == 1.0

    def test_single_mutation(self):
        """Detects one change with attribution."""
        pdb = _make_pdb(
            [(1, "ALA"), (2, "GLY"), (3, "VAL")], chain="A"
        )
        meta = _make_history(pdb)
        result_meta = {"sequence": "AGW"}  # V→W at pos 3
        updated = update_design_history(
            meta, result_meta, "design", cycle=2
        )
        history = updated[WORKFLOW_NAMESPACE]["design_history"]
        assert len(history["mutations"]) == 1
        mut = history["mutations"][0]
        assert mut["chain"] == "A"
        assert mut["position"] == 3
        assert mut["from_aa"] == "V"
        assert mut["to_aa"] == "W"
        assert mut["introduced_by"] == "design"
        assert mut["introduced_at_cycle"] == 2
        assert history["total_mutations"] == 1
        assert history["current_sequences"]["A"] == "AGW"

    def test_reversion(self):
        """Mutation removed when position reverts to original."""
        pdb = _make_pdb(
            [(1, "ALA"), (2, "GLY"), (3, "VAL")], chain="A"
        )
        # Start with existing mutation at pos 3
        meta = _make_history(
            pdb,
            mutations=[
                {
                    "chain": "A",
                    "position": 3,
                    "icode": " ",
                    "from_aa": "V",
                    "to_aa": "W",
                    "introduced_by": "design",
                    "introduced_at_cycle": 1,
                }
            ],
        )
        # Sequence reverts pos 3 back to V
        result_meta = {"sequence": "AGV"}
        updated = update_design_history(
            meta, result_meta, "mpnn", cycle=2
        )
        history = updated[WORKFLOW_NAMESPACE]["design_history"]
        assert history["mutations"] == []
        assert history["total_mutations"] == 0
        assert history["sequence_recovery"] == 1.0

    def test_multi_step_attribution(self):
        """First step recorded; re-mutation updates to_aa and
        last_modified."""
        pdb = _make_pdb(
            [(1, "ALA"), (2, "GLY"), (3, "VAL")], chain="A"
        )
        meta = _make_history(
            pdb,
            mutations=[
                {
                    "chain": "A",
                    "position": 3,
                    "icode": " ",
                    "from_aa": "V",
                    "to_aa": "W",
                    "introduced_by": "design",
                    "introduced_at_cycle": 1,
                }
            ],
        )
        # Re-mutate pos 3 from W to L
        result_meta = {"sequence": "AGL"}
        updated = update_design_history(
            meta, result_meta, "mpnn", cycle=3
        )
        history = updated[WORKFLOW_NAMESPACE]["design_history"]
        assert len(history["mutations"]) == 1
        mut = history["mutations"][0]
        assert mut["to_aa"] == "L"
        assert mut["introduced_by"] == "design"  # original
        assert mut["introduced_at_cycle"] == 1  # original
        assert mut["last_modified_by"] == "mpnn"
        assert mut["last_modified_at_cycle"] == 3

    def test_length_mismatch(self, caplog):
        """Warns and returns unchanged when lengths don't match."""
        pdb = _make_pdb(
            [(1, "ALA"), (2, "GLY")], chain="A"
        )
        meta = _make_history(pdb)
        result_meta = {"sequence": "AGVL"}  # too long
        import logging

        with caplog.at_level(logging.WARNING):
            updated = update_design_history(
                meta, result_meta, "design", cycle=1
            )
        assert "length mismatch" in caplog.text.lower()
        # Returned unchanged
        assert updated is meta

    def test_no_history_noop(self):
        """No crash when design_history not initialized."""
        meta = {"some_key": "value"}
        result_meta = {"sequence": "AGV"}
        updated = update_design_history(
            meta, result_meta, "design", cycle=1
        )
        assert updated is meta

    def test_no_sequence_noop(self):
        """No crash when result has no sequence."""
        pdb = _make_pdb([(1, "ALA")], chain="A")
        meta = _make_history(pdb)
        result_meta = {"some_other": "data"}
        updated = update_design_history(
            meta, result_meta, "design", cycle=1
        )
        assert updated is meta

    def test_does_not_mutate_input(self):
        """update_design_history returns a new dict."""
        pdb = _make_pdb(
            [(1, "ALA"), (2, "GLY")], chain="A"
        )
        meta = _make_history(pdb)
        original_meta = copy.deepcopy(meta)
        result_meta = {"sequence": "AW"}
        updated = update_design_history(
            meta, result_meta, "design", cycle=1
        )
        # Original should be unchanged
        assert meta == original_meta
        # Updated should differ
        assert (
            updated[WORKFLOW_NAMESPACE]["design_history"][
                "total_mutations"
            ]
            == 1
        )


# ------------------------------------------------------------------
# merge_metadata preserves design_history
# ------------------------------------------------------------------


class TestMergeMetadataDesignHistory:
    def test_preserves_design_history(self):
        """design_history survives merge with deep copy."""
        pdb = _make_pdb([(1, "ALA"), (2, "GLY")], chain="A")
        history_data = _make_history(pdb)[WORKFLOW_NAMESPACE][
            "design_history"
        ]
        previous = {
            WORKFLOW_NAMESPACE: {
                "design_history": history_data
            }
        }
        new_values = {"final_energy": -100.0}
        merged = merge_metadata(
            previous, new_values, operation="minimize"
        )
        assert (
            "design_history"
            in merged[WORKFLOW_NAMESPACE]
        )
        merged_history = merged[WORKFLOW_NAMESPACE][
            "design_history"
        ]
        assert (
            merged_history["original_sequences"]
            == history_data["original_sequences"]
        )

    def test_no_aliasing(self):
        """Modifying returned history doesn't affect original."""
        pdb = _make_pdb([(1, "ALA")], chain="A")
        history_data = _make_history(pdb)[WORKFLOW_NAMESPACE][
            "design_history"
        ]
        previous = {
            WORKFLOW_NAMESPACE: {
                "design_history": history_data
            }
        }
        new_values = {"energy": -50.0}
        merged = merge_metadata(
            previous, new_values, operation="relax"
        )

        # Mutate the merged copy
        merged[WORKFLOW_NAMESPACE]["design_history"][
            "mutations"
        ].append({"fake": True})

        # Original should be untouched
        assert history_data["mutations"] == []
