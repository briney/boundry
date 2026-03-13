"""Residue-map and sequence helpers extracted from workflow_metadata.

These utilities are used by ``optimize.py`` (and potentially other modules)
for PDB-level residue identity extraction and per-chain sequence derivation.
They have no workflow semantics.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Standard amino acid 3-letter → 1-letter mapping (matches LigandMPNN)
_RESTYPE_3TO1: Dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def extract_residue_map(
    pdb_string: str,
) -> List[List[Any]]:
    """Extract per-residue identity from CA atoms in a PDB string.

    Matches LigandMPNN's CA-driven sequence semantics. Returns a list
    of ``[chain, resnum, icode, one_letter_aa]`` lists (not tuples,
    for JSON serializability). Deduplicates by (chain, resnum, icode).
    """
    residue_map: List[List[Any]] = []
    seen: set = set()
    for line in pdb_string.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        chain = line[21]
        try:
            resnum = int(line[22:26].strip())
        except ValueError:
            continue
        icode = line[26] if len(line) > 26 else " "
        resname = line[17:20].strip()
        key = (chain, resnum, icode)
        if key in seen:
            continue
        seen.add(key)
        one_letter = _RESTYPE_3TO1.get(resname, "X")
        residue_map.append([chain, resnum, icode, one_letter])
    return residue_map


def _residue_map_to_sequences(
    residue_map: List[List[Any]],
) -> Dict[str, str]:
    """Group a residue map into per-chain sequence strings."""
    chains: Dict[str, List[str]] = {}
    for chain, _resnum, _icode, aa in residue_map:
        chains.setdefault(chain, []).append(aa)
    return {ch: "".join(aas) for ch, aas in chains.items()}
