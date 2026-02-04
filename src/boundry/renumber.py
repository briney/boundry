"""Renumber PDB residues to remove insertion codes.

PDB files with Kabat-numbered antibodies use insertion codes (e.g.,
100, 100A, 100B, 101).  The vendored OpenFold code rejects these,
blocking constrained minimization and related operations.  This module
provides helpers to transparently renumber residues sequentially and
restore the original numbering afterward.

PDB fixed-width layout:
  - Columns 23-26 (0-indexed 22-25): residue sequence number (right-justified int)
  - Column 27 (0-indexed 26): insertion code (single character or space)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# PDB record types that carry residue numbering
_COORD_PREFIXES = ("ATOM  ", "HETATM", "TER   ")


def has_insertion_codes(pdb_string: str) -> bool:
    """Check whether *pdb_string* contains residue insertion codes.

    Scans ATOM/HETATM/TER lines for a non-space character at column 27
    (0-indexed position 26).
    """
    for line in pdb_string.splitlines():
        if len(line) >= 27 and line[:6] in _COORD_PREFIXES:
            if line[26] != " ":
                return True
    return False


@dataclass
class RenumberMapping:
    """Bidirectional mapping between sequential and original numbering.

    Attributes:
        forward: Maps ``(chain_id, sequential_num)`` to
            ``(chain_id, orig_resnum, orig_icode)``.
        reverse: Maps ``(chain_id, orig_resnum, orig_icode)`` to
            ``(chain_id, sequential_num)``.
    """

    forward: Dict[Tuple[str, int], Tuple[str, int, str]] = field(
        default_factory=dict
    )
    reverse: Dict[Tuple[str, int, str], Tuple[str, int]] = field(
        default_factory=dict
    )


def renumber_pdb(
    pdb_string: str,
) -> Tuple[str, RenumberMapping]:
    """Renumber residues sequentially per chain, removing insertion codes.

    Each chain's residues are numbered starting from 1.  The mapping
    between original and sequential numbering is returned so that the
    original numbering can be restored later.

    Args:
        pdb_string: PDB-format string (may contain insertion codes).

    Returns:
        A tuple of *(renumbered_pdb_string, mapping)*.
    """
    mapping = RenumberMapping()

    # First pass: discover unique (chain, resnum, icode) tuples in order
    chain_residues: Dict[str, List[Tuple[int, str]]] = {}
    for line in pdb_string.splitlines():
        if len(line) < 27:
            continue
        if line[:6] not in _COORD_PREFIXES:
            continue
        chain_id = line[21]
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        icode = line[26]

        key = (resnum, icode)
        if chain_id not in chain_residues:
            chain_residues[chain_id] = []
        if not chain_residues[chain_id] or chain_residues[chain_id][-1] != key:
            # Only append if it's a new residue (avoid duplicates from
            # multiple atoms in the same residue)
            if key not in chain_residues[chain_id]:
                chain_residues[chain_id].append(key)

    # Build the mapping: sequential number for each (chain, resnum, icode)
    chain_seq_map: Dict[str, Dict[Tuple[int, str], int]] = {}
    for chain_id, residues in chain_residues.items():
        chain_seq_map[chain_id] = {}
        for seq_num, (orig_resnum, orig_icode) in enumerate(residues, 1):
            chain_seq_map[chain_id][(orig_resnum, orig_icode)] = seq_num
            mapping.forward[(chain_id, seq_num)] = (
                chain_id,
                orig_resnum,
                orig_icode,
            )
            mapping.reverse[(chain_id, orig_resnum, orig_icode)] = (
                chain_id,
                seq_num,
            )

    # Second pass: rewrite the PDB lines
    out_lines = []
    for line in pdb_string.splitlines(True):
        stripped = line.rstrip("\n").rstrip("\r")
        if len(stripped) >= 27 and stripped[:6] in _COORD_PREFIXES:
            chain_id = stripped[21]
            try:
                resnum = int(stripped[22:26])
            except ValueError:
                out_lines.append(line)
                continue
            icode = stripped[26]

            lookup = chain_seq_map.get(chain_id)
            if lookup is not None:
                new_num = lookup.get((resnum, icode))
                if new_num is not None:
                    # Pad line to at least 27 chars if needed
                    padded = stripped.ljust(80)
                    new_resnum_str = f"{new_num:>4}"
                    new_line = (
                        padded[:22]
                        + new_resnum_str
                        + " "  # clear insertion code
                        + padded[27:]
                    )
                    # Preserve original line ending
                    ending = line[len(stripped):]
                    out_lines.append(new_line.rstrip() + ending)
                    continue

        out_lines.append(line)

    return "".join(out_lines), mapping


def restore_numbering(
    pdb_string: str,
    mapping: RenumberMapping,
) -> str:
    """Restore original residue numbering using a :class:`RenumberMapping`.

    Args:
        pdb_string: PDB-format string with sequential numbering.
        mapping: Mapping produced by :func:`renumber_pdb`.

    Returns:
        PDB string with original residue numbers and insertion codes.
    """
    out_lines = []
    for line in pdb_string.splitlines(True):
        stripped = line.rstrip("\n").rstrip("\r")
        if len(stripped) >= 27 and stripped[:6] in _COORD_PREFIXES:
            chain_id = stripped[21]
            try:
                resnum = int(stripped[22:26])
            except ValueError:
                out_lines.append(line)
                continue

            entry = mapping.forward.get((chain_id, resnum))
            if entry is not None:
                _, orig_resnum, orig_icode = entry
                padded = stripped.ljust(80)
                new_resnum_str = f"{orig_resnum:>4}"
                new_line = (
                    padded[:22]
                    + new_resnum_str
                    + orig_icode
                    + padded[27:]
                )
                ending = line[len(stripped):]
                out_lines.append(new_line.rstrip() + ending)
                continue

        out_lines.append(line)

    return "".join(out_lines)
