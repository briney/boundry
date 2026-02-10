"""Helpers for workflow metadata merging and metric resolution."""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

WORKFLOW_NAMESPACE = "_workflow"

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


def merge_metadata(
    previous: Dict[str, Any],
    new_values: Dict[str, Any],
    operation: Optional[str] = None,
) -> Dict[str, Any]:
    """Shallow-merge metadata and maintain workflow namespace helpers."""
    merged = dict(previous)
    merged.update(new_values)

    workflow_ns = dict(merged.get(WORKFLOW_NAMESPACE) or {})
    metrics = dict(workflow_ns.get("metrics") or {})
    state = dict(workflow_ns.get("state") or {})
    provenance = dict(workflow_ns.get("provenance") or {})

    _collect_numeric_values(new_values, prefix="", target=metrics)

    if operation is not None:
        state["last_operation"] = operation
        history = list(provenance.get("operations") or [])
        history.append(operation)
        provenance["operations"] = history

    workflow_ns["metrics"] = metrics
    workflow_ns["state"] = state
    workflow_ns["provenance"] = provenance

    # Carry forward design_history from previous metadata
    # (deepcopy to prevent aliasing with the original)
    prev_wf = previous.get(WORKFLOW_NAMESPACE)
    if isinstance(prev_wf, dict):
        prev_history = prev_wf.get("design_history")
        if prev_history is not None:
            workflow_ns["design_history"] = copy.deepcopy(
                prev_history
            )

    merged[WORKFLOW_NAMESPACE] = workflow_ns
    return merged


def extract_numeric_metric(
    metadata: Dict[str, Any], path: str
) -> Optional[float]:
    """Extract a numeric metric from metadata by dotted path."""
    value = resolve_path(metadata, path)
    if _is_numeric(value):
        return float(value)

    workflow_ns = metadata.get(WORKFLOW_NAMESPACE)
    if isinstance(workflow_ns, dict):
        metrics = workflow_ns.get("metrics")
        if isinstance(metrics, dict):
            alt = metrics.get(path)
            if _is_numeric(alt):
                return float(alt)
    return None


def resolve_path(root: Any, path: str) -> Any:
    """Resolve a dotted path through nested dicts and public attributes."""
    current = root
    for segment in path.split("."):
        if not segment or segment.startswith("_"):
            return None
        if isinstance(current, dict):
            if segment not in current:
                return None
            current = current[segment]
            continue
        if not hasattr(current, segment):
            return None
        current = getattr(current, segment)
    return current


def _collect_numeric_values(
    data: Any, prefix: str, target: Dict[str, float]
) -> None:
    if isinstance(data, dict):
        for key, value in data.items():
            if key.startswith("_"):
                continue
            path = f"{prefix}.{key}" if prefix else key
            _collect_numeric_values(value, path, target)
        return
    if _is_numeric(data) and prefix:
        target[prefix] = float(data)


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ------------------------------------------------------------------
# Design history / mutation tracking
# ------------------------------------------------------------------


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


def update_design_history(
    merged_metadata: Dict[str, Any],
    result_metadata: Dict[str, Any],
    operation: str,
    cycle: Optional[int] = None,
) -> Dict[str, Any]:
    """Update mutation tracking after a design/mpnn step.

    Parameters
    ----------
    merged_metadata : dict
        Metadata after ``merge_metadata()``, containing
        ``_workflow.design_history``.
    result_metadata : dict
        Pre-merge metadata from the design/mpnn operation (must
        contain ``sequence``).
    operation : str
        The operation name (``"design"`` or ``"mpnn"``).
    cycle : int, optional
        The iterate cycle or beam round number.

    Returns
    -------
    dict
        New metadata dict with updated design_history. Does not
        mutate the input.
    """
    wf = merged_metadata.get(WORKFLOW_NAMESPACE)
    if not isinstance(wf, dict):
        return merged_metadata
    history = wf.get("design_history")
    if history is None:
        return merged_metadata

    sequence = result_metadata.get("sequence")
    if sequence is None:
        return merged_metadata

    original_map = history.get("original_residue_map", [])
    if len(sequence) != len(original_map):
        logger.warning(
            "Design history length mismatch: sequence=%d, "
            "residue_map=%d. Skipping mutation tracking.",
            len(sequence),
            len(original_map),
        )
        return merged_metadata

    # Build lookup of existing mutations by position key
    existing_mutations = history.get("mutations", [])
    mut_by_pos: Dict[
        Tuple[str, int, str], Dict[str, Any]
    ] = {}
    for m in existing_mutations:
        key = (m["chain"], m["position"], m["icode"])
        mut_by_pos[key] = m

    new_mutations: List[Dict[str, Any]] = []
    for i, (chain, resnum, icode, orig_aa) in enumerate(
        original_map
    ):
        current_aa = sequence[i]
        key = (chain, resnum, icode)
        prev_mut = mut_by_pos.get(key)

        if current_aa != orig_aa:
            if prev_mut is None:
                # New mutation
                new_mutations.append(
                    {
                        "chain": chain,
                        "position": resnum,
                        "icode": icode,
                        "from_aa": orig_aa,
                        "to_aa": current_aa,
                        "introduced_by": operation,
                        "introduced_at_cycle": cycle,
                    }
                )
            elif prev_mut["to_aa"] != current_aa:
                # Re-mutation at same position
                updated = dict(prev_mut)
                updated["to_aa"] = current_aa
                updated["last_modified_by"] = operation
                updated["last_modified_at_cycle"] = cycle
                new_mutations.append(updated)
            else:
                # Same mutation, keep as-is
                new_mutations.append(prev_mut)
        # else: matches original → reversion, drop from list

    # Build updated sequences
    current_sequences = _residue_map_to_sequences(
        [
            [chain, resnum, icode, sequence[i]]
            for i, (chain, resnum, icode, _) in enumerate(
                original_map
            )
        ]
    )

    total = len(new_mutations)
    recovery = (
        1.0 - total / len(original_map)
        if original_map
        else 1.0
    )

    # Build new metadata (don't mutate in place)
    new_meta = dict(merged_metadata)
    new_wf = dict(wf)
    new_history = dict(history)
    new_history["mutations"] = new_mutations
    new_history["current_sequences"] = current_sequences
    new_history["total_mutations"] = total
    new_history["sequence_recovery"] = round(recovery, 6)
    new_wf["design_history"] = new_history
    new_meta[WORKFLOW_NAMESPACE] = new_wf
    return new_meta
