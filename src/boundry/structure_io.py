"""Unified structure I/O utilities for PDB and CIF formats."""

import io
import logging
import tempfile
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

from Bio.PDB import MMCIFIO, PDBIO, MMCIFParser, PDBParser

logger = logging.getLogger(__name__)

# Type alias: mapping from PDB single-char chain ID to original CIF chain ID
ChainIdMapping = Dict[str, str]

# Pool of single-character chain IDs available for PDB format
_ALL_SINGLE_CHAR_IDS: List[str] = (
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abcdefghijklmnopqrstuvwxyz")
    + list("0123456789")
)


class StructureFormat(Enum):
    """Supported structure file formats."""

    PDB = "pdb"
    CIF = "cif"


def detect_format(path: Path) -> StructureFormat:
    """
    Auto-detect structure format from file extension.

    Args:
        path: Path to structure file

    Returns:
        StructureFormat enum value

    Raises:
        ValueError: If extension is not recognized
    """
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        return StructureFormat.PDB
    elif suffix in (".cif", ".mmcif"):
        return StructureFormat.CIF
    else:
        raise ValueError(
            f"Unknown structure format for extension '{suffix}'. "
            "Supported: .pdb, .cif, .mmcif"
        )


def read_structure(path: Path) -> str:
    """
    Read structure file contents as string.

    Args:
        path: Path to structure file

    Returns:
        File contents as string
    """
    with open(path) as f:
        return f.read()


def write_structure(content: str, path: Path, fmt: StructureFormat = None):
    """
    Write structure string to file.

    Args:
        content: Structure content as string
        path: Output file path
        fmt: Target format (auto-detected from path if not provided)
    """
    if fmt is None:
        fmt = detect_format(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    logger.info(f"Saved structure to {path}")


def _needs_chain_remapping(structure) -> bool:
    """Check if any chain in *structure* has a multi-character ID."""
    for chain in structure.get_chains():
        if len(chain.id) > 1:
            return True
    return False


def _remap_chain_ids(structure) -> ChainIdMapping:
    """In-place reassign multi-char chain IDs to single characters.

    Single-character chains keep their IDs. Multi-character chains are
    assigned from the pool of available single-character IDs.

    Returns a mapping ``{new_pdb_id: original_cif_id}``.

    Raises:
        ValueError: If there are more than 62 chains.
    """
    chains = list(structure.get_chains())
    if len(chains) > len(_ALL_SINGLE_CHAR_IDS):
        raise ValueError(
            f"Structure has {len(chains)} chains but PDB format "
            f"supports at most {len(_ALL_SINGLE_CHAR_IDS)}. "
            f"Cannot remap chain IDs."
        )

    # Collect existing single-char IDs that must be preserved
    existing_single = {c.id for c in chains if len(c.id) == 1}

    # Build pool of available single-char IDs
    available = [
        cid for cid in _ALL_SINGLE_CHAR_IDS if cid not in existing_single
    ]

    mapping: ChainIdMapping = {}
    pool_idx = 0

    for chain in chains:
        if len(chain.id) > 1:
            new_id = available[pool_idx]
            pool_idx += 1
            mapping[new_id] = chain.id
            chain.id = new_id

    return mapping


def restore_cif_chain_ids(
    structure, mapping: ChainIdMapping
) -> None:
    """In-place restore original CIF chain IDs on a Biopython Structure."""
    for chain in structure.get_chains():
        if chain.id in mapping:
            chain.id = mapping[chain.id]


def convert_pdb_to_cif(
    pdb_string: str,
    chain_id_mapping: ChainIdMapping = None,
) -> str:
    """
    Convert PDB format string to CIF format string.

    Args:
        pdb_string: Structure in PDB format
        chain_id_mapping: Optional mapping from PDB chain IDs to
            original CIF chain IDs.  When provided, chain IDs are
            restored before writing CIF output.

    Returns:
        Structure in CIF format
    """
    parser = PDBParser(QUIET=True)
    handle = io.StringIO(pdb_string)
    structure = parser.get_structure("structure", handle)

    if chain_id_mapping:
        restore_cif_chain_ids(structure, chain_id_mapping)

    cif_io = MMCIFIO()
    cif_io.set_structure(structure)

    output = io.StringIO()
    cif_io.save(output)
    return output.getvalue()


def convert_cif_to_pdb(
    cif_string: str,
) -> Tuple[str, ChainIdMapping]:
    """
    Convert CIF format string to PDB format string.

    Multi-character chain IDs (common in AlphaFold3 predictions) are
    remapped to single characters for PDB compatibility.

    Args:
        cif_string: Structure in CIF format

    Returns:
        Tuple of (PDB format string, chain ID mapping).  The mapping
        is ``{pdb_id: original_cif_id}`` and is empty when no
        remapping was needed.
    """
    # MMCIFParser requires a file path, so use temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cif", delete=False
    ) as tmp:
        tmp.write(cif_string)
        tmp_path = tmp.name

    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("structure", tmp_path)

        mapping: ChainIdMapping = {}
        if _needs_chain_remapping(structure):
            mapping = _remap_chain_ids(structure)

        pdb_io = PDBIO()
        pdb_io.set_structure(structure)

        output = io.StringIO()
        pdb_io.save(output)
        return output.getvalue(), mapping
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def ensure_pdb_format(
    content: str, source_path: Path
) -> Tuple[str, ChainIdMapping]:
    """
    Ensure structure content is in PDB format.

    If source is CIF, converts to PDB (remapping multi-character
    chain IDs if necessary). Otherwise returns as-is with an empty
    mapping.

    Args:
        content: Structure content string
        source_path: Original file path (for format detection)

    Returns:
        Tuple of (PDB format string, chain ID mapping).
    """
    fmt = detect_format(source_path)
    if fmt == StructureFormat.CIF:
        logger.debug("Converting CIF to PDB for internal processing")
        return convert_cif_to_pdb(content)
    return content, {}


def convert_to_format(
    pdb_string: str,
    target_format: StructureFormat,
    chain_id_mapping: ChainIdMapping = None,
) -> str:
    """
    Convert PDB string to target format.

    Args:
        pdb_string: Structure in PDB format
        target_format: Desired output format
        chain_id_mapping: Optional mapping to restore original CIF
            chain IDs when converting to CIF format.

    Returns:
        Structure in target format
    """
    if target_format == StructureFormat.PDB:
        return pdb_string
    elif target_format == StructureFormat.CIF:
        return convert_pdb_to_cif(
            pdb_string, chain_id_mapping=chain_id_mapping
        )
    else:
        raise ValueError(f"Unknown target format: {target_format}")


def get_output_format(
    input_path: Path,
    output_path: Path,
) -> StructureFormat:
    """
    Determine output format based on output path extension.

    Falls back to input format if output path has no recognized extension.

    Args:
        input_path: Input file path
        output_path: Output file path

    Returns:
        StructureFormat for output
    """
    # Try to detect from output path extension
    try:
        return detect_format(output_path)
    except ValueError:
        pass

    # Fall back to input format
    return detect_format(input_path)
