"""Surface area and shape complementarity calculations for interfaces."""

import io
import logging
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

from boundry.interface import InterfaceResidue, _get_protein_chains
from boundry.utils import filter_protein_only

logger = logging.getLogger(__name__)

try:  # Optional Rosetta-like SASA using FreeSASA
    import freesasa

    _HAS_FREESASA = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_FREESASA = False


@dataclass
class SurfaceAreaResult:
    """Surface area calculations for interface analysis."""

    complex_sasa: float = 0.0
    chain_sasa: Dict[str, float] = field(default_factory=dict)  # unbound chains
    buried_sasa: float = 0.0  # dSASA_int (Rosetta naming)
    interface_residue_sasa: Dict[str, float] = field(
        default_factory=dict
    )  # bound
    interface_residue_delta_sasa: Dict[str, float] = field(
        default_factory=dict
    )  # unbound - bound


@dataclass
class ShapeComplementarityResult:
    """Shape complementarity analysis."""

    sc_score: float = 0.0
    interface_area: float = 0.0
    interface_gap_volume: float = 0.0


def _compute_structure_sasa(
    structure,
    probe_radius: float = 1.4,
) -> float:
    """
    Compute total SASA for a BioPython structure.

    Args:
        structure: BioPython Structure object
        probe_radius: Water probe radius in angstroms

    Returns:
        Total SASA in angstroms squared
    """
    sr = ShrakeRupley(probe_radius=probe_radius)
    sr.compute(structure, level="S")
    return structure.sasa


def _compute_sasa_freesasa(
    pdb_string: str,
    probe_radius: float = 1.4,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Compute SASA using FreeSASA for closer alignment with Rosetta.

    Returns:
        total_sasa, chain_sasa, residue_sasa
    """
    with tempfile.NamedTemporaryFile(
        suffix=".pdb", delete=False, mode="w"
    ) as tmp:
        tmp.write(pdb_string)
        tmp_path = tmp.name

    try:
        structure = freesasa.Structure(tmp_path)
        parameters = freesasa.Parameters()
        parameters.setProbeRadius(probe_radius)
        result = freesasa.calc(structure, parameters)

        chain_sasa: Dict[str, float] = defaultdict(float)
        residue_sasa: Dict[str, float] = defaultdict(float)

        for i in range(structure.nAtoms()):
            area = result.atomArea(i)
            chain = structure.chainLabel(i)
            resnum = structure.residueNumber(i)
            _get_icode = getattr(
                structure, "residueInsertionCode", lambda *_: ""
            )
            icode = _get_icode(i)  # type: ignore
            key = f"{chain}{resnum}{icode}"
            chain_sasa[chain] += area
            residue_sasa[key] += area

        return result.totalArea(), dict(chain_sasa), dict(residue_sasa)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _compute_residue_sasa(
    structure,
    probe_radius: float = 1.4,
) -> Dict[str, float]:
    """
    Compute per-residue SASA for a BioPython structure.

    Args:
        structure: BioPython Structure object
        probe_radius: Water probe radius in angstroms

    Returns:
        Dictionary mapping "chain_resnum" -> SASA value
    """
    sr = ShrakeRupley(probe_radius=probe_radius)
    sr.compute(structure, level="R")
    residue_sasa = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                het_flag = residue.id[0]
                if het_flag != " ":
                    continue
                resnum = residue.id[1]
                icode = residue.id[2].strip()
                key = f"{chain.id}{resnum}{icode}"
                residue_sasa[key] = residue.sasa
        break  # Only first model
    return residue_sasa


def _extract_chain_pdb_string(
    pdb_string: str,
    chain_id: str,
    protein_only: bool = True,
) -> str:
    """
    Extract a single chain's PDB string from a multi-chain PDB.

    Args:
        pdb_string: Full PDB file contents
        chain_id: Chain ID to extract
        protein_only: If True, only include ATOM records (skip HETATM)

    Returns:
        PDB string for the specified chain with proper TER/END records
    """
    lines = []
    for line in pdb_string.splitlines():
        if line.startswith("ATOM"):
            if len(line) > 21 and line[21] == chain_id:
                lines.append(line)
        elif not protein_only and line.startswith("HETATM"):
            if len(line) > 21 and line[21] == chain_id:
                lines.append(line)
    if lines:
        lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _filter_protein_pdb_string(pdb_string: str) -> str:
    """
    Filter a PDB string to only include protein ATOM records.

    Thin wrapper around :func:`boundry.utils.filter_protein_only` for
    backwards compatibility within this module.

    Args:
        pdb_string: Full PDB file contents

    Returns:
        PDB string with only ATOM records and proper TER/END
    """
    return filter_protein_only(pdb_string)


def calculate_surface_area(
    pdb_string: str,
    interface_residues: List[InterfaceResidue],
    probe_radius: float = 1.4,
) -> SurfaceAreaResult:
    """
    Calculate solvent-accessible surface area (SASA) for interface analysis.

    Uses BioPython's SASA module (Shrake-Rupley algorithm) to compute:
    - Complex SASA (whole structure)
    - Per-chain SASA (each chain alone)
    - Buried SASA = sum(chain SASA) - complex SASA
    - Per-residue SASA for interface residues

    Args:
        pdb_string: PDB structure
        interface_residues: Interface residues from identify_interface_residues
        probe_radius: Water probe radius in angstroms

    Returns:
        SurfaceAreaResult with SASA values
    """
    parser = PDBParser(QUIET=True)
    # Filter to protein-only ATOM records for consistent SASA across
    # complex and unbound calculations (excludes glycans, ligands, etc.)
    protein_pdb_string = _filter_protein_pdb_string(pdb_string)
    structure = parser.get_structure(
        "complex", io.StringIO(protein_pdb_string)
    )

    # Choose SASA backend
    if _HAS_FREESASA:
        complex_sasa, _, complex_res_sasa = _compute_sasa_freesasa(
            protein_pdb_string, probe_radius
        )
    else:
        complex_sasa = _compute_structure_sasa(structure, probe_radius)
        complex_res_sasa = _compute_residue_sasa(structure, probe_radius)

    # Compute per-chain unbound SASA and per-residue unbound SASA
    chain_sasa_unbound: Dict[str, float] = {}
    residue_sasa_unbound: Dict[str, float] = {}

    protein_chains = _get_protein_chains(structure, exclude_ligands=True)
    for chain_id, _residues in protein_chains.items():
        chain_pdb_str = _extract_chain_pdb_string(
            protein_pdb_string, chain_id, protein_only=True
        )

        if _HAS_FREESASA:
            total, _, res_sasa = _compute_sasa_freesasa(
                chain_pdb_str, probe_radius
            )
        else:
            chain_structure = parser.get_structure(
                f"chain_{chain_id}", io.StringIO(chain_pdb_str)
            )
            total = _compute_structure_sasa(chain_structure, probe_radius)
            res_sasa = _compute_residue_sasa(chain_structure, probe_radius)

        chain_sasa_unbound[chain_id] = total
        residue_sasa_unbound.update(res_sasa)

    # Buried SASA = sum(unbound chains) - complex SASA (Rosetta dSASA_int)
    total_chain_sasa = sum(chain_sasa_unbound.values())
    buried_sasa = total_chain_sasa - complex_sasa

    # Interface residue SASA (bound) and burial (unbound - bound)
    interface_residue_sasa = {}
    interface_residue_delta_sasa = {}
    for ir in interface_residues:
        key = f"{ir.chain_id}{ir.residue_number}{ir.insertion_code}"
        bound_val = complex_res_sasa.get(key, 0.0)
        unbound_val = residue_sasa_unbound.get(key, 0.0)
        interface_residue_sasa[key] = bound_val
        interface_residue_delta_sasa[key] = max(0.0, unbound_val - bound_val)

    logger.info(
        f"  SASA (FreeSASA={_HAS_FREESASA}): complex={complex_sasa:.1f}, "
        f"chains_total={total_chain_sasa:.1f}, "
        f"dSASA_int={buried_sasa:.1f} sq. angstroms"
    )

    return SurfaceAreaResult(
        complex_sasa=complex_sasa,
        chain_sasa=chain_sasa_unbound,
        buried_sasa=buried_sasa,
        interface_residue_sasa=interface_residue_sasa,
        interface_residue_delta_sasa=interface_residue_delta_sasa,
    )


def calculate_shape_complementarity(
    pdb_string: str,
    chain_pairs: List[Tuple[str, str]],
    interface_residues: List[InterfaceResidue],
) -> ShapeComplementarityResult:
    """
    Calculate simplified shape complementarity for interface.

    This is a simplified implementation based on interface gap volume.
    Uses average inter-surface distance as a proxy for shape complementarity.
    Lower gap volume indicates better complementarity.

    Full Lawrence & Colman (1993) Sc calculation would require surface
    normal computation and curvature correlation.

    Args:
        pdb_string: PDB structure
        chain_pairs: Interface chain pairs
        interface_residues: Interface residues

    Returns:
        ShapeComplementarityResult with simplified Sc score and metrics
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", io.StringIO(pdb_string))

    # Group interface residues by chain
    chain_interface = {}
    for ir in interface_residues:
        if ir.chain_id not in chain_interface:
            chain_interface[ir.chain_id] = []
        chain_interface[ir.chain_id].append(ir)

    # Collect interface atom coordinates per chain
    chain_coords = {}
    for model in structure:
        for chain in model:
            if chain.id not in chain_interface:
                continue
            interface_resnums = {
                (ir.residue_number, ir.insertion_code)
                for ir in chain_interface[chain.id]
            }
            coords = []
            for residue in chain:
                key = (residue.id[1], residue.id[2].strip())
                if key in interface_resnums:
                    for atom in residue.get_atoms():
                        coords.append(atom.get_vector().get_array())
            if coords:
                chain_coords[chain.id] = np.array(coords)
        break  # Only first model

    if len(chain_coords) < 2:
        logger.warning("  Insufficient chains for shape complementarity")
        return ShapeComplementarityResult()

    # Compute average minimum distance between interface surfaces
    # This serves as a proxy for gap volume
    total_gap = 0.0
    total_points = 0
    interface_area = 0.0

    for chain_a, chain_b in chain_pairs:
        if chain_a not in chain_coords or chain_b not in chain_coords:
            continue

        coords_a = chain_coords[chain_a]
        coords_b = chain_coords[chain_b]

        # For each atom on side A, find minimum distance to side B
        # Using chunked computation to manage memory
        chunk_size = 500
        min_dists_a = []
        for i in range(0, len(coords_a), chunk_size):
            chunk = coords_a[i : i + chunk_size]
            diff = chunk[:, np.newaxis, :] - coords_b[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            min_dists_a.extend(np.min(dists, axis=1).tolist())

        min_dists_b = []
        for i in range(0, len(coords_b), chunk_size):
            chunk = coords_b[i : i + chunk_size]
            diff = chunk[:, np.newaxis, :] - coords_a[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            min_dists_b.extend(np.min(dists, axis=1).tolist())

        all_min_dists = min_dists_a + min_dists_b
        total_gap += sum(all_min_dists)
        total_points += len(all_min_dists)

    if total_points == 0:
        return ShapeComplementarityResult()

    avg_gap = total_gap / total_points

    # Convert average gap distance to a 0-1 score.
    # Perfect complementarity (avg_gap ~1.5A for vdW contact) -> score ~1.0
    # Poor complementarity (avg_gap > 5A) -> score ~0.0
    # Use a sigmoid-like transformation centered at typical contact distance
    typical_contact = 2.0  # angstroms (typical vdW contact distance)
    sc_score = max(0.0, min(1.0, 1.0 - (avg_gap - typical_contact) / 4.0))

    # Estimate interface area from buried SASA (approximation)
    # Use number of interface atoms as a rough proxy
    interface_area = (
        float(total_points) * 5.0
    )  # ~5 sq A per atom, rough estimate

    # Gap volume: average gap * interface area (very rough approximation)
    gap_volume = avg_gap * interface_area

    logger.info(
        f"  Shape complementarity: Sc={sc_score:.3f}, "
        f"avg_gap={avg_gap:.2f} A, "
        f"interface_area~{interface_area:.0f} sq. angstroms"
    )

    return ShapeComplementarityResult(
        sc_score=sc_score,
        interface_area=interface_area,
        interface_gap_volume=gap_volume,
    )
