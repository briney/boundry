"""Interface residue identification for protein-protein interfaces."""

import io
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from Bio.PDB import PDBParser

logger = logging.getLogger(__name__)

# Standard amino acid residue names
STANDARD_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}

# Water residue names to exclude
WATER_RESIDUES = {"HOH", "WAT", "SOL", "TIP3", "TIP4", "SPC"}


@dataclass
class InterfaceResidue:
    """Represents a single interface residue."""

    chain_id: str
    residue_number: int
    residue_name: str
    insertion_code: str
    partner_chain: str
    min_distance: float
    num_contacts: int


@dataclass
class InterfaceInfo:
    """Complete interface analysis results."""

    interface_residues: List[InterfaceResidue] = field(default_factory=list)
    chain_pairs: List[Tuple[str, str]] = field(default_factory=list)
    interface_area: Optional[float] = None
    shape_complementarity: Optional[float] = None
    interface_energy: Optional[float] = None

    @property
    def n_interface_residues(self) -> int:
        return len(self.interface_residues)

    @property
    def summary(self) -> str:
        pairs = ", ".join(f"{a}-{b}" for a, b in self.chain_pairs)
        return f"{self.n_interface_residues} interface residues across {pairs}"


def _is_protein_residue(residue) -> bool:
    """Check if a residue is a protein residue (ATOM record, not water/ligand)."""
    het_flag = residue.id[0]
    # Standard protein residues have het_flag == " "
    # HETATM residues have het_flag starting with "H_" or "W"
    if het_flag != " ":
        return False
    return True


def _get_protein_chains(structure, exclude_ligands: bool = True) -> dict:
    """
    Get protein chains from a BioPython structure.

    Args:
        structure: BioPython Structure object
        exclude_ligands: If True, only include residues with ATOM records

    Returns:
        Dictionary mapping chain_id -> list of residues
    """
    chains = {}
    for model in structure:
        for chain in model:
            residues = []
            for residue in chain:
                if exclude_ligands:
                    if not _is_protein_residue(residue):
                        continue
                else:
                    # Still skip water
                    if residue.resname.strip() in WATER_RESIDUES:
                        continue
                residues.append(residue)
            if residues:
                chains[chain.id] = residues
        break  # Only first model
    return chains


def _get_residue_atoms(residue) -> list:
    """Get all atoms from a residue as coordinate arrays."""
    return [atom.get_vector().get_array() for atom in residue.get_atoms()]


def identify_interface_residues(
    pdb_string: str,
    distance_cutoff: float = 8.0,
    chain_pairs: Optional[List[Tuple[str, str]]] = None,
    exclude_ligands: bool = True,
) -> InterfaceInfo:
    """
    Identify interface residues between protein chains.

    Args:
        pdb_string: PDB file contents
        distance_cutoff: Maximum distance (angstroms) for interface definition
        chain_pairs: Specific chain pairs to analyze (auto-detect if None)
        exclude_ligands: If True, ignore HETATM records (default: True)

    Returns:
        InterfaceInfo with identified residues and metadata
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", io.StringIO(pdb_string))

    # Get protein chains
    protein_chains = _get_protein_chains(structure, exclude_ligands)
    chain_ids = list(protein_chains.keys())

    if len(chain_ids) < 2:
        logger.warning(
            f"Only {len(chain_ids)} chain(s) found - "
            "need at least 2 for interface analysis"
        )
        return InterfaceInfo()

    # Determine chain pairs to analyze
    if chain_pairs is None:
        # Auto-detect: all unique pairs of protein chains
        pairs = []
        for i in range(len(chain_ids)):
            for j in range(i + 1, len(chain_ids)):
                pairs.append((chain_ids[i], chain_ids[j]))
    else:
        # Validate requested pairs exist
        pairs = []
        for a, b in chain_pairs:
            if a in protein_chains and b in protein_chains:
                pairs.append((a, b))
            else:
                missing = []
                if a not in protein_chains:
                    missing.append(a)
                if b not in protein_chains:
                    missing.append(b)
                logger.warning(
                    f"Chain(s) {', '.join(missing)} not found in structure - "
                    f"skipping pair {a}-{b}"
                )

    if not pairs:
        logger.warning("No valid chain pairs to analyze")
        return InterfaceInfo()

    # Precompute atom coordinates per residue per chain
    chain_residue_atoms = {}
    for chain_id in chain_ids:
        residue_atoms = []
        for residue in protein_chains[chain_id]:
            coords = np.array(_get_residue_atoms(residue))
            residue_atoms.append((residue, coords))
        chain_residue_atoms[chain_id] = residue_atoms

    # Find interface residues for each pair
    interface_residues = []
    seen_residues = set()  # Track (chain_id, resnum, icode) to avoid duplicates
    active_pairs = []

    for chain_a, chain_b in pairs:
        pair_has_contacts = False

        for res_a, coords_a in chain_residue_atoms[chain_a]:
            for res_b, coords_b in chain_residue_atoms[chain_b]:
                # Compute pairwise distances between all atoms
                # Using broadcasting: (N, 1, 3) - (1, M, 3) -> (N, M, 3)
                diff = coords_a[:, np.newaxis, :] - coords_b[np.newaxis, :, :]
                distances = np.sqrt(np.sum(diff**2, axis=2))

                min_dist = float(np.min(distances))
                if min_dist > distance_cutoff:
                    continue

                pair_has_contacts = True
                n_contacts = int(np.sum(distances < distance_cutoff))

                # Add residue A as interface residue
                key_a = (
                    chain_a,
                    res_a.id[1],
                    res_a.id[2].strip(),
                )
                if key_a not in seen_residues:
                    seen_residues.add(key_a)
                    interface_residues.append(
                        InterfaceResidue(
                            chain_id=chain_a,
                            residue_number=res_a.id[1],
                            residue_name=res_a.resname.strip(),
                            insertion_code=res_a.id[2].strip(),
                            partner_chain=chain_b,
                            min_distance=min_dist,
                            num_contacts=n_contacts,
                        )
                    )
                else:
                    # Update existing entry if this contact is closer
                    for ir in interface_residues:
                        if (
                            ir.chain_id == chain_a
                            and ir.residue_number == res_a.id[1]
                            and ir.insertion_code == res_a.id[2].strip()
                        ):
                            if min_dist < ir.min_distance:
                                ir.min_distance = min_dist
                            ir.num_contacts += n_contacts
                            break

                # Add residue B as interface residue
                key_b = (
                    chain_b,
                    res_b.id[1],
                    res_b.id[2].strip(),
                )
                if key_b not in seen_residues:
                    seen_residues.add(key_b)
                    interface_residues.append(
                        InterfaceResidue(
                            chain_id=chain_b,
                            residue_number=res_b.id[1],
                            residue_name=res_b.resname.strip(),
                            insertion_code=res_b.id[2].strip(),
                            partner_chain=chain_a,
                            min_distance=min_dist,
                            num_contacts=n_contacts,
                        )
                    )
                else:
                    for ir in interface_residues:
                        if (
                            ir.chain_id == chain_b
                            and ir.residue_number == res_b.id[1]
                            and ir.insertion_code == res_b.id[2].strip()
                        ):
                            if min_dist < ir.min_distance:
                                ir.min_distance = min_dist
                            ir.num_contacts += n_contacts
                            break

        if pair_has_contacts:
            active_pairs.append((chain_a, chain_b))

    logger.info(
        f"Found {len(interface_residues)} interface residues "
        f"across {len(active_pairs)} chain pair(s)"
    )

    return InterfaceInfo(
        interface_residues=interface_residues,
        chain_pairs=active_pairs,
    )
