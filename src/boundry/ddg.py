"""ddG mutation scoring pipeline.

Provides mutation specification, parsing, validation utilities, and the
full compute pipeline (ensemble generation, four-state scoring,
aggregation) for ddG calculations.
"""

import logging
import re
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── Amino acid code mappings ──────────────────────────────────────

_ONE_TO_THREE: Dict[str, str] = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

_THREE_TO_ONE: Dict[str, str] = {v: k for k, v in _ONE_TO_THREE.items()}

_STANDARD_THREE: set = set(_THREE_TO_ONE)


# ── Data classes ──────────────────────────────────────────────────


@dataclass(frozen=True)
class MutationSpec:
    """Specification for a single point mutation.

    Attributes:
        chain_id: Chain containing the residue.
        residue_number: PDB residue sequence number.
        wild_type: Wild-type residue as a 3-letter code (e.g. ``"LEU"``).
        mutant: Mutant residue as a 3-letter code (e.g. ``"ALA"``).
        insertion_code: PDB insertion code (empty string if none).
    """

    chain_id: str
    residue_number: int
    wild_type: str
    mutant: str
    insertion_code: str = ""

    def __str__(self) -> str:
        wt = _THREE_TO_ONE.get(self.wild_type, "?")
        mut = _THREE_TO_ONE.get(self.mutant, "?")
        icode = self.insertion_code or ""
        return f"{self.chain_id}:{wt}{self.residue_number}{icode}{mut}"


@dataclass
class EnsembleMemberResult:
    """Result from scoring a single ensemble member.

    Energy values are populated by the worker; ``wt_bound_energy_rank``
    is assigned after all members are collected.
    """

    member_index: int
    bound_wt_energy: Optional[float] = None
    unbound_wt_energy: Optional[float] = None
    bound_mut_energy: Optional[float] = None
    unbound_mut_energy: Optional[float] = None
    wt_bound_energy_rank: Optional[int] = None

    @property
    def dG_wt(self) -> Optional[float]:
        if self.bound_wt_energy is None or self.unbound_wt_energy is None:
            return None
        return self.bound_wt_energy - self.unbound_wt_energy

    @property
    def dG_mut(self) -> Optional[float]:
        if (
            self.bound_mut_energy is None
            or self.unbound_mut_energy is None
        ):
            return None
        return self.bound_mut_energy - self.unbound_mut_energy

    @property
    def ddG(self) -> Optional[float]:
        dg_wt = self.dG_wt
        dg_mut = self.dG_mut
        if dg_wt is None or dg_mut is None:
            return None
        return dg_mut - dg_wt


@dataclass
class DdGResult:
    """Aggregated ddG result across ensemble members."""

    mutations: List[MutationSpec]
    member_results: List[EnsembleMemberResult]
    mean_ddG: Optional[float]
    std_ddG: Optional[float]
    mean_dG_wt: Optional[float]
    mean_dG_mut: Optional[float]
    n_successful: int
    n_ensemble: int
    ensemble_ddGs: List[float]
    minimized_pdb: Optional[str] = None
    sorted_by_wt_energy: bool = False
    top_n_applied: Optional[int] = None

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""
        members = []
        for m in self.member_results:
            members.append(
                {
                    "member_index": m.member_index,
                    "bound_wt_energy": m.bound_wt_energy,
                    "unbound_wt_energy": m.unbound_wt_energy,
                    "bound_mut_energy": m.bound_mut_energy,
                    "unbound_mut_energy": m.unbound_mut_energy,
                    "wt_bound_energy_rank": m.wt_bound_energy_rank,
                    "dG_wt": m.dG_wt,
                    "dG_mut": m.dG_mut,
                    "ddG": m.ddG,
                }
            )
        return {
            "mutations": [str(m) for m in self.mutations],
            "mean_ddG": self.mean_ddG,
            "std_ddG": self.std_ddG,
            "mean_dG_wt": self.mean_dG_wt,
            "mean_dG_mut": self.mean_dG_mut,
            "n_successful": self.n_successful,
            "n_ensemble": self.n_ensemble,
            "ensemble_ddGs": self.ensemble_ddGs,
            "sorted_by_wt_energy": self.sorted_by_wt_energy,
            "top_n_applied": self.top_n_applied,
            "member_results": members,
        }


@dataclass
class InterfaceDgResult:
    """Result from interface dG (binding energy) computation."""

    dG: float
    minimized_pdb: str


# ── Parsing helpers ───────────────────────────────────────────────

# Pattern: <chain>:<wt_1letter><resnum>[<icode>]<mut_1letter>
# Examples: "A:L5A", "B:W100aG"
_MUTATION_RE = re.compile(
    r"^([A-Za-z]):([A-Za-z])(\d+)([A-Za-z]?)([A-Za-z])$"
)


def _normalize_aa(code: str) -> str:
    """Normalize an amino acid code to a 3-letter uppercase string.

    Accepts either a 1-letter or 3-letter code (case-insensitive).

    Raises:
        ValueError: If the code is not a recognized amino acid.
    """
    upper = code.upper()
    if upper in _STANDARD_THREE:
        return upper
    if upper in _ONE_TO_THREE:
        return _ONE_TO_THREE[upper]
    raise ValueError(
        f"Unrecognized amino acid code: {code!r}. "
        f"Expected a 1-letter or 3-letter standard amino acid code."
    )


def parse_mutation_string(mutation_string: str) -> List[MutationSpec]:
    """Parse a comma-separated mutation string into :class:`MutationSpec` objects.

    Format per mutation: ``<chain>:<wt><resnum>[<icode>]<mut>``

    - ``<chain>``: single-letter chain ID
    - ``<wt>``: wild-type residue (1-letter code)
    - ``<resnum>``: integer residue number
    - ``<icode>``: optional single-letter insertion code
    - ``<mut>``: mutant residue (1-letter code)

    Examples::

        "A:L5A"           -> [MutationSpec("A", 5, "LEU", "ALA")]
        "A:L5A,B:W10G"    -> [MutationSpec("A", 5, ...), MutationSpec("B", 10, ...)]
        "H:S100aA"        -> [MutationSpec("H", 100, "SER", "ALA", "a")]

    Args:
        mutation_string: Comma-separated mutation descriptors.

    Returns:
        List of parsed :class:`MutationSpec` objects.

    Raises:
        ValueError: If any mutation descriptor cannot be parsed.
    """
    specs: List[MutationSpec] = []
    for token in mutation_string.split(","):
        token = token.strip()
        if not token:
            continue
        m = _MUTATION_RE.match(token)
        if m is None:
            raise ValueError(
                f"Cannot parse mutation {token!r}. "
                f"Expected format: <chain>:<wt><resnum>[<icode>]<mut> "
                f"(e.g. 'A:L5A' or 'H:S100aA')."
            )
        chain, wt_letter, resnum_str, icode, mut_letter = m.groups()
        wt_three = _normalize_aa(wt_letter)
        mut_three = _normalize_aa(mut_letter)
        specs.append(
            MutationSpec(
                chain_id=chain,
                residue_number=int(resnum_str),
                wild_type=wt_three,
                mutant=mut_three,
                insertion_code=icode,
            )
        )
    if not specs:
        raise ValueError(
            f"No mutations found in string: {mutation_string!r}"
        )
    return specs


def parse_mutation_dict(d: Dict[str, str]) -> MutationSpec:
    """Parse a dictionary into a :class:`MutationSpec`.

    Accepted keys (case-insensitive):

    - ``chain`` — chain ID (required)
    - ``resnum`` or ``residue_number`` — residue number (required)
    - ``wt`` or ``wild_type`` — wild-type residue, 1- or 3-letter
      (required)
    - ``mut`` or ``mutant`` — mutant residue, 1- or 3-letter (required)
    - ``icode`` or ``insertion_code`` — insertion code (optional)

    Raises:
        ValueError: If required keys are missing or codes are invalid.
    """
    # Normalize keys to lowercase
    lower = {k.lower(): v for k, v in d.items()}

    chain = lower.get("chain")
    if chain is None:
        raise ValueError(f"Missing 'chain' key in mutation dict: {d!r}")

    resnum = lower.get("resnum") or lower.get("residue_number")
    if resnum is None:
        raise ValueError(
            f"Missing 'resnum' or 'residue_number' key in "
            f"mutation dict: {d!r}"
        )

    wt = lower.get("wt") or lower.get("wild_type")
    if wt is None:
        raise ValueError(
            f"Missing 'wt' or 'wild_type' key in mutation dict: {d!r}"
        )

    mut = lower.get("mut") or lower.get("mutant")
    if mut is None:
        raise ValueError(
            f"Missing 'mut' or 'mutant' key in mutation dict: {d!r}"
        )

    icode = lower.get("icode", "") or lower.get("insertion_code", "")

    return MutationSpec(
        chain_id=str(chain),
        residue_number=int(resnum),
        wild_type=_normalize_aa(str(wt)),
        mutant=_normalize_aa(str(mut)),
        insertion_code=str(icode),
    )


def parse_mutations(
    mutations: Optional[List[Dict[str, str]]] = None,
    mutation_string: Optional[str] = None,
) -> List[MutationSpec]:
    """Parse mutations from either dict list or string format.

    Exactly one of *mutations* or *mutation_string* must be provided.

    Args:
        mutations: List of mutation dictionaries (see
            :func:`parse_mutation_dict`).
        mutation_string: Comma-separated mutation string (see
            :func:`parse_mutation_string`).

    Returns:
        List of :class:`MutationSpec` objects.

    Raises:
        ValueError: If neither or both arguments are provided, or if
            parsing fails.
    """
    if mutations is not None and mutation_string is not None:
        raise ValueError(
            "Provide either 'mutations' or 'mutation_string', not both."
        )
    if mutations is not None:
        return [parse_mutation_dict(d) for d in mutations]
    if mutation_string is not None:
        return parse_mutation_string(mutation_string)
    raise ValueError(
        "Must provide either 'mutations' or 'mutation_string'."
    )


# ── WT validation ─────────────────────────────────────────────────


def validate_mutations(
    pdb_string: str,
    specs: List[MutationSpec],
) -> None:
    """Validate that wild-type residues match the input structure.

    Scans the PDB string for the residue at each mutation position
    and checks that the actual residue name matches the declared
    wild-type.

    Args:
        pdb_string: PDB file contents.
        specs: Mutation specifications to validate.

    Raises:
        ValueError: If any wild-type residue does not match, or if a
            specified position is not found in the structure.
    """
    # Build lookup: (chain, resnum, icode) -> resname
    residue_map: Dict[Tuple[str, int, str], str] = {}
    for line in pdb_string.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if len(line) < 27:
            continue
        chain = line[21]
        try:
            resnum = int(line[22:26].strip())
        except (ValueError, IndexError):
            continue
        icode = line[26].strip()
        resname = line[17:20].strip()
        key = (chain, resnum, icode)
        if key not in residue_map:
            residue_map[key] = resname

    errors: List[str] = []
    for spec in specs:
        key = (spec.chain_id, spec.residue_number, spec.insertion_code)
        actual = residue_map.get(key)
        if actual is None:
            errors.append(
                f"Position {spec.chain_id}:{spec.residue_number}"
                f"{spec.insertion_code} not found in structure."
            )
        elif actual != spec.wild_type:
            errors.append(
                f"Position {spec.chain_id}:{spec.residue_number}"
                f"{spec.insertion_code}: expected {spec.wild_type}, "
                f"found {actual}."
            )

    if errors:
        raise ValueError(
            "Wild-type validation failed:\n  " + "\n  ".join(errors)
        )


# ── Neighborhood helpers ─────────────────────────────────────────


def _find_neighborhood_residues(
    pdb_string: str,
    mutation_sites: List[Tuple[str, int, str]],
    neighborhood_radius: float = 8.0,
    sequence_window: int = 1,
) -> Set[Tuple[str, int, str]]:
    """Find residues near mutation sites by CB/CA distance.

    Pure ATOM-record parsing — no BioPython or OpenMM.

    Args:
        pdb_string: PDB file contents.
        mutation_sites: ``(chain, resnum, icode)`` for each site.
        neighborhood_radius: Distance cutoff in angstroms.
        sequence_window: Sequence-position expansion around each
            neighbor (0 = no expansion, 1 = +/- 1 residue).

    Returns:
        Set of ``(chain, resnum, icode)`` tuples in the neighborhood.
    """
    # Parse representative atom per residue (prefer CB, fall back CA)
    residue_coords: Dict[Tuple[str, int, str], List[float]] = {}
    ca_coords: Dict[Tuple[str, int, str], List[float]] = {}

    for line in pdb_string.splitlines():
        if not line.startswith("ATOM"):
            continue
        if len(line) < 54:
            continue
        atom_name = line[12:16].strip()
        if atom_name not in ("CA", "CB"):
            continue
        chain = line[21]
        try:
            resnum = int(line[22:26].strip())
        except (ValueError, IndexError):
            continue
        icode = line[26].strip()
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except (ValueError, IndexError):
            continue
        key = (chain, resnum, icode)
        if atom_name == "CB":
            residue_coords[key] = [x, y, z]
        elif atom_name == "CA":
            ca_coords[key] = [x, y, z]

    # Fall back to CA for residues without CB (GLY, etc.)
    for key, coords in ca_coords.items():
        if key not in residue_coords:
            residue_coords[key] = coords

    # Build per-chain sorted residue lists for window expansion
    chain_residues: Dict[str, List[Tuple[int, str]]] = {}
    for chain, resnum, icode in residue_coords:
        chain_residues.setdefault(chain, []).append((resnum, icode))
    for chain in chain_residues:
        chain_residues[chain].sort()

    # Find mutation site coordinates
    site_set = set(mutation_sites)
    site_coords = []
    for site in site_set:
        if site in residue_coords:
            site_coords.append(residue_coords[site])

    if not site_coords:
        return set()

    # Distance search
    neighbors: Set[Tuple[str, int, str]] = set()
    radius_sq = neighborhood_radius * neighborhood_radius
    for key, coords in residue_coords.items():
        for sc in site_coords:
            dx = coords[0] - sc[0]
            dy = coords[1] - sc[1]
            dz = coords[2] - sc[2]
            if dx * dx + dy * dy + dz * dz <= radius_sq:
                neighbors.add(key)
                break

    # Sequence window expansion
    if sequence_window > 0:
        expanded: Set[Tuple[str, int, str]] = set()
        for chain, resnum, icode in neighbors:
            residues = chain_residues.get(chain, [])
            try:
                idx = residues.index((resnum, icode))
            except ValueError:
                continue
            for offset in range(
                -sequence_window, sequence_window + 1
            ):
                new_idx = idx + offset
                if 0 <= new_idx < len(residues):
                    rn, ic = residues[new_idx]
                    expanded.add((chain, rn, ic))
        neighbors = expanded

    return neighbors


def build_neighborhood_spec(
    pdb_string: str,
    mutation_sites: List[Tuple[str, int, str]],
    neighborhood_radius: float = 8.0,
    sequence_window: int = 1,
) -> "DesignSpec":
    """Build a ``DesignSpec`` for repacking near mutation sites.

    Residues in the neighborhood are set to ``NATAA`` (repack only);
    all others default to ``NATRO`` (frozen).

    Args:
        pdb_string: PDB file contents.
        mutation_sites: ``(chain, resnum, icode)`` per site.
        neighborhood_radius: CB/CA distance cutoff (angstroms).
        sequence_window: Sequence expansion window.

    Returns:
        A :class:`~boundry.resfile.DesignSpec`.
    """
    from boundry.resfile import DesignSpec, ResidueMode, ResidueSpec

    neighbors = _find_neighborhood_residues(
        pdb_string, mutation_sites, neighborhood_radius, sequence_window
    )
    specs: Dict[str, ResidueSpec] = {}
    for chain, resnum, icode in neighbors:
        key = f"{chain}{resnum}{icode}"
        specs[key] = ResidueSpec(
            chain=chain,
            resnum=resnum,
            icode=icode,
            mode=ResidueMode.NATAA,
        )
    return DesignSpec(
        residue_specs=specs, default_mode=ResidueMode.NATRO
    )


def build_sampling_neighborhood(
    pdb_string: str,
    mutation_sites: List[Tuple[str, int, str]],
    neighborhood_radius: float = 8.0,
    sequence_window: int = 1,
) -> List[Tuple[str, int]]:
    """Build ``(chain, resnum)`` list for MD ensemble sampling.

    Suitable for passing to
    :meth:`~boundry.relaxer.Relaxer.generate_local_md_ensemble`
    as the *mutation_sites* parameter.
    """
    neighbors = _find_neighborhood_residues(
        pdb_string, mutation_sites, neighborhood_radius, sequence_window
    )
    return sorted({(c, r) for c, r, _ic in neighbors})


# ── DesignSpec serialization ─────────────────────────────────────


def _serialize_design_spec(spec: "DesignSpec") -> dict:
    """Convert a DesignSpec to a pickle-safe dict."""
    residue_specs = {}
    for key, rs in spec.residue_specs.items():
        residue_specs[key] = {
            "chain": rs.chain,
            "resnum": rs.resnum,
            "icode": rs.icode,
            "mode": rs.mode.value,
            "allowed_aas": (
                sorted(rs.allowed_aas) if rs.allowed_aas else None
            ),
        }
    return {
        "residue_specs": residue_specs,
        "default_mode": spec.default_mode.value,
    }


def _deserialize_design_spec(d: dict) -> "DesignSpec":
    """Reconstruct a DesignSpec from a serialized dict."""
    from boundry.resfile import DesignSpec, ResidueMode, ResidueSpec

    residue_specs = {}
    for key, rs_dict in d["residue_specs"].items():
        residue_specs[key] = ResidueSpec(
            chain=rs_dict["chain"],
            resnum=rs_dict["resnum"],
            icode=rs_dict["icode"],
            mode=ResidueMode(rs_dict["mode"]),
            allowed_aas=(
                set(rs_dict["allowed_aas"])
                if rs_dict["allowed_aas"]
                else None
            ),
        )
    return DesignSpec(
        residue_specs=residue_specs,
        default_mode=ResidueMode(d["default_mode"]),
    )


# ── Worker task / result types ───────────────────────────────────


@dataclass(frozen=True)
class _DdGMemberTask:
    """Pickle-safe task for scoring one ensemble member."""

    member_index: int
    member_pdb_string: str
    mutations: Tuple[MutationSpec, ...]
    neighborhood_spec_dict: dict
    chain_groups: Tuple[Tuple[str, ...], ...]
    separation_distance: float
    relax_config_dict: dict
    design_config_dict: dict
    implicit_solvent: bool
    ca_cutoff: float
    restraint_sd: float
    quiet: bool


@dataclass
class _DdGMemberResult:
    """Result from scoring one ensemble member."""

    member_index: int
    bound_wt_energy: Optional[float] = None
    unbound_wt_energy: Optional[float] = None
    bound_mut_energy: Optional[float] = None
    unbound_mut_energy: Optional[float] = None
    error: Optional[str] = None


# ── Ensemble member worker ───────────────────────────────────────

_ddg_worker_cache: Dict[str, Any] = {}


def _process_ensemble_member(task: _DdGMemberTask) -> _DdGMemberResult:
    """Score one ensemble member (WT + optional mutant, bound + unbound).

    Top-level function for pickle compatibility with ``spawn`` workers.
    """
    try:
        from boundry._parallel import (
            _config_fingerprint,
            _suppress_worker_warnings,
        )

        _suppress_worker_warnings()

        # Lazy-init Relaxer
        relax_key = _config_fingerprint(task.relax_config_dict)
        if _ddg_worker_cache.get("relax_key") != relax_key:
            from boundry.config import RelaxConfig
            from boundry.relaxer import Relaxer

            _ddg_worker_cache["relaxer"] = Relaxer(
                RelaxConfig(**task.relax_config_dict)
            )
            _ddg_worker_cache["relax_key"] = relax_key

        # Lazy-init Designer
        design_key = _config_fingerprint(task.design_config_dict)
        if _ddg_worker_cache.get("design_key") != design_key:
            from boundry.config import DesignConfig
            from boundry.designer import Designer

            _ddg_worker_cache["designer"] = Designer(
                DesignConfig(**task.design_config_dict)
            )
            _ddg_worker_cache["design_key"] = design_key

        relaxer = _ddg_worker_cache["relaxer"]
        designer = _ddg_worker_cache["designer"]

        design_spec = _deserialize_design_spec(
            task.neighborhood_spec_dict
        )
        chain_groups = [list(g) for g in task.chain_groups]

        # --- WT bound: repack → minimise → score --------------------
        wt_bound_pdb = _repack_and_minimize(
            task.member_pdb_string,
            design_spec,
            designer,
            relaxer,
            task.ca_cutoff,
            task.restraint_sd,
            task.implicit_solvent,
        )
        bound_wt_energy = relaxer.get_energy_breakdown(
            wt_bound_pdb
        )["total_energy"]

        # --- WT unbound: separate → score ---------------------------
        from boundry.relaxer import separate_interface_rigid_body

        unbound_pdb = separate_interface_rigid_body(
            wt_bound_pdb,
            chain_groups,
            task.separation_distance,
        )
        unbound_wt_energy = relaxer.get_energy_breakdown(
            unbound_pdb
        )["total_energy"]

        bound_mut_energy = None
        unbound_mut_energy = None

        # --- Mutant (only if mutations provided) --------------------
        if task.mutations:
            from boundry.interface_position_energetics import (
                mutate_residue,
            )

            mut_pdb = task.member_pdb_string
            for mut in task.mutations:
                mut_pdb = mutate_residue(
                    mut_pdb,
                    mut.chain_id,
                    mut.residue_number,
                    mut.mutant,
                    mut.insertion_code,
                )

            mut_bound_pdb = _repack_and_minimize(
                mut_pdb,
                design_spec,
                designer,
                relaxer,
                task.ca_cutoff,
                task.restraint_sd,
                task.implicit_solvent,
            )
            bound_mut_energy = relaxer.get_energy_breakdown(
                mut_bound_pdb
            )["total_energy"]

            mut_unbound_pdb = separate_interface_rigid_body(
                mut_bound_pdb,
                chain_groups,
                task.separation_distance,
            )
            unbound_mut_energy = relaxer.get_energy_breakdown(
                mut_unbound_pdb
            )["total_energy"]

        return _DdGMemberResult(
            member_index=task.member_index,
            bound_wt_energy=bound_wt_energy,
            unbound_wt_energy=unbound_wt_energy,
            bound_mut_energy=bound_mut_energy,
            unbound_mut_energy=unbound_mut_energy,
        )

    except Exception as exc:
        return _DdGMemberResult(
            member_index=task.member_index,
            error=f"{type(exc).__name__}: {exc}",
        )


def _repack_and_minimize(
    pdb_string: str,
    design_spec: "DesignSpec",
    designer: Any,
    relaxer: Any,
    ca_cutoff: float,
    restraint_sd: float,
    implicit_solvent: bool,
) -> str:
    """Write PDB to temp → repack → minimise → return PDB string."""
    with tempfile.NamedTemporaryFile(
        suffix=".pdb", mode="w", delete=False
    ) as f:
        f.write(pdb_string)
        tmp_path = Path(f.name)
    try:
        result = designer.repack(tmp_path, design_spec=design_spec)
        repacked_pdb = designer.result_to_pdb_string(result)
    finally:
        tmp_path.unlink(missing_ok=True)

    minimized_pdb = relaxer.minimize_with_pair_restraints(
        repacked_pdb,
        ca_cutoff=ca_cutoff,
        restraint_sd=restraint_sd,
        implicit_solvent=implicit_solvent,
    )
    return minimized_pdb


# ── Public API ───────────────────────────────────────────────────


def compute_ddg(
    pdb_string: str,
    mutations: List[MutationSpec],
    config: "DdGConfig",
    relaxer: Any = None,
    designer: Any = None,
    pool: Any = None,
) -> DdGResult:
    """Compute ddG for a set of mutations.

    Pipeline:
    1. Validate mutations against structure.
    2. Minimise with CA pair restraints.
    3. Build neighborhood DesignSpec and sampling neighborhood.
    4. Generate MD ensemble.
    5. Score each member (WT bound/unbound + mut bound/unbound).
    6. Aggregate ddG values.

    Args:
        pdb_string: PDB file contents.
        mutations: Mutations to score.
        config: ddG configuration.
        relaxer: Pre-built Relaxer (created if ``None``).
        designer: Pre-built Designer (created if ``None``).
        pool: :class:`~boundry._parallel.WorkPool` for parallel
            dispatch (``None`` = sequential).

    Returns:
        :class:`DdGResult` with per-member and aggregated energies.

    Raises:
        ValueError: If ``config.chain_pairs`` is ``None``.
    """
    from boundry.binding_energy import _get_interface_chain_groups
    from boundry.config import DdGConfig

    if config.chain_pairs is None:
        raise ValueError(
            "chain_pairs is required for ddG computation"
        )

    validate_mutations(pdb_string, mutations)

    # Lazy-create Relaxer / Designer
    if relaxer is None:
        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(
            RelaxConfig(implicit_solvent=config.implicit_solvent)
        )
    if designer is None:
        from boundry.config import DesignConfig
        from boundry.designer import Designer

        designer = Designer(DesignConfig())

    # Step 1: minimise with restraints
    minimized_pdb = relaxer.minimize_with_pair_restraints(
        pdb_string,
        ca_cutoff=config.ca_cutoff,
        restraint_sd=config.restraint_sd,
        implicit_solvent=config.implicit_solvent,
    )

    # Step 2: build neighborhoods
    mutation_sites = [
        (m.chain_id, m.residue_number, m.insertion_code)
        for m in mutations
    ]
    neighborhood_spec = build_neighborhood_spec(
        minimized_pdb,
        mutation_sites,
        config.neighborhood_radius,
        config.sequence_window,
    )
    sampling_neighborhood = build_sampling_neighborhood(
        minimized_pdb,
        mutation_sites,
        config.neighborhood_radius,
        config.sequence_window,
    )

    # Step 3: generate ensemble
    ensemble = relaxer.generate_local_md_ensemble(
        minimized_pdb,
        sampling_neighborhood,
        n_members=config.n_ensemble,
        md_total_steps=config.md_total_steps,
        md_equilibration_steps=config.md_equilibration_steps,
        md_temperature=config.md_temperature,
        md_friction=config.md_friction,
        neighborhood_radius=config.neighborhood_radius,
        sequence_window=config.sequence_window,
        ca_cutoff=config.ca_cutoff,
        restraint_sd=config.restraint_sd,
        implicit_solvent=config.implicit_solvent,
        seed=config.seed,
    )

    # Optionally cache ensemble
    if config.cache_ensemble and config.ensemble_dir is not None:
        config.ensemble_dir.mkdir(parents=True, exist_ok=True)
        for i, member_pdb in enumerate(ensemble):
            path = config.ensemble_dir / f"member_{i:04d}.pdb"
            path.write_text(member_pdb)

    # Step 4: build tasks
    chain_groups = _get_interface_chain_groups(config.chain_pairs)
    spec_dict = _serialize_design_spec(neighborhood_spec)
    relax_config_dict = {
        "implicit_solvent": config.implicit_solvent,
    }
    design_config_dict: Dict[str, Any] = {}

    tasks = [
        _DdGMemberTask(
            member_index=i,
            member_pdb_string=member_pdb,
            mutations=tuple(mutations),
            neighborhood_spec_dict=spec_dict,
            chain_groups=tuple(
                tuple(g) for g in chain_groups
            ),
            separation_distance=config.separation_distance,
            relax_config_dict=relax_config_dict,
            design_config_dict=design_config_dict,
            implicit_solvent=config.implicit_solvent,
            ca_cutoff=config.ca_cutoff,
            restraint_sd=config.restraint_sd,
            quiet=config.quiet,
        )
        for i, member_pdb in enumerate(ensemble)
    ]

    # Step 5: dispatch
    if pool is not None and pool.active:
        raw_results = pool.map(_process_ensemble_member, tasks)
    else:
        raw_results = [_process_ensemble_member(t) for t in tasks]

    # Step 6: convert and aggregate
    member_results: List[EnsembleMemberResult] = []
    for r in raw_results:
        if r.error is not None:
            logger.warning(
                "Ensemble member %d failed: %s",
                r.member_index,
                r.error,
            )
        member_results.append(
            EnsembleMemberResult(
                member_index=r.member_index,
                bound_wt_energy=r.bound_wt_energy,
                unbound_wt_energy=r.unbound_wt_energy,
                bound_mut_energy=r.bound_mut_energy,
                unbound_mut_energy=r.unbound_mut_energy,
            )
        )

    # Sort by WT bound energy and assign ranks
    if config.sort_members_by_wt_bound_energy:
        ranked = [
            m
            for m in member_results
            if m.bound_wt_energy is not None
        ]
        ranked.sort(key=lambda m: m.bound_wt_energy)
        for rank, m in enumerate(ranked):
            m.wt_bound_energy_rank = rank

    # Collect successful ddGs
    ensemble_ddGs: List[float] = []
    dg_wt_values: List[float] = []
    dg_mut_values: List[float] = []
    for m in member_results:
        ddg_val = m.ddG
        if ddg_val is not None:
            # If sorting enabled and top_n set, filter by rank
            if (
                config.sort_members_by_wt_bound_energy
                and config.average_top_n is not None
                and m.wt_bound_energy_rank is not None
                and m.wt_bound_energy_rank >= config.average_top_n
            ):
                continue
            ensemble_ddGs.append(ddg_val)
        dg_wt = m.dG_wt
        if dg_wt is not None:
            dg_wt_values.append(dg_wt)
        dg_mut = m.dG_mut
        if dg_mut is not None:
            dg_mut_values.append(dg_mut)

    mean_ddG = (
        statistics.mean(ensemble_ddGs) if ensemble_ddGs else None
    )
    std_ddG = (
        statistics.stdev(ensemble_ddGs)
        if len(ensemble_ddGs) >= 2
        else None
    )
    mean_dG_wt = (
        statistics.mean(dg_wt_values) if dg_wt_values else None
    )
    mean_dG_mut = (
        statistics.mean(dg_mut_values) if dg_mut_values else None
    )

    return DdGResult(
        mutations=mutations,
        member_results=member_results,
        mean_ddG=mean_ddG,
        std_ddG=std_ddG,
        mean_dG_wt=mean_dG_wt,
        mean_dG_mut=mean_dG_mut,
        n_successful=len(ensemble_ddGs),
        n_ensemble=len(ensemble),
        ensemble_ddGs=ensemble_ddGs,
        minimized_pdb=minimized_pdb,
        sorted_by_wt_energy=config.sort_members_by_wt_bound_energy,
        top_n_applied=(
            config.average_top_n
            if config.sort_members_by_wt_bound_energy
            and config.average_top_n is not None
            else None
        ),
    )


def compute_interface_dg(
    pdb_string: str,
    config: "DdGConfig",
    relaxer: Any = None,
    designer: Any = None,
    pool: Any = None,
) -> InterfaceDgResult:
    """Compute interface dG (binding energy) without mutations.

    Uses the same ensemble pipeline as :func:`compute_ddg` but with
    no mutations — only the WT bound/unbound energies are computed.

    Args:
        pdb_string: PDB file contents.
        config: ddG configuration.
        relaxer: Pre-built Relaxer.
        designer: Pre-built Designer.
        pool: WorkPool for parallel dispatch.

    Returns:
        :class:`InterfaceDgResult` with mean dG and minimized PDB.

    Raises:
        ValueError: If ``config.chain_pairs`` is ``None``.
        RuntimeError: If no ensemble members succeed.
    """
    from boundry.binding_energy import _get_interface_chain_groups
    from boundry.config import DdGConfig

    if config.chain_pairs is None:
        raise ValueError(
            "chain_pairs is required for dG computation"
        )

    # Lazy-create Relaxer / Designer
    if relaxer is None:
        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(
            RelaxConfig(implicit_solvent=config.implicit_solvent)
        )
    if designer is None:
        from boundry.config import DesignConfig
        from boundry.designer import Designer

        designer = Designer(DesignConfig())

    # Minimise
    minimized_pdb = relaxer.minimize_with_pair_restraints(
        pdb_string,
        ca_cutoff=config.ca_cutoff,
        restraint_sd=config.restraint_sd,
        implicit_solvent=config.implicit_solvent,
    )

    # Use empty mutation sites — sample around interface center
    # For dG-only, we still need a sampling neighborhood; use an
    # empty mutation list so all chains get sampled.
    sampling_neighborhood: List[Tuple[str, int]] = []

    # Generate ensemble
    ensemble = relaxer.generate_local_md_ensemble(
        minimized_pdb,
        sampling_neighborhood,
        n_members=config.n_ensemble,
        md_total_steps=config.md_total_steps,
        md_equilibration_steps=config.md_equilibration_steps,
        md_temperature=config.md_temperature,
        md_friction=config.md_friction,
        neighborhood_radius=config.neighborhood_radius,
        sequence_window=config.sequence_window,
        ca_cutoff=config.ca_cutoff,
        restraint_sd=config.restraint_sd,
        implicit_solvent=config.implicit_solvent,
        seed=config.seed,
    )

    # Optionally cache ensemble
    if config.cache_ensemble and config.ensemble_dir is not None:
        config.ensemble_dir.mkdir(parents=True, exist_ok=True)
        for i, member_pdb in enumerate(ensemble):
            path = config.ensemble_dir / f"member_{i:04d}.pdb"
            path.write_text(member_pdb)

    # Build empty neighborhood spec (repack nothing)
    from boundry.resfile import DesignSpec, ResidueMode

    empty_spec = DesignSpec(
        residue_specs={}, default_mode=ResidueMode.NATRO
    )
    spec_dict = _serialize_design_spec(empty_spec)

    chain_groups = _get_interface_chain_groups(config.chain_pairs)
    relax_config_dict = {
        "implicit_solvent": config.implicit_solvent,
    }
    design_config_dict: Dict[str, Any] = {}

    tasks = [
        _DdGMemberTask(
            member_index=i,
            member_pdb_string=member_pdb,
            mutations=(),
            neighborhood_spec_dict=spec_dict,
            chain_groups=tuple(
                tuple(g) for g in chain_groups
            ),
            separation_distance=config.separation_distance,
            relax_config_dict=relax_config_dict,
            design_config_dict=design_config_dict,
            implicit_solvent=config.implicit_solvent,
            ca_cutoff=config.ca_cutoff,
            restraint_sd=config.restraint_sd,
            quiet=config.quiet,
        )
        for i, member_pdb in enumerate(ensemble)
    ]

    # Dispatch
    if pool is not None and pool.active:
        raw_results = pool.map(_process_ensemble_member, tasks)
    else:
        raw_results = [_process_ensemble_member(t) for t in tasks]

    # Aggregate WT dG values
    dg_values: List[float] = []
    for r in raw_results:
        if r.error is not None:
            logger.warning(
                "Ensemble member %d failed: %s",
                r.member_index,
                r.error,
            )
            continue
        if (
            r.bound_wt_energy is not None
            and r.unbound_wt_energy is not None
        ):
            dg_values.append(
                r.bound_wt_energy - r.unbound_wt_energy
            )

    if not dg_values:
        raise RuntimeError(
            "All ensemble members failed — cannot compute dG"
        )

    return InterfaceDgResult(
        dG=statistics.mean(dg_values),
        minimized_pdb=minimized_pdb,
    )
