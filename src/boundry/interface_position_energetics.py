"""Per-position interface energetics: residue removal and alanine scanning.

Provides per-residue binding energy decomposition via two complementary
analyses:

- **Per-position** (residue removal): removes each interface residue and
  recomputes binding energy.
- **Alanine scan**: mutates each interface residue to alanine and
  recomputes binding energy.

Both analyses produce :class:`PositionResult` objects with identical
column schemas:

- ``dG``: full binding energy of the modified system
- ``ddG``: change from wild-type (``dG - dG_wt``); positive = the
  modification is destabilising, indicating a binding hotspot
"""

import contextlib
import csv
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
)

from boundry.binding_energy import (
    _get_interface_chain_groups,
    _repack_with_designer,
    calculate_binding_energy,
    extract_chain,
)
from boundry.interface import InterfaceResidue
from boundry.utils import suppress_stderr as _suppress_stderr

if TYPE_CHECKING:
    from boundry.designer import Designer
    from boundry.relaxer import Relaxer

logger = logging.getLogger(__name__)

# Residues skipped during alanine scanning
_ALANINE_SCAN_SKIP = {"ALA", "GLY", "PRO"}

# Atoms retained when mutating to alanine
_ALA_ATOMS = {"N", "CA", "C", "O", "CB", "OXT", "H", "HA"}


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ResidueKey:
    """Immutable identifier for a protein residue."""

    chain_id: str
    residue_number: int
    insertion_code: str = ""

    def __str__(self) -> str:
        icode = self.insertion_code or ""
        return f"{self.chain_id}{self.residue_number}{icode}"


@dataclass
class PositionRow:
    """One row of position-level energetics for a single interface residue.

    Used by both per-position (residue removal) and alanine scan
    analyses.  The ``dG`` and ``ddG`` fields have the same meaning in
    both contexts:

    - ``dG``: full binding energy of the modified system
      (per-position: ``dG_without_i``; alanine scan: ``dG_ala``)
    - ``ddG``: change from wild-type (``dG - dG_wt``); positive means
      the modification is destabilising (i.e. the residue is a
      binding hotspot)
    """

    # Identifiers / context
    chain_id: str
    residue_number: int
    insertion_code: str
    wt_resname: str
    partner_chain: str
    min_distance: float
    num_contacts: int

    # Burial
    delta_sasa: Optional[float] = None

    # WT binding energy (repeated per row for convenience)
    dG_wt: Optional[float] = None

    # Modified-system binding energy
    dG: Optional[float] = None

    # Change from wild-type: dG - dG_wt
    ddG: Optional[float] = None

    # Scan-skip info (relevant to alanine scan: GLY/PRO/ALA)
    scan_skipped: bool = False
    skip_reason: Optional[str] = None


@dataclass
class PositionResult:
    """Complete position-level energetics output.

    Shared container for both per-position and alanine scan analyses.
    """

    rows: List[PositionRow] = field(default_factory=list)
    dG_wt: Optional[float] = None
    distance_cutoff: float = 8.0
    chain_pairs: List[Tuple[str, str]] = field(default_factory=list)
    position_relax: str = "none"
    relax_separated: bool = False
    constrained: bool = False


class PositionEnergeticsResult(NamedTuple):
    """Return type for :func:`compute_position_energetics`."""

    per_position: Optional[PositionResult]
    alanine_scan: Optional[PositionResult]


# ------------------------------------------------------------------
# PDB mutation / removal helpers
# ------------------------------------------------------------------


def mutate_to_alanine(
    pdb_string: str,
    chain_id: str,
    resnum: int,
    icode: str = "",
) -> str:
    """Mutate a single residue to alanine in a PDB string.

    Deletes side-chain atoms beyond the alanine atom set and renames
    the residue to ``ALA``.  Downstream tools (PDBFixer / OpenMM)
    handle hydrogen rebuilding.

    Args:
        pdb_string: PDB file contents.
        chain_id: Chain containing the target residue.
        resnum: Residue sequence number.
        icode: Insertion code (empty string if none).

    Returns:
        Modified PDB string with the residue mutated to ALA.
    """
    out_lines: list[str] = []
    for line in pdb_string.splitlines(keepends=True):
        if not line.startswith(("ATOM", "HETATM")):
            out_lines.append(line)
            continue

        # Parse PDB fixed-width fields
        line_chain = line[21] if len(line) > 21 else ""
        try:
            line_resnum = int(line[22:26].strip())
        except (ValueError, IndexError):
            out_lines.append(line)
            continue
        line_icode = line[26].strip() if len(line) > 26 else ""
        atom_name = line[12:16].strip() if len(line) > 15 else ""

        # Check if this line belongs to the target residue
        if (
            line_chain == chain_id
            and line_resnum == resnum
            and line_icode == (icode or "")
        ):
            # Drop atoms not in alanine
            if atom_name not in _ALA_ATOMS:
                continue
            # Rename residue to ALA
            line = line[:17] + "ALA" + line[20:]

        out_lines.append(line)

    return "".join(out_lines)


def remove_residue(
    pdb_string: str,
    chain_id: str,
    resnum: int,
    icode: str = "",
) -> str:
    """Remove all ATOM/HETATM records for a residue from a PDB string.

    Args:
        pdb_string: PDB file contents.
        chain_id: Chain containing the target residue.
        resnum: Residue sequence number.
        icode: Insertion code (empty string if none).

    Returns:
        PDB string with the residue removed.
    """
    out_lines: list[str] = []
    for line in pdb_string.splitlines(keepends=True):
        if line.startswith(("ATOM", "HETATM")):
            line_chain = line[21] if len(line) > 21 else ""
            try:
                line_resnum = int(line[22:26].strip())
            except (ValueError, IndexError):
                out_lines.append(line)
                continue
            line_icode = line[26].strip() if len(line) > 26 else ""

            if (
                line_chain == chain_id
                and line_resnum == resnum
                and line_icode == (icode or "")
            ):
                continue  # skip this residue's atoms

        out_lines.append(line)

    return "".join(out_lines)


# ------------------------------------------------------------------
# Energy helpers
# ------------------------------------------------------------------


def _compute_rosetta_dG(
    pdb_string: str,
    relaxer: "Relaxer",
    chain_pairs: List[Tuple[str, str]],
    distance_cutoff: float = 8.0,
    relax_separated: bool = False,
    designer: Optional["Designer"] = None,
) -> float:
    """Compute binding energy dG = E_bound - E_unbound (negative = favorable).

    Wraps :func:`calculate_binding_energy`.

    Raises:
        RuntimeError: If binding energy calculation fails (returns None).
    """
    result = calculate_binding_energy(
        pdb_string,
        relaxer,
        chain_pairs=chain_pairs,
        distance_cutoff=distance_cutoff,
        relax_separated=relax_separated,
        designer=designer,
    )
    if result.binding_energy is None:
        raise RuntimeError(
            "Binding energy calculation failed "
            "(energy returned None)"
        )
    return result.binding_energy


def _apply_relax_policy(
    pdb_string: str,
    chain_groups: List[List[str]],
    position_relax: str,
    designer: Optional["Designer"],
) -> Tuple[str, List[str]]:
    """Apply the relax policy and return (bound_pdb, [unbound_pdbs]).

    ``position_relax`` is one of ``"both"``, ``"unbound"``, ``"none"``.
    """
    bound_pdb = pdb_string
    if position_relax == "both" and designer is not None:
        try:
            bound_pdb = _repack_with_designer(pdb_string, designer)
        except Exception as e:
            logger.warning(f"Failed to repack bound complex: {e}")

    unbound_pdbs: list[str] = []
    for group in chain_groups:
        chain_pdb = extract_chain(pdb_string, group)
        if position_relax in ("both", "unbound") and designer is not None:
            try:
                chain_pdb = _repack_with_designer(chain_pdb, designer)
            except Exception as e:
                label = "+".join(group)
                logger.warning(f"Failed to repack chain(s) {label}: {e}")
        unbound_pdbs.append(chain_pdb)

    return bound_pdb, unbound_pdbs


# ------------------------------------------------------------------
# Alanine scan
# ------------------------------------------------------------------


def compute_alanine_scan(
    pdb_string: str,
    interface_residues: List[InterfaceResidue],
    chain_pairs: List[Tuple[str, str]],
    relaxer: "Relaxer",
    dG_wt: float,
    designer: Optional["Designer"] = None,
    distance_cutoff: float = 8.0,
    position_relax: str = "none",
    relax_separated: bool = False,
    scan_chains: Optional[List[str]] = None,
    max_scan_sites: Optional[int] = None,
    show_progress: bool = False,
    quiet: bool = False,
) -> Dict[ResidueKey, Tuple[Optional[float], Optional[float]]]:
    """Run alanine scanning on interface residues.

    Returns a dict mapping ``ResidueKey`` -> ``(dG_ala, ddG)`` for each
    scanned residue.  Residues that are GLY, PRO, or already ALA are
    skipped and mapped to ``(None, None)``.
    """
    relax_sep = position_relax in ("both", "unbound")
    relax_designer = designer if relax_sep else None

    scan_sites = _select_scan_sites(
        interface_residues, scan_chains, max_scan_sites
    )

    results: Dict[ResidueKey, Tuple[Optional[float], Optional[float]]] = {}

    iterable = scan_sites
    bar = None
    if show_progress:
        from tqdm import tqdm

        bar = tqdm(
            scan_sites, desc="Alanine scan", unit="res"
        )
        iterable = bar

    for idx, ir in enumerate(iterable):
        key = ResidueKey(ir.chain_id, ir.residue_number, ir.insertion_code)

        if bar is not None:
            bar.set_postfix_str(
                f"{ir.residue_name} {key}"
            )

        if ir.residue_name in _ALANINE_SCAN_SKIP:
            results[key] = (None, None)
            continue

        logger.info(
            f"  AlaScan [{idx + 1}/{len(scan_sites)}]: "
            f"{ir.residue_name} {key}"
        )

        mutant_pdb = mutate_to_alanine(
            pdb_string, ir.chain_id, ir.residue_number, ir.insertion_code
        )

        try:
            ctx = _suppress_stderr() if quiet else contextlib.nullcontext()
            with ctx:
                dG_ala = _compute_rosetta_dG(
                    mutant_pdb,
                    relaxer,
                    chain_pairs=chain_pairs,
                    distance_cutoff=distance_cutoff,
                    relax_separated=relax_sep or relax_separated,
                    designer=relax_designer,
                )
            ddG = dG_ala - dG_wt
            results[key] = (dG_ala, ddG)
            logger.info(
                f"    dG_ala={dG_ala:.2f}, ddG={ddG:.2f} kcal/mol"
            )
        except Exception as e:
            logger.warning(f"    AlaScan failed for {key}: {e}")
            results[key] = (None, None)

    return results


# ------------------------------------------------------------------
# Per-position (residue removal)
# ------------------------------------------------------------------


def compute_per_position_dG(
    pdb_string: str,
    interface_residues: List[InterfaceResidue],
    chain_pairs: List[Tuple[str, str]],
    relaxer: "Relaxer",
    dG_wt: float,
    designer: Optional["Designer"] = None,
    distance_cutoff: float = 8.0,
    position_relax: str = "none",
    relax_separated: bool = False,
    scan_chains: Optional[List[str]] = None,
    max_scan_sites: Optional[int] = None,
    show_progress: bool = False,
    quiet: bool = False,
) -> Dict[ResidueKey, Optional[float]]:
    """Compute per-residue binding energy with residue removed.

    Returns ``dG_without_i`` for each residue — the full binding energy
    of the complex with residue *i* removed.  The ``ddG`` is computed
    downstream as ``dG_without_i - dG_wt`` (positive = removing the
    residue is destabilising, i.e. the residue is a binding hotspot).
    """
    relax_sep = position_relax in ("both", "unbound")
    relax_designer = designer if relax_sep else None

    scan_sites = _select_scan_sites(
        interface_residues, scan_chains, max_scan_sites
    )

    results: Dict[ResidueKey, Optional[float]] = {}

    iterable = scan_sites
    bar = None
    if show_progress:
        from tqdm import tqdm

        bar = tqdm(
            scan_sites, desc="Per-position dG", unit="res"
        )
        iterable = bar

    for idx, ir in enumerate(iterable):
        key = ResidueKey(ir.chain_id, ir.residue_number, ir.insertion_code)

        if bar is not None:
            bar.set_postfix_str(
                f"{ir.residue_name} {key}"
            )

        logger.info(
            f"  PerPosition [{idx + 1}/{len(scan_sites)}]: "
            f"{ir.residue_name} {key}"
        )

        removed_pdb = remove_residue(
            pdb_string, ir.chain_id, ir.residue_number, ir.insertion_code
        )

        try:
            ctx = _suppress_stderr() if quiet else contextlib.nullcontext()
            with ctx:
                dG_without_i = _compute_rosetta_dG(
                    removed_pdb,
                    relaxer,
                    chain_pairs=chain_pairs,
                    distance_cutoff=distance_cutoff,
                    relax_separated=relax_sep or relax_separated,
                    designer=relax_designer,
                )
            results[key] = dG_without_i
            ddG = dG_without_i - dG_wt
            logger.info(
                f"    dG_without_i={dG_without_i:.2f}, "
                f"ddG={ddG:.2f} kcal/mol"
            )
        except Exception as e:
            logger.warning(f"    Per-position failed for {key}: {e}")
            results[key] = None

    return results


# ------------------------------------------------------------------
# Scan-site selection
# ------------------------------------------------------------------


def _select_scan_sites(
    interface_residues: List[InterfaceResidue],
    scan_chains: Optional[List[str]] = None,
    max_scan_sites: Optional[int] = None,
) -> List[InterfaceResidue]:
    """Filter and deduplicate interface residues for scanning."""
    seen: set[ResidueKey] = set()
    sites: list[InterfaceResidue] = []
    for ir in interface_residues:
        key = ResidueKey(ir.chain_id, ir.residue_number, ir.insertion_code)
        if key in seen:
            continue
        seen.add(key)
        if scan_chains and ir.chain_id not in scan_chains:
            continue
        sites.append(ir)

    # Stable ordering: chain, resnum, icode
    sites.sort(
        key=lambda r: (r.chain_id, r.residue_number, r.insertion_code)
    )

    if max_scan_sites is not None and len(sites) > max_scan_sites:
        logger.info(
            f"  Limiting scan to {max_scan_sites} of "
            f"{len(sites)} interface residues"
        )
        sites = sites[:max_scan_sites]

    return sites


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------


def _build_position_rows(
    scan_sites: List[InterfaceResidue],
    dG_wt: float,
    sasa_delta: Optional[Dict[str, float]],
) -> List[PositionRow]:
    """Build base PositionRow list with identifiers and SASA."""
    rows: list[PositionRow] = []
    for ir in scan_sites:
        sasa_key = (
            f"{ir.chain_id}{ir.residue_number}{ir.insertion_code}"
        )
        row = PositionRow(
            chain_id=ir.chain_id,
            residue_number=ir.residue_number,
            insertion_code=ir.insertion_code,
            wt_resname=ir.residue_name,
            partner_chain=ir.partner_chain,
            min_distance=ir.min_distance,
            num_contacts=ir.num_contacts,
            delta_sasa=(
                sasa_delta.get(sasa_key) if sasa_delta else None
            ),
            dG_wt=dG_wt,
        )
        rows.append(row)
    return rows


def compute_position_energetics(
    pdb_string: str,
    interface_residues: List[InterfaceResidue],
    chain_pairs: List[Tuple[str, str]],
    relaxer: "Relaxer",
    designer: Optional["Designer"] = None,
    distance_cutoff: float = 8.0,
    position_relax: str = "none",
    relax_separated: bool = False,
    scan_chains: Optional[List[str]] = None,
    max_scan_sites: Optional[int] = None,
    run_per_position: bool = False,
    run_alanine_scan: bool = False,
    sasa_delta: Optional[Dict[str, float]] = None,
    show_progress: bool = False,
    quiet: bool = False,
) -> PositionEnergeticsResult:
    """Run the full per-position energetics pipeline.

    Returns a :class:`PositionEnergeticsResult` with separate
    :class:`PositionResult` objects for per-position and alanine scan
    analyses.  Each result uses identical column schemas (``dG``,
    ``ddG``).

    Requires exactly two interface sides (chain groups). Raises
    ``ValueError`` if the chain pairs cannot be cleanly partitioned.
    """
    # Validate two-sided interface
    chain_groups = _get_interface_chain_groups(chain_pairs)
    if len(chain_groups) != 2:
        raise ValueError(
            f"Per-position energetics require a two-sided interface "
            f"(got {len(chain_groups)} groups: "
            f"{', '.join('+'.join(g) for g in chain_groups)}). "
            f"Provide --chains that cleanly partitions "
            f"(e.g. 'H:A,L:A')."
        )

    # Compute WT dG
    relax_sep = position_relax in ("both", "unbound")
    relax_designer = designer if relax_sep else None

    logger.info("Computing WT binding energy...")
    ctx = _suppress_stderr() if quiet else contextlib.nullcontext()
    with ctx:
        dG_wt = _compute_rosetta_dG(
            pdb_string,
            relaxer,
            chain_pairs=chain_pairs,
            distance_cutoff=distance_cutoff,
            relax_separated=relax_sep or relax_separated,
            designer=relax_designer,
        )
    logger.info(f"  dG_wt = {dG_wt:.2f} kcal/mol")

    # Alanine scan
    ala_results: Dict[
        ResidueKey, Tuple[Optional[float], Optional[float]]
    ] = {}
    if run_alanine_scan:
        logger.info("Running alanine scan...")
        ala_results = compute_alanine_scan(
            pdb_string,
            interface_residues,
            chain_pairs,
            relaxer,
            dG_wt,
            designer=designer,
            distance_cutoff=distance_cutoff,
            position_relax=position_relax,
            relax_separated=relax_separated,
            scan_chains=scan_chains,
            max_scan_sites=max_scan_sites,
            show_progress=show_progress,
            quiet=quiet,
        )

    # Per-position dG (residue removal)
    pp_dG_results: Dict[ResidueKey, Optional[float]] = {}
    if run_per_position:
        logger.info("Computing per-position dG...")
        pp_dG_results = compute_per_position_dG(
            pdb_string,
            interface_residues,
            chain_pairs,
            relaxer,
            dG_wt,
            designer=designer,
            distance_cutoff=distance_cutoff,
            position_relax=position_relax,
            relax_separated=relax_separated,
            scan_chains=scan_chains,
            max_scan_sites=max_scan_sites,
            show_progress=show_progress,
            quiet=quiet,
        )

    scan_sites = _select_scan_sites(
        interface_residues, scan_chains, max_scan_sites
    )

    result_kwargs = dict(
        dG_wt=dG_wt,
        distance_cutoff=distance_cutoff,
        chain_pairs=chain_pairs,
        position_relax=position_relax,
        relax_separated=relax_separated,
    )

    # Build per-position result
    pp_result: Optional[PositionResult] = None
    if run_per_position:
        pp_rows = _build_position_rows(scan_sites, dG_wt, sasa_delta)
        for row, ir in zip(pp_rows, scan_sites):
            key = ResidueKey(
                ir.chain_id, ir.residue_number, ir.insertion_code
            )
            dG_without_i = pp_dG_results.get(key)
            if dG_without_i is not None:
                row.dG = dG_without_i
                row.ddG = dG_without_i - dG_wt
        pp_result = PositionResult(rows=pp_rows, **result_kwargs)

    # Build alanine scan result
    ala_result: Optional[PositionResult] = None
    if run_alanine_scan:
        ala_rows = _build_position_rows(scan_sites, dG_wt, sasa_delta)
        for row, ir in zip(ala_rows, scan_sites):
            key = ResidueKey(
                ir.chain_id, ir.residue_number, ir.insertion_code
            )
            if ir.residue_name in _ALANINE_SCAN_SKIP:
                row.scan_skipped = True
                row.skip_reason = (
                    f"Skipped: {ir.residue_name} "
                    f"(GLY/PRO/ALA excluded from alanine scan)"
                )
            elif key in ala_results:
                dG_ala, ddG = ala_results[key]
                row.dG = dG_ala
                row.ddG = ddG
        ala_result = PositionResult(rows=ala_rows, **result_kwargs)

    return PositionEnergeticsResult(
        per_position=pp_result,
        alanine_scan=ala_result,
    )


# ------------------------------------------------------------------
# CSV output
# ------------------------------------------------------------------

_CSV_COLUMNS = [
    "chain_id",
    "residue_number",
    "insertion_code",
    "wt_resname",
    "partner_chain",
    "min_distance",
    "num_contacts",
    "delta_sasa",
    "dG_wt",
    "dG",
    "ddG",
    "scan_skipped",
    "skip_reason",
]

_METADATA_KEYS = [
    "distance_cutoff",
    "chain_pairs",
    "position_relax",
    "relax_separated",
]


def write_position_csv(
    result: PositionResult,
    path: Path,
) -> None:
    """Write position-level results to a CSV file.

    Metadata is written as ``#``-prefixed comment lines before the
    header.
    """
    path = Path(path)
    with open(path, "w", newline="") as f:
        # Metadata header
        f.write(f"# distance_cutoff={result.distance_cutoff}\n")
        pairs_str = ",".join(
            f"{a}:{b}" for a, b in result.chain_pairs
        )
        f.write(f"# chain_pairs={pairs_str}\n")
        f.write(f"# position_relax={result.position_relax}\n")
        f.write(f"# relax_separated={result.relax_separated}\n")
        if result.dG_wt is not None:
            f.write(f"# dG_wt={result.dG_wt:.4f}\n")

        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()

        for row in result.rows:
            d: dict = {}
            for col in _CSV_COLUMNS:
                val = getattr(row, col)
                if val is None:
                    d[col] = ""
                elif isinstance(val, float):
                    d[col] = f"{val:.4f}"
                elif isinstance(val, bool):
                    d[col] = str(val)
                else:
                    d[col] = str(val)
                # Represent empty insertion codes explicitly
                if col == "insertion_code" and not val:
                    d[col] = ""
            writer.writerow(d)

    logger.info(f"Wrote position CSV to {path}")


def format_position_table(
    result: PositionResult,
    top_n: int = 10,
    label: str = "hotspots",
) -> str:
    """Format a short text table of top position results.

    Sorts by ``ddG`` descending (most destabilising first).
    """
    rows_with_ddg = [
        r for r in result.rows if r.ddG is not None
    ]
    rows_with_ddg.sort(key=lambda r: r.ddG or 0.0, reverse=True)
    rows_with_ddg = rows_with_ddg[:top_n]

    if not rows_with_ddg:
        return ""

    lines = [
        f"Top {min(top_n, len(rows_with_ddg))} {label} "
        f"(ddG, kcal/mol):"
    ]
    lines.append(
        f"  {'Residue':<12} {'ddG':>8}  {'dG':>8}  {'dG_wt':>8}"
    )
    lines.append("  " + "-" * 42)
    for r in rows_with_ddg:
        res_label = f"{r.wt_resname} {r.chain_id}{r.residue_number}"
        ddg = f"{r.ddG:.2f}" if r.ddG is not None else "N/A"
        dg = f"{r.dG:.2f}" if r.dG is not None else "N/A"
        dgw = f"{r.dG_wt:.2f}" if r.dG_wt is not None else "N/A"
        lines.append(
            f"  {res_label:<12} {ddg:>8}  {dg:>8}  {dgw:>8}"
        )

    return "\n".join(lines)


def format_hotspot_table(
    result: PositionResult,
    top_n: int = 10,
) -> str:
    """Format a short text table of top AlaScan hotspots.

    .. deprecated::
        Use :func:`format_position_table` instead.
    """
    return format_position_table(
        result, top_n=top_n, label="AlaScan hotspots"
    )
