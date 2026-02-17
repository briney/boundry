"""ddG mutation scoring pipeline.

Provides mutation specification, parsing, and validation utilities
for the ddG protocol.  The full compute pipeline (ensemble generation,
four-state scoring, aggregation) will be added in later phases.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

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
