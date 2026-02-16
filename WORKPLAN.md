# Flex ddG-Aligned Scoring Pipeline: Technical Implementation Plan

## Background

### Problem

Boundry's current interface scoring pipeline computes ddG via single-point rigid-body chain separation (`E_complex - sum(E_separated)`) with no conformational sampling or ensemble averaging. This produces a single energy value from a single minimized structure, making predictions sensitive to the specific starting conformation and missing the conformational entropy effects of mutations.

### Gold Standard: Rosetta Flex ddG

The flex ddG protocol (Barlow et al., *JPCB* 2018, R=0.68 vs experimental data) achieves high accuracy through:

1. **Constrained minimization** of the input complex with harmonic CA-CA pair distance restraints (9A cutoff, 0.5A SD)
2. **Backrub ensemble generation**: Monte Carlo backbone sampling (35,000 trials, kT=1.2) on residues within 8A of the mutation site, producing ~35 independent backbone conformations
3. **Paired WT/mutant scoring**: Each ensemble member forks into WT and mutant branches from the **same backbone**, so systematic backbone geometry errors cancel
4. **Four-state thermodynamic cycle**: `ddG = (E_bound_mut - E_unbound_mut) - (E_bound_wt - E_unbound_wt)` per member
5. **Ensemble averaging**: Final ddG = arithmetic mean across all members

Key design choices in flex ddG: no soft-rep ramping (standard full-weight scoring throughout), neighborhood-restricted repacking (8A radius + 1 sequence neighbor), rigid-body separation for unbound states (no re-relaxation), multi-cool simulated annealing for rotamer packing.

### Approach

Replace Rosetta-specific tools with Boundry's existing stack:
- **Backrub MC** -> **Restrained Langevin MD** (OpenMM native, arguably better physics)
- **Dunbrack rotamer packing** -> **LigandMPNN side-chain repacking** (neural network-based, context-aware)
- **Talaris2014/REF2015** -> **AMBER14 + GBn2 implicit solvent** (already used by Boundry)

### Conceptual Comparison

| Aspect | Current Boundry | Flex ddG | Planned Boundry |
|--------|----------------|----------|-----------------|
| Backbone sampling | None (L-BFGS only) | Backrub MC (35K trials) | Restrained Langevin MD |
| Ensemble size | 1 structure | 35 structures | Configurable (default 35) |
| WT/mutant scoring | Independent | Paired from same backbone | Paired from same backbone |
| Thermodynamic cycle | `dG = E_complex - E_sep` | `(bound_mut - unbound_mut) - (bound_wt - unbound_wt)` | Same four-state cycle |
| Minimization restraints | Unconstrained L-BFGS | Harmonic CA-CA pair distances | Harmonic CA-CA pair distances |
| Neighborhood repacking | Full structure | 8A around mutation + 1 seq neighbor | Same neighborhood restriction |
| Result aggregation | Single energy | Mean over ensemble | Mean + std over ensemble |
| Unbound state | Rigid separation, optional repack | Rigid separation only | Rigid separation only |

---

## Phase 1: Relaxer Enhancements

### 1.1 Add CA Atom Index Helper

**File:** `src/boundry/relaxer.py`

Add a private helper method to extract CA atom indices from an OpenMM Modeller. This is reused by both constrained minimization (1.2) and MD ensemble generation (1.3).

```python
def _get_ca_atom_indices(
    self, modeller: openmm_app.Modeller
) -> List[int]:
    """Return indices of C-alpha atoms in the modeller topology."""
    return [
        i
        for i, atom in enumerate(modeller.topology.atoms())
        if atom.name == "CA"
    ]
```

**Placement:** After `_add_restraints()` (currently ends at line 415).

### 1.2 Add CA-CA Pair Distance Restraint Minimization

**File:** `src/boundry/relaxer.py`

Add a new public method that performs L-BFGS minimization under harmonic CA-CA pair distance restraints. This differs fundamentally from the existing `_add_restraints()` method (line 392), which uses `CustomExternalForce` to pin each heavy atom to its initial **absolute position**. Flex ddG instead uses **pairwise distance** restraints between CA atoms -- this preserves relative backbone geometry while allowing global translation/rotation.

```python
def minimize_with_pair_restraints(
    self,
    pdb_string: str,
    ca_cutoff: float = 9.0,
    restraint_sd: float = 0.5,
) -> Tuple[str, dict, np.ndarray]:
    """L-BFGS minimization with harmonic CA-CA pair distance restraints.

    Matches the flex ddG initial constrained minimization protocol:
    harmonic restraints on all CA-CA pairs within ca_cutoff angstroms,
    with spring constant k = 1 / (restraint_sd^2).

    Args:
        pdb_string: PDB file contents as string.
        ca_cutoff: Maximum CA-CA distance (A) for adding restraints.
        restraint_sd: Standard deviation (A) of the harmonic restraint.

    Returns:
        Tuple of (relaxed_pdb_string, debug_info, violations).
        debug_info includes 'n_restraints', 'initial_energy', 'final_energy', 'rmsd'.
    """
```

**Implementation details:**

1. Prepare structure via PDBFixer (same pattern as `_relax_unconstrained`, lines 209-228: find missing residues, clear `missingResidues`, find/add missing atoms)
2. Create force field and Modeller, add hydrogens (same as lines 222-228)
3. Create system with `HBonds` constraints
4. Get CA atom indices via `_get_ca_atom_indices()`
5. Compute initial CA-CA distances from `modeller.positions`
6. Create `CustomBondForce("0.5 * k * (r - r0)^2")` with:
   - Global parameter `k = 1.0 / (restraint_sd ** 2)` (converted to OpenMM internal units: kJ/mol/nm^2)
   - Per-bond parameter `r0` (initial distance, in nm)
   - Add bonds only for pairs where initial distance < `ca_cutoff`
7. Add the force to the system
8. Create simulation, set positions, run `minimizeEnergy(maxIterations=self.config.max_iterations)`
9. Write output PDB and return (same pattern as `_relax_unconstrained` lines 256-286)

**Unit conversions:**
- Distances: user-facing API in angstroms, converted to nanometers for OpenMM
- Spring constant: `k = 1 / sd^2` in kcal/mol/A^2, converted to kJ/mol/nm^2 via `unit` module

**Placement:** After the `_get_ca_atom_indices()` helper.

### 1.3 Add MD Ensemble Generation

**File:** `src/boundry/relaxer.py`

Add a new public method that runs restrained Langevin dynamics and extracts snapshot conformations:

```python
def generate_ensemble(
    self,
    pdb_string: str,
    n_members: int = 35,
    temperature_kelvin: float = 300.0,
    total_steps: int = 50000,
    equilibration_steps: int = 5000,
    ca_cutoff: float = 9.0,
    restraint_sd: float = 0.5,
) -> List[str]:
    """Generate a conformational ensemble via restrained Langevin MD.

    Runs short molecular dynamics with CA-CA pair distance restraints
    to sample backbone conformational diversity around the input
    structure. This replaces Rosetta's backrub Monte Carlo sampling
    in the flex ddG protocol.

    Args:
        pdb_string: PDB file contents (should already be minimized).
        n_members: Number of ensemble members to generate.
        temperature_kelvin: Simulation temperature.
        total_steps: Total MD production steps (after equilibration).
        equilibration_steps: Equilibration steps (discarded).
        ca_cutoff: CA-CA pair restraint cutoff (A).
        restraint_sd: Restraint standard deviation (A).

    Returns:
        List of PDB strings, one per ensemble member.
    """
```

**Implementation details:**

1. Prepare structure (same PDBFixer + Modeller pattern)
2. Create AMBER14 system with `HBonds` constraints
3. Add CA-CA pair distance restraints (same scheme as Step 1.2)
4. Create `LangevinMiddleIntegrator`:
   - `temperature`: `temperature_kelvin * unit.kelvin`
   - `friction`: `1.0 / unit.picosecond`
   - `timestep`: `2.0 * unit.femtoseconds`
   - This is a **real** integrator for dynamics (unlike the dummy `LangevinIntegrator(0, 0.01, 0.0)` used in existing minimization code)
5. Create simulation, set positions
6. Run equilibration: `simulation.step(equilibration_steps)`
7. Compute snapshot interval: `interval = total_steps // n_members`
8. Production loop: for each member, run `simulation.step(interval)`, then extract positions and write PDB string via `openmm_app.PDBFile.writeFile()`
9. Return list of PDB strings

**Why `LangevinMiddleIntegrator`:** This is OpenMM's recommended integrator for Langevin dynamics sampling. The BAOAB splitting scheme has better configurational sampling properties than the standard `LangevinIntegrator`. Friction coefficient of 1/ps provides good thermalization without over-damping conformational transitions.

**Total simulation time:** 50,000 steps * 2 fs = 100 ps. With CA-CA pair restraints, this is sufficient to explore local backbone torsional space while keeping the overall fold stable. Comparable to flex ddG's 35,000 backrub trials which sample ~3-12 residue backbone segments.

**Placement:** After `minimize_with_pair_restraints()`.

---

## Phase 2: General Point Mutation Utility

### 2.1 Generalize `mutate_to_alanine` into `mutate_residue`

**File:** `src/boundry/interface_position_energetics.py`

Refactor the existing `mutate_to_alanine()` function (lines 143-194) into a general `mutate_residue()` that handles any target amino acid. The existing function already implements the core logic: scan PDB lines, match the target residue by chain/resnum/icode, strip non-backbone atoms, rename the residue. The generalization extends this to arbitrary target residues.

**New function:**

```python
# Backbone atoms retained for all mutations
_BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT", "H", "HA"}

# Standard amino acid 1-to-3 letter code mapping
_AA_1TO3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}
_AA_3TO1 = {v: k for k, v in _AA_1TO3.items()}

def mutate_residue(
    pdb_string: str,
    chain_id: str,
    resnum: int,
    target_aa: str,
    icode: str = "",
) -> str:
    """Mutate a single residue to any amino acid in a PDB string.

    Strips side-chain atoms beyond the backbone (+ CB for non-GLY targets)
    and renames the residue. PDBFixer rebuilds the correct side-chain
    geometry downstream during energy evaluation or minimization.

    Args:
        pdb_string: PDB file contents.
        chain_id: Chain containing the target residue.
        resnum: Residue sequence number.
        target_aa: Target amino acid (1-letter or 3-letter code).
        icode: Insertion code (empty string if none).

    Returns:
        Modified PDB string with the residue mutated.
    """
```

**Logic:**
1. Normalize `target_aa`: accept both 1-letter ("K") and 3-letter ("LYS") codes, convert to 3-letter for PDB output
2. Determine retained atom set: `_BACKBONE_ATOMS` for GLY targets, `_BACKBONE_ATOMS | {"CB"}` for all others (same as current `_ALA_ATOMS` minus the ALA-specific set)
3. Scan PDB lines (same pattern as current `mutate_to_alanine`, lines 165-192)
4. For matching residue lines: skip atoms not in retained set, rename residue to target 3-letter code

**Wrapper for backward compatibility:**

```python
def mutate_to_alanine(
    pdb_string: str,
    chain_id: str,
    resnum: int,
    icode: str = "",
) -> str:
    """Mutate a single residue to alanine in a PDB string.

    Thin wrapper around :func:`mutate_residue`. Deletes side-chain atoms
    beyond the alanine atom set and renames the residue to ``ALA``.
    """
    return mutate_residue(pdb_string, chain_id, resnum, "ALA", icode)
```

The existing `_ALA_ATOMS` constant and all callers of `mutate_to_alanine()` (in `compute_alanine_scan`, line 366) remain unchanged in behavior.

---

## Phase 3: Configuration

### 3.1 Add `FlexDdGConfig` Dataclass

**File:** `src/boundry/config.py`

Add after `OptimizeConfig` (after line 135):

```python
@dataclass
class FlexDdGConfig:
    """Configuration for flex ddG-style ensemble ddG calculation.

    Implements a protocol conceptually aligned with Rosetta's flex ddG
    (Barlow et al., JPCB 2018) using OpenMM for energy evaluation and
    conformational sampling, and LigandMPNN for side-chain repacking.
    """

    # Ensemble generation (flex ddG Stage 2)
    n_ensemble: int = 35               # Number of ensemble members
    md_temperature: float = 300.0      # Simulation temperature (Kelvin)
    md_total_steps: int = 50000        # Production MD steps
    md_equilibration_steps: int = 5000 # Equilibration steps (discarded)

    # CA-CA pair restraints (flex ddG Stages 1-2)
    ca_cutoff: float = 9.0             # Maximum CA-CA distance for restraints (A)
    restraint_sd: float = 0.5          # Restraint standard deviation (A)

    # Neighborhood repacking (flex ddG Stages 3-4)
    neighborhood_radius: float = 8.0   # Repack radius around mutation site (A)
    sequence_window: int = 1           # +/- residues in sequence to include

    # Sub-configs for Relaxer/Designer
    design: DesignConfig = field(default_factory=DesignConfig)
    relax: RelaxConfig = field(default_factory=RelaxConfig)

    # Interface definition
    chain_pairs: Optional[List[Tuple[str, str]]] = None  # Required at runtime
    distance_cutoff: float = 8.0       # Interface residue detection cutoff (A)

    # Execution
    seed: Optional[int] = None
    workers: int = 1                   # Parallel workers for ensemble members
    quiet: bool = True

    # Ensemble caching
    cache_ensemble: bool = False       # Save/load ensemble PDBs to disk
    ensemble_dir: Optional[Path] = None
```

**Defaults rationale:**
- `n_ensemble=35`: matches flex ddG production recommendation (performance plateaus around 30-35)
- `md_total_steps=50000` at 2fs timestep = 100ps: sufficient for local backbone torsional sampling under restraints
- `ca_cutoff=9.0`, `restraint_sd=0.5`: identical to flex ddG parameters
- `neighborhood_radius=8.0`, `sequence_window=1`: identical to flex ddG's Neighborhood + PrimarySequenceNeighborhood selectors

---

## Phase 4: Core Flex ddG Module

### 4.1 Create `flex_ddg.py`

**File:** `src/boundry/flex_ddg.py` (new)

This is the main implementation module. It contains data classes, the neighborhood builder, the core compute function, and parallel worker types.

#### 4.1a Data Classes

```python
@dataclass
class MutationSpec:
    """A single point mutation specification."""
    chain_id: str
    residue_number: int
    wt_residue: str         # 3-letter code (e.g. "ALA")
    mut_residue: str        # 3-letter code (e.g. "LYS")
    insertion_code: str = ""

    def __str__(self) -> str:
        wt1 = _AA_3TO1.get(self.wt_residue, "?")
        mut1 = _AA_3TO1.get(self.mut_residue, "?")
        return f"{self.chain_id}:{wt1}{self.residue_number}{mut1}"


@dataclass
class EnsembleMemberResult:
    """Four-state scoring results for one ensemble member."""
    member_index: int
    bound_wt_energy: Optional[float] = None
    unbound_wt_energy: Optional[float] = None
    bound_mut_energy: Optional[float] = None
    unbound_mut_energy: Optional[float] = None

    @property
    def dG_wt(self) -> Optional[float]:
        """WT binding energy: E_bound_wt - E_unbound_wt."""
        if self.bound_wt_energy is not None and self.unbound_wt_energy is not None:
            return self.bound_wt_energy - self.unbound_wt_energy
        return None

    @property
    def dG_mut(self) -> Optional[float]:
        """Mutant binding energy: E_bound_mut - E_unbound_mut."""
        if self.bound_mut_energy is not None and self.unbound_mut_energy is not None:
            return self.bound_mut_energy - self.unbound_mut_energy
        return None

    @property
    def ddG(self) -> Optional[float]:
        """Change in binding energy: dG_mut - dG_wt (positive = destabilizing)."""
        if self.dG_wt is not None and self.dG_mut is not None:
            return self.dG_mut - self.dG_wt
        return None


@dataclass
class FlexDdGResult:
    """Complete flex ddG calculation result with ensemble statistics."""
    mutations: List[MutationSpec]
    member_results: List[EnsembleMemberResult]
    mean_ddG: Optional[float] = None
    std_ddG: Optional[float] = None
    mean_dG_wt: Optional[float] = None
    mean_dG_mut: Optional[float] = None
    n_successful: int = 0
    n_ensemble: int = 0
    ensemble_ddGs: List[float] = field(default_factory=list)
```

#### 4.1b Neighborhood DesignSpec Builder

```python
def build_neighborhood_spec(
    pdb_string: str,
    mutation_sites: List[Tuple[str, int, str]],  # (chain, resnum, icode)
    neighborhood_radius: float = 8.0,
    sequence_window: int = 1,
) -> "DesignSpec":
    """Build a DesignSpec that restricts repacking to the mutation neighborhood.

    Mirrors flex ddG's Neighborhood (8A) + PrimarySequenceNeighborhood (+/-1)
    residue selectors. Neighborhood residues get NATAA (repack with same
    identity), all others get NATRO (completely frozen).

    Args:
        pdb_string: PDB file contents.
        mutation_sites: List of (chain_id, residue_number, insertion_code).
        neighborhood_radius: Radius (A) around mutation site CA atoms.
        sequence_window: +/- residues in sequence to include.

    Returns:
        DesignSpec with per-residue modes.
    """
```

**Implementation:**
1. Parse PDB with BioPython `PDBParser`
2. For each mutation site, find the CA atom coordinates
3. For every residue in the structure, check if any heavy atom is within `neighborhood_radius` of any mutation site CA
4. For each selected residue, extend by `sequence_window` positions in each direction along the same chain (by sorting residues per chain by residue number and taking neighbors)
5. Import `DesignSpec` from `boundry.resfile` (lazy import)
6. Build a DesignSpec where:
   - Neighborhood residues (including extensions): `mode = "NATAA"` (same amino acid, optimized rotamer)
   - All other residues: `mode = "NATRO"` (completely fixed, no repacking)
7. Return the DesignSpec

**Note:** The `DesignSpec` and `ResidueMode` types from `boundry/resfile.py` already support `NATAA` and `NATRO` modes, so this is a natural fit.

#### 4.1c Parallel Worker Types and Function

```python
@dataclass(frozen=True)
class _FlexDdGMemberTask:
    """Pickle-safe task for processing one ensemble member.

    Follows the same pattern as _BeamExpansionTask in optimize.py
    and ScanTask in _parallel.py: all data is serializable,
    heavy objects (Relaxer, Designer) are lazily initialized in
    the worker function.
    """
    member_index: int
    ensemble_pdb: str
    mutations: Tuple[Tuple[str, int, str, str, str], ...]  # (chain, resnum, icode, wt_3, mut_3)
    neighborhood_spec_data: Dict[str, Any]  # serialized DesignSpec
    chain_pairs: Tuple[Tuple[str, str], ...]
    distance_cutoff: float
    relax_config_dict: Dict[str, Any]
    design_config_dict: Dict[str, Any]
    ca_cutoff: float
    restraint_sd: float
    quiet: bool


@dataclass
class _FlexDdGMemberResult:
    """Result from processing one ensemble member."""
    member_index: int
    bound_wt_energy: Optional[float] = None
    unbound_wt_energy: Optional[float] = None
    bound_mut_energy: Optional[float] = None
    unbound_mut_energy: Optional[float] = None
    error: Optional[str] = None


# Module-level cache for worker process reuse (same pattern as _parallel.py)
_worker_cache: Dict[str, Any] = {}


def _process_ensemble_member(
    task: _FlexDdGMemberTask,
) -> _FlexDdGMemberResult:
    """Top-level pickle-safe worker for one ensemble member.

    Implements flex ddG Stages 3-5 for a single backbone conformation:
    1. WT branch: repack neighborhood -> constrained min -> score 4 states
    2. Mutant branch: apply mutations -> repack neighborhood -> constrained min -> score 4 states
    """
```

**Worker implementation:**
1. Lazy-initialize Relaxer and Designer from config dicts (with fingerprint-keyed caching, same as `_execute_scan_worker` in `_parallel.py:302`)
2. Reconstruct DesignSpec from `neighborhood_spec_data`
3. **WT branch:**
   a. Repack the ensemble PDB using `designer.repack()` with the neighborhood DesignSpec
   b. Constrained minimize via `relaxer.minimize_with_pair_restraints()`
   c. Score bound state: `relaxer.get_energy_breakdown(wt_bound_pdb)["total_energy"]`
   d. For each chain group (from `_get_interface_chain_groups(chain_pairs)`): extract chains via `extract_chain()`, score: `relaxer.get_energy_breakdown(chain_pdb)["total_energy"]`
   e. `unbound_wt_energy = sum(chain_group_energies)`
4. **Mutant branch:**
   a. Apply mutations to the ensemble PDB via `mutate_residue()` for each mutation
   b. Build a mutant DesignSpec: same neighborhood, but mutation sites get the target amino acid identity (modify the DesignSpec to use PIKAA for mutation sites)
   c. Repack via `designer.repack()` with mutant DesignSpec
   d. Constrained minimize
   e. Score bound and unbound states (same as WT branch)
5. Return `_FlexDdGMemberResult` with all four energies

**Important detail for mutant repacking:** After applying mutations at the PDB level (stripping side chains), the neighborhood DesignSpec is modified for the mutation sites specifically:
- Mutation sites: `mode = "PIKAA"` with the target amino acid, so LigandMPNN places the correct mutant rotamer
- Other neighborhood residues: `mode = "NATAA"` (repack to native identity)
- Non-neighborhood: `mode = "NATRO"` (frozen)

This matches flex ddG's approach where the resfile has `NATAA` as default and `PIKAA X` for mutation sites, with non-neighborhood residues frozen.

#### 4.1d Core `compute_flex_ddg()` Function

```python
def compute_flex_ddg(
    pdb_string: str,
    mutations: List[MutationSpec],
    config: "FlexDdGConfig",
    relaxer: Optional["Relaxer"] = None,
    designer: Optional["Designer"] = None,
    pool: Optional["WorkPool"] = None,
) -> FlexDdGResult:
    """Compute ensemble-averaged ddG via flex ddG-style protocol.

    Protocol stages (matching Barlow et al. 2018):
    1. Constrained minimization with CA-CA pair distance restraints
    2. Ensemble generation via restrained Langevin MD
    3. Per-member paired WT/mutant scoring with four-state thermodynamic cycle
    4. Ensemble averaging of ddG values

    Args:
        pdb_string: Input PDB string (protein complex).
        mutations: List of point mutations to evaluate.
        config: FlexDdGConfig with all parameters.
        relaxer: Pre-configured Relaxer (created from config.relax if None).
        designer: Pre-configured Designer (created from config.design if None).
        pool: WorkPool for parallel ensemble member processing.

    Returns:
        FlexDdGResult with ensemble-averaged ddG and per-member details.
    """
```

**Implementation:**

1. Create Relaxer/Designer if not provided (lazy imports, same pattern as `operations.py`)

2. **Stage 1 -- Constrained minimization:**
   ```python
   minimized_pdb, min_info, _ = relaxer.minimize_with_pair_restraints(
       pdb_string,
       ca_cutoff=config.ca_cutoff,
       restraint_sd=config.restraint_sd,
   )
   ```

3. **Stage 2 -- Ensemble generation:**
   ```python
   # Check cache first
   if config.cache_ensemble and config.ensemble_dir and _ensemble_cached(config.ensemble_dir, config.n_ensemble):
       ensemble_pdbs = _load_ensemble(config.ensemble_dir)
   else:
       ensemble_pdbs = relaxer.generate_ensemble(
           minimized_pdb,
           n_members=config.n_ensemble,
           temperature_kelvin=config.md_temperature,
           total_steps=config.md_total_steps,
           equilibration_steps=config.md_equilibration_steps,
           ca_cutoff=config.ca_cutoff,
           restraint_sd=config.restraint_sd,
       )
       if config.cache_ensemble and config.ensemble_dir:
           _save_ensemble(ensemble_pdbs, config.ensemble_dir)
   ```

4. **Build neighborhood DesignSpec:**
   ```python
   mutation_sites = [(m.chain_id, m.residue_number, m.insertion_code) for m in mutations]
   neighborhood_spec = build_neighborhood_spec(
       minimized_pdb, mutation_sites,
       neighborhood_radius=config.neighborhood_radius,
       sequence_window=config.sequence_window,
   )
   ```

5. **Stage 3-5 -- Per-member scoring:**
   Build `_FlexDdGMemberTask` for each ensemble member, dispatch via `pool.map(_process_ensemble_member, tasks)` if pool is active, otherwise process sequentially.

6. **Stage 6 -- Ensemble averaging:**
   ```python
   valid_ddGs = [r.ddG for r in member_results if r.ddG is not None]
   result = FlexDdGResult(
       mutations=mutations,
       member_results=member_results,
       mean_ddG=statistics.mean(valid_ddGs) if valid_ddGs else None,
       std_ddG=statistics.stdev(valid_ddGs) if len(valid_ddGs) > 1 else None,
       mean_dG_wt=statistics.mean([r.dG_wt for r in member_results if r.dG_wt is not None]),
       mean_dG_mut=statistics.mean([r.dG_mut for r in member_results if r.dG_mut is not None]),
       n_successful=len(valid_ddGs),
       n_ensemble=config.n_ensemble,
       ensemble_ddGs=valid_ddGs,
   )
   ```

#### 4.1e Mutation String Parser

```python
def parse_mutations(mutation_string: str, pdb_string: str) -> List[MutationSpec]:
    """Parse a mutation specification string into MutationSpec objects.

    Format: "H:A100K,L:T52S" where each entry is chain:WTresnumMUT.
    WT and MUT are 1-letter amino acid codes, resnum is the PDB residue number.

    Validates that WT residues match the input PDB structure.

    Args:
        mutation_string: Comma-separated mutation specs.
        pdb_string: Input PDB for WT residue validation.

    Returns:
        List of MutationSpec objects.

    Raises:
        ValueError: If format is invalid or WT residue doesn't match PDB.
    """
```

#### 4.1f Ensemble Cache Helpers

```python
def _ensemble_cached(ensemble_dir: Path, n_members: int) -> bool:
    """Check if a complete ensemble cache exists."""

def _save_ensemble(ensemble_pdbs: List[str], ensemble_dir: Path) -> None:
    """Save ensemble PDBs to disk."""

def _load_ensemble(ensemble_dir: Path) -> List[str]:
    """Load cached ensemble PDBs from disk."""
```

---

## Phase 5: Operation and CLI Integration

### 5.1 Add `flex_ddg` Operation

**File:** `src/boundry/operations.py`

Add a new operation function following the established pattern:

```python
def flex_ddg(
    structure: StructureInput,
    mutations: Optional[List[Dict[str, str]]] = None,
    mutation_string: Optional[str] = None,
    config: Optional["FlexDdGConfig"] = None,
    output_path: Optional[Path] = None,
) -> "Structure":
    """Compute ensemble-averaged ddG of mutation via flex ddG protocol.

    Uses restrained Langevin MD to generate a conformational ensemble,
    then evaluates WT and mutant from each backbone conformation using
    a four-state thermodynamic cycle. Returns the input structure with
    flex ddG results in metadata.

    Args:
        structure: Input structure (file path, PDB string, or Structure).
        mutations: List of mutation dicts with keys: chain, residue_number,
            wt_residue, mut_residue, and optionally insertion_code.
        mutation_string: Alternative mutation spec as string ("H:A100K,L:T52S").
        config: FlexDdGConfig (must include chain_pairs).
        output_path: Optional output path for PDB + results JSON.

    Returns:
        Structure with flex_ddg results in metadata["flex_ddg"].
    """
```

**Implementation:** Lazy imports, `_resolve_input()`, create Relaxer/Designer, validate chain_pairs, call `compute_flex_ddg()`, build `Structure` with metadata, optionally write output.

**Metadata structure:**
```python
metadata["flex_ddg"] = {
    "mean_ddG": result.mean_ddG,
    "std_ddG": result.std_ddG,
    "mean_dG_wt": result.mean_dG_wt,
    "mean_dG_mut": result.mean_dG_mut,
    "n_ensemble": result.n_ensemble,
    "n_successful": result.n_successful,
    "mutations": [{"chain": m.chain_id, "residue_number": m.residue_number,
                    "wt_residue": m.wt_residue, "mut_residue": m.mut_residue} for m in result.mutations],
    "ensemble_ddGs": result.ensemble_ddGs,
    "per_member": [{"index": r.member_index, "ddG": r.ddG, "dG_wt": r.dG_wt, "dG_mut": r.dG_mut}
                   for r in result.member_results],
}
```

### 5.2 Add CLI Subcommand

**File:** `src/boundry/cli.py`

```python
@app.command("flex-ddg")
def flex_ddg_cmd(
    input_file: Path = typer.Argument(..., help="Input PDB/CIF file"),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    interface: str = typer.Option(..., "--interface", "-i", help="Chain pairs (e.g. H:A,L:A)"),
    mutations: str = typer.Option(..., "--mutations", "-m", help="Mutations (e.g. H:A100K,L:T52S)"),
    ensemble_size: int = typer.Option(35, "--ensemble-size", "-n", help="Ensemble members"),
    md_steps: int = typer.Option(50000, "--md-steps", help="MD production steps"),
    md_temperature: float = typer.Option(300.0, "--md-temperature", help="MD temperature (K)"),
    neighborhood_radius: float = typer.Option(8.0, "--neighborhood-radius", help="Repack neighborhood (A)"),
    workers: int = typer.Option(1, "--workers", "-w", help="Parallel workers"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    cache_ensemble: bool = typer.Option(False, "--cache-ensemble", help="Cache ensemble to disk"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Predict ddG of mutation using ensemble-based flex ddG protocol."""
```

**Output:** Writes to the output directory:
- `input_minimized.pdb`: Constrained-minimized input structure
- `flex_ddg_results.json`: Full results including per-member details
- `ensemble/member_{i}.pdb` (if `--cache-ensemble`): Ensemble PDB files

### 5.3 Register Exports

**File:** `src/boundry/__init__.py`

Add to the import section and `__all__`:
```python
from boundry.config import FlexDdGConfig
from boundry.flex_ddg import FlexDdGResult, MutationSpec, EnsembleMemberResult, compute_flex_ddg
from boundry.operations import flex_ddg
```

---

## Phase 6: Testing

### 6.1 Test Data

**New fixture in `tests/conftest.py`:**

The existing `small_peptide_pdb_string` fixture provides a single-chain structure, which is insufficient for interface testing. The existing `antibody_antigen_pdb_string` fixture provides a multi-chain antibody-antigen complex suitable for flex ddG testing.

Add a new fixture that provides a minimal two-chain complex for fast unit tests:

```python
@pytest.fixture
def two_chain_complex_pdb_string():
    """Minimal two-chain complex for flex ddG unit tests.

    A small helical dimer (~20 residues per chain) with a defined
    interface. Small enough for fast mock-based tests.
    """
    # Return a PDB string with chains A and B forming a dimer interface
```

This can be constructed programmatically (two short helices packed together) or extracted from a small known structure. The key requirement is that it has:
- Two chains with a clear interface (>5 interface residues)
- Small enough that energy evaluations are fast
- Known residue identities for mutation testing

### 6.2 New Test File: `tests/test_flex_ddg.py`

**Test classes and methods:**

#### Configuration Tests

```python
class TestFlexDdGConfig:
    def test_default_values(self):
        """FlexDdGConfig has correct defaults matching flex ddG paper."""

    def test_custom_values(self):
        """FlexDdGConfig accepts custom parameter values."""

    def test_nested_sub_configs(self):
        """FlexDdGConfig correctly propagates DesignConfig and RelaxConfig."""
```

#### MutationSpec Tests

```python
class TestMutationSpec:
    def test_str_representation(self):
        """MutationSpec.__str__ produces 'chain:WTresnumMUT' format."""

    def test_mutation_string_parsing(self):
        """parse_mutations correctly parses 'H:A100K,L:T52S' format."""

    def test_invalid_mutation_string(self):
        """parse_mutations raises ValueError for invalid formats."""

    def test_wt_validation_against_pdb(self):
        """parse_mutations validates WT residue matches input PDB."""
```

#### EnsembleMemberResult Tests

```python
class TestEnsembleMemberResult:
    def test_dG_wt_computation(self):
        """dG_wt = bound_wt - unbound_wt."""
        result = EnsembleMemberResult(
            member_index=0,
            bound_wt_energy=-500.0,
            unbound_wt_energy=-480.0,
        )
        assert result.dG_wt == pytest.approx(-20.0)

    def test_dG_mut_computation(self):
        """dG_mut = bound_mut - unbound_mut."""

    def test_ddG_computation(self):
        """ddG = dG_mut - dG_wt (positive = destabilizing)."""
        result = EnsembleMemberResult(
            member_index=0,
            bound_wt_energy=-500.0, unbound_wt_energy=-480.0,   # dG_wt = -20
            bound_mut_energy=-490.0, unbound_mut_energy=-478.0,  # dG_mut = -12
        )
        assert result.ddG == pytest.approx(8.0)  # destabilizing

    def test_ddG_none_when_incomplete(self):
        """ddG is None when any energy is missing."""

    def test_four_state_cancellation(self):
        """Verify ddG = (bound_mut + unbound_wt) - (unbound_mut + bound_wt).

        The algebraic equivalence used by flex ddG's analysis script.
        """
```

#### Neighborhood DesignSpec Tests

```python
class TestBuildNeighborhoodSpec:
    def test_returns_design_spec(self, two_chain_complex_pdb_string):
        """Returns a valid DesignSpec object."""

    def test_neighborhood_residues_are_nataa(self, two_chain_complex_pdb_string):
        """Residues within radius have NATAA mode."""

    def test_non_neighborhood_residues_are_natro(self, two_chain_complex_pdb_string):
        """Residues outside radius have NATRO mode."""

    def test_sequence_window_extends_selection(self, two_chain_complex_pdb_string):
        """Sequence neighbors are included even if beyond distance cutoff."""

    def test_multiple_mutation_sites(self, two_chain_complex_pdb_string):
        """Neighborhood includes union of all mutation site neighborhoods."""

    def test_zero_radius_returns_only_mutation_sites(self, two_chain_complex_pdb_string):
        """With radius=0 and window=0, only mutation site(s) are NATAA."""
```

#### General Mutation Tests

```python
class TestMutateResidue:
    def test_mutate_to_lysine(self, small_peptide_pdb_string):
        """Residue is renamed to LYS and side-chain atoms stripped."""

    def test_mutate_to_glycine_strips_cb(self, small_peptide_pdb_string):
        """GLY mutation strips CB atom as well."""

    def test_backbone_atoms_preserved(self, small_peptide_pdb_string):
        """N, CA, C, O are always preserved after mutation."""

    def test_accepts_one_letter_code(self, small_peptide_pdb_string):
        """Accepts 'K' as well as 'LYS'."""

    def test_accepts_three_letter_code(self, small_peptide_pdb_string):
        """Accepts 'LYS' as well as 'K'."""

    def test_non_target_residues_unchanged(self, small_peptide_pdb_string):
        """Other residues in the PDB are untouched."""

    def test_mutate_to_alanine_wrapper_still_works(self, small_peptide_pdb_string):
        """mutate_to_alanine() produces identical output to mutate_residue(..., 'ALA')."""
```

#### Core compute_flex_ddg Tests (Mocked)

```python
class TestComputeFlexDdG:
    """Unit tests with mocked Relaxer/Designer."""

    def _make_mock_relaxer(self):
        """Create a Relaxer mock with deterministic energy returns."""
        relaxer = MagicMock()
        relaxer.minimize_with_pair_restraints.return_value = ("pdb", {}, np.zeros(0))
        relaxer.generate_ensemble.return_value = ["pdb_1", "pdb_2", "pdb_3"]
        relaxer.get_energy_breakdown.return_value = {"total_energy": -100.0}
        relaxer.relax.return_value = ("pdb", {}, np.zeros(0))
        relaxer.config = RelaxConfig()
        return relaxer

    def _make_mock_designer(self):
        """Create a Designer mock that returns input unchanged."""
        designer = MagicMock()
        designer.repack.return_value = MagicMock()
        designer.result_to_pdb_string.return_value = "pdb"
        designer.config = DesignConfig()
        return designer

    def test_calls_constrained_minimization(self):
        """Stage 1: minimize_with_pair_restraints is called on input."""

    def test_calls_ensemble_generation(self):
        """Stage 2: generate_ensemble is called on minimized structure."""

    def test_ensemble_size_matches_config(self):
        """generate_ensemble receives n_members from config."""

    def test_four_state_scoring_per_member(self):
        """Stages 3-5: each member produces bound_wt, unbound_wt, bound_mut, unbound_mut."""

    def test_ensemble_averaging(self):
        """Stage 6: mean and std computed from per-member ddGs."""

    def test_handles_failed_members(self):
        """Failed ensemble members are excluded from averaging."""

    def test_result_contains_all_mutations(self):
        """FlexDdGResult.mutations matches input mutation list."""

    def test_ensemble_caching_saves(self, tmp_path):
        """With cache_ensemble=True, ensemble PDBs are saved to disk."""

    def test_ensemble_caching_loads(self, tmp_path):
        """With cache_ensemble=True, existing cache is loaded instead of re-running MD."""
```

#### Parallel Worker Tests

```python
class TestFlexDdGWorker:
    def test_task_is_pickle_safe(self):
        """_FlexDdGMemberTask can be pickled and unpickled."""
        task = _FlexDdGMemberTask(
            member_index=0, ensemble_pdb="ATOM...", mutations=(("A", 100, "", "ALA", "LYS"),),
            neighborhood_spec_data={}, chain_pairs=(("H", "A"),),
            distance_cutoff=8.0, relax_config_dict={}, design_config_dict={},
            ca_cutoff=9.0, restraint_sd=0.5, quiet=True,
        )
        import pickle
        roundtripped = pickle.loads(pickle.dumps(task))
        assert roundtripped.member_index == 0

    def test_result_is_pickle_safe(self):
        """_FlexDdGMemberResult can be pickled and unpickled."""
```

#### CLI Tests

```python
class TestFlexDdGCli:
    def test_flex_ddg_subcommand_exists(self):
        """'flex-ddg' appears in CLI help."""

    @patch("boundry.operations.flex_ddg")
    def test_flex_ddg_parses_mutations(self, mock_op, tmp_path):
        """CLI correctly parses --mutations flag and passes to operation."""

    @patch("boundry.operations.flex_ddg")
    def test_flex_ddg_parses_interface(self, mock_op, tmp_path):
        """CLI correctly parses --interface flag into chain_pairs."""

    @patch("boundry.operations.flex_ddg")
    def test_flex_ddg_custom_ensemble_size(self, mock_op, tmp_path):
        """--ensemble-size flag is passed through to config."""
```

### 6.3 Integration Tests (Require OpenMM)

**File:** `tests/test_flex_ddg.py` (marked with `@pytest.mark.integration`)

```python
@pytest.mark.integration
class TestRelaxerPairRestraints:
    def test_minimization_preserves_ca_geometry(self, antibody_antigen_pdb_string):
        """CA RMSD after constrained minimization is < 1.0 A."""

    def test_restraint_count_matches_pairs(self, antibody_antigen_pdb_string):
        """Number of restraints matches CA-CA pairs within cutoff."""

    def test_energy_decreases(self, antibody_antigen_pdb_string):
        """Final energy is lower than initial energy."""


@pytest.mark.integration
class TestRelaxerEnsembleGeneration:
    def test_returns_correct_count(self, antibody_antigen_pdb_string):
        """Returns exactly n_members PDB strings."""

    def test_members_are_distinct(self, antibody_antigen_pdb_string):
        """Ensemble members have different backbone coordinates."""

    def test_ca_rmsd_from_input_is_bounded(self, antibody_antigen_pdb_string):
        """Each member has CA RMSD < 2.0 A from input (restraints working)."""

    def test_small_ensemble_for_speed(self, antibody_antigen_pdb_string):
        """3-member ensemble with 1000 steps completes in < 60s."""


@pytest.mark.integration
class TestFlexDdGEndToEnd:
    def test_alanine_mutation_produces_result(self, antibody_antigen_pdb_string):
        """End-to-end: single alanine mutation produces valid FlexDdGResult."""

    def test_destabilizing_mutation_positive_ddg(self, antibody_antigen_pdb_string):
        """Mutation of a core hydrophobic to charged residue has positive ddG."""
```

### 6.4 Modifications to Existing Test Files

**`tests/test_operations.py`:**
```python
class TestFlexDdGOperation:
    @patch("boundry.flex_ddg.compute_flex_ddg")
    def test_returns_structure(self, mock_compute):
        """flex_ddg() returns a Structure with metadata."""

    @patch("boundry.flex_ddg.compute_flex_ddg")
    def test_metadata_contains_flex_ddg_key(self, mock_compute):
        """Returned Structure.metadata has 'flex_ddg' key."""
```

**`tests/test_cli.py`:**
```python
class TestFlexDdGSubcommand:
    # Same tests as TestFlexDdGCli in test_flex_ddg.py
    # (can be consolidated into one location)
```

---

## File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `src/boundry/relaxer.py` | MODIFY | Add `_get_ca_atom_indices()`, `minimize_with_pair_restraints()`, `generate_ensemble()` |
| `src/boundry/interface_position_energetics.py` | MODIFY | Add `mutate_residue()`, refactor `mutate_to_alanine()` to wrapper |
| `src/boundry/config.py` | MODIFY | Add `FlexDdGConfig` dataclass |
| `src/boundry/flex_ddg.py` | NEW | Core module: data classes, neighborhood builder, compute function, parallel worker |
| `src/boundry/operations.py` | MODIFY | Add `flex_ddg()` operation function |
| `src/boundry/cli.py` | MODIFY | Add `flex-ddg` CLI subcommand |
| `src/boundry/__init__.py` | MODIFY | Export new types and functions |
| `tests/conftest.py` | MODIFY | Add `two_chain_complex_pdb_string` fixture |
| `tests/test_flex_ddg.py` | NEW | Comprehensive unit and integration tests |
| `tests/test_operations.py` | MODIFY | Add flex_ddg operation tests |
| `tests/test_cli.py` | MODIFY | Add flex-ddg CLI tests |
