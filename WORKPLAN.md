# ddG-Aligned Scoring Pipeline: Revised Technical Implementation Plan

## Background

### Problem

Boundry's current interface scoring in `optimize` uses single-structure rigid chain extraction (`E_complex - sum(E_separated)`) without ensemble averaging. This is fast but sensitive to one conformation and not conceptually aligned with flex-ddG-style paired ensemble scoring.

### Target

Implement a ddG scoring pipeline that is conceptually aligned with Rosetta flex-ddG while remaining native to Boundry/OpenMM/LigandMPNN:

1. Constrained minimization with CA-CA pair restraints (9 A cutoff, 0.5 A SD)
2. Independent restrained OpenMM ensemble generation around mutation neighborhoods (backrub-like concept)
3. Paired WT/mutant evaluation from the same backbone member
4. Four-state thermodynamic cycle per member
5. Arithmetic ensemble averaging
6. Strict rigid-body separation for unbound scoring (no unbound repack/min)

### Scope (Updated)

- Add a standalone CLI command named `ddg`
- Use the same ddG engine as the interface scoring backend in `optimize` (default)
- Keep focused commands (`relax`, `minimize`, etc.) for advanced convenience workflows

---

## Alignment Corrections (From Prior Draft)

This revision explicitly addresses the previously identified major deviations.

### A. Ensemble sampling must be local, independent, and OpenMM-native

**Issue in prior draft:** one restrained MD trajectory with evenly spaced snapshots.

**Revision:** generate members as **independent short restrained OpenMM trajectories** seeded per member, with mutation-neighborhood-focused restraints/bias and per-member checkpoint minimization.

This keeps the concept close to backrub sampling (local neighborhood exploration + independent members) without introducing a custom Monte Carlo/backrub engine.

### B. Unbound scoring must be strict rigid-body separation from the same pose

**Issue in prior draft:** extracted-chain rescoring as separate systems.

**Revision:** score unbound by translating one interface side away from the other in the **same complex pose**, then rescore once. No unbound repack/minimize.

### C. Neighborhood-only repacking must be enforced in code

**Issue in prior draft:** plan relied on `DesignSpec`, but `Designer.repack()` currently repacks everything.

**Revision:** explicitly modify `src/boundry/designer.py` so `repack()` honors `DesignSpec`/mask, with `NATRO` fixed and only neighborhood residues packable.

### D. Sampling/minimization/scoring energy model consistency

**Issue in prior draft:** minimization/MD and scoring were implicitly allowed to use different system definitions.

**Revision:** all ddG protocol stages use a shared system-construction path in `Relaxer` controlled by one force-field/solvation policy from `DdGConfig`.

### E. Published-comparison mode

**Issue in prior draft:** defaults and aggregation details did not cleanly expose paper-comparable settings.

**Revision:** support practical defaults and explicit `paper_mode` preset (`n_ensemble=50` with higher trajectory sampling depth), with optional post-hoc model sorting/selection for analysis parity.

### F. Scope mismatch (`flex_ddg` standalone only)

**Issue in prior draft:** new operation/CLI not integrated into `optimize`.

**Revision:** rename to `ddg`, expose CLI `boundry ddg`, and route `optimize` interface scoring + position scoring through the ddG backend by default.

---

## Protocol Mapping

| Aspect | flex-ddG concept | Revised Boundry ddG |
|---|---|---|
| Initial minimization | CA pair restraints (9 A, 0.5 A) | Same |
| Ensemble sampling | Backrub local MC | Independent restrained OpenMM trajectories (backrub-like neighborhood focus) |
| Member independence | Independent trajectories | Independent seeded trajectories |
| WT/mut pairing | Same backbone member | Same |
| Repacking scope | 8 A + seq +/-1 | Same neighborhood selectors |
| Unbound state | Rigid-body separated, no further relax | Same |
| ddG aggregation | Arithmetic mean | Same |
| Primary entrypoint | Rosetta protocol scripts | `optimize` (default) + `ddg` CLI |

---

## Phase 1: Relaxer and Sampling Enhancements

### 1.1 Shared system-construction path

**File:** `src/boundry/relaxer.py`

Add internal helper used by minimization, local sampling, and scoring prep:

```python
def _build_system_for_ddg(
    self,
    topology,
    positions,
    *,
    implicit_solvent: bool,
    constraints=openmm_app.HBonds,
):
    ...
```

Goals:
- Single force-field + solvent policy for all ddG stages
- Eliminate hidden drift between minimized/sampled/scored structures

### 1.2 CA pair-restraint minimization (retained)

**File:** `src/boundry/relaxer.py`

Keep `minimize_with_pair_restraints()` from prior plan, but route it through `_build_system_for_ddg()`.

### 1.3 Replace snapshot-MD ensemble with independent local OpenMM trajectory ensemble

**File:** `src/boundry/relaxer.py`

Replace prior `generate_ensemble()` plan with:

```python
def generate_local_md_ensemble(
    self,
    pdb_string: str,
    mutation_sites: List[Tuple[str, int, str]],
    n_members: int = 35,
    md_total_steps: int = 50000,
    md_equilibration_steps: int = 5000,
    md_temperature: float = 300.0,
    md_friction: float = 1.0,
    neighborhood_radius: float = 8.0,
    sequence_window: int = 1,
    ca_cutoff: float = 9.0,
    restraint_sd: float = 0.5,
    seed: Optional[int] = None,
) -> List[str]:
    ...
```

Implementation details:
1. Build mutation neighborhood (CB distance to mutation sites + sequence +/-1)
2. For each ensemble member:
   - Start from the same restrained-minimized input
   - Initialize unique velocities with deterministic per-member seeds
   - Run short restrained OpenMM dynamics on GPU
   - Apply additional neighborhood-focused restraint/bias so sampling remains local
3. Run short constrained minimization after each trajectory
4. Emit one final member structure per trajectory

Notes:
- This is not Rosetta backrub implementation, but conceptually aligned local sampling.
- Uses OpenMM integrators/forces directly (GPU-accelerated path).
- Independent trajectories avoid highly correlated members.

### 1.4 Rigid-body separation helper

**File:** `src/boundry/relaxer.py` or `src/boundry/ddg.py`

Add utility:

```python
def separate_interface_rigid_body(
    pdb_string: str,
    chain_groups: List[List[str]],
    separation_distance: float = 100.0,
) -> str:
    ...
```

Behavior:
- Translate one chain group as a rigid body
- Keep internal coordinates unchanged
- No repack/minimize on unbound state

---

## Phase 2: Mutation Utilities

### 2.1 General mutation helper (retained)

**File:** `src/boundry/interface_position_energetics.py`

Retain/refine:

```python
def mutate_residue(...)
```

and keep `mutate_to_alanine()` wrapper for backward compatibility.

### 2.2 Mutation parsing + WT validation

**File:** `src/boundry/ddg.py`

Retain robust mutation parsing with WT residue validation against input pose.

---

## Phase 3: Designer Repacking Enforcement

### 3.1 Make `Designer.repack()` honor `DesignSpec`

**File:** `src/boundry/designer.py`

Current behavior repacks all residues. Update `repack()` to respect neighborhood masks:

```python
def repack(
    self,
    pdb_path: Path,
    design_spec: Optional[DesignSpec] = None,
    repack_mask: Optional[torch.Tensor] = None,
) -> dict:
    ...
```

Implementation requirements:
1. Build residue-key mapping
2. Convert `DesignSpec` to per-residue pack mask (`1 = repack`, `0 = fixed`)
3. Call `pack_side_chains(..., repack_everything=False)`
4. Ensure fixed residues retain original side-chain torsions

Rules for ddG workflow:
- WT branch: neighborhood residues repackable (`NATAA` behavior)
- Mut branch: apply PDB-level mutation first, then same neighborhood repack mask
- Non-neighborhood residues (`NATRO`) must remain fixed

### 3.2 Add explicit guardrails

If `design_spec` is provided but would repack all residues, log warning in verbose mode and optionally fail in strict mode.

---

## Phase 4: Configuration

### 4.1 Add `DdGConfig`

**File:** `src/boundry/config.py`

```python
@dataclass
class DdGConfig:
    # Ensemble
    n_ensemble: int = 35
    md_total_steps: int = 50000
    md_equilibration_steps: int = 5000
    md_temperature: float = 300.0
    md_friction: float = 1.0
    neighborhood_sampling_bias: float = 1.0

    # Restraints
    ca_cutoff: float = 9.0
    restraint_sd: float = 0.5

    # Neighborhood
    neighborhood_radius: float = 8.0
    sequence_window: int = 1

    # Interface definition
    chain_pairs: Optional[List[Tuple[str, str]]] = None
    separation_distance: float = 100.0

    # Energy model consistency
    implicit_solvent: bool = True

    # Execution
    workers: int = 1
    seed: Optional[int] = None
    quiet: bool = True

    # Output / cache
    cache_ensemble: bool = False
    ensemble_dir: Optional[Path] = None

    # Optional analysis parity controls
    sort_members_by_wt_bound_energy: bool = False
    average_top_n: Optional[int] = None

    # Convenience preset switch
    paper_mode: bool = False
```

`paper_mode=True` sets:
- `n_ensemble=50`
- higher sampling-depth trajectory settings (e.g., increased `md_total_steps`)

### 4.2 Extend `OptimizeConfig` to use ddG backend

**File:** `src/boundry/config.py`

Add:

```python
interface_scoring_backend: Literal["ddg", "legacy"] = "ddg"
ddg: DdGConfig = field(default_factory=DdGConfig)
```

This makes `optimize` default to the new backend while retaining a temporary fallback.

---

## Phase 5: Core ddG Module

### 5.1 Create `src/boundry/ddg.py`

This replaces the prior `flex_ddg.py` naming.

#### 5.1a Data classes

```python
@dataclass
class MutationSpec: ...

@dataclass
class EnsembleMemberResult:
    member_index: int
    bound_wt_energy: Optional[float]
    unbound_wt_energy: Optional[float]
    bound_mut_energy: Optional[float]
    unbound_mut_energy: Optional[float]
    wt_bound_energy_rank: Optional[int] = None

    @property
    def dG_wt(self): ...
    @property
    def dG_mut(self): ...
    @property
    def ddG(self): ...

@dataclass
class DdGResult:
    mutations: List[MutationSpec]
    member_results: List[EnsembleMemberResult]
    mean_ddG: Optional[float]
    std_ddG: Optional[float]
    mean_dG_wt: Optional[float]
    mean_dG_mut: Optional[float]
    n_successful: int
    n_ensemble: int
    ensemble_ddGs: List[float]
```

#### 5.1b Neighborhood builders

- `build_neighborhood_spec()` for repacking mask
- `build_sampling_neighborhood()` for neighborhood-focused trajectory restraints

Distance logic should match flex-ddG concept:
- neighborhood by mutation-site CB distance (CA for glycine)
- then sequence +/-1 expansion

#### 5.1c Member worker

```python
def _process_ensemble_member(task: _DdGMemberTask) -> _DdGMemberResult:
    ...
```

Per member:
1. WT branch: neighborhood repack -> restrained min -> bound score
2. WT unbound: rigid-body separation of WT bound pose -> unbound score
3. Mut branch: apply mutation(s) -> neighborhood repack -> restrained min -> bound score
4. Mut unbound: rigid-body separation of mutant bound pose -> unbound score
5. Return all four energies

#### 5.1d Core function

```python
def compute_ddg(
    pdb_string: str,
    mutations: List[MutationSpec],
    config: DdGConfig,
    relaxer: Optional[Relaxer] = None,
    designer: Optional[Designer] = None,
    pool: Optional[WorkPool] = None,
) -> DdGResult:
    ...
```

Stages:
1. Constrained minimization
2. Independent local OpenMM ensemble generation
3. Paired WT/mutant four-state scoring
4. Ensemble aggregation (optionally sorted/top-N if configured)

#### 5.1e Binding-only helper for optimize

Add helper for no-mutation interface dG using same unbound separation semantics:

```python
def compute_interface_dg(
    pdb_string: str,
    config: DdGConfig,
    ...
) -> float:
    ...
```

This returns WT-like `dG` from the same backend used by mutation ddG.

---

## Phase 6: Operations + CLI + Optimize Integration

### 6.1 Add `ddg()` operation

**File:** `src/boundry/operations.py`

```python
def ddg(
    structure: StructureInput,
    mutations: Optional[List[Dict[str, str]]] = None,
    mutation_string: Optional[str] = None,
    config: Optional[DdGConfig] = None,
    output_path: Optional[Path] = None,
) -> Structure:
    ...
```

Semantics:
- If mutations provided: run full mutation ddG protocol
- If no mutations: run `compute_interface_dg()` and return `dG` only

Metadata key:

```python
metadata["ddg"] = {...}
```

### 6.2 CLI command: `boundry ddg`

**File:** `src/boundry/cli.py`

```python
@app.command("ddg")
def ddg_cmd(...):
    ...
```

Notes:
- `--mutations` optional; omit for WT interface dG
- Keep backward-compatibility alias `flex-ddg` as hidden/deprecated one release, then remove

### 6.3 Integrate into `optimize` (primary user path)

**File:** `src/boundry/optimize.py`

Replace legacy binding-energy path with ddG backend by default.

#### 6.3a Replace `_score_interface`

Current:
- Calls `calculate_binding_energy()`

Revised:
- Calls `compute_interface_dg()` when `interface_scoring_backend == "ddg"`
- Keeps legacy fallback behind config switch

#### 6.3b Position scoring in optimize

Current:
- `analyze_interface(..., alanine_scan=True)`

Revised:
- Add ddG-backed position scan helper that uses mutation protocol per position (typically alanine substitution) with small ensemble for speed
- Use this helper in `_analyze_and_find_positions()` when backend is `ddg`

Recommended defaults for optimize scan speed:
- `n_ensemble=10`
- `md_total_steps` reduced for scan stage only
- Preserve full settings for final campaign scoring/reporting

### 6.4 Keep `analyze_interface` available

- Do not remove `analyze_interface`
- It remains available for direct use, but `optimize` defaults to ddG backend for scoring decisions

---

## Phase 7: Output and Reporting

### 7.1 Standalone `ddg` outputs

- `input_minimized.pdb`
- `ddg_results.json`
- `ensemble/member_{i}.pdb` (optional)

### 7.2 Include analysis-oriented fields

`ddg_results.json` includes:
- `mean_ddG`, `std_ddG`
- `mean_dG_wt`, `mean_dG_mut`
- `n_ensemble`, `n_successful`
- `ensemble_ddGs`
- per-member four-state energies
- flags showing whether sorted/top-N aggregation was applied

---

## Phase 8: Testing Plan

### 8.1 New tests: `tests/test_ddg.py`

1. Config tests (`DdGConfig`, `paper_mode`)
2. Mutation parsing/validation tests
3. Four-state algebra tests
4. Neighborhood selection tests (CB/CA + sequence window)
5. Rigid-body unbound separation tests
6. Worker tests (pickle safety, error handling)
7. `compute_ddg` unit tests (mocked relaxer/designer)
8. Standalone operation tests (`operations.ddg`)
9. CLI tests (`boundry ddg`)

### 8.2 New tests: `tests/test_designer.py` or `tests/test_designer_unit.py`

Add repack-mask enforcement tests:
- `NATRO` residues remain fixed
- only neighborhood residues repacked
- mutation-site side chains rebuilt while non-neighborhood residues unchanged

### 8.3 `optimize` integration tests

**File:** `tests/test_optimize.py`

Add/modify tests to assert:
- `_score_interface` routes to ddG backend by default
- `_analyze_and_find_positions` uses ddG-backed scan path by default
- legacy backend still works when configured

### 8.4 Integration tests (OpenMM)

- Constrained minimization preserves CA geometry
- Local-backbone ensemble members are non-identical and bounded from input
- Rigid-body separation leaves internal coordinates unchanged
- End-to-end mutation sanity: core hydrophobic->charged often positive ddG

---

## Migration and Compatibility

1. New canonical module/API names use `ddg` instead of `flex_ddg`.
2. Optional temporary shim:
   - `boundry.flex_ddg` imports from `boundry.ddg` with deprecation warning.
3. CLI alias:
   - `flex-ddg` hidden alias for one release, then remove.
4. Metadata key standardized to `metadata["ddg"]`.

---

## File Change Summary (Revised)

| File | Action | Description |
|---|---|---|
| `src/boundry/relaxer.py` | MODIFY | Shared ddG system builder, pair-restraint minimization, local OpenMM ensemble generation, rigid-body separation helper |
| `src/boundry/designer.py` | MODIFY | Make `repack()` honor `DesignSpec`/mask (`NATRO` fixed, neighborhood-only repack) |
| `src/boundry/interface_position_energetics.py` | MODIFY | General `mutate_residue()` plus alanine wrapper compatibility |
| `src/boundry/config.py` | MODIFY | Add `DdGConfig`; add `optimize` backend selector and ddG config |
| `src/boundry/ddg.py` | NEW | Core ddG module (data classes, neighborhood builders, worker, compute functions) |
| `src/boundry/operations.py` | MODIFY | Add `ddg()` operation; expose mutation and binding-only modes |
| `src/boundry/cli.py` | MODIFY | Add `ddg` command; optional deprecated `flex-ddg` alias |
| `src/boundry/optimize.py` | MODIFY | Use ddG backend for interface scoring and position scoring by default |
| `src/boundry/__init__.py` | MODIFY | Export new ddG types/functions |
| `tests/test_ddg.py` | NEW | Comprehensive unit/integration tests for ddG pipeline |
| `tests/test_designer_unit.py` | MODIFY | Add neighborhood repack enforcement tests |
| `tests/test_optimize.py` | MODIFY | Verify optimize default ddG backend and fallback behavior |
| `tests/test_cli.py` | MODIFY | Add `ddg` CLI tests and alias/deprecation tests |

---

## Execution Order

1. `designer.py` repack enforcement + tests (unblocks neighborhood correctness)
2. `relaxer.py` system consistency + local sampling + rigid-body separation
3. `ddg.py` core compute pipeline + tests
4. `operations.py` and `cli.py` (`ddg` command)
5. `optimize.py` backend integration + tests
6. compatibility shims and deprecation notices

This order minimizes risk: scoring correctness is validated before optimize behavior is switched.
