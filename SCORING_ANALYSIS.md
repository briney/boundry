# Scoring Function Analysis: Boundry Optimize vs. Rosetta Flex-ddG

A systematic comparison of the scoring methodology in `boundry optimize` against
Rosetta's flex-ddG protocol ([Barlow et al., J. Phys. Chem. B 2018](https://doi.org/10.1021/acs.jpcb.7b11367);
[tutorial](https://github.com/Kortemme-Lab/flex_ddG_tutorial)).

---

## 1. Protocol Overview

### Flex-ddG (Rosetta)

1. **Prepare**: Add CA-CA distance constraints (coord_dev=0.5 A, max_dist=9.0 A).
   Minimize backbone + chi angles with L-BFGS (5,000 iterations, tolerance 1e-6).
   Clear constraints.
2. **Ensemble generation**: Backrub Monte Carlo (35,000 trials, kT=1.2) samples
   3-residue backbone segment rotations. Snapshots are extracted at stride
   intervals (typically every 7,000 steps, yielding ~5 structures per
   trajectory across ~35 independent trajectories).
3. **Per-snapshot scoring** (at each backrub checkpoint):
   - **WT path**: Repack neighbor shell (8 A bubble, MultiCoolAnnealer 6 states,
     -ex1 -ex2) -> Minimize (constrained) -> InterfaceDdGMover scores
     `bound_wt` and `unbound_wt`.
   - **Mutant path**: Introduce mutation via resfile -> Repack -> Minimize ->
     InterfaceDdGMover scores `bound_mut` and `unbound_mut`.
4. **Aggregate**: ddG per snapshot = (E_bound_mut - E_unbound_mut) - (E_bound_wt - E_unbound_wt).
   Average over ensemble. Optionally apply GAM reweighting.

### Boundry Optimize

1. **Prepare**: Optionally idealize backbone geometry. Run N repack+minimize
   cycles (default 10) to relax the starting structure.
2. **Per-cycle** (beam search, default 10 cycles):
   - **Alanine scan**: Identify destabilizing positions (ddG > threshold).
     For each interface residue, mutate to Ala and compute
     dG = E_complex - sum(E_separated). ddG = dG_ala - dG_wt.
   - **Design**: Sample positions (softmax-weighted by ddG). For each
     position, LigandMPNN designs + OpenMM minimizes. Score via
     binding energy.
   - **Select**: Keep top `beam_width` structures by dG. Regression guard.
3. **Campaign aggregation**: Run multiple independent campaigns from
   different seeds. Keep the overall best structure.

---

## 2. Energy Function

| Aspect | Flex-ddG (Rosetta) | Boundry (OpenMM) |
|---|---|---|
| Force field | talaris2014 (or REF2015) | AMBER14 |
| Type | Knowledge-based + physics hybrid | Pure physics (molecular mechanics) |
| Key energy terms | fa_atr, fa_rep, fa_sol, fa_elec, hbond_sc, hbond_bb_sc, hbond_lr_bb, fa_dun, rama, omega, p_aa_pp, ref | bond, angle, dihedral, nonbonded (LJ + Coulomb), solvation (GBn2) |
| H-bonds | Explicit orientation-dependent terms (3 categories: sc, bb_sc, lr_bb) | Implicit via electrostatics in NonbondedForce |
| Rotamer energy | fa_dun (Dunbrack statistical potential) | Not present — rotamer quality assessed only via LigandMPNN log-probability |
| Reference energy | Per-residue-type reference energies (ref term) | Not present |

### Impact

Rosetta's score function includes several knowledge-based terms that have no
AMBER equivalent:

- **fa_dun** (Dunbrack rotamer probability): Penalizes rotamers that deviate from
  statistically preferred chi angles. AMBER has torsion terms but no statistical
  rotamer potential. This means boundry may accept side-chain conformations that
  Rosetta would penalize.

- **ref** (reference energies): Per-amino-acid-type correction that adjusts the
  baseline energy of each residue type. Without this, the relative energies of
  different amino acids at a given position are not calibrated, potentially
  biasing sequence design preferences.

- **rama / p_aa_pp** (Ramachandran / sequence-dependent backbone preferences):
  Rosetta uses knowledge-based backbone potentials. AMBER's dihedral terms encode
  some of this but are less discriminating for unusual phi/psi combinations.

- **Explicit H-bond terms**: Rosetta separates H-bond contributions by
  donor/acceptor type (sidechain, backbone-sidechain, long-range backbone),
  each with distinct geometry-dependent scoring. AMBER treats all H-bonds
  through generic electrostatics. This gives Rosetta finer control over how
  H-bond networks contribute to interface ddG.

---

## 3. Solvation Model

| Aspect | Flex-ddG | Boundry |
|---|---|---|
| Solvation type | Lazaridis-Karplus implicit (fa_sol) | GBn2 implicit (`get_energy_breakdown`) |
| Relaxation solvation | Same (LK is part of the score function) | **None** — vacuum (amber14 + tip3pfb, no explicit waters) |
| Consistency | Relaxation and scoring use the same energy function | **Mismatch**: relaxation in vacuum, scoring with GBn2 |

### Critical Issue: Solvation Model Mismatch

This is one of the most significant misalignments in the current codebase:

- **`Relaxer.relax()` / `_relax_unconstrained()`** (`relaxer.py:222-223`):
  Always creates the system with `amber14-all.xml + amber14/tip3pfb.xml`.
  Since no explicit water molecules are added to the system, the protein
  is effectively minimized in **vacuum** — electrostatics and van der Waals
  are computed in the absence of solvent.

- **`Relaxer.get_energy_breakdown()`** (`relaxer.py:459-465`): When
  `implicit_solvent=True` (the default), uses `amber14-all.xml + implicit/gbn2.xml`,
  which adds a `CustomGBForce` (Generalized Born polar solvation + nonpolar
  surface area term).

The consequence: structures are relaxed on a vacuum potential energy surface
but scored on a different (solvated) surface. Minimum-energy conformations under
vacuum electrostatics can differ substantially from those under GB solvation,
particularly for charged and polar residues at the interface. This can lead to:

1. Structures that are "optimized" for vacuum but suboptimal under solvated scoring.
2. Inconsistent energy differences between complex and separated states.
3. Spurious ddG values for mutations that change the charge state at the interface.

**In flex-ddG**, the Lazaridis-Karplus solvation term (fa_sol) is part of the
score function used for both minimization and scoring — there is no mismatch.

### Recommendation

Add GBn2 implicit solvation to `_relax_unconstrained()` when
`config.implicit_solvent` is True:
```python
if self.config.implicit_solvent:
    force_field = openmm_app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
else:
    force_field = openmm_app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
```
This ensures structures are minimized on the same energy surface used for scoring.

---

## 4. Separated Chain (Unbound State) Treatment

| Aspect | Flex-ddG | Boundry Optimize |
|---|---|---|
| Chain separation | InterfaceDdGMover translates chains apart | `extract_chain()` extracts PDB records |
| Repack unbound | Yes — always repacks separated chains | **No** — `relax_separated=False` by default |
| Minimize unbound | Yes — minimizes after repacking | **No** — not called in optimize scoring |
| States scored | 4: bound_wt, unbound_wt, bound_mut, unbound_mut | 2: complex, separated (no repacking/minimization) |

### Critical Issue: No Unbound-State Relaxation

In `_score_interface()` (`optimize.py:359-380`) and `_execute_beam_expansion()`
(`optimize.py:293-298`), `calculate_binding_energy()` is called with the
default `relax_separated=False`. This means:

1. The complex is scored.
2. Chains are extracted by filtering PDB ATOM records.
3. **The separated chains are scored as-is** — in the conformation they had
   in the bound complex, with no opportunity to relieve steric strain from
   the absent binding partner.

This is physically unrealistic. When proteins dissociate, side chains at the
former interface relax into new conformations. Scoring the unbound state in
the bound-state conformation systematically overestimates the unbound energy,
which artificially inflates (makes more negative) the binding energy dG.

More critically, the magnitude of this artifact varies by mutation. Mutations
that cause different degrees of steric strain at the interface will have
variable artifacts in their separated-state energies, distorting the ddG
ranking.

**In flex-ddG**, the InterfaceDdGMover always repacks and minimizes the
separated state before scoring.

### Recommendation

Enable separated-chain relaxation in the optimize scoring path:
```python
be_result = calculate_binding_energy(
    current_pdb,
    relaxer,
    chain_pairs=task.chain_pairs,
    distance_cutoff=8.0,
    relax_separated=True,      # Repack + minimize unbound state
    designer=designer,          # Required for repacking
)
```

This is the single highest-impact change for bringing boundry's ddG values
closer to flex-ddG behavior.

---

## 5. Backbone Conformational Sampling

| Aspect | Flex-ddG | Boundry Optimize |
|---|---|---|
| Backbone sampling method | Backrub Monte Carlo | None |
| Sampling extent | 35,000 trials per trajectory, 35 trajectories | Single structure per cycle |
| Ensemble averaging | Yes — ddG averaged over ~35 structures | No — single-structure scoring |
| Backbone flexibility during design | Backrub samples before scoring | LigandMPNN is backbone-fixed; only OpenMM moves backbone during minimization |

### Impact

Backrub ensemble generation is the defining feature of flex-ddG and the
primary reason for its superior performance over single-structure protocols.
By generating a conformational ensemble:

1. **Reduced noise**: Averaging over structures reduces sensitivity to the
   starting conformation.
2. **Accessible conformational states**: Backbone movement allows side chains
   to find conformations not accessible from the starting backbone.
3. **Tolerance modeling**: Small backbone adjustments accommodate mutations
   that would be highly strained on a rigid backbone.

Boundry's approach uses OpenMM minimization, which finds the nearest local
minimum but does not cross energy barriers. This means:

- The backbone is essentially rigid (minimization moves it only slightly).
- Each design cycle scores a single backbone conformation.
- There is no conformational averaging to reduce noise.

### Mitigation

Boundry's iterative design-relax cycles partially compensate: each cycle
redesigns and reminimizes, allowing some backbone adjustment over multiple
rounds. The beam search (multiple parents) provides limited conformational
diversity. However, this is not equivalent to systematic backbone sampling.

A closer approximation to flex-ddG would require either:
- OpenMM molecular dynamics at elevated temperature (analogous to backrub kT=1.2).
- Multiple minimizations from perturbed starting conformations.
- Explicit backbone sampling moves (e.g., phi/psi perturbations with
  Metropolis acceptance).

---

## 6. Minimization Protocol

| Aspect | Flex-ddG | Boundry (default) |
|---|---|---|
| Algorithm | L-BFGS | L-BFGS |
| Max iterations | 5,000 | 0 (unlimited) |
| Convergence tolerance | 1e-6 | OpenMM default (10 kJ/mol/nm) |
| Degrees of freedom | Backbone (bb) + sidechain (chi) | All atoms (unconstrained) |
| Position restraints | CA distance constraints (coord_dev=0.5 A, max_dist=9.0 A) | **None** (default unconstrained mode) |
| H-bond constraints | Not specified | HBonds (rigid H-X bonds) |
| Neighbor shell restriction | 8 A bubble around mutation site | No restriction — whole structure minimized |

### Issue: Unconstrained Minimization

Flex-ddG applies CA-CA distance constraints during minimization to prevent
the structure from drifting far from the starting conformation. This is
important because:

1. The scoring function is calibrated for near-native conformations.
2. Large backbone movements can create artifacts where the structure "finds"
   a non-physical energy minimum.
3. The backrub ensemble is meant to provide controlled backbone diversity —
   minimization should refine within, not escape, the local basin.

Boundry's default unconstrained minimization has no position restraints.
While `config.constrained=True` enables the AlphaFold-style AmberRelaxation
(which does use position restraints), the default optimize configuration
uses unconstrained minimization.

### Recommendation

Consider making constrained minimization the default for optimize, or at
minimum applying harmonic restraints to CA atoms (the `_add_restraints()`
method in `relaxer.py` already supports this via the `stiffness` parameter,
but it's only used in `_relax_direct()`, not in `_relax_unconstrained()`).

---

## 7. Side-Chain Packing

| Aspect | Flex-ddG (Rosetta) | Boundry (LigandMPNN) |
|---|---|---|
| Method | Discrete rotamer library (Dunbrack) | Neural network diffusion-based packer |
| Sampling | MultiCoolAnnealer (6 states), -ex1 -ex2 subrotamers | 16 samples, 3 denoising steps |
| Neighbor shell | 8 A bubble around mutation + adjacent residues | All residues (no spatial restriction) |
| Statistical potential | fa_dun penalizes non-preferred rotamers | No explicit rotamer penalty |
| Sequence dependence | Dunbrack library is backbone-dependent | Context from graph neural network |

### Impact

The packing approaches are fundamentally different but serve similar purposes.
Key differences relevant to scoring accuracy:

1. **Discrete vs. continuous**: Rosetta samples discrete rotamers from a
   library; LigandMPNN predicts continuous coordinates via diffusion.
   LigandMPNN may find conformations outside the rotamer library but lacks
   the statistical penalty for unusual conformations.

2. **Neighbor shell restriction**: Flex-ddG only repacks within 8 A of the
   mutation site. This prevents global repacking from introducing noise.
   Boundry's LigandMPNN repacking operates on the full structure, which
   can introduce unrelated conformational changes.

3. **Coupling with scoring**: Rosetta's packer optimizes the same score
   function used for evaluation. LigandMPNN optimizes its own objective
   (sequence log-probability), which does not directly correspond to
   AMBER energies. A conformation that LigandMPNN considers favorable
   may not be an AMBER energy minimum, and vice versa.

---

## 8. ddG Calculation

### Flex-ddG Formula

Per structure:
```
dG_wt  = E(bound_wt)  - E(unbound_wt)
dG_mut = E(bound_mut) - E(unbound_mut)
ddG    = dG_mut - dG_wt
```
Where each state (bound_wt, unbound_wt, bound_mut, unbound_mut) is
independently repacked, minimized, and scored.

Across ensemble:
```
ddG_final = mean(ddG_i for i in ensemble)
```

With optional GAM reweighting (applied per score term before summation):
```
gam(x, term) = -exp(a) + 2*exp(a) / (1 + exp(-x * exp(b)))
```
Where `(a, b)` are fitted parameters for each of 7 energy terms:
`fa_sol`, `hbond_sc`, `hbond_bb_sc`, `fa_rep`, `fa_elec`, `hbond_lr_bb`, `fa_atr`.

### Boundry Optimize ddG

For alanine scanning (position identification):
```
dG_wt  = E_complex(wt) - sum(E_separated(wt))
dG_ala = E_complex(ala) - sum(E_separated(ala))
ddG    = dG_ala - dG_wt
```

For design scoring (beam ranking):
```
dG = E_complex - sum(E_separated)
```
Structures ranked by dG directly (lower = better).

### Key Differences

1. **No ensemble averaging**: Boundry scores a single structure per design
   variant. Flex-ddG averages over ~35 backbone conformations.

2. **No GAM reweighting**: Flex-ddG applies a fitted nonlinear transformation
   to individual energy terms. This is a form of learned calibration that
   improves correlation with experimental ddG. Boundry uses raw AMBER energies.

3. **No four-state scoring for design**: During beam expansion, boundry
   only scores the designed structure's dG. It does not compare WT and
   mutant states in both bound and unbound conformations. The alanine scan
   does compare WT and mutant (Ala), but only in the context of position
   identification, not for final ranking.

4. **Single-model scoring**: Each separated chain group is scored once.
   Flex-ddG's InterfaceDdGMover repacks and minimizes each state independently.

---

## 9. Interface Residue Definition

| Aspect | Flex-ddG | Boundry |
|---|---|---|
| Method | Chain-based (chainstomove parameter) + 8 A bubble for repacking | Distance-based (all-atom pairwise, 8 A cutoff) |
| Scope of mutations | Specified by resfile | Identified by alanine scan ddG > threshold |
| Design shell | 8 A neighbor bubble + 1 sequence neighbor | No spatial restriction on design |

Both approaches use 8 A as the relevant distance scale, which is consistent.
Boundry's all-atom pairwise distance is more rigorous than chain-based
assignment and should identify interface residues at least as accurately.

---

## 10. Summary of Misalignments

Ranked by expected impact on scoring accuracy:

### Critical

| # | Issue | Flex-ddG Behavior | Boundry Behavior | Impact |
|---|---|---|---|---|
| 1 | **Solvation mismatch** | Consistent solvation (LK) for minimization and scoring | Vacuum minimization, GBn2 scoring | Structures relaxed on wrong energy surface; distorts ddG for polar/charged mutations |
| 2 | **No unbound-state relaxation** | Always repacks + minimizes separated chains | Scores separated chains in bound conformation | Systematically overestimates unbound energy; variable artifact by mutation |
| 3 | **No ensemble averaging** | 35 structures averaged | Single structure scored | High noise; sensitive to starting conformation |

### Significant

| # | Issue | Flex-ddG Behavior | Boundry Behavior | Impact |
|---|---|---|---|---|
| 4 | **No backbone sampling** | Backrub MC (35k trials) | Minimization only | Cannot cross energy barriers; misses accessible backbone states |
| 5 | **No GAM reweighting** | Fitted nonlinear per-term transformation | Raw AMBER energies | Reduced correlation with experimental ddG |
| 6 | **Unconstrained minimization** | CA constraints (0.5 A, 9 A max) | No restraints (default) | Structure may drift to non-physical minima; large RMSD from starting conformation |

### Moderate

| # | Issue | Flex-ddG Behavior | Boundry Behavior | Impact |
|---|---|---|---|---|
| 7 | **No explicit H-bond scoring** | 3 orientation-dependent H-bond terms | Generic electrostatics only | Reduced sensitivity to H-bond network changes at interface |
| 8 | **No rotamer statistical potential** | fa_dun (Dunbrack) | LigandMPNN log-probability (different objective) | May accept rotamers that deviate from statistical preferences |
| 9 | **No reference energies** | Per-amino-acid ref term | Not present | Uncalibrated relative amino acid energies |
| 10 | **Packer-scorer coupling** | Rosetta packer optimizes the score function | LigandMPNN optimizes its own objective, scored by AMBER | Optimal LigandMPNN conformations may not be AMBER energy minima |

### Minor / Acceptable

| # | Issue | Flex-ddG Behavior | Boundry Behavior | Notes |
|---|---|---|---|---|
| 11 | **Force field family** | Rosetta talaris2014 | AMBER14 | Expected difference; both are validated for protein energetics |
| 12 | **Solvation model type** | Lazaridis-Karplus | GBn2 | Both are implicit solvation; GBn2 is more modern |
| 13 | **Interface definition** | Chain-based + neighbor shell | All-atom pairwise distance | Boundry's approach is more rigorous |

---

## 11. Prioritized Recommendations

### Phase 1: Fix Scoring Inconsistencies (Critical)

1. **Unify solvation model**: Use implicit solvent (GBn2) in `_relax_unconstrained()`
   when `config.implicit_solvent` is True. This is a ~5-line change in `relaxer.py`
   that eliminates the vacuum/solvated mismatch.

2. **Enable unbound-state relaxation in optimize**: Pass `relax_separated=True`
   and provide the Designer instance when computing binding energy in
   `_score_interface()` and `_execute_beam_expansion()`. This ensures
   separated chains are repacked and minimized before scoring.

3. **Apply position restraints during minimize**: Use `_add_restraints()` in
   `_relax_unconstrained()` when `config.stiffness > 0`, analogous to
   flex-ddG's CA constraints. This prevents excessive backbone drift.

### Phase 2: Improve Sampling (Significant)

4. **Multi-structure scoring**: Score each design variant from 3-5 minimized
   conformations (different LigandMPNN seeds) and average. This approximates
   ensemble averaging without requiring MD.

5. **Neighbor-shell-restricted repacking**: Add a spatial mask to LigandMPNN
   packing (only repack within 8 A of the mutation site) to reduce noise
   from global repacking.

### Phase 3: Calibration (Enhancement)

6. **Learned reweighting**: Fit a per-energy-term transformation (linear
   or GAM-style) on a benchmark set of experimental ddG values. This is the
   approach flex-ddG uses to improve correlation with experiment.

7. **Rotamer quality penalty**: Add a term that penalizes low-probability
   rotamers, either from the Dunbrack library or from LigandMPNN's own
   per-residue log-probabilities.

---

## 12. What Boundry Does Better

Despite the misalignments above, boundry's approach has several advantages:

1. **Modern neural-network packing**: LigandMPNN's diffusion-based packer
   can find conformations outside discrete rotamer libraries, potentially
   producing more realistic side-chain conformations for non-canonical
   geometries.

2. **Ligand context**: LigandMPNN's `ligand_mpnn` model type can incorporate
   ligand atoms as context, enabling design near cofactors and small
   molecules — something flex-ddG does not natively support.

3. **Iterative design-relax**: The beam search with iterative redesign
   allows coupled sequence-structure optimization, whereas flex-ddG
   evaluates pre-specified point mutations without design exploration.

4. **Physics-based minimization**: AMBER14 with OpenMM provides higher-
   fidelity local geometry optimization than Rosetta's minimizer, with
   proper treatment of bond lengths, angles, and non-bonded interactions.

5. **Parallelism**: The WorkPool architecture enables efficient parallel
   evaluation of design variants, making the beam search practical for
   real-time use.
