# `analyze-interface`: Rosetta-like per-position energetics

## Summary of decisions (confirmed)
- Implement **both**:
  - **IAM-like per-residue energetics**: per-position `dG` derived from **bound vs separated** states.
  - **AlaScan-style per-residue ΔΔG**: alanine mutation effect per position.
- `boundry analyze-interface` should follow **Rosetta conventions**:
  - `dG = E_bound - E_unbound` (favorable binding → **negative**).
  - `ΔΔG = dG_mut(Ala) - dG_wt` (hotspot → **positive**).
- Default scan set: **all detected interface residues**, with an option to restrict by explicit chains.
- Default per-position energy protocol: **repack bound + unbound**.
- Output: **CSV** (primary deliverable).

---

## Rosetta protocol notes (ground truth)

### InterfaceAnalyzerMover (IAM) is *not* alanine scanning
From Rosetta source (`protocols/analysis/InterfaceAnalyzerMover.cc`):
- Creates a **separated pose** by rigid-body translating one side far away (~1000 Å).
- Optional packing:
  - `pack_input` = repack **bound** complex
  - `pack_separated` = repack **separated** pose
- Computes:
  - total `dG = E_bound - E_unbound` (negative favorable)
  - per-residue `dG_i` using the same sign, comparing per-residue energies in bound vs separated.

### AlaScan is a separate protocol (uses DdgFilter)
From Rosetta source (`protocols/simple_ddg/AlaScan.cc`, `protocols/simple_ddg/DdgFilter.cc`):
- Identify interface residues by distance cutoff.
- Compute WT `dG`/`ddG` with repacking enabled by default in `AlaScan` (Rosetta “ddG mode”).
- For each interface residue:
  - mutate to alanine
  - compute mutant `dG`
  - report `ΔΔG = dG_mut - dG_wt` (positive usually destabilizing/hotspot).

---

## Boundry reality check (current implementation vs Rosetta)

### Current Boundry ddG sign
`boundry.binding_energy.calculate_binding_energy()` currently computes:
- `binding_energy = E_unbound - E_bound` (positive favorable), and tests expect that.

### Plan for Rosetta-aligned `analyze-interface`
To avoid breaking other commands/tests, we will:
- Keep the internal `BindingEnergyResult.binding_energy` as-is for now.
- In `boundry analyze-interface`, report Rosetta-aligned values by converting sign:
  - `dG_rosetta = -binding_energy` (i.e., `E_bound - E_unbound`)
- All new per-position outputs and AlaScan deltas will use `dG_rosetta`.

If you later want the whole library to switch to Rosetta sign globally, we can do that in a separate change with coordinated test updates.

---

## User-facing CLI design

### New options on `boundry analyze-interface`
Add options that apply only to this command:

1. `--per-position`
   - Enables per-position energetics output (IAM-like per-residue `dG_i`) plus context columns.

2. `--alanine-scan`
   - Enables AlaScan output per interface residue:
     - compute WT `dG_rosetta`
     - compute Ala mutant `dG_rosetta`
     - report `ΔΔG = dG_mut - dG_wt`

3. `--scan-chains A,B,C`
   - Optional restriction on which residues are included in `--per-position` and/or `--alanine-scan`.
   - Default: scan all detected interface residues.

4. `--position-repack {both,unbound,none}`
   - Default: `both` (Rosetta-like; mirrors `pack_input=true`, `pack_separated=true`).
   - `both`: repack bound complex and unbound partners.
   - `unbound`: repack unbound partners only.
   - `none`: no repacking (fastest).

5. `--position-csv OUT.csv`
   - Write results to CSV (required output mode per requirement).
   - If not provided, default behavior is stdout-only (keep it, but encourage CSV).

6. Optional perf controls (recommended)
   - `--max-scan-sites N` (limit number of residues scanned; helpful for large interfaces)
   - `--scan-workers N` (default 1; consider later after correctness)

### Existing options interactions
- Continue to support existing `analyze-interface` flags (`--chains`, `--distance-cutoff`, `--relax-separated`, `--constrained`, etc.).
- For per-position computations:
  - ignore the legacy `--pack-separated` flag (to avoid mixed semantics)
  - use `--position-repack` instead
- Document clearly in help text which knobs apply to which computation.

---

## Output specification (CSV-first)

### CSV columns (single combined file)
One row per interface residue (stable ordering: chain, resnum, icode):

**Identifiers / context**
- `chain_id`
- `residue_number`
- `insertion_code`
- `wt_resname`
- `partner_chain` (from `InterfaceResidue.partner_chain`)
- `min_distance`
- `num_contacts`

**Burial (if SASA enabled)**
- `delta_sasa` (unbound - bound for that residue; already computed in `surface_area`)

**WT interface energetics (Rosetta sign)**
- `dG_wt` (same value repeated per row for convenience; or a header-level metadata section—CSV can’t easily represent that, so repeat is pragmatic)

**IAM-like per-residue energetics (`--per-position`)**
- `dG_i` (per-residue contribution estimate; see definition below)

**AlaScan (`--alanine-scan`)**
- `dG_ala`
- `delta_ddG` (= `dG_ala - dG_wt`)

**Metadata (repeat per row or emit as commented header lines)**
- `distance_cutoff`
- `chain_pairs`
- `position_repack`
- `relax_separated`
- `constrained`

Stdout can print:
- existing summary lines
- paths to the CSV
- optionally a short top-N table sorted by `delta_ddG` (hotspots) when enabled.

---

## Core algorithm design

### 1) Determine the interface definition
Use existing:
- `boundry.interface.identify_interface_residues(pdb_string, distance_cutoff, chain_pairs)`

Default scan set:
- all unique interface residues (dedupe by `(chain_id, residue_number, insertion_code)`).
Optional restriction:
- filter by `--scan-chains`.

### 2) Define the interface “sides” (required for energies)
Reuse existing chain grouping logic:
- `boundry.binding_energy._get_interface_chain_groups(interface_info.chain_pairs)`

Guardrail:
- For per-position energetics and AlaScan, require **exactly 2 groups**.
- If grouping yields >2 (ambiguous sides), raise an actionable error:
  - “Per-position energetics require a two-sided interface; provide `--chains` that cleanly partitions (e.g. `H:A,L:A`).”

### 3) Repacking protocol (Rosetta-like default)
Implement a repacking helper that works on PDB strings:
- Reuse and generalize existing `_repack_with_designer(pdb_string, designer)` in `src/boundry/binding_energy.py`:
  - add a new public helper in a new module (or move to a shared util) so both binding energy and per-position can use it.

For per-position computations, apply per `--position-repack`:
- `both`:
  - repack bound complex PDB string
  - then generate unbound partner-group PDBs and repack each
- `unbound`:
  - only repack unbound partner-group PDBs
- `none`:
  - skip repacking entirely

### 4) WT `dG` (Rosetta sign) computation
Compute once:
- Call existing `calculate_binding_energy(...)` to get `binding_energy = E_unbound - E_bound`
- Convert:
  - `dG_wt = -binding_energy`

### 5) AlaScan per-residue ΔΔG (`--alanine-scan`)
For each scan site:
- Mutate residue to alanine (see mutation helper below)
- Apply repack policy (`both`/`unbound`/`none`)
- Compute mutant:
  - `dG_ala = -calculate_binding_energy(...).binding_energy`
- Compute:
  - `ΔΔG = dG_ala - dG_wt`

### 6) IAM-like per-residue `dG_i` (`--per-position`)
Rosetta can compute per-residue energies directly; OpenMM cannot easily provide a faithful per-residue decomposition.

We will implement a well-defined OpenMM analogue that is stable and testable:

**Definition (marginal contribution by residue removal)**
- Let `dG_total = E_bound - E_unbound` (Rosetta sign).
- Let `dG_without_i` be the same binding energy computed after removing residue *i* from both bound and unbound states (coordinates fixed; PDBFixer handles missing atoms/termini).
- Define:
  - `dG_i = dG_total - dG_without_i`

This yields a per-residue “contribution” consistent with the chosen forcefield and solvation model, and mirrors the intent of “how much does this residue matter for binding energy” (but is not identical to Rosetta’s per-residue scoring).

**Protocol**
- WT (with repacking policy applied) → compute `dG_total`.
- For each residue *i*:
  - produce a “residue-removed” bound complex PDB string
  - compute `dG_without_i` with the same repack policy (note: with residue removed, repacking might need to be disabled or limited—see risk section)
  - compute `dG_i`

If this is too slow, future optimization paths:
- compute `dG_i` only for top-N AlaScan hotspots
- parallelize over sites with process-level OpenMM contexts
- implement a cross-interface interaction-energy calculator (more complex)

### 7) Mutation helper (alanine)
Add:
`mutate_to_alanine(pdb_string, chain_id, resnum, icode) -> str`

Requirements:
- protein residues only
- preserve chain and residue identifiers
- set residue name to `ALA`
- delete atoms not in alanine (`N`, `CA`, `C`, `O`, `CB`, `OXT` if present)
- rely on PDBFixer inside Relaxer/OpenMM path to rebuild missing hydrogens/atoms

### 8) Residue removal helper
Add:
`remove_residue(pdb_string, chain_id, resnum, icode) -> str`

Requirements:
- remove all `ATOM/HETATM` records for that residue
- preserve the rest of the file
- ensure resulting PDB is still parseable (END record, TER handling)

---

## Concrete implementation steps (repo changes)

### Step 1: Add a new module for per-position energetics
Add `src/boundry/interface_position_energetics.py`:
- dataclasses:
  - `ResidueKey`
  - `PerPositionRow`
  - `PerPositionResult`
- functions:
  - `mutate_to_alanine(...)`
  - `remove_residue(...)`
  - `compute_rosetta_dG(...)` (wraps `calculate_binding_energy` and inverts sign)
  - `compute_alanine_scan(...)`
  - `compute_per_position_dG(...)` (residue removal marginal contributions)
  - `write_position_csv(result, path)`

### Step 2: Add CLI wiring
Modify `src/boundry/cli.py` `analyze-interface`:
- parse new options
- ensure weights/designer are available when repacking is requested (`position_repack != none`)
- compute interface residues as before
- compute WT `dG_wt` via sign-inverted binding energy
- if `--per-position`, compute `dG_i` values and attach to rows
- if `--alanine-scan`, compute `dG_ala`/`ΔΔG` and attach to rows
- write CSV if `--position-csv`

### Step 3: Add tests (unit + lightweight integration)
Add `tests/test_interface_position_energetics.py`:
- unit: `mutate_to_alanine` correctness on small synthetic PDB snippets
- unit: `remove_residue` correctness (residue records removed, rest intact)
- mocked: patch `boundry.binding_energy.calculate_binding_energy` to validate:
  - `dG_rosetta = -binding_energy`
  - `ΔΔG = dG_ala - dG_wt`
  - `dG_i = dG_total - dG_without_i`

Update existing CLI help tests to ensure new flags appear in `boundry analyze-interface --help`.

### Step 4: Documentation
Update `README.md`:
- add an example for per-position CSV:
  - `boundry analyze-interface complex.pdb --chains H:A,L:A --per-position --alanine-scan --position-csv out.csv`
- explain sign conventions (Rosetta-style) and interpretations.

---

## Performance expectations
- AlaScan: O(N) binding-energy evaluations (N = number of interface residues).
- Per-position `dG_i` via residue removal: O(N) additional evaluations on top of WT.
- With repack bound+unbound, each evaluation also includes LigandMPNN packing time.

Mitigations (implement early):
- `--max-scan-sites`
- optionally support “AlaScan only” without per-position `dG_i` for large interfaces
- consider `--scan-workers` after correctness (process-based, not threads)

---

## Risks & mitigations
- **Repacking + residue removal interaction**: removing residues can create chain termini/geometry that makes repacking/minimization less stable.
  - Mitigation: for `dG_i` calculations, default `position_repack=unbound` or `none` unless explicitly set to `both` (but you requested default `both`; we can keep default but add a warning when `--per-position` is enabled because it may be slow/fragile).
- **OpenMM “per-residue” mismatch vs Rosetta**: Boundry’s `dG_i` will be defined by the chosen OpenMM forcefield + GBn2 model and the “residue removal marginal” definition, not Rosetta’s residue total energy decomposition.
  - Mitigation: document the definition; keep the naming clear (`dG_i_marginal` if needed).
- **Multichain ambiguity**: require exactly two sides for per-position energetics; error otherwise with suggestions.

---

## Acceptance criteria
- Running `boundry analyze-interface` without new flags is unchanged.
- `--per-position --position-csv out.csv` writes CSV including per-residue `dG_i` (and context columns).
- `--alanine-scan --position-csv out.csv` writes CSV including `dG_ala` and `ΔΔG`.
- Default energetics follow Rosetta sign conventions in `analyze-interface` outputs.
- Tests cover mutation/removal helpers and the key math/sign conventions.

