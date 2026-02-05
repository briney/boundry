# Boundry Workflow Reference

This directory contains example YAML workflows for `boundry run`.

This README documents:

- The general workflow schema.
- All supported step/block types.
- Every workflow operation and its workflow `params`.
- Condition syntax and valid variable references.
- Output filename template variables (`{cycle}`, `{round}`, `{rank}`).

## 1. Workflow Structure

All workflows use this top-level shape:

```yaml
workflow_version: 1          # optional, defaults to 1
input: input.pdb             # required
output: final.pdb            # optional
steps:                       # required, non-empty
  - operation: idealize
```

Top-level keys are strict:

- Allowed: `workflow_version`, `input`, `output`, `steps`
- Unknown keys raise an error.

## 2. Step Node Types

Each item in `steps:` must contain exactly one of:

- `operation` (single operation step)
- `iterate` (fixed or convergence loop)
- `beam` (population beam search)

### 2.1 Operation Step

```yaml
- operation: relax
  params:
    n_iterations: 3
    constrained: true
  output: relax_out.pdb
```

Allowed keys:

- `operation` (required, string)
- `params` (optional mapping; `null` is treated as `{}`)
- `output` (optional string path template)

### 2.2 Iterate Block

```yaml
- iterate:
    n: 5
    seed_param: seed
    output: iterate_cycle_{cycle}.pdb
    steps:
      - operation: relax
```

Or convergence mode:

```yaml
- iterate:
    until: "{dG} < -12.0"
    max_n: 50
    steps:
      - operation: analyze_interface
```

Fields:

- `steps` (required, non-empty list of nested node types)
- `n` (default `1`) for fixed-count mode (`until` omitted)
- `until` (optional condition string) for convergence mode
- `max_n` (default `100`) safety cap when `until` is set
- `seed_param` (optional) explicit seed param name to inject each cycle
- `output` (optional path template)

Notes:

- If `until` uses `delta(...)`, cycle 1 is treated as "not yet converged" (bootstrap).
- Seed injection only happens for iterate when `seed_param` is set.

### 2.3 Beam Block

```yaml
- beam:
    width: 3
    rounds: 10
    expand: 5
    metric: dG
    direction: min
    until: "{dG} < -15.0"
    output: beam_round_{round}_rank_{rank}.pdb
    steps:
      - operation: design
      - operation: analyze_interface
```

Fields:

- `steps` (required, non-empty list)
- `width` (default `5`) candidates kept per round
- `rounds` (default `10`) max rounds
- `expand` (default `1`) expansions per candidate per round (minimum)
- `metric` (default `"dG"`) dotted path metric to score by
- `direction` (`"min"` or `"max"`, default `"min"`)
- `until` (optional condition string checked on best candidate each round)
- `output` (optional path template)

Notes:

- Beam is first-class population flow: top-K candidates continue into later steps.
- `Workflow.run()` returns best rank-1 candidate.
- `Workflow.run_population()` returns final kept population.

## 3. Output Template Variables

Any `output` field uses Python `str.format(...)` templating.

### 3.1 Available Variables by Context

- Top-level `output` and operation-step `output`:
  - `{rank}` (always available; defaults to `1` for single structure)
- Iterate block `output`:
  - `{cycle}` (1-based iterate cycle index)
  - `{rank}` (if population size > 1, or explicit use)
- Beam block `output`:
  - `{round}` (1-based beam round index)
  - `{rank}` (1-based candidate rank)

### 3.2 Multi-candidate Naming

If more than one structure is written and no `{rank}` placeholder is present,
Boundry auto-appends `_rankN` before the extension:

- `out.pdb` -> `out_rank1.pdb`, `out_rank2.pdb`, ...

## 4. Condition Expressions (`until`)

`iterate.until` and `beam.until` use a safe parser (no `eval`).

### 4.1 Syntax

```text
condition := expr COMPARE expr
COMPARE   := < | > | <= | >= | == | !=
expr      := term ((+|-) term)*
term      := factor ((*|/) factor)*
factor    := NUMBER | VARIABLE | FUNCTION("(" expr ")") | "(" expr ")" | "-" factor
VARIABLE  := {dotted.path}
FUNCTION  := abs | delta
```

### 4.2 Examples

- `"{dG} < -10.0"`
- `"abs({final_energy}) < 1000"`
- `"delta({final_energy}) < 0.5"`
- `"{metrics.interface.dG} < -12.0"`

### 4.3 Variable Rules

- Variables are wrapped in `{...}`.
- Dotted paths are allowed (`{a.b.c}`).
- Private path segments are blocked (segments starting with `_`).
- Values must resolve to numeric (`int`/`float`) for arithmetic/comparisons.
- `delta(x)` compares current vs previous metadata value:
  - `abs(current(x) - previous(x))`
  - Outside loop/round contexts without previous metadata, this raises an error.

## 5. Operations and `params` Reference

Supported operation names:

- `idealize`
- `minimize`
- `repack`
- `relax`
- `mpnn`
- `design`
- `renumber`
- `analyze_interface`

## 5.1 `idealize`

Purpose: backbone geometry idealization.

Workflow `params`:

- `fix_cis_omega` (bool, default `true`)
- `post_idealize_stiffness` (float, default `10.0`)
- `add_missing_residues` (bool, default `true`)
- `close_chainbreaks` (bool, default `true`)

Notes:

- `enabled` is internally forced to `true` in workflows.

## 5.2 `minimize`

Purpose: AMBER minimization.

Workflow `params`:

- `pre_idealize` (bool, default `false`) [workflow-only]
- `constrained` (bool, default `false`)
- `max_iterations` (int, default `0`)
- `tolerance` (float, default `2.39`)
- `stiffness` (float, default `10.0`)
- `max_outer_iterations` (int, default `3`)
- `split_chains_at_gaps` (bool, default `true`)
- `implicit_solvent` (bool, default `true`)

## 5.3 `repack`

Purpose: side-chain repacking (LigandMPNN repack mode).

Workflow `params`:

- `pre_idealize` (bool, default `false`) [workflow-only]
- `resfile` (string path, optional) [workflow-only]
- `model_type` (`protein_mpnn` | `ligand_mpnn` | `soluble_mpnn`, default `ligand_mpnn`)
- `temperature` (float, default `0.1`)
- `pack_side_chains` (bool, default `true`)
- `seed` (int, optional)
- `use_ligand_context` (bool, default `true`)
- `sc_num_denoising_steps` (int, default `3`)
- `sc_num_samples` (int, default `16`)

## 5.4 `relax`

Purpose: iterative repack + minimize cycles.

Workflow `params`:

- `pre_idealize` (bool, default `false`) [workflow-only]
- `resfile` (string path, optional) [workflow-only]
- `n_iterations` (int, default `5`) [workflow-only]
- Any `DesignConfig` field (same as `repack` list above)
- Any `RelaxConfig` field (same as `minimize` list above except `pre_idealize`)

Behavior note:

- Unknown keys are ignored with a warning (they do not fail parsing).

## 5.5 `mpnn`

Purpose: sequence design (single-pass MPNN).

Workflow `params`:

- `pre_idealize` (bool, default `false`) [workflow-only]
- `resfile` (string path, optional) [workflow-only]
- All `DesignConfig` fields (same list as `repack`)

## 5.6 `design`

Purpose: iterative design + minimize cycles.

Workflow `params`:

- `pre_idealize` (bool, default `false`) [workflow-only]
- `resfile` (string path, optional) [workflow-only]
- `n_iterations` (int, default `5`) [workflow-only]
- Any `DesignConfig` field
- Any `RelaxConfig` field

Behavior note:

- Unknown keys are ignored with a warning.

## 5.7 `renumber`

Purpose: remove insertion codes / renumber residues.

Workflow `params`:

- None required.

## 5.8 `analyze_interface`

Purpose: interface metrics and optional energetics analyses.

Workflow `params`:

- `constrained` (bool, default `false`) [workflow-only; controls internal relaxer]
- `chain_pairs`:
  - `"H:A,L:A"` string, or
  - list form like `[["H","A"], ["L","A"]]`
- `distance_cutoff` (float, default `8.0`)
- `calculate_binding_energy` (bool, default `true`)
- `calculate_sasa` (bool, default `false`)
- `calculate_shape_complementarity` (bool, default `false`)
- `relax_separated` (bool, default `false`)
- `relax_separated_iterations` (int, default `1`)
- `relax_separated_seed` (int, optional)
- `sasa_probe_radius` (float, default `1.4`)
- `per_position` (bool, default `false`)
- `alanine_scan` (bool, default `false`)
- `scan_chains` (list of chain IDs, optional)
- `position_relax` (`both` | `unbound` | `none`, default `none`)
- `position_csv` (path, optional)
- `max_scan_sites` (int, optional)
- `show_progress` (bool, default `false`)
- `quiet` (bool, default `false`)

## 6. Variables You Can Use in Conditions

Conditions read from structure metadata produced by prior operations.

Common keys:

- From `idealize`: `{chain_gaps}`
- From `minimize`/`relax`/`design`: `{final_energy}`
- From `repack`/`mpnn`/`design`/`relax`: `{sequence}`, `{native_sequence}`
- From `analyze_interface`:
  - `{dG}`
  - `{complex_energy}`
  - `{buried_sasa}`
  - `{sc_score}`
  - `{n_interface_residues}`
  - `{metrics.interface.dG}` (and related nested metrics keys)

You can also use arithmetic expressions combining variables, for example:

- `"{dG} < -12 and ..."` is not supported (`and`/`or` are not in grammar).
- Use numeric comparison only, e.g. `"{dG} + 0.1 * {buried_sasa} < 50"`.

## 7. Strictness and Validation Summary

- Workflow-level and block-level unknown keys raise errors.
- Node must be exactly one of `operation`, `iterate`, `beam`.
- `steps` lists must be non-empty.
- Numeric block controls (`n`, `max_n`, `width`, `rounds`, `expand`) must be `>= 1`.
- `beam.direction` must be `min` or `max`.
- Invalid `until` syntax fails at parse time.

## 8. Example Files in This Directory

- `simple_relax.yaml`: basic linear workflow
- `design_and_analyze.yaml`: linear design + interface analysis
- `multi_step_pipeline.yaml`: linear multi-operation pipeline
- `iterate_relax.yaml`: fixed iterate loop
- `converge_design.yaml`: convergence iterate loop
- `beam_design.yaml`: beam search with pruning
- `full_pipeline.yaml`: combined iterate + beam pipeline
