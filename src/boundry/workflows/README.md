# Boundry Workflow Reference

This directory contains bundled YAML workflows for `boundry run`.

This document covers:

- The general workflow schema.
- Variable interpolation (`${key}`) and user-defined variables.
- CLI overrides (`key=value`).
- All supported step/block types.
- Every workflow operation and its workflow `params`.
- Condition syntax and valid variable references.
- Auto-generated output directory structure.

## 1. Workflow Structure

All workflows use this top-level shape:

```yaml
workflow_version: 1          # optional, defaults to 1
input: input.pdb             # required
project_path: results/       # optional, output root directory
seed: 42                     # optional, workflow-level seed
workers: 4                   # optional, parallel workers (default 1)
resfile_path: design.resfile # user-defined variable (see §2)
steps:                       # required, non-empty
  - operation: design
    params:
      resfile: ${resfile_path}
```

Reserved top-level keys: `workflow_version`, `input`, `project_path`, `seed`,
`workers`, `steps`. Any other top-level key is treated as a user-defined
variable (see §2).

### Deterministic Seeds

Set `seed` at the workflow level to enable deterministic seed derivation
for all stochastic operations (`repack`, `mpnn`, `relax`, `design`,
`analyze_interface`). Seeds are composed hierarchically:

- Top-level step: seed = `seed`
- Iterate cycle *c*: seed = `seed * 100000 + c`
- Beam round *r*, candidate *k*, expansion *e*: seed = `seed * 100000 + (r*10000 + k*100 + e)`

If a step explicitly sets `seed` in its `params`, that value takes
precedence over the workflow-derived seed. Omit the workflow `seed`
(or set it to `null`) for fully stochastic runs.

The `--seed` CLI flag overrides the YAML `seed` when both are present.

### Parallel Workers

Set `workers` at the workflow level to enable process-level parallelism.
A single shared process pool is created once at workflow start and reused
for all parallel operations (beam steps, population steps, and
per-position interface scans):

```yaml
workers: 4
steps:
  - beam:
      width: 3
      rounds: 10
      steps:
        - operation: design
        - operation: analyze_interface
```

How it works:

- `workers: 1` (default) runs everything sequentially — no process pool
  is created.
- `workers: N` (N > 1) creates a single shared `ProcessPoolExecutor`
  with the `spawn` start method. All parallel operations submit tasks to
  this pool.
- Beam steps execute each operation across all branches in parallel
  (step-level parallelism), with a barrier between steps.
- `analyze_interface` always runs in the main process so that
  per-position scans can fan out to the shared pool.
- The `--workers` / `-j` CLI flag overrides the YAML `workers` value.

Memory note: each worker process imports PyTorch/OpenMM independently
(~500MB–1GB each). Use a worker count appropriate for your system.

## 2. Variable Interpolation

Workflow YAML files support `${key}` variable interpolation (powered by
OmegaConf). Variables are resolved at load time, before any steps are
parsed or executed.

### 2.1 Built-in References

All reserved top-level keys with scalar values can be referenced:

- `${input}` — the input file path
- `${project_path}` — the output root directory
- `${seed}` — the workflow seed

```yaml
input: structures/my_protein.pdb
project_path: results
seed: 42
steps:
  - operation: idealize
  - operation: relax
```

### 2.2 User-Defined Variables

Any top-level key that is not a reserved key (`workflow_version`, `input`,
`project_path`, `seed`, `workers`, `steps`) is treated as a user-defined
variable. These are resolved identically to built-in references.

```yaml
input: input.pdb
project_path: ${project}/output       # cross-reference a user variable
project: my_project
resfile_path: design.resfile
steps:
  - operation: design
    params:
      resfile: ${resfile_path}         # -> design.resfile
```

Rules for user-defined variables:

- **Names** must be valid Python identifiers (letters, digits, underscores;
  cannot start with a digit). Hyphens are not allowed (`my-var` is
  invalid; use `my_var`).
- **Values** must be scalars (string, int, or float). Lists, dicts, and
  booleans are rejected.
- Numeric values are coerced to strings for interpolation, so `run_id: 42`
  makes `${run_id}` resolve to `"42"`.

### 2.3 Cross-References

Variables can reference other variables:

```yaml
input: input.pdb
base_dir: results
project_path: ${base_dir}/run_1       # -> results/run_1
steps:
  - operation: idealize
```

Circular references (e.g. `a: ${b}`, `b: ${a}`) are detected and raise
an error.

### 2.4 Environment Variables

The `${env:VAR}` and `${env:VAR,default}` resolver reads from
environment variables:

```yaml
input: ${env:INPUT_PDB,input.pdb}
project_path: ${env:OUTPUT_DIR,results}
steps:
  - operation: idealize
```

If the environment variable is unset and no default is provided, an
empty string is used.

### 2.5 Runtime Directory Naming

Output directories are created automatically using an opinionated naming
scheme (see §5). Runtime tokens like `{cycle}`, `{round}`, and `{rank}`
are handled internally and do not appear in workflow YAML files.

### 2.6 Errors

- Referencing an undefined variable (`${nonexistent}`) raises an error.
- Malformed syntax (`${_incomplete`) raises an error.
- `${steps}` is not referenceable (it is a non-scalar reserved key).

## 3. CLI Overrides

Extra arguments passed to `boundry run` are applied as config overrides
using `key=value` syntax. Overrides are merged into the YAML before
variable resolution, so they can change variable values that propagate
through `${...}` references.

```bash
boundry run workflow.yaml project_path=custom_results/ project=my_proj seed=99
```

- Any top-level key (reserved or user-defined) can be overridden.
- The `--seed` CLI flag takes precedence over a `seed=N` override.

Overrides can also be passed programmatically:

```python
from boundry import Workflow

wf = Workflow.from_yaml(
    "workflow.yaml",
    overrides=["project_path=results/", "project=custom_proj"],
)
```

## 4. Step Node Types

Each item in `steps:` must contain exactly one of:

- `operation` (single operation step)
- `iterate` (fixed or convergence loop)
- `beam` (population beam search)
- `checkpoint` (save named snapshot)
- `compare` (compute deltas against a checkpoint)

### 4.1 Operation Step

```yaml
- operation: relax
  params:
    n_iterations: 3
    constrained: true
```

Allowed keys:

- `operation` (required, string)
- `params` (optional mapping; `null` is treated as `{}`)

### 4.2 Iterate Block

```yaml
- iterate:
    n: 5
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

Notes:

- If `until` uses `delta(...)`, cycle 1 is treated as "not yet converged" (bootstrap).
- Seed injection is controlled by the workflow-level `seed` (see §1).

### 4.3 Beam Block

```yaml
- beam:
    width: 3
    rounds: 10
    expand: 5
    metric: dG
    direction: min
    until: "{dG} < -15.0"
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

Notes:

- Beam is first-class population flow: top-K candidates continue into later steps.
- `Workflow.run()` returns best rank-1 candidate.
- `Workflow.run_population()` returns final kept population.

### 4.4 Checkpoint Step

```yaml
- checkpoint: parent
```

Saves a snapshot of the current structure's metadata under a named key.
The name must be a valid Python identifier and cannot start with an
underscore. Checkpoint is a passthrough — it does not modify the
structure or its metadata.

Use checkpoints to capture a baseline for later comparison (see §4.5).

### 4.5 Compare Step

```yaml
- compare: parent
```

Computes numeric deltas between the current structure's metrics and the
metrics saved at the named checkpoint. The named checkpoint must have
been saved by a prior `checkpoint` step, otherwise an error is raised.

Compare stores results in metadata under the checkpoint name:

- `{name.delta.key}` — difference between current and checkpoint value
  for any numeric metric (e.g., `{parent.delta.dG}`)
- `{name.ref.key}` — the reference value from the checkpoint
  (e.g., `{parent.ref.dG}`)

Compare also writes a `compare.json` file to the step output directory
containing the delta and reference values.

Example — iterate until binding energy improves by at least 5 kcal/mol
relative to a checkpoint:

```yaml
- operation: relax
- checkpoint: parent
- iterate:
    until: "{parent.delta.dG} < -5.0"
    max_n: 10
    steps:
      - operation: design
      - operation: analyze_interface
      - compare: parent
```

## 5. Output Directory Structure

Output directories are created automatically under `project_path` (defaults
to the current working directory when omitted). You do not need to specify
per-step output paths — Boundry uses an opinionated naming scheme.

### 5.1 Top-level Steps

Each step creates a `{index}.{operation_name}/` directory:

```
project_path/
├── 0.idealize/
│   ├── idealized.pdb
│   └── metrics.json
├── 1.design/
│   ├── designed.pdb
│   ├── metrics.json
│   └── energy.json
└── 2.analyze_interface/
    ├── interface.json
    ├── metrics.json
    └── per_position.csv
```

### 5.2 Iterate Blocks

Iterate blocks create an `{index}.iterate/` directory with `cycle_{n}/`
subdirectories (1-indexed). Each cycle contains its own numbered step
directories:

```
project_path/
├── 0.idealize/
│   └── ...
└── 1.iterate/
    ├── cycle_1/
    │   ├── 0.relax/
    │   │   ├── relaxed.pdb
    │   │   ├── metrics.json
    │   │   └── energy.json
    │   └── 1.analyze_interface/
    │       ├── interface.json
    │       └── metrics.json
    ├── cycle_2/
    │   └── ...
    └── cycle_3/
        └── ...
```

### 5.3 Beam Blocks

Beam blocks create an `{index}.beam/` directory with `round_{n}/` and
`rank_{k}/` subdirectories. Candidates that are pruned (not in the top
`width`) are placed under `others/rank_{k}/`:

```
project_path/
└── 1.beam/
    ├── round_1/
    │   ├── rank_1/
    │   │   ├── 0.design/
    │   │   │   ├── designed.pdb
    │   │   │   ├── metrics.json
    │   │   │   └── energy.json
    │   │   └── 1.analyze_interface/
    │   │       ├── interface.json
    │   │       └── metrics.json
    │   ├── rank_2/
    │   │   └── ...
    │   └── others/
    │       ├── rank_3/
    │       │   └── ...
    │       └── rank_4/
    │           └── ...
    └── round_2/
        └── ...
```

### 5.4 Operation Output Files

Each operation writes a standard set of files:

| Operation | PDB stem | Additional files |
|-----------|----------|-----------------|
| `idealize` | `idealized.pdb` | `metrics.json` |
| `minimize` | `minimized.pdb` | `metrics.json` |
| `repack` | `repacked.pdb` | `metrics.json` |
| `mpnn` | `designed_mpnn.pdb` | `metrics.json` |
| `relax` | `relaxed.pdb` | `metrics.json`, `energy.json` |
| `design` | `designed.pdb` | `metrics.json`, `energy.json` |
| `renumber` | `renumbered.pdb` | `metrics.json` |
| `select_positions` | — | `metrics.json`, `selected_positions.json` |
| `analyze_interface` | — | `metrics.json`, `interface.json`, `per_position.csv`*, `alanine_scan.csv`* |
| `checkpoint` | — | — |
| `compare` | — | `compare.json` |

\* Written only when `per_position: true` or `alanine_scan: true` is set.

## 6. Condition Expressions (`until`)

`iterate.until` and `beam.until` use a safe parser (no `eval`).

### 6.1 Syntax

```text
condition := expr COMPARE expr
COMPARE   := < | > | <= | >= | == | !=
expr      := term ((+|-) term)*
term      := factor ((*|/) factor)*
factor    := NUMBER | VARIABLE | FUNCTION("(" expr ")") | "(" expr ")" | "-" factor
VARIABLE  := {dotted.path}
FUNCTION  := abs | delta
```

### 6.2 Examples

- `"{dG} < -10.0"`
- `"abs({final_energy}) < 1000"`
- `"delta({final_energy}) < 0.5"`
- `"{metrics.interface.dG} < -12.0"`

### 6.3 Variable Rules

- Variables are wrapped in `{...}`.
- Dotted paths are allowed (`{a.b.c}`).
- Private path segments are blocked (segments starting with `_`).
- Values must resolve to numeric (`int`/`float`) for arithmetic/comparisons.
- `delta(x)` compares current vs previous metadata value:
  - `abs(current(x) - previous(x))`
  - Outside loop/round contexts without previous metadata, this raises an error.

## 7. Operations and `params` Reference

Supported operation names:

- `idealize`
- `minimize`
- `repack`
- `relax`
- `mpnn`
- `design`
- `renumber`
- `analyze_interface`
- `select_positions`

## 7.1 `idealize`

Purpose: backbone geometry idealization.

Workflow `params`:

- `fix_cis_omega` (bool, default `true`)
- `post_idealize_stiffness` (float, default `10.0`)
- `add_missing_residues` (bool, default `true`)
- `close_chainbreaks` (bool, default `true`)

Notes:

- `enabled` is internally forced to `true` in workflows.

## 7.2 `minimize`

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

## 7.3 `repack`

Purpose: side-chain repacking (LigandMPNN repack mode).

Workflow `params`:

- `pre_idealize` (bool, default `false`) [workflow-only]
- `resfile` (string path, optional) [workflow-only]
- `design_spec` (auto-linked from metadata if omitted; see `select_positions`) [workflow-only]
- `model_type` (`protein_mpnn` | `ligand_mpnn` | `soluble_mpnn`, default `ligand_mpnn`)
- `temperature` (float, default `0.1`)
- `pack_side_chains` (bool, default `true`)
- `seed` (int, optional)
- `use_ligand_context` (bool, default `true`)
- `sc_num_denoising_steps` (int, default `3`)
- `sc_num_samples` (int, default `16`)

## 7.4 `relax`

Purpose: iterative repack + minimize cycles.

Workflow `params`:

- `pre_idealize` (bool, default `false`) [workflow-only]
- `resfile` (string path, optional) [workflow-only]
- `design_spec` (auto-linked from metadata if omitted; see `select_positions`) [workflow-only]
- `n_iterations` (int, default `5`) [workflow-only]
- Any `DesignConfig` field (same as `repack` list above)
- Any `RelaxConfig` field (same as `minimize` list above except `pre_idealize`)

Behavior note:

- Unknown keys are ignored with a warning (they do not fail parsing).

## 7.5 `mpnn`

Purpose: sequence design (single-pass MPNN).

Workflow `params`:

- `pre_idealize` (bool, default `false`) [workflow-only]
- `resfile` (string path, optional) [workflow-only]
- `design_spec` (auto-linked from metadata if omitted; see `select_positions`) [workflow-only]
- All `DesignConfig` fields (same list as `repack`)

## 7.6 `design`

Purpose: iterative design + minimize cycles.

Workflow `params`:

- `pre_idealize` (bool, default `false`) [workflow-only]
- `resfile` (string path, optional) [workflow-only]
- `design_spec` (auto-linked from metadata if omitted; see `select_positions`) [workflow-only]
- `n_iterations` (int, default `5`) [workflow-only]
- Any `DesignConfig` field
- Any `RelaxConfig` field

Behavior note:

- Unknown keys are ignored with a warning.

## 7.7 `renumber`

Purpose: remove insertion codes / renumber residues.

Workflow `params`:

- None required.

## 7.8 `analyze_interface`

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
- `seed` (int, optional)
- `sasa_probe_radius` (float, default `1.4`)
- `per_position` (bool, default `false`)
- `alanine_scan` (bool, default `false`)
- `scan_chains`:
  - `"H,L"` string, or
  - list form like `["H", "L"]`
- `position_relax` (`both` | `unbound` | `none`, default `none`)
- `position_csv` (path, optional)
- `max_scan_sites` (int, optional)
- `show_progress` (bool, default `false`)
- `quiet` (bool, default `false`)

## 7.9 `select_positions`

Purpose: select interface positions for targeted redesign based on
per-position energetics from a prior `analyze_interface` step.

Workflow `params`:

- `source` (`per_position` | `alanine_scan`, default `alanine_scan`)
- `metric` (string, default `ddG`) — `PositionRow` field to threshold on
- `threshold` (float, default `1.0`)
- `direction` (`above` | `below`, default `above`)
- `mode` (string, default `ALLAA`) — `ResidueMode` for selected positions
- `default_mode` (string, default `NATAA`) — `ResidueMode` for non-selected positions
- `allowed_aas` (string, optional) — amino acid letters for `PIKAA` mode (e.g. `"ACDEF"`)

Behavior:

- Reads a `PositionResult` from the structure's metadata (produced by
  `analyze_interface` with `per_position: true` or `alanine_scan: true`).
- Filters rows where the metric exceeds the threshold, builds a
  `DesignSpec`, and stores it in metadata under `design_spec`.
- Downstream `design`, `mpnn`, `relax`, and `repack` steps automatically
  pick up the `design_spec` from metadata when no explicit `resfile` is
  provided in their `params`.
- Stores `selected_positions` (int) in metadata for convergence conditions.

Example — converge when no positions remain above threshold:

```yaml
- beam:
    until: "{selected_positions} == 0"
    steps:
      - operation: analyze_interface
        params:
          alanine_scan: true
      - operation: select_positions
        params:
          source: alanine_scan
          metric: ddG
          threshold: 1.0
      - operation: design
```

Notes:

- `select_positions` is a workflow-only and Python API operation; there is
  no CLI subcommand.
- Rows where the metric value is `None` or where `scan_skipped` is `true`
  are silently excluded from selection.
- If zero positions are selected, a valid but empty `DesignSpec` is produced
  (downstream design steps will use the default mode for all positions).

## 8. Variables You Can Use in Conditions

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
- From `select_positions`:
  - `{selected_positions}` (int, number of positions selected for design)
  - `{selection_source}`, `{selection_metric}`, `{selection_threshold}`
- From `compare` (where *name* is the checkpoint name):
  - `{name.delta.key}` — delta between current and checkpoint for any
    numeric metric (e.g., `{parent.delta.dG}`)
  - `{name.ref.key}` — the reference value from the checkpoint
    (e.g., `{parent.ref.dG}`)

You can also use arithmetic expressions combining variables, for example:

- `"{dG} < -12 and ..."` is not supported (`and`/`or` are not in grammar).
- Use numeric comparison only, e.g. `"{dG} + 0.1 * {buried_sasa} < 50"`.

## 9. Strictness and Validation Summary

- Unknown top-level keys are treated as user-defined variables (see §2).
  They must be valid Python identifiers with scalar values.
- Block-level unknown keys raise errors.
- Node must be exactly one of `operation`, `iterate`, `beam`, `checkpoint`, `compare`.
- `steps` lists must be non-empty.
- Numeric block controls (`n`, `max_n`, `width`, `rounds`, `expand`) must be `>= 1`.
- Block-level `workers` is accepted but deprecated (emits a warning); use
  the top-level `workers` setting instead.
- `beam.direction` must be `min` or `max`.
- Invalid `until` syntax fails at parse time.
- Undefined `${...}` references, circular references, and malformed
  interpolation syntax all fail at load time.

## 10. Bundled Workflow Files

- `simple_relax.yaml`: basic linear workflow
- `design_and_analyze.yaml`: linear design + interface analysis
- `multi_step_pipeline.yaml`: linear multi-operation pipeline
- `iterate_relax.yaml`: fixed iterate loop
- `converge_design.yaml`: convergence iterate loop
- `beam_design.yaml`: beam search with pruning
- `beam_optimize.yaml`: beam search with adaptive position selection
- `full_pipeline.yaml`: combined iterate + beam pipeline
