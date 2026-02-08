# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Boundry is a Python package for protein engineering that combines neural network-based sequence design (LigandMPNN) with physics-based energy minimization (OpenMM AMBER). It operates on protein structures (PDB/CIF) through iterative design-relax cycles, similar to Rosetta FastRelax/Design protocols.

## Build & Development Commands

```bash
# Install in development mode
pip install -e .

# Run unit tests (skips integration tests by default)
python -m pytest tests/

# Run a specific test file
python -m pytest tests/test_operations.py

# Run a specific test class or method
python -m pytest tests/test_operations.py::TestIdealize::test_returns_structure

# Run integration tests (require OpenMM and LigandMPNN weights)
python -m pytest tests/ -m integration

# Run all tests except slow
python -m pytest tests/ -m 'not slow'

# Run with coverage
pytest --cov=boundry tests/

# Build distributions
python -m build
```

## Code Style

- Black formatter with 80-character line length
- isort with black-compatible profile
- The `LigandMPNN/` directory is excluded from formatting

## Architecture

### Operations Pattern
Core functionality is exposed as standalone functions in `operations.py`. Each function accepts flexible input (file path, PDB string, or `Structure` object), wraps the underlying `Designer`/`Relaxer`/analysis modules, and returns a `Structure` with metadata. Heavy dependencies are lazy-loaded inside each function body to keep import times fast.

Available operations:
- `idealize` — Fix backbone geometry
- `minimize` — Energy minimization (OpenMM AMBER)
- `repack` — Repack side chains (LigandMPNN, preserves sequence)
- `relax` — Iterative repack + minimize cycles
- `mpnn` — Sequence design (LigandMPNN)
- `design` — Iterative design + minimize cycles
- `renumber` — Remove PDB insertion codes (Kabat numbering)
- `select_positions` — Filter interface positions by metric and build a `DesignSpec` for downstream design ops (workflow/API only, no CLI subcommand)
- `analyze_interface` — Interface residue identification, binding energy, SASA, shape complementarity

Operations that accept a `design_spec` parameter (`repack`, `relax`, `mpnn`, `design`) will also read it from `structure.metadata["design_spec"]` when not explicitly provided, enabling auto-linking from `select_positions`.

### Configuration System
All settings are defined as dataclasses in `config.py`: `DesignConfig`, `RelaxConfig`, `IdealizeConfig`, `InterfaceConfig`, `SelectPositionsConfig`, and `PipelineConfig` (bundles design + relax for iterative operations). Workflow-specific: `WorkflowStep`, `WorkflowConfig`, `IterateBlock`, `BeamBlock`, `CheckpointStep`, `CompareStep`.

### CLI
Subcommand-based CLI built with Typer in `cli.py`, registered as `boundry = "boundry.cli:main"` in pyproject.toml. Each subcommand builds the appropriate config and calls the corresponding operation function from `operations.py`.

Subcommands: `idealize`, `minimize`, `repack`, `relax`, `mpnn`, `design`, `renumber`, `analyze-interface`, `run`.

### Workflow System
`workflow.py` provides a `Workflow` class that loads YAML files describing multi-step pipelines. Each step's output feeds as input to the next step. Configuration parameters are passed via `params` mapping in each step.

**Operation dispatch:** `_OPERATION_REGISTRY` maps operation names to `_SimpleSpec`/`_PipelineSpec` objects, dispatched by a single `_run_operation()` method. `_run_analyze_interface` remains separate due to its unique chain_pairs parsing and conditional Relaxer/Designer creation.

**Block types:**
- `iterate` — Repeat nested steps for `n` cycles or until a convergence condition (`until`). Supports `max_n` safety limit.
- `beam` — Population-based search with `width` candidates, `rounds` of expansion, metric-based ranking/pruning, and `expand` factor. Uses deferred output writing (two-phase execution).
- `checkpoint` — Save named snapshot of structure metadata for later comparison.
- `compare` — Compute deltas between current metadata and a named checkpoint.

**Convergence conditions:** Parsed by `condition.py` — a safe expression evaluator (no `eval`) supporting comparisons, arithmetic, and functions (`abs`, `delta`). Variable references use `{dotted.path}` syntax. Used in `until` fields for iterate and beam blocks. Convergence checking is deduplicated via `_check_convergence()` used by both block types.

**Variable interpolation:** OmegaConf-powered `${key}` resolution with `${env:VAR}` for environment variables. User-defined variables are any top-level key not in the known workflow schema keys. CLI overrides via dotlist syntax.

**Parallel execution:** `_parallel.py` provides `ProcessPoolExecutor` with `spawn` context (avoids CUDA fork hazards). `BranchTask`/`BranchResult` for beam-level parallelism, `StepTask`/`StepResult` for population-level parallelism. Dispatched by `_expand_beam_sequential()`/`_expand_beam_parallel()` and `_execute_step_sequential()`/`_execute_step_parallel()` based on effective worker count. Workers field: `WorkflowConfig.workers` (global default=1), `BeamBlock.workers`/`IterateBlock.workers` (per-block override, `None`=use global). Nested blocks in beam steps trigger fallback to sequential with warning.

**Seed composition:** Hierarchical deterministic seed derivation via `_compose_seed()`. Top-level steps inherit workflow seed; iterate cycles derive `seed * 100000 + cycle`; beam branches derive `seed * 100000 + (round * 10000 + candidate * 100 + expansion)`. Step-level seed params take precedence.

**Progress monitoring:** `_progress.py` provides `WorkflowProgress` with Rich-based multi-level progress bars (workflow → block → inner). Enabled via `show_progress` parameter.

**Output specs:** `_OPERATION_OUTPUT_SPECS` defines what each operation writes (PDB, metrics JSON, CSV). Native writer functions handle operation-specific output files.

**Bundled workflows:** `src/boundry/workflows/` contains example YAML files and a comprehensive `README.md` reference guide covering the full workflow schema.

### Key Modules
- **`operations.py`** — Core Python API. Standalone functions for each operation plus the `Structure` and `InterfaceAnalysisResult` data classes. This is the primary interface for programmatic use.
- **`workflow.py`** — YAML workflow runner. `Workflow.from_yaml()` loads and validates, `Workflow.run()` executes. Handles iterate/beam blocks, checkpoints, and parallel dispatch.
- **`_parallel.py`** — Parallel execution for workflows. `ProcessPoolExecutor` with `spawn` context. `BranchTask`/`BranchResult` for beam, `StepTask`/`StepResult` for steps, `ScanTask`/`ScanResult` for per-position interface scans.
- **`condition.py`** — Safe condition expression parser for workflow convergence (`until` fields). Grammar supports comparisons, arithmetic, `abs()`, `delta()`, `{dotted.path}` variable references.
- **`_progress.py`** — Rich-based workflow progress monitoring with multi-level progress bars.
- **`workflow_metadata.py`** — Metadata merge strategy (`merge_metadata()`), numeric metric extraction (`extract_numeric_metric()`), dotted path resolution.
- **`designer.py`** — Wraps vendored LigandMPNN for sequence design. Supports three model variants: `protein_mpnn`, `ligand_mpnn`, `soluble_mpnn`. Integrates with Rosetta-style resfiles for residue-specific design control.
- **`relaxer.py`** — Wraps OpenMM AMBER for energy minimization. Two modes: unconstrained L-BFGS and constrained AmberRelaxation (AlphaFold-style). Automatically splits chains at gaps to prevent artificial gap closure.
- **`idealize.py`** — Optional preprocessing to fix backbone geometry while preserving dihedral angles.
- **`renumber.py`** — PDB insertion code handling. `has_insertion_codes()`, `renumber_pdb()`, `restore_numbering()`. Operations that need sequential numbering (minimize, relax, design) auto-renumber and restore.
- **`interface.py` / `binding_energy.py` / `surface_area.py`** — Interface analysis: residue identification, ddG calculation, SASA, shape complementarity.
- **`chain_gaps.py`** — Detects missing residues via residue number discontinuities and large C-N distances.
- **`resfile.py`** — Parses Rosetta-style resfiles (NATRO, NATAA, ALLAA, PIKAA, NOTAA, POLAR, APOLAR).
- **`structure_io.py`** — Unified PDB/CIF I/O with auto-detection and format conversion.
- **`weights.py`** — Manages LigandMPNN model weight downloads to `~/.boundry/weights/` (or `BOUNDRY_WEIGHTS_DIR`).
- **`config.py`** — All configuration dataclasses.
- **`cli.py`** — Typer-based CLI with subcommands.

### Vendored Code
`src/boundry/LigandMPNN/` contains the vendored LigandMPNN implementation and OpenFold utilities. This code is excluded from formatting rules and should be modified carefully.

### Lazy Loading
Heavy dependencies (PyTorch, OpenMM) are loaded on first use inside function bodies in `operations.py` and other modules to keep `import boundry` fast. The `__init__.py` re-exports operations and data classes from `operations.py` without triggering heavy imports.

## Testing

Tests are in `tests/` using pytest. Two custom markers:
- `integration` — requires OpenMM and/or LigandMPNN weights (skipped by default via `addopts = "-m 'not integration'"`)
- `slow` — long-running tests

### Test Patterns
- Workflow execution tests mock `Workflow._run_operation` with `side_effect` dispatchers when multiple operations are involved.
- Operation runner unit tests call `Workflow._run_operation("name", struct, params)` directly and mock the underlying `boundry.operations.*` or `boundry.config.*`.
- `_run_analyze_interface` tests call the method directly since it remains a separate static method.

### Test Files
- **`test_operations.py`** — Tests for all operation functions, `Structure`, `InterfaceAnalysisResult`, input resolution helpers, and top-level imports. Uses `unittest.mock.patch` to mock heavy dependencies (Designer, Relaxer, OpenMM).
- **`test_cli.py`** — Tests for all CLI subcommands using `typer.testing.CliRunner`.
- **`test_workflow.py`** — Tests for YAML parsing, validation, step dispatching, and workflow execution including iterate/beam blocks.
- **`test_parallel.py`** — Tests for `_parallel.py` parallel execution module.
- **`test_condition.py`** — Tests for condition expression parser.
- **`test_progress.py`** — Tests for `WorkflowProgress` monitoring.
- **`test_select_positions.py`** — Tests for `select_positions` operation.
- **`test_idealize.py`** — Tests for backbone idealization and dihedral extraction.
- **`test_chain_gaps.py`** — Tests for chain gap detection.
- **`test_resfile.py`** — Tests for resfile parsing.
- **`test_structure_io.py`** — Tests for PDB/CIF I/O.
- **`test_surface_area.py`** — Tests for SASA and shape complementarity.
- **`test_config.py`** — Tests for configuration dataclasses.
- **`test_interface_scoring_integration.py`** — Integration tests using real PDB structures (1VFB).
- **`test_pipeline_interface.py`** — Interface analysis integration tests.

Test fixtures in `conftest.py` provide: `weights_available`, `test_data_dir`, `small_peptide_pdb_string`, `small_peptide_pdb`, `small_peptide_cif`, `sample_resfile_content`, `sample_resfile`, `ubiquitin_pdb`, `antibody_antigen_pdb`, `antibody_antigen_pdb_string`, `heme_protein_pdb`.
