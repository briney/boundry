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

**Parallel execution:** A single `WorkPool` (context manager wrapping `ProcessPoolExecutor` with `spawn` context to avoid CUDA fork hazards) is created once in `run_population()` and stored as `self._pool`. `OperationTask`/`OperationResult` are the unified task/result types for all parallel dispatch via `_execute_operation_worker()`. Beam uses step-level parallelism: all branches execute step N in parallel (barrier sync), then step N+1. `_BranchState` tracks per-branch structure, seed, snapshots, and branch-local checkpoints. `analyze_interface` always runs in main process so per-position scans can fan out to the shared pool. `WorkflowConfig.workers` is the single parallelism control (global default=1); block-level `workers` is deprecated.

**Seed composition:** Hierarchical deterministic seed derivation via `_compose_seed()`. Top-level steps inherit workflow seed; iterate cycles derive `seed * 100000 + cycle`; beam branches derive `seed * 100000 + (round * 10000 + candidate * 100 + expansion)`. Step-level seed params take precedence.

**Progress monitoring:** `_progress.py` provides `WorkflowProgress` with Rich-based multi-level progress bars (workflow → block → inner). Enabled via `show_progress` parameter.

**Output specs:** `_OPERATION_OUTPUT_SPECS` defines what each operation writes (PDB, metrics JSON, CSV). Native writer functions handle operation-specific output files.

**Bundled workflows:** `src/boundry/workflows/` contains example YAML files and a comprehensive `README.md` reference guide covering the full workflow schema.

### Key Modules
- **`operations.py`** — Core Python API. Standalone functions for each operation plus the `Structure` and `InterfaceAnalysisResult` data classes. This is the primary interface for programmatic use.
- **`workflow.py`** — YAML workflow runner. `Workflow.from_yaml()` loads and validates, `Workflow.run()` executes. Handles iterate/beam blocks, checkpoints, and parallel dispatch.
- **`_parallel.py`** — Parallel execution for workflows. `WorkPool` wraps `ProcessPoolExecutor` with `spawn` context. `OperationTask`/`OperationResult` for unified operation dispatch, `ScanTask`/`ScanResult` for per-position interface scans. Scan workers use config-fingerprint-keyed `_worker_cache` for lazy Relaxer/Designer init.
- **`condition.py`** — Safe condition expression parser for workflow convergence (`until` fields). Grammar supports comparisons, arithmetic, `abs()`, `delta()`, `{dotted.path}` variable references.
- **`_progress.py`** — Rich-based workflow progress monitoring with multi-level progress bars.
- **`workflow_metadata.py`** — Metadata merge strategy (`merge_metadata()`), numeric metric extraction (`extract_numeric_metric()`), dotted path resolution.
- **`designer.py`** — Wraps vendored LigandMPNN for sequence design. Supports three model variants: `protein_mpnn`, `ligand_mpnn`, `soluble_mpnn`. Integrates with Rosetta-style resfiles for residue-specific design control.
- **`relaxer.py`** — Wraps OpenMM AMBER for energy minimization. Two modes: unconstrained L-BFGS and constrained AmberRelaxation (AlphaFold-style). Automatically splits chains at gaps to prevent artificial gap closure.
- **`idealize.py`** — Optional preprocessing to fix backbone geometry while preserving dihedral angles.
- **`renumber.py`** — PDB insertion code handling. `has_insertion_codes()`, `renumber_pdb()`, `restore_numbering()`. Operations that need sequential numbering (minimize, relax, design) auto-renumber and restore.
- **`interface.py` / `binding_energy.py` / `surface_area.py`** — Interface analysis: residue identification, ddG calculation, SASA, shape complementarity.
- **`interface_position_energetics.py`** — Per-position interface energetics (residue removal and alanine scanning). `compute_position_energetics()` with sequential and parallel scan paths via the shared pool.
- **`runner.py`** — Shared operation runners with invocation-aware output handling. `run_structure_operation()` and `run_interface_operation()` unify execution across API/CLI/workflow modes.
- **`invocation.py`** — Invocation/output policy helpers. `InvocationMode`, `OperationKind`, `OutputRequirement`, `OutputPolicy` manage output-path requirements across calling contexts.
- **`result_io.py`** — Result serialization and output-path helpers. `write_structure_output()`, `write_interface_json()`, `write_interface_csv()`.
- **`chain_gaps.py`** — Detects missing residues via residue number discontinuities and large C-N distances.
- **`resfile.py`** — Parses Rosetta-style resfiles (NATRO, NATAA, ALLAA, PIKAA, NOTAA, POLAR, APOLAR).
- **`structure_io.py`** — Unified PDB/CIF I/O with auto-detection and format conversion.
- **`weights.py`** — Manages LigandMPNN model weight downloads to `~/.boundry/weights/` (or `BOUNDRY_WEIGHTS_DIR`).
- **`utils.py`** — Scoring and I/O utilities: `suppress_stderr()`, `remove_waters()`, `filter_protein_only()`, `compute_sequence_recovery()`, `write_scorefile()`.
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
- Workflow execution tests mock `Workflow._run_operation` with `side_effect` dispatchers when multiple operations are involved. Dispatchers must accept `**kwargs` (for `pool` parameter).
- Operation runner unit tests call `Workflow._run_operation("name", struct, params)` directly and mock the underlying `boundry.operations.*` or `boundry.config.*`.
- `_run_analyze_interface` tests call the method directly since it remains a separate static method.
- `WorkPool` tests need top-level functions (not lambdas) for pickle compatibility with `spawn` context.

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
- **`test_renumber.py`** — Tests for PDB insertion code handling.
- **`test_structure_io.py`** — Tests for PDB/CIF I/O.
- **`test_surface_area.py`** — Tests for SASA and shape complementarity.
- **`test_binding_energy.py`** — Tests for binding energy calculation.
- **`test_interface.py`** — Tests for interface residue identification.
- **`test_interface_position_energetics.py`** — Tests for per-position energetics.
- **`test_config.py`** — Tests for configuration dataclasses.
- **`test_runner.py`** — Tests for operation runner functions.
- **`test_result_io.py`** — Tests for result serialization.
- **`test_utils.py`** — Tests for utility functions.
- **`test_weights.py`** — Tests for LigandMPNN weight management.
- **`test_designer_unit.py`** — Unit tests for Designer.
- **`test_designer_integration.py`** — Integration tests for Designer/LigandMPNN.
- **`test_relaxer_integration.py`** — Integration tests for Relaxer/OpenMM.
- **`test_cli_integration.py`** — Integration tests for CLI.
- **`test_interface_scoring_integration.py`** — Integration tests using real PDB structures (1VFB).
- **`test_pipeline_interface.py`** — Interface analysis integration tests.

Test fixtures in `conftest.py` provide: `weights_available`, `test_data_dir`, `small_peptide_pdb_string`, `small_peptide_pdb`, `small_peptide_cif`, `sample_resfile_content`, `sample_resfile`, `ubiquitin_pdb`, `antibody_antigen_pdb`, `antibody_antigen_pdb_string`, `heme_protein_pdb`.
