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
- `select_positions` — Filter interface positions by metric and build a `DesignSpec` for downstream design ops (API only, no CLI subcommand)
- `analyze_interface` — Interface residue identification, binding energy, SASA, shape complementarity
- `optimize` — Beam-search interface optimization (alanine scan → design → score cycles)
- `ddg` — MD-ensemble mutation scoring / binding energy via four-state thermodynamic cycle

Operations that accept a `design_spec` parameter (`repack`, `relax`, `mpnn`, `design`) will also read it from `structure.metadata["design_spec"]` when not explicitly provided, enabling auto-linking from `select_positions`.

### Configuration System
All settings are defined as dataclasses in `config.py`: `DesignConfig`, `RelaxConfig`, `IdealizeConfig`, `InterfaceConfig`, `SelectPositionsConfig`, `PipelineConfig` (bundles design + relax for iterative operations), `OptimizeConfig`, and `DdGConfig`.

### CLI
Subcommand-based CLI built with Typer in `cli.py`, registered as `boundry = "boundry.cli:main"` in pyproject.toml. Each subcommand builds the appropriate config and calls the corresponding operation function from `operations.py`.

Subcommands: `idealize`, `minimize`, `repack`, `relax`, `mpnn`, `design`, `renumber`, `analyze-interface`, `optimize`, `ddg`.

### Key Modules
- **`operations.py`** — Core Python API. Standalone functions for each operation plus the `Structure` and `InterfaceAnalysisResult` data classes. This is the primary interface for programmatic use.
- **`optimize.py`** — Beam-search interface optimization. Iterative alanine scan → LigandMPNN design → AMBER minimization → binding energy scoring cycles with beam-width pruning and multi-campaign support.
- **`ddg.py`** — MD-ensemble ddG scoring. Constrained minimization, local OpenMM ensemble generation, four-state thermodynamic cycle (bound/unbound × WT/mutant), arithmetic averaging.
- **`_parallel.py`** — Shared process-level parallelism. `WorkPool` wraps `ProcessPoolExecutor` with `spawn` context. `ScanTask`/`ScanResult` for per-position interface scans. Scan workers use config-fingerprint-keyed `_worker_cache` for lazy Relaxer/Designer init.
- **`_sequence.py`** — Residue-map and per-chain sequence helpers (`_RESTYPE_3TO1`, `extract_residue_map()`, `_residue_map_to_sequences()`).
- **`designer.py`** — Wraps vendored LigandMPNN for sequence design. Supports three model variants: `protein_mpnn`, `ligand_mpnn`, `soluble_mpnn`. Integrates with Rosetta-style resfiles for residue-specific design control.
- **`relaxer.py`** — Wraps OpenMM AMBER for energy minimization. Two modes: unconstrained L-BFGS and constrained AmberRelaxation (AlphaFold-style). Automatically splits chains at gaps to prevent artificial gap closure.
- **`idealize.py`** — Optional preprocessing to fix backbone geometry while preserving dihedral angles.
- **`renumber.py`** — PDB insertion code handling. `has_insertion_codes()`, `renumber_pdb()`, `restore_numbering()`. Operations that need sequential numbering (minimize, relax, design) auto-renumber and restore.
- **`interface.py` / `binding_energy.py` / `surface_area.py`** — Interface analysis: residue identification, ddG calculation, SASA, shape complementarity.
- **`interface_position_energetics.py`** — Per-position interface energetics (residue removal and alanine scanning). `compute_position_energetics()` with sequential and parallel scan paths via the shared pool.
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
- `WorkPool` tests need top-level functions (not lambdas) for pickle compatibility with `spawn` context.

### Test Files
- **`test_operations.py`** — Tests for all operation functions, `Structure`, `InterfaceAnalysisResult`, input resolution helpers, and top-level imports. Uses `unittest.mock.patch` to mock heavy dependencies (Designer, Relaxer, OpenMM).
- **`test_cli.py`** — Tests for all CLI subcommands using `typer.testing.CliRunner`.
- **`test_cli_integration.py`** — Integration tests for CLI.
- **`test_parallel.py`** — Tests for `_parallel.py` parallel execution module.
- **`test_optimize.py`** — Tests for the optimize pipeline.
- **`test_ddg.py`** — Tests for the ddG scoring module.
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
- **`test_result_io.py`** — Tests for result serialization.
- **`test_utils.py`** — Tests for utility functions.
- **`test_weights.py`** — Tests for LigandMPNN weight management.
- **`test_designer_unit.py`** — Unit tests for Designer.
- **`test_relaxer_unit.py`** — Unit tests for Relaxer.
- **`test_designer_integration.py`** — Integration tests for Designer/LigandMPNN.
- **`test_relaxer_integration.py`** — Integration tests for Relaxer/OpenMM.
- **`test_interface_scoring_integration.py`** — Integration tests using real PDB structures (1VFB).
- **`test_pipeline_interface.py`** — Interface analysis integration tests.

Test fixtures in `conftest.py` provide: `weights_available`, `test_data_dir`, `small_peptide_pdb_string`, `small_peptide_pdb`, `small_peptide_cif`, `sample_resfile_content`, `sample_resfile`, `ubiquitin_pdb`, `antibody_antigen_pdb`, `antibody_antigen_pdb_string`, `heme_protein_pdb`.
