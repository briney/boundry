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
- `analyze_interface` — Interface residue identification, binding energy, SASA, shape complementarity

### Configuration System
All settings are defined as dataclasses in `config.py`: `DesignConfig`, `RelaxConfig`, `IdealizeConfig`, `InterfaceConfig`, and `PipelineConfig` (bundles design + relax for iterative operations). Workflow-specific: `WorkflowStep`, `WorkflowConfig`.

### CLI
Subcommand-based CLI built with Typer in `cli.py`, registered as `boundry = "boundry.cli:main"` in pyproject.toml. Each subcommand builds the appropriate config and calls the corresponding operation function from `operations.py`.

Subcommands: `idealize`, `minimize`, `repack`, `relax`, `mpnn`, `design`, `analyze-interface`, `run`.

### Workflow System
`workflow.py` provides a `Workflow` class that loads YAML files describing a linear sequence of operations. Each step's output feeds as input to the next step. Supported operations match the core operations. Configuration parameters are passed via `params` mapping in each step.

### Key Modules
- **`operations.py`** — Core Python API. Standalone functions for each operation plus the `Structure` and `InterfaceAnalysisResult` data classes. This is the primary interface for programmatic use.
- **`workflow.py`** — YAML workflow runner. `Workflow.from_yaml()` loads and validates, `Workflow.run()` executes sequentially.
- **`designer.py`** — Wraps vendored LigandMPNN for sequence design. Supports three model variants: `protein_mpnn`, `ligand_mpnn`, `soluble_mpnn`. Integrates with Rosetta-style resfiles for residue-specific design control.
- **`relaxer.py`** — Wraps OpenMM AMBER for energy minimization. Two modes: unconstrained L-BFGS and constrained AmberRelaxation (AlphaFold-style). Automatically splits chains at gaps to prevent artificial gap closure.
- **`idealize.py`** — Optional preprocessing to fix backbone geometry while preserving dihedral angles.
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

### Test Files
- **`test_operations.py`** — Tests for all operation functions, `Structure`, `InterfaceAnalysisResult`, input resolution helpers, and top-level imports. Uses `unittest.mock.patch` to mock heavy dependencies (Designer, Relaxer, OpenMM).
- **`test_cli.py`** — Tests for all CLI subcommands using `typer.testing.CliRunner`.
- **`test_workflow.py`** — Tests for YAML parsing, validation, step dispatching, and workflow execution.
- **`test_idealize.py`** — Tests for backbone idealization and dihedral extraction.
- **`test_chain_gaps.py`** — Tests for chain gap detection.
- **`test_resfile.py`** — Tests for resfile parsing.
- **`test_structure_io.py`** — Tests for PDB/CIF I/O.
- **`test_surface_area.py`** — Tests for SASA and shape complementarity.
- **`test_config.py`** — Tests for configuration dataclasses.
- **`test_interface_scoring_integration.py`** — Integration tests using real PDB structures (1VFB).
- **`test_pipeline_interface.py`** — Interface analysis integration tests.

Test fixtures in `conftest.py` provide: `weights_available`, `test_data_dir`, `small_peptide_pdb_string`, `small_peptide_pdb`, `small_peptide_cif`, `sample_resfile_content`, `sample_resfile`, `ubiquitin_pdb`, `antibody_antigen_pdb`, `antibody_antigen_pdb_string`, `heme_protein_pdb`.
