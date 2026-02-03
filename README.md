# Boundry

Protein engineering with LigandMPNN sequence design and OpenMM AMBER relaxation.

Boundry combines neural network-based sequence design ([LigandMPNN](https://github.com/dauparas/LigandMPNN)) with physics-based energy minimization (OpenMM AMBER), similar to Rosetta FastRelax and FastDesign protocols. It operates on protein structures in PDB or CIF format.

## Installation

Boundry requires `pdbfixer`, which is not available on PyPI and must be installed via `conda` or `mamba` from the `conda-forge` channel:

```bash
conda install -c conda-forge pdbfixer
```

Then install Boundry:

```bash
pip install boundry
```

For development:

```bash
conda install -c conda-forge pdbfixer
git clone https://github.com/bryanbriney/boundry.git
cd boundry
pip install -e ".[dev]"
```

## CLI

Boundry provides a subcommand-based CLI. Run `boundry --help` for a full list of commands.

```
boundry
├── idealize              Fix backbone geometry
├── minimize              Energy minimization (OpenMM AMBER)
├── repack                Repack side chains (LigandMPNN)
├── relax                 Iterative repack + minimize cycles
├── mpnn                  Sequence design (LigandMPNN)
├── design                Iterative design + minimize cycles
├── analyze-interface     Interface scoring and analysis
└── run                   Execute a YAML workflow
```

### Examples

```bash
# Fix backbone geometry
boundry idealize input.pdb idealized.pdb

# Energy minimization
boundry minimize input.pdb minimized.pdb
boundry minimize input.pdb minimized.pdb --constrained --pre-idealize

# Repack side chains (preserves sequence)
boundry repack input.pdb repacked.pdb
boundry repack input.pdb repacked.pdb --resfile design.resfile

# Iterative repack + minimize (like Rosetta FastRelax)
boundry relax input.pdb relaxed.pdb --n-iter 5

# Sequence design with LigandMPNN
boundry mpnn input.pdb designed.pdb
boundry mpnn input.pdb designed.pdb --temperature 0.2 --resfile design.resfile

# Iterative design + minimize (like Rosetta FastDesign)
boundry design input.pdb designed.pdb --n-iter 5

# Interface analysis
boundry analyze-interface complex.pdb
boundry analyze-interface complex.pdb --chains H:A,L:A --distance-cutoff 10.0

# Per-position interface energetics
boundry analyze-interface complex.pdb --per-position --alanine-scan
boundry analyze-interface complex.pdb --per-position --scan-chains A,B --position-csv results.csv

# Execute a YAML workflow
boundry run workflow.yaml
```

All commands that include energy minimization (`minimize`, `relax`, `design`) support `--pre-idealize` to fix backbone geometry before processing. Use `--verbose` or `-v` on any command for detailed logging.

## Python API

Core operations are available as standalone functions:

```python
from boundry import idealize, minimize, repack, relax, mpnn, design
from boundry import analyze_interface
from boundry import Structure, Workflow
```

### Operations

Each operation accepts a file path, PDB string, or `Structure` object and returns a `Structure` with metadata:

```python
from boundry import relax, design, analyze_interface
from boundry import Structure
from boundry.config import PipelineConfig, RelaxConfig

# Relax a structure (repack + minimize cycles)
result = relax("input.pdb", n_iterations=5)
result.write("relaxed.pdb")
print(f"Final energy: {result.metadata['final_energy']:.2f} kcal/mol")

# Design with custom config
config = PipelineConfig(
    design=DesignConfig(temperature=0.2, model_type="ligand_mpnn"),
    relax=RelaxConfig(constrained=True),
)
result = design("input.pdb", config=config, n_iterations=3)
result.write("designed.pdb")

# Chain operations
struct = Structure.from_file("input.pdb")
struct = idealize(struct)
struct = minimize(struct, pre_idealize=False)
struct.write("processed.pdb")
```

### Structure

The `Structure` class wraps a PDB string with metadata from operations:

```python
from boundry import Structure

# Load from file
struct = Structure.from_file("input.pdb")

# Create from string
struct = Structure(pdb_string="ATOM...")

# Write to file (auto-detects PDB/CIF from extension)
struct.write("output.pdb")
struct.write("output.cif")

# Access metadata from operations
print(struct.metadata)       # {"energy": ..., "sequence": ..., ...}
print(struct.source_path)    # Original file path (if loaded from file)
```

### Interface Analysis

```python
from boundry import analyze_interface
from boundry.config import InterfaceConfig, RelaxConfig
from boundry.relaxer import Relaxer

# Basic interface analysis (residues + SASA)
result = analyze_interface("complex.pdb")
print(result.interface_info.summary)
print(f"Buried SASA: {result.sasa.buried_sasa:.1f} sq. angstroms")

# Full analysis including binding energy
relaxer = Relaxer(RelaxConfig())
config = InterfaceConfig(
    enabled=True,
    chain_pairs=[("H", "A"), ("L", "A")],
    calculate_binding_energy=True,
    calculate_sasa=True,
    calculate_shape_complementarity=True,
)
result = analyze_interface("complex.pdb", config=config, relaxer=relaxer)
print(f"ddG: {result.binding_energy.binding_energy:.2f} kcal/mol")

# Per-position energetics (IAM-like dG_i and alanine scanning)
from pathlib import Path

config = InterfaceConfig(
    enabled=True,
    chain_pairs=[("H", "A"), ("L", "A")],
    calculate_binding_energy=True,
    calculate_sasa=True,
    per_position=True,
    alanine_scan=True,
    position_csv=Path("per_position_results.csv"),
)
result = analyze_interface("complex.pdb", config=config, relaxer=relaxer)
for row in result.per_position.rows:
    print(f"  {row.wt_resname} {row.chain_id}{row.residue_number}: "
          f"dG_i={row.dG_i}, ΔΔG={row.delta_ddG}")
```

## Workflows

Workflows define a linear sequence of operations in YAML. Each step's output is fed as input to the next step.

```yaml
# workflow.yaml
input: input.pdb
output: final.pdb

steps:
  - operation: idealize
    output: idealized.pdb  # optional intermediate output

  - operation: relax
    params:
      n_iterations: 3
      constrained: true
```

Run with the CLI or Python:

```bash
boundry run workflow.yaml
```

```python
from boundry import Workflow

workflow = Workflow.from_yaml("workflow.yaml")
result = workflow.run()
```

### Supported Operations

| Operation           | Description                              | Key Parameters                             |
| ------------------- | ---------------------------------------- | ------------------------------------------ |
| `idealize`          | Fix backbone geometry                    | `fix_cis_omega`, `add_missing_residues`    |
| `minimize`          | Energy minimization                      | `constrained`, `max_iterations`            |
| `repack`            | Repack side chains                       | `temperature`, `model_type`, `resfile`     |
| `relax`             | Iterative repack + minimize              | `n_iterations`, `temperature`, `constrained` |
| `mpnn`              | Sequence design                          | `temperature`, `model_type`, `resfile`     |
| `design`            | Iterative design + minimize              | `n_iterations`, `temperature`, `constrained` |
| `analyze_interface` | Interface scoring                        | `chain_pairs`, `distance_cutoff`, `per_position`, `alanine_scan` |

See `examples/workflows/` for more workflow examples.

## Resfiles

Boundry supports Rosetta-style resfiles for residue-level control during repacking and design:

```
NATAA          # Default: repack all residues (native amino acid types)
START
10 A ALLAA     # Design position 10 on chain A to any amino acid
15 A PIKAA HYW # Design position 15 to only His, Tyr, or Trp
20 A NOTAA CP  # Exclude Cys and Pro at position 20
25 A NATRO     # Keep position 25 fixed (no repacking)
30 A POLAR     # Restrict to polar amino acids
35 A APOLAR    # Restrict to apolar amino acids
```

Use with any repacking or design command:

```bash
boundry repack input.pdb output.pdb --resfile design.resfile
boundry design input.pdb output.pdb --resfile design.resfile
```

## Configuration

All configuration is done through dataclasses in `boundry.config`:

- **`DesignConfig`** — LigandMPNN settings (model type, temperature, seed)
- **`RelaxConfig`** — AMBER minimization settings (constrained, stiffness, solvent)
- **`IdealizeConfig`** — Backbone idealization settings
- **`InterfaceConfig`** — Interface analysis settings (cutoff, chain pairs, metrics)
- **`PipelineConfig`** — Bundles design + relax configs for iterative operations

## Development

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests (require OpenMM and LigandMPNN weights)
python -m pytest tests/ -m integration

# Run with coverage
pytest --cov=boundry tests/
```

## License

MIT
