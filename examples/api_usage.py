"""Example usage of the Boundry Python API.

This script demonstrates the core operations and workflow system.
Requires OpenMM and LigandMPNN weights to be installed.
"""

from boundry import (
    Structure,
    Workflow,
    analyze_interface,
    design,
    idealize,
    minimize,
    repack,
    relax,
)
from boundry.config import (
    DesignConfig,
    InterfaceConfig,
    PipelineConfig,
    RelaxConfig,
)

# ---------------------------------------------------------------
# 1. Basic operations
# ---------------------------------------------------------------

# Load a structure
struct = Structure.from_file("input.pdb")

# Idealize backbone geometry
idealized = idealize(struct)
idealized.write("idealized.pdb")
print(f"Chain gaps detected: {idealized.metadata['chain_gaps']}")

# Energy minimization
minimized = minimize("input.pdb")
minimized.write("minimized.pdb")
print(f"Energy: {minimized.metadata['final_energy']:.2f} kcal/mol")

# Repack side chains (preserves sequence)
repacked = repack("input.pdb")
repacked.write("repacked.pdb")
print(f"Sequence: {repacked.metadata['sequence']}")

# ---------------------------------------------------------------
# 2. Iterative operations
# ---------------------------------------------------------------

# Relax: iterative repack + minimize (like Rosetta FastRelax)
relaxed = relax("input.pdb", n_iterations=5)
relaxed.write("relaxed.pdb")
print(f"Final energy: {relaxed.metadata['final_energy']:.2f} kcal/mol")

# Design: iterative sequence design + minimize (like Rosetta FastDesign)
designed = design("input.pdb", n_iterations=3)
designed.write("designed.pdb")
print(f"Designed sequence: {designed.metadata['sequence']}")
print(f"Native sequence:   {designed.metadata['native_sequence']}")

# ---------------------------------------------------------------
# 3. Custom configuration
# ---------------------------------------------------------------

config = PipelineConfig(
    design=DesignConfig(
        model_type="ligand_mpnn",
        temperature=0.2,
        seed=42,
    ),
    relax=RelaxConfig(
        constrained=True,
        stiffness=10.0,
        implicit_solvent=True,
    ),
)

result = design("input.pdb", config=config, n_iterations=3)
result.write("designed_custom.pdb")

# ---------------------------------------------------------------
# 4. Pre-idealization
# ---------------------------------------------------------------

# Any operation supports pre_idealize to fix backbone first
result = relax("input.pdb", pre_idealize=True, n_iterations=3)

# ---------------------------------------------------------------
# 5. Resfile-controlled design
# ---------------------------------------------------------------

# Design only specific positions using a resfile
result = design("input.pdb", resfile="design.resfile", n_iterations=3)

# ---------------------------------------------------------------
# 6. Interface analysis
# ---------------------------------------------------------------

# Basic analysis (interface residues + SASA)
result = analyze_interface("complex.pdb")
print(result.interface_info.summary)
print(f"Buried SASA: {result.sasa.buried_sasa:.1f} sq. angstroms")

# Full analysis with binding energy (requires relaxer)
from boundry.relaxer import Relaxer

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
print(f"Shape complementarity: {result.shape_complementarity.sc_score:.3f}")

# ---------------------------------------------------------------
# 7. Chaining operations
# ---------------------------------------------------------------

# Operations can be chained by passing Structure objects
struct = Structure.from_file("input.pdb")
struct = idealize(struct)
struct = minimize(struct)
struct.write("processed.pdb")

# ---------------------------------------------------------------
# 8. Workflow from YAML
# ---------------------------------------------------------------

workflow = Workflow.from_yaml("src/boundry/workflows/simple_relax.yaml")
result = workflow.run()
print(f"Workflow complete. Final structure has {len(result.pdb_string)} chars")
