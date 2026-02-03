"""Example: interface analysis with per-position energetics.

Demonstrates the analyze_interface operation with per-position dG_i
(residue removal) and alanine scanning. These follow Rosetta sign
conventions:

- dG = E_bound - E_unbound  (negative = favorable binding)
- ddG = dG_ala - dG_wt      (positive = destabilising hotspot)

Requires OpenMM to be installed.
"""

from pathlib import Path

from boundry import analyze_interface
from boundry.config import InterfaceConfig, RelaxConfig
from boundry.relaxer import Relaxer

INPUT_PDB = "complex.pdb"

# ---------------------------------------------------------------
# 1. Basic interface analysis
# ---------------------------------------------------------------

result = analyze_interface(INPUT_PDB)
print(result.interface_info.summary)
print(f"Buried SASA: {result.sasa.buried_sasa:.1f} sq. angstroms")

# ---------------------------------------------------------------
# 2. Full analysis with binding energy
# ---------------------------------------------------------------

relaxer = Relaxer(RelaxConfig())
config = InterfaceConfig(
    enabled=True,
    chain_pairs=[("H", "A"), ("L", "A")],
    calculate_binding_energy=True,
    calculate_sasa=True,
    calculate_shape_complementarity=True,
)

result = analyze_interface(INPUT_PDB, config=config, relaxer=relaxer)
print(f"Binding energy: {result.binding_energy.binding_energy:.2f} kcal/mol")
print(f"Shape complementarity: {result.shape_complementarity.sc_score:.3f}")

# ---------------------------------------------------------------
# 3. Per-position energetics (dG_i and alanine scan)
# ---------------------------------------------------------------

config = InterfaceConfig(
    enabled=True,
    chain_pairs=[("H", "A"), ("L", "A")],
    calculate_binding_energy=True,
    calculate_sasa=True,
    per_position=True,          # IAM-like per-residue dG_i
    alanine_scan=True,          # per-residue AlaScan ddG
    position_repack="both",     # repack bound + unbound states
    position_csv=Path("per_position_results.csv"),
)

result = analyze_interface(INPUT_PDB, config=config, relaxer=relaxer)

# Overall WT binding energy (Rosetta sign)
print(f"dG_wt: {result.per_position.dG_wt:.2f} kcal/mol")

# Per-residue results
for row in result.per_position.rows:
    label = f"{row.wt_resname} {row.chain_id}{row.residue_number}"
    parts = [f"{label:<10}"]
    if row.dG_i is not None:
        parts.append(f"dG_i={row.dG_i:+.2f}")
    if row.delta_ddG is not None:
        parts.append(f"ddG={row.delta_ddG:+.2f}")
    elif row.scan_skipped:
        parts.append(row.skip_reason)
    print("  ".join(parts))

print(f"\nResults written to: per_position_results.csv")

# ---------------------------------------------------------------
# 4. Restrict scanning to specific chains
# ---------------------------------------------------------------

config = InterfaceConfig(
    enabled=True,
    chain_pairs=[("H", "A"), ("L", "A")],
    calculate_binding_energy=True,
    calculate_sasa=True,
    per_position=True,
    alanine_scan=True,
    scan_chains=["A"],          # only scan antigen chain
    max_scan_sites=20,          # limit to 20 residues
)

result = analyze_interface(INPUT_PDB, config=config, relaxer=relaxer)
print(f"\nScanned {len(result.per_position.rows)} residues on chain A")
