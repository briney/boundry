"""OpenMM AMBER relaxation wrapper."""

import io
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from openmm import Platform
from openmm import app as openmm_app
from openmm import openmm, unit
from pdbfixer import PDBFixer

from boundry.chain_gaps import (
    detect_chain_gaps,
    get_gap_summary,
    restore_chain_ids,
    split_chains_at_gaps,
)
from boundry.config import RelaxConfig
from boundry.idealize import extract_ligands, restore_ligands
from boundry.utils import filter_protein_only

# Add vendored LigandMPNN to path for OpenFold imports
# Must happen before importing from openfold
LIGANDMPNN_PATH = Path(__file__).parent / "LigandMPNN"
if str(LIGANDMPNN_PATH) not in sys.path:
    sys.path.insert(0, str(LIGANDMPNN_PATH))

from openfold.np import protein  # noqa: E402
from openfold.np.relax.relax import AmberRelaxation  # noqa: E402

logger = logging.getLogger(__name__)


class Relaxer:
    """Wrapper for OpenMM AMBER relaxation."""

    def __init__(self, config: RelaxConfig):
        self.config = config
        self._use_gpu: Optional[bool] = None

    def _check_gpu_available(self) -> bool:
        """Check if CUDA is available for OpenMM."""
        if self._use_gpu is not None:
            return self._use_gpu

        for i in range(Platform.getNumPlatforms()):
            if Platform.getPlatform(i).getName() == "CUDA":
                try:
                    platform = Platform.getPlatformByName("CUDA")
                    system = openmm.System()
                    system.addParticle(1.0)
                    integrator = openmm.VerletIntegrator(0.001)
                    _ctx = openmm.Context(  # noqa: F841
                        system, integrator, platform
                    )
                    del _ctx
                    self._use_gpu = True
                    logger.info("OpenMM CUDA platform detected, using GPU")
                    return True
                except Exception:
                    logger.warning(
                        "OpenMM CUDA platform found but not functional, "
                        "falling back to CPU"
                    )
                    break

        self._use_gpu = False
        logger.info("OpenMM CUDA not available, using CPU")
        return False

    def relax(self, pdb_string: str) -> Tuple[str, dict, np.ndarray]:
        """
        Relax a structure from PDB string.

        Uses unconstrained minimization by default, or constrained
        AmberRelaxation if config.constrained is True.

        If split_chains_at_gaps is enabled, chains will be split at detected
        gaps before minimization to prevent artificial gap closure.

        Ligands (non-water HETATM records) are extracted before relaxation
        and restored afterward, since standard AMBER force fields cannot
        parameterize arbitrary ligands.

        Args:
            pdb_string: PDB file contents as string

        Returns:
            Tuple of (relaxed_pdb_string, debug_info, violations)
        """
        # Extract ligands before relaxation (AMBER can't parameterize them)
        protein_pdb, ligand_lines = extract_ligands(pdb_string)
        if ligand_lines.strip():
            logger.debug(
                "Extracted ligands for separate handling during relaxation"
            )

        # Detect and handle chain gaps if configured
        chain_mapping = {}
        if self.config.split_chains_at_gaps:
            gaps = detect_chain_gaps(protein_pdb)
            if gaps:
                logger.info(get_gap_summary(gaps))
                protein_pdb, chain_mapping = split_chains_at_gaps(
                    protein_pdb, gaps
                )

        if self.config.constrained:
            prot = protein.from_pdb_string(protein_pdb)
            relaxed_pdb, debug_info, violations = self.relax_protein(prot)
        else:
            relaxed_pdb, debug_info, violations = self._relax_unconstrained(
                protein_pdb
            )

        # Restore original chain IDs if chains were split
        if chain_mapping:
            relaxed_pdb = restore_chain_ids(relaxed_pdb, chain_mapping)
            debug_info["chains_split"] = True
            debug_info["gaps_detected"] = len(
                [k for k, v in chain_mapping.items() if k != v]
            )

        # Restore ligands after relaxation
        relaxed_pdb = restore_ligands(relaxed_pdb, ligand_lines)

        return relaxed_pdb, debug_info, violations

    def relax_pdb_file(self, pdb_path: Path) -> Tuple[str, dict, np.ndarray]:
        """
        Relax a PDB file.

        Args:
            pdb_path: Path to input PDB file

        Returns:
            Tuple of (relaxed_pdb_string, debug_info, violations)
        """
        with open(pdb_path) as f:
            pdb_string = f.read()
        return self.relax(pdb_string)

    def relax_protein(self, prot) -> Tuple[str, dict, np.ndarray]:
        """
        Relax a Protein object using OpenFold's AmberRelaxation.

        Args:
            prot: OpenFold Protein object

        Returns:
            Tuple of (relaxed_pdb_string, debug_info, violations)
        """
        use_gpu = self._check_gpu_available()

        relaxer = AmberRelaxation(
            max_iterations=self.config.max_iterations,
            tolerance=self.config.tolerance,
            stiffness=self.config.stiffness,
            exclude_residues=[],
            max_outer_iterations=self.config.max_outer_iterations,
            use_gpu=use_gpu,
        )

        logger.info(
            f"Running AMBER relaxation (max_iter={self.config.max_iterations}, "
            f"stiffness={self.config.stiffness}, gpu={use_gpu})"
        )

        relaxed_pdb, debug_data, violations = relaxer.process(prot=prot)

        logger.info(
            f"Relaxation complete: E_init={debug_data['initial_energy']:.2f}, "
            f"E_final={debug_data['final_energy']:.2f}, "
            f"RMSD={debug_data['rmsd']:.3f} A"
        )

        return relaxed_pdb, debug_data, violations

    def _relax_unconstrained(
        self, pdb_string: str
    ) -> Tuple[str, dict, np.ndarray]:
        """
        Bare-bones unconstrained OpenMM minimization.

        No position restraints, no violation checking, uses OpenMM defaults.
        This is the default minimization mode.

        Note: Ligands are extracted at the relax() level before calling this.

        Args:
            pdb_string: PDB file contents as string (protein-only)

        Returns:
            Tuple of (relaxed_pdb_string, debug_info, violations)
        """
        ENERGY = unit.kilocalories_per_mole
        LENGTH = unit.angstroms

        use_gpu = self._check_gpu_available()

        logger.info(
            f"Running unconstrained OpenMM minimization "
            f"(max_iter={self.config.max_iterations}, gpu={use_gpu})"
        )

        # Use pdbfixer to add missing atoms and terminal groups
        fixer = PDBFixer(pdbfile=io.StringIO(pdb_string))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

        # Create force field and system
        force_field = openmm_app.ForceField(
            "amber14-all.xml", "amber14/tip3pfb.xml"
        )

        # Use Modeller to add hydrogens
        modeller = openmm_app.Modeller(fixer.topology, fixer.positions)
        modeller.addHydrogens(force_field)

        # Create system with HBonds constraints (standard for minimization)
        system = force_field.createSystem(
            modeller.topology, constraints=openmm_app.HBonds
        )

        # Create integrator and simulation
        integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
        platform = openmm.Platform.getPlatformByName(
            "CUDA" if use_gpu else "CPU"
        )
        simulation = openmm_app.Simulation(
            modeller.topology, system, integrator, platform
        )
        simulation.context.setPositions(modeller.positions)

        # Get initial energy
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        einit = state.getPotentialEnergy().value_in_unit(ENERGY)
        posinit = state.getPositions(asNumpy=True).value_in_unit(LENGTH)

        # Minimize with default tolerance
        if self.config.max_iterations > 0:
            simulation.minimizeEnergy(maxIterations=self.config.max_iterations)
        else:
            simulation.minimizeEnergy()

        # Get final state
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        efinal = state.getPotentialEnergy().value_in_unit(ENERGY)
        pos = state.getPositions(asNumpy=True).value_in_unit(LENGTH)

        # Calculate RMSD
        rmsd = np.sqrt(np.sum((posinit - pos) ** 2) / len(posinit))

        # Write output PDB
        output = io.StringIO()
        openmm_app.PDBFile.writeFile(
            simulation.topology, state.getPositions(), output
        )
        relaxed_pdb = output.getvalue()

        debug_data = {
            "initial_energy": einit,
            "final_energy": efinal,
            "rmsd": rmsd,
            "attempts": 1,
        }

        logger.info(
            f"Minimization complete: E_init={einit:.2f}, "
            f"E_final={efinal:.2f}, RMSD={rmsd:.3f} A"
        )

        # No violations tracking in unconstrained mode
        violations = np.zeros(0)

        return relaxed_pdb, debug_data, violations

    def _relax_direct(self, pdb_string: str) -> Tuple[str, dict, np.ndarray]:
        """
        Direct OpenMM minimization without pdbfixer.

        This is a simpler approach that works for already-complete structures
        (like those from LigandMPNN with packed side chains).

        Args:
            pdb_string: PDB file contents as string

        Returns:
            Tuple of (relaxed_pdb_string, debug_info, violations)
        """
        ENERGY = unit.kilocalories_per_mole
        LENGTH = unit.angstroms

        use_gpu = self._check_gpu_available()

        logger.info(
            f"Running direct OpenMM minimization "
            f"(max_iter={self.config.max_iterations}, "
            f"stiffness={self.config.stiffness}, gpu={use_gpu})"
        )

        # Parse PDB
        pdb_file = io.StringIO(pdb_string)
        pdb = openmm_app.PDBFile(pdb_file)

        # Create force field and system
        force_field = openmm_app.ForceField(
            "amber14-all.xml", "amber14/tip3pfb.xml"
        )

        # Use Modeller to add hydrogens (doesn't require pdbfixer)
        modeller = openmm_app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(force_field)

        # Create system with constraints on hydrogen bonds
        system = force_field.createSystem(
            modeller.topology, constraints=openmm_app.HBonds
        )

        # Add position restraints if stiffness > 0
        if self.config.stiffness > 0:
            self._add_restraints(
                system, modeller, self.config.stiffness * ENERGY / (LENGTH**2)
            )

        # Create integrator and simulation
        integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
        platform = openmm.Platform.getPlatformByName(
            "CUDA" if use_gpu else "CPU"
        )
        simulation = openmm_app.Simulation(
            modeller.topology, system, integrator, platform
        )
        simulation.context.setPositions(modeller.positions)

        # Get initial energy
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        einit = state.getPotentialEnergy().value_in_unit(ENERGY)
        posinit = state.getPositions(asNumpy=True).value_in_unit(LENGTH)

        # Minimize
        # OpenMM minimizeEnergy tolerance is in kJ/mol/nm (gradient threshold)
        tolerance = (
            self.config.tolerance * unit.kilojoules_per_mole / unit.nanometer
        )
        simulation.minimizeEnergy(
            maxIterations=self.config.max_iterations, tolerance=tolerance
        )

        # Get final state
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        efinal = state.getPotentialEnergy().value_in_unit(ENERGY)
        pos = state.getPositions(asNumpy=True).value_in_unit(LENGTH)

        # Calculate RMSD
        rmsd = np.sqrt(np.sum((posinit - pos) ** 2) / len(posinit))

        # Write output PDB
        output = io.StringIO()
        openmm_app.PDBFile.writeFile(
            simulation.topology, state.getPositions(), output
        )
        relaxed_pdb = output.getvalue()

        debug_data = {
            "initial_energy": einit,
            "final_energy": efinal,
            "rmsd": rmsd,
            "attempts": 1,
        }

        logger.info(
            f"Relaxation complete: E_init={einit:.2f}, "
            f"E_final={efinal:.2f}, RMSD={rmsd:.3f} A"
        )

        # No violations tracking in direct mode
        violations = np.zeros(0)

        return relaxed_pdb, debug_data, violations

    def _add_restraints(self, system, modeller, stiffness):
        """Add harmonic position restraints to heavy atoms."""
        force = openmm.CustomExternalForce(
            "0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
        )
        # Convert stiffness to OpenMM internal units (kJ/mol/nm^2)
        stiffness_value = stiffness.value_in_unit(
            unit.kilojoules_per_mole / unit.nanometers**2
        )
        force.addGlobalParameter("k", stiffness_value)
        for p in ["x0", "y0", "z0"]:
            force.addPerParticleParameter(p)

        for i, atom in enumerate(modeller.topology.atoms()):
            if atom.element.name != "hydrogen":
                # Convert positions to nanometers (OpenMM internal units)
                pos = modeller.positions[i].value_in_unit(unit.nanometers)
                force.addParticle(i, pos)

        logger.debug(
            f"Added restraints to {force.getNumParticles()} / "
            f"{system.getNumParticles()} atoms"
        )
        system.addForce(force)

    def get_energy_breakdown(self, pdb_string: str) -> dict:
        """
        Get individual force field energy terms for a structure.

        Uses PDBFixer for robust hydrogen handling and amber14-all.xml for
        consistency with the relaxation force field. When implicit_solvent
        is enabled (default), includes GBn2 implicit solvation which adds
        a CustomGBForce (polar GB solvation + nonpolar SA term).

        Args:
            pdb_string: PDB file contents as string

        Returns:
            Dictionary with energy breakdown by force type
        """
        try:
            ENERGY = unit.kilocalories_per_mole

            # Strip non-protein content (ligands, glycans, waters)
            # before OpenMM processing -- AMBER can't parameterize them
            pdb_string = filter_protein_only(pdb_string)

            # Use pdbfixer to handle missing atoms/H
            fixer = PDBFixer(pdbfile=io.StringIO(pdb_string))
            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()

            # Force field selection
            if self.config.implicit_solvent:
                force_field = openmm_app.ForceField(
                    "amber14-all.xml", "implicit/gbn2.xml"
                )
            else:
                force_field = openmm_app.ForceField(
                    "amber14-all.xml", "amber14/tip3pfb.xml"
                )

            # Use Modeller to add hydrogens
            modeller = openmm_app.Modeller(fixer.topology, fixer.positions)
            modeller.addHydrogens(force_field)

            # Create system
            system = force_field.createSystem(
                modeller.topology,
                constraints=openmm_app.HBonds,
                soluteDielectric=1.0,
                solventDielectric=78.5,
            )

            # Map force types to names
            force_names = {
                "HarmonicBondForce": "bond_energy",
                "HarmonicAngleForce": "angle_energy",
                "PeriodicTorsionForce": "dihedral_energy",
                "NonbondedForce": "nonbonded_energy",
                "CustomGBForce": "solvation_energy",
                "CMMotionRemover": None,
            }

            for i in range(system.getNumForces()):
                force = system.getForce(i)
                force.setForceGroup(i)

            # Create simulation
            use_gpu = self._check_gpu_available()
            platform = openmm.Platform.getPlatformByName(
                "CUDA" if use_gpu else "CPU"
            )
            integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
            simulation = openmm_app.Simulation(
                modeller.topology, system, integrator, platform
            )
            simulation.context.setPositions(modeller.positions)

            # Get total energy
            state = simulation.context.getState(getEnergy=True)
            total_energy = state.getPotentialEnergy().value_in_unit(ENERGY)

            energy_breakdown = {"total_energy": total_energy}

            # Get energy for each force group
            for i in range(system.getNumForces()):
                force = system.getForce(i)
                force_type = force.__class__.__name__

                name = force_names.get(force_type, force_type.lower())
                if name is None:
                    continue

                state = simulation.context.getState(getEnergy=True, groups={i})
                energy = state.getPotentialEnergy().value_in_unit(ENERGY)
                energy_breakdown[name] = energy

            return energy_breakdown

        except Exception as e:
            logger.warning(f"Could not compute energy breakdown: {e}")
            return {"total_energy": None}
