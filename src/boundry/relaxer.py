"""OpenMM AMBER relaxation wrapper."""

import io
import logging
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

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
        # Clear missing residues — we only want to add missing atoms
        # to existing residues, not rebuild entire loop segments.
        # Rebuilding missing residues into chain gaps causes OpenMM
        # template errors ("missing 1 C atom") because the inserted
        # residues create malformed bond topologies at gap boundaries.
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

        # Create force field and system (uses implicit solvation config)
        force_field, create_kwargs = self._make_force_field()
        modeller = openmm_app.Modeller(fixer.topology, fixer.positions)
        modeller.addHydrogens(force_field)
        system = force_field.createSystem(
            modeller.topology, **create_kwargs
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

        # Create force field and system (uses implicit solvation config)
        force_field, create_kwargs = self._make_force_field()
        modeller = openmm_app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(force_field)
        system = force_field.createSystem(
            modeller.topology, **create_kwargs
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

    def _make_force_field(
        self, *, implicit_solvent: bool | None = None
    ) -> tuple:
        """Select force field and createSystem kwargs for solvation model.

        Centralises force field + solvation model selection so that
        relaxation and scoring use the same energy surface.

        Args:
            implicit_solvent: Override for ``self.config.implicit_solvent``.
                When *None*, reads from config.

        Returns:
            ``(force_field, create_kwargs)`` — *create_kwargs* includes
            ``constraints=HBonds`` and dielectric parameters when implicit
            solvation is active.
        """
        use_implicit = (
            implicit_solvent
            if implicit_solvent is not None
            else self.config.implicit_solvent
        )
        if use_implicit:
            force_field = openmm_app.ForceField(
                "amber14-all.xml", "implicit/gbn2.xml"
            )
        else:
            force_field = openmm_app.ForceField(
                "amber14-all.xml", "amber14/tip3pfb.xml"
            )
        create_kwargs: dict = {"constraints": openmm_app.HBonds}
        if use_implicit:
            create_kwargs["soluteDielectric"] = 1.0
            create_kwargs["solventDielectric"] = 78.5
        return force_field, create_kwargs

    def _build_system_for_ddg(
        self,
        topology,
        positions,
        *,
        implicit_solvent: bool = True,
        constraints=openmm_app.HBonds,
    ):
        """Build an OpenMM System for ddG calculations.

        Shared system construction used by ``get_energy_breakdown()``,
        ``minimize_with_pair_restraints()``, and
        ``generate_local_md_ensemble()``.

        Args:
            topology: OpenMM Topology (from PDBFixer or Modeller).
            positions: Matching positions.
            implicit_solvent: When *True* use ``implicit/gbn2.xml``
                solvation model; otherwise ``amber14/tip3pfb.xml``.
            constraints: Bond constraint scheme (default HBonds).

        Returns:
            ``(system, modeller)`` tuple.  The caller is responsible for
            creating an integrator and simulation.
        """
        force_field, create_kwargs = self._make_force_field(
            implicit_solvent=implicit_solvent
        )
        if constraints is not openmm_app.HBonds:
            create_kwargs["constraints"] = constraints

        modeller = openmm_app.Modeller(topology, positions)
        modeller.addHydrogens(force_field)

        system = force_field.createSystem(
            modeller.topology, **create_kwargs
        )
        return system, modeller

    def _prepare_structure(self, pdb_string: str):
        """Prepare a PDB string with PDBFixer for OpenMM.

        Applies gap splitting, finds missing atoms, and adds them.
        Returns ``(fixer, pdb_string_after_split)`` — the *fixer*
        has its topology/positions ready for downstream use.
        """
        if self.config.split_chains_at_gaps:
            gaps = detect_chain_gaps(pdb_string)
            if gaps:
                logger.info(get_gap_summary(gaps))
                pdb_string, _ = split_chains_at_gaps(pdb_string, gaps)

        fixer = PDBFixer(pdbfile=io.StringIO(pdb_string))
        fixer.findMissingResidues()
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        return fixer

    def minimize_with_pair_restraints(
        self,
        pdb_string: str,
        *,
        ca_cutoff: float = 9.0,
        restraint_sd: float = 0.5,
        implicit_solvent: bool = True,
        max_iterations: int = 0,
    ) -> str:
        """Minimise a structure with CA–CA pair distance restraints.

        All CA–CA pairs within *ca_cutoff* angstroms are restrained with
        a harmonic potential ``0.5 * k * (r - r0)^2`` where
        ``k = 1 / sd^2`` (in kJ/mol/nm^2) and *r0* is the initial
        distance.  This keeps global backbone geometry near the input
        while allowing local relaxation.

        Args:
            pdb_string: PDB file contents.
            ca_cutoff: Distance cutoff (angstroms) for CA pair selection.
            restraint_sd: Standard deviation of the harmonic restraint
                (angstroms).
            implicit_solvent: Use GBn2 implicit solvation.
            max_iterations: L-BFGS iterations (0 = until convergence).

        Returns:
            Minimised PDB string.
        """
        ENERGY = unit.kilocalories_per_mole
        LENGTH = unit.angstroms

        pdb_string = filter_protein_only(pdb_string)
        fixer = self._prepare_structure(pdb_string)
        system, modeller = self._build_system_for_ddg(
            fixer.topology,
            fixer.positions,
            implicit_solvent=implicit_solvent,
        )

        # --- find CA–CA pairs within cutoff -------------------------
        positions_nm = np.array(
            [
                v.value_in_unit(unit.nanometers)
                for v in modeller.positions
            ]
        )
        ca_indices = [
            i
            for i, atom in enumerate(modeller.topology.atoms())
            if atom.name == "CA"
        ]
        cutoff_nm = ca_cutoff / 10.0  # angstroms → nanometers

        # Add CustomBondForce with per-bond parameters
        # k in kJ/mol/nm^2 so restraint_sd (angstroms) must convert
        k_value = (
            1.0
            / (restraint_sd * 0.1) ** 2  # sd Å → nm, then 1/sd^2
        )
        restraint_force = openmm.CustomBondForce(
            "0.5 * k * (r - r0)^2"
        )
        restraint_force.addGlobalParameter("k", k_value)
        restraint_force.addPerBondParameter("r0")

        n_restraints = 0
        for idx_a in range(len(ca_indices)):
            for idx_b in range(idx_a + 1, len(ca_indices)):
                i, j = ca_indices[idx_a], ca_indices[idx_b]
                diff = positions_nm[i] - positions_nm[j]
                dist = float(np.sqrt(np.dot(diff, diff)))
                if dist <= cutoff_nm:
                    restraint_force.addBond(i, j, [dist])
                    n_restraints += 1

        system.addForce(restraint_force)
        logger.debug(
            f"Added {n_restraints} CA–CA pair restraints "
            f"(cutoff={ca_cutoff} Å, sd={restraint_sd} Å)"
        )

        # --- minimise ------------------------------------------------
        use_gpu = self._check_gpu_available()
        integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
        platform = openmm.Platform.getPlatformByName(
            "CUDA" if use_gpu else "CPU"
        )
        simulation = openmm_app.Simulation(
            modeller.topology, system, integrator, platform
        )
        simulation.context.setPositions(modeller.positions)

        state = simulation.context.getState(getEnergy=True)
        einit = state.getPotentialEnergy().value_in_unit(ENERGY)

        if max_iterations > 0:
            simulation.minimizeEnergy(maxIterations=max_iterations)
        else:
            simulation.minimizeEnergy()

        state = simulation.context.getState(
            getEnergy=True, getPositions=True
        )
        efinal = state.getPotentialEnergy().value_in_unit(ENERGY)

        logger.info(
            f"Pair-restraint minimisation: "
            f"E_init={einit:.2f}, E_final={efinal:.2f} kcal/mol"
        )

        output = io.StringIO()
        openmm_app.PDBFile.writeFile(
            simulation.topology, state.getPositions(), output
        )
        return output.getvalue()

    # ----------------------------------------------------------------
    # Mutation neighbourhood
    # ----------------------------------------------------------------

    @staticmethod
    def _build_mutation_neighborhood(
        topology,
        positions,
        mutation_sites: List[Tuple[str, int]],
        neighborhood_radius: float = 8.0,
        sequence_window: int = 1,
    ) -> Set[Tuple[str, int]]:
        """Identify residues near mutation sites.

        Uses CB distance (CA for GLY) from each mutation site to find
        nearby residues, then expands by *sequence_window* positions
        in each direction along each chain.

        Args:
            topology: OpenMM Topology.
            positions: Matching positions (OpenMM Quantity list).
            mutation_sites: ``(chain_id, resnum)`` pairs.
            neighborhood_radius: Distance cutoff in angstroms.
            sequence_window: Sequence positions to expand in each
                direction.

        Returns:
            Set of ``(chain_id, resnum)`` tuples.
        """
        positions_nm = np.array(
            [v.value_in_unit(unit.nanometers) for v in positions]
        )
        cutoff_nm = neighborhood_radius / 10.0

        # Map (chain_id, resnum) → representative atom index
        # Prefer CB; fall back to CA for GLY.
        residue_atoms: dict = {}  # (chain, resnum) → atom_idx
        chain_residues: dict = {}  # chain → sorted list of resnums

        for atom in topology.atoms():
            chain_id = atom.residue.chain.id
            resnum = atom.residue.index + 1  # 0-indexed → 1-indexed
            key = (chain_id, resnum)

            if chain_id not in chain_residues:
                chain_residues[chain_id] = set()
            chain_residues[chain_id].add(resnum)

            if atom.name == "CB":
                residue_atoms[key] = atom.index
            elif atom.name == "CA" and key not in residue_atoms:
                residue_atoms[key] = atom.index

        # Sort chain residue lists for window expansion
        chain_resnums_sorted = {
            c: sorted(rn) for c, rn in chain_residues.items()
        }

        # Collect mutation-site atom positions
        site_positions = []
        for chain_id, resnum in mutation_sites:
            key = (chain_id, resnum)
            if key in residue_atoms:
                site_positions.append(positions_nm[residue_atoms[key]])

        if not site_positions:
            return set()
        site_positions = np.array(site_positions)

        # Find all residues within cutoff of any mutation site
        neighborhood: Set[Tuple[str, int]] = set()
        for key, atom_idx in residue_atoms.items():
            pos = positions_nm[atom_idx]
            diffs = site_positions - pos
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            if np.min(dists) <= cutoff_nm:
                neighborhood.add(key)

        # Expand by sequence window
        expanded = set(neighborhood)
        for chain_id, resnum in neighborhood:
            resnums = chain_resnums_sorted.get(chain_id, [])
            if not resnums:
                continue
            try:
                idx = resnums.index(resnum)
            except ValueError:
                continue
            for offset in range(-sequence_window, sequence_window + 1):
                new_idx = idx + offset
                if 0 <= new_idx < len(resnums):
                    expanded.add((chain_id, resnums[new_idx]))

        return expanded

    # ----------------------------------------------------------------
    # Local MD ensemble
    # ----------------------------------------------------------------

    def generate_local_md_ensemble(
        self,
        pdb_string: str,
        mutation_sites: List[Tuple[str, int]],
        *,
        n_members: int = 35,
        md_total_steps: int = 50000,
        md_equilibration_steps: int = 5000,
        md_temperature: float = 300.0,
        md_friction: float = 1.0,
        neighborhood_radius: float = 8.0,
        sequence_window: int = 1,
        ca_cutoff: float = 9.0,
        restraint_sd: float = 0.5,
        implicit_solvent: bool = True,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Generate an ensemble of structures via independent MD runs.

        Each member runs short Langevin dynamics from the input
        structure with CA–CA pair restraints to maintain the global
        fold, followed by a brief minimisation.

        Args:
            pdb_string: PDB file contents.
            mutation_sites: ``(chain_id, resnum)`` for each mutated
                position.
            n_members: Number of ensemble members.
            md_total_steps: Production MD steps per member (2 fs
                timestep).
            md_equilibration_steps: Equilibration steps (discarded).
            md_temperature: Langevin temperature in Kelvin.
            md_friction: Langevin friction coefficient (1/ps).
            neighborhood_radius: Å radius for mutation-site
                neighbourhood.
            sequence_window: Sequence expansion window around
                neighbourhood residues.
            ca_cutoff: CA–CA pair restraint cutoff (Å).
            restraint_sd: Restraint standard deviation (Å).
            implicit_solvent: Use GBn2 implicit solvation.
            seed: Base random seed; per-member seed is derived as
                ``seed * 100_000 + i``.

        Returns:
            List of PDB strings, one per ensemble member.
        """
        pdb_string = filter_protein_only(pdb_string)
        fixer = self._prepare_structure(pdb_string)
        system, modeller = self._build_system_for_ddg(
            fixer.topology,
            fixer.positions,
            implicit_solvent=implicit_solvent,
        )

        # Build mutation neighbourhood (informational / for Phase 3)
        _neighborhood = self._build_mutation_neighborhood(
            modeller.topology,
            modeller.positions,
            mutation_sites,
            neighborhood_radius=neighborhood_radius,
            sequence_window=sequence_window,
        )
        logger.debug(
            f"Mutation neighbourhood: {len(_neighborhood)} residues"
        )

        # --- CA–CA pair restraints (same logic as minimize_with_pair_restraints)
        positions_nm = np.array(
            [
                v.value_in_unit(unit.nanometers)
                for v in modeller.positions
            ]
        )
        ca_indices = [
            i
            for i, atom in enumerate(modeller.topology.atoms())
            if atom.name == "CA"
        ]
        cutoff_nm = ca_cutoff / 10.0
        k_value = 1.0 / (restraint_sd * 0.1) ** 2

        restraint_force = openmm.CustomBondForce(
            "0.5 * k * (r - r0)^2"
        )
        restraint_force.addGlobalParameter("k", k_value)
        restraint_force.addPerBondParameter("r0")

        n_restraints = 0
        for idx_a in range(len(ca_indices)):
            for idx_b in range(idx_a + 1, len(ca_indices)):
                i, j = ca_indices[idx_a], ca_indices[idx_b]
                diff = positions_nm[i] - positions_nm[j]
                dist = float(np.sqrt(np.dot(diff, diff)))
                if dist <= cutoff_nm:
                    restraint_force.addBond(i, j, [dist])
                    n_restraints += 1

        system.addForce(restraint_force)
        logger.debug(
            f"Added {n_restraints} CA–CA pair restraints for MD "
            f"ensemble (cutoff={ca_cutoff} Å)"
        )

        # --- initial restrained minimisation -------------------------
        use_gpu = self._check_gpu_available()
        platform = openmm.Platform.getPlatformByName(
            "CUDA" if use_gpu else "CPU"
        )
        init_integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
        init_sim = openmm_app.Simulation(
            modeller.topology, system, init_integrator, platform
        )
        init_sim.context.setPositions(modeller.positions)
        init_sim.minimizeEnergy()
        minimised_positions = (
            init_sim.context.getState(getPositions=True).getPositions()
        )
        del init_sim

        # --- per-member MD -------------------------------------------
        TIMESTEP_PS = 0.002  # 2 fs
        base_seed = seed or 0
        members: List[str] = []

        for i in range(n_members):
            member_seed = base_seed * 100_000 + i
            integrator = openmm.LangevinMiddleIntegrator(
                md_temperature * unit.kelvin,
                md_friction / unit.picosecond,
                TIMESTEP_PS * unit.picoseconds,
            )
            integrator.setRandomNumberSeed(member_seed)

            sim = openmm_app.Simulation(
                modeller.topology, system, integrator, platform
            )
            sim.context.setPositions(minimised_positions)
            sim.context.setVelocitiesToTemperature(
                md_temperature * unit.kelvin, member_seed
            )

            # Equilibration (discarded)
            sim.step(md_equilibration_steps)

            # Production
            sim.step(md_total_steps)

            # Brief post-MD minimisation
            sim.minimizeEnergy(maxIterations=100)

            state = sim.context.getState(getPositions=True)
            output = io.StringIO()
            openmm_app.PDBFile.writeFile(
                sim.topology, state.getPositions(), output
            )
            members.append(output.getvalue())
            del sim

            logger.debug(
                f"Ensemble member {i + 1}/{n_members} complete "
                f"(seed={member_seed})"
            )

        logger.info(
            f"Generated {len(members)}-member MD ensemble"
        )
        return members

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

            fixer = self._prepare_structure(pdb_string)
            system, modeller = self._build_system_for_ddg(
                fixer.topology,
                fixer.positions,
                implicit_solvent=self.config.implicit_solvent,
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


# --------------------------------------------------------------------
# Module-level helpers
# --------------------------------------------------------------------


def separate_interface_rigid_body(
    pdb_string: str,
    chain_groups: List[List[str]],
    separation_distance: float = 100.0,
) -> str:
    """Translate chain groups apart by rigid-body separation.

    The first group is left in place.  The second group is translated
    along the axis connecting the two group centres of mass by
    *separation_distance* angstroms.

    This is a pure coordinate transformation — no repacking and no
    minimisation.  Internal coordinates within each group are preserved
    exactly.

    Args:
        pdb_string: PDB file contents.
        chain_groups: Exactly two groups of chain IDs, e.g.
            ``[["H", "L"], ["A"]]``.
        separation_distance: How far (angstroms) to translate the
            second group.

    Returns:
        Modified PDB string.

    Raises:
        ValueError: If ``chain_groups`` does not contain exactly two
            groups.
    """
    if len(chain_groups) != 2:
        raise ValueError(
            f"Expected exactly 2 chain groups, got {len(chain_groups)}"
        )

    group1_chains = set(chain_groups[0])
    group2_chains = set(chain_groups[1])

    # --- parse CA positions per group to compute centres of mass ------
    group1_ca: list = []
    group2_ca: list = []

    for line in pdb_string.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if len(line) < 54:
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        chain_id = line[21]
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except (ValueError, IndexError):
            continue

        if chain_id in group1_chains:
            group1_ca.append(np.array([x, y, z]))
        elif chain_id in group2_chains:
            group2_ca.append(np.array([x, y, z]))

    if not group1_ca or not group2_ca:
        return pdb_string  # nothing to separate

    center1 = np.mean(group1_ca, axis=0)
    center2 = np.mean(group2_ca, axis=0)

    direction = center2 - center1
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        # Degenerate — pick an arbitrary axis
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = direction / norm

    translation = direction * separation_distance

    # --- rewrite PDB with shifted coordinates for group2 --------------
    out_lines: list = []
    for line in pdb_string.splitlines(keepends=True):
        if line.rstrip().startswith(("ATOM", "HETATM")) and len(line) >= 54:
            chain_id = line[21]
            if chain_id in group2_chains:
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    out_lines.append(line)
                    continue
                x += translation[0]
                y += translation[1]
                z += translation[2]
                line = (
                    line[:30]
                    + f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    + line[54:]
                )
        out_lines.append(line)

    return "".join(out_lines)
