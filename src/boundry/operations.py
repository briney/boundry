"""Core operations for Boundry - standalone functions for protein engineering.

This module provides the primary Python API for Boundry. Each function
wraps the underlying Designer, Relaxer, and analysis modules into
self-contained operations that accept flexible input types and return
Structure objects with metadata.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

if TYPE_CHECKING:
    from boundry.binding_energy import BindingEnergyResult
    from boundry.config import (
        DesignConfig,
        IdealizeConfig,
        InterfaceConfig,
        PipelineConfig,
        RelaxConfig,
    )
    from boundry.designer import Designer
    from boundry.interface import InterfaceInfo
    from boundry.interface_position_energetics import PerPositionResult
    from boundry.relaxer import Relaxer
    from boundry.renumber import RenumberMapping
    from boundry.surface_area import (
        ShapeComplementarityResult,
        SurfaceAreaResult,
    )

logger = logging.getLogger(__name__)

StructureInput = Union[str, Path, "Structure"]


# -------------------------------------------------------------------
# Data classes
# -------------------------------------------------------------------


@dataclass
class Structure:
    """Wrapper around a protein structure with metadata.

    Holds a PDB-format string along with optional metadata from
    operations (energy, sequence, scores, etc.) and the source
    file path.
    """

    pdb_string: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None

    def write(self, path: Union[str, Path]) -> None:
        """Write structure to file.

        Format is auto-detected from the file extension. PDB strings
        are converted to CIF when writing to ``.cif`` or ``.mmcif``
        files.
        """
        from boundry.structure_io import (
            StructureFormat,
            convert_to_format,
            detect_format,
            write_structure,
        )

        path = Path(path)
        try:
            target_format = detect_format(path)
        except ValueError:
            target_format = StructureFormat.PDB

        content = self.pdb_string
        if target_format != StructureFormat.PDB:
            content = convert_to_format(content, target_format)

        write_structure(content, path, target_format)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> Structure:
        """Load structure from a PDB or CIF file.

        CIF files are converted to PDB format internally.
        """
        from boundry.structure_io import (
            ensure_pdb_format,
            read_structure,
        )

        path = Path(path)
        content = read_structure(path)
        pdb_string = ensure_pdb_format(content, path)
        return cls(pdb_string=pdb_string, source_path=str(path))


@dataclass
class InterfaceAnalysisResult:
    """Results from comprehensive interface analysis."""

    interface_info: Optional[InterfaceInfo] = None
    binding_energy: Optional[BindingEnergyResult] = None
    sasa: Optional[SurfaceAreaResult] = None
    shape_complementarity: Optional[ShapeComplementarityResult] = None
    per_position: Optional[PerPositionResult] = None


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _resolve_input(
    structure: StructureInput,
) -> Tuple[str, Optional[str]]:
    """Resolve flexible structure input to *(pdb_string, source_path)*.

    Accepts:
    * :class:`Structure` -- uses its ``pdb_string`` directly.
    * :class:`~pathlib.Path` -- reads and converts to PDB.
    * ``str`` -- treated as a file path if the file exists, otherwise
      assumed to be a PDB-format string.
    """
    if isinstance(structure, Structure):
        return structure.pdb_string, structure.source_path
    if isinstance(structure, Path):
        s = Structure.from_file(structure)
        return s.pdb_string, str(structure)
    if isinstance(structure, str):
        # Strings containing newlines are PDB content, not file paths
        if "\n" not in structure:
            try:
                p = Path(structure)
                if p.exists() and p.is_file():
                    s = Structure.from_file(p)
                    return s.pdb_string, structure
            except OSError:
                pass
        return structure, None
    raise TypeError(f"Expected str, Path, or Structure, got {type(structure)}")


def _write_temp_pdb(pdb_string: str) -> Path:
    """Write *pdb_string* to a temporary file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
    tmp.write(pdb_string)
    tmp.close()
    return Path(tmp.name)


def _auto_renumber(
    pdb_string: str,
) -> Tuple[str, Optional["RenumberMapping"]]:
    """Renumber residues if insertion codes are present.

    Returns *(pdb_string, mapping)*.  If no insertion codes are found
    the input is returned unchanged with ``mapping=None``.
    """
    from boundry.renumber import has_insertion_codes, renumber_pdb

    if not has_insertion_codes(pdb_string):
        return pdb_string, None
    logger.info("Insertion codes detected; renumbering residues")
    return renumber_pdb(pdb_string)


def _maybe_restore(
    pdb_string: str,
    mapping: Optional["RenumberMapping"],
) -> str:
    """Restore original numbering if *mapping* is not ``None``."""
    if mapping is None:
        return pdb_string
    from boundry.renumber import restore_numbering

    logger.info("Restoring original residue numbering")
    return restore_numbering(pdb_string, mapping)


# -------------------------------------------------------------------
# Operations
# -------------------------------------------------------------------


def idealize(
    structure: StructureInput,
    config: Optional[IdealizeConfig] = None,
) -> Structure:
    """Fix backbone geometry while preserving dihedral angles.

    Runs constrained minimization to relieve local strain while
    keeping the overall structure close to the original.  Optionally
    corrects cis-omega angles and adds missing residues.

    Args:
        structure: Input structure (file path, PDB string, or
            Structure).
        config: Idealization configuration.  Uses defaults if not
            provided.

    Returns:
        Structure with idealized backbone geometry.
    """
    from boundry.config import IdealizeConfig
    from boundry.idealize import idealize_structure

    if config is None:
        config = IdealizeConfig(enabled=True)

    pdb_string, source_path = _resolve_input(structure)
    pdb_string, renumber_mapping = _auto_renumber(pdb_string)
    idealized_pdb, gaps = idealize_structure(pdb_string, config)
    idealized_pdb = _maybe_restore(idealized_pdb, renumber_mapping)

    metadata: Dict[str, Any] = {
        "chain_gaps": len(gaps),
        "operation": "idealize",
    }
    if renumber_mapping is not None:
        metadata["renumber_mapping"] = renumber_mapping

    return Structure(
        pdb_string=idealized_pdb,
        metadata=metadata,
        source_path=source_path,
    )


def minimize(
    structure: StructureInput,
    config: Optional[RelaxConfig] = None,
    pre_idealize: bool = False,
    idealize_config: Optional[IdealizeConfig] = None,
) -> Structure:
    """Energy minimization using OpenMM AMBER force field.

    Runs energy minimization without repacking side chains.  Supports
    both unconstrained L-BFGS and constrained AmberRelaxation modes.

    Args:
        structure: Input structure (file path, PDB string, or
            Structure).
        config: Relaxation configuration.  Uses defaults if not
            provided.
        pre_idealize: Run idealization before minimization.
        idealize_config: Configuration for pre-idealization step.

    Returns:
        Structure with minimized energy.
    """
    from boundry.config import IdealizeConfig, RelaxConfig
    from boundry.relaxer import Relaxer

    if config is None:
        config = RelaxConfig()

    pdb_string, source_path = _resolve_input(structure)
    pdb_string, renumber_mapping = _auto_renumber(pdb_string)

    if pre_idealize:
        pre = idealize(
            pdb_string,
            config=idealize_config or IdealizeConfig(enabled=True),
        )
        pdb_string = pre.pdb_string

    relaxer = Relaxer(config)
    relaxed_pdb, relax_info, _ = relaxer.relax(pdb_string)
    relaxed_pdb = _maybe_restore(relaxed_pdb, renumber_mapping)

    metadata: Dict[str, Any] = {
        "initial_energy": relax_info["initial_energy"],
        "final_energy": relax_info["final_energy"],
        "rmsd": relax_info["rmsd"],
        "operation": "minimize",
    }
    if renumber_mapping is not None:
        metadata["renumber_mapping"] = renumber_mapping

    return Structure(
        pdb_string=relaxed_pdb,
        metadata=metadata,
        source_path=source_path,
    )


def repack(
    structure: StructureInput,
    config: Optional[DesignConfig] = None,
    resfile: Optional[Union[str, Path]] = None,
    pre_idealize: bool = False,
    idealize_config: Optional[IdealizeConfig] = None,
) -> Structure:
    """Repack side chains without changing sequence.

    Uses LigandMPNN's side-chain packing to optimise rotamers while
    keeping the amino acid sequence fixed.

    Args:
        structure: Input structure (file path, PDB string, or
            Structure).
        config: Design/packing configuration.  Uses defaults if not
            provided.
        resfile: Optional Rosetta-style resfile for residue control.
        pre_idealize: Run idealization before repacking.
        idealize_config: Configuration for pre-idealization step.

    Returns:
        Structure with repacked side chains.
    """
    from boundry.config import DesignConfig, IdealizeConfig
    from boundry.designer import Designer
    from boundry.resfile import ResfileParser

    if config is None:
        config = DesignConfig()

    pdb_string, source_path = _resolve_input(structure)

    if pre_idealize:
        pre = idealize(
            pdb_string,
            config=idealize_config or IdealizeConfig(enabled=True),
        )
        pdb_string = pre.pdb_string

    design_spec = None
    if resfile is not None:
        design_spec = ResfileParser().parse(resfile)

    pdb_path = _write_temp_pdb(pdb_string)
    try:
        designer = Designer(config)
        repack_result = designer.repack(pdb_path, design_spec=design_spec)
        repacked_pdb = designer.result_to_pdb_string(repack_result)
    finally:
        pdb_path.unlink(missing_ok=True)

    return Structure(
        pdb_string=repacked_pdb,
        metadata={
            "sequence": repack_result["sequence"],
            "native_sequence": repack_result["native_sequence"],
            "ligandmpnn_loss": float(repack_result["loss"][0]),
            "operation": "repack",
        },
        source_path=source_path,
    )


def relax(
    structure: StructureInput,
    config: Optional[PipelineConfig] = None,
    resfile: Optional[Union[str, Path]] = None,
    pre_idealize: bool = False,
    n_iterations: int = 5,
) -> Structure:
    """Iterative side-chain repacking and energy minimization.

    Alternates between LigandMPNN side-chain repacking and OpenMM
    AMBER energy minimization for the specified number of iterations,
    similar to Rosetta FastRelax.

    Args:
        structure: Input structure (file path, PDB string, or
            Structure).
        config: Pipeline configuration (bundles design and relax
            configs).  Uses defaults if not provided.
        resfile: Optional Rosetta-style resfile for residue control.
        pre_idealize: Run idealization before the relax cycles.
        n_iterations: Number of repack + minimize cycles.

    Returns:
        Structure with relaxed energy and optimised side chains.
    """
    from boundry.config import IdealizeConfig, PipelineConfig
    from boundry.designer import Designer
    from boundry.relaxer import Relaxer
    from boundry.resfile import ResfileParser

    if config is None:
        config = PipelineConfig()

    pdb_string, source_path = _resolve_input(structure)
    pdb_string, renumber_mapping = _auto_renumber(pdb_string)

    if pre_idealize:
        ide_cfg = (
            config.idealize
            if config.idealize.enabled
            else IdealizeConfig(enabled=True)
        )
        pre = idealize(pdb_string, config=ide_cfg)
        pdb_string = pre.pdb_string

    design_spec = None
    if resfile is not None:
        design_spec = ResfileParser().parse(resfile)

    designer = Designer(config.design)
    relaxer = Relaxer(config.relax)

    iterations: list[dict[str, Any]] = []
    current_pdb = pdb_string

    # Set up optional progress bar
    iterable = range(1, n_iterations + 1)
    pbar = None
    if config.show_progress:
        from tqdm import tqdm

        pbar = tqdm(iterable, desc="Relaxing", unit="iter")
        iterable = pbar

    for i in iterable:
        logger.info(f"Iteration {i}/{n_iterations}")

        # Repack
        pdb_path = _write_temp_pdb(current_pdb)
        try:
            repack_result = designer.repack(pdb_path, design_spec=design_spec)
            current_pdb = designer.result_to_pdb_string(repack_result)
        finally:
            pdb_path.unlink(missing_ok=True)

        # Minimize
        relaxed_pdb, relax_info, _ = relaxer.relax(current_pdb)
        current_pdb = relaxed_pdb

        iterations.append(
            {
                "iteration": i,
                "initial_energy": relax_info["initial_energy"],
                "final_energy": relax_info["final_energy"],
                "rmsd": relax_info["rmsd"],
                "sequence": repack_result["sequence"],
            }
        )

        # Update progress bar with energy
        if pbar is not None:
            pbar.set_postfix(E=f"{relax_info['final_energy']:.1f}")

        logger.info(
            f"  E_init={relax_info['initial_energy']:.2f}, "
            f"E_final={relax_info['final_energy']:.2f}, "
            f"RMSD={relax_info['rmsd']:.3f}"
        )

    energy_breakdown = relaxer.get_energy_breakdown(current_pdb)
    current_pdb = _maybe_restore(current_pdb, renumber_mapping)

    metadata: Dict[str, Any] = {
        "iterations": iterations,
        "final_energy": (
            iterations[-1]["final_energy"] if iterations else None
        ),
        "energy_breakdown": energy_breakdown,
        "sequence": (iterations[-1]["sequence"] if iterations else None),
        "operation": "relax",
    }
    if renumber_mapping is not None:
        metadata["renumber_mapping"] = renumber_mapping

    return Structure(
        pdb_string=current_pdb,
        metadata=metadata,
        source_path=source_path,
    )


def mpnn(
    structure: StructureInput,
    config: Optional[DesignConfig] = None,
    resfile: Optional[Union[str, Path]] = None,
    pre_idealize: bool = False,
    idealize_config: Optional[IdealizeConfig] = None,
) -> Structure:
    """Sequence design using LigandMPNN.

    Runs neural network-based sequence design.  By default, designs
    all residues.  Use a resfile to control which residues are
    designed.

    Args:
        structure: Input structure (file path, PDB string, or
            Structure).
        config: Design configuration.  Uses defaults if not provided.
        resfile: Optional Rosetta-style resfile for residue control.
            When provided, only residues specified in the resfile are
            designed.
        pre_idealize: Run idealization before design.
        idealize_config: Configuration for pre-idealization step.

    Returns:
        Structure with designed sequence and packed side chains.
    """
    from boundry.config import DesignConfig, IdealizeConfig
    from boundry.designer import Designer
    from boundry.resfile import ResfileParser
    from boundry.utils import format_sequence_alignment

    if config is None:
        config = DesignConfig()

    pdb_string, source_path = _resolve_input(structure)

    if pre_idealize:
        pre = idealize(
            pdb_string,
            config=idealize_config or IdealizeConfig(enabled=True),
        )
        pdb_string = pre.pdb_string

    design_spec = None
    design_all = True
    if resfile is not None:
        design_spec = ResfileParser().parse(resfile)
        design_all = False

    pdb_path = _write_temp_pdb(pdb_string)
    try:
        designer = Designer(config)
        design_result = designer.design(
            pdb_path,
            design_spec=design_spec,
            design_all=design_all,
        )
        designed_pdb = designer.result_to_pdb_string(design_result)
    finally:
        pdb_path.unlink(missing_ok=True)

    alignment = format_sequence_alignment(
        design_result["native_sequence"],
        design_result["sequence"],
    )
    logger.info(f"Sequence design result:\n{alignment}")

    return Structure(
        pdb_string=designed_pdb,
        metadata={
            "sequence": design_result["sequence"],
            "native_sequence": design_result["native_sequence"],
            "ligandmpnn_loss": float(design_result["loss"][0]),
            "operation": "mpnn",
        },
        source_path=source_path,
    )


def design(
    structure: StructureInput,
    config: Optional[PipelineConfig] = None,
    resfile: Optional[Union[str, Path]] = None,
    pre_idealize: bool = False,
    n_iterations: int = 5,
) -> Structure:
    """Iterative sequence design and energy minimization.

    Alternates between LigandMPNN sequence design and OpenMM AMBER
    energy minimization for the specified number of iterations,
    similar to Rosetta FastDesign.

    Args:
        structure: Input structure (file path, PDB string, or
            Structure).
        config: Pipeline configuration (bundles design and relax
            configs).  Uses defaults if not provided.
        resfile: Optional Rosetta-style resfile for residue control.
        pre_idealize: Run idealization before the design cycles.
        n_iterations: Number of design + minimize cycles.

    Returns:
        Structure with designed sequence and minimized energy.
    """
    from boundry.config import IdealizeConfig, PipelineConfig
    from boundry.designer import Designer
    from boundry.relaxer import Relaxer
    from boundry.resfile import ResfileParser
    from boundry.utils import format_sequence_alignment

    if config is None:
        config = PipelineConfig()

    pdb_string, source_path = _resolve_input(structure)
    pdb_string, renumber_mapping = _auto_renumber(pdb_string)

    if pre_idealize:
        ide_cfg = (
            config.idealize
            if config.idealize.enabled
            else IdealizeConfig(enabled=True)
        )
        pre = idealize(pdb_string, config=ide_cfg)
        pdb_string = pre.pdb_string

    design_spec = None
    design_all = True
    if resfile is not None:
        design_spec = ResfileParser().parse(resfile)
        design_all = False

    designer = Designer(config.design)
    relaxer = Relaxer(config.relax)

    iterations: list[dict[str, Any]] = []
    current_pdb = pdb_string
    original_native_sequence: Optional[str] = None

    # Set up optional progress bar
    iterable = range(1, n_iterations + 1)
    pbar = None
    if config.show_progress:
        from tqdm import tqdm

        pbar = tqdm(iterable, desc="Designing", unit="iter")
        iterable = pbar

    for i in iterable:
        logger.info(f"Iteration {i}/{n_iterations}")

        # Design
        pdb_path = _write_temp_pdb(current_pdb)
        try:
            design_result = designer.design(
                pdb_path,
                design_spec=design_spec,
                design_all=design_all,
            )
            current_pdb = designer.result_to_pdb_string(design_result)
        finally:
            pdb_path.unlink(missing_ok=True)

        if original_native_sequence is None:
            original_native_sequence = design_result["native_sequence"]

        compare_to = (
            original_native_sequence or design_result["native_sequence"]
        )
        alignment = format_sequence_alignment(
            compare_to, design_result["sequence"]
        )
        logger.info(f"  Sequence design result:\n{alignment}")

        # Minimize
        relaxed_pdb, relax_info, _ = relaxer.relax(current_pdb)
        current_pdb = relaxed_pdb

        iterations.append(
            {
                "iteration": i,
                "initial_energy": relax_info["initial_energy"],
                "final_energy": relax_info["final_energy"],
                "rmsd": relax_info["rmsd"],
                "sequence": design_result["sequence"],
                "native_sequence": design_result["native_sequence"],
                "ligandmpnn_loss": float(design_result["loss"][0]),
            }
        )

        # Update progress bar with energy
        if pbar is not None:
            pbar.set_postfix(E=f"{relax_info['final_energy']:.1f}")

        logger.info(
            f"  E_init={relax_info['initial_energy']:.2f}, "
            f"E_final={relax_info['final_energy']:.2f}, "
            f"RMSD={relax_info['rmsd']:.3f}"
        )

    energy_breakdown = relaxer.get_energy_breakdown(current_pdb)
    current_pdb = _maybe_restore(current_pdb, renumber_mapping)

    metadata: Dict[str, Any] = {
        "iterations": iterations,
        "final_energy": (
            iterations[-1]["final_energy"] if iterations else None
        ),
        "energy_breakdown": energy_breakdown,
        "sequence": (iterations[-1]["sequence"] if iterations else None),
        "native_sequence": original_native_sequence,
        "ligandmpnn_loss": (
            iterations[-1]["ligandmpnn_loss"] if iterations else None
        ),
        "operation": "design",
    }
    if renumber_mapping is not None:
        metadata["renumber_mapping"] = renumber_mapping

    return Structure(
        pdb_string=current_pdb,
        metadata=metadata,
        source_path=source_path,
    )


def renumber(structure: StructureInput) -> Structure:
    """Renumber residues sequentially, removing insertion codes.

    Useful for preparing Kabat-numbered antibody structures for
    operations that cannot handle insertion codes.  The original
    numbering mapping is stored in ``metadata["renumber_mapping"]``
    so it can be restored later via
    :func:`boundry.renumber.restore_numbering`.

    Args:
        structure: Input structure (file path, PDB string, or
            Structure).

    Returns:
        Structure with sequentially numbered residues.
    """
    from boundry.renumber import renumber_pdb

    pdb_string, source_path = _resolve_input(structure)
    renumbered_pdb, mapping = renumber_pdb(pdb_string)

    return Structure(
        pdb_string=renumbered_pdb,
        metadata={
            "renumber_mapping": mapping,
            "operation": "renumber",
        },
        source_path=source_path,
    )


def analyze_interface(
    structure: StructureInput,
    config: Optional[InterfaceConfig] = None,
    relaxer: Optional[Relaxer] = None,
    designer: Optional[Designer] = None,
) -> InterfaceAnalysisResult:
    """Analyse protein-protein interface properties.

    Identifies interface residues and optionally computes binding
    energy, buried surface area, and shape complementarity.

    Args:
        structure: Input structure (file path, PDB string, or
            Structure).
        config: Interface analysis configuration.  Uses defaults if
            not provided.
        relaxer: Relaxer instance (required for binding energy
            calculation).
        designer: Designer instance (required if ``relax_separated``
            is enabled or per-position scans use relaxation).

    Returns:
        InterfaceAnalysisResult with interface properties.
    """
    from boundry.binding_energy import calculate_binding_energy
    from boundry.config import InterfaceConfig
    from boundry.interface import identify_interface_residues
    from boundry.surface_area import (
        calculate_shape_complementarity,
        calculate_surface_area,
    )

    if config is None:
        config = InterfaceConfig(enabled=True)

    pdb_string, _ = _resolve_input(structure)

    result = InterfaceAnalysisResult()

    # Identify interface residues
    interface_info = identify_interface_residues(
        pdb_string,
        distance_cutoff=config.distance_cutoff,
        chain_pairs=config.chain_pairs,
    )
    result.interface_info = interface_info
    logger.info(interface_info.summary)

    if not interface_info.interface_residues:
        logger.warning("No interface residues found")
        return result

    # Binding energy (requires relaxer)
    if config.calculate_binding_energy and relaxer is not None:
        logger.info("Calculating binding energy...")
        result.binding_energy = calculate_binding_energy(
            pdb_string,
            relaxer,
            chain_pairs=interface_info.chain_pairs,
            distance_cutoff=config.distance_cutoff,
            relax_separated=config.relax_separated,
            designer=designer,
            relax_separated_iterations=config.relax_separated_iterations,
            relax_separated_seed=config.relax_separated_seed,
        )
        if result.binding_energy.binding_energy is not None:
            logger.info(
                f"dG = "
                f"{result.binding_energy.binding_energy:.2f} "
                f"kcal/mol"
            )
        else:
            logger.warning("dG calculation failed (energy returned None)")

    # Surface area
    if config.calculate_sasa:
        logger.info("Calculating buried surface area...")
        result.sasa = calculate_surface_area(
            pdb_string,
            interface_info.interface_residues,
            probe_radius=config.sasa_probe_radius,
        )
        logger.info(
            f"Buried SASA: " f"{result.sasa.buried_sasa:.1f} sq. angstroms"
        )

    # Shape complementarity
    if config.calculate_shape_complementarity:
        logger.info("Calculating shape complementarity...")
        result.shape_complementarity = calculate_shape_complementarity(
            pdb_string,
            interface_info.chain_pairs,
            interface_info.interface_residues,
        )
        logger.info(
            f"Shape complementarity: "
            f"{result.shape_complementarity.sc_score:.3f}"
        )

    # Per-position energetics (alanine scan and/or dG_i)
    if (config.per_position or config.alanine_scan) and relaxer is not None:
        from boundry.interface_position_energetics import (
            compute_position_energetics,
            write_position_csv,
        )

        sasa_delta = None
        if result.sasa is not None:
            sasa_delta = result.sasa.interface_residue_delta_sasa

        logger.info("Computing per-position energetics...")
        result.per_position = compute_position_energetics(
            pdb_string,
            interface_info.interface_residues,
            interface_info.chain_pairs,
            relaxer,
            designer=designer,
            distance_cutoff=config.distance_cutoff,
            position_relax=config.position_relax,
            relax_separated=config.relax_separated,
            scan_chains=config.scan_chains,
            max_scan_sites=config.max_scan_sites,
            run_per_position=config.per_position,
            run_alanine_scan=config.alanine_scan,
            sasa_delta=sasa_delta,
            show_progress=config.show_progress,
            quiet=config.quiet,
        )

        if config.position_csv is not None:
            write_position_csv(result.per_position, config.position_csv)

    return result
