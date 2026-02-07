#!/usr/bin/env python
"""Boundry CLI - Protein engineering with LigandMPNN and AMBER relaxation."""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="boundry",
    help=(
        "Protein engineering with LigandMPNN design and AMBER relaxation.\n\n"
        "Boundry combines neural network-based sequence design (LigandMPNN) "
        "with physics-based energy minimization (OpenMM AMBER), similar to "
        "Rosetta FastRelax and FastDesign protocols.\n\n"
        "By default, commands run quietly with minimal output. Use --verbose "
        "to enable detailed logging from OpenMM, ProDy, and other dependencies."
    ),
    no_args_is_help=True,
    add_completion=False,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    """Configure logging for the CLI.

    Default (non-verbose) runs are quiet - only warnings and errors from
    Boundry itself are shown. Third-party dependency logging, Python
    warnings from OpenMM/PDBFixer/FreeSASA/torch, and absl logging from
    OpenFold are all suppressed.

    With --verbose, detailed logs from all components are enabled.
    """
    import warnings

    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Override any existing handlers
    )

    if verbose:
        # In verbose mode, capture Python warnings into logging
        logging.captureWarnings(True)
    else:
        # Suppress noisy dependency loggers
        for name in (
            "openmm",
            "pdbfixer",
            "Bio",
            "freesasa",
            "torch",
            "absl",
            "prody",
            "py.warnings",
        ):
            logging.getLogger(name).setLevel(logging.ERROR)

        # Suppress Python warnings from dependencies
        warnings.filterwarnings(
            "ignore",
            module=r"(openmm|pdbfixer|Bio|freesasa|torch|absl|openfold|simtk)",
        )
        # Suppress simtk deprecation warnings specifically
        warnings.filterwarnings(
            "ignore",
            message=r".*simtk.*",
            category=DeprecationWarning,
        )

        # Suppress absl logging (used by OpenFold)
        try:
            from absl import logging as absl_logging

            absl_logging.set_verbosity(absl_logging.ERROR)
            absl_logging.set_stderrthreshold(absl_logging.ERROR)
        except ImportError:
            pass  # absl not installed, nothing to suppress

        # Suppress ProDy logger explicitly
        try:
            import prody

            prody.confProDy(verbosity="none")
        except ImportError:
            pass  # prody not installed


def _quiet_context(verbose: bool):
    """Return context manager to suppress C-level stderr when not verbose.

    C libraries (OpenMM, PDBFixer, FreeSASA) write directly to stderr,
    bypassing Python's logging system. This function returns:
    - nullcontext() if verbose=True (stderr passes through)
    - suppress_stderr() if verbose=False (stderr redirected to /dev/null)
    """
    import contextlib

    if verbose:
        return contextlib.nullcontext()
    else:
        from boundry.utils import suppress_stderr

        return suppress_stderr()


def _validate_input(path: Path) -> None:
    """Validate that *path* exists and has a supported extension."""
    if not path.exists():
        typer.echo(f"Error: Input file not found: {path}", err=True)
        raise typer.Exit(code=1)
    ext = path.suffix.lower()
    if ext not in (".pdb", ".cif", ".mmcif"):
        typer.echo(
            f"Error: Unsupported format '{ext}'. "
            "Supported: .pdb, .cif, .mmcif",
            err=True,
        )
        raise typer.Exit(code=1)


def _check_for_ligands(input_path: Path) -> bool:
    """Check if structure has non-water HETATM records."""
    from boundry.structure_io import StructureFormat, detect_format

    fmt = detect_format(input_path)
    water_residues = {"HOH", "WAT", "SOL", "TIP3", "TIP4", "SPC"}

    if fmt == StructureFormat.PDB:
        with open(input_path) as f:
            for line in f:
                if line.startswith("HETATM"):
                    resname = line[17:20].strip()
                    if resname not in water_residues:
                        return True
    else:
        from Bio.PDB import MMCIFParser

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("check", str(input_path))
        for model in structure:
            for chain in model:
                for residue in chain:
                    hetflag = residue.id[0]
                    if hetflag.startswith("H_"):
                        resname = residue.resname.strip()
                        if resname not in water_residues:
                            return True
    return False


def _validate_minimization_ligands(
    input_path: Path, constrained: bool
) -> None:
    """Error if structure has ligands but constrained mode is off."""
    if not constrained and _check_for_ligands(input_path):
        typer.echo(
            "Error: Input contains ligands (HETATM records). "
            "Unconstrained minimization cannot handle non-standard "
            "residues. Use --constrained flag.",
            err=True,
        )
        raise typer.Exit(code=1)


def _parse_chain_pairs(chain_string: str) -> list:
    """Parse ``'H:L,H:A'`` into ``[('H', 'L'), ('H', 'A')]``."""
    pairs = []
    for pair in chain_string.split(","):
        if ":" in pair:
            a, b = pair.strip().split(":")
            pairs.append((a.strip(), b.strip()))
    return pairs


def _run_structure_command(
    *,
    operation_name: str,
    operation,
    input_file: Path,
    output_file: Path,
    operation_kwargs: Optional[dict] = None,
):
    """Run a structure-producing operation with CLI output policy."""
    from boundry.invocation import (
        InvocationMode,
        OutputPolicy,
        OutputRequirement,
    )
    from boundry.runner import run_structure_operation

    return run_structure_operation(
        name=operation_name,
        operation=operation,
        structure=input_file,
        output=output_file,
        mode=InvocationMode.CLI,
        output_policy=OutputPolicy(OutputRequirement.REQUIRED),
        **(operation_kwargs or {}),
    )


# -------------------------------------------------------------------
# Commands
# -------------------------------------------------------------------


@app.command()
def idealize(
    input_file: Path = typer.Argument(
        ..., metavar="INPUT", help="Input structure file (PDB or CIF)"
    ),
    output_file: Path = typer.Argument(
        ..., metavar="OUTPUT", help="Output structure file"
    ),
    fix_cis_omega: bool = typer.Option(
        True, help="Correct non-trans peptide bonds (except Pro)"
    ),
    add_missing_residues: bool = typer.Option(
        True, help="Add missing residues from SEQRES"
    ),
    close_chainbreaks: bool = typer.Option(
        True, help="Close chain breaks during idealization"
    ),
    stiffness: float = typer.Option(
        10.0, help="Restraint stiffness in kcal/mol/A^2"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable detailed logging from all components"
    ),
):
    """Fix backbone geometry while preserving dihedral angles."""
    _setup_logging(verbose)
    _validate_input(input_file)

    from boundry.config import IdealizeConfig
    from boundry.operations import idealize as _idealize

    config = IdealizeConfig(
        enabled=True,
        fix_cis_omega=fix_cis_omega,
        add_missing_residues=add_missing_residues,
        close_chainbreaks=close_chainbreaks,
        post_idealize_stiffness=stiffness,
    )

    logger.info(f"Idealizing {input_file} -> {output_file}")
    with _quiet_context(verbose):
        result = _run_structure_command(
            operation_name="idealize",
            operation=_idealize,
            input_file=input_file,
            output_file=output_file,
            operation_kwargs={"config": config},
        )
    logger.info(f"Wrote idealized structure to {output_file}")


@app.command()
def minimize(
    input_file: Path = typer.Argument(
        ..., metavar="INPUT", help="Input structure file (PDB or CIF)"
    ),
    output_file: Path = typer.Argument(
        ..., metavar="OUTPUT", help="Output structure file"
    ),
    pre_idealize: bool = typer.Option(
        False, help="Idealize backbone geometry before minimization"
    ),
    constrained: bool = typer.Option(
        False,
        help="Use constrained minimization with position restraints "
        "(AlphaFold-style)",
    ),
    stiffness: float = typer.Option(
        10.0, help="Restraint stiffness in kcal/mol/A^2"
    ),
    max_iterations: int = typer.Option(
        0, help="Max L-BFGS iterations (0 = unlimited)"
    ),
    no_split_gaps: bool = typer.Option(
        False, help="Disable chain splitting at gaps"
    ),
    no_implicit_solvent: bool = typer.Option(
        False, help="Disable GBn2 implicit solvation"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable detailed logging from all components"
    ),
):
    """Energy minimization using OpenMM AMBER force field."""
    _setup_logging(verbose)
    _validate_input(input_file)
    _validate_minimization_ligands(input_file, constrained)

    from boundry.config import RelaxConfig
    from boundry.operations import minimize as _minimize

    config = RelaxConfig(
        constrained=constrained,
        stiffness=stiffness,
        max_iterations=max_iterations,
        split_chains_at_gaps=not no_split_gaps,
        implicit_solvent=not no_implicit_solvent,
    )

    logger.info(f"Minimizing {input_file} -> {output_file}")
    with _quiet_context(verbose):
        result = _run_structure_command(
            operation_name="minimize",
            operation=_minimize,
            input_file=input_file,
            output_file=output_file,
            operation_kwargs={
                "config": config,
                "pre_idealize": pre_idealize,
            },
        )

    energy = result.metadata.get("final_energy")
    if energy is not None:
        logger.info(f"Final energy: {energy:.2f} kcal/mol")
    logger.info(f"Wrote minimized structure to {output_file}")


@app.command()
def repack(
    input_file: Path = typer.Argument(
        ..., metavar="INPUT", help="Input structure file (PDB or CIF)"
    ),
    output_file: Path = typer.Argument(
        ..., metavar="OUTPUT", help="Output structure file"
    ),
    pre_idealize: bool = typer.Option(
        False, help="Idealize backbone geometry before repacking"
    ),
    resfile: Optional[Path] = typer.Option(
        None, help="Rosetta-style resfile for residue control"
    ),
    temperature: float = typer.Option(
        0.1, help="LigandMPNN sampling temperature"
    ),
    model_type: str = typer.Option(
        "ligand_mpnn",
        help="Model variant: protein_mpnn, ligand_mpnn, soluble_mpnn",
    ),
    seed: Optional[int] = typer.Option(
        None, help="Random seed for reproducibility"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable detailed logging from all components"
    ),
):
    """Repack side chains without changing sequence."""
    _setup_logging(verbose)
    _validate_input(input_file)

    from boundry.config import DesignConfig
    from boundry.operations import repack as _repack
    from boundry.weights import ensure_weights

    ensure_weights(verbose=verbose)

    config = DesignConfig(
        model_type=model_type,
        temperature=temperature,
        seed=seed,
    )

    logger.info(f"Repacking {input_file} -> {output_file}")
    with _quiet_context(verbose):
        result = _run_structure_command(
            operation_name="repack",
            operation=_repack,
            input_file=input_file,
            output_file=output_file,
            operation_kwargs={
                "config": config,
                "resfile": resfile,
                "pre_idealize": pre_idealize,
            },
        )
    logger.info(f"Wrote repacked structure to {output_file}")


@app.command()
def relax(
    input_file: Path = typer.Argument(
        ..., metavar="INPUT", help="Input structure file (PDB or CIF)"
    ),
    output_file: Path = typer.Argument(
        ..., metavar="OUTPUT", help="Output structure file"
    ),
    pre_idealize: bool = typer.Option(
        False, help="Idealize backbone geometry before relaxation"
    ),
    n_iter: int = typer.Option(
        5, help="Number of repack + minimize cycles"
    ),
    resfile: Optional[Path] = typer.Option(
        None, help="Rosetta-style resfile for residue control"
    ),
    temperature: float = typer.Option(
        0.1, help="LigandMPNN sampling temperature"
    ),
    model_type: str = typer.Option(
        "ligand_mpnn",
        help="Model variant: protein_mpnn, ligand_mpnn, soluble_mpnn",
    ),
    seed: Optional[int] = typer.Option(
        None, help="Random seed for reproducibility"
    ),
    constrained: bool = typer.Option(
        False,
        help="Use constrained minimization with position restraints "
        "(AlphaFold-style)",
    ),
    stiffness: float = typer.Option(
        10.0, help="Restraint stiffness in kcal/mol/A^2"
    ),
    max_iterations: int = typer.Option(
        0, help="Max L-BFGS iterations (0 = unlimited)"
    ),
    no_split_gaps: bool = typer.Option(
        False, help="Disable chain splitting at gaps"
    ),
    no_implicit_solvent: bool = typer.Option(
        False, help="Disable GBn2 implicit solvation"
    ),
    no_progress: bool = typer.Option(
        False, "--no-progress", help="Suppress progress bar"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable detailed logging from all components"
    ),
):
    """Iterative side-chain repacking and energy minimization.

    Alternates between LigandMPNN repacking and OpenMM AMBER minimization,
    similar to Rosetta FastRelax.
    """
    _setup_logging(verbose)
    _validate_input(input_file)
    _validate_minimization_ligands(input_file, constrained)

    from boundry.config import DesignConfig, PipelineConfig, RelaxConfig
    from boundry.operations import relax as _relax
    from boundry.weights import ensure_weights

    ensure_weights(verbose=verbose)

    config = PipelineConfig(
        design=DesignConfig(
            model_type=model_type,
            temperature=temperature,
            seed=seed,
        ),
        relax=RelaxConfig(
            constrained=constrained,
            stiffness=stiffness,
            max_iterations=max_iterations,
            split_chains_at_gaps=not no_split_gaps,
            implicit_solvent=not no_implicit_solvent,
        ),
        show_progress=not no_progress,
    )

    logger.info(
        f"Relaxing {input_file} -> {output_file} ({n_iter} iterations)"
    )
    with _quiet_context(verbose):
        result = _run_structure_command(
            operation_name="relax",
            operation=_relax,
            input_file=input_file,
            output_file=output_file,
            operation_kwargs={
                "config": config,
                "resfile": resfile,
                "pre_idealize": pre_idealize,
                "n_iterations": n_iter,
            },
        )

    energy = result.metadata.get("final_energy")
    if energy is not None:
        logger.info(f"Final energy: {energy:.2f} kcal/mol")
    logger.info(f"Wrote relaxed structure to {output_file}")


@app.command()
def mpnn(
    input_file: Path = typer.Argument(
        ..., metavar="INPUT", help="Input structure file (PDB or CIF)"
    ),
    output_file: Path = typer.Argument(
        ..., metavar="OUTPUT", help="Output structure file"
    ),
    pre_idealize: bool = typer.Option(
        False, help="Idealize backbone geometry before design"
    ),
    resfile: Optional[Path] = typer.Option(
        None, help="Rosetta-style resfile for residue control"
    ),
    temperature: float = typer.Option(
        0.1, help="LigandMPNN sampling temperature"
    ),
    model_type: str = typer.Option(
        "ligand_mpnn",
        help="Model variant: protein_mpnn, ligand_mpnn, soluble_mpnn",
    ),
    seed: Optional[int] = typer.Option(
        None, help="Random seed for reproducibility"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable detailed logging from all components"
    ),
):
    """Sequence design using LigandMPNN."""
    _setup_logging(verbose)
    _validate_input(input_file)

    from boundry.config import DesignConfig
    from boundry.operations import mpnn as _mpnn
    from boundry.weights import ensure_weights

    ensure_weights(verbose=verbose)

    config = DesignConfig(
        model_type=model_type,
        temperature=temperature,
        seed=seed,
    )

    logger.info(f"Designing {input_file} -> {output_file}")
    with _quiet_context(verbose):
        result = _run_structure_command(
            operation_name="mpnn",
            operation=_mpnn,
            input_file=input_file,
            output_file=output_file,
            operation_kwargs={
                "config": config,
                "resfile": resfile,
                "pre_idealize": pre_idealize,
            },
        )
    logger.info(f"Wrote designed structure to {output_file}")


@app.command()
def design(
    input_file: Path = typer.Argument(
        ..., metavar="INPUT", help="Input structure file (PDB or CIF)"
    ),
    output_file: Path = typer.Argument(
        ..., metavar="OUTPUT", help="Output structure file"
    ),
    pre_idealize: bool = typer.Option(
        False, help="Idealize backbone geometry before design"
    ),
    n_iter: int = typer.Option(
        5, help="Number of design + minimize cycles"
    ),
    resfile: Optional[Path] = typer.Option(
        None, help="Rosetta-style resfile for residue control"
    ),
    temperature: float = typer.Option(
        0.1, help="LigandMPNN sampling temperature"
    ),
    model_type: str = typer.Option(
        "ligand_mpnn",
        help="Model variant: protein_mpnn, ligand_mpnn, soluble_mpnn",
    ),
    seed: Optional[int] = typer.Option(
        None, help="Random seed for reproducibility"
    ),
    constrained: bool = typer.Option(
        False,
        help="Use constrained minimization with position restraints "
        "(AlphaFold-style)",
    ),
    stiffness: float = typer.Option(
        10.0, help="Restraint stiffness in kcal/mol/A^2"
    ),
    max_iterations: int = typer.Option(
        0, help="Max L-BFGS iterations (0 = unlimited)"
    ),
    no_split_gaps: bool = typer.Option(
        False, help="Disable chain splitting at gaps"
    ),
    no_implicit_solvent: bool = typer.Option(
        False, help="Disable GBn2 implicit solvation"
    ),
    no_progress: bool = typer.Option(
        False, "--no-progress", help="Suppress progress bar"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable detailed logging from all components"
    ),
):
    """Iterative sequence design and energy minimization.

    Alternates between LigandMPNN design and OpenMM AMBER minimization,
    similar to Rosetta FastDesign.
    """
    _setup_logging(verbose)
    _validate_input(input_file)
    _validate_minimization_ligands(input_file, constrained)

    from boundry.config import DesignConfig, PipelineConfig, RelaxConfig
    from boundry.operations import design as _design
    from boundry.weights import ensure_weights

    ensure_weights(verbose=verbose)

    config = PipelineConfig(
        design=DesignConfig(
            model_type=model_type,
            temperature=temperature,
            seed=seed,
        ),
        relax=RelaxConfig(
            constrained=constrained,
            stiffness=stiffness,
            max_iterations=max_iterations,
            split_chains_at_gaps=not no_split_gaps,
            implicit_solvent=not no_implicit_solvent,
        ),
        show_progress=not no_progress,
    )

    logger.info(
        f"Designing {input_file} -> {output_file} ({n_iter} iterations)"
    )
    with _quiet_context(verbose):
        result = _run_structure_command(
            operation_name="design",
            operation=_design,
            input_file=input_file,
            output_file=output_file,
            operation_kwargs={
                "config": config,
                "resfile": resfile,
                "pre_idealize": pre_idealize,
                "n_iterations": n_iter,
            },
        )

    energy = result.metadata.get("final_energy")
    if energy is not None:
        logger.info(f"Final energy: {energy:.2f} kcal/mol")
    logger.info(f"Wrote designed structure to {output_file}")


@app.command()
def renumber(
    input_file: Path = typer.Argument(
        ..., metavar="INPUT", help="Input structure file (PDB or CIF)"
    ),
    output_file: Path = typer.Argument(
        ..., metavar="OUTPUT", help="Output structure file"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable detailed logging from all components"
    ),
):
    """Renumber residues sequentially, removing insertion codes.

    Useful for preparing Kabat-numbered antibody structures for
    downstream operations that cannot handle insertion codes.
    """
    _setup_logging(verbose)
    _validate_input(input_file)

    from boundry.operations import renumber as _renumber

    logger.info(f"Renumbering {input_file} -> {output_file}")
    _run_structure_command(
        operation_name="renumber",
        operation=_renumber,
        input_file=input_file,
        output_file=output_file,
    )
    logger.info(f"Wrote renumbered structure to {output_file}")


@app.command("analyze-interface")
def analyze_interface(
    input_file: Path = typer.Argument(
        ..., metavar="INPUT", help="Input structure file (PDB or CIF)"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write interface summary JSON (.json) or to a directory",
    ),
    chains: Optional[str] = typer.Option(
        None,
        help="Chain pairs to analyze, e.g. 'H:L,H:A' "
        "(auto-detect if omitted)",
    ),
    distance_cutoff: float = typer.Option(
        8.0, help="Distance cutoff (angstroms) for interface residues"
    ),
    no_binding_energy: bool = typer.Option(
        False, help="Skip binding energy (dG) calculation"
    ),
    sasa: bool = typer.Option(
        False, "--sasa", help="Calculate buried surface area (SASA)"
    ),
    shape_complementarity: bool = typer.Option(
        False,
        "--shape-complementarity",
        help="Calculate shape complementarity (experimental)",
    ),
    relax_separated: bool = typer.Option(
        False, help="Repack and minimize separated partners (full relax)"
    ),
    relax_separated_iterations: int = typer.Option(
        1,
        "--relax-separated-iterations",
        help="Number of repack+minimize iterations "
        "(samples rotamer space, selects lowest energy)",
    ),
    relax_separated_seed: Optional[int] = typer.Option(
        None,
        "--relax-separated-seed",
        help="Base random seed for relax-separated iterations",
    ),
    constrained: bool = typer.Option(
        False,
        help="Use constrained minimization for binding energy calculation",
    ),
    per_position: bool = typer.Option(
        False,
        "--per-position",
        help="Compute per-residue dG via residue removal "
        "(ddG > 0 = hotspot)",
    ),
    alanine_scan: bool = typer.Option(
        False,
        "--alanine-scan",
        help="Compute per-residue AlaScan ddG "
        "(GLY/PRO/ALA skipped; ddG > 0 = hotspot)",
    ),
    scan_chains: Optional[str] = typer.Option(
        None,
        "--scan-chains",
        help="Restrict per-position / alanine scan to these chains "
        "(comma-separated, e.g. 'A,B')",
    ),
    position_relax: str = typer.Option(
        "none",
        "--position-relax",
        help="Relax policy for per-position scans: "
        "both (bound+unbound), unbound, none",
    ),
    per_position_csv: Optional[Path] = typer.Option(
        None,
        "--per-position-csv",
        help="Write per-position results to CSV file",
    ),
    alanine_scan_csv: Optional[Path] = typer.Option(
        None,
        "--alanine-scan-csv",
        help="Write alanine scan results to CSV file",
    ),
    max_scan_sites: Optional[int] = typer.Option(
        None,
        "--max-scan-sites",
        help="Limit number of residues scanned "
        "(useful for large interfaces)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable detailed logging from all components"
    ),
):
    """Analyze protein-protein interface properties.

    Identifies interface residues by distance cutoff, computes binding
    energy (dG), and optionally buried SASA and shape complementarity.

    Per-position flags (--per-position, --alanine-scan) enable per-residue
    energetics. Both produce dG (binding energy of modified system) and
    ddG = dG - dG_wt (positive = destabilising, i.e. a binding hotspot).
    dG = E_bound - E_unbound (negative = favorable binding).
    """
    _setup_logging(verbose)
    _validate_input(input_file)

    if position_relax not in ("both", "unbound", "none"):
        typer.echo(
            f"Error: --position-relax must be 'both', 'unbound', "
            f"or 'none' (got '{position_relax}')",
            err=True,
        )
        raise typer.Exit(code=1)
    if per_position_csv is not None and not per_position:
        typer.echo(
            "Error: --per-position-csv requires --per-position",
            err=True,
        )
        raise typer.Exit(code=1)
    if alanine_scan_csv is not None and not alanine_scan:
        typer.echo(
            "Error: --alanine-scan-csv requires --alanine-scan",
            err=True,
        )
        raise typer.Exit(code=1)

    from boundry.config import DesignConfig, InterfaceConfig, RelaxConfig
    from boundry.operations import analyze_interface as _analyze
    from boundry.runner import run_interface_operation

    parsed_scan_chains = None
    if scan_chains:
        parsed_scan_chains = [
            c.strip() for c in scan_chains.split(",") if c.strip()
        ]

    interface_config = InterfaceConfig(
        enabled=True,
        distance_cutoff=distance_cutoff,
        chain_pairs=(
            _parse_chain_pairs(chains) if chains else None
        ),
        calculate_binding_energy=not no_binding_energy,
        calculate_sasa=sasa,
        calculate_shape_complementarity=shape_complementarity,
        relax_separated=relax_separated,
        relax_separated_iterations=relax_separated_iterations,
        seed=relax_separated_seed,
        per_position=per_position,
        alanine_scan=alanine_scan,
        scan_chains=parsed_scan_chains,
        position_relax=position_relax,
        max_scan_sites=max_scan_sites,
        show_progress=per_position or alanine_scan,
        quiet=not verbose,
    )

    relaxer = None
    designer = None

    needs_relaxer = not no_binding_energy or per_position or alanine_scan
    needs_designer = (
        relax_separated
        or (
            (per_position or alanine_scan)
            and position_relax != "none"
        )
    )

    if needs_relaxer:
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(RelaxConfig(constrained=constrained))

    if needs_designer:
        from boundry.designer import Designer
        from boundry.weights import ensure_weights

        ensure_weights(verbose=verbose)
        designer = Designer(DesignConfig())

    result, outputs = run_interface_operation(
        operation=_analyze,
        structure=input_file,
        output=output,
        per_position_csv=per_position_csv,
        alanine_scan_csv=alanine_scan_csv,
        include_per_position_csv=per_position,
        include_alanine_scan_csv=alanine_scan,
        config=interface_config,
        relaxer=relaxer,
        designer=designer,
    )

    if output is None:
        # Print results to stdout
        if result.interface_info:
            typer.echo(result.interface_info.summary)
        if result.binding_energy:
            if result.binding_energy.binding_energy is not None:
                dG = result.binding_energy.binding_energy
                typer.echo(f"dG: {dG:.2f} kcal/mol")
            else:
                typer.echo("dG: could not be computed")
        if result.sasa:
            typer.echo(
                f"Buried SASA: {result.sasa.buried_sasa:.1f} sq. angstroms"
            )
        if result.shape_complementarity:
            typer.echo(
                f"Shape complementarity: "
                f"{result.shape_complementarity.sc_score:.3f}"
            )
        if result.per_position:
            from boundry.interface_position_energetics import (
                format_position_table,
            )

            if result.per_position.dG_wt is not None:
                typer.echo(
                    f"dG_wt: "
                    f"{result.per_position.dG_wt:.2f} kcal/mol"
                )
            table = format_position_table(
                result.per_position,
                label="per-position hotspots",
            )
            if table:
                typer.echo(table)
            if outputs.per_position_csv is not None:
                typer.echo(
                    f"Per-position CSV: {outputs.per_position_csv}"
                )
        if result.alanine_scan:
            from boundry.interface_position_energetics import (
                format_position_table as _fmt_table,
            )

            table = _fmt_table(
                result.alanine_scan,
                label="AlaScan hotspots",
            )
            if table:
                typer.echo(table)
            if outputs.alanine_scan_csv is not None:
                typer.echo(
                    f"Alanine scan CSV: {outputs.alanine_scan_csv}"
                )
    else:
        if outputs.summary_json is not None:
            typer.echo(f"Summary JSON: {outputs.summary_json}")
        if outputs.per_position_csv is not None:
            typer.echo(f"Per-position CSV: {outputs.per_position_csv}")
        if outputs.alanine_scan_csv is not None:
            typer.echo(f"Alanine scan CSV: {outputs.alanine_scan_csv}")


def _resolve_workflow(name_or_path: str) -> Path:
    """Resolve a workflow file path or built-in name."""
    path = Path(name_or_path)
    if path.exists():
        return path

    # Search user workflows directory
    user_dir = Path.home() / ".boundry" / "workflows"
    for ext in (".yaml", ".yml"):
        candidate = user_dir / f"{name_or_path}{ext}"
        if candidate.exists():
            return candidate

    # Search package built-in workflows
    builtin_dir = Path(__file__).parent / "workflows"
    for ext in (".yaml", ".yml"):
        candidate = builtin_dir / f"{name_or_path}{ext}"
        if candidate.exists():
            return candidate

    raise typer.BadParameter(
        f"Workflow '{name_or_path}' not found. Checked:\n"
        f"  - {path}\n"
        f"  - {user_dir}/\n"
        f"  - {builtin_dir}/"
    )


@app.command(
    context_settings={
        "allow_extra_args": True,
        "allow_interspersed_args": False,
    }
)
def run(
    ctx: typer.Context,
    workflow_file: str = typer.Argument(
        ...,
        metavar="WORKFLOW",
        help="YAML workflow file or built-in name",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed for reproducibility (overrides YAML seed)",
    ),
    workers: Optional[int] = typer.Option(
        None,
        "--workers",
        "-j",
        help="Number of parallel worker processes "
        "(overrides YAML workers; default 1 = sequential)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable detailed logging from all components",
    ),
):
    """Execute a YAML workflow.

    Extra arguments are applied as config overrides (key=value syntax).
    Example: boundry run workflow.yaml seed=42 project_path=results/
    """
    _setup_logging(verbose)

    try:
        resolved = _resolve_workflow(workflow_file)
    except typer.BadParameter as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    try:
        from boundry.workflow import Workflow
    except ImportError:
        typer.echo(
            "Error: Workflow system not available. "
            "Install pyyaml: pip install pyyaml",
            err=True,
        )
        raise typer.Exit(code=1)

    overrides = ctx.args or None
    workflow = Workflow.from_yaml(
        resolved,
        seed=seed,
        workers=workers,
        overrides=overrides,
    )
    workflow.run()


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------


def main():
    """CLI entry point (called by ``boundry`` console script)."""
    app()


if __name__ == "__main__":
    main()
