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
        "Rosetta FastRelax and FastDesign protocols."
    ),
    no_args_is_help=True,
    add_completion=False,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not verbose:
        # Suppress noisy dependency loggers
        import warnings

        for name in (
            "openmm",
            "pdbfixer",
            "Bio",
            "freesasa",
            "torch",
            "absl",
        ):
            logging.getLogger(name).setLevel(logging.ERROR)
        warnings.filterwarnings(
            "ignore",
            module=r"(openmm|pdbfixer|Bio|freesasa|torch|absl)",
        )


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
        False, "--verbose", "-v", help="Enable verbose output"
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
    result = _idealize(input_file, config=config)
    result.write(output_file)
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
        False, "--verbose", "-v", help="Enable verbose output"
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
    result = _minimize(
        input_file, config=config, pre_idealize=pre_idealize
    )
    result.write(output_file)

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
        False, "--verbose", "-v", help="Enable verbose output"
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
    result = _repack(
        input_file,
        config=config,
        resfile=resfile,
        pre_idealize=pre_idealize,
    )
    result.write(output_file)
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
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
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
    )

    logger.info(
        f"Relaxing {input_file} -> {output_file} ({n_iter} iterations)"
    )
    result = _relax(
        input_file,
        config=config,
        resfile=resfile,
        pre_idealize=pre_idealize,
        n_iterations=n_iter,
    )
    result.write(output_file)

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
        False, "--verbose", "-v", help="Enable verbose output"
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
    result = _mpnn(
        input_file,
        config=config,
        resfile=resfile,
        pre_idealize=pre_idealize,
    )
    result.write(output_file)
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
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
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
    )

    logger.info(
        f"Designing {input_file} -> {output_file} ({n_iter} iterations)"
    )
    result = _design(
        input_file,
        config=config,
        resfile=resfile,
        pre_idealize=pre_idealize,
        n_iterations=n_iter,
    )
    result.write(output_file)

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
        False, "--verbose", "-v", help="Enable verbose output"
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
    result = _renumber(input_file)
    result.write(output_file)
    logger.info(f"Wrote renumbered structure to {output_file}")


@app.command("analyze-interface")
def analyze_interface(
    input_file: Path = typer.Argument(
        ..., metavar="INPUT", help="Input structure file (PDB or CIF)"
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
        help="Compute IAM-like per-residue dG_i via residue removal",
    ),
    alanine_scan: bool = typer.Option(
        False,
        "--alanine-scan",
        help="Compute per-residue AlaScan ΔΔG (GLY/PRO/ALA skipped)",
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
    position_csv: Optional[Path] = typer.Option(
        None,
        "--position-csv",
        help="Write per-position results to CSV file",
    ),
    max_scan_sites: Optional[int] = typer.Option(
        None,
        "--max-scan-sites",
        help="Limit number of residues scanned "
        "(useful for large interfaces)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """Analyze protein-protein interface properties.

    Identifies interface residues by distance cutoff, computes binding
    energy (dG), and optionally buried SASA and shape complementarity.

    Per-position flags (--per-position, --alanine-scan) enable per-residue
    energetics. dG = E_bound - E_unbound (negative = favorable binding),
    and ΔΔG = dG_ala - dG_wt (positive = destabilising hotspot).
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

    from boundry.config import DesignConfig, InterfaceConfig, RelaxConfig
    from boundry.operations import analyze_interface as _analyze

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
        relax_separated_seed=relax_separated_seed,
        per_position=per_position,
        alanine_scan=alanine_scan,
        scan_chains=parsed_scan_chains,
        position_relax=position_relax,
        position_csv=position_csv,
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

    result = _analyze(
        input_file,
        config=interface_config,
        relaxer=relaxer,
        designer=designer,
    )

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
            format_hotspot_table,
        )

        if result.per_position.dG_wt is not None:
            typer.echo(
                f"dG_wt (per-position): "
                f"{result.per_position.dG_wt:.2f} kcal/mol"
            )
        hotspot_table = format_hotspot_table(result.per_position)
        if hotspot_table:
            typer.echo(hotspot_table)
        if position_csv:
            typer.echo(f"Per-position CSV: {position_csv}")


@app.command()
def run(
    workflow_file: Path = typer.Argument(
        ..., metavar="WORKFLOW", help="YAML workflow file"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """Execute a YAML workflow."""
    _setup_logging(verbose)

    if not workflow_file.exists():
        typer.echo(
            f"Error: Workflow file not found: {workflow_file}", err=True
        )
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

    workflow = Workflow.from_yaml(workflow_file)
    workflow.run()


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------


def main():
    """CLI entry point (called by ``boundry`` console script)."""
    app()


if __name__ == "__main__":
    main()
