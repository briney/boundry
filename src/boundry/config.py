"""Configuration dataclasses for Boundry pipeline and workflows."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union


ModelType = Literal["protein_mpnn", "ligand_mpnn", "soluble_mpnn"]
PositionRelaxMode = Literal["both", "unbound", "none"]


@dataclass
class DesignConfig:
    """Configuration for LigandMPNN design/repacking."""

    model_type: ModelType = "ligand_mpnn"
    temperature: float = 0.1
    pack_side_chains: bool = True
    seed: Optional[int] = None
    use_ligand_context: bool = True
    sc_num_denoising_steps: int = 3
    sc_num_samples: int = 16


@dataclass
class RelaxConfig:
    """Configuration for AMBER relaxation."""

    max_iterations: int = 0  # 0 = no limit
    tolerance: float = 2.39  # kcal/mol (OpenMM default)
    stiffness: float = 10.0  # kcal/mol/A^2
    max_outer_iterations: int = 3  # Violation-fixing iterations
    constrained: bool = False  # Use constrained (AmberRelaxation) minimization
    split_chains_at_gaps: bool = True  # Split chains at gaps to prevent closure
    implicit_solvent: bool = (
        True  # Use GBn2 implicit solvation for relaxation and energy evaluation
    )
    # GPU is auto-detected and used when available


@dataclass
class IdealizeConfig:
    """Configuration for structure idealization preprocessing."""

    enabled: bool = False  # Idealization disabled by default
    fix_cis_omega: bool = True  # Correct non-trans peptide bonds (except Pro)
    post_idealize_stiffness: float = 10.0  # kcal/mol/A^2 for restraint
    add_missing_residues: bool = True  # Add missing residues from SEQRES
    close_chainbreaks: bool = True  # Close chain breaks during idealization


@dataclass
class InterfaceConfig:
    """Configuration for interface analysis."""

    enabled: bool = False  # Enable interface analysis
    distance_cutoff: float = 8.0  # Distance cutoff (A) for interface residues
    chain_pairs: Optional[List[Tuple[str, str]]] = None  # Auto-detect if None
    calculate_binding_energy: bool = True  # Calculate ddG via chain separation
    calculate_sasa: bool = False  # Calculate buried surface area
    calculate_shape_complementarity: bool = (
        False  # Simplified Sc (experimental)
    )
    # Rosetta InterfaceAnalyzer default: rigid-body separation (no repack/min)
    relax_separated: bool = False  # Repack and minimize separated partners
    relax_separated_iterations: int = 1  # Number of repack+min iterations
    seed: Optional[int] = None  # Base seed for iterations
    sasa_probe_radius: float = 1.4  # Probe radius for SASA (A)
    # Per-position energetics
    per_position: bool = False  # Per-residue dG via residue removal
    alanine_scan: bool = False  # Per-residue ddG via alanine mutation
    scan_chains: Optional[List[str]] = None  # Restrict scan to these chains
    position_relax: PositionRelaxMode = "none"  # Relax policy for scans
    max_scan_sites: Optional[int] = None  # Limit number of residues scanned
    show_progress: bool = False  # Show tqdm progress bar for per-position scans
    quiet: bool = False  # Suppress dependency logging/stderr noise
    workers: int = 1  # Parallel workers for per-position scans (1 = sequential)


@dataclass
class SelectPositionsConfig:
    """Configuration for position selection from interface analysis."""

    source: str = "alanine_scan"  # "per_position" or "alanine_scan"
    metric: str = "ddG"  # PositionRow field to threshold on
    threshold: Optional[float] = None  # Threshold value (None = no filtering)
    direction: str = "above"  # "above" (metric > threshold) or "below"
    top_k: Optional[int] = None  # Select top-k positions (None = no limit)
    order: str = "descending"  # "ascending" | "descending" | "random"
    mode: str = "ALLAA"  # ResidueMode for selected positions
    default_mode: str = "NATAA"  # ResidueMode for non-selected positions
    allowed_aas: Optional[str] = None  # For PIKAA mode: e.g. "ACDEF"


@dataclass
class DdGConfig:
    """Configuration for ddG scoring pipeline.

    When ``paper_mode=True``, ``n_ensemble`` and ``md_total_steps`` are
    overridden to 50 and 100_000 respectively — but only when they are
    still at their default values.  If the user explicitly sets either
    field, the explicit value is preserved even with ``paper_mode=True``.
    """

    # Ensemble
    n_ensemble: int = 35
    md_total_steps: int = 50000
    md_equilibration_steps: int = 5000
    md_temperature: float = 300.0  # Kelvin
    md_friction: float = 1.0  # 1/ps
    neighborhood_sampling_bias: float = 1.0

    # Restraints
    ca_cutoff: float = 9.0  # Angstroms
    restraint_sd: float = 0.5  # Angstroms

    # Neighborhood
    neighborhood_radius: float = 8.0  # Angstroms
    sequence_window: int = 1

    # Interface definition
    chain_pairs: Optional[List[Tuple[str, str]]] = None
    separation_distance: float = 100.0  # Angstroms

    # Energy model
    implicit_solvent: bool = True

    # Unbound-state relaxation
    relax_separated: bool = True
    relax_separated_iterations: int = 1

    # Execution
    workers: int = 1
    seed: Optional[int] = None
    quiet: bool = True

    # Output / cache
    cache_ensemble: bool = False
    ensemble_dir: Optional[Path] = None

    # Analysis parity controls
    sort_members_by_wt_bound_energy: bool = False
    average_top_n: Optional[int] = None

    # Convenience preset
    paper_mode: bool = False

    def __post_init__(self):
        if self.paper_mode:
            if self.n_ensemble == 35:
                self.n_ensemble = 50
            if self.md_total_steps == 50000:
                self.md_total_steps = 100000


@dataclass
class OptimizeConfig:
    """Configuration for the optimize command."""

    chain_pairs: List[Tuple[str, str]]  # REQUIRED, no default
    scan_chains: Optional[List[str]] = None
    n_campaigns: int = 1
    relax_iterations: int = 10
    design_cycles: int = 10
    beam_width: int = 4
    beam_expansion: int = 25
    ddg_threshold: float = 1.0
    position_sampling: str = "weighted"  # "weighted" | "threshold"
    sampling_temperature: float = 1.0  # softmax temperature (> 0)
    regression_tolerance: float = 0.0  # max dG increase allowed
    exclude_native: bool = False
    relax_separated: bool = True
    relax_separated_iterations: int = 1
    relax_separated_scan: bool = True
    design: "DesignConfig" = field(default_factory=lambda: DesignConfig())
    relax: "RelaxConfig" = field(default_factory=lambda: RelaxConfig())
    idealize: "IdealizeConfig" = field(
        default_factory=lambda: IdealizeConfig(
            enabled=True, add_missing_residues=False
        )
    )
    interface_scoring_backend: str = "ddg"  # "ddg" | "legacy"
    ddg: "DdGConfig" = field(default_factory=DdGConfig)
    seed: Optional[int] = None
    workers: int = 1
    show_progress: bool = False
    quiet: bool = True

    def __post_init__(self):
        valid_sampling = ("weighted", "threshold")
        if self.position_sampling not in valid_sampling:
            raise ValueError(
                f"position_sampling must be one of {valid_sampling}, "
                f"got {self.position_sampling!r}"
            )
        if self.sampling_temperature <= 0:
            raise ValueError(
                f"sampling_temperature must be > 0, "
                f"got {self.sampling_temperature}"
            )
        valid_backends = ("ddg", "legacy")
        if self.interface_scoring_backend not in valid_backends:
            raise ValueError(
                f"interface_scoring_backend must be one of "
                f"{valid_backends}, "
                f"got {self.interface_scoring_backend!r}"
            )


@dataclass
class PipelineConfig:
    """Configuration for composite operations (relax, design).

    Bundles the sub-configs needed by :func:`boundry.operations.relax`
    and :func:`boundry.operations.design`, which iterate over
    repack/design + minimize cycles.
    """

    n_iterations: int = 5  # Number of repack/design + minimize cycles
    n_outputs: int = 1  # Number of output models to generate
    scorefile: Optional[Path] = None  # If set, write scores to this file
    verbose: bool = False
    remove_waters: bool = True  # Remove water molecules from input
    show_progress: bool = False  # Show tqdm progress bar for iterations
    design: DesignConfig = field(default_factory=DesignConfig)
    relax: RelaxConfig = field(default_factory=RelaxConfig)
    idealize: IdealizeConfig = field(default_factory=IdealizeConfig)
    interface: InterfaceConfig = field(default_factory=InterfaceConfig)


# -------------------------------------------------------------------
# Workflow configuration
# -------------------------------------------------------------------


@dataclass
class WorkflowStep:
    """A single step in a Boundry workflow.

    Each step maps to one of the core operations (idealize, minimize,
    repack, relax, mpnn, design, analyze_interface) and carries
    operation-specific parameters.
    """

    operation: str  # Operation name (e.g. 'idealize', 'minimize')
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IterateBlock:
    """Repeat a group of steps for a fixed count or until convergence."""

    steps: List["WorkflowStepOrBlock"]
    n: int = 1
    max_n: int = 100
    until: Optional[str] = None


@dataclass
class BeamBlock:
    """Population-based beam search over a nested group of steps."""

    steps: List["WorkflowStepOrBlock"]
    width: int = 5
    rounds: int = 10
    metric: str = "dG"
    direction: Literal["min", "max"] = "min"
    until: Optional[str] = None
    expand: int = 1


@dataclass
class CheckpointStep:
    """Save the current structure under a named checkpoint."""

    name: str


@dataclass
class CompareStep:
    """Compute deltas between current structure and a named checkpoint."""

    name: str


WorkflowStepOrBlock = Union[
    WorkflowStep, IterateBlock, BeamBlock, CheckpointStep, CompareStep
]


@dataclass
class WorkflowConfig:
    """Configuration for a YAML-based workflow.

    Workflows are versioned schemas so the parser can evolve safely.
    """

    input: str  # Input PDB/CIF path
    project_path: Optional[str] = None  # Base output directory (default: cwd)
    seed: Optional[int] = None  # Workflow-level seed for reproducibility
    workers: int = 1  # Global default; 1 = sequential (no pool)
    workflow_version: int = 1
    steps: List[WorkflowStepOrBlock] = field(default_factory=list)
    vars: Dict[str, str] = field(default_factory=dict)
