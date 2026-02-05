"""
Boundry: Combine LigandMPNN sequence design with AMBER relaxation.

This package provides tools for protein engineering that alternate between
neural network-based sequence design/repacking (LigandMPNN) and physics-based
energy minimization (OpenMM AMBER), similar to Rosetta FastRelax and Design
protocols.

Core operations are available as top-level imports::

    from boundry import idealize, minimize, repack, relax, mpnn, design
    from boundry import analyze_interface
    from boundry import Structure, Workflow
"""

try:
    from boundry._version import __version__
except ImportError:
    # Package not installed (running from source without build)
    __version__ = "0.0.0.dev0"

# Core operations (heavy deps are lazy-loaded inside each function)
from boundry.operations import (
    InterfaceAnalysisResult,
    Structure,
    analyze_interface,
    design,
    idealize,
    minimize,
    mpnn,
    repack,
    relax,
    renumber,
)

# Workflow system
from boundry.workflow import Workflow

# Configuration dataclasses (lightweight, no heavy deps)
from boundry.config import (
    BeamBlock,
    DesignConfig,
    IdealizeConfig,
    InterfaceConfig,
    IterateBlock,
    PipelineConfig,
    RelaxConfig,
    WorkflowConfig,
    WorkflowStep,
)

# Resfile parsing
from boundry.resfile import (
    DesignSpec,
    ResfileParser,
    ResidueMode,
    ResidueSpec,
)

__all__ = [
    # Operations
    "idealize",
    "minimize",
    "repack",
    "relax",
    "mpnn",
    "design",
    "renumber",
    "analyze_interface",
    # Data classes
    "Structure",
    "InterfaceAnalysisResult",
    # Workflow
    "Workflow",
    # Configuration
    "PipelineConfig",
    "DesignConfig",
    "RelaxConfig",
    "IdealizeConfig",
    "InterfaceConfig",
    "WorkflowConfig",
    "WorkflowStep",
    "IterateBlock",
    "BeamBlock",
    # Resfile
    "ResidueMode",
    "ResidueSpec",
    "DesignSpec",
    "ResfileParser",
]
