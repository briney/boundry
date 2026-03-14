"""
Boundry: Combine LigandMPNN sequence design with AMBER relaxation.

This package provides tools for protein engineering that alternate between
neural network-based sequence design/repacking (LigandMPNN) and physics-based
energy minimization (OpenMM AMBER), similar to Rosetta FastRelax and Design
protocols.

Core operations are available as top-level imports::

    from boundry import idealize, minimize, repack, relax, mpnn, design
    from boundry import analyze_interface, ddg
    from boundry import Structure
"""

try:
    from boundry._version import __version__
except ImportError:
    # Package not installed (running from source without build)
    __version__ = "0.0.0.dev0"

# ddG types — imported BEFORE operations so the ``ddg`` function
# from ``boundry.operations`` overwrites the submodule reference
# that Python auto-binds when importing ``boundry.ddg``.
from boundry.ddg import DdGResult, MutationSpec

# Core operations (heavy deps are lazy-loaded inside each function)
from boundry.operations import (
    InterfaceAnalysisResult,
    Structure,
    analyze_interface,
    ddg,
    design,
    idealize,
    minimize,
    mpnn,
    repack,
    relax,
    renumber,
    select_positions,
)

# Optimize
from boundry.optimize import OptimizeResult, optimize

# Configuration dataclasses (lightweight, no heavy deps)
from boundry.config import (
    DdGConfig,
    DesignConfig,
    IdealizeConfig,
    InterfaceConfig,
    OptimizeConfig,
    PipelineConfig,
    RelaxConfig,
    SelectPositionsConfig,
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
    "select_positions",
    "ddg",
    # Data classes
    "Structure",
    "InterfaceAnalysisResult",
    "DdGResult",
    "MutationSpec",
    # Optimize
    "optimize",
    "OptimizeResult",
    "OptimizeConfig",
    # Configuration
    "PipelineConfig",
    "DesignConfig",
    "RelaxConfig",
    "IdealizeConfig",
    "InterfaceConfig",
    "SelectPositionsConfig",
    "DdGConfig",
    # Resfile
    "ResidueMode",
    "ResidueSpec",
    "DesignSpec",
    "ResfileParser",
]
