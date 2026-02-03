"""
Boundry: Combine LigandMPNN sequence design with AMBER relaxation.

This package provides a CLI tool that alternates between neural network-based
sequence design/repacking (LigandMPNN) and physics-based energy minimization
(OpenMM AMBER), similar to Rosetta FastRelax and Design protocols.
"""

try:
    from boundry._version import __version__
except ImportError:
    # Package not installed (running from source without build)
    __version__ = "0.0.0.dev0"


def __getattr__(name):
    """Lazy import heavy modules only when accessed."""
    if name == "Pipeline":
        from boundry.pipeline import Pipeline

        return Pipeline
    elif name == "PipelineMode":
        from boundry.config import PipelineMode

        return PipelineMode
    elif name == "PipelineConfig":
        from boundry.config import PipelineConfig

        return PipelineConfig
    elif name == "DesignConfig":
        from boundry.config import DesignConfig

        return DesignConfig
    elif name == "RelaxConfig":
        from boundry.config import RelaxConfig

        return RelaxConfig
    elif name == "InterfaceConfig":
        from boundry.config import InterfaceConfig

        return InterfaceConfig
    elif name == "ResidueMode":
        from boundry.resfile import ResidueMode

        return ResidueMode
    elif name == "ResidueSpec":
        from boundry.resfile import ResidueSpec

        return ResidueSpec
    elif name == "DesignSpec":
        from boundry.resfile import DesignSpec

        return DesignSpec
    elif name == "ResfileParser":
        from boundry.resfile import ResfileParser

        return ResfileParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PipelineMode",
    "PipelineConfig",
    "DesignConfig",
    "RelaxConfig",
    "InterfaceConfig",
    "ResidueMode",
    "ResidueSpec",
    "DesignSpec",
    "ResfileParser",
    "Pipeline",
]
