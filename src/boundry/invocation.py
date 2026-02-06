"""Invocation/output policy helpers for API, CLI, and workflow entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union


class InvocationMode(str, Enum):
    """Supported caller modes."""

    API = "api"
    CLI = "cli"
    WORKFLOW = "workflow"


class OperationKind(str, Enum):
    """High-level operation categories."""

    STRUCTURE = "structure"
    ANALYSIS = "analysis"


class OutputRequirement(str, Enum):
    """Whether an output path is required by policy."""

    OPTIONAL = "optional"
    REQUIRED = "required"


PathLike = Union[str, Path]


@dataclass(frozen=True)
class OutputPolicy:
    """Output-path policy for a specific operation call."""

    requirement: OutputRequirement = OutputRequirement.OPTIONAL

    def validate(
        self,
        output: Optional[PathLike],
        *,
        operation: str,
        mode: InvocationMode,
    ) -> None:
        """Validate output-path presence for this policy."""
        if (
            self.requirement == OutputRequirement.REQUIRED
            and output is None
        ):
            raise ValueError(
                f"{operation} requires an output path in {mode.value} mode"
            )
