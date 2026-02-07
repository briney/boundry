"""YAML-based workflow system for Boundry."""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from boundry._progress import WorkflowProgress, _extract_metric_names
from boundry.condition import ConditionError, check_condition, parse_condition
from boundry.config import (
    BeamBlock,
    IterateBlock,
    WorkflowConfig,
    WorkflowStep,
    WorkflowStepOrBlock,
)
from boundry.workflow_metadata import extract_numeric_metric, merge_metadata

logger = logging.getLogger(__name__)

SEED_PARAM_BY_OPERATION: Dict[str, str] = {
    "repack": "seed",
    "mpnn": "seed",
    "relax": "seed",
    "design": "seed",
    "analyze_interface": "seed",
}


class WorkflowError(Exception):
    """Raised for workflow validation or execution failures."""


def _extract_fields(params: Dict[str, Any], cls: type) -> Dict[str, Any]:
    """Pop keys from *params* that match fields of dataclass *cls*."""
    fields = {f.name for f in dataclasses.fields(cls)}
    return {k: params.pop(k) for k in list(params) if k in fields}


_KNOWN_KEYS = frozenset(
    {
        "workflow_version",
        "input",
        "project_path",
        "seed",
        "workers",
        "steps",
    }
)


# ------------------------------------------------------------------
# Output path context
# ------------------------------------------------------------------


@dataclass(frozen=True)
class OutputPathContext:
    """Tracks the current position in the workflow output directory tree.

    Immutable — each method returns a new child context with an
    additional path segment appended.
    """

    base_path: Path
    segments: tuple[str, ...] = ()

    def child(self, segment: str) -> "OutputPathContext":
        """Create a child context with an additional path segment."""
        return OutputPathContext(
            base_path=self.base_path,
            segments=self.segments + (segment,),
        )

    def step_dir(self, index: int, name: str) -> "OutputPathContext":
        """Create a child context for a workflow step or block."""
        return self.child(f"{index}.{name}")

    def cycle_dir(self, cycle: int) -> "OutputPathContext":
        """Create a child context for an iterate cycle (1-indexed)."""
        return self.child(f"cycle_{cycle}")

    def round_dir(self, round_num: int) -> "OutputPathContext":
        """Create a child context for a beam round (1-indexed)."""
        return self.child(f"round_{round_num}")

    def rank_dir(self, rank: int) -> "OutputPathContext":
        """Create a child context for a selected beam candidate."""
        return self.child(f"rank_{rank}")

    def others_rank_dir(self, rank: int) -> "OutputPathContext":
        """Create a child context for a non-selected beam candidate."""
        return self.child("others").child(f"rank_{rank}")

    def resolve(self) -> Path:
        """Resolve to a full filesystem path."""
        return self.base_path.joinpath(*self.segments)


# ------------------------------------------------------------------
# Operation output spec — what each operation natively writes
# ------------------------------------------------------------------


@dataclass(frozen=True)
class _OperationOutputSpec:
    """Specification for what an operation writes to disk."""

    writes_pdb: bool
    pdb_stem: str  # e.g. "idealized", "relaxed" (unused if writes_pdb=False)
    native_writers: tuple[str, ...]  # keys into _NATIVE_WRITERS


_OPERATION_OUTPUT_SPECS: Dict[str, _OperationOutputSpec] = {
    "idealize": _OperationOutputSpec(
        True, "idealized", ("metrics",)
    ),
    "minimize": _OperationOutputSpec(
        True, "minimized", ("metrics",)
    ),
    "repack": _OperationOutputSpec(
        True, "repacked", ("metrics",)
    ),
    "mpnn": _OperationOutputSpec(
        True, "designed_mpnn", ("metrics",)
    ),
    "relax": _OperationOutputSpec(
        True, "relaxed", ("metrics", "energy")
    ),
    "design": _OperationOutputSpec(
        True, "designed", ("metrics", "energy")
    ),
    "renumber": _OperationOutputSpec(
        True, "renumbered", ("metrics",)
    ),
    "select_positions": _OperationOutputSpec(
        False, "", ("metrics", "selected_positions")
    ),
    "analyze_interface": _OperationOutputSpec(
        False,
        "",
        ("metrics", "interface", "per_position_csv", "alanine_scan_csv"),
    ),
}


# Which top-level metadata keys are native to each operation.
# Only these are included in that operation's metrics.json.
_NATIVE_METRIC_KEYS: Dict[str, tuple[str, ...]] = {
    "idealize": ("chain_gaps",),
    "minimize": ("initial_energy", "final_energy", "rmsd"),
    "repack": ("ligandmpnn_loss",),
    "mpnn": ("ligandmpnn_loss",),
    "relax": ("final_energy",),
    "design": ("final_energy", "ligandmpnn_loss"),
    "renumber": (),
    "select_positions": (
        "selected_positions",
        "selection_source",
        "selection_metric",
        "selection_threshold",
        "selection_direction",
        "selection_mode",
        "selection_top_k",
        "selection_order",
    ),
    "analyze_interface": (
        "dG",
        "complex_energy",
        "buried_sasa",
        "sc_score",
        "n_interface_residues",
    ),
}


# ------------------------------------------------------------------
# Native auxiliary file writers
# ------------------------------------------------------------------


def _write_native_metrics(
    result_metadata: Dict[str, Any],
    dir_path: Path,
    operation: str,
) -> None:
    """Write operation-specific metrics to metrics.json."""
    native_keys = _NATIVE_METRIC_KEYS.get(operation, ())
    metrics = {}
    for key in native_keys:
        if key in result_metadata:
            value = result_metadata[key]
            if isinstance(value, (int, float)) and not isinstance(
                value, bool
            ):
                metrics[key] = value
            elif isinstance(value, str):
                metrics[key] = value
    if metrics:
        json_path = dir_path / "metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"  Wrote metrics: {json_path}")


def _write_energy_json(
    result_metadata: Dict[str, Any],
    dir_path: Path,
    operation: str,
) -> None:
    """Write energy breakdown from relax/design operations."""
    energy = result_metadata.get("energy_breakdown")
    if energy:
        json_path = dir_path / "energy.json"
        with open(json_path, "w") as f:
            json.dump(energy, f, indent=2, default=str)
        logger.info(f"  Wrote energy breakdown: {json_path}")


def _write_selected_positions_json(
    result_metadata: Dict[str, Any],
    dir_path: Path,
    operation: str,
) -> None:
    """Write selected positions from select_positions operation."""
    design_spec = result_metadata.get("design_spec")
    if design_spec is None:
        return
    data: Dict[str, Any] = {
        "selected_count": result_metadata.get(
            "selected_positions", 0
        ),
        "selection_criteria": {
            "source": result_metadata.get("selection_source"),
            "metric": result_metadata.get("selection_metric"),
            "threshold": result_metadata.get("selection_threshold"),
            "direction": result_metadata.get("selection_direction"),
            "mode": result_metadata.get("selection_mode"),
        },
        "positions": [
            {
                "chain": spec.chain,
                "resnum": spec.resnum,
                "icode": spec.icode,
            }
            for spec in design_spec.residue_specs.values()
        ],
    }
    json_path = dir_path / "selected_positions.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"  Wrote selected positions: {json_path}")


def _write_interface_json(
    result_metadata: Dict[str, Any],
    dir_path: Path,
    operation: str,
) -> None:
    """Write interface analysis summary metrics."""
    keys = (
        "dG",
        "complex_energy",
        "buried_sasa",
        "sc_score",
        "n_interface_residues",
    )
    data = {}
    for key in keys:
        if key in result_metadata:
            data[key] = result_metadata[key]
    if data:
        json_path = dir_path / "interface.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"  Wrote interface metrics: {json_path}")


def _write_per_position_csv(
    result_metadata: Dict[str, Any],
    dir_path: Path,
    operation: str,
) -> None:
    """Write per-position energetics CSV."""
    per_position = result_metadata.get("per_position")
    if per_position is not None:
        from boundry.interface_position_energetics import (
            write_position_csv,
        )

        csv_path = dir_path / "per_position.csv"
        write_position_csv(per_position, csv_path)
        logger.info(f"  Wrote per-position CSV: {csv_path}")


def _write_alanine_scan_csv(
    result_metadata: Dict[str, Any],
    dir_path: Path,
    operation: str,
) -> None:
    """Write alanine scan CSV."""
    alanine_scan = result_metadata.get("alanine_scan")
    if alanine_scan is not None:
        from boundry.interface_position_energetics import (
            write_position_csv,
        )

        csv_path = dir_path / "alanine_scan.csv"
        write_position_csv(alanine_scan, csv_path)
        logger.info(f"  Wrote alanine scan CSV: {csv_path}")


# Writer function signature: (result_metadata, dir_path, operation) -> None
_NativeWriter = Callable[[Dict[str, Any], Path, str], None]

_NATIVE_WRITERS: Dict[str, _NativeWriter] = {
    "metrics": _write_native_metrics,
    "energy": _write_energy_json,
    "selected_positions": _write_selected_positions_json,
    "interface": _write_interface_json,
    "per_position_csv": _write_per_position_csv,
    "alanine_scan_csv": _write_alanine_scan_csv,
}


# ------------------------------------------------------------------
# YAML parsing
# ------------------------------------------------------------------


ParserFn = Callable[[Dict[str, Any], str], WorkflowStepOrBlock]
_NODE_PARSERS: Dict[str, ParserFn] = {}


def _register_parser(key: str) -> Callable[[ParserFn], ParserFn]:
    def _decorator(func: ParserFn) -> ParserFn:
        _NODE_PARSERS[key] = func
        return func

    return _decorator


def _parse_steps(
    raw_steps: Any,
    prefix: str = "",
) -> List[WorkflowStepOrBlock]:
    """Parse raw YAML step nodes recursively."""
    if not isinstance(raw_steps, list):
        raise WorkflowError(
            f"{prefix}Expected a list of steps, got "
            f"{type(raw_steps).__name__}"
        )
    if not raw_steps:
        raise WorkflowError(f"{prefix}steps list must not be empty")

    parsed: List[WorkflowStepOrBlock] = []
    valid_keys = sorted(_NODE_PARSERS)

    for i, step_data in enumerate(raw_steps, 1):
        label = f"{prefix}Step {i}"
        if not isinstance(step_data, dict):
            raise WorkflowError(
                f"{label}: expected a mapping, "
                f"got {type(step_data).__name__}"
            )
        found_keys = [k for k in valid_keys if k in step_data]
        if not found_keys:
            raise WorkflowError(
                f"{label}: expected one of "
                f"{', '.join(repr(k) for k in valid_keys)}"
            )
        if len(found_keys) > 1:
            raise WorkflowError(
                f"{label}: found multiple node keys {found_keys}; "
                "use exactly one"
            )
        parser = _NODE_PARSERS[found_keys[0]]
        parsed.append(parser(step_data, label))

    return parsed


def _validate_unknown_keys(
    data: Dict[str, Any], allowed: set[str], label: str
) -> None:
    extra = sorted(k for k in data if k not in allowed)
    if extra:
        raise WorkflowError(
            f"{label}: unknown fields {extra}. Allowed: {sorted(allowed)}"
        )


def _parse_positive_int(value: Any, label: str, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkflowError(
            f"{label}: '{field_name}' must be an integer, "
            f"got {type(value).__name__}"
        )
    if value < 1:
        raise WorkflowError(
            f"{label}: '{field_name}' must be >= 1, got {value}"
        )
    return value


def _parse_optional_positive_int(
    value: Any, label: str, field_name: str
) -> Optional[int]:
    if value is None:
        return None
    return _parse_positive_int(value, label, field_name)


def _parse_optional_str(
    value: Any, label: str, field_name: str
) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise WorkflowError(
            f"{label}: '{field_name}' must be a string, "
            f"got {type(value).__name__}"
        )
    return value


@_register_parser("operation")
def _parse_operation(
    step_data: Dict[str, Any], label: str
) -> WorkflowStep:
    _validate_unknown_keys(
        step_data,
        {"operation", "params"},
        label,
    )

    operation = step_data["operation"]
    if not isinstance(operation, str):
        raise WorkflowError(
            f"{label}: 'operation' must be a string, "
            f"got {type(operation).__name__}"
        )

    params = step_data.get("params") or {}
    if not isinstance(params, dict):
        raise WorkflowError(
            f"{label}: 'params' must be a mapping, "
            f"got {type(params).__name__}"
        )

    return WorkflowStep(operation=operation, params=params)


@_register_parser("iterate")
def _parse_iterate(
    step_data: Dict[str, Any], label: str
) -> IterateBlock:
    _validate_unknown_keys(step_data, {"iterate"}, label)
    block_data = step_data["iterate"]
    if not isinstance(block_data, dict):
        raise WorkflowError(
            f"{label}.iterate: expected mapping, "
            f"got {type(block_data).__name__}"
        )

    _validate_unknown_keys(
        block_data,
        {"steps", "n", "max_n", "until", "workers"},
        f"{label}.iterate",
    )

    steps = _parse_steps(
        block_data.get("steps"),
        prefix=f"{label}.iterate.",
    )

    n = _parse_positive_int(
        block_data.get("n", 1),
        f"{label}.iterate",
        "n",
    )
    max_n = _parse_positive_int(
        block_data.get("max_n", 100),
        f"{label}.iterate",
        "max_n",
    )
    until = _parse_optional_str(
        block_data.get("until"),
        f"{label}.iterate",
        "until",
    )
    workers = _parse_optional_positive_int(
        block_data.get("workers"),
        f"{label}.iterate",
        "workers",
    )

    if until is None and n < 1:
        raise WorkflowError(f"{label}.iterate: n must be >= 1")
    if until is not None:
        parse_condition(until)
        if max_n < 1:
            raise WorkflowError(
                f"{label}.iterate: max_n must be >= 1"
            )

    return IterateBlock(
        steps=steps,
        n=n,
        max_n=max_n,
        until=until,
        workers=workers,
    )


@_register_parser("beam")
def _parse_beam(
    step_data: Dict[str, Any], label: str
) -> BeamBlock:
    _validate_unknown_keys(step_data, {"beam"}, label)
    block_data = step_data["beam"]
    if not isinstance(block_data, dict):
        raise WorkflowError(
            f"{label}.beam: expected mapping, got "
            f"{type(block_data).__name__}"
        )

    _validate_unknown_keys(
        block_data,
        {
            "steps",
            "width",
            "rounds",
            "metric",
            "direction",
            "until",
            "expand",
            "workers",
        },
        f"{label}.beam",
    )

    steps = _parse_steps(
        block_data.get("steps"),
        prefix=f"{label}.beam.",
    )
    width = _parse_positive_int(
        block_data.get("width", 5),
        f"{label}.beam",
        "width",
    )
    rounds = _parse_positive_int(
        block_data.get("rounds", 10),
        f"{label}.beam",
        "rounds",
    )
    expand = _parse_positive_int(
        block_data.get("expand", 1),
        f"{label}.beam",
        "expand",
    )
    metric = _parse_optional_str(
        block_data.get("metric", "dG"),
        f"{label}.beam",
        "metric",
    )
    if metric is None:
        raise WorkflowError(
            f"{label}.beam: metric must not be null"
        )
    direction = block_data.get("direction", "min")
    if direction not in {"min", "max"}:
        raise WorkflowError(
            f"{label}.beam: direction must be 'min' or 'max', "
            f"got '{direction}'"
        )
    until = _parse_optional_str(
        block_data.get("until"),
        f"{label}.beam",
        "until",
    )
    if until is not None:
        parse_condition(until)
    workers = _parse_optional_positive_int(
        block_data.get("workers"),
        f"{label}.beam",
        "workers",
    )

    return BeamBlock(
        steps=steps,
        width=width,
        rounds=rounds,
        metric=metric,
        direction=direction,
        until=until,
        expand=expand,
        workers=workers,
    )


def _ensure_resolvers() -> None:
    """Register custom OmegaConf resolvers (idempotent)."""
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver(
            "env",
            lambda key, default="": os.environ.get(key, default),
        )


# ------------------------------------------------------------------
# Operation dispatch registry
# ------------------------------------------------------------------


@dataclass(frozen=True)
class _SimpleSpec:
    """Spec for operations that use a single config class."""

    op_name: str
    config_cls: Optional[str]
    config_overrides: Dict[str, Any]
    extra_params: tuple[str, ...]
    needs_weights: bool


@dataclass(frozen=True)
class _PipelineSpec:
    """Spec for operations that use PipelineConfig (design + relax)."""

    op_name: str
    extra_params: tuple[str, ...]
    needs_weights: bool = True


_OPERATION_REGISTRY: Dict[
    str, Optional[Union[_SimpleSpec, _PipelineSpec]]
] = {
    "idealize": _SimpleSpec(
        "idealize", "IdealizeConfig", {"enabled": True}, (), False
    ),
    "minimize": _SimpleSpec(
        "minimize", "RelaxConfig", {}, ("pre_idealize",), False
    ),
    "repack": _SimpleSpec(
        "repack",
        "DesignConfig",
        {},
        ("pre_idealize", "resfile", "design_spec"),
        True,
    ),
    "mpnn": _SimpleSpec(
        "mpnn",
        "DesignConfig",
        {},
        ("pre_idealize", "resfile", "design_spec"),
        True,
    ),
    "relax": _PipelineSpec(
        "relax",
        ("pre_idealize", "resfile", "design_spec", "n_iterations"),
    ),
    "design": _PipelineSpec(
        "design",
        ("pre_idealize", "resfile", "design_spec", "n_iterations"),
    ),
    "renumber": _SimpleSpec("renumber", None, {}, (), False),
    "select_positions": _SimpleSpec(
        "select_positions", "SelectPositionsConfig", {}, (), False
    ),
    "analyze_interface": None,  # special-cased
}

VALID_OPERATIONS = frozenset(_OPERATION_REGISTRY)


# ------------------------------------------------------------------
# Execution context and snapshot for beam deferred-write
# ------------------------------------------------------------------


@dataclass
class _ExecutionContext:
    population: List["Structure"]
    last_operation: Optional[str] = None
    output_context: Optional[OutputPathContext] = None


@dataclass
class _StepSnapshot:
    """Captures a single step's result for deferred writing (beam)."""

    operation: str
    step_index: int
    result_metadata: Dict[str, Any]  # pre-merge metadata from this step
    pdb_string: str
    structure: "Structure"  # the merged structure after this step


# ------------------------------------------------------------------
# Workflow class
# ------------------------------------------------------------------


class Workflow:
    """YAML-defined workflow runner with composable compound blocks."""

    def __init__(
        self,
        config: WorkflowConfig,
    ):
        self.config = config
        self.last_population: List["Structure"] = []
        self._progress = WorkflowProgress(enabled=False)
        self._validate()

    @classmethod
    def from_yaml(
        cls,
        path: Union[str, Path],
        seed: Optional[int] = None,
        workers: Optional[int] = None,
        overrides: Optional[List[str]] = None,
    ) -> "Workflow":
        """Load a workflow from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to the YAML workflow file.
        seed : int, optional
            Workflow-level seed for reproducibility. Overrides YAML
            ``seed`` when both are present.
        workers : int, optional
            Number of parallel workers. Overrides YAML ``workers``
            when both are present.
        overrides : list of str, optional
            Dotlist-style overrides (e.g. ``["project_path=results/",
            "seed=42"]``).  Applied after YAML loading but before
            variable resolution.
        """
        from omegaconf import DictConfig, OmegaConf
        from omegaconf.errors import OmegaConfBaseException

        _ensure_resolvers()

        path = Path(path)
        try:
            cfg = OmegaConf.load(path)
        except OmegaConfBaseException as exc:
            raise WorkflowError(
                f"Variable resolution failed: {exc}"
            ) from exc

        if not isinstance(cfg, DictConfig):
            raise WorkflowError(
                "Workflow YAML must be a mapping, "
                f"got {type(cfg).__name__}"
            )

        # Apply CLI overrides before resolution
        if overrides:
            try:
                override_cfg = OmegaConf.from_dotlist(overrides)
                cfg = OmegaConf.merge(cfg, override_cfg)
            except (OmegaConfBaseException, ValueError) as exc:
                raise WorkflowError(
                    f"Invalid override: {exc}"
                ) from exc

        # Resolve all ${...} interpolations → plain Python dict
        try:
            data = OmegaConf.to_container(cfg, resolve=True)
        except OmegaConfBaseException as exc:
            msg = str(exc)
            if (
                "circular" in msg.lower()
                or "recursion" in msg.lower()
                or "recursive" in msg.lower()
            ):
                raise WorkflowError(
                    f"Circular variable reference detected: {exc}"
                ) from exc
            raise WorkflowError(
                f"Variable resolution failed: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise WorkflowError(
                "Workflow YAML must be a mapping, "
                f"got {type(data).__name__}"
            )

        if "input" not in data:
            raise WorkflowError(
                "Workflow must specify 'input' field"
            )
        if "steps" not in data:
            raise WorkflowError(
                "Workflow must specify 'steps' field"
            )

        # Deprecation warning for old 'output' key
        if "output" in data:
            import warnings

            warnings.warn(
                "The top-level 'output' key is deprecated. "
                "Use 'project_path' instead. Per-step output "
                "paths are no longer supported.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Treat as project_path if project_path not already set
            if "project_path" not in data:
                data["project_path"] = data.pop("output")
            else:
                data.pop("output")

        # Validate user-defined keys (non-known keys must be valid
        # identifiers with scalar values).
        user_vars: Dict[str, str] = {}
        for key in list(data):
            if key in _KNOWN_KEYS:
                continue
            if not key.isidentifier():
                raise WorkflowError(
                    f"Invalid variable name '{key}'. "
                    "User-defined keys must be valid Python "
                    "identifiers (letters, digits, underscores; "
                    "cannot start with a digit)."
                )
            value = data[key]
            if isinstance(value, bool) or not isinstance(
                value, (str, int, float)
            ):
                raise WorkflowError(
                    f"User-defined variable '{key}' must have "
                    f"a scalar value (string, int, or float), "
                    f"got {type(value).__name__}"
                )

        # Separate user-defined keys before parsing known fields.
        for key in list(data):
            if key not in _KNOWN_KEYS:
                user_vars[key] = str(data.pop(key))

        input_path = data["input"]
        if not isinstance(input_path, str):
            raise WorkflowError(
                "Workflow 'input' must be a string"
            )

        project_path = _parse_optional_str(
            data.get("project_path"), "Workflow", "project_path"
        )
        version = data.get("workflow_version", 1)
        if isinstance(version, bool) or not isinstance(version, int):
            raise WorkflowError(
                "workflow_version must be an integer"
            )

        # Parse seed: CLI argument overrides YAML value
        yaml_seed = data.get("seed")
        if yaml_seed is not None:
            if isinstance(yaml_seed, bool) or not isinstance(
                yaml_seed, int
            ):
                raise WorkflowError(
                    "Workflow 'seed' must be an integer, "
                    f"got {type(yaml_seed).__name__}"
                )
        effective_seed = seed if seed is not None else yaml_seed

        # Parse workers: CLI argument overrides YAML value
        yaml_workers = data.get("workers", 1)
        if isinstance(yaml_workers, bool) or not isinstance(
            yaml_workers, int
        ):
            raise WorkflowError(
                "Workflow 'workers' must be an integer, "
                f"got {type(yaml_workers).__name__}"
            )
        if yaml_workers < 1:
            raise WorkflowError(
                "Workflow 'workers' must be >= 1, "
                f"got {yaml_workers}"
            )
        effective_workers = (
            workers if workers is not None else yaml_workers
        )

        steps = _parse_steps(data["steps"])
        config = WorkflowConfig(
            input=input_path,
            project_path=project_path,
            seed=effective_seed,
            workers=effective_workers,
            workflow_version=version,
            steps=steps,
            vars=user_vars,
        )
        return cls(config)

    def _validate(self) -> None:
        """Validate workflow configuration recursively."""
        if self.config.workflow_version != 1:
            raise WorkflowError(
                f"Unsupported workflow_version="
                f"{self.config.workflow_version}. "
                "Only version 1 is supported."
            )
        if not self.config.steps:
            raise WorkflowError(
                "Workflow must contain at least one step"
            )
        self._validate_steps(self.config.steps, prefix="")

    def _validate_steps(
        self, steps: List[WorkflowStepOrBlock], prefix: str
    ) -> None:
        for i, item in enumerate(steps, 1):
            label = f"{prefix}Step {i}"
            if isinstance(item, WorkflowStep):
                self._validate_step(item, label)
            elif isinstance(item, IterateBlock):
                self._validate_iterate(item, label)
            elif isinstance(item, BeamBlock):
                self._validate_beam(item, label)
            else:
                raise WorkflowError(
                    f"{label}: unsupported node type "
                    f"'{type(item).__name__}'"
                )

    def _validate_step(
        self, step: WorkflowStep, label: str
    ) -> None:
        if step.operation not in VALID_OPERATIONS:
            raise WorkflowError(
                f"{label}: unknown operation '{step.operation}'. "
                f"Valid: {', '.join(sorted(VALID_OPERATIONS))}"
            )

    def _validate_iterate(
        self, block: IterateBlock, label: str
    ) -> None:
        if block.until is None and block.n < 1:
            raise WorkflowError(
                f"{label}: iterate n must be >= 1"
            )
        if block.until is not None and block.max_n < 1:
            raise WorkflowError(
                f"{label}: iterate max_n must be >= 1"
            )
        self._validate_steps(
            block.steps, prefix=f"{label}.iterate."
        )

    def _validate_beam(
        self, block: BeamBlock, label: str
    ) -> None:
        if (
            block.width < 1
            or block.rounds < 1
            or block.expand < 1
        ):
            raise WorkflowError(
                f"{label}: beam width/rounds/expand must be >= 1"
            )
        if block.direction not in {"min", "max"}:
            raise WorkflowError(
                f"{label}: beam direction must be 'min' or 'max'"
            )
        self._validate_steps(
            block.steps, prefix=f"{label}.beam."
        )

    # ------------------------------------------------------------------
    # Parallelism helpers
    # ------------------------------------------------------------------

    def _effective_workers(
        self, block_workers: Optional[int]
    ) -> int:
        """Resolve the effective worker count for a block.

        Per-block ``workers`` overrides the global
        ``WorkflowConfig.workers``.
        """
        if block_workers is not None:
            return block_workers
        return self.config.workers

    @staticmethod
    def _has_nested_blocks(
        steps: List[WorkflowStepOrBlock],
    ) -> bool:
        """Check if *steps* contain any nested IterateBlock or
        BeamBlock."""
        return any(
            isinstance(s, (IterateBlock, BeamBlock))
            for s in steps
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self, show_progress: bool = False
    ) -> "Structure":
        """Execute workflow and return the best final structure."""
        population = self.run_population(
            show_progress=show_progress
        )
        return population[0]

    def run_population(
        self, show_progress: bool = False
    ) -> List["Structure"]:
        """Execute workflow and return the full final candidate
        population."""
        from boundry.operations import Structure

        input_path = Path(self.config.input)
        if not input_path.exists():
            raise WorkflowError(
                f"Input file not found: {input_path}"
            )

        # Resolve project_path (defaults to current working directory)
        project_path = Path(self.config.project_path or ".")
        project_path.mkdir(parents=True, exist_ok=True)
        output_ctx = OutputPathContext(base_path=project_path)

        current = _ExecutionContext(
            population=[Structure.from_file(input_path)],
            output_context=output_ctx,
        )
        logger.info(f"Loaded input: {input_path}")
        logger.info(f"Project path: {project_path.resolve()}")

        total = len(self.config.steps)

        with WorkflowProgress(enabled=show_progress) as progress:
            self._progress = progress
            progress.start_workflow(total)

            for i, item in enumerate(self.config.steps):
                desc = self._describe_item(item)
                logger.info(
                    f"Step {i + 1}/{total}: {desc}"
                )
                progress.update_workflow_status(desc)
                current = self._execute_item(
                    item,
                    current,
                    seed_base=self.config.seed,
                    step_index=i,
                )
                progress.advance_workflow(desc)

            progress.finish_workflow()
            self._progress = WorkflowProgress(enabled=False)

        self.last_population = list(current.population)
        return list(current.population)

    def _execute_item(
        self,
        item: WorkflowStepOrBlock,
        context: _ExecutionContext,
        seed_base: Optional[int],
        step_index: int = 0,
    ) -> _ExecutionContext:
        if isinstance(item, WorkflowStep):
            return self._execute_step(
                item, context, seed_base, step_index
            )
        if isinstance(item, IterateBlock):
            return self._execute_iterate(
                item, context, seed_base, step_index
            )
        if isinstance(item, BeamBlock):
            return self._execute_beam(
                item, context, seed_base, step_index
            )
        raise WorkflowError(
            f"Unsupported node type '{type(item).__name__}'"
        )

    def _execute_step(
        self,
        step: WorkflowStep,
        context: _ExecutionContext,
        seed_base: Optional[int],
        step_index: int,
    ) -> _ExecutionContext:
        from boundry.operations import Structure

        if step.operation not in _OPERATION_REGISTRY:
            raise WorkflowError(
                f"Unknown operation '{step.operation}'"
            )

        self._progress.start_inner_step(step.operation)

        # Build output context for this step
        step_ctx = (
            context.output_context.step_dir(
                step_index, step.operation
            )
            if context.output_context is not None
            else None
        )

        pop_size = len(context.population)
        workers = self.config.workers
        use_parallel = workers > 1 and pop_size > 1

        if use_parallel:
            results = self._execute_step_parallel(
                step, context.population, seed_base, workers
            )
        else:
            results = self._execute_step_sequential(
                step, context.population, seed_base
            )

        # Write outputs and merge metadata (always sequential)
        updated: List[Structure] = []
        for idx, (structure, result) in enumerate(
            zip(context.population, results)
        ):
            if step_ctx is not None:
                if pop_size > 1:
                    write_ctx = step_ctx.rank_dir(idx + 1)
                else:
                    write_ctx = step_ctx
                self._write_step_output(
                    result, write_ctx, step.operation
                )

            merged = merge_metadata(
                structure.metadata,
                result.metadata,
                operation=step.operation,
            )
            updated.append(
                Structure(
                    pdb_string=result.pdb_string,
                    metadata=merged,
                    source_path=(
                        result.source_path or structure.source_path
                    ),
                )
            )

        self._progress.finish_inner_step()

        return _ExecutionContext(
            population=updated,
            last_operation=step.operation,
            output_context=context.output_context,
        )

    def _execute_step_sequential(
        self,
        step: WorkflowStep,
        population: List["Structure"],
        seed_base: Optional[int],
    ) -> List["Structure"]:
        """Run an operation on each population member sequentially."""
        results = []
        for idx, structure in enumerate(population):
            params = self._with_seed(
                step.operation,
                dict(step.params),
                seed_base,
                idx,
            )
            result = self._run_operation(
                step.operation, structure, params
            )
            results.append(result)
        return results

    def _execute_step_parallel(
        self,
        step: WorkflowStep,
        population: List["Structure"],
        seed_base: Optional[int],
        max_workers: int,
    ) -> List["Structure"]:
        """Run an operation on each population member in parallel."""
        from concurrent.futures import as_completed

        from boundry._parallel import (
            StepTask,
            _execute_step_worker,
            get_pool,
        )
        from boundry.operations import Structure

        tasks = []
        for idx, structure in enumerate(population):
            params = self._with_seed(
                step.operation,
                dict(step.params),
                seed_base,
                idx,
            )
            tasks.append(
                StepTask(
                    pdb_string=structure.pdb_string,
                    metadata=dict(structure.metadata),
                    source_path=structure.source_path,
                    operation=step.operation,
                    params=params,
                )
            )

        total = len(tasks)
        results: List[Structure] = [None] * total  # type: ignore

        pool = get_pool(max_workers)
        try:
            future_to_idx = {
                pool.submit(_execute_step_worker, task): idx
                for idx, task in enumerate(tasks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                step_result = future.result()

                if step_result.error is not None:
                    raise WorkflowError(
                        f"Step '{step.operation}' member "
                        f"{idx + 1}/{total} failed: "
                        f"{step_result.error}"
                    )

                results[idx] = Structure(
                    pdb_string=step_result.pdb_string,
                    metadata=step_result.metadata,
                    source_path=step_result.source_path,
                )
        except Exception:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)

        return results

    def _execute_iterate(
        self,
        block: IterateBlock,
        context: _ExecutionContext,
        seed_base: Optional[int],
        step_index: int,
    ) -> _ExecutionContext:
        # Build iterate block output directory context
        block_ctx = (
            context.output_context.step_dir(step_index, "iterate")
            if context.output_context is not None
            else None
        )

        current = context
        previous_metadata: Optional[Dict[str, Any]] = None
        convergence = block.until is not None
        max_iters = block.max_n if convergence else block.n
        converged = False

        # Extract metric name from convergence condition for display
        metric_names = _extract_metric_names(block.until)
        metric_name = metric_names[0] if metric_names else ""

        self._progress.start_iterate(
            max_iters, convergence, metric_name
        )

        for cycle in range(1, max_iters + 1):
            # Create cycle context
            cycle_ctx = (
                block_ctx.cycle_dir(cycle)
                if block_ctx is not None
                else None
            )

            cycle_seed = _compose_seed(seed_base, cycle)
            inner_seed = (
                cycle_seed if seed_base is not None else None
            )

            # Update context for inner steps to use the cycle path
            current = _ExecutionContext(
                population=current.population,
                last_operation=current.last_operation,
                output_context=cycle_ctx,
            )

            for inner_idx, inner in enumerate(block.steps):
                current = self._execute_item(
                    inner,
                    current,
                    inner_seed,
                    step_index=inner_idx,
                )

            # Extract metric value for progress display
            metric_value = None
            if metric_name:
                metric_value = extract_numeric_metric(
                    current.population[0].metadata, metric_name
                )
            self._progress.advance_iterate(cycle, metric_value)

            if convergence:
                converged, previous_metadata = (
                    self._check_convergence(
                        block.until,
                        current.population[0].metadata,
                        previous_metadata,
                        cycle,
                        "Iterate",
                    )
                )
                if converged:
                    logger.info(
                        f"Iterate block converged at cycle {cycle}"
                    )
                    break

        if convergence and not converged:
            logger.warning(
                f"Iterate block reached max_n={block.max_n} "
                "without meeting convergence condition"
            )

        self._progress.finish_iterate()

        # Restore parent output context
        return _ExecutionContext(
            population=current.population,
            last_operation=current.last_operation,
            output_context=context.output_context,
        )

    def _execute_beam(
        self,
        block: BeamBlock,
        context: _ExecutionContext,
        seed_base: Optional[int],
        step_index: int,
    ) -> _ExecutionContext:
        # Build beam block output directory context
        block_ctx = (
            context.output_context.step_dir(step_index, "beam")
            if context.output_context is not None
            else None
        )

        population = list(context.population)
        previous_best_metadata: Optional[Dict[str, Any]] = None

        self._progress.start_beam(block.rounds)

        for round_num in range(1, block.rounds + 1):
            if not population:
                raise WorkflowError("Beam population is empty")

            round_ctx = (
                block_ctx.round_dir(round_num)
                if block_ctx is not None
                else None
            )

            expand_per = max(
                block.expand,
                _ceil_div(block.width, len(population)),
            )

            # Phase 1: Execute all branches without writing,
            # collecting snapshots
            workers = self._effective_workers(block.workers)
            use_parallel = (
                workers > 1
                and not self._has_nested_blocks(block.steps)
            )
            if (
                workers > 1
                and self._has_nested_blocks(block.steps)
            ):
                logger.warning(
                    "Beam steps contain nested blocks; "
                    "falling back to sequential execution"
                )

            expanded: List[
                tuple["Structure", List[_StepSnapshot]]
            ] = []

            if use_parallel:
                expanded = self._expand_beam_parallel(
                    population,
                    block,
                    seed_base,
                    round_num,
                    expand_per,
                    workers,
                )
            else:
                expanded = self._expand_beam_sequential(
                    population,
                    block,
                    seed_base,
                    round_num,
                    expand_per,
                )

            # Score candidates
            scored: List[
                tuple[
                    float,
                    "Structure",
                    List[_StepSnapshot],
                ]
            ] = []
            for candidate, snaps in expanded:
                metric_value = extract_numeric_metric(
                    candidate.metadata,
                    block.metric,
                )
                if metric_value is None:
                    logger.warning(
                        f"Dropping beam candidate missing metric "
                        f"'{block.metric}'"
                    )
                    continue
                scored.append((metric_value, candidate, snaps))

            if not scored:
                raise WorkflowError(
                    "Beam search could not score any candidates. "
                    f"Missing metric '{block.metric}'."
                )

            scored.sort(
                key=lambda item: item[0],
                reverse=(block.direction == "max"),
            )

            best_metric = scored[0][0]
            logger.info(
                f"Beam round {round_num}/{block.rounds}: "
                f"best {block.metric}={best_metric:.4f}"
            )
            self._progress.advance_beam_round(
                round_num, best_metric, block.metric
            )

            # Phase 2: Write outputs to ranked directories
            if round_ctx is not None:
                for rank, (_, cand, snaps) in enumerate(
                    scored, 1
                ):
                    is_selected = rank <= block.width
                    if is_selected:
                        rank_ctx = round_ctx.rank_dir(rank)
                    else:
                        rank_ctx = round_ctx.others_rank_dir(rank)

                    for snap in snaps:
                        snap_ctx = rank_ctx.step_dir(
                            snap.step_index, snap.operation
                        )
                        self._write_step_output(
                            snap.structure,
                            snap_ctx,
                            snap.operation,
                            result_metadata=snap.result_metadata,
                        )

            population = [
                cand for _, cand, _ in scored[: block.width]
            ]

            if block.until is not None:
                done, previous_best_metadata = (
                    self._check_convergence(
                        block.until,
                        population[0].metadata,
                        previous_best_metadata,
                        round_num,
                        "Beam",
                    )
                )
                if done:
                    logger.info(
                        f"Beam block converged at round "
                        f"{round_num}"
                    )
                    break

        self._progress.finish_beam()

        # Restore parent output context
        return _ExecutionContext(
            population=population,
            output_context=context.output_context,
        )

    def _expand_beam_sequential(
        self,
        population: List["Structure"],
        block: BeamBlock,
        seed_base: Optional[int],
        round_num: int,
        expand_per: int,
    ) -> List[tuple["Structure", List[_StepSnapshot]]]:
        """Expand beam branches sequentially (original path)."""
        total_branches = len(population) * expand_per
        self._progress.start_branches(total_branches)

        expanded: List[
            tuple["Structure", List[_StepSnapshot]]
        ] = []
        for cand_idx, candidate in enumerate(population, 1):
            for exp_idx in range(expand_per):
                branch_seed = _compose_seed(
                    seed_base,
                    (round_num * 10000)
                    + (cand_idx * 100)
                    + exp_idx,
                )
                snapshots: List[_StepSnapshot] = []
                branch = _ExecutionContext(
                    population=[_clone_structure(candidate)],
                    output_context=None,
                )
                for inner_idx, inner in enumerate(
                    block.steps
                ):
                    branch = (
                        self._execute_item_with_snapshots(
                            inner,
                            branch,
                            branch_seed,
                            step_index=inner_idx,
                            snapshots=snapshots,
                        )
                    )
                for struct in branch.population:
                    expanded.append((struct, snapshots))
                self._progress.advance_branch()

        self._progress.finish_branches()
        return expanded

    def _expand_beam_parallel(
        self,
        population: List["Structure"],
        block: BeamBlock,
        seed_base: Optional[int],
        round_num: int,
        expand_per: int,
        max_workers: int,
    ) -> List[tuple["Structure", List[_StepSnapshot]]]:
        """Expand beam branches in parallel using a process pool."""
        from concurrent.futures import as_completed

        from boundry._parallel import (
            BranchTask,
            _execute_branch_worker,
            get_pool,
        )
        from boundry.operations import Structure

        # Build tasks
        tasks: List[BranchTask] = []
        for cand_idx, candidate in enumerate(population, 1):
            for exp_idx in range(expand_per):
                branch_seed = _compose_seed(
                    seed_base,
                    (round_num * 10000)
                    + (cand_idx * 100)
                    + exp_idx,
                )
                tasks.append(
                    BranchTask(
                        candidate_pdb_string=(
                            candidate.pdb_string
                        ),
                        candidate_metadata=dict(
                            candidate.metadata
                        ),
                        candidate_source_path=(
                            candidate.source_path
                        ),
                        steps=[
                            (s.operation, dict(s.params))
                            for s in block.steps
                        ],
                        branch_seed=branch_seed,
                    )
                )

        total = len(tasks)
        logger.info(
            f"  Beam expansion: {total} branches "
            f"across {max_workers} workers"
        )

        self._progress.start_branches(total)

        # Submit to pool
        expanded: List[
            tuple["Structure", List[_StepSnapshot]]
        ] = [None] * total  # type: ignore[list-item]

        pool = get_pool(max_workers)
        try:
            future_to_idx = {
                pool.submit(
                    _execute_branch_worker, task
                ): idx
                for idx, task in enumerate(tasks)
            }
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                result = future.result()

                if result.error is not None:
                    raise WorkflowError(
                        f"Beam branch {idx + 1}/{total} "
                        f"failed: {result.error}"
                    )

                completed += 1
                logger.info(
                    f"  Branch {completed}/{total} completed"
                )
                self._progress.advance_branch()

                # Reconstruct Structure + snapshots
                struct = Structure(
                    pdb_string=result.pdb_string,
                    metadata=result.metadata,
                    source_path=result.source_path,
                )
                snapshots = [
                    _StepSnapshot(
                        operation=s.operation,
                        step_index=s.step_index,
                        result_metadata=s.result_metadata,
                        pdb_string=s.pdb_string,
                        structure=Structure(
                            pdb_string=s.merged_pdb_string,
                            metadata=s.merged_metadata,
                            source_path=s.merged_source_path,
                        ),
                    )
                    for s in result.snapshots
                ]
                expanded[idx] = (struct, snapshots)
        except Exception:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)

        self._progress.finish_branches()
        return expanded

    def _execute_item_with_snapshots(
        self,
        item: WorkflowStepOrBlock,
        context: _ExecutionContext,
        seed_base: Optional[int],
        step_index: int,
        snapshots: List[_StepSnapshot],
    ) -> _ExecutionContext:
        """Execute an item and capture step snapshots for deferred
        writing (used by beam blocks)."""
        if isinstance(item, WorkflowStep):
            return self._execute_step_with_snapshot(
                item, context, seed_base, step_index, snapshots
            )
        # For nested iterate/beam blocks inside beam steps,
        # execute normally without snapshots (the block-level
        # output directories are handled by the block itself).
        if isinstance(item, IterateBlock):
            return self._execute_iterate(
                item, context, seed_base, step_index
            )
        if isinstance(item, BeamBlock):
            return self._execute_beam(
                item, context, seed_base, step_index
            )
        raise WorkflowError(
            f"Unsupported node type '{type(item).__name__}'"
        )

    def _execute_step_with_snapshot(
        self,
        step: WorkflowStep,
        context: _ExecutionContext,
        seed_base: Optional[int],
        step_index: int,
        snapshots: List[_StepSnapshot],
    ) -> _ExecutionContext:
        """Execute a step and record a snapshot for deferred writing."""
        from boundry.operations import Structure

        if step.operation not in _OPERATION_REGISTRY:
            raise WorkflowError(
                f"Unknown operation '{step.operation}'"
            )

        updated: List[Structure] = []
        for idx, structure in enumerate(context.population):
            params = self._with_seed(
                step.operation,
                dict(step.params),
                seed_base,
                idx,
            )
            result = self._run_operation(
                step.operation, structure, params
            )
            merged = merge_metadata(
                structure.metadata,
                result.metadata,
                operation=step.operation,
            )
            merged_struct = Structure(
                pdb_string=result.pdb_string,
                metadata=merged,
                source_path=(
                    result.source_path or structure.source_path
                ),
            )
            updated.append(merged_struct)

            # Record snapshot with pre-merge metadata
            snapshots.append(
                _StepSnapshot(
                    operation=step.operation,
                    step_index=step_index,
                    result_metadata=dict(result.metadata),
                    pdb_string=result.pdb_string,
                    structure=merged_struct,
                )
            )

        return _ExecutionContext(
            population=updated,
            last_operation=step.operation,
            output_context=context.output_context,
        )

    # ------------------------------------------------------------------
    # Output writing
    # ------------------------------------------------------------------

    @staticmethod
    def _write_step_output(
        structure: "Structure",
        output_ctx: OutputPathContext,
        operation: str,
        result_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write operation-native artifacts to the step directory.

        Parameters
        ----------
        structure : Structure
            The structure (with merged metadata) after the operation.
        output_ctx : OutputPathContext
            The output directory context for this step.
        operation : str
            The operation name.
        result_metadata : dict, optional
            The pre-merge metadata from this operation only. If not
            provided, falls back to ``structure.metadata`` (which
            may contain accumulated data from prior steps).
        """
        spec = _OPERATION_OUTPUT_SPECS.get(operation)
        if spec is None:
            return

        dir_path = output_ctx.resolve()
        dir_path.mkdir(parents=True, exist_ok=True)

        # Use result_metadata for native writers to avoid leaking
        # accumulated metadata. Fall back to structure.metadata if
        # result_metadata is not provided.
        meta = (
            result_metadata
            if result_metadata is not None
            else structure.metadata
        )

        # Write PDB if this operation modifies the structure
        if spec.writes_pdb:
            pdb_path = dir_path / f"{spec.pdb_stem}.pdb"
            structure.write(pdb_path)
            logger.info(f"  Wrote output: {pdb_path}")

        # Write auxiliary files
        for writer_key in spec.native_writers:
            writer = _NATIVE_WRITERS.get(writer_key)
            if writer is not None:
                try:
                    writer(meta, dir_path, operation)
                except Exception as exc:
                    logger.warning(
                        f"Failed to write {writer_key}: {exc}"
                    )

    # ------------------------------------------------------------------
    # Convergence and helpers
    # ------------------------------------------------------------------

    def _check_convergence(
        self,
        condition: str,
        best_metadata: Dict[str, Any],
        previous_metadata: Optional[Dict[str, Any]],
        cycle: int,
        block_label: str,
    ) -> tuple[bool, Dict[str, Any]]:
        """Check convergence condition.

        Returns ``(converged, new_previous_metadata)``.
        """
        try:
            converged = check_condition(
                condition,
                best_metadata,
                previous_metadata,
            )
        except ConditionError as exc:
            if self._is_initial_delta_bootstrap(
                condition,
                cycle,
                previous_metadata,
            ):
                converged = False
            else:
                raise WorkflowError(
                    f"{block_label} condition failed: {exc}"
                ) from exc

        return converged, copy.deepcopy(best_metadata)

    @staticmethod
    def _with_seed(
        operation: str,
        params: Dict[str, Any],
        seed_base: Optional[int],
        candidate_offset: int,
    ) -> Dict[str, Any]:
        if seed_base is None:
            return params
        seed_param = SEED_PARAM_BY_OPERATION.get(operation)
        if seed_param is None:
            return params
        # Explicit step-level seed takes precedence over workflow seed
        if seed_param not in params:
            params[seed_param] = seed_base + candidate_offset
        return params

    @staticmethod
    def _is_initial_delta_bootstrap(
        condition: str,
        cycle: int,
        previous_metadata: Optional[Dict[str, Any]],
    ) -> bool:
        return (
            cycle == 1
            and previous_metadata is None
            and "delta(" in condition
        )

    @staticmethod
    def _describe_item(item: WorkflowStepOrBlock) -> str:
        if isinstance(item, WorkflowStep):
            return item.operation
        if isinstance(item, IterateBlock):
            if item.until is not None:
                return "iterate (convergence)"
            return f"iterate (n={item.n})"
        if isinstance(item, BeamBlock):
            return (
                f"beam (width={item.width}, "
                f"rounds={item.rounds}, "
                f"metric={item.metric})"
            )
        return type(item).__name__

    # ------------------------------------------------------------------
    # Operation runners
    # ------------------------------------------------------------------

    @staticmethod
    def _run_operation(name, structure, params):
        """Unified operation dispatch driven by
        _OPERATION_REGISTRY."""
        import boundry.config as _cfg
        import boundry.operations as _ops

        spec = _OPERATION_REGISTRY.get(name)

        # analyze_interface is special-cased
        if spec is None:
            return Workflow._run_analyze_interface(
                structure, params
            )

        if spec.needs_weights:
            from boundry.weights import ensure_weights

            ensure_weights()

        # Pop extra params before extracting config fields
        extras: Dict[str, Any] = {}
        for key in spec.extra_params:
            if key in params:
                extras[key] = params.pop(key)

        if isinstance(spec, _PipelineSpec):
            design_params = _extract_fields(
                params, _cfg.DesignConfig
            )
            relax_params = _extract_fields(
                params, _cfg.RelaxConfig
            )
            if params:
                logger.warning(
                    f"Unknown {name} params ignored: {params}"
                )
            config = _cfg.PipelineConfig(
                design=_cfg.DesignConfig(**design_params),
                relax=_cfg.RelaxConfig(**relax_params),
            )

            # Auto-link design_spec from metadata when no explicit
            # resfile or design_spec is provided in step params.
            _ds = extras.pop("design_spec", None)
            if _ds is None and "resfile" not in extras:
                _ds = getattr(
                    structure, "metadata", {}
                ).get("design_spec")

            op_fn = getattr(_ops, spec.op_name)
            return op_fn(
                structure,
                config=config,
                pre_idealize=extras.get("pre_idealize", False),
                resfile=extras.get("resfile"),
                design_spec=_ds,
                n_iterations=extras.get("n_iterations", 5),
            )

        # _SimpleSpec
        if spec.config_cls is None:
            # e.g. renumber — no config
            op_fn = getattr(_ops, spec.op_name)
            return op_fn(structure)

        config_class = getattr(_cfg, spec.config_cls)
        config_fields = _extract_fields(params, config_class)
        if params:
            logger.warning(
                f"Unknown {name} params ignored: {params}"
            )
        config = config_class(
            **spec.config_overrides, **config_fields
        )
        op_fn = getattr(_ops, spec.op_name)

        # Auto-link design_spec from metadata for ops that accept it
        if "design_spec" in spec.extra_params:
            if (
                "design_spec" not in extras
                and "resfile" not in extras
            ):
                meta_spec = getattr(
                    structure, "metadata", {}
                ).get("design_spec")
                if meta_spec is not None:
                    extras["design_spec"] = meta_spec

        kwargs: Dict[str, Any] = {"config": config}
        for key in spec.extra_params:
            if key in extras:
                kwargs[key] = extras[key]
            elif key == "pre_idealize":
                kwargs[key] = False

        return op_fn(structure, **kwargs)

    @staticmethod
    def _run_analyze_interface(structure, params):
        from boundry.config import (
            DesignConfig,
            InterfaceConfig,
            RelaxConfig,
        )
        from boundry.operations import Structure, analyze_interface

        constrained = params.pop("constrained", False)

        # Support chain_pairs as "H:A,L:A" or [["H","A"],["L","A"]]
        chain_pairs = params.pop("chain_pairs", None)
        if isinstance(chain_pairs, str):
            pairs = []
            for pair in chain_pairs.split(","):
                if ":" in pair:
                    a, b = pair.strip().split(":")
                    pairs.append((a.strip(), b.strip()))
            chain_pairs = pairs or None
        elif isinstance(chain_pairs, list):
            chain_pairs = [tuple(p) for p in chain_pairs]

        if chain_pairs is not None:
            params["chain_pairs"] = chain_pairs

        # Support scan_chains as "H,L" or ["H", "L"]
        scan_chains = params.pop("scan_chains", None)
        if isinstance(scan_chains, str):
            scan_chains = [
                c.strip()
                for c in scan_chains.split(",")
                if c.strip()
            ]
        if scan_chains is not None:
            params["scan_chains"] = scan_chains

        interface_params = _extract_fields(
            params, InterfaceConfig
        )
        if params:
            logger.warning(
                f"Unknown analyze_interface params ignored: "
                f"{params}"
            )
        config = InterfaceConfig(
            enabled=True, **interface_params
        )

        relaxer = None
        designer = None

        if config.calculate_binding_energy:
            from boundry.relaxer import Relaxer

            relaxer = Relaxer(
                RelaxConfig(constrained=constrained)
            )

        needs_designer = config.relax_separated or (
            (config.per_position or config.alanine_scan)
            and config.position_relax != "none"
        )
        if needs_designer:
            from boundry.designer import Designer
            from boundry.weights import ensure_weights

            ensure_weights()
            designer = Designer(DesignConfig())

        result = analyze_interface(
            structure,
            config=config,
            relaxer=relaxer,
            designer=designer,
        )

        if result.interface_info:
            logger.info(result.interface_info.summary)
        if result.binding_energy:
            logger.info(
                f"  ddG: "
                f"{result.binding_energy.binding_energy:.2f} "
                f"kcal/mol"
            )
        if result.sasa:
            logger.info(
                f"  Buried SASA: "
                f"{result.sasa.buried_sasa:.1f} sq. angstroms"
            )

        return result.to_structure(
            structure.pdb_string,
            source_path=structure.source_path,
            base_metadata=structure.metadata,
        )


def _compose_seed(
    seed_base: Optional[int], local_seed: int
) -> int:
    if seed_base is None:
        return local_seed
    return (seed_base * 100000) + local_seed


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _clone_structure(structure: "Structure") -> "Structure":
    from boundry.operations import Structure

    return Structure(
        pdb_string=structure.pdb_string,
        metadata=copy.deepcopy(structure.metadata),
        source_path=structure.source_path,
    )
