"""YAML-based workflow system for Boundry."""

from __future__ import annotations

import copy
import dataclasses
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

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

VALID_OPERATIONS = frozenset(
    {
        "idealize",
        "minimize",
        "repack",
        "relax",
        "mpnn",
        "design",
        "renumber",
        "analyze_interface",
    }
)

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


_STRUCTURE_EXTENSIONS = frozenset({".pdb", ".cif", ".mmcif"})

_KNOWN_KEYS = frozenset(
    {"workflow_version", "input", "output", "seed", "steps"}
)


def _is_directory_output(path_template: str) -> bool:
    """A path is a directory if it ends with '/' or has no recognized file extension."""
    if path_template.endswith("/"):
        return True
    stripped = path_template.replace("{", "").replace("}", "")
    return Path(stripped).suffix.lower() not in _STRUCTURE_EXTENSIONS


_DEFAULT_STEM: Dict[str, str] = {
    "idealize": "idealized",
    "minimize": "minimized",
    "repack": "repacked",
    "relax": "relaxed",
    "mpnn": "designed_mpnn",
    "design": "designed",
    "renumber": "renumbered",
    "analyze_interface": "interface_analysis",
}


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
def _parse_operation(step_data: Dict[str, Any], label: str) -> WorkflowStep:
    _validate_unknown_keys(
        step_data,
        {"operation", "params", "output"},
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

    output = _parse_optional_str(
        step_data.get("output"),
        label,
        "output",
    )
    return WorkflowStep(operation=operation, params=params, output=output)


@_register_parser("iterate")
def _parse_iterate(step_data: Dict[str, Any], label: str) -> IterateBlock:
    _validate_unknown_keys(step_data, {"iterate"}, label)
    block_data = step_data["iterate"]
    if not isinstance(block_data, dict):
        raise WorkflowError(
            f"{label}.iterate: expected mapping, "
            f"got {type(block_data).__name__}"
        )

    _validate_unknown_keys(
        block_data,
        {"steps", "n", "max_n", "until", "output"},
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
    output = _parse_optional_str(
        block_data.get("output"),
        f"{label}.iterate",
        "output",
    )

    if until is None and n < 1:
        raise WorkflowError(f"{label}.iterate: n must be >= 1")
    if until is not None:
        parse_condition(until)
        if max_n < 1:
            raise WorkflowError(f"{label}.iterate: max_n must be >= 1")

    return IterateBlock(
        steps=steps,
        n=n,
        max_n=max_n,
        until=until,
        output=output,
    )


@_register_parser("beam")
def _parse_beam(step_data: Dict[str, Any], label: str) -> BeamBlock:
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
            "output",
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
        raise WorkflowError(f"{label}.beam: metric must not be null")
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
    output = _parse_optional_str(
        block_data.get("output"),
        f"{label}.beam",
        "output",
    )

    return BeamBlock(
        steps=steps,
        width=width,
        rounds=rounds,
        metric=metric,
        direction=direction,
        until=until,
        expand=expand,
        output=output,
    )


def _ensure_resolvers() -> None:
    """Register custom OmegaConf resolvers (idempotent)."""
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver(
            "env",
            lambda key, default="": os.environ.get(key, default),
        )


@dataclass
class _ExecutionContext:
    population: List["Structure"]


class Workflow:
    """YAML-defined workflow runner with composable compound blocks."""

    _VALIDATE_DISPATCH = {
        WorkflowStep: "_validate_step",
        IterateBlock: "_validate_iterate",
        BeamBlock: "_validate_beam",
    }
    _EXECUTE_DISPATCH = {
        WorkflowStep: "_execute_step",
        IterateBlock: "_execute_iterate",
        BeamBlock: "_execute_beam",
    }

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.last_population: List["Structure"] = []
        self._validate()

    @classmethod
    def from_yaml(
        cls,
        path: Union[str, Path],
        seed: Optional[int] = None,
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
        overrides : list of str, optional
            Dotlist-style overrides (e.g. ``["output=results/",
            "seed=42"]``).  Applied after YAML loading but before
            variable resolution.
        """
        from omegaconf import OmegaConf, DictConfig
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
            raise WorkflowError("Workflow must specify 'input' field")
        if "steps" not in data:
            raise WorkflowError("Workflow must specify 'steps' field")

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
            raise WorkflowError("Workflow 'input' must be a string")

        output_path = _parse_optional_str(
            data.get("output"), "Workflow", "output"
        )
        version = data.get("workflow_version", 1)
        if isinstance(version, bool) or not isinstance(version, int):
            raise WorkflowError("workflow_version must be an integer")

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

        steps = _parse_steps(data["steps"])
        config = WorkflowConfig(
            input=input_path,
            output=output_path,
            seed=effective_seed,
            workflow_version=version,
            steps=steps,
            vars=user_vars,
        )
        return cls(config)

    def _validate(self) -> None:
        """Validate workflow configuration recursively."""
        if self.config.workflow_version != 1:
            raise WorkflowError(
                f"Unsupported workflow_version={self.config.workflow_version}. "
                "Only version 1 is supported."
            )
        if not self.config.steps:
            raise WorkflowError("Workflow must contain at least one step")
        self._validate_steps(self.config.steps, prefix="")

    def _validate_steps(
        self, steps: List[WorkflowStepOrBlock], prefix: str
    ) -> None:
        for i, item in enumerate(steps, 1):
            label = f"{prefix}Step {i}"
            method_name = self._VALIDATE_DISPATCH.get(type(item))
            if method_name is None:
                raise WorkflowError(
                    f"{label}: unsupported node type "
                    f"'{type(item).__name__}'"
                )
            getattr(self, method_name)(item, label)

    def _validate_step(self, step: WorkflowStep, label: str) -> None:
        if step.operation not in VALID_OPERATIONS:
            raise WorkflowError(
                f"{label}: unknown operation '{step.operation}'. "
                f"Valid: {', '.join(sorted(VALID_OPERATIONS))}"
            )
        if not isinstance(step.params, dict):
            raise WorkflowError(f"{label}: params must be a mapping")

    def _validate_iterate(self, block: IterateBlock, label: str) -> None:
        if block.until is None and block.n < 1:
            raise WorkflowError(f"{label}: iterate n must be >= 1")
        if block.until is not None:
            parse_condition(block.until)
            if block.max_n < 1:
                raise WorkflowError(f"{label}: iterate max_n must be >= 1")
        self._validate_steps(block.steps, prefix=f"{label}.iterate.")

    def _validate_beam(self, block: BeamBlock, label: str) -> None:
        if block.width < 1 or block.rounds < 1 or block.expand < 1:
            raise WorkflowError(
                f"{label}: beam width/rounds/expand must be >= 1"
            )
        if block.direction not in {"min", "max"}:
            raise WorkflowError(
                f"{label}: beam direction must be 'min' or 'max'"
            )
        if block.until is not None:
            parse_condition(block.until)
        self._validate_steps(block.steps, prefix=f"{label}.beam.")

    def run(self) -> "Structure":
        """Execute workflow and return the best final structure."""
        population = self.run_population()
        return population[0]

    def run_population(self) -> List["Structure"]:
        """Execute workflow and return the full final candidate population."""
        from boundry.operations import Structure

        input_path = Path(self.config.input)
        if not input_path.exists():
            raise WorkflowError(f"Input file not found: {input_path}")

        current = _ExecutionContext(population=[Structure.from_file(input_path)])
        logger.info(f"Loaded input: {input_path}")

        total = len(self.config.steps)
        for i, item in enumerate(self.config.steps, 1):
            logger.info(f"Step {i}/{total}: {self._describe_item(item)}")
            current = self._execute_item(item, current, seed_base=self.config.seed)
            if isinstance(item, WorkflowStep) and item.output is not None:
                self._write_population(
                    current.population, item.output, operation=item.operation
                )

        if self.config.output is not None:
            last_op = None
            if self.config.steps:
                last = self.config.steps[-1]
                if isinstance(last, WorkflowStep):
                    last_op = last.operation
            self._write_population(
                current.population, self.config.output, operation=last_op
            )

        self.last_population = list(current.population)
        return list(current.population)

    def _execute_item(
        self,
        item: WorkflowStepOrBlock,
        context: _ExecutionContext,
        seed_base: Optional[int],
    ) -> _ExecutionContext:
        method_name = self._EXECUTE_DISPATCH.get(type(item))
        if method_name is None:
            raise WorkflowError(
                f"Unsupported node type '{type(item).__name__}'"
            )
        handler = getattr(self, method_name)
        return handler(item, context, seed_base)

    def _execute_step(
        self,
        step: WorkflowStep,
        context: _ExecutionContext,
        seed_base: Optional[int],
    ) -> _ExecutionContext:
        from boundry.operations import Structure

        dispatch = {
            "idealize": self._run_idealize,
            "minimize": self._run_minimize,
            "repack": self._run_repack,
            "relax": self._run_relax,
            "mpnn": self._run_mpnn,
            "design": self._run_design,
            "renumber": self._run_renumber,
            "analyze_interface": self._run_analyze_interface,
        }
        handler = dispatch.get(step.operation)
        if handler is None:
            raise WorkflowError(f"Unknown operation '{step.operation}'")

        updated: List[Structure] = []
        for idx, structure in enumerate(context.population):
            params = self._with_seed(
                step.operation,
                dict(step.params),
                seed_base,
                idx,
            )
            result = handler(structure, params)
            merged = merge_metadata(
                structure.metadata,
                result.metadata,
                operation=step.operation,
            )
            updated.append(
                Structure(
                    pdb_string=result.pdb_string,
                    metadata=merged,
                    source_path=result.source_path or structure.source_path,
                )
            )
        return _ExecutionContext(population=updated)

    def _execute_iterate(
        self,
        block: IterateBlock,
        context: _ExecutionContext,
        seed_base: Optional[int],
    ) -> _ExecutionContext:
        current = context
        previous_metadata: Optional[Dict[str, Any]] = None
        max_iters = block.max_n if block.until is not None else block.n
        converged = False

        for cycle in range(1, max_iters + 1):
            cycle_seed = _compose_seed(seed_base, cycle)
            inner_seed = cycle_seed if seed_base is not None else None
            for inner in block.steps:
                current = self._execute_item(
                    inner,
                    current,
                    inner_seed,
                )

            if block.output is not None:
                last_op = None
                if block.steps and isinstance(block.steps[-1], WorkflowStep):
                    last_op = block.steps[-1].operation
                self._write_population(
                    current.population,
                    block.output,
                    operation=last_op,
                    cycle=cycle,
                )

            if block.until is not None:
                best = current.population[0]
                try:
                    converged = check_condition(
                        block.until,
                        best.metadata,
                        previous_metadata,
                    )
                except ConditionError as exc:
                    if self._is_initial_delta_bootstrap(
                        block.until,
                        cycle,
                        previous_metadata,
                    ):
                        converged = False
                    else:
                        raise WorkflowError(
                            f"Iterate condition failed: {exc}"
                        ) from exc

                previous_metadata = copy.deepcopy(best.metadata)
                if converged:
                    logger.info(
                        f"Iterate block converged at cycle {cycle}"
                    )
                    break

        if block.until is not None and not converged:
            logger.warning(
                f"Iterate block reached max_n={block.max_n} "
                "without meeting convergence condition"
            )

        return current

    def _execute_beam(
        self,
        block: BeamBlock,
        context: _ExecutionContext,
        seed_base: Optional[int],
    ) -> _ExecutionContext:
        population = list(context.population)
        previous_best_metadata: Optional[Dict[str, Any]] = None

        for round_num in range(1, block.rounds + 1):
            if not population:
                raise WorkflowError("Beam population is empty")

            expand_per = max(
                block.expand,
                _ceil_div(block.width, len(population)),
            )
            expanded: List["Structure"] = []

            for cand_idx, candidate in enumerate(population, 1):
                for exp_idx in range(expand_per):
                    branch_seed = _compose_seed(
                        seed_base,
                        (round_num * 10000) + (cand_idx * 100) + exp_idx,
                    )
                    branch = _ExecutionContext(
                        population=[_clone_structure(candidate)]
                    )
                    for inner in block.steps:
                        branch = self._execute_item(
                            inner,
                            branch,
                            branch_seed,
                        )
                    expanded.extend(branch.population)

            scored: List[tuple[float, "Structure"]] = []
            for candidate in expanded:
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
                scored.append((metric_value, candidate))

            if not scored:
                raise WorkflowError(
                    "Beam search could not score any candidates. "
                    f"Missing metric '{block.metric}'."
                )

            scored.sort(
                key=lambda item: item[0],
                reverse=(block.direction == "max"),
            )
            population = [cand for _, cand in scored[: block.width]]
            best_metric = scored[0][0]
            logger.info(
                f"Beam round {round_num}/{block.rounds}: "
                f"best {block.metric}={best_metric:.4f}"
            )

            if block.output is not None:
                last_op = None
                if block.steps and isinstance(block.steps[-1], WorkflowStep):
                    last_op = block.steps[-1].operation
                self._write_population(
                    population,
                    block.output,
                    operation=last_op,
                    round=round_num,
                )

            if block.until is not None:
                best = population[0]
                try:
                    done = check_condition(
                        block.until,
                        best.metadata,
                        previous_best_metadata,
                    )
                except ConditionError as exc:
                    if self._is_initial_delta_bootstrap(
                        block.until,
                        round_num,
                        previous_best_metadata,
                    ):
                        done = False
                    else:
                        raise WorkflowError(
                            f"Beam condition failed: {exc}"
                        ) from exc

                previous_best_metadata = copy.deepcopy(best.metadata)
                if done:
                    logger.info(
                        f"Beam block converged at round {round_num}"
                    )
                    break

        return _ExecutionContext(population=population)

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
                f"beam (width={item.width}, rounds={item.rounds}, "
                f"metric={item.metric})"
            )
        return type(item).__name__

    def _write_population(
        self,
        population: List["Structure"],
        path_template: str,
        operation: Optional[str] = None,
        **tokens: int,
    ) -> None:
        if not population:
            return
        if _is_directory_output(path_template):
            return self._write_population_to_dir(
                population, path_template, operation, **tokens
            )

        if len(population) == 1:
            path = self._render_path(path_template, rank=1, **tokens)
            population[0].write(path)
            logger.info(f"  Wrote output: {path}")
            return

        has_rank_placeholder = "{rank" in path_template
        for rank, structure in enumerate(population, 1):
            if has_rank_placeholder:
                path = self._render_path(
                    path_template,
                    rank=rank,
                    **tokens,
                )
            else:
                base = self._render_path(path_template, **tokens)
                path = _add_rank_suffix(base, rank)
            structure.write(path)
            logger.info(f"  Wrote output rank {rank}: {path}")

    def _write_population_to_dir(
        self,
        population: List["Structure"],
        dir_template: str,
        operation: Optional[str],
        **tokens: int,
    ) -> None:
        dir_path = Path(self._render_path(dir_template, **tokens))
        dir_path.mkdir(parents=True, exist_ok=True)

        stem = _DEFAULT_STEM.get(operation, "output") if operation else "output"

        # Build token suffix: "cycle_3", "round_2", etc.
        token_parts = [f"{k}_{v}" for k, v in sorted(tokens.items())]
        suffix = "_".join(token_parts)

        for rank, structure in enumerate(population, 1):
            parts = [stem]
            if suffix:
                parts.append(suffix)
            if len(population) > 1:
                parts.append(f"rank_{rank}")
            filename = "_".join(parts)

            # Write structure PDB
            pdb_path = dir_path / f"{filename}.pdb"
            structure.write(pdb_path)
            logger.info(f"  Wrote output: {pdb_path}")

            # Write auxiliary files based on metadata
            self._write_auxiliary_files(structure, dir_path, filename, operation)

    @staticmethod
    def _write_auxiliary_files(structure, dir_path, filename, operation):
        """Auto-write all available auxiliary outputs to the directory."""
        import json

        meta = structure.metadata

        # Per-position CSV (from analyze_interface)
        per_position = meta.get("per_position")
        if per_position is not None:
            from boundry.interface_position_energetics import write_position_csv

            csv_path = dir_path / f"{filename}_positions.csv"
            write_position_csv(per_position, csv_path)
            logger.info(f"  Wrote per-position CSV: {csv_path}")

        # Metrics JSON — write workflow metrics for any operation that produces them
        wf = meta.get("_workflow", {})
        metrics = wf.get("metrics")
        if metrics:
            json_path = dir_path / f"{filename}_metrics.json"
            with open(json_path, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            logger.info(f"  Wrote metrics: {json_path}")

        # Energy breakdown (from relax, design, minimize)
        energy = meta.get("energy_breakdown")
        if energy:
            json_path = dir_path / f"{filename}_energy.json"
            with open(json_path, "w") as f:
                json.dump(energy, f, indent=2, default=str)
            logger.info(f"  Wrote energy breakdown: {json_path}")

        # Interface-specific metrics
        interface_metrics = meta.get("metrics", {}).get("interface")
        if interface_metrics:
            json_path = dir_path / f"{filename}_interface.json"
            with open(json_path, "w") as f:
                json.dump(interface_metrics, f, indent=2, default=str)
            logger.info(f"  Wrote interface metrics: {json_path}")

    @staticmethod
    def _render_path(path_template: str, **tokens: int) -> str:
        try:
            return path_template.format(**tokens)
        except KeyError as exc:
            missing = exc.args[0]
            raise WorkflowError(
                f"Output path template '{path_template}' is missing "
                f"format token '{missing}'"
            ) from exc
        except ValueError as exc:
            raise WorkflowError(
                f"Invalid output template '{path_template}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Operation runners
    # ------------------------------------------------------------------

    @staticmethod
    def _run_idealize(structure, params):
        from boundry.config import IdealizeConfig
        from boundry.operations import idealize

        idealize_params = _extract_fields(params, IdealizeConfig)
        if params:
            logger.warning(f"Unknown idealize params ignored: {params}")
        config = IdealizeConfig(enabled=True, **idealize_params)
        return idealize(structure, config=config)

    @staticmethod
    def _run_minimize(structure, params):
        from boundry.config import RelaxConfig
        from boundry.operations import minimize

        pre_idealize = params.pop("pre_idealize", False)
        relax_params = _extract_fields(params, RelaxConfig)
        if params:
            logger.warning(f"Unknown minimize params ignored: {params}")
        config = RelaxConfig(**relax_params)
        return minimize(
            structure,
            config=config,
            pre_idealize=pre_idealize,
        )

    @staticmethod
    def _run_repack(structure, params):
        from boundry.config import DesignConfig
        from boundry.operations import repack
        from boundry.weights import ensure_weights

        pre_idealize = params.pop("pre_idealize", False)
        resfile = params.pop("resfile", None)
        design_params = _extract_fields(params, DesignConfig)
        if params:
            logger.warning(f"Unknown repack params ignored: {params}")
        ensure_weights()
        config = DesignConfig(**design_params)
        return repack(
            structure,
            config=config,
            resfile=resfile,
            pre_idealize=pre_idealize,
        )

    @staticmethod
    def _run_relax(structure, params):
        from boundry.config import DesignConfig, PipelineConfig, RelaxConfig
        from boundry.operations import relax
        from boundry.weights import ensure_weights

        pre_idealize = params.pop("pre_idealize", False)
        resfile = params.pop("resfile", None)
        n_iterations = params.pop("n_iterations", 5)

        design_params = _extract_fields(params, DesignConfig)
        relax_params = _extract_fields(params, RelaxConfig)

        if params:
            logger.warning(f"Unknown relax params ignored: {params}")

        ensure_weights()
        config = PipelineConfig(
            design=DesignConfig(**design_params),
            relax=RelaxConfig(**relax_params),
        )
        return relax(
            structure,
            config=config,
            resfile=resfile,
            pre_idealize=pre_idealize,
            n_iterations=n_iterations,
        )

    @staticmethod
    def _run_mpnn(structure, params):
        from boundry.config import DesignConfig
        from boundry.operations import mpnn
        from boundry.weights import ensure_weights

        pre_idealize = params.pop("pre_idealize", False)
        resfile = params.pop("resfile", None)
        design_params = _extract_fields(params, DesignConfig)
        if params:
            logger.warning(f"Unknown mpnn params ignored: {params}")
        ensure_weights()
        config = DesignConfig(**design_params)
        return mpnn(
            structure,
            config=config,
            resfile=resfile,
            pre_idealize=pre_idealize,
        )

    @staticmethod
    def _run_design(structure, params):
        from boundry.config import DesignConfig, PipelineConfig, RelaxConfig
        from boundry.operations import design
        from boundry.weights import ensure_weights

        pre_idealize = params.pop("pre_idealize", False)
        resfile = params.pop("resfile", None)
        n_iterations = params.pop("n_iterations", 5)

        design_params = _extract_fields(params, DesignConfig)
        relax_params = _extract_fields(params, RelaxConfig)

        if params:
            logger.warning(f"Unknown design params ignored: {params}")

        ensure_weights()
        config = PipelineConfig(
            design=DesignConfig(**design_params),
            relax=RelaxConfig(**relax_params),
        )
        return design(
            structure,
            config=config,
            resfile=resfile,
            pre_idealize=pre_idealize,
            n_iterations=n_iterations,
        )

    @staticmethod
    def _run_renumber(structure, params):
        from boundry.operations import renumber

        return renumber(structure)

    @staticmethod
    def _run_analyze_interface(structure, params):
        from boundry.config import DesignConfig, InterfaceConfig, RelaxConfig
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

        interface_params = _extract_fields(params, InterfaceConfig)
        if params:
            logger.warning(f"Unknown analyze_interface params ignored: {params}")
        config = InterfaceConfig(enabled=True, **interface_params)

        relaxer = None
        designer = None

        if config.calculate_binding_energy:
            from boundry.relaxer import Relaxer

            relaxer = Relaxer(RelaxConfig(constrained=constrained))

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

        analysis_metadata: Dict[str, Any] = {
            "operation": "analyze_interface",
            "metrics": {"interface": {}},
        }

        if result.binding_energy is not None:
            analysis_metadata["dG"] = result.binding_energy.binding_energy
            analysis_metadata["complex_energy"] = (
                result.binding_energy.complex_energy
            )
            analysis_metadata["metrics"]["interface"]["dG"] = (
                result.binding_energy.binding_energy
            )
            analysis_metadata["metrics"]["interface"]["complex_energy"] = (
                result.binding_energy.complex_energy
            )

        if result.sasa is not None:
            analysis_metadata["buried_sasa"] = result.sasa.buried_sasa
            analysis_metadata["metrics"]["interface"]["buried_sasa"] = (
                result.sasa.buried_sasa
            )

        if result.shape_complementarity is not None:
            analysis_metadata["sc_score"] = (
                result.shape_complementarity.sc_score
            )
            analysis_metadata["metrics"]["interface"]["sc_score"] = (
                result.shape_complementarity.sc_score
            )

        if result.interface_info is not None:
            analysis_metadata["n_interface_residues"] = (
                result.interface_info.n_interface_residues
            )
            analysis_metadata["metrics"]["interface"][
                "n_interface_residues"
            ] = result.interface_info.n_interface_residues

        if result.per_position is not None:
            analysis_metadata["per_position"] = result.per_position

        merged = dict(structure.metadata)
        merged.update(analysis_metadata)
        return Structure(
            pdb_string=structure.pdb_string,
            metadata=merged,
            source_path=structure.source_path,
        )


def _compose_seed(seed_base: Optional[int], local_seed: int) -> int:
    if seed_base is None:
        return local_seed
    return (seed_base * 100000) + local_seed


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _add_rank_suffix(path: str, rank: int) -> str:
    p = Path(path)
    if p.suffix:
        return str(p.with_name(f"{p.stem}_rank{rank}{p.suffix}"))
    return f"{path}_rank{rank}"


def _clone_structure(structure: "Structure") -> "Structure":
    from boundry.operations import Structure

    return Structure(
        pdb_string=structure.pdb_string,
        metadata=copy.deepcopy(structure.metadata),
        source_path=structure.source_path,
    )
