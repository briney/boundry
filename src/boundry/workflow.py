"""YAML-based workflow system for Boundry.

Provides a :class:`Workflow` class that loads a YAML file describing
a linear sequence of operations and executes them in order.  Each
step's output is fed as input to the next step.
"""

import dataclasses
import logging
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from boundry.config import WorkflowConfig, WorkflowStep

logger = logging.getLogger(__name__)

VALID_OPERATIONS = frozenset(
    {
        "idealize",
        "minimize",
        "repack",
        "relax",
        "mpnn",
        "design",
        "analyze_interface",
    }
)


class WorkflowError(Exception):
    """Raised for workflow validation or execution failures."""


def _extract_fields(params: Dict[str, Any], cls: type) -> Dict[str, Any]:
    """Pop keys from *params* that match fields of dataclass *cls*."""
    fields = {f.name for f in dataclasses.fields(cls)}
    return {k: params.pop(k) for k in list(params) if k in fields}


class Workflow:
    """YAML-defined linear workflow runner.

    A workflow is a sequence of operations executed in order.  Each
    step feeds its output as input to the next step.

    Example YAML::

        input: input.pdb
        output: final.pdb
        steps:
          - operation: idealize
          - operation: minimize
            params:
              constrained: true
    """

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self._validate()

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Workflow":
        """Load a workflow from a YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise WorkflowError(
                f"Workflow YAML must be a mapping, "
                f"got {type(data).__name__}"
            )

        if "input" not in data:
            raise WorkflowError("Workflow must specify 'input' field")

        if "steps" not in data or not data["steps"]:
            raise WorkflowError(
                "Workflow must have a non-empty 'steps' list"
            )

        steps = []
        for i, step_data in enumerate(data["steps"], 1):
            if not isinstance(step_data, dict):
                raise WorkflowError(
                    f"Step {i}: expected a mapping, "
                    f"got {type(step_data).__name__}"
                )
            if "operation" not in step_data:
                raise WorkflowError(
                    f"Step {i}: missing 'operation' field"
                )
            steps.append(
                WorkflowStep(
                    operation=step_data["operation"],
                    params=step_data.get("params") or {},
                    output=step_data.get("output"),
                )
            )

        config = WorkflowConfig(
            input=data["input"],
            output=data.get("output"),
            steps=steps,
        )
        return cls(config)

    def _validate(self) -> None:
        """Validate the workflow configuration."""
        for i, step in enumerate(self.config.steps, 1):
            if step.operation not in VALID_OPERATIONS:
                raise WorkflowError(
                    f"Step {i}: unknown operation "
                    f"'{step.operation}'. "
                    f"Valid: "
                    f"{', '.join(sorted(VALID_OPERATIONS))}"
                )

    def run(self) -> "Structure":
        """Execute the workflow steps sequentially.

        Returns the final :class:`~boundry.operations.Structure`.
        """
        from boundry.operations import Structure

        input_path = Path(self.config.input)
        if not input_path.exists():
            raise WorkflowError(
                f"Input file not found: {input_path}"
            )

        current = Structure.from_file(input_path)
        logger.info(f"Loaded input: {input_path}")

        for i, step in enumerate(self.config.steps, 1):
            logger.info(
                f"Step {i}/{len(self.config.steps)}: "
                f"{step.operation}"
            )
            current = self._execute_step(step, current)

            if step.output is not None:
                current.write(step.output)
                logger.info(
                    f"  Wrote intermediate output: {step.output}"
                )

        if self.config.output is not None:
            current.write(self.config.output)
            logger.info(
                f"Wrote final output: {self.config.output}"
            )

        return current

    # ------------------------------------------------------------------
    # Step dispatching
    # ------------------------------------------------------------------

    def _execute_step(
        self, step: WorkflowStep, structure: "Structure"
    ) -> "Structure":
        """Execute a single workflow step."""
        dispatch = {
            "idealize": self._run_idealize,
            "minimize": self._run_minimize,
            "repack": self._run_repack,
            "relax": self._run_relax,
            "mpnn": self._run_mpnn,
            "design": self._run_design,
            "analyze_interface": self._run_analyze_interface,
        }
        handler = dispatch[step.operation]
        return handler(structure, dict(step.params))

    # ------------------------------------------------------------------
    # Operation runners
    # ------------------------------------------------------------------

    @staticmethod
    def _run_idealize(structure, params):
        from boundry.config import IdealizeConfig
        from boundry.operations import idealize

        config = IdealizeConfig(enabled=True, **params)
        return idealize(structure, config=config)

    @staticmethod
    def _run_minimize(structure, params):
        from boundry.config import RelaxConfig
        from boundry.operations import minimize

        pre_idealize = params.pop("pre_idealize", False)
        config = RelaxConfig(**params)
        return minimize(
            structure, config=config, pre_idealize=pre_idealize
        )

    @staticmethod
    def _run_repack(structure, params):
        from boundry.config import DesignConfig
        from boundry.operations import repack
        from boundry.weights import ensure_weights

        pre_idealize = params.pop("pre_idealize", False)
        resfile = params.pop("resfile", None)
        ensure_weights()
        config = DesignConfig(**params)
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
        ensure_weights()
        config = DesignConfig(**params)
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
    def _run_analyze_interface(structure, params):
        from boundry.config import DesignConfig, InterfaceConfig, RelaxConfig
        from boundry.operations import analyze_interface

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

        config = InterfaceConfig(enabled=True, **params)

        relaxer = None
        designer = None

        if config.calculate_binding_energy:
            from boundry.relaxer import Relaxer

            relaxer = Relaxer(RelaxConfig(constrained=constrained))

        if config.pack_separated:
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

        # Analysis doesn't modify the structure
        return structure
