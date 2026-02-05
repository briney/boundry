"""Tests for boundry.config module."""

from pathlib import Path

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


class TestDesignConfig:
    """Tests for DesignConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DesignConfig()
        assert config.model_type == "ligand_mpnn"
        assert config.temperature == 0.1
        assert config.pack_side_chains is True
        assert config.seed is None
        assert config.use_ligand_context is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DesignConfig(
            model_type="protein_mpnn",
            temperature=0.5,
            pack_side_chains=False,
            seed=42,
            use_ligand_context=True,
        )
        assert config.model_type == "protein_mpnn"
        assert config.temperature == 0.5
        assert config.pack_side_chains is False
        assert config.seed == 42
        assert config.use_ligand_context is True

    def test_model_type_options(self):
        """Test valid model type options."""
        for model_type in ["protein_mpnn", "ligand_mpnn", "soluble_mpnn"]:
            config = DesignConfig(model_type=model_type)
            assert config.model_type == model_type


class TestRelaxConfig:
    """Tests for RelaxConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RelaxConfig()
        assert config.max_iterations == 0
        assert config.tolerance == 2.39
        assert config.stiffness == 10.0
        assert config.max_outer_iterations == 3
        assert config.implicit_solvent is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RelaxConfig(
            max_iterations=1000,
            tolerance=1.0,
            stiffness=5.0,
            max_outer_iterations=5,
            implicit_solvent=False,
        )
        assert config.max_iterations == 1000
        assert config.tolerance == 1.0
        assert config.stiffness == 5.0
        assert config.max_outer_iterations == 5
        assert config.implicit_solvent is False


class TestIdealizeConfig:
    """Tests for IdealizeConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = IdealizeConfig()
        assert config.enabled is False
        assert config.fix_cis_omega is True
        assert config.post_idealize_stiffness == 10.0
        assert config.add_missing_residues is True
        assert config.close_chainbreaks is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = IdealizeConfig(
            enabled=True,
            fix_cis_omega=False,
            post_idealize_stiffness=5.0,
            add_missing_residues=False,
            close_chainbreaks=False,
        )
        assert config.enabled is True
        assert config.fix_cis_omega is False
        assert config.post_idealize_stiffness == 5.0
        assert config.add_missing_residues is False
        assert config.close_chainbreaks is False


class TestInterfaceConfig:
    """Tests for InterfaceConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = InterfaceConfig()
        assert config.enabled is False
        assert config.distance_cutoff == 8.0
        assert config.chain_pairs is None
        assert config.calculate_binding_energy is True
        assert config.calculate_sasa is False
        assert config.calculate_shape_complementarity is False
        assert config.relax_separated is False
        assert config.sasa_probe_radius == 1.4
        assert config.position_relax == "none"
        assert config.show_progress is False
        assert config.quiet is False

    def test_custom_chain_pairs(self):
        """Test setting chain pairs."""
        config = InterfaceConfig(
            enabled=True,
            chain_pairs=[("H", "A"), ("L", "A")],
        )
        assert config.enabled is True
        assert config.chain_pairs == [("H", "A"), ("L", "A")]

    def test_show_progress_and_quiet(self):
        """Test setting show_progress and quiet flags."""
        config = InterfaceConfig(
            show_progress=True,
            quiet=True,
        )
        assert config.show_progress is True
        assert config.quiet is True


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PipelineConfig()
        assert config.n_iterations == 5
        assert config.n_outputs == 1
        assert config.scorefile is None
        assert config.verbose is False
        assert config.remove_waters is True
        assert config.show_progress is False
        assert isinstance(config.design, DesignConfig)
        assert isinstance(config.relax, RelaxConfig)
        assert isinstance(config.idealize, IdealizeConfig)
        assert isinstance(config.interface, InterfaceConfig)

    def test_no_mode_field(self):
        """Test that PipelineConfig no longer has a mode field."""
        config = PipelineConfig()
        assert not hasattr(config, "mode")

    def test_with_scorefile(self):
        """Test setting scorefile path."""
        config = PipelineConfig(scorefile=Path("/tmp/scores.sc"))
        assert config.scorefile == Path("/tmp/scores.sc")

    def test_nested_configs(self):
        """Test nested config objects."""
        config = PipelineConfig(
            design=DesignConfig(temperature=0.5),
            relax=RelaxConfig(stiffness=20.0),
        )
        assert config.design.temperature == 0.5
        assert config.relax.stiffness == 20.0

    def test_verbose_flag(self):
        """Test verbose flag setting."""
        config = PipelineConfig(verbose=True)
        assert config.verbose is True

    def test_custom_iterations(self):
        """Test setting custom iteration count."""
        config = PipelineConfig(n_iterations=10, n_outputs=5)
        assert config.n_iterations == 10
        assert config.n_outputs == 5

    def test_show_progress_flag(self):
        """Test show_progress flag setting."""
        config = PipelineConfig(show_progress=True)
        assert config.show_progress is True


class TestWorkflowStep:
    """Tests for WorkflowStep dataclass."""

    def test_minimal_step(self):
        """Test creating a step with only required fields."""
        step = WorkflowStep(operation="idealize")
        assert step.operation == "idealize"
        assert step.params == {}
        assert step.output is None

    def test_step_with_params(self):
        """Test creating a step with parameters."""
        step = WorkflowStep(
            operation="minimize",
            params={"constrained": False, "max_iterations": 1000},
        )
        assert step.operation == "minimize"
        assert step.params == {"constrained": False, "max_iterations": 1000}

    def test_step_with_output(self):
        """Test creating a step with intermediate output path."""
        step = WorkflowStep(
            operation="idealize",
            params={"fix_cis_omega": True},
            output="idealized.pdb",
        )
        assert step.output == "idealized.pdb"


class TestWorkflowConfig:
    """Tests for WorkflowConfig dataclass."""

    def test_minimal_config(self):
        """Test creating a workflow config with only input."""
        config = WorkflowConfig(input="input.pdb")
        assert config.input == "input.pdb"
        assert config.output is None
        assert config.seed is None
        assert config.workflow_version == 1
        assert config.steps == []

    def test_full_config(self):
        """Test creating a workflow config with all fields."""
        steps = [
            WorkflowStep(
                operation="idealize",
                params={"fix_cis_omega": True},
                output="idealized.pdb",
            ),
            WorkflowStep(
                operation="minimize",
                params={"constrained": False},
            ),
        ]
        config = WorkflowConfig(
            input="input.pdb",
            output="final.pdb",
            steps=steps,
        )
        assert config.input == "input.pdb"
        assert config.output == "final.pdb"
        assert len(config.steps) == 2
        assert config.steps[0].operation == "idealize"
        assert config.steps[1].operation == "minimize"

    def test_default_steps_are_independent(self):
        """Test that default steps list is not shared between instances."""
        config1 = WorkflowConfig(input="a.pdb")
        config2 = WorkflowConfig(input="b.pdb")
        config1.steps.append(WorkflowStep(operation="idealize"))
        assert len(config2.steps) == 0


class TestIterateBlock:
    """Tests for IterateBlock dataclass."""

    def test_defaults(self):
        block = IterateBlock(steps=[WorkflowStep(operation="idealize")])
        assert block.n == 1
        assert block.max_n == 100
        assert block.until is None
        assert block.output is None


class TestBeamBlock:
    """Tests for BeamBlock dataclass."""

    def test_defaults(self):
        block = BeamBlock(steps=[WorkflowStep(operation="relax")])
        assert block.width == 5
        assert block.rounds == 10
        assert block.metric == "dG"
        assert block.direction == "min"
        assert block.until is None
        assert block.expand == 1
        assert block.output is None
