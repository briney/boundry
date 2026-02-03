"""Tests for boundry.config module."""

from pathlib import Path

from boundry.config import (
    DesignConfig,
    PipelineConfig,
    PipelineMode,
    RelaxConfig,
)


class TestPipelineMode:
    """Tests for PipelineMode enum."""

    def test_all_modes_exist(self):
        """Test that all expected modes are defined."""
        assert PipelineMode.RELAX.value == "relax"
        assert PipelineMode.REPACK_ONLY.value == "repack_only"
        assert PipelineMode.NO_REPACK.value == "no_repack"
        assert PipelineMode.DESIGN.value == "design"
        assert PipelineMode.DESIGN_ONLY.value == "design_only"

    def test_mode_count(self):
        """Test that we have exactly 5 modes."""
        assert len(PipelineMode) == 5


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


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PipelineConfig()
        assert config.mode == PipelineMode.RELAX
        assert config.n_iterations == 5
        assert config.n_outputs == 1
        assert config.scorefile is None
        assert config.verbose is False
        assert isinstance(config.design, DesignConfig)
        assert isinstance(config.relax, RelaxConfig)

    def test_custom_mode(self):
        """Test setting different modes."""
        for mode in PipelineMode:
            config = PipelineConfig(mode=mode)
            assert config.mode == mode

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
