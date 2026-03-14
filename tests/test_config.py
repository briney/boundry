"""Tests for boundry.config module."""

from pathlib import Path

from boundry.config import (
    DdGConfig,
    DesignConfig,
    IdealizeConfig,
    InterfaceConfig,
    PipelineConfig,
    RelaxConfig,
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


class TestDdGConfig:
    """Tests for DdGConfig dataclass."""

    def test_default_values(self):
        """Test all default configuration values."""
        config = DdGConfig()
        assert config.n_ensemble == 35
        assert config.md_total_steps == 50000
        assert config.md_equilibration_steps == 5000
        assert config.md_temperature == 300.0
        assert config.md_friction == 1.0
        assert config.neighborhood_sampling_bias == 1.0
        assert config.ca_cutoff == 9.0
        assert config.restraint_sd == 0.5
        assert config.neighborhood_radius == 8.0
        assert config.sequence_window == 1
        assert config.chain_pairs is None
        assert config.separation_distance == 100.0
        assert config.implicit_solvent is True
        assert config.workers == 1
        assert config.seed is None
        assert config.quiet is True
        assert config.cache_ensemble is False
        assert config.ensemble_dir is None
        assert config.sort_members_by_wt_bound_energy is False
        assert config.average_top_n is None
        assert config.paper_mode is False
        assert config.relax_separated is True
        assert config.relax_separated_iterations == 1

    def test_custom_values(self):
        """Test constructor with overrides."""
        config = DdGConfig(
            n_ensemble=50,
            md_total_steps=100000,
            ca_cutoff=12.0,
            chain_pairs=[("H", "A")],
            workers=4,
            seed=42,
            quiet=False,
        )
        assert config.n_ensemble == 50
        assert config.md_total_steps == 100000
        assert config.ca_cutoff == 12.0
        assert config.chain_pairs == [("H", "A")]
        assert config.workers == 4
        assert config.seed == 42
        assert config.quiet is False

    def test_paper_mode_overrides(self):
        """paper_mode=True sets n_ensemble=50, md_total_steps=100000."""
        config = DdGConfig(paper_mode=True)
        assert config.n_ensemble == 50
        assert config.md_total_steps == 100000

    def test_paper_mode_preserves_explicit(self):
        """paper_mode=True does not override explicitly set values."""
        config = DdGConfig(
            paper_mode=True, n_ensemble=20, md_total_steps=75000
        )
        assert config.n_ensemble == 20
        assert config.md_total_steps == 75000

    def test_ensemble_dir_accepts_path(self):
        """ensemble_dir accepts a Path object."""
        config = DdGConfig(ensemble_dir=Path("/tmp/ensembles"))
        assert config.ensemble_dir == Path("/tmp/ensembles")


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
