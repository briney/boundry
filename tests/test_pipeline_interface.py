"""Tests for pipeline integration with interface analysis."""

import pytest
from typer.testing import CliRunner

from boundry.cli import _parse_chain_pairs, app
from boundry.config import InterfaceConfig, PipelineConfig

runner = CliRunner()


class TestParseChainPairs:
    def test_single_pair(self):
        """Test parsing a single chain pair."""
        result = _parse_chain_pairs("H:A")
        assert result == [("H", "A")]

    def test_multiple_pairs(self):
        """Test parsing multiple chain pairs."""
        result = _parse_chain_pairs("H:A,L:A")
        assert result == [("H", "A"), ("L", "A")]

    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        result = _parse_chain_pairs(" H : A , L : A ")
        assert result == [("H", "A"), ("L", "A")]

    def test_empty_string(self):
        """Test parsing empty string."""
        result = _parse_chain_pairs("")
        assert result == []


class TestInterfaceConfig:
    def test_default_values(self):
        """Test default InterfaceConfig values."""
        config = InterfaceConfig()
        assert config.enabled is False
        assert config.distance_cutoff == 8.0
        assert config.chain_pairs is None
        assert config.calculate_binding_energy is True
        assert config.calculate_sasa is True
        assert config.calculate_shape_complementarity is False
        assert config.pack_separated is False
        assert config.relax_separated_chains is False
        assert config.sasa_probe_radius == 1.4

    def test_custom_values(self):
        """Test InterfaceConfig with custom values."""
        config = InterfaceConfig(
            enabled=True,
            distance_cutoff=6.0,
            chain_pairs=[("H", "A")],
            calculate_binding_energy=False,
        )
        assert config.enabled is True
        assert config.distance_cutoff == 6.0
        assert config.chain_pairs == [("H", "A")]
        assert config.calculate_binding_energy is False


class TestPipelineConfigWithInterface:
    def test_pipeline_config_has_interface(self):
        """Test that PipelineConfig includes InterfaceConfig."""
        config = PipelineConfig()
        assert hasattr(config, "interface")
        assert isinstance(config.interface, InterfaceConfig)
        assert config.interface.enabled is False

    def test_pipeline_config_with_interface_enabled(self):
        """Test PipelineConfig with interface analysis enabled."""
        config = PipelineConfig(
            interface=InterfaceConfig(
                enabled=True,
                chain_pairs=[("H", "A"), ("L", "A")],
            )
        )
        assert config.interface.enabled is True
        assert len(config.interface.chain_pairs) == 2


class TestCLIInterfaceArgs:
    """Tests for analyze-interface CLI subcommand options."""

    def test_analyze_interface_help(self):
        """Test analyze-interface --help shows interface options."""
        result = runner.invoke(app, ["analyze-interface", "--help"])
        assert result.exit_code == 0
        assert "interface" in result.output.lower()

    def test_distance_cutoff_option(self):
        """Test --distance-cutoff option is available."""
        result = runner.invoke(app, ["analyze-interface", "--help"])
        assert "--distance-cutoff" in result.output

    def test_chains_option(self):
        """Test --chains option is available."""
        result = runner.invoke(app, ["analyze-interface", "--help"])
        assert "--chains" in result.output

    def test_no_binding_energy_option(self):
        """Test --no-binding-energy option is available."""
        result = runner.invoke(app, ["analyze-interface", "--help"])
        assert "--no-binding-energy" in result.output

    def test_shape_complementarity_option(self):
        """Test --shape-complementarity option is available."""
        result = runner.invoke(app, ["analyze-interface", "--help"])
        # Rich may truncate long option names
        assert "--shape-complementa" in result.output

    def test_pack_separated_option(self):
        """Test --pack-separated option is available."""
        result = runner.invoke(app, ["analyze-interface", "--help"])
        assert "--pack-separated" in result.output

    def test_relax_separated_option(self):
        """Test --relax-separated option is available."""
        result = runner.invoke(app, ["analyze-interface", "--help"])
        assert "--relax-separated" in result.output

    def test_constrained_option(self):
        """Test --constrained option is available."""
        result = runner.invoke(app, ["analyze-interface", "--help"])
        assert "--constrained" in result.output
