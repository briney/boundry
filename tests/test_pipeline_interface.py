"""Tests for pipeline integration with interface analysis."""

import pytest

from boundry.cli import _parse_chain_pairs, create_parser
from boundry.config import InterfaceConfig, PipelineConfig


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
    def test_analyze_interface_flag(self):
        """Test --analyze-interface flag is parsed."""
        parser = create_parser()
        args = parser.parse_args(
            ["-i", "input.pdb", "-o", "output.pdb", "--analyze-interface"]
        )
        assert args.analyze_interface is True

    def test_interface_distance_cutoff(self):
        """Test --interface-distance-cutoff argument."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "-i",
                "input.pdb",
                "-o",
                "output.pdb",
                "--analyze-interface",
                "--interface-distance-cutoff",
                "6.0",
            ]
        )
        assert args.interface_distance_cutoff == 6.0

    def test_interface_chains(self):
        """Test --interface-chains argument."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "-i",
                "input.pdb",
                "-o",
                "output.pdb",
                "--analyze-interface",
                "--interface-chains",
                "H:A,L:A",
            ]
        )
        assert args.interface_chains == "H:A,L:A"

    def test_no_binding_energy(self):
        """Test --no-binding-energy flag."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "-i",
                "input.pdb",
                "-o",
                "output.pdb",
                "--analyze-interface",
                "--no-binding-energy",
            ]
        )
        assert args.no_binding_energy is True

    def test_calculate_shape_complementarity(self):
        """Test --calculate-shape-complementarity flag."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "-i",
                "input.pdb",
                "-o",
                "output.pdb",
                "--analyze-interface",
                "--calculate-shape-complementarity",
            ]
        )
        assert args.calculate_shape_complementarity is True

    def test_pack_separated_flag(self):
        """Test --pack-separated flag."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "-i",
                "input.pdb",
                "-o",
                "output.pdb",
                "--analyze-interface",
                "--pack-separated",
            ]
        )
        assert args.pack_separated is True

    def test_relax_separated_flag(self):
        """Test --relax-separated flag."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "-i",
                "input.pdb",
                "-o",
                "output.pdb",
                "--analyze-interface",
                "--relax-separated",
            ]
        )
        assert args.relax_separated is True

    def test_default_interface_values(self):
        """Test default values for interface arguments."""
        parser = create_parser()
        args = parser.parse_args(["-i", "input.pdb", "-o", "output.pdb"])
        assert args.analyze_interface is False
        assert args.interface_distance_cutoff == 8.0
        assert args.interface_chains is None
        assert args.no_binding_energy is False
        assert args.calculate_shape_complementarity is False
        assert args.pack_separated is False
        assert args.relax_separated is False
