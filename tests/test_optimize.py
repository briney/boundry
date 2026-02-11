"""Tests for boundry.optimize module and CLI subcommand."""

import json
import pickle
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from boundry.cli import app
from boundry.config import (
    DesignConfig,
    IdealizeConfig,
    OptimizeConfig,
    RelaxConfig,
)
from boundry.optimize import (
    CampaignResult,
    CycleResult,
    OptimizeResult,
    _BeamExpansionResult,
    _BeamExpansionTask,
    _compose_seed,
    _filter_sequences,
    _get_aa_at_position,
)

runner = CliRunner()


# ------------------------------------------------------------------
# OptimizeConfig
# ------------------------------------------------------------------


class TestOptimizeConfig:
    """Tests for the OptimizeConfig dataclass."""

    def test_defaults(self):
        cfg = OptimizeConfig(chain_pairs=[("H", "L")])
        assert cfg.n_campaigns == 1
        assert cfg.relax_iterations == 10
        assert cfg.design_cycles == 10
        assert cfg.beam_width == 4
        assert cfg.beam_expansion == 25
        assert cfg.ddg_threshold == 1.0
        assert cfg.workers == 1
        assert cfg.show_progress is False
        assert cfg.quiet is True
        assert cfg.seed is None
        assert cfg.scan_chains is None

    def test_chain_pairs_required(self):
        """OptimizeConfig requires chain_pairs."""
        with pytest.raises(TypeError):
            OptimizeConfig()

    def test_custom_values(self):
        cfg = OptimizeConfig(
            chain_pairs=[("A", "B"), ("A", "C")],
            n_campaigns=3,
            beam_width=8,
            seed=42,
        )
        assert len(cfg.chain_pairs) == 2
        assert cfg.n_campaigns == 3
        assert cfg.beam_width == 8
        assert cfg.seed == 42

    def test_nested_configs_defaults(self):
        cfg = OptimizeConfig(chain_pairs=[("H", "L")])
        assert isinstance(cfg.design, DesignConfig)
        assert isinstance(cfg.relax, RelaxConfig)
        assert isinstance(cfg.idealize, IdealizeConfig)
        assert cfg.idealize.enabled is True


# ------------------------------------------------------------------
# _compose_seed
# ------------------------------------------------------------------


class TestComposeSeed:
    """Tests for deterministic seed composition."""

    def test_deterministic(self):
        assert _compose_seed(42, 0) == _compose_seed(42, 0)

    def test_different_inputs(self):
        assert _compose_seed(42, 0) != _compose_seed(42, 1)
        assert _compose_seed(42, 0) != _compose_seed(43, 0)

    def test_formula(self):
        assert _compose_seed(42, 5) == 42 * 100000 + 5


# ------------------------------------------------------------------
# _BeamExpansionTask
# ------------------------------------------------------------------


class TestBeamExpansionTask:
    """Tests for the beam expansion task dataclass."""

    def test_frozen(self):
        task = _BeamExpansionTask(
            parent_pdb_string="ATOM...",
            target_chain="H",
            target_resnum=52,
            target_icode="",
            relax_config_dict={},
            design_config_dict={},
            chain_pairs=[("H", "L")],
            seed=42,
        )
        with pytest.raises(AttributeError):
            task.seed = 99

    def test_pickle_safe(self):
        task = _BeamExpansionTask(
            parent_pdb_string="ATOM...",
            target_chain="H",
            target_resnum=52,
            target_icode="",
            relax_config_dict={"constrained": False},
            design_config_dict={"temperature": 0.1},
            chain_pairs=[("H", "L")],
            seed=42,
        )
        roundtripped = pickle.loads(pickle.dumps(task))
        assert roundtripped.target_chain == "H"
        assert roundtripped.seed == 42
        assert roundtripped.chain_pairs == [("H", "L")]


# ------------------------------------------------------------------
# OptimizeResult
# ------------------------------------------------------------------


class TestOptimizeResult:
    """Tests for the OptimizeResult dataclass."""

    def test_delta_dG_property(self):
        result = OptimizeResult(
            structure=MagicMock(),
            initial_dG=-10.0,
            final_dG=-15.0,
        )
        assert result.delta_dG == pytest.approx(-5.0)

    def test_delta_dG_none_when_missing(self):
        result = OptimizeResult(structure=MagicMock())
        assert result.delta_dG is None

    def test_delta_dG_none_partial(self):
        result = OptimizeResult(
            structure=MagicMock(), initial_dG=-10.0
        )
        assert result.delta_dG is None


# ------------------------------------------------------------------
# _analyze_and_find_bad
# ------------------------------------------------------------------


class TestAnalyzeAndFindBad:
    """Tests for the alanine scan filtering helper."""

    def test_filters_by_threshold(self):
        from boundry.optimize import _analyze_and_find_bad

        mock_be = MagicMock()
        mock_be.binding_energy = -15.0

        mock_row_bad = MagicMock()
        mock_row_bad.scan_skipped = False
        mock_row_bad.ddG = 2.5
        mock_row_bad.chain_id = "H"
        mock_row_bad.residue_number = 52
        mock_row_bad.insertion_code = ""

        mock_row_good = MagicMock()
        mock_row_good.scan_skipped = False
        mock_row_good.ddG = 0.3
        mock_row_good.chain_id = "H"
        mock_row_good.residue_number = 53
        mock_row_good.insertion_code = ""

        mock_row_skipped = MagicMock()
        mock_row_skipped.scan_skipped = True

        mock_ala = MagicMock()
        mock_ala.rows = [mock_row_bad, mock_row_good, mock_row_skipped]

        mock_result = MagicMock()
        mock_result.binding_energy = mock_be
        mock_result.alanine_scan = mock_ala

        config = OptimizeConfig(
            chain_pairs=[("H", "L")], ddg_threshold=1.0
        )
        mock_relaxer = MagicMock()

        with patch(
            "boundry.operations.analyze_interface",
            return_value=mock_result,
        ):
            dG, bad = _analyze_and_find_bad(
                "ATOM...", config, mock_relaxer
            )

        assert dG == -15.0
        assert len(bad) == 1
        assert bad[0] == ("H", 52, "")

    def test_no_bad_positions(self):
        from boundry.optimize import _analyze_and_find_bad

        mock_be = MagicMock()
        mock_be.binding_energy = -15.0

        mock_row = MagicMock()
        mock_row.scan_skipped = False
        mock_row.ddG = 0.5
        mock_row.chain_id = "H"
        mock_row.residue_number = 52
        mock_row.insertion_code = ""

        mock_ala = MagicMock()
        mock_ala.rows = [mock_row]

        mock_result = MagicMock()
        mock_result.binding_energy = mock_be
        mock_result.alanine_scan = mock_ala

        config = OptimizeConfig(
            chain_pairs=[("H", "L")], ddg_threshold=1.0
        )

        with patch(
            "boundry.operations.analyze_interface",
            return_value=mock_result,
        ):
            dG, bad = _analyze_and_find_bad(
                "ATOM...", config, MagicMock()
            )

        assert len(bad) == 0


# ------------------------------------------------------------------
# _score_interface
# ------------------------------------------------------------------


class TestScoreInterface:
    """Tests for the binding energy scoring helper."""

    def test_returns_dG(self):
        from boundry.optimize import _score_interface

        mock_be_result = MagicMock()
        mock_be_result.binding_energy = -12.5

        config = OptimizeConfig(chain_pairs=[("H", "L")])

        with patch(
            "boundry.binding_energy.calculate_binding_energy",
            return_value=mock_be_result,
        ):
            dG = _score_interface("ATOM...", config, MagicMock())

        assert dG == -12.5

    def test_raises_on_none(self):
        from boundry.optimize import _score_interface

        mock_be_result = MagicMock()
        mock_be_result.binding_energy = None

        config = OptimizeConfig(chain_pairs=[("H", "L")])

        with patch(
            "boundry.binding_energy.calculate_binding_energy",
            return_value=mock_be_result,
        ):
            with pytest.raises(RuntimeError, match="returned None"):
                _score_interface("ATOM...", config, MagicMock())


# ------------------------------------------------------------------
# Full optimize (mocked)
# ------------------------------------------------------------------


class TestOptimize:
    """End-to-end tests with mocked operations."""

    def _make_mock_be(self, dG=-15.0):
        be = MagicMock()
        be.binding_energy = dG
        return be

    def _make_mock_ala_result(self, bad_positions):
        rows = []
        for chain, resnum, icode, ddG_val in bad_positions:
            row = MagicMock()
            row.scan_skipped = False
            row.ddG = ddG_val
            row.chain_id = chain
            row.residue_number = resnum
            row.insertion_code = icode
            rows.append(row)
        ala = MagicMock()
        ala.rows = rows
        return ala

    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.weights.ensure_weights")
    @patch("boundry.optimize._score_interface")
    @patch("boundry.optimize._analyze_and_find_bad")
    @patch("boundry.operations.relax")
    @patch("boundry.operations.idealize")
    @patch("boundry._parallel.WorkPool")
    def test_basic_flow(
        self,
        mock_pool_cls,
        mock_idealize,
        mock_relax,
        mock_analyze,
        mock_score,
        mock_ensure_weights,
        mock_relaxer_cls,
        tmp_path,
    ):
        from boundry.operations import Structure

        pdb = "ATOM mock pdb\nEND\n"

        # Mock idealize
        mock_idealize.return_value = Structure(pdb_string=pdb)

        # Mock relax
        mock_relax.return_value = Structure(
            pdb_string=pdb,
            metadata={"final_energy": -100.0},
        )

        # Mock score_interface: initial=-10, final=-15
        mock_score.side_effect = [-10.0, -15.0]

        # Mock analyze: 1 bad position per cycle, then no bad
        mock_analyze.side_effect = [
            (-10.0, [("H", 52, "")]),
            (-12.0, []),  # no bad -> skip
        ]

        # Mock pool
        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)

        expansion_result = _BeamExpansionResult(
            pdb_string=pdb,
            metadata={
                "sequence": "AAA",
                "old_aa": "G",
                "new_aa": "S",
                "sequences": {"H": "MKTLV", "L": "DIQMT"},
            },
            dG=-12.0,
            target_chain="H",
            target_resnum=52,
        )
        mock_pool.map.return_value = [expansion_result]
        mock_pool_cls.return_value = mock_pool

        config = OptimizeConfig(
            chain_pairs=[("H", "L")],
            design_cycles=2,
            beam_expansion=1,
            beam_width=1,
            seed=42,
        )

        from boundry.optimize import optimize

        result = optimize(pdb, config=config, output_dir=tmp_path)

        assert isinstance(result, OptimizeResult)
        assert result.initial_dG == -10.0
        assert result.final_dG == -15.0
        assert result.delta_dG == pytest.approx(-5.0)
        assert len(result.campaigns) == 1
        assert len(result.campaigns[0].cycles) == 2

        # Check output files
        assert (tmp_path / "final.pdb").exists()
        assert (tmp_path / "summary.json").exists()

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["initial_dG"] == -10.0
        assert summary["final_dG"] == -15.0

    def test_requires_config(self):
        from boundry.optimize import optimize

        with pytest.raises(ValueError, match="OptimizeConfig is required"):
            optimize("ATOM...", config=None)

    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.weights.ensure_weights")
    @patch("boundry.optimize._score_interface")
    @patch("boundry.optimize._analyze_and_find_bad")
    @patch("boundry.operations.relax")
    @patch("boundry.operations.idealize")
    @patch("boundry._parallel.WorkPool")
    def test_multi_campaign(
        self,
        mock_pool_cls,
        mock_idealize,
        mock_relax,
        mock_analyze,
        mock_score,
        mock_ensure_weights,
        mock_relaxer_cls,
        tmp_path,
    ):
        from boundry.operations import Structure

        pdb = "ATOM mock pdb\nEND\n"

        mock_idealize.return_value = Structure(pdb_string=pdb)
        mock_relax.return_value = Structure(pdb_string=pdb)

        # Two campaigns, each with 1 cycle
        # score calls: initial_c1, final_c1, initial_c2, final_c2
        mock_score.side_effect = [-10.0, -14.0, -10.0, -16.0]
        # Each campaign: 1 cycle with no bad positions (skip)
        mock_analyze.side_effect = [
            (-10.0, []),
            (-10.0, []),
        ]

        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool_cls.return_value = mock_pool

        config = OptimizeConfig(
            chain_pairs=[("H", "L")],
            n_campaigns=2,
            design_cycles=1,
            seed=42,
        )

        from boundry.optimize import optimize

        result = optimize(pdb, config=config, output_dir=tmp_path)

        assert len(result.campaigns) == 2
        # Best campaign should be campaign 2 (dG=-16)
        assert result.final_dG == -16.0

        # Multi-campaign layout
        assert (tmp_path / "campaign_01").is_dir()
        assert (tmp_path / "campaign_02").is_dir()
        assert (tmp_path / "final.pdb").exists()


# ------------------------------------------------------------------
# Strict chain pair parsing
# ------------------------------------------------------------------


class TestStrictChainParsing:
    """Tests for _parse_chain_pairs_strict."""

    def test_valid_single(self):
        from boundry.cli import _parse_chain_pairs_strict

        result = _parse_chain_pairs_strict("H:L")
        assert result == [("H", "L")]

    def test_valid_multiple(self):
        from boundry.cli import _parse_chain_pairs_strict

        result = _parse_chain_pairs_strict("H:L,H:A")
        assert result == [("H", "L"), ("H", "A")]

    def test_whitespace_handling(self):
        from boundry.cli import _parse_chain_pairs_strict

        result = _parse_chain_pairs_strict(" H : L , H : A ")
        assert result == [("H", "L"), ("H", "A")]

    def test_missing_colon(self):
        from boundry.cli import _parse_chain_pairs_strict

        with pytest.raises(Exception, match="Invalid chain pair"):
            _parse_chain_pairs_strict("HL")

    def test_empty_chain_id(self):
        from boundry.cli import _parse_chain_pairs_strict

        with pytest.raises(Exception, match="must not be empty"):
            _parse_chain_pairs_strict("H:")

    def test_empty_string(self):
        from boundry.cli import _parse_chain_pairs_strict

        with pytest.raises(Exception, match="No valid chain pairs"):
            _parse_chain_pairs_strict("")

    def test_multiple_colons(self):
        from boundry.cli import _parse_chain_pairs_strict

        with pytest.raises(Exception, match="exactly one"):
            _parse_chain_pairs_strict("H:L:A")


# ------------------------------------------------------------------
# CLI tests
# ------------------------------------------------------------------


class TestCLIOptimize:
    """Tests for the optimize CLI subcommand."""

    def test_help(self):
        result = runner.invoke(app, ["optimize", "--help"])
        assert result.exit_code == 0
        assert "interface" in result.output.lower()
        assert "beam" in result.output.lower()

    def test_in_command_list(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "optimize" in result.output

    def test_missing_required_args(self):
        result = runner.invoke(app, ["optimize"])
        assert result.exit_code != 0

    def test_missing_interface_flag(self, tmp_path):
        pdb = tmp_path / "in.pdb"
        pdb.write_text("ATOM mock\nEND\n")
        result = runner.invoke(
            app,
            ["optimize", str(pdb), str(tmp_path / "out")],
        )
        assert result.exit_code != 0

    def test_invalid_interface_format(self, tmp_path):
        pdb = tmp_path / "in.pdb"
        pdb.write_text("ATOM mock\nEND\n")
        result = runner.invoke(
            app,
            [
                "optimize",
                str(pdb),
                str(tmp_path / "out"),
                "--interface",
                "HL",
            ],
        )
        assert result.exit_code != 0
        assert "Invalid chain pair" in result.output

    def test_nonexistent_input(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "optimize",
                str(tmp_path / "nonexistent.pdb"),
                str(tmp_path / "out"),
                "--interface",
                "H:L",
            ],
        )
        assert result.exit_code != 0


# ------------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------------


class TestWriteCycleOutput:
    """Tests for cycle output writing."""

    def test_writes_ranked_pdbs(self, tmp_path):
        from boundry.optimize import _write_cycle_output

        results = []
        for i in range(5):
            r = _BeamExpansionResult(
                pdb_string=f"ATOM {i}\nEND\n",
                dG=-15.0 + i,
                target_chain="H",
                target_resnum=50 + i,
                metadata={
                    "old_aa": "G",
                    "new_aa": chr(ord("A") + i),
                    "sequences": {"H": f"SEQ{i}", "L": "DIQMT"},
                },
            )
            results.append((r, i))

        cycle_dir = tmp_path / "cycle_01"
        dG_before = -10.0
        seq_before = {"H": "BEFORE", "L": "DIQMT"}
        _write_cycle_output(
            cycle_dir,
            results,
            beam_width=3,
            cycle_num=1,
            dG_before=dG_before,
            n_bad_positions=4,
            sequences_before=seq_before,
        )

        # Top 3 in cycle dir
        assert (cycle_dir / "rank_01.pdb").exists()
        assert (cycle_dir / "rank_02.pdb").exists()
        assert (cycle_dir / "rank_03.pdb").exists()

        # Others in other/
        assert (cycle_dir / "other" / "rank_04.pdb").exists()
        assert (cycle_dir / "other" / "rank_05.pdb").exists()

        # Summary JSON
        summary = json.loads(
            (cycle_dir / "cycle_summary.json").read_text()
        )
        assert summary["cycle"] == 1
        assert summary["dG_before"] == dG_before
        assert summary["dG_after"] == -15.0
        assert summary["delta_dG"] == -15.0 - dG_before
        assert summary["n_bad_positions"] == 4
        assert summary["sequences_before"] == seq_before
        assert len(summary["rankings"]) == 5
        assert summary["rankings"][0]["rank"] == 1
        assert summary["rankings"][0]["delta_dG"] == -15.0 - dG_before
        assert summary["rankings"][0]["old_aa"] == "G"
        assert summary["rankings"][0]["new_aa"] == "A"
        assert summary["rankings"][0]["sequences_after"] == {
            "H": "SEQ0",
            "L": "DIQMT",
        }


class TestWriteSummaryJson:
    """Tests for aggregate summary writing."""

    def test_writes_valid_json(self, tmp_path):
        from boundry.optimize import _write_summary_json

        result = OptimizeResult(
            structure=MagicMock(),
            campaigns=[
                CampaignResult(
                    campaign=1,
                    initial_dG=-10.0,
                    final_dG=-15.0,
                    cycles=[
                        CycleResult(
                            cycle=1,
                            dG_before=-10.0,
                            dG_after=-12.0,
                            delta_dG=-2.0,
                            n_expansions=25,
                            n_bad_positions=5,
                            selected_position="H:52",
                        ),
                    ],
                ),
            ],
            initial_dG=-10.0,
            final_dG=-15.0,
        )

        config = OptimizeConfig(chain_pairs=[("H", "L")])
        path = tmp_path / "summary.json"
        _write_summary_json(path, result, config)

        data = json.loads(path.read_text())
        assert data["initial_dG"] == -10.0
        assert data["final_dG"] == -15.0
        assert data["delta_dG"] == -5.0
        assert len(data["campaigns"]) == 1
        assert len(data["campaigns"][0]["cycles"]) == 1


# ------------------------------------------------------------------
# _get_aa_at_position
# ------------------------------------------------------------------


class TestGetAaAtPosition:
    """Tests for the per-position AA lookup helper."""

    PDB = (
        "ATOM      1  N   ALA H  10       0.0  0.0  0.0  1.00  0.00\n"
        "ATOM      2  CA  ALA H  10       1.0  0.0  0.0  1.00  0.00\n"
        "ATOM      3  N   GLY H  11       2.0  0.0  0.0  1.00  0.00\n"
        "ATOM      4  CA  GLY H  11       3.0  0.0  0.0  1.00  0.00\n"
        "ATOM      5  CA  SER L  20       4.0  0.0  0.0  1.00  0.00\n"
        "END\n"
    )

    def test_finds_residue(self):
        assert _get_aa_at_position(self.PDB, "H", 10, "") == "A"
        assert _get_aa_at_position(self.PDB, "H", 11, "") == "G"
        assert _get_aa_at_position(self.PDB, "L", 20, "") == "S"

    def test_missing_returns_x(self):
        assert _get_aa_at_position(self.PDB, "H", 99, "") == "X"
        assert _get_aa_at_position(self.PDB, "Z", 10, "") == "X"

    def test_icode_matching(self):
        pdb_icode = (
            "ATOM      1  CA  ALA H  10A      0.0  0.0  0.0  1.00  0.00\n"
            "ATOM      2  CA  GLY H  10       1.0  0.0  0.0  1.00  0.00\n"
        )
        assert _get_aa_at_position(pdb_icode, "H", 10, "A") == "A"
        assert _get_aa_at_position(pdb_icode, "H", 10, "") == "G"


# ------------------------------------------------------------------
# _filter_sequences
# ------------------------------------------------------------------


class TestFilterSequences:
    """Tests for the scan_chains filtering helper."""

    def test_none_passthrough(self):
        seqs = {"H": "AAA", "L": "BBB", "A": "CCC"}
        assert _filter_sequences(seqs, None) == seqs

    def test_filters_to_scan_chains(self):
        seqs = {"H": "AAA", "L": "BBB", "A": "CCC"}
        assert _filter_sequences(seqs, ["H", "L"]) == {
            "H": "AAA",
            "L": "BBB",
        }

    def test_missing_chain_ignored(self):
        seqs = {"H": "AAA"}
        assert _filter_sequences(seqs, ["H", "L"]) == {"H": "AAA"}


# ------------------------------------------------------------------
# scan_chains filtering in _write_cycle_output
# ------------------------------------------------------------------


class TestWriteCycleOutputScanChains:
    """Tests that scan_chains filters sequences_after."""

    def test_sequences_after_filtered(self, tmp_path):
        from boundry.optimize import _write_cycle_output

        r = _BeamExpansionResult(
            pdb_string="ATOM 0\nEND\n",
            dG=-15.0,
            target_chain="H",
            target_resnum=50,
            metadata={
                "old_aa": "G",
                "new_aa": "S",
                "sequences": {"H": "AAA", "L": "BBB", "A": "CCC"},
            },
        )

        cycle_dir = tmp_path / "cycle_filtered"
        _write_cycle_output(
            cycle_dir,
            [(r, 0)],
            beam_width=1,
            cycle_num=1,
            dG_before=-10.0,
            n_bad_positions=1,
            sequences_before={"H": "OLD", "L": "OLD2"},
            scan_chains=["H", "L"],
        )

        summary = json.loads(
            (cycle_dir / "cycle_summary.json").read_text()
        )
        after = summary["rankings"][0]["sequences_after"]
        assert "H" in after
        assert "L" in after
        assert "A" not in after
