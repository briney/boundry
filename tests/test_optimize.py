"""Tests for boundry.optimize module and CLI subcommand."""

import json
import pickle
import random
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from boundry.cli import app
from boundry.config import (
    DdGConfig,
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
    _PositionInfo,
    _compose_seed,
    _cycle_to_dict,
    _filter_sequences,
    _get_aa_at_position,
    _softmax_sample,
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

    def test_ddg_backend_default(self):
        cfg = OptimizeConfig(chain_pairs=[("H", "L")])
        assert cfg.interface_scoring_backend == "ddg"
        assert isinstance(cfg.ddg, DdGConfig)

    def test_legacy_backend(self):
        cfg = OptimizeConfig(
            chain_pairs=[("H", "L")],
            interface_scoring_backend="legacy",
        )
        assert cfg.interface_scoring_backend == "legacy"


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
    @patch("boundry.optimize._analyze_and_find_positions")
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

        # Mock analyze: 1 position per cycle, then none
        mock_analyze.side_effect = [
            (-10.0, [_PositionInfo("H", 52, "", 2.5)]),
            (-12.0, []),  # no positions -> skip
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
        assert (tmp_path / "campaign_summary.json").exists()

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["initial_dG"] == -10.0
        assert summary["final_dG"] == -15.0

        # Verify summary has new fields
        camp = summary["campaigns"][0]
        assert "delta_dG" in camp
        assert "sequences_before" in camp
        assert "sequences_after" in camp
        cycle = camp["cycles"][0]
        assert "old_aa" in cycle
        assert "new_aa" in cycle
        assert "sequences_after" in cycle

        # Verify campaign_summary.json
        cs = json.loads(
            (tmp_path / "campaign_summary.json").read_text()
        )
        assert cs["campaign"] == 1
        assert cs["initial_dG"] == -10.0
        assert cs["final_dG"] == -15.0
        assert "delta_dG" in cs
        assert "sequences_before" in cs
        assert "sequences_after" in cs
        assert len(cs["cycles"]) == 2

    def test_requires_config(self):
        from boundry.optimize import optimize

        with pytest.raises(ValueError, match="OptimizeConfig is required"):
            optimize("ATOM...", config=None)

    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.weights.ensure_weights")
    @patch("boundry.optimize._score_interface")
    @patch("boundry.optimize._analyze_and_find_positions")
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
        # Each campaign: 1 cycle with no positions (skip)
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

        # Each campaign directory should have campaign_summary.json
        for camp_dir in ["campaign_01", "campaign_02"]:
            cs_path = tmp_path / camp_dir / "campaign_summary.json"
            assert cs_path.exists()
            cs = json.loads(cs_path.read_text())
            assert "campaign" in cs
            assert "initial_dG" in cs
            assert "final_dG" in cs
            assert "delta_dG" in cs
            assert "sequences_before" in cs
            assert "sequences_after" in cs
            assert "cycles" in cs


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
        # Cycle 1 (no parent_rank set) -> prior_rank is null
        for entry in summary["rankings"]:
            assert entry["prior_rank"] is None

    def test_prior_rank_in_cycle_summary(self, tmp_path):
        """Results with parent_rank set should produce integer prior_rank."""
        from boundry.optimize import _write_cycle_output

        results = []
        for i, prank in enumerate([1, 1, 2, 2]):
            r = _BeamExpansionResult(
                pdb_string=f"ATOM {i}\nEND\n",
                dG=-15.0 + i,
                target_chain="H",
                target_resnum=50 + i,
                metadata={
                    "old_aa": "G",
                    "new_aa": "S",
                    "sequences": {"H": f"SEQ{i}"},
                },
                parent_rank=prank,
            )
            results.append((r, i))

        cycle_dir = tmp_path / "cycle_02"
        _write_cycle_output(
            cycle_dir,
            results,
            beam_width=2,
            cycle_num=2,
            dG_before=-10.0,
            n_bad_positions=3,
        )

        summary = json.loads(
            (cycle_dir / "cycle_summary.json").read_text()
        )
        prior_ranks = [e["prior_rank"] for e in summary["rankings"]]
        # All should be integers (not None)
        assert all(isinstance(pr, int) for pr in prior_ranks)
        assert set(prior_ranks) == {1, 2}


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
                            old_aa="G",
                            new_aa="S",
                            sequence="MKTLV",
                            sequences_after={
                                "H": "MKTLV",
                                "L": "DIQMT",
                            },
                        ),
                    ],
                    sequences_before={
                        "H": "BEFORE",
                        "L": "DIQMT",
                    },
                    sequences_after={
                        "H": "MKTLV",
                        "L": "DIQMT",
                    },
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

        camp = data["campaigns"][0]
        assert camp["delta_dG"] == -5.0
        assert camp["sequences_before"] == {
            "H": "BEFORE",
            "L": "DIQMT",
        }
        assert camp["sequences_after"] == {
            "H": "MKTLV",
            "L": "DIQMT",
        }

        cycle = camp["cycles"][0]
        assert cycle["old_aa"] == "G"
        assert cycle["new_aa"] == "S"
        assert cycle["sequence"] == "MKTLV"
        assert cycle["sequences_after"] == {
            "H": "MKTLV",
            "L": "DIQMT",
        }

    def test_skipped_cycle_has_null_fields(self, tmp_path):
        from boundry.optimize import _write_summary_json

        result = OptimizeResult(
            structure=MagicMock(),
            campaigns=[
                CampaignResult(
                    campaign=1,
                    initial_dG=-10.0,
                    final_dG=-10.0,
                    cycles=[
                        CycleResult(
                            cycle=1,
                            dG_before=-10.0,
                            dG_after=-10.0,
                            delta_dG=0.0,
                            n_expansions=0,
                            n_bad_positions=0,
                            selected_position=None,
                        ),
                    ],
                ),
            ],
            initial_dG=-10.0,
            final_dG=-10.0,
        )

        config = OptimizeConfig(chain_pairs=[("H", "L")])
        path = tmp_path / "summary.json"
        _write_summary_json(path, result, config)

        data = json.loads(path.read_text())
        cycle = data["campaigns"][0]["cycles"][0]
        assert cycle["old_aa"] is None
        assert cycle["new_aa"] is None
        assert cycle["sequence"] is None
        assert cycle["sequences_after"] is None


# ------------------------------------------------------------------
# _write_campaign_summary
# ------------------------------------------------------------------


class TestWriteCampaignSummary:
    """Tests for per-campaign summary writing."""

    def test_writes_valid_json(self, tmp_path):
        from boundry.optimize import _write_campaign_summary

        cr = CampaignResult(
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
                    old_aa="G",
                    new_aa="S",
                    sequence="MKTLV",
                    sequences_after={"H": "MKTLV", "L": "DIQMT"},
                ),
            ],
            sequences_before={"H": "BEFORE", "L": "DIQMT"},
            sequences_after={"H": "MKTLV", "L": "DIQMT"},
        )

        path = tmp_path / "campaign_summary.json"
        _write_campaign_summary(path, cr)

        data = json.loads(path.read_text())
        assert data["campaign"] == 1
        assert data["initial_dG"] == -10.0
        assert data["final_dG"] == -15.0
        assert data["delta_dG"] == -5.0
        assert data["sequences_before"] == {
            "H": "BEFORE",
            "L": "DIQMT",
        }
        assert data["sequences_after"] == {
            "H": "MKTLV",
            "L": "DIQMT",
        }
        assert len(data["cycles"]) == 1

        cycle = data["cycles"][0]
        assert cycle["old_aa"] == "G"
        assert cycle["new_aa"] == "S"
        assert cycle["sequence"] == "MKTLV"
        assert cycle["sequences_after"] == {
            "H": "MKTLV",
            "L": "DIQMT",
        }

    def test_skipped_cycle_has_null_fields(self, tmp_path):
        from boundry.optimize import _write_campaign_summary

        cr = CampaignResult(
            campaign=1,
            initial_dG=-10.0,
            final_dG=-10.0,
            cycles=[
                CycleResult(
                    cycle=1,
                    dG_before=-10.0,
                    dG_after=-10.0,
                    delta_dG=0.0,
                    n_expansions=0,
                    n_bad_positions=0,
                    selected_position=None,
                ),
            ],
        )

        path = tmp_path / "campaign_summary.json"
        _write_campaign_summary(path, cr)

        data = json.loads(path.read_text())
        cycle = data["cycles"][0]
        assert cycle["old_aa"] is None
        assert cycle["new_aa"] is None
        assert cycle["sequence"] is None
        assert cycle["sequences_after"] is None
        assert data["sequences_before"] is None
        assert data["sequences_after"] is None


# ------------------------------------------------------------------
# _cycle_to_dict
# ------------------------------------------------------------------


class TestCycleToDict:
    """Tests for the cycle serialization helper."""

    def test_all_fields_present(self):
        cy = CycleResult(
            cycle=1,
            dG_before=-10.0,
            dG_after=-12.0,
            delta_dG=-2.0,
            n_expansions=25,
            n_bad_positions=5,
            selected_position="H:52",
            old_aa="G",
            new_aa="S",
            sequence="MKTLV",
            sequences_after={"H": "MKTLV"},
        )
        d = _cycle_to_dict(cy)
        assert d["cycle"] == 1
        assert d["dG_before"] == -10.0
        assert d["dG_after"] == -12.0
        assert d["delta_dG"] == -2.0
        assert d["n_expansions"] == 25
        assert d["n_bad_positions"] == 5
        assert d["selected_position"] == "H:52"
        assert d["old_aa"] == "G"
        assert d["new_aa"] == "S"
        assert d["sequence"] == "MKTLV"
        assert d["sequences_after"] == {"H": "MKTLV"}

    def test_none_fields(self):
        cy = CycleResult(
            cycle=1,
            dG_before=-10.0,
            dG_after=-10.0,
            delta_dG=0.0,
            n_expansions=0,
            n_bad_positions=0,
            selected_position=None,
        )
        d = _cycle_to_dict(cy)
        assert d["old_aa"] is None
        assert d["new_aa"] is None
        assert d["sequence"] is None
        assert d["sequences_after"] is None


# ------------------------------------------------------------------
# Dataclass defaults
# ------------------------------------------------------------------


class TestCycleResultDefaults:
    """Verify backward compatibility of new CycleResult fields."""

    def test_new_fields_default_to_none(self):
        cy = CycleResult(
            cycle=1,
            dG_before=-10.0,
            dG_after=-10.0,
            delta_dG=0.0,
            n_expansions=0,
            n_bad_positions=0,
            selected_position=None,
        )
        assert cy.old_aa is None
        assert cy.new_aa is None
        assert cy.sequences_after is None


class TestCampaignResultDefaults:
    """Verify backward compatibility of new CampaignResult fields."""

    def test_new_fields_default_to_none(self):
        cr = CampaignResult(
            campaign=1,
            initial_dG=-10.0,
            final_dG=-15.0,
        )
        assert cr.sequences_before is None
        assert cr.sequences_after is None


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


# ------------------------------------------------------------------
# Sampling without replacement
# ------------------------------------------------------------------


class TestSamplingWithoutReplacement:
    """Tests that beam expansion samples positions without replacement."""

    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.weights.ensure_weights")
    @patch("boundry.optimize._score_interface")
    @patch("boundry.optimize._analyze_and_find_positions")
    @patch("boundry.operations.relax")
    @patch("boundry.operations.idealize")
    @patch("boundry._parallel.WorkPool")
    def test_caps_tasks_at_bad_positions_count(
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
        """With beam_expansion=10 and 3 positions, only 3 tasks
        should be submitted, each targeting a unique position."""
        from boundry.operations import Structure
        from boundry.optimize import optimize

        pdb = "ATOM mock pdb\nEND\n"

        mock_idealize.return_value = Structure(pdb_string=pdb)
        mock_relax.return_value = Structure(pdb_string=pdb)

        # score: initial, final
        mock_score.side_effect = [-10.0, -15.0]

        # 3 positions, then none (to end after 1 active cycle)
        mock_analyze.side_effect = [
            (
                -10.0,
                [
                    _PositionInfo("H", 50, "", 3.0),
                    _PositionInfo("H", 51, "", 2.0),
                    _PositionInfo("H", 52, "", 1.5),
                ],
            ),
            (-12.0, []),
        ]

        # Capture tasks submitted to pool.map
        captured_tasks = []

        def capture_map(fn, tasks):
            captured_tasks.extend(tasks)
            return [
                _BeamExpansionResult(
                    pdb_string=pdb,
                    metadata={
                        "sequence": "AAA",
                        "old_aa": "G",
                        "new_aa": "S",
                        "sequences": {"H": "MKTLV"},
                    },
                    dG=-12.0,
                    target_chain=t.target_chain,
                    target_resnum=t.target_resnum,
                    target_icode=t.target_icode,
                )
                for t in tasks
            ]

        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.map.side_effect = capture_map
        mock_pool_cls.return_value = mock_pool

        config = OptimizeConfig(
            chain_pairs=[("H", "L")],
            design_cycles=2,
            beam_expansion=10,
            beam_width=1,
            seed=42,
        )

        result = optimize(pdb, config=config, output_dir=tmp_path)

        # Only 3 tasks (capped by number of bad positions)
        assert len(captured_tasks) == 3

        # All target positions are unique
        positions = [
            (t.target_chain, t.target_resnum, t.target_icode)
            for t in captured_tasks
        ]
        assert len(set(positions)) == 3

        # CycleResult records actual task count, not beam_expansion
        active_cycle = result.campaigns[0].cycles[0]
        assert active_cycle.n_expansions == 3


# ------------------------------------------------------------------
# OptimizeConfig validation (new fields)
# ------------------------------------------------------------------


class TestOptimizeConfigValidation:
    """Tests for the new OptimizeConfig fields and validation."""

    def test_position_sampling_default(self):
        cfg = OptimizeConfig(chain_pairs=[("H", "L")])
        assert cfg.position_sampling == "weighted"
        assert cfg.sampling_temperature == 1.0
        assert cfg.regression_tolerance == 0.0

    def test_position_sampling_invalid(self):
        with pytest.raises(ValueError, match="position_sampling"):
            OptimizeConfig(
                chain_pairs=[("H", "L")],
                position_sampling="invalid",
            )

    def test_sampling_temperature_invalid(self):
        with pytest.raises(ValueError, match="sampling_temperature"):
            OptimizeConfig(
                chain_pairs=[("H", "L")],
                sampling_temperature=0.0,
            )
        with pytest.raises(ValueError, match="sampling_temperature"):
            OptimizeConfig(
                chain_pairs=[("H", "L")],
                sampling_temperature=-1.0,
            )

    def test_invalid_backend(self):
        with pytest.raises(
            ValueError, match="interface_scoring_backend"
        ):
            OptimizeConfig(
                chain_pairs=[("H", "L")],
                interface_scoring_backend="invalid",
            )

    def test_threshold_mode_compat(self):
        cfg = OptimizeConfig(
            chain_pairs=[("H", "L")],
            position_sampling="threshold",
            ddg_threshold=2.0,
        )
        assert cfg.position_sampling == "threshold"
        assert cfg.ddg_threshold == 2.0


# ------------------------------------------------------------------
# _softmax_sample
# ------------------------------------------------------------------


class TestSoftmaxSample:
    """Tests for softmax-weighted sampling."""

    def _positions(self):
        return [
            _PositionInfo("H", 50, "", 5.0),
            _PositionInfo("H", 51, "", 3.0),
            _PositionInfo("H", 52, "", 1.0),
            _PositionInfo("H", 53, "", 0.5),
            _PositionInfo("H", 54, "", 0.1),
        ]

    def test_returns_k_positions(self):
        rng = random.Random(42)
        result = _softmax_sample(self._positions(), 3, 1.0, rng)
        assert len(result) == 3

    def test_no_duplicates(self):
        rng = random.Random(42)
        result = _softmax_sample(self._positions(), 4, 1.0, rng)
        keys = [(p.chain_id, p.resnum) for p in result]
        assert len(set(keys)) == len(keys)

    def test_returns_all_when_k_ge_n(self):
        rng = random.Random(42)
        positions = self._positions()
        result = _softmax_sample(positions, 10, 1.0, rng)
        assert len(result) == len(positions)

    def test_deterministic_with_same_seed(self):
        positions = self._positions()
        r1 = _softmax_sample(positions, 3, 1.0, random.Random(99))
        r2 = _softmax_sample(positions, 3, 1.0, random.Random(99))
        assert r1 == r2

    def test_low_temperature_biases_high_ddg(self):
        """Very low temperature should almost always pick the
        highest-ddG position first."""
        positions = self._positions()
        result = _softmax_sample(
            positions, 1, 0.01, random.Random(42)
        )
        assert result[0].ddG == 5.0  # highest ddG

    def test_high_temperature_approaches_uniform(self):
        """At very high temperature, no single position should
        dominate across many trials."""
        positions = self._positions()
        counts = {p.resnum: 0 for p in positions}
        for seed in range(300):
            result = _softmax_sample(
                positions, 1, 1000.0, random.Random(seed)
            )
            counts[result[0].resnum] += 1
        # Each position should be picked at least once in 300 trials
        for resnum, count in counts.items():
            assert count > 0, (
                f"Position {resnum} never sampled at T=1000"
            )


# ------------------------------------------------------------------
# _analyze_and_find_positions
# ------------------------------------------------------------------


class TestAnalyzeAndFindPositions:
    """Tests for the position collection function."""

    def _mock_scan_result(self, rows_data):
        """Build a mock interface analysis result.

        rows_data: list of (chain, resnum, icode, ddG, skipped)
        """
        mock_be = MagicMock()
        mock_be.binding_energy = -15.0

        rows = []
        for chain, resnum, icode, ddG, skipped in rows_data:
            row = MagicMock()
            row.scan_skipped = skipped
            row.ddG = ddG
            row.chain_id = chain
            row.residue_number = resnum
            row.insertion_code = icode
            rows.append(row)
        mock_ala = MagicMock()
        mock_ala.rows = rows

        mock_result = MagicMock()
        mock_result.binding_energy = mock_be
        mock_result.alanine_scan = mock_ala
        return mock_result

    def test_weighted_returns_all_non_skipped(self):
        from boundry.optimize import _analyze_and_find_positions

        mock_result = self._mock_scan_result([
            ("H", 50, "", 3.0, False),  # high ddG
            ("H", 51, "", 0.3, False),  # low ddG
        ])

        config = OptimizeConfig(
            chain_pairs=[("H", "L")],
            position_sampling="weighted",
        )

        with patch(
            "boundry.operations.analyze_interface",
            return_value=mock_result,
        ):
            dG, positions = _analyze_and_find_positions(
                "ATOM...", config, MagicMock()
            )

        assert dG == -15.0
        assert len(positions) == 2

    def test_threshold_filters_by_ddg(self):
        from boundry.optimize import _analyze_and_find_positions

        mock_result = self._mock_scan_result([
            ("H", 50, "", 3.0, False),
            ("H", 51, "", 0.3, False),
        ])

        config = OptimizeConfig(
            chain_pairs=[("H", "L")],
            position_sampling="threshold",
            ddg_threshold=1.0,
        )

        with patch(
            "boundry.operations.analyze_interface",
            return_value=mock_result,
        ):
            dG, positions = _analyze_and_find_positions(
                "ATOM...", config, MagicMock()
            )

        assert len(positions) == 1
        assert positions[0].resnum == 50

    def test_skipped_excluded_from_both_modes(self):
        from boundry.optimize import _analyze_and_find_positions

        mock_result = self._mock_scan_result([
            ("H", 50, "", 3.0, False),
            ("H", 51, "", 5.0, True),  # skipped
        ])

        for mode in ("weighted", "threshold"):
            config = OptimizeConfig(
                chain_pairs=[("H", "L")],
                position_sampling=mode,
            )
            with patch(
                "boundry.operations.analyze_interface",
                return_value=mock_result,
            ):
                _, positions = _analyze_and_find_positions(
                    "ATOM...", config, MagicMock()
                )
            assert len(positions) == 1
            assert positions[0].resnum == 50


# ------------------------------------------------------------------
# Regression guard
# ------------------------------------------------------------------


class TestRegressionGuard:
    """Tests for the regression guard in the optimize loop."""

    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.weights.ensure_weights")
    @patch("boundry.optimize._score_interface")
    @patch("boundry.optimize._analyze_and_find_positions")
    @patch("boundry.operations.relax")
    @patch("boundry.operations.idealize")
    @patch("boundry._parallel.WorkPool")
    def test_rejects_worse_design(
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
        """If the best expansion is worse than the parent, the parent
        structure should be kept."""
        from boundry.operations import Structure
        from boundry.optimize import optimize

        pdb = "ATOM mock pdb\nEND\n"
        mock_idealize.return_value = Structure(pdb_string=pdb)
        mock_relax.return_value = Structure(pdb_string=pdb)

        # score: initial, final
        mock_score.side_effect = [-10.0, -10.0]

        # 1 position, then none
        mock_analyze.side_effect = [
            (-10.0, [_PositionInfo("H", 52, "", 2.0)]),
            (-10.0, []),
        ]

        # Expansion result is WORSE: dG=-8 vs parent dG=-10
        expansion_result = _BeamExpansionResult(
            pdb_string="ATOM worse\nEND\n",
            metadata={
                "sequence": "BAD",
                "old_aa": "G",
                "new_aa": "D",
                "sequences": {"H": "XXX"},
            },
            dG=-8.0,
            target_chain="H",
            target_resnum=52,
        )

        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.map.return_value = [expansion_result]
        mock_pool_cls.return_value = mock_pool

        config = OptimizeConfig(
            chain_pairs=[("H", "L")],
            design_cycles=2,
            beam_expansion=1,
            beam_width=1,
            seed=42,
            regression_tolerance=0.0,
        )

        result = optimize(pdb, config=config, output_dir=tmp_path)

        # Parent should be kept: dG_after == dG_before
        cycle = result.campaigns[0].cycles[0]
        assert cycle.dG_after == -10.0
        assert cycle.selected_position is None
        assert cycle.sequence is None

    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.weights.ensure_weights")
    @patch("boundry.optimize._score_interface")
    @patch("boundry.optimize._analyze_and_find_positions")
    @patch("boundry.operations.relax")
    @patch("boundry.operations.idealize")
    @patch("boundry._parallel.WorkPool")
    def test_tolerance_allows_small_regression(
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
        """With regression_tolerance=1.0, a small regression should
        be accepted."""
        from boundry.operations import Structure
        from boundry.optimize import optimize

        pdb = "ATOM mock pdb\nEND\n"
        mock_idealize.return_value = Structure(pdb_string=pdb)
        mock_relax.return_value = Structure(pdb_string=pdb)

        # score: initial, final
        mock_score.side_effect = [-10.0, -9.5]

        # 1 position, then none
        mock_analyze.side_effect = [
            (-10.0, [_PositionInfo("H", 52, "", 2.0)]),
            (-9.5, []),
        ]

        # Expansion is slightly worse: dG=-9.5 vs parent dG=-10.0
        expansion_result = _BeamExpansionResult(
            pdb_string="ATOM slightly worse\nEND\n",
            metadata={
                "sequence": "OK",
                "old_aa": "G",
                "new_aa": "S",
                "sequences": {"H": "YYY"},
            },
            dG=-9.5,
            target_chain="H",
            target_resnum=52,
        )

        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.map.return_value = [expansion_result]
        mock_pool_cls.return_value = mock_pool

        config = OptimizeConfig(
            chain_pairs=[("H", "L")],
            design_cycles=2,
            beam_expansion=1,
            beam_width=1,
            seed=42,
            regression_tolerance=1.0,
        )

        result = optimize(pdb, config=config, output_dir=tmp_path)

        # Accepted: 0.5 increase < 1.0 tolerance
        cycle = result.campaigns[0].cycles[0]
        assert cycle.dG_after == -9.5
        assert cycle.selected_position is not None
        assert cycle.sequence == "OK"


# ------------------------------------------------------------------
# CLI new flags
# ------------------------------------------------------------------


class TestCLIOptimizeNewFlags:
    """Tests that new CLI flags appear in help output."""

    def test_new_flags_in_help(self):
        result = runner.invoke(app, ["optimize", "--help"])
        assert result.exit_code == 0
        assert "--position-sampling" in result.output
        assert "--sampling-temperature" in result.output
        assert "--regression-tolerance" in result.output


# ------------------------------------------------------------------
# Multi-parent beam expansion
# ------------------------------------------------------------------


class TestMultiParentExpansion:
    """Tests for multi-parent beam search expansion."""

    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.weights.ensure_weights")
    @patch("boundry.optimize._score_interface")
    @patch("boundry.optimize._analyze_and_find_positions")
    @patch("boundry.operations.relax")
    @patch("boundry.operations.idealize")
    @patch("boundry._parallel.WorkPool")
    def test_multi_parent_expansion_count(
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
        """With beam_width=2, beam_expansion=3, cycle 2 should create
        6 tasks (2 parents x 3 expansions each)."""
        from boundry.operations import Structure
        from boundry.optimize import optimize

        pdb = "ATOM mock pdb\nEND\n"
        pdb_a = "ATOM parent A\nEND\n"
        pdb_b = "ATOM parent B\nEND\n"

        mock_idealize.return_value = Structure(pdb_string=pdb)
        mock_relax.return_value = Structure(pdb_string=pdb)

        # score: initial, final
        mock_score.side_effect = [-10.0, -18.0]

        # Cycle 1: 5 positions, cycle 2: 5 positions
        positions = [
            _PositionInfo("H", 50, "", 3.0),
            _PositionInfo("H", 51, "", 2.5),
            _PositionInfo("H", 52, "", 2.0),
            _PositionInfo("H", 53, "", 1.5),
            _PositionInfo("H", 54, "", 1.0),
        ]
        mock_analyze.side_effect = [
            (-10.0, list(positions)),
            (-14.0, list(positions)),
        ]

        captured_tasks_per_cycle = []

        def capture_map(fn, tasks):
            tasks = list(tasks)
            captured_tasks_per_cycle.append(tasks)
            # Return 2 good results so beam_width=2 keeps 2 parents
            results = []
            for i, t in enumerate(tasks):
                results.append(
                    _BeamExpansionResult(
                        pdb_string=(
                            pdb_a if i % 2 == 0 else pdb_b
                        ),
                        metadata={
                            "sequence": f"SEQ{i}",
                            "old_aa": "G",
                            "new_aa": "S",
                            "sequences": {"H": f"MKTLV{i}"},
                        },
                        dG=-14.0 - i * 0.1,
                        target_chain=t.target_chain,
                        target_resnum=t.target_resnum,
                        target_icode=t.target_icode,
                    )
                )
            return results

        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.map.side_effect = capture_map
        mock_pool_cls.return_value = mock_pool

        config = OptimizeConfig(
            chain_pairs=[("H", "L")],
            design_cycles=2,
            beam_expansion=3,
            beam_width=2,
            seed=42,
        )

        optimize(pdb, config=config, output_dir=tmp_path)

        # Cycle 1: 1 parent x 3 = 3 tasks
        assert len(captured_tasks_per_cycle[0]) == 3
        # All tasks share same parent PDB
        parent_pdbs_c1 = set(
            t.parent_pdb_string
            for t in captured_tasks_per_cycle[0]
        )
        assert len(parent_pdbs_c1) == 1

        # Cycle 2: 2 parents x 3 = 6 tasks
        assert len(captured_tasks_per_cycle[1]) == 6
        # Tasks should have 2 distinct parent PDB strings
        parent_pdbs_c2 = set(
            t.parent_pdb_string
            for t in captured_tasks_per_cycle[1]
        )
        assert len(parent_pdbs_c2) == 2

    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.weights.ensure_weights")
    @patch("boundry.optimize._score_interface")
    @patch("boundry.optimize._analyze_and_find_positions")
    @patch("boundry.operations.relax")
    @patch("boundry.operations.idealize")
    @patch("boundry._parallel.WorkPool")
    def test_prior_rank_in_cycle_summary_json(
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
        """Cycle 1 rankings have prior_rank: null, cycle 2 have
        integer prior_rank values."""
        from boundry.operations import Structure
        from boundry.optimize import optimize

        pdb = "ATOM mock pdb\nEND\n"

        mock_idealize.return_value = Structure(pdb_string=pdb)
        mock_relax.return_value = Structure(pdb_string=pdb)

        mock_score.side_effect = [-10.0, -18.0]

        positions = [
            _PositionInfo("H", 50, "", 3.0),
            _PositionInfo("H", 51, "", 2.5),
            _PositionInfo("H", 52, "", 2.0),
        ]
        mock_analyze.side_effect = [
            (-10.0, list(positions)),
            (-14.0, list(positions)),
        ]

        call_count = [0]

        def capture_map(fn, tasks):
            tasks = list(tasks)
            call_count[0] += 1
            results = []
            for i, t in enumerate(tasks):
                results.append(
                    _BeamExpansionResult(
                        pdb_string=f"ATOM result {call_count[0]}_{i}\nEND\n",
                        metadata={
                            "sequence": f"SEQ{i}",
                            "old_aa": "G",
                            "new_aa": "S",
                            "sequences": {"H": f"MKTLV{i}"},
                        },
                        dG=-14.0 - i * 0.1,
                        target_chain=t.target_chain,
                        target_resnum=t.target_resnum,
                        target_icode=t.target_icode,
                    )
                )
            return results

        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.map.side_effect = capture_map
        mock_pool_cls.return_value = mock_pool

        config = OptimizeConfig(
            chain_pairs=[("H", "L")],
            design_cycles=2,
            beam_expansion=2,
            beam_width=2,
            seed=42,
        )

        optimize(pdb, config=config, output_dir=tmp_path)

        # Cycle 1: prior_rank should be null (parent_rank=0)
        c1_summary = json.loads(
            (tmp_path / "cycle_01" / "cycle_summary.json").read_text()
        )
        for entry in c1_summary["rankings"]:
            assert entry["prior_rank"] is None

        # Cycle 2: prior_rank should be integers
        c2_summary = json.loads(
            (tmp_path / "cycle_02" / "cycle_summary.json").read_text()
        )
        for entry in c2_summary["rankings"]:
            assert isinstance(entry["prior_rank"], int)
            assert entry["prior_rank"] >= 1

    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.weights.ensure_weights")
    @patch("boundry.optimize._score_interface")
    @patch("boundry.optimize._analyze_and_find_positions")
    @patch("boundry.operations.relax")
    @patch("boundry.operations.idealize")
    @patch("boundry._parallel.WorkPool")
    def test_regression_guard_preserves_all_parents(
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
        """When a cycle is rejected, all parents should be preserved
        for the next cycle."""
        from boundry.operations import Structure
        from boundry.optimize import optimize

        pdb = "ATOM mock pdb\nEND\n"
        pdb_a = "ATOM parent A\nEND\n"
        pdb_b = "ATOM parent B\nEND\n"

        mock_idealize.return_value = Structure(pdb_string=pdb)
        mock_relax.return_value = Structure(pdb_string=pdb)

        mock_score.side_effect = [-10.0, -14.0]

        positions = [
            _PositionInfo("H", 50, "", 3.0),
            _PositionInfo("H", 51, "", 2.5),
            _PositionInfo("H", 52, "", 2.0),
        ]
        mock_analyze.side_effect = [
            (-10.0, list(positions)),
            (-14.0, list(positions)),  # cycle 2 (rejected)
            (-14.0, list(positions)),  # cycle 3
        ]

        captured_tasks_per_cycle = []
        call_count = [0]

        def capture_map(fn, tasks):
            tasks = list(tasks)
            captured_tasks_per_cycle.append(tasks)
            call_count[0] += 1
            results = []
            for i, t in enumerate(tasks):
                if call_count[0] == 1:
                    # Cycle 1: good results (accepted)
                    dG = -14.0 - i * 0.1
                    pdb_out = pdb_a if i % 2 == 0 else pdb_b
                elif call_count[0] == 2:
                    # Cycle 2: worse results (rejected)
                    dG = -5.0
                    pdb_out = f"ATOM worse {i}\nEND\n"
                else:
                    # Cycle 3: good results
                    dG = -16.0 - i * 0.1
                    pdb_out = f"ATOM better {i}\nEND\n"
                results.append(
                    _BeamExpansionResult(
                        pdb_string=pdb_out,
                        metadata={
                            "sequence": f"SEQ{i}",
                            "old_aa": "G",
                            "new_aa": "S",
                            "sequences": {"H": f"MKTLV{i}"},
                        },
                        dG=dG,
                        target_chain=t.target_chain,
                        target_resnum=t.target_resnum,
                        target_icode=t.target_icode,
                    )
                )
            return results

        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.map.side_effect = capture_map
        mock_pool_cls.return_value = mock_pool

        config = OptimizeConfig(
            chain_pairs=[("H", "L")],
            design_cycles=3,
            beam_expansion=2,
            beam_width=2,
            seed=42,
            regression_tolerance=0.0,
        )

        optimize(pdb, config=config, output_dir=tmp_path)

        # Cycle 1: 1 parent x 2 = 2 tasks
        assert len(captured_tasks_per_cycle[0]) == 2

        # Cycle 2: 2 parents x 2 = 4 tasks (parents from cycle 1)
        assert len(captured_tasks_per_cycle[1]) == 4

        # Cycle 3: still 2 parents x 2 = 4 tasks
        # (cycle 2 was rejected, so original 2 parents preserved)
        assert len(captured_tasks_per_cycle[2]) == 4
        parent_pdbs_c3 = set(
            t.parent_pdb_string
            for t in captured_tasks_per_cycle[2]
        )
        assert len(parent_pdbs_c3) == 2


# ------------------------------------------------------------------
# exclude_native
# ------------------------------------------------------------------


class TestExcludeNative:
    """Tests for the exclude_native feature."""

    def test_config_default_false(self):
        cfg = OptimizeConfig(chain_pairs=[("H", "L")])
        assert cfg.exclude_native is False

    def test_config_explicit_true(self):
        cfg = OptimizeConfig(
            chain_pairs=[("H", "L")], exclude_native=True
        )
        assert cfg.exclude_native is True

    def test_task_default_false(self):
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
        assert task.exclude_native is False

    def test_task_explicit_true(self):
        task = _BeamExpansionTask(
            parent_pdb_string="ATOM...",
            target_chain="H",
            target_resnum=52,
            target_icode="",
            relax_config_dict={},
            design_config_dict={},
            chain_pairs=[("H", "L")],
            seed=42,
            exclude_native=True,
        )
        assert task.exclude_native is True

    def test_pickle_safe_with_exclude_native(self):
        task = _BeamExpansionTask(
            parent_pdb_string="ATOM...",
            target_chain="H",
            target_resnum=52,
            target_icode="",
            relax_config_dict={},
            design_config_dict={},
            chain_pairs=[("H", "L")],
            seed=42,
            exclude_native=True,
        )
        roundtripped = pickle.loads(pickle.dumps(task))
        assert roundtripped.exclude_native is True

    def test_summary_json_includes_field(self, tmp_path):
        from boundry.optimize import _write_summary_json

        result = OptimizeResult(
            structure=MagicMock(),
            campaigns=[],
            initial_dG=-10.0,
            final_dG=-15.0,
        )
        config = OptimizeConfig(
            chain_pairs=[("H", "L")], exclude_native=True
        )
        path = tmp_path / "summary.json"
        _write_summary_json(path, result, config)

        data = json.loads(path.read_text())
        assert data["exclude_native"] is True

    def test_cli_help_includes_flag(self):
        result = runner.invoke(app, ["optimize", "--help"])
        assert result.exit_code == 0
        assert "--exclude-native" in result.output

    @patch("boundry.relaxer.Relaxer")
    @patch("boundry.weights.ensure_weights")
    @patch("boundry.optimize._score_interface")
    @patch("boundry.optimize._analyze_and_find_positions")
    @patch("boundry.operations.relax")
    @patch("boundry.operations.idealize")
    @patch("boundry._parallel.WorkPool")
    def test_tasks_carry_exclude_native(
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
        """Tasks created by optimize() carry exclude_native from config."""
        from boundry.operations import Structure
        from boundry.optimize import optimize

        pdb = "ATOM mock pdb\nEND\n"
        mock_idealize.return_value = Structure(pdb_string=pdb)
        mock_relax.return_value = Structure(pdb_string=pdb)

        mock_score.side_effect = [-10.0, -18.0]

        positions = [
            _PositionInfo("H", 50, "", 3.0),
            _PositionInfo("H", 51, "", 2.5),
        ]
        mock_analyze.return_value = (-10.0, list(positions))

        captured_tasks = []

        def capture_map(fn, tasks):
            tasks = list(tasks)
            captured_tasks.extend(tasks)
            results = []
            for i, t in enumerate(tasks):
                results.append(
                    _BeamExpansionResult(
                        pdb_string=pdb,
                        metadata={
                            "sequence": f"SEQ{i}",
                            "old_aa": "G",
                            "new_aa": "S",
                            "sequences": {"H": f"MKTLV{i}"},
                        },
                        dG=-14.0 - i * 0.1,
                        target_chain=t.target_chain,
                        target_resnum=t.target_resnum,
                        target_icode=t.target_icode,
                    )
                )
            return results

        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.map.side_effect = capture_map
        mock_pool_cls.return_value = mock_pool

        config = OptimizeConfig(
            chain_pairs=[("H", "L")],
            design_cycles=1,
            beam_expansion=2,
            seed=42,
            exclude_native=True,
        )

        optimize(pdb, config=config, output_dir=tmp_path)

        assert len(captured_tasks) > 0
        for task in captured_tasks:
            assert task.exclude_native is True
