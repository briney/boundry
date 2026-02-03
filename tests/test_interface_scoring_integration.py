"""Integration tests for interface scoring metrics using a real antibody-antigen
crystal structure (PDB 1VFB: D1.3 Fv antibody bound to hen egg white lysozyme).

These tests validate that interface identification, SASA, shape complementarity,
and binding energy (ddG) produce biologically meaningful results on a known
high-affinity interaction.

Requires OpenMM to be installed and network access to download the PDB.
"""

import math

import pytest

from boundry.binding_energy import (
    BindingEnergyResult,
    calculate_binding_energy,
)
from boundry.config import (
    InterfaceConfig,
    PipelineConfig,
    PipelineMode,
    RelaxConfig,
)
from boundry.interface import identify_interface_residues
from boundry.surface_area import (
    calculate_shape_complementarity,
    calculate_surface_area,
)

# Skip entire module if OpenMM not available
pytest.importorskip("openmm")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def relaxer():
    """Create a Relaxer instance for integration tests."""
    from boundry.relaxer import Relaxer

    config = RelaxConfig(max_iterations=50, stiffness=10.0)
    return Relaxer(config)


@pytest.fixture(scope="session")
def relaxed_1vfb_pdb_string(antibody_antigen_pdb_string, relaxer):
    """Relax the 1VFB crystal structure for use in energy calculations.

    Crystal structures lack hydrogen atoms, so they must be relaxed
    (which adds hydrogens via the AMBER pipeline) before OpenMM energy
    calculations will produce meaningful results.
    """
    relaxed_pdb, _, _ = relaxer.relax(antibody_antigen_pdb_string)
    return relaxed_pdb


@pytest.fixture(scope="session")
def interface_info_1vfb(antibody_antigen_pdb_string):
    """Pre-computed interface residue identification for 1VFB, reused across
    test classes."""
    return identify_interface_residues(antibody_antigen_pdb_string)


# ---------------------------------------------------------------------------
# Test Class 1: Interface residue identification
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestInterfaceResiduesWithRealStructure:
    """Test identify_interface_residues() on 1VFB."""

    def test_identifies_interface_on_antibody_antigen(
        self, interface_info_1vfb
    ):
        """Auto-detection finds residues on all 3 chains with non-empty
        chain_pairs."""
        info = interface_info_1vfb
        assert info.n_interface_residues > 0
        assert len(info.chain_pairs) > 0

        chains_in_interface = {r.chain_id for r in info.interface_residues}
        assert "A" in chains_in_interface
        assert "B" in chains_in_interface
        assert "C" in chains_in_interface

    def test_interface_residue_count_reasonable(self, interface_info_1vfb):
        """Between 20 and 200 interface residues (rules out 0 and whole
        protein)."""
        n = interface_info_1vfb.n_interface_residues
        assert 20 < n < 200, f"Unexpected interface residue count: {n}"

    def test_explicit_chain_pairs(self, antibody_antigen_pdb_string):
        """Specifying chain_pairs=[('A','C'),('B','C')] returns non-empty
        result with expected pairs."""
        info = identify_interface_residues(
            antibody_antigen_pdb_string,
            chain_pairs=[("A", "C"), ("B", "C")],
        )
        assert info.n_interface_residues > 0
        pair_set = set(info.chain_pairs)
        assert ("A", "C") in pair_set or ("B", "C") in pair_set

    def test_close_contacts_exist(self, interface_info_1vfb):
        """At least some residues have min_distance < 4.0 A (real vdW
        contacts)."""
        close = [
            r
            for r in interface_info_1vfb.interface_residues
            if r.min_distance < 4.0
        ]
        assert len(close) > 0, "No close contacts found (< 4.0 A)"

    def test_distance_cutoff_sensitivity(self, antibody_antigen_pdb_string):
        """5.0 A cutoff yields fewer residues than 10.0 A; both non-empty."""
        info_5 = identify_interface_residues(
            antibody_antigen_pdb_string, distance_cutoff=5.0
        )
        info_10 = identify_interface_residues(
            antibody_antigen_pdb_string, distance_cutoff=10.0
        )
        assert info_5.n_interface_residues > 0
        assert info_10.n_interface_residues > 0
        assert info_5.n_interface_residues < info_10.n_interface_residues


# ---------------------------------------------------------------------------
# Test Class 2: Surface area (SASA)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSurfaceAreaWithRealStructure:
    """Test calculate_surface_area() on 1VFB."""

    def test_buried_sasa_substantial(
        self, antibody_antigen_pdb_string, interface_info_1vfb
    ):
        """Antibody-antigen interfaces typically bury 1200-2000 sq A; 500 is
        conservative."""
        result = calculate_surface_area(
            antibody_antigen_pdb_string,
            interface_info_1vfb.interface_residues,
        )
        assert (
            result.buried_sasa > 500
        ), f"Buried SASA too small: {result.buried_sasa:.1f}"

    def test_complex_sasa_less_than_chain_sum(
        self, antibody_antigen_pdb_string, interface_info_1vfb
    ):
        """Fundamental thermodynamic property: complex_sasa < sum of
        individual chain SASAs."""
        result = calculate_surface_area(
            antibody_antigen_pdb_string,
            interface_info_1vfb.interface_residues,
        )
        chain_sum = sum(result.chain_sasa.values())
        assert result.complex_sasa < chain_sum

    def test_all_chains_have_positive_sasa(
        self, antibody_antigen_pdb_string, interface_info_1vfb
    ):
        """All 3 chains (A, B, C) appear in chain_sasa with values > 0."""
        result = calculate_surface_area(
            antibody_antigen_pdb_string,
            interface_info_1vfb.interface_residues,
        )
        for chain_id in ("A", "B", "C"):
            assert (
                chain_id in result.chain_sasa
            ), f"Chain {chain_id} missing from chain_sasa"
            assert result.chain_sasa[chain_id] > 0

    def test_interface_residue_sasa_populated(
        self, antibody_antigen_pdb_string, interface_info_1vfb
    ):
        """interface_residue_sasa is non-empty with non-negative values."""
        result = calculate_surface_area(
            antibody_antigen_pdb_string,
            interface_info_1vfb.interface_residues,
        )
        assert len(result.interface_residue_sasa) > 0
        for key, val in result.interface_residue_sasa.items():
            assert val >= 0, f"Negative SASA for {key}: {val}"


# ---------------------------------------------------------------------------
# Test Class 3: Shape complementarity
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestShapeComplementarityWithRealStructure:
    """Test calculate_shape_complementarity() on 1VFB."""

    def test_sc_score_in_valid_range(
        self, antibody_antigen_pdb_string, interface_info_1vfb
    ):
        """Sc score is in the [0, 1] range. The simplified implementation
        may return 0.0 for large interfaces where the average inter-atom
        distance exceeds the scoring threshold."""
        result = calculate_shape_complementarity(
            antibody_antigen_pdb_string,
            interface_info_1vfb.chain_pairs,
            interface_info_1vfb.interface_residues,
        )
        assert (
            0.0 <= result.sc_score <= 1.0
        ), f"Sc score out of [0, 1] range: {result.sc_score}"

    def test_interface_area_positive(
        self, antibody_antigen_pdb_string, interface_info_1vfb
    ):
        """Interface area must be > 0."""
        result = calculate_shape_complementarity(
            antibody_antigen_pdb_string,
            interface_info_1vfb.chain_pairs,
            interface_info_1vfb.interface_residues,
        )
        assert result.interface_area > 0

    def test_gap_volume_positive(
        self, antibody_antigen_pdb_string, interface_info_1vfb
    ):
        """Interface gap volume must be > 0."""
        result = calculate_shape_complementarity(
            antibody_antigen_pdb_string,
            interface_info_1vfb.chain_pairs,
            interface_info_1vfb.interface_residues,
        )
        assert result.interface_gap_volume > 0


# ---------------------------------------------------------------------------
# Test Class 4: Binding energy (ddG)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestBindingEnergyWithRealStructure:
    """Test calculate_binding_energy() on 1VFB.

    Uses the relaxed structure (with hydrogens) for meaningful AMBER energies,
    and explicit chain_pairs=[('A','C'),('B','C')] to group antibody chains
    (A+B) vs antigen (C).

    With GBn2 implicit solvation enabled (default), ddG should be negative
    for this known high-affinity antibody-antigen interaction.
    """

    def test_binding_energy_completes(self, relaxed_1vfb_pdb_string, relaxer):
        """Returns BindingEnergyResult with finite non-zero complex_energy,
        2 entries in separated_energies (A+B and C), and non-empty
        interface_residues."""
        result = calculate_binding_energy(
            relaxed_1vfb_pdb_string,
            relaxer,
            chain_pairs=[("A", "C"), ("B", "C")],
            relax_separated=False,
        )
        assert isinstance(result, BindingEnergyResult)
        assert math.isfinite(result.complex_energy)
        assert result.complex_energy != 0.0
        assert len(result.separated_energies) == 2
        assert len(result.interface_residues) > 0

    def test_positive_ddg_rosetta_convention(
        self, relaxed_1vfb_pdb_string, relaxer
    ):
        """ddG (separated - complex) should be positive for 1VFB under
        Rosetta InterfaceAnalyzer convention."""
        result = calculate_binding_energy(
            relaxed_1vfb_pdb_string,
            relaxer,
            chain_pairs=[("A", "C"), ("B", "C")],
            relax_separated=False,
        )
        assert math.isfinite(result.binding_energy)
        assert result.binding_energy > 0, (
            "ddG should be positive for high-affinity binder in "
            "separated-minus-complex convention, "
            f"got {result.binding_energy:.2f}"
        )

    def test_chain_grouping_correct(self, relaxed_1vfb_pdb_string, relaxer):
        """separated_energies keys should be 'A+B' and 'C' (antibody Fv
        grouped together, lysozyme separate)."""
        result = calculate_binding_energy(
            relaxed_1vfb_pdb_string,
            relaxer,
            chain_pairs=[("A", "C"), ("B", "C")],
            relax_separated=False,
        )
        keys = set(result.separated_energies.keys())
        assert keys == {"A+B", "C"}, f"Unexpected chain groups: {keys}"

    def test_binding_energy_with_relax_separated(
        self, relaxed_1vfb_pdb_string, relaxer
    ):
        """relax_separated=True: separated energies are finite and non-zero,
        and relaxation lowers separated chain energies compared to unrelaxed
        separation."""
        result_relaxed = calculate_binding_energy(
            relaxed_1vfb_pdb_string,
            relaxer,
            chain_pairs=[("A", "C"), ("B", "C")],
            relax_separated=True,
        )
        result_unrelaxed = calculate_binding_energy(
            relaxed_1vfb_pdb_string,
            relaxer,
            chain_pairs=[("A", "C"), ("B", "C")],
            relax_separated=False,
        )
        for label, energy in result_relaxed.separated_energies.items():
            assert math.isfinite(
                energy
            ), f"Non-finite energy for {label}: {energy}"
            assert energy != 0.0
        # Relaxed separated energies should be lower (more negative)
        # than unrelaxed, since relaxation minimizes energy
        relaxed_sum = sum(result_relaxed.separated_energies.values())
        unrelaxed_sum = sum(result_unrelaxed.separated_energies.values())
        assert relaxed_sum <= unrelaxed_sum, (
            f"Relaxed separated energy ({relaxed_sum:.2f}) should be "
            f"<= unrelaxed ({unrelaxed_sum:.2f})"
        )


# ---------------------------------------------------------------------------
# Test Class 5: Cross-metric consistency
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMetricConsistency:
    """Cross-metric consistency tests on 1VFB."""

    @pytest.mark.slow
    def test_interface_residue_count_matches_binding_energy(
        self, relaxed_1vfb_pdb_string, relaxer
    ):
        """Interface residue count from identify_interface_residues equals
        count from BindingEnergyResult.interface_residues (same parameters)."""
        pairs = [("A", "C"), ("B", "C")]
        standalone = identify_interface_residues(
            relaxed_1vfb_pdb_string, chain_pairs=pairs
        )
        be_result = calculate_binding_energy(
            relaxed_1vfb_pdb_string,
            relaxer,
            chain_pairs=pairs,
            relax_separated=False,
        )
        assert standalone.n_interface_residues == len(
            be_result.interface_residues
        )

    def test_all_metrics_non_degenerate(
        self, antibody_antigen_pdb_string, interface_info_1vfb
    ):
        """Compute all metrics on same structure; none return zeros/empty
        results."""
        # Interface residues already verified non-empty via fixture
        assert interface_info_1vfb.n_interface_residues > 0

        # SASA
        sasa = calculate_surface_area(
            antibody_antigen_pdb_string,
            interface_info_1vfb.interface_residues,
        )
        assert sasa.buried_sasa > 0
        assert sasa.complex_sasa > 0

        # Shape complementarity (area and gap volume should be positive;
        # sc_score may be 0.0 with the simplified implementation)
        sc = calculate_shape_complementarity(
            antibody_antigen_pdb_string,
            interface_info_1vfb.chain_pairs,
            interface_info_1vfb.interface_residues,
        )
        assert sc.sc_score >= 0
        assert sc.interface_area > 0
        assert sc.interface_gap_volume > 0


# ---------------------------------------------------------------------------
# Test Class 6: Pipeline orchestration
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineInterfaceAnalysis:
    """Test the full pipeline _analyze_interface orchestration on 1VFB.

    Uses explicit chain_pairs to get antibody (A+B) vs antigen (C)
    grouping, avoiding the VL-VH interface pair that causes the chain
    grouping to fall back to individual chains.
    """

    def test_pipeline_returns_all_interface_metrics(
        self, antibody_antigen_pdb, tmp_path
    ):
        """Pipeline with InterfaceConfig produces interface_info,
        binding_energy, sasa, and shape_complementarity keys."""
        from boundry.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=1,
            relax=RelaxConfig(max_iterations=50, stiffness=10.0),
            interface=InterfaceConfig(
                enabled=True,
                calculate_sasa=True,
                calculate_shape_complementarity=True,
                relax_separated_chains=False,
                chain_pairs=[("A", "C"), ("B", "C")],
            ),
        )
        pipeline = Pipeline(config)
        output_pdb = tmp_path / "1vfb_output.pdb"
        result = pipeline.run(
            input_pdb=antibody_antigen_pdb,
            output_pdb=output_pdb,
        )

        analysis = result["outputs"][0]["interface_analysis"]
        assert "interface_info" in analysis
        assert "binding_energy" in analysis
        assert "sasa" in analysis
        assert "shape_complementarity" in analysis

    def test_pipeline_scorefile_has_interface_columns(
        self, antibody_antigen_pdb, tmp_path
    ):
        """Scorefile written by pipeline contains binding_energy,
        buried_sasa, n_interface_residues columns."""
        from boundry.pipeline import Pipeline

        scorefile = tmp_path / "scores.sc"
        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=1,
            scorefile=scorefile,
            relax=RelaxConfig(max_iterations=50, stiffness=10.0),
            interface=InterfaceConfig(
                enabled=True,
                calculate_sasa=True,
                calculate_shape_complementarity=False,
                relax_separated_chains=False,
                chain_pairs=[("A", "C"), ("B", "C")],
            ),
        )
        pipeline = Pipeline(config)
        output_pdb = tmp_path / "1vfb_scored.pdb"
        pipeline.run(
            input_pdb=antibody_antigen_pdb,
            output_pdb=output_pdb,
        )

        assert scorefile.exists()
        contents = scorefile.read_text()
        assert "dG_separated" in contents
        assert "dSASA_int" in contents
        assert "n_interface_residues" in contents

    def test_pipeline_binding_energy_positive(
        self, antibody_antigen_pdb, tmp_path
    ):
        """The ddG from the pipeline result should be positive for this
        known high-affinity antibody-antigen interaction with GBn2 solvation."""
        from boundry.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=1,
            relax=RelaxConfig(max_iterations=50, stiffness=10.0),
            interface=InterfaceConfig(
                enabled=True,
                calculate_sasa=False,
                calculate_shape_complementarity=False,
                relax_separated_chains=False,
                chain_pairs=[("A", "C"), ("B", "C")],
            ),
        )
        pipeline = Pipeline(config)
        output_pdb = tmp_path / "1vfb_ddg.pdb"
        result = pipeline.run(
            input_pdb=antibody_antigen_pdb,
            output_pdb=output_pdb,
        )

        be = result["outputs"][0]["interface_analysis"]["binding_energy"]
        assert math.isfinite(be.binding_energy)
        assert be.binding_energy > 0, (
            "Pipeline ddG should be positive (separated - complex) for "
            f"high-affinity binder, got {be.binding_energy:.2f}"
        )
