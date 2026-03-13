"""Unit tests for boundry.relaxer ddG pipeline methods.

Tests use mocked OpenMM dependencies so they run without heavy
computational backends.
"""

import io
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest


# ----------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------


@pytest.fixture
def two_chain_pdb():
    """A minimal two-chain PDB for rigid-body separation tests.

    Chain A: 3 residues near origin.
    Chain B: 3 residues offset by ~20 Å along X.
    """
    # fmt: off
    return (
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C\n"
        "ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O\n"
        "ATOM      5  CB  ALA A   1       1.986  -0.760  -1.216  1.00  0.00           C\n"
        "ATOM      6  N   ALA A   2       3.326   1.540   0.000  1.00  0.00           N\n"
        "ATOM      7  CA  ALA A   2       3.941   2.861   0.000  1.00  0.00           C\n"
        "ATOM      8  C   ALA A   2       5.459   2.789   0.000  1.00  0.00           C\n"
        "ATOM      9  O   ALA A   2       6.065   1.719   0.000  1.00  0.00           O\n"
        "ATOM     10  CB  ALA A   2       3.473   3.699   1.186  1.00  0.00           C\n"
        "ATOM     11  N   ALA A   3       6.063   3.970   0.000  1.00  0.00           N\n"
        "ATOM     12  CA  ALA A   3       7.510   4.096   0.000  1.00  0.00           C\n"
        "ATOM     13  C   ALA A   3       8.061   5.516   0.000  1.00  0.00           C\n"
        "ATOM     14  O   ALA A   3       7.298   6.486   0.000  1.00  0.00           O\n"
        "ATOM     15  CB  ALA A   3       8.038   3.336  -1.216  1.00  0.00           C\n"
        "TER      16      ALA A   3\n"
        "ATOM     17  N   ALA B   1      20.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM     18  CA  ALA B   1      21.458   0.000   0.000  1.00  0.00           C\n"
        "ATOM     19  C   ALA B   1      22.009   1.420   0.000  1.00  0.00           C\n"
        "ATOM     20  O   ALA B   1      21.246   2.390   0.000  1.00  0.00           O\n"
        "ATOM     21  CB  ALA B   1      21.986  -0.760  -1.216  1.00  0.00           C\n"
        "ATOM     22  N   ALA B   2      23.326   1.540   0.000  1.00  0.00           N\n"
        "ATOM     23  CA  ALA B   2      23.941   2.861   0.000  1.00  0.00           C\n"
        "ATOM     24  C   ALA B   2      25.459   2.789   0.000  1.00  0.00           C\n"
        "ATOM     25  O   ALA B   2      26.065   1.719   0.000  1.00  0.00           O\n"
        "ATOM     26  CB  ALA B   2      23.473   3.699   1.186  1.00  0.00           C\n"
        "ATOM     27  N   ALA B   3      26.063   3.970   0.000  1.00  0.00           N\n"
        "ATOM     28  CA  ALA B   3      27.510   4.096   0.000  1.00  0.00           C\n"
        "ATOM     29  C   ALA B   3      28.061   5.516   0.000  1.00  0.00           C\n"
        "ATOM     30  O   ALA B   3      27.298   6.486   0.000  1.00  0.00           O\n"
        "ATOM     31  CB  ALA B   3      28.038   3.336  -1.216  1.00  0.00           C\n"
        "TER      32      ALA B   3\n"
        "END\n"
    )
    # fmt: on


@pytest.fixture
def three_chain_pdb(two_chain_pdb):
    """A three-chain PDB (A, B, C) for multi-chain group tests."""
    extra = (
        "ATOM     33  N   ALA C   1      40.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM     34  CA  ALA C   1      41.458   0.000   0.000  1.00  0.00           C\n"
        "ATOM     35  C   ALA C   1      42.009   1.420   0.000  1.00  0.00           C\n"
        "ATOM     36  O   ALA C   1      41.246   2.390   0.000  1.00  0.00           O\n"
        "TER      37      ALA C   1\n"
        "END\n"
    )
    # Replace END with chain C + END
    return two_chain_pdb.replace("END\n", extra)


# ----------------------------------------------------------------
# Helpers for parsing PDB coordinates
# ----------------------------------------------------------------


def _parse_atom_coords(pdb_string, chain_filter=None):
    """Return list of (chain, resnum, atom_name, x, y, z) tuples."""
    records = []
    for line in pdb_string.splitlines():
        if not line.startswith("ATOM"):
            continue
        chain = line[21]
        if chain_filter and chain not in chain_filter:
            continue
        resnum = int(line[22:26].strip())
        atom = line[12:16].strip()
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        records.append((chain, resnum, atom, x, y, z))
    return records


def _extract_ca_coords(pdb_string, chain_id):
    """Return Nx3 array of CA coords for *chain_id*."""
    coords = []
    for chain, _rn, atom, x, y, z in _parse_atom_coords(pdb_string):
        if chain == chain_id and atom == "CA":
            coords.append([x, y, z])
    return np.array(coords)


# ================================================================
# Tests for separate_interface_rigid_body (module-level function)
# ================================================================


class TestSeparateInterfaceRigidBody:
    """Tests for the pure-coordinate rigid-body separation."""

    def test_second_group_translated(self, two_chain_pdb):
        from boundry.relaxer import separate_interface_rigid_body

        result = separate_interface_rigid_body(
            two_chain_pdb, [["A"], ["B"]], separation_distance=100.0
        )

        orig_b = _extract_ca_coords(two_chain_pdb, "B")
        new_b = _extract_ca_coords(result, "B")

        # The translation should be roughly 100 Å
        deltas = new_b - orig_b
        per_atom_dist = np.sqrt(np.sum(deltas**2, axis=1))
        np.testing.assert_allclose(
            per_atom_dist, 100.0, atol=0.01
        )

    def test_first_group_unchanged(self, two_chain_pdb):
        from boundry.relaxer import separate_interface_rigid_body

        result = separate_interface_rigid_body(
            two_chain_pdb, [["A"], ["B"]], separation_distance=100.0
        )

        orig_a = _extract_ca_coords(two_chain_pdb, "A")
        new_a = _extract_ca_coords(result, "A")
        np.testing.assert_allclose(new_a, orig_a, atol=1e-3)

    def test_internal_distances_preserved(self, two_chain_pdb):
        """Pairwise distances within group B must be identical."""
        from boundry.relaxer import separate_interface_rigid_body

        result = separate_interface_rigid_body(
            two_chain_pdb, [["A"], ["B"]], separation_distance=50.0
        )

        orig_b = _extract_ca_coords(two_chain_pdb, "B")
        new_b = _extract_ca_coords(result, "B")

        from scipy.spatial.distance import pdist

        np.testing.assert_allclose(pdist(new_b), pdist(orig_b), atol=1e-3)

    def test_separation_distance_parameter(self, two_chain_pdb):
        from boundry.relaxer import separate_interface_rigid_body

        for sep in [50.0, 200.0]:
            result = separate_interface_rigid_body(
                two_chain_pdb, [["A"], ["B"]], separation_distance=sep
            )
            orig_b = _extract_ca_coords(two_chain_pdb, "B")
            new_b = _extract_ca_coords(result, "B")
            deltas = new_b - orig_b
            per_atom_dist = np.sqrt(np.sum(deltas**2, axis=1))
            np.testing.assert_allclose(per_atom_dist, sep, atol=0.01)

    def test_multi_chain_groups(self, three_chain_pdb):
        """Group 1 = [A, B], Group 2 = [C]."""
        from boundry.relaxer import separate_interface_rigid_body

        result = separate_interface_rigid_body(
            three_chain_pdb,
            [["A", "B"], ["C"]],
            separation_distance=100.0,
        )

        # A and B should be untouched
        orig_a = _parse_atom_coords(three_chain_pdb, {"A"})
        new_a = _parse_atom_coords(result, {"A"})
        for (_, _, _, ox, oy, oz), (_, _, _, nx, ny, nz) in zip(
            orig_a, new_a
        ):
            assert abs(ox - nx) < 1e-3
            assert abs(oy - ny) < 1e-3
            assert abs(oz - nz) < 1e-3

        orig_b = _parse_atom_coords(three_chain_pdb, {"B"})
        new_b = _parse_atom_coords(result, {"B"})
        for (_, _, _, ox, oy, oz), (_, _, _, nx, ny, nz) in zip(
            orig_b, new_b
        ):
            assert abs(ox - nx) < 1e-3
            assert abs(oy - ny) < 1e-3
            assert abs(oz - nz) < 1e-3

        # C should be translated
        orig_c_ca = _extract_ca_coords(three_chain_pdb, "C")
        new_c_ca = _extract_ca_coords(result, "C")
        shift = np.sqrt(np.sum((new_c_ca - orig_c_ca) ** 2, axis=1))
        np.testing.assert_allclose(shift, 100.0, atol=0.01)

    def test_rejects_wrong_group_count(self, two_chain_pdb):
        from boundry.relaxer import separate_interface_rigid_body

        with pytest.raises(ValueError, match="exactly 2"):
            separate_interface_rigid_body(
                two_chain_pdb, [["A"], ["B"], ["C"]]
            )

    def test_no_op_when_no_ca_atoms(self):
        """If no CA atoms exist, returns input unchanged."""
        from boundry.relaxer import separate_interface_rigid_body

        pdb = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000"
            "  1.00  0.00           N\n"
            "END\n"
        )
        result = separate_interface_rigid_body(
            pdb, [["A"], ["B"]], separation_distance=100.0
        )
        assert result == pdb


# ================================================================
# Tests for _make_force_field
# ================================================================


class TestMakeForceField:
    """Tests for the centralised force field / solvation helper."""

    @patch("boundry.relaxer.openmm_app")
    def test_implicit_solvent_from_config(self, mock_app):
        """When config.implicit_solvent=True, selects gbn2."""
        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(RelaxConfig(implicit_solvent=True))
        mock_ff = MagicMock()
        mock_app.ForceField.return_value = mock_ff

        ff, kwargs = relaxer._make_force_field()

        mock_app.ForceField.assert_called_once_with(
            "amber14-all.xml", "implicit/gbn2.xml"
        )
        assert ff is mock_ff

    @patch("boundry.relaxer.openmm_app")
    def test_explicit_solvent_from_config(self, mock_app):
        """When config.implicit_solvent=False, selects tip3pfb."""
        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(RelaxConfig(implicit_solvent=False))
        mock_ff = MagicMock()
        mock_app.ForceField.return_value = mock_ff

        ff, kwargs = relaxer._make_force_field()

        mock_app.ForceField.assert_called_once_with(
            "amber14-all.xml", "amber14/tip3pfb.xml"
        )

    @patch("boundry.relaxer.openmm_app")
    def test_explicit_override_beats_config(self, mock_app):
        """Passing implicit_solvent=False overrides config=True."""
        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(RelaxConfig(implicit_solvent=True))
        mock_app.ForceField.return_value = MagicMock()

        relaxer._make_force_field(implicit_solvent=False)

        mock_app.ForceField.assert_called_once_with(
            "amber14-all.xml", "amber14/tip3pfb.xml"
        )

    @patch("boundry.relaxer.openmm_app")
    def test_implicit_override_beats_config(self, mock_app):
        """Passing implicit_solvent=True overrides config=False."""
        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(RelaxConfig(implicit_solvent=False))
        mock_app.ForceField.return_value = MagicMock()

        relaxer._make_force_field(implicit_solvent=True)

        mock_app.ForceField.assert_called_once_with(
            "amber14-all.xml", "implicit/gbn2.xml"
        )

    @patch("boundry.relaxer.openmm_app")
    def test_dielectric_params_when_implicit(self, mock_app):
        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(RelaxConfig(implicit_solvent=True))
        mock_app.ForceField.return_value = MagicMock()

        _, kwargs = relaxer._make_force_field()

        assert kwargs["soluteDielectric"] == 1.0
        assert kwargs["solventDielectric"] == 78.5

    @patch("boundry.relaxer.openmm_app")
    def test_no_dielectric_params_when_explicit(self, mock_app):
        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(RelaxConfig(implicit_solvent=False))
        mock_app.ForceField.return_value = MagicMock()

        _, kwargs = relaxer._make_force_field()

        assert "soluteDielectric" not in kwargs
        assert "solventDielectric" not in kwargs

    @patch("boundry.relaxer.openmm_app")
    def test_hbonds_constraint_always_present(self, mock_app):
        from openmm import app as real_app

        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        for implicit in (True, False):
            mock_app.reset_mock()
            mock_app.HBonds = real_app.HBonds
            mock_app.ForceField.return_value = MagicMock()

            relaxer = Relaxer(RelaxConfig(implicit_solvent=implicit))
            _, kwargs = relaxer._make_force_field()

            assert kwargs["constraints"] is real_app.HBonds


# ================================================================
# Tests for _relax_unconstrained force field selection
# ================================================================


class TestRelaxUnconstrainedForceField:
    """Verify _relax_unconstrained uses _make_force_field."""

    @patch("boundry.relaxer.openmm_app")
    @patch("boundry.relaxer.openmm")
    @patch("boundry.relaxer.PDBFixer")
    def test_implicit_solvent_used(
        self, mock_fixer_cls, mock_openmm, mock_app
    ):
        """With implicit_solvent=True, _relax_unconstrained uses gbn2."""
        from openmm import unit

        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(RelaxConfig(implicit_solvent=True))
        relaxer._use_gpu = False

        mock_fixer = MagicMock()
        mock_fixer_cls.return_value = mock_fixer

        mock_ff = MagicMock()
        mock_modeller = MagicMock()
        mock_app.ForceField.return_value = mock_ff
        mock_ff.createSystem.return_value = MagicMock()
        mock_app.Modeller.return_value = mock_modeller

        mock_sim = MagicMock()
        mock_app.Simulation.return_value = mock_sim
        mock_state = MagicMock()
        mock_state.getPotentialEnergy.return_value = unit.Quantity(
            -100.0, unit.kilocalories_per_mole
        )
        mock_state.getPositions.return_value = unit.Quantity(
            np.array([[0.0, 0.0, 0.0]]), unit.angstroms
        )
        mock_sim.context.getState.return_value = mock_state
        mock_app.PDBFile.writeFile.side_effect = (
            lambda t, p, o: o.write("END\n")
        )

        relaxer._relax_unconstrained("ATOM dummy\nEND\n")

        mock_app.ForceField.assert_called_once_with(
            "amber14-all.xml", "implicit/gbn2.xml"
        )
        create_kwargs = mock_ff.createSystem.call_args[1]
        assert "soluteDielectric" in create_kwargs
        assert "solventDielectric" in create_kwargs

    @patch("boundry.relaxer.openmm_app")
    @patch("boundry.relaxer.openmm")
    @patch("boundry.relaxer.PDBFixer")
    def test_explicit_solvent_used(
        self, mock_fixer_cls, mock_openmm, mock_app
    ):
        """With implicit_solvent=False, _relax_unconstrained uses tip3pfb."""
        from openmm import unit

        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(RelaxConfig(implicit_solvent=False))
        relaxer._use_gpu = False

        mock_fixer = MagicMock()
        mock_fixer_cls.return_value = mock_fixer

        mock_ff = MagicMock()
        mock_modeller = MagicMock()
        mock_app.ForceField.return_value = mock_ff
        mock_ff.createSystem.return_value = MagicMock()
        mock_app.Modeller.return_value = mock_modeller

        mock_sim = MagicMock()
        mock_app.Simulation.return_value = mock_sim
        mock_state = MagicMock()
        mock_state.getPotentialEnergy.return_value = unit.Quantity(
            -100.0, unit.kilocalories_per_mole
        )
        mock_state.getPositions.return_value = unit.Quantity(
            np.array([[0.0, 0.0, 0.0]]), unit.angstroms
        )
        mock_sim.context.getState.return_value = mock_state
        mock_app.PDBFile.writeFile.side_effect = (
            lambda t, p, o: o.write("END\n")
        )

        relaxer._relax_unconstrained("ATOM dummy\nEND\n")

        mock_app.ForceField.assert_called_once_with(
            "amber14-all.xml", "amber14/tip3pfb.xml"
        )
        create_kwargs = mock_ff.createSystem.call_args[1]
        assert "soluteDielectric" not in create_kwargs


# ================================================================
# Tests for _relax_direct force field selection
# ================================================================


class TestRelaxDirectForceField:
    """Verify _relax_direct uses _make_force_field."""

    @patch("boundry.relaxer.openmm_app")
    @patch("boundry.relaxer.openmm")
    def test_implicit_solvent_used(self, mock_openmm, mock_app):
        """With implicit_solvent=True, _relax_direct uses gbn2."""
        from openmm import unit

        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(RelaxConfig(implicit_solvent=True))
        relaxer._use_gpu = False

        mock_pdb = MagicMock()
        mock_app.PDBFile.return_value = mock_pdb

        mock_ff = MagicMock()
        mock_modeller = MagicMock()
        mock_app.ForceField.return_value = mock_ff
        mock_ff.createSystem.return_value = MagicMock()
        mock_app.Modeller.return_value = mock_modeller

        mock_sim = MagicMock()
        mock_app.Simulation.return_value = mock_sim
        mock_state = MagicMock()
        mock_state.getPotentialEnergy.return_value = unit.Quantity(
            -100.0, unit.kilocalories_per_mole
        )
        mock_state.getPositions.return_value = unit.Quantity(
            np.array([[0.0, 0.0, 0.0]]), unit.angstroms
        )
        mock_sim.context.getState.return_value = mock_state

        # writeFile is called at end to produce output
        def write_pdb(t, p, o):
            o.write("END\n")

        mock_app.PDBFile.writeFile = MagicMock(side_effect=write_pdb)

        relaxer._relax_direct("ATOM dummy\nEND\n")

        mock_app.ForceField.assert_called_once_with(
            "amber14-all.xml", "implicit/gbn2.xml"
        )
        create_kwargs = mock_ff.createSystem.call_args[1]
        assert "soluteDielectric" in create_kwargs
        assert "solventDielectric" in create_kwargs

    @patch("boundry.relaxer.openmm_app")
    @patch("boundry.relaxer.openmm")
    def test_explicit_solvent_used(self, mock_openmm, mock_app):
        """With implicit_solvent=False, _relax_direct uses tip3pfb."""
        from openmm import unit

        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(RelaxConfig(implicit_solvent=False))
        relaxer._use_gpu = False

        mock_pdb = MagicMock()
        mock_app.PDBFile.return_value = mock_pdb

        mock_ff = MagicMock()
        mock_modeller = MagicMock()
        mock_app.ForceField.return_value = mock_ff
        mock_ff.createSystem.return_value = MagicMock()
        mock_app.Modeller.return_value = mock_modeller

        mock_sim = MagicMock()
        mock_app.Simulation.return_value = mock_sim
        mock_state = MagicMock()
        mock_state.getPotentialEnergy.return_value = unit.Quantity(
            -100.0, unit.kilocalories_per_mole
        )
        mock_state.getPositions.return_value = unit.Quantity(
            np.array([[0.0, 0.0, 0.0]]), unit.angstroms
        )
        mock_sim.context.getState.return_value = mock_state

        def write_pdb(t, p, o):
            o.write("END\n")

        mock_app.PDBFile.writeFile = MagicMock(side_effect=write_pdb)

        relaxer._relax_direct("ATOM dummy\nEND\n")

        mock_app.ForceField.assert_called_once_with(
            "amber14-all.xml", "amber14/tip3pfb.xml"
        )
        create_kwargs = mock_ff.createSystem.call_args[1]
        assert "soluteDielectric" not in create_kwargs


# ================================================================
# Tests for _build_system_for_ddg (mocked OpenMM)
# ================================================================


class TestBuildSystemForDdg:
    """Tests for the shared system construction method."""

    @patch("boundry.relaxer.openmm_app")
    def test_implicit_solvent_selects_gbn2(self, mock_app):
        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(RelaxConfig())

        mock_topology = MagicMock()
        mock_positions = MagicMock()
        mock_system = MagicMock()
        mock_modeller = MagicMock()

        mock_ff = MagicMock()
        mock_app.ForceField.return_value = mock_ff
        mock_ff.createSystem.return_value = mock_system
        mock_app.Modeller.return_value = mock_modeller

        system, modeller = relaxer._build_system_for_ddg(
            mock_topology, mock_positions, implicit_solvent=True
        )

        mock_app.ForceField.assert_called_once_with(
            "amber14-all.xml", "implicit/gbn2.xml"
        )
        assert system is mock_system
        assert modeller is mock_modeller

    @patch("boundry.relaxer.openmm_app")
    def test_explicit_solvent_selects_tip3pfb(self, mock_app):
        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(RelaxConfig())

        mock_ff = MagicMock()
        mock_app.ForceField.return_value = mock_ff
        mock_ff.createSystem.return_value = MagicMock()
        mock_app.Modeller.return_value = MagicMock()

        relaxer._build_system_for_ddg(
            MagicMock(), MagicMock(), implicit_solvent=False
        )

        mock_app.ForceField.assert_called_once_with(
            "amber14-all.xml", "amber14/tip3pfb.xml"
        )

    @patch("boundry.relaxer.openmm_app")
    def test_returns_system_modeller_tuple(self, mock_app):
        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(RelaxConfig())

        mock_system = MagicMock(name="system")
        mock_modeller = MagicMock(name="modeller")
        mock_ff = MagicMock()
        mock_app.ForceField.return_value = mock_ff
        mock_ff.createSystem.return_value = mock_system
        mock_app.Modeller.return_value = mock_modeller

        result = relaxer._build_system_for_ddg(
            MagicMock(), MagicMock()
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] is mock_system
        assert result[1] is mock_modeller

    @patch("boundry.relaxer.openmm_app")
    def test_hbonds_constraints_applied(self, mock_app):
        from openmm import app as real_app

        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(RelaxConfig())

        mock_ff = MagicMock()
        mock_app.ForceField.return_value = mock_ff
        mock_ff.createSystem.return_value = MagicMock()
        mock_app.Modeller.return_value = MagicMock()

        relaxer._build_system_for_ddg(MagicMock(), MagicMock())

        create_kwargs = mock_ff.createSystem.call_args[1]
        # The default parameter captures the real HBonds at import time
        assert create_kwargs["constraints"] is real_app.HBonds

    @patch("boundry.relaxer.openmm_app")
    def test_implicit_solvent_dielectric_params(self, mock_app):
        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(RelaxConfig())

        mock_ff = MagicMock()
        mock_app.ForceField.return_value = mock_ff
        mock_ff.createSystem.return_value = MagicMock()
        mock_app.Modeller.return_value = MagicMock()

        relaxer._build_system_for_ddg(
            MagicMock(), MagicMock(), implicit_solvent=True
        )

        create_kwargs = mock_ff.createSystem.call_args[1]
        assert create_kwargs["soluteDielectric"] == 1.0
        assert create_kwargs["solventDielectric"] == 78.5

    @patch("boundry.relaxer.openmm_app")
    def test_explicit_solvent_no_dielectric_params(self, mock_app):
        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(RelaxConfig())

        mock_ff = MagicMock()
        mock_app.ForceField.return_value = mock_ff
        mock_ff.createSystem.return_value = MagicMock()
        mock_app.Modeller.return_value = MagicMock()

        relaxer._build_system_for_ddg(
            MagicMock(), MagicMock(), implicit_solvent=False
        )

        create_kwargs = mock_ff.createSystem.call_args[1]
        assert "soluteDielectric" not in create_kwargs
        assert "solventDielectric" not in create_kwargs

    @patch("boundry.relaxer.openmm_app")
    def test_modeller_add_hydrogens_called(self, mock_app):
        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(RelaxConfig())

        mock_ff = MagicMock()
        mock_modeller = MagicMock()
        mock_app.ForceField.return_value = mock_ff
        mock_ff.createSystem.return_value = MagicMock()
        mock_app.Modeller.return_value = mock_modeller

        relaxer._build_system_for_ddg(MagicMock(), MagicMock())

        mock_modeller.addHydrogens.assert_called_once_with(mock_ff)


# ================================================================
# Tests for minimize_with_pair_restraints (mocked OpenMM)
# ================================================================


class TestMinimizeWithPairRestraints:
    """Tests for CA pair-restraint minimisation."""

    def _make_mock_relaxer(self):
        """Create a Relaxer with mocked internals."""
        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(RelaxConfig())
        relaxer._use_gpu = False
        return relaxer

    @patch("boundry.relaxer.openmm_app")
    @patch("boundry.relaxer.openmm")
    @patch("boundry.relaxer.PDBFixer")
    @patch("boundry.relaxer.filter_protein_only", side_effect=lambda x: x)
    @patch(
        "boundry.relaxer.detect_chain_gaps", return_value=[]
    )
    def test_returns_pdb_string(
        self,
        mock_gaps,
        mock_filter,
        mock_fixer_cls,
        mock_openmm,
        mock_app,
    ):
        from openmm import unit

        relaxer = self._make_mock_relaxer()

        # Set up mock fixer
        mock_fixer = MagicMock()
        mock_fixer_cls.return_value = mock_fixer

        # Set up mock modeller with fake CA positions
        mock_modeller = MagicMock()
        # Two CA atoms close together, one far away
        pos_data = [
            [0.1, 0.0, 0.0],  # atom 0 (CA)
            [0.2, 0.0, 0.0],  # atom 1 (CA) — 1nm from atom 0
            [5.0, 0.0, 0.0],  # atom 2 (CA) — far away
        ]
        mock_positions = [
            unit.Quantity(p, unit.nanometers) for p in pos_data
        ]
        mock_modeller.positions = mock_positions

        # Set up topology atoms
        atoms = []
        for i, name in enumerate(["CA", "CA", "CA"]):
            atom = MagicMock()
            atom.name = name
            atom.index = i
            atoms.append(atom)
        mock_modeller.topology.atoms.return_value = atoms

        # Mock system/simulation
        mock_system = MagicMock()
        mock_ff = MagicMock()
        mock_app.ForceField.return_value = mock_ff
        mock_ff.createSystem.return_value = mock_system
        mock_app.Modeller.return_value = mock_modeller

        mock_sim = MagicMock()
        mock_app.Simulation.return_value = mock_sim

        # State with positions
        mock_state = MagicMock()
        mock_state.getPositions.return_value = mock_positions
        mock_state.getPotentialEnergy.return_value = unit.Quantity(
            -100.0, unit.kilocalories_per_mole
        )
        mock_sim.context.getState.return_value = mock_state

        # Mock PDBFile.writeFile to produce output
        def write_file(topology, positions, output):
            output.write("ATOM      1  CA  ALA A   1\nEND\n")

        mock_app.PDBFile.writeFile.side_effect = write_file

        # Mock CustomBondForce
        mock_restraint = MagicMock()
        mock_openmm.CustomBondForce.return_value = mock_restraint

        result = relaxer.minimize_with_pair_restraints(
            "ATOM  dummy PDB\nEND\n",
            ca_cutoff=9.0,
            restraint_sd=0.5,
        )

        assert isinstance(result, str)
        assert "ATOM" in result

    @patch("boundry.relaxer.openmm_app")
    @patch("boundry.relaxer.openmm")
    @patch("boundry.relaxer.PDBFixer")
    @patch("boundry.relaxer.filter_protein_only", side_effect=lambda x: x)
    @patch(
        "boundry.relaxer.detect_chain_gaps", return_value=[]
    )
    def test_ca_pairs_within_cutoff(
        self,
        mock_gaps,
        mock_filter,
        mock_fixer_cls,
        mock_openmm,
        mock_app,
    ):
        """Verify that only CA pairs within cutoff get restraints."""
        from openmm import unit

        relaxer = self._make_mock_relaxer()

        mock_fixer = MagicMock()
        mock_fixer_cls.return_value = mock_fixer

        mock_modeller = MagicMock()
        # Three CA atoms: 0 and 1 are 5Å apart, 2 is 100Å away
        pos_data = [
            [0.0, 0.0, 0.0],  # CA 0
            [0.5, 0.0, 0.0],  # CA 1 (5 Å from 0)
            [10.0, 0.0, 0.0],  # CA 2 (100 Å from 0)
        ]
        mock_positions = [
            unit.Quantity(p, unit.nanometers) for p in pos_data
        ]
        mock_modeller.positions = mock_positions

        atoms = []
        for i in range(3):
            atom = MagicMock()
            atom.name = "CA"
            atom.index = i
            atoms.append(atom)
        mock_modeller.topology.atoms.return_value = atoms

        mock_system = MagicMock()
        mock_ff = MagicMock()
        mock_app.ForceField.return_value = mock_ff
        mock_ff.createSystem.return_value = mock_system
        mock_app.Modeller.return_value = mock_modeller

        mock_sim = MagicMock()
        mock_app.Simulation.return_value = mock_sim

        mock_state = MagicMock()
        mock_state.getPositions.return_value = mock_positions
        mock_state.getPotentialEnergy.return_value = unit.Quantity(
            -50.0, unit.kilocalories_per_mole
        )
        mock_sim.context.getState.return_value = mock_state

        mock_app.PDBFile.writeFile.side_effect = (
            lambda t, p, o: o.write("END\n")
        )

        mock_restraint = MagicMock()
        mock_openmm.CustomBondForce.return_value = mock_restraint

        relaxer.minimize_with_pair_restraints(
            "ATOM dummy\nEND\n",
            ca_cutoff=9.0,  # 0.9 nm
        )

        # With cutoff 9Å (0.9nm): pair (0,1) at 0.5nm is within,
        # pairs with atom 2 at 10nm are not.
        assert mock_restraint.addBond.call_count == 1

    @patch("boundry.relaxer.openmm_app")
    @patch("boundry.relaxer.openmm")
    @patch("boundry.relaxer.PDBFixer")
    @patch("boundry.relaxer.filter_protein_only", side_effect=lambda x: x)
    @patch(
        "boundry.relaxer.detect_chain_gaps", return_value=[]
    )
    def test_restraint_sd_affects_k(
        self,
        mock_gaps,
        mock_filter,
        mock_fixer_cls,
        mock_openmm,
        mock_app,
    ):
        """k = 1 / (sd_nm)^2 should change with restraint_sd."""
        from openmm import unit

        relaxer = self._make_mock_relaxer()

        mock_fixer = MagicMock()
        mock_fixer_cls.return_value = mock_fixer

        mock_modeller = MagicMock()
        mock_modeller.positions = [
            unit.Quantity([0.0, 0.0, 0.0], unit.nanometers)
        ]
        atom = MagicMock()
        atom.name = "CA"
        atom.index = 0
        mock_modeller.topology.atoms.return_value = [atom]

        mock_system = MagicMock()
        mock_ff = MagicMock()
        mock_app.ForceField.return_value = mock_ff
        mock_ff.createSystem.return_value = mock_system
        mock_app.Modeller.return_value = mock_modeller

        mock_sim = MagicMock()
        mock_app.Simulation.return_value = mock_sim
        mock_state = MagicMock()
        mock_state.getPositions.return_value = mock_modeller.positions
        mock_state.getPotentialEnergy.return_value = unit.Quantity(
            0.0, unit.kilocalories_per_mole
        )
        mock_sim.context.getState.return_value = mock_state
        mock_app.PDBFile.writeFile.side_effect = (
            lambda t, p, o: o.write("END\n")
        )

        mock_restraint = MagicMock()
        mock_openmm.CustomBondForce.return_value = mock_restraint

        relaxer.minimize_with_pair_restraints(
            "END\n", restraint_sd=0.5
        )

        # k = 1 / (0.5 * 0.1)^2 = 1 / 0.0025 = 400
        k_call = mock_restraint.addGlobalParameter.call_args
        assert k_call[0][0] == "k"
        np.testing.assert_allclose(k_call[0][1], 400.0, rtol=1e-6)


# ================================================================
# Tests for _build_mutation_neighborhood
# ================================================================


class TestBuildMutationNeighborhood:
    """Tests for the static mutation neighbourhood builder."""

    def _make_topology_and_positions(self, residues):
        """Build mock topology + positions for neighbourhood tests.

        *residues* is a list of (chain_id, resnum, atom_name, x, y, z)
        tuples.  Positions are in angstroms.
        """
        from openmm import unit

        atoms = []
        positions = []
        for i, (chain_id, resnum, atom_name, x, y, z) in enumerate(
            residues
        ):
            atom = MagicMock()
            atom.name = atom_name
            atom.index = i
            atom.residue = MagicMock()
            atom.residue.chain = MagicMock()
            atom.residue.chain.id = chain_id
            # OpenMM topology uses 0-based index; our method adds 1
            atom.residue.index = resnum - 1
            atoms.append(atom)
            positions.append(
                unit.Quantity(
                    [x / 10.0, y / 10.0, z / 10.0], unit.nanometers
                )
            )

        topology = MagicMock()
        topology.atoms.return_value = atoms
        return topology, positions

    def test_finds_nearby_residues(self):
        from boundry.relaxer import Relaxer

        # Residue 1 at origin (CB), residue 2 at 5Å (CB), residue 3
        # at 50Å
        topology, positions = self._make_topology_and_positions(
            [
                ("A", 1, "CA", 0.0, 0.0, 0.0),
                ("A", 1, "CB", 0.5, 0.0, 0.0),
                ("A", 2, "CA", 5.0, 0.0, 0.0),
                ("A", 2, "CB", 5.5, 0.0, 0.0),
                ("A", 3, "CA", 50.0, 0.0, 0.0),
                ("A", 3, "CB", 50.5, 0.0, 0.0),
            ]
        )

        result = Relaxer._build_mutation_neighborhood(
            topology,
            positions,
            mutation_sites=[("A", 1)],
            neighborhood_radius=8.0,
            sequence_window=0,
        )

        assert ("A", 1) in result
        assert ("A", 2) in result
        assert ("A", 3) not in result

    def test_sequence_window_expansion(self):
        from boundry.relaxer import Relaxer

        # Residues 1-5 on chain A; only residue 3 is within radius
        topology, positions = self._make_topology_and_positions(
            [
                ("A", 1, "CB", 50.0, 0.0, 0.0),
                ("A", 2, "CB", 40.0, 0.0, 0.0),
                ("A", 3, "CB", 5.0, 0.0, 0.0),
                ("A", 4, "CB", 40.0, 0.0, 0.0),
                ("A", 5, "CB", 50.0, 0.0, 0.0),
                ("B", 1, "CB", 0.0, 0.0, 0.0),  # mutation site
            ]
        )

        result = Relaxer._build_mutation_neighborhood(
            topology,
            positions,
            mutation_sites=[("B", 1)],
            neighborhood_radius=8.0,
            sequence_window=1,
        )

        # Residue A3 within 8Å; window=1 expands to A2 and A4
        assert ("A", 3) in result
        assert ("A", 2) in result
        assert ("A", 4) in result
        # A1 and A5 are too far and outside window
        assert ("A", 1) not in result
        assert ("A", 5) not in result

    def test_uses_cb_over_ca(self):
        """Should prefer CB atom, falling back to CA for GLY."""
        from boundry.relaxer import Relaxer

        # Residue 1 has both CA and CB; residue 2 only has CA (GLY)
        topology, positions = self._make_topology_and_positions(
            [
                ("A", 1, "CA", 0.0, 0.0, 0.0),
                ("A", 1, "CB", 1.0, 0.0, 0.0),  # should use this
                ("A", 2, "CA", 3.0, 0.0, 0.0),  # GLY, no CB
            ]
        )

        result = Relaxer._build_mutation_neighborhood(
            topology,
            positions,
            mutation_sites=[("A", 1)],
            neighborhood_radius=8.0,
            sequence_window=0,
        )

        # Both should be found: A1 is mutation site, A2 CB (CA) at
        # distance = |3 - 1| = 2Å from CB of A1
        assert ("A", 1) in result
        assert ("A", 2) in result

    def test_empty_mutation_sites(self):
        from boundry.relaxer import Relaxer

        topology, positions = self._make_topology_and_positions(
            [("A", 1, "CB", 0.0, 0.0, 0.0)]
        )

        result = Relaxer._build_mutation_neighborhood(
            topology, positions, mutation_sites=[]
        )
        assert result == set()

    def test_missing_mutation_site_atoms(self):
        """If mutation site has no atoms in topology, returns empty."""
        from boundry.relaxer import Relaxer

        topology, positions = self._make_topology_and_positions(
            [("A", 1, "CB", 0.0, 0.0, 0.0)]
        )

        result = Relaxer._build_mutation_neighborhood(
            topology,
            positions,
            mutation_sites=[("Z", 99)],  # doesn't exist
        )
        assert result == set()


# ================================================================
# Tests for generate_local_md_ensemble (mocked OpenMM)
# ================================================================


class TestGenerateLocalMdEnsemble:
    """Tests for local MD ensemble generation."""

    def _make_mock_relaxer(self):
        from boundry.config import RelaxConfig
        from boundry.relaxer import Relaxer

        relaxer = Relaxer(RelaxConfig())
        relaxer._use_gpu = False
        return relaxer

    @patch("boundry.relaxer.openmm_app")
    @patch("boundry.relaxer.openmm")
    @patch("boundry.relaxer.PDBFixer")
    @patch("boundry.relaxer.filter_protein_only", side_effect=lambda x: x)
    @patch(
        "boundry.relaxer.detect_chain_gaps", return_value=[]
    )
    def test_returns_correct_number_of_members(
        self,
        mock_gaps,
        mock_filter,
        mock_fixer_cls,
        mock_openmm,
        mock_app,
    ):
        from openmm import unit

        relaxer = self._make_mock_relaxer()

        mock_fixer = MagicMock()
        mock_fixer_cls.return_value = mock_fixer

        mock_modeller = MagicMock()
        pos = [unit.Quantity([0.0, 0.0, 0.0], unit.nanometers)]
        mock_modeller.positions = pos

        atom = MagicMock()
        atom.name = "CA"
        atom.index = 0
        atom.residue = MagicMock()
        atom.residue.chain = MagicMock()
        atom.residue.chain.id = "A"
        atom.residue.index = 0
        mock_modeller.topology.atoms.return_value = [atom]

        mock_ff = MagicMock()
        mock_app.ForceField.return_value = mock_ff
        mock_ff.createSystem.return_value = MagicMock()
        mock_app.Modeller.return_value = mock_modeller

        mock_restraint = MagicMock()
        mock_openmm.CustomBondForce.return_value = mock_restraint

        # Initial minimisation simulation
        mock_init_sim = MagicMock()
        mock_init_state = MagicMock()
        mock_init_state.getPositions.return_value = pos
        mock_init_sim.context.getState.return_value = mock_init_state

        # Per-member simulation
        mock_member_sim = MagicMock()
        mock_member_state = MagicMock()
        mock_member_state.getPositions.return_value = pos
        mock_member_sim.context.getState.return_value = mock_member_state

        # Simulation() calls: first is init, rest are per-member
        mock_app.Simulation.side_effect = [
            mock_init_sim
        ] + [mock_member_sim] * 5

        mock_app.PDBFile.writeFile.side_effect = (
            lambda t, p, o: o.write("ATOM  member PDB\nEND\n")
        )

        mock_openmm.LangevinMiddleIntegrator.return_value = MagicMock()

        result = relaxer.generate_local_md_ensemble(
            "ATOM dummy\nEND\n",
            mutation_sites=[("A", 1)],
            n_members=5,
        )

        assert len(result) == 5
        assert all(isinstance(m, str) for m in result)
        assert all("ATOM" in m for m in result)

    @patch("boundry.relaxer.openmm_app")
    @patch("boundry.relaxer.openmm")
    @patch("boundry.relaxer.PDBFixer")
    @patch("boundry.relaxer.filter_protein_only", side_effect=lambda x: x)
    @patch(
        "boundry.relaxer.detect_chain_gaps", return_value=[]
    )
    def test_deterministic_seeds(
        self,
        mock_gaps,
        mock_filter,
        mock_fixer_cls,
        mock_openmm,
        mock_app,
    ):
        """Per-member seed is base_seed * 100000 + i."""
        from openmm import unit

        relaxer = self._make_mock_relaxer()

        mock_fixer = MagicMock()
        mock_fixer_cls.return_value = mock_fixer

        mock_modeller = MagicMock()
        pos = [unit.Quantity([0.0, 0.0, 0.0], unit.nanometers)]
        mock_modeller.positions = pos
        atom = MagicMock()
        atom.name = "CA"
        atom.index = 0
        atom.residue = MagicMock()
        atom.residue.chain = MagicMock()
        atom.residue.chain.id = "A"
        atom.residue.index = 0
        mock_modeller.topology.atoms.return_value = [atom]

        mock_ff = MagicMock()
        mock_app.ForceField.return_value = mock_ff
        mock_ff.createSystem.return_value = MagicMock()
        mock_app.Modeller.return_value = mock_modeller

        mock_openmm.CustomBondForce.return_value = MagicMock()

        # Mock sims
        mock_init_sim = MagicMock()
        mock_init_state = MagicMock()
        mock_init_state.getPositions.return_value = pos
        mock_init_sim.context.getState.return_value = mock_init_state

        member_sims = [MagicMock() for _ in range(3)]
        for sim in member_sims:
            state = MagicMock()
            state.getPositions.return_value = pos
            sim.context.getState.return_value = state

        mock_app.Simulation.side_effect = [mock_init_sim] + member_sims

        mock_app.PDBFile.writeFile.side_effect = (
            lambda t, p, o: o.write("END\n")
        )

        mock_integrators = [MagicMock() for _ in range(3)]
        mock_openmm.LangevinMiddleIntegrator.side_effect = (
            mock_integrators
        )

        relaxer.generate_local_md_ensemble(
            "END\n",
            mutation_sites=[("A", 1)],
            n_members=3,
            seed=42,
        )

        # Check seeds: 42 * 100000 + i for i in 0..2
        for i, integrator in enumerate(mock_integrators):
            expected_seed = 42 * 100_000 + i
            integrator.setRandomNumberSeed.assert_called_once_with(
                expected_seed
            )

    @patch("boundry.relaxer.openmm_app")
    @patch("boundry.relaxer.openmm")
    @patch("boundry.relaxer.PDBFixer")
    @patch("boundry.relaxer.filter_protein_only", side_effect=lambda x: x)
    @patch(
        "boundry.relaxer.detect_chain_gaps", return_value=[]
    )
    def test_md_steps_executed(
        self,
        mock_gaps,
        mock_filter,
        mock_fixer_cls,
        mock_openmm,
        mock_app,
    ):
        """Verify equilibration + production steps are run."""
        from openmm import unit

        relaxer = self._make_mock_relaxer()

        mock_fixer = MagicMock()
        mock_fixer_cls.return_value = mock_fixer

        mock_modeller = MagicMock()
        pos = [unit.Quantity([0.0, 0.0, 0.0], unit.nanometers)]
        mock_modeller.positions = pos
        atom = MagicMock()
        atom.name = "CA"
        atom.index = 0
        atom.residue = MagicMock()
        atom.residue.chain = MagicMock()
        atom.residue.chain.id = "A"
        atom.residue.index = 0
        mock_modeller.topology.atoms.return_value = [atom]

        mock_ff = MagicMock()
        mock_app.ForceField.return_value = mock_ff
        mock_ff.createSystem.return_value = MagicMock()
        mock_app.Modeller.return_value = mock_modeller

        mock_openmm.CustomBondForce.return_value = MagicMock()

        mock_init_sim = MagicMock()
        mock_init_state = MagicMock()
        mock_init_state.getPositions.return_value = pos
        mock_init_sim.context.getState.return_value = mock_init_state

        mock_member_sim = MagicMock()
        mock_member_state = MagicMock()
        mock_member_state.getPositions.return_value = pos
        mock_member_sim.context.getState.return_value = mock_member_state

        mock_app.Simulation.side_effect = [
            mock_init_sim,
            mock_member_sim,
        ]
        mock_app.PDBFile.writeFile.side_effect = (
            lambda t, p, o: o.write("END\n")
        )
        mock_openmm.LangevinMiddleIntegrator.return_value = MagicMock()

        relaxer.generate_local_md_ensemble(
            "END\n",
            mutation_sites=[("A", 1)],
            n_members=1,
            md_equilibration_steps=1000,
            md_total_steps=5000,
        )

        # Should call step(1000) then step(5000) then minimizeEnergy
        step_calls = mock_member_sim.step.call_args_list
        assert step_calls[0] == call(1000)
        assert step_calls[1] == call(5000)
        mock_member_sim.minimizeEnergy.assert_called_once_with(
            maxIterations=100
        )
