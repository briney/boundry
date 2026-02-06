"""Tests for invocation-aware operation runners."""

from pathlib import Path

import pytest

from boundry.interface_position_energetics import (
    PerPositionResult,
    PerPositionRow,
)
from boundry.invocation import InvocationMode, OutputPolicy, OutputRequirement
from boundry.operations import InterfaceAnalysisResult, Structure
from boundry.runner import run_interface_operation, run_structure_operation


_PDB = (
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000"
    "  1.00  0.00           N\nEND\n"
)


def test_run_structure_operation_writes_output(tmp_path):
    output = tmp_path / "out.pdb"

    def _op(_structure):
        return Structure(pdb_string=_PDB)

    result = run_structure_operation(
        name="idealize",
        operation=_op,
        structure="ignored",
        output=output,
        mode=InvocationMode.CLI,
        output_policy=OutputPolicy(OutputRequirement.REQUIRED),
    )

    assert isinstance(result, Structure)
    assert output.exists()


def test_run_structure_operation_requires_output():
    def _op(_structure):
        return Structure(pdb_string=_PDB)

    with pytest.raises(ValueError, match="requires an output path"):
        run_structure_operation(
            name="idealize",
            operation=_op,
            structure="ignored",
            output=None,
            mode=InvocationMode.CLI,
            output_policy=OutputPolicy(OutputRequirement.REQUIRED),
        )


def test_run_interface_operation_writes_json_and_csv(tmp_path):
    def _op(_structure):
        return InterfaceAnalysisResult(
            per_position=PerPositionResult(
                rows=[
                    PerPositionRow(
                        chain_id="A",
                        residue_number=1,
                        insertion_code="",
                        wt_resname="ALA",
                        partner_chain="B",
                        min_distance=4.0,
                        num_contacts=2,
                    )
                ]
            )
        )

    summary = tmp_path / "interface.json"
    result, outputs = run_interface_operation(
        operation=_op,
        structure="ignored",
        output=summary,
        include_position_csv=True,
    )

    assert isinstance(result, InterfaceAnalysisResult)
    assert outputs.summary_json == summary
    assert outputs.position_csv == tmp_path / "interface_positions.csv"
    assert outputs.summary_json.exists()
    assert outputs.position_csv.exists()


def test_run_interface_operation_position_csv_only(tmp_path):
    def _op(_structure):
        return InterfaceAnalysisResult(
            per_position=PerPositionResult(
                rows=[
                    PerPositionRow(
                        chain_id="A",
                        residue_number=1,
                        insertion_code="",
                        wt_resname="ALA",
                        partner_chain="B",
                        min_distance=4.0,
                        num_contacts=2,
                    )
                ]
            )
        )

    csv_path = tmp_path / "positions.csv"
    _, outputs = run_interface_operation(
        operation=_op,
        structure="ignored",
        position_csv=csv_path,
    )

    assert outputs.summary_json is None
    assert outputs.position_csv == csv_path
    assert csv_path.exists()
