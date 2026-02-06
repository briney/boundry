"""Tests for invocation-aware operation runners."""

from pathlib import Path

import pytest

from boundry.interface_position_energetics import (
    PositionResult,
    PositionRow,
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


def _make_interface_result(per_position=True, alanine_scan=False):
    """Helper to build an InterfaceAnalysisResult with optional fields."""
    pp = (
        PositionResult(
            rows=[
                PositionRow(
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
        if per_position
        else None
    )
    ala = (
        PositionResult(
            rows=[
                PositionRow(
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
        if alanine_scan
        else None
    )
    return InterfaceAnalysisResult(per_position=pp, alanine_scan=ala)


def test_run_interface_operation_writes_json_and_per_position_csv(tmp_path):
    def _op(_structure):
        return _make_interface_result(per_position=True)

    summary = tmp_path / "interface.json"
    result, outputs = run_interface_operation(
        operation=_op,
        structure="ignored",
        output=summary,
        include_per_position_csv=True,
    )

    assert isinstance(result, InterfaceAnalysisResult)
    assert outputs.summary_json == summary
    assert outputs.per_position_csv == tmp_path / "interface_per_position.csv"
    assert outputs.alanine_scan_csv is None
    assert outputs.summary_json.exists()
    assert outputs.per_position_csv.exists()


def test_run_interface_operation_writes_json_and_alanine_scan_csv(tmp_path):
    def _op(_structure):
        return _make_interface_result(alanine_scan=True)

    summary = tmp_path / "interface.json"
    result, outputs = run_interface_operation(
        operation=_op,
        structure="ignored",
        output=summary,
        include_alanine_scan_csv=True,
    )

    assert isinstance(result, InterfaceAnalysisResult)
    assert outputs.summary_json == summary
    assert outputs.alanine_scan_csv == (
        tmp_path / "interface_alanine_scan.csv"
    )
    assert outputs.per_position_csv is None
    assert outputs.summary_json.exists()
    assert outputs.alanine_scan_csv.exists()


def test_run_interface_operation_writes_both_csvs(tmp_path):
    def _op(_structure):
        return _make_interface_result(per_position=True, alanine_scan=True)

    summary = tmp_path / "interface.json"
    result, outputs = run_interface_operation(
        operation=_op,
        structure="ignored",
        output=summary,
        include_per_position_csv=True,
        include_alanine_scan_csv=True,
    )

    assert outputs.summary_json == summary
    assert outputs.per_position_csv == tmp_path / "interface_per_position.csv"
    assert outputs.alanine_scan_csv == (
        tmp_path / "interface_alanine_scan.csv"
    )
    assert outputs.summary_json.exists()
    assert outputs.per_position_csv.exists()
    assert outputs.alanine_scan_csv.exists()


def test_run_interface_operation_per_position_csv_only(tmp_path):
    def _op(_structure):
        return _make_interface_result(per_position=True)

    csv_path = tmp_path / "per_position.csv"
    _, outputs = run_interface_operation(
        operation=_op,
        structure="ignored",
        per_position_csv=csv_path,
    )

    assert outputs.summary_json is None
    assert outputs.per_position_csv == csv_path
    assert outputs.alanine_scan_csv is None
    assert csv_path.exists()


def test_run_interface_operation_alanine_scan_csv_only(tmp_path):
    def _op(_structure):
        return _make_interface_result(alanine_scan=True)

    csv_path = tmp_path / "alanine_scan.csv"
    _, outputs = run_interface_operation(
        operation=_op,
        structure="ignored",
        alanine_scan_csv=csv_path,
    )

    assert outputs.summary_json is None
    assert outputs.per_position_csv is None
    assert outputs.alanine_scan_csv == csv_path
    assert csv_path.exists()
