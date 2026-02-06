"""Tests for shared result serialization/output helpers."""

from pathlib import Path

from boundry.binding_energy import BindingEnergyResult
from boundry.interface import InterfaceInfo
from boundry.interface_position_energetics import (
    PerPositionResult,
    PerPositionRow,
)
from boundry.operations import InterfaceAnalysisResult
from boundry.result_io import (
    interface_result_to_dict,
    resolve_interface_output_paths,
    write_interface_csv,
    write_interface_json,
)


class TestResolveInterfaceOutputPaths:
    def test_json_file_output(self):
        summary, csv = resolve_interface_output_paths(
            "results/interface.json",
            include_position_csv=True,
        )
        assert summary == Path("results/interface.json")
        assert csv == Path("results/interface_positions.csv")

    def test_directory_output(self):
        summary, csv = resolve_interface_output_paths(
            "results",
            include_position_csv=True,
        )
        assert summary == Path("results/interface_analysis.json")
        assert csv == Path("results/interface_positions.csv")

    def test_invalid_file_extension_raises(self):
        import pytest

        with pytest.raises(ValueError, match="must be a .json file"):
            resolve_interface_output_paths(
                "results/interface.txt",
                include_position_csv=False,
            )


class TestInterfaceSerialization:
    def test_interface_result_to_dict(self):
        result = InterfaceAnalysisResult(
            interface_info=InterfaceInfo(chain_pairs=[("A", "B")]),
            binding_energy=BindingEnergyResult(binding_energy=-1.23),
            per_position=PerPositionResult(
                rows=[
                    PerPositionRow(
                        chain_id="A",
                        residue_number=10,
                        insertion_code="",
                        wt_resname="TYR",
                        partner_chain="B",
                        min_distance=3.4,
                        num_contacts=5,
                        delta_ddG=1.1,
                    )
                ],
                dG_wt=-5.0,
            ),
        )
        payload = interface_result_to_dict(result)

        assert payload["binding_energy"]["binding_energy"] == -1.23
        assert payload["interface_info"]["chain_pairs"] == [["A", "B"]]
        assert payload["per_position"]["rows"][0]["residue_number"] == 10

    def test_write_json_and_csv(self, tmp_path):
        result = InterfaceAnalysisResult(
            per_position=PerPositionResult(
                rows=[
                    PerPositionRow(
                        chain_id="A",
                        residue_number=1,
                        insertion_code="",
                        wt_resname="ALA",
                        partner_chain="B",
                        min_distance=4.2,
                        num_contacts=2,
                    )
                ]
            )
        )
        json_path = tmp_path / "interface.json"
        csv_path = tmp_path / "positions.csv"

        write_interface_json(result, json_path)
        write_interface_csv(result, csv_path)

        assert json_path.exists()
        assert csv_path.exists()
        assert "per_position" in json_path.read_text()
        assert "chain_id" in csv_path.read_text()
