"""Tests for shared result serialization/output helpers."""

from pathlib import Path

from boundry.binding_energy import BindingEnergyResult
from boundry.interface import InterfaceInfo
from boundry.interface_position_energetics import (
    PositionResult,
    PositionRow,
)
from boundry.operations import InterfaceAnalysisResult
from boundry.result_io import (
    interface_result_to_dict,
    resolve_interface_output_paths,
    write_interface_csv,
    write_interface_json,
)


class TestResolveInterfaceOutputPaths:
    def test_json_file_output_per_position(self):
        summary, pp_csv, ala_csv = resolve_interface_output_paths(
            "results/interface.json",
            include_per_position_csv=True,
        )
        assert summary == Path("results/interface.json")
        assert pp_csv == Path("results/interface_per_position.csv")
        assert ala_csv is None

    def test_json_file_output_alanine_scan(self):
        summary, pp_csv, ala_csv = resolve_interface_output_paths(
            "results/interface.json",
            include_alanine_scan_csv=True,
        )
        assert summary == Path("results/interface.json")
        assert pp_csv is None
        assert ala_csv == Path("results/interface_alanine_scan.csv")

    def test_json_file_output_both(self):
        summary, pp_csv, ala_csv = resolve_interface_output_paths(
            "results/interface.json",
            include_per_position_csv=True,
            include_alanine_scan_csv=True,
        )
        assert summary == Path("results/interface.json")
        assert pp_csv == Path("results/interface_per_position.csv")
        assert ala_csv == Path("results/interface_alanine_scan.csv")

    def test_directory_output(self):
        summary, pp_csv, ala_csv = resolve_interface_output_paths(
            "results",
            include_per_position_csv=True,
            include_alanine_scan_csv=True,
        )
        assert summary == Path("results/interface_analysis.json")
        assert pp_csv == Path("results/interface_per_position.csv")
        assert ala_csv == Path("results/interface_alanine_scan.csv")

    def test_explicit_csv_paths_override_defaults(self):
        summary, pp_csv, ala_csv = resolve_interface_output_paths(
            "results/interface.json",
            include_per_position_csv=True,
            include_alanine_scan_csv=True,
            per_position_csv="custom/pp.csv",
            alanine_scan_csv="custom/ala.csv",
        )
        assert summary == Path("results/interface.json")
        assert pp_csv == Path("custom/pp.csv")
        assert ala_csv == Path("custom/ala.csv")

    def test_no_csv_flags(self):
        summary, pp_csv, ala_csv = resolve_interface_output_paths(
            "results/interface.json",
        )
        assert summary == Path("results/interface.json")
        assert pp_csv is None
        assert ala_csv is None

    def test_invalid_file_extension_raises(self):
        import pytest

        with pytest.raises(ValueError, match="must be a .json file"):
            resolve_interface_output_paths(
                "results/interface.txt",
                include_per_position_csv=False,
            )


class TestInterfaceSerialization:
    def test_interface_result_to_dict(self):
        result = InterfaceAnalysisResult(
            interface_info=InterfaceInfo(chain_pairs=[("A", "B")]),
            binding_energy=BindingEnergyResult(binding_energy=-1.23),
            per_position=PositionResult(
                rows=[
                    PositionRow(
                        chain_id="A",
                        residue_number=10,
                        insertion_code="",
                        wt_resname="TYR",
                        partner_chain="B",
                        min_distance=3.4,
                        num_contacts=5,
                        ddG=1.1,
                    )
                ],
                dG_wt=-5.0,
            ),
            alanine_scan=PositionResult(
                rows=[
                    PositionRow(
                        chain_id="A",
                        residue_number=10,
                        insertion_code="",
                        wt_resname="TYR",
                        partner_chain="B",
                        min_distance=3.4,
                        num_contacts=5,
                        ddG=2.3,
                    )
                ],
                dG_wt=-5.0,
            ),
        )
        payload = interface_result_to_dict(result)

        assert payload["binding_energy"]["binding_energy"] == -1.23
        assert payload["interface_info"]["chain_pairs"] == [["A", "B"]]
        assert payload["per_position"]["rows"][0]["residue_number"] == 10
        assert payload["per_position"]["rows"][0]["ddG"] == 1.1
        assert payload["alanine_scan"]["rows"][0]["ddG"] == 2.3

    def test_write_json_and_csv(self, tmp_path):
        result = InterfaceAnalysisResult(
            per_position=PositionResult(
                rows=[
                    PositionRow(
                        chain_id="A",
                        residue_number=1,
                        insertion_code="",
                        wt_resname="ALA",
                        partner_chain="B",
                        min_distance=4.2,
                        num_contacts=2,
                    )
                ]
            ),
            alanine_scan=PositionResult(
                rows=[
                    PositionRow(
                        chain_id="A",
                        residue_number=1,
                        insertion_code="",
                        wt_resname="ALA",
                        partner_chain="B",
                        min_distance=4.2,
                        num_contacts=2,
                    )
                ]
            ),
        )
        json_path = tmp_path / "interface.json"
        pp_csv_path = tmp_path / "per_position.csv"
        ala_csv_path = tmp_path / "alanine_scan.csv"

        write_interface_json(result, json_path)
        pp_written, ala_written = write_interface_csv(
            result,
            per_position_path=pp_csv_path,
            alanine_scan_path=ala_csv_path,
        )

        assert json_path.exists()
        assert pp_written == pp_csv_path
        assert ala_written == ala_csv_path
        assert pp_csv_path.exists()
        assert ala_csv_path.exists()
        assert "per_position" in json_path.read_text()
        assert "alanine_scan" in json_path.read_text()
        assert "chain_id" in pp_csv_path.read_text()
        assert "chain_id" in ala_csv_path.read_text()

    def test_write_csv_returns_none_when_no_paths(self):
        result = InterfaceAnalysisResult(
            per_position=PositionResult(
                rows=[
                    PositionRow(
                        chain_id="A",
                        residue_number=1,
                        insertion_code="",
                        wt_resname="ALA",
                        partner_chain="B",
                        min_distance=4.2,
                        num_contacts=2,
                    )
                ]
            ),
        )
        pp_written, ala_written = write_interface_csv(result)
        assert pp_written is None
        assert ala_written is None
