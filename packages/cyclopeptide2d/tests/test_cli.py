import json

from cyclopeptide2d.cli import main

SMILES = "N1CC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC1=O"


def test_cli_writes_accepted_structure(tmp_path):
    assert main(["--smiles", SMILES, "--maxiter", "150", "--out", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "report.json").read_text())["qa_pass"]
    assert (tmp_path / "structure.mol").is_file()
    assert "<svg" in (tmp_path / "structure.svg").read_text()


def test_cli_invalid_input_writes_report_only(tmp_path):
    assert main(["--smiles", "NCC(=O)*", "--out", str(tmp_path)]) == 2
    report = json.loads((tmp_path / "report.json").read_text())
    assert report == {"qa_pass": False, "error": "input_contains_unresolved_atoms"}
    assert not (tmp_path / "structure.mol").exists()
    assert not (tmp_path / "structure.svg").exists()
