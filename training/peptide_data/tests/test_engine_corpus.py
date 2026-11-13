import hashlib

import pytest
from peptidecore import from_helm
from rdkit import Chem

from training.peptide_data.engine_corpus import _worker, build_corpus


def test_layout_artifacts_preserve_chemistry_and_coordinates(tmp_path):
    helm = "PEPTIDE1{A.G.G.G.G.G}$PEPTIDE1,PEPTIDE1,6:R2-1:R1$$$V2.0"
    chemical = from_helm(helm)
    for name in ("svg", "mol"):
        (tmp_path / name).mkdir()
    job = (0, chemical.smiles, helm, "test", 42, 15, str(tmp_path))
    first, second = _worker(job), _worker(job)
    assert first["status"] == second["status"] == "accepted"
    assert first["record_key"] == second["record_key"]
    assert not first["training_admitted"]
    for name in ("svg", "mol"):
        path = tmp_path / name / first[name]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == first[name + "_sha256"]
    molecule = Chem.MolFromMolFile(str(tmp_path / "mol" / first["mol"]))
    assert Chem.MolToSmiles(molecule) == chemical.smiles
    assert molecule.GetNumConformers() == 1
    assert not molecule.GetConformer().Is3D()
    assert _worker((0, "CC", helm, "test", 42, 15, str(tmp_path)))["status"] == "failed"


def test_existing_corpus_cannot_be_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        build_corpus(
            out=tmp_path,
            seed=1,
            per_stratum=1,
            strata=(),
            workers=1,
            deadline_seconds=1,
        )
