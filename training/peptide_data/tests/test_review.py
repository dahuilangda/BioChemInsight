from pathlib import Path

from training.peptide_data.design import build_design
from training.peptide_data.review import _review


def test_review_images_keep_atom_coordinates_and_remain_unapproved(tmp_path):
    candidate = build_design(per_stratum=1)["records"][0]
    result = _review((candidate, str(tmp_path), 17, 50, 30, [384], "rdkit"))
    assert result["status"] == "rendered_for_review"
    assert not result["training_release_approved"]
    assert len(result["images"]) == 2
    for style in result["images"]:
        assert len(style["atom_positions"]) == candidate["atom_count"]
        sample = style["inputs"][0]
        assert Path(sample["file"]).is_file()
        assert len(sample["atom_positions"]) == candidate["atom_count"]
        assert not sample["training_release_approved"]
        assert sample["glyph_review_required"]


def test_unsupported_topology_is_reported(tmp_path):
    candidate = next(
        row
        for row in build_design(per_stratum=1)["records"]
        if row["stratum"] == "multiple_closures"
    )
    result = _review((candidate, str(tmp_path), 17, 50, 30, [384], "residue"))
    assert result["status"] == "unsupported_topology"
    assert not list(tmp_path.iterdir())
