import pytest

from training.peptide_data.artifacts import digest
from training.peptide_data.render_corpus import source_file


def test_source_artifacts_require_matching_hashes_and_corpus_paths(tmp_path):
    path = tmp_path / "drawing.mol"
    path.write_text("coordinate artifact")
    assert source_file(tmp_path, path.name, digest(path)) == path.resolve()
    with pytest.raises(ValueError, match="changed"):
        source_file(tmp_path, path.name, "invalid")
    with pytest.raises(ValueError, match="leaves"):
        source_file(tmp_path, "../drawing.mol", digest(path))
    link = tmp_path / "outside"
    link.symlink_to(tmp_path.parent)
    with pytest.raises(ValueError, match="leaves"):
        source_file(tmp_path, "outside/drawing.mol", digest(path))
