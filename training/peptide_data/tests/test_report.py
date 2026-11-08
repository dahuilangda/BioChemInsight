import hashlib

import pytest

from training.peptide_data.report import summarize


def test_review_report_checks_image_hashes(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"original")
    design = {"records": [{"id": "a"}]}
    manifest = {
        "sources_unchanged": True,
        "records": [
            {
                "id": "a",
                "stratum": "linear_short",
                "status": "rendered_for_review",
                "images": [
                    {
                        "inputs": [
                            {
                                "file": str(image),
                                "sha256": hashlib.sha256(b"original").hexdigest(),
                                "size": 384,
                                "resolution_screen_pass": False,
                            }
                        ]
                    }
                ],
            }
        ],
    }
    result = summarize(design, {"run": manifest})
    assert result["structures_with_review_images"] == 1
    assert result["resolution_screen"]["run"]["384"] == {"fail": 1}
    assert not result["training_release_approved"]
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        summarize(design, {"run": manifest})


def test_rejected_records_remain_in_coverage():
    design = {"records": [{"id": "a"}, {"id": "b"}]}
    manifest = {
        "sources_unchanged": True,
        "records": [
            {"id": "a", "stratum": "linear_short", "status": "geometry_rejected"},
            {"id": "b", "stratum": "head_to_tail_small", "status": "layout_rejected"},
        ],
    }
    result = summarize(design, {"run": manifest})
    assert result["structures_with_review_images"] == 0
    assert result["coverage"]["run"]["linear_short"] == {"geometry_rejected": 1}
    assert result["coverage"]["run"]["head_to_tail_small"] == {"layout_rejected": 1}
