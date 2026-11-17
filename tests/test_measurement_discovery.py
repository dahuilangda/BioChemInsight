"""Deterministic page discovery independent of completion order."""

import concurrent.futures

import fitz
import pytest

import constants
from utils import measurement_evidence


@pytest.fixture
def source_pdf(tmp_path):
    path = tmp_path / "source.pdf"
    with fitz.open() as document:
        for index in range(3):
            page = document.new_page()
            page.insert_text((50, 50), f"Page {index + 1}")
        document.save(path)
    return str(path)


def _read_page(image, task, *, metadata, **kwargs):
    page = metadata["page"]
    return {
        "has_result_records": page != 2,
        "confidence": "high",
        "endpoints": [
            {"column": f"page {page} endpoint"},
            {"column": "shared endpoint"},
        ],
        "evidence": f"page {page} result evidence",
    }


@pytest.mark.parametrize("reverse", [False, True])
def test_candidate_order_follows_source_pages(monkeypatch, source_pdf, reverse):
    monkeypatch.setattr(constants, "VISION_MODEL_CONCURRENCY", 3)
    monkeypatch.setattr(measurement_evidence, "read_evidence", _read_page)

    def completed(futures):
        ordered = list(futures)
        for future in ordered:
            future.result()
        return iter(reversed(ordered) if reverse else ordered)

    monkeypatch.setattr(concurrent.futures, "as_completed", completed)
    progress = []
    pages, names, decisions, markdowns = measurement_evidence.discover_pdf_measurements(
        source_pdf, [1, 2, 3], progress_callback=progress.append
    )
    assert pages == [1, 3]
    assert names == ["page 1 endpoint", "shared endpoint", "page 3 endpoint"]
    assert list(decisions) == list(markdowns) == [1, 2, 3]
    assert [event["current"] for event in progress] == [1, 2, 3]
    assert [decision["assay_names"] for decision in decisions.values()] == [
        [f"page {page} endpoint", "shared endpoint"] for page in (1, 2, 3)
    ]


def test_page_failure_aborts_discovery(monkeypatch, source_pdf):
    def fail(image, task, *, metadata, **kwargs):
        if metadata["page"] == 2:
            raise RuntimeError("unreadable page")
        return _read_page(image, task, metadata=metadata)

    monkeypatch.setattr(measurement_evidence, "read_evidence", fail)
    with pytest.raises(RuntimeError, match="pages \\[2\\]"):
        measurement_evidence.discover_pdf_measurements(source_pdf, [1, 2, 3])
