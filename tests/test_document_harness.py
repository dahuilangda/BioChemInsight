"""Required document evidence and ledger persistence."""

import pytest

from services import document_ledger
from utils.document_harness import DocumentHarness
from utils.model_harness import (
    ModelContractError,
    ModelEvidenceError,
    ModelProviderError,
)
from utils.page_extractor import PageEntityExtractor


def _record(page):
    return {
        "page_num": page,
        "ok": True,
        "compounds": [],
        "sequences": [],
        "sequence_entries": [],
        "rgroup_definitions": [],
        "page_summary": "",
    }


def test_ledger_initialization_failure_propagates(monkeypatch):
    error = RuntimeError("ledger unavailable")

    def fail(*args):
        raise error

    monkeypatch.setattr(document_ledger, "DocumentLedger", fail)
    with pytest.raises(RuntimeError) as caught:
        DocumentHarness("document.pdf")
    assert caught.value is error


def test_page_read_failure_propagates_without_caching(monkeypatch):
    error = ModelProviderError("offline")

    def fail(*args):
        raise error

    monkeypatch.setattr(PageEntityExtractor, "extract_pages", fail)
    harness = DocumentHarness("document.pdf", structure_pages=[1])
    with pytest.raises(ModelProviderError) as caught:
        harness.ensure_page_entities()
    assert caught.value is error and harness.page_entities == []


@pytest.mark.parametrize(
    "records,error",
    [
        ([], ModelEvidenceError),
        ([_record(1)], ModelEvidenceError),
        ([_record(1), _record(1)], ModelContractError),
        ([_record(1), _record(3)], ModelContractError),
        (
            [_record(1), dict(_record(2), ok=False, error="unreadable")],
            ModelEvidenceError,
        ),
        ([_record(1), dict(_record(2), compounds=None)], ModelContractError),
        ([_record(1), dict(_record(2), page_summary=None)], ModelContractError),
        (None, ModelContractError),
        ([None], ModelContractError),
        ([_record(1), dict(_record(2), page_num=True)], ModelContractError),
    ],
)
def test_incomplete_page_inventory_is_rejected(monkeypatch, records, error):
    monkeypatch.setattr(PageEntityExtractor, "extract_pages", lambda *args: records)
    harness = DocumentHarness("document.pdf", structure_pages=[1, 2])
    with pytest.raises(error):
        harness.ensure_page_entities()
    assert harness.page_entities == []


def test_verified_pages_are_sorted_cached_and_audited(monkeypatch, tmp_path):
    calls = []

    def read(self, pdf, pages):
        calls.append((self.audit_path, pdf, pages))
        return [_record(2), _record(1)]

    monkeypatch.setattr(PageEntityExtractor, "extract_pages", read)
    harness = DocumentHarness("document.pdf", str(tmp_path), structure_pages=[2, 1])
    first = harness.ensure_page_entities()
    assert [record["page_num"] for record in first] == [1, 2]
    assert harness.ensure_page_entities() is first
    assert calls == [(str(tmp_path / "model_calls.jsonl"), "document.pdf", [2, 1])]


def test_cached_invalid_records_are_not_accepted():
    harness = DocumentHarness("document.pdf", structure_pages=[1])
    harness.page_entities = [dict(_record(1), ok=False)]
    with pytest.raises(ModelEvidenceError):
        harness.ensure_page_entities()


def test_no_requested_pages_does_not_invoke_reader(monkeypatch):
    def fail(*args):
        pytest.fail("reader called without requested pages")

    monkeypatch.setattr(PageEntityExtractor, "extract_pages", fail)
    assert DocumentHarness("document.pdf").ensure_page_entities() == []


def test_ledger_persistence_failure_propagates(monkeypatch, tmp_path):
    harness = DocumentHarness("document.pdf", str(tmp_path))
    monkeypatch.setattr(harness, "service_report", dict)
    error = OSError("disk full")

    def fail(path):
        raise error

    monkeypatch.setattr(harness._ledger, "save", fail)
    with pytest.raises(OSError) as caught:
        harness.write_outputs([])
    assert caught.value is error


def test_required_ledger_is_written(monkeypatch, tmp_path):
    harness = DocumentHarness("document.pdf", str(tmp_path))
    monkeypatch.setattr(harness, "service_report", dict)
    paths = harness.write_outputs([])
    assert paths["ledger"] == str(tmp_path / "document_ledger.json")
    assert (tmp_path / "document_ledger.json").is_file()


def test_cached_inventory_requires_every_requested_page():
    harness = DocumentHarness("document.pdf", structure_pages=[1, 2])
    harness.page_entities = [_record(1)]
    with pytest.raises(ModelEvidenceError, match="Missing page evidence"):
        harness.ensure_page_entities()


@pytest.mark.parametrize(
    "sequence", ["A[Me]G", "acg", "A-G", "A G", "AβG", "ＡＧ", None]
)
def test_sequence_index_does_not_strip_or_uppercase_notation(sequence):
    harness = DocumentHarness("document.pdf")
    record = _record(1)
    record["sequences"] = [
        {
            "seq_id": "SEQ 1",
            "sequence_one_letter": sequence,
            "sequence_text": "H-Ala-[NMeGly]-OH",
            "cyclic": False,
        }
    ]
    harness.page_entities = [record]
    with pytest.raises(ModelContractError):
        harness.by_id()
    assert record["sequences"][0]["sequence_one_letter"] == sequence
    assert record["sequences"][0]["sequence_text"] == "H-Ala-[NMeGly]-OH"


def test_raw_modified_sequence_is_retained_without_canonical_derivation():
    harness = DocumentHarness("document.pdf")
    record = _record(1)
    record["sequences"] = [
        {
            "seq_id": "SEQ 1",
            "sequence_one_letter": "",
            "sequence_text": "H-Ala-[NMeGly]-OH",
            "cyclic": False,
        }
    ]
    harness.page_entities = [record]
    assert harness.by_id() == {}
    assert record["sequences"][0]["sequence_text"] == "H-Ala-[NMeGly]-OH"


def test_sequence_index_retains_canonical_residue_codes():
    harness = DocumentHarness("document.pdf")
    record = _record(1)
    record["sequences"] = [
        {
            "seq_id": "SEQ 1",
            "sequence_one_letter": "ACGT",
            "sequence_text": "ACGT",
            "cyclic": False,
        }
    ]
    harness.page_entities = [record]
    assert harness.by_id()["1"][0].sequence == "ACGT"


@pytest.mark.parametrize(
    "field,value", [("cyclic", "false"), ("cyclic", 0), ("seq_id", 1), ("seq_id", None)]
)
def test_sequence_index_rejects_implicit_type_conversion(field, value):
    harness = DocumentHarness("document.pdf")
    record = _record(1)
    sequence = {
        "seq_id": "1",
        "sequence_one_letter": "ACGT",
        "sequence_text": "ACGT",
        "cyclic": False,
    }
    sequence[field] = value
    record["sequences"] = [sequence]
    harness.page_entities = [record]
    with pytest.raises(ModelContractError):
        harness.by_id()
