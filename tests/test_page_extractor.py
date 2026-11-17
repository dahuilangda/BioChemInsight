"""Page evidence contracts and model-call routing."""

import json

import pytest

from utils import llm_utils, page_extractor
from utils.model_harness import (
    ModelContractError,
    ModelEvidenceError,
    ModelProviderError,
)


def _page_payload():
    return {
        "compounds": [
            {
                "label_in_page": "6a",
                "name": "",
                "structure_type": "complete_compound",
                "evidence_kind": "drawn_structure",
                "evidence": "Label 6a below the structure.",
                "sequence_one_letter": "",
            }
        ],
        "sequences": [],
        "has_sequence_table": False,
        "rgroup_definitions": [],
        "page_summary": "Substituent R{1 appears in the figure.",
    }


def _sequence_payload():
    return {
        "sequence_entries": [
            {
                "seq_id": "SEQ 1",
                "sequence_text": "H-Ala-[NMePhe]-OH",
                "procedure": "",
                "molecular_formula": "",
                "observed_ion": "",
                "molecular_weight": None,
                "exact_mass": 250.1,
                "purity_percent": 97.2,
                "measured_mass": None,
            }
        ]
    }


def test_valid_json_preserves_unmatched_braces_in_strings():
    payload = _page_payload()
    assert page_extractor.parse_page_entities(json.dumps(payload)) == payload


@pytest.mark.parametrize(
    "text",
    [
        "",
        "{}",
        '{"unexpected":"content"}',
        "[]",
        "```json\n{}\n```",
        '{"compounds":[],"compounds":[]}',
    ],
)
def test_invalid_page_payload_is_rejected(text):
    with pytest.raises(ModelContractError):
        page_extractor.parse_page_entities(text)


@pytest.mark.parametrize(
    "field,value",
    [
        ("compounds", None),
        ("sequences", {}),
        ("rgroup_definitions", ""),
        ("has_sequence_table", "false"),
        ("page_summary", 1),
    ],
)
def test_page_field_types_are_required(field, value):
    payload = _page_payload()
    payload[field] = value
    with pytest.raises(ModelContractError):
        page_extractor.parse_page_entities(json.dumps(payload))


@pytest.mark.parametrize(
    "field,value",
    [
        ("label_in_page", None),
        ("structure_type", "result"),
        ("evidence_kind", "inferred"),
        ("evidence", ""),
        ("name", []),
        ("sequence_one_letter", 1),
    ],
)
def test_compound_field_types_and_evidence_are_required(field, value):
    payload = _page_payload()
    payload["compounds"][0][field] = value
    with pytest.raises(ModelContractError):
        page_extractor.parse_page_entities(json.dumps(payload))


def test_modified_sequence_text_is_preserved():
    payload = _page_payload()
    payload["sequences"] = [
        {
            "seq_id": "1",
            "sequence_one_letter": "",
            "sequence_text": "H-Ala-[NMePhe]-OH",
            "cyclic": False,
        }
    ]
    assert page_extractor.parse_page_entities(json.dumps(payload)) == payload
    table = _sequence_payload()
    assert page_extractor.parse_sequence_entries(json.dumps(table)) == table


@pytest.mark.parametrize(
    "sequence",
    [
        None,
        {},
        {
            "seq_id": "1",
            "sequence_one_letter": "",
            "sequence_text": "",
            "cyclic": False,
        },
        {
            "seq_id": "1",
            "sequence_one_letter": "AC",
            "sequence_text": "AC",
            "cyclic": "false",
        },
    ],
)
def test_invalid_sequence_records_are_rejected(sequence):
    payload = _page_payload()
    payload["sequences"] = [sequence]
    with pytest.raises(ModelContractError):
        page_extractor.parse_page_entities(json.dumps(payload))


@pytest.mark.parametrize(
    "definition",
    [
        None,
        {"variable": "", "substituents": []},
        {"variable": "R1", "substituents": [3]},
        {"variable": "R1"},
    ],
)
def test_invalid_substituent_records_are_rejected(definition):
    payload = _page_payload()
    payload["rgroup_definitions"] = [definition]
    with pytest.raises(ModelContractError):
        page_extractor.parse_page_entities(json.dumps(payload))


@pytest.mark.parametrize(
    "field,value",
    [
        ("exact_mass", True),
        ("purity_percent", "97.2"),
        ("molecular_weight", []),
        ("sequence_text", ""),
        ("observed_ion", None),
    ],
)
def test_sequence_table_types_are_required(field, value):
    payload = _sequence_payload()
    payload["sequence_entries"][0][field] = value
    with pytest.raises(ModelContractError):
        page_extractor.parse_sequence_entries(json.dumps(payload))


def test_sequence_table_rejects_nonfinite_numbers():
    payload = _sequence_payload()
    payload["sequence_entries"][0]["exact_mass"] = float("inf")
    with pytest.raises(ModelContractError):
        page_extractor.parse_sequence_entries(json.dumps(payload))


def _install_reader(monkeypatch, tmp_path, responses):
    image = tmp_path / "page.png"
    image.write_bytes(b"fixture")
    monkeypatch.setattr(page_extractor, "_render_page", lambda *args: str(image))
    calls = []

    def invoke(**kwargs):
        calls.append(kwargs)
        response = responses[len(calls) - 1]
        if isinstance(response, Exception):
            raise response
        return kwargs["parser"](json.dumps(response))

    monkeypatch.setattr(llm_utils, "run_vision_json_task", invoke)
    return image, calls


def test_page_reads_use_shared_visual_harness(monkeypatch, tmp_path):
    payload = _page_payload()
    image, calls = _install_reader(monkeypatch, tmp_path, [payload])
    audit = tmp_path / "model_calls.jsonl"
    result = page_extractor.PageEntityExtractor(audit_path=audit).extract_page(
        "document.pdf", 3
    )
    assert result["ok"] is True and result["compounds"] == payload["compounds"]
    assert calls[0]["task_name"] == "extract_page_entities"
    assert calls[0]["preserve_response"] is True
    assert calls[0]["metadata"]["page"] == 3 and calls[0]["audit_path"] == audit
    assert not image.exists()


def test_sequence_table_is_a_separate_validated_call(monkeypatch, tmp_path):
    payload = _page_payload()
    payload["has_sequence_table"] = True
    image, calls = _install_reader(
        monkeypatch, tmp_path, [payload, _sequence_payload()]
    )
    result = page_extractor.PageEntityExtractor().extract_page("document.pdf", 2)
    assert [call["task_name"] for call in calls] == [
        "extract_page_entities",
        "extract_sequence_table",
    ]
    assert result["sequence_entries"] == _sequence_payload()["sequence_entries"]
    assert all(call["preserve_response"] is True for call in calls)
    assert not image.exists()


@pytest.mark.parametrize(
    "error", [ModelProviderError("offline"), ModelContractError("invalid output")]
)
def test_failed_read_propagates_and_cleans_image(monkeypatch, tmp_path, error):
    image, _ = _install_reader(monkeypatch, tmp_path, [error])
    with pytest.raises(type(error)) as caught:
        page_extractor.PageEntityExtractor().extract_page("document.pdf", 1)
    assert caught.value is error and not image.exists()


def test_failed_sequence_read_does_not_return_partial_page(monkeypatch, tmp_path):
    payload = _page_payload()
    payload["has_sequence_table"] = True
    image, _ = _install_reader(
        monkeypatch, tmp_path, [payload, ModelProviderError("offline")]
    )
    with pytest.raises(ModelProviderError):
        page_extractor.PageEntityExtractor().extract_page("document.pdf", 1)
    assert not image.exists()


def test_claimed_sequence_table_requires_rows(monkeypatch, tmp_path):
    payload = _page_payload()
    payload["has_sequence_table"] = True
    image, _ = _install_reader(
        monkeypatch, tmp_path, [payload, {"sequence_entries": []}]
    )
    with pytest.raises(ModelEvidenceError):
        page_extractor.PageEntityExtractor().extract_page("document.pdf", 1)
    assert not image.exists()


def test_page_schema_failure_never_becomes_an_empty_success(monkeypatch, tmp_path):
    image, _ = _install_reader(monkeypatch, tmp_path, [{"unexpected": "content"}])
    with pytest.raises(ModelContractError):
        page_extractor.PageEntityExtractor().extract_page("document.pdf", 1)
    assert not image.exists()


def test_source_pages_are_ordered_and_prior_entities_are_context(monkeypatch):
    calls = []
    extractor = page_extractor.PageEntityExtractor()

    def read(pdf, page, digest):
        calls.append((page, digest))
        return {
            "page_num": page,
            "compounds": _page_payload()["compounds"],
            "sequences": [],
        }

    monkeypatch.setattr(extractor, "extract_page", read)
    result = extractor.extract_pages("document.pdf", [3, 1, 3])
    assert [record["page_num"] for record in result] == [1, 3]
    assert calls[0] == (1, "") and "page 1: [6a]" in calls[1][1]


def _source_pdf(tmp_path):
    import fitz

    path = tmp_path / "source.pdf"
    with fitz.open() as document:
        document.new_page()
        document.save(path)
    return str(path)


def test_rendered_page_files_are_unique(tmp_path):
    from pathlib import Path

    source = _source_pdf(tmp_path)
    first = Path(page_extractor._render_page(source, 1))
    second = Path(page_extractor._render_page(source, 1))
    try:
        assert first != second and first.is_file() and second.is_file()
    finally:
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)


@pytest.mark.parametrize("page", [0, 2, True])
def test_invalid_source_page_is_rejected_before_render(monkeypatch, tmp_path, page):
    source = _source_pdf(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("temporary image created for an invalid page")

    monkeypatch.setattr(page_extractor.tempfile, "mkstemp", forbidden)
    with pytest.raises(ModelEvidenceError):
        page_extractor._render_page(source, page)


@pytest.mark.parametrize(
    "sequence",
    [
        "A[Me]G",
        "aCg",
        "acg",
        "A-G",
        "A G",
        "A\nG",
        "AβG",
        "ＡＣＧ",
        "A1G",
        "A(G)",
        " ACG",
        "ACG ",
    ],
)
def test_one_letter_notation_is_not_cleaned(sequence):
    payload = _page_payload()
    payload["sequences"] = [
        {
            "seq_id": "1",
            "sequence_one_letter": sequence,
            "sequence_text": sequence,
            "cyclic": False,
        }
    ]
    with pytest.raises(ModelContractError):
        page_extractor.parse_page_entities(json.dumps(payload))
    assert payload["sequences"][0]["sequence_text"] == sequence


@pytest.mark.parametrize("sequence", ["A[Me]G", "acg", "AβG"])
def test_compound_one_letter_notation_is_validated(sequence):
    payload = _page_payload()
    payload["compounds"][0]["sequence_one_letter"] = sequence
    with pytest.raises(ModelContractError):
        page_extractor.parse_page_entities(json.dumps(payload))


def test_canonical_sequence_is_preserved_without_case_conversion():
    payload = _page_payload()
    payload["sequences"] = [
        {
            "seq_id": "1",
            "sequence_one_letter": "ACGT",
            "sequence_text": "ACGT",
            "cyclic": False,
        }
    ]
    assert page_extractor.parse_page_entities(json.dumps(payload)) == payload


def test_page_reader_preserves_literal_markup(monkeypatch, tmp_path):
    payload = _page_payload()
    payload["page_summary"] = "<think>printed label R{1</think>"
    _, calls = _install_reader(monkeypatch, tmp_path, [payload])
    result = page_extractor.PageEntityExtractor().extract_page("document.pdf", 1)
    assert result["page_summary"] == payload["page_summary"]
    assert calls[0]["preserve_response"] is True
