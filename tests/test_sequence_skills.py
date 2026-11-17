"""Sequence evidence, image transport, and listing integrity."""

import asyncio
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from skills.seq_id_vision import skill as vision
from skills.sequence_assembler.skill import assemble
from utils import llm_utils as llm
from utils import model_harness as harness


@pytest.fixture(autouse=True)
def isolated_circuits():
    harness._TASK_CIRCUIT_STATE.clear()
    yield
    harness._TASK_CIRCUIT_STATE.clear()


def annotation(**changes):
    return {
        "status": "found",
        "seq_id": 17,
        "label_text": "SEQ ID NO: 17",
        "confidence": 0.9,
        **changes,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", False),
        ("status", []),
        ("status", "missing"),
        ("seq_id", True),
        ("seq_id", "17"),
        ("seq_id", 17.0),
        ("seq_id", 0),
        ("seq_id", -1),
        ("label_text", ""),
        ("label_text", None),
        ("label_text", 17),
        ("confidence", True),
        ("confidence", "0.9"),
        ("confidence", None),
        ("confidence", -0.1),
        ("confidence", 1.1),
        ("confidence", float("nan")),
    ],
)
def test_annotation_rejects_invalid_fields(field, value):
    with pytest.raises(harness.ModelContractError):
        vision.parse_annotation(json.dumps(annotation(**{field: value})))


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "{}",
        "[]",
        "```json\n{}\n```",
        '{"status":"absent","status":"found"}',
        "prefix {}",
    ],
)
def test_malformed_response_cannot_become_absence(raw):
    with pytest.raises(harness.ModelContractError):
        vision.parse_annotation(raw)


@pytest.mark.parametrize("status", ["absent", "uncertain"])
@pytest.mark.parametrize(
    "field,value",
    [("seq_id", 17), ("label_text", "SEQ ID NO: 17"), ("confidence", 0.9)],
)
def test_inconsistent_status_rejected(status, field, value):
    result = annotation(status=status, seq_id=None, label_text=None, confidence=None)
    result[field] = value
    with pytest.raises(harness.ModelContractError):
        vision.parse_annotation(json.dumps(result))


def test_ambiguous_annotation_is_terminal_evidence_error():
    calls = []

    def operation():
        calls.append(1)
        return json.dumps(
            annotation(
                status="uncertain", seq_id=None, label_text=None, confidence=None
            )
        )

    with pytest.raises(harness.ModelEvidenceError):
        harness.run_json_task(
            task_name="sequence-evidence-test",
            channel="vision",
            operation=operation,
            parser=vision.parse_annotation,
            retry=3,
            retry_delays=[0, 0],
        )
    assert calls == [1]


def test_absent_annotation_is_explicit():
    result = annotation(status="absent", seq_id=None, label_text=None, confidence=None)
    assert vision.parse_annotation(json.dumps(result)) == result


@pytest.mark.parametrize("status", ["found", "absent"])
def test_views_share_one_request_and_are_cleaned_up(monkeypatch, tmp_path, status):
    source = tmp_path / "source.png"
    assert cv2.imwrite(str(source), np.full((300, 200, 3), 255, dtype=np.uint8))
    observed = []

    def run(**kwargs):
        observed.extend(kwargs["image_files"])
        assert [path.stem for path in observed] == ["band_bottom", "band_top", "full"]
        assert all(path.is_file() for path in observed)
        assert kwargs["preserve_response"] is True
        assert kwargs["response_schema"] == vision._RESPONSE_SCHEMA
        result = (
            annotation()
            if status == "found"
            else annotation(
                status="absent", seq_id=None, label_text=None, confidence=None
            )
        )
        return kwargs["parser"](json.dumps(result))

    monkeypatch.setattr(llm, "run_vision_json_task", run)
    result = asyncio.run(vision.read_seq_id_label({"image_path": str(source)}))
    assert result["ok"] and result["found"] is (status == "found")
    assert len(observed) == 3 and all(not path.exists() for path in observed)


def test_failed_request_cleans_up_views_without_replacement(monkeypatch, tmp_path):
    source = tmp_path / "source.png"
    assert cv2.imwrite(str(source), np.full((300, 200, 3), 255, dtype=np.uint8))
    paths = []

    def fail(**kwargs):
        paths.extend(kwargs["image_files"])
        raise harness.ModelProviderError("unavailable")

    monkeypatch.setattr(llm, "run_vision_json_task", fail)
    with pytest.raises(harness.ModelProviderError):
        asyncio.run(vision.read_seq_id_label({"crop_path": str(source)}))
    assert paths and all(not path.exists() for path in paths)


def test_multi_image_transport_preserves_order_and_schema(monkeypatch, tmp_path):
    paths = [tmp_path / f"view-{index}.png" for index in range(3)]
    for index, path in enumerate(paths):
        path.write_bytes(bytes([index]))
    requests, closed = [], []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(annotation()))
                )
            ],
        )

    def factory(**kwargs):
        assert kwargs["max_retries"] == 0
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=lambda: closed.append(1),
        )

    monkeypatch.setattr(llm, "require_openai", lambda: factory)
    monkeypatch.setattr(llm, "VISUAL_MODEL_KEY", "test")
    monkeypatch.setattr(llm, "VISUAL_MODEL_NAME", "test")
    monkeypatch.setattr(llm, "encode_image_to_base64_data_uri", lambda path: str(path))
    result = llm.run_vision_json_task(
        task_name="multi-image-test",
        image_files=paths,
        prompt="read",
        parser=vision.parse_annotation,
        preserve_response=True,
        response_schema=vision._RESPONSE_SCHEMA,
        retry=1,
    )
    assert result["seq_id"] == 17
    assert len(requests) == len(closed) == 1
    content = requests[0]["messages"][1]["content"]
    assert [item["image_url"]["url"] for item in content[1:]] == [
        str(path) for path in paths
    ]
    assert (
        requests[0]["response_format"]["json_schema"]["schema"]
        == vision._RESPONSE_SCHEMA
    )


@pytest.mark.parametrize("paths", [[], "source.png", [None], [3]])
def test_invalid_image_sets_never_call_provider(paths):
    with pytest.raises(ValueError):
        llm.run_vision_json_task(
            task_name="image-input-test",
            image_files=paths,
            prompt="read",
            parser=lambda value: value,
        )


@pytest.mark.parametrize(
    "pages",
    [
        None,
        [],
        "pages",
        [None],
        [{"page": True, "text": ""}],
        [{"page": 0, "text": ""}],
        [{"page": 1}],
        [{"page": 1, "text": []}],
        [{"page": 1, "text": ""}, {"page": 1, "text": ""}],
    ],
)
def test_listing_input_rejects_invalid_pages(pages):
    assert asyncio.run(assemble({"pages": pages}))["ok"] is False


@pytest.mark.parametrize("length", [400, 401, 800, 4096])
def test_long_sequences_are_not_truncated(length):
    sequence = ("ACDEFGHIKLMNPQRSTVWY" * (length // 19 + 1))[:length]
    result = asyncio.run(
        assemble(
            {
                "pages": [
                    {
                        "page": 1,
                        "text": f"SEQ ID NO: 7\n<211> LENGTH: {length}\n{sequence}",
                    }
                ]
            }
        )
    )
    row = result["sequences"][0]
    assert row["length"] == len(row["sequence"]) == length
    assert row["sequence"] == sequence and row["length_ok"]


def test_cross_page_order_does_not_depend_on_input_order():
    prefix, suffix = "ACDEFGHIKLMNP", "QRSTVWYACDEFG"
    pages = [
        {"page": 3, "text": suffix},
        {"page": 2, "text": f"SEQ ID NO: 7\n{prefix}"},
    ]
    result = asyncio.run(assemble({"pages": pages}))
    assert result["sequences"][0]["sequence"] == prefix + suffix
    assert result["sequences"][0]["pages"] == [2, 3]
