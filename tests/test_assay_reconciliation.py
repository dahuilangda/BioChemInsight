"""Assay decision contracts."""

import copy
import json
import random

import pytest

import pipeline
from utils import llm_utils
from utils import model_harness as h
from utils.assay_reconciliation import AssayReconciliationRequest

NAMES = [
    "α-glucosidase · RMSD (Å) · calculated",
    "β-glucosidase · RMSD (Å) · calculated",
    "α-glucosidase · RMSD",
    "α-glucosidase protocol",
]
RESULT_TEXT = (
    "Compound 6a: calculated α-glucosidase RMSD = 1.2 Å; β-glucosidase RMSD = 2.4 Å."
)
PROTOCOL = "Protocol: buffer was prepared at pH 7.4."
SOURCES = {1: RESULT_TEXT + "\n" + PROTOCOL}


def _make_decision(cid, action="keep", canonical=None, line=1):
    return {
        "candidate_id": cid,
        "action": action,
        "canonical_candidate_id": (cid if action == "keep" else "a1")
        if canonical is None
        else canonical,
        "confidence": "high",
        "evidence_kind": "result_column",
        "evidence": [
            {
                "source_id": "p1",
                "line_start": line,
                "line_end": line,
            }
        ],
        "reason": "The source explicitly records this compound-level endpoint.",
    }


def _make_response():
    excluded = _make_decision("a4", "exclude", line=2)
    excluded.update(
        canonical_candidate_id=None,
        evidence_kind="non_result_context",
        reason="This candidate denotes buffer preparation only, not an independent result.",
    )
    return {
        "status": "completed",
        "reason": "Every candidate has been adjudicated.",
        "decisions": [
            _make_decision("a1"),
            _make_decision("a2"),
            _make_decision("a3", "merge"),
            excluded,
        ],
    }


def _make_request(names=None, sources=None):
    return AssayReconciliationRequest.from_pages(
        NAMES if names is None else names, SOURCES if sources is None else sources
    )


def _mock_model_runner(monkeypatch, responses, *, audit_path=None):
    calls = []

    def runner(**kwargs):
        schema = kwargs["response_schema"]
        assert schema["additionalProperties"] is False

        def operation():
            item = responses[min(len(calls), len(responses) - 1)]
            calls.append(item)
            if isinstance(item, BaseException):
                raise item
            return (
                item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            )

        return h.run_json_task(
            task_name=kwargs["task_name"],
            channel="text",
            operation=operation,
            parser=kwargs["parser"],
            retry=kwargs["retry"],
            retry_delays=[0] * kwargs["retry"],
            audit_path=audit_path,
            metadata=kwargs["metadata"],
        )

    monkeypatch.setattr(llm_utils, "run_text_json_task", runner)
    return calls


def test_valid_keep_merge_exclude_and_output_order():
    obj = _make_response()
    obj["decisions"].reverse()
    assert _make_request().parse_response(json.dumps(obj)) == NAMES[:2]


def test_excluded_candidates_return_empty_list():
    obj = _make_response()
    for item in obj["decisions"]:
        item.update(
            action="exclude",
            canonical_candidate_id=None,
            evidence_kind="non_result_context",
            evidence=[{"source_id": "p1", "line_start": 2, "line_end": 2}],
        )
    assert _make_request().parse_response(json.dumps(obj)) == []


def test_candidate_names_are_preserved():
    names = ["A·B", "AB", "A·B", "A B", "a·b", "α", "β", "Å", "A", "%", "μM", "uM"]
    obj = {
        "status": "completed",
        "reason": "",
        "decisions": [_make_decision(f"a{i}") for i in range(1, len(names) + 1)],
    }
    assert _make_request(names).parse_response(json.dumps(obj)) == names


def test_single_candidate_does_not_bypass_model(monkeypatch):
    obj = {"status": "completed", "reason": "", "decisions": [_make_decision("a1")]}
    calls = _mock_model_runner(monkeypatch, [obj])
    assert (
        pipeline.reconcile_assay_names(NAMES[:1], page_markdowns=SOURCES) == NAMES[:1]
    )
    assert len(calls) == 1


def test_missing_sources_fails_before_model(monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("missing evidence initiated a model call")

    monkeypatch.setattr(llm_utils, "run_text_json_task", forbidden)
    with pytest.raises(h.ModelEvidenceError):
        pipeline.reconcile_assay_names(NAMES, page_markdowns={})


def test_source_tail_is_not_truncated(monkeypatch):
    large = {1: "prefix " * 10000, 9: RESULT_TEXT + "\n" + PROTOCOL}
    obj = _make_response()
    for item in obj["decisions"]:
        item["evidence"][0]["source_id"] = "p9"

    def runner(**kwargs):
        assert large[1] in kwargs["prompt"] and RESULT_TEXT in kwargs["prompt"]
        return kwargs["parser"](json.dumps(obj))

    monkeypatch.setattr(llm_utils, "run_text_json_task", runner)
    assert pipeline.reconcile_assay_names(NAMES, page_markdowns=large) == NAMES[:2]


@pytest.mark.parametrize(
    "error",
    [
        h.ModelContractError("invalid"),
        h.ModelTimeoutError("timeout"),
        h.ModelProviderError("offline"),
        h.ModelCircuitOpen("task", 10),
        h.ModelEvidenceError("missing sources"),
    ],
)
def test_model_errors_propagate(monkeypatch, error):
    def fail(**kwargs):
        raise error

    monkeypatch.setattr(llm_utils, "run_text_json_task", fail)
    with pytest.raises(type(error)) as caught:
        pipeline.reconcile_assay_names(NAMES, page_markdowns=SOURCES)
    assert caught.value is error


def test_insufficient_evidence_is_explicit_terminal_failure(monkeypatch):
    obj = {
        "status": "insufficient_evidence",
        "decisions": [],
        "reason": "Missing results for a2.",
    }
    calls = _mock_model_runner(monkeypatch, [obj])
    with pytest.raises(h.ModelEvidenceError, match="a2"):
        pipeline.reconcile_assay_names(NAMES, page_markdowns=SOURCES)
    assert len(calls) == 1


def test_corrected_model_response_succeeds_after_invalid_decision(
    monkeypatch, tmp_path
):
    broken = _make_response()
    broken["decisions"][2]["canonical_candidate_id"] = ""
    path = tmp_path / "model_calls.jsonl"
    calls = _mock_model_runner(monkeypatch, [broken, _make_response()], audit_path=path)
    assert pipeline.reconcile_assay_names(NAMES, page_markdowns=SOURCES) == NAMES[:2]
    assert len(calls) == 2
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["ok"] for row in rows] == [False, True]
    assert [row["error_type"] for row in rows] == ["contract_error", ""]
    assert rows[0]["call_id"] == rows[1]["call_id"]
    assert all(row.get("event") != "conservative_fallback" for row in rows)


def test_invalid_responses_fail_after_retry_limit(monkeypatch, tmp_path):
    broken = _make_response()
    broken["decisions"][2]["canonical_candidate_id"] = ""
    path = tmp_path / "model_calls.jsonl"
    calls = _mock_model_runner(monkeypatch, [broken], audit_path=path)
    with pytest.raises(h.ModelContractError, match="canonical_candidate_id"):
        pipeline.reconcile_assay_names(NAMES, page_markdowns=SOURCES)
    assert len(calls) == pipeline.ASSAY_AUTO_DETECT_LLM_MAX_RETRIES + 1
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows and all(row["ok"] is False for row in rows)


MUTATIONS = (
    "missing_candidate",
    "duplicate_candidate",
    "unknown_candidate",
    "name_as_id",
    "empty_canonical",
    "null_merge_canonical",
    "unknown_canonical",
    "self_merge",
    "merge_chain",
    "merge_cycle",
    "missing_canonical",
    "wrong_keep_canonical",
    "exclude_nonnull_canonical",
    "low_merge_confidence",
    "medium_exclude_confidence",
    "unknown_action",
    "boolean_action",
    "missing_reason",
    "empty_reason",
    "wrong_reason_type",
    "unknown_evidence_kind",
    "wrong_exclude_kind",
    "missing_evidence",
    "empty_evidence",
    "unknown_source",
    "wrong_source_type",
    "beyond_page_end",
    "reversed_range",
    "zero_line_start",
    "line_wrong_type",
    "line_beyond_source",
    "missing_line_end",
    "extra_evidence_key",
    "decision_wrong_type",
    "decisions_wrong_type",
    "empty_decisions",
    "extra_decision_key",
    "extra_root_key",
    "missing_status",
    "unknown_status",
    "incomplete_evidence_status",
    "legacy_contract",
    "unknown_confidence",
    "canonical_wrong_type",
    "malformed_evidence",
    "reason_wrong_type",
    "root_reason_missing",
    "wrong_keep_kind",
    "missing_candidate_id",
    "source_as_candidate",
    "valid",
    "valid_reordered",
    "valid_all_keep",
    "valid_all_exclude",
    "valid_string_page_keys",
    "valid_unicode_distinct",
    "valid_duplicate_names",
    "valid_long_source",
    "valid_candidate_count",
    "valid_empty_reason",
    "valid_different_units",
    "valid_greek_names",
    "valid_multiple_ranges",
    "valid_two_sources",
)


def _mutate_response(obj, mutation, rng):
    a, _other, merged, excluded = obj["decisions"]
    if mutation == "missing_candidate":
        obj["decisions"].pop(1)
    elif mutation == "duplicate_candidate":
        obj["decisions"][1] = copy.deepcopy(a)
    elif mutation == "unknown_candidate":
        a["candidate_id"] = "a999"
    elif mutation == "name_as_id":
        a["candidate_id"] = NAMES[0]
    elif mutation == "empty_canonical":
        merged["canonical_candidate_id"] = ""
    elif mutation == "null_merge_canonical":
        merged["canonical_candidate_id"] = None
    elif mutation == "unknown_canonical":
        merged["canonical_candidate_id"] = "a999"
    elif mutation == "self_merge":
        merged["canonical_candidate_id"] = "a3"
    elif mutation == "merge_chain":
        a.update(action="merge", canonical_candidate_id="a2")
    elif mutation == "merge_cycle":
        a.update(action="merge", canonical_candidate_id="a3")
    elif mutation == "missing_canonical":
        merged.pop("canonical_candidate_id")
    elif mutation == "wrong_keep_canonical":
        a["canonical_candidate_id"] = "a2"
    elif mutation == "exclude_nonnull_canonical":
        excluded["canonical_candidate_id"] = "a1"
    elif mutation == "low_merge_confidence":
        merged["confidence"] = "low"
    elif mutation == "medium_exclude_confidence":
        excluded["confidence"] = "medium"
    elif mutation == "unknown_action":
        a["action"] = "drop_by_regex"
    elif mutation == "boolean_action":
        a["action"] = False
    elif mutation == "missing_reason":
        a.pop("reason")
    elif mutation == "empty_reason":
        a["reason"] = "  "
    elif mutation == "wrong_reason_type":
        a["reason"] = rng.choice([None, [], {}, 42])
    elif mutation == "unknown_evidence_kind":
        a["evidence_kind"] = "heuristic"
    elif mutation == "wrong_exclude_kind":
        excluded["evidence_kind"] = "result_column"
    elif mutation == "missing_evidence":
        a.pop("evidence")
    elif mutation == "empty_evidence":
        a["evidence"] = []
    elif mutation == "unknown_source":
        a["evidence"][0]["source_id"] = "p99"
    elif mutation == "wrong_source_type":
        a["evidence"][0]["source_id"] = rng.choice([1, None, [], {}])
    elif mutation == "beyond_page_end":
        a["evidence"][0]["line_end"] = 9999
    elif mutation == "reversed_range":
        a["evidence"][0].update(line_start=2, line_end=1)
    elif mutation == "zero_line_start":
        a["evidence"][0]["line_start"] = 0
    elif mutation == "line_wrong_type":
        a["evidence"][0]["line_start"] = rng.choice([None, True, [], {}])
    elif mutation == "line_beyond_source":
        a["evidence"][0].update(source_id="p2", line_start=2, line_end=2)
    elif mutation == "missing_line_end":
        a["evidence"][0].pop("line_end")
    elif mutation == "extra_evidence_key":
        a["evidence"][0]["page"] = 1
    elif mutation == "decision_wrong_type":
        obj["decisions"][0] = rng.choice([None, [], 1, "a1"])
    elif mutation == "decisions_wrong_type":
        obj["decisions"] = rng.choice([None, {}, 1, "a1"])
    elif mutation == "empty_decisions":
        obj["decisions"] = []
    elif mutation == "extra_decision_key":
        a["keep"] = True
    elif mutation == "extra_root_key":
        obj["assay_names"] = NAMES
    elif mutation == "missing_status":
        obj.pop("status")
    elif mutation == "unknown_status":
        obj["status"] = "degraded"
    elif mutation == "incomplete_evidence_status":
        obj["status"] = "insufficient_evidence"
    elif mutation == "legacy_contract":
        obj.clear()
        obj.update(assay_names=NAMES, decisions={})
    elif mutation == "unknown_confidence":
        a["confidence"] = "certain"
    elif mutation == "canonical_wrong_type":
        merged["canonical_candidate_id"] = rng.choice([[], {}, 1, False])
    elif mutation == "malformed_evidence":
        a["evidence"] = [rng.choice([None, [], 1, "quote"])]
    elif mutation == "reason_wrong_type":
        obj["reason"] = rng.choice([None, [], {}, 1])
    elif mutation == "root_reason_missing":
        obj.pop("reason")
    elif mutation == "wrong_keep_kind":
        a["evidence_kind"] = "non_result_context"
    elif mutation == "missing_candidate_id":
        a.pop("candidate_id")
    elif mutation == "source_as_candidate":
        a["candidate_id"] = "p1"
    elif mutation == "valid_reordered":
        rng.shuffle(obj["decisions"])
    elif mutation == "valid_all_keep":
        for item in obj["decisions"]:
            item.update(
                action="keep",
                canonical_candidate_id=item["candidate_id"],
                evidence_kind="result_column",
            )
    elif mutation == "valid_all_exclude":
        for item in obj["decisions"]:
            item.update(
                action="exclude",
                canonical_candidate_id=None,
                evidence_kind="non_result_context",
            )
    elif mutation == "valid_empty_reason":
        obj["reason"] = ""
    elif mutation == "valid_multiple_ranges":
        a["evidence"].append({"source_id": "p1", "line_start": 2, "line_end": 2})
    elif mutation == "valid_two_sources":
        a["evidence"].append({"source_id": "p2", "line_start": 1, "line_end": 1})


@pytest.mark.parametrize("seed", range(256))
def test_response_mutations(seed):
    """32,768 responses: valid exact results or explicit contract failure."""
    rng = random.Random(seed)
    mutations = list(MUTATIONS)
    for index in range(128):
        if index % len(mutations) == 0:
            rng.shuffle(mutations)
        mutation = mutations[index % len(mutations)]
        obj = _make_response()
        _mutate_response(obj, mutation, rng)
        names = list(NAMES)
        sources = {1: SOURCES[1], 2: "Independent source text."}
        if mutation == "valid_string_page_keys":
            sources = {str(key): value for key, value in sources.items()}
        elif mutation == "valid_unicode_distinct":
            names = ["α", "β", "Å", "A"]
        elif mutation == "valid_duplicate_names":
            names = ["same"] * 4
        elif mutation == "valid_long_source":
            sources[1] = "context " * rng.randint(2000, 4000) + SOURCES[1]
        elif mutation == "valid_different_units":
            names = ["IC50 (µM)", "IC50 (nM)", "pIC50", "pH"]
        elif mutation == "valid_greek_names":
            names = ["α-glucosidase", "β-glucosidase", "γ-glucosidase", "δ-glucosidase"]
        elif mutation == "valid_candidate_count":
            for extra in range(rng.randint(1, 12)):
                cid = f"a{len(names) + 1}"
                names.append(f"additional measured endpoint {extra}")
                obj["decisions"].append(_make_decision(cid))
        current = _make_request(names, sources)
        text = json.dumps(obj, ensure_ascii=False)
        if mutation.startswith("valid"):
            expected = names[:2]
            if mutation == "valid_all_keep":
                expected = names
            elif mutation == "valid_all_exclude":
                expected = []
            elif mutation == "valid_candidate_count":
                expected = names[:2] + names[4:]
            assert current.parse_response(text) == expected, (seed, index, mutation)
        else:
            with pytest.raises(h.ModelContractError):
                current.parse_response(text)


@pytest.mark.parametrize("seed", range(64))
def test_invalid_decision_types(seed):
    rng = random.Random(seed)
    for _ in range(64):
        obj = _make_response()
        item = rng.choice(obj["decisions"])
        key = rng.choice(sorted(item))
        item[key] = (
            {"unexpected": [None, {"nested": ["invalid"]}]}
            if key != "evidence"
            else ["invalid"]
        )
        with pytest.raises(h.ModelContractError):
            _make_request().parse_response(json.dumps(obj))


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not JSON",
        "[]",
        "null",
        "{}",
        '{"decisions":[]}',
        '{"status":"completed","reason":"","decisions":null}',
        '{"assay_names":[],"decisions":{}}',
    ],
)
def test_malformed_responses_exhaust_and_fail(monkeypatch, raw):
    calls = _mock_model_runner(monkeypatch, [raw])
    with pytest.raises(h.ModelContractError):
        pipeline.reconcile_assay_names(NAMES, page_markdowns=SOURCES)
    assert len(calls) == pipeline.ASSAY_AUTO_DETECT_LLM_MAX_RETRIES + 1


@pytest.mark.parametrize(
    "names,sources,error",
    [
        ([None], SOURCES, TypeError),
        ([""], SOURCES, ValueError),
        ("one name", SOURCES, TypeError),
        (NAMES, {True: RESULT_TEXT}, TypeError),
        (NAMES, {0: RESULT_TEXT}, ValueError),
        (NAMES, {1: None}, TypeError),
        (NAMES, {1: RESULT_TEXT, "1": RESULT_TEXT}, ValueError),
        (NAMES, {"p1": RESULT_TEXT}, ValueError),
    ],
)
def test_invalid_inputs_are_not_silently_normalized(names, sources, error):
    with pytest.raises(error):
        _make_request(names, sources)


@pytest.mark.parametrize("seed", range(128))
def test_retry_outcomes(monkeypatch, seed):
    rng = random.Random(seed)
    invalids = [name for name in MUTATIONS if not name.startswith("valid")]
    for case in range(8):
        broken = _make_response()
        _mutate_response(broken, rng.choice(invalids), rng)
        if case % 4 == 0:
            responses = [broken, _make_response()]
            expected_attempts, error = 2, None
        elif case % 4 == 1:
            responses = [broken, broken]
            expected_attempts, error = 2, h.ModelContractError
        elif case % 4 == 2:
            responses = [
                TimeoutError("injected transient transport timeout"),
                _make_response(),
            ]
            expected_attempts, error = 2, None
        else:
            responses = [
                {
                    "status": "insufficient_evidence",
                    "decisions": [],
                    "reason": "Source for a2 is missing.",
                }
            ]
            expected_attempts, error = 1, h.ModelEvidenceError
        calls = _mock_model_runner(monkeypatch, responses)
        if error:
            with pytest.raises(error):
                pipeline.reconcile_assay_names(NAMES, page_markdowns=SOURCES)
        else:
            assert (
                pipeline.reconcile_assay_names(NAMES, page_markdowns=SOURCES)
                == NAMES[:2]
            )
        assert len(calls) == expected_attempts, (seed, case)


def test_concurrent_requests_have_independent_state():
    from concurrent.futures import ThreadPoolExecutor

    def parse_one(index):
        names = [f"{name} / document {index}" for name in NAMES]
        return _make_request(names).parse_response(json.dumps(_make_response()))

    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(parse_one, range(1024)))
    for index, result in enumerate(results):
        assert result == [f"{name} / document {index}" for name in NAMES[:2]]


@pytest.mark.parametrize(
    "prefix,suffix",
    [
        ("```json\n", "\n```"),
        ("Here is the result: ", ""),
        ("<think>reasoning</think>", ""),
        ("", "\nDone."),
    ],
)
def test_strict_response_does_not_extract_json_from_wrappers(prefix, suffix):
    with pytest.raises(h.ModelContractError):
        _make_request().parse_response(prefix + json.dumps(_make_response()) + suffix)


def test_literal_markup_in_source_is_preserved():
    source = "Result: <think>label</think> RMSD = 1.2 Å."
    obj = {
        "status": "completed",
        "reason": "",
        "decisions": [_make_decision("a1")],
    }
    assert _make_request(["calculated RMSD"], {1: source}).parse_response(
        json.dumps(obj)
    ) == ["calculated RMSD"]


@pytest.mark.parametrize(
    "start,end",
    [
        (0, 1),
        (-1, 1),
        (2, 1),
        (1, 3),
        (3, 3),
        (True, 1),
        (1, False),
        (1.0, 1),
        (1, 1.0),
        ("1", 1),
        (None, 1),
    ],
)
def test_line_ranges_require_valid_integer_bounds(start, end):
    obj = _make_response()
    obj["decisions"][0]["evidence"][0].update(line_start=start, line_end=end)
    with pytest.raises(h.ModelContractError):
        _make_request().parse_response(json.dumps(obj))


def test_blank_line_range_does_not_count_as_evidence():
    obj = _make_response()
    obj["decisions"][0]["evidence"][0].update(source_id="p2", line_start=1, line_end=1)
    with pytest.raises(h.ModelContractError):
        _make_request(
            sources={1: SOURCES[1], 2: "\nValid source below.\n"}
        ).parse_response(json.dumps(obj))


def test_source_numbering_preserves_every_character():
    text = "Header\r\nRMSD\t1.2 Å\rValue\nEnd\fNote\u2028α\x85β"
    current = _make_request(["RMSD"], {4: text})
    lines = current.build_source_context()[0]["lines"]
    assert "".join(line["text"] for line in lines) == text
    assert [line["line"] for line in lines] == list(range(1, len(lines) + 1))


def test_one_invalid_range_rejects_the_entire_response():
    obj = _make_response()
    obj["decisions"][0]["evidence"].append(
        {"source_id": "p99", "line_start": 1, "line_end": 1}
    )
    with pytest.raises(h.ModelContractError):
        _make_request().parse_response(json.dumps(obj))


def test_candidate_deduplication_preserves_case_spacing_and_unicode():
    names = ["a b", "a  b", " a b", "A B", "α", "β", "Å", "A", "a b"]
    assert pipeline.normalize_detected_assay_names(names) == names[:-1]


@pytest.mark.parametrize("name", ["", " ", None, 1])
def test_candidate_deduplication_rejects_invalid_input(name):
    with pytest.raises((ValueError, TypeError)):
        pipeline.normalize_detected_assay_names(["valid endpoint", name])
