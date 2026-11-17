"""Measurement corrections, omissions, conflicts, and source support."""

import copy
import json

import pytest

from utils.measurement_contract import MeasurementRequest
from utils.measurement_review import parse_review, review_measurements
from utils.model_harness import (
    ModelContractError,
    ModelDownstreamVerificationError,
    ModelEvidenceError,
    ModelProviderError,
    ModelTimeoutError,
)


def test_workflow_deadline_stops_later_stages(monkeypatch):
    from utils import measurement_review as module

    clock = [0.0]
    calls = []
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    def run(**kwargs):
        calls.append(kwargs)
        clock[0] = 51
        return kwargs["parser"](json.dumps(candidate(observation())))

    with pytest.raises(ModelTimeoutError):
        review_measurements(SOURCE, timeout_seconds=50, runner=run)
    assert len(calls) == 1
    assert calls[0]["timeout_seconds"] == 50


def test_call_budget_shrinks_across_review_stages(monkeypatch):
    from utils import measurement_review as module

    clock = [0.0]
    calls = []
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    responses = iter([candidate(observation()), review()])

    def run(**kwargs):
        calls.append(kwargs)
        clock[0] += 20
        return kwargs["parser"](json.dumps(next(responses)))

    review_measurements(SOURCE, timeout_seconds=50, call_timeout_seconds=40, runner=run)
    assert [call["timeout_seconds"] for call in calls] == [40, 30]
    assert all(call["retry"] == 2 for call in calls)


SOURCE = "Compound | IC50 (nM) | calculated RMSD (Å)\n6a | 12 | 1.2\n6b | 24 | 2.4"
REFERENCE = {"line_start": 1, "line_end": 3}


def observation(compound="6a", raw="12", **changes):
    return {
        "compound_id": compound,
        "id_kind": "compound",
        "assay_name": "IC50 (nM)",
        "endpoint": "IC50",
        "target": None,
        "conditions": None,
        "source_kind": "experimental",
        "raw": raw,
        "value": float(raw),
        "operator": "",
        "unit": "nM",
        "evidence": [copy.deepcopy(REFERENCE)],
        **changes,
    }


def candidate(*items):
    return {
        "status": "complete" if items else "absent",
        "reason": "",
        "observations": list(items),
        "source_issues": [],
    }


def review(status="accepted", kind="omission"):
    return {
        "status": status,
        "issues": []
        if status == "accepted"
        else [
            {
                "kind": kind,
                "description": "Check row 6b against the source",
                "resolution": "source_review"
                if status == "unresolved"
                else "correctable",
                "subjects": [{"id_kind": "compound", "compound_id": "6b"}],
                "scope": {
                    "endpoint": "IC50",
                    "method": None,
                    "reported_values": ["24"],
                },
                "evidence": [copy.deepcopy(REFERENCE)],
            }
        ],
        "coverage": [copy.deepcopy(REFERENCE)],
        "question_decisions": [],
    }


def source_issue(kind="ambiguity"):
    issue = review("unresolved", kind)["issues"][0]
    return {key: value for key, value in issue.items() if key != "resolution"}


def question_decision(question_id, decision):
    return {
        "question_id": question_id,
        "decision": decision,
        "explanation": "The source establishes this assignment.",
        "evidence": [copy.deepcopy(REFERENCE)],
    }


def runner_for(responses, calls):
    responses = iter(responses)

    def run(**kwargs):
        calls.append(kwargs)
        response = next(responses)
        if isinstance(response, BaseException):
            raise response
        return kwargs["parser"](json.dumps(response))

    return run


@pytest.mark.parametrize(
    "kind", ["omission", "conflict", "unsupported", "transcription"]
)
def test_source_supported_revision_must_pass_fresh_review(kind):
    calls = []
    original = candidate(observation())
    corrected = candidate(observation(), observation("6b", "24"))
    result = review_measurements(
        SOURCE,
        runner=runner_for(
            [original, review("revise", kind), corrected, review()], calls
        ),
    )
    assert [row["compound_id"] for row in result["rows"]] == ["6a", "6b"]
    assert [call["metadata"]["stage"] for call in calls] == [
        "extract",
        "review",
        "revise",
        "review",
    ]
    assert result["review"]["revisions"] == 1
    assert (
        result["review"]["history"][0]["candidate"]["observations"][0]["raw"]
        == original["observations"][0]["raw"]
    )
    assert len(result["review"]["history"][1]["candidate"]["observations"]) == len(
        corrected["observations"]
    )
    assert SOURCE.splitlines()[0] in calls[1]["prompt"]
    assert "SOURCE-SUPPORTED REVIEW" not in calls[3]["prompt"]
    assert "SOURCE-SUPPORTED REVIEW" in calls[2]["prompt"]


def test_absence_is_reviewed_for_missed_results():
    calls = []
    result = review_measurements(
        SOURCE,
        runner=runner_for(
            [candidate(), review("revise"), candidate(observation()), review()], calls
        ),
    )
    assert result["rows"] and len(calls) == 4


@pytest.mark.parametrize("phase", [0, 1, 2, 3])
def test_failed_stage_never_returns_an_earlier_candidate(phase):
    calls = []
    responses = [
        candidate(observation()),
        review("revise"),
        candidate(observation("6b", "24")),
        review(),
    ]
    responses[phase] = ModelProviderError("provider rejected request")
    with pytest.raises(ModelProviderError):
        review_measurements(SOURCE, runner=runner_for(responses, calls))
    assert len(calls) == phase + 1


@pytest.mark.parametrize("budget", [0, 1, 2])
def test_unresolved_revision_budget_is_terminal(budget):
    calls = []
    responses = [candidate(observation())]
    for index in range(budget + 1):
        responses.append(review("revise"))
        if index < budget:
            responses.append(candidate(observation()))
    with pytest.raises(ModelDownstreamVerificationError):
        review_measurements(
            SOURCE, max_revisions=budget, runner=runner_for(responses, calls)
        )
    assert len(calls) == 2 + 2 * budget


def test_ambiguous_source_conflict_is_not_forced_to_a_value():
    calls = []
    with pytest.raises(ModelEvidenceError):
        review_measurements(
            SOURCE,
            runner=runner_for(
                [candidate(observation()), review("unresolved", "conflict")], calls
            ),
        )
    assert len(calls) == 2


@pytest.mark.parametrize(
    "endpoint,unit,kind",
    [
        ("RMSD", "Å", "calculated"),
        ("Caco-2 Papp", "cm/s", "experimental"),
        ("Half-life", "h", "experimental"),
        ("hERG risk", None, "predicted"),
        ("Solubility", "mg/mL", "experimental"),
    ],
)
def test_endpoint_vocabulary_is_not_restricted(endpoint, unit, kind):
    source = f"Compound 6a: {endpoint} = 12 {unit or ''} ({kind})"
    item = observation(
        endpoint=endpoint,
        assay_name=endpoint,
        source_kind=kind,
        unit=unit,
        evidence=[{"line_start": 1, "line_end": 1}],
    )
    result = MeasurementRequest(source).parse(json.dumps(candidate(item)))
    assert result["observations"][0]["endpoint"] == endpoint


@pytest.mark.parametrize(
    "field,value",
    [
        ("compound_id", ""),
        ("id_kind", None),
        ("endpoint", ""),
        ("assay_name", 3),
        ("raw", None),
        ("target", []),
        ("unit", ""),
        ("source_kind", "inferred"),
        ("operator", "?"),
        ("value", True),
        ("value", "12"),
        ("value", float("inf")),
    ],
)
def test_invalid_observation_is_rejected(field, value):
    item = observation()
    item[field] = value
    with pytest.raises(ModelContractError):
        MeasurementRequest(SOURCE).parse(json.dumps(candidate(item)))


@pytest.mark.parametrize(
    "reference",
    [
        None,
        [],
        [{"line_start": 1, "line_end": 3, "quote": "invented"}],
        [{"line_start": 2, "line_end": 2, "quote": "6b | 24 | 2.4"}],
    ],
)
def test_evidence_cannot_be_missing_or_attached_to_wrong_lines(reference):
    with pytest.raises(ModelContractError):
        MeasurementRequest(SOURCE).parse(
            json.dumps(candidate(observation(evidence=reference)))
        )


@pytest.mark.parametrize("start,end", [(True, 1), (0, 1), (1, 4), (3, 2), ("1", 2)])
def test_source_ranges_are_strict(start, end):
    with pytest.raises(ModelContractError):
        MeasurementRequest(SOURCE).validate_evidence(
            [{"line_start": start, "line_end": end}]
        )


def test_same_compound_distinct_conditions_are_preserved():
    request = MeasurementRequest(SOURCE)
    result = request.parse(
        json.dumps(
            candidate(
                observation(), observation(assay_name="IC50 at pH 6", conditions="pH 6")
            )
        )
    )
    assert len(result["observations"]) == 2
    with pytest.raises(ModelContractError):
        request.parse(json.dumps(candidate(observation(), observation())))


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "accepted", "issues": review("revise")["issues"]},
        {"status": "revise", "issues": []},
        {"status": "unknown"},
        {"coverage": []},
        {"coverage": [{"line_start": 1, "line_end": 1, "quote": "fabricated"}]},
    ],
)
def test_review_cannot_accept_inconsistent_or_ungrounded_claims(changes):
    with pytest.raises((ModelContractError, ModelEvidenceError)):
        parse_review(json.dumps({**review(), **changes}), MeasurementRequest(SOURCE))


def test_accepted_model_review_cannot_override_reference_scoring(monkeypatch, tmp_path):
    from scripts.check_measurement_review import evaluate_case
    from utils import measurement_review as module

    result = {
        "rows": [
            {
                "compound_id": "6a",
                "id_kind": "compound",
                "assays": {"IC50": observation(raw="120")},
            }
        ],
        "review": {"revisions": 0},
    }
    monkeypatch.setattr(module, "review_measurements", lambda *args, **kwargs: result)
    report = evaluate_case(
        {"name": "wrong-accepted-value", "source": SOURCE, "expected": [observation()]},
        tmp_path,
    )
    assert not report["passed"] and report["missing"] and report["unexpected"]


def test_expected_ambiguity_does_not_hide_provider_failure(monkeypatch, tmp_path):
    from scripts.check_measurement_review import evaluate_case
    from utils import measurement_review as module

    def fail(*args, **kwargs):
        raise ModelProviderError("service unavailable")

    monkeypatch.setattr(module, "review_measurements", fail)
    report = evaluate_case(
        {
            "name": "unavailable",
            "source": SOURCE,
            "expected": [],
            "expect_evidence_error": True,
        },
        tmp_path,
    )
    assert not report["passed"]


def test_duplicate_reference_rows_are_rejected_before_evaluation():
    from scripts.check_measurement_review import validate_cases

    with pytest.raises(ValueError, match="Duplicate reference"):
        validate_cases(
            [
                {
                    "name": "duplicate",
                    "source": SOURCE,
                    "expected": [observation(), observation()],
                }
            ]
        )


@pytest.mark.parametrize(
    "error",
    [ModelEvidenceError("missing quote"), ModelEvidenceError("fabricated source")],
)
def test_invalid_grounding_cannot_pass_expected_ambiguity(monkeypatch, tmp_path, error):
    from scripts.check_measurement_review import evaluate_case
    from utils import measurement_review as module

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(module, "review_measurements", fail)
    report = evaluate_case(
        {
            "name": "invalid-evidence",
            "source": SOURCE,
            "expected": [],
            "expect_evidence_error": True,
        },
        tmp_path,
    )
    assert not report["passed"]


def test_identifier_namespaces_do_not_collide():
    from utils.measurement_contract import measurement_rows

    request = MeasurementRequest(SOURCE)
    result = request.parse(
        json.dumps(
            candidate(
                observation(compound="1"), observation(compound="1", id_kind="seq_id")
            )
        )
    )
    rows = measurement_rows(result)["rows"]
    assert len(rows) == 2
    assert {row["id_kind"] for row in rows} == {"compound", "seq_id"}


def test_swapped_conditions_fail_even_when_value_sets_match(monkeypatch, tmp_path):
    from scripts.check_measurement_review import evaluate_case
    from utils import measurement_review as module

    expected = [
        observation(conditions="pH 6.0"),
        observation(raw="24", conditions="pH 8.0"),
    ]
    result = {
        "rows": [
            {
                "compound_id": "6a",
                "id_kind": "compound",
                "assays": {
                    "IC50 at pH 6.0": observation(raw="24", conditions="pH 6.0"),
                    "IC50 at pH 8.0": observation(conditions="pH 8.0"),
                },
            }
        ],
        "review": {"revisions": 0},
    }
    monkeypatch.setattr(module, "review_measurements", lambda *args, **kwargs: result)
    report = evaluate_case(
        {"name": "swapped-conditions", "source": SOURCE, "expected": expected}, tmp_path
    )
    assert not report["passed"] and len(report["missing"]) == 2


def test_scorer_accepts_real_measurement_row_shape(monkeypatch, tmp_path):
    from scripts.check_measurement_review import evaluate_case
    from utils import measurement_review as module
    from utils.measurement_contract import measurement_rows

    result = measurement_rows(
        MeasurementRequest(SOURCE).parse(json.dumps(candidate(observation())))
    )
    result["review"] = {"revisions": 0}
    monkeypatch.setattr(module, "review_measurements", lambda *args, **kwargs: result)
    report = evaluate_case(
        {"name": "valid-shape", "source": SOURCE, "expected": [observation()]}, tmp_path
    )
    assert report["passed"] and not report["missing"] and not report["unexpected"]


def test_cli_runs_keep_separate_artifacts(monkeypatch, tmp_path):
    from scripts import check_measurement_review as script

    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps([{"name": "absence", "source": "No results.", "expected": []}])
    )
    output = tmp_path / "results"

    def evaluate(case, directory, timeout, call_timeout, trial):
        return {
            "name": case["name"],
            "trial": trial,
            "passed": True,
            "elapsed_seconds": 0,
            "result": {},
        }

    monkeypatch.setattr(script, "evaluate_case", evaluate)
    monkeypatch.setattr(
        script.sys,
        "argv",
        ["check", "--cases", str(cases), "--out", str(output), "--repeats", "2"],
    )
    assert script.main() == script.main() == 0
    reports = list(output.glob("*/report.json"))
    assert len(reports) == 2
    for path in reports:
        report = json.loads(path.read_text())
        assert report["cases"] == 2 and report["consistent_cases"] == 1
        assert [item["trial"] for item in report["results"]] == [1, 2]


def test_conflicting_aliases_cannot_duplicate_typed_measurement():
    request = MeasurementRequest(SOURCE)
    with pytest.raises(ModelContractError, match="measurement identity"):
        request.parse(
            json.dumps(
                candidate(
                    observation(),
                    observation(raw="24", assay_name="Alternate IC50 label"),
                )
            )
        )


def test_unresolved_review_retains_actionable_details():
    responses = [candidate(observation()), review("unresolved", "conflict")]
    with pytest.raises(ModelEvidenceError) as caught:
        review_measurements(SOURCE, runner=runner_for(responses, []))
    assert caught.value.details["review"]["issues"][0]["kind"] == "conflict"
    assert (
        caught.value.details["history"][0]["candidate"]["observations"][0]["raw"]
        == responses[0]["observations"][0]["raw"]
    )


def test_exhausted_review_retains_correction_request():
    with pytest.raises(ModelDownstreamVerificationError) as caught:
        review_measurements(
            SOURCE,
            max_revisions=0,
            runner=runner_for([candidate(observation()), review("revise")], []),
        )
    assert caught.value.details["review"]["issues"][0]["evidence"] == [
        {**REFERENCE, "quote": SOURCE}
    ]


def test_uncertain_transcription_requires_a_reason():
    request = MeasurementRequest(SOURCE)
    with pytest.raises(ModelContractError):
        request.parse(
            json.dumps({"status": "uncertain", "reason": "", "observations": []})
        )
    result = request.parse(
        json.dumps(
            {
                "status": "uncertain",
                "reason": "Two values lack compound ownership",
                "observations": [],
                "source_issues": [source_issue()],
            }
        )
    )
    assert result["status"] == "uncertain" and result["reason"]


def test_conflicting_reference_values_are_rejected_before_calls():
    from scripts.check_measurement_review import validate_cases

    with pytest.raises(ValueError, match="Duplicate reference measurement identity"):
        validate_cases(
            [
                {
                    "name": "conflicting-gold",
                    "source": SOURCE,
                    "expected": [observation(), observation(raw="24")],
                }
            ]
        )


def test_rescoring_is_traceable_and_cannot_change_model_input(tmp_path):
    from scripts.check_measurement_review import rescore_run
    from utils.measurement_contract import measurement_rows

    case = {
        "name": "reference-review",
        "source": SOURCE,
        "expected": [observation(conditions="rat plasma")],
    }
    directory = tmp_path / "trial_1" / case["name"]
    directory.mkdir(parents=True)
    original = {**case, "expected": [observation()]}
    (directory / "input.json").write_text(json.dumps(original))
    result = measurement_rows(
        MeasurementRequest(SOURCE).parse(
            json.dumps(candidate(observation(conditions="rat plasma")))
        )
    )
    report = {
        "repeats": 1,
        "results": [
            {"name": case["name"], "trial": 1, "passed": False, "result": result}
        ],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    derived = tmp_path / "derived"
    results, repeats = rescore_run(tmp_path, [case], output=derived)
    assert repeats == 1 and results[0]["passed"]
    (derived / "report.json").write_text(
        json.dumps({"repeats": repeats, "results": results})
    )
    assert rescore_run(derived, [case])[0][0]["passed"]
    assert json.loads(path.read_text()) == report
    with pytest.raises(ValueError, match="model inputs"):
        rescore_run(tmp_path, [{**case, "source": SOURCE + " changed"}])


def test_rescoring_cannot_accept_an_incomplete_repeat_set(tmp_path):
    from scripts.check_measurement_review import rescore_run

    (tmp_path / "report.json").write_text(
        json.dumps({"repeats": 2, "results": [{"name": "incomplete", "trial": 1}]})
    )
    with pytest.raises(ValueError, match="incomplete"):
        rescore_run(tmp_path, [{"name": "incomplete"}])


def test_source_excerpt_is_derived_without_retyping_unicode():
    source = "Compound | IC50 (µM)\r\n6a | 12\u00a0±\u20090.3\r\n"
    result = MeasurementRequest(source).parse(
        json.dumps(candidate(observation(evidence=[{"line_start": 2, "line_end": 2}])))
    )
    assert (
        result["observations"][0]["evidence"][0]["quote"]
        == "6a | 12\u00a0±\u20090.3\r\n"
    )


def test_model_cannot_supply_or_rewrite_source_excerpt():
    with pytest.raises(ModelContractError, match="reference fields"):
        MeasurementRequest(SOURCE).parse(
            json.dumps(
                candidate(observation(evidence=[{**REFERENCE, "quote": "invented"}]))
            )
        )


def test_approved_context_wording_does_not_allow_missing_or_swapped_conditions():
    from scripts.check_measurement_review import score_result, validate_cases
    from utils.measurement_contract import measurement_rows

    case = {
        "name": "context-alias",
        "source": SOURCE,
        "expected": [observation(conditions="rat plasma")],
        "condition_aliases": {"rat plasma": ["Experimental rat plasma measurements"]},
    }
    validate_cases([case])
    for condition, expected in [
        ("Experimental rat plasma measurements", True),
        (None, False),
        ("human plasma", False),
    ]:
        result = measurement_rows(
            MeasurementRequest(SOURCE).parse(
                json.dumps(candidate(observation(conditions=condition)))
            )
        )
        assert score_result(result, case)["passed"] is expected
        assert result["rows"][0]["assays"]["IC50 (nM)"]["conditions"] == condition


def test_ambiguous_context_aliases_are_rejected():
    from scripts.check_measurement_review import validate_cases

    with pytest.raises(ValueError, match="unambiguous"):
        validate_cases(
            [
                {
                    "name": "bad-aliases",
                    "source": SOURCE,
                    "expected": [],
                    "condition_aliases": {
                        "rat plasma": ["plasma"],
                        "human plasma": ["plasma"],
                    },
                }
            ]
        )


def test_alias_equivalent_references_cannot_shrink_denominator():
    from scripts.check_measurement_review import score_result, validate_cases
    from utils.measurement_contract import measurement_rows

    case = {
        "name": "duplicate-alias-reference",
        "source": SOURCE,
        "expected": [
            observation(conditions="rat plasma"),
            observation(conditions="rat plasma measurements"),
        ],
        "condition_aliases": {"rat plasma": ["rat plasma measurements"]},
    }
    with pytest.raises(ValueError, match="Duplicate reference"):
        validate_cases([case])
    result = measurement_rows(
        MeasurementRequest(SOURCE).parse(
            json.dumps(candidate(observation(conditions="rat plasma")))
        )
    )
    assert not score_result(result, case)["passed"]


def test_uncertain_extraction_can_be_repaired_after_independent_source_review():
    calls = []
    draft = {
        "status": "uncertain",
        "reason": "A data table is required",
        "observations": [],
        "source_issues": [source_issue("unsupported")],
    }
    correction = review("revise")
    correction["question_decisions"] = [question_decision("q1", "refuted")]
    result = review_measurements(
        SOURCE,
        runner=runner_for(
            [draft, correction, candidate(observation()), review()], calls
        ),
    )
    assert result["rows"] and len(calls) == 4
    assert "A data table is required" not in calls[1]["prompt"]


def test_uncertain_draft_cannot_be_accepted_without_resolution():
    request = MeasurementRequest(SOURCE)
    with pytest.raises(ModelContractError, match="uncertain candidate"):
        parse_review(json.dumps(review()), request, candidate={"status": "uncertain"})


def test_ambiguity_grading_requires_the_correct_conflict_and_subjects():
    from scripts.check_measurement_review import ambiguity_matches

    case = {
        "expect_evidence_error": True,
        "expected_issue_kinds": ["conflict"],
        "expected_subjects": [
            {"id_kind": "compound", "compound_id": "6h"},
            {"id_kind": "compound", "compound_id": "6j"},
        ],
    }
    assert not ambiguity_matches({"reason": "No data table"}, case)
    details = {"review": review("unresolved", "unsupported")}
    assert not ambiguity_matches(details, case)
    details["review"]["issues"][0]["kind"] = "conflict"
    assert not ambiguity_matches(details, case)
    details["review"]["issues"][0]["subjects"] = case["expected_subjects"]
    assert ambiguity_matches(details, case)


@pytest.mark.parametrize(
    "details",
    [
        None,
        [],
        {"review": None},
        {"review": {"status": "unresolved", "issues": [None]}},
        {
            "review": {
                "status": "unresolved",
                "issues": [
                    {
                        "kind": "conflict",
                        "subjects": [{"id_kind": [], "compound_id": "6a"}],
                    }
                ],
            }
        },
    ],
)
def test_malformed_ambiguity_proof_is_rejected_without_crashing(details):
    from scripts.check_measurement_review import ambiguity_matches

    assert not ambiguity_matches(details, {"expect_evidence_error": True})


@pytest.mark.parametrize(
    "status,resolution", [("revise", "source_review"), ("unresolved", "correctable")]
)
def test_review_verdict_must_match_source_resolvability(status, resolution):
    value = review(status)
    value["issues"][0]["resolution"] = resolution
    with pytest.raises(ModelContractError, match="Source-review"):
        parse_review(json.dumps(value), MeasurementRequest(SOURCE))


def test_valid_source_question_survives_unrelated_contract_repair():
    request = MeasurementRequest(SOURCE)
    malformed = {
        "status": "complete",
        "reason": "",
        "observations": "invalid",
        "source_issues": [source_issue("conflict")],
    }
    with pytest.raises(ModelContractError):
        request.parse(json.dumps(malformed))
    questions = request.source_questions(pending_only=True)
    assert len(questions) == 1 and questions[0]["question_id"] == "q1"
    request.parse(json.dumps(candidate(observation())))
    assert request.source_questions(pending_only=True) == questions
    accepted = review()
    with pytest.raises(ModelContractError, match="Missing decisions"):
        parse_review(
            json.dumps(accepted),
            request,
            candidate=candidate(observation()),
            questions=questions,
        )


def test_question_snapshots_cannot_rewrite_host_claims():
    request = MeasurementRequest(SOURCE)
    issue = request.validate_source_issue(source_issue())
    request.register_source_issue(issue, "response")
    snapshot = request.source_questions()
    snapshot[0]["issue"]["description"] = "overwritten"
    assert request.source_questions()[0]["issue"]["description"] == issue["description"]


def test_confirmed_source_question_blocks_repair_and_acceptance():
    request = MeasurementRequest(SOURCE)
    request.register_source_issue(
        request.validate_source_issue(source_issue("conflict")), "response"
    )
    value = review("revise", "conflict")
    value["question_decisions"] = [question_decision("q1", "confirmed")]
    with pytest.raises(ModelContractError, match="confirmed source question"):
        parse_review(
            json.dumps(value),
            request,
            questions=request.source_questions(pending_only=True),
        )


def test_source_question_decisions_require_exact_coverage():
    request = MeasurementRequest(SOURCE)
    request.register_source_issue(
        request.validate_source_issue(source_issue()), "response"
    )
    for decisions in [
        [],
        [question_decision("unknown", "refuted")],
        [question_decision("q1", "refuted"), question_decision("q1", "refuted")],
    ]:
        value = review()
        value["question_decisions"] = decisions
        with pytest.raises(ModelContractError):
            parse_review(
                json.dumps(value),
                request,
                questions=request.source_questions(pending_only=True),
            )


@pytest.mark.parametrize("stage", ["extract", "review"])
def test_source_questions_survive_provider_contract_repair(
    monkeypatch, tmp_path, stage
):
    from types import SimpleNamespace

    from utils import llm_utils as llm
    from utils import model_harness as harness

    harness._TASK_CIRCUIT_STATE.clear()
    concern = source_issue("conflict")
    unresolved = review("unresolved", "conflict")
    unresolved["question_decisions"] = [question_decision("q1", "confirmed")]
    bad_review = review("unresolved", "conflict")
    bad_review["status"] = "accepted"
    bad_extract = {
        "status": "complete",
        "reason": "",
        "observations": "bad",
        "source_issues": [concern],
    }
    initial = candidate(observation())
    responses = (
        [bad_extract, initial, unresolved]
        if stage == "extract"
        else [initial, bad_review, unresolved]
    )
    raw_responses = [json.dumps(item) for item in responses]
    replies = iter(raw_responses)
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content=next(replies)))],
        )

    def factory(**kwargs):
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=lambda: None,
        )

    monkeypatch.setattr(llm, "require_openai", lambda: factory)
    monkeypatch.setattr(llm, "get_retry_delays", lambda *args, **kwargs: [0])
    with pytest.raises(ModelEvidenceError) as caught:
        review_measurements(SOURCE, audit_path=tmp_path / "calls.jsonl")
    questions = caught.value.details["source_questions"]
    assert questions[0]["question_id"] == "q1" and questions[0]["state"] == "confirmed"
    assert questions[0]["issue"]["scope"]["reported_values"] == ["24"]
    assert len(requests) == 3
    assert '"question_id": "q1"' in requests[-1]["messages"][1]["content"]
    schema = requests[-1]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["question_decisions"]["items"]["properties"][
        "question_id"
    ]["enum"] == ["q1"]
    if stage == "extract":
        assert not caught.value.details["history"][0]["candidate"]["source_issues"]
    else:
        first_schema = requests[1]["response_format"]["json_schema"]["schema"]
        assert first_schema["properties"]["question_decisions"]["maxItems"] == 0
    records = [
        json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    assert len(records) == 3 and [item["ok"] for item in records].count(False) == 1


def test_reference_expression_variants_do_not_rewrite_values_or_units():
    from scripts.check_measurement_review import score_result
    from utils.measurement_contract import measurement_rows

    case = {
        "name": "expression-variants",
        "source": SOURCE,
        "expected": [observation(raw="12")],
        "raw_aliases": {"12": ["12 nM"]},
    }
    for value, unit, expected in [
        (12, "nM", True),
        (120, "nM", False),
        (12, "µM", False),
    ]:
        item = observation(value=value, unit=unit)
        item["raw"] = "12 nM"
        result = measurement_rows(
            MeasurementRequest(SOURCE).parse(json.dumps(candidate(item)))
        )
        assert score_result(result, case)["passed"] is expected
        assert result["rows"][0]["assays"]["IC50 (nM)"]["raw"] == "12 nM"


def test_question_id_enum_does_not_constrain_scientific_identifiers():
    from utils.measurement_review import build_review_schema

    first = build_review_schema([{"question_id": "q1"}])
    second = build_review_schema([{"question_id": "q2"}])
    for schema, identifier in [(first, "q1"), (second, "q2")]:
        properties = schema["properties"]
        assert properties["question_decisions"]["items"]["properties"]["question_id"][
            "enum"
        ] == [identifier]
        issue = properties["issues"]["items"]["properties"]
        assert issue["subjects"]["items"]["properties"]["compound_id"] == {
            "type": "string",
            "minLength": 1,
        }
        assert issue["scope"]["properties"]["reported_values"]["items"] == {
            "type": "string",
            "minLength": 1,
        }


def test_wrong_response_envelope_cannot_erase_valid_source_question():
    request = MeasurementRequest(SOURCE)
    with pytest.raises(ModelContractError):
        parse_review(json.dumps({"source_issues": [source_issue("conflict")]}), request)
    assert len(request.source_questions(pending_only=True)) == 1
    with pytest.raises(ModelContractError):
        parse_review(
            json.dumps(review()),
            request,
            questions=request.source_questions(pending_only=True),
        )


def test_adjudication_may_refine_issue_kind_without_losing_subject_identity():
    request = MeasurementRequest(SOURCE)
    request.register_source_issue(
        request.validate_source_issue(source_issue("ambiguity")), "source-response"
    )
    value = review("unresolved", "conflict")
    value["question_decisions"] = [question_decision("q1", "confirmed")]
    result = parse_review(
        json.dumps(value),
        request,
        questions=request.source_questions(pending_only=True),
    )
    assert result["status"] == "unresolved"
    assert result["question_decisions"][0]["decision"] == "confirmed"


def test_final_result_assembly_cannot_accept_after_workflow_deadline(monkeypatch):
    from utils import measurement_review as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    assemble = module.measurement_rows

    def late(result):
        clock[0] = 51
        return assemble(result)

    monkeypatch.setattr(module, "measurement_rows", late)
    with pytest.raises(ModelTimeoutError):
        review_measurements(
            SOURCE,
            timeout_seconds=50,
            runner=runner_for([candidate(observation()), review()], []),
        )


@pytest.mark.parametrize("invalid_first", [True, False])
@pytest.mark.parametrize("separate_arrays", [True, False])
def test_invalid_source_question_cannot_hide_other_valid_questions(
    invalid_first, separate_arrays
):
    request = MeasurementRequest(SOURCE)
    invalid, valid = source_issue(), source_issue("conflict")
    invalid["evidence"][0]["line_end"] = 10000
    proposals = [invalid, valid] if invalid_first else [valid, invalid]
    payload = {"source_issues": proposals}
    if separate_arrays:
        payload = {
            "source_issues": proposals[:1],
            "issues": [{**proposals[1], "resolution": "source_review"}],
        }
    with pytest.raises(ModelContractError, match="Invalid source questions"):
        request.parse(json.dumps(payload))
    questions = request.source_questions(pending_only=True)
    assert len(questions) == 1
    assert questions[0]["issue"]["kind"] == "conflict"
    request.parse(json.dumps(candidate(observation())))
    with pytest.raises(ModelContractError, match="Missing decisions"):
        parse_review(json.dumps(review()), request)
