"""Evidence-based measurement review and bounded revision."""

from __future__ import annotations

import json
import math
import time
from copy import deepcopy

from utils.measurement_contract import (
    REFERENCE_SCHEMA,
    RESPONSE_SCHEMA,
    SOURCE_ISSUE_SCHEMA,
    TEXT,
    MeasurementRequest,
    measurement_rows,
    object_schema,
)
from utils.model_harness import (
    ModelContractError,
    ModelDownstreamVerificationError,
    ModelHarnessError,
    ModelSourceAmbiguityError,
    ModelTimeoutError,
    require_json_object,
)
from utils.skill_prompt_loader import render_skill_reference

_ISSUE_SCHEMA = object_schema(
    {
        **SOURCE_ISSUE_SCHEMA["properties"],
        "resolution": {"type": "string", "enum": ["correctable", "source_review"]},
    }
)
_QUESTION_DECISION_SCHEMA = object_schema(
    {
        "question_id": TEXT,
        "decision": {"type": "string", "enum": ["confirmed", "refuted"]},
        "explanation": {**TEXT, "maxLength": 600},
        "evidence": {"type": "array", "minItems": 1, "items": REFERENCE_SCHEMA},
    }
)
REVIEW_SCHEMA = object_schema(
    {
        "status": {"type": "string", "enum": ["accepted", "revise", "unresolved"]},
        "issues": {"type": "array", "items": _ISSUE_SCHEMA},
        "coverage": {"type": "array", "minItems": 1, "items": REFERENCE_SCHEMA},
        "question_decisions": {"type": "array", "items": _QUESTION_DECISION_SCHEMA},
    }
)

_EXTRACT = (
    render_skill_reference(
        "biocheminsight-text-models",
        "references/extract_compound_measurements_prompt.md",
    )
    + "\n"
)

_REVIEW = (
    render_skill_reference(
        "biocheminsight-text-models",
        "references/review_compound_measurements_prompt.md",
    )
    + "\n"
)


def build_review_schema(questions):
    schema = deepcopy(REVIEW_SCHEMA)
    decisions = schema["properties"]["question_decisions"]
    decisions.update(minItems=len(questions), maxItems=len(questions))
    if questions:
        fields = decisions["items"]["properties"]
        fields["question_id"] = {
            **fields["question_id"],
            "enum": [q["question_id"] for q in questions],
        }
    return schema


def _review_candidate(request, candidate, context, run, audit_path, revision):
    state = {}

    def build_request():
        questions = request.source_questions(pending_only=True)
        state["questions"] = questions
        prompt = (
            _REVIEW
            + context
            + "\nCANDIDATE:\n"
            + json.dumps(
                {
                    key: value
                    for key, value in candidate.items()
                    if key not in {"reason", "source_issues"}
                },
                ensure_ascii=False,
            )
        )
        prompt += "\nSOURCE QUESTIONS:\n" + json.dumps(
            [
                {"question_id": question["question_id"], "issue": question["issue"]}
                for question in questions
            ],
            ensure_ascii=False,
        )
        return prompt, build_review_schema(questions)

    prompt, schema = build_request()
    return run(
        task_name="review_compound_measurements",
        prompt=prompt,
        request_factory=build_request,
        parser=lambda text: parse_review(
            text, request, candidate=candidate, questions=state["questions"]
        ),
        response_schema=schema,
        audit_path=audit_path,
        metadata={"stage": "review", "revision": revision},
    )


class ModelPromptNeeded(ModelHarnessError):
    """The workflow reached a call that must be answered externally."""

    error_type = "model_prompt_needed"

    def __init__(self, call: dict):
        super().__init__("model call required", details=dict(call))
        self.call = dict(call)


class MeasurementReviewSession:
    """The measurement review workflow with every model answer supplied
    externally.

    Each ``next_prompt`` runs the workflow on a replay runner: recorded
    answers are served back, and the first call without a recorded answer
    raises out as the pending prompt. Submitting an answer advances the
    replay by one call. The final ``next_prompt`` returns None and the
    result is available on ``result``."""

    def __init__(self, source: str, *, audit_path=None, max_revisions: int = 1):
        self._source = source
        self._audit_path = audit_path
        self._max_revisions = max_revisions
        self._answers: list[Any] = []
        self._calls: list[dict] = []
        self.result: dict | None = None

    def _invoke(self) -> dict:
        answers = list(self._answers)

        def replay(**kwargs):
            index = len(self._calls)
            self._calls.append(dict(kwargs))
            if index < len(answers):
                parser = kwargs.get("parser")
                raw = answers[index]
                return parser(raw) if parser else raw
            raise ModelPromptNeeded(dict(kwargs))

        return review_measurements(
            self._source,
            audit_path=self._audit_path,
            max_revisions=self._max_revisions,
            runner=replay,
            timeout_seconds=10 ** 6,
            call_timeout_seconds=10 ** 6,
        )

    def next_prompt(self) -> dict | None:
        self._calls = []
        try:
            self.result = self._invoke()
        except ModelPromptNeeded as needed:
            return needed.call
        return None

    def submit(self, answer: Any) -> dict | None:
        self._answers.append(answer)
        return self.next_prompt()


def review_measurements(
    source: str,
    *,
    audit_path=None,
    max_revisions=1,
    runner=None,
    timeout_seconds=240,
    call_timeout_seconds=90,
) -> dict:
    if type(max_revisions) is not int or not 0 <= max_revisions <= 2:
        raise ValueError("max_revisions must be an integer between zero and two")
    for value in (timeout_seconds, call_timeout_seconds):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Measurement timeouts must be positive finite numbers")
    if runner is None:
        from utils.llm_utils import run_text_json_task

        runner = run_text_json_task
    deadline = time.monotonic() + timeout_seconds
    request = MeasurementRequest(source)
    history = []

    def diagnostics(**extra):
        return {
            **extra,
            "source_questions": request.source_questions(),
            "history": deepcopy(history),
        }

    def run(**kwargs):
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModelTimeoutError(
                    "Measurement review exceeded its workflow deadline"
                )
            result = runner(
                **kwargs,
                retry=2,
                timeout_seconds=min(remaining, call_timeout_seconds),
                deadline=deadline,
                max_tokens=4096,
            )
            if time.monotonic() >= deadline:
                raise ModelTimeoutError(
                    "Measurement review exceeded its workflow deadline"
                )
            return result
        except ModelHarnessError as exc:
            exc.details = diagnostics(**(exc.details or {}))
            raise

    context = "\nSOURCE:\n" + request.source_context()
    candidate = run(
        task_name="extract_compound_measurements",
        prompt=_EXTRACT + context,
        parser=request.parse,
        response_schema=RESPONSE_SCHEMA,
        audit_path=audit_path,
        metadata={"stage": "extract", "revision": 0},
    )
    for revision in range(max_revisions + 1):
        review = _review_candidate(
            request, candidate, context, run, audit_path, revision
        )
        request.record_question_decisions(review["question_decisions"])
        direct = []
        questions_by_id = {
            question["question_id"]: question for question in request.source_questions()
        }
        decided = {decision["question_id"] for decision in review["question_decisions"]}
        for question_id in review["source_question_ids"]:
            if question_id in decided:
                continue
            question = questions_by_id[question_id]
            direct.append(
                {
                    "question_id": question_id,
                    "decision": "confirmed",
                    "explanation": question["issue"]["description"],
                    "evidence": question["issue"]["evidence"],
                    "origin": "review_issue",
                }
            )
        request.record_question_decisions(direct)
        history.append(
            {
                "candidate": candidate,
                "review": review,
                "source_questions": request.source_questions(),
            }
        )
        if review["status"] == "accepted":
            if request.source_questions(pending_only=True):
                raise ModelDownstreamVerificationError(
                    "Source questions remain unadjudicated",
                    details=diagnostics(review=review),
                )
            result = measurement_rows(candidate)
            result["review"] = {
                "status": "accepted",
                "revisions": revision,
                "history": history,
            }
            if time.monotonic() >= deadline:
                raise ModelTimeoutError(
                    "Measurement review exceeded its workflow deadline",
                    details=diagnostics(review=review),
                )
            return result
        if review["status"] == "unresolved":
            raise ModelSourceAmbiguityError(
                "Source conflicts or measurement ownership remain unresolved",
                details=diagnostics(review=review),
            )
        if revision == max_revisions:
            raise ModelDownstreamVerificationError(
                "Measurement review still requires correction after the revision budget",
                details=diagnostics(review=review),
            )
        candidate = run(
            task_name="revise_compound_measurements",
            prompt=_EXTRACT
            + context
            + "\nPREVIOUS CANDIDATE:\n"
            + json.dumps(candidate, ensure_ascii=False)
            + "\nSOURCE-SUPPORTED REVIEW:\n"
            + json.dumps(review, ensure_ascii=False)
            + "\nCorrect the identified problems and return the complete transcription, including unaffected results.",
            parser=request.parse,
            response_schema=RESPONSE_SCHEMA,
            audit_path=audit_path,
            metadata={"stage": "revise", "revision": revision + 1},
        )
    raise AssertionError("Unreachable review state")


def parse_review(
    text: str, request: MeasurementRequest, *, candidate=None, questions=None
) -> dict:
    expected_questions = (
        request.source_questions(pending_only=True) if questions is None else questions
    )
    result = require_json_object(
        text, "review_compound_measurements", allow_wrapped=False
    )
    source_question_ids = request.capture_source_questions(result, text)
    if set(result) != set(REVIEW_SCHEMA["required"]):
        raise ModelContractError("Measurement review fields do not match the contract")
    status, issues = result["status"], result["issues"]
    if not isinstance(status, str) or status not in {
        "accepted",
        "revise",
        "unresolved",
    }:
        raise ModelContractError("Unknown measurement review status")
    if not isinstance(issues, list) or ((status == "accepted") != (len(issues) == 0)):
        raise ModelContractError("Measurement review status and issues disagree")
    if (
        status == "accepted"
        and candidate is not None
        and candidate["status"] == "uncertain"
    ):
        raise ModelContractError(
            "An uncertain candidate requires revision or an unresolved source decision"
        )
    result["coverage"] = request.validate_evidence(result["coverage"])
    needs_source_review = False
    for index, issue in enumerate(issues):
        if not isinstance(issue, dict) or set(issue) != set(_ISSUE_SCHEMA["required"]):
            raise ModelContractError(
                "Measurement review issue fields do not match the contract"
            )
        if issue["resolution"] not in ("correctable", "source_review"):
            raise ModelContractError("Unknown issue resolution")
        needs_source_review = (
            needs_source_review or issue["resolution"] == "source_review"
        )
        validated = request.validate_source_issue(
            {key: value for key, value in issue.items() if key != "resolution"}
        )
        issues[index] = {**validated, "resolution": issue["resolution"]}
    if needs_source_review != (status == "unresolved"):
        raise ModelContractError(
            "Source-review issues require unresolved; correctable issues require revise"
        )
    _validate_question_decisions(
        result, expected_questions, request, source_question_ids
    )
    result["source_question_ids"] = list(dict.fromkeys(source_question_ids))
    return result


def _validate_question_decisions(result, questions, request, source_question_ids):
    expected = {question["question_id"]: question for question in questions}
    decisions, seen = result["question_decisions"], set()
    if not isinstance(decisions, list):
        raise ModelContractError("question_decisions must be an array")
    for decision in decisions:
        if not isinstance(decision, dict) or set(decision) != set(
            _QUESTION_DECISION_SCHEMA["required"]
        ):
            raise ModelContractError(
                "Question decision fields do not match the contract"
            )
        question_id = decision["question_id"]
        if (
            not isinstance(question_id, str)
            or question_id not in expected
            or question_id in seen
        ):
            raise ModelContractError(
                "Question IDs must cover every registered question exactly once"
            )
        seen.add(question_id)
        if decision["decision"] not in ("confirmed", "refuted"):
            raise ModelContractError("Question decision must be confirmed or refuted")
        explanation = decision["explanation"]
        if (
            not isinstance(explanation, str)
            or not explanation.strip()
            or len(explanation) > 600
        ):
            raise ModelContractError(
                "Question explanation must contain 1 to 600 characters"
            )
        decision["evidence"] = request.validate_evidence(decision["evidence"])
        if decision["decision"] == "confirmed":
            if result["status"] != "unresolved":
                raise ModelContractError(
                    "A confirmed source question requires an unresolved verdict"
                )
            concern = expected[question_id]["issue"]
            subjects = {
                (subject["id_kind"], subject["compound_id"])
                for subject in concern["subjects"]
            }
            matching = [
                issue
                for issue in result["issues"]
                if issue["resolution"] == "source_review"
                and subjects
                <= {
                    (subject["id_kind"], subject["compound_id"])
                    for subject in issue["subjects"]
                }
            ]
            if not matching:
                raise ModelContractError(
                    "Confirmed questions require a matching source issue and affected subjects"
                )
        elif question_id in source_question_ids:
            raise ModelContractError(
                "A refuted question cannot also be an unresolved source issue"
            )
    if seen != set(expected):
        raise ModelContractError(
            f"Missing decisions for source questions: {sorted(set(expected) - seen)}"
        )
