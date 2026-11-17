"""Contracts for source-linked compound measurements."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy

from utils.model_harness import (
    ModelContractError,
    require_json_object,
)


def object_schema(properties: dict) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


TEXT = {"type": "string", "minLength": 1}
IDENTIFIER_KINDS = ("compound", "seq_id", "example", "unknown")
ISSUE_KINDS = ("omission", "conflict", "unsupported", "transcription", "ambiguity")
REFERENCE_SCHEMA = object_schema(
    {
        "line_start": {"type": "integer", "minimum": 1},
        "line_end": {"type": "integer", "minimum": 1},
    }
)
SUBJECT_SCHEMA = object_schema(
    {
        "id_kind": {"type": "string", "enum": list(IDENTIFIER_KINDS)},
        "compound_id": TEXT,
    }
)
SOURCE_ISSUE_SCHEMA = object_schema(
    {
        "kind": {"type": "string", "enum": list(ISSUE_KINDS)},
        "description": {**TEXT, "maxLength": 600},
        "subjects": {"type": "array", "items": SUBJECT_SCHEMA},
        "scope": object_schema(
            {
                "endpoint": {"type": ["string", "null"]},
                "method": {"type": ["string", "null"]},
                "reported_values": {"type": "array", "items": TEXT},
            }
        ),
        "evidence": {"type": "array", "minItems": 1, "items": REFERENCE_SCHEMA},
    }
)
OBSERVATION_SCHEMA = object_schema(
    {
        "compound_id": TEXT,
        "id_kind": {
            "type": "string",
            "enum": list(IDENTIFIER_KINDS),
        },
        "assay_name": TEXT,
        "endpoint": TEXT,
        "target": {"type": ["string", "null"]},
        "conditions": {"type": ["string", "null"]},
        "source_kind": {
            "enum": ["experimental", "calculated", "predicted", "unknown"],
            "type": "string",
            "description": "The source-stated method of this result column or record, including cells with missing values.",
        },
        "raw": TEXT,
        "value": {"type": ["number", "null"]},
        "operator": {"type": "string", "enum": ["", "<", ">", "≤", "≥", "~", "="]},
        "unit": {"type": ["string", "null"]},
        "evidence": {"type": "array", "minItems": 1, "items": REFERENCE_SCHEMA},
    }
)
RESPONSE_SCHEMA = object_schema(
    {
        "status": {"type": "string", "enum": ["complete", "absent", "uncertain"]},
        "reason": {"type": "string"},
        "observations": {"type": "array", "items": OBSERVATION_SCHEMA},
        "source_issues": {"type": "array", "items": SOURCE_ISSUE_SCHEMA},
    }
)


def measurement_identity(item: dict) -> tuple:
    return tuple(
        item[field]
        for field in (
            "id_kind",
            "compound_id",
            "endpoint",
            "target",
            "conditions",
            "unit",
            "source_kind",
        )
    )


class MeasurementRequest:
    def __init__(self, source: str):
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Non-empty measurement source required")
        self.lines = tuple(source.splitlines())
        self.source_lines = tuple(source.splitlines(keepends=True))
        self._questions = {}
        self._question_keys = {}

    def register_source_issue(self, issue: dict, response_text: str) -> str:
        key = json.dumps(issue, ensure_ascii=False, sort_keys=True)
        origin = hashlib.sha256(response_text.encode()).hexdigest()
        question_id = self._question_keys.get(key)
        if question_id is None:
            question_id = f"q{len(self._questions) + 1}"
            self._question_keys[key] = question_id
            self._questions[question_id] = {
                "question_id": question_id,
                "issue": deepcopy(issue),
                "state": "pending",
                "response_hashes": [],
            }
        if origin not in self._questions[question_id]["response_hashes"]:
            self._questions[question_id]["response_hashes"].append(origin)
            if self._questions[question_id]["state"] == "refuted":
                self._questions[question_id]["state"] = "pending"
        return question_id

    def source_questions(self, *, pending_only=False) -> list[dict]:
        return deepcopy(
            [
                question
                for question in self._questions.values()
                if not pending_only or question["state"] == "pending"
            ]
        )

    def record_question_decisions(self, decisions: list[dict]) -> None:
        for decision in decisions:
            question = self._questions[decision["question_id"]]
            question["state"] = decision["decision"]
            question.setdefault("decisions", []).append(deepcopy(decision))

    def capture_source_questions(self, payload: dict, response_text: str) -> list[str]:
        identifiers, errors = [], []
        proposals = payload.get("source_issues")
        proposals = list(proposals) if isinstance(proposals, list) else []
        issues = payload.get("issues")
        if isinstance(issues, list):
            proposals.extend(
                {key: value for key, value in issue.items() if key != "resolution"}
                for issue in issues
                if isinstance(issue, dict)
                and issue.get("resolution") == "source_review"
            )
        for proposal in proposals:
            try:
                validated = self.validate_source_issue(proposal)
            except ModelContractError as exc:
                errors.append(str(exc))
            else:
                identifiers.append(self.register_source_issue(validated, response_text))
        if errors:
            raise ModelContractError("Invalid source questions: " + "; ".join(errors))
        return list(dict.fromkeys(identifiers))

    def source_context(self) -> str:
        return "\n".join(f"{index}: {line}" for index, line in enumerate(self.lines, 1))

    def parse(self, text: str) -> dict:
        result = require_json_object(text, "activity_tables", allow_wrapped=False)
        self.capture_source_questions(result, text)
        source_issues = result.get("source_issues")
        if set(result) != set(RESPONSE_SCHEMA["required"]):
            raise ModelContractError(
                "Measurement response fields do not match the contract"
            )
        status, observations = result["status"], result["observations"]
        if not isinstance(status, str) or status not in {
            "complete",
            "absent",
            "uncertain",
        }:
            raise ModelContractError("Unknown measurement transcription status")
        if not isinstance(observations, list):
            raise ModelContractError("observations must be an array")
        if not isinstance(source_issues, list):
            raise ModelContractError("source_issues must be an array")
        if status == "uncertain":
            if observations:
                raise ModelContractError(
                    "Uncertain transcription requires empty observations"
                )
            if not isinstance(result["reason"], str) or not result["reason"].strip():
                raise ModelContractError(
                    "Uncertain transcription requires a source-based reason"
                )
            if not source_issues:
                raise ModelContractError(
                    "Uncertain transcription requires specific source issues"
                )
            result["source_issues"] = [
                self.validate_source_issue(issue) for issue in source_issues
            ]
            return result
        if source_issues:
            raise ModelContractError(
                "A completed or absent draft cannot claim unresolved source issues"
            )
        if not isinstance(result["reason"], str):
            raise ModelContractError("Transcription reason must be a string")
        if (status == "absent") != (len(observations) == 0):
            raise ModelContractError(
                "Measurement status and observation count disagree"
            )
        seen, labels = set(), set()
        for item in observations:
            self.validate_observation(item)
            key = measurement_identity(item)
            if key in seen:
                raise ModelContractError(
                    "Repeated measurement identity; resolve duplicated or conflicting source assignments"
                )
            seen.add(key)
            label = (item["id_kind"], item["compound_id"], item["assay_name"])
            if label in labels:
                raise ModelContractError(
                    "Distinct measurements require distinct assay names"
                )
            labels.add(label)
        return result

    def validate_source_issue(self, issue) -> dict:
        if not isinstance(issue, dict) or set(issue) != set(
            SOURCE_ISSUE_SCHEMA["required"]
        ):
            raise ModelContractError("Source issue fields do not match the contract")
        if issue["kind"] not in ISSUE_KINDS:
            raise ModelContractError("Unknown source issue kind")
        if (
            not isinstance(issue["description"], str)
            or not issue["description"].strip()
            or len(issue["description"]) > 600
        ):
            raise ModelContractError(
                "Source issue description must contain 1 to 600 characters"
            )
        subjects = issue["subjects"]
        if not isinstance(subjects, list):
            raise ModelContractError("Issue subjects must be an array")
        for subject in subjects:
            if not isinstance(subject, dict) or set(subject) != set(
                SUBJECT_SCHEMA["required"]
            ):
                raise ModelContractError(
                    "Issue subject fields do not match the contract"
                )
            if (
                subject["id_kind"] not in IDENTIFIER_KINDS
                or not isinstance(subject["compound_id"], str)
                or not subject["compound_id"].strip()
            ):
                raise ModelContractError("Invalid issue subject identity")
        scope = issue["scope"]
        if not isinstance(scope, dict) or set(scope) != {
            "endpoint",
            "method",
            "reported_values",
        }:
            raise ModelContractError(
                "Issue scope requires endpoint, method, and reported_values"
            )
        for field in ("endpoint", "method"):
            if scope[field] is not None and (
                not isinstance(scope[field], str) or not scope[field].strip()
            ):
                raise ModelContractError(
                    "Issue scope fields must be non-empty strings or null"
                )
        if not isinstance(scope["reported_values"], list) or any(
            not isinstance(value, str) or not value.strip()
            for value in scope["reported_values"]
        ):
            raise ModelContractError("Issue reported_values must be non-empty strings")
        return {**issue, "evidence": self.validate_evidence(issue["evidence"])}

    def validate_evidence(self, references) -> list[dict]:
        if not isinstance(references, list) or not references:
            raise ModelContractError("Source evidence is missing")
        grounded = []
        for reference in references:
            if not isinstance(reference, dict) or set(reference) != set(
                REFERENCE_SCHEMA["required"]
            ):
                raise ModelContractError(
                    "Source reference fields do not match the contract"
                )
            start, end = reference["line_start"], reference["line_end"]
            if (
                type(start) is not int
                or type(end) is not int
                or not 1 <= start <= end <= len(self.lines)
            ):
                raise ModelContractError(
                    f"Source line range must satisfy 1 <= start <= end <= {len(self.lines)}"
                )
            quote = "".join(self.source_lines[start - 1 : end])
            if not quote.strip():
                raise ModelContractError("Source range must contain evidence text")
            grounded.append({**reference, "quote": quote})
        return grounded

    def validate_observation(self, item) -> None:
        if not isinstance(item, dict) or set(item) != set(
            OBSERVATION_SCHEMA["required"]
        ):
            raise ModelContractError("Observation fields do not match the contract")
        for field in ("compound_id", "id_kind", "assay_name", "endpoint", "raw"):
            if not isinstance(item[field], str) or not item[field].strip():
                raise ModelContractError(f"Observation requires non-empty {field}")
        for field in ("target", "unit", "conditions"):
            if item[field] is not None and (
                not isinstance(item[field], str) or not item[field].strip()
            ):
                raise ModelContractError(f"{field} must be a non-empty string or null")
        if item["id_kind"] not in IDENTIFIER_KINDS:
            raise ModelContractError("Unknown compound identifier namespace")
        if item["source_kind"] not in (
            "experimental",
            "calculated",
            "predicted",
            "unknown",
        ):
            raise ModelContractError("Unknown measurement provenance")
        if item["operator"] not in ("", "<", ">", "≤", "≥", "~", "="):
            raise ModelContractError("Unknown measurement comparison operator")
        value = item["value"]
        if value is not None and (
            type(value) not in (int, float) or not math.isfinite(value)
        ):
            raise ModelContractError("Measurement value must be finite or null")
        item["evidence"] = self.validate_evidence(item["evidence"])


def measurement_rows(result: dict) -> dict:
    rows = {}
    observations = result["observations"]
    for item in observations:
        row = rows.setdefault(
            (item["id_kind"], item["compound_id"]),
            {
                "compound_id": item["compound_id"],
                "id_kind": item["id_kind"],
                "assays": {},
            },
        )
        row["assays"][item["assay_name"]] = {
            key: value
            for key, value in item.items()
            if key not in {"compound_id", "id_kind", "assay_name"}
        }
        row["assays"][item["assay_name"]]["unit_missing"] = item["unit"] is None
    missing_units = sum(item["unit"] is None for item in observations)
    return {
        "ok": True,
        "status": result["status"],
        "rows": list(rows.values()),
        "assay_types": sorted({item["assay_name"] for item in observations}),
        "gaps": [f"{missing_units} measurements have no stated unit"]
        if missing_units
        else [],
    }
