"""Source-grounded assay decisions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

from utils.model_harness import (
    ModelContractError,
    ModelEvidenceError,
    require_json_object,
)

TASK = "reconcile_detected_assay_names"
ACTIONS = ("keep", "merge", "exclude")
RESULT_KINDS = (
    "result_column",
    "result_table",
    "result_record",
    "result_value",
    "result_unit",
)
DECISION_KEYS = {
    "candidate_id",
    "action",
    "canonical_candidate_id",
    "confidence",
    "evidence_kind",
    "evidence",
    "reason",
}


@dataclass(frozen=True)
class AssayReconciliationRequest:
    candidates: list[dict[str, str]]
    sources: list[dict[str, Any]]

    @classmethod
    def from_pages(
        cls,
        assay_names: list[str] | tuple[str, ...],
        page_markdowns: dict[int | str, str],
    ) -> AssayReconciliationRequest:
        if not isinstance(assay_names, (list, tuple)):
            raise TypeError(
                "assay_names must be a list or tuple of original candidate strings"
            )
        candidates = []
        for index, name in enumerate(assay_names, 1):
            if not isinstance(name, str):
                raise TypeError(f"assay_names[{index - 1}] must be a string")
            if not name.strip():
                raise ValueError(f"assay_names[{index - 1}] must not be empty")
            candidates.append({"candidate_id": f"a{index}", "assay_name": name})
        if not isinstance(page_markdowns, dict):
            raise TypeError("page_markdowns must be a page-to-source-text object")
        pages = {}
        for raw_page, text in page_markdowns.items():
            if isinstance(raw_page, bool) or not isinstance(raw_page, (int, str)):
                raise TypeError(f"invalid source page type: {raw_page!r}")
            if isinstance(raw_page, str) and not raw_page.isdecimal():
                raise ValueError(f"invalid source page: {raw_page!r}")
            page = int(raw_page)
            if page < 1 or page in pages:
                raise ValueError(f"invalid or duplicate source page: {raw_page!r}")
            if not isinstance(text, str):
                raise TypeError(f"source page {page} must contain text")
            pages[page] = text
        sources = [
            {"source_id": f"p{page}", "page": page, "text": text}
            for page, text in sorted(pages.items())
            if text.strip()
        ]
        if candidates and not sources:
            raise ModelEvidenceError(
                f"{TASK} requires original source text; no non-empty pages were supplied"
            )
        return cls(candidates, sources)

    @cached_property
    def source_lines(self) -> dict[str, list[str]]:
        return {
            source["source_id"]: source["text"].splitlines(keepends=True)
            for source in self.sources
        }

    def build_source_context(self) -> list[dict[str, Any]]:
        return [
            {
                "source_id": source["source_id"],
                "page": source["page"],
                "lines": [
                    {"line": index, "text": text}
                    for index, text in enumerate(
                        self.source_lines[source["source_id"]], 1
                    )
                ],
            }
            for source in self.sources
        ]

    def build_response_schema(self) -> dict[str, Any]:
        ids = [candidate["candidate_id"] for candidate in self.candidates]
        source_ids = [source["source_id"] for source in self.sources]
        evidence = {
            "type": "object",
            "additionalProperties": False,
            "required": ["source_id", "line_start", "line_end"],
            "properties": {
                "source_id": {"type": "string", "enum": source_ids},
                "line_start": {"type": "integer", "minimum": 1},
                "line_end": {"type": "integer", "minimum": 1},
            },
        }
        decision = {
            "type": "object",
            "additionalProperties": False,
            "required": sorted(DECISION_KEYS),
            "properties": {
                "candidate_id": {"type": "string", "enum": ids},
                "action": {"type": "string", "enum": list(ACTIONS)},
                "canonical_candidate_id": {
                    "type": ["string", "null"],
                    "enum": [*ids, None],
                },
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "evidence_kind": {
                    "type": "string",
                    "enum": [*RESULT_KINDS, "non_result_context"],
                },
                "evidence": {"type": "array", "minItems": 1, "items": evidence},
                "reason": {"type": "string", "minLength": 1},
            },
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["status", "decisions", "reason"],
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["completed", "insufficient_evidence"],
                },
                "decisions": {"type": "array", "maxItems": len(ids), "items": decision},
                "reason": {"type": "string"},
            },
        }

    def parse_response(self, response_text: str) -> list[str]:
        payload = require_json_object(response_text, TASK, allow_wrapped=False)
        errors = []

        def error(path, message):
            errors.append(f"{path}: {message}")

        if set(payload) != {"status", "decisions", "reason"}:
            error("$", "expected exactly status, decisions, reason")
        status = payload.get("status")
        raw = payload.get("decisions")
        reason = payload.get("reason")
        if not isinstance(reason, str):
            error("$.reason", "must be a string")
        if status == "insufficient_evidence":
            if raw != [] or not isinstance(reason, str) or not reason.strip() or errors:
                raise ModelContractError(
                    f"{TASK} insufficient_evidence requires empty decisions and a non-empty reason"
                )
            raise ModelEvidenceError(f"{TASK}: {reason}")
        if status != "completed":
            error("$.status", "must be completed or insufficient_evidence")
        if not isinstance(raw, list):
            error(
                "$.decisions", "must be a list with exactly one decision per candidate"
            )
            raw = []
        candidates = {
            candidate["candidate_id"]: candidate["assay_name"]
            for candidate in self.candidates
        }
        sources = self.source_lines
        decisions = {}
        for index, decision in enumerate(raw):
            path = f"$.decisions[{index}]"
            if not isinstance(decision, dict):
                error(path, "must be an object")
                continue
            if set(decision) != DECISION_KEYS:
                error(path, "expected exactly " + ", ".join(sorted(DECISION_KEYS)))
            cid = decision.get("candidate_id")
            if not isinstance(cid, str) or cid not in candidates:
                error(path + ".candidate_id", "must reference an original candidate ID")
            elif cid in decisions:
                error(path + ".candidate_id", f"duplicate decision for {cid}")
            else:
                decisions[cid] = decision
            action = decision.get("action")
            confidence = decision.get("confidence")
            canonical = decision.get("canonical_candidate_id")
            if action not in ACTIONS:
                error(path + ".action", "must be keep, merge or exclude")
            if confidence not in ("high", "medium", "low"):
                error(path + ".confidence", "must be high, medium or low")
            if (
                not isinstance(decision.get("reason"), str)
                or not decision["reason"].strip()
            ):
                error(path + ".reason", "must be a non-empty source-based explanation")
            if action in ("merge", "exclude") and confidence != "high":
                error(path + ".confidence", "merge/exclude requires high confidence")
            if action == "exclude":
                if canonical is not None:
                    error(
                        path + ".canonical_candidate_id",
                        "exclude must explicitly use null",
                    )
                if decision.get("evidence_kind") != "non_result_context":
                    error(
                        path + ".evidence_kind", "exclude requires non_result_context"
                    )
            else:
                if not isinstance(canonical, str) or canonical not in candidates:
                    error(
                        path + ".canonical_candidate_id",
                        "must reference an original candidate ID",
                    )
                if action == "keep" and canonical != cid:
                    error(
                        path + ".canonical_candidate_id",
                        "keep must reference its own candidate ID",
                    )
                if action == "merge" and canonical == cid:
                    error(
                        path + ".canonical_candidate_id",
                        "merge must reference a different kept candidate",
                    )
                if decision.get("evidence_kind") not in RESULT_KINDS:
                    error(
                        path + ".evidence_kind",
                        "keep/merge requires a result evidence kind",
                    )
            evidence = decision.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                error(path + ".evidence", "must contain at least one source line range")
                continue
            for evidence_index, item in enumerate(evidence):
                epath = f"{path}.evidence[{evidence_index}]"
                if not isinstance(item, dict) or set(item) != {
                    "source_id",
                    "line_start",
                    "line_end",
                }:
                    error(epath, "expected exactly source_id, line_start, line_end")
                    continue
                source_id = item["source_id"]
                start, end = item["line_start"], item["line_end"]
                if not isinstance(source_id, str) or source_id not in sources:
                    error(epath + ".source_id", "unknown original source ID")
                elif type(start) is not int or type(end) is not int:
                    error(epath, "line_start and line_end must be integers")
                elif not 1 <= start <= end <= len(sources[source_id]):
                    error(
                        epath,
                        f"line range must satisfy 1 <= line_start <= line_end <= {len(sources[source_id])}",
                    )
                elif not "".join(sources[source_id][start - 1 : end]).strip():
                    error(epath, "line range must cite non-empty original source text")
        missing = candidates.keys() - decisions.keys()
        if missing:
            error("$.decisions", "missing candidate IDs: " + ", ".join(sorted(missing)))
        for cid, decision in decisions.items():
            if decision.get("action") == "merge":
                canonical = decision.get("canonical_candidate_id")
                target = (
                    decisions.get(canonical) if isinstance(canonical, str) else None
                )
                if not target or target.get("action") != "keep":
                    error(
                        f"candidate {cid}",
                        "merge destination must have an explicit keep decision; chains/cycles are invalid",
                    )
        if errors:
            raise ModelContractError(
                f"{TASK} invalid response ({len(errors)} violations):\n"
                + "\n".join(errors[:40])
            )
        return [
            candidates[cid] for cid in candidates if decisions[cid]["action"] == "keep"
        ]
