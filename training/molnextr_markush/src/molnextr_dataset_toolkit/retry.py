from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


csv.field_size_limit(sys.maxsize)


@dataclass(frozen=True)
class RetryAttempt:
    primary_row_id: str
    attempt_index: int
    attempt_count: int
    row_id: str
    render_seed: int
    accepted: bool = False
    stage: str = ""
    error: str = ""
    error_class: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_row_id": self.primary_row_id,
            "attempt_index": self.attempt_index,
            "attempt_count": self.attempt_count,
            "row_id": self.row_id,
            "render_seed": self.render_seed,
            "accepted": self.accepted,
            "stage": self.stage,
            "error": self.error,
            "error_class": self.error_class,
        }


@dataclass(frozen=True)
class RetryPolicy:
    attempt_count: int = 1
    policy: str = "renderer_level_layout_seed_retry_no_threshold_relaxation"
    deterministic_seed_namespace: str = "molnextr_markush_renderer_attempt_v1"

    def __post_init__(self) -> None:
        if self.attempt_count < 1:
            raise ValueError("attempt_count must be >= 1")

    def row_id_for_attempt(self, primary_row_id: str, attempt_index: int) -> str:
        if attempt_index < 0 or attempt_index >= self.attempt_count:
            raise ValueError("attempt_index outside retry policy attempt_count")
        return primary_row_id if attempt_index == 0 else f"{primary_row_id}_r{attempt_index:02d}"

    def seed_for_attempt(
        self,
        *,
        base_seed: int,
        selection_hash: str,
        primary_row_id: str,
        attempt_index: int,
        cxsmiles: str,
    ) -> int:
        payload = "\x1f".join(
            [
                self.deterministic_seed_namespace,
                str(base_seed),
                str(selection_hash or ""),
                str(primary_row_id),
                str(attempt_index),
                str(cxsmiles),
            ]
        )
        return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:15], 16) % (2**31 - 1)

    def attempts_for_candidate(
        self,
        *,
        base_seed: int,
        selection_hash: str,
        primary_row_id: str,
        cxsmiles: str,
    ) -> list[RetryAttempt]:
        return [
            RetryAttempt(
                primary_row_id=primary_row_id,
                attempt_index=index,
                attempt_count=self.attempt_count,
                row_id=self.row_id_for_attempt(primary_row_id, index),
                render_seed=self.seed_for_attempt(
                    base_seed=base_seed,
                    selection_hash=selection_hash,
                    primary_row_id=primary_row_id,
                    attempt_index=index,
                    cxsmiles=cxsmiles,
                ),
            )
            for index in range(self.attempt_count)
        ]

    def contract(self) -> dict[str, Any]:
        return {
            "schema_version": "markush_renderer_seed_retry_v1",
            "enabled": self.attempt_count > 1,
            "policy": self.policy,
            "attempt_count_per_source": self.attempt_count,
            "thresholds_unchanged": True,
            "coordinate_recomputed_from_selected_renderer_output": True,
            "same_source_group_for_all_attempts": True,
            "diagnostic_not_quality_relaxation": True,
        }


@dataclass
class RetryDiagnostics:
    csv_path: Path | None = None
    manifest_path: Path | None = None
    accepted_rows: int = 0
    selected_attempt_counts: Counter[int] = field(default_factory=Counter)
    final_failure_count: int = 0
    intermediate_retry_failure_count: int = 0
    failure_class_counts: Counter[str] = field(default_factory=Counter)
    retry_saved_failure_class_counts: Counter[str] = field(default_factory=Counter)
    policy_issues: list[str] = field(default_factory=list)

    @property
    def retry_selected_rows(self) -> int:
        return sum(count for attempt, count in self.selected_attempt_counts.items() if int(attempt) > 0)

    @property
    def retry_selected_fraction(self) -> float:
        return float(self.retry_selected_rows / self.accepted_rows) if self.accepted_rows else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "markush_renderer_retry_diagnostics_v1",
            "csv_path": str(self.csv_path or ""),
            "manifest_path": str(self.manifest_path or ""),
            "accepted_rows": int(self.accepted_rows),
            "selected_attempt_counts": {str(k): int(v) for k, v in sorted(self.selected_attempt_counts.items())},
            "retry_selected_rows": int(self.retry_selected_rows),
            "retry_selected_fraction": self.retry_selected_fraction,
            "final_failure_count": int(self.final_failure_count),
            "intermediate_retry_failure_count": int(self.intermediate_retry_failure_count),
            "failure_class_counts": dict(self.failure_class_counts),
            "retry_saved_failure_class_counts": dict(self.retry_saved_failure_class_counts),
            "policy_issues": list(self.policy_issues),
            "deterministic_layout_recommendations": deterministic_layout_recommendations(self),
            "engineering_policy": (
                "Renderer retry is allowed only as renderer-level layout diversity with unchanged gates. "
                "Frequent retry classes must feed deterministic source/layout policy improvements."
            ),
        }


def classify_generation_error(error: str) -> str:
    text = str(error or "").lower()
    if "nonlinear" in text and ("pose" in text or "warp" in text):
        return "formal_nonlinear_pose_preservation"
    if "line residual" in text or "line_constraint" in text or "line rmse" in text:
        return "svg_line_constraint_pose"
    if "intersection" in text or "anchor count" in text:
        return "svg_intersection_anchor_pose"
    if "scale ratio" in text or "affine" in text:
        return "affine_seed_or_scale"
    if "atom pair" in text or "too close" in text or "min_atom_pair" in text:
        return "atom_pair_too_close"
    if "accepted_candidate_filter" in text or "dummy_atom_coordinate" in text or "ocr" in text:
        return "accepted_candidate_or_substitution_anchor"
    if "kekule" in text or "smiles" in text or "cdk" in text or "depictor" in text:
        return "chemistry_or_depictor"
    if "blank" in text or "dense" in text or "readable" in text:
        return "visual_readability"
    return "other"


def deterministic_layout_recommendations(diagnostics: RetryDiagnostics) -> list[dict[str, Any]]:
    counts = diagnostics.retry_saved_failure_class_counts + diagnostics.failure_class_counts
    recommendations: list[dict[str, Any]] = []
    for failure_class, count in counts.most_common():
        if int(count) <= 0:
            continue
        action = _DETERMINISTIC_ACTIONS.get(
            failure_class,
            {
                "priority": "low",
                "component": "failure_audit",
                "action": "collect more examples and add a specific classifier before changing generation policy",
            },
        )
        recommendations.append(
            {
                "failure_class": failure_class,
                "count": int(count),
                "priority": action["priority"],
                "component": action["component"],
                "recommended_action": action["action"],
                "must_not_do": [
                    "do not relax MolNexTR pose/substitution/min-distance gates",
                    "do not copy failed retry artifacts into accepted data",
                    "do not split retry attempts across source-disjoint splits",
                ],
            }
        )
    if diagnostics.retry_selected_fraction >= 0.05:
        recommendations.insert(
            0,
            {
                "failure_class": "retry_selected_fraction_high",
                "count": int(diagnostics.retry_selected_rows),
                "priority": "high",
                "component": "deterministic_layout_policy",
                "recommended_action": (
                    "use the retry-saved failure mix to add pre-render source risk scoring or deterministic "
                    "layout seed selection before increasing retry count"
                ),
                "must_not_do": [
                    "do not treat retry as acceptance-rate tuning",
                    "do not raise renderer_seed_retries without a failure-class audit",
                ],
            }
        )
    return recommendations


_DETERMINISTIC_ACTIONS: dict[str, dict[str, str]] = {
    "formal_nonlinear_pose_preservation": {
        "priority": "high",
        "component": "nonlinear_warp_policy",
        "action": (
            "predict warp-risk from SVG bond geometry density and bbox margins, then select a documented "
            "formal amplitude before rendering the final row"
        ),
    },
    "svg_line_constraint_pose": {
        "priority": "high",
        "component": "svg_bond_axis_parser",
        "action": (
            "audit multiline bond axis selection, endpoint closure, and clipped selected-axis polylines for "
            "the failing source class"
        ),
    },
    "svg_intersection_anchor_pose": {
        "priority": "medium",
        "component": "source_geometry_filter",
        "action": (
            "detect too-few-intersection or rank-deficient structures before full rendering and top up from "
            "later source candidates"
        ),
    },
    "affine_seed_or_scale": {
        "priority": "high",
        "component": "pre_render_layout_selection",
        "action": (
            "classify extreme aspect-ratio/orientation sources and select a deterministic renderer layout seed "
            "that keeps SVG/image scale within the formal ratio gate"
        ),
    },
    "atom_pair_too_close": {
        "priority": "high",
        "component": "source_capacity_filter",
        "action": (
            "preflight dense fused-ring and high-variable-count sources for normalized atom spacing risk, then "
            "top up the same bucket from later candidates instead of lowering min distance"
        ),
    },
    "accepted_candidate_or_substitution_anchor": {
        "priority": "high",
        "component": "substitution_anchor_parser",
        "action": (
            "audit dummy/atomLabel OCR bbox closure, fixed-abbreviation classification, and atom-index alignment "
            "for the repeated source pattern"
        ),
    },
    "chemistry_or_depictor": {
        "priority": "medium",
        "component": "source_chemistry_normalizer",
        "action": (
            "separate invalid chemistry from depictor limitations before candidate planning; only normalized rows "
            "that preserve dummy/atomLabel semantics may re-enter"
        ),
    },
    "visual_readability": {
        "priority": "medium",
        "component": "document_realism_policy",
        "action": (
            "tighten machine readability and background overlap preflight before final accepted-candidate gates"
        ),
    },
}


def analyze_retry_diagnostics(csv_path: Path | None = None, manifest_path: Path | None = None) -> RetryDiagnostics:
    diagnostics = RetryDiagnostics(csv_path=csv_path, manifest_path=manifest_path)
    if csv_path is not None and csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                diagnostics.accepted_rows += 1
                render_quality = _json_object(row.get("render_quality"))
                retry = render_quality.get("renderer_seed_retry")
                if isinstance(retry, dict):
                    try:
                        diagnostics.selected_attempt_counts[int(retry.get("selected_attempt_index") or 0)] += 1
                    except (TypeError, ValueError):
                        diagnostics.policy_issues.append("renderer_seed_retry.selected_attempt_index is not an integer")
                    for failure in retry.get("previous_attempt_failures") or []:
                        if isinstance(failure, dict):
                            diagnostics.retry_saved_failure_class_counts[classify_generation_error(str(failure.get("error") or ""))] += 1
                else:
                    diagnostics.policy_issues.append("accepted row missing renderer_seed_retry contract")
    if manifest_path is not None and manifest_path.exists():
        manifest = _json_object(manifest_path.read_text(encoding="utf-8"))
        generation = manifest.get("generation") if isinstance(manifest.get("generation"), dict) else {}
        retry = generation.get("renderer_seed_retry") if isinstance(generation.get("renderer_seed_retry"), dict) else {}
        diagnostics.final_failure_count = int(manifest.get("failure_count") or generation.get("failure_count") or 0)
        diagnostics.intermediate_retry_failure_count = int(retry.get("intermediate_retry_failure_count") or 0)
        if retry and retry.get("thresholds_unchanged") is not True:
            diagnostics.policy_issues.append("manifest renderer retry does not prove thresholds_unchanged=true")
        for failure in manifest.get("failures") or []:
            if isinstance(failure, dict):
                diagnostics.failure_class_counts[classify_generation_error(str(failure.get("error") or ""))] += 1
                for previous in failure.get("previous_renderer_attempt_failures") or []:
                    if isinstance(previous, dict):
                        diagnostics.failure_class_counts[classify_generation_error(str(previous.get("error") or ""))] += 1
    return diagnostics


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
