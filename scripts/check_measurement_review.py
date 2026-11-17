"""Evaluate measurement transcription and review against explicit references."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIELDS = (
    "compound_id",
    "id_kind",
    "endpoint",
    "target",
    "conditions",
    "raw",
    "value",
    "operator",
    "unit",
    "source_kind",
)


def observation_key(item, case=None):
    reference = case or {}
    values = {
        **item,
        "conditions": text_identity(
            item["conditions"], reference.get("condition_aliases", {})
        ),
        "raw": text_identity(item["raw"], reference.get("raw_aliases", {})),
    }
    return tuple(values[field] for field in FIELDS)


def text_identity(value, aliases):
    for canonical, alternatives in aliases.items():
        if value == canonical or value in alternatives:
            return canonical
    return value


def _validate_text_aliases(aliases):
    if not isinstance(aliases, dict):
        raise TypeError("Reference aliases must be an object")
    assigned = {}
    for canonical, alternatives in aliases.items():
        if (
            not isinstance(canonical, str)
            or not canonical.strip()
            or not isinstance(alternatives, list)
        ):
            raise ValueError(
                "Reference aliases require a non-empty canonical string and a list"
            )
        for value in [canonical, *alternatives]:
            if (
                not isinstance(value, str)
                or not value.strip()
                or (value in assigned and assigned[value] != canonical)
            ):
                raise ValueError("Reference aliases must be non-empty and unambiguous")
            assigned[value] = canonical


def ambiguity_matches(details, case):
    if not isinstance(details, dict):
        return False
    review = (details or {}).get("review", {})
    if not isinstance(review, dict):
        return False
    issues = review.get("issues", [])
    if (
        not case.get("expect_evidence_error", False)
        or review.get("status") != "unresolved"
        or not isinstance(issues, list)
        or not issues
        or not all(isinstance(issue, dict) for issue in issues)
    ):
        return False
    expected_kinds = case.get("expected_issue_kinds", [])
    if expected_kinds:
        issues = [issue for issue in issues if issue.get("kind") in expected_kinds]
    issues = [issue for issue in issues if issue.get("resolution") == "source_review"]
    subjects = set()
    for issue in issues:
        if not isinstance(issue.get("subjects"), list):
            return False
        for subject in issue["subjects"]:
            if (
                not isinstance(subject, dict)
                or not isinstance(subject.get("id_kind"), str)
                or not isinstance(subject.get("compound_id"), str)
            ):
                return False
            subjects.add((subject["id_kind"], subject["compound_id"]))
    expected = {
        (subject["id_kind"], subject["compound_id"])
        for subject in case.get("expected_subjects", [])
    }
    return bool(issues) and expected <= subjects


def score_result(result, case):
    actual = [
        {"compound_id": row["compound_id"], "id_kind": row["id_kind"], **item}
        for row in result["rows"]
        for item in row["assays"].values()
    ]
    expected = {observation_key(item, case) for item in case["expected"]}
    observed = {observation_key(item, case) for item in actual}
    return {
        "passed": not case.get("expect_evidence_error", False)
        and expected == observed
        and len(actual) == len(case["expected"]),
        "missing": [
            {field: item[field] for field in FIELDS}
            for item in case["expected"]
            if observation_key(item, case) not in observed
        ],
        "unexpected": [
            {field: item[field] for field in FIELDS}
            for item in actual
            if observation_key(item, case) not in expected
        ],
    }


def rescore_run(directory, cases, *, output=None):
    report = json.loads((directory / "report.json").read_text())
    references = {case["name"]: case for case in cases}
    if {item["name"] for item in report["results"]} != set(references):
        raise ValueError("Rescoring requires the same case names")
    repeats = report["repeats"]
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Invalid repetition count in the original report")
    expected_trials = {
        (name, trial) for name in references for trial in range(1, repeats + 1)
    }
    actual_trials = [(item["name"], item["trial"]) for item in report["results"]]
    if set(actual_trials) != expected_trials or len(actual_trials) != len(
        expected_trials
    ):
        raise ValueError("Cannot rescore an incomplete or duplicated trial set")
    results = copy.deepcopy(report["results"])
    for item in results:
        case = references[item["name"]]
        input_path = directory / f"trial_{item['trial']}" / item["name"] / "input.json"
        original = json.loads(input_path.read_text())
        if (
            original["source"] != case["source"]
            or original.get("initial_observations") != case.get("initial_observations")
            or original.get("initial_source_issues")
            != case.get("initial_source_issues")
        ):
            raise ValueError("Rescoring cannot change model inputs")
        if "result" in item:
            item.update(score_result(item["result"], case))
        elif item.get("error_type") == "source_ambiguity":
            item["passed"] = ambiguity_matches(item.get("details"), case)
        else:
            item["passed"] = False
        if output is not None:
            destination = output / f"trial_{item['trial']}" / item["name"]
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "input.json").write_bytes(input_path.read_bytes())
            (destination / "result.json").write_text(
                json.dumps(item, ensure_ascii=False, indent=2)
            )
    return results, repeats


def validate_cases(cases):
    from utils.measurement_contract import (
        ISSUE_KINDS,
        MeasurementRequest,
        measurement_identity,
    )

    if not isinstance(cases, list) or not cases:
        raise ValueError("The evaluation requires a non-empty case list")
    names = set()
    for case in cases:
        name = case["name"]
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in {".", ".."}
            or name in names
        ):
            raise ValueError("Case names must be unique directory names")
        names.add(name)
        aliases = case.get("condition_aliases", {})
        _validate_text_aliases(aliases)
        _validate_text_aliases(case.get("raw_aliases", {}))
        if not isinstance(case["source"], str) or not case["source"].strip():
            raise ValueError(f"Missing source for {name}")
        if not isinstance(case["expected"], list):
            raise TypeError(f"Expected observations must be a list for {name}")
        if type(case.get("expect_evidence_error", False)) is not bool:
            raise ValueError(f"Expected ambiguity must be boolean for {name}")
        if "expected_issue_kind" in case:
            raise ValueError("Use expected_issue_kinds for reference issue classes")
        expected_kinds = case.get("expected_issue_kinds", [])
        if not isinstance(expected_kinds, list) or any(
            kind not in ISSUE_KINDS for kind in expected_kinds
        ):
            raise ValueError("Unknown expected issue kinds")
        subjects = case.get("expected_subjects", [])
        if not isinstance(subjects, list):
            raise TypeError("expected_subjects must be an array")
        for subject in subjects:
            if not isinstance(subject, dict) or set(subject) != {
                "id_kind",
                "compound_id",
            }:
                raise ValueError("Invalid expected subject fields")
            if (
                subject["id_kind"] not in ("compound", "seq_id", "example", "unknown")
                or not isinstance(subject["compound_id"], str)
                or not subject["compound_id"].strip()
            ):
                raise ValueError("Invalid expected subject identity")
        if (subjects or case.get("expected_issue_kinds")) and not case.get(
            "expect_evidence_error", False
        ):
            raise ValueError("Expected ambiguity details require an ambiguity case")
        request = MeasurementRequest(case["source"])
        if "initial_source_issues" in case:
            proposals = case["initial_source_issues"]
            if (
                "initial_observations" in case
                or not isinstance(proposals, list)
                or not proposals
            ):
                raise ValueError(
                    "Source-issue injection requires non-empty proposals and no observation injection"
                )
            for proposal in proposals:
                request.validate_source_issue(proposal)
        for key in ("expected", "initial_observations"):
            items = case.get(key, [])
            if not isinstance(items, list):
                raise TypeError(f"{key} must be a list for {name}")
            for item in items:
                request.validate_observation(
                    {
                        **item,
                        "evidence": [
                            {
                                "line_start": 1,
                                "line_end": len(request.lines),
                            }
                        ],
                    }
                )
        expected = [
            measurement_identity(
                {**item, "conditions": text_identity(item["conditions"], aliases)}
            )
            for item in case["expected"]
        ]
        if len(set(expected)) != len(expected):
            raise ValueError(f"Duplicate reference measurement identity in {name}")
        if case.get("expect_evidence_error", False) and expected:
            raise ValueError(
                f"Unresolved cases cannot require accepted observations: {name}"
            )


def evaluate_case(case, output, timeout_seconds=240, call_timeout_seconds=90, trial=1):
    from utils.llm_utils import run_text_json_task
    from utils.measurement_review import review_measurements
    from utils.model_harness import ModelSourceAmbiguityError

    directory = output / case["name"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "input.json").write_text(
        json.dumps(case, ensure_ascii=False, indent=2)
    )
    started = time.monotonic()
    injected = False

    def runner(**kwargs):
        nonlocal injected
        if kwargs["metadata"]["stage"] == "extract" and "initial_source_issues" in case:
            injected = True
            proposals = copy.deepcopy(case["initial_source_issues"])
            return kwargs["parser"](
                json.dumps(
                    {
                        "status": "uncertain",
                        "reason": "; ".join(
                            issue["description"] for issue in proposals
                        ),
                        "observations": [],
                        "source_issues": proposals,
                    }
                )
            )
        if kwargs["metadata"]["stage"] == "extract" and "initial_observations" in case:
            injected = True
            observations = copy.deepcopy(case["initial_observations"])
            for item in observations:
                item["evidence"] = [
                    {
                        "line_start": 1,
                        "line_end": len(case["source"].splitlines()),
                    }
                ]
            return kwargs["parser"](
                json.dumps(
                    {
                        "status": "complete" if observations else "absent",
                        "reason": "",
                        "observations": observations,
                        "source_issues": [],
                    }
                )
            )
        return run_text_json_task(**kwargs)

    report = {"name": case["name"], "trial": trial, "passed": False}
    try:
        result = review_measurements(
            case["source"],
            audit_path=directory / "model_calls.jsonl",
            runner=runner,
            timeout_seconds=timeout_seconds,
            call_timeout_seconds=call_timeout_seconds,
        )
        report.update(
            **score_result(result, case),
            revisions=result["review"]["revisions"],
            result=result,
        )
    except ModelSourceAmbiguityError as exc:
        report.update(
            passed=ambiguity_matches(exc.details, case),
            error_type=exc.error_type,
            error=str(exc),
            details=exc.details,
        )
    except Exception as exc:  # noqa: BLE001
        report.update(
            error_type=getattr(exc, "error_type", type(exc).__name__),
            error=str(exc),
            details=getattr(exc, "details", None),
        )
    report.update(
        injected=injected, elapsed_seconds=round(time.monotonic() - started, 3)
    )
    (directory / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "evals/measurement_review_cases.json"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "output/measurement_review")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--call-timeout", type=float, default=90)
    parser.add_argument("--repeats", type=int, choices=range(1, 21), default=1)
    parser.add_argument("--rescore", type=Path)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    try:
        validate_cases(cases)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if any(
        not math.isfinite(value) or value <= 0
        for value in (args.timeout, args.call_timeout)
    ):
        parser.error("Timeouts must be positive finite numbers")
    run_id = uuid.uuid4().hex
    run_directory = args.out / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    print(json.dumps({"run_id": run_id, "directory": str(run_directory)}), flush=True)
    (run_directory / "references.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2)
    )
    if args.rescore is not None:
        try:
            results, args.repeats = rescore_run(
                args.rescore, cases, output=run_directory
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
    else:
        results = run_cases(cases, args, run_directory)
    return write_report(results, cases, args, run_directory, run_id)


def run_cases(cases, args, run_directory):
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                evaluate_case,
                case,
                run_directory / f"trial_{trial}",
                args.timeout,
                args.call_timeout,
                trial,
            )
            for trial in range(1, args.repeats + 1)
            for case in cases
        ]
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        key: result[key]
                        for key in ("name", "trial", "passed", "elapsed_seconds")
                    }
                ),
                flush=True,
            )
    return results


def write_report(results, cases, args, run_directory, run_id):
    results.sort(key=lambda item: (item["name"], item["trial"]))
    scored = [
        item
        for item in results
        if "result" in item or item.get("error_type") == "source_ambiguity"
    ]
    report = {
        "run_id": run_id,
        "source_run": str(args.rescore.resolve()) if args.rescore is not None else None,
        "unique_cases": len(cases),
        "repeats": args.repeats,
        "cases": len(results),
        "passed": sum(item["passed"] for item in results),
        "case_success_rate": sum(item["passed"] for item in results) / len(results),
        "scored_cases": len(scored),
        "execution_failures": len(results) - len(scored),
        "reference_agreement": sum(item["passed"] for item in scored) / len(scored)
        if scored
        else None,
        "results": results,
    }
    report["consistent_cases"] = sum(
        all(item["passed"] for item in results if item["name"] == case["name"])
        for case in cases
    )
    (run_directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    return 0 if report["passed"] == report["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
