"""Literature extraction evaluation against annotated values and structures."""

import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem

from utils.extraction_metrics import (
    combine_evaluation_statuses,
    score_assay_values,
    score_structure_core,
    score_structure_matches,
)
from utils.series_id_utils import parse_series_range_id
from utils.series_member_synthesis import name_to_molecule

ROOT = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH = json.load(
    open(os.path.join(ROOT, "literature_ground_truth.json"), encoding="utf-8")
)
ARTIFACTS = os.path.join(ROOT, "artifacts")


def page_contexts_from_pdf(pdf_path, pages, cache_path):
    import fitz

    cached = {}
    if os.path.exists(cache_path):
        cached = json.load(open(cache_path, encoding="utf-8"))
    contexts = []
    doc = fitz.open(pdf_path)
    for page in pages:
        key = str(page)
        if key not in cached:
            cached[key] = doc[page - 1].get_text()
            json.dump(
                cached, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False
            )
        contexts.append({"page": page, "markdown": cached[key]})
    return contexts


def judge_assay(extracted, expected_members):
    return score_assay_values(extracted, expected_members)


def _structure_member_ids(truth):
    series = parse_series_range_id(truth.get("series_id"))
    return series["members"] if series is not None else None


def _unannotated_structure_report(member_rows):
    return {
        "status": "inconclusive",
        "expected_members": None,
        "emitted": len(member_rows),
        "missing": None,
        "coverage": None,
        "full_structure_accuracy": None,
        "reference_errors": ["Expected structure member IDs are not annotated"],
    }


def judge_structure_benzimidazole(member_rows, truth):
    expected = _structure_member_ids(truth)
    if expected is None:
        return _unannotated_structure_report(member_rows)
    references = {}
    resolution_errors = []
    for member_id in expected:
        name = truth["members"].get(member_id, {}).get("name")
        if not isinstance(name, str) or not name.strip():
            references[member_id] = None
            resolution_errors.append(
                {"id": member_id, "reason": "Complete chemical name is not annotated"}
            )
            continue
        try:
            molecule, reason = name_to_molecule(name)
        except Exception as exc:  # noqa: BLE001
            molecule, reason = None, str(exc)
        references[member_id] = (
            Chem.MolToSmiles(molecule, isomericSmiles=True)
            if molecule is not None
            else None
        )
        if molecule is None:
            resolution_errors.append({"id": member_id, "reason": reason})
    result = score_structure_matches(member_rows, references, expected_ids=expected)
    result["reference_resolution_errors"] = resolution_errors
    return result


def judge_structure_quinoline(member_rows, truth):
    expected = _structure_member_ids(truth)
    if expected is None:
        return _unannotated_structure_report(member_rows)
    return score_structure_core(member_rows, expected, truth["core_smarts"])


def run_assay_eval(paper, pdf_path, cache_path, out_dir):
    from activity_parser import extract_activity_data_multi

    truth = GROUND_TRUTH[paper]
    result = extract_activity_data_multi(
        pdf_file=pdf_path,
        assay_page_start=min(truth["assay_pages"]),
        assay_page_end=max(truth["assay_pages"]),
        assay_names=[truth["assay_name"]],
        pages_per_chunk=2,
        # Production passes the structure-side series identifier; page-level
        # series expansion then supplies the member allowlist.
        compound_id_list=[truth["series_id"]],
        output_dir=out_dir,
        structure_records=None,
    )
    extracted = result.get(truth["assay_name"], {})
    return judge_assay(extracted, truth["members"])


def run_structure_eval(paper, pdf_path, cache_path, out_dir):
    from utils.series_member_synthesis import synthesize_series_members

    truth = GROUND_TRUTH[paper]
    contexts = page_contexts_from_pdf(pdf_path, truth["assay_pages"], cache_path)
    import fitz

    total_pages = fitz.open(pdf_path).page_count
    pages_all = page_contexts_from_pdf(
        pdf_path, list(range(1, total_pages + 1)), cache_path
    )
    member_rows, report = synthesize_series_members(
        [], pages_all, audit_path=os.path.join(out_dir, "model_calls.jsonl")
    )
    rows = [
        r
        for r in member_rows
        if str(r.get("SERIES_COMPOUND_ID") or "") == truth["series_id"]
    ]
    if paper == "benzimidazole":
        verdict = judge_structure_benzimidazole(rows, truth)
    else:
        verdict = judge_structure_quinoline(rows, truth)
    verdict["report"] = [
        {
            "series_id": s.get("series_id"),
            "defined": s.get("defined_members"),
            "emitted": s.get("emitted"),
            "quarantined": s.get("quarantined", 0),
            "consistency": s.get("consistency"),
        }
        for s in report.get("series", [])
    ]
    return verdict


def main():
    rolls = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    papers = (
        sys.argv[2].split(",") if len(sys.argv) > 2 else ["benzimidazole", "quinoline"]
    )
    if rolls < 1 or not papers:
        raise ValueError("At least one paper and one structure trial are required")
    os.makedirs(ARTIFACTS, exist_ok=True)
    pdfs = {
        "benzimidazole": os.path.join(
            ROOT, "..", "data", "literature_test", "scirep_benzimidazole.pdf"
        ),
        "quinoline": os.path.join(
            ROOT, "..", "data", "literature_test", "scirep_quinoline.pdf"
        ),
    }
    overall = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "rolls": rolls, "papers": {}}
    for paper in papers:
        paper_result = {"assay": None, "structure_rolls": []}
        out_dir = os.path.join(ARTIFACTS, f"{paper}_{int(time.time())}")
        os.makedirs(out_dir, exist_ok=True)
        cache_path = os.path.join(ARTIFACTS, f"{paper}_pages.json")
        try:
            paper_result["assay"] = run_assay_eval(
                paper, pdfs[paper], cache_path, out_dir
            )
        except Exception:
            paper_result["assay"] = {
                "status": "failed",
                "error": traceback.format_exc()[-800:],
            }
        for roll in range(1, rolls + 1):
            try:
                verdict = run_structure_eval(paper, pdfs[paper], cache_path, out_dir)
                verdict["roll_pass"] = verdict["status"] == "passed"
                paper_result["structure_rolls"].append(verdict)
                print(
                    f"[{paper}] roll {roll}: status={verdict['status']} coverage={verdict.get('coverage')}"
                )
            except Exception:
                paper_result["structure_rolls"].append(
                    {
                        "status": "failed",
                        "error": traceback.format_exc()[-800:],
                        "roll_pass": False,
                    }
                )
        if paper_result["assay"] and "wrong" in paper_result["assay"]:
            a = paper_result["assay"]
            print(
                f"[{paper}] assay: {a['correct']}/{a['expected']} correct, "
                f"missing={len(a['missing'])}, wrong={len(a['wrong'])}, unexpected={a['unexpected_keys']}"
            )
        paper_result["status"] = combine_evaluation_statuses(
            [
                paper_result["assay"]["status"],
                *(row["status"] for row in paper_result["structure_rolls"]),
            ]
        )
        paper_result["pass"] = paper_result["status"] == "passed"
        overall["papers"][paper] = paper_result
    overall["status"] = combine_evaluation_statuses(
        row["status"] for row in overall["papers"].values()
    )
    overall["pass"] = overall["status"] == "passed"
    with open(os.path.join(ARTIFACTS, "runs.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(overall, ensure_ascii=False) + "\n")
    print(overall["status"].upper())
    return 0 if overall["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
