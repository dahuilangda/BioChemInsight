from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.tools.check_split_pose_preservation_contract import (
    compact_pose_report,
    inspect_csv,
    load_json,
)
from training.molnextr_markush.src.pose_factory import validate_pose_factory_shard


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate that Markush-only train/calibration splits preserve images, pose, graph, OCR labels, and provenance."
    )
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-issues", type=int, default=200)
    args = parser.parse_args()

    manifest_path = Path(args.split_manifest)
    manifest = load_json(manifest_path)
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    blockers: list[str] = []
    reports: dict[str, Any] = {}

    if manifest.get("schema_version") != "markush_only_split_manifest_v1":
        blockers.append("split manifest schema_version is not markush_only_split_manifest_v1")
    if manifest.get("accepted_for_training") is not True:
        blockers.append("Markush-only split manifest accepted_for_training is not true")
    policy = manifest.get("policy") if isinstance(manifest.get("policy"), dict) else {}
    for key in [
        "source_document_disjoint_train_calibration",
        "markush_rows_are_routed_expert_branch_only",
        "complete_molecules_remain_on_original_molnextr_path",
        "does_not_mix_legacy_fragment_or_ordinary_generated_data",
        "positive_only_markush_layout_training",
        "router_negative_evidence_not_claimed",
    ]:
        if policy.get(key) is not True:
            blockers.append(f"split manifest policy.{key} is not true")

    for key in ["markush_train_csv", "markush_calibration_csv"]:
        path_text = str(outputs.get(key) or "")
        if not path_text:
            blockers.append(f"split manifest missing outputs.{key}")
            continue
        path = Path(path_text)
        if not path.exists():
            blockers.append(f"split output does not exist for {key}: {path}")
            continue
        pose_report = validate_pose_factory_shard(path, check_images=True, max_issues=int(args.max_issues))
        preservation_report = inspect_csv(path, role="markush", max_examples=int(args.max_issues))
        reports[key] = {
            "pose_factory_validation": compact_pose_report(pose_report),
            "preservation": preservation_report,
        }
        if pose_report.trainable is not True:
            blockers.append(f"{key} failed pose-factory validation")
        if preservation_report.get("passed") is not True:
            blockers.append(f"{key} failed Markush preservation checks")

    report = {
        "schema_version": "markush_only_split_preservation_v1",
        "split_manifest": str(manifest_path),
        "split_stage": manifest.get("stage"),
        "split_accepted_for_training": manifest.get("accepted_for_training") is True,
        "image_checks_enabled": True,
        "passed": not blockers,
        "blockers": blockers,
        "reports": reports,
        "policy": {
            "split_outputs_must_preserve_images": True,
            "split_outputs_must_preserve_graph_and_atom_coordinates": True,
            "markush_rows_must_preserve_ocr_cells_r_tag_labels_annotations_and_render_provenance": True,
            "split_rows_must_preserve_image_to_graph_orientation_alignment": True,
            "does_not_require_fragment_or_ordinary_rows": True,
            "does_not_claim_router_negative_evidence": True,
            "complete_molecules_remain_on_original_molnextr_path": True,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
