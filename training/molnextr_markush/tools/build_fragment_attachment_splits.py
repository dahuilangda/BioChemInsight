from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.tools.build_formal_sidecar_splits import (
    collect_source_group_index,
    file_sha256,
    split_source_group_index,
    write_streaming_split_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build source-disjoint train/calibration splits for the fragment attachment-edit expert."
    )
    parser.add_argument("--fragment-csv", required=True)
    parser.add_argument("--ordinary-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--ordinary-calibration-fraction", type=float, default=0.2)
    parser.add_argument("--min-calibration-per-bucket", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026070601)
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    output_dir = Path(args.output_dir)
    paths = {
        "fragment_train_csv": output_dir / "fragment_train.csv",
        "fragment_calibration_csv": output_dir / "fragment_calibration.csv",
        "ordinary_train_csv": output_dir / "ordinary_train.csv",
        "ordinary_calibration_csv": output_dir / "ordinary_calibration.csv",
    }

    fragment_fields, fragment_groups, _fragment_raw_rows, fragment_row_count = collect_source_group_index(
        args.fragment_csv,
        kind="fragment",
    )
    ordinary_fields, ordinary_groups, _ordinary_raw_rows, ordinary_row_count = collect_source_group_index(
        args.ordinary_csv,
        kind="ordinary",
    )
    fragment_train_keys, fragment_calib_keys, fragment_report = split_source_group_index(
        fragment_groups,
        kind="fragment",
        trainable_rows=fragment_row_count,
        calibration_fraction=float(args.calibration_fraction),
        min_calibration_per_bucket=int(args.min_calibration_per_bucket),
        rng=rng,
    )
    ordinary_train_keys, ordinary_calib_keys, ordinary_report = split_source_group_index(
        ordinary_groups,
        kind="ordinary",
        trainable_rows=ordinary_row_count,
        calibration_fraction=float(args.ordinary_calibration_fraction),
        min_calibration_per_bucket=int(args.min_calibration_per_bucket),
        rng=rng,
    )

    fragment_write_report = write_streaming_split_rows(
        args.fragment_csv,
        kind="fragment",
        fieldnames=fragment_fields,
        train_keys=fragment_train_keys,
        calibration_keys=fragment_calib_keys,
        train_csv=paths["fragment_train_csv"],
        calibration_csv=paths["fragment_calibration_csv"],
    )
    ordinary_write_report = write_streaming_split_rows(
        args.ordinary_csv,
        kind="ordinary",
        fieldnames=ordinary_fields,
        train_keys=ordinary_train_keys,
        calibration_keys=ordinary_calib_keys,
        train_csv=paths["ordinary_train_csv"],
        calibration_csv=paths["ordinary_calibration_csv"],
    )

    blockers: list[str] = []
    if fragment_report.get("source_group_overlap"):
        blockers.append("fragment source group overlap between train and calibration")
    if ordinary_report.get("source_group_overlap"):
        blockers.append("ordinary source group overlap between train and calibration")
    for name, path in paths.items():
        if not path.exists():
            blockers.append(f"missing split output: {name}")

    report = {
        "schema_version": "fragment_attachment_source_disjoint_split_manifest_v1",
        "seed": int(args.seed),
        "accepted_for_training": not blockers,
        "split_policy": "source_group_disjoint_stratified_by_fragment_side_anchor_mode_and_ordinary_source",
        "purpose": {
            "fragment_attachment_expert": "train on fragment_train + ordinary_train; calibrate confidence on fragment_calibration + ordinary_calibration",
            "complete_molecule_path": "ordinary rows are negatives only; complete molecules continue through original MolNexTR",
            "markush_layout": "not included; Markush layout is a separate expert/task",
        },
        "inputs": {
            "fragment_csv": str(args.fragment_csv),
            "ordinary_csv": str(args.ordinary_csv),
        },
        "outputs": {name: str(path) for name, path in paths.items()},
        "output_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "fragment": fragment_report | fragment_write_report,
        "ordinary": ordinary_report | ordinary_write_report,
        "passed": not blockers,
        "blockers": blockers,
    }
    report_path = output_dir / "formal_sidecar_split_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
