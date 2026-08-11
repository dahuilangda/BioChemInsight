#!/usr/bin/env python
"""Measure attachment detector agreement with gold dummy positions and emit pseudo-label parquet.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from utils.MolNexTR.attachment_detector import AttachmentDetector  # noqa: E402

_DUMMY_TOKEN_RE = re.compile(r"\[?\d*\*\]?")


def gold_dummy_count(smiles: str) -> int:
    """Count attachment dummies in a gold SMILES ([1*], [2*], bare *)."""
    if not smiles:
        return 0
    count = 0
    for match in re.finditer(r"\[[^\]]*\*[^\]]*\]|\*", smiles):
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--df-cache", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--out-dir", default="experiments/moe/detector_pseudo_labels")
    parser.add_argument("--max-rows", type=int, default=400)
    parser.add_argument("--fragment-fraction", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--extra-fragment-csv",
        action="append",
        default=[],
        help="CSV/parquet with file_path+SMILES of fragment rows (repeatable)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.df_cache, columns=[
        "file_path", "SMILES", "structure_type_label", "image_domain",
        "node_coords", "source_dummy_atom_index",
    ])
    rows = []
    for label_id in (1, 2):
        sub = df[df["structure_type_label"] == label_id]
        cap = (
            int(args.max_rows * (1.0 - args.fragment_fraction))
            if label_id == 1
            else int(args.max_rows * args.fragment_fraction)
        )
        real = sub[sub["image_domain"] == "real_original"]
        synth = sub[sub["image_domain"] != "real_original"]
        candidates = []
        for part in (real, synth):
            for p, s in zip(part["file_path"].tolist(), part["SMILES"].tolist()):
                if os.path.isfile(p):
                    candidates.append((p, s, int(label_id)))
        rng = random.Random(args.seed)
        rng.shuffle(candidates)
        rows.extend(candidates[:cap])
    # Fragment rows are absent from the df-cache on this machine; add real
    # fragment sources with images (contract-verified parquet or eval CSVs).
    seen = {r[0] for r in rows}
    for csv_path in args.extra_fragment_csv:
        if not os.path.exists(csv_path):
            print(f"warning: extra fragment csv not found: {csv_path}")
            continue
        extra = (
            pd.read_parquet(csv_path)
            if str(csv_path).endswith(".parquet")
            else pd.read_csv(csv_path)
        )
        if not {"file_path", "SMILES"}.issubset(extra.columns):
            print(f"warning: {csv_path} lacks file_path/SMILES columns")
            continue
        for p, s in zip(extra["file_path"].tolist(), extra["SMILES"].tolist()):
            if os.path.isfile(p) and p not in seen:
                seen.add(p)
                rows.append((p, s, 2))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.max_rows]
    print(f"sampled {len(rows)} rows "
          f"({sum(1 for r in rows if r[2] == 1)} markush, "
          f"{sum(1 for r in rows if r[2] == 2)} fragment)")

    detector = AttachmentDetector(
        checkpoint_path=args.detector_checkpoint,
        device=None,
        num_classes=5,  # bg + wavy + rgroup + asterisk + dashed
    )
    detector._ensure_loaded()

    import cv2
    results = []
    count_matches = 0
    frag_localization_distances = []
    mark_classes: dict[str, int] = {}
    for file_path, smiles, label_id in rows:
        img = cv2.imread(file_path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            dets = detector.detect(rgb, score_threshold=args.score_threshold)
        except Exception:
            dets = []
        gold_count = gold_dummy_count(smiles)
        marks = [
            {"class": d.class_name, "cx": float(d.cx), "cy": float(d.cy),
             "confidence": float(d.confidence)}
            for d in dets
        ]
        for d in marks:
            mark_classes[d["class"]] = mark_classes.get(d["class"], 0) + 1
        count_match = len(marks) == gold_count
        count_matches += int(count_match)
        min_dist = None
        if label_id == 2 and gold_count == 1 and marks and "node_coords" in df.columns:
            # Fragment rows: source_dummy_atom_index locates the gold dummy in
            # normalized image coords (node_coords_space=normalized_image).
            row_meta = df[(df["file_path"] == file_path) & (df["SMILES"] == smiles)]
            if len(row_meta):
                meta = row_meta.iloc[0]
                idx = meta.get("source_dummy_atom_index")
                coords = meta.get("node_coords")
                if isinstance(idx, (int, float)) and not pd.isna(idx) and isinstance(coords, list):
                    i = int(idx)
                    if 0 <= i < len(coords):
                        gx, gy = float(coords[i][0]), float(coords[i][1])
                        min_dist = min(
                            (d["cx"] - gx) ** 2 + (d["cy"] - gy) ** 2
                            for d in marks
                        ) ** 0.5
                        frag_localization_distances.append(min_dist)
        results.append({
            "file_path": file_path,
            "SMILES": smiles,
            "structure_type_label": label_id,
            "gold_dummy_count": gold_count,
            "detector_mark_count": len(marks),
            "count_match": count_match,
            "min_normalized_distance": min_dist,
            "detections": marks,
        })

    summary = {
        "n": len(results),
        "count_match_rate": count_matches / max(1, len(results)),
        "fragment_localization": {
            "n": len(frag_localization_distances),
            "median_normalized_distance": float(np.median(frag_localization_distances)) if frag_localization_distances else None,
            "within_0_05_rate": (
                float(np.mean([d <= 0.05 for d in frag_localization_distances]))
                if frag_localization_distances else None
            ),
        },
        "mark_classes": dict(sorted(mark_classes.items())),
        "score_threshold": args.score_threshold,
    }
    out_df = pd.DataFrame(results)
    out_df.to_parquet(out_dir / "pseudo_labels.parquet", index=False)
    with open(out_dir / "report.json", "w", encoding="utf-8") as handle:
        json.dump({"schema_version": "detector_pseudo_label_agreement_v1",
                   **summary}, handle, ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'pseudo_labels.parquet'} and {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
