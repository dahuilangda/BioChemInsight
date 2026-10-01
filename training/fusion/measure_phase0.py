#!/usr/bin/env python
"""A/B measure of the two Phase-0 efficiency switches on real patent pages.

1. Tile rescan: legacy (always) vs conditional — wall-clock per page on the
   detection stage + detection-set equality (gate: no structures lost).
2. Box-gate short circuit: replay over the segments produced in (1) —
   skip-rate of the VLM call (score >= tau AND no border ink) and the
   projected seconds saved using the production audit's per-call latency
   (mean 4.41 s, 3.35 calls/structure, 0.54 structures/page).
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np


def load_pages(count=48):
    import pyarrow.parquet as pq

    rows = []
    paths = sorted((ROOT / "training/molnextr_markush/data/raw/markushgrapher2/m2s").glob("*.parquet"))
    pf = pq.ParquetFile(paths[0])
    batch = next(pf.iter_batches(columns=["page_image"], batch_size=count))
    for payload in batch.column("page_image").to_pylist()[:count]:
        raw = payload["bytes"] if isinstance(payload, dict) else payload
        import cv2

        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        rows.append(image)
    return rows


def main() -> None:
    import cv2

    pages = load_pages(24)
    from utils import molecule_segmentation as seg

    recognizer_holder = {}

    def detect_all(mode):
        seg.DET_TILE_MODE = mode
        seg._TILE_STATS.update({"skipped_pages": 0, "ran_pages": 0})
        results = []
        started = time.time()
        for page in pages:
            masks = seg.get_expanded_masks(page)
            segments, bboxes = seg.apply_masks(page, masks)
            results.append((len(segments), [tuple(int(v) for v in b) for b in bboxes], masks))
        elapsed = time.time() - started
        return results, elapsed

    legacy_results, legacy_time = detect_all("legacy")
    cond_results, cond_time = detect_all("conditional")
    stats = dict(seg._TILE_STATS)

    legacy_total = sum(r[0] for r in legacy_results)
    cond_total = sum(r[0] for r in cond_results)
    same_pages = sum(1 for a, b in zip(legacy_results, cond_results) if a[0] == b[0] and a[1] == b[1])
    lost = [(i, a[0], b[0]) for i, (a, b) in enumerate(zip(legacy_results, cond_results)) if b[0] < a[0]]
    gained = [(i, a[0], b[0]) for i, (a, b) in enumerate(zip(legacy_results, cond_results)) if b[0] > a[0]]

    report = {
        "pages": len(pages),
        "legacy_seconds": round(legacy_time, 2),
        "conditional_seconds": round(cond_time, 2),
        "seconds_per_page_legacy": round(legacy_time / len(pages), 3),
        "seconds_per_page_conditional": round(cond_time / len(pages), 3),
        "detections_legacy": legacy_total,
        "detections_conditional": cond_total,
        "pages_with_identical_detection": same_pages,
        "pages_lost_detections": lost,
        "pages_gained_detections": gained,
        "tile_stats_conditional": stats,
        "gate_pass": cond_total >= legacy_total,
    }

    # --- box-gate short-circuit replay (no VLM calls; pure policy check) ---
    threshold = 0.12
    margin = 4
    calls = skipped = no_score = border_ink = 0
    saved_seconds = 0.0
    MEAN_VLM_SECONDS = 4.41  # production audit: 78,127 calls, mean 4.41s
    for page_idx, (count, bboxes, masks) in enumerate(cond_results):
        page = pages[page_idx]
        for idx in range(count):
            calls += 1
            y0, x0, y1, x1 = bboxes[idx]
            m = masks[:, :, idx]
            area = max(1, (y1 - y0) * (x1 - x0))
            score = float(m[y0:y1, x0:x1].sum()) / area
            crop = page[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            band = max(1, margin)
            band_pixels = max(1, 2 * band * (gray.shape[0] + gray.shape[1]))
            dark = int((gray[:band, :] < 235).sum() + (gray[-band:, :] < 235).sum()
                       + (gray[:, :band] < 235).sum() + (gray[:, -band:] < 235).sum())
            if score < threshold:
                no_score += 1
                continue
            if dark / band_pixels > 0.03:
                border_ink += 1
                continue
            skipped += 1
            saved_seconds += MEAN_VLM_SECONDS

    report["box_gate"] = {
        "segments": calls,
        "short_circuited": skipped,
        "skip_rate": round(skipped / max(1, calls), 4),
        "rejected_low_score": no_score,
        "rejected_border_ink": border_ink,
        "projected_saved_seconds_total": round(saved_seconds, 1),
        "projected_saved_seconds_per_page": round(saved_seconds / len(pages), 2),
    }

    out = ROOT / "experiments/fusion/phase0_ab.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
