"""Build OURS-v3 SFT dataset (§44.67): OCR-in-prompt.

The §44.66 failure decomposition showed 41.3% of identity failures are label
VALUES read wrong and 26.8% are missed dummy positions — while wrong ORDER is
only 2.0%. The model was never shown the label strings; it had to infer
numbering from nothing. v3 gives the model the OCR-read label texts in the
prompt ("The following text labels appear in the image: ..."), reducing the
task to assigning already-visible text to positions. Oracle-label upper bound
measured at 50.7% identity.

Sources:
  - base rows: /data/glyph_data/ours_v2/train.jsonl (v2 messages format)
  - OCR cells: shard + real-train + real-eval precompute parquets
    ({file_path, ocr_cells: JSON [{bbox, text}]})

Prompt format (markush rows with cells):
  "<image>\\nExtract the chemical structure from this image.\\n"
  "Text labels visible in the image: R4, Z, Y, SCH3, X\\n"
  "Target format: return a SMILES string."

Label filtering (noise control — ChemicalOCR also reads atom symbols and
figure annotations): keep texts of 1-6 chars containing at least one
alphanumeric character; drop pure numbers >4 digits (page/figure numbers);
cap at 24 labels sorted top-to-bottom then left-to-right.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

BASE = Path("/data/glyph_data/ours_v2/train.jsonl")
CELLS_GLOBS = [
    "/tmp/ocr_cells_precompute.shard*.parquet",
    "/tmp/ocr_cells_precompute_t2.parquet",
    "/tmp/ocr_cells_precompute_t3.parquet",
    "/tmp/ocr_cells_real_train.parquet",
]
OUT = Path("/data/glyph_data/ours_v3/train.jsonl")
MAX_LABELS = 24
LABEL_DROPOUT = 0.10  # fraction of markush rows that get NO labels (fallback robustness)


def label_like(text: str) -> bool:
    t = str(text or "").strip()
    if not (1 <= len(t) <= 6):
        return False
    if not re.search(r"[A-Za-z0-9]", t):
        return False
    if re.fullmatch(r"\d{5,}", t):
        return False
    return True


def load_cells_map() -> dict[str, list[dict]]:
    import pandas as pd
    import glob
    frames = []
    for pattern in CELLS_GLOBS:
        for path in sorted(glob.glob(pattern)):
            try:
                frames.append(pd.read_parquet(path))
            except Exception:
                pass
    if not frames:
        raise SystemExit("no cells parquets found")
    cells = pd.concat(frames, ignore_index=True)
    cells = cells.drop_duplicates(subset=["file_path"], keep="last")
    out = {}
    for _, row in cells.iterrows():
        raw = str(row.get("ocr_cells") or "")
        try:
            parsed = json.loads(raw) if raw else []
        except (ValueError, TypeError):
            parsed = []
        out[str(row["file_path"])] = parsed
    return out


def labels_for(image_path: str, cells_map: dict[str, list[dict]]) -> list[str]:
    cells = cells_map.get(str(image_path)) or []
    labelled = [
        (float(c["bbox"][1]), float(c["bbox"][0]), str(c["text"]).strip())
        for c in cells
        if label_like(c.get("text"))
    ]
    labelled.sort(key=lambda t: (t[0], t[1]))  # top-to-bottom, left-to-right
    seen, out = set(), []
    for _y, _x, text in labelled:
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out[:MAX_LABELS]


def main() -> None:
    cells_map = load_cells_map()
    print(f"cells map: {len(cells_map)} images", flush=True)

    import random
    rng = random.Random(20261022)

    train_in = open(BASE)
    out = open(OUT, "w")
    stats = {"markush_with_labels": 0, "markush_no_labels": 0,
             "complete": 0, "fragment": 0}
    for line in train_in:
        r = json.loads(line)
        content = r["messages"][0]["content"]
        target = r["messages"][1]["content"]
        image = r["images"][0]
        is_starred = "*" in target
        if not is_starred:
            stats["complete" if "*" not in target else "fragment"] += 1
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            continue
        if r.get("regime") == "fragment":
            stats["fragment"] += 1
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            continue
        labels = labels_for(image, cells_map)
        if labels and rng.random() >= LABEL_DROPOUT:
            stats["markush_with_labels"] += 1
            user = (
                "<image>\nExtract the chemical structure from this image.\n"
                "Text labels visible in the image: " + ", ".join(labels) + "\n"
                "Target format: return a SMILES string."
            )
        else:
            stats["markush_no_labels"] += 1
            user = (
                "<image>\nExtract the chemical structure from this image.\n"
                "Target format: return a SMILES string."
            )
        rec = {
            "id": r.get("id", ""),
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": target},
            ],
            "images": [image],
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.close()
    print(f"stats: {stats} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
