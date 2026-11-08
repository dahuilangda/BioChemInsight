"""Build OUR OWN three-regime SFT dataset (§44.63) for the glyph trainer.

Unified output contract: image → isotope-numbered SMILES, for ALL regimes
(complete, markush, fragment) — matches the deployed identity metric and the
downstream numbered-molblock consumers. Regime is recorded as `task`
("ocsr" for complete/fragment, "ocsr_markush" tagging for markush rows — the
trainer's ocsr branch takes the canonical SMILES either way; the tag rides
along for data accounting).

Sources: uspto_label_ocr_full_df3.parquet (405,605 rows: 301,711 complete /
103,017 markush incl. 54.7k real-cell-backed shard rows / 925 fragment).

Sampling (v1):
  - all markush rows (the capacity-critical regime)
  - 100k complete rows (subsample)
  - all fragment rows x5 oversample
  - 1% held-out validation split
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

DF = "experiments/moe/uspto_label_ocr_full_df3.parquet"
OUT_DIR = Path("/data/glyph_data/ours_v2")
MAX_TARGET_TOKENS = 250
MAX_TARGET_TOKENS_MARKUSH = 400
COMPLETE_SAMPLE = 100_000
FRAGMENT_OVERSAMPLE = 5


def canonical_target(smiles: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(str(smiles or ""))
        if mol is None:
            return None
        out = Chem.MolToSmiles(mol)
        if len(out) > MAX_TARGET_TOKENS:
            return None
        return out
    except Exception:
        return None


def main() -> None:
    df = pd.read_parquet(DF)
    print(f"source rows: {len(df)}", flush=True)

    rng = random.Random(20261011)
    records: list[dict] = []
    stats = {"complete": 0, "markush": 0, "fragment": 0, "skipped": 0}

    star_mask = df["SMILES"].fillna("").astype(str).str.contains(r"\*", regex=True)
    frag_mask = df["structure_type_label"].fillna(-1).astype(int) == 2

    # \u00a744.64: per-regime target caps \u2014 markush rows keep large
    # structures (the eval subset is dummy-heavy: 34% have >=6 dummies vs
    # 15% in v1's corpus; v1's 250-char cap silently dropped most of them).
    def target_ok(smiles: str, regime: str) -> bool:
        limit = (MAX_TARGET_TOKENS if regime == "complete"
                 else MAX_TARGET_TOKENS_MARKUSH)
        return len(smiles) <= limit

    def dummy_count(smiles: str) -> int:
        return smiles.count("*")

    complete_pool = df.index[~star_mask & ~frag_mask].tolist()
    rng.shuffle(complete_pool)
    complete_pool = set(complete_pool[:COMPLETE_SAMPLE])

    def emit(idx, row, regime, repeat=1):
        smiles = str(row.get("SMILES") or "")
        image = str(row.get("file_path") or "")
        if not target_ok(smiles, regime):
            stats["skipped"] += 1
            return
        target = canonical_target(smiles)
        if target is None or not image:
            stats["skipped"] += 1
            return
        for _ in range(repeat):
            records.append({
                "id": f"ours-{regime}-{idx}",
                "messages": [
                    {"role": "user",
                     "content": "<image>\nExtract the chemical structure from this image.\n"
                                "Target format: return a SMILES string."},
                    {"role": "assistant", "content": target},
                ],
                "images": [image],
            })
        stats[regime] += 1

    for idx, row in df.iterrows():
        smiles = str(row.get("SMILES") or "")
        if frag_mask.loc[idx]:
            emit(idx, row, "fragment", repeat=FRAGMENT_OVERSAMPLE)
        elif idx in complete_pool:
            emit(idx, row, "complete")
        elif star_mask.loc[idx]:
            nd = dummy_count(smiles)
            emit(idx, row, "markush", repeat=2 if nd >= 6 else 1)
        else:
            stats["skipped"] += 1

    rng.shuffle(records)
    n_val = max(500, len(records) // 100)
    val, train = records[:n_val], records[n_val:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "train.jsonl", "w") as fh:
        for r in train:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "val.jsonl", "w") as fh:
        for r in val:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train={len(train)} val={len(val)} stats={stats}", flush=True)


if __name__ == "__main__":
    main()
