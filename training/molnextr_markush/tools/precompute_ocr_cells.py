"""Precompute ChemicalOCR cells for training images (§44.53).

Reads a pose_factory df parquet, runs docling ChemicalOCR on every unique
``file_path``, and writes a {file_path, ocr_cells} parquet for the merge into
the training frame. Resume-safe: existing rows are skipped, results flush
incrementally.

Usage:
  python tools/precompute_ocr_cells.py \
      --df experiments/moe/uspto_label_v1_train_df.parquet \
      --filter-starred --out experiments/moe/ocr_cells_shard.parquet \
      --device cuda --longest-edge 768
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--df", required=True, help="training df parquet (needs file_path)")
    p.add_argument("--out", required=True, help="output parquet {file_path, ocr_cells}")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0, help="cap on new images (0 = all)")
    p.add_argument("--longest-edge", type=int, default=768,
                   help="ChemicalOCR processor longest edge; 1820 validated on real "
                        "page crops, synthetic renders are smaller and fast at 768")
    p.add_argument("--filter-starred", action="store_true",
                   help="only rows whose SMILES contains '*' (markush rows)")
    p.add_argument("--model", default="docling-project/ChemicalOCR")
    p.add_argument("--flush-every", type=int, default=200)
    p.add_argument("--max-cells", type=int, default=24,
                   help="keep at most this many cells per image (JSON)")
    p.add_argument("--max-new-tokens", type=int, default=96,
                   help="ChemicalOCR generate cap; structure crops need <64 "
                        "tokens and the cap neutralises repetition loops")
    p.add_argument("--shard", type=int, default=0,
                   help="process only images with hash(path) %% num_shards == shard")
    p.add_argument("--num-shards", type=int, default=1,
                   help="total parallel shards (each writes its own out file)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import pandas as pd

    df = pd.read_parquet(args.df)
    paths = df["file_path"].dropna().astype(str).unique().tolist()
    if args.filter_starred and "SMILES" in df.columns:
        starred = set(
            df.loc[df["SMILES"].fillna("").astype(str).str.contains(r"\*", regex=True),
                   "file_path"].astype(str))
        paths = [p for p in paths if p in starred]
    done = set()
    out_path = Path(args.out)
    if args.num_shards > 1:
        out_path = out_path.with_name(
            f"{out_path.stem}.shard{args.shard}{out_path.suffix}")
    if out_path.exists():
        done = set(pd.read_parquet(out_path)["file_path"].astype(str))
        print(f"resume: {len(done)} images already done", flush=True)
    todo = [p for p in paths if p not in done and os.path.isfile(p)]
    if args.num_shards > 1:
        # crc32: Python's str hash is salted per process and must never be
        # used to partition work across workers.
        import zlib
        todo = [p for p in todo
                if zlib.crc32(p.encode("utf-8")) % args.num_shards == args.shard]
    missing = sum(1 for p in paths if p not in done and not os.path.isfile(p))
    if args.limit > 0:
        todo = todo[: args.limit]
    print(f"targets: {len(todo)} todo / {len(paths)} unique "
          f"(shard {args.shard}/{args.num_shards}, {missing} missing files), "
          f"out={out_path}", flush=True)

    from utils.chemical_ocr_service import ChemicalOCRService
    from PIL import Image

    service = ChemicalOCRService(args.model, device=args.device,
                                 longest_edge=int(args.longest_edge),
                                 max_new_tokens=int(args.max_new_tokens))

    import torch
    records: list[dict] = []
    t0 = time.time()
    errors = 0
    for index, path in enumerate(todo):
        try:
            image = Image.open(path).convert("RGB")
            cells = service(image)
            cells = cells[: args.max_cells]
            payload = json.dumps(
                [{"bbox": [round(float(v), 5) for v in c["bbox"]],
                  "text": str(c["text"])} for c in cells])
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # OOM / corrupt image / transient CUDA error
            errors += 1
            print(f"[{index}] ERROR {type(exc).__name__}: {str(exc)[:90]} {path}",
                  flush=True)
            if "out of memory" in str(exc).lower() and hasattr(torch, "cuda"):
                torch.cuda.empty_cache()
            payload = json.dumps([])
        records.append({"file_path": path, "ocr_cells": payload})
        if len(records) % int(args.flush_every) == 0:
            _flush(records, done, out_path)
            rate = len(records) / max(1e-9, time.time() - t0)
            remaining = (len(todo) - index) / max(rate, 1e-9) / 3600
            print(f"[{index + 1}/{len(todo)}] {rate:.2f} img/s "
                  f"eta={remaining:.1f}h errors={errors}", flush=True)
    _flush(records, done, out_path)
    print(f"done: {len(done)} total, errors={errors}, "
          f"wall={(time.time() - t0) / 3600:.2f}h", flush=True)


def _flush(records: list[dict], done: set, out_path: Path) -> None:
    import pandas as pd
    if not records:
        return
    frame = pd.DataFrame(records)
    if out_path.exists():
        frame = pd.concat([pd.read_parquet(out_path), frame], ignore_index=True)
    frame = frame.drop_duplicates(subset=["file_path"], keep="last")
    frame.to_parquet(out_path, index=False)
    done.update(frame["file_path"].astype(str))
    records.clear()


if __name__ == "__main__":
    main()
