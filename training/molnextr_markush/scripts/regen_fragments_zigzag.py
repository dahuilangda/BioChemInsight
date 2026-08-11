#!/usr/bin/env python3
"""Regenerate all fragment shards with the corrected zigzag wavy code.

Each shard is an independent call to run_molnextr_dataset_pipeline.py with
--branch fragment. Shards run in parallel; progress + failures are logged.
Resumable: skips shards whose CSV already has >= 240 rows (256 - tolerance)
when --skip-existing is set.

Usage:
    python3 training/molnextr_markush/scripts/regen_fragments_zigzag.py \
        --shards 0-1959 --jobs 8
"""
import argparse
import concurrent.futures
import os
import subprocess
import sys
import time

ROOT = "/data/BioChemInsight"
PY = os.environ.get("REGEN_PY", "/home/dahuilangda/miniconda3/envs/llm/bin/python")
FRAG_ROOT = os.path.join(
    ROOT,
    "training/molnextr_markush/data/generated/pose_factory/"
    "molnextr_moe_production_v1/fragment",
)


def shard_csv_path(shard):
    return os.path.join(FRAG_ROOT, f"s{shard:03d}", "attachment_fragment_positive.csv")


def csv_row_count(shard):
    p = shard_csv_path(shard)
    if not os.path.exists(p):
        return 0
    try:
        with open(p) as f:
            return sum(1 for _ in f) - 1  # minus header
    except Exception:
        return 0


def gen_shard(shard, rows, seed):
    """Regenerate one fragment shard. Returns (shard, ok, msg)."""
    start = time.time()
    # remove old shard dir to guarantee clean zigzag regen
    sdir = os.path.join(FRAG_ROOT, f"s{shard:03d}")
    if os.path.isdir(sdir):
        # keep dir, but the pipeline overwrites images+csv; explicit clean avoids stale files
        import shutil

        shutil.rmtree(sdir)
    cmd = [
        PY,
        os.path.join(
            ROOT,
            "training/molnextr_markush/tools/run_molnextr_dataset_pipeline.py",
        ),
        "--dataset-id",
        "molnextr_moe_production_v1",
        "--branch",
        "fragment",
        "--start-shard",
        str(shard),
        "--shards",
        "1",
        "--fragment-rows",
        str(rows),
        "--seed",
        str(seed),
        "--execute",
    ]
    try:
        r = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return (shard, False, "timeout")
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-200:]
        return (shard, False, f"rc={r.returncode}: {tail}")
    n = csv_row_count(shard)
    if n < rows - 20:
        return (shard, False, f"low_rows={n}")
    return (shard, True, f"{n}rows {time.time()-start:.0f}s")


def parse_range(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="0-1959")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip shards with >= rows-20 positives already")
    args = ap.parse_args()

    shards = parse_range(args.shards)
    todo = []
    for s in shards:
        if args.skip_existing and csv_row_count(s) >= args.rows - 20:
            continue
        todo.append(s)
    print(f"shards total={len(shards)} todo={len(todo)} jobs={args.jobs}", flush=True)
    if not todo:
        print("nothing to do")
        return

    done = 0
    failed = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(gen_shard, s, args.rows, args.seed): s for s in todo}
        for fut in concurrent.futures.as_completed(futs):
            s = futs[fut]
            try:
                shard, ok, msg = fut.result()
            except Exception as e:
                shard, ok, msg = s, False, f"exc={e}"
            done += 1
            status = "OK " if ok else "FAIL"
            print(f"[{done}/{len(todo)}] s{shard:03d} {status} {msg}", flush=True)
            if not ok:
                failed.append(shard)
    elapsed = time.time() - t0
    print(f"\nDONE {done}/{len(todo)} in {elapsed:.0f}s  failed={failed}")
    if failed:
        print("retry with: --shards " + ",".join(str(s) for s in failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
