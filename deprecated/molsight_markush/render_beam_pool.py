"""Re-render the round-1 synthetic pool with the beam side-chain layout.

Round-4 (beam) real-domain SFT: the synthetic corpus's spatial preparation
switches to beam_sidechain_layout.py — joint beam search (beam=7) over the
side-chain DOFs of a saw-tooth / ragged zig-zag macroring, CDK-refiner
second stage, force-field fallback — the layout whose renders matched the
real patent crop morphology in the visual nucleus comparison
(images/beam_sidechain/compare_*.png).  SAME SMILES pools as round-4z/4cdk:

  - round-1 synthetic pool: /data/molsight_markush_v13/uspto_mol/
    train_680k.csv (3496 rows -> 3482 unique SMILES)
  - round-3 vocabulary pool: 467 unique SMILES of
    /data/molsight_realpatent_sft_r3/aligned_synth_vocab/train_680k.csv
    (fluoro / quaternary / proline motifs), x2 depiction variants

Render pipeline identical in spirit to the whole corpus:
  beam_layout_coords(seed loop)          — accept only physically clean
                                           (0 collapse pairs, 0 crossings)
                                           AND stereo-roundtrip layouts
  random upright rotation                — as render_beam() does
  style_calibration.draw_molecule()      — ALIGNED profile (beam profile
                                           pins backbone bond to 42 px);
                                           palette + grayscale asserts
  render_stage11.finish()                — patent-scan post-chain

Every failure is NAMED in the manifest: rdkit_parse / beam_no_clean_layout /
ink_rejected / timeout / exception, each with the offending SMILES.
Nothing is silently dropped or silently degraded: unlike round-4z/4cdk
there is NO coordgen/circle fallback — a molecule the beam engine cannot
lay out cleanly is a counted skip.

Throughput note: the joint beam search costs 20-330 s per seed on the
180-300 atom pool molecules (congested scenes escalate tiers and can run
past 10 min), so each seed try is bounded by a wall-clock budget
(signal.alarm — an over-budget try aborts and the next seed runs) and the
pool render is a multi-hour job — run it under nohup (see
run_r4beam_all.sh).

Usage:
  python render_beam_pool.py --out /data/molsight_realpatent_sft_r4beam/beam_synth \
      --variants 1 --vocab-variants 2 --workers 12
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import signal
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

# render_stage11.py parses sys.argv[1] at import time — hide our CLI from it
_argv = sys.argv
sys.argv = sys.argv[:1]
try:
    from render_stage11 import finish          # patent-scan post-chain
finally:
    sys.argv = _argv
from style_calibration import draw_molecule
import zigzag_layout_v2 as zzv2
import beam_sidechain_layout as bsl

RDLogger.DisableLog("rdApp.*")

ROUND1_POOL = "/data/molsight_markush_v13/uspto_mol/train_680k.csv"
R3_VOCAB = "/data/molsight_realpatent_sft_r3/aligned_synth_vocab/train_680k.csv"

RENDER_TIMEOUT_S = 900       # per job; worst case 2 bounded seeds + slack
SKIP_LIST_CAP = 6000         # manifest keeps every named skip
LAYOUT_SEEDS = 2             # clean+stereo accepted seeds tried per molecule
SEED_BUDGET_S = 360          # wall-clock cap per seed try (SIGALRM); the
                             # slowest seed-42 pass measured on the pool was
                             # 321 s, so 360 s keeps every observable pass


def _read_smiles(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        fh.readline()
        for r in csv.reader(fh):
            if len(r) >= 2:
                out.append(r[1])
    return out


def _init_worker():
    # fork() copies the parent numpy RNG state — reseed per worker so
    # finish() noise fields differ across children
    np.random.seed((os.getpid() * 2654435761 + time.time_ns()) % (2 ** 32))


class SeedBudgetExceeded(Exception):
    """A single beam_layout_coords try blew its wall-clock budget."""


def _alarm_handler(signum, frame):
    raise SeedBudgetExceeded()


def _layout_clean(mol, seed, seeds, budget_s=SEED_BUDGET_S):
    """Seed loop with the pool acceptance gate: 0 collapse pairs +
    0 bond crossings + stereo round-trip.  Each try is bounded to
    `budget_s` seconds (SIGALRM — the beam search is a Python-level round
    loop, so the alarm fires between rounds).  Returns
    (mol_with_coords, rep, tag) with tag in {beam_first_seed,
    beam_retry_seed, beam_no_clean_layout}; the molecule is None only for
    the named skip."""
    signal.signal(signal.SIGALRM, _alarm_handler)
    best = None                     # (rank, laid, rep) best-effort kept
    n_over = 0
    for k in range(seeds):
        signal.alarm(int(budget_s))
        try:
            laid, rep = bsl.beam_layout_coords(mol, seed=seed + k)
        except SeedBudgetExceeded:
            n_over += 1
            continue
        finally:
            signal.alarm(0)
        if laid is None:
            continue
        m = rep["metrics"]
        clean = (m["bond_crossings"] == 0 and m["n_close_pairs"] == 0)
        stereo = bool(rep.get("stereo_roundtrip"))
        if clean and stereo:
            return laid, rep, ("beam_first_seed" if k == 0
                               else "beam_retry_seed")
        rank = (clean, stereo, -m["n_close_pairs"],
                -m["bond_crossings"], m["min_nonbonded_ratio"])
        if best is None or rank > best[0]:
            best = (rank, laid, rep)
    return None, (best[2] if best else {"error": "backbone"}), \
        ("beam_no_clean_layout",
         {"over_budget": n_over, "seeds": seeds,
          "budget_s": budget_s} if n_over else
         {"seeds": seeds, "budget_s": budget_s})


def _render_one(job):
    idx, smiles, out_dir, seed = job
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return idx, None, ("rdkit_parse", smiles)

        laid, rep, tag = _layout_clean(mol, seed, LAYOUT_SEEDS)
        if laid is None:
            reason, info = (tag if isinstance(tag, tuple)
                            else (tag, {}))
            m = rep.get("metrics", {})
            detail = (f"{smiles} :: seeds={info.get('seeds')} "
                      f"over_budget={info.get('over_budget', 0)}"
                      f"/{info.get('budget_s')}s "
                      f"best seed={rep.get('seed_used')} "
                      f"close={m.get('n_close_pairs')} "
                      f"cross={m.get('bond_crossings')} "
                      f"stereo={rep.get('stereo_roundtrip')}")
            return idx, None, (reason, detail)

        # ---- render: upright rotation + ALIGNED draw, as render_beam() ----
        seed_used = rep["seed_used"]
        rng_draw = random.Random(seed_used + 101)
        np.random.seed(seed_used + 101)
        rad = math.radians(-rng_draw.uniform(0, 360))
        render_mol = bsl.patent_label_style(Chem.Mol(laid))
        conf = render_mol.GetConformer()
        for i in range(render_mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, Point3D(
                p.x * math.cos(rad) - p.y * math.sin(rad),
                p.x * math.sin(rad) + p.y * math.cos(rad), 0.0))
        profile = zzv2.render_profile()          # ALIGNED, bond pinned 42 px
        zzv2._assert_black_white(profile)
        img = draw_molecule(render_mol, profile, rng_draw)
        zzv2._assert_grayscale(img)

        path = os.path.join(out_dir, "images", f"{idx:06d}.png")
        rng_finish = random.Random(seed_used * 1000003 + idx)
        if not finish(img, rng_finish, path):
            return idx, None, ("ink_rejected", smiles)
        return idx, path, tag
    except Exception as exc:
        return idx, None, ("exception", f"{smiles} :: "
                                        f"{type(exc).__name__}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True,
                    help="output root; images/ + train_680k.csv + manifest")
    ap.add_argument("--pool", default=ROUND1_POOL)
    ap.add_argument("--vocab", default=R3_VOCAB)
    ap.add_argument("--variants", type=int, default=1,
                    help="renders per round-1 pool SMILES")
    ap.add_argument("--vocab-variants", type=int, default=2,
                    help="renders per round-3 vocabulary SMILES")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=LAYOUT_SEEDS,
                    help="clean-layout seed tries per molecule")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0,
                    help="pilot: cap total jobs (0 = full pool)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore any partial CSV from an earlier run")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    base, seen = [], set()
    for smi in _read_smiles(args.pool):
        if smi not in seen:
            seen.add(smi)
            base.append(smi)
    n_pool = len(base)
    vocab = []
    for smi in _read_smiles(args.vocab):
        if smi not in seen:
            seen.add(smi)
            vocab.append(smi)
    print(f"pool: round-1 synth {n_pool} + r3 vocab {len(vocab)} unique "
          f"SMILES (beam side-chain layout, ALIGNED style)", flush=True)

    jobs, smi_of_job = [], {}
    idx = 0
    for smi in base:
        for v in range(args.variants):
            smi_of_job[idx] = smi
            jobs.append((idx, smi, args.out, args.seed + v))
            idx += 1
    for smi in vocab:
        for v in range(args.vocab_variants):
            smi_of_job[idx] = smi
            jobs.append((idx, smi, args.out, args.seed + v))
            idx += 1
    if args.limit:
        jobs = jobs[:args.limit]

    # ---- crash resume: the partial CSV is flushed every 25 completions;
    #      re-running skips images already rendered ----
    csv_path = os.path.join(args.out, "train_680k.csv")
    partial_path = csv_path + ".partial"
    rows = []
    if not args.no_resume and os.path.isfile(partial_path):
        with open(partial_path, encoding="utf-8") as fh:
            fh.readline()
            for r in csv.reader(fh):
                if len(r) >= 2:
                    rows.append({"file_path": r[0], "SMILES": r[1]})
        done_ids = {int(os.path.splitext(os.path.basename(r["file_path"]))[0])
                    for r in rows}
        jobs = [j for j in jobs if j[0] not in done_ids]
        print(f"resume: {len(rows)} rows already rendered, "
              f"{len(jobs)} jobs remain", flush=True)
    print(f"{len(jobs)} render jobs ({args.variants}x pool + "
          f"{args.vocab_variants}x vocab, {args.seeds} seed tries), "
          f"{args.workers} workers", flush=True)

    skip_counts, layout_counts = {}, {}
    named_skips, job_seconds = [], []
    t0 = time.time()
    with Pool(processes=args.workers, maxtasksperchild=4,
              initializer=_init_worker) as mp:
        asyncs = [(j[0], mp.apply_async(_render_one, (j,))) for j in jobs]
        done = 0
        for idx, a in asyncs:
            tj = time.time()
            try:
                _, path, tag = a.get(timeout=RENDER_TIMEOUT_S)
                job_seconds.append(time.time() - tj)
            except Exception as exc:
                path, tag = None, None
                timed_out = "timed out" in str(exc) or "Timeout" in \
                    type(exc).__name__
                if timed_out:
                    named_skips.append(
                        {"idx": idx, "reason": "timeout",
                         "detail": f"{smi_of_job[idx]} :: job exceeded "
                                   f"{RENDER_TIMEOUT_S}s"})
                    skip_counts["timeout"] = \
                        skip_counts.get("timeout", 0) + 1
                else:
                    named_skips.append(
                        {"idx": idx, "reason": "exception",
                         "detail": f"{smi_of_job[idx]} :: worker: {exc}"})
                    skip_counts["exception"] = \
                        skip_counts.get("exception", 0) + 1
            if path:
                rows.append({"file_path": path, "SMILES": smi_of_job[idx]})
                layout_counts[tag] = layout_counts.get(tag, 0) + 1
            elif tag is None:
                pass                      # already counted above
            else:
                reason, detail = tag, tag
                if isinstance(tag, tuple):
                    reason, detail = tag
                skip_counts[reason] = skip_counts.get(reason, 0) + 1
                if len(named_skips) < SKIP_LIST_CAP:
                    named_skips.append({"idx": idx, "reason": reason,
                                        "detail": detail})
            done += 1
            if done % 25 == 0:
                el = time.time() - t0
                mean = (sum(job_seconds) / len(job_seconds)
                        if job_seconds else 0.0)
                eta = mean * (len(jobs) - done) / max(args.workers, 1)
                print(f"  {done}/{len(jobs)} rendered, {len(rows)} kept, "
                      f"{done - len(rows)} skipped, mean job {mean:.0f}s, "
                      f"eta ~{eta / 60:.0f} min ({el:.0f}s) "
                      f"layout={layout_counts} skips={skip_counts}",
                      flush=True)
                with open(partial_path, "w", encoding="utf-8",
                          newline="") as fh:
                    fh.write("file_path,SMILES\n")
                    for r in rows:
                        smi = r["SMILES"]
                        if "," in smi or '"' in smi:
                            smi = '"' + smi.replace('"', '""') + '"'
                        fh.write(f"{r['file_path']},{smi}\n")

    csv_path = os.path.join(args.out, "train_680k.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("file_path,SMILES\n")
        for r in rows:
            smi = r["SMILES"]
            if "," in smi or '"' in smi:
                smi = '"' + smi.replace('"', '""') + '"'
            fh.write(f"{r['file_path']},{smi}\n")

    manifest = {
        "engine": "beam_sidechain_layout.py (joint beam search beam=7 + "
                  "CDK-style refiner + force-field fallback)",
        "profile": "ALIGNED (style_calibration.draw_molecule, beam profile "
                   "pins backbone bond to 42 px) + render_stage11.finish()",
        "pipeline": "beam_layout_coords seed loop gated on physical-clean "
                    "(0 collapse pairs, 0 crossings) + stereo roundtrip -> "
                    "upright rotation -> draw_molecule(ALIGNED) -> finish(), "
                    "via render_beam_pool.py",
        "layout_seed_policy": {
            "seed_tries": args.seeds,
            "seed_budget_s": SEED_BUDGET_S,
            "accept": "bond_crossings == 0 and n_close_pairs == 0 and "
                      "stereo_roundtrip",
            "fallback": "none — unclean molecules are named skips",
        },
        "layout_counts": layout_counts,
        "skip_counts": skip_counts,
        "named_skips": named_skips,
        "pools": {
            "round1_synth_csv": args.pool,
            "round1_synth_smiles": n_pool,
            "r3_vocab_csv": args.vocab,
            "r3_vocab_smiles": len(vocab),
        },
        "seed": args.seed,
        "variants_per_smiles": args.variants,
        "vocab_variants": args.vocab_variants,
        "jobs": len(jobs),
        "rows_rendered": len(rows),
        "mean_job_seconds": round(sum(job_seconds) / len(job_seconds), 1)
                            if job_seconds else None,
        "wall_seconds": round(time.time() - t0, 1),
        "csv": csv_path,
        "images_dir": os.path.join(args.out, "images"),
    }
    with open(os.path.join(args.out, "render_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    if os.path.isfile(partial_path):
        os.remove(partial_path)
    print(json.dumps({k: v for k, v in manifest.items()
                      if k != "named_skips"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
