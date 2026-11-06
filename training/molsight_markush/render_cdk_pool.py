"""Re-render the round-1 synthetic pool with CDK saw-tooth engine layout.

Round-4 real-domain SFT: the synthetic corpus's spatial preparation moves
from the CoordGen circular-wheel layouts (every prior round) to the real-
patent saw-tooth macroring polygon.  Primary engine is the CDK
StructureDiagramGenerator — the only mature EXTERNAL engine whose macroring
reproduces the patent saw-tooth (engine study record in render_via_engine.py;
visual confirmation experiments/engine_layout_eval/comparison_sheet.png).

Layout-source policy (measured, not assumed — pool shape audit, see the
manifest): CDK's polygonal macroring layout holds for the smaller rings
(mean interior angle ~120 deg, zigzag alternation > 0) but degrades to a
near-perfect circle for most large peptide macrorings (angle ~175 deg,
zigzag 0 — 21/24 molecules of a seeded 24-molecule pool audit).  So each
molecule takes the FIRST layout source that actually yields the saw-tooth:

  1. cdk_sawtooth   — CDK layout whose macroring passes the shape gate
                      (mean angle <= 145 deg); also cdk_smallring for
                      molecules without a >= 10-atom macroring, where CDK
                      is already clean
  2. zigzag_template — zigzag_layout.zigzag_coords (hex-lattice all-120-deg
                      ragged-box contour), 3 seeded retries; the incumbent
                      saw-tooth template kept "in the pipeline" per the
                      render_via_engine.py study record
  3. cdk_circle     — molecule whose CDK layout stayed circular and whose
                      template attempts all failed QA (counted, never silent)
  4. coordgen       — final fallback when both CDK and the template fail
                      outright (skips stay named)

Every failure is NAMED: the manifest counts skips per exception class and
lists each skipped SMILES with its reason (rdkit_parse / cdk_invalid_input
/ cdk_layout_failed / engine_not_installed / ink_rejected / exception).
Engine coordinates feed the SAME single render path as the whole corpus:

  layout coords -> style_calibration.draw_molecule(ALIGNED)   (style overlay)
               -> render_stage11.finish()                     (patent scan)

SMILES pools (full re-render, no subsampling):
  - round-1 synthetic pool: /data/molsight_markush_v13/uspto_mol/
    train_680k.csv (3496 rows -> 3482 unique SMILES, the pool the round-2
    aligned corpus and the round-1 replay rows were drawn from)
  - round-3 vocabulary pool: 467 unique SMILES of
    /data/molsight_realpatent_sft_r3/aligned_synth_vocab/train_680k.csv
    (fluoro / quaternary / proline motifs), x2 depiction variants

Usage:
  python render_cdk_pool.py --out /data/molsight_realpatent_sft_r4cdk/cdk_synth \
      --variants 1 --vocab-variants 2 --workers 10
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem, RDLogger

# render_stage11.py parses sys.argv[1] at import time — hide our CLI from it
_argv = sys.argv
sys.argv = sys.argv[:1]
try:
    from render_stage11 import finish          # patent-scan post-chain
finally:
    sys.argv = _argv
from style_calibration import ALIGNED, draw_molecule, coordgen_coords
import render_via_engine as rve
from zigzag_layout import zigzag_coords, find_macroring, layout_metrics

RDLogger.DisableLog("rdApp.*")

ROUND1_POOL = "/data/molsight_markush_v13/uspto_mol/train_680k.csv"
R3_VOCAB = "/data/molsight_realpatent_sft_r3/aligned_synth_vocab/train_680k.csv"
ENGINE = "cdk"
RENDER_TIMEOUT_S = 300
SKIP_LIST_CAP = 4000     # manifest keeps every named skip up to this many
TEMPLATE_SEED_RETRIES = 3
SAWTOOTH_MAX_MEAN_ANGLE = 145.0   # circles measure ~173-177 deg


def _macroring_sawtooth(mol):
    """True when the placed macroring passes the saw-tooth shape gate.
    None when the molecule has no >= 10-atom macroring (small-ring case)."""
    try:
        ring = find_macroring(mol)
    except ValueError:
        return None
    if not ring or len(ring) < 10:
        return None
    metrics = layout_metrics(mol, ring)
    return metrics["mean_angle"] <= SAWTOOTH_MAX_MEAN_ANGLE


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


def _render_one(job):
    idx, smiles, out_dir, seed = job
    cdk_mol = None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return idx, None, ("rdkit_parse", smiles)

        # ---- 1. CDK engine layout ----
        try:
            cdk_mol = rve._load_engine_coords(smiles, ENGINE)
        except rve.EngineInvalidInputError as exc:
            return idx, None, ("cdk_invalid_input", f"{smiles} :: {exc}")
        except rve.EngineNotInstalledError as exc:
            return idx, None, ("engine_not_installed", f"{smiles} :: {exc}")
        except rve.EngineLayoutFailedError as exc:
            cdk_mol = None
            cdk_error = f"{smiles} :: {exc}"
        else:
            cdk_error = None

        sawtooth = _macroring_sawtooth(cdk_mol) if cdk_mol else None
        if sawtooth:
            layout_used = "cdk_sawtooth"       # gate passed at ~120 deg
            laid = cdk_mol
        elif cdk_mol is not None and sawtooth is None:
            layout_used = "cdk_smallring"      # no big macroring: CDK is clean
            laid = cdk_mol
        else:
            # ---- 2. zig-zag template fallback (CDK went circular) ----
            laid = None
            for k in range(TEMPLATE_SEED_RETRIES):
                laid = zigzag_coords(mol,
                                     random.Random(seed + k), seed=seed + k)
                if laid is not None:
                    layout_used = "zigzag_template"
                    break
            if laid is None:
                # ---- 3. counted CDK circle; ----
                # ---- 4. coordgen only if CDK failed outright ----
                if cdk_mol is not None:
                    layout_used = "cdk_circle"
                    laid = cdk_mol
                else:
                    laid = coordgen_coords(mol)
                    if laid is None:
                        return idx, None, (
                            "cdk_layout_failed",
                            cdk_error or f"{smiles} :: no usable layout")
                    layout_used = "coordgen"

        rng = random.Random(seed * 1000003 + idx)
        img = draw_molecule(laid, ALIGNED, rng)
        path = os.path.join(out_dir, "images", f"{idx:06d}.png")
        if not finish(img, rng, path):
            return idx, None, ("ink_rejected", smiles)  # over the envelope
        return idx, path, layout_used
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
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
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
          f"SMILES ({ENGINE} engine, ALIGNED style overlay)", flush=True)

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
    print(f"{len(jobs)} render jobs ({args.variants}x pool + "
          f"{args.vocab_variants}x vocab), {args.workers} workers",
          flush=True)

    rows, skip_counts = [], {}
    layout_counts = {}
    named_skips = []
    t0 = time.time()
    with Pool(processes=args.workers, maxtasksperchild=8,
              initializer=_init_worker) as mp:
        asyncs = [(j[0], mp.apply_async(_render_one, (j,))) for j in jobs]
        done = 0
        for idx, a in asyncs:
            try:
                _, path, info = a.get(timeout=RENDER_TIMEOUT_S)
            except Exception as exc:
                path, info = None, (
                    "exception", f"{smi_of_job[idx]} :: worker: {exc}")
            if path:
                rows.append({"file_path": path, "SMILES": smi_of_job[idx]})
                layout_counts[info] = layout_counts.get(info, 0) + 1
            else:
                reason, detail = info
                skip_counts[reason] = skip_counts.get(reason, 0) + 1
                if len(named_skips) < SKIP_LIST_CAP:
                    named_skips.append({"idx": idx, "reason": reason,
                                        "detail": detail})
            done += 1
            if done % 250 == 0:
                print(f"  {done}/{len(jobs)} rendered, {len(rows)} kept, "
                      f"{done - len(rows)} skipped "
                      f"({time.time() - t0:.0f}s) "
                      f"layout={layout_counts}", flush=True)

    csv_path = os.path.join(args.out, "train_680k.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("file_path,SMILES\n")
        for r in rows:
            smi = r["SMILES"]
            if "," in smi or '"' in smi:
                smi = '"' + smi.replace('"', '""') + '"'
            fh.write(f"{r['file_path']},{smi}\n")

    manifest = {
        "engine": ENGINE,
        "engine_bundle": str(rve.cdk_bundle_jar()),
        "profile": "ALIGNED (style_calibration.py, commit 59ba6dd) "
                   "style overlay + render_stage11.finish()",
        "pipeline": "layout-source ladder (cdk saw-tooth gate -> "
                    "zigzag_template -> cdk_circle -> coordgen) -> "
                    "draw_molecule(ALIGNED) -> finish(), via render_cdk_pool.py",
        "layout_source_policy": {
            "sawtooth_gate": f"macroring mean interior angle <= "
                             f"{SAWTOOTH_MAX_MEAN_ANGLE} deg (circles measure "
                             f"~173-177 deg)",
            "template_seed_retries": TEMPLATE_SEED_RETRIES,
            "audit_note": "CDK's polygonal macroring layout holds only for "
                          "smaller rings; a seeded 24-molecule pool audit "
                          "measured 21/24 CDK macrorings circular "
                          "(angle ~175 deg), hence the template fallback",
        },
        "pools": {"round1_synth_csv": args.pool, "round1_synth_smiles": n_pool,
                  "r3_vocab_csv": args.vocab, "r3_vocab_smiles": len(vocab)},
        "seed": args.seed,
        "variants_per_smiles": args.variants,
        "vocab_variants": args.vocab_variants,
        "jobs": len(jobs),
        "rows_rendered": len(rows),
        "layout_counts": layout_counts,
        "skips_total": len(jobs) - len(rows),
        "skip_counts_by_class": skip_counts,
        "named_skips": named_skips,
        "csv": csv_path,
        "images_dir": os.path.join(args.out, "images"),
    }
    with open(os.path.join(args.out, "render_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps({k: v for k, v in manifest.items()
                      if k != "named_skips"}, indent=2), flush=True)
    if named_skips:
        print(f"named skips (first 10 of {len(named_skips)}):",
              flush=True)
        for s in named_skips[:10]:
            print(f"  [{s['reason']}] {s['detail'][:140]}", flush=True)


if __name__ == "__main__":
    main()
