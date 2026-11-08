"""Real-vs-synthetic render style comparison for the cyclopeptide pipeline.

Measures every image of a molecule-paired dataset (see render_style_pairs.py)
with the raster style metrics of style_metrics.py and reports, per metric,
real vs synthetic distribution statistics: mean/std per side, Wasserstein-1
distance, and smoothed KL divergence (real || synth).

Metrics (all length ratios are normalized by the image's median bond
length, so dpi / crop scale differences cancel):

  bond_angle            pooled junction-angle histogram (KL + Wasserstein
                        on degrees)
  angle_120_frac        per-image share of angles within 120 +/- 8 deg
                        (regular-hexagon / ring uniformity sharpness)
  bond_len_cv           per-image bond length coefficient of variation
                        (ring-evenness proxy)
  junction_len_ratio    per-image median max/min incident bond length
  stroke_ratio          ink thickness / bond length (line weight)
  ink_gray_mean         mean gray of ink pixels (ink darkness)
  ink_frac              ink fraction inside the ink bounding box (density)
  glyph_height_ratio    label glyph height / bond length
  glyph_clearance_ratio label-to-bond-ink gap / bond length
  labels_per_bond       glyph clusters per bond (abbreviation density)
  dbl_gap_ratio         double-bond parallel line gap / bond length
  dbl_inner_outer       inner/outer line length ratio of double bonds
  scale_density         ink-bbox diagonal / bond length (fill scale)

Usage:
  python compare_render_style.py --pairs /tmp/style_pairs_baseline/pairs.json \
      [--out /tmp/style_report_baseline.json]
  python compare_render_style.py --pairs-after /tmp/style_pairs_aligned/pairs.json \
      --report-before /tmp/style_report_baseline.json \
      [--out /tmp/style_report_aligned.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style_metrics import measure_style, pooled_angle_stats

# metric key -> (label, per-image or pooled)
METRICS = [
    ("angle_120_frac", "ring-angle uniformity (120+/-8 deg share)"),
    ("bond_len_cv", "bond length CV (ring evenness)"),
    ("junction_len_ratio", "junction max/min bond length"),
    ("stroke_ratio", "stroke width / bond length"),
    ("ink_gray_mean", "ink gray level (0-1)"),
    ("ink_frac", "ink fraction in bbox"),
    ("glyph_height_ratio", "label height / bond length"),
    ("glyph_clearance_ratio", "label-to-bond gap / bond length"),
    ("labels_per_bond", "glyph clusters per bond"),
    ("dbl_gap_ratio", "double-bond gap / bond length"),
    ("dbl_inner_outer", "double-bond inner/outer length"),
    ("scale_density", "bbox diagonal / bond length"),
    ("aspect", "bbox width/height"),
]


def _measure_one(job):
    path = job
    try:
        m = measure_style(path)
        return path, m.to_dict()
    except Exception as exc:
        return path, {"error": str(exc)[:120]}


def measure_pairs(pairs_file: str, workers: int = 8) -> dict:
    pairs = json.load(open(pairs_file))
    jobs = []
    for p in pairs:
        jobs.append(p["real"])
        jobs.append(p["synth"])
    results = {}
    with Pool(processes=workers) as pool:
        for path, d in pool.imap_unordered(_measure_one, jobs, chunksize=2):
            results[path] = d
    real, synth = [], []
    for p in pairs:
        mr, ms = results.get(p["real"]), results.get(p["synth"])
        if not mr or not ms or "error" in mr or "error" in ms:
            continue
        real.append(mr)
        synth.append(ms)
    return {"pairs_file": pairs_file, "n_pairs_measured": len(real),
            "real": real, "synth": synth}


def build_report(measured: dict) -> dict:
    real, synth = measured["real"], measured["synth"]
    report = {"n_pairs": measured["n_pairs_measured"], "metrics": {}}
    for key, _label in METRICS:
        rv = [m[key] for m in real if m.get(key)]
        sv = [m[key] for m in synth if m.get(key)]
        from style_metrics import distribution_stats
        report["metrics"][key] = distribution_stats(rv, sv)
    ra = [m["angles_deg"] for m in real]
    sa = [m["angles_deg"] for m in synth]
    report["metrics"]["bond_angle"] = pooled_angle_stats(ra, sa)
    return report


def render_table(before: dict, after: dict | None) -> str:
    """Markdown table; `after` optional for the aligned pass."""
    def row(key, label, rep):
        m = rep["metrics"].get(key, {})
        mr = m.get("mean_real"), m.get("std_real")
        ms = m.get("mean_synth"), m.get("std_synth")
        w = m.get("wasserstein")
        k = m.get("kl_real_to_synth")
        fmt = lambda v, n=3: ("-" if v is None else f"{v:.{n}f}")
        return (f"| {label} | {fmt(mr[0])} ± {fmt(mr[1])} | "
                f"{fmt(ms[0])} ± {fmt(ms[1])} | {fmt(w, 4)} | {fmt(k, 3)} |")

    header = ("| metric | real mean ± std | synth mean ± std | "
              "Wasserstein-1 | KL(real||synth) |")
    sep = "|---|---|---|---|---|"
    lines = [header, sep]
    for key, label in METRICS:
        lines.append(row(key, label, before))
    lines.append(row("bond_angle", "bond angle (deg, pooled)", before))
    if after is not None:
        lines += ["", "**After alignment (aligned profile)**", header, sep]
        for key, label in METRICS:
            lines.append(row(key, label, after))
        lines.append(row("bond_angle", "bond angle (deg, pooled)", after))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", help="pairs.json of a baseline render pass")
    ap.add_argument("--pairs-after", help="pairs.json of an aligned pass")
    ap.add_argument("--report-before",
                    help="existing baseline report to diff against")
    ap.add_argument("--out", help="write the report JSON here")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.pairs:
        measured = measure_pairs(args.pairs, args.workers)
        report = build_report(measured)
        report["pairs_file"] = args.pairs
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(report, fh, indent=1)
        print(render_table(report, None))
        if args.out:
            print(f"\nreport -> {args.out}")
    elif args.pairs_after and args.report_before:
        before = json.load(open(args.report_before))
        measured = measure_pairs(args.pairs_after, args.workers)
        after = build_report(measured)
        after["pairs_file"] = args.pairs_after
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(after, fh, indent=1)
        print(render_table(before, after))
        if args.out:
            print(f"\nreport -> {args.out}")
    else:
        ap.error("give --pairs, or --pairs-after with --report-before")


if __name__ == "__main__":
    main()
