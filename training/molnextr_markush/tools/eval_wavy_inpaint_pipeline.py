#!/usr/bin/env python3
"""Phase 2 evaluation: full detect -> inpaint -> re-decode pipeline.

Runs the trained wavy Mask R-CNN on real fragments, inpaints detected wavy
pixels, re-decodes with MolNexTR, and compares ghost-carbon count / tanimoto
against the baseline (no inpaint). This is the end-to-end gate for Phase 2.
"""
from __future__ import annotations

import csv
import os
import re
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training" / "molnextr_markush" / "tools"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from utils.structure_recognition import StructureRecognizer  # noqa: E402


def count_carbon(s: str) -> int:
    return len(re.findall(r"C(?![lorpsnb])", s.replace("Cl", "").replace("Br", "")))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(ROOT / "experiments/wavy_seg/wavy_maskrcnn_best.pth"))
    parser.add_argument("--moe-config", default=str(ROOT / "experiments/moe/molnextr_zigzag_cn_retrain/moe_config.json"))
    parser.add_argument("--eval-csv", default=str(ROOT / "training/molnextr_markush/data/real_wavy_hard_eval/eval.csv"))
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rec = StructureRecognizer(moe_config_path=args.moe_config, device=str(device))

    from train_wavy_maskrcnn import build_wavy_maskrcnn
    det = build_wavy_maskrcnn(num_classes=2, pretrained_backbone=False).to(device)
    det.load_state_dict(torch.load(args.ckpt, map_location=device))
    det.eval()
    from torchvision.transforms import functional as TF

    rows = list(csv.DictReader(open(args.eval_csv)))
    frags = [r for r in rows if r["structure_type"] == "fragment"]
    if args.limit:
        frags = frags[: args.limit]

    base_ghost = 0
    inpaint_fixed = 0
    inpaint_same = 0
    inpaint_worse = 0
    detected = 0
    for r in frags:
        path = r["file_path"].replace("/data/BioChemInsight", str(ROOT))
        img = cv2.imread(path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gold = r["SMILES"]
        gold_c = count_carbon(gold)

        # baseline decode
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            cv2.imwrite(tf.name, img)
            tmp = tf.name
        try:
            bp = rec.predict_segment_files([tmp], expected_structure_types=["fragment"])[0]
        finally:
            os.unlink(tmp)
        base_smi = bp.raw.get("predicted_smiles", "") if isinstance(bp.raw, dict) else ""
        base_c = count_carbon(base_smi)
        is_ghost = base_c > gold_c
        if is_ghost:
            base_ghost += 1

        # detect wavy
        pil = Image.fromarray(rgb).resize((384, 384), Image.LANCZOS)
        with torch.no_grad():
            out = det(TF.to_tensor(pil).unsqueeze(0).to(device))[0]
        scores = out["scores"].cpu().numpy()
        masks = out["masks"].cpu().numpy()[:, 0]
        keep = scores >= args.score_threshold
        n_det = int(keep.sum())
        if n_det > 0:
            detected += 1

        if n_det == 0:
            continue
        # inpaint all detected wavy masks (resized to original size)
        h, w = rgb.shape[:2]
        mask_full = np.zeros((h, w), np.uint8)
        for i in np.where(keep)[0]:
            m = (masks[i] > 0.5).astype(np.uint8)
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_full |= m
        if mask_full.sum() > 0:
            mask_full = cv2.dilate(mask_full, np.ones((3, 3), np.uint8), 1)
            inpainted = cv2.inpaint(rgb, mask_full, 3, cv2.INPAINT_TELEA)
        else:
            inpainted = rgb

        # re-decode
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            cv2.imwrite(tf.name, cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR))
            tmp2 = tf.name
        try:
            ip = rec.predict_segment_files([tmp2], expected_structure_types=["fragment"])[0]
        finally:
            os.unlink(tmp2)
        in_smi = ip.raw.get("predicted_smiles", "") if isinstance(ip.raw, dict) else ""
        in_c = count_carbon(in_smi)

        if is_ghost:
            if in_c <= gold_c:
                inpaint_fixed += 1
                tag = "FIXED"
            elif in_c < base_c:
                inpaint_same += 1
                tag = "better"
            else:
                inpaint_worse += 1
                tag = "worse"
        else:
            tag = "no-ghost"
        print(f"{r['segment']:<16} det={n_det} gold_c={gold_c} base_c={base_c} inp_c={in_c} {tag}")
        print(f"   gold={gold[:24]}")
        print(f"   base={base_smi[:24]}")
        print(f"   inp ={in_smi[:24]}")

    print()
    print(f"=== 汇总 ({len(frags)} fragments) ===")
    print(f"检出率: {detected}/{len(frags)} = {detected/max(1,len(frags)):.0%}")
    print(f"baseline 幽灵碳: {base_ghost}/{len(frags)}")
    print(f"inpaint 修复(碳数纠正): {inpaint_fixed}/{base_ghost}")
    print(f"inpaint 部分改善: {inpaint_same}/{base_ghost}")
    print(f"inpaint 变差: {inpaint_worse}/{base_ghost}")


if __name__ == "__main__":
    main()
