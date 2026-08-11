#!/usr/bin/env python3
"""Phase 1: Mask R-CNN wavy-bond detector trainer.

A dedicated instance-segmentation model for wavy attachment bonds, reusing the
DECIMER Mask R-CNN's ResNet+FPN backbone as a pretrained feature extractor
but with standard torchvision detection heads (trainable, no Matterport weight
compatibility baggage). This is the production detector (Phase 0 validated that
synthetic->real transfer works; Phase 1 gives it object-level bbox+mask
discrimination the U-Net lacked).

Pipeline: synthetic COCO data (gen_wavy_seg_data.py --coco) -> Mask R-CNN
(num_classes=2: background + wavy) -> evaluate on real_wavy_hard fragments.

Usage:
    python train_wavy_maskrcnn.py --data experiments/wavy_seg/train --epochs 20
    python train_wavy_maskrcnn.py --eval-only --ckpt <path> --eval-csv <real.csv>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.src.base_trainer import (  # noqa: E402
    BaseCocoMaskRCNNTrainer,
    CocoDetectionDataset,
    coco_collate,
    build_maskrcnn,
)
from torchvision.transforms import functional as TF  # noqa: E402


class WavyMaskRCNNTrainer(BaseCocoMaskRCNNTrainer):
    """Mask R-CNN trainer for single-class wavy-bond detection."""

    NUM_CLASSES = 2
    CHECKPOINT_NAME = "wavy_maskrcnn_best.pth"

    def build_scheduler(self, optimizer):
        # Original used step_size=6 (not the base class default of 8)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader | None]:
        data_dir = Path(self.args.data)
        coco = COCO(str(data_dir / "annotations.json"))
        img_ids = list(coco.imgs.keys())
        rng = np.random.RandomState(7)
        rng.shuffle(img_ids)
        n_val = max(1, int(len(img_ids) * self.args.val_frac))
        val_ids = img_ids[:n_val]
        # positives + 1/3 negatives for false-positive suppression
        pos_train = [i for i in img_ids[n_val:] if len(coco.getAnnIds(imgIds=i)) > 0]
        neg_train = [i for i in img_ids[n_val:] if len(coco.getAnnIds(imgIds=i)) == 0]
        train_ids = pos_train + neg_train[: max(1, len(pos_train) // 3)]

        # fixed_label=1: original WavyCocoDataset always used label 1
        train_ds = CocoDetectionDataset(data_dir / "images", coco, augment=True, fixed_label=1)
        train_ds.ids = train_ids
        val_ds = CocoDetectionDataset(data_dir / "images", coco, augment=False, fixed_label=1)
        val_ds.ids = val_ids
        train_dl = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=coco_collate,
        )
        val_dl = DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=coco_collate
        )
        return train_dl, val_dl


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate detection on real fragments: recall of wavy presence."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_maskrcnn(num_classes=2, pretrained_backbone=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    import csv

    rows = list(csv.DictReader(open(args.eval_csv)))
    frags = [r for r in rows if r["structure_type"] == "fragment"]
    detected = 0
    correct_edge = 0
    for r in frags:
        img = (
            Image.open(r["file_path"].replace("/data/BioChemInsight", str(ROOT)))
            .convert("RGB")
            .resize((384, 384), Image.LANCZOS)
        )
        img_t = TF.to_tensor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img_t)[0]
        scores = out["scores"].cpu().numpy()
        boxes = out["boxes"].cpu().numpy()
        keep = scores > 0.3
        n_det = int(keep.sum())
        if n_det > 0:
            detected += 1
            bx = boxes[keep][0]
            cx = (bx[0] + bx[2]) / 2
            gt_side = "L" if r["SMILES"].startswith("*") else "R"
            pred_side = "L" if cx < 192 else "R"
            if pred_side == gt_side:
                correct_edge += 1
        print(f"  {r['segment']:<16} detections={n_det} max_score={scores.max() if len(scores) else 0:.2f}")
    print(f"\n检出率: {detected}/{len(frags)} = {detected / max(1, len(frags)):.0%}")
    print(f"边缘正确: {correct_edge}/{len(frags)} = {correct_edge / max(1, len(frags)):.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    BaseCocoMaskRCNNTrainer.add_common_args(parser)
    parser.set_defaults(
        data="experiments/wavy_seg/train",
        out="experiments/wavy_seg",
        ckpt="experiments/wavy_seg/wavy_maskrcnn_best.pth",
        eval_csv=str(ROOT / "training/molnextr_markush/data/real_wavy_hard_eval/eval.csv"),
    )
    args = parser.parse_args()
    if args.eval_only:
        evaluate(args)
    else:
        WavyMaskRCNNTrainer(args).train()


if __name__ == "__main__":
    main()
