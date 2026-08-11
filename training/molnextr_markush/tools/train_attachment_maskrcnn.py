#!/usr/bin/env python3
"""Stage A: multi-class attachment-point Mask R-CNN trainer.

Trains a single detector for wavy bonds, R-group labels, asterisks, and dashed
bonds (num_classes = 1 bg + 4 fg = 5). This detector's masks drive the
mask-guided graph repair (Stage B) — the DECIMER+MolNexTR organic fusion.

Reuses the architecture/loss of train_wavy_maskrcnn.py via the shared
``BaseCocoMaskRCNNTrainer``; only num_classes and the dataset split differ.
Anchors tuned for small attachment glyphs.

Usage:
    python train_attachment_maskrcnn.py --data experiments/attachment_seg/train --epochs 25
    python train_attachment_maskrcnn.py --eval-only --ckpt <path> --eval-csv <real.csv>
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

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

import numpy as np  # noqa: E402

NUM_CLASSES = 5  # bg + wavy + rgroup + asterisk + dashed
CATEGORY_NAMES = {1: "wavy", 2: "rgroup", 3: "asterisk", 4: "dashed"}


class AttachmentMaskRCNNTrainer(BaseCocoMaskRCNNTrainer):
    """Multi-class Mask R-CNN trainer for attachment-point detection."""

    NUM_CLASSES = NUM_CLASSES
    CHECKPOINT_NAME = "attachment_maskrcnn_best.pth"

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader | None]:
        data_dir = Path(self.args.data)
        coco = COCO(str(data_dir / "annotations.json"))
        img_ids = list(coco.imgs.keys())
        rng = np.random.RandomState(11)
        rng.shuffle(img_ids)
        n_val = max(1, int(len(img_ids) * self.args.val_frac))
        val_ids = img_ids[:n_val]
        pos_train = [i for i in img_ids[n_val:] if len(coco.getAnnIds(imgIds=i)) > 0]
        neg_train = [i for i in img_ids[n_val:] if len(coco.getAnnIds(imgIds=i)) == 0]
        train_ids = pos_train + neg_train[: max(1, len(pos_train) // 3)]

        train_ds = CocoDetectionDataset(data_dir / "images", coco, augment=True)
        train_ds.ids = train_ids
        val_ds = CocoDetectionDataset(data_dir / "images", coco, augment=False)
        val_ds.ids = val_ids
        train_dl = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=coco_collate,
        )
        val_dl = DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=coco_collate
        )
        return train_dl, val_dl


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate multi-class detection on real fragments."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_maskrcnn(num_classes=NUM_CLASSES, pretrained_backbone=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    import csv

    rows = list(csv.DictReader(open(args.eval_csv)))
    frags = [r for r in rows if r["structure_type"] == "fragment"]
    detected_by_cat: Counter = Counter()
    for r in frags:
        img = (
            Image.open(r["file_path"].replace("/data/BioChemInsight", str(ROOT)))
            .convert("RGB")
            .resize((384, 384), Image.LANCZOS)
        )
        with torch.no_grad():
            out = model(TF.to_tensor(img).unsqueeze(0).to(device))[0]
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        for i in range(len(scores)):
            if scores[i] >= args.score_threshold:
                detected_by_cat[int(labels[i])] += 1
    print(f"真实 fragment 检出 (>={args.score_threshold}): {dict(detected_by_cat)}")
    print(f"类别名: {CATEGORY_NAMES}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    BaseCocoMaskRCNNTrainer.add_common_args(parser)
    parser.set_defaults(
        data="experiments/attachment_seg/train",
        out="experiments/attachment_seg",
        ckpt="experiments/attachment_seg/attachment_maskrcnn_best.pth",
        eval_csv=str(ROOT / "training/molnextr_markush/data/real_wavy_hard_eval/eval.csv"),
    )
    args = parser.parse_args()
    if args.eval_only:
        evaluate(args)
    else:
        AttachmentMaskRCNNTrainer(args).train()


if __name__ == "__main__":
    main()
