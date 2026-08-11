#!/usr/bin/env python3
"""Phase 1: Mask R-CNN wavy-bond detector trainer.

A dedicated instance-segmentation model for wavy attachment bonds, reusing the
DECIMER Mask R-CNN's ResNet101+FPN backbone as a pretrained feature extractor
but with standard torchvision detection heads (trainable, no Matterport weight
compatibility baggage). This is the production detector (Phase 0 validated that
synthetic->real transfer works; Phase 1 gives it object-level bbox+mask
discrimination the U-Net lacked).

Pipeline: synthetic COCO data (gen_wavy_seg_data.py --coco) -> Mask R-CNN
(num_classes=2: background + wavy) -> evaluate on real_wavy_hard fragments.

Usage:
    python train_wavy_maskrcnn.py --data experiments/wavy_seg/train --epochs 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def build_wavy_maskrcnn(num_classes: int = 2, pretrained_backbone: bool = True) -> MaskRCNN:
    """Mask R-CNN with resnet50+FPN backbone (lighter than DECIMER's resnet101,
    sufficient for the small wavy object). Backbone optionally warm-started from
    ImageNet; heads trained from scratch. num_classes=2 (bg + wavy)."""
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights="DEFAULT" if pretrained_backbone else None,
        trainable_layers=5,
    )
    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),  # wavy is small (~20px)
        aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),) * 5,  # wavy is elongated
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
    )
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2
    )
    model = MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler,
        min_size=384,
        max_size=384,
    )
    return model


class WavyCocoDataset(Dataset):
    def __init__(self, img_dir: Path, coco: COCO, augment: bool = False):
        self.img_dir = img_dir
        self.coco = coco
        # Start with all image ids; the train/eval split reassigns self.ids.
        self.ids = list(coco.imgs.keys())
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.imgs[img_id]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        masks = []
        boxes = []
        labels = []
        for ann in anns:
            m = self.coco.annToMask(ann)
            masks.append(m)
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(1)
        if not boxes:
            # negative sample: single dummy "background-only" target
            masks = [np.zeros((img.size[1], img.size[0]), dtype=np.uint8)]
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks_t = torch.zeros((0, img.size[1], img.size[0]), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks_t = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        if self.augment and boxes.shape[0] > 0 and np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w = img.size[0]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            masks_t = torch.flip(masks_t, dims=[2])
        img_t = TF.to_tensor(img)
        target = {"boxes": boxes, "labels": labels, "masks": masks_t,
                  "image_id": torch.tensor([img_id])}
        return img_t, target


def collate(batch):
    return tuple(zip(*batch))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data)
    coco = COCO(str(data_dir / "annotations.json"))
    img_ids = list(coco.imgs.keys())
    rng = np.random.RandomState(7)
    rng.shuffle(img_ids)
    n_val = max(1, int(len(img_ids) * args.val_frac))
    val_ids = img_ids[:n_val]
    train_ids = [i for i in img_ids[n_val:] if len(coco.getAnnIds(imgIds=i)) > 0]
    # add negatives to training for false-positive suppression
    neg_ids = [i for i in img_ids[n_val:] if len(coco.getAnnIds(imgIds=i)) == 0]
    train_ids = train_ids + neg_ids[: max(1, len(train_ids) // 3)]

    train_ds = WavyCocoDataset(data_dir / "images", coco, augment=True)
    train_ds.ids = train_ids
    val_ds = WavyCocoDataset(data_dir / "images", coco, augment=False)
    val_ds.ids = val_ids
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate)

    model = build_wavy_maskrcnn(num_classes=2, pretrained_backbone=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=6, gamma=0.5)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    best_loss = 1e9
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0; nb = 0
        for imgs, targets in train_dl:
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # skip degenerate all-empty batches (no positive boxes)
            if all(t["boxes"].numel() == 0 for t in targets):
                continue
            loss_dict = model(imgs, targets)
            loss = sum(loss for loss in loss_dict.values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            epoch_loss += loss.item(); nb += 1
        scheduler.step()
        avg = epoch_loss / max(1, nb)
        print(f"epoch {epoch}: train_loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), Path(args.out) / "wavy_maskrcnn_best.pth")
            print(f"  saved best -> {Path(args.out)/'wavy_maskrcnn_best.pth'}")
    print("done")


def evaluate(args):
    """Evaluate detection on real fragments: recall of wavy presence."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_wavy_maskrcnn(num_classes=2, pretrained_backbone=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    import csv
    rows = list(csv.DictReader(open(args.eval_csv)))
    frags = [r for r in rows if r["structure_type"] == "fragment"]
    detected = 0; correct_edge = 0
    for r in frags:
        img = Image.open(r["file_path"].replace("/data/BioChemInsight", str(ROOT))).convert("RGB").resize((384, 384), Image.LANCZOS)
        img_t = TF.to_tensor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img_t)[0]
        scores = out["scores"].cpu().numpy()
        masks = out["masks"].cpu().numpy()[:, 0]
        boxes = out["boxes"].cpu().numpy()
        keep = scores > 0.3
        n_det = int(keep.sum())
        if n_det > 0:
            detected += 1
            # predicted edge from the highest-score detection center
            bx = boxes[keep][0]
            cx = (bx[0] + bx[2]) / 2
            gt_side = "L" if r["SMILES"].startswith("*") else "R"
            pred_side = "L" if cx < 192 else "R"
            if pred_side == gt_side:
                correct_edge += 1
        print(f"  {r['segment']:<16} detections={n_det} max_score={scores.max() if len(scores) else 0:.2f}")
    print(f"\n检出率: {detected}/{len(frags)} = {detected/max(1,len(frags)):.0%}")
    print(f"边缘正确: {correct_edge}/{len(frags)} = {correct_edge/max(1,len(frags)):.0%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="experiments/wavy_seg/train")
    parser.add_argument("--out", default="experiments/wavy_seg")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ckpt", default="experiments/wavy_seg/wavy_maskrcnn_best.pth")
    parser.add_argument("--eval-csv", default=str(ROOT / "training/molnextr_markush/data/real_wavy_hard_eval/eval.csv"))
    args = parser.parse_args()
    if args.eval_only:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
