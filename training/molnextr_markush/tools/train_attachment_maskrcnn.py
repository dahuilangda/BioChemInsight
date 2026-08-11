#!/usr/bin/env python3
"""Stage A: multi-class attachment-point Mask R-CNN trainer.

Trains a single detector for wavy bonds, R-group labels, asterisks, and dashed
bonds (num_classes = 1 bg + 4 fg = 5). This detector's masks drive the
mask-guided graph repair (Stage B) — the DECIMER+MolNexTR organic fusion.

Reuses the architecture/loss of train_wavy_maskrcnn.py; only num_classes and
the dataset source differ. Anchors tuned for small attachment glyphs.

Usage:
    python train_attachment_maskrcnn.py --data experiments/attachment_seg/train --epochs 25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

NUM_CLASSES = 5  # bg + wavy + rgroup + asterisk + dashed
CATEGORY_NAMES = {1: "wavy", 2: "rgroup", 3: "asterisk", 4: "dashed"}


def build_attachment_maskrcnn(num_classes: int = NUM_CLASSES, pretrained_backbone: bool = True) -> MaskRCNN:
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights="DEFAULT" if pretrained_backbone else None,
        trainable_layers=5,
    )
    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),) * 5,
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)
    return MaskRCNN(
        backbone=backbone, num_classes=num_classes,
        rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler, min_size=384, max_size=384,
    )


class AttachmentCocoDataset(Dataset):
    def __init__(self, img_dir: Path, coco: COCO, augment: bool = False):
        self.img_dir = img_dir; self.coco = coco; self.ids = list(coco.imgs.keys()); self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.imgs[img_id]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        masks, boxes, labels = [], [], []
        for ann in anns:
            m = self.coco.annToMask(ann)
            masks.append(m)
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(int(ann["category_id"]))
        if not boxes:
            masks = [np.zeros((img.size[1], img.size[0]), dtype=np.uint8)]
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks_t = torch.zeros((0, img.size[1], img.size[0]), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks_t = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        if self.augment and boxes.shape[0] > 0 and np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT); w = img.size[0]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]; masks_t = torch.flip(masks_t, dims=[2])
        target = {"boxes": boxes, "labels": labels, "masks": masks_t, "image_id": torch.tensor([img_id])}
        return TF.to_tensor(img), target


def collate(batch):
    return tuple(zip(*batch))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data)
    coco = COCO(str(data_dir / "annotations.json"))
    img_ids = list(coco.imgs.keys())
    rng = np.random.RandomState(11)
    rng.shuffle(img_ids)
    n_val = max(1, int(len(img_ids) * args.val_frac))
    val_ids = img_ids[:n_val]
    pos_train = [i for i in img_ids[n_val:] if len(coco.getAnnIds(imgIds=i)) > 0]
    neg_train = [i for i in img_ids[n_val:] if len(coco.getAnnIds(imgIds=i)) == 0]
    train_ids = pos_train + neg_train[: max(1, len(pos_train) // 3)]
    train_ds = AttachmentCocoDataset(data_dir / "images", coco, augment=True); train_ds.ids = train_ids
    val_ds = AttachmentCocoDataset(data_dir / "images", coco, augment=False); val_ds.ids = val_ids
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    model = build_attachment_maskrcnn(NUM_CLASSES, pretrained_backbone=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=8, gamma=0.5)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    best = 1e9
    for epoch in range(args.epochs):
        model.train(); el = 0.0; nb = 0
        for imgs, targets in train_dl:
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            if all(t["boxes"].numel() == 0 for t in targets):
                continue
            loss = sum(model(imgs, targets).values())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
            el += loss.item(); nb += 1
        sched.step()
        avg = el / max(1, nb)
        print(f"epoch {epoch}: train_loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e}")
        if avg < best:
            best = avg
            torch.save(model.state_dict(), Path(args.out) / "attachment_maskrcnn_best.pth")
            print(f"  saved best -> {Path(args.out)/'attachment_maskrcnn_best.pth'}")
    print("done")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_attachment_maskrcnn(NUM_CLASSES, pretrained_backbone=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    import csv
    rows = list(csv.DictReader(open(args.eval_csv)))
    frags = [r for r in rows if r["structure_type"] == "fragment"]
    from collections import Counter
    detected_by_cat = Counter(); gt_by_cat = Counter()
    for r in frags:
        img = Image.open(r["file_path"].replace("/data/BioChemInsight", str(ROOT))).convert("RGB").resize((384, 384), Image.LANCZOS)
        with torch.no_grad():
            out = model(TF.to_tensor(img).unsqueeze(0).to(device))[0]
        scores = out["scores"].cpu().numpy(); labels = out["labels"].cpu().numpy()
        for i in range(len(scores)):
            if scores[i] >= args.score_threshold:
                detected_by_cat[int(labels[i])] += 1
    # every fragment has a wavy (cat 1) by construction in real_wavy_hard
    print(f"真实 fragment 检出 (>{args.score_threshold}): {dict(detected_by_cat)}")
    print(f"类别名: {CATEGORY_NAMES}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="experiments/attachment_seg/train")
    parser.add_argument("--out", default="experiments/attachment_seg")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ckpt", default="experiments/attachment_seg/attachment_maskrcnn_best.pth")
    parser.add_argument("--eval-csv", default=str(ROOT / "training/molnextr_markush/data/real_wavy_hard_eval/eval.csv"))
    parser.add_argument("--score-threshold", type=float, default=0.05)
    args = parser.parse_args()
    if args.eval_only:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
