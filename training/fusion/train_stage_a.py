#!/usr/bin/env python
"""Stage-A training: freeze Swin, train detection head + RoI projection.

Data: MG2 page-level (one structure per page, box ~= full page). This
bootstraps the fused model's detection branch before any recognition
fine-tuning (design §5 Stage A; Frozen-DETR memory lesson: frozen backbone
keeps 2 img/GPU well inside 16GB).

Usage:
  python training/fusion/train_stage_a.py --epochs 1 --limit 0 \
      --output experiments/fusion/stage_a
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from utils.fusion.fusion_model import FusionConfig, FusedStructureRecognizer
from utils.fusion.page_dataset import MG2PageDataset


def collate(batch):
    images = torch.stack([torch.from_numpy(b["image_resized"]) for b in batch])
    gt_boxes = [torch.from_numpy(b["boxes"]) for b in batch]
    return images, gt_boxes


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResizedPages(torch.utils.data.Dataset):
    """Pages preprocessed exactly like the two-stage crop path.

    get_transforms (dataset.py) does ToGray + ImageNet normalize + Resize;
    the encoder therefore expects a normalized *grayscale* replicated to
    three channels, laid out CHW.
    """

    def __init__(self, inner: MG2PageDataset, size: int = 384) -> None:
        self.inner = inner
        self.size = size

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int):
        import cv2
        import numpy as np

        item = self.inner[index]
        resized = cv2.resize(item["image"], (self.size, self.size),
                             interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        gray = resized.mean(axis=2, keepdims=True)          # ToGray
        normalized = (gray - sum(IMAGENET_MEAN) / 3) / (sum(IMAGENET_STD) / 3)
        chw = np.repeat(normalized, 3, axis=2).transpose(2, 0, 1)  # CHW, 3ch
        return {"image_resized": chw, "boxes": item["boxes"]}


def iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    return inter / (area_a + area_b - inter + 1e-6)


@torch.no_grad()
def evaluate_f1(model, loader, device, iou_threshold=0.5, max_batches=50):
    """Detection precision/recall/F1@IoU vs GT + no-detection (cascade) rate."""
    model.eval()
    tp = fp = fn = 0
    pages = no_detection = 0
    for batch_index, (images, gt_boxes) in enumerate(loader):
        if batch_index >= max_batches:
            break
        images = images.to(device)
        features, hiddens = model.encoder(images)
        pred_boxes, _, _, _ = model.detect(hiddens)
        for pred, gt in zip(pred_boxes, gt_boxes):
            pages += 1
            pred = pred.to(device).float()
            gt = gt.to(device).float()
            if pred.numel() == 0:
                no_detection += gt.shape[0]
                fn += gt.shape[0]
                continue
            ious = iou_xyxy(gt[:, None, :], pred[None, :, :])  # (G, P)
            matched_gt = set()
            for g in range(gt.shape[0]):
                best = ious[g].argmax()
                if ious[g, best] >= iou_threshold and best not in matched_gt:
                    matched_gt.add(int(best))
                    tp += 1
                else:
                    fn += 1
            fp += pred.shape[0] - len(matched_gt)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    return {
        "pages": pages, "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "no_detection_pages_ratio": round(no_detection / max(1, pages), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="cap pages (0 = all 54k)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--holdout", type=int, default=512, help="tail pages held out for F1")
    parser.add_argument("--output", default="experiments/fusion/stage_a")
    parser.add_argument("--data", choices=["mg2", "synth", "crops", "mix"], default="mg2",
                        help="mg2 = single-structure pages; synth = multi-structure pastes (P2)")
    parser.add_argument("--pages", type=int, default=20000, help="synthetic page count")
    args = parser.parse_args()

    from utils.structure_recognition import _load_molnextr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    mol = _load_molnextr(None, moe_config_path="experiments/moe/production/moe_config.json")
    model = FusedStructureRecognizer(mol.encoder, mol.moe_decoder,
                                     FusionConfig(stage="a")).to(device)
    # freeze encoder + decoder; only the detection head + RoI projection train
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    covered = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any("_head" in n for n in covered) and any("_collector" in n for n in covered), covered

    if args.data == "crops" or args.data == "mix":
        import csv as _csv
        import os as _os
        import random as _random

        _csv.field_size_limit(10 ** 9)
        pool = []
        with open(ROOT / "training/molnextr_markush/data/raw/molscribe/uspto_mol/train_200k.csv") as fh:
            for _row in _csv.DictReader(fh):
                _smi = (_row.get("SMILES") or "").strip()
                _rel = (_row.get("file_path") or "").strip()
                if _smi and _rel and "*" not in _smi and "[R" not in _smi:
                    if _os.path.exists(ROOT / "training/molnextr_markush/data/raw/molscribe" / _rel):
                        pool.append((ROOT / "training/molnextr_markush/data/raw/molscribe" / _rel, _smi))
        _random.Random(5).shuffle(pool)

        class _CropPages(torch.utils.data.Dataset):
            """USPTO TIF crops treated as single-structure pages (box ~= full)."""

            def __init__(self, items, margin=0.04):
                self.items = items
                self.margin = margin

            def __len__(self):
                return len(self.items)

            def __getitem__(self, index):
                import cv2 as _cv2

                path, _ = self.items[index]
                img = _cv2.imread(str(path))
                if img is None:
                    img = _np().zeros((384, 384, 3), dtype=_np().uint8)
                boxes = _np().array([[self.margin, self.margin,
                                      1 - self.margin, 1 - self.margin]],
                                    dtype=_np().float32)
                return {"image": img[..., ::-1], "boxes": boxes}  # BGR->RGB

        def _np():
            import numpy as __np
            return __np

        if args.data == "crops":
            total = (args.limit + args.holdout) if args.limit else (args.pages + args.holdout)
            full = _CropPages(pool[:total])
        else:  # mix: half crops, half synth
            synth_total = (args.pages + args.holdout) // 2
            from utils.fusion.synth_pages import (
                SynthPageConfig, SyntheticMultiStructureDataset,
            )

            synth_full = SyntheticMultiStructureDataset(
                ROOT / "training/molnextr_markush/data/raw/molscribe/uspto_mol/train_200k.csv",
                ROOT / "training/molnextr_markush/data/raw/molscribe",
                pages=synth_total, config=SynthPageConfig(seed=31),
            )
            crop_total = synth_total
            crops_ds = _CropPages(pool[:crop_total])
            full = torch.utils.data.ConcatDataset([synth_full, crops_ds])
    elif args.data == "synth":
        from utils.fusion.synth_pages import (
            SynthPageConfig, SyntheticMultiStructureDataset,
        )

        total = args.pages + args.holdout
        full = SyntheticMultiStructureDataset(
            ROOT / "training/molnextr_markush/data/raw/molscribe/uspto_mol/train_200k.csv",
            ROOT / "training/molnextr_markush/data/raw/molscribe",
            pages=total, config=SynthPageConfig(seed=11),
        )
    else:
        full = MG2PageDataset(ROOT / "training/molnextr_markush/data/raw/markushgrapher2",
                              limit=(args.limit + args.holdout) if args.limit else 0)
    train_inner = torch.utils.data.Subset(full, range(0, len(full) - args.holdout))
    eval_inner = torch.utils.data.Subset(full, range(len(full) - args.holdout, len(full)))
    train_loader = DataLoader(ResizedPages(train_inner), batch_size=args.batch_size,
                              shuffle=True, num_workers=4, collate_fn=collate)
    eval_loader = DataLoader(ResizedPages(eval_inner), batch_size=args.batch_size,
                             shuffle=False, num_workers=2, collate_fn=collate)
    print(f"stage A [{args.data}]: {len(train_inner)} train pages, {len(eval_inner)} holdout, "
          f"trainable tensors={len(trainable)}", flush=True)

    metrics = evaluate_f1(model, eval_loader, device, max_batches=8)
    print(f"pre-training metrics: {json.dumps(metrics)}", flush=True)

    model.train()
    step = 0
    started = time.time()
    for epoch in range(args.epochs):
        for images, gt_boxes in train_loader:
            images = images.to(device)
            gt_boxes = [b.to(device) for b in gt_boxes]
            optimizer.zero_grad(set_to_none=True)
            # fp32 end-to-end: bf16 autocast drove the detection head to NaN
            # by step ~50-100 on full data (fp32 loop verified clean for 60+
            # steps, loss 0.90 -> 0.21); the frozen-encoder forward is cheap
            # enough that the precision is worth the speed.
            output = model(images, refs={}, gt_boxes=gt_boxes)
            loss = output["losses"]["loss_det"]
            loss.backward()
            optimizer.step()
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss_det={float(loss):.4f} "
                      f"({time.time() - started:.0f}s)", flush=True)
            if step % args.eval_every == 0:
                metrics = evaluate_f1(model, eval_loader, device)
                print(f"epoch {epoch} step {step} METRICS {json.dumps(metrics)}", flush=True)
                model.train()

    metrics = evaluate_f1(model, eval_loader, device)
    torch.save({"head": model._head.state_dict(),
                "collector": model._collector.state_dict()},
               out_dir / "stage_a_heads.pth")
    (out_dir / "metrics.json").write_text(json.dumps({
        "pre_train_hint": "see log", "final": metrics,
        "steps": step, "epochs": args.epochs,
    }, indent=2))
    print(f"FINAL METRICS {json.dumps(metrics)}", flush=True)
    print(f"saved heads -> {out_dir / 'stage_a_heads.pth'}", flush=True)


if __name__ == "__main__":
    main()
