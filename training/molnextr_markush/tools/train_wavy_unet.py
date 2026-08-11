#!/usr/bin/env python3
"""Phase 0: lightweight U-Net wavy-bond segmentation trainer.

A fast, deliberately small segmentation model to validate the core hypothesis:
can a detector trained on SYNTHETIC wavy bonds localize REAL patent wavy bonds?
This is a gate — if this passes, we invest in the full Mask R-CNN (Phase 1).

Architecture: U-Net with a ResNet18 encoder (torchvision, ImageNet-pretrained),
single binary output channel (wavy vs not-wavy). Trained with BCE+Dice loss on
the synthetic dataset, then evaluated on the real real_wavy_hard fragments.

Usage:
    python train_wavy_unet.py --data experiments/wavy_seg/train --epochs 8
    python train_wavy_unet.py --eval-only --ckpt <path> --eval-csv <real.csv>
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

IMG_SIZE = 384


class WavySegDataset(Dataset):
    def __init__(self, data_dir: Path, file_list: list[str], augment: bool = False):
        self.images = data_dir / "images"
        self.masks = data_dir / "masks"
        self.files = file_list
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(self.images / name).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        mask = Image.open(self.masks / name).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        img = np.asarray(img, dtype=np.float32) / 255.0
        mask = (np.asarray(mask) > 127).astype(np.float32)
        if self.augment and np.random.rand() < 0.5:
            img = img[:, ::-1].copy(); mask = mask[:, ::-1].copy()
        if self.augment and np.random.rand() < 0.5:
            img = img[::-1].copy(); mask = mask[::-1].copy()
        # add mild blur/jitter
        if self.augment:
            img = img + np.random.normal(0, 0.01, img.shape).astype(np.float32)
            img = np.clip(img, 0, 1)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return img, mask, name


class UNet(nn.Module):
    """U-Net with a torchvision ResNet18 encoder."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # /2
        self.pool = nn.MaxPool2d(2)
        self.e1 = resnet.layer1   # /2  64
        self.e2 = resnet.layer2   # /4  128
        self.e3 = resnet.layer3   # /8  256
        self.e4 = resnet.layer4   # /16 512
        self.up4 = self._up(512 + 256, 256)   # e4(512) + e3(256)
        self.up3 = self._up(256 + 128, 128)   # d4(256) + e2(128)
        self.up2 = self._up(128 + 64, 64)     # d3(128) + e1(64)
        self.up1 = self._up(64 + 64, 32)      # d2(64) + s0(64)
        self.out = nn.Conv2d(32, 1, 1)

    @staticmethod
    def _up(in_c, out_c):
        # Operates at the encoder-skip resolution (bilinear upsampling done
        # outside); two convs refine the concatenated features.
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        s0 = self.stem(x)        # /2 64
        e1 = self.e1(s0)         # /2 64
        e2 = self.e2(e1)         # /4 128
        e3 = self.e3(e2)         # /8 256
        e4 = self.e4(e3)         # /16 512
        # up4: e4(/16) -> /8, concat e3
        u = F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.up4(torch.cat([u, e3], 1))
        u = F.interpolate(d4, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.up3(torch.cat([u, e2], 1))
        u = F.interpolate(d3, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.up2(torch.cat([u, e1], 1))
        u = F.interpolate(d2, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.up1(torch.cat([u, s0], 1))
        out = F.interpolate(self.out(d1), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


def bce_dice(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    union = prob.sum() + target.sum() + 1e-6
    return bce + 1.0 - 2.0 * inter / union


def split_manifest(manifest: Path, val_frac: float = 0.1):
    rows = list(csv.DictReader(manifest.open()))
    pos = [r["file"] for r in rows if int(r["has_wavy"]) == 1]
    neg = [r["file"] for r in rows if int(r["has_wavy"]) == 0]
    rng = np.random.RandomState(7)
    rng.shuffle(pos); rng.shuffle(neg)
    n_val = max(1, int(len(pos) * val_frac))
    val = pos[:n_val] + neg[: max(1, int(len(neg) * val_frac))]
    train = pos[n_val:] + neg[max(1, int(len(neg) * val_frac)):]
    return train, val


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data)
    train_files, val_files = split_manifest(data_dir / "manifest.csv", args.val_frac)
    train_dl = DataLoader(WavySegDataset(data_dir, train_files, augment=True), batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(WavySegDataset(data_dir, val_files, augment=False), batch_size=args.batch_size, num_workers=2)
    model = UNet(pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = 1e9
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train(); tot = 0.0
        for img, mask, _ in train_dl:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad()
            loss = bce_dice(model(img), mask)
            loss.backward(); opt.step()
            tot += loss.item()
        # val
        model.eval(); vloss = 0.0; viou = 0.0; vn = 0
        with torch.no_grad():
            for img, mask, _ in val_dl:
                img, mask = img.to(device), mask.to(device)
                logits = model(img)
                vloss += bce_dice(logits, mask).item()
                prob = torch.sigmoid(logits)
                inter = ((prob > 0.5) & (mask > 0.5)).sum().item()
                uni = ((prob > 0.5) | (mask > 0.5)).sum().item() + 1e-6
                viou += inter / uni; vn += 1
        print(f"epoch {epoch}: train_loss={tot/len(train_dl):.4f} val_loss={vloss/max(1,len(val_dl)):.4f} val_iou={viou/max(1,vn):.4f}")
        if vloss < best_val:
            best_val = vloss
            torch.save(model.state_dict(), Path(args.out) / "wavy_unet_best.pth")
            print(f"  saved best -> {Path(args.out)/'wavy_unet_best.pth'}")
    print("done")


def _load_real_for_eval(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS))


def evaluate(args):
    """Evaluate on real patent fragments: localization accuracy by edge."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    rows = list(csv.DictReader(open(args.eval_csv)))
    frags = [r for r in rows if r["structure_type"] == "fragment"]
    correct = 0; total = 0
    print(f"Evaluating on {len(frags)} real fragments...")
    for r in frags:
        img = _load_real_for_eval(r["file_path"].replace("/data/BioChemInsight", str(ROOT)))
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0))[0, 0].cpu().numpy()
        pred_mask = prob > 0.3
        ys, xs = np.where(pred_mask)
        gt_side = "L" if r["SMILES"].startswith("*") else "R"
        if len(xs) == 0:
            pred_side = "none"
        else:
            # predicted edge = side with most predicted wavy ink
            H, W = pred_mask.shape
            left = pred_mask[:, : W // 4].sum()
            right = pred_mask[:, 3 * W // 4 :].sum()
            top = pred_mask[: H // 4, :].sum()
            bot = pred_mask[3 * H // 4 :, :].sum()
            scores = {"L": left, "R": right, "T": top, "B": bot}
            pred_side = max(scores, key=scores.get) if max(scores.values()) > 0 else "none"
            pred_side = pred_side[0]  # L/R/T/B
        ok = pred_side == gt_side
        correct += ok; total += 1
        print(f"  {r['segment']:<16} gt={gt_side} pred={pred_side} ink={int(pred_mask.sum()):>5} {'OK' if ok else 'XX'}")
    print(f"\n定位准确率: {correct}/{total} = {correct/max(1,total):.0%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="experiments/wavy_seg/train")
    parser.add_argument("--out", default="experiments/wavy_seg")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ckpt", default="experiments/wavy_seg/wavy_unet_best.pth")
    parser.add_argument("--eval-csv", default=str(ROOT / "training/molnextr_markush/data/real_wavy_hard_eval/eval.csv"))
    args = parser.parse_args()
    if args.eval_only:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
