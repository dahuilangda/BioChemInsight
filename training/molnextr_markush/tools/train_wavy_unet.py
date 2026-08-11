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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.src.base_trainer import BaseTrainer  # noqa: E402

IMG_SIZE = 384


class WavySegDataset(Dataset):
    """Image + binary-mask dataset from ``images/`` and ``masks/`` directories."""

    def __init__(self, data_dir: Path, file_list: list[str], augment: bool = False):
        self.images = Path(data_dir) / "images"
        self.masks = Path(data_dir) / "masks"
        self.files = file_list
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(self.images / name).convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.LANCZOS
        )
        mask = Image.open(self.masks / name).convert("L").resize(
            (IMG_SIZE, IMG_SIZE), Image.NEAREST
        )
        img = np.asarray(img, dtype=np.float32) / 255.0
        mask = (np.asarray(mask) > 127).astype(np.float32)
        if self.augment and np.random.rand() < 0.5:
            img = img[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
        if self.augment and np.random.rand() < 0.5:
            img = img[::-1].copy()
            mask = mask[::-1].copy()
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
        self.e1 = resnet.layer1  # /2  64
        self.e2 = resnet.layer2  # /4  128
        self.e3 = resnet.layer3  # /8  256
        self.e4 = resnet.layer4  # /16 512
        self.up4 = self._up(512 + 256, 256)
        self.up3 = self._up(256 + 128, 128)
        self.up2 = self._up(128 + 64, 64)
        self.up1 = self._up(64 + 64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    @staticmethod
    def _up(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        s0 = self.stem(x)
        e1 = self.e1(s0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
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


def bce_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combined BCE + Dice loss for binary segmentation."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    union = prob.sum() + target.sum() + 1e-6
    return bce + 1.0 - 2.0 * inter / union


def split_manifest(manifest: Path, val_frac: float = 0.1) -> tuple[list[str], list[str]]:
    """Split a manifest CSV into (train, val) file lists, balanced by has_wavy."""
    rows = list(csv.DictReader(manifest.open()))
    pos = [r["file"] for r in rows if int(r["has_wavy"]) == 1]
    neg = [r["file"] for r in rows if int(r["has_wavy"]) == 0]
    rng = np.random.RandomState(7)
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_val = max(1, int(len(pos) * val_frac))
    val = pos[:n_val] + neg[: max(1, int(len(neg) * val_frac))]
    train = pos[n_val:] + neg[max(1, int(len(neg) * val_frac)):]
    return train, val


class WavyUNetTrainer(BaseTrainer):
    """U-Net trainer for binary wavy-bond segmentation."""

    def build_model(self) -> nn.Module:
        return UNet(pretrained=True)

    def build_optimizer(self, params: list) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=self.args.lr)

    def build_scheduler(self, optimizer):
        # Original used no scheduler — just Adam with fixed LR
        return None

    def checkpoint_name(self) -> str:
        return "wavy_unet_best.pth"

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader | None]:
        data_dir = Path(self.args.data)
        train_files, val_files = split_manifest(data_dir / "manifest.csv", self.args.val_frac)
        train_dl = DataLoader(
            WavySegDataset(data_dir, train_files, augment=True),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
        )
        val_dl = DataLoader(
            WavySegDataset(data_dir, val_files, augment=False),
            batch_size=self.args.batch_size,
            num_workers=2,
        )
        return train_dl, val_dl

    def compute_loss(self, model: nn.Module, batch: tuple) -> torch.Tensor:
        img, mask, _ = batch
        return bce_dice(model(img.to(self.device)), mask.to(self.device))

    def validate(self, model: nn.Module, val_dl: DataLoader | None) -> float:
        if val_dl is None:
            return 1e9
        model.eval()
        vloss = 0.0
        viou = 0.0
        vn = 0
        with torch.no_grad():
            for img, mask, _ in val_dl:
                img, mask = img.to(self.device), mask.to(self.device)
                logits = model(img)
                vloss += bce_dice(logits, mask).item()
                prob = torch.sigmoid(logits)
                inter = ((prob > 0.5) & (mask > 0.5)).sum().item()
                uni = ((prob > 0.5) | (mask > 0.5)).sum().item() + 1e-6
                viou += inter / uni
                vn += 1
        model.train()
        self._last_val_iou = viou / max(1, vn)
        return vloss / max(1, len(val_dl))

    def epoch_suffix(self) -> str:
        """Report validation IoU alongside the val_metric in the epoch log."""
        iou = getattr(self, "_last_val_iou", None)
        return f" val_iou={iou:.4f}" if iou is not None else ""
        return vloss / max(1, len(val_dl))


def _load_real_for_eval(path: str) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    )


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate on real patent fragments: localization accuracy by edge."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    rows = list(csv.DictReader(open(args.eval_csv)))
    frags = [r for r in rows if r["structure_type"] == "fragment"]
    correct = 0
    total = 0
    print(f"Evaluating on {len(frags)} real fragments...")
    for r in frags:
        img = _load_real_for_eval(r["file_path"].replace("/data/BioChemInsight", str(ROOT)))
        with torch.no_grad():
            prob = torch.sigmoid(
                model(
                    torch.from_numpy(img)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    .to(device)
                    / 255.0
                )[0, 0]
            ).cpu().numpy()
        pred_mask = prob > 0.3
        ys, xs = np.where(pred_mask)
        gt_side = "L" if r["SMILES"].startswith("*") else "R"
        if len(xs) == 0:
            pred_side = "none"
        else:
            h, w = pred_mask.shape
            left = pred_mask[:, : w // 4].sum()
            right = pred_mask[:, 3 * w // 4 :].sum()
            top = pred_mask[: h // 4, :].sum()
            bot = pred_mask[3 * h // 4 :, :].sum()
            scores = {"L": left, "R": right, "T": top, "B": bot}
            pred_side = max(scores, key=scores.get) if max(scores.values()) > 0 else "none"
            pred_side = pred_side[0]  # L/R/T/B
        ok = pred_side == gt_side
        correct += ok
        total += 1
        print(
            f"  {r['segment']:<16} gt={gt_side} pred={pred_side} "
            f"ink={int(pred_mask.sum()):>5} {'OK' if ok else 'XX'}"
        )
    print(f"\n定位准确率: {correct}/{total} = {correct / max(1, total):.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    BaseTrainer.add_common_args(parser)
    parser.set_defaults(
        data="experiments/wavy_seg/train",
        out="experiments/wavy_seg",
        epochs=8,
        batch_size=16,
        lr=1e-4,
        ckpt="experiments/wavy_seg/wavy_unet_best.pth",
        eval_csv=str(ROOT / "training/molnextr_markush/data/real_wavy_hard_eval/eval.csv"),
    )
    args = parser.parse_args()
    if args.eval_only:
        evaluate(args)
    else:
        WavyUNetTrainer(args).train()


if __name__ == "__main__":
    main()
