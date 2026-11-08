"""Label reader head v2: digit-decomposition multi-task CNN (§44.41).

v1 classified 290 whole-label classes (val 95.8%) but memorization dominated
rare classes. v2 decomposes R<n> into three small heads over a shared trunk:
  - magnitude (n >= 10): binary
  - units digit 0-9
  - tens digit 0-9 (0 when n < 10)
Combined number accuracy is the metric; shared glyph features should
generalize to rare classes instead of memorizing whole-label appearances.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelReaderCNNV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        flat = 128 * 4 * 4
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(flat, 256), nn.ReLU(),
                                   nn.Dropout(0.3))
        self.head_magnitude = nn.Linear(256, 2)   # n < 10 vs n >= 10
        self.head_units = nn.Linear(256, 10)      # units digit 0-9
        self.head_tens = nn.Linear(256, 10)       # tens digit 0-9

    def forward(self, x):
        features = self.trunk(self.features(x))
        return {
            "magnitude": self.head_magnitude(features),
            "units": self.head_units(features),
            "tens": self.head_tens(features),
        }


def decompose(number: int):
    tens, units = divmod(int(number), 10)
    return (1 if tens > 0 else 0), units, tens


def load_pairs(labels_path: str, size: int = 64):
    labels = json.loads(Path(labels_path).read_text())
    crops_dir = Path(labels_path).parent / "crops"
    samples = []
    for file_name, number in labels.items():
        image = cv2.imread(str(crops_dir / file_name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, (size, size))
        samples.append((image, int(number)))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.05)
    args = parser.parse_args()

    torch.manual_seed(17)
    random.seed(17)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = [(image, number) for image, number in load_pairs(args.labels)
               if 1 <= number <= 99]
    dropped = len(load_pairs(args.labels)) - len(samples) if False else 0
    if not samples:
        raise SystemExit("no training pairs found")
    random.shuffle(samples)
    val_count = max(1, int(len(samples) * args.val_frac))
    val, train = samples[:val_count], samples[val_count:]
    print(f"samples={len(samples)} train={len(train)} val={len(val)}", flush=True)

    model = LabelReaderCNNV2().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def batch_tensors(batch):
        images = np.stack([
            cv2.resize(image, (64, 64)) for image, _ in batch
        ]).astype(np.float32) / 255.0
        images = torch.from_numpy(images).unsqueeze(1)
        mag = torch.tensor([decompose(n)[0] for _, n in batch])
        uni = torch.tensor([decompose(n)[1] for _, n in batch])
        ten = torch.tensor([decompose(n)[2] for _, n in batch])
        return images, mag, uni, ten

    for epoch in range(args.epochs):
        random.shuffle(train)
        model.train()
        total_loss = seen = correct = 0
        for start in range(0, len(train) - len(train) % args.batch_size,
                           args.batch_size):
            images, mag, uni, ten = batch_tensors(train[start:start + args.batch_size])
            images, mag, uni, ten = (images.to(device), mag.to(device),
                                     uni.to(device), ten.to(device))
            optimizer.zero_grad()
            out = model(images)
            loss = (F.cross_entropy(out["magnitude"], mag)
                    + F.cross_entropy(out["units"], uni)
                    + F.cross_entropy(out["tens"], ten))
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(mag)
            seen += len(mag)
            correct += int(((out["magnitude"].argmax(1) == mag)
                            & (out["units"].argmax(1) == uni)
                            & (out["tens"].argmax(1) == ten)).sum())
        print(f"epoch {epoch}: loss={total_loss/max(1,seen):.4f} "
              f"train_exact={correct/max(1,seen):.3f}", flush=True)
        model.eval()
        vcorrect = vtotal = 0
        with torch.no_grad():
            for start in range(0, len(val), args.batch_size):
                images, mag, uni, ten = batch_tensors(val[start:start + args.batch_size])
                images, mag, uni, ten = (images.to(device), mag.to(device),
                                         uni.to(device), ten.to(device))
                out = model(images)
                vtotal += len(mag)
                vcorrect += int(((out["magnitude"].argmax(1) == mag)
                                 & (out["units"].argmax(1) == uni)
                                 & (out["tens"].argmax(1) == ten)).sum())
        print(f"epoch {epoch}: val_exact={vcorrect}/{len(val)}="
              f"{vcorrect/max(1,len(val)):.3f}", flush=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "arch": "label_reader_cnn_v2",
        "input_size": 64,
    }
    torch.save(checkpoint, args.out)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
