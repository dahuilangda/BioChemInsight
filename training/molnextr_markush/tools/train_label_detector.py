"""Train the R-group label position detector (§44.28, position mapping).

Small heatmap head over the structure crop: Gaussian peaks at label centers
(the cells-bbox supervision), local maxima at inference are label positions.
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

INPUT = 256      # network input size
STRIDE = 8       # heatmap downsample (3 max-pools)
HEATMAP = INPUT // STRIDE


class LabelDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x):
        return self.features(x)


def load_samples(centers_path: str, images_dir: str, limit: int = 0):
    centers = json.loads(Path(centers_path).read_text())
    samples = []
    for file_name, points in centers.items():
        path = Path(images_dir) / file_name
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        samples.append((file_name, image, points))
        if limit and len(samples) >= limit:
            break
    return samples


def make_target(points):
    heat = np.zeros((HEATMAP, HEATMAP), dtype=np.float32)
    radius = 2
    for px, py, _ in points:
        cx = int(px * HEATMAP)
        cy = int(py * HEATMAP)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < HEATMAP and 0 <= y < HEATMAP:
                    heat[y, x] = max(heat[y, x],
                                     float(np.exp(-(dx * dx + dy * dy) / (2 * 1.5 ** 2))))
    return heat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--centers", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.05)
    args = parser.parse_args()

    torch.manual_seed(17)
    random.seed(17)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = load_samples(args.centers, args.images_dir, args.limit)
    random.shuffle(samples)
    val_count = max(1, int(len(samples) * args.val_frac))
    val, train = samples[:val_count], samples[val_count:]
    print(f"samples={len(samples)} train={len(train)} val={len(val)}", flush=True)

    model = LabelDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def batch_tensors(batch):
        images = np.stack([
            cv2.resize(image, (INPUT, INPUT)).astype(np.float32) / 255.0
            for _, image, _ in batch
        ])
        images = torch.from_numpy(images).unsqueeze(1)
        targets = torch.from_numpy(np.stack([make_target(p) for _, _, p in batch]))
        return images, targets

    for epoch in range(args.epochs):
        random.shuffle(train)
        model.train()
        total = seen = 0
        for start in range(0, len(train), args.batch_size):
            images, targets = batch_tensors(train[start:start + args.batch_size])
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            heat = model(images)
            loss = F.mse_loss(heat, targets)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(targets)
            seen += len(targets)
        print(f"epoch {epoch}: mse={total/max(1,seen):.4f}", flush=True)

        # peak recall: fraction of val images where every gold peak has a
        # detected peak within 3 heatmap cells
        model.eval()
        recall_hits = recall_total = 0
        with torch.no_grad():
            for start in range(0, len(val), args.batch_size):
                images, targets = batch_tensors(val[start:start + args.batch_size])
                pred = model(images.to(device)).cpu()
                for b in range(len(val[start:start + args.batch_size])):
                    gold_points = val[start + b][2]
                    heat = pred[b, 0].numpy()
                    peaks = []
                    thresh = heat.max() * 0.3 if heat.max() > 0 else 0
                    while True:
                        if heat.max() <= thresh or heat.max() <= 0:
                            break
                        y, x = np.unravel_index(heat.argmax(), heat.shape)
                        peaks.append((x, y))
                        y0, y1 = max(0, y - 3), min(HEATMAP, y + 4)
                        x0, x1 = max(0, x - 3), min(HEATMAP, x + 4)
                        heat[y0:y1, x0:x1] = 0
                    matched = 0
                    for px, py, _ in gold_points:
                        gx, gy = int(px * HEATMAP), int(py * HEATMAP)
                        if any(abs(px_ - gx) <= 3 and abs(py_ - gy) <= 3
                               for px_, py_ in peaks):
                            matched += 1
                    recall_total += len(gold_points)
                    recall_hits += matched
        print(f"epoch {epoch}: peak_recall={recall_hits}/{recall_total}="
              f"{recall_hits/max(1,recall_total):.3f}", flush=True)

    torch.save({"state_dict": model.state_dict()}, args.out)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
