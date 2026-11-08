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
    for point in points:
        px, py = point[0], point[1]
        cx = int(px * HEATMAP)
        cy = int(py * HEATMAP)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < HEATMAP and 0 <= y < HEATMAP:
                    heat[y, x] = max(heat[y, x],
                                     float(np.exp(-(dx * dx + dy * dy) / (2 * 1.5 ** 2))))
    return heat


def augment_sample(image, points):
    """Domain-alignment augmentation (§44.29): random loose/tight padding,
    scale jitter, and mild blur — bridging the USPTO full-page domain and
    the eval tight-crop domain."""
    height, width = image.shape[:2]
    pad_frac = random.uniform(0.0, 0.6)
    pad_x, pad_y = int(width * pad_frac), int(height * pad_frac)
    padded = np.full((height + 2 * pad_y, width + 2 * pad_x), 255, dtype=np.uint8)
    padded[pad_y:pad_y + height, pad_x:pad_x + width] = image
    scale = random.uniform(0.5, 1.6)
    resized = cv2.resize(padded, (max(32, int(padded.shape[1] * scale)),
                                  max(32, int(padded.shape[0] * scale))))
    if random.random() < 0.3:
        resized = cv2.GaussianBlur(resized, (3, 3), 0)
    new_h, new_w = resized.shape[:2]
    # px/py were normalized to the UNpadded crop; the padded normalized
    # position is (pad + px*old_size)/new_size.
    new_points = [
        [(pad_x + point[0] * width) / new_w,
         (pad_y + point[1] * height) / new_h]
        for point in points
    ]
    return resized, new_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--centers", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--pseudo-centers", default="",
                        help="target-domain pseudo centers json (self-training)")
    parser.add_argument("--pseudo-images-dir", default="")
    parser.add_argument("--pseudo-weight", type=int, default=2,
                        help="times each pseudo sample repeats in the train list")
    args = parser.parse_args()

    torch.manual_seed(17)
    random.seed(17)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = load_samples(args.centers, args.images_dir, args.limit)
    # audit §44.34: split by patent page (image base) to prevent page-group
    # leakage inflating val recall.
    groups = {}
    for sample in samples:
        base = sample[0].rsplit('_', 2)[0]
        groups.setdefault(base, []).append(sample)
    group_items = list(groups.values())
    random.shuffle(group_items)
    val_count = max(1, int(len(group_items) * args.val_frac))
    val = [s for g in group_items[:val_count] for s in g]
    train = [s for g in group_items[val_count:] for s in g]
    if args.pseudo_centers:
        pseudo = load_samples(args.pseudo_centers, args.pseudo_images_dir)
        train = train + pseudo * max(1, args.pseudo_weight)
        print(f"pseudo samples added: {len(pseudo)} x{args.pseudo_weight}", flush=True)
    print(f"samples={len(samples)} train={len(train)} val={len(val)}", flush=True)

    model = LabelDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def batch_tensors(batch, do_augment=False):
        images = []
        points_list = []
        for file_name, image, points in batch:
            if do_augment:
                image, points = augment_sample(image, points)
            images.append(cv2.resize(image, (INPUT, INPUT)).astype(np.float32) / 255.0)
            points_list.append(points)
        images = torch.from_numpy(np.stack(images)).unsqueeze(1)
        targets = torch.from_numpy(np.stack([make_target(p) for p in points_list]))
        return images, targets

    for epoch in range(args.epochs):
        random.shuffle(train)
        model.train()
        total = seen = 0
        for start in range(0, len(train), args.batch_size):
            images, targets = batch_tensors(train[start:start + args.batch_size],
                                            do_augment=args.augment)
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
        recall_hits = recall_total = precision_hits = precision_total = 0
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
                    for gold_point in gold_points:
                        gx, gy = int(gold_point[0] * HEATMAP), int(gold_point[1] * HEATMAP)
                        if any(abs(px_ - gx) <= 3 and abs(py_ - gy) <= 3
                               for px_, py_ in peaks):
                            matched += 1
                    precision_total += len(peaks)
                    precision_hits += sum(
                        1 for px_, py_ in peaks
                        if any(abs(px_ - gx) <= 3 and abs(py_ - gy) <= 3
                               for gold_point in gold_points))
                    recall_total += len(gold_points)
                    recall_hits += matched
        print(f"epoch {epoch}: peak_recall={recall_hits}/{recall_total}="
              f"{recall_hits/max(1,recall_total):.3f} "
              f"precision={precision_hits}/{max(1,precision_total)}="
              f"{precision_hits/max(1,precision_total):.3f}", flush=True)

    torch.save({"state_dict": model.state_dict()}, args.out)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
