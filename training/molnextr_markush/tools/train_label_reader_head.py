"""Train the R-group label reader head: small CNN over 88x88 label crops.

Architecture-clean root cure (§44.25): the decoder stays frozen; a compact
classifier reads label glyphs from cropped windows, trained on the 54k
alias supervision pairs (build_label_reader_pairs.py). At inference this
classifier replaces the VLM reader in the label rewrite.
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


class LabelCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_pairs(labels_path: str, size: int = 64):
    labels = json.loads(Path(labels_path).read_text())
    crops_dir = Path(labels_path).parent / "crops"
    samples = []
    for file_name, number in labels.items():
        path = crops_dir / file_name
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, (size, size))
        samples.append((image, int(number)))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.05)
    args = parser.parse_args()

    torch.manual_seed(17)
    random.seed(17)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = load_pairs(args.labels)
    if not samples:
        raise SystemExit("no training pairs found")
    classes = sorted({number for _, number in samples})
    class_to_index = {number: index for index, number in enumerate(classes)}
    index_to_number = {index: number for number, index in class_to_index.items()}
    random.shuffle(samples)
    val_count = max(1, int(len(samples) * args.val_frac))
    val, train = samples[:val_count], samples[val_count:]
    print(f"samples={len(samples)} train={len(train)} val={len(val)} "
          f"classes={len(classes)}", flush=True)

    def batch_tensor(batch):
        images = np.stack([
            cv2.resize(image, (64, 64)) for image, _ in batch
        ]).astype(np.float32) / 255.0
        images = torch.from_numpy(images).unsqueeze(1)
        targets = torch.tensor([class_to_index[number] for _, number in batch])
        return images, targets

    model = LabelCNN(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        random.shuffle(train)
        model.train()
        total_loss = seen = 0
        correct = 0
        for start in range(0, len(train) - len(train) % args.batch_size,
                           args.batch_size):
            images, targets = batch_tensor(train[start:start + args.batch_size])
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(targets)
            seen += len(targets)
            correct += int((logits.argmax(1) == targets).sum())
        print(f"epoch {epoch}: loss={total_loss/max(1,seen):.4f} "
              f"train_acc={correct/max(1,seen):.3f}", flush=True)
        model.eval()
        vcorrect = 0
        with torch.no_grad():
            for start in range(0, len(val), args.batch_size):
                images, targets = batch_tensor(val[start:start + args.batch_size])
                images, targets = images.to(device), targets.to(device)
                vcorrect += int((model(images).argmax(1) == targets).sum())
        print(f"epoch {epoch}: val_acc={vcorrect}/{len(val)}="
              f"{vcorrect/max(1,len(val)):.3f}", flush=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "classes": classes,
        "index_to_number": {str(k): v for k, v in index_to_number.items()},
        "input_size": 64,
    }
    torch.save(checkpoint, args.out)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
