"""Shared base classes for single-GPU detection/segmentation trainers.

The three Mask R-CNN / U-Net trainers (``train_wavy_maskrcnn``,
``train_attachment_maskrcnn``, ``train_wavy_unet``) share ~80 % of their
training-loop boilerplate.  This module factors that into two abstract base
classes so subclasses only override the parts that genuinely differ: model
architecture, dataset format, and loss computation.

Class hierarchy::

    BaseTrainer                     (abstract training loop + CLI)
    ├── BaseCocoMaskRCNNTrainer     (COCO-format Mask R-CNN specifics)
    │   ├── WavyMaskRCNNTrainer
    │   └── AttachmentMaskRCNNTrainer
    └── WavyUNetTrainer             (image+mask directory format)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as TF


# --------------------------------------------------------------------------- #
# Mask R-CNN factory
# --------------------------------------------------------------------------- #

def build_maskrcnn(
    num_classes: int = 2,
    pretrained_backbone: bool = True,
    backbone_name: str = "resnet50",
) -> MaskRCNN:
    """Construct a Mask R-CNN with FPN backbone and small-object anchors.

    Parameters
    ----------
    num_classes
        ``1 + num_foreground_classes`` (background is implicit).
    pretrained_backbone
        Warm-start the ResNet+FPN backbone from ImageNet weights.
    backbone_name
        Torchvision ResNet variant (``"resnet50"`` recommended).
    """
    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights="DEFAULT" if pretrained_backbone else None,
        trainable_layers=5,
    )
    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),) * 5,
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
    )
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2
    )
    return MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pooler=mask_roi_pooler,
        min_size=384,
        max_size=384,
    )


# --------------------------------------------------------------------------- #
# COCO collate
# --------------------------------------------------------------------------- #

def coco_collate(batch):
    """Variable-size batch collate for detection datasets."""
    return tuple(zip(*batch))


# --------------------------------------------------------------------------- #
# BaseTrainer
# --------------------------------------------------------------------------- #

class BaseTrainer:
    """Abstract single-GPU trainer with best-loss checkpointing.

    Subclasses must override :meth:`build_model`, :meth:`build_dataloaders`,
    :meth:`compute_loss`, and :meth:`checkpoint_name`.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(args.out)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric: float = 1e9

    # -- abstract methods ------------------------------------------------ #

    def build_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader | None]:
        """Return (train_loader, val_loader_or_None)."""
        raise NotImplementedError

    def compute_loss(self, model: torch.nn.Module, batch: Any) -> torch.Tensor | None:
        """Return loss tensor, or *None* to skip the batch."""
        raise NotImplementedError

    def checkpoint_name(self) -> str:
        raise NotImplementedError

    # -- overridable hooks ---------------------------------------------- #

    def build_optimizer(self, params: list) -> torch.optim.Optimizer:
        return torch.optim.SGD(params, lr=self.args.lr, momentum=0.9, weight_decay=1e-4)

    def build_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler | None:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    def clip_gradients(self, params: list) -> None:
        """Override to apply gradient clipping (e.g. ``clip_grad_norm_``)."""
        pass

    def is_better(self, metric: float) -> bool:
        """Return *True* if *metric* improves on the best so far."""
        return metric < self.best_metric

    def epoch_suffix(self) -> str:
        """Override to append extra fields to the per-epoch log line."""
        return ""

    def validate(self, model: torch.nn.Module, val_dl: DataLoader | None) -> float:
        """Return a validation metric for checkpoint selection.

        Default: no validation — the training loss is used instead.
        """
        return 1e9

    # -- training loop --------------------------------------------------- #

    def train(self) -> None:
        train_dl, val_dl = self.build_dataloaders()
        model = self.build_model().to(self.device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self.build_optimizer(params)
        scheduler = self.build_scheduler(optimizer)

        for epoch in range(self.args.epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_dl:
                loss = self.compute_loss(model, batch)
                if loss is None:
                    continue
                optimizer.zero_grad()
                loss.backward()
                self.clip_gradients(params)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            if scheduler is not None:
                scheduler.step()

            avg_loss = epoch_loss / max(1, n_batches)
            lr = (
                scheduler.get_last_lr()[0]
                if scheduler is not None
                else optimizer.param_groups[0]["lr"]
            )

            # Only use validation metric for checkpointing when the subclass
            # has actually overridden validate().  Otherwise fall back to
            # training loss (the default no-validation contract).
            has_real_validation = type(self).validate is not BaseTrainer.validate
            if val_dl is not None and has_real_validation:
                val_metric = self.validate(model, val_dl)
                checkpoint_metric = val_metric
                val_tag = f" val_metric={val_metric:.4f}"
            else:
                checkpoint_metric = avg_loss
                val_tag = ""
            print(f"epoch {epoch}: train_loss={avg_loss:.4f} lr={lr:.2e}{val_tag}{self.epoch_suffix()}")

            if self.is_better(checkpoint_metric):
                self.best_metric = checkpoint_metric
                path = self.output_dir / self.checkpoint_name()
                torch.save(model.state_dict(), path)
                print(f"  saved best -> {path}")

        print("done")

    # -- checkpoint I/O -------------------------------------------------- #

    def load_model(self, ckpt_path: str | Path, pretrained_backbone: bool = False) -> torch.nn.Module:
        """Load a checkpoint into a fresh model and set eval mode."""
        model = self.build_model().to(self.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        model.eval()
        return model

    # -- shared CLI ------------------------------------------------------ #

    @classmethod
    def add_common_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--data", default="experiments/seg/train")
        parser.add_argument("--out", default="experiments/seg")
        parser.add_argument("--epochs", type=int, default=20)
        parser.add_argument("--batch-size", type=int, default=6)
        parser.add_argument("--lr", type=float, default=0.005)
        parser.add_argument("--val-frac", type=float, default=0.1)
        parser.add_argument("--num-workers", type=int, default=0)
        parser.add_argument("--eval-only", action="store_true")
        parser.add_argument("--ckpt", default="")
        parser.add_argument("--eval-csv", default="")
        parser.add_argument("--score-threshold", type=float, default=0.05)


# --------------------------------------------------------------------------- #
# BaseCocoMaskRCNNTrainer
# --------------------------------------------------------------------------- #

class BaseCocoMaskRCNNTrainer(BaseTrainer):
    """Common base for COCO-format Mask R-CNN trainers.

    Subclasses set :attr:`NUM_CLASSES` and :attr:`CHECKPOINT_NAME`, and
    override :meth:`build_dataloaders` to wire the specific COCO split.
    """

    NUM_CLASSES: int = 2
    CHECKPOINT_NAME: str = "maskrcnn_best.pth"

    def build_model(self) -> MaskRCNN:
        return build_maskrcnn(
            num_classes=self.NUM_CLASSES,
            pretrained_backbone=True,
        )

    def clip_gradients(self, params: list) -> None:
        torch.nn.utils.clip_grad_norm_(params, 5.0)

    def compute_loss(self, model: MaskRCNN, batch: tuple) -> torch.Tensor | None:
        imgs, targets = batch
        imgs = [im.to(self.device) for im in imgs]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        if all(t["boxes"].numel() == 0 for t in targets):
            return None
        loss_dict = model(imgs, targets)
        return sum(loss for loss in loss_dict.values())

    def checkpoint_name(self) -> str:
        return self.CHECKPOINT_NAME


# --------------------------------------------------------------------------- #
# COCO dataset (shared)
# --------------------------------------------------------------------------- #

class CocoDetectionDataset(torch.utils.data.Dataset):
    """COCO-format instance-segmentation dataset with optional augmentation.

    Works for both single-class (wavy) and multi-class (attachment) COCO
    annotation files.  When ``fixed_label`` is set (e.g. 1 for single-class
    wavy detection), all annotations use that label regardless of
    ``category_id``; otherwise ``category_id`` is preserved as-is.
    """

    def __init__(self, img_dir: Path, coco, augment: bool = False, fixed_label: int | None = None):
        self.img_dir = Path(img_dir)
        self.coco = coco
        self.ids = list(coco.imgs.keys())
        self.augment = augment
        self.fixed_label = fixed_label

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
            masks.append(self.coco.annToMask(ann))
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.fixed_label if self.fixed_label is not None else int(ann["category_id"]))
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks_t = torch.zeros((0, img.size[1], img.size[0]), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks_t = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        if self.augment and boxes.shape[0] > 0 and np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w_img = img.size[0]
            boxes[:, [0, 2]] = w_img - boxes[:, [2, 0]]
            masks_t = torch.flip(masks_t, dims=[2])
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks_t,
            "image_id": torch.tensor([img_id]),
        }
        return TF.to_tensor(img), target
