#!/usr/bin/env python3
"""
Convert DECIMER Mask R-CNN weights from TensorFlow/Keras (.h5) to PyTorch (.pth).

Source: Matterport Mask R-CNN with ResNet-101 backbone + FPN
Target: torchvision MaskRCNN with resnet101_fpn_backbone (num_classes=2)

Usage:
    python convert_decimer_weights.py [h5_input] [pth_output]

Defaults:
    h5_input:  /tmp/decimer_weights/mask_rcnn_molecule.h5
    pth_output: /tmp/decimer_weights/mask_rcnn_molecule.pth
"""

import sys
import os
import re
import numpy as np
import h5py
import torch
import torchvision
from torch import nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.misc import Conv2dNormActivation


# ---------------------------------------------------------------------------
# 1. Load all weight tensors from .h5 into a flat dict
# ---------------------------------------------------------------------------

def load_h5_weights(h5_path):
    """Load all weight tensors from Keras .h5 file into a dict.

    TF key format after cleaning: 'group/subgroup/param'
    where group==subgroup, so we strip the duplicate prefix.
    E.g. 'res2a_branch2a/res2a_branch2a/kernel' -> 'res2a_branch2a/kernel'
         'rpn_model/rpn_conv_shared/rpn_conv_shared/kernel' -> 'rpn_conv_shared/kernel'
    But:  'rpn_model/rpn_bbox_pred/rpn_bbox_pred/kernel' -> 'rpn_bbox_pred/kernel'
    """
    weights = {}
    with h5py.File(h5_path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Strip trailing :N suffix
                clean_name = re.sub(r":\d+$", "", name)
                # Strip duplicate prefix: "a/b/c" where a==b -> "b/c"
                # But also handle "rpn_model/rpn_bbox_pred/rpn_bbox_pred/kernel"
                parts = clean_name.split("/")
                if len(parts) == 3 and parts[0] == parts[1]:
                    # e.g. "res2a_branch2a/res2a_branch2a/kernel" -> "res2a_branch2a/kernel"
                    clean_name = "/".join(parts[1:])
                elif len(parts) >= 3 and parts[1] == parts[2]:
                    # e.g. "rpn_model/rpn_bbox_pred/rpn_bbox_pred/kernel" -> "rpn_bbox_pred/kernel"
                    clean_name = "/".join(parts[1:])
                weights[clean_name] = np.array(obj)
        f.visititems(visitor)
    return weights


# ---------------------------------------------------------------------------
# 2. Build the PyTorch Mask R-CNN model
# ---------------------------------------------------------------------------

class MatterportRPNHead(nn.Module):
    """RPN head matching Matterport/DECIMER: 256-channel FPN -> 512-channel shared conv."""

    def __init__(self, in_channels=256, hidden_channels=512, num_anchors=3):
        super().__init__()
        self.conv = nn.Sequential(
            Conv2dNormActivation(
                in_channels,
                hidden_channels,
                kernel_size=3,
                norm_layer=None,
            )
        )
        self.cls_logits = nn.Conv2d(hidden_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(hidden_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            hidden = self.conv(feature)
            logits.append(self.cls_logits(hidden))
            bbox_reg.append(self.bbox_pred(hidden))
        return logits, bbox_reg


def _first_existing(candidates, tf_keys_set):
    for key in candidates:
        if key in tf_keys_set:
            return key
    return candidates[0]


def build_pytorch_model():
    """Build a Mask R-CNN model with ResNet-101 + FPN backbone."""
    backbone = resnet_fpn_backbone("resnet101", pretrained=False, trainable_layers=5)
    # FPN returns {"0": C2, "1": C3, "2": C4, "3": C5}
    # 3 anchors per location, 1 scale per level on P2-P6
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
    )
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2
    )

    rpn_head = MatterportRPNHead()

    model = MaskRCNN(
        backbone=backbone,
        num_classes=2,
        rpn_anchor_generator=anchor_generator,
        rpn_head=rpn_head,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler,
    )
    return model


# ---------------------------------------------------------------------------
# 3. Dynamic key mapping: for each PyTorch key, find the TF key
# ---------------------------------------------------------------------------

# Stage mapping: PyTorch layer index -> Matterport stage number + block count
STAGE_INFO = {
    1: (2, 3),   # layer1 -> res2, 3 blocks (a,b,c)
    2: (3, 4),   # layer2 -> res3, 4 blocks (a,b,c,d)
    3: (4, 23),  # layer3 -> res4, 23 blocks (a..w)
    4: (5, 3),   # layer4 -> res5, 3 blocks (a,b,c)
}


def idx_to_letter(idx):
    """Convert block index to letter: 0->a, 1->b, ..."""
    return chr(ord('a') + idx)


def find_tf_key_for_pt_key(pt_key, tf_keys_set):
    """Given a PyTorch state dict key, construct the expected TF key and look it up.

    Returns (tf_key, needs_special_handling) or (None, None) if not found.
    """
    # --- Backbone stem: conv1, bn1 ---
    if pt_key == "backbone.body.conv1.weight":
        return "conv1/kernel", "conv4d"
    if pt_key == "backbone.body.conv1.bias":
        return "conv1/bias", "direct"
    if pt_key == "backbone.body.bn1.weight":
        return "bn_conv1/gamma", "direct"
    if pt_key == "backbone.body.bn1.bias":
        return "bn_conv1/beta", "direct"
    if pt_key == "backbone.body.bn1.running_mean":
        return "bn_conv1/moving_mean", "direct"
    if pt_key == "backbone.body.bn1.running_var":
        return "bn_conv1/moving_variance", "direct"

    # --- Backbone residual blocks ---
    # Pattern: backbone.body.layer{L}.{B}.{conv|bn|downsample}.{...}
    m = re.match(
        r"backbone\.body\.layer(\d+)\.(\d+)\.(conv\d|bn\d|downsample)\.(.+)", pt_key
    )
    if m:
        layer_idx = int(m.group(1))
        block_idx = int(m.group(2))
        sub = m.group(3)   # conv1/conv2/conv3, bn1/bn2/bn3, downsample
        suffix = m.group(4)  # weight, bias, running_mean, running_var, or "0.weight", "1.weight", etc.

        if layer_idx not in STAGE_INFO:
            return None, None
        stage_num, n_blocks = STAGE_INFO[layer_idx]
        letter = idx_to_letter(block_idx)
        block_prefix = f"res{stage_num}{letter}"

        # conv1 -> branch2a, conv2 -> branch2b, conv3 -> branch2c
        conv_to_branch = {"conv1": "branch2a", "conv2": "branch2b", "conv3": "branch2c"}
        bn_to_branch = {"bn1": "branch2a", "bn2": "branch2b", "bn3": "branch2c"}

        if sub.startswith("conv"):
            branch = conv_to_branch[sub]
            tf_key = f"{block_prefix}_{branch}/{suffix}" if suffix == "bias" else None
            if suffix == "weight":
                tf_key = f"{block_prefix}_{branch}/kernel"
            else:
                tf_key = f"{block_prefix}_{branch}/{suffix}"
            tf_type = "conv4d" if suffix == "weight" else "direct"
            return tf_key, tf_type

        elif sub.startswith("bn"):
            branch = bn_to_branch[sub]
            # TF BN naming: bn{stage_num}{letter}_{branch}, e.g. bn2a_branch2a
            bn_name = f"bn{stage_num}{letter}_{branch}"
            param_map = {
                "weight": "gamma",
                "bias": "beta",
                "running_mean": "moving_mean",
                "running_var": "moving_variance",
            }
            tf_key = f"{bn_name}/{param_map[suffix]}"
            return tf_key, "direct"

        elif sub == "downsample":
            # downsample.0 = conv, downsample.1 = bn
            if suffix.startswith("0."):
                param = suffix[2:]  # "weight" or "bias"
                tf_key = f"{block_prefix}_branch1/{param}" if param == "bias" else f"{block_prefix}_branch1/kernel"
                tf_type = "conv4d" if param == "weight" else "direct"
                return tf_key, tf_type
            elif suffix.startswith("1."):
                param = suffix[2:]
                bn_name = f"bn{stage_num}{letter}_branch1"
                param_map = {
                    "weight": "gamma",
                    "bias": "beta",
                    "running_mean": "moving_mean",
                    "running_var": "moving_variance",
                }
                tf_key = f"{bn_name}/{param_map[param]}"
                return tf_key, "direct"

    # --- FPN lateral (inner_blocks) ---
    # torchvision FPN order: inner_blocks.0->C2(256ch), .1->C3(512ch), .2->C4(1024ch), .3->C5(2048ch)
    # TF/Matterport naming: fpn_c2p2(Cin=256), fpn_c3p3(Cin=512), fpn_c4p4(Cin=1024), fpn_c5p5(Cin=2048)
    m = re.match(r"backbone\.fpn\.inner_blocks\.(\d+)\.0\.(weight|bias)", pt_key)
    if m:
        idx = int(m.group(1))
        param = m.group(2)
        c_num = 2 + idx  # 0->2, 1->3, 2->4, 3->5
        p_num = c_num
        tf_name = f"fpn_c{c_num}p{p_num}"
        tf_key = f"{tf_name}/{param}" if param == "bias" else f"{tf_name}/kernel"
        tf_type = "conv4d" if param == "weight" else "direct"
        return tf_key, tf_type

    # --- FPN output (layer_blocks) ---
    # torchvision FPN order: layer_blocks.0->P2, .1->P3, .2->P4, .3->P5
    # TF/Matterport naming: fpn_p2, fpn_p3, fpn_p4, fpn_p5
    m = re.match(r"backbone\.fpn\.layer_blocks\.(\d+)\.0\.(weight|bias)", pt_key)
    if m:
        idx = int(m.group(1))
        param = m.group(2)
        p_num = 2 + idx  # 0->2, 1->3, 2->4, 3->5
        tf_name = f"fpn_p{p_num}"
        tf_key = f"{tf_name}/{param}" if param == "bias" else f"{tf_name}/kernel"
        tf_type = "conv4d" if param == "weight" else "direct"
        return tf_key, tf_type

    # --- RPN shared conv ---
    if pt_key == "rpn.head.conv.0.0.weight":
        return "rpn_model/rpn_conv_shared/kernel", "conv4d"
    if pt_key == "rpn.head.conv.0.0.bias":
        return "rpn_model/rpn_conv_shared/bias", "direct"

    # --- RPN class logits ---
    # TF: rpn_class_raw (1,1,512,6) 3 anchors * 2 class logits (bg/fg).
    # PT: rpn.head.cls_logits (3,512,1,1) 3 anchors * foreground score.
    if pt_key == "rpn.head.cls_logits.weight":
        return "rpn_model/rpn_class_raw/kernel", "rpn_cls_special"
    if pt_key == "rpn.head.cls_logits.bias":
        return "rpn_model/rpn_class_raw/bias", "rpn_cls_bias_special"
    if pt_key == "rpn.head.bbox_pred.weight":
        return "rpn_model/rpn_bbox_pred/kernel", "conv4d"
    if pt_key == "rpn.head.bbox_pred.bias":
        return "rpn_model/rpn_bbox_pred/bias", "direct"

    # --- Box head fc layers ---
    # fc6: TF conv (7,7,256,1024) -> PT linear (1024, 7*7*256=12544)
    if pt_key == "roi_heads.box_head.fc6.weight":
        return "mrcnn_class_conv1/kernel", "conv4d_to_linear_fc6"
    if pt_key == "roi_heads.box_head.fc6.bias":
        return "mrcnn_class_conv1/bias", "direct"
    # fc7: TF conv (1,1,1024,1024) -> PT linear (1024, 1024)
    if pt_key == "roi_heads.box_head.fc7.weight":
        return "mrcnn_class_conv2/kernel", "conv4d_to_linear_fc7"
    if pt_key == "roi_heads.box_head.fc7.bias":
        return "mrcnn_class_conv2/bias", "direct"

    # --- Box predictor ---
    if pt_key == "roi_heads.box_predictor.cls_score.weight":
        return "mrcnn_class_logits/kernel", "linear2d_transpose"
    if pt_key == "roi_heads.box_predictor.cls_score.bias":
        return "mrcnn_class_logits/bias", "direct"
    if pt_key == "roi_heads.box_predictor.bbox_pred.weight":
        return "mrcnn_bbox_fc/kernel", "linear2d_transpose"
    if pt_key == "roi_heads.box_predictor.bbox_pred.bias":
        return "mrcnn_bbox_fc/bias", "direct"

    # --- Mask head conv layers (4 convs, no BN in torchvision default) ---
    # PT: roi_heads.mask_head.{0,1,2,3}.0.weight/bias
    # TF: mrcnn_mask_conv{1,2,3,4}/kernel or bias
    m = re.match(r"roi_heads\.mask_head\.(\d+)\.0\.(weight|bias)", pt_key)
    if m:
        idx = int(m.group(1))  # 0,1,2,3
        param = m.group(2)
        tf_name = f"mrcnn_mask_conv{idx + 1}"
        tf_key = f"{tf_name}/kernel" if param == "weight" else f"{tf_name}/bias"
        tf_type = "conv4d" if param == "weight" else "direct"
        return tf_key, tf_type

    # --- Mask deconv ---
    # PT: roi_heads.mask_predictor.conv5_mask (256,256,2,2)
    # TF: mrcnn_mask_deconv (2,2,256,256)
    if pt_key == "roi_heads.mask_predictor.conv5_mask.weight":
        return "mrcnn_mask_deconv/kernel", "conv4d"
    if pt_key == "roi_heads.mask_predictor.conv5_mask.bias":
        return "mrcnn_mask_deconv/bias", "direct"

    # --- Mask final conv ---
    # PT: roi_heads.mask_predictor.mask_fcn_logits (2,256,1,1)
    # TF: mrcnn_mask (1,1,256,2)
    if pt_key == "roi_heads.mask_predictor.mask_fcn_logits.weight":
        return "mrcnn_mask/kernel", "conv4d"
    if pt_key == "roi_heads.mask_predictor.mask_fcn_logits.bias":
        return "mrcnn_mask/bias", "direct"

    return None, None


# ---------------------------------------------------------------------------
# 4. Convert weights
# ---------------------------------------------------------------------------

def convert_array(arr, conversion_type):
    """Apply the appropriate conversion to a numpy array."""
    if conversion_type == "direct":
        return arr
    elif conversion_type == "conv4d":
        # TF conv: (H, W, Cin, Cout) -> PyTorch: (Cout, Cin, H, W)
        return arr.transpose(3, 2, 0, 1)
    elif conversion_type == "linear2d_transpose":
        # TF dense: (Cin, Cout) -> PyTorch Linear: (Cout, Cin)
        return arr.transpose(1, 0)
    elif conversion_type == "conv4d_to_linear_fc6":
        # TF Conv2D (7,7,256,1024) -> PT Linear (1024, 7*7*256=12544)
        # First transpose to PyTorch conv format: (Cout, Cin, H, W) = (1024, 256, 7, 7)
        arr = arr.transpose(3, 2, 0, 1)
        # Then reshape: (1024, 256*7*7) = (1024, 12544)
        cout, cin, h, w = arr.shape
        return arr.reshape(cout, cin * h * w)
    elif conversion_type == "conv4d_to_linear_fc7":
        # TF Conv2D (1,1,1024,1024) -> PT Linear (1024, 1024)
        # First transpose: (1024, 1024, 1, 1)
        arr = arr.transpose(3, 2, 0, 1)
        # Then reshape: (1024, 1024*1*1) = (1024, 1024)
        cout, cin, h, w = arr.shape
        return arr.reshape(cout, cin * h * w)
    elif conversion_type == "rpn_cls_special":
        # Matterport predicts [bg, fg] logits for each anchor.  Torchvision's
        # RPN uses one binary objectness logit, equivalent to fg - bg.
        # TF: (1,1,512,6) -> per-anchor difference -> (3,512,1,1)
        arr = arr.reshape(1, 1, arr.shape[2], 3, 2)
        arr = arr[:, :, :, :, 1] - arr[:, :, :, :, 0]
        return arr.transpose(3, 2, 0, 1)
    elif conversion_type == "rpn_cls_bias_special":
        arr = arr.reshape(3, 2)
        return arr[:, 1] - arr[:, 0]
    else:
        return arr


def convert_weights(h5_path, pth_path):
    """Main conversion function."""
    print("=" * 70)
    print("DECIMER Mask R-CNN Weight Converter")
    print("TF/Keras (.h5) -> PyTorch (.pth)")
    print("=" * 70)

    # Load TF weights
    print(f"\n[1/4] Loading TF weights from: {h5_path}")
    tf_weights = load_h5_weights(h5_path)
    tf_keys = set(tf_weights.keys())
    print(f"      Found {len(tf_weights)} weight tensors")

    # Build PyTorch model
    print("\n[2/4] Building PyTorch Mask R-CNN model (ResNet-101 + FPN)...")
    model = build_pytorch_model()
    pt_sd = model.state_dict()
    pt_keys = list(pt_sd.keys())
    print(f"      Model has {len(pt_keys)} parameters")

    # Map and convert
    print("\n[3/4] Converting weights...")
    converted = {}
    matched_report = []
    unmatched_pt = []
    unmatched_tf = set(tf_keys)

    for pt_key in pt_keys:
        tf_key, conv_type = find_tf_key_for_pt_key(pt_key, tf_keys)

        if tf_key is None:
            unmatched_pt.append((pt_key, "no mapping rule"))
            continue

        if tf_key not in tf_weights:
            unmatched_pt.append((pt_key, f"TF key '{tf_key}' not found in h5"))
            continue

        arr = tf_weights[tf_key]
        converted_arr = convert_array(arr, conv_type)

        if converted_arr is None:
            # Special case: architecture mismatch, keep random init
            unmatched_pt.append((pt_key, f"architecture mismatch ({conv_type}), keeping random init"))
            unmatched_tf.discard(tf_key)
            continue

        t = torch.from_numpy(converted_arr.copy()).float()
        expected_shape = pt_sd[pt_key].shape

        if t.shape == expected_shape:
            converted[pt_key] = t
            matched_report.append((pt_key, tf_key, arr.shape, tuple(t.shape)))
            unmatched_tf.discard(tf_key)
        else:
            unmatched_pt.append((
                pt_key,
                f"shape mismatch: TF {arr.shape} -> converted {t.shape}, expected {expected_shape}"
            ))
            unmatched_tf.discard(tf_key)

    # Fill unmatched PT keys with original random init
    for pt_key in pt_keys:
        if pt_key not in converted:
            converted[pt_key] = pt_sd[pt_key]

    # Print report
    print(f"\n      Successfully converted: {len(matched_report)} / {len(pt_keys)} parameters")

    if matched_report:
        print("\n      --- Matched weights (sample) ---")
        for pt_key, tf_key, tf_shape, pt_shape in matched_report[:15]:
            print(f"      {pt_key}")
            print(f"        <- {tf_key}  {tf_shape} -> {pt_shape}")
        if len(matched_report) > 15:
            print(f"      ... and {len(matched_report) - 15} more")

    if unmatched_pt:
        print(f"\n      --- Unmatched PyTorch keys ({len(unmatched_pt)}) ---")
        for pt_key, reason in unmatched_pt:
            print(f"      {pt_key}")
            print(f"        Reason: {reason}")

    # Filter out non-weight TF keys (like anchors)
    unmatched_tf_weights = {k for k in unmatched_tf if "anchor" not in k.lower()}
    if unmatched_tf_weights:
        print(f"\n      --- Unmatched TF keys ({len(unmatched_tf_weights)}) ---")
        for k in sorted(unmatched_tf_weights):
            print(f"      {k}: {tf_weights[k].shape}")

    # Save
    print(f"\n[4/4] Saving to: {pth_path}")
    os.makedirs(os.path.dirname(pth_path), exist_ok=True)
    torch.save(converted, pth_path)
    file_size = os.path.getsize(pth_path) / (1024 * 1024)
    print(f"      File size: {file_size:.1f} MB")

    # Summary
    print("\n" + "=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)
    total = len(pt_keys)
    converted_count = len(matched_report)
    random_init = total - converted_count
    print(f"  Total PyTorch parameters:    {total}")
    print(f"  Converted from TF:           {converted_count} ({100*converted_count/total:.1f}%)")
    print(f"  Random init (unmatched):     {random_init} ({100*random_init/total:.1f}%)")

    if random_init > 0:
        print(f"\n  NOTE: {random_init} parameters could not be transferred from TF.")
        print("  These are primarily:")
        rpn_keys = [k for k, r in unmatched_pt if "rpn" in k]
        other_keys = [k for k, r in unmatched_pt if "rpn" not in k]
        if rpn_keys:
            print(f"    - RPN head ({len(rpn_keys)} params): architecture mismatch")
            print(f"      Matterport RPN uses 512-ch shared conv, torchvision uses 256-ch.")
            print(f"      The RPN will use random initialization and should be fine-tuned.")
        if other_keys:
            for k in other_keys:
                print(f"    - {k}")

    print(f"\n  Output saved to: {pth_path}")
    print("=" * 70)

    return converted


if __name__ == "__main__":
    h5_in = sys.argv[1] if len(sys.argv) > 1 else "/tmp/decimer_weights/mask_rcnn_molecule.h5"
    pth_out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/decimer_weights/mask_rcnn_molecule.pth"
    convert_weights(h5_in, pth_out)
