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
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.misc import Conv2dNormActivation


MATTERPORT_BOX_STD = (0.1, 0.1, 0.2, 0.2)
TORCHVISION_BOX_CODER_WEIGHTS = tuple(1.0 / value for value in MATTERPORT_BOX_STD)


def load_h5_weights(h5_path):
    weights = {}
    with h5py.File(h5_path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                clean_name = re.sub(r":\d+$", "", name)
                parts = clean_name.split("/")
                if len(parts) == 3 and parts[0] == parts[1]:
                    clean_name = "/".join(parts[1:])
                elif len(parts) >= 3 and parts[1] == parts[2]:
                    clean_name = "/".join(parts[1:])
                weights[clean_name] = np.array(obj)
        f.visititems(visitor)
    return weights


class MatterportRPNHead(nn.Module):
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


class MatterportBoxHead(nn.Module):
    def __init__(self, in_channels=256, resolution=7, representation_size=1024):
        super().__init__()
        self.fc6 = nn.Linear(in_channels * resolution * resolution, representation_size)
        self.bn6 = nn.BatchNorm1d(representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.bn7 = nn.BatchNorm1d(representation_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.bn6(self.fc6(x)))
        x = self.relu(self.bn7(self.fc7(x)))
        return x


class MatterportMaskHead(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        return x


def _first_existing(candidates, tf_keys_set):
    for key in candidates:
        if key in tf_keys_set:
            return key
    return candidates[0]


def build_pytorch_model():
    backbone = resnet_fpn_backbone("resnet101", pretrained=False, trainable_layers=5)
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
    box_head = MatterportBoxHead()
    mask_head = MatterportMaskHead()

    model = MaskRCNN(
        backbone=backbone,
        num_classes=2,
        rpn_anchor_generator=anchor_generator,
        rpn_head=rpn_head,
        box_head=box_head,
        box_roi_pool=roi_pooler,
        mask_head=mask_head,
        mask_roi_pool=mask_roi_pooler,
    )
    model.transform = GeneralizedRCNNTransform(
        min_size=1024,
        max_size=1024,
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
        size_divisible=1,
    )
    model.rpn.box_coder.weights = TORCHVISION_BOX_CODER_WEIGHTS
    model.roi_heads.box_coder.weights = TORCHVISION_BOX_CODER_WEIGHTS
    return model


STAGE_INFO = {
    1: (2, 3),
    2: (3, 4),
    3: (4, 23),
    4: (5, 3),
}


def idx_to_letter(idx):
    return chr(ord('a') + idx)


def find_tf_key_for_pt_key(pt_key, tf_keys_set):
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

    m = re.match(
        r"backbone\.body\.layer(\d+)\.(\d+)\.(conv\d|bn\d|downsample)\.(.+)", pt_key
    )
    if m:
        layer_idx = int(m.group(1))
        block_idx = int(m.group(2))
        sub = m.group(3)
        suffix = m.group(4)

        if layer_idx not in STAGE_INFO:
            return None, None
        stage_num, n_blocks = STAGE_INFO[layer_idx]
        letter = idx_to_letter(block_idx)
        block_prefix = f"res{stage_num}{letter}"

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
            if suffix.startswith("0."):
                param = suffix[2:]
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

    m = re.match(r"backbone\.fpn\.inner_blocks\.(\d+)\.0\.(weight|bias)", pt_key)
    if m:
        idx = int(m.group(1))
        param = m.group(2)
        c_num = 2 + idx
        p_num = c_num
        tf_name = f"fpn_c{c_num}p{p_num}"
        tf_key = f"{tf_name}/{param}" if param == "bias" else f"{tf_name}/kernel"
        tf_type = "conv4d" if param == "weight" else "direct"
        return tf_key, tf_type

    m = re.match(r"backbone\.fpn\.layer_blocks\.(\d+)\.0\.(weight|bias)", pt_key)
    if m:
        idx = int(m.group(1))
        param = m.group(2)
        p_num = 2 + idx
        tf_name = f"fpn_p{p_num}"
        tf_key = f"{tf_name}/{param}" if param == "bias" else f"{tf_name}/kernel"
        tf_type = "conv4d" if param == "weight" else "direct"
        return tf_key, tf_type

    if pt_key == "rpn.head.conv.0.0.weight":
        return "rpn_model/rpn_conv_shared/kernel", "conv4d"
    if pt_key == "rpn.head.conv.0.0.bias":
        return "rpn_model/rpn_conv_shared/bias", "direct"

    if pt_key == "rpn.head.cls_logits.weight":
        return "rpn_model/rpn_class_raw/kernel", "rpn_cls_special"
    if pt_key == "rpn.head.cls_logits.bias":
        return "rpn_model/rpn_class_raw/bias", "rpn_cls_bias_special"
    if pt_key == "rpn.head.bbox_pred.weight":
        return "rpn_model/rpn_bbox_pred/kernel", "bbox_conv4d_dy_dx_dh_dw_to_dx_dy_dw_dh"
    if pt_key == "rpn.head.bbox_pred.bias":
        return "rpn_model/rpn_bbox_pred/bias", "bbox_bias_dy_dx_dh_dw_to_dx_dy_dw_dh"

    if pt_key == "roi_heads.box_head.fc6.weight":
        return "mrcnn_class_conv1/kernel", "conv4d_to_linear_fc6"
    if pt_key == "roi_heads.box_head.fc6.bias":
        return "mrcnn_class_conv1/bias", "direct"
    if pt_key == "roi_heads.box_head.fc7.weight":
        return "mrcnn_class_conv2/kernel", "conv4d_to_linear_fc7"
    if pt_key == "roi_heads.box_head.fc7.bias":
        return "mrcnn_class_conv2/bias", "direct"
    m = re.match(r"roi_heads\.box_head\.bn([67])\.(weight|bias|running_mean|running_var)", pt_key)
    if m:
        bn_num = int(m.group(1)) - 5
        param = m.group(2)
        param_map = {
            "weight": "gamma",
            "bias": "beta",
            "running_mean": "moving_mean",
            "running_var": "moving_variance",
        }
        return f"mrcnn_class_bn{bn_num}/{param_map[param]}", "direct"

    if pt_key == "roi_heads.box_predictor.cls_score.weight":
        return "mrcnn_class_logits/kernel", "linear2d_transpose"
    if pt_key == "roi_heads.box_predictor.cls_score.bias":
        return "mrcnn_class_logits/bias", "direct"
    if pt_key == "roi_heads.box_predictor.bbox_pred.weight":
        return "mrcnn_bbox_fc/kernel", "bbox_linear_dy_dx_dh_dw_to_dx_dy_dw_dh"
    if pt_key == "roi_heads.box_predictor.bbox_pred.bias":
        return "mrcnn_bbox_fc/bias", "bbox_bias_dy_dx_dh_dw_to_dx_dy_dw_dh"

    m = re.match(r"roi_heads\.mask_head\.conv(\d+)\.(weight|bias)", pt_key)
    if m:
        idx = int(m.group(1)) - 1
        param = m.group(2)
        tf_name = f"mrcnn_mask_conv{idx + 1}"
        tf_key = f"{tf_name}/kernel" if param == "weight" else f"{tf_name}/bias"
        tf_type = "conv4d" if param == "weight" else "direct"
        return tf_key, tf_type
    m = re.match(r"roi_heads\.mask_head\.bn(\d+)\.(weight|bias|running_mean|running_var)", pt_key)
    if m:
        bn_num = int(m.group(1))
        param = m.group(2)
        param_map = {
            "weight": "gamma",
            "bias": "beta",
            "running_mean": "moving_mean",
            "running_var": "moving_variance",
        }
        return f"mrcnn_mask_bn{bn_num}/{param_map[param]}", "direct"

    if pt_key == "roi_heads.mask_predictor.conv5_mask.weight":
        return "mrcnn_mask_deconv/kernel", "conv4d"
    if pt_key == "roi_heads.mask_predictor.conv5_mask.bias":
        return "mrcnn_mask_deconv/bias", "direct"

    if pt_key == "roi_heads.mask_predictor.mask_fcn_logits.weight":
        return "mrcnn_mask/kernel", "conv4d"
    if pt_key == "roi_heads.mask_predictor.mask_fcn_logits.bias":
        return "mrcnn_mask/bias", "direct"

    return None, None


def convert_array(arr, conversion_type):
    if conversion_type == "direct":
        return arr
    elif conversion_type == "conv4d":
        return arr.transpose(3, 2, 0, 1)
    elif conversion_type == "linear2d_transpose":
        return arr.transpose(1, 0)
    elif conversion_type == "conv4d_to_linear_fc6":
        arr = arr.transpose(3, 2, 0, 1)
        cout, cin, h, w = arr.shape
        return arr.reshape(cout, cin * h * w)
    elif conversion_type == "conv4d_to_linear_fc7":
        arr = arr.transpose(3, 2, 0, 1)
        cout, cin, h, w = arr.shape
        return arr.reshape(cout, cin * h * w)
    elif conversion_type == "rpn_cls_special":
        arr = arr.reshape(1, 1, arr.shape[2], 3, 2)
        arr = arr[:, :, :, :, 1] - arr[:, :, :, :, 0]
        return arr.transpose(3, 2, 0, 1)
    elif conversion_type == "rpn_cls_bias_special":
        arr = arr.reshape(3, 2)
        return arr[:, 1] - arr[:, 0]
    elif conversion_type == "bbox_conv4d_dy_dx_dh_dw_to_dx_dy_dw_dh":
        arr = arr.transpose(3, 2, 0, 1)
        arr = arr.reshape(3, 4, arr.shape[1], arr.shape[2], arr.shape[3])
        arr = arr[:, [1, 0, 3, 2], :, :, :]
        return arr.reshape(12, arr.shape[2], arr.shape[3], arr.shape[4])
    elif conversion_type == "bbox_linear_dy_dx_dh_dw_to_dx_dy_dw_dh":
        arr = arr.transpose(1, 0)
        arr = arr.reshape(2, 4, arr.shape[1])
        arr = arr[:, [1, 0, 3, 2], :]
        return arr.reshape(8, arr.shape[2])
    elif conversion_type == "bbox_bias_dy_dx_dh_dw_to_dx_dy_dw_dh":
        arr = arr.reshape(-1, 4)
        arr = arr[:, [1, 0, 3, 2]]
        return arr.reshape(-1)
    else:
        return arr


def convert_weights(h5_path, pth_path):
    print("=" * 70)
    print("DECIMER Mask R-CNN Weight Converter")
    print("TF/Keras (.h5) -> PyTorch (.pth)")
    print("=" * 70)

    print(f"\n[1/4] Loading TF weights from: {h5_path}")
    tf_weights = load_h5_weights(h5_path)
    tf_keys = set(tf_weights.keys())
    print(f"      Found {len(tf_weights)} weight tensors")

    print("\n[2/4] Building PyTorch Mask R-CNN model (ResNet-101 + FPN)...")
    model = build_pytorch_model()
    pt_sd = model.state_dict()
    pt_keys = list(pt_sd.keys())
    print(f"      Model has {len(pt_keys)} parameters")

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

    for pt_key in pt_keys:
        if pt_key not in converted:
            converted[pt_key] = pt_sd[pt_key]

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

    unmatched_tf_weights = {k for k in unmatched_tf if "anchor" not in k.lower()}
    if unmatched_tf_weights:
        print(f"\n      --- Unmatched TF keys ({len(unmatched_tf_weights)}) ---")
        for k in sorted(unmatched_tf_weights):
            print(f"      {k}: {tf_weights[k].shape}")

    print(f"\n[4/4] Saving to: {pth_path}")
    os.makedirs(os.path.dirname(pth_path), exist_ok=True)
    torch.save(converted, pth_path)
    file_size = os.path.getsize(pth_path) / (1024 * 1024)
    print(f"      File size: {file_size:.1f} MB")

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
