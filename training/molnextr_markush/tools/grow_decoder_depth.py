"""Depth-grow a sidecar decoder checkpoint (§44.55).

Expands ``decoder.chartok_coords.decoder.transformer_layers.*`` from the base
depth to N by replicating the last warm layer for each new layer index. All
non-layer keys (embeddings, output_layer, enc_trans_layer, edges head) copy
through unchanged — the grown decoder stays fully warm-started.

The replicated layers start as EXACT copies of the last warm layer: the grown
decoder's function differs from the base only by the extra residual blocks,
so the warm decode behaviour degrades gracefully and the dedicated
--new-layer-lr group differentiates the copies during training.

Usage:
  python tools/grow_decoder_depth.py \
      --expert experiments/moe/uspto_label_v2_ft/moe_expert1.pth \
      --layers 12 --out experiments/moe/uspto_label_depth12/moe_expert1.pth
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def grow_state_dict(state: dict, target_layers: int) -> dict:
    layer_re = re.compile(
        r"^(.*transformer_layers\.)(\d+)(\..*)$")
    indices = set()
    for key in state:
        m = layer_re.match(key)
        if m:
            indices.add(int(m.group(2)))
    if not indices:
        raise SystemExit("no transformer_layers keys found — not a decoder checkpoint")
    base_depth = max(indices) + 1
    if target_layers < base_depth:
        raise SystemExit(f"target {target_layers} < base depth {base_depth}")
    if target_layers == base_depth:
        print(f"already at depth {base_depth}; nothing to do")
        return dict(state)

    last = max(indices)
    grown = dict(state)
    replicated = 0
    # CREATE the new layer indices by replicating every last-warm-layer tensor
    for key, value in state.items():
        m = layer_re.match(key)
        if not m or int(m.group(2)) != last:
            continue
        for new_idx in range(last + 1, target_layers):
            grown[f"{m.group(1)}{new_idx}{m.group(3)}"] = value.clone()
            replicated += 1
    print(f"grown {base_depth} -> {target_layers} layers "
          f"({replicated} tensors replicated from layer {last})")
    return grown


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--expert", required=True, help="sidecar .pth ({'decoder': sd})")
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import torch
    payload = torch.load(args.expert, map_location="cpu", weights_only=False)
    state = payload.get("decoder", payload)
    grown = grow_state_dict(state, args.layers)
    out_payload = {"decoder": grown}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
