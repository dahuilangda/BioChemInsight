"""Depth-grow the MolSight decoder from 6 to 12 layers by replicating the
last warm layer. Non-layer keys copy through unchanged. The grown decoder
stays warm-started (the function changes only by extra residual blocks).

Usage: python grow_decoder.py --ckpt <in.pth> --layers 12 --out <out.pth>
"""
import argparse, torch, re, os

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--layers", type=int, default=12)
ap.add_argument("--out", required=True)
args = ap.parse_args()

ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
sd = ck.get('model', ck)
layer_re = re.compile(r'^(decoder\.blocks\.)(\d+)(\..*)$')
indices = set()
for key in sd:
    m = layer_re.match(key)
    if m: indices.add(int(m.group(2)))
base_depth = max(indices) + 1
print(f"base depth: {base_depth}")
if args.layers <= base_depth:
    print(f"already at depth {base_depth}"); os._exit(0)

last = max(indices)
grown = dict(sd)
replicated = 0
for key, value in sd.items():
    m = layer_re.match(key)
    if not m or int(m.group(2)) != last: continue
    for new_idx in range(last + 1, args.layers):
        grown[f"{m.group(1)}{new_idx}{m.group(3)}"] = value.clone()
        replicated += 1
print(f"grown {base_depth} -> {args.layers} ({replicated} tensors replicated)")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
torch.save({'epoch': -1, 'model': grown}, args.out)
print(f"saved {args.out} ({os.path.getsize(args.out)/1e6:.0f} MB)")
