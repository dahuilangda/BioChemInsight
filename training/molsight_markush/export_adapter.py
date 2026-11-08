"""Export a fine-tuned MolSight checkpoint as a residual adapter on the
published base checkpoint.

For every weight the fine-tune delta (W_ft - W_base) is stored; 2D deltas
are SVD-compressed only when that actually saves storage (k*(out+in) <
out*in) AND the reconstruction stays within tolerance -- otherwise the full
delta ships. The adapter loads on top of the pristine base; rollback is
"do not apply it".

Fails hard on reconstruction error above max_rel_err (a silent lossy
export is worse than a failed one).
"""
import sys
import json
import hashlib
import os
import torch
from safetensors.torch import save_file

FT = sys.argv[1]
BASE = sys.argv[2]
OUT_DIR = sys.argv[3]
MAX_REL_ERR = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3
ENERGY = float(sys.argv[5]) if len(sys.argv) > 5 else 0.999
MAX_RANK = int(sys.argv[6]) if len(sys.argv) > 6 else 1000000000

# Base-only tensors that the char-only architecture does not build.
INERT_BASE_KEYS = {'decoder.edge_predictor.mlp.0.weight',
                   'decoder.edge_predictor.mlp.0.bias',
                   'decoder.edge_predictor.mlp.2.weight',
                   'decoder.edge_predictor.mlp.2.bias'}


def load(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck.get('model', ck)
    return {(k[7:] if k.startswith('module.') else k): v.float() for k, v in sd.items()}


ft = load(FT)
base = load(BASE)
ft_only = sorted(set(ft) - set(base))
base_only = sorted(set(base) - set(ft) - INERT_BASE_KEYS)
if ft_only or base_only:
    raise SystemExit(f"key-set mismatch: ft_only={ft_only} base_only={base_only}")
shape_mismatch = [k for k in ft if ft[k].shape != base[k].shape]
if shape_mismatch:
    raise SystemExit(f"shape mismatch: {shape_mismatch}")

adapter = {}
ranks = {}
errors = {}
n_lowrank = n_full = n_zero = 0
for name, w_ft in ft.items():
    delta = w_ft - base[name]
    if delta.abs().max() == 0:
        n_zero += 1
        continue
    out_n = delta.shape[0] if delta.dim() == 2 else 0
    in_n = delta.shape[1] if delta.dim() == 2 else 0
    use_lowrank = False
    if delta.dim() == 2 and min(delta.shape) >= 2:
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        energy = (S ** 2).cumsum(0) / (S ** 2).sum().clamp_min(1e-12)
        k = min(int((energy < ENERGY).sum().item()) + 1, MAX_RANK, S.numel())
        if k * (out_n + in_n) < out_n * in_n:
            use_lowrank = True
    if use_lowrank:
        A = U[:, :k] * S[:k].sqrt()
        B = S[:k].sqrt()[:, None] * Vh[:k, :]
        err = (delta - A @ B).abs().max().item()
        adapter[f'{name}.A'] = A.contiguous()
        adapter[f'{name}.B'] = B.contiguous()
        ranks[name] = k
        n_lowrank += 1
    else:
        adapter[f'{name}.full'] = delta.contiguous()
        ranks[name] = 'full'
        err = 0.0
        n_full += 1
    errors[name] = err
    if err > MAX_REL_ERR * delta.abs().max().clamp_min(1e-9):
        raise SystemExit(
            f"reconstruction error too high for {name}: {err:.3e} "
            f"(delta max {delta.abs().max():.3e}); refusing to ship a lossy "
            f"adapter -- raise MAX_RANK or ship the full delta")

os.makedirs(OUT_DIR, exist_ok=True)
staging = os.path.join(OUT_DIR, 'staging')
os.makedirs(staging, exist_ok=True)
st_path = os.path.join(staging, 'adapter.safetensors')
save_file(adapter, st_path)
with open(st_path, 'rb') as sfh:
    sha = hashlib.sha256(sfh.read()).hexdigest()
config = {
    'base_checkpoint': BASE,
    'full_checkpoint': FT,
    'energy_keep': ENERGY,
    'max_rank': MAX_RANK,
    'max_relative_reconstruction_error': max(errors.values(), default=0.0),
    'lowrank_tensors': n_lowrank,
    'full_delta_tensors': n_full,
    'zero_delta_tensors': n_zero,
    'adapter_sha256': sha,
    'ranks': ranks,
    'errors': errors,
}
with open(os.path.join(staging, 'adapter_config.json'), 'w') as fh:
    json.dump(config, fh, indent=1)
os.replace(st_path, os.path.join(OUT_DIR, 'adapter.safetensors'))
os.replace(os.path.join(staging, 'adapter_config.json'),
           os.path.join(OUT_DIR, 'adapter_config.json'))
os.rmdir(staging)
size_mb = os.path.getsize(os.path.join(OUT_DIR, 'adapter.safetensors')) / 1e6
print(f"adapter: {n_lowrank} low-rank + {n_full} full deltas ({n_zero} zero "
      f"skipped), {size_mb:.1f} MB, max reconstruction error "
      f"{max(errors.values(), default=0.0):.2e}, sha256 {sha[:16]}...")
