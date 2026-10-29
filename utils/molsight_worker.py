"""MolSight inference service (§44.70) — OCSR for complete + large macrocycle.

MolSight (hustvl, AAAI 2026): EfficientViT-L1 encoder + 6-layer char-level
autoregressive decoder + edge predictor. Benchmark: USPTO ~94%, Stereo-200k
0.84 ACC. On our cyclic-peptide test set it was the ONLY model that decoded a
23-membered macrocycle (gramicidin S) exactly — where keeper broke the ring
and MarkushGlyph got the composition wrong.

Runs INSIDE the molsight_venv (separate from production llm env and glyph venv).
JSON-line protocol on stdin/stdout, same pattern as markushglyph_worker.py.
"""
import argparse
import json
import sys
import os

os.environ.setdefault("HF_HOME", "/data/LLM_Models/huggingface_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.environ.get(
        "MOLSIGHT_CHECKPOINT",
        "/data/molsight_repo/runs/markush_ft_v4_tier2/ckpt_model/epoch_0.pth"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    sys.path.insert(0, "/data/molsight_repo")
    import torch
    import cv2
    from molsight.dataset import get_transforms
    from molsight.model import MolsightModel
    from molsight.tokenizer import CharTokenizer
    from molsight.chemistry import _postprocess_smiles

    class A:
        pass
    a = A()
    a.encoder = 'efficientvit'
    a.use_checkpoint = False
    a.embed_dim = 512
    a.dec_n_layer = 6
    a.dec_n_head = 8
    a.use_qknorm = True
    a.use_swiglu = True
    a.use_rmsnorm = True
    a.lora = False
    a.regression = False
    a.resume = True
    a.input_size = 512
    a.formats = ['char']
    a.vocab_file = '/data/molsight_repo/vocab/vocab_chars.json'
    a.max_len = 320

    tokenizer = CharTokenizer(a.vocab_file)
    model = MolsightModel(a, tokenizer)
    model.to(args.device)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    mc = ckpt.get("model", ckpt)
    mc = {(k[7:] if k.startswith('module.') else k): v for k, v in mc.items()}
    missing, unexpected = model.load_state_dict(mc, strict=False)
    if hasattr(model, 'module'):
        model = model.module
    model.eval()
    print(json.dumps({"event": "ready", "missing": len(missing),
                       "unexpected": len(unexpected)}), flush=True)

    transform = get_transforms(a, augment=False, rotate=False)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"ok": False, "error": "bad json"}), flush=True)
            continue
        image_path = req.get("image_path", "")
        if not image_path or not os.path.exists(image_path):
            print(json.dumps({"ok": False, "error": "image not found"}), flush=True)
            continue
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            aug = transform(image=image)
            tensor = aug['image'].unsqueeze(0).to(args.device)
            with torch.no_grad():
                kv_cache, hooks = model.install_kv_cache_hooks()
                preds, _ = model.generate(image=tensor, kv_cache=kv_cache)
                for hook in hooks:
                    hook.remove()
            raw_smiles = preds.get('smiles', [''])[0]
            try:
                smi, _, pp_ok = _postprocess_smiles(raw_smiles)
            except Exception:
                smi, pp_ok = raw_smiles, False
            print(json.dumps({"ok": True, "smiles": smi, "raw": raw_smiles,
                               "postprocess_ok": bool(pp_ok)}), flush=True)
        except Exception as exc:
            print(json.dumps({"ok": False,
                               "error": f"{type(exc).__name__}: {str(exc)[:200]}"}), flush=True)
    # CUDA teardown can hang after EOF; exit hard once stdin closes.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
