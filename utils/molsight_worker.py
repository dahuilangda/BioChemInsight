"""MolSight inference worker: OCSR for complete and large-macrocycle
structures.

MolSight (hustvl): EfficientViT-L1 encoder + char-level autoregressive
decoder + edge predictor. Runs inside the molsight_venv, separate from the
production llm env and glyph venv. JSON-line protocol on stdin/stdout.
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
        "/data/molsight_repo/runs/markush_ft_v12_3ep_clean/ckpt_model/epoch_0.pth"))
    parser.add_argument("--adapter", default=os.environ.get(
        "MOLSIGHT_ADAPTER", ""),
        help="Adapter directory (adapter.safetensors + adapter_config.json) "
             "applied on top of the checkpoint. Empty disables it — the v8 "
             "checkpoint is already fine-tuned, no adapter needed.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    sys.path.insert(0, "/data/molsight_repo")
    import torch
    import cv2
    from molsight.dataset import get_transforms
    from molsight.model import MolsightModel
    from molsight.tokenizer import CharTokenizer
    from molsight.chemistry import _postprocess_smiles

    from types import SimpleNamespace
    cfg = SimpleNamespace(
        encoder='efficientvit', use_checkpoint=False, embed_dim=512,
        dec_n_layer=6, dec_n_head=8, use_qknorm=True, use_swiglu=True,
        use_rmsnorm=True, lora=False, regression=False, resume=True,
        input_size=512, formats=['char'],
        vocab_file='/data/molsight_repo/vocab/vocab_chars.json',
        max_len=1024)

    tokenizer = CharTokenizer(cfg.vocab_file)
    model = MolsightModel(cfg, tokenizer)
    model.to(args.device)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    mc = ckpt.get("model", ckpt)
    mc = {(k[7:] if k.startswith('module.') else k): v for k, v in mc.items()}

    if args.adapter:
        # The adapter is a residual delta against the base checkpoint named
        # in its config. Refuse to start on any other checkpoint — applying
        # it elsewhere serves base + delta + (checkpoint - base), a model
        # nobody trained.
        from safetensors.torch import load_file
        with open(os.path.join(args.adapter, 'adapter_config.json')) as fh:
            adapter_cfg = json.load(fh)
        expected_base = os.path.basename(adapter_cfg['base_checkpoint'])
        if os.path.basename(args.checkpoint) != expected_base:
            raise SystemExit(
                f"adapter base mismatch: adapter expects {expected_base}, "
                f"checkpoint is {os.path.basename(args.checkpoint)}")
        adapter = load_file(os.path.join(args.adapter, 'adapter.safetensors'))
        for key, tensor in adapter.items():
            if key.endswith('.A'):
                name = key[:-2]
                mc[name] = mc[name].float() + adapter[f'{name}.A'].float() @ adapter[f'{name}.B'].float()
            elif key.endswith('.full'):
                mc[key[:-5]] = mc[key[:-5]].float() + tensor.float()
    missing, unexpected = model.load_state_dict(mc, strict=False)
    model.eval()
    print(json.dumps({"event": "ready", "missing": len(missing),
                       "unexpected": len(unexpected)}), flush=True)

    transform = get_transforms(cfg, augment=False, rotate=False)

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
