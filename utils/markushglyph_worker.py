"""MarkushGlyph persistent worker — runs inside the glyph venv.

Loads Qwen3.5-2B + the MarkushGlyph LoRA once, then answers JSON-line
requests on stdin: {"image_path": "..."} → {"cxsmiles": "...", "ok": true}.
The pipeline's client (utils/markush_glyph_service.py) spawns this script
with the glyph venv's python, keeping the transformers-5.x dependency tree
fully isolated from the production llm environment.

Usage: python markushglyph_worker.py [--checkpoint SNAP] [--base-model ID]
"""
from __future__ import annotations

import argparse
import json
import sys

os_defaults = {
    "HF_HOME": "/data/LLM_Models/huggingface_cache",
    "HF_HUB_OFFLINE": "1",
}
for key, value in os_defaults.items():
    import os
    os.environ.setdefault(key, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=(
        "/data/LLM_Models/huggingface_cache/hub/"
        "models--EdisonScientific--MarkushGlyph/snapshots/"
        "2879b0380c2687a1bbdb2312ac4a810ed4887893"))
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    sys.path.insert(0, "/data/glyph_repo")
    from glyph.markush.inference import MarkushPredictor

    predictor = MarkushPredictor(
        checkpoint=args.checkpoint,
        base_model=args.base_model,
        device=args.device,
        dtype="bfloat16",
    )
    predictor.load()
    print(json.dumps({"event": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            print(json.dumps({"ok": False, "error": f"bad_request: {exc}"}), flush=True)
            continue
        image_path = str(request.get("image_path") or "")
        if not image_path:
            print(json.dumps({"ok": False, "error": "missing image_path"}), flush=True)
            continue
        try:
            prediction = predictor.predict(image_path)
            print(json.dumps({
                "ok": True,
                "cxsmiles": prediction.cxsmiles,
                "cxsmiles_opt": prediction.cxsmiles_opt,
                "raw": prediction.raw,
            }, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({
                "ok": False,
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            }, flush=True))


if __name__ == "__main__":
    main()
