from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def namespace_from_config(config: dict[str, Any], checkpoint_args: dict[str, Any] | None = None) -> argparse.Namespace:
    defaults = {
        "encoder": "swin_base",
        "decoder": "transformer",
        "trunc_encoder": False,
        "no_pretrained": True,
        "use_checkpoint": True,
        "dropout": 0.5,
        "embed_dim": 256,
        "enc_pos_emb": False,
        "dec_num_layers": 6,
        "dec_hidden_size": 256,
        "dec_attn_heads": 8,
        "dec_num_queries": 128,
        "hidden_dropout": 0.1,
        "attn_dropout": 0.1,
        "max_relative_positions": 0,
        "continuous_coords": False,
        "compute_confidence": False,
        "input_size": 384,
        "vocab_file": None,
        "coord_bins": 64,
        "sep_xy": True,
        "test_file": "",
        "data_path": "",
        "coords_file": None,
        "pseudo_coords": False,
        "predict_coords": False,
        "augment": True,
        "mol_augment": False,
        "default_option": False,
        "shuffle_nodes": False,
        "include_condensed": True,
        "save_image": False,
        "save_path": str(Path(config.get("output_dir", "runs/dev"))),
        "mask_ratio": 0.0,
        "label_smoothing": 0.0,
        "formats": ["chartok_coords", "edges"],
    }
    if checkpoint_args:
        defaults.update(checkpoint_args)
    defaults.update(config)
    return argparse.Namespace(**defaults)
