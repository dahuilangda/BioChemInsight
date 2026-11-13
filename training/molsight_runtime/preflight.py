"""Validate a production checkpoint against actual recognition targets."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from molsight.checkpoint import model_configuration, load_parameters, model_state
from molsight.dataset import get_transforms
from molsight.loss import Criterion
from molsight.model import MolsightModel
from molsight.structure import parse_structure
from molsight.tokenizer import CharTokenizer
from PIL import Image


def validate(
    checkpoint_path,
    source_vocabulary,
    target_vocabulary,
    pairs,
    *,
    device,
    max_len,
    input_size,
    train_encoder=False,
):
    tokenizer = CharTokenizer(target_vocabulary)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    state = model_state(checkpoint)
    configuration = model_configuration(state)
    args = SimpleNamespace(
        **configuration,
        dec_n_head=8,
        max_len=max_len,
        input_size=input_size,
        encoder="efficientvit",
        resume=True,
        lora=False,
        use_checkpoint=True,
        formats=["char"],
        label_smoothing=0.1,
        regression=False,
    )
    model = MolsightModel(args, tokenizer)
    load_parameters(
        model,
        state,
        source_vocabulary=json.loads(source_vocabulary.read_text()),
        target_vocabulary=tokenizer.stoi,
    )
    model.image_encoder.requires_grad_(train_encoder)
    model.train()
    model.image_encoder.eval()
    model.to(device)
    criterion = Criterion(args, tokenizer).to(device)
    transform = get_transforms(args, augment=False, rotate=False)
    results = []
    for pair in pairs:
        label = parse_structure(pair["SMILES"]).model_text
        ids, indices = tokenizer.smiles_to_sequence(label)
        if len(ids) > max_len:
            raise ValueError("Target exceeds the model context")
        image = np.asarray(Image.open(pair["file_path"]).convert("RGB"))
        image = transform(image=image, keypoints=[])["image"][None].to(device)
        batch = {
            "image": image,
            "label": torch.tensor([ids], device=device),
            "atom_indices": torch.tensor([indices], device=device),
            "edges": torch.full((1, len(indices), len(indices)), -100, device=device),
            "coords": torch.full((1, len(indices), 2), -100, device=device),
        }
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = model(**batch)
            losses = criterion(**predictions, **batch)
        loss = sum(losses.values())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float("inf"), error_if_nonfinite=True
        )
        encoder_norm = (
            torch.linalg.vector_norm(
                torch.stack(
                    [
                        parameter.grad.detach().float().norm()
                        for parameter in model.image_encoder.parameters()
                        if parameter.grad is not None
                    ]
                )
            )
            if train_encoder
            else torch.tensor(0.0)
        )
        if train_encoder and (not torch.isfinite(encoder_norm) or encoder_norm <= 0):
            raise FloatingPointError("Encoder gradients are missing or nonfinite")
        if not torch.isfinite(loss) or norm <= 0:
            raise FloatingPointError(
                "Production model has an invalid objective or missing gradients"
            )
        torch.cuda.synchronize(device)
        results.append(
            {
                "file_path": pair["file_path"],
                "token_length": len(ids),
                "sequence_loss": float(loss.detach()),
                "gradient_norm": float(norm),
                "encoder_gradient_norm": float(encoder_norm),
                "seconds": time.monotonic() - started,
            }
        )
    return {
        "checkpoint": str(checkpoint_path),
        "configuration": vars(args),
        "source_vocabulary_sha256": hashlib.sha256(
            source_vocabulary.read_bytes()
        ).hexdigest(),
        "target_vocabulary_sha256": hashlib.sha256(
            target_vocabulary.read_bytes()
        ).hexdigest(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "examples": results,
        "optimizer_updates": 0,
        "train_encoder": train_encoder,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-vocab", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--train-encoder", action="store_true")
    args = parser.parse_args()
    result = validate(
        args.checkpoint,
        args.source_vocab,
        args.vocab,
        json.loads(args.pairs.read_text()),
        device=torch.device(args.device),
        max_len=args.max_len,
        input_size=args.input_size,
        train_encoder=args.train_encoder,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
