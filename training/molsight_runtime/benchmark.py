"""Measure unassisted image recognition on a frozen, traceable sample."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from molsight.abbreviations import REGISTRY_SHA256
from molsight.checkpoint import load_parameters, model_configuration, model_state
from molsight.cpu import loader_options
from molsight.inference import predict_images
from molsight.evaluation import aggregate, compare
from molsight.model import MolsightModel
from molsight.prepared import ImageReader
from molsight.tokenizer import CharTokenizer
from torch.utils.data import DataLoader, Dataset


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class RecognitionImages(Dataset):
    def __init__(self, pairs, input_size):
        self.pairs = pairs
        self.input_size = input_size
        self.reader = ImageReader()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return index, self.reader(self.pairs[index], self.input_size)


def checkpoint_formats(checkpoint, state):
    """Keep trained auxiliary heads present while decoding SMILES only."""
    actual = ['char']
    if any(key.startswith('decoder.edge_predictor.') for key in state):
        actual.append('edges')
    if any(key.startswith('decoder.loc_predictor.') for key in state):
        raise ValueError('Coordinate-head benchmarking needs its recorded bin configuration')
    declared = checkpoint.get('plan', {}).get('configuration', {}).get('formats')
    if declared is not None and set(declared) != set(actual):
        raise ValueError('Recorded task formats contradict checkpoint parameters')
    return actual


def run(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=False)
    pairs = json.loads(args.pairs.read_text())
    if not pairs:
        raise ValueError("Recognition benchmark requires nonempty gold data")
    tokenizer = CharTokenizer(args.vocab)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    state = model_state(checkpoint)
    configuration = SimpleNamespace(
        **model_configuration(state),
        dec_n_head=args.heads,
        max_len=args.max_len,
        input_size=args.input_size,
        encoder="efficientvit",
        resume=True,
        lora=False,
        use_checkpoint=False,
        formats=checkpoint_formats(checkpoint, state),
        regression=False,
    )
    model = MolsightModel(configuration, tokenizer)
    load_parameters(
        model,
        state,
        source_vocabulary=json.loads(args.source_vocab.read_text()),
        target_vocabulary=tokenizer.stoi,
    )
    model.requires_grad_(False).eval().to(args.device)
    loader = DataLoader(
        RecognitionImages(pairs, args.input_size),
        batch_size=args.batch_size,
        pin_memory=True,
        **loader_options(args.workers),
    )
    timings, rows = [], []
    started = time.monotonic()
    with (args.out / "predictions.jsonl").open("x") as stream:
        for indices, images in loader:
            images = images.to(args.device, non_blocking=True)
            torch.cuda.synchronize(args.device)
            begin = time.monotonic()
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"
            ):
                predictions = predict_images(model, images)
            torch.cuda.synchronize(args.device)
            timings.append(time.monotonic() - begin)
            for offset, index in enumerate(indices.tolist()):
                pair = pairs[index]
                row = {
                    "index": index,
                    "record_key": pair.get("record_key"),
                    "category": pair["category"],
                    "gold": pair["SMILES"],
                    "prediction": predictions["smiles"][offset],
                    "tokens": len(predictions["tokens"][offset]),
                    "terminated": predictions["terminated"][offset],
                    **compare(pair["SMILES"], predictions["smiles"][offset]),
                }
                rows.append(row)
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {"examples": len(rows), "batch_seconds": round(timings[-1], 3)}
                ),
                flush=True,
            )
    metrics = aggregate(rows)
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": digest(args.checkpoint),
        "pairs_sha256": digest(args.pairs),
        "source_vocab_sha256": digest(args.source_vocab),
        "vocab_sha256": digest(args.vocab),
        "abbreviation_registry_sha256": REGISTRY_SHA256,
        "configuration": vars(configuration),
        "examples": len(rows),
        "metrics": metrics,
        "batch_seconds": timings,
        "warmup_batches": 1,
        "seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
        "optimizer_updates": 0,
        "precision": args.precision,
        "deterministic": True,
        "allow_tf32": False,
        "auxiliary_formats_loaded": configuration.formats,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "source-vocab", "vocab", "pairs", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:0"))
    args = parser.parse_args()
    if min(args.heads, args.input_size, args.max_len, args.batch_size) < 1:
        parser.error("Model and batch dimensions must be positive")
    print(json.dumps(run(args)))


if __name__ == "__main__":
    main()
