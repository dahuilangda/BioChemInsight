"""Measure full-parameter training throughput on verified native-resolution images."""

import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.compute as pc
import torch
from molsight.checkpoint import load_parameters, model_configuration, model_state
from molsight.loss import Criterion
from molsight.model import MolsightModel
from molsight.optimization import supervised_update
from molsight.prepared import PreparedDataset, collate_characters, file_hash
from molsight.tokenizer import CharTokenizer


def measure(args, dataset, tokenizer, state, size, microbatch, checkpointing):
    dataset._open()
    table = dataset.table
    candidates = pc.indices_nonzero(
        pc.and_(
            pc.equal(table["category"], args.category),
            pc.equal(table["input_size"], size),
        )
    ).to_pylist()
    lengths = pc.list_value_length(table["token_ids"]).to_numpy()
    candidates.sort(key=lambda index: lengths[index])
    if len(candidates) < args.effective_batch:
        raise ValueError(
            "Insufficient verified peptide images at the requested resolution"
        )
    positions = np.linspace(
        len(candidates) // 2, len(candidates) - 1, args.effective_batch, dtype=int
    )
    if args.longest:
        positions = np.arange(len(candidates) - args.effective_batch, len(candidates))
    indices = [candidates[index] for index in positions]
    rows = [dataset[index] for index in indices]
    batches = [
        collate_characters(rows[start : start + microbatch])
        for start in range(0, len(rows), microbatch)
    ]
    for batch in batches:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.pin_memory()
    configuration = SimpleNamespace(
        **model_configuration(state),
        dec_n_head=8,
        max_len=2048,
        input_size=512,
        encoder="efficientvit",
        resume=True,
        lora=False,
        use_checkpoint=checkpointing,
        formats=["char"],
        regression=False,
        label_smoothing=0.1,
    )
    model = MolsightModel(configuration, tokenizer)
    load_parameters(
        model, state, source_vocabulary=tokenizer.stoi, target_vocabulary=tokenizer.stoi
    )
    model.train().to(args.device)
    model.image_encoder.eval()
    criterion = Criterion(configuration, tokenizer).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    result = {
        "resolution": size,
        "microbatch": microbatch,
        "effective_batch_per_gpu": args.effective_batch,
        "checkpointing": checkpointing,
        "record_indices": indices,
        "target_lengths": [len(row["label"]) for row in rows],
    }
    times = []
    torch.cuda.reset_peak_memory_stats(args.device)
    try:
        for step in range(args.warmup + args.steps):
            torch.cuda.synchronize(args.device)
            started = time.perf_counter()
            losses, norm, stepped = supervised_update(
                model,
                criterion,
                batches,
                optimizer,
                scheduler,
                scaler,
                args.device,
                1.0,
                amp=True,
            )
            torch.cuda.synchronize(args.device)
            if not stepped or not torch.isfinite(norm):
                raise FloatingPointError("Invalid throughput update")
            if step >= args.warmup:
                times.append(time.perf_counter() - started)
        median = float(np.median(times))
        result.update(
            status="passed",
            median_update_seconds=median,
            images_per_second=args.effective_batch / median,
            target_tokens_per_second=sum(len(row["label"]) - 1 for row in rows)
            / median,
            update_seconds=times,
            last_sequence_loss=losses["sequence"],
        )
        if args.trace and not checkpointing and microbatch == args.microbatches[-1]:
            trace_path = args.out / f"trace_{size}_{microbatch}.json"
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profiler:
                supervised_update(
                    model,
                    criterion,
                    batches,
                    optimizer,
                    scheduler,
                    scaler,
                    args.device,
                    1.0,
                    amp=True,
                )
                torch.cuda.synchronize(args.device)
            profiler.export_chrome_trace(str(trace_path))
            result["trace"] = str(trace_path.resolve())
    except torch.cuda.OutOfMemoryError:
        result["status"] = "out_of_memory"
    finally:
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(args.device)
        model.zero_grad(set_to_none=True)
        del model, optimizer, scheduler, criterion, scaler, batches
        gc.collect()
        torch.cuda.empty_cache()
    return result


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(args.device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    dataset = PreparedDataset(args.manifest, "train", max_len=2048)
    tokenizer = CharTokenizer(dataset.specification["vocabulary"])
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=False
    )
    if checkpoint["vocabulary"] != tokenizer.stoi:
        raise ValueError("Profiling requires the recorded supervised vocabulary")
    state = model_state(checkpoint)
    with (args.out / "gpu_activity.csv").open("x") as activity:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw",
                "--format=csv,noheader,nounits",
                "-lms",
                "200",
            ],
            stdout=activity,
        )
        results = []
        try:
            with (args.out / "measurements.jsonl").open("x") as stream:
                for size in args.sizes:
                    for mode in args.checkpoint_modes:
                        checkpointing = mode == "on"
                        for microbatch in args.microbatches:
                            if args.effective_batch % microbatch:
                                raise ValueError(
                                    "Microbatch must divide the per-GPU effective batch"
                                )
                            result = measure(
                                args,
                                dataset,
                                tokenizer,
                                state,
                                size,
                                microbatch,
                                checkpointing,
                            )
                            results.append(result)
                            stream.write(json.dumps(result) + "\n")
                            stream.flush()
                            print(json.dumps(result), flush=True)
                            if result["status"] == "out_of_memory":
                                break
        finally:
            monitor.terminate()
            monitor.wait()
    report = {
        "checkpoint_sha256": file_hash(args.checkpoint),
        "manifest_sha256": file_hash(args.manifest),
        "code_sha256": file_hash(__file__),
        "gpu_name": torch.cuda.get_device_name(args.device),
        "gpu_total_bytes": torch.cuda.get_device_properties(args.device).total_memory,
        "warmup_updates": args.warmup,
        "measured_updates": args.steps,
        "measurements": results,
        "scope": "Single-GPU full-parameter training with prefetched verified peptide images. Activation checkpointing and microbatch comparisons use identical inputs. These measurements exclude data-loader stalls and DDP communication and do not measure recognition quality or exact SM occupancy.",
        "category": args.category,
        "longest_targets": args.longest,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    for name in ("manifest", "checkpoint", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:0"))
    parser.add_argument("--sizes", nargs="+", type=int, default=[512, 768, 1024])
    parser.add_argument("--microbatches", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--effective-batch", type=int, default=8)
    parser.add_argument(
        "--category", choices=("complete", "peptide"), default="peptide"
    )
    parser.add_argument("--longest", action="store_true")
    parser.add_argument(
        "--checkpoint-modes", nargs="+", choices=("on", "off"), default=["on", "off"]
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if min(args.effective_batch, args.steps) < 1 or args.warmup < 0:
        parser.error("Invalid benchmark update counts")
    run(args)


if __name__ == "__main__":
    main()
