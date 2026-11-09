"""Supervised adaptation from admitted recognition data."""

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from molsight.checkpoint import (
    capture_rng,
    load_parameters,
    model_configuration,
    model_state,
    restore_rng,
)
from molsight.cpu import loader_options
from molsight.evaluation import aggregate, compare
from molsight.inference import predict_images
from molsight.loss import Criterion, GRPOLoss
from molsight.model import MolsightModel
from molsight.metrics import TokenMetrics
from molsight.optimization import supervised_update, policy_update
from molsight.reward import tanimoto_reward_fn, stereochem_reward_fn
from molsight.prepared import PreparedDataset, collate_characters, file_hash
from molsight.sampling import DomainBatchSampler
from molsight.retention import CompleteAnchorSampler, RetentionObjective
from molsight.tokenizer import CharTokenizer
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


def runtime_signature():
    root = Path(__file__).resolve().parent
    sources = [Path(__file__).resolve(), *(root / "molsight").rglob("*.py")]
    driver = Path("/proc/driver/nvidia/version")
    devices = [
        torch.cuda.get_device_properties(index)
        for index in range(torch.cuda.device_count())
    ]
    return {
        "packages": {
            name: version(name)
            for name in (
                "torch",
                "numpy",
                "timm",
                "albumentations",
                "Pillow",
                "pyarrow",
                "rdkit",
                "SmilesPE",
                "opencv-python",
            )
        },
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "driver": driver.read_text().strip() if driver.is_file() else None,
        "devices": [
            {
                "name": device.name,
                "uuid": str(getattr(device, "uuid", None)),
                "capability": [device.major, device.minor],
            }
            for device in devices
        ],
        "sources": {
            str(path.relative_to(root)): file_hash(path) for path in sorted(sources)
        },
    }


def validation_indices(dataset, limits, peptide_tags=None):
    dataset._open()
    rows = dataset.table.select(
        ["record_key", "component", "category", "input_size", "peptide_stratum"]
    ).to_pylist()
    if peptide_tags is not None:
        for row, tags in zip(rows, dataset.metadata(peptide_tags), strict=True):
            row.update(tags)
    order = sorted(range(len(rows)), key=lambda index: rows[index]["record_key"])
    selected, groups, counts = [], set(), defaultdict(int)
    for index in order:
        row = rows[index]
        if (
            counts[row["category"]] >= limits.get(row["category"], 0)
            or row["component"] in groups
        ):
            continue
        selected.append(index)
        groups.add(row["component"])
        counts[row["category"]] += 1
    if any(not counts[category] for category in limits):
        raise ValueError("Validation lacks a required recognition domain")
    discovery_counts = defaultdict(int)
    for index in order:
        row = rows[index]
        if (
            not row.get("peptide_candidate")
            or row["category"] == "peptide"
            or row["category"] not in limits
        ):
            continue
        kind = (
            "discovered_cyclic"
            if row.get("cyclic_peptide_candidate")
            else "discovered_linear"
        )
        if row["component"] in groups or discovery_counts[kind] >= 32:
            continue
        selected.append(index)
        groups.add(row["component"])
        discovery_counts[kind] += 1
        row["evaluation_category"] = "discovered_peptide"
        row["peptide_stratum"] = kind
    return selected, rows


def evaluate(
    model,
    dataset,
    selected,
    metadata,
    *,
    device,
    rank,
    world_size,
    batch_size,
    precision,
):
    buckets = defaultdict(list)
    for index in selected:
        buckets[metadata[index]["input_size"]].append(index)
    batches = []
    for _, values in sorted(buckets.items()):
        local = values[rank::world_size]
        batches.extend(
            local[start : start + batch_size]
            for start in range(0, len(local), batch_size)
        )
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(
        dataset,
        batch_sampler=batches,
        collate_fn=collate_characters,
        num_workers=0,
        generator=generator,
    )
    rows = []
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            predictions = predict_images(model, batch["image"].to(device))
        for offset, index in enumerate(batch["idx"].tolist()):
            prediction = predictions["smiles"][offset]
            rows.append(
                {
                    "index": index,
                    "record_key": metadata[index]["record_key"],
                    "category": metadata[index].get(
                        "evaluation_category", metadata[index]["category"]
                    ),
                    "source_category": metadata[index]["category"],
                    "input_size": metadata[index]["input_size"],
                    "peptide_stratum": metadata[index]["peptide_stratum"],
                    "gold": batch["smiles"][offset],
                    "prediction": prediction,
                    "terminated": predictions["terminated"][offset],
                    **compare(batch["smiles"][offset], prediction),
                }
            )
    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, rows)
        rows = [row for part in gathered for row in part]
    if len(rows) != len(selected) or len({row["index"] for row in rows}) != len(
        selected
    ):
        raise ValueError("Validation predictions have missing or repeated records")
    return aggregate(rows), sorted(rows, key=lambda row: row["index"])


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    update,
    plan,
    vocabulary,
    baseline,
    selected,
    *,
    rank,
    world_size,
    device,
):
    rng = capture_rng(device)
    states = [rng]
    if world_size > 1:
        states = [None] * world_size
        dist.all_gather_object(states, rng)
    if rank == 0:
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "update": update,
                    "plan": plan,
                    "vocabulary": vocabulary,
                    "rank_rng": states,
                    "baseline": baseline,
                    "validation_indices": selected,
                },
                stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    if world_size > 1:
        dist.barrier()


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA runtime initialization failed")
    if not args.allow_nondeterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(not args.allow_nondeterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = not args.allow_nondeterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group("nccl", device_id=device)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=args.resume)
    if world_size > 1:
        dist.barrier()
    specification = json.loads(args.manifest.read_text())
    vocabulary = Path(specification["vocabulary"])
    if file_hash(vocabulary) != specification["vocabulary_sha256"]:
        raise ValueError("Vocabulary changed after dataset admission")
    tokenizer = CharTokenizer(vocabulary)
    training = PreparedDataset(args.manifest, "train", max_len=args.max_len)
    validation = PreparedDataset(args.manifest, "validation", max_len=args.max_len)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=False
    )
    state = model_state(checkpoint)
    configuration = SimpleNamespace(
        **model_configuration(state),
        dec_n_head=args.heads,
        max_len=args.max_len,
        input_size=512,
        encoder="efficientvit",
        resume=True,
        lora=False,
        use_checkpoint=not args.no_gradient_checkpointing,
        formats=["char"],
        regression=False,
        label_smoothing=args.label_smoothing,
        compile_layers=args.compile_layers,
    )
    model = MolsightModel(configuration, tokenizer)
    source_vocabulary = (
        checkpoint["vocabulary"]
        if args.resume
        else json.loads(args.source_vocab.read_text())
    )
    load_parameters(
        model,
        state,
        source_vocabulary=source_vocabulary,
        target_vocabulary=tokenizer.stoi,
    )
    model.image_encoder.requires_grad_(args.encoder_lr > 0)
    reference = None
    if args.retention_checkpoint is not None:
        reference_state = model_state(
            torch.load(
                args.retention_checkpoint,
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )
        )
        reference_tokenizer = CharTokenizer(args.retention_vocab)
        # The KL term compares full token distributions: identifiers must
        # align regardless of which parameters train.
        if any(
            tokenizer.stoi.get(token) != index
            for token, index in reference_tokenizer.stoi.items()
        ):
            raise ValueError(
                "Reference token identifiers differ from the student vocabulary"
            )
        reference_configuration = SimpleNamespace(**vars(configuration))
        for key, value in model_configuration(reference_state).items():
            setattr(reference_configuration, key, value)
        reference_configuration.compile_layers = False
        reference = MolsightModel(reference_configuration, reference_tokenizer)
        load_parameters(
            reference,
            reference_state,
            source_vocabulary=reference_tokenizer.stoi,
            target_vocabulary=reference_tokenizer.stoi,
        )
        current = model.state_dict()
        if any(
            key not in current or not torch.equal(value, current[key])
            for key, value in reference.state_dict().items()
            if not key.startswith("decoder.")
        ):
            raise ValueError(
                "Reference image features differ from the frozen training encoder"
            )
        if args.encoder_lr == 0:
            reference.image_encoder = model.image_encoder
        reference.requires_grad_(False).eval()
        del current, reference_state
    model.to(device)
    if reference is not None:
        reference.to(device)
    policy_reference = None
    if args.objective == "grpo":
        frozen_checkpoint = torch.load(
            args.policy_reference, map_location="cpu", mmap=True, weights_only=False
        )
        if frozen_checkpoint.get("vocabulary") != tokenizer.stoi:
            raise ValueError(
                "Policy reference must record the exact training vocabulary"
            )
        policy_reference = MolsightModel(configuration, tokenizer)
        load_parameters(
            policy_reference,
            model_state(frozen_checkpoint),
            source_vocabulary=tokenizer.stoi,
            target_vocabulary=tokenizer.stoi,
        )
        policy_reference.requires_grad_(False).eval().to(device)
        del frozen_checkpoint
    encoder_parameters = list(model.image_encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    parameter_groups = [
        {
            "params": [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad and id(parameter) not in encoder_ids
            ],
            "lr": args.lr,
        }
    ]
    if args.encoder_lr > 0:
        parameter_groups.append({"params": encoder_parameters, "lr": args.encoder_lr})
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def schedule(step):
        if step < args.warmup:
            return (step + 1) / max(1, args.warmup)
        fraction = min(1, (step - args.warmup) / max(1, args.updates - args.warmup))
        return 0.5 * (1 + math.cos(math.pi * fraction))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    start = checkpoint["update"] if args.resume else 0
    anchor_sampler, anchor_loader, retention = None, None, None
    if reference is not None:
        anchor_sampler = CompleteAnchorSampler(
            training,
            vocabulary_size=len(reference_tokenizer),
            updates=args.updates,
            batch_size=args.retention_batch_size,
            rank=rank,
            world_size=world_size,
            seed=args.seed,
            start_update=start,
        )
        anchor_loader = DataLoader(
            training,
            batch_sampler=anchor_sampler,
            collate_fn=collate_characters,
            num_workers=0,
            generator=torch.Generator().manual_seed(args.seed + rank + 0xA11CE),
        )
        retention = RetentionObjective(
            reference,
            weight=args.retention_weight,
            supervised_weight=args.retention_supervised_weight,
            temperature=args.retention_temperature,
        )
    training_metadata = training.metadata(args.peptide_tags)
    sampler = DomainBatchSampler(
        training_metadata,
        updates=args.updates,
        accumulation=args.accumulation,
        batch_size=args.batch_size,
        weights=json.loads(args.weights),
        rank=rank,
        world_size=world_size,
        seed=args.seed,
        start_update=start,
        excluded_domains=args.excluded_domains,
        peptide_group_weights=(
            json.loads(args.peptide_group_weights)
            if args.peptide_group_weights
            else None
        ),
        stratum_weights=(
            json.loads(args.stratum_weights) if args.stratum_weights else None
        ),
    )
    plan = {
        "retention": None
        if reference is None
        else {
            "checkpoint_sha256": file_hash(args.retention_checkpoint),
            "vocabulary_sha256": file_hash(args.retention_vocab),
            "sampler_sha256": anchor_sampler.definition_sha256,
            "anchor_records": len(anchor_sampler.indices),
            "weight": args.retention_weight,
            "supervised_weight": args.retention_supervised_weight,
            "temperature": args.retention_temperature,
            "batch_size": args.retention_batch_size,
        },
        "manifest_sha256": file_hash(args.manifest),
        "peptide_tags_sha256": file_hash(args.peptide_tags)
        if args.peptide_tags
        else None,
        "sampler_sha256": sampler.definition_sha256,
        "peptide_group_weights": sampler.peptide_group_weights,
        "configuration": vars(configuration),
        "world_size": world_size,
        "seed": args.seed,
        "updates": args.updates,
        "lr": args.lr,
        "encoder_lr": args.encoder_lr,
        "objective": args.objective,
        "policy": None
        if policy_reference is None
        else {
            "reference_sha256": file_hash(args.policy_reference),
            "samples": args.n_samples,
            "kl_beta": args.kl_beta,
            "clip_epsilon": args.clip_epsilon,
            "temperature": args.policy_temperature,
            "reward": "0.4_fingerprint_plus_0.6_exact_chemical_identity",
        },
        "warmup": args.warmup,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "vocabulary_sha256": file_hash(vocabulary),
        "retention_tolerance": args.retention_tolerance,
        "evaluation_batch_size": args.eval_batch_size,
        "evaluation_precision": args.eval_precision,
        "deterministic": not args.allow_nondeterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "runtime": runtime_signature(),
        "training_precision": "bf16",
        "initial_checkpoint_sha256": checkpoint["plan"]["initial_checkpoint_sha256"]
        if args.resume
        else file_hash(args.checkpoint),
    }
    selected, metadata = validation_indices(
        validation,
        {
            domain: 128 if domain == "complete" else 64
            for domain in json.loads(args.weights)
        },
        args.peptide_tags,
    )
    baseline = None
    if args.resume:
        if checkpoint["plan"] != plan or selected != checkpoint["validation_indices"]:
            raise ValueError("Resume changes the training plan or validation cohort")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        baseline = checkpoint["baseline"]
    if world_size > 1:
        distributed_model = DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False
        )
    else:
        distributed_model = model
    criterion = Criterion(configuration, tokenizer).to(device)
    policy_criterion = (
        GRPOLoss(args, reference_model=policy_reference)
        if policy_reference is not None
        else None
    )
    loader = DataLoader(
        training,
        batch_sampler=sampler,
        collate_fn=collate_characters,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed + rank),
        **loader_options(args.workers),
    )
    if baseline is None:
        baseline, rows = evaluate(
            reference if reference is not None else model,
            validation,
            selected,
            metadata,
            device=device,
            rank=rank,
            world_size=world_size,
            batch_size=args.eval_batch_size,
            precision=args.eval_precision,
        )
        if rank == 0:
            (args.out / "baseline.json").write_text(
                json.dumps(
                    {
                        "metrics": baseline,
                        "records": rows,
                        "peptide_strata": aggregate(rows, group_by="peptide_stratum"),
                        "input_resolutions": aggregate(rows, group_by="input_size"),
                    },
                    indent=2,
                )
                + "\n"
            )
    if args.resume:
        restore_rng(checkpoint["rank_rng"][rank], device)
    if rank == 0:
        (args.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        (args.out / "execution.json").write_text(
            json.dumps(
                {
                    "workers_per_rank": loader.num_workers,
                    "prefetch_factor": loader.prefetch_factor,
                    "world_size": world_size,
                    "start_update": start,
                    "checkpoint": str(args.checkpoint),
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps({"start_update": start, "baseline": baseline}), flush=True)
    iterator = iter(loader)
    anchor_iterator = iter(anchor_loader) if anchor_loader is not None else None
    started = time.monotonic()
    for update in range(start, args.updates):
        distributed_model.train(args.objective == "supervised")
        model.image_encoder.eval()
        batches = [next(iterator) for _ in range(args.accumulation)]
        token_metrics = None
        if policy_criterion is not None:
            losses, norm, stepped = policy_update(
                distributed_model,
                policy_criterion,
                batches,
                lambda predictions, targets: (
                    tanimoto_reward_fn(predictions, targets)
                    + stereochem_reward_fn(predictions, targets)
                ),
                optimizer,
                scheduler,
                scaler,
                device,
                args.max_grad_norm,
                amp=True,
            )
        else:
            token_metrics = TokenMetrics(
                criterion.criterion["sequence"].criterion.ignore_indices, device
            )
            losses, norm, stepped = supervised_update(
                distributed_model,
                criterion,
                batches,
                optimizer,
                scheduler,
                scaler,
                device,
                args.max_grad_norm,
                amp=True,
                retention=retention,
                anchor_batch=next(anchor_iterator)
                if anchor_iterator is not None
                else None,
                observer=token_metrics,
            )
        if not stepped:
            raise FloatingPointError("Training update did not advance the optimizer")
        record = {
            "update": update + 1,
            "losses": losses,
            "metrics": token_metrics.compute() if token_metrics is not None else {},
            "gradient_norm": float(norm),
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.monotonic() - started,
            "domain": sampler.domain_for_update(update),
            "rank0_source_categories": sorted(
                {
                    training_metadata[int(index)]["category"]
                    for batch in batches
                    for index in batch["idx"]
                }
            ),
            "input_resolutions": sorted(
                {batch["image"].shape[-1] for batch in batches}
            ),
            "global_images": sum(len(batch["idx"]) for batch in batches) * world_size,
            "replay_images": args.retention_batch_size * world_size
            if retention is not None
            else 0,
            "rank0_record_indices": [
                int(index) for batch in batches for index in batch["idx"]
            ],
        }
        if rank == 0:
            with (args.out / "updates.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if (update + 1) % 10 == 0:
                print(json.dumps(record), flush=True)
        evaluation = (update + 1) % args.eval_every == 0 or update + 1 == args.updates
        regression = False
        if evaluation:
            scores, rows = evaluate(
                model,
                validation,
                selected,
                metadata,
                device=device,
                rank=rank,
                world_size=world_size,
                batch_size=args.eval_batch_size,
                precision=args.eval_precision,
            )
            regression = (
                scores["complete"]["resolved_exact"]
                < baseline["complete"]["resolved_exact"] - args.retention_tolerance
            )
            if rank == 0:
                (args.out / f"validation_{update + 1:06d}.json").write_text(
                    json.dumps(
                        {
                            "metrics": scores,
                            "records": rows,
                            "peptide_strata": aggregate(
                                rows, group_by="peptide_stratum"
                            ),
                            "input_resolutions": aggregate(rows, group_by="input_size"),
                            "retention_regression": regression,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(
                    json.dumps(
                        {
                            "update": update + 1,
                            "validation": scores,
                            "retention_regression": regression,
                        }
                    ),
                    flush=True,
                )
        stopping = args.stop_after is not None and update + 1 >= args.stop_after
        if evaluation or (update + 1) % args.save_every == 0 or regression or stopping:
            # A regressing checkpoint must never look resumable: keep it for
            # forensics under an explicit suffix and stop before it can be
            # picked up by "resume from latest".
            name = (
                f"update_{update + 1:06d}.regressed.pth"
                if regression
                else f"update_{update + 1:06d}.pth"
            )
            save_checkpoint(
                args.out / name,
                model,
                optimizer,
                scheduler,
                scaler,
                update + 1,
                plan,
                tokenizer.stoi,
                baseline,
                selected,
                rank=rank,
                world_size=world_size,
                device=device,
            )
        if regression:
            raise RuntimeError(
                "Complete-molecule retention fell below the configured tolerance"
            )
        if stopping:
            break
    if world_size > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    for name in ("manifest", "checkpoint", "source-vocab", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--peptide-tags", type=Path)
    parser.add_argument("--excluded-domains", nargs="*", default=[])
    parser.add_argument("--peptide-group-weights")
    parser.add_argument("--stratum-weights")
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--encoder-lr", type=float, default=0.0)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--objective", choices=("supervised", "grpo"), default="supervised"
    )
    parser.add_argument("--policy-reference", type=Path)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--kl-beta", type=float, default=0.04)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1)
    parser.add_argument("--retention-tolerance", type=float, default=0.03)
    parser.add_argument("--retention-checkpoint", type=Path)
    parser.add_argument("--retention-vocab", type=Path)
    parser.add_argument("--retention-batch-size", type=int, default=1)
    parser.add_argument("--retention-weight", type=float, default=1.0)
    parser.add_argument("--retention-supervised-weight", type=float, default=0.25)
    parser.add_argument("--retention-temperature", type=float, default=2.0)
    parser.add_argument(
        "--weights",
        default='{"complete":0.4,"markush":0.35,"attachment":0.1,"peptide":0.15}',
    )
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--allow-nondeterministic", action="store_true")
    parser.add_argument("--compile-layers", action="store_true")
    args = parser.parse_args()
    if args.objective == "grpo":
        if not math.isfinite(args.policy_temperature) or args.policy_temperature <= 0:
            parser.error("Policy temperature must be finite and positive")
        if args.policy_reference is None or args.retention_checkpoint is not None:
            parser.error(
                "GRPO requires an independent policy reference and does not support supervised anchor replay"
            )
        if (
            args.n_samples < 2
            or not math.isfinite(args.kl_beta)
            or args.kl_beta < 0
            or not math.isfinite(args.clip_epsilon)
            or not 0 < args.clip_epsilon < 1
        ):
            parser.error("Invalid GRPO sampling or objective settings")
    elif args.policy_reference is not None:
        parser.error("A policy reference requires the GRPO objective")
    if not math.isfinite(args.encoder_lr) or args.encoder_lr < 0:
        parser.error("Encoder learning rate must be finite and nonnegative")
    if (args.retention_checkpoint is None) != (args.retention_vocab is None):
        parser.error("Retention requires a reference checkpoint and vocabulary")
    if (
        args.retention_batch_size < 1
        or not math.isfinite(args.retention_temperature)
        or args.retention_temperature <= 0
        or any(
            not math.isfinite(value) or value < 0
            for value in (args.retention_weight, args.retention_supervised_weight)
        )
    ):
        parser.error("Invalid retention configuration")
    if (
        min(
            args.heads,
            args.updates,
            args.batch_size,
            args.eval_batch_size,
            args.accumulation,
            args.max_len,
            args.eval_every,
            args.save_every,
        )
        < 1
    ):
        parser.error("Training dimensions and intervals must be positive")
    if not 0 <= args.warmup <= args.updates or args.lr <= 0 or args.max_grad_norm <= 0:
        parser.error("Invalid warmup, learning rate, or gradient limit")
    if (
        not all(
            math.isfinite(value)
            for value in (
                args.lr,
                args.weight_decay,
                args.label_smoothing,
                args.max_grad_norm,
                args.retention_tolerance,
            )
        )
        or args.weight_decay < 0
        or not 0 <= args.label_smoothing <= 1
        or not 0 <= args.retention_tolerance <= 1
    ):
        parser.error("Invalid optimizer, smoothing, or retention settings")
    if args.stop_after is not None and not 1 <= args.stop_after <= args.updates:
        parser.error("Stop position must fall within the training plan")
    if args.compile_layers and not args.allow_nondeterministic:
        parser.error("Compiled layers require the nondeterministic execution option")
    run(args)


if __name__ == "__main__":
    main()
