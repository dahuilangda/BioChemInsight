"""Exercise a full-parameter policy update on recorded recognition images."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from molsight.checkpoint import load_parameters, model_configuration, model_state
from molsight.loss import GRPOLoss
from molsight.model import MolsightModel
from molsight.optimization import policy_update
from molsight.prepared import ImageReader, file_hash
from molsight.reward import stereochem_reward_fn, tanimoto_reward_fn
from molsight.tokenizer import CharTokenizer


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(args):
    torch.manual_seed(42)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=False
    )
    tokenizer = CharTokenizer(args.vocab)
    if checkpoint["vocabulary"] != tokenizer.stoi:
        raise ValueError("Policy preflight requires the recorded supervised vocabulary")
    state = model_state(checkpoint)
    configuration = SimpleNamespace(
        **model_configuration(state),
        dec_n_head=8,
        max_len=2048,
        input_size=512,
        encoder="efficientvit",
        resume=True,
        lora=False,
        use_checkpoint=True,
        formats=["char"],
        regression=False,
    )
    model = MolsightModel(configuration, tokenizer)
    load_parameters(
        model, state, source_vocabulary=tokenizer.stoi, target_vocabulary=tokenizer.stoi
    )
    model.requires_grad_(True).eval()
    reference = copy.deepcopy(model).requires_grad_(False).eval()
    initial_reference = state_digest(reference)
    encoder_before = next(model.image_encoder.parameters()).detach().clone()
    decoder_before = model.decoder.token_embedding.weight.detach().clone()
    model.to(args.device)
    reference.to(args.device)
    criterion = GRPOLoss(
        SimpleNamespace(
            n_samples=4,
            kl_beta=0.04,
            clip_epsilon=0.2,
            policy_temperature=args.temperature,
        ),
        reference_model=reference,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=1
    )
    reader = ImageReader()
    pairs = json.loads(args.pairs.read_text())
    batches = [
        {"image": reader(pair, 512)[None], "smiles": [pair["SMILES"]]} for pair in pairs
    ]

    def reward(predictions, targets):
        return tanimoto_reward_fn(predictions, targets) + stereochem_reward_fn(
            predictions, targets
        )

    metrics, norm, stepped = policy_update(
        model,
        criterion,
        batches,
        reward,
        optimizer,
        scheduler,
        torch.amp.GradScaler("cuda", enabled=False),
        args.device,
        1.0,
        amp=True,
    )
    encoder_delta = (
        (next(model.image_encoder.parameters()).detach().cpu() - encoder_before)
        .abs()
        .max()
        .item()
    )
    decoder_delta = (
        (model.decoder.token_embedding.weight.detach().cpu() - decoder_before)
        .abs()
        .max()
        .item()
    )
    if (
        not stepped
        or not torch.isfinite(norm)
        or state_digest(reference) != initial_reference
    ):
        raise RuntimeError("Policy update or reference immutability check failed")
    result = {
        "checkpoint_sha256": file_hash(args.checkpoint),
        "pairs_sha256": hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
        "vocabulary_sha256": hashlib.sha256(args.vocab.read_bytes()).hexdigest(),
        "metrics": metrics,
        "gradient_norm": float(norm),
        "encoder_max_abs_change": encoder_delta,
        "decoder_max_abs_change": decoder_delta,
        "reference_unchanged": True,
        "optimizer_updates": 1,
        "temperature": args.temperature,
        "effective_update": encoder_delta > 0 and decoder_delta > 0,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
        "scope": "Production execution check, not a capability evaluation. No checkpoint is promoted.",
    }
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "vocab", "pairs", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:0"))
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(run(args)))


if __name__ == "__main__":
    main()
