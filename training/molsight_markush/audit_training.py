"""Reproduce objective and reward defects in an existing MolSight checkout."""

import argparse
import ast
import hashlib
import json
import platform
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs, rdBase


def definitions(path, names, scope):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    if {node.name for node in nodes} != set(names):
        raise ValueError(f"Missing definitions in {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return {node.name: node.lineno for node in nodes}


def audit(repository):
    loss_path = repository / "molsight/loss.py"
    scope = {"torch": torch, "nn": nn, "F": F}
    lines = definitions(loss_path, ("SmoothedKLDLoss", "SequenceLoss"), scope)
    smooth = scope["SmoothedKLDLoss"](0.1, 5, 0)
    distribution = smooth.uniform.clone()
    distribution.scatter_(1, torch.tensor([[1]]), smooth.confidence)
    mass = distribution.sum().item()

    sequence = scope["SequenceLoss"](0.0, 5, ignore_index=0, ignore_indices=[0, 3])
    logits = torch.zeros(1, 3, 5, requires_grad=True)
    sequence(logits, torch.tensor([[1, 3, 2]])).backward()
    masked_gradient = logits.grad[0, 0].abs().sum().item()

    reward_path = repository / "molsight/reward.py"
    rewards = {"torch": torch, "Chem": Chem, "DataStructs": DataStructs}
    reward_lines = definitions(reward_path, ("stereochem_reward_fn",), rewards)
    incorrect = rewards["stereochem_reward_fn"](["N[*:2]"], ["C[*:1]"]).item()
    correct = rewards["stereochem_reward_fn"](["C[*:1]"], ["C[*:1]"]).item()

    criterion = scope["SequenceLoss"](0.0, 5, ignore_index=0)
    first = torch.zeros(1, 4, 5)
    second = first.clone()
    second[:, :, 2] = 3.0
    target_a, target_b = torch.tensor([[1, 2, 0, 0]]), torch.tensor([[1, 2, 2, 2]])
    accumulated = (criterion(first, target_a) + criterion(second, target_b)) / 2
    combined = criterion(torch.cat((first, second)), torch.cat((target_a, target_b)))
    checks = [
        {
            "id": "normalized_smoothing",
            "passed": abs(mass - 1.0) < 1e-6,
            "observed_probability_mass": mass,
            "expected_probability_mass": 1.0,
            "file": str(loss_path),
            "line": lines["SmoothedKLDLoss"],
        },
        {
            "id": "ignored_token_gradient",
            "passed": masked_gradient < 1e-8,
            "observed_gradient_l1": masked_gradient,
            "expected_gradient_l1": 0.0,
            "file": str(loss_path),
            "line": lines["SequenceLoss"],
        },
        {
            "id": "attachment_reward_identity",
            "passed": incorrect < correct,
            "incorrect_structure_reward": incorrect,
            "correct_structure_reward": correct,
            "prediction": "N[*:2]",
            "target": "C[*:1]",
            "file": str(reward_path),
            "line": reward_lines["stereochem_reward_fn"],
        },
    ]
    source_files = (
        "train.py",
        "post_train.py",
        "evaluate.py",
        "molsight/loss.py",
        "molsight/reward.py",
        "molsight/dataset.py",
        "molsight/tokenizer.py",
        "molsight/augment.py",
        "molsight/model.py",
        "molsight/chemistry.py",
    )
    return {
        "schema_version": 1,
        "repository": str(repository.resolve()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "rdkit_version": rdBase.rdkitVersion,
        "source_sha256": {
            name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
            for name in source_files
        },
        "checks": checks,
        "observations": [
            {
                "id": "microbatch_normalization",
                "status": "objective_review_required",
                "mean_of_microbatch_losses": accumulated.item(),
                "combined_token_mean_loss": combined.item(),
                "equivalent": torch.allclose(accumulated, combined),
                "interpretation": "A mean of microbatch token means differs from the token mean of the combined batch when token counts differ.",
            }
        ],
        "status": "issues_found"
        if any(not check["passed"] for check in checks)
        else "checks_passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.repository)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": report["checks"]}))
    return 2 if report["status"] == "issues_found" else 0


if __name__ == "__main__":
    raise SystemExit(main())
