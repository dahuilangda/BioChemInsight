"""Compare optimizer-boundary restart states without integer-buffer norms."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def equal(first, second):
    if isinstance(first, torch.Tensor):
        return (
            isinstance(second, torch.Tensor)
            and first.dtype == second.dtype
            and first.shape == second.shape
            and torch.equal(first, second)
        )
    if isinstance(first, np.ndarray):
        return (
            isinstance(second, np.ndarray)
            and first.dtype == second.dtype
            and np.array_equal(first, second)
        )
    if isinstance(first, dict):
        return (
            isinstance(second, dict)
            and first.keys() == second.keys()
            and all(equal(value, second[key]) for key, value in first.items())
        )
    if isinstance(first, (tuple, list)):
        return (
            type(first) is type(second)
            and len(first) == len(second)
            and all(
                equal(left, right) for left, right in zip(first, second, strict=True)
            )
        )
    return type(first) is type(second) and first == second


def compare(first_path, second_path):
    first = torch.load(first_path, map_location="cpu", mmap=True, weights_only=False)
    second = torch.load(second_path, map_location="cpu", mmap=True, weights_only=False)
    checks = {
        key: equal(first[key], second[key])
        for key in (
            "model",
            "optimizer",
            "scheduler",
            "scaler",
            "rank_rng",
            "plan",
            "update",
            "vocabulary",
            "baseline",
            "validation_indices",
        )
    }
    changed, maximum, numerator, denominator = [], 0.0, 0.0, 0.0
    for key, value in first["model"].items():
        other = second["model"].get(key)
        if not equal(value, other):
            changed.append(key)
        if (
            key.startswith("decoder.")
            and other is not None
            and value.shape == other.shape
        ):
            delta = value.double() - other.double()
            maximum = max(maximum, delta.abs().max().item())
            numerator += delta.square().sum().item()
            denominator += value.double().square().sum().item()
    update = first["update"]
    predictions = [
        Path(path).parent / f"validation_{update:06d}.json"
        for path in (first_path, second_path)
    ]
    if all(path.is_file() for path in predictions):
        checks["validation_predictions"] = equal(
            json.loads(predictions[0].read_text()),
            json.loads(predictions[1].read_text()),
        )
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "different_model_tensors": len(changed),
        "different_model_keys": changed[:20],
        "decoder_max_abs_error": maximum,
        "decoder_relative_l2": (numerator / max(denominator, 1e-300)) ** 0.5,
        "checkpoints": [
            str(Path(path).resolve()) for path in (first_path, second_path)
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.first, args.second)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
