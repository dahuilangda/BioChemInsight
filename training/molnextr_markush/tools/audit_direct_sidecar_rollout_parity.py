#!/usr/bin/env python
"""Prove that greedy train-time rollout matches deployed direct Expert2 decode."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from utils.MolNexTR.model import molnextr, resize_small_image_to_long_edge  # noqa: E402
from utils.MolNexTR.moe import ATOM_FORMAT  # noqa: E402
from utils.MolNexTR.moe_confidence import fragment_graph_reward  # noqa: E402
from utils.MolNexTR.chemical import convert_graph_to_smiles  # noqa: E402
from training.molnextr_markush.tools.train_moe import (  # noqa: E402
    terminal_dummy_edge_expected_reward,
)


def prediction_smiles(prediction: dict) -> str:
    if prediction.get("decode_quality_issue"):
        return ""
    atoms = prediction.get(ATOM_FORMAT, {})
    smiles, _, _, issues = convert_graph_to_smiles(
        [atoms.get("coords") or []],
        [atoms.get("symbols") or []],
        [prediction.get("edges") or []],
        num_workers=1,
    )
    return "" if issues and issues[0] else (smiles[0] if smiles else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moe-config", required=True)
    parser.add_argument(
        "--base-checkpoint",
        default=str(REPO / "models/molnextr_best.pth"),
    )
    parser.add_argument(
        "--eval-csv",
        default=str(
            REPO
            / "training/molnextr_markush/data/real_wavy_hard_eval/eval.csv"
        ),
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with Path(args.eval_csv).open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)]
    row = next(
        (item for item in rows if item.get("structure_type") == "fragment"),
        None,
    )
    if row is None:
        raise ValueError("parity audit requires a fragment row")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = molnextr(
        args.base_checkpoint,
        device=device,
        moe_config_path=args.moe_config,
    )
    image = cv2.imread(row["file_path"])
    if image is None:
        raise FileNotFoundError(row["file_path"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_small_image_to_long_edge(image, runtime.preprocess_long_edge)
    tensor = runtime.transform(image=image, keypoints=[])["image"].unsqueeze(0)
    tensor = tensor.to(device)
    with torch.inference_mode():
        features, encoder_hiddens = runtime.encoder(tensor)
    features = features.clone().detach()

    moe = runtime.moe_decoder
    if moe is None:
        raise ValueError("parity audit requires a MoE config")
    constraints = moe._decode_constraints_for_expected("fragment", {}, device)
    with torch.inference_mode():
        deployed = moe.decode(
            features,
            hiddens=encoder_hiddens,
            beam_size=1,
            n_best=1,
            decode_constraints=constraints,
            expected_structure_types=["fragment"],
        )[0]
    with torch.enable_grad():
        rollout, sequence_log_prob, _, rollout_info = (
            moe.decode_direct_sidecar_rl(
                features.detach(),
                constraints,
                sidecar_expert_idx=2,
                sample=False,
                reference_kl=False,
                return_terminal_action_logits=True,
            )
        )
    boundary_pair = rollout_info.pop("_boundary_action_logits", None)
    final_pair = rollout_info.pop("_terminal_action_logits", None)
    final_stop_pair = rollout_info.pop("_terminal_stop_logits", None)
    rollout_info["boundary_star_minus_eos_logit"] = (
        None
        if boundary_pair is None
        else float((boundary_pair[1] - boundary_pair[0]).detach().item())
    )
    rollout_info["final_star_minus_eos_logit"] = (
        None
        if final_pair is None
        else float((final_pair[1] - final_pair[0]).detach().item())
    )
    rollout_info["final_stop_minus_continue_logit"] = (
        None
        if final_stop_pair is None
        else float(
            (final_stop_pair[1] - final_stop_pair[0]).detach().item()
        )
    )
    counterfactual = {}
    if (
        rollout_info["eos_found"]
        and rollout_info["generated_stars"] == 0
        and rollout_info["sequence"]
    ):
        star_token_id = int(constraints[ATOM_FORMAT]["star_token_id"])
        eos_step = len(rollout_info["sequence"]) - 1
        with torch.enable_grad():
            alternative, _, _, alternative_info = moe.decode_direct_sidecar_rl(
                features.detach(),
                constraints,
                sidecar_expert_idx=2,
                sample=False,
                reference_kl=False,
                forced_token_at_step=(eos_step, star_token_id),
            )
        greedy_reward = fragment_graph_reward(
            prediction_smiles(rollout),
            row.get("SMILES", ""),
        )
        alternative_reward = fragment_graph_reward(
            prediction_smiles(alternative),
            row.get("SMILES", ""),
        )
        counterfactual = {
            "eos_step": eos_step,
            "greedy_reward": greedy_reward,
            "alternative_reward": alternative_reward,
            "reward_delta": alternative_reward["reward"]
            - greedy_reward["reward"],
            "alternative_rollout": alternative_info,
        }
        edge_loss, best_edge_metrics, edge_audit = (
            terminal_dummy_edge_expected_reward(
                alternative,
                row.get("SMILES", ""),
            )
        )
        counterfactual["edge_search"] = {
            **edge_audit,
            "best_metrics": best_edge_metrics,
            "loss_finite": bool(
                edge_loss is not None
                and torch.isfinite(edge_loss.detach()).item()
            ),
        }

    deployed_atoms = deployed.get(ATOM_FORMAT, {})
    rollout_atoms = rollout.get(ATOM_FORMAT, {})
    comparisons = {
        "symbols": deployed_atoms.get("symbols") == rollout_atoms.get("symbols"),
        "coords": deployed_atoms.get("coords") == rollout_atoms.get("coords"),
        "indices": deployed_atoms.get("indices") == rollout_atoms.get("indices"),
        "token_smiles": deployed_atoms.get("smiles") == rollout_atoms.get("smiles"),
        "edges": deployed.get("edges") == rollout.get("edges"),
        "quality_issue": deployed.get("decode_quality_issue")
        == rollout.get("decode_quality_issue"),
    }
    report = {
        "schema_version": "molnextr_direct_sidecar_rollout_parity_v1",
        "source_id": row.get("source_id", ""),
        "moe_config": str(args.moe_config),
        "comparisons": comparisons,
        "passed": all(comparisons.values()),
        "rollout": rollout_info,
        "counterfactual": counterfactual,
        "sequence_log_prob_finite": bool(
            torch.isfinite(sequence_log_prob.detach()).item()
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
