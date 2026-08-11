#!/usr/bin/env python
"""Train the calibrated confidence head (E[Tanimoto]) on a frozen MoE.

The MoE decoder computes ``pred["confidence"]`` only when ``confidence_head``
is present, but NO production checkpoint in experiments/moe ever enabled it
(``use_confidence_head: false`` everywhere), so MOLNEXTR_CONFIDENCE was always
-1.0 and the assembly confidence gate (markush_assembly.py) had no signal.

This script trains the small ConfidenceHead on top of a FROZEN deployed MoE
(encoder + experts + router untouched) so the calibrated confidence becomes a
reliable backbone-correctness proxy for the precision-first assembly gate:

  - feature extraction pass: decode each training image once with the frozen
    model (production path: expected structure type forces the router), and
    record the pooled encoder features + forced gate weights;
  - target: dummy-stripped Morgan Tanimoto(pred, gold) — the exact signal the
    gate must predict (backbone correctness of the decoded graph);
  - the inference-time token/edge NLL fallbacks (5.0, 2.0) are used at train
    time too, because the direct_sidecar decode emits no atom/edge scores;
  - train the head with the ordinal-smoothed bin CE (confidence_loss), report
    calibration (ECE + threshold table) on a held-out split;
  - write moe_confidence.pt + a copy of the source moe_config.json with
    use_confidence_head=true and confidence_path set, ready to deploy.

Usage (llm env, GPU):
  python training/molnextr_markush/tools/train_confidence_head.py \
      --moe-config experiments/moe/production/moe_config.json \
      --df-cache experiments/moe/bond_finetune_cache.parquet \
      --out-dir experiments/moe/confidence_head_output
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from utils.MolNexTR.moe_confidence import (  # noqa: E402
    ConfidenceHead,
    confidence_loss,
)
from utils.MolNexTR.model import resize_small_image_to_long_edge  # noqa: E402
from utils.structure_recognition import _load_molnextr  # noqa: E402

LABEL_TO_TYPE = {0: "complete", 1: "markush", 2: "fragment"}
TYPE_TO_LABEL = {name: idx for idx, name in LABEL_TO_TYPE.items()}

# Inference-time fallbacks used by MoEDecoder._attach_confidence when the
# sidecar decode emits no atom/edge scores (moe.py). The head must be trained
# with the same constants so train/inference inputs match.
FALLBACK_TOKEN_NLL = 5.0
FALLBACK_EDGE_NLL = 2.0


def strip_dummy_mol(mol: Chem.Mol) -> Chem.Mol | None:
    if mol is None:
        return None
    editable = Chem.RWMol(mol)
    dummy_indices = [
        atom.GetIdx()
        for atom in editable.GetAtoms()
        if atom.GetAtomicNum() == 0 or "*" in atom.GetSymbol()
    ]
    for atom_index in sorted(dummy_indices, reverse=True):
        editable.RemoveAtom(atom_index)
    stripped = editable.GetMol()
    if stripped.GetNumAtoms() == 0:
        return None
    try:
        Chem.SanitizeMol(stripped)
    except Exception:
        try:
            stripped.UpdatePropertyCache(strict=False)
        except Exception:
            return None
    return stripped


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is not None:
        return mol
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False) if smiles else None
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def backbone_tanimoto(pred_smiles: str, gold_smiles: str) -> float:
    """Dummy-stripped Morgan Tanimoto — the assembly-gate correctness signal."""
    pred = mol_from_smiles(pred_smiles)
    gold = mol_from_smiles(gold_smiles)
    if pred is None or gold is None:
        return 0.0
    pred = strip_dummy_mol(pred)
    gold = strip_dummy_mol(gold)
    if pred is None or gold is None:
        return 0.0
    try:
        fp_p = AllChem.GetMorganFingerprintAsBitVect(pred, 2, nBits=1024)
        fp_g = AllChem.GetMorganFingerprintAsBitVect(gold, 2, nBits=1024)
    except Exception:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp_p, fp_g))


def load_rows(df_path: str, max_rows_per_label: dict[str, int], seed: int,
              extra_fragment_csvs: list[str] | None = None) -> list[dict]:
    import pandas as pd
    rng = random.Random(seed)
    df = pd.read_parquet(df_path, columns=[
        "file_path", "SMILES", "structure_type_label", "image_domain",
    ])
    rows = []
    seen = set()
    for label_id, type_name in LABEL_TO_TYPE.items():
        cap = int(max_rows_per_label.get(type_name, 0) or 0)
        if cap <= 0:
            continue
        sub = df[df["structure_type_label"] == label_id]
        # Real patent crops first (they dominate the production distribution).
        real = sub[sub["image_domain"] == "real_original"]
        synth = sub[sub["image_domain"] != "real_original"]
        candidates = []
        for part in (real, synth):
            paths = part["file_path"].tolist()
            smiles = part["SMILES"].tolist()
            candidates.extend(
                {"file_path": p, "SMILES": s, "type": type_name}
                for p, s in zip(paths, smiles)
                if os.path.isfile(p)
            )
        rng.shuffle(candidates)
        taken = []
        for row in candidates:
            if len(taken) >= cap:
                break
            if row["file_path"] in seen:
                continue
            seen.add(row["file_path"])
            taken.append(row)
        print(f"{type_name}: {len(taken)} rows "
              f"({sum(1 for r in taken if 'real_original' in r['file_path'])} real)")
        rows.extend(taken)
    # Fragment rows are absent from the big df-cache on this machine (391k
    # unmounted images); merge real fragment sources with images instead.
    for csv_path in extra_fragment_csvs or []:
        if not os.path.exists(csv_path):
            print(f"warning: extra fragment csv not found: {csv_path}")
            continue
        extra = (
            pd.read_parquet(csv_path)
            if str(csv_path).endswith(".parquet")
            else pd.read_csv(csv_path)
        )
        if not {"file_path", "SMILES"}.issubset(extra.columns):
            print(f"warning: {csv_path} lacks file_path/SMILES columns")
            continue
        for p, s in zip(extra["file_path"].tolist(), extra["SMILES"].tolist()):
            if not os.path.isfile(p) or p in seen:
                continue
            seen.add(p)
            rows.append({"file_path": p, "SMILES": s, "type": "fragment"})
    rng.shuffle(rows)
    return rows


def _row_nlls(pred: dict) -> tuple[float, float]:
    """Token/edge NLLs exactly like MoEDecoder._attach_confidence computes at
    inference (real atom_scores/edge_scores when present, else the same
    fallbacks). The head must be trained with the SAME inputs inference uses —
    the earlier head trained on fallback-only inputs was miscalibrated."""
    cc = pred.get("chartok_coords") or {}
    atom_scores = cc.get("atom_scores") or []
    if atom_scores:
        token_nll = float(np.mean([-math.log(max(float(s), 1e-6)) for s in atom_scores]))
    else:
        token_nll = FALLBACK_TOKEN_NLL
    edge_scores = pred.get("edge_scores") or {}
    if edge_scores:
        edge_nll = float(np.mean([-math.log(max(float(v), 1e-6)) for v in edge_scores.values()]))
    else:
        edge_nll = FALLBACK_EDGE_NLL
    return token_nll, edge_nll


def _compute_scalars(pred: dict, fallback_t: float = FALLBACK_TOKEN_NLL,
                     fallback_e: float = FALLBACK_EDGE_NLL):
    """Compute the scalar confidence features from a decoded prediction dict,
    matching MoEDecoder._attach_confidence exactly."""
    cc = pred.get("chartok_coords") or {}
    atom_scores = cc.get("atom_scores") or []
    edge_scores = pred.get("edge_scores") or {}
    num_tokens = len(atom_scores) if atom_scores else 1
    if atom_scores:
        token_probs = [max(float(s), 1e-6) for s in atom_scores]
        token_nll = float(np.mean([-math.log(s) for s in token_probs]))
        avg_token = float(np.prod(token_probs) ** (1.0 / len(token_probs)))
        pos_entropy = float(np.mean([-s * math.log(s) for s in token_probs]))
    else:
        token_nll = fallback_t
        avg_token = 0.0
        pos_entropy = math.log(100)
    if edge_scores:
        edge_probs = [max(float(v), 1e-6) for v in edge_scores.values()]
        edge_nll = float(np.mean([-math.log(s) for s in edge_probs]))
        edge_geom = float(np.prod(edge_probs) ** (1.0 / max(1, len(edge_probs))))
    else:
        edge_nll = fallback_e
        edge_geom = 0.0
    overall_score = avg_token * (edge_geom ** 0.5) if edge_geom > 0 else avg_token
    return token_nll, edge_nll, overall_score, pos_entropy, float(num_tokens)


def forward_features(model, batch_rows: list[dict], batch_size: int):
    """Decode a batch through the frozen production path; return pooled
    features, SOFT gate weights, predicted SMILES, and all scalar features."""
    import cv2
    pooled_all, gate_all, smiles_all = [], [], []
    token_nll_all, edge_nll_all = [], []
    overall_all, posent_all, ntok_all = [], [], []
    for start in range(0, len(batch_rows), batch_size):
        chunk = batch_rows[start:start + batch_size]
        images = []
        for row in chunk:
            img = cv2.imread(row["file_path"])
            if img is None:
                images.append(None)
                continue
            img = resize_small_image_to_long_edge(img, model.preprocess_long_edge)
            images.append(model.transform(image=img, keypoints=[])["image"])
        valid = [i for i, im in enumerate(images) if im is not None]
        if not valid:
            continue
        tensor = torch.stack([images[i] for i in valid]).to(model.device)
        types = [batch_rows[i]["type"] for i in valid]
        model.moe_decoder.compute_confidence = True
        for _expert in model.moe_decoder.experts:
            _expert.compute_confidence = True
        with torch.inference_mode():
            features, hiddens = model.encoder(tensor)
            # Get SOFT gate weights from the router (not forced one-hot)
            router_out = model.moe_decoder.router(features) if hasattr(model.moe_decoder, 'router') else None
            if router_out is not None and hasattr(router_out, 'dim'):
                soft_gates = torch.softmax(router_out.float(), dim=-1).cpu().numpy()
            elif isinstance(router_out, dict) and 'weights' in router_out:
                soft_gates = router_out['weights'].float().cpu().numpy()
            else:
                # fallback: use one-hot from expected type
                soft_gates = np.zeros((len(valid), 3), dtype=np.float32)
                for i, t in enumerate(types):
                    soft_gates[i][TYPE_TO_LABEL[t]] = 1.0
            preds = model.moe_decoder.decode(
                features, hiddens,
                expected_structure_types=types,
            )
            pooled = features.mean(dim=1).float().cpu().numpy()
        from utils.MolNexTR.chemical import convert_graph_to_smiles
        ok_preds = [
            pred for pred in preds
            if isinstance(pred, dict) and not pred.get("decode_quality_issue")
        ]
        smiles_list = []
        if ok_preds:
            smiles_list, _molblocks, _success, _issues = convert_graph_to_smiles(
                [pred["chartok_coords"]["coords"] for pred in ok_preds],
                [pred["chartok_coords"]["symbols"] for pred in ok_preds],
                [pred["edges"] for pred in ok_preds],
                images=[tensor[i] for i, pred in enumerate(preds) if pred in ok_preds],
                num_workers=1,
            )
        smiles_iter = iter(smiles_list)
        for local_idx, row_index in enumerate(valid):
            pred = preds[local_idx]
            if isinstance(pred, dict) and not pred.get("decode_quality_issue"):
                smiles = str(next(smiles_iter, "") or "")
                token_nll, edge_nll, overall, posent, ntok = _compute_scalars(pred)
            else:
                smiles = ""
                token_nll, edge_nll = FALLBACK_TOKEN_NLL, FALLBACK_EDGE_NLL
                overall, posent, ntok = 0.0, math.log(100), 1.0
            pooled_all.append(pooled[local_idx])
            gate_all.append(soft_gates[local_idx] if local_idx < len(soft_gates) else np.zeros(3, dtype=np.float32))
            smiles_all.append(smiles)
            token_nll_all.append(token_nll)
            edge_nll_all.append(edge_nll)
            overall_all.append(overall)
            posent_all.append(posent)
            ntok_all.append(ntok)
    return pooled_all, gate_all, smiles_all, token_nll_all, edge_nll_all, overall_all, posent_all, ntok_all


def evaluate_head(head: torch.nn.Module, pooled, gates, targets, threshold: float,
                 token_nlls=None, edge_nlls=None, overalls=None, posents=None, ntoks=None) -> dict:
    head.eval()
    with torch.inference_mode():
        x_pooled = torch.tensor(np.asarray(pooled), dtype=torch.float32)
        x_gates = torch.tensor(np.asarray(gates), dtype=torch.float32)
        bs = len(pooled)
        token_nll = torch.tensor(np.asarray(token_nlls, dtype=np.float32) if token_nlls else [FALLBACK_TOKEN_NLL]*bs)
        edge_nll = torch.tensor(np.asarray(edge_nlls, dtype=np.float32) if edge_nlls else [FALLBACK_EDGE_NLL]*bs)
        overall = torch.tensor(np.asarray(overalls, dtype=np.float32) if overalls else [0.0]*bs)
        posent = torch.tensor(np.asarray(posents, dtype=np.float32) if posents else [0.0]*bs)
        ntok = torch.tensor(np.asarray(ntoks, dtype=np.float32) if ntoks else [1.0]*bs)
        conf = head.expected_confidence(head(x_pooled, x_gates, token_nll, edge_nll, overall, posent, ntok)).numpy()
    targets = np.asarray(targets, dtype=np.float32)
    kept = conf >= threshold
    wrong_kept = int(np.sum(kept & (targets < 0.8))) if kept.any() else 0
    correct_kept = int(np.sum(kept & (targets >= 0.8)))
    correct_total = int(np.sum(targets >= 0.8))
    kept_total = int(kept.sum())
    corr = float(np.corrcoef(conf, targets)[0, 1]) if len(conf) > 2 else float("nan")
    mae = float(np.mean(np.abs(conf - targets))) if len(conf) else float("nan")
    return {
        "n": len(conf),
        "correlation": corr,
        "mae": mae,
        "threshold": threshold,
        "kept": kept_total,
        "wrong_kept": wrong_kept,
        "correct_kept": correct_kept,
        "correct_total": correct_total,
        "precision": (correct_kept / kept_total) if kept_total else None,
        "coverage": (correct_kept / correct_total) if correct_total else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moe-config", required=True)
    parser.add_argument("--df-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--max-rows-per-label",
        default='{"complete": 800, "markush": 2000, "fragment": 3000}',
        help="JSON dict complete/markush/fragment row caps",
    )
    parser.add_argument(
        "--extra-fragment-csv",
        action="append",
        default=[],
        help="CSV with file_path+SMILES columns of fragment rows (repeatable)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dec-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.df_cache, json.loads(args.max_rows_per_label), args.seed,
                     extra_fragment_csvs=args.extra_fragment_csv)
    if not rows:
        raise SystemExit("no usable rows")
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_count = max(1, int(len(rows) * args.val_fraction))
    train_rows, val_rows = rows[val_count:], rows[:val_count]
    print(f"train {len(train_rows)} rows, val {len(val_rows)} rows")

    model = _load_molnextr(moe_config_path=args.moe_config)
    model.encoder.eval()
    model.moe_decoder.eval()
    feature_dim = int(model._args.encoder_dim)
    print(f"encoder_dim={feature_dim}")

    def collect(rows_):
        pooled, gates, smiles, token_nlls, edge_nlls, overalls, posents, ntoks = forward_features(model, rows_, args.dec_batch_size)
        targets = []
        for pred_smi, row in zip(smiles, rows_):
            gold = row["SMILES"]
            if row["type"] == "fragment":
                from utils.MolNexTR.moe_confidence import fragment_graph_reward
                reward = fragment_graph_reward(pred_smi, gold)
                targets.append(reward["reward"])
            else:
                targets.append(backbone_tanimoto(pred_smi, gold) if pred_smi else 0.0)
        return pooled, gates, targets, token_nlls, edge_nlls, overalls, posents, ntoks

    def save_features(pooled, gates, targets, token_nlls, edge_nlls, overalls, posents, ntoks, suffix):
        np.savez(
            out_dir / f"features_{suffix}.npz",
            pooled=np.asarray(pooled, dtype=np.float32),
            gates=np.asarray(gates, dtype=np.float32),
            targets=np.asarray(targets, dtype=np.float32),
            token_nlls=np.asarray(token_nlls, dtype=np.float32),
            edge_nlls=np.asarray(edge_nlls, dtype=np.float32),
            overalls=np.asarray(overalls, dtype=np.float32),
            posents=np.asarray(posents, dtype=np.float32),
            ntoks=np.asarray(ntoks, dtype=np.float32),
        )

    def load_features(suffix):
        path = out_dir / f"features_{suffix}.npz"
        if not path.exists():
            return None
        data = np.load(path)
        if "overalls" not in data:
            return None  # stale cache without new features
        return (
            [np.asarray(row, dtype=np.float32) for row in data["pooled"]],
            [np.asarray(row, dtype=np.float32) for row in data["gates"]],
            [float(v) for v in data["targets"]],
            [float(v) for v in data["token_nlls"]],
            [float(v) for v in data["edge_nlls"]],
            [float(v) for v in data["overalls"]],
            [float(v) for v in data["posents"]],
            [float(v) for v in data["ntoks"]],
        )

    tr_cached = load_features("train")
    va_cached = load_features("val")
    if tr_cached is None:
        print("feature extraction over train split ...")
        result = collect(train_rows)
        tr_pooled, tr_gates, tr_targets, tr_tnll, tr_enll, tr_overall, tr_posent, tr_ntok = result
        save_features(*result, "train")
    else:
        tr_pooled, tr_gates, tr_targets, tr_tnll, tr_enll, tr_overall, tr_posent, tr_ntok = tr_cached
        print(f"loaded cached train features ({len(tr_pooled)} rows)")
    if va_cached is None:
        print("feature extraction over val split ...")
        result = collect(val_rows)
        va_pooled, va_gates, va_targets, va_tnll, va_enll, va_overall, va_posent, va_ntok = result
        save_features(*result, "val")
    else:
        va_pooled, va_gates, va_targets, va_tnll, va_enll, va_overall, va_posent, va_ntok = va_cached
        print(f"loaded cached val features ({len(va_pooled)} rows)")

    head = ConfidenceHead(feature_dim, 3)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    def batches():
        idx = list(range(len(tr_pooled)))
        rng.shuffle(idx)
        for start in range(0, len(idx), args.batch_size):
            yield idx[start:start + args.batch_size]

    best_mae = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        head.train()
        losses = []
        for batch_idx in batches():
            x_pooled = torch.tensor(np.asarray([tr_pooled[i] for i in batch_idx]), dtype=torch.float32)
            x_gates = torch.tensor(np.asarray([tr_gates[i] for i in batch_idx]), dtype=torch.float32)
            t = torch.tensor([tr_targets[i] for i in batch_idx], dtype=torch.float32)
            token_nll = torch.tensor([tr_tnll[i] for i in batch_idx], dtype=torch.float32)
            edge_nll = torch.tensor([tr_enll[i] for i in batch_idx], dtype=torch.float32)
            overall = torch.tensor([tr_overall[i] for i in batch_idx], dtype=torch.float32)
            posent = torch.tensor([tr_posent[i] for i in batch_idx], dtype=torch.float32)
            ntok = torch.tensor([tr_ntok[i] for i in batch_idx], dtype=torch.float32)
            logits = head(x_pooled, x_gates, token_nll, edge_nll, overall, posent, ntok)
            loss = confidence_loss(logits, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        val_metrics = evaluate_head(head, va_pooled, va_gates, va_targets, threshold=0.8,
                                    token_nlls=va_tnll, edge_nlls=va_enll,
                                    overalls=va_overall, posents=va_posent, ntoks=va_ntok)
        print(f"epoch {epoch:02d} loss={np.mean(losses):.4f} "
              f"val corr={val_metrics['correlation']:.3f} mae={val_metrics['mae']:.3f} "
              f"wrong_kept@0.8={val_metrics['wrong_kept']}/{val_metrics['kept']}")
        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)

    # Calibration sweep: precision-first threshold (zero wrong-backbone rows kept).
    threshold_table = []
    head.eval()
    with torch.inference_mode():
        x_pooled = torch.tensor(np.asarray(va_pooled), dtype=torch.float32)
        x_gates = torch.tensor(np.asarray(va_gates), dtype=torch.float32)
        bs = len(va_pooled)
        token_nll = torch.tensor(np.asarray(va_tnll, dtype=np.float32))
        edge_nll = torch.tensor(np.asarray(va_enll, dtype=np.float32))
        overall = torch.tensor(np.asarray(va_overall, dtype=np.float32))
        posent = torch.tensor(np.asarray(va_posent, dtype=np.float32))
        ntok = torch.tensor(np.asarray(va_ntok, dtype=np.float32))
        conf = head.expected_confidence(head(x_pooled, x_gates, token_nll, edge_nll, overall, posent, ntok)).numpy()
    targets = np.asarray(va_targets, dtype=np.float32)
    for t in sorted(set(np.round(conf, 3)) | {0.0}):
        threshold_table.append(evaluate_head(head, va_pooled, va_gates, va_targets,
                                             threshold=float(t), token_nlls=va_tnll, edge_nlls=va_enll,
                                             overalls=va_overall, posents=va_posent, ntoks=va_ntok))
    zero_wrong = [e for e in threshold_table if e["wrong_kept"] == 0]
    recommended = max(
        zero_wrong,
        key=lambda e: (e["coverage"] or 0.0, e["threshold"]),
    ) if zero_wrong else max(threshold_table, key=lambda e: e["coverage"] or 0.0)

    torch.save({"confidence_head": best_state}, out_dir / "moe_confidence.pt")
    source_config = json.load(open(args.moe_config))
    deploy_config = dict(source_config)
    deploy_config["use_confidence_head"] = True
    deploy_config["confidence_path"] = str((out_dir / "moe_confidence.pt").resolve())
    # Artifact paths in the deployed config must resolve from the DEPLOY dir;
    # absolutize the source artifacts so the new config works standalone.
    source_dir = os.path.dirname(os.path.abspath(args.moe_config))
    for key in ("encoder_path", "expert0_path", "expert1_path", "expert2_path",
                "router_path", "regime_path"):
        raw = deploy_config.get(key)
        if raw and isinstance(raw, str):
            p = raw if os.path.isabs(raw) else os.path.join(source_dir, raw)
            deploy_config[key] = os.path.abspath(p)
    if isinstance(deploy_config.get("expert_paths"), list):
        deploy_config["expert_paths"] = [
            (v if os.path.isabs(v) else os.path.abspath(os.path.join(source_dir, v)))
            if isinstance(v, str) else v
            for v in deploy_config["expert_paths"]
        ]
    with open(out_dir / "moe_config.json", "w", encoding="utf-8") as handle:
        json.dump(deploy_config, handle, ensure_ascii=False, indent=2)

    report = {
        "schema_version": "confidence_head_training_v1",
        "source_moe_config": args.moe_config,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "best_val_mae": best_mae,
        "recommended_threshold": recommended,
        "threshold_table": threshold_table,
    }
    with open(out_dir / "confidence_head_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print(json.dumps({"recommended": recommended}, indent=2))
    print(f"wrote {out_dir / 'moe_confidence.pt'} and {out_dir / 'moe_config.json'}")


if __name__ == "__main__":
    main()
