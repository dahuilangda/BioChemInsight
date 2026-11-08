"""COMO-style Mixed Reward Training (MRT) for the markush sidecar (§44.54).

Samples K rollouts per markush training image at temperature > 0, scores them
with a chemistry reward (isotope-identity sequence match + Tanimoto + multiset
F1 + validity), and applies group-relative leave-one-out policy gradient
(``group_relative_rloo_loss``) on the teacher-forced sequence log-probs.
The MRT loss is ADDED to the CE loss (mixed objective): the reward shapes the
decoder's label-emission behaviour that the drift-suppressed CE lr cannot
reach, while CE keeps the general decoding distribution anchored.

Reward convention: the deployed identity metric compares the dummy-isotope
SEQUENCE in dummy-atom order, so the dominant term uses ``GetIsotope()`` in
atom order — not a multiset.
"""
from __future__ import annotations

from collections import Counter

import torch

from utils.MolNexTR.moe import ATOM_FORMAT
from utils.MolNexTR.ocr_conditioning import build_ocr_memory
from utils.MolNexTR.tokenization import PAD_ID
from .losses import group_relative_rloo_loss


def identity_reward(pred_smiles: str, gold_smiles: str) -> float:
    """Chemistry reward: identity (1.0) + Tanimoto (0.2) + isotope multiset F1 (0.1)."""
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    pred_mol = Chem.MolFromSmiles(str(pred_smiles or ""))
    if pred_mol is None:
        return 0.0
    gold_mol = Chem.MolFromSmiles(str(gold_smiles or ""))
    if gold_mol is None:
        return 0.1  # unparseable gold: credit validity only

    def iso_seq(mol):
        return [atom.GetIsotope() for atom in mol.GetAtoms() if atom.GetIsotope() > 0]

    identity = float(iso_seq(pred_mol) == iso_seq(gold_mol))
    try:
        fp_p = AllChem.GetMorganFingerprintAsBitVect(pred_mol, 2, 512)
        fp_g = AllChem.GetMorganFingerprintAsBitVect(gold_mol, 2, 512)
        tanimoto = DataStructs.TanimotoSimilarity(fp_p, fp_g)
    except Exception:
        tanimoto = 0.0
    p_seq, g_seq = iso_seq(pred_mol), iso_seq(gold_mol)
    if p_seq and g_seq:
        overlap = sum((Counter(p_seq) & Counter(g_seq)).values())
        precision = overlap / max(1, len(p_seq))
        recall = overlap / max(1, len(g_seq))
        multiset_f1 = 2 * precision * recall / max(1e-9, precision + recall)
    else:
        multiset_f1 = 0.0
    return float(identity) + 0.2 * float(tanimoto) + 0.1 * float(multiset_f1)


def _sequence_log_probs(decoder_ar, features_rep, labels, lengths, ocr_memory_pair):
    """Teacher-forced per-sequence sum log-prob of the given token rows.

    ``labels`` rows are ``[BOS] + generated tokens``; the decoder's forward
    shifts internally (logits[:, :-1] predict labels[:, 1:]).
    """
    logits, targets, _ = decoder_ar(
        features_rep, labels, lengths, ocr_memory=ocr_memory_pair)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = targets.ne(PAD_ID)
    return (token_lp * mask).sum(dim=1)


def mrt_loss_for_markush_batch(
    model,
    features: torch.Tensor,
    encoder_hiddens,
    refs: dict,
    gold_smiles,
    *,
    k: int = 4,
    temperature: float = 0.8,
    markush_mask: torch.Tensor | None = None,
    bos_id: int | None = None,
    expected_structure_types=None,
):
    """One MRT step on the markush subset of a training batch.

    Returns ``(loss | None, stats)``. ``loss`` is None when no markush rows are
    present or no group produced a non-degenerate reward spread.
    """
    moe = model.moe
    device = features.device
    if markush_mask is None:
        markush_mask = torch.ones(features.size(0), dtype=torch.bool, device=device)
    rows = torch.nonzero(markush_mask).reshape(-1).tolist()
    if not rows:
        return None, {"mrt_skipped": "no_markush_rows"}
    feats = features[markush_mask]
    golds = [gold_smiles[i] for i in rows]
    B = feats.size(0)
    hiddens_sub = None
    if encoder_hiddens:
        hiddens_sub = [
            h[markush_mask] if isinstance(h, torch.Tensor) and h.size(0) == features.size(0)
            else h
            for h in encoder_hiddens
        ]
    expected_full = list(expected_structure_types or ["markush"] * features.size(0))
    expected = [expected_full[i] if i < len(expected_full) else "markush" for i in rows]

    ocr_pair_sub = None
    cells = refs.get("ocr_cells")
    if getattr(moe, "ocr_conditioning", None) is not None and isinstance(cells, torch.Tensor):
        pair = build_ocr_memory(
            moe.ocr_conditioning, cells[markush_mask], device, feats.dtype)
        if pair is not None:
            ocr_pair_sub = pair

    # Rollouts: replicate the sub-batch K times, decode once at temperature.
    feats_rep = feats.repeat(k, 1, 1)
    if ocr_pair_sub is not None:
        moe.set_ocr_memory(
            ocr_pair_sub[0].repeat(k, 1, 1), ocr_pair_sub[1].repeat(k, 1, 1))
    try:
        with torch.no_grad():
            preds = moe.decode(
                feats_rep,
                hiddens_sub,
                beam_size=1,
                n_best=1,
                expected_structure_types=expected * k,
                sampling_temp=float(temperature),
                keep_topk=1,
            )
    finally:
        moe.clear_ocr_memory()

    sequences, rewards = [], []
    for idx, pred in enumerate(preds):
        atom_data = pred.get(ATOM_FORMAT) or {}
        sequences.append(list(atom_data.get("_token_sequence") or []))
        rewards.append(identity_reward(str(atom_data.get("smiles") or ""), golds[idx % B]))

    bos = int(bos_id) if bos_id is not None else 1  # tokenization SOS default
    max_len = max((len(s) for s in sequences), default=0) + 1
    labels = torch.full((len(sequences), max_len), PAD_ID, dtype=torch.long)
    for i, seq in enumerate(sequences):
        row = [bos] + list(seq)
        labels[i, :len(row)] = torch.tensor(row, dtype=torch.long)
    lengths = torch.ones((len(sequences), 1), dtype=torch.long)
    for i, seq in enumerate(sequences):
        lengths[i, 0] = len(seq) + 1  # BOS + generated tokens

    labels_dev = labels.to(device)
    lengths_dev = lengths.to(device)
    mem_pair_rep = (
        (ocr_pair_sub[0].repeat(k, 1, 1), ocr_pair_sub[1].repeat(k, 1, 1))
        if ocr_pair_sub is not None else None
    )
    seq_lp = _sequence_log_probs(
        moe.experts[moe.ocr_conditioning_expert].decoder[ATOM_FORMAT],
        feats_rep, labels_dev, lengths_dev, mem_pair_rep)

    per_group_losses = []
    groups_with_signal = 0
    for b in range(B):
        group_lp = [seq_lp[b + j * B] for j in range(k)]
        group_rewards = [rewards[b + j * B] for j in range(k)]
        try:
            loss_b, _adv = group_relative_rloo_loss(group_lp, group_rewards)
        except ValueError:
            continue
        if loss_b is not None:
            groups_with_signal += 1
            per_group_losses.append(loss_b)
    stats = {
        "mrt_rows": int(B),
        "mrt_mean_reward": sum(rewards) / max(1, len(rewards)),
        "mrt_groups_with_signal": groups_with_signal,
    }
    if not per_group_losses:
        stats["mrt_skipped"] = "no_reward_spread"
        return None, stats
    return torch.stack(per_group_losses).mean(), stats
