"""Confidence head for the MolNexTR MoE.

Predicts a distribution over bins of a structural accuracy metric; the
calibrated confidence is the expected bin value.

  * metric : Tanimoto similarity between predicted and gold molecular graphs
             (Morgan fingerprints), in [0, 1].
  * target : the true Tanimoto of a free-running greedy decode vs gold,
             computed periodically during training.
  * head   : MLP over [pooled encoder features, gate weights, gold-token
             mixture-NLL, edge-NLL] -> logits over ``NUM_BINS`` bins.
  * output : confidence = softmax(logits) . bin_centers, in [0, 1].
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_BINS = 20                      # Tanimoto bins over [0, 1]
BIN_WIDTH = 1.0 / NUM_BINS


def bin_centers(num_bins: int = NUM_BINS) -> torch.Tensor:
    return torch.linspace(BIN_WIDTH / 2, 1.0 - BIN_WIDTH / 2, num_bins)


def value_to_bin(value: float, num_bins: int = NUM_BINS) -> int:
    """Map a Tanimoto value in [0,1] to a bin index."""
    v = max(0.0, min(1.0, float(value)))
    return min(num_bins - 1, int(v / BIN_WIDTH))


def _safe_morgan(smiles: str, radius: int = 2, nbits: int = 1024):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception:
        return None
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    except Exception:
        return None


def smi_tanimoto(pred_smiles: str, gold_smiles: str) -> float:
    """Tanimoto between predicted and gold SMILES (0 if either is invalid)."""
    fp_a = _safe_morgan(pred_smiles)
    fp_b = _safe_morgan(gold_smiles)
    if fp_a is None or fp_b is None:
        return 0.0
    try:
        from rdkit import DataStructs
        return max(0.0, min(1.0, float(DataStructs.TanimotoSimilarity(fp_a, fp_b))))
    except Exception:
        return 0.0


def _fragment_reward_molecule(smiles: str):
    try:
        from rdkit import Chem
    except Exception:
        return None
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def _fragment_backbone_smiles(mol) -> str:
    if mol is None:
        return ""
    try:
        from rdkit import Chem

        editable = Chem.RWMol(mol)
        dummy_indices = [
            atom.GetIdx()
            for atom in editable.GetAtoms()
            if atom.GetAtomicNum() == 0
        ]
        for atom_index in sorted(dummy_indices, reverse=True):
            editable.RemoveAtom(atom_index)
        backbone = editable.GetMol()
        if backbone.GetNumAtoms() == 0:
            return ""
        Chem.SanitizeMol(backbone)
        return Chem.MolToSmiles(backbone, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def _fragment_attachment_site_smiles(mol) -> str:
    if mol is None:
        return ""
    try:
        from rdkit import Chem

        dummy_atoms = [
            atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0
        ]
        if len(dummy_atoms) != 1 or dummy_atoms[0].GetDegree() != 1:
            return ""
        dummy_index = dummy_atoms[0].GetIdx()
        anchor_index = dummy_atoms[0].GetNeighbors()[0].GetIdx()
        attachment_bond = dummy_atoms[0].GetBonds()[0]
        bond_code = int(round(float(attachment_bond.GetBondTypeAsDouble()) * 10))
        editable = Chem.RWMol(mol)
        editable.GetAtomWithIdx(anchor_index).SetIsotope(900 + bond_code)
        editable.RemoveAtom(dummy_index)
        marked = editable.GetMol()
        Chem.SanitizeMol(marked)
        return Chem.MolToSmiles(marked, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def fragment_graph_reward(pred_smiles: str, gold_smiles: str) -> dict[str, float | bool]:
    """Dense deployment-aligned reward: a high score requires both the
    dummy-stripped backbone and one bonded terminal dummy in a single
    connected graph; a missing attachment caps the reward at 0.55.
    """
    pred = _fragment_reward_molecule(pred_smiles)
    gold = _fragment_reward_molecule(gold_smiles)
    valid = pred is not None and gold is not None
    backbone_tanimoto = 0.0
    attachment_valid = False
    attachment_site_exact = False
    if valid:
        pred_backbone = _fragment_backbone_smiles(pred)
        gold_backbone = _fragment_backbone_smiles(gold)
        backbone_tanimoto = smi_tanimoto(pred_backbone, gold_backbone)
        dummy_atoms = [
            atom for atom in pred.GetAtoms() if atom.GetAtomicNum() == 0
        ]
        try:
            from rdkit import Chem

            connected = len(Chem.GetMolFrags(pred)) == 1
        except Exception:
            connected = False
        attachment_valid = bool(
            connected
            and len(dummy_atoms) == 1
            and dummy_atoms[0].GetDegree() == 1
        )
        if attachment_valid:
            pred_site = _fragment_attachment_site_smiles(pred)
            gold_site = _fragment_attachment_site_smiles(gold)
            attachment_site_exact = bool(
                pred_site and gold_site and pred_site == gold_site
            )
    attachment_score = 1.0 if attachment_valid else 0.0
    site_score = 1.0 if attachment_site_exact else 0.0
    reward = (
        0.55 * backbone_tanimoto
        + 0.15 * attachment_score
        + 0.10 * backbone_tanimoto * attachment_score
        + 0.20 * site_score
    )
    return {
        "reward": float(max(0.0, min(1.0, reward))),
        "valid": bool(valid),
        "backbone_tanimoto": float(backbone_tanimoto),
        "attachment_valid": bool(attachment_valid),
        "attachment_site_exact": bool(attachment_site_exact),
    }


class ConfidenceHead(nn.Module):
    """Predicts a distribution over accuracy bins from pooled encoder
    features plus gate/entropy/NLL/overall-score scalar features.
    """

    def __init__(self, feature_dim: int, num_experts: int, num_bins: int = NUM_BINS,
                 hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.num_bins = num_bins
        # attention pooling over encoder features
        self.attn_pool = nn.Linear(feature_dim, 1)
        scalar_dim = num_experts + 1 + 1 + 1 + 1 + 1 + 1  # gate + entropy + token_nll + edge_nll + overall + pos_entropy + num_tokens
        in_dim = feature_dim + scalar_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_bins),
        )
        self.register_buffer("bin_centers", bin_centers(num_bins), persistent=False)

    def attention_pool(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """Attention-weighted pooling over spatial tokens."""
        weights = F.softmax(self.attn_pool(encoder_features.float()).squeeze(-1), dim=-1)
        return (encoder_features.float() * weights.unsqueeze(-1)).sum(dim=1)

    def forward(self, pooled_features: torch.Tensor, gate_weights: torch.Tensor,
                token_nll: torch.Tensor, edge_nll: torch.Tensor,
                overall_score: torch.Tensor | None = None,
                pos_entropy: torch.Tensor | None = None,
                num_tokens: torch.Tensor | None = None) -> torch.Tensor:
        gate_entropy = (-(gate_weights.float() * (gate_weights.float() + 1e-8).log()).sum(dim=-1, keepdim=True))
        bs = pooled_features.shape[0]
        device = pooled_features.device
        def _col(t):
            if t is None:
                return torch.zeros(bs, 1, device=device)
            return t.float().to(device).reshape(bs, -1)[:, 0:1]
        x = torch.cat([
            pooled_features.float(),
            gate_weights.float(),
            gate_entropy,
            _col(token_nll),
            _col(edge_nll),
            _col(overall_score),
            _col(pos_entropy),
            _col(num_tokens),
        ], dim=-1)
        return self.net(x)

    def expected_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """E[bin value] under softmax -> calibrated confidence in [0,1]."""
        probs = F.softmax(logits.float(), dim=-1)
        centers = self.bin_centers.to(probs.device)
        return (probs * centers).sum(dim=-1)      # (B,)


def confidence_loss(logits: torch.Tensor, tanimoto_targets: torch.Tensor,
                    num_bins: int = NUM_BINS) -> torch.Tensor:
    """Ordinal-smoothed cross-entropy: each continuous Tanimoto target splits
    its mass between the two adjacent bin centers (ordinal soft-label).
    """
    targets = tanimoto_targets.float().clamp(0.0, 0.9999)
    lower = (targets / BIN_WIDTH).floor().long().clamp(0, num_bins - 1)
    upper = (lower + 1).clamp(max=num_bins - 1)
    frac = (targets - lower.float() * BIN_WIDTH) / BIN_WIDTH      # weight on `upper`
    log_probs = F.log_softmax(logits.float(), dim=-1)
    loss = -(1.0 - frac) * log_probs.gather(1, lower.unsqueeze(1)).squeeze(1) \
           - frac * log_probs.gather(1, upper.unsqueeze(1)).squeeze(1)
    return loss.mean()
