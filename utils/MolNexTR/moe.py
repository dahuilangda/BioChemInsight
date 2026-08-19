"""Mixture-of-experts decoder for MolNexTR.

Extends MolNexTR to recognize Markush structures and attachment-bearing
fragments while keeping the complete-molecule prediction byte-identical to the
frozen base decoder.

Architecture
------------
- A frozen base decoder (``expert0``) handles complete molecules; its decode
  path is unmodified, so complete output is identical to single-decoder MolNexTR.
- Trained sidecar decoders (``expert1`` = markush, ``expert2`` = fragment) add
  attachment/R-group recognition.
- A router predicts ``[complete, markush, fragment]``; a confident-complete
  prediction forces ``expert0`` (the safety floor).
- An attachment-set head (``residual_base`` decode mode) predicts attachment
  points and grafts them onto the expert0 graph for markush/fragment outputs.
"""

from __future__ import annotations

import json
import math
import os
import re as _re
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import (
    Decoder,
    decode_terminal_dummy_single_bond_map,
    get_edge_prediction,
    decode_bond_limit,
    invalid_decode_symbol,
    MAX_DECODE_ATOMS,
)
from .attachment_set import (
    AttachmentSetHead,
    MultiScaleAttachmentSetHead,
    MultiScaleAttachmentPointerHead,
    MultiScaleAttachmentHeatmapPointerHead,
    attachment_set_loss,
    select_attachment_queries,
)
from .moe_confidence import ConfidenceHead
from .tokenization import PAD_ID, SOS_ID, EOS_ID, MASK_ID
from .utils import FORMAT_INFO
from .abbrs import VALENCES
from ..markush_labels import is_markush_label, normalize_label

EXPERT_NAMES = ["complete", "markush", "fragment"]
ATOM_FORMAT = "chartok_coords"  # the only atom-stream format in the shipped checkpoint


# ---------------------------------------------------------------------------
# Valence-aware graph helpers (module-level so they can be unit-tested).
# Edge codes match _convert_graph_to_smiles: 0=no bond, 1=single, 2=double,
# 3=triple, 4=aromatic, 5/6=wedge/dash singles.
def bond_valence_weight(edge_value: int) -> float:
    # Aromatic bonds count 1.0 toward valence; 1.5 would wrongly flag valid
    # fused-ring junction atoms.
    v = int(edge_value)
    if v == 2:
        return 2.0
    if v == 3:
        return 3.0
    if v in (1, 4, 5, 6):
        return 1.0
    return 0.0


def used_valence(edges: list[list[int]], atom_index: int) -> float:
    if atom_index < 0 or atom_index >= len(edges):
        return 0.0
    return float(sum(bond_valence_weight(v) for v in edges[atom_index]))


def max_valence_for_symbol(symbol: Any) -> float:
    """Max permitted valence for a predicted atom symbol. Lowercase c/n/o/s
    are aromatic spellings of the same element; '*', '[n*]' and 'Rn' are
    one-bond wildcards. Unknown symbols get 99.0 (sanitization catches them).
    """
    text = str(symbol or "").strip()
    if not text:
        return 0.0
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1]
        if inner.endswith("*"):
            return 1.0
        m = _re.match(r"([A-Z][a-z]?)", inner)
        if not m:
            m = _re.match(r"([a-z])", inner)
        element = m.group(1) if m else ""
    else:
        element = text
    if element == "*" or element.startswith("R") or element == "":
        return 1.0
    if len(element) == 1:
        element = element.upper()
    allowed = VALENCES.get(element)
    if not allowed:
        return 99.0
    return float(max(allowed))


def remaining_valence(
    symbols: list[Any], edges: list[list[int]], atom_index: int
) -> float:
    if atom_index < 0 or atom_index >= len(symbols):
        return 0.0
    return max_valence_for_symbol(symbols[atom_index]) - used_valence(
        edges, atom_index
    )


def has_valence_room(
    symbols: list[Any],
    edges: list[list[int]],
    atom_index: int,
    extra_bond_value: int = 1,
) -> bool:
    # 1e-6 epsilon absorbs the 0.5 rounding introduced by aromatic edges.
    return (
        remaining_valence(symbols, edges, atom_index) + 1e-6
        >= bond_valence_weight(extra_bond_value)
    )


# --------------------------------------------------------------------------- config
def load_moe_config(path: str) -> dict:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"MoE config not found: {path}")
    with open(path, "r") as handle:
        config = json.load(handle)
    config_dir = os.path.dirname(os.path.abspath(path))

    def resolve_artifact(raw):
        if not raw or not isinstance(raw, str):
            return raw
        if os.path.isabs(raw):
            return raw
        # Legacy configs stored repo-relative paths; also try the basename
        # beside the config before falling back to the process cwd.
        candidates = [
            os.path.join(config_dir, raw),
            os.path.join(config_dir, os.path.basename(raw)),
            raw,
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
        return os.path.abspath(candidates[0])

    artifact_keys = (
        "adapter_path",
        "expert0_path",
        "expert1_path",
        "expert2_path",
        "router_path",
        "confidence_path",
        "regime_path",
        "encoder_path",
    )
    for key in artifact_keys:
        if key in config:
            config[key] = resolve_artifact(config.get(key))
    if isinstance(config.get("expert_paths"), list):
        config["expert_paths"] = [
            resolve_artifact(value) for value in config["expert_paths"]
        ]
    return config


# --------------------------------------------------------------------------- repair
def valence_aware_edge_repair(symbols, edges, edge_scores):
    """Drop the lowest-priority bonds from atoms RDKit flags as over-valent
    until the molecule sanitizes; a fully-valid graph is returned unchanged.
    Drop key: dummy/R-group bonds first, then lowest bond order, then lowest
    learned ``edge_scores`` (the tiebreak works without ``compute_confidence``).
    """
    n = len(symbols)
    if n == 0 or edges is None:
        return edges, edge_scores
    try:
        from rdkit import Chem
        from .chemical import _atom_from_predicted_symbol
    except Exception:
        return edges, edge_scores

    code_to_type = {
        1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC, 5: Chem.BondType.SINGLE, 6: Chem.BondType.SINGLE,
    }

    def build_mol():
        mol = Chem.RWMol()
        idx = []
        for s in symbols:
            try:
                atom = _atom_from_predicted_symbol(s)
            except Exception:
                atom = Chem.Atom(0) if s == "*" else Chem.Atom(6)
            idx.append(mol.AddAtom(atom))
        for i in range(n):
            for j in range(i + 1, n):
                bt = code_to_type.get(int(edges[i][j]))
                if bt is not None:
                    mol.AddBond(idx[i], idx[j], bt)
        return mol

    try:
        Chem.SanitizeMol(build_mol())
        return edges, edge_scores
    except Exception:
        pass

    def _is_attachment_symbol(symbol: Any) -> bool:
        text = str(symbol or "").strip()
        if not text:
            return False
        if text.startswith("[") and text.endswith("]") and text[1:-1].endswith("*"):
            return True
        return text == "*" or text.startswith("R")

    # Bond-order drop preference: lower = remove first. Single/wedge are the most
    # disposable; aromatic/double/triple are preserved in preference to single.
    bond_drop_rank = {1: 0, 5: 1, 6: 1, 4: 2, 2: 3, 3: 4}

    def _drop_key(worst: int, j: int):
        """Composite key: smaller = drop first. (attachment_to_dummy, bond_rank, score)."""
        a, b = (worst, j) if worst < j else (j, worst)
        code = int(edges[a][b])
        neighbor = j
        attachment = 0 if _is_attachment_symbol(symbols[neighbor]) else 1
        rank = bond_drop_rank.get(code, 0)
        sc = float(scores.get((a, b), 0.0))
        return (attachment, rank, sc)

    edges = [list(row) for row in edges]
    scores = dict(edge_scores or {})
    for _ in range(2 * n + 4):
        try:
            Chem.SanitizeMol(build_mol())
            break
        except Exception as e:
            msg = str(e)
        if "valence" not in msg.lower() and "greater than permitted" not in msg.lower():
            break
        am = _re.search(r"atom\s*#\s*(\d+)", msg)
        if not am:
            break
        worst = int(am.group(1))
        if worst >= n:
            break
        best_key, best_drop = None, None
        for j in range(n):
            if j == worst:
                continue
            a, b = (worst, j) if worst < j else (j, worst)
            if edges[a][b]:
                key = _drop_key(worst, j)
                if best_drop is None or key < best_drop:
                    best_drop, best_key = key, (a, b)
        if best_key is None:
            break
        a, b = best_key
        edges[a][b] = 0
        edges[b][a] = 0
        scores.pop((a, b), None)
    return edges, scores


# --------------------------------------------------------------------------- router
class StructureRouter(nn.Module):
    """Learned image-level expert router: ``features`` (B, L, C) -> raw
    logits (B, K); the caller applies softmax. ``mean_mlp`` is the legacy
    checkpoint-compatible kind.
    """

    def __init__(self, feature_dim: int, num_experts: int = 3,
                 hidden: int = 256, dropout: float = 0.1,
                 kind: str = "mean_mlp"):
        super().__init__()
        self.num_experts = num_experts
        self.kind = str(kind or "mean_mlp").lower()
        if self.kind == "mean_mlp":
            self.token_score = None
            self.net = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_experts),
            )
        elif self.kind == "attention_pool":
            self.token_norm = nn.LayerNorm(feature_dim)
            self.token_score = nn.Sequential(
                nn.Linear(feature_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1, bias=False),
            )
            self.net = nn.Sequential(
                nn.LayerNorm(feature_dim * 3),
                nn.Linear(feature_dim * 3, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_experts),
            )
        else:
            raise ValueError(f"unsupported router kind: {kind!r}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.kind == "mean_mlp":
            pooled = features.mean(dim=1)
        else:
            normalized = self.token_norm(features)
            scores = self.token_score(normalized).squeeze(-1)
            attention = torch.softmax(scores.float(), dim=1).to(dtype=features.dtype)
            attended = torch.sum(features * attention.unsqueeze(-1), dim=1)
            pooled = torch.cat(
                [attended, features.mean(dim=1), features.amax(dim=1)],
                dim=-1,
            )
        return self.net(pooled)                 # (B, K)


class TokenFusionGate(nn.Module):
    """Per-step gate between frozen expert0 and one sidecar, predicted from
    decoder context plus gold-free distribution evidence (entropy, margin,
    Jensen-Shannon disagreement).
    """

    def __init__(self, hidden_size: int, initial_sidecar_weight: float = 0.2):
        super().__init__()
        hidden_size = int(hidden_size)
        inner = max(128, hidden_size // 2)
        evidence_hidden = 32
        self.context_norm = nn.LayerNorm(hidden_size * 4)
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_size * 4, inner),
            nn.SiLU(),
            nn.Linear(inner, inner),
        )
        self.evidence_proj = nn.Sequential(
            nn.LayerNorm(7),
            nn.Linear(7, evidence_hidden),
            nn.SiLU(),
            nn.Linear(evidence_hidden, evidence_hidden),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(inner + evidence_hidden),
            nn.Linear(inner + evidence_hidden, inner),
            nn.SiLU(),
            nn.Linear(inner, 1),
        )
        initial = min(0.95, max(0.05, float(initial_sidecar_weight)))
        nn.init.normal_(self.proj[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.proj[-1].bias, math.log(initial / (1.0 - initial)))

    @staticmethod
    def _distribution_evidence(
        base_log_probs: torch.Tensor,
        sidecar_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        if base_log_probs.shape != sidecar_log_probs.shape:
            raise ValueError(
                "fusion log-prob shape mismatch: "
                f"{tuple(base_log_probs.shape)} vs {tuple(sidecar_log_probs.shape)}"
            )
        lp0 = base_log_probs.detach().float()
        lp1 = sidecar_log_probs.detach().float()
        p0 = lp0.exp()
        p1 = lp1.exp()
        vocab = max(2, int(lp0.size(-1)))
        entropy_scale = math.log(vocab)
        entropy0 = -(p0 * lp0).sum(dim=-1) / entropy_scale
        entropy1 = -(p1 * lp1).sum(dim=-1) / entropy_scale
        k = min(2, vocab)
        top0 = p0.topk(k, dim=-1).values
        top1 = p1.topk(k, dim=-1).values
        confidence0 = top0[..., 0]
        confidence1 = top1[..., 0]
        margin0 = top0[..., 0] - (top0[..., 1] if k > 1 else 0.0)
        margin1 = top1[..., 0] - (top1[..., 1] if k > 1 else 0.0)
        log_mean = torch.logaddexp(lp0, lp1) - math.log(2.0)
        js = 0.5 * (
            (p0 * (lp0 - log_mean)).sum(dim=-1)
            + (p1 * (lp1 - log_mean)).sum(dim=-1)
        ) / math.log(2.0)
        return torch.stack(
            [
                entropy0,
                entropy1,
                confidence0,
                confidence1,
                margin0,
                margin1,
                js,
            ],
            dim=-1,
        )

    def forward(
        self,
        base_hidden: torch.Tensor,
        sidecar_hidden: torch.Tensor,
        base_log_probs: torch.Tensor,
        sidecar_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        if base_hidden.shape != sidecar_hidden.shape:
            raise ValueError(
                f"fusion hidden shape mismatch: {tuple(base_hidden.shape)} vs "
                f"{tuple(sidecar_hidden.shape)}"
            )
        context = torch.cat(
            [
                base_hidden,
                sidecar_hidden,
                sidecar_hidden - base_hidden,
                base_hidden * sidecar_hidden,
            ],
            dim=-1,
        )
        context_features = self.context_proj(self.context_norm(context))
        evidence = self._distribution_evidence(base_log_probs, sidecar_log_probs)
        evidence_features = self.evidence_proj(
            evidence.to(dtype=context_features.dtype)
        )
        return self.proj(
            torch.cat([context_features, evidence_features], dim=-1)
        ).squeeze(-1)


class FragmentTerminalActionHead(nn.Module):
    """Residual on the EOS-vs-``*`` terminal action, consulted only when the
    constrained top action is one of that pair. Inputs are detached, so this
    supervision cannot reshape the encoder or backbone token policy.
    """

    def __init__(self, hidden_size: int, head_hidden_size: int = 128):
        super().__init__()
        hidden_size = int(hidden_size)
        head_hidden_size = max(32, int(head_hidden_size))
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(hidden_size + 3, head_hidden_size),
            nn.SiLU(),
            nn.Linear(head_hidden_size, head_hidden_size),
            nn.SiLU(),
            nn.Linear(head_hidden_size, 1),
        )
        self.stop_net = nn.Sequential(
            nn.Linear(hidden_size + 3, head_hidden_size),
            nn.SiLU(),
            nn.Linear(head_hidden_size, head_hidden_size),
            nn.SiLU(),
            nn.Linear(head_hidden_size, 1),
        )
        # Zero-init keeps enabling the head behavior-preserving at start.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.zeros_(self.stop_net[-1].weight)
        nn.init.zeros_(self.stop_net[-1].bias)

    def _features(
        self,
        hidden: torch.Tensor,
        base_pair_logits: torch.Tensor,
        has_attachment: torch.Tensor,
        step_fraction: torch.Tensor,
    ) -> torch.Tensor:
        if base_pair_logits.shape[:-1] != hidden.shape[:-1] or base_pair_logits.size(-1) != 2:
            raise ValueError(
                "terminal action head shape mismatch: "
                f"hidden={tuple(hidden.shape)} pair={tuple(base_pair_logits.shape)}"
            )
        base_pair = base_pair_logits.detach().float()
        base_margin = torch.tanh(
            (base_pair[..., 1] - base_pair[..., 0]) / 8.0
        ).unsqueeze(-1)
        state = torch.stack(
            [
                has_attachment.detach().float(),
                step_fraction.detach().float().clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        return torch.cat(
            [
                self.hidden_norm(hidden.detach().float()),
                base_margin,
                state,
            ],
            dim=-1,
        )

    def residual(
        self,
        hidden: torch.Tensor,
        base_pair_logits: torch.Tensor,
        has_attachment: torch.Tensor,
        step_fraction: torch.Tensor,
    ) -> torch.Tensor:
        features = self._features(
            hidden,
            base_pair_logits,
            has_attachment,
            step_fraction,
        )
        return self.net(features).squeeze(-1)

    def stop_residual(
        self,
        hidden: torch.Tensor,
        base_pair_logits: torch.Tensor,
        has_attachment: torch.Tensor,
        step_fraction: torch.Tensor,
    ) -> torch.Tensor:
        features = self._features(
            hidden,
            base_pair_logits,
            has_attachment,
            step_fraction,
        )
        return self.stop_net(features).squeeze(-1)

    def policy_pair_logits(
        self,
        hidden: torch.Tensor,
        base_pair_logits: torch.Tensor,
        has_attachment: torch.Tensor,
        step_fraction: torch.Tensor,
    ) -> torch.Tensor:
        """Return differentiable ``[EOS, *]`` logits with a detached base."""
        base_pair = base_pair_logits.detach().float()
        delta = self.residual(
            hidden,
            base_pair,
            has_attachment,
            step_fraction,
        )
        return torch.stack(
            [
                base_pair[..., 0] - 0.5 * delta,
                base_pair[..., 1] + 0.5 * delta,
            ],
            dim=-1,
        )

    def stop_policy_logits(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        *,
        eos_token_id: int,
        has_attachment: torch.Tensor,
        step_fraction: torch.Tensor,
    ) -> torch.Tensor:
        """Return detached-base ``[continue, stop]`` action logits."""
        base_logits = logits.detach().float()
        eos_token_id = int(eos_token_id)
        non_eos = torch.ones(
            base_logits.size(-1),
            dtype=torch.bool,
            device=base_logits.device,
        )
        non_eos[eos_token_id] = False
        strongest_continue = base_logits[..., non_eos].amax(dim=-1)
        base_pair = torch.stack(
            [strongest_continue, base_logits[..., eos_token_id]],
            dim=-1,
        )
        delta = self.stop_residual(
            hidden,
            base_pair,
            has_attachment,
            step_fraction,
        )
        return torch.stack(
            [base_pair[..., 0], base_pair[..., 1] + delta],
            dim=-1,
        )

    def adjust_logits(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        *,
        eos_token_id: int,
        star_token_id: int,
        has_attachment: torch.Tensor,
        step_fraction: torch.Tensor,
        eligible: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair = torch.stack(
            [logits[..., int(eos_token_id)], logits[..., int(star_token_id)]],
            dim=-1,
        )
        policy_pair = self.policy_pair_logits(
            hidden,
            pair,
            has_attachment,
            step_fraction,
        )
        delta = policy_pair[..., 1] - pair.detach().float()[..., 1]
        delta = 2.0 * delta * eligible.detach().float()
        adjusted = logits.clone()
        adjusted[..., int(eos_token_id)] = (
            adjusted[..., int(eos_token_id)] - 0.5 * delta.to(adjusted.dtype)
        )
        adjusted[..., int(star_token_id)] = (
            adjusted[..., int(star_token_id)] + 0.5 * delta.to(adjusted.dtype)
        )
        stop_pair = self.stop_policy_logits(
            hidden,
            logits,
            eos_token_id=eos_token_id,
            has_attachment=has_attachment,
            step_fraction=step_fraction,
        )
        stop_delta = (
            stop_pair[..., 1]
            - logits.detach().float()[..., int(eos_token_id)]
        ) * has_attachment.detach().float()
        adjusted[..., int(eos_token_id)] = (
            adjusted[..., int(eos_token_id)]
            + stop_delta.to(adjusted.dtype)
        )
        return adjusted, policy_pair, stop_pair


def atom_symbol_scores_from_token_probs(
    symbols: list[str],
    indices: list[int],
    token_probs: list[float] | np.ndarray,
) -> list[float]:
    """Match ``Decoder.decode`` atom confidence indexing for mixture decoding."""
    probs = np.asarray(token_probs, dtype=np.float64)
    scores = []
    for symbol, index in zip(symbols, indices):
        symbol_end = int(index) - 3
        symbol_start = symbol_end - len(str(symbol)) + 1
        if symbol_start < 0 or symbol_end < symbol_start or symbol_end >= len(probs):
            scores.append(0.0)
            continue
        span = probs[symbol_start:symbol_end + 1]
        scores.append(float(np.prod(span) ** (1.0 / max(1, len(span)))))
    return scores


# --------------------------------------------------------------- LoRA adapter layer
class LoRAMoELinear(nn.Module):
    """``nn.Linear`` drop-in: frozen ``W0`` + N gate-weighted LoRA adapters.
    ``base_weight``/``base_bias`` are persistent buffers so the base is
    serialized with the module. A zero/absent ``_gate`` short-circuits the
    residual (bit-identical to ``nn.Linear(weight=W0, bias=b0)``).
    """

    def __init__(self, in_features: int, out_features: int, num_experts: int,
                 rank: int, alpha: float, bias: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_experts = int(num_experts)
        self.rank = int(rank)
        self.scaling = float(alpha) / float(max(rank, 1))
        self.register_buffer(
            "base_weight",
            torch.zeros(self.out_features, self.in_features, device=device, dtype=dtype),
        )
        if bias:
            self.register_buffer(
                "base_bias",
                torch.zeros(self.out_features, device=device, dtype=dtype),
            )
        else:
            self.base_bias = None
        # Stacked LoRA params: A (N,r,in), B (N,out,r).
        self.A = nn.Parameter(torch.empty(self.num_experts, self.rank, self.in_features))
        self.B = nn.Parameter(torch.empty(self.num_experts, self.out_features, self.rank))
        self.reset_lora_parameters()
        self._gate: torch.Tensor | None = None

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def set_gate(self, gate: torch.Tensor | None) -> None:
        if gate is None:
            self._gate = None
            return
        gate = gate.to(device=self.base_weight.device, dtype=self.base_weight.dtype)
        # Back-compat: drop the leading complete-class column from old
        # router-class gates passed to a sidecar-only adapter stack.
        if gate.shape[-1] == self.num_experts + 1:
            gate = gate[..., 1:]
        if gate.shape[-1] != self.num_experts:
            raise ValueError(
                f"LoRA gate width {gate.shape[-1]} does not match adapter experts {self.num_experts}"
            )
        self._gate = gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.base_weight, self.base_bias)
        g = self._gate
        if g is None or g.numel() == 0 or not bool(g.any()):
            return base                                    # gate inactive fast path
        gs = g * self.scaling
        if g.dim() == 1:                                  # (N,) broadcast over all of x
            ax = torch.einsum("nri,...i->n...r", self.A, x)
            bax = torch.einsum("nor,n...r->n...o", self.B, ax)
            residual = torch.einsum("n,n...o->...o", gs, bax)
        else:                                             # (B, N) per-sample gate
            B = g.size(0)
            x2 = x.reshape(B, -1, self.in_features)
            ax = torch.einsum("nri,bsi->bnsr", self.A, x2)
            bax = torch.einsum("nor,bnsr->bnso", self.B, ax)
            residual = torch.einsum("bn,bnso->bso", gs, bax)
            residual = residual.reshape(x.shape[:-1] + (self.out_features,))
        return base + residual


def _parent_and_child(root: nn.Module, dotted: str):
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


# default decoder-side Linear targets (regex on module name within a Decoder)
_DEFAULT_LORA_TARGETS = [
    r"chartok_coords\.decoder\.transformer_layers\.\d+\.self_attn\.(linear_query|linear_keys|linear_values|final_linear)",
    r"chartok_coords\.decoder\.transformer_layers\.\d+\.context_attn\.(linear_query|linear_keys|linear_values|final_linear)",
    r"chartok_coords\.decoder\.transformer_layers\.\d+\.feed_forward\.(w_1|w_2)",
    r"chartok_coords\.enc_trans_layer\.0",
]
_OUTPUT_LAYER_TARGET = r"chartok_coords\.output_layer"
_EDGES_TARGET = r"edges\.mlp\.\d+"


def wrap_decoder_with_lora(
    decoder: Decoder,
    num_experts: int,
    rank: int,
    alpha: float,
    *,
    include_output_layer: bool = True,
    include_edges: bool = False,
) -> int:
    """In-place replace the target ``nn.Linear`` modules of ``decoder`` with
    :class:`LoRAMoELinear`, copying the frozen ``W0``/bias from the source.
    Idempotent; returns the count replaced.
    """
    patterns = list(_DEFAULT_LORA_TARGETS)
    if include_output_layer:
        patterns.append(_OUTPUT_LAYER_TARGET)
    if include_edges:
        patterns.append(_EDGES_TARGET)
    combined = _re.compile("|".join(f"(?:{p})" for p in patterns))

    targets = []
    for name, mod in decoder.named_modules():
        if isinstance(mod, LoRAMoELinear):
            continue
        if isinstance(mod, nn.Linear) and combined.search(name):
            targets.append(name)

    for name in targets:
        parent, child = _parent_and_child(decoder, name)
        linear = getattr(parent, child)
        new = LoRAMoELinear(
            linear.in_features, linear.out_features, num_experts, rank, alpha,
            bias=linear.bias is not None,
            device=linear.weight.device, dtype=linear.weight.dtype,
        )
        new.base_weight.copy_(linear.weight.detach())
        if linear.bias is not None and new.base_bias is not None:
            new.base_bias.copy_(linear.bias.detach())
        new.reset_lora_parameters()
        setattr(parent, child, new)
    return len(targets)


def lora_state_dict(module: nn.Module) -> dict:
    """Adapter-only state: the LoRA ``.A`` / ``.B`` parameters (excludes the
    frozen ``base_weight``/``base_bias`` buffers and every non-LoRA param)."""
    return {n: p.detach().cpu() for n, p in module.named_parameters()
            if n.endswith(".A") or n.endswith(".B")}


def pad_chartok_for_sep(module: nn.Module, states: dict) -> dict:
    """Pad the chartok_coords output_layer + embedding +1 row for the <sep>
    token when loading a pre-<sep> checkpoint (vocab 229 -> 230). Only pads
    on an exact 1-row dim-0 shortfall."""
    out_layer_keys = (
        "chartok_coords.output_layer.weight",
        "chartok_coords.output_layer.bias",
    )
    for name, param in module.named_parameters():
        is_out = any(name == k or name.endswith("." + k) for k in out_layer_keys)
        is_emb = (
            "chartok_coords" in name
            and "emb_luts" in name
            and name.endswith(".weight")
        )
        if not (is_out or is_emb):
            continue
        st = states.get(name)
        if st is None or st.dim() < 1:
            continue
        if st.shape[0] == param.shape[0] - 1:
            # Init <sep> row = copy of EOS row (id 2); see model.py for rationale.
            pad = (st[2:3]).clone()
            states[name] = torch.cat([st, pad], dim=0)
    return states


def load_base_weights_into_lora(module: nn.Module, base_states: dict) -> None:
    """Copy frozen ``nn.Linear`` weights from a vanilla decoder state-dict into
    every :class:`LoRAMoELinear` buffer (``foo.weight`` -> ``foo.base_weight``).
    Needed because ``loading()`` uses ``strict=False`` and the keys differ."""
    for name, sub in module.named_modules():
        if isinstance(sub, LoRAMoELinear):
            w = base_states.get(name + ".weight")
            b = base_states.get(name + ".bias")
            if w is not None:
                sub.base_weight.copy_(w)
            if b is not None and sub.base_bias is not None:
                sub.base_bias.copy_(b)


# --------------------------------------------------------------------------- MoE
class MoEDecoder(nn.Module):
    """Frozen-backbone decoder + LoRA adapter experts + a ``StructureRouter``.
    ``decode(...)`` mirrors ``components.Decoder.decode``'s contract and
    attaches MoE routing metadata to each prediction dict.
    """

    def __init__(
        self,
        args,
        tokenizer,
        num_experts: int = 3,
        expert_names: list[str] | None = None,
        *,
        expert_kind: str = "lora",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        include_output_layer: bool = True,
        include_edges: bool = False,
        # router / calibration (used)
        sidecar_confidence_threshold: float = 1.01,
        sidecar_confidence_thresholds: list[float] | dict[str, float] | None = None,
        routing_strategy: str = "soft_mixture",
        router_kind: str = "mean_mlp",
        use_confidence_head: bool = False,
        # legacy / back-compat kwargs (accepted, ignored)
        complete_confidence_threshold: float = 0.9,
        shared_expert0_weight: float = 0.0,
        frozen_expert0: bool = True,
        preserve_expert0: bool = False,
        detach_expert0_on_fragment: bool = False,
        regime_conditioning: bool = False,
        per_expert_routing: list[str] | None = None,
        mix_space: str = "logit",
        rerank_enabled: bool = False,
        rerank_w0_grid: tuple[float, ...] | list[float] | None = None,
        rerank_include_standalone: bool = True,
        rerank_include_expert0: bool = True,
        valence_repair_enabled: bool = True,
        expected_fragment_star_logit_bias: float = 0.0,
        expected_fragment_star_budget: int = 1,
        expected_fragment_max_atoms: int = 30,
        # full_mixture: floor on the frozen complete expert's per-step mixture
        # weight; without it a confident router lets an undertrained
        # specialist dominate and corrupt structure. Default 0.65.
        mixture_complete_floor: float = 0.65,
        full_mixture_sidecar_mode: str = "collapsed",
        token_fusion_mode: str = "fixed",
        token_fusion_initial_sidecar_weight: float = 0.2,
        token_fusion_dispatch: str = "soft",
        token_fusion_hard_threshold: float = 0.5,
        specialist_ownership_scope: str = "full",
        decouple_fusion_policy_optimization: bool = True,
        one_sided_fusion_oracle: bool = True,
        attachment_repair_enabled: bool = False,
        attachment_repair_probe_expert: int = 2,
        attachment_repair_min_probe_prob: float = 0.25,
        attachment_repair_max_dummies: int = 1,
        attachment_set_enabled: bool = False,
        attachment_set_hidden_dim: int = 256,
        attachment_set_num_queries: int = 32,
        attachment_set_num_layers: int = 3,
        attachment_set_num_heads: int = 8,
        attachment_set_max_count: int = 30,
        attachment_set_min_confidence: float = 0.5,
        attachment_set_max_anchor_distance: float = 0.25,
        attachment_set_decode_mode: str = "token_mixture",
        attachment_set_feature_mode: str = "single_scale",
        attachment_set_feature_levels: int = 3,
        attachment_set_max_feature_size: int = 32,
        attachment_set_min_pointer_confidence: float = 0.15,
        attachment_set_heatmap_logit_scale: float = 4.0,
        attachment_set_heatmap_prior_precision: float = 8.0,
        sidecar_coordinate_context: str = "full",
        fragment_terminal_action_head_enabled: bool = False,
        fragment_terminal_action_head_hidden_dim: int = 128,
        fragment_structured_terminal_edge_enabled: bool = False,
    ):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.num_experts = int(num_experts)
        self.expert_names = list(expert_names or EXPERT_NAMES)[: self.num_experts]
        while len(self.expert_names) < self.num_experts:
            self.expert_names.append(f"expert{len(self.expert_names)}")
        self.formats = args.formats

        self.expert_kind = str(expert_kind or "lora").lower()
        if self.expert_kind not in {"lora", "full_mixture"}:
            raise ValueError(
                "expert_kind must be 'lora' (frozen base + LoRA sidecar adapters) "
                f"or 'full_mixture' (frozen complete decoder + a full-decoder "
                f"specialist whose logits are mixed per step), got {expert_kind!r}"
            )
        # LoRA adapters cover only the sidecar classes [markush, fragment];
        # complete is the frozen base path. full_mixture has no adapter
        # columns: expert0 is frozen and expert1 (w_specialist = p_markush +
        # p_fragment) covers both sidecar classes.
        self.num_adapter_experts = (
            max(0, self.num_experts - 1) if self.expert_kind == "lora" else 0
        )
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.include_output_layer = bool(include_output_layer)
        self.include_edges = bool(include_edges)

        self.routing_strategy = str(routing_strategy or "soft_mixture").lower()
        self.router_kind = str(router_kind or "mean_mlp").lower()
        if self.routing_strategy == "shared_routed_top1":
            # Frozen W0 is already the always-on shared expert; fall back to
            # the soft-mixture gate.
            self.routing_strategy = "soft_mixture"
        if self.routing_strategy not in {"soft_mixture", "sparse_top1"}:
            raise ValueError(
                "routing_strategy must be 'soft_mixture' or 'sparse_top1', "
                f"got {routing_strategy!r}"
            )

        self.sidecar_confidence_threshold = float(sidecar_confidence_threshold)
        self.sidecar_confidence_thresholds = self._normalize_sidecar_thresholds(
            sidecar_confidence_thresholds,
            fallback=float(sidecar_confidence_threshold),
        )
        self.valence_repair_enabled = bool(valence_repair_enabled)
        self.mixture_complete_floor = float(min(0.999, max(0.0, mixture_complete_floor)))
        self.expected_fragment_star_logit_bias = float(expected_fragment_star_logit_bias or 0.0)
        self.expected_fragment_star_budget = max(0, int(expected_fragment_star_budget or 0))
        self.expected_fragment_max_atoms = max(0, int(expected_fragment_max_atoms or 0))
        self.full_mixture_sidecar_mode = str(full_mixture_sidecar_mode or "collapsed").lower()
        self.token_fusion_mode = str(token_fusion_mode or "fixed").lower()
        if self.token_fusion_mode not in {"fixed", "adaptive"}:
            raise ValueError(
                "token_fusion_mode must be 'fixed' or 'adaptive', "
                f"got {token_fusion_mode!r}"
            )
        self.token_fusion_initial_sidecar_weight = float(
            token_fusion_initial_sidecar_weight or 0.2)
        self.token_fusion_dispatch = str(token_fusion_dispatch or "soft").lower()
        if self.token_fusion_dispatch not in {"soft", "hard"}:
            raise ValueError(
                "token_fusion_dispatch must be 'soft' or 'hard', "
                f"got {token_fusion_dispatch!r}"
            )
        self.token_fusion_hard_threshold = float(token_fusion_hard_threshold)
        if not 0.0 < self.token_fusion_hard_threshold < 1.0:
            raise ValueError(
                "token_fusion_hard_threshold must be strictly between 0 and 1, "
                f"got {token_fusion_hard_threshold!r}"
            )
        self.specialist_ownership_scope = str(
            specialist_ownership_scope or "full"
        ).lower()
        if self.specialist_ownership_scope not in {"full", "attachment"}:
            raise ValueError(
                "specialist_ownership_scope must be 'full' or 'attachment', "
                f"got {specialist_ownership_scope!r}"
            )
        self.decouple_fusion_policy_optimization = bool(
            decouple_fusion_policy_optimization
        )
        self.one_sided_fusion_oracle = bool(one_sided_fusion_oracle)
        self.attachment_set_enabled = bool(attachment_set_enabled)
        self.attachment_set_hidden_dim = int(attachment_set_hidden_dim)
        self.attachment_set_num_queries = int(attachment_set_num_queries)
        self.attachment_set_num_layers = int(attachment_set_num_layers)
        self.attachment_set_num_heads = int(attachment_set_num_heads)
        self.attachment_set_max_count = int(attachment_set_max_count)
        self.attachment_set_feature_mode = str(
            attachment_set_feature_mode or "single_scale"
        ).lower()
        if self.attachment_set_feature_mode not in {
            "single_scale",
            "multiscale",
            "multiscale_pointer",
            "multiscale_pointer_heatmap",
        }:
            raise ValueError(
                "attachment_set_feature_mode must be 'single_scale', 'multiscale', "
                "'multiscale_pointer', or 'multiscale_pointer_heatmap', "
                f"got {attachment_set_feature_mode!r}"
            )
        self.attachment_set_feature_levels = max(
            2, int(attachment_set_feature_levels)
        )
        self.attachment_set_max_feature_size = max(
            4, int(attachment_set_max_feature_size)
        )
        self.attachment_set_min_confidence = float(
            attachment_set_min_confidence
        )
        self.attachment_set_max_anchor_distance = float(
            attachment_set_max_anchor_distance
        )
        self.attachment_set_min_pointer_confidence = float(
            attachment_set_min_pointer_confidence
        )
        self.attachment_set_heatmap_logit_scale = float(
            attachment_set_heatmap_logit_scale
        )
        self.attachment_set_heatmap_prior_precision = float(
            attachment_set_heatmap_prior_precision
        )
        self.attachment_set_decode_mode = str(
            attachment_set_decode_mode or "token_mixture"
        ).lower()
        if self.attachment_set_decode_mode not in {
            "direct_sidecar",
            "token_mixture",
            "residual_base",
        }:
            raise ValueError(
                "attachment_set_decode_mode must be 'direct_sidecar', "
                f"'token_mixture', or 'residual_base', got {attachment_set_decode_mode!r}"
            )
        self.sidecar_coordinate_context = str(
            sidecar_coordinate_context or "full"
        ).lower()
        if self.sidecar_coordinate_context not in {"full", "mask_y"}:
            raise ValueError(
                "sidecar_coordinate_context must be 'full' or 'mask_y', "
                f"got {sidecar_coordinate_context!r}"
            )
        self.fragment_terminal_action_head_enabled = bool(
            fragment_terminal_action_head_enabled
        )
        self.fragment_terminal_action_head_hidden_dim = max(
            32, int(fragment_terminal_action_head_hidden_dim)
        )
        self.fragment_structured_terminal_edge_enabled = bool(
            fragment_structured_terminal_edge_enabled
        )
        if self.fragment_terminal_action_head_enabled and (
            self.attachment_set_decode_mode != "direct_sidecar"
            or self.expert_kind != "full_mixture"
            or self.full_mixture_sidecar_mode != "per_sidecar"
        ):
            raise ValueError(
                "fragment terminal action head requires direct_sidecar "
                "full_mixture per_sidecar decoding"
            )
        aliases = {
            "single": "collapsed",
            "single_specialist": "collapsed",
            "shared_specialist": "collapsed",
            "separate": "per_sidecar",
            "per-sidecar": "per_sidecar",
            "per_label": "per_sidecar",
            "three_expert": "per_sidecar",
        }
        self.full_mixture_sidecar_mode = aliases.get(
            self.full_mixture_sidecar_mode,
            self.full_mixture_sidecar_mode,
        )
        if self.expert_kind == "full_mixture" and self.full_mixture_sidecar_mode not in {
            "collapsed",
            "per_sidecar",
        }:
            raise ValueError(
                "full_mixture_sidecar_mode must be 'collapsed' or 'per_sidecar', "
                f"got {full_mixture_sidecar_mode!r}"
            )
        if (
            self.attachment_set_decode_mode == "direct_sidecar"
            and (
                self.expert_kind != "full_mixture"
                or self.full_mixture_sidecar_mode != "per_sidecar"
            )
        ):
            raise ValueError(
                "direct_sidecar decoding requires full_mixture with per_sidecar experts"
            )
        self.compute_confidence = getattr(args, "compute_confidence", False)

        # --- Decoder experts ---
        # expert0 = complete decoder (frozen unless frozen_expert0=False);
        # per-sidecar mode adds expert1=markush, expert2=fragment.
        self.num_replaced = 0
        decoder_count = (
            self.num_experts
            if self.full_mixture_sidecar_mode == "per_sidecar"
            else 2
        )
        self.experts = nn.ModuleList([Decoder(args, tokenizer) for _ in range(decoder_count)])
        self.unfreeze_expert0 = (not bool(frozen_expert0))
        if not self.unfreeze_expert0:
            for _p in self.experts[0].parameters():
                _p.requires_grad_(False)
        # expert1 always trainable.
        self.token_fusion_gates = nn.ModuleList()
        if self.expert_kind == "full_mixture" and self.token_fusion_mode == "adaptive":
            self.token_fusion_gates.extend(
                TokenFusionGate(
                    int(args.dec_hidden_size),
                    initial_sidecar_weight=self.token_fusion_initial_sidecar_weight,
                )
                for _ in range(max(0, len(self.experts) - 1))
            )
        self.attachment_set_heads = nn.ModuleList()
        if self.expert_kind == "full_mixture" and self.attachment_set_enabled:
            if self.attachment_set_feature_mode in {
                "multiscale",
                "multiscale_pointer",
                "multiscale_pointer_heatmap",
            }:
                encoder_dim = int(args.encoder_dim)
                feature_dims = tuple(
                    encoder_dim // (2 ** level)
                    for level in reversed(range(self.attachment_set_feature_levels))
                )
                if self.attachment_set_feature_mode == "multiscale_pointer_heatmap":
                    head_factory = lambda: MultiScaleAttachmentHeatmapPointerHead(
                        feature_dims,
                        atom_feature_dim=int(args.dec_hidden_size),
                        hidden_dim=self.attachment_set_hidden_dim,
                        num_queries=self.attachment_set_num_queries,
                        num_layers=self.attachment_set_num_layers,
                        num_heads=self.attachment_set_num_heads,
                        max_count=self.attachment_set_max_count,
                        max_feature_size=self.attachment_set_max_feature_size,
                        heatmap_logit_scale=self.attachment_set_heatmap_logit_scale,
                        heatmap_prior_precision=(
                            self.attachment_set_heatmap_prior_precision
                        ),
                    )
                elif self.attachment_set_feature_mode == "multiscale_pointer":
                    head_factory = lambda: MultiScaleAttachmentPointerHead(
                        feature_dims,
                        atom_feature_dim=int(args.dec_hidden_size),
                        hidden_dim=self.attachment_set_hidden_dim,
                        num_queries=self.attachment_set_num_queries,
                        num_layers=self.attachment_set_num_layers,
                        num_heads=self.attachment_set_num_heads,
                        max_count=self.attachment_set_max_count,
                        max_feature_size=self.attachment_set_max_feature_size,
                    )
                else:
                    head_factory = lambda: MultiScaleAttachmentSetHead(
                        feature_dims,
                        hidden_dim=self.attachment_set_hidden_dim,
                        num_queries=self.attachment_set_num_queries,
                        num_layers=self.attachment_set_num_layers,
                        num_heads=self.attachment_set_num_heads,
                        max_count=self.attachment_set_max_count,
                        max_feature_size=self.attachment_set_max_feature_size,
                    )
            else:
                head_factory = lambda: AttachmentSetHead(
                    int(args.encoder_dim),
                    hidden_dim=self.attachment_set_hidden_dim,
                    num_queries=self.attachment_set_num_queries,
                    num_layers=self.attachment_set_num_layers,
                    num_heads=self.attachment_set_num_heads,
                    max_count=self.attachment_set_max_count,
                )
            self.attachment_set_heads.extend(
                head_factory()
                for _ in range(max(0, len(self.experts) - 1))
            )
        self.fragment_terminal_action_head = None
        if self.fragment_terminal_action_head_enabled:
            # Keep RNG state so a new auxiliary module does not perturb the
            # initialization of existing router/heads.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20260717)
                self.fragment_terminal_action_head = FragmentTerminalActionHead(
                    int(args.dec_hidden_size),
                    self.fragment_terminal_action_head_hidden_dim,
                )
        # Router + confidence head + (kept-buf) reference of the dominant route.
        self.router = StructureRouter(
            args.encoder_dim,
            self.num_experts,
            kind=self.router_kind,
        )
        self.confidence_head = (
            ConfidenceHead(args.encoder_dim, self.num_experts) if use_confidence_head else None
        )
        # No regime conditioning, no expert0_reference; complete rows short-circuit
        # to the frozen expert0 (lora: LoRAMoELinear buffers; full_mixture: experts[0]).
        self.regime_conditioning = False
        self.regime_emb = None
        self.expert0_reference = None

    @property
    def _decoder(self) -> Decoder:
        """The single LoRA-wrapped base decoder (= ``experts[0]``). Read-only
        property so the module is registered exactly once."""
        return self.experts[0]

    def train(self, mode: bool = True):
        """Keep expert0 behaviorally frozen during training: ``train()`` would
        otherwise re-enable its dropout while deployment runs it in eval mode.
        """
        super().train(mode)
        expert0_is_frozen = (
            self.expert_kind == "lora"
            or not bool(getattr(self, "unfreeze_expert0", False))
        )
        if mode and expert0_is_frozen and len(self.experts):
            self.experts[0].eval()
        if (
            mode
            and self.expert_kind == "full_mixture"
            and getattr(self, "full_mixture_sidecar_mode", "collapsed") == "per_sidecar"
            and getattr(self, "fragment_decoder_train_scope", "full") in {
                "output_edge",
                "last_cross_output_edge",
            }
            and len(self.experts) > 2
        ):
            # Output/edge heads have no train/eval-dependent layers; eval mode
            # disables dropout in the frozen core while keeping gradients.
            self.experts[2].eval()
        reference = getattr(self, "expert0_reference", None)
        if reference is not None:
            reference.eval()
        return self

    # -------------------------------------------------- thresholds (kept verbatim)
    def _normalize_sidecar_thresholds(
        self,
        value: list[float] | dict[str, float] | None,
        *,
        fallback: float,
    ) -> list[float]:
        thresholds = [float("inf")] + [float(fallback)] * max(0, self.num_experts - 1)
        if value is None:
            return thresholds
        if isinstance(value, dict):
            for key, raw in value.items():
                idx = None
                key_s = str(key)
                if key_s.isdigit():
                    idx = int(key_s)
                elif key_s in self.expert_names:
                    idx = self.expert_names.index(key_s)
                if idx is not None and 0 <= idx < self.num_experts:
                    thresholds[idx] = float(raw)
            return thresholds
        for idx, raw in enumerate(list(value)[: self.num_experts]):
            thresholds[idx] = float(raw)
        return thresholds

    # -------------------------------------------------- gate plumbing
    def _set_gate_all(self, gate: torch.Tensor | None) -> None:
        for m in self._decoder.modules():
            if isinstance(m, LoRAMoELinear):
                m.set_gate(gate)

    def _gate_from_expected_labels(
        self,
        weights: torch.Tensor,
        structure_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Build the supervised sidecar gate from the known structure labels:
        complete rows zero-gated, markush/fragment rows use their adapter.
        """
        if self.num_adapter_experts <= 0:
            return weights.new_zeros((weights.size(0), 0))
        labels = structure_labels.to(weights.device).long()
        gate = weights.new_zeros((weights.size(0), self.num_adapter_experts))
        sidecar_rows = labels.gt(0)
        if bool(sidecar_rows.any()):
            adapter_idx = (labels[sidecar_rows] - 1).clamp(
                min=0, max=self.num_adapter_experts - 1)
            gate[sidecar_rows] = F.one_hot(
                adapter_idx, self.num_adapter_experts).to(dtype=weights.dtype)
        return gate

    def _gate_from_weights(
        self,
        weights: torch.Tensor,
        *,
        structure_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Translate router softmax into the inference LoRA gate. Training rows
        pass ``structure_labels`` and use the explicit expected gate instead.
        """
        if structure_labels is not None:
            return self._gate_from_expected_labels(weights, structure_labels)
        if self.num_adapter_experts <= 0:
            return weights.new_zeros((weights.size(0), 0))
        if self.routing_strategy == "sparse_top1":
            idx = weights.argmax(dim=-1)
            sidecar_idx = (idx - 1).clamp(min=0, max=self.num_adapter_experts - 1)
            gate = F.one_hot(sidecar_idx, self.num_adapter_experts).to(weights)
            gate = torch.where(idx.unsqueeze(1) > 0, gate, torch.zeros_like(gate))
        else:
            gate = weights[:, 1: 1 + self.num_adapter_experts].clone()
        return gate

    # -------------------------------------------------- loading
    def load_expert0_from_base(self, base_decoder_states: dict) -> None:
        """Load the frozen base decoder into the LoRA-wrapped decoder. Strips
        a DDP ``module.`` prefix, uses ``strict=False`` for non-wrapped
        submodules plus :func:`load_base_weights_into_lora` for the buffers.
        """
        cleaned = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in base_decoder_states.items()
        }
        # Non-default encoder variants (e.g. swin_large, dim=1536) mismatch
        # enc_trans_layer; skip those keys to keep the fresh Linear.
        if os.environ.get("MOLNEXTR_ENCODER_VARIANT", "").strip() not in ("", "swin_base"):
            cleaned = {k: v for k, v in cleaned.items() if "enc_trans_layer" not in k}
        cleaned = pad_chartok_for_sep(self._decoder, cleaned)
        self._decoder.load_state_dict(cleaned, strict=False)
        load_base_weights_into_lora(self._decoder, cleaned)
        # Unfrozen expert0 needs a frozen reference copy for the KL anchor.
        if getattr(self, "unfreeze_expert0", False) and self.expert_kind == "full_mixture":
            import copy
            self.expert0_reference = copy.deepcopy(self.experts[0])
            for _p in self.expert0_reference.parameters():
                _p.requires_grad_(False)
            self.expert0_reference.eval()

    def load_adapter(self, adapter_states: dict) -> None:
        """Load LoRA ``A``/``B`` params. Older experimental checkpoints stored a
        redundant complete adapter at index 0; slice that column away.
        """
        converted = {}
        for name, tensor in adapter_states.items():
            if (
                (name.endswith(".A") or name.endswith(".B"))
                and isinstance(tensor, torch.Tensor)
                and tensor.ndim >= 1
                and int(tensor.shape[0]) == self.num_experts
                and self.num_adapter_experts == self.num_experts - 1
            ):
                converted[name] = tensor[1:].contiguous()
            else:
                converted[name] = tensor
        self._decoder.load_state_dict(converted, strict=False)

    def load_expert(self, idx: int, decoder_states: dict) -> None:
        """Load a per-expert checkpoint. ``full_mixture``: full-decoder state
        dict into ``experts[idx]``. ``lora``: back-compat shim that loads LoRA
        A/B keys if present (legacy full-decoder checkpoints are ignored).
        """
        if self.expert_kind == "full_mixture":
            cleaned = {
                (k[len("module."):] if k.startswith("module.") else k): v
                for k, v in decoder_states.items()
            }
            cleaned = pad_chartok_for_sep(self.experts[idx], cleaned)
            self.experts[idx].load_state_dict(cleaned, strict=False)
            return
        if any(k.endswith(".A") or k.endswith(".B") for k in decoder_states):
            self.load_adapter(decoder_states)

    def warm_start_specialist_from_base(self, base_decoder_states: dict) -> None:
        """``full_mixture`` only: warm-start trainable sidecar decoder(s) from
        the frozen base so each sidecar only learns its structure-type delta.
        """
        if self.expert_kind != "full_mixture":
            raise RuntimeError("warm_start_specialist_from_base is full_mixture-only")
        cleaned = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in base_decoder_states.items()
        }
        # Skip enc_trans_layer when encoder variant differs (dim mismatch)
        if os.environ.get("MOLNEXTR_ENCODER_VARIANT", "").strip() not in ("", "swin_base"):
            cleaned = {k: v for k, v in cleaned.items() if "enc_trans_layer" not in k}
        if len(self.experts) > 1:
            cleaned = pad_chartok_for_sep(self.experts[1], cleaned)
        for idx in range(1, len(self.experts)):
            self.experts[idx].load_state_dict(cleaned, strict=False)

    def warm_start_from_expert0(self, idx: int) -> None:
        """No-op for LoRA: experts are slices of the shared A/B stack, all
        initialised with B=0 ⇒ already equal to the frozen base."""
        return None

    def load_router(
        self,
        router_states: dict,
        *,
        require_fusion_state: bool = False,
        require_attachment_set_state: bool = False,
        require_terminal_action_state: bool = False,
        allow_attachment_set_pointer_upgrade: bool = False,
        allow_attachment_set_heatmap_upgrade: bool = False,
        allow_direct_graph_upgrade: bool = False,
    ) -> None:
        """Load router plus optional token-fusion gates, validating checkpoint
        architecture metadata before loading so mismatched checkpoints fail
        loudly instead of being misinterpreted.
        """
        payload = router_states
        pointer_upgrade = False
        heatmap_upgrade = False
        dummy_pointer_upgrade = False
        if isinstance(payload, dict):
            checkpoint_router_kind = payload.get("router_kind")
            if (
                checkpoint_router_kind is not None
                and str(checkpoint_router_kind) != self.router_kind
            ):
                raise ValueError(
                    "router checkpoint kind mismatch: "
                    f"checkpoint={checkpoint_router_kind!r}, model={self.router_kind!r}"
                )
            checkpoint_fusion_mode = payload.get("token_fusion_mode")
            if (
                checkpoint_fusion_mode is not None
                and str(checkpoint_fusion_mode) != self.token_fusion_mode
                and not (
                    allow_direct_graph_upgrade
                    and self.attachment_set_decode_mode == "direct_sidecar"
                )
            ):
                raise ValueError(
                    "router checkpoint token fusion mismatch: "
                    f"checkpoint={checkpoint_fusion_mode!r}, "
                    f"model={self.token_fusion_mode!r}"
                )
            checkpoint_decoupled = payload.get("decouple_fusion_policy_optimization")
            if (
                checkpoint_decoupled is not None
                and bool(checkpoint_decoupled)
                != self.decouple_fusion_policy_optimization
            ):
                raise ValueError(
                    "router checkpoint fusion optimization mismatch: "
                    f"checkpoint={bool(checkpoint_decoupled)!r}, "
                    f"model={self.decouple_fusion_policy_optimization!r}"
                )
            checkpoint_dispatch = payload.get("token_fusion_dispatch")
            if (
                checkpoint_dispatch is not None
                and str(checkpoint_dispatch) != self.token_fusion_dispatch
                and not (
                    allow_direct_graph_upgrade
                    and self.attachment_set_decode_mode == "direct_sidecar"
                )
            ):
                raise ValueError(
                    "router checkpoint fusion dispatch mismatch: "
                    f"checkpoint={checkpoint_dispatch!r}, "
                    f"model={self.token_fusion_dispatch!r}"
                )
            checkpoint_hard_threshold = payload.get(
                "token_fusion_hard_threshold"
            )
            if (
                checkpoint_hard_threshold is not None
                and abs(
                    float(checkpoint_hard_threshold)
                    - self.token_fusion_hard_threshold
                )
                > 1e-8
                and not (
                    allow_direct_graph_upgrade
                    and self.attachment_set_decode_mode == "direct_sidecar"
                )
            ):
                raise ValueError(
                    "router checkpoint fusion hard-threshold mismatch: "
                    f"checkpoint={float(checkpoint_hard_threshold)!r}, "
                    f"model={self.token_fusion_hard_threshold!r}"
                )
            checkpoint_ownership = payload.get("specialist_ownership_scope")
            if (
                checkpoint_ownership is not None
                and str(checkpoint_ownership) != self.specialist_ownership_scope
                and not (
                    allow_direct_graph_upgrade
                    and self.attachment_set_decode_mode == "direct_sidecar"
                    and str(checkpoint_ownership) == "attachment"
                    and self.specialist_ownership_scope == "full"
                )
            ):
                raise ValueError(
                    "router checkpoint specialist ownership mismatch: "
                    f"checkpoint={checkpoint_ownership!r}, "
                    f"model={self.specialist_ownership_scope!r}"
                )
            checkpoint_one_sided = payload.get("one_sided_fusion_oracle")
            if (
                checkpoint_one_sided is not None
                and bool(checkpoint_one_sided) != self.one_sided_fusion_oracle
            ):
                raise ValueError(
                    "router checkpoint fusion oracle mismatch: "
                    f"checkpoint={bool(checkpoint_one_sided)!r}, "
                    f"model={self.one_sided_fusion_oracle!r}"
                )
            checkpoint_terminal_enabled = payload.get(
                "fragment_terminal_action_head_enabled"
            )
            if (
                checkpoint_terminal_enabled is not None
                and bool(checkpoint_terminal_enabled)
                != self.fragment_terminal_action_head_enabled
            ):
                raise ValueError(
                    "router checkpoint fragment terminal action head mismatch: "
                    f"checkpoint={bool(checkpoint_terminal_enabled)!r}, "
                    f"model={self.fragment_terminal_action_head_enabled!r}"
                )
            checkpoint_terminal_hidden = payload.get(
                "fragment_terminal_action_head_hidden_dim"
            )
            if (
                self.fragment_terminal_action_head_enabled
                and checkpoint_terminal_hidden is not None
                and int(checkpoint_terminal_hidden)
                != self.fragment_terminal_action_head_hidden_dim
            ):
                raise ValueError(
                    "router checkpoint fragment terminal head width mismatch: "
                    f"checkpoint={int(checkpoint_terminal_hidden)!r}, "
                    f"model={self.fragment_terminal_action_head_hidden_dim!r}"
                )
            checkpoint_structured_terminal_edge = payload.get(
                "fragment_structured_terminal_edge_enabled"
            )
            if (
                checkpoint_structured_terminal_edge is not None
                and bool(checkpoint_structured_terminal_edge)
                != self.fragment_structured_terminal_edge_enabled
            ):
                raise ValueError(
                    "router checkpoint fragment structured terminal edge mismatch: "
                    f"checkpoint={bool(checkpoint_structured_terminal_edge)!r}, "
                    f"model={self.fragment_structured_terminal_edge_enabled!r}"
                )
            checkpoint_attachment_mode = payload.get(
                "attachment_set_feature_mode"
            )
            pointer_upgrade = bool(
                allow_attachment_set_pointer_upgrade
                and str(checkpoint_attachment_mode) == "multiscale"
                and self.attachment_set_feature_mode == "multiscale_pointer"
            )
            heatmap_upgrade = bool(
                allow_attachment_set_heatmap_upgrade
                and str(checkpoint_attachment_mode) == "multiscale_pointer"
                and self.attachment_set_feature_mode
                == "multiscale_pointer_heatmap"
            )
            if (
                checkpoint_attachment_mode is not None
                and str(checkpoint_attachment_mode)
                != self.attachment_set_feature_mode
                and not pointer_upgrade
                and not heatmap_upgrade
            ):
                raise ValueError(
                    "router checkpoint attachment-set feature mismatch: "
                    f"checkpoint={checkpoint_attachment_mode!r}, "
                    f"model={self.attachment_set_feature_mode!r}"
                )
            if self.attachment_set_feature_mode == "multiscale_pointer_heatmap":
                for metadata_key, model_value in (
                    (
                        "attachment_set_heatmap_logit_scale",
                        self.attachment_set_heatmap_logit_scale,
                    ),
                    (
                        "attachment_set_heatmap_prior_precision",
                        self.attachment_set_heatmap_prior_precision,
                    ),
                ):
                    checkpoint_value = payload.get(metadata_key)
                    if (
                        checkpoint_value is not None
                        and abs(float(checkpoint_value) - float(model_value)) > 1e-8
                    ):
                        raise ValueError(
                            f"router checkpoint {metadata_key} mismatch: "
                            f"checkpoint={float(checkpoint_value)!r}, "
                            f"model={float(model_value)!r}"
                        )
        if isinstance(payload, dict) and "router" in payload:
            router_state = payload["router"]
        else:
            router_state = payload
        self.router.load_state_dict(router_state)
        fusion_state = payload.get("token_fusion_gates") if isinstance(payload, dict) else None
        if fusion_state is not None and len(self.token_fusion_gates):
            self.token_fusion_gates.load_state_dict(fusion_state, strict=True)
        elif require_fusion_state and len(self.token_fusion_gates):
            raise ValueError(
                "adaptive token fusion is enabled but the router checkpoint "
                "does not contain token_fusion_gates"
            )
        attachment_set_state = (
            payload.get("attachment_set_heads")
            if isinstance(payload, dict)
            else None
        )
        if attachment_set_state is not None and len(self.attachment_set_heads):
            dummy_pointer_upgrade = not any(
                ".dummy_pointer_" in str(key)
                for key in attachment_set_state.keys()
            ) and any(
                isinstance(head, MultiScaleAttachmentPointerHead)
                for head in self.attachment_set_heads
            )
            if pointer_upgrade or heatmap_upgrade or dummy_pointer_upgrade:
                incompatible = self.attachment_set_heads.load_state_dict(
                    attachment_set_state, strict=False
                )
                unexpected = list(incompatible.unexpected_keys)
                allowed_missing_markers = []
                if pointer_upgrade:
                    allowed_missing_markers.extend(
                        (".pointer_query.", ".pointer_atom.", ".pointer_geometry.")
                    )
                if heatmap_upgrade:
                    allowed_missing_markers.extend(
                        (".point_query.", ".point_memory.", ".point_offset.")
                    )
                if dummy_pointer_upgrade:
                    allowed_missing_markers.extend(
                        (
                            ".dummy_pointer_query.",
                            ".dummy_pointer_atom.",
                            ".dummy_pointer_geometry.",
                        )
                    )
                disallowed_missing = [
                    key
                    for key in incompatible.missing_keys
                    if not any(
                        marker in key
                        for marker in allowed_missing_markers
                    )
                ]
                if unexpected or disallowed_missing:
                    raise ValueError(
                        "attachment set upgrade checkpoint mismatch: "
                        f"unexpected={unexpected}, missing={disallowed_missing}"
                    )
            else:
                self.attachment_set_heads.load_state_dict(
                    attachment_set_state, strict=True
                )
        elif require_attachment_set_state and len(self.attachment_set_heads):
            raise ValueError(
                "attachment set prediction is enabled but the router checkpoint "
                "does not contain attachment_set_heads"
            )
        terminal_action_state = (
            payload.get("fragment_terminal_action_head")
            if isinstance(payload, dict)
            else None
        )
        if (
            terminal_action_state is not None
            and self.fragment_terminal_action_head is not None
        ):
            incompatible = self.fragment_terminal_action_head.load_state_dict(
                terminal_action_state,
                strict=False,
            )
            disallowed_missing = [
                key for key in incompatible.missing_keys
                if not str(key).startswith("stop_net.")
            ]
            if incompatible.unexpected_keys or disallowed_missing:
                raise ValueError(
                    "fragment terminal action head checkpoint mismatch: "
                    f"unexpected={list(incompatible.unexpected_keys)}, "
                    f"missing={disallowed_missing}"
                )
        elif (
            require_terminal_action_state
            and self.fragment_terminal_action_head is not None
        ):
            raise ValueError(
                "fragment terminal action head is enabled but the router "
                "checkpoint does not contain its state"
            )

    def _token_fusion_gate(self, sidecar_expert_idx: int) -> TokenFusionGate | None:
        gate_index = int(sidecar_expert_idx) - 1
        if (
            self.token_fusion_mode != "adaptive"
            or gate_index < 0
            or gate_index >= len(self.token_fusion_gates)
        ):
            return None
        return self.token_fusion_gates[gate_index]

    def _dispatch_fusion_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        """Convert calibrated gate probabilities into the deployed policy:
        ``hard`` quantizes so every step is emitted by exactly one expert.
        """
        if self.token_fusion_dispatch == "hard":
            return alpha.ge(self.token_fusion_hard_threshold).to(dtype=alpha.dtype)
        return alpha

    # -------------------------------------------------- training forward
    def forward(self, features: torch.Tensor, refs: dict,
                structure_labels: torch.Tensor | None = None,
                encoder_hiddens: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None) -> dict:
        """DDP-compatible training entry point."""
        return self.training_forward(
            features,
            refs,
            structure_labels=structure_labels,
            encoder_hiddens=encoder_hiddens,
        )

    def _sidecar_context_labels(self, labels, atom_indices):
        """Replace y-coordinate label inputs with MASK (the context used by
        coordinate-free rows and direct inference) while ``token_target``
        keeps the original labels.
        """
        if (
            self.attachment_set_decode_mode != "direct_sidecar"
            or self.sidecar_coordinate_context != "mask_y"
            or atom_indices is None
        ):
            return labels
        indices = (
            atom_indices[0]
            if isinstance(atom_indices, (list, tuple))
            else atom_indices
        )
        if not isinstance(indices, torch.Tensor) or indices.numel() == 0:
            return labels
        indices = indices.to(device=labels.device, dtype=torch.long)
        result = labels.clone()
        valid = indices.gt(0) & indices.lt(labels.size(1))
        if bool(valid.any()):
            batch = torch.arange(
                labels.size(0), device=labels.device
            ).unsqueeze(1).expand_as(indices)
            result[batch[valid], indices[valid]] = MASK_ID
        return result

    def _fragment_terminal_teacher_outputs(
        self,
        *,
        expert_logits_by_idx,
        expert_dec_out_by_idx,
        labels: torch.Tensor,
        token_target: torch.Tensor,
        structure_labels: torch.Tensor | None,
    ) -> dict | None:
        """Build head-only EOS/attachment supervision on canonical prefixes."""
        head = self.fragment_terminal_action_head
        if (
            head is None
            or structure_labels is None
            or expert_logits_by_idx is None
            or expert_dec_out_by_idx is None
            or len(expert_logits_by_idx) <= 2
            or len(expert_dec_out_by_idx) <= 2
        ):
            return None
        star_id = getattr(self.tokenizer[ATOM_FORMAT], "stoi", {}).get("*")
        if star_id is None:
            raise RuntimeError("fragment terminal action head requires a '*' token")
        token_steps = min(
            token_target.size(1),
            expert_logits_by_idx[2].size(1),
            expert_dec_out_by_idx[2].size(1),
        )
        if token_steps <= 0:
            return None
        base_logits = expert_logits_by_idx[2][:, :token_steps]
        hidden = expert_dec_out_by_idx[2][:, :token_steps]
        base_pair = torch.stack(
            [
                base_logits[..., EOS_ID],
                base_logits[..., int(star_id)],
            ],
            dim=-1,
        )
        # Prediction t conditions on labels through position t; the cumulative
        # flag separates the first terminal dummy from the EOS after it.
        prefix = labels[:, :token_steps]
        has_attachment = prefix.eq(int(star_id)).cumsum(dim=1).gt(0)
        step_fraction = torch.arange(
            token_steps,
            device=labels.device,
            dtype=torch.float32,
        ).view(1, -1).expand(labels.size(0), -1)
        step_fraction = step_fraction / float(max(1, token_steps - 1))
        pair_logits = head.policy_pair_logits(
            hidden,
            base_pair,
            has_attachment,
            step_fraction,
        )
        stop_logits = head.stop_policy_logits(
            hidden,
            base_logits,
            eos_token_id=EOS_ID,
            has_attachment=has_attachment,
            step_fraction=step_fraction,
        )
        targets = token_target[:, :token_steps]
        fragment_mask = structure_labels.to(labels.device).long().eq(2).unsqueeze(1)
        mask = fragment_mask & (targets.eq(EOS_ID) | targets.eq(int(star_id)))
        stop_mask = fragment_mask & targets.eq(EOS_ID) & has_attachment
        return {
            "logits": pair_logits,
            "targets": targets.eq(int(star_id)).long(),
            "mask": mask,
            "stop_logits": stop_logits,
            "stop_targets": torch.ones_like(targets, dtype=torch.long),
            "stop_mask": stop_mask,
        }

    def training_forward(self, features: torch.Tensor, refs: dict,
                         structure_labels: torch.Tensor | None = None,
                         encoder_hiddens: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None) -> dict:
        """Teacher-forced forward. The router emits its class distribution for
        auxiliary supervision; the decoder gate follows the explicit
        structure-label contract (the same path expected_structure_types
        forces at no-fallback inference).
        """
        atom_format = ATOM_FORMAT
        device = features.device
        labels = refs[atom_format][0].to(device)
        label_lengths = refs[atom_format][1].to(device)
        refs = dict(refs)
        refs[atom_format] = [labels, label_lengths]
        if "edges" in refs and isinstance(refs["edges"], torch.Tensor):
            refs["edges"] = refs["edges"].to(device)
        decoder_task_mask = refs.get("decoder_task_mask")
        if isinstance(decoder_task_mask, torch.Tensor):
            decoder_task_mask = decoder_task_mask.to(device, dtype=torch.bool)
            refs["decoder_task_mask"] = decoder_task_mask
        for key, dtype in (
            ("attachment_points", torch.float32),
            ("attachment_point_mask", torch.bool),
            ("attachment_set_complete", torch.bool),
            ("attachment_count", torch.long),
            ("attachment_bonded", torch.long),
            ("attachment_bond_type", torch.long),
            ("attachment_dummy_indices", torch.long),
            ("attachment_anchor_points", torch.float32),
            ("attachment_anchor_mask", torch.bool),
            ("attachment_anchor_indices", torch.long),
            ("attachment_atom_coords", torch.float32),
        ):
            value = refs.get(key)
            if isinstance(value, torch.Tensor):
                refs[key] = value.to(device=device, dtype=dtype)
        atom_indices = refs.get("atom_indices")
        if atom_indices is not None:
            ai = atom_indices
            refs["atom_indices"] = [ai[0].to(device) if isinstance(ai[0], torch.Tensor) else ai[0],
                                    ai[1] if len(ai) > 1 else None]

        router_logits = self.router(features)               # (B, K)
        weights = router_logits.softmax(dim=-1)             # (B, K)
        token_target = labels[:, 1:]
        sidecar_labels = self._sidecar_context_labels(
            labels,
            refs.get("atom_indices"),
        )
        attachment_set_outputs = [
            (
                head(features, encoder_hiddens)
                if self.attachment_set_feature_mode in {
                    "multiscale",
                    "multiscale_pointer",
                    "multiscale_pointer_heatmap",
                }
                else head(features)
            )
            for head in self.attachment_set_heads
        ]

        token_logits_0 = None                                # KL teacher (full_mixture only)
        ref_logits_0 = None
        expert_logits_by_idx = None
        expert_dec_out_by_idx = None
        if self.expert_kind == "full_mixture":
            if getattr(self, "unfreeze_expert0", False) and self.expert0_reference is not None:
                # v8-style: expert0 TRAINABLE (fine-tuned on fragments for backbone).
                # Run WITH grad so the preserve-loss can update it.
                token_logits_0, _target0, dec_out_0 = self.experts[0].decoder[atom_format](
                    features, labels, label_lengths, logit_bias=None)
                # Frozen reference for the KL anchor that preserves complete.
                with torch.no_grad():
                    ref_logits_0, _, _ = self.expert0_reference.decoder[atom_format](
                        features, labels, label_lengths, logit_bias=None)
                ref_logits_0 = ref_logits_0.detach()
            else:
                # Expert0 frozen, but a trainable encoder's complete-row loss must
                # still propagate through this fixed decoder.
                if features.requires_grad:
                    token_logits_0, _target0, dec_out_0 = self.experts[0].decoder[atom_format](
                        features, labels, label_lengths, logit_bias=None)
                else:
                    with torch.no_grad():
                        token_logits_0, _target0, dec_out_0 = self.experts[0].decoder[atom_format](
                            features, labels, label_lengths, logit_bias=None)
                    token_logits_0 = token_logits_0.detach()
            if (
                self.full_mixture_sidecar_mode == "per_sidecar"
                and structure_labels is not None
                and len(self.experts) >= self.num_experts
            ):
                # Each sidecar trains only on its regime (1 -> markush,
                # 2 -> fragment); complete rows use expert0 logits.
                expert_logits = [token_logits_0]
                expert_dec_out_by_idx = [dec_out_0]
                for expert_idx in range(1, self.num_experts):
                    logits_i, _tgt_i, dec_out_i = self.experts[expert_idx].decoder[atom_format](
                        features, sidecar_labels, label_lengths, logit_bias=None)
                    expert_logits.append(logits_i)
                    expert_dec_out_by_idx.append(dec_out_i)
                expert_logits_by_idx = expert_logits
                labels_dev = structure_labels.to(device).long().clamp(min=0, max=self.num_experts - 1)
                token_logits = torch.empty_like(expert_logits[0])
                dec_out = torch.empty_like(expert_dec_out_by_idx[0])
                for expert_idx, logits_i in enumerate(expert_logits):
                    row_mask = labels_dev.eq(expert_idx).view(-1, 1, 1)
                    token_logits = torch.where(row_mask, logits_i, token_logits)
                    dec_out = torch.where(row_mask, expert_dec_out_by_idx[expert_idx], dec_out)
                if "edges" in refs and refs.get("atom_indices") is not None:
                    edge_head_owner = None
                else:
                    edge_head_owner = self.experts[1].decoder["edges"]
            else:
                # Legacy collapsed mode: one specialist learns both markush and
                # fragment. Kept for old checkpoint compatibility only.
                token_logits, _tgt, dec_out = self.experts[1].decoder[atom_format](
                    features, sidecar_labels, label_lengths, logit_bias=None)
                expert_logits_by_idx = [token_logits_0, token_logits]
                expert_dec_out_by_idx = [dec_out_0, dec_out]
                edge_head_owner = self.experts[1].decoder["edges"]
            gate = None
        else:
            gate = self._gate_from_weights(
                weights, structure_labels=structure_labels)  # (B, K-1), complete rows = 0
            self._set_gate_all(gate)
            token_logits, _tgt, dec_out = self._decoder.decoder[atom_format](
                features, labels, label_lengths, logit_bias=None)
            edge_head_owner = self._decoder.decoder["edges"]

        attachment_atom_candidate_mask = None
        if (
            self.attachment_set_feature_mode in {
                "multiscale_pointer",
                "multiscale_pointer_heatmap",
            }
            and attachment_set_outputs
            and expert_dec_out_by_idx is not None
            and refs.get("atom_indices") is not None
        ):
            atom_index_values, atom_index_lengths = refs["atom_indices"]
            atom_index_values = atom_index_values.to(device=device, dtype=torch.long)
            lengths = atom_index_lengths.to(device=device).reshape(-1).long()
            attachment_atom_candidate_mask = (
                torch.arange(atom_index_values.size(1), device=device).unsqueeze(0)
                < lengths.unsqueeze(1)
            )
            dummy_mask = self._dummy_atom_mask_from_labels(
                labels,
                atom_index_values,
            )
            dummy_mask = (
                dummy_mask.to(device=device, dtype=torch.bool)
                if dummy_mask is not None
                else torch.zeros_like(attachment_atom_candidate_mask)
            )
            anchor_candidate_mask = attachment_atom_candidate_mask & ~dummy_mask
            atom_coords = refs.get("attachment_atom_coords")
            for sidecar_idx, (head, outputs) in enumerate(
                zip(self.attachment_set_heads, attachment_set_outputs),
                start=1,
            ):
                if isinstance(head, MultiScaleAttachmentPointerHead):
                    if sidecar_idx >= len(expert_dec_out_by_idx):
                        raise RuntimeError(
                            "attachment head has no matching sidecar decoder hidden state"
                        )
                    sidecar_hidden = expert_dec_out_by_idx[sidecar_idx]
                    gather_indices = atom_index_values.clamp(
                        min=0,
                        max=max(0, sidecar_hidden.size(1) - 1),
                    )
                    gather_shape = gather_indices.unsqueeze(-1).expand(
                        -1, -1, sidecar_hidden.size(-1)
                    )
                    atom_features = sidecar_hidden.gather(1, gather_shape)
                    outputs["anchor_pointer_logits"] = head.anchor_pointer_logits(
                        outputs,
                        atom_features,
                        atom_coords=atom_coords,
                        atom_mask=anchor_candidate_mask,
                    )
                    outputs["dummy_pointer_logits"] = head.dummy_pointer_logits(
                        outputs,
                        atom_features,
                        atom_coords=atom_coords,
                        atom_mask=attachment_atom_candidate_mask,
                    )

        token_fusion_logits = None
        token_fusion_alpha = None
        token_mixed_log_probs = None
        if (
            self.expert_kind == "full_mixture"
            and self.token_fusion_mode == "adaptive"
            and token_logits_0 is not None
            and expert_dec_out_by_idx is not None
        ):
            token_steps = token_logits.size(1)
            lp0 = F.log_softmax(token_logits_0.float(), dim=-1)
            lp1 = F.log_softmax(token_logits.float(), dim=-1)
            token_fusion_logits = token_logits.new_zeros(token_logits.shape[:2])
            if structure_labels is None:
                selected_experts = weights[:, 1:].argmax(dim=1) + 1
            else:
                selected_experts = structure_labels.to(device).long().clamp(
                    min=0, max=len(self.experts) - 1)
            max_sidecar_idx = min(len(self.experts), len(expert_dec_out_by_idx))
            for expert_idx in range(1, max_sidecar_idx):
                fusion_gate = self._token_fusion_gate(expert_idx)
                if fusion_gate is None:
                    continue
                base_hidden = dec_out_0[:, :token_steps]
                sidecar_hidden = expert_dec_out_by_idx[expert_idx][:, :token_steps]
                if self.decouple_fusion_policy_optimization:
                    # Detach so the fusion policy loss cannot reshape decoder
                    # hidden states merely to ease routing.
                    base_hidden = base_hidden.detach()
                    sidecar_hidden = sidecar_hidden.detach()
                gate_logits_i = fusion_gate(
                    base_hidden,
                    sidecar_hidden,
                    lp0,
                    lp1,
                )
                row_mask = selected_experts.eq(expert_idx).view(-1, 1)
                token_fusion_logits = torch.where(
                    row_mask,
                    gate_logits_i,
                    token_fusion_logits,
                )
            token_fusion_alpha = torch.sigmoid(token_fusion_logits.float()).to(
                dtype=token_logits.dtype)
            task_alpha = self._dispatch_fusion_alpha(token_fusion_alpha.float())
            if self.decouple_fusion_policy_optimization:
                # Mixture CE trains the specialist under the deployed policy;
                # the likelihood-ratio objective trains the policy itself.
                task_alpha = task_alpha.detach()
            log_alpha = torch.log(task_alpha.clamp(min=1e-5)).unsqueeze(-1)
            log_base = torch.log((1.0 - task_alpha).clamp(min=1e-5)).unsqueeze(-1)
            token_mixed_log_probs = torch.logsumexp(
                torch.stack([lp0 + log_base, lp1 + log_alpha], dim=0),
                dim=0,
            )

        # Edge heads use the same expert pair as token decoding; the fusion
        # alpha is projected onto atom pairs and trained against the edge mixture.
        edge_logits, edge_target = None, refs.get("edges")
        edge_logits_expert0 = None
        edge_mixed_log_probs = None
        edge_fusion_alpha = None
        ai = refs.get("atom_indices")
        if ai is not None and edge_target is not None:
            idx = ai[0] if isinstance(ai, (list, tuple)) else ai
            if (
                self.expert_kind == "full_mixture"
                and self.full_mixture_sidecar_mode == "per_sidecar"
                and structure_labels is not None
                and len(self.experts) >= self.num_experts
            ):
                labels_dev = structure_labels.to(device).long().clamp(min=0, max=self.num_experts - 1)
                edge_logits = None
                for expert_idx in range(self.num_experts):
                    hidden_i = (
                        expert_dec_out_by_idx[expert_idx]
                        if expert_dec_out_by_idx is not None
                        else dec_out
                    )
                    edge_pred_i = self.experts[expert_idx].decoder["edges"](hidden_i, indices=idx)
                    edge_logits_i = edge_pred_i["edges"]
                    if expert_idx == 0:
                        edge_logits_expert0 = edge_logits_i
                    if edge_logits is None:
                        edge_logits = torch.empty_like(edge_logits_i)
                    row_mask = labels_dev.eq(expert_idx).view(-1, 1, 1, 1)
                    edge_logits = torch.where(row_mask, edge_logits_i, edge_logits)
            else:
                edge_pred = edge_head_owner(dec_out, indices=idx)
                edge_logits = edge_pred["edges"]                # (B, 7, n, n)
                if self.expert_kind == "full_mixture" and token_logits_0 is not None:
                    edge_logits_expert0 = self.experts[0].decoder["edges"](
                        dec_out_0,
                        indices=idx,
                    )["edges"]

            if (
                self.expert_kind == "full_mixture"
                and edge_logits_expert0 is not None
                and edge_logits is not None
            ):
                if token_fusion_alpha is not None:
                    token_steps = token_fusion_alpha.size(1)
                    gather_idx = (idx.long() - 1).clamp(
                        min=0,
                        max=max(0, token_steps - 1),
                    )
                    atom_alpha = token_fusion_alpha.float().gather(1, gather_idx)
                    edge_fusion_alpha = torch.maximum(
                        atom_alpha.unsqueeze(2),
                        atom_alpha.unsqueeze(1),
                    )
                    edge_fusion_alpha = self._dispatch_fusion_alpha(
                        edge_fusion_alpha
                    )
                else:
                    w2 = self._weights_to_mixture_2_batch(
                        weights,
                        structure_labels=(
                            structure_labels
                            if self.full_mixture_sidecar_mode == "per_sidecar"
                            else None
                        ),
                    )
                    edge_fusion_alpha = w2[:, 1].view(-1, 1, 1).expand(
                        -1,
                        edge_logits.size(2),
                        edge_logits.size(3),
                    )
                edge_fusion_alpha = edge_fusion_alpha.clamp(min=1e-5, max=1.0 - 1e-5)
                edge_task_alpha = edge_fusion_alpha
                if self.decouple_fusion_policy_optimization:
                    edge_task_alpha = edge_task_alpha.detach()
                edge_lp0 = F.log_softmax(edge_logits_expert0.float(), dim=1)
                edge_lp1 = F.log_softmax(edge_logits.float(), dim=1)
                log_sidecar = torch.log(edge_task_alpha).unsqueeze(1)
                log_base = torch.log1p(-edge_task_alpha).unsqueeze(1)
                edge_mixed_log_probs = torch.logsumexp(
                    torch.stack(
                        [edge_lp0 + log_base, edge_lp1 + log_sidecar],
                        dim=0,
                    ),
                    dim=0,
                )

        # Per-sample NLL features for the confidence head (fp32, detached grad).
        B = features.size(0)
        tok_v = token_logits.size(-1)
        tok_ce = F.cross_entropy(
            token_logits.float().reshape(-1, tok_v), token_target.reshape(-1),
            ignore_index=PAD_ID, reduction="none").view(B, -1)
        tok_mask = (token_target != PAD_ID).logical_and(
            token_target != MASK_ID).float()
        token_nll = (tok_ce * tok_mask).sum(dim=1) / tok_mask.sum(dim=1).clamp(min=1.0)
        edge_nll = torch.zeros(B, device=device)
        if edge_logits is not None and edge_target is not None:
            e_v = edge_logits.size(1)
            e_ce = F.cross_entropy(
                edge_logits.float().permute(0, 2, 3, 1).reshape(-1, e_v),
                edge_target.reshape(-1), ignore_index=-100, reduction="none")
            e_ce = e_ce.view(B, edge_target.size(1), edge_target.size(2))
            e_mask = (edge_target != -100).float()
            denom = e_mask.sum(dim=(1, 2)).clamp(min=1.0)
            edge_nll = (e_ce * e_mask).sum(dim=(1, 2)) / denom

        pooled_features = features.mean(dim=1)
        ai_out = refs.get("atom_indices")
        fragment_terminal_teacher = self._fragment_terminal_teacher_outputs(
            expert_logits_by_idx=expert_logits_by_idx,
            expert_dec_out_by_idx=expert_dec_out_by_idx,
            labels=labels,
            token_target=token_target,
            structure_labels=structure_labels,
        )
        return {
            "token_logits": token_logits,
            "token_logits_expert0": token_logits_0,
            "token_logits_expert0_ref": ref_logits_0,
            "token_fusion_logits": token_fusion_logits,
            "token_fusion_alpha": token_fusion_alpha,
            "token_mixed_log_probs": token_mixed_log_probs,
            "token_target": token_target,
            "edge_logits": edge_logits,
            "edge_logits_expert0": edge_logits_expert0,
            "edge_mixed_log_probs": edge_mixed_log_probs,
            "edge_fusion_alpha": edge_fusion_alpha,
            "edge_target": edge_target,
            "labels": labels,
            "atom_indices": ai_out[0] if ai_out is not None else None,
            "router_logits": router_logits,
            "weights": weights,
            "gate": gate,
            "pooled_features": pooled_features,
            "token_nll": token_nll,
            "edge_nll": edge_nll,
            "decoder_task_mask": decoder_task_mask,
            "encoder_features_require_grad": bool(features.requires_grad),
            "attachment_set_outputs": attachment_set_outputs,
            "attachment_points": refs.get("attachment_points"),
            "attachment_point_mask": refs.get("attachment_point_mask"),
            "attachment_set_complete": refs.get("attachment_set_complete"),
            "attachment_count": refs.get("attachment_count"),
            "attachment_bonded": refs.get("attachment_bonded"),
            "attachment_bond_type": refs.get("attachment_bond_type"),
            "attachment_dummy_indices": refs.get("attachment_dummy_indices"),
            "attachment_anchor_points": refs.get("attachment_anchor_points"),
            "attachment_anchor_mask": refs.get("attachment_anchor_mask"),
            "attachment_anchor_indices": refs.get("attachment_anchor_indices"),
            "attachment_atom_candidate_mask": attachment_atom_candidate_mask,
            "attachment_anchor_candidate_mask": (
                anchor_candidate_mask
                if attachment_atom_candidate_mask is not None else None
            ),
            "fragment_terminal_action_logits": (
                None
                if fragment_terminal_teacher is None
                else fragment_terminal_teacher["logits"]
            ),
            "fragment_terminal_action_targets": (
                None
                if fragment_terminal_teacher is None
                else fragment_terminal_teacher["targets"]
            ),
            "fragment_terminal_action_mask": (
                None
                if fragment_terminal_teacher is None
                else fragment_terminal_teacher["mask"]
            ),
            "fragment_terminal_stop_logits": (
                None
                if fragment_terminal_teacher is None
                else fragment_terminal_teacher["stop_logits"]
            ),
            "fragment_terminal_stop_targets": (
                None
                if fragment_terminal_teacher is None
                else fragment_terminal_teacher["stop_targets"]
            ),
            "fragment_terminal_stop_mask": (
                None
                if fragment_terminal_teacher is None
                else fragment_terminal_teacher["stop_mask"]
            ),
        }

    def confidence_head_forward(self, fwd: dict) -> torch.Tensor | None:
        if self.confidence_head is None:
            return None
        pf = fwd["pooled_features"].clone()
        w = fwd["weights"].clone()
        tn = fwd["token_nll"].clone()
        en = fwd["edge_nll"].clone()
        return self.confidence_head(pf, w, tn, en)

    # -------------------------------------------------- loss weighting (kept verbatim)
    def _token_loss_weights(
        self, target, structure_labels, *, markush_symbol_weight,
        fragment_symbol_weight, aromatic_symbol_weight, unsaturated_symbol_weight,
        attachment_span_mask=None, fragment_eos_weight: float = 1.0,
    ):
        if structure_labels is None:
            return None
        weights = torch.ones_like(target, dtype=torch.float32)
        labels = structure_labels.to(target.device)
        markush_rows = labels.eq(1).view(-1, 1)
        fragment_rows = labels.eq(2).view(-1, 1)
        if not bool((markush_rows | fragment_rows).any()):
            return weights
        stoi = getattr(self.tokenizer[ATOM_FORMAT], "stoi", {})

        def ids_for(chars):
            return [int(stoi[ch]) for ch in chars if ch in stoi]

        aromatic_ids = ids_for(["c", "n", "o", "p", "s"])
        unsat_ids = ids_for(["=", "#", "/", "\\"])

        def apply(ids, row_mask, value):
            if not ids or float(value) <= 1.0:
                return
            token_mask = torch.zeros_like(target, dtype=torch.bool)
            for token_id in ids:
                token_mask |= target.eq(int(token_id))
            boosted = torch.full_like(weights, float(value))
            weights.copy_(torch.where(row_mask & token_mask, torch.maximum(weights, boosted), weights))

        if attachment_span_mask is not None:
            span = attachment_span_mask.to(target.device, dtype=torch.bool)
            markush_boost = torch.full_like(weights, float(markush_symbol_weight))
            fragment_boost = torch.full_like(weights, float(fragment_symbol_weight))
            weights = torch.where(
                markush_rows & span,
                torch.maximum(weights, markush_boost),
                weights,
            )
            weights = torch.where(
                fragment_rows & span,
                torch.maximum(weights, fragment_boost),
                weights,
            )
        apply(aromatic_ids, markush_rows | fragment_rows, aromatic_symbol_weight)
        apply(unsat_ids, markush_rows | fragment_rows, unsaturated_symbol_weight)
        if float(fragment_eos_weight) > 1.0:
            eos = target.eq(EOS_ID)
            boosted = torch.full_like(weights, float(fragment_eos_weight))
            weights = torch.where(
                fragment_rows & eos,
                torch.maximum(weights, boosted),
                weights,
            )
        return weights

    @staticmethod
    def _masked_loss_mean(values, weights, *, reduction: str):
        reduction = str(reduction or "per_sample")
        if reduction == "global":
            return (values * weights).sum() / weights.sum().clamp(min=1.0)
        if reduction != "per_sample":
            raise ValueError(f"unsupported loss reduction: {reduction!r}")
        dims = tuple(range(1, values.ndim))
        numer = (values * weights).sum(dim=dims)
        denom = weights.sum(dim=dims)
        active = denom.gt(0)
        per_sample = numer / denom.clamp(min=1.0)
        return per_sample[active].mean() if bool(active.any()) else numer.sum() * 0.0

    def _weighted_token_ce(self, logits, target, weights, *, pad_index,
                           reduction: str = "per_sample"):
        vocab = logits.size(-1)
        ce = F.cross_entropy(
            logits.reshape(-1, vocab), target.reshape(-1),
            ignore_index=pad_index, reduction="none",
        ).view_as(target)
        mask = target.ne(pad_index).logical_and(target.ne(MASK_ID)).float()
        if weights is None:
            weights_f = mask
        else:
            weights_f = weights.to(ce.device, dtype=ce.dtype) * mask
        return self._masked_loss_mean(ce, weights_f, reduction=reduction)

    def _edge_loss_weights(
        self, target, structure_labels, *, dummy_atom_mask=None,
        sidecar_nonzero_edge_weight, dummy_edge_weight, multiple_edge_weight,
        aromatic_edge_weight, edge_ignore,
    ):
        if structure_labels is None:
            return None
        weights = torch.ones_like(target, dtype=torch.float32)
        labels = structure_labels.to(target.device)
        sidecar_rows = labels.gt(0).view(-1, 1, 1)
        if not bool(sidecar_rows.any()):
            return weights
        valid = target.ne(edge_ignore)
        nonzero = valid & target.ne(0)
        if float(sidecar_nonzero_edge_weight) > 1.0:
            boosted = torch.full_like(weights, float(sidecar_nonzero_edge_weight))
            weights = torch.where(sidecar_rows & nonzero, torch.maximum(weights, boosted), weights)
        if dummy_atom_mask is not None and float(dummy_edge_weight) > 1.0:
            dummy = dummy_atom_mask.to(target.device, dtype=torch.bool)
            if dummy.ndim == 2 and dummy.shape[:2] == target.shape[:2]:
                dummy_incident = dummy.unsqueeze(2) | dummy.unsqueeze(1)
                boosted = torch.full_like(weights, float(dummy_edge_weight))
                weights = torch.where(
                    sidecar_rows & nonzero & dummy_incident,
                    torch.maximum(weights, boosted), weights,
                )
        if float(multiple_edge_weight) > 1.0:
            multiple = target.eq(2) | target.eq(3)
            boosted = torch.full_like(weights, float(multiple_edge_weight))
            weights = torch.where(sidecar_rows & multiple, torch.maximum(weights, boosted), weights)
        if float(aromatic_edge_weight) > 1.0:
            aromatic = target.eq(4)
            boosted = torch.full_like(weights, float(aromatic_edge_weight))
            weights = torch.where(sidecar_rows & aromatic, torch.maximum(weights, boosted), weights)
        weights = torch.where(target.eq(edge_ignore), torch.zeros_like(weights), weights)
        return weights

    def _weighted_edge_ce(self, logits, target, weights, *, edge_ignore,
                          reduction: str = "per_sample"):
        classes = logits.size(1)
        ce = F.cross_entropy(
            logits.float().permute(0, 2, 3, 1).reshape(-1, classes),
            target.reshape(-1), ignore_index=edge_ignore, reduction="none",
        ).view_as(target)
        mask = target.ne(edge_ignore).float()
        if weights is None:
            weights_f = mask
        else:
            weights_f = weights.to(ce.device, dtype=ce.dtype) * mask
        return self._masked_loss_mean(ce, weights_f, reduction=reduction)

    def _valence_violation_loss(
        self,
        edge_logits: torch.Tensor,
        edge_target: torch.Tensor,
        symbols_per_row: list[list[str]],
        *,
        edge_ignore: int = -100,
        reduction: str = "per_sample",
    ) -> torch.Tensor | None:
        """Differentiable over-valence penalty: penalize the expected used
        valence (from the edge softmax) in excess of each atom's permitted
        maximum. Returns None when inputs are unusable; caller skips the term.
        """
        if edge_logits is None or edge_target is None or not symbols_per_row:
            return None
        B, C, N, _ = edge_logits.shape
        if C != 7 or N == 0 or len(symbols_per_row) != B:
            return None
        device = edge_logits.device

        # Per-class valence weight: [no, single, double, triple, aromatic, wedge, dash]
        # (matches bond_valence_weight; aromatic = 1.0).
        class_w = torch.tensor(
            [0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0],
            device=device, dtype=torch.float32,
        )

        # Expected valence contribution per pair must condition on a bond
        # existing: E[contribution | i,j] = P(bond) * E[order | bond], with the
        # order re-normalized over classes 1..6; otherwise softmax background on
        # padded pairs leaks spurious valence into every off-diagonal slot.
        probs = F.softmax(edge_logits.float(), dim=1)              # (B, 7, N, N)
        p_bond = 1.0 - probs[:, 0]                                 # (B, N, N)
        p_bond_safe = p_bond.clamp(min=1e-6)
        numerator = (probs * class_w.view(1, 7, 1, 1)).sum(dim=1)  # (B, N, N)
        cond_order = numerator / p_bond_safe                       # (B, N, N)
        per_bond_val = p_bond * cond_order
        # Zero the diagonal to exclude self-bond phantom valence.
        eye = torch.eye(N, device=device).unsqueeze(0)  # (1, N, N)
        per_bond_val = per_bond_val * (1.0 - eye)
        expected_used = per_bond_val.sum(dim=2)                    # (B, N)

        # Build the per-atom valence cap from symbol strings.
        cap = torch.zeros(B, N, device=device, dtype=torch.float32)
        valid = torch.zeros(B, N, device=device, dtype=torch.float32)
        for b in range(B):
            syms = symbols_per_row[b]
            for n in range(min(N, len(syms))):
                # Skip atoms whose edges are entirely padding (no valid targets).
                row = edge_target[b, n]
                if bool((row.eq(edge_ignore)).all()):
                    continue
                cap[b, n] = float(max_valence_for_symbol(syms[n]))
                # Atoms we deliberately treat as permissive (unknown abbreviation
                # returns 99.0) get zero weight so they do not dilute the mean.
                if cap[b, n] >= 90.0:
                    continue
                valid[b, n] = 1.0
        if float(valid.sum().item()) < 0.5:
            return None

        # Soft hinge: only penalize expected_used > cap.
        excess = torch.clamp(expected_used - cap, min=0.0)          # (B, N)
        weighted = excess * valid                                   # (B, N)
        if reduction == "per_sample":
            # Mean over atoms within each row, then mean over rows that have any
            # valid atom (matches _masked_loss_mean's per_sample semantics).
            per_row_denom = valid.sum(dim=1).clamp(min=1.0)
            per_row = weighted.sum(dim=1) / per_row_denom           # (B,)
            active_rows = (valid.sum(dim=1) > 0).float()
            denom = active_rows.sum().clamp(min=1.0)
            return (per_row * active_rows).sum() / denom
        # sum / scalar fallback
        return weighted.sum() / valid.sum().clamp(min=1.0)

    def _dummy_symbol_bounds(self, labels_row, y_index: int):
        """Return inclusive label bounds for a dummy/R-group atom symbol."""
        tokenizer = self.tokenizer[ATOM_FORMAT]
        stoi = getattr(tokenizer, "stoi", {})
        if y_index <= 1 or y_index >= len(labels_row):
            return None
        symbol_end = int(y_index) - 2
        symbol_start = symbol_end
        close_id = stoi.get("]")
        open_id = stoi.get("[")
        if close_id is not None and int(labels_row[symbol_end]) == int(close_id):
            while symbol_start >= 0:
                if open_id is not None and int(labels_row[symbol_start]) == int(open_id):
                    break
                symbol_start -= 1
            if symbol_start < 0:
                return None
        itos = getattr(tokenizer, "itos", {})
        chars = []
        for position in range(symbol_start, symbol_end + 1):
            token_id = int(labels_row[position])
            token = itos.get(token_id, "") if isinstance(itos, dict) else ""
            chars.append(str(token))
        symbol = "".join(chars)
        star_id = stoi.get("*")
        has_star_id = star_id is not None and any(
            int(labels_row[position]) == int(star_id)
            for position in range(symbol_start, symbol_end + 1)
        )
        if (
            not has_star_id
            and "*" not in symbol
            and not is_markush_label(normalize_label(symbol))
        ):
            return None
        return symbol_start, symbol_end

    def _atom_symbols_from_labels(self, labels, atom_indices):
        """Recover each atom slot's symbol string from the chartok_coords
        layout ``[x, y, <symbol_chars...>]`` (``y_idx - 2`` is the last symbol
        char; ``]`` starts a walk back to the matching ``[``).
        """
        if labels is None or atom_indices is None:
            return []
        tokenizer = self.tokenizer[ATOM_FORMAT]
        itos = getattr(tokenizer, "itos", {})
        stoi = getattr(tokenizer, "stoi", {})
        open_id = stoi.get("[")
        close_id = stoi.get("]")
        labels_cpu = labels.detach().cpu()
        atom_indices_cpu = atom_indices.detach().cpu()
        B, N = atom_indices_cpu.shape
        rows: list[list[str]] = []
        for b in range(B):
            row_labels = labels_cpu[b]
            symbols: list[str] = []
            for n in range(N):
                y_idx = int(atom_indices_cpu[b, n])
                symbol = ""
                if 1 < y_idx < len(row_labels):
                    last_pos = y_idx - 2
                    last_id = int(row_labels[last_pos])
                    if close_id is not None and last_id == int(close_id):
                        # Bracketed atom: walk back to matching '['.
                        start = last_pos - 1
                        while start >= 0:
                            if (
                                open_id is not None
                                and int(row_labels[start]) == int(open_id)
                            ):
                                break
                            start -= 1
                        if start < 0:
                            start = last_pos  # fallback: just the ']'
                        symbol = "".join(
                            str(itos.get(int(row_labels[p]), ""))
                            for p in range(start, last_pos + 1)
                        )
                    else:
                        # Non-bracketed: may be multi-char (Cl, Br, Si).
                        # Walk back while preceding tokens are symbol chars.
                        start = last_pos
                        while start - 1 >= 0:
                            prev_id = int(row_labels[start - 1])
                            prev_char = str(itos.get(prev_id, ""))
                            if not prev_char or prev_char.isalpha():
                                start -= 1
                            else:
                                break
                        symbol = "".join(
                            str(itos.get(int(row_labels[p]), ""))
                            for p in range(start, last_pos + 1)
                        )
                symbols.append(symbol)
            rows.append(symbols)
        return rows

    def _dummy_token_span_mask(self, labels, atom_indices, target_length: int | None = None):
        """Mark symbol and x/y target positions for every dummy/R-group atom."""
        if labels is None or atom_indices is None:
            return None
        if getattr(self.tokenizer[ATOM_FORMAT], "stoi", {}).get("*") is None:
            return None
        labels_cpu = labels.detach().cpu()
        atom_indices_cpu = atom_indices.detach().cpu()
        length = int(target_length or max(0, labels.size(1) - 1))
        mask = torch.zeros(labels.size(0), length, dtype=torch.bool, device=labels.device)
        for batch_index in range(atom_indices_cpu.size(0)):
            for atom_index in range(atom_indices_cpu.size(1)):
                y_index = int(atom_indices_cpu[batch_index, atom_index])
                bounds = self._dummy_symbol_bounds(
                    labels_cpu[batch_index],
                    y_index,
                )
                if bounds is None:
                    continue
                symbol_start, _symbol_end = bounds
                # labels carry SOS at position 0; shift symbol_start by one into
                # token_target, with y_index as the exclusive end.
                target_start = max(0, symbol_start - 1)
                target_end = min(length, y_index)
                if target_start < target_end:
                    mask[batch_index, target_start:target_end] = True
        # Keep the literal attachment char supervised even with incomplete
        # atom-index metadata.
        star_id = int(self.tokenizer[ATOM_FORMAT].stoi["*"])
        target_tokens = labels[:, 1 : 1 + length]
        mask[:, : target_tokens.size(1)] |= target_tokens.eq(star_id)
        return mask

    def _attachment_extension_span_mask(
        self,
        labels,
        target_length: int | None = None,
    ):
        """Mark ``<sep>[anchor:*]`` tokens, excluding the trailing EOS."""

        if labels is None:
            return None
        sep_id = getattr(self.tokenizer[ATOM_FORMAT], "sep_id", None)
        if sep_id is None:
            return None
        length = int(target_length or max(0, labels.size(1) - 1))
        mask = torch.zeros(
            labels.size(0), length, dtype=torch.bool, device=labels.device
        )
        labels_cpu = labels.detach().cpu()
        for batch_index, row in enumerate(labels_cpu):
            separators = row.eq(int(sep_id)).nonzero(as_tuple=False).view(-1)
            if separators.numel() != 1:
                continue
            separator_position = int(separators[0])
            eos_positions = (
                row[separator_position + 1 :]
                .eq(EOS_ID)
                .nonzero(as_tuple=False)
                .view(-1)
            )
            if eos_positions.numel() == 0:
                continue
            eos_position = separator_position + 1 + int(eos_positions[0])
            target_start = max(0, separator_position - 1)
            target_end = min(length, eos_position - 1)
            if target_start < target_end:
                mask[batch_index, target_start:target_end] = True
        return mask

    def _attachment_extension_identity_mask(
        self,
        labels,
        target_length: int | None = None,
    ):
        """Mark the anchor-index digits inside a native attachment extension."""

        if labels is None:
            return None
        tokenizer = self.tokenizer[ATOM_FORMAT]
        sep_id = getattr(tokenizer, "sep_id", None)
        stoi = getattr(tokenizer, "stoi", {})
        colon_id = stoi.get(":")
        digit_ids = {
            int(stoi[str(value)])
            for value in range(10)
            if str(value) in stoi
        }
        if sep_id is None or colon_id is None or not digit_ids:
            return None
        length = int(target_length or max(0, labels.size(1) - 1))
        mask = torch.zeros(
            labels.size(0), length, dtype=torch.bool, device=labels.device
        )
        labels_cpu = labels.detach().cpu()
        for batch_index, row in enumerate(labels_cpu):
            separators = row.eq(int(sep_id)).nonzero(as_tuple=False).view(-1)
            if separators.numel() != 1:
                continue
            for label_position in range(int(separators[0]) + 1, len(row)):
                token_id = int(row[label_position])
                if token_id == int(colon_id) or token_id == EOS_ID:
                    break
                target_position = label_position - 1
                if token_id in digit_ids and 0 <= target_position < length:
                    mask[batch_index, target_position] = True
        return mask

    def _variable_identity_token_mask(
        self,
        labels,
        atom_indices,
        target_length: int | None = None,
    ):
        """Mask label-identity characters such as ``1`` in ``[1*]``/``R1``."""
        if labels is None or atom_indices is None:
            return None
        tokenizer = self.tokenizer[ATOM_FORMAT]
        itos = getattr(tokenizer, "itos", {})
        if not isinstance(itos, dict):
            return None
        labels_cpu = labels.detach().cpu()
        atom_indices_cpu = atom_indices.detach().cpu()
        length = int(target_length or max(0, labels.size(1) - 1))
        mask = torch.zeros(labels.size(0), length, dtype=torch.bool, device=labels.device)
        structural_chars = {"", "[", "]", "*", ":", "+", "-"}
        for batch_index in range(atom_indices_cpu.size(0)):
            for atom_index in range(atom_indices_cpu.size(1)):
                y_index = int(atom_indices_cpu[batch_index, atom_index])
                bounds = self._dummy_symbol_bounds(labels_cpu[batch_index], y_index)
                if bounds is None:
                    continue
                symbol_start, symbol_end = bounds
                for label_position in range(symbol_start, symbol_end + 1):
                    token = str(itos.get(int(labels_cpu[batch_index, label_position]), ""))
                    if token in structural_chars:
                        continue
                    target_position = label_position - 1
                    if 0 <= target_position < length:
                        mask[batch_index, target_position] = True
        return mask

    def _dummy_atom_mask_from_labels(self, labels, atom_indices):
        if labels is None or atom_indices is None:
            return None
        if getattr(self.tokenizer[ATOM_FORMAT], "stoi", {}).get("*") is None:
            return None
        labels_cpu = labels.detach().cpu()
        atom_indices_cpu = atom_indices.detach().cpu()
        B, N = atom_indices_cpu.shape
        mask = torch.zeros(B, N, dtype=torch.bool, device=labels.device)
        for b in range(B):
            for n in range(N):
                y_idx = int(atom_indices_cpu[b, n])
                if self._dummy_symbol_bounds(labels_cpu[b], y_idx) is not None:
                    mask[b, n] = True
        return mask

    # -------------------------------------------------- loss
    def compute_moe_loss(
        self, fwd: dict, structure_labels: torch.Tensor | None = None,
        load_balance_weight: float = 0.05, z_loss_weight: float = 1e-3,
        structure_weight: float = 0.5,
        router_margin_weight: float = 0.0, router_margin: float = 2.0,
        expert_diversity_weight: float = 0.0,
        markush_symbol_weight: float = 1.0, fragment_symbol_weight: float = 1.0,
        aromatic_symbol_weight: float = 1.0, unsaturated_symbol_weight: float = 1.0,
        attachment_symbol_loss_weight: float = 0.0,
        variable_identity_loss_weight: float = 0.0,
        attachment_cardinality_loss_weight: float = 0.0,
        fragment_eos_weight: float = 1.0,
        fragment_terminal_dummy_margin_weight: float = 0.0,
        fragment_terminal_dummy_margin: float = 2.0,
        fragment_premature_eos_unlikelihood_weight: float = 0.0,
        fragment_terminal_action_head_loss_weight: float = 0.0,
        token_fusion_supervision_weight: float = 0.0,
        token_fusion_attachment_target: float = 0.95,
        token_fusion_backbone_target: float = 0.15,
        token_fusion_oracle_temperature: float = 1.0,
        sidecar_nonzero_edge_weight: float = 1.0, dummy_edge_weight: float = 1.0,
        multiple_edge_weight: float = 1.0, aromatic_edge_weight: float = 1.0,
        pad_index: int = PAD_ID, edge_ignore: int = -100,
        # legacy args accepted & ignored (trainer back-compat)
        sidecar_dense_weight: float = 0.0, sidecar_specialist_weight: float = 0.0,
        distill_complete_weight: float = 0.0, anchor_l2_weight: float = 0.0,
        expert0_preserve_weight: float = 0.0, distill_temperature: float = 2.0,
        routed_mixture_weight: float = 0.0,
        mixture_token_ce_weight: float = 0.0,
        mixture_edge_ce_weight: float = 0.0,
        edge_distill_weight: float = 0.0,
        attachment_set_loss_weight: float = 0.0,
        attachment_set_point_loss_weight: float = 5.0,
        attachment_set_cardinality_loss_weight: float = 1.0,
        attachment_set_relation_loss_weight: float = 1.0,
        attachment_set_anchor_loss_weight: float = 5.0,
        attachment_set_pointer_loss_weight: float = 3.0,
        attachment_set_dummy_pointer_loss_weight: float = 3.0,
        attachment_set_heatmap_loss_weight: float = 2.0,
        attachment_set_heatmap_cost_weight: float = 2.0,
        attachment_set_heatmap_diversity_loss_weight: float = 0.5,
        edge_valence_loss_weight: float = 0.0,
        loss_reduction: str = "per_sample",
    ) -> dict:
        """Task token/edge CE plus router auxiliaries (load balance, z-loss,
        structure CE, margins, optional diversity/distillation terms).
        """
        weights = fwd["weights"].float()
        router_logits = fwd["router_logits"].float()
        K = weights.size(1)

        attachment_span_mask = self._dummy_token_span_mask(
            fwd.get("labels"),
            fwd.get("atom_indices"),
            target_length=fwd["token_target"].size(1),
        )
        extension_span_mask = self._attachment_extension_span_mask(
            fwd.get("labels"),
            target_length=fwd["token_target"].size(1),
        )
        if extension_span_mask is not None:
            attachment_span_mask = (
                extension_span_mask
                if attachment_span_mask is None
                else attachment_span_mask | extension_span_mask
            )
        variable_identity_mask = self._variable_identity_token_mask(
            fwd.get("labels"),
            fwd.get("atom_indices"),
            target_length=fwd["token_target"].size(1),
        )
        extension_identity_mask = self._attachment_extension_identity_mask(
            fwd.get("labels"),
            target_length=fwd["token_target"].size(1),
        )
        if extension_identity_mask is not None:
            variable_identity_mask = (
                extension_identity_mask
                if variable_identity_mask is None
                else variable_identity_mask | extension_identity_mask
            )
        token_weights = self._token_loss_weights(
            fwd["token_target"], structure_labels,
            markush_symbol_weight=markush_symbol_weight,
            fragment_symbol_weight=fragment_symbol_weight,
            aromatic_symbol_weight=aromatic_symbol_weight,
            unsaturated_symbol_weight=unsaturated_symbol_weight,
            attachment_span_mask=attachment_span_mask,
            fragment_eos_weight=fragment_eos_weight,
        )
        decoder_task_mask = fwd.get("decoder_task_mask")
        if decoder_task_mask is None:
            decoder_task_mask = torch.ones(
                fwd["token_target"].size(0),
                dtype=torch.bool,
                device=fwd["token_target"].device,
            )
        else:
            decoder_task_mask = decoder_task_mask.to(
                fwd["token_target"].device,
                dtype=torch.bool,
            )
        sidecar_task_rows = None
        if (
            self.expert_kind == "full_mixture"
            and self.full_mixture_sidecar_mode == "per_sidecar"
            and structure_labels is not None
            and not getattr(self, "unfreeze_expert0", False)
            and not bool(fwd.get("encoder_features_require_grad", False))
        ):
            # Complete rows short-circuit to frozen expert0 at inference; mask
            # them out of task CE so they don't dilute sidecar gradients.
            sidecar_task_rows = (
                structure_labels.to(fwd["token_target"].device).gt(0)
                & decoder_task_mask
            )
            sidecar_token_mask = sidecar_task_rows.view(-1, 1).to(dtype=torch.float32)
            token_weights = (
                sidecar_token_mask
                if token_weights is None
                else token_weights * sidecar_token_mask
            )
            if self.specialist_ownership_scope == "attachment":
                ownership_mask = torch.zeros_like(
                    fwd["token_target"], dtype=torch.bool
                )
                if attachment_span_mask is not None:
                    ownership_mask |= attachment_span_mask.to(
                        ownership_mask.device, dtype=torch.bool
                    )
                ownership_mask |= (
                    structure_labels.to(ownership_mask.device).eq(2).view(-1, 1)
                    & fwd["token_target"].eq(EOS_ID)
                )
                token_weights = token_weights * ownership_mask.to(
                    dtype=token_weights.dtype
                )
        edge_weights = None
        dummy_atom_mask = None
        if fwd["edge_target"] is not None:
            dummy_atom_mask = self._dummy_atom_mask_from_labels(
                fwd.get("labels"), fwd.get("atom_indices"))
            edge_weights = self._edge_loss_weights(
                fwd["edge_target"], structure_labels,
                dummy_atom_mask=dummy_atom_mask,
                sidecar_nonzero_edge_weight=sidecar_nonzero_edge_weight,
                dummy_edge_weight=dummy_edge_weight,
                multiple_edge_weight=multiple_edge_weight,
                aromatic_edge_weight=aromatic_edge_weight,
                edge_ignore=edge_ignore,
            )
            if sidecar_task_rows is not None:
                sidecar_edge_mask = sidecar_task_rows.view(-1, 1, 1).to(dtype=torch.float32)
                edge_weights = (
                    sidecar_edge_mask
                    if edge_weights is None
                    else edge_weights * sidecar_edge_mask
                )
            if (
                self.specialist_ownership_scope == "attachment"
                and dummy_atom_mask is not None
            ):
                dummy = dummy_atom_mask.to(
                    fwd["edge_target"].device, dtype=torch.bool
                )
                dummy_incident = dummy.unsqueeze(2) | dummy.unsqueeze(1)
                edge_weights = edge_weights * dummy_incident.to(
                    dtype=edge_weights.dtype
                )

        token_loss = self._weighted_token_ce(
            fwd["token_logits"], fwd["token_target"], token_weights,
            pad_index=pad_index, reduction=loss_reduction)
        mixture_loss = token_loss
        loss = token_loss
        attachment_token_loss = None
        if (
            float(attachment_symbol_loss_weight) > 0.0
            and structure_labels is not None
            and attachment_span_mask is not None
        ):
            labels_dev = structure_labels.to(fwd["token_target"].device)
            sidecar_span = (
                (labels_dev.gt(0) & decoder_task_mask).view(-1, 1)
                & attachment_span_mask
                & fwd["token_target"].ne(MASK_ID)
            )
            if bool(sidecar_span.any()):
                vocab = fwd["token_logits"].size(-1)
                span_ce = F.cross_entropy(
                    fwd["token_logits"].float().reshape(-1, vocab),
                    fwd["token_target"].reshape(-1),
                    ignore_index=pad_index,
                    reduction="none",
                ).view_as(fwd["token_target"])
                attachment_token_loss = self._masked_loss_mean(
                    span_ce,
                    sidecar_span.to(dtype=span_ce.dtype),
                    reduction=loss_reduction,
                )
                loss = loss + float(attachment_symbol_loss_weight) * attachment_token_loss

        terminal_dummy_margin_loss = None
        premature_eos_unlikelihood_loss = None
        if structure_labels is not None:
            target = fwd["token_target"]
            fragment_rows = (
                structure_labels.to(target.device).eq(2)
                & decoder_task_mask.to(target.device)
            ).view(-1, 1)
            tokenizer = self.tokenizer[ATOM_FORMAT]
            star_id = getattr(tokenizer, "stoi", {}).get("*")
            sep_id = getattr(tokenizer, "sep_id", None)
            if (
                star_id is not None
                and float(fragment_terminal_dummy_margin_weight) > 0.0
            ):
                row_has_sep = (
                    target.eq(int(sep_id)).any(dim=1, keepdim=True)
                    if sep_id is not None
                    else torch.zeros(
                        target.size(0), 1, dtype=torch.bool, device=target.device
                    )
                )
                boundary = fragment_rows & torch.where(
                    row_has_sep,
                    target.eq(int(sep_id)) if sep_id is not None else target.eq(-1),
                    target.eq(int(star_id)),
                )
                if bool(boundary.any()):
                    logits = fwd["token_logits"].float()
                    action_logit = logits.gather(
                        dim=-1,
                        index=target.clamp(
                            min=0, max=logits.size(-1) - 1
                        ).unsqueeze(-1),
                    ).squeeze(-1)
                    eos_logit = logits[..., EOS_ID]
                    boundary_hinge = F.relu(
                        float(fragment_terminal_dummy_margin)
                        - action_logit
                        + eos_logit
                    )
                    terminal_dummy_margin_loss = self._masked_loss_mean(
                        boundary_hinge,
                        boundary.to(dtype=boundary_hinge.dtype),
                        reduction=loss_reduction,
                    )
                    loss = loss + (
                        float(fragment_terminal_dummy_margin_weight)
                        * terminal_dummy_margin_loss
                    )
            if float(fragment_premature_eos_unlikelihood_weight) > 0.0:
                before_eos = (
                    fragment_rows
                    & target.ne(EOS_ID)
                    & target.ne(pad_index)
                    & target.ne(MASK_ID)
                )
                if bool(before_eos.any()):
                    eos_probability = F.softmax(
                        fwd["token_logits"].float(), dim=-1
                    )[..., EOS_ID]
                    eos_unlikelihood = -torch.log1p(
                        -eos_probability.clamp(max=1.0 - 1.0e-6)
                    )
                    premature_eos_unlikelihood_loss = self._masked_loss_mean(
                        eos_unlikelihood,
                        before_eos.to(dtype=eos_unlikelihood.dtype),
                        reduction=loss_reduction,
                    )
                    loss = loss + (
                        float(fragment_premature_eos_unlikelihood_weight)
                        * premature_eos_unlikelihood_loss
                    )

        fragment_terminal_action_head_loss = None
        terminal_head_components = []
        terminal_pair_logits = fwd.get("fragment_terminal_action_logits")
        terminal_pair_targets = fwd.get("fragment_terminal_action_targets")
        terminal_pair_mask = fwd.get("fragment_terminal_action_mask")
        if (
            float(fragment_terminal_action_head_loss_weight) > 0.0
            and isinstance(terminal_pair_logits, torch.Tensor)
            and isinstance(terminal_pair_targets, torch.Tensor)
            and isinstance(terminal_pair_mask, torch.Tensor)
            and bool(terminal_pair_mask.any())
        ):
            pair_ce = F.cross_entropy(
                terminal_pair_logits.float().reshape(-1, 2),
                terminal_pair_targets.reshape(-1),
                reduction="none",
            ).view_as(terminal_pair_targets)
            terminal_head_components.append(self._masked_loss_mean(
                pair_ce,
                terminal_pair_mask.to(dtype=pair_ce.dtype),
                reduction=loss_reduction,
            ))
        terminal_stop_logits = fwd.get("fragment_terminal_stop_logits")
        terminal_stop_targets = fwd.get("fragment_terminal_stop_targets")
        terminal_stop_mask = fwd.get("fragment_terminal_stop_mask")
        if (
            float(fragment_terminal_action_head_loss_weight) > 0.0
            and isinstance(terminal_stop_logits, torch.Tensor)
            and isinstance(terminal_stop_targets, torch.Tensor)
            and isinstance(terminal_stop_mask, torch.Tensor)
            and bool(terminal_stop_mask.any())
        ):
            stop_ce = F.cross_entropy(
                terminal_stop_logits.float().reshape(-1, 2),
                terminal_stop_targets.reshape(-1),
                reduction="none",
            ).view_as(terminal_stop_targets)
            terminal_head_components.append(self._masked_loss_mean(
                stop_ce,
                terminal_stop_mask.to(dtype=stop_ce.dtype),
                reduction=loss_reduction,
            ))
        if terminal_head_components:
            fragment_terminal_action_head_loss = torch.stack(
                terminal_head_components
            ).mean()
            loss = loss + (
                float(fragment_terminal_action_head_loss_weight)
                * fragment_terminal_action_head_loss
            )

        variable_identity_loss = None
        if (
            float(variable_identity_loss_weight) > 0.0
            and structure_labels is not None
            and variable_identity_mask is not None
        ):
            target = fwd["token_target"]
            markush_identity = (
                structure_labels.to(target.device).eq(1).view(-1, 1)
                & decoder_task_mask.to(target.device).view(-1, 1)
                & variable_identity_mask.to(target.device)
                & target.ne(MASK_ID)
            )
            if bool(markush_identity.any()):
                vocab = fwd["token_logits"].size(-1)
                identity_ce = F.cross_entropy(
                    fwd["token_logits"].float().reshape(-1, vocab),
                    target.reshape(-1),
                    ignore_index=pad_index,
                    reduction="none",
                ).view_as(target)
                variable_identity_loss = self._masked_loss_mean(
                    identity_ce,
                    markush_identity.to(dtype=identity_ce.dtype),
                    reduction=loss_reduction,
                )
                loss = loss + (
                    float(variable_identity_loss_weight) * variable_identity_loss
                )

        attachment_cardinality_loss = None
        if float(attachment_cardinality_loss_weight) > 0.0 and structure_labels is not None:
            star_id = getattr(self.tokenizer[ATOM_FORMAT], "stoi", {}).get("*")
            if star_id is not None:
                policy_lp = fwd.get("token_mixed_log_probs")
                if policy_lp is None:
                    policy_lp = F.log_softmax(fwd["token_logits"].float(), dim=-1)
                target = fwd["token_target"]
                valid = target.ne(pad_index) & target.ne(MASK_ID)
                sidecar_rows = (
                    structure_labels.to(target.device).gt(0)
                    & decoder_task_mask.to(target.device)
                )
                expected_count = (policy_lp[..., int(star_id)].exp() * valid).sum(dim=1)
                target_count = target.eq(int(star_id)).sum(dim=1).to(dtype=expected_count.dtype)
                if bool(sidecar_rows.any()):
                    attachment_cardinality_loss = F.smooth_l1_loss(
                        expected_count[sidecar_rows],
                        target_count[sidecar_rows],
                    )
                    loss = loss + (
                        float(attachment_cardinality_loss_weight) * attachment_cardinality_loss
                    )
        edge_loss = None
        if fwd["edge_logits"] is not None and fwd["edge_target"] is not None:
            edge_loss = self._weighted_edge_ce(
                fwd["edge_logits"], fwd["edge_target"], edge_weights,
                edge_ignore=edge_ignore, reduction=loss_reduction)
            mixture_loss = mixture_loss + edge_loss
            loss = loss + edge_loss

        # Valence penalty on predicted edge probabilities; sidecar rows only.
        edge_valence_loss = None
        # Gate on structure_labels > 0 directly: sidecar_task_rows is
        # intentionally None during encoder fine-tuning.
        sidecar_valence_rows = None
        if structure_labels is not None and decoder_task_mask is not None:
            sidecar_valence_rows = (
                structure_labels.to(fwd["edge_target"].device).gt(0)
                & decoder_task_mask.to(fwd["edge_target"].device)
            )
        if (
            fwd["edge_logits"] is not None
            and fwd["edge_target"] is not None
            and float(edge_valence_loss_weight) > 0.0
            and sidecar_valence_rows is not None
            and bool(sidecar_valence_rows.any())
        ):
            symbols_per_row = self._atom_symbols_from_labels(
                fwd.get("labels"), fwd.get("atom_indices")
            )
            edge_valence_loss = self._valence_violation_loss(
                fwd["edge_logits"], fwd["edge_target"], symbols_per_row,
                edge_ignore=edge_ignore, reduction=loss_reduction,
            )
            if edge_valence_loss is not None:
                loss = loss + float(edge_valence_loss_weight) * edge_valence_loss


        mixture_edge_ce_loss = None
        edge_mixed_lp = fwd.get("edge_mixed_log_probs")
        if (
            edge_mixed_lp is not None
            and fwd["edge_target"] is not None
            and float(mixture_edge_ce_weight) > 0.0
        ):
            edge_target = fwd["edge_target"]
            edge_classes = edge_mixed_lp.size(1)
            mixed_edge_nll = F.nll_loss(
                edge_mixed_lp.permute(0, 2, 3, 1).reshape(-1, edge_classes),
                edge_target.reshape(-1),
                ignore_index=edge_ignore,
                reduction="none",
            ).view_as(edge_target)
            mixed_edge_weights = edge_target.ne(edge_ignore).to(
                dtype=mixed_edge_nll.dtype)
            if edge_weights is not None:
                mixed_edge_weights = edge_weights.to(
                    mixed_edge_nll.device,
                    dtype=mixed_edge_nll.dtype,
                )
            mixture_edge_ce_loss = self._masked_loss_mean(
                mixed_edge_nll,
                mixed_edge_weights,
                reduction=loss_reduction,
            )
            loss = loss + float(mixture_edge_ce_weight) * mixture_edge_ce_loss

        edge_distill_loss = None
        edge_teacher = fwd.get("edge_logits_expert0")
        edge_student = fwd.get("edge_logits")
        if (
            edge_teacher is not None
            and edge_student is not None
            and fwd.get("edge_target") is not None
            and float(edge_distill_weight) > 0.0
            and structure_labels is not None
        ):
            edge_target = fwd["edge_target"]
            non_dummy = edge_target.ne(edge_ignore)
            if dummy_atom_mask is not None:
                dummy = dummy_atom_mask.to(edge_target.device, dtype=torch.bool)
                non_dummy &= ~(
                    dummy.unsqueeze(2) | dummy.unsqueeze(1)
                )
            non_dummy &= (
                structure_labels.to(edge_target.device).gt(0)
                & decoder_task_mask.to(edge_target.device)
            ).view(-1, 1, 1)
            if bool(non_dummy.any()):
                T = max(1e-3, float(distill_temperature))
                student_lp = F.log_softmax(edge_student.float() / T, dim=1)
                teacher_prob = F.softmax(
                    edge_teacher.float().detach() / T, dim=1
                )
                edge_kl = F.kl_div(
                    student_lp, teacher_prob, reduction="none"
                ).sum(dim=1)
                edge_distill_loss = (T * T) * self._masked_loss_mean(
                    edge_kl,
                    non_dummy.to(dtype=edge_kl.dtype),
                    reduction=loss_reduction,
                )
                loss = loss + float(edge_distill_weight) * edge_distill_loss

        # KL anchor: keep the specialist's non-attachment tokens close to the
        # frozen complete expert; attachment/pad positions stay free.
        token_distill_loss = None
        teacher = fwd.get("token_logits_expert0")
        if teacher is not None and float(distill_complete_weight) > 0.0:
            target = fwd["token_target"]
            non_attach = (target != pad_index) & (target != MASK_ID)
            if attachment_span_mask is not None:
                non_attach = non_attach & ~attachment_span_mask
            if (
                self.expert_kind == "full_mixture"
                and self.full_mixture_sidecar_mode == "per_sidecar"
                and structure_labels is not None
            ):
                # In per-sidecar mode the anchor regularizes sidecar rows only.
                non_attach = (
                    non_attach
                    & structure_labels.to(target.device).gt(0).view(-1, 1)
                    & decoder_task_mask.to(target.device).view(-1, 1)
                )
            if bool(non_attach.any()):
                T = max(1e-3, float(distill_temperature))
                student = F.log_softmax(fwd["token_logits"].float() / T, dim=-1)
                teach = F.softmax(teacher.float().detach() / T, dim=-1)
                kl = F.kl_div(student, teach, reduction="none").sum(dim=-1)   # (B, L)
                m = non_attach.float()
                token_distill_loss = (T * T) * self._masked_loss_mean(
                    kl, m, reduction=loss_reduction)
                loss = loss + float(distill_complete_weight) * token_distill_loss

        # expert0 preserve loss: KL(expert0 || frozen reference) on complete
        # rows only, used when expert0 is unfrozen.
        expert0_preserve_loss = None
        ref = fwd.get("token_logits_expert0_ref")
        e0_logits = fwd.get("token_logits_expert0")
        if (ref is not None and e0_logits is not None
                and float(expert0_preserve_weight) > 0.0
                and structure_labels is not None):
            complete_mask = (structure_labels.to(e0_logits.device) == 0)
            if bool(complete_mask.any()):
                target = fwd["token_target"]
                pad_mask = (target != pad_index) & (target != MASK_ID)
                m = (complete_mask.view(-1, 1).expand_as(pad_mask) & pad_mask).float()
                T = max(1e-3, float(distill_temperature))
                student0 = F.log_softmax(e0_logits.float() / T, dim=-1)
                ref_teach = F.softmax(ref.float().detach() / T, dim=-1)
                kl0 = F.kl_div(student0, ref_teach, reduction="none").sum(dim=-1)
                expert0_preserve_loss = (T * T) * self._masked_loss_mean(
                    kl0, m, reduction=loss_reduction)
                loss = loss + float(expert0_preserve_weight) * expert0_preserve_loss

        # Mixture token CE: train the actual inference mixture distribution so
        # the optimized objective matches what decode emits (complete rows have
        # w_complete=1, i.e. the frozen expert0 under no_grad).
        mixture_token_ce_loss = None
        teacher_logits = fwd.get("token_logits_expert0")
        student_logits = fwd.get("token_logits")
        if (self.expert_kind == "full_mixture" and teacher_logits is not None
                and student_logits is not None and float(mixture_token_ce_weight) > 0.0):
            mixed_lp = fwd.get("token_mixed_log_probs")
            if mixed_lp is None:
                w2 = self._weights_to_mixture_2_batch(
                    weights,
                    structure_labels=structure_labels
                    if self.full_mixture_sidecar_mode == "per_sidecar"
                    else None,
                )        # (B, 2) detached, floored
                log_w0 = torch.log(w2[:, 0]).clamp(min=-30.0).view(-1, 1, 1)
                log_w1 = torch.log(w2[:, 1]).clamp(min=-30.0).view(-1, 1, 1)
                lp0 = F.log_softmax(teacher_logits.float(), dim=-1)
                lp1 = F.log_softmax(student_logits.float(), dim=-1)
                mixed_lp = torch.logsumexp(
                    torch.stack([lp0 + log_w0, lp1 + log_w1], dim=0), dim=0)
            target = fwd["token_target"]
            v_mixed = mixed_lp.size(-1)
            mce = F.nll_loss(
                mixed_lp.reshape(-1, v_mixed), target.reshape(-1),
                ignore_index=pad_index, reduction="none").view_as(target)
            mce_mask = ((target != pad_index) & (target != MASK_ID)).float()
            if sidecar_task_rows is not None:
                mce_mask = mce_mask * sidecar_task_rows.view(-1, 1).to(dtype=mce_mask.dtype)
            mixture_token_ce_loss = self._masked_loss_mean(
                mce, mce_mask, reduction=loss_reduction)
            loss = loss + float(mixture_token_ce_weight) * mixture_token_ce_loss

        token_fusion_supervision_loss = None
        fusion_alpha_attachment = None
        fusion_alpha_backbone = None
        fusion_oracle_attachment = None
        fusion_oracle_backbone = None
        fusion_logits = fwd.get("token_fusion_logits")
        fusion_alpha = fwd.get("token_fusion_alpha")
        if (
            fusion_logits is not None
            and structure_labels is not None
            and float(token_fusion_supervision_weight) > 0.0
        ):
            target = fwd["token_target"]
            labels_dev = structure_labels.to(target.device)
            sidecar = labels_dev.gt(0).view(-1, 1)
            fragment = labels_dev.eq(2).view(-1, 1)
            valid = (
                target.ne(pad_index)
                & target.ne(MASK_ID)
                & sidecar
                & decoder_task_mask.to(target.device).view(-1, 1)
            )
            high_gate = target.eq(EOS_ID) & fragment
            if attachment_span_mask is not None:
                high_gate = high_gate | (attachment_span_mask & sidecar)
            prior = torch.full_like(
                fusion_logits.float(), float(token_fusion_backbone_target))
            prior = torch.where(
                high_gate,
                torch.full_like(prior, float(token_fusion_attachment_target)),
                prior,
            )
            # Competence-aware oracle: semantic prior plus the gold-token
            # likelihood ratio moves the target toward the better expert.
            prior = prior.clamp(min=1e-4, max=1.0 - 1e-4)
            soft_target = prior
            teacher_logits = fwd.get("token_logits_expert0")
            student_logits = fwd.get("token_logits")
            if teacher_logits is not None and student_logits is not None:
                with torch.no_grad():
                    gather_target = target.clamp(
                        min=0,
                        max=student_logits.size(-1) - 1,
                    ).unsqueeze(-1)
                    base_gold_lp = F.log_softmax(
                        teacher_logits.float(), dim=-1).gather(
                            -1, gather_target).squeeze(-1)
                    sidecar_gold_lp = F.log_softmax(
                        student_logits.float(), dim=-1).gather(
                            -1, gather_target).squeeze(-1)
                    prior_logit = torch.logit(prior)
                    oracle_temperature = max(
                        1e-3,
                        float(token_fusion_oracle_temperature),
                    )
                    soft_target = torch.sigmoid(
                        prior_logit
                        + (sidecar_gold_lp - base_gold_lp) / oracle_temperature
                    )
                    if self.one_sided_fusion_oracle:
                        # Likelihood evidence may sharpen the prior direction
                        # (base for backbone, sidecar for attachment) but never
                        # reverse it.
                        soft_target = torch.where(
                            high_gate,
                            torch.maximum(soft_target, prior),
                            torch.minimum(soft_target, prior),
                        )
            gate_bce = F.binary_cross_entropy_with_logits(
                fusion_logits.float(), soft_target, reduction="none")
            gate_weights = valid.to(dtype=gate_bce.dtype) * (
                1.0 + 4.0 * high_gate.to(dtype=gate_bce.dtype))
            token_fusion_supervision_loss = self._masked_loss_mean(
                gate_bce, gate_weights, reduction=loss_reduction)
            loss = loss + (
                float(token_fusion_supervision_weight) * token_fusion_supervision_loss
            )
            with torch.no_grad():
                if fusion_alpha is not None and bool((high_gate & valid).any()):
                    fusion_alpha_attachment = fusion_alpha.float()[high_gate & valid].mean()
                backbone_gate = valid & ~high_gate
                if fusion_alpha is not None and bool(backbone_gate.any()):
                    fusion_alpha_backbone = fusion_alpha.float()[backbone_gate].mean()
                if bool((high_gate & valid).any()):
                    fusion_oracle_attachment = soft_target[high_gate & valid].mean()
                if bool(backbone_gate.any()):
                    fusion_oracle_backbone = soft_target[backbone_gate].mean()

        attachment_set_total_loss = None
        attachment_set_components: dict[str, torch.Tensor | None] = {}
        if (
            float(attachment_set_loss_weight) > 0.0
            and structure_labels is not None
            and fwd.get("attachment_set_outputs")
            and isinstance(fwd.get("attachment_points"), torch.Tensor)
            and isinstance(fwd.get("attachment_point_mask"), torch.Tensor)
            and isinstance(fwd.get("attachment_set_complete"), torch.Tensor)
            and isinstance(fwd.get("attachment_count"), torch.Tensor)
        ):
            attachment_set_total_loss, attachment_set_components = attachment_set_loss(
                fwd["attachment_set_outputs"],
                structure_labels=structure_labels,
                points=fwd["attachment_points"],
                point_mask=fwd["attachment_point_mask"],
                set_complete=fwd["attachment_set_complete"],
                counts=fwd["attachment_count"],
                bonded=fwd.get("attachment_bonded"),
                bond_types=fwd.get("attachment_bond_type"),
                anchor_points=fwd.get("attachment_anchor_points"),
                anchor_mask=fwd.get("attachment_anchor_mask"),
                anchor_indices=fwd.get("attachment_anchor_indices"),
                dummy_indices=fwd.get("attachment_dummy_indices"),
                atom_candidate_mask=fwd.get("attachment_atom_candidate_mask"),
                anchor_candidate_mask=fwd.get(
                    "attachment_anchor_candidate_mask"
                ),
                dummy_candidate_mask=fwd.get(
                    "attachment_atom_candidate_mask"
                ),
                point_loss_weight=float(attachment_set_point_loss_weight),
                cardinality_loss_weight=float(
                    attachment_set_cardinality_loss_weight
                ),
                relation_loss_weight=float(
                    attachment_set_relation_loss_weight
                ),
                anchor_loss_weight=float(attachment_set_anchor_loss_weight),
                pointer_loss_weight=float(attachment_set_pointer_loss_weight),
                dummy_pointer_loss_weight=float(
                    attachment_set_dummy_pointer_loss_weight
                ),
                heatmap_loss_weight=float(attachment_set_heatmap_loss_weight),
                heatmap_cost_weight=float(attachment_set_heatmap_cost_weight),
                heatmap_diversity_loss_weight=float(
                    attachment_set_heatmap_diversity_loss_weight
                ),
            )
            loss = loss + float(attachment_set_loss_weight) * attachment_set_total_loss

        # Standard Switch/GShard load balance over all experts; excluding
        # expert0 would reward routing every row to complete.
        with torch.no_grad():
            assignments = weights.argmax(-1)
            f = torch.stack([(assignments == i).float().mean() for i in range(K)])
        # Stop gradient through the discrete frequencies only; P stays
        # differentiable or the term never trains the router.
        P = weights.mean(dim=0)
        lb_loss = K * (f * P).sum()
        if load_balance_weight > 0:
            loss = loss + load_balance_weight * lb_loss

        # ST-MoE z-loss squares the log normalizer, not the raw logits.
        z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        loss = loss + z_loss_weight * z_loss

        # Structure-type CE.
        struct_loss = None
        if structure_labels is not None:
            struct_loss = F.cross_entropy(router_logits, structure_labels.to(router_logits.device))
            loss = loss + structure_weight * struct_loss

        # Router max-margin (multi-class hinge).
        router_margin_loss = None
        if router_margin_weight > 0 and structure_labels is not None:
            labels_dev = structure_labels.to(router_logits.device).clamp(min=0, max=K - 1)
            true_logit = router_logits.gather(1, labels_dev.view(-1, 1)).squeeze(1)
            competitor = router_logits.clone()
            competitor.scatter_(1, labels_dev.view(-1, 1), float("-inf"))
            max_competitor = competitor.max(dim=1).values
            router_margin_loss = torch.clamp(
                router_margin - (true_logit - max_competitor), min=0.0).mean()
            loss = loss + router_margin_weight * router_margin_loss

        # Optional expert-diversity (OMoE): cosine penalty between the LoRA
        # deltas (B_i A_i).
        expert_diversity_loss = None
        if expert_diversity_weight > 0:
            sims = []
            for m in self._decoder.modules():
                if isinstance(m, LoRAMoELinear) and m.num_experts > 1:
                    delta = torch.matmul(m.B, m.A).reshape(m.num_experts, -1)  # (N, out*in)
                    norm = delta.norm(dim=1, keepdim=True).clamp(min=1e-6)
                    unit = delta / norm
                    sim = unit @ unit.t()
                    off = sim.sum() - sim.diag().sum()
                    sims.append(off / (m.num_experts * (m.num_experts - 1)))
            if sims:
                expert_diversity_loss = torch.stack(sims).mean()
                loss = loss + expert_diversity_weight * expert_diversity_loss

        return {
            "loss": loss,
            "mixture_loss": float(mixture_loss.detach()),
            "token_loss": float(token_loss.detach()),
            "mixture_token_ce_loss": (
                None if mixture_token_ce_loss is None else float(mixture_token_ce_loss.detach())),
            "mixture_edge_ce_loss": (
                None if mixture_edge_ce_loss is None else float(mixture_edge_ce_loss.detach())),
            "attachment_token_loss": (
                None if attachment_token_loss is None else float(attachment_token_loss.detach())),
            "terminal_dummy_margin_loss": (
                None
                if terminal_dummy_margin_loss is None
                else float(terminal_dummy_margin_loss.detach())
            ),
            "premature_eos_unlikelihood_loss": (
                None
                if premature_eos_unlikelihood_loss is None
                else float(premature_eos_unlikelihood_loss.detach())
            ),
            "fragment_terminal_action_head_loss": (
                None
                if fragment_terminal_action_head_loss is None
                else float(fragment_terminal_action_head_loss.detach())
            ),
            "variable_identity_loss": (
                None
                if variable_identity_loss is None
                else float(variable_identity_loss.detach())
            ),
            "attachment_cardinality_loss": (
                None if attachment_cardinality_loss is None
                else float(attachment_cardinality_loss.detach())),
            "token_fusion_supervision_loss": (
                None if token_fusion_supervision_loss is None
                else float(token_fusion_supervision_loss.detach())),
            "fusion_alpha_attachment": (
                None if fusion_alpha_attachment is None
                else float(fusion_alpha_attachment.detach())),
            "fusion_alpha_backbone": (
                None if fusion_alpha_backbone is None
                else float(fusion_alpha_backbone.detach())),
            "fusion_oracle_attachment": (
                None if fusion_oracle_attachment is None
                else float(fusion_oracle_attachment.detach())),
            "fusion_oracle_backbone": (
                None if fusion_oracle_backbone is None
                else float(fusion_oracle_backbone.detach())),
            "attachment_set_loss": (
                None if attachment_set_total_loss is None
                else float(attachment_set_total_loss.detach())
            ),
            **{
                name: (
                    None if value is None else float(value.detach())
                )
                for name, value in attachment_set_components.items()
            },
            "edge_loss": None if edge_loss is None else float(edge_loss.detach()),
            "edge_valence_loss": (
                None if edge_valence_loss is None
                else float(edge_valence_loss.detach())
            ),
            "dummy_atom_count": (
                None if dummy_atom_mask is None else float(dummy_atom_mask.float().sum().detach())),
            "sidecar_dense_loss": None, "sidecar_token_loss": None, "sidecar_edge_loss": None,
            "specialist_loss": None, "specialist_token_loss": None, "specialist_edge_loss": None,
            "distill_loss": (None if token_distill_loss is None else float(token_distill_loss.detach())),
            "token_distill_loss": (None if token_distill_loss is None else float(token_distill_loss.detach())),
            "edge_distill_loss": (
                None if edge_distill_loss is None
                else float(edge_distill_loss.detach())
            ),
            "anchor_loss": None, "expert0_preserve_loss": (
                None if expert0_preserve_loss is None else float(expert0_preserve_loss.detach())),
            "load_balance_loss": float(lb_loss.detach()),
            "z_loss": float(z_loss.detach()),
            "structure_loss": None if struct_loss is None else float(struct_loss.detach()),
            "routed_mixture_loss": None,
            "router_margin_loss": (
                None if router_margin_loss is None else float(router_margin_loss.detach())),
            "expert_diversity_loss": (
                None if expert_diversity_loss is None else float(expert_diversity_loss.detach())),
        }

    # -------------------------------------------------- routing (inference)
    def _compute_weights(self, features: torch.Tensor):
        """Per-image routing tensors for inference. A sidecar activates only
        when it is the argmax AND clears its accept threshold; everything else
        is ``forced_default`` (zero gate => frozen base).
        """
        gate_logits = self.router(features)                 # (B, K)
        probs = gate_logits.softmax(dim=-1)                 # (B, K)
        argmax_expert = probs.argmax(dim=-1)                # (B,)
        top_conf = probs.max(dim=-1).values                 # (B,)
        thresholds = torch.tensor(
            self.sidecar_confidence_thresholds, dtype=probs.dtype, device=probs.device)
        required_threshold = thresholds.gather(0, argmax_expert)
        if self.attachment_set_decode_mode == "direct_sidecar" or getattr(
            self, "per_bucket_decode_dispatch", False
        ):
            # Direct experts own the final graph: confidence is metadata and
            # never replaces a routed sidecar graph.
            use_sidecar = argmax_expert != 0
        else:
            use_sidecar = (argmax_expert != 0) & (top_conf >= required_threshold)
        forced_default = ~use_sidecar
        # weights are for the gate / metadata; forced_default rows get gate=0 in decode.
        weights = probs.clone()
        if self.routing_strategy == "sparse_top1":
            onehot = torch.zeros_like(weights)
            onehot.scatter_(1, argmax_expert.view(-1, 1), 1.0)
            weights = torch.where(use_sidecar.unsqueeze(1), onehot, weights)
        return weights, argmax_expert, top_conf, forced_default, required_threshold, probs

    def _compute_expected_weights(self, features: torch.Tensor, expected_structure_types) -> tuple:
        """Routing with an explicit upstream structure contract: known
        complete/markush/fragment crops are forced to their expert instead of
        silently falling back to complete; unknown entries use router confidence.
        """
        normal = self._compute_weights(features)
        if expected_structure_types is None:
            return normal
        weights, argmax_expert, top_conf, forced_default, required_threshold, probs = normal
        expected = list(expected_structure_types)
        if len(expected) != features.size(0):
            raise ValueError(
                f"expected_structure_types length {len(expected)} != batch size {features.size(0)}"
            )
        forced_weights = weights.clone()
        forced_argmax = argmax_expert.clone()
        forced_top = top_conf.clone()
        forced_default = forced_default.clone()
        forced_required = required_threshold.clone()
        name_to_idx = {name: idx for idx, name in enumerate(self.expert_names)}
        aliases = {"complete_compound": "complete", "ordinary": "complete"}
        for i, raw in enumerate(expected):
            name = aliases.get(str(raw or "").strip().lower(), str(raw or "").strip().lower())
            if name not in name_to_idx:
                continue
            idx = int(name_to_idx[name])
            forced_weights[i].zero_()
            forced_weights[i, idx] = 1.0
            forced_argmax[i] = idx
            forced_top[i] = 1.0
            forced_required[i] = 0.0
            forced_default[i] = idx == 0
        return forced_weights, forced_argmax, forced_top, forced_default, forced_required, probs

    def _decode_constraints_for_expected(
        self,
        expected_structure_type,
        base_constraints: dict,
        device: torch.device,
    ) -> dict:
        """Per-sample decode constraints for explicit upstream structure types.
        Fragment crops get a star budget (and optional bias/atom cap); markush
        keeps the unconstrained decode.
        """
        constraints = dict(base_constraints or {})
        name = str(expected_structure_type or "").strip().lower()
        if name != "fragment":
            return constraints
        atom_constraints = dict(constraints.get(ATOM_FORMAT) or {})
        max_atoms = max(0, int(getattr(self, "expected_fragment_max_atoms", 0) or 0))
        star_budget = max(
            0, int(getattr(self, "expected_fragment_star_budget", 0) or 0)
        )
        star_bias = float(
            getattr(self, "expected_fragment_star_logit_bias", 0.0) or 0.0
        )
        if max_atoms > 0:
            atom_constraints.setdefault(
                "max_decode_atoms",
                max_atoms,
            )
        star_id = getattr(self.tokenizer[ATOM_FORMAT], "stoi", {}).get("*")
        if star_id is None or star_budget <= 0:
            if atom_constraints:
                constraints[ATOM_FORMAT] = atom_constraints
            return constraints
        atom_constraints.setdefault("star_token_id", int(star_id))
        atom_constraints.setdefault(
            "star_budgets",
            torch.tensor([star_budget], device=device, dtype=torch.long),
        )
        if star_bias != 0.0:
            atom_constraints.setdefault(
                "star_logit_bias",
                torch.tensor([star_bias], device=device, dtype=torch.float32),
            )
        constraints[ATOM_FORMAT] = atom_constraints
        return constraints

    # -------------------------------------------------- decode
    def _apply_mixture_floor(self, w_complete: float, w_sidecar: float) -> tuple[float, float]:
        total = float(w_complete) + float(w_sidecar)
        if total <= 0:
            w_complete, w_sidecar = 1.0, 0.0
        else:
            w_complete, w_sidecar = float(w_complete) / total, float(w_sidecar) / total
        # Optional floor on the complete expert weight (always-on shared
        # expert); 0.0 disables it.
        floor = float(getattr(self, "mixture_complete_floor", 0.0))
        if floor > 0.0 and w_complete < floor:
            w_complete = floor
            w_sidecar = max(0.0, 1.0 - floor)
        return w_complete, w_sidecar

    def _mixture_weights_2(self, weights_b: torch.Tensor) -> torch.Tensor:
        """Collapse the K=3 router distribution into the 2-decoder mixture
        weights ``[w_complete, w_specialist]`` (markush + fragment sum into
        the single specialist). Normalized by construction.
        """
        w = weights_b.detach().float()
        n = w.numel()
        w_complete = float(w[0]) if n > 0 else 1.0
        # markush + fragment both route to the single specialist (expert1).
        if n > 2:
            w_specialist = float(w[1]) + float(w[2])
        elif n > 1:
            w_specialist = float(w[1])
        else:
            w_specialist = 0.0
        w_complete, w_specialist = self._apply_mixture_floor(w_complete, w_specialist)
        return torch.tensor([w_complete, w_specialist], device=weights_b.device)

    def _mixture_weights_for_sidecar(
        self,
        weights_b: torch.Tensor,
        sidecar_expert_idx: int,
    ) -> torch.Tensor:
        """Per-sidecar full-mixture weights for expert0 + the selected sidecar
        (class 1 uses expert1 only, class 2 uses expert2 only).
        """
        w = weights_b.detach().float()
        idx = int(sidecar_expert_idx)
        w_complete = float(w[0]) if w.numel() > 0 else 1.0
        w_sidecar = float(w[idx]) if 0 <= idx < w.numel() else 0.0
        w_complete, w_sidecar = self._apply_mixture_floor(w_complete, w_sidecar)
        return torch.tensor([w_complete, w_sidecar], device=weights_b.device)

    def _weights_to_mixture_2_batch(
        self,
        weights: torch.Tensor,
        structure_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Batched counterpart of :meth:`_mixture_weights_2` with the
        complete-floor applied, **detached** so mixture-CE trains the experts.
        ``weights``: (B, K); returns (B, 2)."""
        w = weights.detach().float()
        if structure_labels is not None:
            labels = structure_labels.to(w.device).long().clamp(min=0, max=w.size(1) - 1)
            w_complete = labels.eq(0).to(dtype=w.dtype)
            w_specialist = labels.gt(0).to(dtype=w.dtype)
        elif self.full_mixture_sidecar_mode == "per_sidecar":
            sidecar = w[:, 1:]
            idx = sidecar.argmax(dim=1) + 1 if sidecar.numel() else torch.zeros(
                w.size(0), dtype=torch.long, device=w.device)
            w_complete = w[:, 0]
            w_specialist = w.gather(1, idx.view(-1, 1)).squeeze(1)
        else:
            w_complete = w[:, 0]
            w_specialist = w[:, 1:].sum(dim=1) if w.size(1) > 1 else torch.zeros_like(w_complete)
        total = w_complete + w_specialist
        safe = total.clamp(min=1e-8)
        w_complete = torch.where(total > 0, w_complete / safe, torch.ones_like(w_complete))
        w_specialist = torch.where(total > 0, w_specialist / safe, torch.zeros_like(w_specialist))
        floor = float(getattr(self, "mixture_complete_floor", 0.0))
        if floor > 0.0:
            w_complete = torch.clamp(w_complete, min=floor)
            w_specialist = 1.0 - w_complete
        return torch.stack([w_complete, w_specialist], dim=1)

    @torch.inference_mode()
    def _decode_mixture_single(self, enc_b, w2, sample_constraints, beam_size=1, n_best=1,
                               sidecar_expert_idx: int = 1):
        """Per-sample (B=1) probability-space mixture of expert0 and the
        sidecar, decoded in lockstep on a SHARED token sequence: both experts
        advance on the mixed argmax, hard masks are applied AFTER the mixture,
        and ``w=[1,0]`` reproduces expert0 greedy exactly (byte-identical).
        """
        if int(beam_size or 1) != 1:
            import warnings
            warnings.warn("full_mixture decode is greedy-only; clamping beam_size to 1.")
        device = enc_b.device
        ar0 = self.experts[0].decoder[ATOM_FORMAT]
        sidecar_expert_idx = int(sidecar_expert_idx)
        if sidecar_expert_idx <= 0 or sidecar_expert_idx >= len(self.experts):
            sidecar_expert_idx = 1
        ar1 = self.experts[sidecar_expert_idx].decoder[ATOM_FORMAT]
        tok = self.tokenizer[ATOM_FORMAT]
        mb0 = ar0.enc_transform(enc_b)
        mb1 = ar1.enc_transform(enc_b)

        ac = dict((sample_constraints or {}).get(ATOM_FORMAT) or {})
        star_token_id = ac.get("star_token_id")
        star_budgets = ac.get("star_budgets")
        star_logit_bias = ac.get("star_logit_bias")
        star_budget = (int(star_budgets[0].item())
                       if (star_budgets is not None and torch.numel(star_budgets)) else None)
        star_bias = (float(star_logit_bias[0].item())
                     if (star_logit_bias is not None and torch.numel(star_logit_bias)) else 0.0)

        max_len = int(FORMAT_INFO[ATOM_FORMAT]["max_len"])
        min_length = 1
        logit_bias = None  # MoE inference passes no per-token logit_bias
        # Fragment-only guard; markush graphs may legitimately exceed 30
        # atoms/dummies and must not inherit it.
        max_fragment_atoms = ac.get("max_decode_atoms")
        max_fragment_atoms = (
            max(1, int(max_fragment_atoms))
            if max_fragment_atoms is not None and int(max_fragment_atoms) > 0
            else None
        )

        w0 = float(w2[0])
        w1 = float(w2[1])
        fusion_gate = self._token_fusion_gate(sidecar_expert_idx)
        adaptive_fusion = fusion_gate is not None
        static_log_w0 = math.log(w0) if w0 > 0 else float("-inf")
        static_log_w1 = math.log(w1) if w1 > 0 else float("-inf")

        tgt = torch.tensor([SOS_ID], device=device, dtype=torch.long)
        seq = [SOS_ID]
        pos_prob = [1.0]                 # mixed prob per sequence position (SOS given)
        dec_out_0_steps, dec_out_1_steps = [], []
        fusion_alpha_steps = []
        generated_stars = 0
        generated_atoms = 0  # over-generation guard
        eos_found = False

        for step in range(max_len):
            z0, d0 = ar0._step_logits(tgt, mb0, step, logit_bias)
            z1, d1 = ar1._step_logits(tgt, mb1, step, logit_bias)
            # star_logit_bias applied to BOTH experts' logits BEFORE log_softmax
            # (mirrors components.py:350-358).
            if (star_token_id is not None and star_bias != 0.0
                    and 0 <= int(star_token_id) < z0.size(-1)):
                z0[:, int(star_token_id)] += star_bias
                z1[:, int(star_token_id)] += star_bias
            lp0 = F.log_softmax(z0, dim=-1)[0]      # (vocab,)
            lp1 = F.log_softmax(z1, dim=-1)[0]

            if adaptive_fusion:
                soft_alpha = float(torch.sigmoid(
                    fusion_gate(
                        d0,
                        d1,
                        lp0.view(1, 1, -1),
                        lp1.view(1, 1, -1),
                    )
                ).reshape(-1)[0].item())
                soft_alpha = min(1.0 - 1e-5, max(1e-5, soft_alpha))
                if self.token_fusion_dispatch == "hard":
                    alpha = float(
                        soft_alpha >= self.token_fusion_hard_threshold
                    )
                    log_w0 = 0.0 if alpha == 0.0 else float("-inf")
                    log_w1 = 0.0 if alpha == 1.0 else float("-inf")
                else:
                    alpha = soft_alpha
                    log_w0 = math.log(1.0 - alpha)
                    log_w1 = math.log(alpha)
            else:
                alpha = w1
                log_w0 = static_log_w0
                log_w1 = static_log_w1
            fusion_alpha_steps.append(alpha)
            stacked = torch.stack([lp0 + log_w0, lp1 + log_w1], dim=0)   # (2, vocab)
            mixed_lp = torch.logsumexp(stacked, dim=0)                    # (vocab,)
            # HARD constraints applied AFTER the mixture.
            if step < min_length:
                mixed_lp[EOS_ID] = float("-inf")
            if getattr(tok, "output_constraint", False):
                mask = torch.tensor(tok.get_output_mask(int(tgt.item())), device=device)
                mixed_lp = mixed_lp.masked_fill(mask, float("-inf"))
            if (star_token_id is not None and star_budget is not None
                    and generated_stars >= star_budget
                    and 0 <= int(star_token_id) < mixed_lp.numel()):
                mixed_lp[int(star_token_id)] = float("-inf")
            # Fragment over-generation guard: if we've decoded enough atoms,
            # force EOS to prevent tiny-fragment -> big-molecule and spurious splits.
            if (max_fragment_atoms is not None and generated_atoms >= max_fragment_atoms
                    and 0 <= EOS_ID < mixed_lp.numel()):
                mixed_lp[EOS_ID] = float("inf")  # force EOS
            next_id = int(torch.argmax(mixed_lp).item())
            pos_prob.append(float(mixed_lp[next_id].exp().item()))

            seq.append(next_id)
            dec_out_0_steps.append(d0)
            dec_out_1_steps.append(d1)
            if next_id == star_token_id:
                generated_stars += 1
            # Every decoded atom terminates with exactly one y-coordinate token,
            # so this counts atoms without heuristics for multi-character symbols.
            if max_fragment_atoms is not None and tok.is_y(next_id):
                generated_atoms += 1
            if next_id == EOS_ID:
                eos_found = True
                break
            tgt = torch.tensor([next_id], device=device, dtype=torch.long)

        # --- post-processing (mirror Decoder.decode, components.py:651-759) ---
        # Predictions do NOT carry the leading SOS; strip it before
        # sequence_to_smiles (else atom indices shift +1). Hidden stays
        # unstripped (SOS-step matches base's dec_out[0]).
        cc = tok.sequence_to_smiles(seq[1:])
        symbols = cc.get("symbols") or []
        coords = cc.get("coords") or []
        indices = cc.get("indices") or []
        atom_count = len(symbols)
        pred = {
            ATOM_FORMAT: {
                "smiles": cc.get("smiles", ""),
                "symbols": symbols,
                "coords": coords,
                "indices": indices,
            },
            "decode_atom_count": atom_count,
        }
        if not eos_found:
            pred["decode_quality_issue"] = "molnextr_decode_missing_eos"
        elif atom_count > MAX_DECODE_ATOMS:
            pred["decode_quality_issue"] = f"molnextr_decode_atom_limit_exceeded:{atom_count}"
        else:
            bad = [s for s in symbols if invalid_decode_symbol(s)]
            if bad:
                pred["decode_quality_issue"] = "molnextr_decode_invalid_symbols:" + ",".join(bad[:3])

        quality_issue = pred.get("decode_quality_issue")
        if self.compute_confidence and not quality_issue:
            # atom_scores: geometric mean of the mixed probs over each atom's
            # symbol span; pos_prob[0] is the SOS slot, so seq position p maps
            # to pos_prob[p+1].
            pred[ATOM_FORMAT]["atom_scores"] = atom_symbol_scores_from_token_probs(
                symbols,
                indices,
                pos_prob[1:],
            )
        if adaptive_fusion:
            pred["token_fusion_mode"] = "adaptive"
            pred["token_fusion_dispatch"] = self.token_fusion_dispatch
            pred["token_fusion_applied"] = True
            pred["token_fusion_sidecar_mean"] = float(
                np.mean(fusion_alpha_steps) if fusion_alpha_steps else 0.0)

        # --- edges: probability-space mixture of both experts' edge heads ---
        if quality_issue or atom_count == 0 or not dec_out_0_steps:
            pred["edges"] = [[0] * atom_count for _ in range(atom_count)]
        else:
            hidden0 = torch.cat(dec_out_0_steps, dim=1)   # (1, T, dim)
            hidden1 = torch.cat(dec_out_1_steps, dim=1)
            # Atom indices are absolute positions; the last can land on the EOS
            # slot no per-step hidden covers. Pad with the last hidden so the
            # GraphPredictor gather stays in bounds.
            need = int(max(indices)) + 1

            def _pad_to(h):
                T = h.size(1)
                return h if T >= need else torch.cat(
                    [h, h[:, -1:, :].expand(-1, need - T, -1)], dim=1)

            hidden0 = _pad_to(hidden0)
            hidden1 = _pad_to(hidden1)
            idx_t = torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(0)  # (1, k)
            e0 = self.experts[0].decoder["edges"](hidden0, idx_t)["edges"]   # (1,7,k,k)
            e1 = self.experts[sidecar_expert_idx].decoder["edges"](hidden1, idx_t)["edges"]
            prob0 = F.softmax(e0.squeeze(0).permute(1, 2, 0), dim=2).detach().cpu().numpy()
            prob1 = F.softmax(e1.squeeze(0).permute(1, 2, 0), dim=2).detach().cpu().numpy()
            if adaptive_fusion:
                atom_alpha = []
                for index in indices:
                    alpha_index = min(
                        max(0, int(index) - 1),
                        max(0, len(fusion_alpha_steps) - 1),
                    )
                    atom_alpha.append(
                        float(fusion_alpha_steps[alpha_index])
                        if fusion_alpha_steps else float(w1)
                    )
                pair_alpha = np.zeros((atom_count, atom_count), dtype=np.float32)
                for i in range(atom_count):
                    for j in range(atom_count):
                        value = max(atom_alpha[i], atom_alpha[j])
                        pair_alpha[i, j] = min(1.0, max(0.0, value))
                mixed_prob = (
                    (1.0 - pair_alpha[..., None]) * prob0
                    + pair_alpha[..., None] * prob1
                )
            else:
                mixed_prob = w0 * prob0 + w1 * prob1
            edge_pred, edge_score = get_edge_prediction(mixed_prob)
            bond_count = len(edge_score)
            bond_limit = decode_bond_limit(atom_count)
            if bond_count > bond_limit:
                pred["decode_quality_issue"] = f"molnextr_decode_bond_limit_exceeded:{bond_count}>{bond_limit}"
                pred["edges"] = [[0] * atom_count for _ in range(atom_count)]
            else:
                pred["edges"] = edge_pred
                if self.compute_confidence:
                    pred["edge_scores"] = edge_score
        return pred

    def _attach_direct_attachment_diagnostics(
        self,
        pred: dict,
        outputs: dict[str, torch.Tensor],
        *,
        expected_type: str,
        anchor_pointer_logits: torch.Tensor | None = None,
        dummy_pointer_logits: torch.Tensor | None = None,
    ) -> dict:
        """Validate a sidecar-owned graph without changing it."""
        cc = pred.get(ATOM_FORMAT)
        if not isinstance(cc, dict):
            return pred
        symbols = list(cc.get("symbols") or [])
        coords = list(cc.get("coords") or [])
        edges = pred.get("edges")
        if not isinstance(edges, list):
            edges = []
        dummy_indices = [
            index
            for index, symbol in enumerate(symbols)
            if "*" in str(symbol)
            or is_markush_label(normalize_label(str(symbol)))
        ]
        target_cardinality = int(outputs["cardinality_logits"][0].argmax().item())
        proposals = select_attachment_queries(
            outputs,
            batch_index=0,
            expected_type=expected_type,
            min_confidence=self.attachment_set_min_confidence,
            max_count=self.attachment_set_max_count,
        )
        # Diagnostic only: surface detector attachment points for telemetry;
        # the residual path is where detector points graft dummies.
        detector_proposals = self._detector_prior_proposals(
            0,
            expected_type=expected_type,
            min_confidence=self.attachment_set_min_confidence,
        )
        if detector_proposals:
            proposals = proposals + detector_proposals
        diagnostics = []
        mapped_dummies: set[int] = set()
        pointer_consistent = True
        relation_consistent = True
        for proposal in proposals:
            query_index = int(proposal["query_index"])
            dummy_index = -1
            dummy_confidence = 0.0
            if isinstance(dummy_pointer_logits, torch.Tensor) and dummy_pointer_logits.numel():
                probabilities = torch.softmax(
                    dummy_pointer_logits[query_index].float(), dim=-1
                )
                dummy_index = int(probabilities.argmax().item())
                dummy_confidence = float(probabilities[dummy_index].item())
            anchor_index = -1
            anchor_confidence = 0.0
            if isinstance(anchor_pointer_logits, torch.Tensor) and anchor_pointer_logits.numel():
                probabilities = torch.softmax(
                    anchor_pointer_logits[query_index].float(), dim=-1
                )
                anchor_index = int(probabilities.argmax().item())
                anchor_confidence = float(probabilities[anchor_index].item())

            dummy_matches_graph = dummy_index in dummy_indices
            neighbors = []
            if 0 <= dummy_index < len(edges) and isinstance(edges[dummy_index], list):
                neighbors = [
                    index
                    for index, bond_type in enumerate(edges[dummy_index])
                    if index != dummy_index and int(bond_type) > 0
                ]
            relation_matches_graph = bool(
                dummy_matches_graph
                and len(neighbors) == 1
                and anchor_index == neighbors[0]
            )
            if dummy_matches_graph:
                mapped_dummies.add(dummy_index)
            pointer_consistent &= dummy_matches_graph
            relation_consistent &= relation_matches_graph
            point_distance = None
            if 0 <= dummy_index < len(coords):
                try:
                    dx = float(coords[dummy_index][0]) - float(proposal["x"])
                    dy = float(coords[dummy_index][1]) - float(proposal["y"])
                    point_distance = math.sqrt(dx * dx + dy * dy)
                except (IndexError, TypeError, ValueError):
                    point_distance = None
            diagnostics.append(
                {
                    **proposal,
                    "dummy_index": dummy_index,
                    "dummy_pointer_confidence": dummy_confidence,
                    "anchor_index": anchor_index,
                    "anchor_pointer_confidence": anchor_confidence,
                    "dummy_is_graph_attachment": dummy_matches_graph,
                    "graph_neighbors": neighbors,
                    "relation_consistent": relation_matches_graph,
                    "point_distance": point_distance,
                }
            )

        cardinality_consistent = target_cardinality == len(dummy_indices)
        coverage_consistent = len(mapped_dummies) == len(dummy_indices)
        graph_consistent = bool(
            cardinality_consistent
            and len(proposals) == len(dummy_indices)
            and coverage_consistent
            and pointer_consistent
            and relation_consistent
        )
        # Cardinality edit: when decoded dummies exceed the cardinality head's
        # count, prune the excess (pointer confidence first, detector distance
        # tiebreak); keep a floor of 1 on fragment/markush rows.
        dummy_pruned: list[int] = []
        if (
            len(dummy_indices) > target_cardinality >= 0
            and expected_type in {"fragment", "markush"}
            and getattr(self, "attachment_set_dummy_prune_enabled", True)
        ):
            keep_count = max(1, min(target_cardinality, len(dummy_indices)))
            pointer_probs: dict[int, float] = {}
            if isinstance(dummy_pointer_logits, torch.Tensor) and dummy_pointer_logits.numel():
                probs = torch.softmax(dummy_pointer_logits.float(), dim=-1)
                for q in range(probs.size(0)):
                    for idx in dummy_indices:
                        if 0 <= idx < probs.size(1):
                            v = float(probs[q, idx].item())
                            pointer_probs[idx] = max(pointer_probs.get(idx, 0.0), v)
            det_pts: list[tuple[float, float]] = []
            for det in self._detector_prior_proposals(
                0, expected_type=expected_type, min_confidence=0.0
            ):
                try:
                    det_pts.append((float(det["x"]), float(det["y"])))
                except (KeyError, TypeError, ValueError):
                    continue

            def _keep_rank(idx: int) -> tuple[float, float, int]:
                conf = pointer_probs.get(idx, 0.0)
                dist = float("inf")
                if det_pts and 0 <= idx < len(coords):
                    try:
                        dist = min(
                            math.hypot(float(coords[idx][0]) - dx, float(coords[idx][1]) - dy)
                            for dx, dy in det_pts
                        )
                    except (IndexError, TypeError, ValueError):
                        dist = float("inf")
                return (0.0 - conf, dist, idx)

            ranked = sorted(dummy_indices, key=_keep_rank)
            keep = set(ranked[:keep_count])
            dummy_pruned = [idx for idx in dummy_indices if idx not in keep]
            if dummy_pruned:
                prune_set = set(dummy_pruned)
                cc = pred.get(ATOM_FORMAT)
                if isinstance(cc, dict) and isinstance(cc.get("symbols"), list):
                    symbols = list(cc.get("symbols") or [])
                    coords_ = list(cc.get("coords") or [])
                    scores = cc.get("atom_scores")
                    scores_list = list(scores) if isinstance(scores, list) else None
                    cc["symbols"] = [s for i, s in enumerate(symbols) if i not in prune_set]
                    cc["coords"] = [c for i, c in enumerate(coords_) if i not in prune_set]
                    if scores_list is not None:
                        cc["atom_scores"] = [s for i, s in enumerate(scores_list) if i not in prune_set]
                    edges = pred.get("edges")
                    if isinstance(edges, list):
                        pred["edges"] = [
                            [e for j, e in enumerate(row) if j not in prune_set]
                            for i, row in enumerate(edges)
                            if i not in prune_set
                        ]
                    graph_consistent = False
        pred["attachment_set_dummy_prune"] = {
            "pruned": dummy_pruned,
            "decoded_dummy_count": len(dummy_indices),
            "target_cardinality": target_cardinality,
            "expected_type": expected_type,
        }
        pred["attachment_set_predictions"] = diagnostics
        pred["attachment_set_selected_count"] = len(proposals)
        pred["attachment_set_cardinality"] = target_cardinality
        pred["attachment_set_decode_mode"] = "direct_sidecar"
        pred["attachment_graph_consistency"] = {
            "consistent": graph_consistent,
            "decoded_dummy_count": len(dummy_indices),
            "set_cardinality": target_cardinality,
            "selected_query_count": len(proposals),
            "mapped_dummy_count": len(mapped_dummies),
            "cardinality_consistent": cardinality_consistent,
            "query_coverage_consistent": coverage_consistent,
            "pointer_consistent": bool(pointer_consistent),
            "relation_consistent": bool(relation_consistent),
            "graph_modified": bool(dummy_pruned),
            "dummy_prune_count": len(dummy_pruned),
        }
        return pred

    def _detector_prior_proposals(
        self,
        batch_index: int,
        *,
        expected_type: str,
        min_confidence: float,
    ) -> list[dict[str, Any]]:
        """Build full-schema attachment proposals from the detector priors
        (Mask R-CNN attachment marks) for the sample at ``batch_index``.
        Returned in the exact ``select_attachment_queries`` schema; empty
        list when fusion is disabled or no priors are available.
        """
        # Read the current sample's priors; batch_index is always 0 here
        # because the edit runs on sliced single-sample outputs.
        sample_priors = getattr(self, "_current_sample_priors", None)
        if not sample_priors:
            return []
        proposals: list[dict[str, Any]] = []
        for det in sample_priors:
            if not isinstance(det, dict):
                continue
            try:
                cx = float(det.get("cx"))
                cy = float(det.get("cy"))
                confidence = float(det.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            # Same confidence gate the learned queries face, so a weak detection
            # is dropped here rather than surviving to graft a spurious dummy.
            if confidence < float(min_confidence):
                continue
            proposals.append(
                {
                    # Coordinate fields consumed by the unique_proposals loop and
                    # the continuous-anchor resolution downstream.
                    "x": cx,
                    "y": cy,
                    "anchor_x": cx,
                    "anchor_y": cy,
                    "confidence": confidence,
                    # A detected attachment mark denotes a bonded single-bond
                    # attachment point by definition.
                    "bonded": True,
                    "bonded_confidence": confidence,
                    "bond_type": 1,  # Chem.BondType.SINGLE
                    "bond_type_confidence": confidence,
                    # Marker so downstream telemetry/eval can attribute the edit.
                    "query_index": -1,
                    "source": "detector",
                    "point_heatmap_confidence": None,
                    "point_heatmap_peak_probability": None,
                }
            )
        return proposals

    def _apply_attachment_set_residual(
        self,
        pred: dict,
        outputs: dict[str, torch.Tensor],
        *,
        batch_index: int,
        expected_type: str,
        anchor_pointer_logits: torch.Tensor | None = None,
    ) -> dict:
        """Apply cardinality-bounded, transactional edits to expert0's graph."""
        cc = pred.get(ATOM_FORMAT)
        if not isinstance(cc, dict):
            return pred
        symbols = list(cc.get("symbols") or [])
        coords = [list(point) for point in (cc.get("coords") or [])]
        edges = [list(row) for row in (pred.get("edges") or [])]
        if (
            len(symbols) != len(coords)
            or len(edges) != len(symbols)
            or any(len(row) != len(symbols) for row in edges)
        ):
            return pred
        base_graph = {
            "symbols": list(symbols),
            "coords": [list(point) for point in coords],
            "edges": [list(row) for row in edges],
            "atom_scores": list(cc.get("atom_scores") or []),
            "edge_scores": dict(pred.get("edge_scores") or {}),
            "decode_quality_issue": pred.get("decode_quality_issue"),
        }
        proposals = select_attachment_queries(
            outputs,
            batch_index=batch_index,
            expected_type=expected_type,
            min_confidence=self.attachment_set_min_confidence,
            max_count=self.attachment_set_max_count,
        )
        # Append detector proposals; downstream dedup, cardinality cap, and
        # anchor/bonded validation reject anything unsafe to graft.
        detector_proposals = self._detector_prior_proposals(
            batch_index,
            expected_type=expected_type,
            min_confidence=self.attachment_set_min_confidence,
        )
        if detector_proposals:
            proposals = proposals + detector_proposals
        target_cardinality = int(
            outputs["cardinality_logits"][batch_index].argmax().item()
        )
        if str(expected_type or "").strip().lower() == "fragment":
            target_cardinality = min(1, max(0, target_cardinality))

        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        unique_proposals: list[dict[str, Any]] = []
        duplicate_radius = 0.025
        if (
            getattr(self, "attachment_set_feature_mode", "")
            == "multiscale_pointer_heatmap"
        ):
            duplicate_radius = 0.5 / max(
                1, int(getattr(self, "attachment_set_max_feature_size", 48)) - 1
            )
        for proposal in proposals:
            point = [float(proposal["x"]), float(proposal["y"])]
            anchor_point = [
                float(proposal.get("anchor_x", proposal["x"])),
                float(proposal.get("anchor_y", proposal["y"])),
            ]
            if any(
                math.dist(point, previous["point"]) < duplicate_radius
                for previous in unique_proposals
            ):
                rejected.append(
                    {**proposal, "point": point, "reason": "duplicate_point"}
                )
                continue
            unique_proposals.append(
                {**proposal, "point": point, "anchor_point": anchor_point}
            )

        def marker_candidate(symbol: Any) -> bool:
            text = str(symbol or "").strip()
            label = normalize_label(text)
            return bool(
                text != "*"
                and (
                    "*" in text
                    or is_markush_label(label)
                    or invalid_decode_symbol(text)
                )
            )

        def existing_attachment_marker(symbol: Any) -> bool:
            text = str(symbol or "").strip()
            return bool(
                text != "*"
                and ("*" in text or is_markush_label(normalize_label(text)))
            )

        def bonded_degree(atom_index: int) -> int:
            return int(sum(int(value) > 0 for value in edges[atom_index]))

        # Valence-aware anchor selection: skip saturated backbone atoms.
        def _has_valence_room(atom_index: int, extra_bond_value: int = 1) -> bool:
            return has_valence_room(symbols, edges, atom_index, extra_bond_value)

        def distance_matches(
            proposal_indices: list[int],
            atom_indices: list[int],
            *,
            max_distance: float | None = None,
        ) -> list[tuple[int, int, float]]:
            pairs = sorted(
                (
                    math.dist(unique_proposals[pidx]["point"], coords[aidx]),
                    pidx,
                    aidx,
                )
                for pidx in proposal_indices
                for aidx in atom_indices
            )
            used_proposals: set[int] = set()
            used_atoms: set[int] = set()
            matches = []
            for distance, proposal_index, atom_index in pairs:
                if max_distance is not None and distance > max_distance:
                    continue
                if proposal_index in used_proposals or atom_index in used_atoms:
                    continue
                used_proposals.add(proposal_index)
                used_atoms.add(atom_index)
                matches.append((proposal_index, atom_index, float(distance)))
            return matches

        remaining = list(range(len(unique_proposals)))
        exact_dummies = [
            index for index, symbol in enumerate(symbols) if str(symbol) == "*"
        ]
        for proposal_index, atom_index, distance in distance_matches(
            remaining,
            exact_dummies,
        ):
            proposal = unique_proposals[proposal_index]
            accepted.append(
                {
                    **proposal,
                    "action": "preserve_existing",
                    "atom_index": atom_index,
                    "marker_distance": distance,
                }
            )
            remaining.remove(proposal_index)

        evidence_markers = [
            index
            for index, symbol in enumerate(symbols)
            if existing_attachment_marker(symbol) and bonded_degree(index) > 0
        ]
        evidence_matches = distance_matches(
            remaining,
            evidence_markers,
            max_distance=self.attachment_set_max_anchor_distance,
        )
        for proposal_index, atom_index, distance in evidence_matches:
            proposal = unique_proposals[proposal_index]
            symbols[atom_index] = "*"
            accepted.append(
                {
                    **proposal,
                    "action": "replace",
                    "atom_index": atom_index,
                    "marker_distance": distance,
                }
            )
            remaining.remove(proposal_index)

        edit_slots = max(
            0,
            target_cardinality - len(exact_dummies) - len(evidence_markers),
        )
        salvage_candidates = [
            index
            for index, symbol in enumerate(symbols)
            if marker_candidate(symbol)
            and index not in evidence_markers
            and bonded_degree(index) > 0
        ]
        salvage_matches = distance_matches(
            remaining,
            salvage_candidates,
            max_distance=self.attachment_set_max_anchor_distance,
        )[:edit_slots]
        for proposal_index, atom_index, distance in salvage_matches:
            proposal = unique_proposals[proposal_index]
            symbols[atom_index] = "*"
            accepted.append(
                {
                    **proposal,
                    "action": "replace",
                    "atom_index": atom_index,
                    "marker_distance": distance,
                }
            )
            remaining.remove(proposal_index)
        edit_slots -= len(salvage_matches)

        backbone_candidates = [
            index
            for index, symbol in enumerate(symbols)
            if str(symbol) != "*" and not marker_candidate(symbol)
        ]
        for proposal_index in list(remaining):
            proposal = unique_proposals[proposal_index]
            if edit_slots <= 0:
                rejected.append(
                    {**proposal, "reason": "cardinality_satisfied_by_existing"}
                )
                continue
            if not proposal.get("bonded"):
                rejected.append({**proposal, "reason": "unbonded_attachment"})
                continue

            anchor_index = -1
            anchor_distance = float("inf")
            pointer_confidence = None
            pointer_margin = None
            anchor_resolution = None
            # The anchor must have enough remaining valence for this bond
            # order, else the graph goes hypervalent and rdkit refuses it.
            proposed_bond_type = int(proposal.get("bond_type") or 1)
            if proposed_bond_type <= 0 or proposed_bond_type > 6:
                proposed_bond_type = 1
            eligible_candidates = [
                index
                for index in backbone_candidates
                if _has_valence_room(index, proposed_bond_type)
            ]
            eligible_exhausted = (
                bool(backbone_candidates) and not eligible_candidates
            )
            search_pool = eligible_candidates
            if anchor_pointer_logits is not None and search_pool:
                query_index = int(proposal["query_index"])
                pointer_row = anchor_pointer_logits[query_index].float()
                candidate_tensor = torch.tensor(
                    search_pool,
                    dtype=torch.long,
                    device=pointer_row.device,
                )
                candidate_logits = pointer_row.index_select(0, candidate_tensor)
                pointer_probabilities = candidate_logits.softmax(dim=0)
                top_count = min(2, pointer_probabilities.numel())
                top_values, top_positions = pointer_probabilities.topk(top_count)
                anchor_index = search_pool[int(top_positions[0].item())]
                pointer_confidence = float(top_values[0].item())
                pointer_margin = float(
                    top_values[0].item()
                    - (top_values[1].item() if top_count > 1 else 0.0)
                )
                anchor_distance = math.dist(
                    proposal["anchor_point"], coords[anchor_index]
                )
                # search_pool is non-empty here, so eligible_exhausted is False.
                anchor_resolution = "decoder_atom_pointer"
                if pointer_confidence < self.attachment_set_min_pointer_confidence:
                    rejected.append(
                        {
                            **proposal,
                            "reason": "low_anchor_pointer_confidence",
                            "anchor_index": anchor_index,
                            "anchor_distance": anchor_distance,
                            "anchor_pointer_confidence": pointer_confidence,
                            "anchor_pointer_margin": pointer_margin,
                        }
                    )
                    continue
            elif search_pool:
                anchor_index = min(
                    search_pool,
                    key=lambda index: math.dist(
                        proposal["anchor_point"], coords[index]
                    ),
                )
                anchor_distance = math.dist(
                    proposal["anchor_point"], coords[anchor_index]
                )
                # search_pool is non-empty here, so eligible_exhausted is False.
                anchor_resolution = "continuous_nearest_atom"
                if anchor_distance > self.attachment_set_max_anchor_distance:
                    rejected.append(
                        {
                            **proposal,
                            "reason": "anchor_distance_exceeded",
                            "anchor_index": anchor_index,
                            "anchor_distance": anchor_distance,
                        }
                    )
                    continue
            if anchor_index < 0:
                rejected.append(
                    {
                        **proposal,
                        "reason": (
                            "anchor_valence_exceeded"
                            if eligible_exhausted
                            else "no_backbone_anchor"
                        ),
                    }
                )
                continue

            old_count = len(symbols)
            symbols.append("*")
            coords.append(list(proposal["point"]))
            for row in edges:
                row.append(0)
            edges.append([0] * (old_count + 1))
            bond_type = int(proposal["bond_type"])
            if bond_type <= 0 or bond_type > 6:
                bond_type = 1
            edges[anchor_index][old_count] = bond_type
            edges[old_count][anchor_index] = bond_type
            attachment_confidence = min(
                float(proposal["confidence"]),
                float(proposal.get("bonded_confidence", 1.0)),
                float(proposal.get("bond_type_confidence", 1.0)),
                float(pointer_confidence if pointer_confidence is not None else 1.0),
            )
            accepted.append(
                {
                    **proposal,
                    "action": "append",
                    "atom_index": old_count,
                    "anchor_index": anchor_index,
                    "anchor_distance": anchor_distance,
                    "anchor_resolution": anchor_resolution,
                    "anchor_pointer_confidence": pointer_confidence,
                    "anchor_pointer_margin": pointer_margin,
                    "attachment_confidence": attachment_confidence,
                }
            )
            edit_slots -= 1

        cc["symbols"] = symbols
        cc["coords"] = coords
        if self.compute_confidence:
            atom_scores = list(cc.get("atom_scores") or [])
            if len(atom_scores) < len(symbols):
                atom_scores.extend([0.0] * (len(symbols) - len(atom_scores)))
            for item in accepted:
                atom_index = int(item["atom_index"])
                if item["action"] == "append":
                    atom_scores[atom_index] = float(
                        item.get("attachment_confidence", item["confidence"])
                    )
                elif item["action"] == "replace" and atom_index < len(base_graph["atom_scores"]):
                    atom_scores[atom_index] = max(
                        float(base_graph["atom_scores"][atom_index]),
                        float(item["confidence"]),
                    )
            cc["atom_scores"] = atom_scores
        edge_scores = pred.get("edge_scores")
        if isinstance(edge_scores, dict):
            for item in accepted:
                if item["action"] != "append":
                    continue
                edge_scores[(int(item["anchor_index"]), int(item["atom_index"]))] = min(
                    float(item.get("bonded_confidence", 1.0)),
                    float(item.get("bond_type_confidence", 1.0)),
                    float(item.get("anchor_pointer_confidence") or 1.0),
                )
        pred["edges"] = edges
        pred["decode_atom_count"] = len(symbols)
        pred["_attachment_set_base_graph"] = base_graph
        pred["attachment_set_predictions"] = accepted
        pred["attachment_set_rejections"] = rejected
        pred["attachment_set_selected_count"] = len(proposals)
        pred["attachment_set_cardinality"] = target_cardinality
        pred["attachment_set_decode_mode"] = self.attachment_set_decode_mode
        pred["attachment_set_transaction"] = {
            "status": "committed" if any(
                item["action"] in {"replace", "append"} for item in accepted
            ) else "no_op",
            "base_atom_count": len(base_graph["symbols"]),
            "result_atom_count": len(symbols),
            "target_cardinality": target_cardinality,
            "existing_dummy_count": len(exact_dummies),
            "existing_marker_count": len(evidence_markers),
        }
        issue = str(pred.get("decode_quality_issue") or "")
        if issue.startswith("molnextr_decode_invalid_symbols:"):
            remaining = [symbol for symbol in symbols if invalid_decode_symbol(str(symbol))]
            if not remaining and any(any(value for value in row) for row in edges):
                pred.pop("decode_quality_issue", None)
        return pred

    def _sidecar_decode_is_broken(self, pred: dict) -> bool:
        """Fallback predicate: True when a sidecar decode is broken/empty or
        RDKit cannot parse its SMILES (catches assembly-level breaks too).
        """
        if pred.get("decode_quality_issue"):
            return True
        cc = pred.get(ATOM_FORMAT, {}) or {}
        smiles = str(cc.get("smiles", "") or "").strip()
        if not smiles:
            return True
        if int(pred.get("decode_atom_count", 0) or 0) == 0:
            return True
        try:
            from rdkit import Chem
            if Chem.MolFromSmiles(smiles) is None:
                return True
        except Exception:
            pass
        return False

    @torch.inference_mode()
    def decode(self, features, hiddens=None, refs=None, beam_size=1, n_best=1,
               logit_biases=None, decode_constraints=None, expected_structure_types=None):
        beam_size = max(1, int(beam_size or 1))
        n_best = max(1, int(n_best or 1))
        decode_constraints = decode_constraints or {}
        (weights, argmax_expert, top_conf, forced_default,
         required_threshold, _probs) = self._compute_expected_weights(features, expected_structure_types)
        expected_list = list(expected_structure_types or [None] * features.size(0))
        attachment_set_outputs = [
            (
                head(features, hiddens)
                if self.attachment_set_feature_mode in {
                    "multiscale",
                    "multiscale_pointer",
                    "multiscale_pointer_heatmap",
                }
                else head(features)
            )
            for head in self.attachment_set_heads
        ]
        predictions: list[dict] = []
        for b in range(features.size(0)):
            self._clear_caches()
            enc_b = features[b:b + 1]                       # (1, L, C)
            fc = bool(forced_default[b].item())
            routed_idx = int(argmax_expert[b].item())
            # Expose this sample's detector priors; the edit below reads them
            # with batch_index=0 on the sliced single-sample outputs.
            _priors_batch = getattr(self, "_attachment_priors_batch", None)
            self._current_sample_priors = (
                _priors_batch[b] if _priors_batch and b < len(_priors_batch) else None
            )
            # Interim safety: route fragments to the frozen base while the
            # fragment sidecar regresses. Toggle via config.
            if (
                getattr(self, "route_fragment_to_base", False)
                and not fc
                and routed_idx == 2
            ):
                fc = True
                routed_idx = 0
            # Per-bucket dispatch: markush (1) -> direct_sidecar; fragment (2)
            # -> residual_base; everything else falls to the global mode.
            if (
                getattr(self, "per_bucket_decode_dispatch", False)
                and self.expert_kind == "full_mixture"
                and not fc
            ):
                effective_decode_mode = (
                    "direct_sidecar" if routed_idx == 1 else "residual_base"
                )
            else:
                effective_decode_mode = self.attachment_set_decode_mode
            sample_constraints = self._decode_constraints_for_expected(
                expected_list[b],
                decode_constraints,
                features.device,
            )
            if self.expert_kind == "lora":
                # Forced-default rows get the zero gate (frozen base); sidecar
                # rows get the routed adapter gate.
                gate = (torch.zeros(self.num_adapter_experts, device=features.device) if fc
                        else self._gate_from_weights(weights[b:b + 1]).squeeze(0))
                self._set_gate_all(gate.unsqueeze(0))           # (1, K-1)

            if fc:
                # Byte-identical complete path: expert0's exact standalone
                # decode (lora: gate=0; full_mixture: experts[0]).
                pred = self.experts[0].decode(
                    enc_b, hiddens=None, beam_size=beam_size, n_best=n_best,
                    decode_constraints=sample_constraints,
                )[0]
            elif (
                self.expert_kind == "full_mixture"
                and effective_decode_mode == "direct_sidecar"
            ):
                if routed_idx <= 0 or routed_idx >= len(self.experts):
                    raise RuntimeError(
                        f"direct sidecar route {routed_idx} has no decoder expert"
                    )
                pointer_enabled = bool(
                    self.attachment_set_enabled
                    and routed_idx <= len(attachment_set_outputs)
                    and isinstance(
                        self.attachment_set_heads[routed_idx - 1],
                        MultiScaleAttachmentPointerHead,
                    )
                )
                terminal_action_head = (
                    self.fragment_terminal_action_head
                    if self.fragment_terminal_action_head_enabled
                    and routed_idx == 2
                    else None
                )
                terminal_star_token_id = (
                    getattr(self.tokenizer[ATOM_FORMAT], "stoi", {}).get("*")
                    if terminal_action_head is not None
                    else None
                )
                # Sidecars emit the final graph directly. No Expert0 logits,
                # residual atom edits, star bias, or rollback path participates.
                pred = self.experts[routed_idx].decode(
                    enc_b,
                    hiddens=None,
                    beam_size=beam_size,
                    n_best=n_best,
                    decode_constraints=dict(decode_constraints or {}),
                    return_atom_hidden=pointer_enabled,
                    mask_y_coordinate_context=(
                        self.sidecar_coordinate_context == "mask_y"
                    ),
                    terminal_action_head=terminal_action_head,
                    terminal_star_token_id=terminal_star_token_id,
                    structured_terminal_dummy_edge=(
                        self.fragment_structured_terminal_edge_enabled
                        and routed_idx == 2
                    ),
                    allow_sep=(
                        getattr(self, "sep_extension_enabled", False)
                        and routed_idx == 2
                    ),
                )[0]
                pred["token_fusion_applied"] = False
                if self.attachment_set_enabled and routed_idx <= len(attachment_set_outputs):
                    routed_type = (
                        str(expected_list[b] or "").strip().lower()
                        or self.expert_names[routed_idx]
                    )
                    full_set_outputs = attachment_set_outputs[routed_idx - 1]
                    set_outputs = {
                        key: (
                            value[b:b + 1]
                            if isinstance(value, torch.Tensor)
                            and value.ndim > 0
                            and value.size(0) == features.size(0)
                            else value
                        )
                        for key, value in full_set_outputs.items()
                    }
                    anchor_pointer_logits = None
                    dummy_pointer_logits = None
                    atom_hidden = pred.pop("_decoder_atom_hidden", None)
                    pointer_head = self.attachment_set_heads[routed_idx - 1]
                    if (
                        isinstance(pointer_head, MultiScaleAttachmentPointerHead)
                        and isinstance(atom_hidden, torch.Tensor)
                    ):
                        cc = pred.get(ATOM_FORMAT) or {}
                        symbols = list(cc.get("symbols") or [])
                        coords = list(cc.get("coords") or [])
                        if len(symbols) == atom_hidden.size(0) == len(coords):
                            valid_mask = torch.ones(
                                len(symbols), dtype=torch.bool, device=atom_hidden.device
                            ).unsqueeze(0)
                            dummy_mask = torch.tensor(
                                [
                                    "*" in str(symbol)
                                    or is_markush_label(normalize_label(str(symbol)))
                                    for symbol in symbols
                                ],
                                dtype=torch.bool,
                                device=atom_hidden.device,
                            ).unsqueeze(0)
                            atom_coords = torch.tensor(
                                coords,
                                dtype=atom_hidden.dtype,
                                device=atom_hidden.device,
                            ).unsqueeze(0)
                            anchor_pointer_logits = pointer_head.anchor_pointer_logits(
                                set_outputs,
                                atom_hidden.unsqueeze(0),
                                atom_coords=atom_coords,
                                atom_mask=valid_mask & ~dummy_mask,
                            )[0]
                            dummy_pointer_logits = pointer_head.dummy_pointer_logits(
                                set_outputs,
                                atom_hidden.unsqueeze(0),
                                atom_coords=atom_coords,
                                atom_mask=valid_mask,
                            )[0]
                    pred = self._attach_direct_attachment_diagnostics(
                        pred,
                        set_outputs,
                        expected_type=routed_type,
                        anchor_pointer_logits=anchor_pointer_logits,
                        dummy_pointer_logits=dummy_pointer_logits,
                    )
                # Legacy diagnostic fallback only; production never substitutes
                # an expert0 graph for a sidecar result.
                if (
                    getattr(self, "sidecar_broken_decode_fallback", False)
                    and self._sidecar_decode_is_broken(pred)
                ):
                    fb = self.experts[0].decode(
                        enc_b,
                        hiddens=None,
                        beam_size=beam_size,
                        n_best=n_best,
                        decode_constraints=sample_constraints,
                    )[0]
                    fb["fallback_to_expert0"] = str(
                        pred.get("decode_quality_issue", "empty_sidecar_decode")
                    )
                    fb["sidecar_route"] = routed_idx
                    pred = fb
            elif (
                self.expert_kind == "full_mixture"
                and self.attachment_set_enabled
                and effective_decode_mode == "residual_base"
                and 0 < routed_idx <= len(attachment_set_outputs)
            ):
                # The frozen decoder owns the backbone here: fragment-only
                # constraints (star budget/bias, atom cap) are omitted;
                # explicit caller constraints are still honored.
                base_constraints = dict(decode_constraints or {})
                pred = self.experts[0].decode(
                    enc_b,
                    hiddens=None,
                    beam_size=beam_size,
                    n_best=n_best,
                    decode_constraints=base_constraints,
                    return_atom_hidden=(
                        self.attachment_set_feature_mode
                        in {
                            "multiscale_pointer",
                            "multiscale_pointer_heatmap",
                        }
                    ),
                )[0]
                routed_type = (
                    str(expected_list[b] or "").strip().lower()
                    or self.expert_names[routed_idx]
                )
                full_set_outputs = attachment_set_outputs[routed_idx - 1]
                set_outputs = {
                    key: (
                        value[b:b + 1]
                        if isinstance(value, torch.Tensor)
                        and value.ndim > 0
                        and value.size(0) == features.size(0)
                        else value
                    )
                    for key, value in full_set_outputs.items()
                }
                anchor_pointer_logits = None
                atom_hidden = pred.pop("_decoder_atom_hidden", None)
                pointer_head = self.attachment_set_heads[routed_idx - 1]
                if (
                    isinstance(pointer_head, MultiScaleAttachmentPointerHead)
                    and isinstance(atom_hidden, torch.Tensor)
                ):
                    cc = pred.get(ATOM_FORMAT) or {}
                    symbols = list(cc.get("symbols") or [])
                    coords = list(cc.get("coords") or [])
                    if len(symbols) == atom_hidden.size(0) == len(coords):
                        atom_mask = torch.tensor(
                            [
                                str(symbol) != "*"
                                and "*" not in str(symbol)
                                and not is_markush_label(normalize_label(str(symbol)))
                                and not invalid_decode_symbol(str(symbol))
                                for symbol in symbols
                            ],
                            dtype=torch.bool,
                            device=atom_hidden.device,
                        ).unsqueeze(0)
                        atom_coords = torch.tensor(
                            coords,
                            dtype=atom_hidden.dtype,
                            device=atom_hidden.device,
                        ).unsqueeze(0)
                        anchor_pointer_logits = pointer_head.anchor_pointer_logits(
                            set_outputs,
                            atom_hidden.unsqueeze(0),
                            atom_coords=atom_coords,
                            atom_mask=atom_mask,
                        )[0]
                pred = self._apply_attachment_set_residual(
                    pred,
                    set_outputs,
                    batch_index=0,
                    expected_type=routed_type,
                    anchor_pointer_logits=anchor_pointer_logits,
                )
                pred["token_fusion_applied"] = False
            elif self.expert_kind == "full_mixture":
                # Sidecar path: lockstep probability-space mixture of expert0+expert1.
                sidecar_idx = int(routed_idx)
                if self.full_mixture_sidecar_mode == "per_sidecar" and len(self.experts) >= self.num_experts:
                    if sidecar_idx <= 0 or sidecar_idx >= len(self.experts):
                        sidecar_idx = 1
                    w2 = self._mixture_weights_for_sidecar(weights[b], sidecar_idx)
                else:
                    sidecar_idx = 1
                    w2 = self._mixture_weights_2(weights[b])
                pred = self._decode_mixture_single(
                    enc_b,
                    w2,
                    sample_constraints,
                    beam_size,
                    n_best,
                    sidecar_expert_idx=sidecar_idx,
                )
            else:
                # lora sidecar: single LoRA-adapted decode (gate set above).
                pred = self._decoder.decode(
                    enc_b, hiddens=None, beam_size=beam_size, n_best=n_best,
                    decode_constraints=sample_constraints,
                )[0]

            # Valence-aware edge repair, sidecar paths only (complete/forced-
            # default rows never reach here): over-valent edges are repaired in
            # the graph layer, dropping dummy bonds and lowest bond orders first.
            if (not fc and self.valence_repair_enabled
                    and not pred.get("decode_quality_issue")):
                cc = pred.get(ATOM_FORMAT) or {}
                symbols = cc.get("symbols") if isinstance(cc, dict) else []
                edges = pred.get("edges")
                edge_scores = pred.get("edge_scores") or {}
                if symbols and isinstance(edges, list):
                    new_edges, new_scores = valence_aware_edge_repair(symbols, edges, edge_scores)
                    pred["edges"] = new_edges
                    if edge_scores:
                        pred["edge_scores"] = new_scores

            w_meta = weights[b].detach().float().cpu().tolist()
            if not fc and self.token_fusion_mode == "adaptive":
                pred.setdefault("token_fusion_mode", self.token_fusion_mode)
                pred.setdefault("token_fusion_dispatch", self.token_fusion_dispatch)
            pred["expert_weights"] = w_meta
            pred["routed_expert"] = self.expert_names[0 if fc else routed_idx]
            pred["routing_forced_default"] = fc
            pred["routing_forced_complete"] = fc           # legacy metadata name
            pred["routing_strategy"] = self.routing_strategy
            pred["routing_confidence"] = float(top_conf[b].detach().float().cpu())
            pred["routing_required_threshold"] = float(required_threshold[b].detach().float().cpu())
            self._attach_confidence(pred, enc_b, weights[b])
            predictions.append(pred)

        self._clear_caches()
        return predictions

    def _clear_caches(self) -> None:
        for expert in self.experts:
            for dec in expert.decoder.values():
                inner = getattr(dec, "decoder", None)
                state = getattr(inner, "state", None)
                if isinstance(state, dict):
                    state["cache"] = None
                    state["src"] = None

    def _attach_confidence(self, pred: dict, enc_b: torch.Tensor, w_b: torch.Tensor) -> None:
        if self.confidence_head is None:
            return
        cc = pred.get(ATOM_FORMAT, {})
        atom_scores = cc.get("atom_scores") or []
        edge_scores = pred.get("edge_scores") or {}
        num_tokens = len(atom_scores) if atom_scores else 1
        if atom_scores:
            token_probs = [max(float(s), 1e-6) for s in atom_scores]
            token_nll = float(np.mean([-math.log(s) for s in token_probs]))
            # MolScribe overall_score: geometric mean of token probs
            avg_token_score = float(np.prod(token_probs) ** (1.0 / len(token_probs)))
        else:
            token_nll = 5.0
            avg_token_score = 0.0
        if edge_scores:
            edge_probs = [max(float(v), 1e-6) for v in edge_scores.values()]
            edge_nll = float(np.mean([-math.log(s) for s in edge_probs]))
            edge_geom = float(np.prod(edge_probs) ** (1.0 / max(1, len(edge_probs))))
        else:
            edge_nll = 2.0
            edge_geom = 0.0
        overall_score = avg_token_score * (edge_geom ** 0.5) if edge_geom > 0 else avg_token_score
        # per-position entropy from atom scores (higher = more uncertain)
        if atom_scores:
            pos_entropy = float(np.mean([-s * math.log(max(s, 1e-6)) for s in token_probs]))
        else:
            pos_entropy = math.log(100)  # max entropy fallback
        device = enc_b.device
        pooled = self.confidence_head.attention_pool(enc_b.unsqueeze(0) if enc_b.dim() == 2 else enc_b)
        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)
        tnll = torch.tensor([token_nll], device=device)
        enll = torch.tensor([edge_nll], device=device)
        oscore = torch.tensor([overall_score], device=device)
        pentropy = torch.tensor([pos_entropy], device=device)
        ntok = torch.tensor([float(num_tokens)], device=device)
        logits = self.confidence_head(pooled, w_b.view(1, -1), tnll, enll, oscore, pentropy, ntok)
        pred["confidence"] = float(self.confidence_head.expected_confidence(logits)[0])
