from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import cv2
import torch
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Geometry import Point3D

from training.molnextr_markush.tools.train_fragment_attachment_expert import (
    load_fragment_attachment_checkpoint,
    load_frozen_encoder,
    predict_fragment_attachment_batch,
)


@dataclass(frozen=True)
class FragmentAttachmentFusionResult:
    smiles: str
    molblock: str
    ready: bool
    issue: str = ""
    endpoint: dict[str, Any] | None = None
    anchor_atom_index: int = -1
    anchor_distance: float = -1.0


def _mol_from_block_or_smiles(molblock: str, smiles: str) -> Chem.Mol | None:
    mol = None
    if molblock:
        mol = Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                try:
                    mol.UpdatePropertyCache(strict=False)
                except Exception:
                    return None
    if mol is None and smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                rdDepictor.Compute2DCoords(mol)
            except Exception:
                pass
    return mol


def _normalized_atom_positions(mol: Chem.Mol) -> list[tuple[int, float, float]]:
    if mol.GetNumConformers() == 0:
        try:
            Chem.rdDepictor.Compute2DCoords(mol)
        except Exception:
            return []
    conf = mol.GetConformer()
    points = [(idx, float(conf.GetAtomPosition(idx).x), float(conf.GetAtomPosition(idx).y)) for idx in range(mol.GetNumAtoms())]
    if not points:
        return []
    min_x = min(x for _, x, _ in points)
    max_x = max(x for _, x, _ in points)
    min_y = min(y for _, _, y in points)
    max_y = max(y for _, _, y in points)
    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)
    return [(idx, (x - min_x) / span_x, (y - min_y) / span_y) for idx, x, y in points]


def _nearest_atom(mol: Chem.Mol, endpoint_x: float, endpoint_y: float) -> list[tuple[int, float]]:
    """Candidate anchor atoms nearest to the endpoint first (skipping H), as a
    sorted list. Returning the ranking lets ``fuse_dummy_attachment`` skip an
    atom whose valence is already saturated (a graft would exceed it and yield
    an unparseable fused SMILES) and try the next-nearest valid atom instead."""
    return sorted(
        ((idx, sqrt((x - endpoint_x) ** 2 + (y - endpoint_y) ** 2))
         for idx, x, y in _normalized_atom_positions(mol)
         if mol.GetAtomWithIdx(idx).GetAtomicNum() != 1),
        key=lambda item: item[1],
    )


def fuse_dummy_attachment(
    *,
    backbone_smiles: str,
    backbone_molblock: str,
    endpoint: dict[str, Any],
    min_sidecar_confidence: float = 0.20,
    max_anchor_distance: float = 0.35,
) -> FragmentAttachmentFusionResult:
    if not endpoint:
        return FragmentAttachmentFusionResult(backbone_smiles, backbone_molblock, False, "missing_endpoint_prediction")
    if int(endpoint.get("endpoint_presence") or 0) != 1:
        return FragmentAttachmentFusionResult(backbone_smiles, backbone_molblock, False, "endpoint_presence_negative", endpoint)
    confidence = float(endpoint.get("sidecar_confidence") or 0.0)
    if confidence < float(min_sidecar_confidence):
        return FragmentAttachmentFusionResult(backbone_smiles, backbone_molblock, False, "endpoint_confidence_below_threshold", endpoint)
    mol = _mol_from_block_or_smiles(backbone_molblock, backbone_smiles)
    if mol is None:
        return FragmentAttachmentFusionResult(backbone_smiles, backbone_molblock, False, "backbone_mol_unparseable", endpoint)
    endpoint_x = min(1.0, max(0.0, float(endpoint.get("endpoint_x") or 0.0)))
    endpoint_y = min(1.0, max(0.0, float(endpoint.get("endpoint_y") or 0.0)))
    candidates = _nearest_atom(mol, endpoint_x, endpoint_y)
    if not candidates:
        return FragmentAttachmentFusionResult(backbone_smiles, backbone_molblock, False, "no_anchor_atom_candidate", endpoint)
    if candidates[0][1] > float(max_anchor_distance):
        return FragmentAttachmentFusionResult(
            backbone_smiles, backbone_molblock, False,
            "endpoint_anchor_distance_above_threshold", endpoint,
            int(candidates[0][0]), float(candidates[0][1]))
    # Valence-aware graft: try anchors nearest-first, keep the first that yields
    # a chemically valid fused mol. A nearest atom at its valence ceiling (e.g.
    # quaternary C) produces an unparseable fused SMILES; the next-nearest atom
    # that can accept the bond is the correct anchor. When the nearest atom is
    # valid it is used unchanged (no behavior change vs the single-anchor path).
    chosen_smiles = chosen_molblock = ""
    chosen_anchor = -1
    chosen_dist = -1.0
    for anchor_index, anchor_distance in candidates:
        if anchor_distance > float(max_anchor_distance):
            break
        rw = Chem.RWMol(mol)
        dummy_index = rw.AddAtom(Chem.Atom("*"))
        rw.AddBond(int(anchor_index), int(dummy_index), Chem.BondType.SINGLE)
        cand = rw.GetMol()
        try:
            Chem.SanitizeMol(cand)
        except Exception:
            try:
                cand.UpdatePropertyCache(strict=False)
            except Exception:
                continue
        try:
            cand_smiles = Chem.MolToSmiles(cand, canonical=True, isomericSmiles=True)
        except Exception:
            continue
        parsed = Chem.MolFromSmiles(cand_smiles) if cand_smiles else None
        if parsed is None:
            continue
        if sum(1 for atom in parsed.GetAtoms() if atom.GetAtomicNum() == 0) != 1:
            continue
        # Valid graft — place the dummy off the anchor in the endpoint's direction
        if cand.GetNumConformers() == 0:
            rdDepictor.Compute2DCoords(cand)
        conf = cand.GetConformer()
        anchor_pos = conf.GetAtomPosition(int(anchor_index))
        dx = 1.5 if str(endpoint.get("side")) == "right" else -1.5 if str(endpoint.get("side")) == "left" else 0.0
        dy = 1.5 if str(endpoint.get("side")) == "top" else -1.5 if str(endpoint.get("side")) == "bottom" else 0.0
        if dx == 0.0 and dy == 0.0:
            dx = 1.5
        conf.SetAtomPosition(int(dummy_index), Point3D(float(anchor_pos.x + dx), float(anchor_pos.y + dy), 0.0))
        try:
            cand_molblock = Chem.MolToMolBlock(cand, kekulize=False)
        except Exception:
            continue
        chosen_smiles, chosen_molblock = cand_smiles, cand_molblock
        chosen_anchor, chosen_dist = int(anchor_index), float(anchor_distance)
        break
    if chosen_anchor < 0:
        return FragmentAttachmentFusionResult(
            backbone_smiles, backbone_molblock, False,
            "fused_smiles_unparseable", endpoint)
    return FragmentAttachmentFusionResult(
        smiles=chosen_smiles, molblock=chosen_molblock, ready=True,
        endpoint=endpoint, anchor_atom_index=chosen_anchor, anchor_distance=chosen_dist)


class FragmentAttachmentFusionRuntime:
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        base_checkpoint: str | Path | None = None,
        device: torch.device | None = None,
        batch_size: int = 16,
        num_workers: int = 0,
        allow_debug_checkpoint: bool = False,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.expert, self.checkpoint = load_fragment_attachment_checkpoint(self.checkpoint_path, device=self.device)
        validate_fragment_attachment_checkpoint_metadata(
            self.checkpoint,
            allow_debug_checkpoint=bool(allow_debug_checkpoint),
        )
        resolved_base = str(base_checkpoint or self.checkpoint.get("base_checkpoint") or "")
        if not resolved_base:
            raise ValueError("fragment attachment checkpoint does not contain base_checkpoint and none was provided")
        config = {
            "output_dir": str(self.checkpoint_path.parent),
            "augment": False,
        }
        self.encoder, self.model_args = load_frozen_encoder(resolved_base, config, self.device)
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.batch_size = max(1, int(batch_size or 1))
        self.num_workers = max(0, int(num_workers or 0))

    def predict_endpoints(self, image_paths: list[str]) -> list[dict[str, Any]]:
        missing = [path for path in image_paths if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"fragment attachment image paths do not exist: {missing[:5]}")
        unreadable = [path for path in image_paths if cv2.imread(path) is None]
        if unreadable:
            raise ValueError(f"fragment attachment image paths are unreadable: {unreadable[:5]}")
        return predict_fragment_attachment_batch(
            encoder=self.encoder,
            fragment_expert=self.expert,
            model_args=self.model_args,
            image_paths=image_paths,
            device=self.device,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


def validate_fragment_attachment_checkpoint_metadata(
    checkpoint: dict[str, Any],
    *,
    allow_debug_checkpoint: bool,
) -> None:
    metrics = checkpoint.get("metrics") if isinstance(checkpoint.get("metrics"), dict) else {}
    if not metrics:
        raise ValueError("fragment attachment checkpoint is missing metrics; refusing runtime use")
    stage = str(metrics.get("config", {}).get("training_stage") or metrics.get("gate_report", {}).get("training_stage") or "")
    model_scale = metrics.get("model_scale") if isinstance(metrics.get("model_scale"), dict) else {}
    gate_report = metrics.get("gate_report") if isinstance(metrics.get("gate_report"), dict) else {}
    debug_reasons = []
    if metrics.get("debug_only") is True or model_scale.get("debug_only") is True:
        debug_reasons.append("checkpoint is marked debug_only")
    if stage == "micro_smoke":
        debug_reasons.append("checkpoint training_stage is micro_smoke")
    if gate_report.get("enforced") is not True or gate_report.get("accepted") is not True:
        debug_reasons.append("checkpoint training gate was not enforced and accepted")
    if model_scale.get("allowed_for_measured_or_formal_evidence") is not True:
        debug_reasons.append("checkpoint model_scale is not accepted for measured/formal evidence")
    if debug_reasons and not allow_debug_checkpoint:
        raise ValueError(
            "fragment attachment checkpoint is not production-eligible; "
            + "; ".join(debug_reasons)
            + ". Pass --allow-debug-fragment-attachment-checkpoint only for explicit diagnostic runs."
        )


def preflight_fragment_attachment_checkpoint(
    checkpoint_path: str | Path,
    *,
    allow_debug_checkpoint: bool = False,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"fragment attachment checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"fragment attachment checkpoint must be a dict: {path}")
    validate_fragment_attachment_checkpoint_metadata(
        checkpoint,
        allow_debug_checkpoint=bool(allow_debug_checkpoint),
    )
    return checkpoint
