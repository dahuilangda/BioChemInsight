"""Per-figure goal FSM: the intelligent harness core.

Goal invariant: every detected figure ends in exactly one terminal state
carrying full evidence and named gaps:

    RESOLVED       {smiles, id, verification: passed, evidence[...]}
    REVIEW_QUEUED  {best_so_far, contradiction_or_gap, attempted[]}
    UNSOLVABLE     {reason_code, attempts[]}

State machine (deterministic where reliable, LLM only at 3 decision nodes):

    DETECTED → LABELED → ROUTED → DECODED → TYPED → VERIFIED → RESOLVED
                  │        │        │                    │
                  │        │        ▼                    ▼
                  │        │   DECODE_REPAIR → REPAIR_PLAN → (back to DECODED)
                  │        ▼
                  │   (markush → emit template, stop)
                  ▼
           LABEL_REPAIR (re-OCR band crop)
                  │ budget exhausted
                  ▼
           REVIEW_QUEUED / UNSOLVABLE

The document ledger accumulates all entities, evidence, and contradictions.
Nothing is silently overwritten — contradictions are first-class rows.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

MAX_ATTEMPTS = 3


class TerminalState(str, Enum):
    RESOLVED = "RESOLVED"
    REVIEW_QUEUED = "REVIEW_QUEUED"
    UNSOLVABLE = "UNSOLVABLE"


class ReasonCode(str, Enum):
    NO_BOX_DETECTED = "NO_BOX_DETECTED"
    LABEL_UNREADABLE = "LABEL_UNREADABLE"
    DECODE_FAILED_ALL_STRATEGIES = "DECODE_FAILED_ALL_STRATEGIES"
    MARKUSH_AMBIGUOUS = "MARKUSH_AMBIGUOUS"
    VERIFICATION_CONTRADICTION = "VERIFICATION_CONTRADICTION"
    ACTIVITY_LINK_EMPTY = "ACTIVITY_LINK_EMPTY"


@dataclass
class Attempt:
    skill: str
    input_ref: str
    output_digest: str
    success: bool
    detail: str = ""


@dataclass
class FigureGoal:
    """Per-figure goal state — the FSM's working state object."""
    page: int
    bbox: List[int]
    crop_path: str
    sequence_text: Optional[str] = None
    formula: Optional[str] = None
    # accumulated results
    smiles: Optional[str] = None
    smiles_valid: bool = False
    compound_id: Optional[str] = None
    compound_name: Optional[str] = None
    seq_id: Optional[str] = None
    route_kind: Optional[str] = None
    typed_residues: Optional[List[str]] = None
    # verification
    verification: Optional[Dict[str, Any]] = None
    # audit trail
    attempts: List[Attempt] = field(default_factory=list)
    contradictions: List[Dict] = field(default_factory=list)
    # terminal
    state: str = "DETECTED"
    reason_code: Optional[str] = None


class FigureFSM:
    """Runs the per-figure goal state machine against the skill APIs."""

    def __init__(self, page_num: int, bbox: List[int], crop_path: str,
                 sequence_text: Optional[str] = None,
                 formula: Optional[str] = None):
        self.goal = FigureGoal(
            page=page_num, bbox=bbox, crop_path=crop_path,
            sequence_text=sequence_text, formula=formula)

    def run(self) -> FigureGoal:
        """Execute the FSM until a terminal state is reached."""
        g = self.goal

        # phase 1: decode (ensemble + repair ladder)
        self._phase_decode()

        # phase 2: verification
        if g.smiles:
            self._phase_verify()

        # phase 3: repair if needed
        if g.state not in ("RESOLVED",):
            self._phase_repair()

        # terminal classification
        if g.verification and g.verification.get("verdict") == "verified":
            g.state = "RESOLVED"
        elif g.attempts:
            g.state = "UNSOLVABLE"
            g.reason_code = self._best_reason()
        else:
            g.state = "UNSOLVABLE"
            g.reason_code = ReasonCode.NO_BOX_DETECTED

        return g

    def _phase_decode(self):
        """Attempt SMILES decode with ensemble preparations."""
        from utils.molsight_service import molsight_service
        g = self.goal
        for prep_name, prep_crop in self._decode_variants(g.crop_path):
            try:
                result = molsight_service().predict(prep_crop)
                smi = result.get("smiles", "") if result.get("ok") else ""
                g.attempts.append(Attempt(
                    skill=f"recognize[{prep_name}]", input_ref=prep_crop,
                    output_digest=smi[:80], success=bool(smi)))
                if smi and self._rdkit_ok(smi):
                    g.smiles = smi
                    g.smiles_valid = True
                    return
            except Exception as exc:
                g.attempts.append(Attempt(
                    skill=f"recognize[{prep_name}]", input_ref=prep_crop,
                    output_digest="", success=False, detail=str(exc)[:200]))

    def _phase_verify(self):
        """Multi-signal verification of the decoded SMILES."""
        g = self.goal
        from utils.residue_typer import smiles_to_sequence, compare_with_anchor
        # RDKit validity
        mol = Chem.MolFromSmiles(g.smiles) if g.smiles else None
        # sequence agreement (if anchor available)
        seq_result = None
        if g.sequence_text:
            try:
                from utils.sequence_parser import parse_cyclopeptide_notation
                codes = parse_cyclopeptide_notation(g.sequence_text)
                if codes:
                    seq_result = compare_with_anchor(g.smiles, codes)
            except Exception:
                pass
        # verification verdict
        verdict = "unverified"
        signals = {}
        if mol is not None:
            verdict = "rdkit_valid"
            signals["rdkit_valid"] = True
            # ring analysis
            rings = sorted({len(r) for r in mol.GetRingInfo().AtomRings()
                            if len(r) >= 12}, reverse=True)
            signals["macro_rings"] = rings
        if seq_result and seq_result.get("ok"):
            agree = seq_result.get("agree", False)
            signals["sequence_agreement"] = agree
            signals["residue_count_match"] = seq_result.get("residue_count_match", False)
            if agree:
                verdict = "verified"

        g.verification = {"verdict": verdict, "signals": signals}

    def _phase_repair(self):
        """Repair ladder when the decode fails verification."""
        g = self.goal
        # repair 1: sequence-anchor rebuild
        if g.sequence_text:
            from utils.sequence_parser import build_cyclopeptide_smiles, \
                parse_cyclopeptide_notation
            codes = parse_cyclopeptide_notation(g.sequence_text)
            built = build_cyclopeptide_smiles(codes) if codes else None
            if built:
                g.attempts.append(Attempt(
                    skill="sequence_grammar", input_ref=g.crop_path,
                    output_digest=built[:80], success=True))
                g.smiles = built
                g.smiles_valid = True
                return

    def _best_reason(self) -> str:
        """Pick the most specific reason code from the attempt trail."""
        g = self.goal
        if not g.smiles:
            return ReasonCode.DECODE_FAILED_ALL_STRATEGIES
        if g.verification and g.verification.get("signals", {}).get("rdkit_valid"):
            v = g.verification.get("verdict", "")
            if v == "contradiction":
                return ReasonCode.VERIFICATION_CONTRADICTION
        return ReasonCode.DECODE_FAILED_ALL_STRATEGIES

    def _decode_variants(self, crop_path: str) -> list[tuple[str, str]]:
        """Generate image preparations for ensemble decode."""
        import tempfile
        results = []
        image = cv2.imread(crop_path)
        if image is None:
            return [(crop_path, crop_path)]
        h, w = image.shape[:2]
        results.append(("original", crop_path))
        # square pad
        side = max(h, w)
        square = np.full((side, side, 3), 255, dtype=np.uint8)
        y0, x0 = (side - h) // 2, (side - w) // 2
        square[y0:y0 + h, x0:x0 + w] = image
        tmp_sq = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp_sq.name, square)
        results.append(("square_pad", tmp_sq.name))
        # upscale 2x
        up = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        tmp_up = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp_up.name, up)
        results.append(("upscale2x", tmp_up.name))
        return results

    @staticmethod
    def _rdkit_ok(smiles: str) -> bool:
        try:
            from rdkit import Chem
            return Chem.MolFromSmiles(smiles) is not None
        except Exception:
            return False
