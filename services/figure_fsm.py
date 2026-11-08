"""Per-figure goal FSM: decode -> verify -> anchor repair -> one terminal state.

Orchestrates the resolution of a single cropped figure:

    DETECTED
      -> _phase_decode  ensemble decode (original / square pad / 2x upscale /
                        margin trim); distinct candidates deduplicated by
                        canonical SMILES, each verified with the shared
                        multi-signal verifier in structure_api
      -> _phase_repair  deterministic rebuild from the sequence anchor
                        (the rebuild's own self-agreement signal excluded)
      -> _select        best-verified candidate wins; ties go to the
                        anchor-derived rebuild

Terminal states (exactly one per goal):
    RESOLVED       verification verdict "verified"
    REVIEW_QUEUED  a SMILES exists but verification could not confirm it
    UNSOLVABLE     no candidate at all (decode empty, no usable anchor)

The caller owns the input crop's lifetime; this module removes its own
decode temporaries. The resolved figure lands in the per-document
DocumentLedger by the caller (structure_api /v1/harness/resolve-figure).
"""
from __future__ import annotations

import hashlib
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

# budget of distinct decode candidates carried into verification
MAX_ATTEMPTS = 4

_VERIFY_RANK = {"verified": 3, "unverified": 2, "contradiction": 1,
                "rejected": 0}


def _verify_rank(verify: Optional[dict]) -> int:
    """Trustworthiness rank: verified > unverified > contradiction > rejected."""
    return _VERIFY_RANK.get((verify or {}).get("verdict"), 0)


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
    # candidate rows only (decode ensemble and anchor rebuild): the
    # SMILES and its verification; probe rows leave both None
    smiles: Optional[str] = None
    verify: Optional[Dict[str, Any]] = None


@dataclass
class FigureGoal:
    """Per-figure goal state — the FSM's working state object."""
    page: int
    bbox: List[int]
    crop_path: str
    sequence_text: Optional[str] = None
    formula: Optional[str] = None
    mw: Optional[float] = None
    # accumulated results
    smiles: Optional[str] = None
    smiles_valid: bool = False
    compound_id: Optional[str] = None
    compound_name: Optional[str] = None
    seq_id: Optional[str] = None
    route_kind: Optional[str] = None
    typed_residues: Optional[List[str]] = None
    # verification of `smiles`, and the skill that produced it
    verification: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
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
                 formula: Optional[str] = None,
                 mw: Optional[float] = None):
        self.goal = FigureGoal(
            page=page_num, bbox=bbox, crop_path=crop_path,
            sequence_text=sequence_text, formula=formula, mw=mw)

    def run(self) -> FigureGoal:
        """Execute the FSM until a terminal state is reached."""
        g = self.goal
        self._phase_decode()
        self._phase_repair()
        self._select()

        # terminal classification: every path ends in exactly one state
        verdict = (g.verification or {}).get("verdict")
        if verdict == "verified":
            g.state = TerminalState.RESOLVED.value
        elif g.smiles:
            g.state = TerminalState.REVIEW_QUEUED.value
            g.reason_code = self._best_reason()
        elif g.attempts:
            g.state = TerminalState.UNSOLVABLE.value
            g.reason_code = ReasonCode.DECODE_FAILED_ALL_STRATEGIES
        else:
            g.state = TerminalState.UNSOLVABLE.value
            g.reason_code = ReasonCode.NO_BOX_DETECTED
        return g

    # phases ---------------------------------------------------------------

    def _phase_decode(self):
        """Ensemble decode over image preparations. Distinct candidates are
        deduplicated by canonical SMILES (agreement across preparations is
        evidence) and each is verified; selection happens in _select."""
        g = self.goal
        from utils.molsight_service import molsight_service
        candidates: Dict[str, dict] = {}
        for prep_name, prep_path in self._decode_variants(g.crop_path):
            try:
                result = molsight_service().predict(prep_path)
                smi = result.get("smiles", "") if result.get("ok") else ""
                g.attempts.append(Attempt(
                    skill=f"recognize[{prep_name}]", input_ref=prep_path,
                    output_digest=smi[:80], success=bool(smi)))
                if smi:
                    key, valid = self._canonical_key(smi)
                    cand = candidates.get(key)
                    if cand is None:
                        candidates[key] = {"smiles": smi, "valid": valid,
                                           "preparations": [prep_name]}
                    else:
                        cand["preparations"].append(prep_name)
            except Exception as exc:
                g.attempts.append(Attempt(
                    skill=f"recognize[{prep_name}]", input_ref=prep_path,
                    output_digest="", success=False, detail=str(exc)[:200]))
            finally:
                # this module owns the preparation temporaries, not the
                # caller's input crop
                if prep_path != g.crop_path and os.path.exists(prep_path):
                    try:
                        os.unlink(prep_path)
                    except OSError:
                        pass
        ranked = sorted(candidates.values(),
                        key=lambda c: (not c["valid"], -len(c["preparations"])))
        for cand in ranked[:MAX_ATTEMPTS]:
            method = f"molsight[{'+'.join(cand['preparations'])}]"
            g.attempts.append(Attempt(
                skill=method, input_ref=g.crop_path,
                output_digest=cand["smiles"][:80], success=cand["valid"],
                smiles=cand["smiles"],
                verify=self._phase_verify(cand["smiles"])))

    def _phase_verify(self, smiles: str,
                      exclude_anchor: bool = False) -> Dict[str, Any]:
        """Single verification entry point, delegated to the shared
        multi-signal implementation in structure_api. Lazy import here:
        structure_api imports this module, so a module-level import would
        be circular. The anchor-derived rebuild excludes the sequence
        self-agreement signal — it cannot testify for itself."""
        g = self.goal
        anchor = None if exclude_anchor else g.sequence_text
        from services.structure_api import _verify_structure_impl
        return _verify_structure_impl(g.crop_path, smiles, g.formula, g.mw,
                                      anchor)

    def _phase_repair(self):
        """Repair rung: deterministic rebuild from the sequence anchor. The
        rebuild IS derived from the anchor, so the sequence self-agreement
        signal is excluded from its verification — RDKit validity, declared
        anchors, ink geometry and the arbiter judge it. A rebuild identical
        (canonical SMILES) to a decoded candidate reuses that candidate's
        verification instead of re-paying the VLM call."""
        g = self.goal
        if not g.sequence_text:
            return
        from utils.sequence_parser import rebuild_from_anchor
        built = rebuild_from_anchor(g.sequence_text)
        if not built:
            return
        twin = self._canonical_twin(built)
        verify = twin.verify if twin is not None \
            else self._phase_verify(built, exclude_anchor=True)
        g.attempts.append(Attempt(
            skill="sequence_grammar", input_ref=g.crop_path,
            output_digest=built[:80], success=True,
            smiles=built, verify=verify))

    def _select(self):
        """Best-verified candidate becomes the goal's result; among equals
        the candidate that CAME from the sequence anchor (the patent's own
        notation) outranks a decoded guess."""
        g = self.goal

        def _rank(a: Attempt) -> tuple:
            anchor_bonus = 1 if a.skill.startswith("sequence_grammar") else 0
            return (_verify_rank(a.verify), anchor_bonus)

        best = None
        for a in g.attempts:
            if a.smiles is None:
                continue
            if best is None or _rank(a) > _rank(best):
                best = a
        if best is None:
            return
        g.smiles = best.smiles
        g.smiles_valid = bool(best.success)
        g.verification = best.verify
        g.source = best.skill

    def _best_reason(self) -> str:
        """Pick the most specific reason code from the attempt trail."""
        g = self.goal
        if not g.smiles:
            return ReasonCode.DECODE_FAILED_ALL_STRATEGIES
        if g.verification and g.verification.get("verdict") == "contradiction":
            return ReasonCode.VERIFICATION_CONTRADICTION
        return ReasonCode.DECODE_FAILED_ALL_STRATEGIES

    # helpers --------------------------------------------------------------

    def _canonical_twin(self, smiles: str) -> Optional[Attempt]:
        """Verified decode candidate with the same canonical SMILES."""
        key, _ = self._canonical_key(smiles)
        for a in self.goal.attempts:
            if a.smiles is None or a.verify is None \
                    or not a.skill.startswith("molsight["):
                continue
            if self._canonical_key(a.smiles)[0] == key:
                return a
        return None

    @staticmethod
    def _canonical_key(smiles: str) -> tuple[str, bool]:
        """Dedup key: canonical SMILES for parseable output, a content hash
        otherwise (two distinct-but-equal-length invalid outputs must not
        collide). An RDKit fault on one SMILES is a per-input fact here:
        the explicit invalid branch below still keys the string and marks
        it unparseable, so nothing is silently dropped."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol), True
        except Exception:
            pass  # safe: the invalid branch is this function's named outcome
        return f"invalid:{hashlib.sha1(smiles.encode()).hexdigest()[:12]}", False

    def _decode_variants(self, crop_path: str) -> list[tuple[str, str]]:
        """Image preparations for the decode ensemble: original, square pad
        (avoid aspect distortion in the 512 resize), 2x upscale (sharper
        bonds) and white-margin trim (big borders resize the molecule
        smaller). The returned paths belong to the decode phase, which
        removes them after decoding; the original stays on the caller."""
        image = cv2.imread(crop_path)
        if image is None:
            return []
        h, w = image.shape[:2]
        preps: list[tuple[str, np.ndarray]] = [("original", image)]

        side = max(h, w)
        square = np.full((side, side, 3), 255, dtype=np.uint8)
        y0, x0 = (side - h) // 2, (side - w) // 2
        square[y0:y0 + h, x0:x0 + w] = image
        preps.append(("square_pad", square))

        preps.append(("upscale2x", cv2.resize(image, (w * 2, h * 2),
                                              interpolation=cv2.INTER_CUBIC)))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(gray < 235)
        if len(ys) > 0:
            ty0, ty1, tx0, tx1 = ys.min(), ys.max(), xs.min(), xs.max()
            if (ty1 - ty0) < h * 0.9 or (tx1 - tx0) < w * 0.9:
                pad = 12
                trimmed = image[max(0, ty0 - pad):min(h, ty1 + pad),
                                max(0, tx0 - pad):min(w, tx1 + pad)]
                if trimmed.size:
                    preps.append(("trimmed", trimmed))

        out: list[tuple[str, str]] = []
        for name, prep in preps:
            if prep.size == 0:
                continue
            if name == "original":
                out.append((name, crop_path))
                continue
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            cv2.imwrite(tmp.name, prep)
            out.append((name, tmp.name))
        return out
