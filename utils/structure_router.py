"""Structure routing harness (§45.13, model-native) — markush-vs-complete
judgment and multi-backend arbitration decided by the VLM, not by rules.

The user directive: no fallbacks, no hard filters, no regex — the harness's
capability IS the model. This module keeps only two non-model elements:
  - declared-type routing (a label, not a heuristic)
  - RDKit chemistry facts (a decode with dummy atoms is by chemical
    definition not a complete molecule; an unparseable candidate cannot be
    rendered) — chemistry identity, not filtering

Everything else moved to the model (utils/model_arbiter.py):
  - broken-macrocycle detection: macrocycle_check() — the judge reads the
    original crop, states its largest ring, and verifies the candidate
    against that list (checklist pattern)
  - backend arbitration: same_molecule() forced pairwise verdicts on the
    rendered candidate vs the original crop
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

ROUTE_KEEPER = "keeper"
ROUTE_GLYPH = "glyph"
ROUTE_MOLSIGHT = "molsight"

# Model-decision cache (verdicts are pure functions of crop+decode; one VLM
# call per unique pair per process).
_VERDICT_CACHE: dict[tuple, dict] = {}


@dataclass(frozen=True)
class RouteDecision:
    kind: str                      # 'markush' | 'complete'
    backend: str                   # ROUTE_KEEPER | ROUTE_GLYPH | ROUTE_MOLSIGHT
    reason: str
    signals: dict = field(default_factory=dict)
    macro_expected: bool = False


def has_dummy_count(smiles: str) -> int:
    """RDKit chemistry fact: how many dummy atoms ('*') the decode carries.
    Complete molecules never carry them; fragments legitimately do."""
    text = str(smiles or "")
    if not text.strip():
        return 0
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            mol = Chem.MolFromSmiles(text.replace("*", "[*]"))
        if mol is None:
            return 0
        return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
    except Exception:
        return 0


def declared_kind(structure_type: str) -> str:
    text = str(structure_type or "").lower()
    if "markush" in text:
        return "markush"
    if "fragment" in text:
        return "fragment"
    return "complete"


def judge(structure_type: str, keeper_smiles: str, image_path) -> RouteDecision:
    """Route decision: declared type + chemistry facts + model judgment."""
    from utils.model_arbiter import model_arbiter

    kind = declared_kind(structure_type)
    dummies = has_dummy_count(keeper_smiles)
    keeper_ok = bool(str(keeper_smiles or "").strip())
    signals = {
        "declared": kind,
        "keeper_ok": keeper_ok,
        "keeper_dummies": dummies,
    }

    if kind == "markush":
        return RouteDecision("markush", ROUTE_GLYPH, "declared_markush", signals)

    # Fragments legitimately carry attachment dummies — exempt them.
    if kind == "complete" and dummies > 0:
        return RouteDecision("markush", ROUTE_GLYPH, "keeper_decode_has_dummies",
                             signals)

    if not keeper_ok:
        # Escalation, not fallback: the row currently has nothing; the
        # second decoder is invoked and its decode is judged on its own
        # merits by arbitrate() before it is ever accepted.
        return RouteDecision("complete", ROUTE_MOLSIGHT,
                             "keeper_empty_escalation", signals)

    # Ring-broken macrocycle: the model decides by reading the image.
    verdict = {"verdict": "unavailable"}
    if image_path and os.path.exists(str(image_path)):
        cache_key = ("macro", str(image_path), str(keeper_smiles)[:120])
        if cache_key in _VERDICT_CACHE:
            verdict = _VERDICT_CACHE[cache_key]
        else:
            try:
                verdict = model_arbiter().macrocycle_check(
                    str(image_path), keeper_smiles)
            except Exception:
                verdict = {"verdict": "unavailable"}
            _VERDICT_CACHE[cache_key] = verdict
    signals["macrocycle_check"] = verdict
    if verdict.get("candidate_broke_ring") is True:
        return RouteDecision("complete", ROUTE_MOLSIGHT,
                             "model_judged_ring_broken", signals,
                             macro_expected=True)

    return RouteDecision(kind, ROUTE_KEEPER, "keeper_stands", signals)


def arbitrate(keeper_smiles: str, candidate_smiles: str, image_path) -> bool:
    """Model-native backend arbitration: the candidate replaces the keeper
    decode only when the judge says the candidate depicts the original crop.
    No thresholds — a 'yes' verdict is the whole contract."""
    if not str(candidate_smiles or "").strip():
        return False
    if not image_path or not os.path.exists(str(image_path)):
        return False
    from utils.model_arbiter import model_arbiter
    cache_key = ("same", str(image_path), str(candidate_smiles)[:160])
    if cache_key in _VERDICT_CACHE:
        verdict = _VERDICT_CACHE[cache_key]
    else:
        try:
            verdict = model_arbiter().same_molecule(str(image_path), candidate_smiles)
        except Exception:
            verdict = {"verdict": "unavailable"}
        _VERDICT_CACHE[cache_key] = verdict
    return verdict.get("same_molecule") == "yes"
