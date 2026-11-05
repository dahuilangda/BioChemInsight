"""Structure routing: markush-vs-complete judgment and multi-backend
arbitration.

Non-model elements: declared-type routing (a label) and RDKit chemistry
facts (a complete molecule carries no dummy atoms; an unparseable candidate
cannot be rendered). Judgment lives in utils/model_arbiter.py.
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


def has_dummy_count(smiles: str) -> int:
    """Number of dummy atoms ('*') in the decode. Complete molecules never
    carry them; fragments legitimately do."""
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
    kind = declared_kind(structure_type)
    dummies = has_dummy_count(keeper_smiles)
    keeper_ok = bool(str(keeper_smiles or "").strip())
    signals = {"declared": kind, "dummy_atoms": dummies, "keeper_decode_ok": keeper_ok}

    if kind == "markush":
        return RouteDecision("markush", ROUTE_GLYPH, "declared_markush", signals)

    # Fragments legitimately carry attachment dummies — exempt them.
    if kind == "complete" and dummies > 0:
        return RouteDecision("markush", ROUTE_GLYPH, "keeper_decode_has_dummies",
                             signals)

    if not keeper_ok:
        # Empty keeper decode: consult the second decoder; its decode is
        # accepted only if arbitrate() approves it.
        return RouteDecision("complete", ROUTE_MOLSIGHT,
                             "keeper_empty_escalation", signals)

    # Ring-broken macrocycle: the model decides by reading the image.
    verdict = {"verdict": "unavailable"}
    if image_path and os.path.exists(str(image_path)):
        from utils.model_arbiter import model_arbiter
        cache_key = ("macro", str(image_path), str(keeper_smiles)[:120])
        if cache_key in _VERDICT_CACHE:
            verdict = _VERDICT_CACHE[cache_key]
        else:
            try:
                verdict = model_arbiter().macrocycle_check(
                    str(image_path), keeper_smiles)
            except Exception as arbiter_exc:
                print(f"structure router: arbiter unavailable "
                      f"({type(arbiter_exc).__name__}: {arbiter_exc})")
                verdict = {"verdict": "unavailable"}
            # an unavailable verdict is a failed probe, not an answer: keep
            # it out of the cache so the next call retries the model
            if verdict.get("verdict") != "unavailable":
                _VERDICT_CACHE[cache_key] = verdict
    # The checklist probe returns ring facts, same_molecule returns a pairwise
    # verdict — surface whichever fields the model actually produced.
    for key in ("candidate_broke_ring", "same_molecule", "verdict"):
        if key in verdict:
            signals["model_verdict"] = f"{key}={verdict[key]}"
            break
    else:
        signals["model_verdict"] = "unavailable"
    if verdict.get("candidate_broke_ring") is True:
        return RouteDecision("complete", ROUTE_MOLSIGHT,
                             "model_judged_ring_broken", signals)

    return RouteDecision(kind, ROUTE_KEEPER, "keeper_stands", signals)


def arbitrate(keeper_smiles: str, candidate_smiles: str, image_path) -> bool:
    """Arbitration: the candidate replaces the keeper decode only when the
    judge confirms it depicts the original crop (a 'yes' verdict)."""
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
        except Exception as exc:
            print(f"structure router: arbitrate unavailable "
                  f"({type(exc).__name__}: {exc})")
            return False               # failed probe: not cached, retried next call
        _VERDICT_CACHE[cache_key] = verdict
    return verdict.get("same_molecule") == "yes"
