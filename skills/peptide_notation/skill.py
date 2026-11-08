"""Atomic skill: cyclopeptide notation -> built, mass-checked SMILES.

Audit-and-correct loop:

The language model reads a cyclopeptide line notation (any patent style,
OCR noise included) and proposes unit definitions; RDKit validates each
fragment and the assembler builds the molecule; the declared formula /
exact mass from the patent table closes the loop. Every rejection feeds
back to the model as a specific correction request, and every attempt is
recorded — nothing silently overwritten, no silent fallback.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from skills.registry import register as _register_skill
from skills.residue_lexicon import (_LOCK, _load, _save, _canonical_fragment,
                                   lexicon_snapshot, NotBackboneForm)

RDLogger.DisableLog("rdApp.*")

SKILL = "peptide_notation"
MAX_ATTEMPTS = 6   # composite rows need extra correction rounds

_PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "prompt.md")


def _load_prompt() -> str:
    """The model contract lives in prompt.md — one source, no drift."""
    text = open(_PROMPT_PATH, encoding="utf-8").read()
    return text.split("<<<PROMPT", 1)[1].rsplit("PROMPT>>>", 1)[0].strip()


NOTATION_PROMPT = _load_prompt()


def _client():
    from constants import (LLM_OPENAI_COMPATIBLE_MODEL_KEY,
                           LLM_OPENAI_COMPATIBLE_MODEL_NAME,
                           LLM_OPENAI_COMPATIBLE_MODEL_URL)
    from openai import OpenAI
    return OpenAI(api_key=LLM_OPENAI_COMPATIBLE_MODEL_KEY,
                  base_url=LLM_OPENAI_COMPATIBLE_MODEL_URL), \
        LLM_OPENAI_COMPATIBLE_MODEL_NAME


def _vocabulary_hint() -> str:
    """Known monomer codes, as working memory for the reader."""
    lex = lexicon_snapshot()
    codes = sorted(k for k, v in lex.items() if v.get("note") != ""
                   and len(k) <= 20 and k[0].isupper())
    return "Known monomer codes: " + ", ".join(codes[:80])


class ModelCallError(RuntimeError):
    """LLM API call failed, or the reply was broken JSON; named apart from
    chemistry rejections so the attempt trail records the real cause."""


def _ask_model(notation: str, context: str, feedback: str) -> dict | None:
    try:
        client, model = _client()
        user = (f"Notation: {notation}\n{_vocabulary_hint()}"
                "\nIf a token is several known codes joined without '-', split "
                "it into separate units.")
        if context:
            user += f"\nContext: {context}"
        if feedback:
            user += f"\nYour previous answer had errors — correct it:\n{feedback}"
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": NOTATION_PROMPT},
                      {"role": "user", "content": user}],
            temperature=0.15, max_tokens=2500, timeout=180)
    except Exception as exc:
        raise ModelCallError(f"LLM API call failed: "
                             f"{type(exc).__name__}: {str(exc)[:160]}") from exc
    text = resp.choices[0].message.content.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None                      # genuinely no JSON in the reply
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ModelCallError(f"model emitted broken JSON at offset {exc.pos}: "
                             f"{exc.msg}") from exc


def _fragment_ok(smiles: str) -> str | None:
    """RDKit gate + canonicalization to backbone form (amine first, alpha
    carboxyl stripped to C(=O)). None when not an amino-acid skeleton."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 2:
        return None
    return _canonical_fragment(mol)


def _validate_units(parsed: dict) -> tuple[list[dict], list[str]]:
    """Per-unit resolution: the lexicon (previously verified definitions)
    outranks the model's fresh proposal; both pass the RDKit gate."""
    lex = {k.casefold(): v for k, v in lexicon_snapshot().items()}
    units, failures = [], []
    for u in parsed.get("units", []):
        code = str(u.get("code", "")).strip()
        name = str(u.get("name", "")).strip()
        if not code:
            failures.append("a unit is missing its code")
            continue
        frag = None
        for key in (code, name):
            cached = lex.get(key.casefold())
            if cached and cached.get("fragment_smiles"):
                frag = cached["fragment_smiles"]   # stored canonical form
                break
        if frag is None:
            try:
                frag = _fragment_ok(u.get("fragment_smiles", ""))
            except NotBackboneForm as exc:
                # the skeleton is an amino acid but sanitization failed —
                # carry the concrete reason so the model fixes the right thing
                failures.append(
                    f"unit '{code}': fragment_smiles "
                    f"'{u.get('fragment_smiles', '')[:60]}' rejected: {exc}")
                continue
        if not frag:
            failures.append(
                f"unit '{code}': fragment_smiles "
                f"'{u.get('fragment_smiles', '')[:60]}' does not parse in "
                "RDKit or does not start at the backbone amine N — rewrite "
                "it as N...C(=O)O (N(C) when N-methyl)")
            continue
        entry = {"fragment_smiles": frag,
                 "n_methyl": bool(u.get("n_methyl", False)),
                 "ring_nitrogen": bool(u.get("ring_nitrogen", False)),
                 "note": str(u.get("name", ""))[:200]}
        units.append({"index": len(units), "code": code, "entry": entry})
        _lexicon_pin(code, entry)
    if not parsed.get("units"):
        failures.append("no units were returned")
    return units, failures


MACRO_DIGIT = 10


def _find_amine_and_carboxyl(mol):
    """(amine N idx, carboxyl C idx) of a backbone-form fragment; either
    may be None for non-amino fragments (caps). The alpha carboxyl is the
    carbonyl adjacent to the carbon that carries the amine — never the
    first carbonyl in atom order (side-chain acids would win that scan).
    An O-leading fragment (hydroxy-acid depsipeptide unit, e.g. the KRAS
    family's phenyllactic FLac) occupies the same chain slot: the incoming
    ester/amide bond lands on its hydroxyl oxygen."""
    amine = carboxyl = None
    head_symbol = mol.GetAtomWithIdx(0).GetSymbol() if mol.GetNumAtoms() else None
    if head_symbol not in ("N", "O"):
        # cap-shaped: first carbon bearing a C=O
        for a in mol.GetAtoms():
            if a.GetSymbol() != "C":
                continue
            if any(b.GetBondTypeAsDouble() == 2 and mol.GetAtomWithIdx(
                    b.GetOtherAtomIdx(a.GetIdx())).GetSymbol() == "O"
                   for b in a.GetBonds()):
                carboxyl = a.GetIdx()
                break
        return amine, carboxyl
    amine = 0
    for alpha in mol.GetAtomWithIdx(0).GetNeighbors():
        if alpha.GetSymbol() != "C":
            continue
        for c2 in alpha.GetNeighbors():
            if c2.GetIdx() == 0 or c2.GetSymbol() != "C":
                continue
            if any(b.GetBondTypeAsDouble() == 2
                   and mol.GetAtomWithIdx(
                       b.GetOtherAtomIdx(c2.GetIdx())).GetSymbol() == "O"
                   for b in c2.GetBonds()):
                carboxyl = c2.GetIdx()
                break
        if carboxyl is not None:
            break
    if carboxyl is None:
        # degenerate ring-carboxyl shape (ACPC-like): any carbonyl carbon
        for a in mol.GetAtoms():
            if a.GetSymbol() == "C" and any(
                    b.GetBondTypeAsDouble() == 2 and mol.GetAtomWithIdx(
                        b.GetOtherAtomIdx(a.GetIdx())).GetSymbol() == "O"
                    for b in a.GetBonds()):
                carboxyl = a.GetIdx()
                break
    return amine, carboxyl


def _assemble(units: list[dict], parsed: dict,
              ) -> tuple[str | None, str, list[str]]:
    """Molecule-graph assembly: fragments are combined with RWMol and the
    backbone bonds (carboxyl C -> amine N, ring closure, cap, tail,
    terminal) are added explicitly. No SMILES text surgery, so ring digits,
    bracket amines and stray hydroxyls cannot corrupt the bond order.
    Returns (smiles, error, notes); notes carry every silent-correction
    decision the caller must surface."""
    notes: list[str] = []
    try:
        frags = []
        for u in units:
            mol = Chem.MolFromSmiles(u["entry"]["fragment_smiles"])
            if mol is None:
                return None, f"unit '{u['code']}' does not parse", notes
            frags.append(mol)
        caps = []
        for c in parsed.get("caps") or []:
            mol = Chem.MolFromSmiles(str(c.get("fragment_smiles", "")))
            if mol is None:
                return None, f"cap '{c.get('code')}' does not parse", notes
            caps.append(mol)

        ring_size = parsed.get("ring_size", len(frags))
        if (not isinstance(ring_size, int) or isinstance(ring_size, bool)
                or not 0 < ring_size <= len(frags)):
            notes.append(f"ring_size {parsed.get('ring_size')!r} out of "
                         f"range for {len(frags)} units; assembled full ring")
            ring_size = len(frags)
        # units beyond the ring are the linear tail; its attachment point is
        # unresolved until connector notes land, so only the ring assembles
        ring = frags[:ring_size]
        tail = []

        em = Chem.RWMol()
        amines, carboxyls = [], []
        for mol in caps + ring + tail:
            offset = em.GetNumAtoms()
            amine, carboxyl = _find_amine_and_carboxyl(mol)
            for a in mol.GetAtoms():
                em.AddAtom(Chem.Atom(a))
            for b in mol.GetBonds():
                em.AddBond(b.GetBeginAtomIdx() + offset,
                           b.GetEndAtomIdx() + offset, b.GetBondType())
            amines.append(offset + amine if amine is not None else None)
            carboxyls.append(offset + carboxyl if carboxyl is not None else None)

        ncap = len(caps)
        chain = list(range(len(ring) + len(tail)))
        head_amine = amines[ncap] if ncap < len(amines) else None
        # cap acyls bond the head amine
        for i in range(ncap):
            if carboxyls[i] is None or head_amine is None:
                return None, "cap cannot anchor: no acyl carbon or no head amine", notes
            em.AddBond(carboxyls[i], head_amine, Chem.BondType.SINGLE)
        # chain bonds: carboxyl[i] -> amine[i+1]
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            if carboxyls[ncap + i] is None or amines[ncap + i + 1] is None:
                return None, ("unit lacks a backbone amine/carboxyl: "
                              f"positions {i + 1}->{i + 2}"), notes
            em.AddBond(carboxyls[ncap + i], amines[ncap + i + 1],
                       Chem.BondType.SINGLE)
        # ring closure: last chain carboxyl -> head amine (only for a true
        # macrocycle; with a tail the closure skips the tail segment)
        if tail:
            closure_from = carboxyls[ncap + len(ring) - 1]
            em.AddBond(closure_from, head_amine, Chem.BondType.SINGLE)
        else:
            em.AddBond(carboxyls[-1], head_amine, Chem.BondType.SINGLE)

        terminal = str(parsed.get("terminal") or "").strip().upper()
        if tail and terminal == "NH2":
            n = em.AddAtom(Chem.Atom("N"))
            em.AddBond(carboxyls[-1], n, Chem.BondType.SINGLE)
        elif tail and terminal == "OH":
            o = em.AddAtom(Chem.Atom("O"))
            em.AddBond(carboxyls[-1], o, Chem.BondType.SINGLE)

        mol = em.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol), "", notes
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:160]}", notes


def _cap_acyl_smiles(cap: dict) -> str:
    smi = str((cap or {}).get("fragment_smiles", "")).strip()
    return smi


def _to_backbone_form(fragment: str) -> str:
    """Safety net: strip the alpha carboxyl's OH, found via the backbone
    amine's alpha carbon (side-chain acids survive). Lexicon entries are
    already backbone form, so this normally passes through."""
    mol = Chem.MolFromSmiles(fragment)
    if mol is None:
        return fragment
    em = Chem.RWMol(mol)
    amine = mol.GetAtomWithIdx(0)
    if amine.GetSymbol() != "N":
        return fragment
    # a free terminal OH would bond the next residue's amine in-chain;
    # only the alpha carboxyl may terminate a fragment
    tail = em.GetAtomWithIdx(em.GetNumAtoms() - 1)
    if (tail.GetSymbol() == "O" and tail.GetDegree() == 1
            and tail.GetFormalCharge() == 0):
        prev = next(iter(tail.GetNeighbors()), None)
        if prev is not None and prev.GetSymbol() != "C":
            pass
        elif prev is not None and not any(
                b.GetBondTypeAsDouble() == 2 for b in prev.GetBonds()):
            em.RemoveAtom(tail.GetIdx())
            try:
                Chem.SanitizeMol(em.GetMol())
                fragment = Chem.MolToSmiles(em.GetMol(), rootedAtAtom=0)
                em = Chem.RWMol(Chem.MolFromSmiles(fragment))
                amine = em.GetMol().GetAtomWithIdx(0)
            except Exception:
                return fragment
    # alpha C: carbon neighbor of the amine that carries a carboxyl
    for alpha in amine.GetNeighbors():
        if alpha.GetSymbol() != "C":
            continue
        for cand in alpha.GetNeighbors():
            if cand.GetSymbol() != "C":
                continue
            dbl_o = oh = None
            for b in cand.GetBonds():
                x = b.GetOtherAtomIdx(cand.GetIdx())
                xa = em.GetAtomWithIdx(x)
                if b.GetBondTypeAsDouble() == 2 and xa.GetSymbol() == "O":
                    dbl_o = x
                elif xa.GetSymbol() == "O" and xa.GetDegree() == 1:
                    oh = x
            if dbl_o is not None and oh is not None:
                em.RemoveAtom(oh)
                out = em.GetMol()
                try:
                    Chem.SanitizeMol(out)
                    return Chem.MolToSmiles(out)
                except Exception:
                    return fragment
    return fragment


def _head_amine(fragment: str, digit: int, branch: str) -> str:
    """Mark the first unit's amine with the ring digit; attach the cap acyl
    as a branch (N,N-diacyl convention). Handles plain and bracket amines."""
    mol = Chem.MolFromSmiles(fragment)
    if mol is None:
        return fragment
    smiles = Chem.MolToSmiles(mol, rootedAtAtom=0)
    if smiles.startswith("N") and not smiles.startswith("["):
        return f"N%{digit}{branch}{smiles[1:]}"
    if smiles.startswith("[N"):                       # [NH] / [N] bracket
        close = smiles.index("]")
        return f"{smiles[:close]}%{digit}{branch}{smiles[close:]}"
    return fragment


def _exact_mass(smiles: str) -> float | None:
    mol = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.CalcExactMolWt(mol) if mol else None


def read_and_build(notation: str, declared_mass: float | None = None,
                   context: str = "") -> dict:
    """Notation -> built SMILES with an audit-and-correct loop.

    Attempts: model proposal -> RDKit unit gate -> assembly -> exact-mass
    gate. Every rejection returns to the model as a specific correction.
    The final record carries the full attempt trail; gaps are named."""
    attempts: list[dict] = []
    feedback = ""
    last: dict[str, Any] = {}
    for n in range(1, MAX_ATTEMPTS + 1):
        try:
            parsed = _ask_model(notation, context, feedback)
        except ModelCallError as exc:
            # an API fault is not a chemistry rejection: record the real
            # cause and abort without burning correction rounds
            attempts.append({"attempt": n, "failure": str(exc)})
            return {"skill": SKILL, "smiles": None, "units": [],
                    "topology": {}, "attempts": attempts,
                    "gaps": [{"reason": "model call failed; correction loop "
                                       "aborted"}]}
        record: dict[str, Any] = {"attempt": n}
        if not parsed:
            record["failure"] = "model returned no parseable JSON"
            attempts.append(record)
            feedback = "Your answer was not valid JSON matching the schema. Output JSON only."
            continue
        units, failures = _validate_units(parsed)
        gaps: list[dict] = []
        tail_units = parsed.get("tail_units")
        if isinstance(tail_units, int) and tail_units > 0 and tail_units < len(units):
            # prompt contract: ring_size units close the macrocycle,
            # tail_units follow it linearly. The linear tail topology is
            # not assembled yet: build the ring, name the dropped tail.
            ring_units = parsed.get("ring_size")
            if not isinstance(ring_units, int) or not 0 < ring_units <= len(units):
                ring_units = len(units) - tail_units
                gaps.append({"reason": f"ring_size missing/out of range; "
                             f"derived as len(units)-tail_units={ring_units}"})
            parsed = dict(parsed)
            parsed["tail_units"] = 0
            parsed["ring_size"] = ring_units
            parsed["terminal"] = None
            dropped = units[ring_units:]
            units = units[:ring_units]
            gaps.append({"reason": "linear tail topology unresolved; "
                         "ring-only molecule assembled",
                         "tail_units": [{"code": u["code"]} for u in dropped]})
        if failures:
            record["failure"] = "unit definitions rejected"
            record["detail"] = failures
            attempts.append(record)
            feedback = "\n".join(f"- {f}" for f in failures)
            continue
        smiles, chemistry_error, notes = _assemble(units, parsed)
        if not smiles:
            record["failure"] = "assembly rejected by RDKit"
            record["detail"] = ([chemistry_error] if chemistry_error else []) + notes
            attempts.append(record)
            feedback = (f"Assembly failed RDKit chemistry: {chemistry_error}. "
                        "Each unit fragment must be the free amino acid "
                        "ending at its alpha carboxyl — no stray terminal "
                        "OH/NH2 that would bond in-chain.")
            continue
        gaps.extend({"reason": n} for n in notes)
        record["smiles"] = smiles[:120]
        if declared_mass:
            built = _exact_mass(smiles)
            delta = abs(built - declared_mass)
            record["mass"] = {"built": round(built, 2),
                              "declared": round(declared_mass, 2),
                              "delta": round(built - declared_mass, 2)}
            if delta > 0.8:
                record["failure"] = "exact mass mismatch"
                _lexicon_evict([u["code"] for u in units])
                attempts.append(record)
                feedback = (
                    f"Assembled exact mass {built:.2f} but the patent "
                    f"declares {declared_mass:.2f} (delta "
                    f"{built - declared_mass:+.2f} Da). Likely causes: a "
                    "missing/extra CH2 or N, an unstated side-chain bridge "
                    "(thioether loses 2 H), a wrong stereo-carbon count, or "
                    "a cap bonded at the wrong place. Re-derive the units.")
                continue
        record["accepted"] = True
        attempts.append(record)
        last = {"skill": SKILL, "smiles": smiles, "units": units,
                "topology": {"ring_size": parsed.get("ring_size"),
                             "tail_units": parsed.get("tail_units"),
                             "terminal": parsed.get("terminal"),
                             "bridges": parsed.get("bridges") or []},
                "gaps": gaps, "attempts": attempts}
        return last
    last = {"skill": SKILL, "smiles": None, "units": [],
            "topology": {}, "attempts": attempts,
            "gaps": [{"reason": "notation unresolved after corrections",
                      "attempts": len(attempts)}]}
    return last


def _lexicon_pin(key: str, entry: dict) -> None:
    """Cache a validated definition for future calls."""
    if not key:
        return
    with _LOCK:
        lex = _load()
        if key not in lex:
            lex[key] = entry
            _save(lex)


def _lexicon_evict(keys: list[str]) -> None:
    """Drop suspect definitions: a mass-gate failure implicates every
    cached entry the failed assembly used."""
    with _LOCK:
        lex = _load()
        removed = False
        for k in keys:
            if k in lex and "verified" not in str(lex[k].get("note", "")):
                del lex[k]
                removed = True
        if removed:
            _save(lex)


@_register_skill("peptide_notation", "model",
                    "cyclopeptide line notation -> built, mass-checked SMILES")
async def invoke(payload: dict) -> dict:
    """Atomic operation: notation (+ optional declared mass) -> SMILES."""
    notation = str(payload.get("notation", "")).strip()
    if not notation:
        # an empty notation would burn real LLM calls for nothing
        return {"skill": SKILL, "smiles": None, "units": [], "topology": {},
                "attempts": [],
                "gaps": [{"reason": "empty notation input"}]}
    return read_and_build(notation,
                          declared_mass=payload.get("declared_mass"),
                          context=payload.get("context", ""))
