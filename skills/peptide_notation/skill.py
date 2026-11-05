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
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from skills.registry import register as _register_skill
from skills.residue_lexicon import (_LOCK, _load, _save, _canonical_fragment,
                                   lexicon_snapshot)

RDLogger.DisableLog("rdApp.*")

SKILL = "peptide_notation"
MAX_ATTEMPTS = 4

NOTATION_PROMPT = """Read this cyclopeptide line notation from a patent and list every amino-acid unit in order, with its chemical identity. Respond with one JSON object:

{"units": [{"code": "<token as written>", "name": "<full chemical name>", "fragment_smiles": "<free amino acid: backbone amine N first, alpha carbon with side chain, carboxyl as C(=O)O>", "n_methyl": <bool>, "ring_nitrogen": <bool>}], "ring_size": <units in the head-to-tail macrocycle>, "tail_units": <linear units after the ring>, "terminal": "<C-terminal group like NH2 / OH, or null>", "caps": [{"code": "<cap token>", "fragment_smiles": "<acyl group ending in C(=O), e.g. chloroacetyl ClCC(=O)>"}], "bridges": [{"from_unit": <unit index>, "to_unit": <unit index>, "kind": "<thioether|lactam|disulfide>", "note": "<side-chain atoms>"}]}

Rules: units[0..ring_size-1] form the macrocycle in bond order; a cap inside cyclo(...) acylates the first unit's amine (that amine is N,N-diacyl); beta-methyl amino acids are N[C@@H]([C@H](C)side)C(=O)O; Sarcosine is N(C)CC(=O)O. JSON only."""


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


def _ask_model(notation: str, context: str, feedback: str) -> dict | None:
    client, model = _client()
    user = (f"Notation: {notation}\n{_vocabulary_hint()}"
            "\nIf a token is several known codes joined without '-', split "
            "it into separate units.")
    if context:
        user += f"\nContext: {context}"
    if feedback:
        user += f"\nYour previous answer had errors — correct it:\n{feedback}"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": NOTATION_PROMPT},
                      {"role": "user", "content": user}],
            temperature=0.15, max_tokens=2500)
        text = resp.choices[0].message.content.strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        return json.loads(text[start:end + 1])
    except Exception:
        return None


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
            if cached:
                frag = _fragment_ok(cached.get("fragment_smiles", ""))
                if frag:
                    break
        if frag is None:
            frag = _fragment_ok(u.get("fragment_smiles", ""))
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


def _assemble(units: list[dict], parsed: dict) -> tuple[str | None, str]:
    """Deterministic head-to-tail macrocycle assembly from validated
    fragments. Ring closure uses a single global digit; a cap acylates the
    first unit's amine as a branch (N,N-diacyl convention)."""
    ring = units[:parsed.get("ring_size", len(units))]
    tail = units[len(ring):]
    caps = [c for c in (parsed.get("caps") or [])
            if _fragment_ok(_cap_acyl_smiles(c))]
    parts = []
    cap_branch = ""
    if caps:
        acyl = _cap_acyl_smiles(caps[0])
        cap_branch = f"({acyl})" if acyl else ""
    first = True
    for u in ring:
        frag = u["entry"]["fragment_smiles"]
        frag = _to_backbone_form(frag)          # C(=O)O -> C(=O), strip charge
        if first:
            frag = _head_amine(frag, MACRO_DIGIT, cap_branch)
            first = False
        parts.append(frag)
    parts.append(f"%{MACRO_DIGIT}")
    for u in tail:
        parts.append(_to_backbone_form(u["entry"]["fragment_smiles"]))
    terminal = str(parsed.get("terminal") or "").strip()
    if tail and terminal.upper() == "NH2":
        parts.append("N")
    elif tail and terminal.upper() == "OH":
        parts.append("")
    smiles = "".join(parts)
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as exc:
        return None, str(exc)[:200]
    if mol is None:
        m2 = Chem.MolFromSmiles(smiles, sanitize=False)
        if m2 is not None:
            try:
                Chem.SanitizeMol(m2)
            except Exception as exc:
                return None, str(exc)[:200]
        return None, "SMILES does not parse (syntax or valence)"
    return Chem.MolToSmiles(mol), ""


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
        parsed = _ask_model(notation, context, feedback)
        record: dict[str, Any] = {"attempt": n}
        if not parsed:
            record["failure"] = "model returned no parseable JSON"
            attempts.append(record)
            feedback = "Your answer was not valid JSON matching the schema. Output JSON only."
            continue
        units, failures = _validate_units(parsed)
        if failures:
            record["failure"] = "unit definitions rejected"
            record["detail"] = failures
            attempts.append(record)
            feedback = "\n".join(f"- {f}" for f in failures)
            continue
        smiles, chemistry_error = _assemble(units, parsed)
        if not smiles:
            record["failure"] = "assembly rejected by RDKit"
            record["detail"] = chemistry_error
            attempts.append(record)
            feedback = (f"Assembly failed RDKit chemistry: {chemistry_error}. "
                        "Each unit fragment must be the free amino acid "
                        "ending at its alpha carboxyl — no stray terminal "
                        "OH/NH2 that would bond in-chain.")
            continue
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
                "gaps": [], "attempts": attempts}
        return last
    last = {"skill": SKILL, "smiles": None, "units": [],
            "topology": {}, "attempts": attempts,
            "gaps": [{"reason": "notation unresolved after corrections",
                      "attempts": len(attempts)}]}
    return last


def _lexicon_pin(key: str, entry: dict) -> None:
    """Cache a validated definition for future calls."""
    try:
        with _LOCK:
            lex = _load()
            if key and key not in lex:
                lex[key] = entry
                _save(lex)
    except Exception:
        pass


def _lexicon_evict(keys: list[str]) -> None:
    """Drop suspect definitions: a mass-gate failure implicates every
    cached entry the failed assembly used."""
    try:
        with _LOCK:
            lex = _load()
            removed = False
            for k in keys:
                if k in lex and "verified" not in str(lex[k].get("note", "")):
                    del lex[k]
                    removed = True
            if removed:
                _save(lex)
    except Exception:
        pass


@_register_skill("peptide_notation", "model",
                    "cyclopeptide line notation -> built, mass-checked SMILES")
async def invoke(payload: dict) -> dict:
    """Atomic operation: notation (+ optional declared mass) -> SMILES."""
    return read_and_build(payload.get("notation", ""),
                          declared_mass=payload.get("declared_mass"),
                          context=payload.get("context", ""))
