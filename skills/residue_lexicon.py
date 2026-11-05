"""Skill storage: residue lexicon. LLM-normalized definitions for chemical residue names
and patent monomer codes, validated by RDKit and cached on disk.

Role split:
  - the language model reads a residue NAME (ST.25/ST.26 SITE note) or a
    patent monomer CODE and emits a canonical JSON definition;
  - RDKit is the gate: the fragment must parse and satisfy the backbone
    contract (amine-first, carboxyl-last) before it enters the lexicon;
  - validated entries persist in data/residue_lexicon.json, so each name
    costs one LLM call ever.

The lexicon is the vocabulary for both directions of the cyclopeptide
harness: sequence -> SMILES assembly and SMILES -> sequence typing.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

LEXICON_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "residue_lexicon.json")

SYSTEM_PROMPT = (
    "You are a peptide chemistry dictionary. Given a chemical residue name "
    "or patent monomer code, output ONE JSON object:\n"
    '{"fragment_smiles": "<SMILES of the free amino acid as backbone-N first, '
    "backbone-carboxyl last; ring amino acids start at the ring N>\", "
    '"n_methyl": <true if the backbone amine is N-methylated>, '
    '"ring_nitrogen": <true if the backbone amine belongs to a ring>, '
    '"note": "<one-line justification>"}\n'
    "Rules: standard alpha-amino acids use the L form (N[C@@H](side)C(=O)); "
    "beta-methyl acids are N[C@@H]([C@H](C)side)C(=O); N-methyl acids write "
    "the amine as N(C); no hydrogen counts on the backbone N. Output JSON "
    "only, no prose."
)

_LOCK = threading.Lock()


def _load() -> dict[str, Any]:
    if os.path.exists(LEXICON_PATH):
        try:
            with open(LEXICON_PATH, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save(lexicon: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(LEXICON_PATH), exist_ok=True)
    with open(LEXICON_PATH, "w", encoding="utf-8") as fh:
        json.dump(lexicon, fh, ensure_ascii=False, indent=1, sort_keys=True)


def _client():
    from constants import (LLM_OPENAI_COMPATIBLE_MODEL_KEY,
                           LLM_OPENAI_COMPATIBLE_MODEL_NAME,
                           LLM_OPENAI_COMPATIBLE_MODEL_URL)
    from openai import OpenAI
    return OpenAI(api_key=LLM_OPENAI_COMPATIBLE_MODEL_KEY,
                  base_url=LLM_OPENAI_COMPATIBLE_MODEL_URL), \
        LLM_OPENAI_COMPATIBLE_MODEL_NAME


def _ask(spec: str, context: str) -> dict | None:
    """One LLM call -> parsed JSON definition, or None."""
    client, model = _client()
    user = f"Residue: {spec}" + (f"\nContext: {context}" if context else "")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user}],
            temperature=0.1, max_tokens=600)
        text = resp.choices[0].message.content.strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def _is_carboxyl(carbon) -> bool:
    """Carbon bearing a C=O and an -O(H) — a carboxylic acid group."""
    dbl = oh = False
    for b in carbon.GetBonds():
        x = b.GetOtherAtomIdx(carbon.GetIdx())
        atom = carbon.GetOwningMol().GetAtomWithIdx(x)
        if atom.GetSymbol() != "O":
            continue
        if b.GetBondTypeAsDouble() == 2:
            dbl = True
        elif atom.GetDegree() == 1:
            oh = True
    return dbl and oh


def _canonical_fragment(mol) -> str | None:
    """Normalize any valid free-amino-acid form to the backbone contract:
    deprotonate/ionize-free the alpha carboxyl, root the SMILES at the
    backbone amine. Returns None when the skeleton is not an amino acid."""
    mol = Chem.Mol(mol)
    # neutralize [O-] to OH so the carboxyl scan sees it
    for a in mol.GetAtoms():
        if a.GetFormalCharge() == -1 and a.GetSymbol() == "O":
            a.SetFormalCharge(0)
            a.SetNumExplicitHs(1)
    # nearest alpha carboxyl within 3 bonds of the amine (2 for normal
    # acids, 3 for ring-amine acids like ACPC where the amine sits in the
    # ring one carbon away from the carboxyl-bearing carbon)
    alpha_n = alpha_carbonyl = None
    amine = next((a for a in mol.GetAtoms() if a.GetSymbol() == "N"), None)
    if amine is None:
        return None
    from collections import deque
    seen = {amine.GetIdx()}
    order = deque([(amine, 0)])
    while order and alpha_carbonyl is None:
        node, depth = order.popleft()
        if depth > 3:
            continue
        for nb in node.GetNeighbors():
            if nb.GetIdx() in seen:
                continue
            seen.add(nb.GetIdx())
            if (nb.GetSymbol() == "C" and depth >= 1
                    and _is_carboxyl(nb)):
                alpha_n, alpha_carbonyl = amine, nb
                break
            if depth + 1 <= 3:
                order.append((nb, depth + 1))
        if alpha_carbonyl is not None:
            break
    if alpha_carbonyl is None:
        return None
    dbl_o = next(b.GetOtherAtomIdx(alpha_carbonyl.GetIdx())
                 for b in alpha_carbonyl.GetBonds()
                 if b.GetBondTypeAsDouble() == 2
                 and mol.GetAtomWithIdx(
                     b.GetOtherAtomIdx(alpha_carbonyl.GetIdx())).GetSymbol() == "O")
    em = Chem.RWMol(mol)
    em.GetAtomWithIdx(alpha_n.GetIdx()).SetProp("_backbone_amine", "1")
    # drop exactly the alpha carboxyl's hydroxyl (side-chain acids survive)
    oh = None
    for nb in alpha_carbonyl.GetNeighbors():
        if (nb.GetSymbol() == "O" and nb.GetIdx() != dbl_o
                and nb.GetDegree() == 1):
            oh = nb.GetIdx()
            break
    if oh is not None:
        em.RemoveAtom(oh)
    out = em.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    roots = [a.GetIdx() for a in out.GetAtoms()
             if a.HasProp("_backbone_amine")]
    if not roots:
        return None
    return Chem.MolToSmiles(out, rootedAtAtom=roots[0])


def _validate(defn: dict | None) -> dict | None:
    """RDKit gate: fragment parses, is an amino-acid skeleton, and is
    canonicalized to amine-first / carboxyl-last form."""
    if not defn:
        return None
    mol = Chem.MolFromSmiles(defn.get("fragment_smiles", ""))
    if mol is None or mol.GetNumAtoms() < 2:
        return None
    smiles = _canonical_fragment(mol)
    if smiles is None:
        return None
    return {
        "fragment_smiles": smiles,
        "n_methyl": bool(defn.get("n_methyl", False)),
        "ring_nitrogen": bool(defn.get("ring_nitrogen", False)),
        "note": str(defn.get("note", ""))[:200],
    }


def normalize_spec(spec: str) -> str:
    """Strip OCR column noise (trailing digits, spacing) from a name."""
    spec = re.sub(r"\s+", " ", str(spec or "")).strip()
    spec = re.sub(r"\s+\d{1,2}$", "", spec)
    return spec.strip()


def define(spec: str, context: str = "", use_cache: bool = True) -> dict | None:
    """Residue name/code -> validated definition (None when unresolvable).
    A failed definition retries once with a corrective hint."""
    spec = normalize_spec(spec)
    if not spec:
        return None
    with _LOCK:
        lexicon = _load() if use_cache else {}
        if spec in lexicon:
            return lexicon[spec]
    defn = _validate(_ask(spec, context))
    if defn is None:
        hint = ("Previous attempt was rejected. The fragment must contain a "
                "backbone amine N bonded to the alpha carbon of a carboxylic "
                "acid. For N-methyl acids write the amine as N(C); for ring "
                "acids the amine is a ring atom. A beta-methyl acid like "
                "'(2S,3S)-2-Amino-3-(1H-indol-3-yl)butanoic acid' is "
                "N[C@@H]([C@H](C)c1c[nH]c2ccccc12)C(=O)O.")
        defn = _validate(_ask(spec, f"{context}\n{hint}" if context else hint))
    if defn is not None:
        with _LOCK:
            lexicon = _load()
            lexicon[spec] = defn
            _save(lexicon)
    return defn


def define_batch(specs: list[str], context: str = "",
                 progress=None) -> dict[str, dict]:
    """Resolve many specs; cached ones cost nothing, misses cost one call."""
    out: dict[str, dict] = {}
    with _LOCK:
        lexicon = _load()
    for i, spec in enumerate(specs):
        spec = normalize_spec(spec)
        if not spec:
            continue
        if spec in lexicon:
            out[spec] = lexicon[spec]
            continue
        defn = _validate(_ask(spec, context))
        if defn is not None:
            out[spec] = defn
            with _LOCK:
                lexicon[spec] = defn
                _save(lexicon)
        if progress:
            progress(i + 1, len(specs), spec, defn is not None)
    return out


def lexicon_snapshot() -> dict[str, Any]:
    with _LOCK:
        return _load()
