"""Document-scoped peptide notation -> SMILES resolution.

Patents define their own monomer codes; the same code means different
skeletons in different documents. Resolution rungs, cheapest evidence
first, every acceptance decided by the fragment contract + assembly +
declared mass (mono or average convention):

  lexicon -> this document's verified memory -> built-in merged dictionary
  (data/monomer_dictionary.json, offline) -> RCSB/PubChem lookups
  (opt-in, rate-limited) -> document table readings (names via OPSIN,
  structures via VLM transcription) -> peptide_notation model loop.

Unverified candidates never mutate state; failures are recorded as named
gaps. Mass cannot arbitrate L/D stereo or asymmetric-cap attachment —
records note which convention held.
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):      # CLI entry: run from any cwd
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from skills.peptide_assembly import (MASS_TOLERANCE, _assemble, exact_mass,
                                     normalize_token, resolve_token,
                                     split_merged, split_notation)
from skills.residue_lexicon import _LOCK, _load as _lexicon_load
from skills.residue_lexicon import _save as _lexicon_save
from utils.peptide_fragment_contract import (backbone_fragment,
                                             cap_fragment)

RUNTIME_DIR = Path(__file__).resolve().parent.parent / "data" / "runtime"
BUILTIN_DICTIONARY_PATH = RUNTIME_DIR.parent / "monomer_dictionary.json"
CCD_CACHE_PATH = RUNTIME_DIR / "rcsb_chemcomp_candidates.json"
CCD_DATA_URL = "https://data.rcsb.org/rest/v1/core/chemcomp"
# Polite pacing for the RCSB Data API: the service publishes no hard
# per-client rate limit today (batch endpoints cap at 1000 IDs) but asks
# for fair use; a small fixed interval stays well inside it.
CCD_MIN_INTERVAL = 0.15
CCD_CACHE_TTL = 30 * 86400
# A catalog returning more than this many distinct readings for one code is
# noise; the curated descriptors come first anyway.
MAX_CANDIDATES_PER_TOKEN = 4

_LEXICON_PIN_LOCK = threading.Lock()

# Evidence routing: cost-ordered rungs per evidence need. The router
# escalates only when a rung yields nothing usable; every escalation and
# its outcome lands in the row audit. Cost is relative (log-scale intent:
# free local lookups, local CPU tools, network calls, GPU inference,
# model loops).
RUNG_COST = {"lexicon": 0, "memory": 0, "builtin": 1, "grammar": 1,
             "opsin": 2, "ccd": 3, "pubchem": 4, "ocsr": 5, "vision": 6,
             "llm": 8}


@dataclass
class EvidencePolicy:
    """Which capabilities exist and in what order they are consulted.

    Defaults express the production hierarchy: persisted knowledge first
    (lexicon, document memory, built-in dictionary), then local tools
    (OPSIN, grammar repairs), then catalogs (network, opt-in), then
    image/model inference (OCSR for drawn cells, VLM for page-level
    transcription), and the model loop last. Each flag turns a rung off
    globally (offline mode, budget caps); a rung with no evidence to read
    (e.g. vision without located structure cells) is skipped per row, not
    per policy.
    """
    lexicon: bool = True
    memory: bool = True
    builtin: bool = True
    opsin: bool = True
    ccd: bool = True
    pubchem: bool = True
    ocsr: bool = True
    vision: bool = True
    llm: bool = True

    def token_rungs(self) -> list[str]:
        """Ordered rungs for resolving one printed code."""
        rungs = []
        if self.memory:
            rungs.append("memory")
        if self.lexicon:
            rungs.append("lexicon")
        if self.builtin:
            rungs.append("builtin")
        rungs += ["opsin", "ccd", "pubchem"]
        return [r for r in rungs if getattr(self, r, True)]

    def structure_rungs(self) -> list[str]:
        """Ordered rungs for reading one drawn structure cell."""
        rungs = []
        if self.ocsr:
            rungs.append("ocsr")
        if self.vision:
            rungs.append("vision")
        return rungs

    def notation_rungs(self) -> list[str]:
        """Ordered rungs for a whole notation that deterministic assembly
        could not verify. The model loop is the final rung."""
        return ["llm"] if self.llm else []

    @classmethod
    def offline(cls) -> "EvidencePolicy":
        return cls(ccd=False, pubchem=False, llm=False)

# C-terminal markers printed by the notation grammar itself (amide/hydroxyl
# end groups, the same role cyclo() plays at the N side). Not a code
# vocabulary: these are suffix markers, like the ring wrapper.
_TERMINAL_MARKERS = {"NH2": "NH2", "AM": "NH2", "NHEt": None, "OH": "OH",
                     "OL": "OH"}
_C_TERMINAL_RE = re.compile(r"-\s*(NH2|am|OH|ol)\s*$", re.IGNORECASE)
# the printed sequence annotation is grammar, not chemistry
_SEQ_ID_SUFFIX_RE = re.compile(r"\s*\(\s*SEQ ID NO:\s*\d+\s*\)\s*$",
                               re.IGNORECASE)
_SEQ_RE = re.compile(
    r"^\s*[A-Za-z][A-Za-z0-9_\-,\(\)\+ ]*\(SEQ ID NO:\s*(\d+)\)\s*$")
_MW_RE = re.compile(r"\b([DM])\s*;\s*([0-9]+\.[0-9]+)")
_BARE_MASS_RE = re.compile(r"\b([0-9]{3,6}\.[0-9]{2,4})\b")

STD_LETTERS = "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------------------
# candidate readings
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """One proposed fragment for one token, with provenance. Candidates are
    inert until arbitration verifies them; a candidate from a source the
    document never sanctioned (a catalog lookup) can still be rejected by
    the declared mass without any side effect."""
    fragment_smiles: str
    source: str                  # 'lexicon'|'document'|'ccd'|'pubchem'|'opsin'
    provenance: dict = field(default_factory=dict)
    role: str = "residue"        # 'residue' | 'cap'


def _memory_candidates(code: str, memory: dict) -> list[Candidate]:
    """L1: fragments this document already verified. Code looked up in its
    printed form first, then casefolded (OCR case is not chemistry)."""
    out = []
    for key in (code, code.casefold()):
        entry = memory.get(key)
        if entry and entry.get("fragment_smiles"):
            out.append(Candidate(entry["fragment_smiles"], "document",
                                 {"verified_rows": entry.get("verified_rows", 0)}))
            break
    return out


def _lexicon_candidates(code: str, flags: dict) -> list[Candidate]:
    """L0: the shared lexicon via the flag-aware resolver."""
    frag, key = resolve_token(code, flags)
    if frag:
        return [Candidate(frag, "lexicon", {"lexicon_key": key})]
    return []


def _ccd_candidates(code: str) -> list[Candidate]:
    """L2a: RCSB chem_comp records matching the printed code, verbatim.
    Lookups go through a small disk cache so one code costs one request
    per month; the returned SMILES are candidates — arbitration decides."""
    fold = code.strip()
    if not fold or len(fold) > 6 or not re.fullmatch(r"[0-9A-Za-z]{1,6}", fold):
        return []
    cache = _load_ccd_cache()
    hit = cache.get(fold)
    if hit and time.time() - hit.get("fetched_at", 0) < CCD_CACHE_TTL:
        smiles_list = hit.get("smiles") or []
    else:
        smiles_list = _fetch_ccd_smiles(fold)
        cache[fold] = {"smiles": smiles_list, "fetched_at": time.time()}
        _save_ccd_cache(cache)
    out = []
    for smiles in smiles_list[:MAX_CANDIDATES_PER_TOKEN]:
        if smiles and Chem.MolFromSmiles(smiles) is not None:
            out.append(Candidate(smiles, "ccd", {"ccd_code": fold}))
    return out


def _load_ccd_cache() -> dict:
    try:
        return json.loads(CCD_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_ccd_cache(cache: dict) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CCD_CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=1, sort_keys=True),
                   encoding="utf-8")
    tmp.replace(CCD_CACHE_PATH)


_CCD_FETCH_LOCK = threading.Lock()
_CCD_LAST_FETCH = [0.0]


def _fetch_ccd_smiles(code: str) -> list[str]:
    """One paced chem_comp lookup. Network faults yield an empty candidate
    list; the audit names the outage — a network error never promotes or
    rejects anything on its own."""
    import urllib.error
    import urllib.parse
    import urllib.request
    with _CCD_FETCH_LOCK:
        wait = CCD_MIN_INTERVAL - (time.monotonic() - _CCD_LAST_FETCH[0])
        if wait > 0:
            time.sleep(wait)
        _CCD_LAST_FETCH[0] = time.monotonic()
        url = f"{CCD_DATA_URL}/{urllib.parse.quote(code)}"
        request = urllib.request.Request(
            url, headers={"Accept": "application/json",
                          "User-Agent": "BioChemInsight peptide harness "
                                        "(RCSB chem_comp candidate lookup)"})
        try:
            # a bounded timeout keeps an unreachable catalog from stalling
            # the ladder; failures cache as empty and are named in the audit
            with urllib.request.urlopen(request, timeout=8) as response:
                payload = json.loads(response.read(2_000_000).decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return []
    return _ccd_smiles_from_record(payload)


def _ccd_smiles_from_record(payload: dict) -> list[str]:
    """Curated SMILES descriptors from a chem_comp record, most specific
    first: SMILES_stereo carries the isomeric form, SMILES the connectivity
    form (the endpoint publishes these two kinds)."""
    descriptors = payload.get("pdbx_chem_comp_descriptor") or []
    stereo, plain = [], []
    for item in descriptors:
        text = str(item.get("descriptor", "")).strip()
        kind = str(item.get("type", ""))
        if not text:
            continue
        if kind == "SMILES_stereo" and text not in stereo:
            stereo.append(text)
        elif kind == "SMILES" and text not in plain:
            plain.append(text)
    return (stereo + plain)[:MAX_CANDIDATES_PER_TOKEN]


def load_builtin_dictionary(path: str | None = None,
                            ) -> dict[str, list[dict]]:
    """The built-in merged vocabulary (skills/build_monomer_dictionary.py
    output) indexed by every alias its sources printed for an entry.
    Entry shape: {"backbone", "cap", "name", "sources"}; either form may
    be null. Offline by construction."""
    path = path or str(BUILTIN_DICTIONARY_PATH)
    try:
        entries = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    index: dict[str, list[dict]] = {}
    for entry in entries:
        for code in entry.get("codes", []):
            for fold in {code.casefold(),
                         re.sub(r"[\s_]+", "", code).casefold()}:
                if not fold:
                    continue
                slot = index.setdefault(fold, [])
                if entry not in slot:
                    slot.append(entry)
    return index


def load_monomer_dictionary(sdf_path: str,
                            code_props=("m_abbr", "symbol", "abbr", "code"),
                            name_prop="m_name") -> dict[str, list[dict]]:
    """Ad-hoc monomer SDFs -> the same index shape as the built-in
    dictionary. RDKit reads the file; the originating package is not a
    runtime dependency."""
    from rdkit import Chem
    dictionary: dict[str, list[dict]] = {}
    for mol in Chem.SDMolSupplier(str(sdf_path), removeHs=True):
        if mol is None:
            continue
        props = mol.GetPropsAsDict()
        smiles = Chem.MolToSmiles(mol)
        name = str(props.get(name_prop, "")).strip()
        codes = [str(props.get(prop, "")).strip() for prop in code_props]
        entry = {"backbone": backbone_fragment(smiles),
                 "cap": cap_fragment(smiles), "name": name,
                 "sources": ["sdf"]}
        for code in codes:
            if not code or code.lower() == "none":
                continue
            fold = re.sub(r"[\s_]+", "", code).casefold()
            if not fold:
                continue
            slot = dictionary.setdefault(fold, [])
            if entry not in slot:
                slot.append(entry)
    return dictionary


def _name_candidates(code: str, names: list[str]) -> list[Candidate]:
    """L2b/L3: chemical NAMES (document abbreviation-table readings, or a
    catalog name search) -> fragment candidates via OPSIN (local) and
    PubChem (shared rate-limited client). Table cells print alternatives
    as 'X or Y' — each alternative is its own candidate."""
    out = []
    seen_names = set()
    for name in names:
        for alt in re.split(r"\s+or\s+", name):
            alt = alt.strip()
            if alt and alt.casefold() not in seen_names:
                seen_names.add(alt.casefold())
                out.extend(_single_name_candidates(alt))
    return out[:MAX_CANDIDATES_PER_TOKEN]


def _single_name_candidates(name: str) -> list[Candidate]:
    out = []
    try:
        from utils.opsin_local import name_to_smiles
        smiles = (name_to_smiles(name) or "").strip()
    except Exception:
        smiles = ""
    if smiles and Chem.MolFromSmiles(smiles) is not None:
        out.append(Candidate(smiles, "opsin", {"name": name}))
    try:
        from utils.pubchem_client import pubchem_lookup_name_details
        detail = pubchem_lookup_name_details(name)
        record = detail.get("record") or {}
        if detail.get("status") == "found" \
                and record.get("isomeric_smiles"):
            out.append(Candidate(record["isomeric_smiles"], "pubchem",
                                 {"name": name, "cid": record.get("cid")}))
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# row routing (structure only — no acceptance decisions here)
# ---------------------------------------------------------------------------

@dataclass
class NotationRow:
    """One compound row: printed notation, declared form/mass, and the
    cyclization/dimerization annotations."""
    notation: str
    seq_id: str = ""
    form: str = ""               # 'M' | 'D' | '' as printed
    declared_mass: float | None = None
    cyclization: dict = field(default_factory=dict)
    page: Any = None
    smiles_reading: str = ""     # candidate reading; arbitration decides
    source: str = ""


def parse_notation(notation: str) -> dict:
    """Notation text -> tokens + terminal marker. Head-cap detection is
    chemistry-driven: whether the first token is a residue or an N-cap is
    decided later from which contract its candidates satisfy, never from a
    spelling whitelist. Both printed grammars are honoured: cyclo()-wrapped
    rings and bare hyphen-joined linear strings."""
    body = _SEQ_ID_SUFFIX_RE.sub("", notation.strip())
    body = re.sub(r"(?<=[A-Za-z0-9)\]])\s+(?=[A-Za-z0-9\[(])", "-", body)
    linear = not re.match(r"^\s*cyclo\s*\(", body, re.IGNORECASE)
    terminal = ""
    match = _C_TERMINAL_RE.search(body)
    if match:
        terminal = _TERMINAL_MARKERS.get(match.group(1).upper()) or ""
        if not terminal:
            # a printed marker outside the amide/hydroxyl pair is data, not
            # noise: keep the row intact and name it
            return {"tokens": _split_tokens(body, linear), "terminal": "",
                    "linear": linear, "unparsed_marker": match.group(1)}
        body = body[:match.start()]
    return {"tokens": _split_tokens(body, linear), "terminal": terminal,
            "linear": linear}


def _split_tokens(body: str, linear: bool) -> list[str]:
    """Dash/paren-aware split (cyclo-wrapper aware) + OCR-fused token
    splitting via the lexicon cover (skills/peptide_assembly.py)."""
    tokens, _tail = split_notation(body, linear_notation=linear)
    return [tok.strip() for chunk in tokens
            for tok in split_merged(chunk) if tok.strip()]


def mine_rows_from_tables(page_tables) -> dict:
    """OCR table grids -> NotationRow slots keyed by printed compound id.

    Routing only: rows group by content signature (attribute / sequence /
    SMILES reading / unclassified). SMILES readings are stored as candidate
    strings — a damaged reading fails arbitration downstream and is
    reported with its reason, never pattern-filtered away here.
    Unclassified rows are retained for the audit to name."""
    compounds: dict[str, dict] = {}
    order: list[str] = []
    unroutable: list[dict] = []
    current = None
    for page_no, table in page_tables:
        for row in table:
            joined = " ".join(row[1:]).strip()
            cid_m = re.match(r"^(\d{1,4})$", row[0].strip()) if row else None
            kind = _route_row(row, joined)
            if kind == "attr":
                mw_m = _MW_RE.search(joined)
                if cid_m:
                    current = cid_m.group(1)
                    slot = compounds.setdefault(current, {})
                    if current not in order:
                        order.append(current)
                    mass = float(mw_m.group(2)) if mw_m else None
                    if mass is None:
                        bare = _BARE_MASS_RE.search(joined)
                        mass = float(bare.group(1)) if bare else None
                    slot["attr"] = {
                        "form": mw_m.group(1) if mw_m else "",
                        "declared_mass": mass,
                        "cyclization": {
                            "positions": row[2] if len(row) > 2 else "",
                            "bond": row[3] if len(row) > 3 else "",
                            "linker": row[4] if len(row) > 4 else "",
                            "dimerization": row[5] if len(row) > 5 else "",
                        },
                        "stability": row[6] if len(row) > 6 else "",
                        "ic50": row[7] if len(row) > 7 else "",
                        "page": page_no,
                    }
            elif kind == "seq":
                m = _SEQ_RE.match(joined)
                if m:
                    if cid_m:
                        current = cid_m.group(1)
                    key = current or "?"
                    slot = compounds.setdefault(key, {})
                    if key not in order:
                        order.append(key)
                    slot["notation"] = m.group(0).strip()
                    slot["seq_id"] = m.group(1)
            elif kind == "smiles":
                if cid_m:
                    current = cid_m.group(1)
                key = current or "?"
                slot = compounds.setdefault(key, {})
                frag = re.sub(r"\s+", "", joined)
                acc = slot.get("smiles_reading", "")
                if frag and frag not in acc:   # page-break duplicate guard
                    slot["smiles_reading"] = acc + frag
            elif joined:
                unroutable.append({"page": page_no, "cells": row[:6]})
    rows = []
    for cid in order:
        slot = compounds[cid]
        if "notation" not in slot:
            continue
        attr = slot.get("attr") or {}
        rows.append(NotationRow(
            notation=slot["notation"], seq_id=slot.get("seq_id", ""),
            form=attr.get("form", ""),
            declared_mass=attr.get("declared_mass"),
            cyclization={**attr.get("cyclization", {}),
                         "stability": attr.get("stability", ""),
                         "ic50": attr.get("ic50", "")},
            page=attr.get("page"),
            smiles_reading=slot.get("smiles_reading", ""),
            source=f"table:compound-{cid}"))
    return {"rows": rows, "unroutable": unroutable}


def _route_row(row: list[str], joined: str) -> str:
    if _SEQ_RE.match(joined):
        return "seq"
    flat = re.sub(r"\s+", "", joined)
    if len(flat) >= 15 and Chem.MolFromSmiles(flat) is not None:
        return "smiles"          # RDKit decides, not a character filter
    if _MW_RE.search(joined) or _BARE_MASS_RE.search(joined):
        return "attr"
    return "unclassified"


# ---------------------------------------------------------------------------
# arbitration
# ---------------------------------------------------------------------------

def _contract_fragment(candidate: Candidate) -> str | None:
    """The candidate's fragment rewritten into the form the assembler
    consumes: backbone contract for residues, acyl contract for caps.
    Returns None when the reading violates its contract."""
    if candidate.role == "cap":
        return cap_fragment(candidate.fragment_smiles)
    return backbone_fragment(candidate.fragment_smiles)


def _declared_mass_matches(smiles: str, declared: float) -> tuple[
        bool, float, str]:
    """Patent MW columns come in both monoisotopic and average
    conventions (this one is average); a reading matches when EITHER
    convention agrees within tolerance. Returns (ok, delta, convention)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, None, ""
    exact = exact_mass(smiles)
    if abs(exact - declared) <= MASS_TOLERANCE:
        return True, round(exact - declared, 4), "mono"
    from rdkit.Chem import Descriptors
    avg = Descriptors.MolWt(mol)
    if abs(avg - declared) <= MASS_TOLERANCE:
        return True, round(avg - declared, 4), "avg"
    delta = round(exact - declared, 4)
    return False, delta, f"exact {exact:.4f} / avg {avg:.4f} vs declared"


def _verify_assembly(tokens: list[tuple[str, dict]], caps: list[str],
                     declared_mass: float | None,
                     fragment_of: dict[str, str],
                     terminal: str = "", linear_chain: bool = False) -> dict:
    units = [{"code": code, "entry": {"fragment_smiles": fragment_of[code]}}
             for code, _ in tokens]
    trial = {"ok": False, "smiles": None, "delta": None,
             "declared_mass": declared_mass, "failure": None}
    smiles, error, _notes = _assemble(
        units, {"ring_size": len(units), "linear_chain": linear_chain,
                "caps": [{"fragment_smiles": c} for c in caps],
                "terminal": terminal})
    if not smiles:
        trial["failure"] = f"assembly rejected: {error}"
        return trial
    if declared_mass is None:
        # provisionally accept; the caller arbitrates via the row's own
        # printed SMILES reading (canonical match), else names the failure
        trial["ok"] = True
        trial["smiles"] = smiles
        trial["mass_convention"] = "reading-match"
        return trial
    ok, delta, convention = _declared_mass_matches(smiles, declared_mass)
    trial["delta"] = delta
    trial["mass_convention"] = convention
    if ok:
        trial["ok"] = True
        trial["smiles"] = smiles
    else:
        trial["failure"] = (f"mass mismatch ({convention})")
    return trial


# ---------------------------------------------------------------------------
# the harness
# ---------------------------------------------------------------------------

class PeptideResolutionHarness:
    """Per-document resolution state.

        harness = PeptideResolutionHarness(doc_key="WO2025...")
        record = harness.resolve_row(row)
        summary = harness.bootstrap(rows)

    Every outcome carries its evidence: verified rows carry the assembled
    SMILES, the mass delta and per-token provenance; rows that could not
    verify carry every named reason in order — which rungs were consulted
    and why each candidate lost.
    """

    def __init__(self, doc_key: str, style_opts: dict | None = None,
                 use_ccd: bool = True, use_name_lookups: bool = True,
                 monomer_dictionary: dict | None = None,
                 policy: EvidencePolicy | None = None):
        self.doc_key = doc_key
        # caller opts merge over the linear-notation defaults; an explicit
        # empty dict is meaningful and must not fall back to defaults
        self.opts = {"d_prefix": True, **(style_opts or {})}
        if policy is not None:
            self.policy = policy
        else:
            self.policy = EvidencePolicy(ccd=use_ccd,
                                         pubchem=use_ccd and use_name_lookups,
                                         opsin=use_name_lookups)
        self.use_ccd = use_ccd
        self.use_name_lookups = use_name_lookups
        # L2: the built-in merged vocabulary, offline by construction.
        # Entries are candidates; the same code means different molecules
        # in different catalogs, so arbitration decides.
        self.monomer_dictionary: dict[str, list[dict]] = \
            monomer_dictionary if monomer_dictionary is not None \
            else load_builtin_dictionary()
        # document memory: printed code -> verified fragment entry, plus
        # 'names:<code>' / 'struct:<code>' slots holding this document's own
        # table readings for the later rungs
        self.memory: dict[str, dict] = {}
        self.quarantine: list[dict] = []
        self.audit: list[dict] = []

    # -- one row ----------------------------------------------------------

    def resolve_row(self, row: NotationRow) -> dict:
        record: dict[str, Any] = {
            "doc_key": self.doc_key, "seq_id": row.seq_id,
            "notation": row.notation, "form": row.form,
            "declared_mass": row.declared_mass, "ok": False,
            "deferred": False, "smiles": None, "delta": None,
            "tokens": [], "unknown": [], "gaps": [], "verified": [],
        }
        reading_mol = None
        if row.smiles_reading:
            reading_mol = Chem.MolFromSmiles(row.smiles_reading)
            if reading_mol is None:
                record["gaps"].append({"reason": "row's printed SMILES "
                                       "reading does not parse; kept as "
                                       "named gap, not used"})
        if row.declared_mass is None and reading_mol is None:
            record["gaps"].append({"reason": "row has neither a declared "
                                   "mass nor a parseable SMILES reading; "
                                   "no arbitration can accept a reading"})
            self.quarantine.append(record)
            self.audit.append(record)
            return record
        # document-SMILES direct verification: the row's own printed SMILES
        # surviving RDKit + matching the declared mass verifies the row by
        # the document itself — notation resolution not required
        if reading_mol is not None and row.declared_mass is not None:
            from rdkit.Chem import Descriptors
            exact = exact_mass(row.smiles_reading)
            avg = Descriptors.MolWt(reading_mol)
            declared = float(row.declared_mass)
            conv = None
            if abs(exact - declared) <= MASS_TOLERANCE:
                conv = "mono"
            elif abs(avg - declared) <= MASS_TOLERANCE:
                conv = "avg"
            if conv:
                record.update({"ok": True, "smiles": row.smiles_reading,
                               "delta": round((exact if conv == "mono"
                                               else avg) - declared, 4),
                               "mass_convention": conv,
                               "verified_by": "document-smiles"})
                self.audit.append(record)
                return record
        parsed = parse_notation(row.notation)
        if parsed.get("unparsed_marker"):
            record["gaps"].append({"reason": "printed terminal marker is "
                                   f"not an amide/hydroxyl pair: "
                                   f"'{parsed['unparsed_marker']}'"})
        raw_tokens = parsed["tokens"]
        terminal = parsed["terminal"]
        # topology: a bare hyphen-joined notation is an open chain (the
        # cyclo() wrapper marks head-to-tail macrocycles); linear chains
        # carry the terminal group instead of closing
        linear_chain = bool(parsed.get("linear"))
        record["linear_chain"] = linear_chain
        record["tokens"] = list(raw_tokens)
        if not raw_tokens:
            record["gaps"].append({"reason": "no tokens parsed from notation"})
            self.quarantine.append(record)
            self.audit.append(record)
            return record

        flags_list = []
        for tok in raw_tokens:
            # OCR prints the D-form marker with unstable case ('dR'/'dr');
            # normalize 'd'+lowercase-letter before flagging
            if self.opts.get("d_prefix"):
                tok = re.sub(r"^d([a-z])$", lambda m: "d" + m.group(1).upper(),
                             tok)
            flags_list.append(normalize_token(tok, self.opts))

        # per-token candidate pools: L0/L1 first, evidence rungs only for
        # what they could not resolve
        pools: dict[str, list[Candidate]] = {}
        unknown: list[str] = []
        for code, flags in flags_list:
            cands = (_lexicon_candidates(code.upper(), flags)
                     if len(code) == 1 and code.upper() in STD_LETTERS
                     else _lexicon_candidates(code, flags))
            cands = _memory_candidates(code, self.memory) + cands
            if not cands:
                unknown.append(code)
            pools[code] = cands
        record["unknown"] = unknown

        # head-cap classification is chemistry, not spelling, and it runs
        # BEFORE the evidence rungs: the head token is an N-cap when none
        # of its readings satisfies the residue contract (a pinned acyl
        # entry from a previously verified row must re-classify as a cap,
        # so "pool is empty" is the wrong test)
        cap_codes: list[str] = []
        if flags_list:
            head = flags_list[0][0]
            head_pool = pools.get(head) or []
            residue_head = any(
                _contract_fragment(c) if c.role != "cap" else None
                for c in head_pool)
            if head_pool and not residue_head:
                cap_readings = self._cap_candidates(head)
                if cap_readings:
                    cap_codes.append(head)
                    pools[head] = cap_readings
                    flags_list = flags_list[1:]
                    record["tokens"] = record["tokens"][1:]
                    record["gaps"].append({
                        "reason": f"head token '{head}' classified as N-cap"})
            elif not head_pool:
                cap_readings = self._cap_candidates(head)
                if cap_readings:
                    cap_codes.append(head)
                    pools[head] = cap_readings
                    flags_list = flags_list[1:]
                    record["tokens"] = record["tokens"][1:]
                    record["gaps"].append({
                        "reason": f"head token '{head}' classified as N-cap"})

        if unknown:
            record["gaps"].append({"reason": "tokens without an L0/L1 "
                                   "reading", "tokens": unknown})
            self._run_evidence_rungs(record, flags_list, pools, row)
            unknown = [code for code, _ in flags_list if not pools.get(code)]
            record["unknown"] = unknown

        if any(not pools.get(c, []) for c in [c for c, _ in flags_list]
               + cap_codes):
            record["gaps"].append({
                "reason": "unresolved tokens after all evidence rungs",
                "tokens": [c for c, _ in flags_list
                           if not pools.get(c)] + [c for c in cap_codes
                                                   if not pools.get(c)]})
            self.quarantine.append(record)
            self.audit.append(record)
            return record

        if row.form.upper() == "D":
            self._resolve_dimer_row(record, flags_list, cap_codes, pools,
                                    terminal, linear_chain)
            self.audit.append(record)
            return record

        verdict = self._assemble_with_candidates(flags_list, cap_codes,
                                                 pools, row.declared_mass,
                                                 terminal, linear_chain)
        record.update({k: verdict.get(k) for k in
                       ("ok", "smiles", "delta", "mass_convention")})
        # per-token provenance travels with the record so callers (skills,
        # pre-pass CLI) can audit exactly which reading verified
        record["used_candidates"] = verdict.get("used_candidates", {})
        record["used_caps"] = verdict.get("used_caps", [])
        # strongest possible cross-check: the document's own printed
        # SMILES vs the assembled connectivity. A match upgrades a
        # reading-arbitrated row to verified; a mismatch on a mass-
        # verified row is recorded as evidence, not silently dropped.
        if record["ok"] and reading_mol is not None and record["smiles"]:
            built = Chem.MolFromSmiles(record["smiles"])
            match = built is not None and \
                Chem.MolToSmiles(built) == Chem.MolToSmiles(reading_mol)
            record["smiles_match"] = match
            if row.declared_mass is None:
                record["ok"] = match
                if not match:
                    record["gaps"].append({
                        "reason": "assembly differs from the row's printed "
                                  "SMILES reading"})
                    self.quarantine.append(record)
                    self.audit.append(record)
                    return record
            elif not match:
                record["gaps"].append({
                    "reason": "mass-verified assembly differs from the "
                              "printed SMILES reading (stereo/connectivity "
                              "divergence kept as evidence)"})
        if verdict.get("failure"):
            record["gaps"].append({"reason": verdict["failure"]})
        if verdict.get("ok"):
            self._commit_verified(record, flags_list, pools, verdict,
                                  cap_codes)
        else:
            self.quarantine.append(record)
        self.audit.append(record)
        return record

    def _cap_candidates(self, code: str) -> list[Candidate]:
        """Acyl-contract candidates for a head token. Each raw candidate is
        tried under the cap contract; candidates that also satisfy the
        backbone contract keep their residue reading upstream, so a genuine
        residue head never loses its reading here."""
        raw = _memory_candidates(code, self.memory) + \
            _lexicon_candidates(code, {})
        fold = re.sub(r"[\s_]+", "", code).casefold()
        raw.extend(Candidate(entry["cap"], "dictionary",
                             {"name": entry.get("name", "")})
                   for entry in self.monomer_dictionary.get(fold, [])
                   if entry.get("cap"))
        if self.use_ccd and not raw:
            raw.extend(_ccd_candidates(code))
        out = []
        for cand in raw:
            out.append(Candidate(cand.fragment_smiles, cand.source,
                                 {**cand.provenance, "code": code},
                                 role="cap"))
        return out

    def _resolve_dimer_row(self, record, flags_list, cap_codes, pools,
                           terminal, linear_chain=True):
        """D rows: the declared mass belongs to monomer x2 + linker, so no
        monomer-only mass verdict exists. Assemble the monomer as a
        deferred candidate and name exactly why it is not verified."""
        all_codes = [c for c, _ in flags_list] + cap_codes
        if any(not pools.get(c) for c in all_codes):
            record["gaps"].append({"reason": "dimer row: tokens unresolved"})
            self.quarantine.append(record)
            return
        fragment_of, missing = {}, []
        for code, _f in flags_list:
            for cand in pools[code]:
                frag = _contract_fragment(cand)
                if frag:
                    fragment_of[code] = frag
                    break
            else:
                missing.append(code)
        if missing:
            record["gaps"].append({"reason": "dimer row: candidate readings "
                                   "violated the fragment contract",
                                   "tokens": missing})
            self.quarantine.append(record)
            return
        units = [{"code": c, "entry": {"fragment_smiles": fragment_of[c]}}
                 for c, _ in flags_list]
        smiles, error, _notes = _assemble(
            units, {"ring_size": len(units), "linear_chain": linear_chain,
                    "caps": [], "terminal": terminal})
        record["deferred"] = True
        record["smiles"] = smiles
        record["gaps"].append({
            "reason": "dimer row: monomer assembled as a deferred candidate; "
            "declared mass includes the dimerization linker so no monomer "
            "mass verdict exists"
            + ("" if smiles else f" (assembly rejected: {error})" )})

    def _run_evidence_rungs(self, record, flags_list, pools, row):
        """Route each unresolved token through the policy's rung order:
        document-own structure readings first (the document is the
        authoritative dictionary for its own codes), then catalogs.
        Candidates join the pool; arbitration still decides. The per-token
        rung trace lands in the record."""
        trace = record.setdefault("rung_trace", [])
        for code, _flags in flags_list:
            if pools.get(code):
                continue
            cands: list[Candidate] = []
            fold = re.sub(r"[\s_]+", "", code).casefold()
            for rung in ["document"] + self.policy.token_rungs():
                before = len(cands)
                if rung == "document":
                    # crop-level OCSR decode leads, page-level VLM
                    # transcription follows (production semantics)
                    for reading in self.memory.get("struct:" + fold, {}).get(
                            "readings", []):
                        cands.append(Candidate(reading["smiles"],
                                               reading["source"],
                                               {"origin": "document table"}))
                elif rung == "memory":
                    cands.extend(_memory_candidates(code, self.memory))
                elif rung == "lexicon":
                    flags = next((f for c, f in flags_list if c == code), {})
                    cands.extend(_lexicon_candidates(code, flags))
                elif rung == "builtin":
                    for entry in self.monomer_dictionary.get(fold, []):
                        if entry.get("backbone"):
                            cands.append(Candidate(
                                entry["backbone"], "dictionary",
                                {"name": entry.get("name", ""),
                                 "sources": entry.get("sources", [])}))
                elif rung == "ccd":
                    cands.extend(_ccd_candidates(code))
                elif rung in ("opsin", "pubchem"):
                    # name resolution is memoized per code: a code printed
                    # 50 times costs one OPSIN/PubChem pass, not fifty
                    slot = self.memory.setdefault("cand:" + fold, {})
                    cached = slot.get("name_candidates")
                    if cached is None:
                        names = self._document_names(code)
                        cached = [(c.fragment_smiles, c.source,
                                   json.dumps(c.provenance, sort_keys=True))
                                  for c in _name_candidates(code, names)] \
                            if names else []
                        slot["name_candidates"] = cached
                    for frag, cand_source, prov in cached:
                        if cand_source == rung:
                            cands.append(Candidate(frag, cand_source,
                                                   json.loads(prov)))
                if len(cands) > before:
                    trace.append({"token": code, "rung": rung,
                                  "candidates": len(cands) - before})
            if cands:
                pools[code] = cands
                record["gaps"].append({
                    "reason": f"token '{code}' carries candidate readings "
                              f"only (sources: {sorted({c.source for c in cands})}); "
                              "awaiting declared-mass arbitration"})

    def ingest_structure_smiles(self, pairs: list[tuple[str, str]],
                                source: str = "vision") -> int:
        """Load (code, SMILES) readings of the document's own structure
        drawings. source 'ocsr' = the trained OCSR decode (primary);
        'vision' = VLM transcription (fallback). Both are document
        evidence ranked above catalog candidates, still candidates: the
        declared mass arbitrates like everything else."""
        count = 0
        for code, smiles in pairs:
            if not code or not smiles:
                continue
            if Chem.MolFromSmiles(smiles) is None:
                continue
            primary = code.split("\n")[0].strip()
            for key in {code.casefold(), primary.casefold(),
                        re.sub(r"[\s_]+", "", primary).casefold()}:
                if not key:
                    continue
                slot = self.memory.setdefault("struct:" + key, {})
                readings = slot.setdefault("readings", [])
                if not any(r["smiles"] == smiles and r["source"] == source
                           for r in readings):
                    readings.append({"smiles": smiles, "source": source})
                    count += 1
        return count

    def _document_names(self, code: str) -> list[str]:
        """The document's own name readings for this code (abbreviation
        table pass). Silent when no table pass has run — the harness never
        invents a name to force a resolution."""
        return self.memory.get("names:" + code.casefold(), {}).get("names", [])

    def _assemble_with_candidates(self, flags_list, cap_codes, pools,
                                  declared_mass, terminal,
                                  linear_chain=False):
        """Walk candidate combinations, strongest provenance first: the
        first combination is the document-memory/lexicon reading, catalog
        candidates follow. Every rejected combination's reason is kept in
        the trial trail."""
        pools_list = []
        for code, _flags in flags_list:
            entries = [(_contract_fragment(c), c)
                       for c in pools.get(code, [])]
            entries = [(frag, c) for frag, c in entries if frag]
            if not entries:
                return {"ok": False,
                        "failure": f"token '{code}' has no contract-valid "
                                   "reading"}
            pools_list.append(entries[:MAX_CANDIDATES_PER_TOKEN])
        cap_entries = []
        for code in cap_codes:
            entries = [(_contract_fragment(c), c)
                       for c in pools.get(code, [])]
            entries = [(frag, c) for frag, c in entries if frag]
            if not entries:
                return {"ok": False,
                        "failure": f"N-cap '{code}' has no contract-valid "
                                   "reading"}
            cap_entries.append(entries[0])

        trail: list[str] = []
        # Greedy first, then single-token alternation: verify the strongest
        # reading of every token, and on failure walk ONE token's
        # alternatives at a time. O(sum(pool sizes)) trials — the cartesian
        # product explodes for long notations with several multi-reading
        # tokens (4^17-style), which is a hang, not a search.
        tried = 0
        max_trials = 64
        current = [pool[0] for pool in pools_list]   # (frag, cand) pairs

        def _verify_with(combo):
            fragment_of = {code: f
                           for (code, _f), (f, _c) in zip(flags_list, combo)}
            return _verify_assembly(list(map(tuple, flags_list)),
                                    [frag for frag, _ in cap_entries],
                                    declared_mass, fragment_of, terminal,
                                    linear_chain)

        verdict = _verify_with(current)
        tried += 1
        while not verdict.get("ok") and tried < max_trials:
            advanced = False
            for idx in range(len(pools_list)):
                for frag, cand in pools_list[idx][1:]:
                    if tried >= max_trials:
                        break
                    tried += 1
                    trial = list(current)
                    trial[idx] = (frag, cand)
                    verdict = _verify_with(trial)
                    if verdict.get("ok"):
                        current = trial
                        break
                    trail.append(verdict.get("failure") or "rejected")
                if verdict.get("ok"):
                    advanced = True
                    break
            if not advanced:
                break
            # still failing after this token's alternatives: keep the best
            # switch and try the next token's alternatives
            if not verdict.get("ok"):
                continue
        if verdict.get("ok"):
            verdict["used_candidates"] = {
                code: {"fragment_smiles": frag, "source": cand.source,
                       "provenance": cand.provenance}
                for (code, _f), (frag, cand) in zip(flags_list, current)}
            verdict["used_caps"] = [
                {"code": code, "fragment_smiles": frag, "source": cand.source}
                for code, (frag, cand) in zip(cap_codes, cap_entries)]
            return verdict
        note = (f"{tried} combinations tried"
                if tried < max_trials else
                f"combination budget {max_trials} exhausted; evidence too "
                "ambiguous — strengthen a rung (document reading, dictionary "
                "entry) rather than widening the search")
        return {"ok": False,
                "failure": "no candidate combination passed arbitration "
                           f"({note})"}

    def _commit_verified(self, record, flags_list, pools, verdict, cap_codes):
        """A verified row upgrades its tokens: document memory immediately,
        shared lexicon with provenance. Only fragments that actually
        passed arbitration are committed — a candidate that merely existed
        is never recorded."""
        used = verdict.get("used_candidates", {})
        for code, info in used.items():
            entry = self.memory.setdefault(code.casefold(), {
                "fragment_smiles": info["fragment_smiles"],
                "verified_rows": 0, "sources": []})
            entry["verified_rows"] += 1
            if info["source"] not in entry["sources"]:
                entry["sources"].append(info["source"])
            record["verified"].append({"code": code,
                                       "source": info["source"]})
            self._pin_to_lexicon(code, info, record)
        for cap in verdict.get("used_caps", []):
            self._pin_to_lexicon(cap["code"],
                                 {"fragment_smiles": cap["fragment_smiles"],
                                  "source": cap["source"]}, record)

    def _pin_to_lexicon(self, code, info, record):
        if not code:
            return
        note = (f"doc-verified via {info.get('source', 'unknown')} "
                f"({self.doc_key})")
        with _LEXICON_PIN_LOCK:
            lex = _lexicon_load()
            existing = lex.get(code)
            if existing is None or not existing.get("fragment_smiles") \
                    or existing["fragment_smiles"] != info["fragment_smiles"]:
                lex[code] = {"fragment_smiles": info["fragment_smiles"],
                             "n_methyl": False, "ring_nitrogen": False,
                             "note": note}
                _lexicon_save(lex)
                record.setdefault("lexicon_pins", []).append(
                    {"code": code, "note": note})

    # -- pre-pass ---------------------------------------------------------

    def bootstrap(self, rows: list[NotationRow]) -> dict:
        """Amortized document pre-pass: resolve every row in order. Later
        rows ride on earlier verified fragments — the point of running
        this before any per-compound work."""
        summary = {"rows": 0, "verified": 0, "deferred": 0, "failed": 0,
                   "lexicon_pins": 0, "gaps": []}
        for row in rows:
            record = self.resolve_row(row)
            summary["rows"] += 1
            if record["ok"]:
                summary["verified"] += 1
            elif record["deferred"]:
                summary["deferred"] += 1
            else:
                summary["failed"] += 1
            summary["lexicon_pins"] += len(record.get("lexicon_pins", []))
            for gap in record["gaps"]:
                summary["gaps"].append({"seq_id": row.seq_id, **gap})
        return summary

    def coverage_report(self) -> dict:
        """Named-gap accounting: what resolved, from which rungs, and what
        still needs evidence."""
        from collections import Counter
        sources = Counter()
        for record in self.audit:
            for entry in record.get("verified", []):
                sources[entry["source"]] += 1
        return {"doc_key": self.doc_key,
                "rows_seen": len(self.audit),
                "rows_verified": sum(1 for r in self.audit if r["ok"]),
                "rows_deferred": sum(1 for r in self.audit
                                     if r.get("deferred")),
                "verified_by_source": dict(sources),
                "memory_codes": len([k for k in self.memory
                                     if not k.startswith("names:")]),
                "quarantined_rows": len(self.quarantine),
                "gap_reasons": Counter(
                    g["reason"].split(':')[0].split(';')[0][:60]
                    for r in self.audit for g in r.get("gaps", []))}

    # -- document table pass (L3) -----------------------------------------

    def ingest_abbreviation_names(self, pairs: list[tuple[str, str]]) -> int:
        """Load (code, chemical name) readings from the document's own
        abbreviation table (a transcription pass upstream). Codes are
        normalized the same way printed codes are matched to sequence
        tokens (annotation lines stripped, casefold); names are stored as
        evidence for the name rung. Nothing resolves until a name actually
        produces a contract-valid, mass-verified fragment."""
        count = 0
        for code, name in pairs:
            if not code or not name:
                continue
            primary = code.split("\n")[0].strip()
            for key in {code.casefold(), primary.casefold()}:
                if not key:
                    continue
                slot = self.memory.setdefault("names:" + key, {})
                names = slot.setdefault("names", [])
                if name not in names:
                    names.append(name)
                    count += 1
        return count


def _combo_iter(pools_list):
    """Candidate combinations in evidence order. Each pool is already
    strongest-first, so the first yielded combination is the
    document-memory/lexicon reading and catalog readings follow. Bounded:
    pools were truncated at MAX_CANDIDATES_PER_TOKEN upstream."""

    def rec(idx, acc):
        if idx == len(pools_list):
            yield list(acc)
            return
        for chosen in pools_list[idx]:
            acc.append(chosen)
            yield from rec(idx + 1, acc)
            acc.pop()

    yield from rec(0, [])


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def rows_from_json(path: str) -> list[NotationRow]:
    """Rebuild NotationRow objects from a mined rows JSON (list of dicts
    with notation/mw/md/seq_id/cyclization fields)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for item in raw:
        attr = item.get("attr") if isinstance(item.get("attr"), dict) else {}
        rows.append(NotationRow(
            notation=item.get("notation", ""),
            seq_id=str(item.get("seq_id", "")),
            form=str(item.get("md", "") or attr.get("form", "")),
            declared_mass=item.get("mw") if item.get("mw") is not None
            else attr.get("declared_mass"),
            cyclization=attr.get("cyclization", {}),
            page=item.get("page"),
            smiles_reading=item.get("smiles_reading", ""),
            source="json"))
    return rows


def write_results_csv(harness: "PeptideResolutionHarness", out_path: str,
                      rows: list[NotationRow]) -> int:
    """The document parse deliverable: one row per compound — identity,
    verified structure, mass arbitration, annotations, named gaps.
    Verified rows carry SMILES; deferred/fail rows carry their reasons so
    the downstream reviewer sees exactly what needs which rung."""
    import csv
    by_seq = {r.seq_id: r for r in rows if r.seq_id}
    written = 0
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["seq_id", "form", "notation", "status", "smiles",
                         "mass_delta", "mass_convention", "cyclization",
                         "dimerization", "stability", "ic50", "gaps",
                         "sources"])
        for record in harness.audit:
            row = by_seq.get(str(record.get("seq_id", "")))
            cyc = (row.cyclization if row else {}) or {}
            status = "verified" if record["ok"] else (
                "deferred" if record.get("deferred") else "failed")
            writer.writerow([
                record.get("seq_id"), record.get("form"),
                record.get("notation"), status, record.get("smiles") or "",
                record.get("delta"), record.get("mass_convention", ""),
                cyc.get("positions", ""), cyc.get("dimerization", ""),
                cyc.get("stability", ""), cyc.get("ic50", ""),
                " | ".join(g["reason"][:160] for g in record.get("gaps", [])),
                ";".join(sorted({v.get("source", "") for v in
                                 record.get("used_candidates", {}).values()
                                 if v.get("source")})),
            ])
            written += 1
    return written


def tables_from_ocr_json(path: str) -> list:
    """OCR job JSON (paddleocr_client output) -> [(page_no, table_cells)].
    The HTML parsing lives with the mining caller in scripts; here we only
    re-open an already-extracted grids file: JSON list of [page, table]."""
    return [tuple(item) for item in
            json.loads(Path(path).read_text(encoding="utf-8"))]


if __name__ == "__main__":  # pragma: no cover - pre-pass CLI entry
    import argparse
    parser = argparse.ArgumentParser(
        description="Peptide notation resolution pre-pass over mined rows")
    parser.add_argument("rows_json")
    parser.add_argument("--doc-key", default="doc")
    parser.add_argument("--no-ccd", action="store_true",
                        help="disable RCSB candidate lookups")
    parser.add_argument("--no-name-lookups", action="store_true",
                        help="disable OPSIN/PubChem name resolution")
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    harness = PeptideResolutionHarness(
        args.doc_key, use_ccd=not args.no_ccd,
        use_name_lookups=not args.no_name_lookups)
    summary = harness.bootstrap(rows_from_json(args.rows_json))
    print(json.dumps(summary, indent=1, default=str)[:2000])
    print(json.dumps(harness.coverage_report(), indent=1, default=str))
    if args.report:
        Path(args.report).write_text(json.dumps(
            {"summary": summary, "coverage": harness.coverage_report(),
             "audit": harness.audit}, indent=1, default=str),
            encoding="utf-8")
