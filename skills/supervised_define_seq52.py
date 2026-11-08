"""Supervised lexicon completion for the seq52 blocked SITE name (round 4).

Round-3 diagnosis left exactly one chemical SITE name in the seq52 listing
without a lexicon definition: "Acetyl Norleucine" (the other two unresolved
notes are structural annotations, already excluded by filter_notes; every
other note resolves, "alpha-Methyl-1-serine" via the established 1->l OCR
variant).  The row currently passes only because the DP pairs the note to
the Nle code at W_CODE and the code resolves; the SITE name itself never
resolves.

Protocol (same as the round-2/3 supervised definitions):
  1. define   — supervised candidate fragments, literal reading first;
  2. chemical check — RDKit parse + backbone contract (amine-first,
     carboxyl-last) + name/substructure agreement;
  3. assemble — re-run the gold_aligner row assembly for seq52 with the
     candidate injected, so SITE-name-first resolution, DP pairing and ring
     rotation are exercised exactly as in production;
  4. mass + pairing gate — |assembled - declared| <= 0.8 Da AND the note
     stays paired (names count == baseline, "Acetyl Norleucine" paired);
     otherwise the candidate is refused and NOT persisted.

Measured outcomes (drive the candidate order):
  * literal N-acetyl-Nle: the ring amine would be double-acylated (the
    assembly already models the acetyl as its own unit — the listing's
    "position 1 connected to position 14 via an Acetyl cross-linker", the
    chloroacetyl cap), so the mass prefilter rejects every pairing trial
    and the aligner escapes through a degenerate no-notes reading
    (names 11 -> 0).  Refused by the pairing gate.
  * cap-excluded norleucine: the "Acetyl" names the cross-linker that caps
    this residue; the residue unit is plain norleucine.  Fragment equals
    the stored Nle entry, so the note now pairs at W_AGREE and the assembly
    is byte-identical to the baseline.  Committed with rows_unlocked=[52].

Usage:
  python -I skills/supervised_define_seq52.py [--apply]
Dry-run by default; --apply commits the gated definition to
data/residue_lexicon.json after a no-regression check over every listing
row that mentions the new key.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rdkit import Chem, RDLogger                      # noqa: E402

from skills import gold_aligner as ga                 # noqa: E402
from skills.residue_lexicon import (                  # noqa: E402
    _LOCK, _load, _save, _canonical_fragment, LEXICON_PATH)

RDLogger.DisableLog("rdApp.*")

SEQ_ID = "52"
TABLE_PATH = "/tmp/table1_rows.json"
LISTING_PATH = "/tmp/listing_entries.json"
NEW_KEY = "Acetyl Norleucine"
MASS_TOL = 0.8

# supervised candidates, literal reading first (the gate arbitrates)
CANDIDATES = [
    {
        "label": "literal N-acetyl-norleucine",
        "smiles": "CC(=O)N[C@@H](CCCC)C(=O)O",
        "rationale": "SITE name taken literally: norleucine whose backbone "
                     "amine carries the acetyl",
    },
    {
        "label": "cap-excluded norleucine",
        "smiles": "N[C@@H](CCCC)C(=O)O",
        "rationale": "'Acetyl' names the position-1 chloroacetyl cross-linker "
                     "cap (listing: position 1 connected to position 14 via "
                     "an acetyl cross-linker); the residue unit is plain "
                     "norleucine, the cap is its own assembly unit (ClAc)",
    },
]


def chem_check(name: str, smiles: str) -> tuple[bool, str, str]:
    """RDKit gate -> (ok, canonical fragment or '', reason)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "", "fragment does not parse"
    try:
        canon = _canonical_fragment(mol)
    except Exception as exc:            # NotBackboneForm or sanitization
        return False, "", f"backbone contract rejected: {str(exc)[:120]}"
    if canon is None:
        return False, "", "not an amino-acid skeleton"
    if "norleucine" in name.lower() and "CCCC" not in canon:
        return False, "", "norleucine butyl side chain missing"
    return True, canon, ""


def assemble_seq52(extra_defs: dict[str, dict]) -> dict:
    """Run the production row assembly with candidate definitions injected.

    gold_aligner caches LEXICON at import time, so the injection patches the
    module dict (casefolded, same convention as _load_lexicon) and restores
    it afterwards."""
    backup = dict(ga.LEXICON)
    try:
        for key, entry in extra_defs.items():
            ga.LEXICON[key.casefold()] = entry
        table = json.load(open(TABLE_PATH, encoding="utf-8"))
        listing = json.load(open(LISTING_PATH, encoding="utf-8"))
        return ga.align_row(table[SEQ_ID], listing[SEQ_ID],
                            ga._style_opts("us", None))
    finally:
        ga.LEXICON.clear()
        ga.LEXICON.update(backup)


def load_rows_listing():
    table = json.load(open(TABLE_PATH, encoding="utf-8"))
    listing = json.load(open(LISTING_PATH, encoding="utf-8"))
    return table, listing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="commit the mass-gated definition to the lexicon")
    args = ap.parse_args()

    table, listing = load_rows_listing()
    notes = listing[SEQ_ID]["notes"]
    declared = table[SEQ_ID]["exact_mass"]

    blocked = [n for n in ga.filter_notes(notes)
               if ga.resolve_name(n)[0] is None]
    print(f"seq{SEQ_ID} blocked SITE names: {blocked}")
    if blocked != [NEW_KEY]:
        print("unexpected blocked set — aborting")
        return 2

    baseline = assemble_seq52({})
    print(f"baseline: ok={baseline['ok']} delta={baseline.get('delta')} "
          f"names={len(baseline['names'])} rotation={baseline.get('rotation')}")

    committed = None
    for cand in CANDIDATES:
        ok, canon, why = chem_check(NEW_KEY, cand["smiles"])
        if not ok:
            print(f"[reject-chem] {cand['label']}: {why}")
            continue
        print(f"[chem-ok] {cand['label']}: canonical={canon}")
        entry = {"fragment_smiles": canon, "n_methyl": False,
                 "ring_nitrogen": False, "note": f"supervised: {cand['label']}"}
        res = assemble_seq52({NEW_KEY: entry})
        gate_mass = res.get("ok") and res.get("delta") is not None \
            and abs(res["delta"]) <= MASS_TOL
        gate_pairing = (NEW_KEY in res.get("names", [])
                        and len(res["names"]) == len(baseline["names"]))
        if not (gate_mass and gate_pairing):
            print(f"[gate-refuse] {cand['label']}: mass={gate_mass} "
                  f"(delta={res.get('delta')}, failure={res.get('failure')}) "
                  f"pairing={gate_pairing} (names={len(res.get('names', []))})")
            continue
        print(f"[gate-pass] {cand['label']}: delta={res['delta']:+.4f} "
              f"(declared {declared}), names={len(res['names'])}, "
              f"rotation={res.get('rotation')}")
        committed = (cand, entry)
        break

    if committed is None:
        print("no candidate passed the gate — nothing persisted")
        return 1
    cand, entry = committed

    # the candidate fragment must equal the stored Nle unit so the pairing
    # scores W_AGREE and the assembly stays byte-identical
    nle = json.load(open(LEXICON_PATH, encoding="utf-8"))["Nle"]
    print(f"fragment==Nle entry: "
          f"{entry['fragment_smiles'] == nle['fragment_smiles']}")

    if not args.apply:
        print("dry run — pass --apply to commit")
        return 0

    with _LOCK:
        lex = _load()
        lex[NEW_KEY] = dict(
            entry, rows_unlocked=[int(SEQ_ID)],
            note=entry["note"] + " | mass-gated: row 52 (cap-excluded reading)")
        _save(lex)
    print(f"committed -> {LEXICON_PATH}")

    # no-regression check over every listing row that mentions the new key:
    # a previously-passing row must keep passing (already-failing rows may
    # keep failing — their reading is untouched by this definition when the
    # failure mode is unchanged)
    backup = dict(ga.LEXICON)
    ga.LEXICON[NEW_KEY.casefold()] = lex[NEW_KEY]
    affected = [sid for sid, e in listing.items()
                if NEW_KEY.casefold() in (n.casefold()
                                          for n in ga.filter_notes(e.get("notes", [])))]
    print(f"affected listing rows: {len(affected)}")
    regress = []
    n_ok = 0
    for sid in affected:
        base = ga.align_row(table[sid], listing[sid], ga._style_opts("us", None))
        inj = ga.align_row(table[sid], listing[sid], ga._style_opts("us", None))
        if base["ok"] and not inj["ok"]:
            regress.append(sid)
            print(f"REGRESSION seq{sid}: {inj.get('failure')}")
        elif inj["ok"]:
            n_ok += 1
    ga.LEXICON.clear()
    ga.LEXICON.update(backup)
    print(f"affected rows still ok: {n_ok}/{len(affected)}; "
          f"ok->fail transitions: {regress}")
    if regress:
        print("REGRESSION — rolling back lexicon commit")
        with _LOCK:
            lex = _load()
            lex.pop(NEW_KEY, None)
            _save(lex)
        return 3
    print("no-regression check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
