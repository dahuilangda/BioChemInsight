"""Fetch per-target active compounds from the ChEMBL web API (ChEMBL_37).

Used for the manuscript target-space comparison figure. One-off download with
a disk cache so reruns do not re-query the API. Politeness: 0.3 s between
requests, two retries per request.

Output: paper/figure_data/chembl_target_compounds.json
  {symbol: {"target_chembl_id": ..., "pref_name": ..., "smiles": [...]}}
"""
import json
import os
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "figure_data")
CACHE = os.path.join(OUT_DIR, "chembl_target_compounds.json")
BASE = "https://www.ebi.ac.uk/chembl/api/data"
MAX_PER_TARGET = 1000

# corpus target groups -> component ChEMBL protein symbols
SYMBOLS = {
    "AXL": "AXL", "MERTK": "MERTK", "TYRO3": "TYRO3", "CDK8": "CDK8",
    "DHODH": "Dihydroorotate dehydrogenase", "EGFR": "EGFR",
    "GSPT1": "GSPT1", "MAP4K1": "MAP4K1", "JAK1": "JAK1", "NLRP3": "NLRP3",
    "TNF": "TNF", "TNKS2": "TNKS2", "WRN": "WRN", "MET": "MET",
    "KDR": "KDR", "RET": "RET",
}

session = requests.Session()
session.headers["Accept"] = "application/json"


def get(url, params=None, retries=2):
    for attempt in range(retries + 1):
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 200:
                time.sleep(0.3)
                return r.json()
            print("HTTP", r.status_code, url, str(r.text)[:120])
        except Exception as exc:
            print("ERR", attempt, url, exc)
        time.sleep(2)
    return None


def resolve_target(query):
    """Return (chembl_id, pref_name) for the best human single-protein match."""
    data = get(f"{BASE}/target.json", {
        "pref_name__contains": query, "limit": 100,
        "only": "target_chembl_id,pref_name,organism,target_type",
    })
    if not data:
        return None, None
    cands = data.get("targets", [])
    def score(t):
        name = (t.get("pref_name") or "").lower()
        return (
            t.get("organism") == "Homo sapiens",
            t.get("target_type") == "SINGLE PROTEIN",
            name == query.lower(),
            name.startswith(query.lower() + " ") or name.startswith(query.lower() + "("),
        )
    cands.sort(key=score, reverse=True)
    for t in cands:
        if t.get("organism") == "Homo sapiens":
            return t["target_chembl_id"], t.get("pref_name")
    return (cands[0]["target_chembl_id"], cands[0].get("pref_name")) if cands else (None, None)


def active_compounds(chembl_id):
    """Distinct molecule ids with direct IC50/EC50/Ki <= 100 uM, up to cap."""
    mols = []
    offset = 0
    while len(mols) < MAX_PER_TARGET:
        data = get(f"{BASE}/activity.json", {
            "target_chembl_id": chembl_id,
            "activity_type__in": "IC50,EC50,Ki",
            "standard_relation__in": "=",
            "standard_units": "nM",
            "target_organism": "Homo sapiens",
            "limit": 200, "offset": offset,
            "only": "molecule_chembl_id,standard_value",
        })
        if not data:
            break
        acts = data.get("activities", [])
        if not acts:
            break
        for a in acts:
            try:
                val = float(a.get("standard_value"))
            except (TypeError, ValueError):
                continue
            if val <= 100000:  # <= 100 uM
                mols.append(a["molecule_chembl_id"])
        mols = list(dict.fromkeys(mols))
        meta = data.get("page_meta", {})
        if offset + 200 >= min(meta.get("total_count", 0), 20000) or not meta.get("next"):
            break
        offset += 200
    return mols[:MAX_PER_TARGET]


def smiles_for(mol_ids):
    out = {}
    for i in range(0, len(mol_ids), 200):
        chunk = mol_ids[i:i + 200]
        data = get(f"{BASE}/molecule.json", {
            "molecule_chembl_id__in": ",".join(chunk),
            "only": "molecule_chembl_id,molecule_structures",
            "limit": 200,
        })
        if not data:
            continue
        for m in data.get("molecules", []):
            smi = ((m.get("molecule_structures") or {}).get("canonical_smiles")) or ""
            if smi:
                out[m["molecule_chembl_id"]] = smi
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(CACHE):
        print("cache exists:", CACHE)
        return
    result = {}
    for sym, query in SYMBOLS.items():
        cid, pref = resolve_target(query)
        if not cid:
            print(f"[{sym}] NOT RESOLVED")
            result[sym] = {"target_chembl_id": None, "pref_name": None, "smiles": []}
            continue
        mols = active_compounds(cid)
        smap = smiles_for(mols)
        smiles = [smap[m] for m in mols if m in smap]
        result[sym] = {"target_chembl_id": cid, "pref_name": pref, "smiles": smiles}
        print(f"[{sym}] {cid} ({pref}): {len(mols)} actives, {len(smiles)} with SMILES")
        json.dump(result, open(CACHE, "w"), indent=1)
    json.dump(result, open(CACHE, "w"), indent=1)
    print("written:", CACHE)


if __name__ == "__main__":
    main()
