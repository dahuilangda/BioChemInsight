"""Pass 2: re-resolve ChEMBL targets that failed or matched the wrong protein.

Exact pref_name matching against canonical names; merges into the existing
cache. Only touches symbols whose current entry is missing or has <10
compounds.
"""
import json
import os
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "figure_data", "chembl_target_compounds.json")
BASE = "https://www.ebi.ac.uk/chembl/api/data"
MAX_PER_TARGET = 1000

EXACT = {
    "AXL": ["Tyrosine-protein kinase receptor UFO"],
    "MERTK": ["Tyrosine-protein kinase MER", "MERTK"],
    "DHODH": ["Dihydroorotate dehydrogenase (quinone), mitochondrial"],
    "EGFR": ["Epidermal growth factor receptor"],
    "GSPT1": ["GSPT1", "Eukaryotic peptide chain release factor GTP-binding subunit ERF3A"],
    "MAP4K1": ["MAP4K1", "Mitogen-activated protein kinase kinase kinase kinase 1"],
    "NLRP3": ["NACHT, LRR and PYD domains-containing protein 3", "NLRP3"],
    "TNF": ["Tumor necrosis factor"],
    "TNKS2": ["Tankyrase 1/2", "Tankyrase-2"],
    "MET": ["Hepatocyte growth factor receptor"],
    "KDR": ["Vascular endothelial growth factor receptor 2"],
}

session = requests.Session()
session.headers["Accept"] = "application/json"


def get(url, params=None):
    for attempt in range(2):
        try:
            r = session.get(url, params=params, timeout=45)
            if r.status_code == 200:
                time.sleep(0.3)
                return r.json()
            print("HTTP", r.status_code, r.url[:120])
        except Exception as exc:
            print("ERR", attempt, exc)
        time.sleep(2)
    return None


def active_compounds(chembl_id):
    mols = []
    offset = 0
    while len(mols) < MAX_PER_TARGET:
        data = get(f"{BASE}/activity.json", {
            "target_chembl_id": chembl_id,
            "activity_type__in": "IC50,EC50,Ki",
            "standard_relation__in": "=",
            "standard_units": "nM",
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
            if val <= 100000:
                mols.append(a["molecule_chembl_id"])
        mols = list(dict.fromkeys(mols))
        meta = data.get("page_meta", {})
        if not meta.get("next"):
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
    cache = json.load(open(CACHE))
    for sym, wants in EXACT.items():
        cur = cache.get(sym) or {}
        if len(cur.get("smiles") or []) >= 10 and cur.get("pref_name") in wants:
            print(f"[{sym}] cached ok ({len(cur['smiles'])})")
            continue
        cand = None
        for want in wants:
            data = get(f"{BASE}/target.json", {
                "pref_name__contains": want, "limit": 50,
                "only": "target_chembl_id,pref_name,organism,target_type",
            })
            cands = (data or {}).get("targets", [])
            exact = [t for t in cands if (t.get("pref_name") or "") == want]
            cand = next((t for t in exact if t.get("organism") == "Homo sapiens"), None) or (exact[0] if exact else None)
            if cand is not None:
                break
        if cand is None:
            print(f"[{sym}] NOT FOUND exact {wants}")
            continue
        cid, pref = cand["target_chembl_id"], cand.get("pref_name")
        mols = active_compounds(cid)
        smap = smiles_for(mols)
        smiles = [smap[m] for m in mols if m in smap]
        old = cache.get(sym) or {}
        if len(smiles) >= len(old.get("smiles") or []):
            cache[sym] = {"target_chembl_id": cid, "pref_name": pref, "smiles": smiles}
        elif old.get("target_chembl_id") in (None, "") and cid:
            old["target_chembl_id"], old["pref_name"] = cid, pref
            cache[sym] = old
        print(f"[{sym}] {cid} ({pref}): {len(mols)} actives, {len(smiles)} SMILES")
        json.dump(cache, open(CACHE, "w"), indent=1)
    json.dump(cache, open(CACHE, "w"), indent=1)
    print("updated:", CACHE)


if __name__ == "__main__":
    main()
