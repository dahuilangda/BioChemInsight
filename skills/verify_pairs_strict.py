"""Strict cross-validation of verified real-patent crop/SMILES pairs.

Label matching alone cannot rule out wrong-row pairing, so every pair is
re-decoded from pixels and compared against the gold SMILES:

  1. label evidence     : seq_id annotation pairing (already in manifests)
  2. decode cross-check : MolSight decode of the crop -> Tanimoto(decoded, gold).
     A near-decode of the *same* molecule keeps skeleton identity, so a
     correctly paired crop scores >= 0.55; low scores indicate wrong-row
     pairing or neighbour bleed.
  3. mass consistency   : decoded exact mass vs declared exact_mass is
     recorded as evidence but never vetoes (bare decode has mass bias).

Grades:
  strict   tanimoto >= 0.65   -> same molecule confirmed
  weak     0.45 <= t < 0.65   -> near match, kept but flagged
  rejected t < 0.45 or decode failure -> not fit for training

Inputs : three real_crops manifests (verified=true rows only, 255 pairs)
API    : POST http://localhost:8011/v1/harness/resolve-figure (multipart)
State  : /tmp/pair_verify_state.json (checkpoint every 25 pairs, resume-safe)

Outputs (on completion):
  - verify_strict field added to every verified row of each manifest
  - data/verified_pairs_strict.json training-pair list (strict only)
  - per-patent strict/weak/rejected statistics on stdout
"""
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

MANIFESTS = [
    ("/data/BioChemInsight/data/real_crops", "/data/BioChemInsight/data/real_crops/real_crops_manifest.json"),
    ("/tmp/crops_kras", "/tmp/crops_kras/real_crops_manifest.json"),
    ("/tmp/crops_gpc3", "/tmp/crops_gpc3/real_crops_manifest.json"),
]
API = "http://localhost:8011"
RESOLVE = "/v1/harness/resolve-figure"
STATE_PATH = "/tmp/pair_verify_state.json"
STRICT_OUT = "/data/BioChemInsight/data/verified_pairs_strict.json"

T_STRICT = 0.65
T_WEAK = 0.45
WORKERS = 2
CHECKPOINT_EVERY = 25
REQ_TIMEOUT = 300
RETRIES = 2

_lock = threading.Lock()


def tanimoto(smiles_a, smiles_b):
    """Morgan r2 2048-bit Tanimoto; chirality-blind (skeleton identity).
    Returns (tanimoto, chiral_tanimoto) or (None, None) on parse failure."""
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        try:  # keep logs free of RDKit deprecation noise
            from rdkit import RDLogger
            RDLogger.DisableLog("rdApp.*")
        except Exception:
            pass

        def fps(smi):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                mol = Chem.MolFromSmiles(smi, sanitize=False)
                if mol is None:
                    return None
                try:
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
                except Exception:
                    return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False)
            fp_c = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=True)
            return fp, fp_c

        fa, fb = fps(smiles_a), fps(smiles_b)
        if fa is None or fb is None:
            return None, None
        return (DataStructs.TanimotoSimilarity(fa[0], fb[0]),
                DataStructs.TanimotoSimilarity(fa[1], fb[1]))
    except Exception:
        return None, None


def exact_mass(smiles):
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return round(float(Descriptors.ExactMolWt(mol)), 4)
    except Exception:
        return None


def decode_crop(crop_abs):
    """Multipart upload -> selected.smiles, or (None, reason)."""
    last_err = None
    for attempt in range(RETRIES + 1):
        try:
            with open(crop_abs, "rb") as fh:
                r = requests.post(
                    API + RESOLVE,
                    files={"file": (os.path.basename(crop_abs), fh, "image/png")},
                    data={"document_id": "verify"},
                    timeout=REQ_TIMEOUT,
                )
            if r.status_code != 200:
                last_err = f"http_{r.status_code}: {r.text[:200]}"
                time.sleep(3)
                continue
            payload = r.json()
            sel = payload.get("selected") or {}
            smi = (sel.get("smiles") or "").strip()
            if not smi:
                return None, "no_selected_smiles(state=%s)" % payload.get("state")
            return {"smiles": smi, "method": sel.get("method"),
                    "verdict": sel.get("verdict"), "confidence": sel.get("confidence"),
                    "state": payload.get("state")}, None
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(5)
    return None, last_err or "unknown_error"


def evaluate_one(pair):
    """Full triple-evidence check for one pair. Returns result dict."""
    res = {
        "verdict": "rejected",
        "reason": None,
        "tanimoto": None,
        "tanimoto_chiral": None,
        "decoded_smiles": None,
        "decoded_exact_mass": None,
        "mass_delta": None,
        "decode_method": None,
        "decode_state": None,
        "elapsed_s": None,
    }
    t0 = time.time()
    dec, err = decode_crop(pair["crop_abs"])
    res["elapsed_s"] = round(time.time() - t0, 1)
    if dec is None:
        res["reason"] = "decode_failed: %s" % err
        return res
    res["decoded_smiles"] = dec["smiles"]
    res["decode_method"] = dec["method"]
    res["decode_state"] = dec["state"]
    t, tc = tanimoto(dec["smiles"], pair["smiles"])
    if t is None:
        res["reason"] = "invalid_decoded_smiles"
        return res
    res["tanimoto"] = round(t, 4)
    res["tanimoto_chiral"] = round(tc, 4)
    dm = exact_mass(dec["smiles"])
    res["decoded_exact_mass"] = dm
    if pair["exact_mass"] is not None and dm is not None:
        res["mass_delta"] = round(dm - pair["exact_mass"], 4)
    if t >= T_STRICT:
        res["verdict"] = "strict"
        res["reason"] = "tanimoto>=0.65_same_skeleton"
    elif t >= T_WEAK:
        res["verdict"] = "weak"
        res["reason"] = "tanimoto_0.45-0.65_near_match"
    else:
        res["verdict"] = "rejected"
        res["reason"] = "tanimoto<0.45_pairing_doubt"
    return res


def load_pairs():
    pairs = []
    for base, mpath in MANIFESTS:
        with open(mpath) as fh:
            man = json.load(fh)
        for row in man["rows"]:
            if row.get("verified") is not True:
                continue
            pairs.append({
                "base": base, "manifest": mpath,
                "crop_rel": row["crop"], "crop_abs": os.path.join(base, row["crop"]),
                "patent": row.get("patent"), "page": row.get("page"),
                "seq_id": row.get("seq_id"), "smiles": row.get("smiles"),
                "exact_mass": row.get("exact_mass"),
                "key": base + "::" + row["crop"],
            })
    return pairs


def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH) as fh:
                return json.load(fh)
        except Exception:
            pass
    return {"results": {}}


def save_state(state):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh)
    os.replace(tmp, STATE_PATH)


def finalize(state):
    """Rewrite manifests with verify_strict, emit strict dataset, print stats."""
    pairs = load_pairs()
    stats = {}
    strict_pairs = []
    per_manifest_rows = {}
    for pair in pairs:
        res = state["results"].get(pair["key"])
        if res is None:
            continue
        per_manifest_rows.setdefault(pair["manifest"], []).append((pair, res))
        st = stats.setdefault(pair["patent"], {"strict": 0, "weak": 0, "rejected": 0})
        st[res["verdict"]] += 1
        if res["verdict"] == "strict":
            strict_pairs.append({
                "crop": pair["crop_abs"],
                "crop_rel": pair["crop_rel"],
                "patent": pair["patent"],
                "page": pair["page"],
                "seq_id": pair["seq_id"],
                "smiles": pair["smiles"],
                "exact_mass": pair["exact_mass"],
                "verify_strict": {
                    "verdict": "strict",
                    "tanimoto": res["tanimoto"],
                    "tanimoto_chiral": res["tanimoto_chiral"],
                    "decoded_smiles": res["decoded_smiles"],
                    "decoded_exact_mass": res["decoded_exact_mass"],
                    "mass_delta": res["mass_delta"],
                    "decode_method": res["decode_method"],
                    "evidence": "label_seq_id + decode_crosscheck(tanimoto>=0.65) + mass_recorded",
                },
            })

    # in-place manifest update (keep every other field)
    for _, mpath in MANIFESTS:
        rows = per_manifest_rows.get(mpath, [])
        if not rows:
            continue
        with open(mpath) as fh:
            man = json.load(fh)
        by_rel = {p["crop_rel"]: r for p, r in rows}
        for row in man["rows"]:
            res = by_rel.get(row.get("crop"))
            if res is not None:
                row["verify_strict"] = {
                    "verdict": res["verdict"],
                    "tanimoto": res["tanimoto"],
                    "tanimoto_chiral": res["tanimoto_chiral"],
                    "decoded_smiles": res["decoded_smiles"],
                    "decoded_exact_mass": res["decoded_exact_mass"],
                    "mass_delta": res["mass_delta"],
                    "reason": res["reason"],
                    "decode_method": res["decode_method"],
                }
        tmp = mpath + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(man, fh, indent=1, ensure_ascii=False)
        os.replace(tmp, mpath)
        print("[finalize] %s: %d rows annotated" % (mpath, len(rows)), flush=True)

    total = {k: sum(v[k] for v in stats.values()) for k in ("strict", "weak", "rejected")}
    doc = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "criteria": {"strict": "tanimoto>=0.65", "weak": "0.45<=tanimoto<0.65",
                     "rejected": "tanimoto<0.45 or decode failure"},
        "decoder": "MolSight via %s (document_id=verify)" % (API + RESOLVE),
        "fingerprint": "Morgan r2 2048bit, chirality-blind primary",
        "stats_per_patent": stats,
        "stats_total": total,
        "n_pairs": len(strict_pairs),
        "pairs": strict_pairs,
    }
    with open(STRICT_OUT, "w") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
    print("[finalize] strict dataset: %s (%d pairs)" % (STRICT_OUT, len(strict_pairs)), flush=True)
    print("[finalize] stats per patent:", flush=True)
    for pat, st in sorted(stats.items()):
        print("  %-16s strict=%d weak=%d rejected=%d" % (pat, st["strict"], st["weak"], st["rejected"]), flush=True)
    print("  %-16s strict=%d weak=%d rejected=%d" % ("TOTAL", total["strict"], total["weak"], total["rejected"]), flush=True)


def main():
    finalize_only = "--finalize" in sys.argv
    pairs = load_pairs()
    state = load_state()
    todo = [p for p in pairs if p["key"] not in state["results"]]
    print("[strict-verify] %d pairs total, %d done, %d to go"
          % (len(pairs), len(pairs) - len(todo), len(todo)), flush=True)
    if finalize_only:
        finalize(state)
        return

    done_since_ckpt = 0
    rejected_log = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(evaluate_one, p): p for p in todo}
        try:
            for fut in as_completed(futs):
                pair = futs[fut]
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001
                    res = {"verdict": "rejected", "reason": "worker_error: %s" % exc,
                           "tanimoto": None, "tanimoto_chiral": None,
                           "decoded_smiles": None, "decoded_exact_mass": None,
                           "mass_delta": None, "decode_method": None,
                           "decode_state": None, "elapsed_s": None}
                with _lock:
                    state["results"][pair["key"]] = res
                    done_since_ckpt += 1
                    n_done = len(state["results"])
                    print("[%d/%d] %-8s t=%s %s :: %s"
                          % (n_done, len(pairs), res["verdict"],
                             ("%.3f" % res["tanimoto"]) if res["tanimoto"] is not None else "-",
                             pair["crop_rel"], res["reason"]), flush=True)
                    if res["verdict"] == "rejected":
                        rejected_log.append({"crop": pair["crop_rel"], "patent": pair["patent"],
                                             "reason": res["reason"], "tanimoto": res["tanimoto"]})
                    if done_since_ckpt >= CHECKPOINT_EVERY:
                        save_state(state)
                        done_since_ckpt = 0
                        print("[checkpoint] %d results saved to %s"
                              % (n_done, STATE_PATH), flush=True)
        finally:
            save_state(state)

    print("[strict-verify] all %d pairs done" % len(state["results"]), flush=True)
    finalize(state)


if __name__ == "__main__":
    main()
