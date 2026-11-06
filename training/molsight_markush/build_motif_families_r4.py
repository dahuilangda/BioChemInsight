"""Round-4 motif-family expansion pool (generalize, don't patch).

Round-3 diagnosis left fig_s52_0 (0.82) and fig_s77_0 (0.85) behind on two
specific motifs.  Round 4 renders the seq52/seq77 golds (render_hardcase_r4)
AND expands the failure modes into full pattern families, so the model sees
each motif across many skeletons instead of memorizing two molecules:

  Q-aryl   — quaternary chiral centre [C@@](C)(aryl)C(=O) with the aryl
             bound directly to the alpha carbon (the fig_s52 gold class,
             SbMeW mis-assignment), >=12 distinct aryl/heterocycle skeletons
  P-sub    — 2/3/4/5-substituted proline & ACPC rings (the
             C2[C@@H](C)CC[C@H]2N class decoded as plain proline),
             Me/F/OH/aryl substituents, >=12 kinds
  bridge   — ClAc-cap + thioether bridge heads (Nle-ClAc-S-CH2-Cys(NH2),
             the fig_s52 gold N-terminus) combined with SbMeW beta-methyl
             aryl residues, >=10 head x body combinations

Outputs (under --out):
  train_680k.csv          rendered family molecules (train side)
  heldout/heldout_680k.csv rendered SAME-FAMILY molecules kept out of
                          training for the pattern-generalization check
  wildmol_replay.csv      small-molecule replay rows sampled from the v10
                          generalist pool (anti-forgetting for the Markush
                          small-molecule domain)
  linear_replay.csv       linear-peptide rows: generated linear chains
                          (rendered ALIGNED) + v10 linear-peptide rows
                          verbatim (anti-forgetting for the peptide domain)
  families_manifest.json  per-family molecule counts, tags, held-out ids
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor

_argv = sys.argv
sys.argv = sys.argv[:1]
try:
    from render_stage11 import finish
finally:
    sys.argv = _argv
from style_calibration import coordgen_coords, draw_molecule, ALIGNED
from build_vocab_r3 import PROLINE_ANALOG_UNITS
from cyclopeptide_generator import (
    _weighted_letter, _residue_fragment, _ring_size, D_STEREO_PROB)

RDLogger.DisableLog("rdApp.*")

OUT_DEFAULT = "/data/molsight_realpatent_sft_r4"
V10_POOL = "/data/molsight_markush_v10/uspto_mol/train_680k.csv"
MARKUSH_POOL = "/data/markushgrapher_v1/tier2_char.csv"

# ---- Q-aryl: aryl bound DIRECTLY to the quaternary alpha carbon ----------
Q_ARYL_SIDES = {
    "aMeInd_5H": "c1c[nH]c2ccccc12",                 # the fig_s52 gold class
    "aMeInd_5OMe": "c1c[nH]c2c(OC)cccc12",
    "aMeInd_7Cl": "c1c[nH]c2c(Cl)cccc12",
    "aMeInd_5CN": "c1c[nH]c2cc(C#N)ccc12",
    "aMeInd_2Me": "c1c(C)[nH]c2ccccc12",
    "aMeInd_5F": "c1c(F)[nH]c2ccccc12",
    "aMeBenzothiene": "c1ccc2sccc2c1",
    "aMeBenzofuran": "c1ccc2occc2c1",
    "aMePyridyl3": "c1cnccc1",
    "aMeThiene2": "c1ccsc1",
    "aMeNaphth1": "c1ccc2ccccc2c1",
    "aMepFPhe_direct": "c1ccc(F)cc1",
}

# ---- P-sub: additional substituted-proline kinds beyond build_vocab_r3 ---
PROLINE_EXTRA = {
    "4PhPro": "N1C[C@@H](c2ccccc2)C[C@H]1C(=O)",
    "5FPro": "N1C[C@H](F)CC[C@H]1C(=O)",
}

# ---- SbMeW: beta-methyl aryl residues (in-chain generator convention) ----
SBMEW_UNITS = {
    "SbMeW7OMe": "N[C@@H]([C@H](C)c1c[nH]c2c(OC)cccc12)C(=O)",
    "SbMeW7Cl": "N[C@@H]([C@H](C)c1c[nH]c2c(Cl)cccc12)C(=O)",
    "SbMeW5H": "N[C@@H]([C@H](C)c1c[nH]c2ccccc12)C(=O)",
    "SbMeW5CN": "N[C@@H]([C@H](C)c1c[nH]c2cc(C#N)ccc12)C(=O)",
    "SbMeW4Pal": "N[C@@H]([C@H](C)c1ccncc1)C(=O)",
    "SbMePhe": "N[C@@H]([C@H](C)c1ccccc1)C(=O)",
    "SbMeW5F": "N[C@@H]([C@H](C)c1c(F)[nH]c2ccccc12)C(=O)",
}

# ---- bridge heads: the fig_s52 gold N-terminus, varied head residues -----
BRIDGE_HEAD_SIDES = ["CCCC", "Cc1ccccc1", "CC(C)C", "CSCC", "CCC", "Cc1ccc(O)cc1",
                     "CC(=O)N", "C[C@H](O)C", "CCCCN", "Cc1ccncc1"]
BRIDGE_SEG = "C(=O)CSC[C@H](C(N)=O)"

HELDOUT = {
    "Q": {"aMeInd_5Br": "c1c(Br)[nH]c2ccccc12"},
    "P": {"3OHPro": "N1C[C@H](O)CC[C@H]1C(=O)"},
    "B": {"SbMeW6Cl": "N[C@@H]([C@H](C)c1c[nH]c2cc(Cl)ccc12)C(=O)"},
}


def _validate(frag: str) -> bool:
    probe = "CC(=O)" + frag + "NCC(=O)O"
    return Chem.MolFromSmiles(probe) is not None


def _quat_unit(side: str, rng: random.Random) -> str:
    w = "[C@@]" if rng.random() < 0.5 else "[C@]"
    return f"N{w}({side})(C)C(=O)"


def build_family_molecules(family: str, n: int, rng: random.Random) -> list:
    """Cyclopeptides forcing 1-2 slots of the requested motif family."""
    prol = [(k, v) for k, v in {**PROLINE_ANALOG_UNITS,
                                **PROLINE_EXTRA}.items() if _validate(v)]
    sbmew = [(k, v) for k, v in SBMEW_UNITS.items() if _validate(v)]
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * 40:
        tries += 1
        size = _ring_size(rng)
        seq = [_weighted_letter(rng) for _ in range(size)]
        if seq[0] in ("P", "HypOCM", "PipOH"):
            seq[0] = "S"
        n_forced = rng.choices([1, 2], weights=[0.6, 0.4])[0]
        n_forced = min(n_forced, size - 1)
        slots = rng.sample(range(1, size), n_forced)

        if family == "B":
            # bridge head replaces the normal head; one slot forced to SbMeW
            head = f"N9[C@@H]({rng.choice(BRIDGE_HEAD_SIDES)}){BRIDGE_SEG}"
            sb_slot = rng.randrange(1, size)
            body = []
            for p, spec in enumerate(seq[1:], start=1):
                if p == sb_slot:
                    k, u = rng.choice(sbmew)
                    body.append(u)
                    continue
                body.append(_residue_fragment(rng, spec))
            smiles = head + "".join(body) + "9"
            tags = ["B:ClAc-bridge", "B:" + "SbMeW"]
        else:
            centre = "[C@@H]" if rng.random() < D_STEREO_PROB else "[C@H]"
            side = "CC(N)=O" if seq[0] == "N" else ""
            if seq[0] == "Sar":
                head = "N9(C)CC(=O)"
            elif seq[0] == "Nle":
                head = "N9[C@@H](CCCC)C(=O)"
            elif seq[0] == "G" or not side:
                head = "N9CC(=O)"
            elif seq[0] == "N":
                head = f"N9{centre}(CC(N)=O)C(=O)"
            else:
                head = f"N9{centre}({side})C(=O)"
            body, tags = [], []
            for p, spec in enumerate(seq[1:], start=1):
                if p in slots:
                    if family == "Q":
                        k, s = rng.choice(list(Q_ARYL_SIDES.items()))
                        body.append(_quat_unit(s, rng))
                        tags.append("Q:" + k)
                    else:
                        k, u = rng.choice(prol)
                        body.append(u)
                        tags.append("P:" + k)
                    continue
                body.append(_residue_fragment(rng, spec))
            smiles = head + "".join(body) + "9"
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol)
        if len(smi) > 1400 or smi in seen:
            continue
        seen.add(smi)
        out.append({"smiles": smi, "family": family, "tags": ";".join(tags)})
    return out


def build_heldout(rng: random.Random) -> list:
    """Same-family molecules excluded from training (pattern check)."""
    out = []
    q_side = HELDOUT["Q"]["aMeInd_5Br"]
    mol = Chem.MolFromSmiles(
        "N9CC(=O)" + _quat_unit(q_side, rng)
        + "N[C@@H](Cc1ccc(C(N)=O)cc1)C(=O)N1CC[C@@H](C)C1C(=O)"
        + "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CCC(N)=O)C(=O)N(C)CC(=O)9")
    assert mol is not None
    out.append({"smiles": Chem.MolToSmiles(mol), "family": "Q",
                "tags": "Q:aMeInd_5Br-heldout"})
    p = HELDOUT["P"]["3OHPro"]
    assert _validate(p)
    mol = Chem.MolFromSmiles(
        "N9[C@@H](CCCC)C(=O)" + p
        + "N[C@@H](Cc1c[nH]c2c(Cl)cccc12)C(=O)N[C@H](CS)C(=O)"
        + "N(C)CC(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)9")
    assert mol is not None
    out.append({"smiles": Chem.MolToSmiles(mol), "family": "P",
                "tags": "P:3OHPro-heldout"})
    b = HELDOUT["B"]["SbMeW6Cl"]
    mol = Chem.MolFromSmiles(
        "N9[C@@H](Cc1ccccc1)" + BRIDGE_SEG + b
        + "N[C@@H](CC(N)=O)C(=O)N1CCC[C@@H](C)C1C(=O)"
        + "N[C@@H](Cc1ccncc1)C(=O)9")
    assert mol is not None
    out.append({"smiles": Chem.MolToSmiles(mol), "family": "B",
                "tags": "B:ClAc-bridge+heldout-SbMeW6Cl"})
    return out


def render_rows(rows, out_dir, seed=42, variants=2, workers=6):
    """ALIGNED renders via render_hardcase_r4._render_one (module-level,
    picklable; molecule-distinct pool so the per-job depiction seed keeps
    layouts diverse).  Files are named mol_v00_<idx>.png."""
    from multiprocessing import Pool
    from render_hardcase_r4 import _render_one

    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    jobs, meta = [], []
    idx = 0
    for i, r in enumerate(rows):
        for v in range(variants):
            jobs.append((idx, r["smiles"], "mol", 0, out_dir, seed))
            meta.append({"idx": idx, "row": i, "variant": v})
            idx += 1

    rows_out = []
    with Pool(processes=workers, maxtasksperchild=8) as mp:
        asyncs = [(m["idx"], mp.apply_async(_render_one, (j,)))
                  for m, j in zip(meta, jobs)]
        for m, (jidx, a) in zip(meta, asyncs):
            try:
                _, path = a.get(timeout=300)
            except Exception as exc:
                print(f"  render fail idx={m['idx']}: {str(exc)[:80]}",
                      flush=True)
                path = None
            if path:
                rows_out.append({"file_path": path, **rows[m["row"]]})
    return rows_out


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("file_path,SMILES\n")
        for r in rows:
            smi = r["SMILES"] if "SMILES" in r else r["smiles"]
            if "," in smi or '"' in smi:
                smi = '"' + smi.replace('"', '""') + '"'
            fh.write(f"{r['file_path']},{smi}\n")


def sample_wildmol_replay(n, seed):
    """Small-molecule replay from the v10 generalist pool: regular small
    molecules (SMILES < 120 chars, RDKit-parseable, no macrocycle)."""
    df = pd.read_csv(V10_POOL)
    df = df[df["SMILES"].str.len() < 120]
    rng = random.Random(seed)
    order = list(range(len(df)))
    rng.shuffle(order)
    out = []
    for pos in order:
        if len(out) >= n:
            break
        smi = df["SMILES"].iloc[pos]
        p = df["file_path"].iloc[pos]
        if not os.path.isfile(p):
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        ri = mol.GetRingInfo()
        if max((len(r) for r in ri.AtomRings()), default=0) > 12:
            continue
        out.append({"file_path": p, "SMILES": smi, "source": "v10_smallmol"})
    return out


def sample_v10_linear(seed):
    """All v10 linear-peptide rows (>=6 amide bonds, no ring past 12)."""
    df = pd.read_csv(V10_POOL)
    df = df[df["SMILES"].str.count(r"NC\(\=O\)") >= 6]
    out = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            continue
        ri = mol.GetRingInfo()
        if max((len(r) for r in ri.AtomRings()), default=0) > 12:
            continue
        if not os.path.isfile(row["file_path"]):
            continue
        out.append({"file_path": row["file_path"], "SMILES": row["SMILES"],
                    "source": "v10_linear"})
    return out


def generate_linear_pool(n, seed):
    """Fresh linear chains from the cyclopeptide grammar (rendered here)."""
    from cyclopeptide_generator import generate_linear
    rng = random.Random(seed)
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * 10:
        tries += 1
        smi = generate_linear(rng)
        if not smi or len(smi) > 1200 or smi in seen:
            continue
        mol = Chem.MolFromSmiles(smi)
        ri = mol.GetRingInfo()
        if max((len(r) for r in ri.AtomRings()), default=0) > 12:
            continue
        seen.add(smi)
        out.append({"smiles": smi, "family": "linear_gen",
                    "tags": "linear:generated"})
    return out


def sample_markush(n, eval_n, seed):
    """Markush R-group/attachment replay from the tier2 pool, plus a
    disjoint eval subset (never trained) for the before/after check."""
    df = pd.read_csv(MARKUSH_POOL)
    df = df[df["SMILES"].str.contains(r"\*|\[\d+\*\]", regex=True)]
    rng = random.Random(seed)
    order = list(range(len(df)))
    rng.shuffle(order)
    replay, evalset = [], []
    for pos in order:
        if len(replay) >= n and len(evalset) >= eval_n:
            break
        smi = df["SMILES"].iloc[pos]
        p = df["file_path"].iloc[pos]
        if not os.path.isfile(p) or Chem.MolFromSmiles(smi) is None:
            continue
        item = {"file_path": p, "SMILES": smi, "source": "tier2_markush"}
        if len(replay) < n:
            replay.append(item)
        elif len(evalset) < eval_n:
            evalset.append(item)
    return replay, evalset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--n_per_family", type=int, default=12)
    ap.add_argument("--n_heldout_variants", type=int, default=2)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--wildmol_replay", type=int, default=1500)
    ap.add_argument("--markush_replay", type=int, default=2000)
    ap.add_argument("--markush_eval_n", type=int, default=200)
    ap.add_argument("--linear_gen", type=int, default=450)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # motif families (train side)
    family_rows = []
    for fam in ("Q", "P", "B"):
        mols = build_family_molecules(fam, args.n_per_family, rng)
        print(f"family {fam}: {len(mols)} molecules", flush=True)
        family_rows.extend(mols)
    rendered = render_rows(family_rows, os.path.join(args.out, "families"),
                           seed=args.seed, variants=args.variants,
                           workers=args.workers)
    write_csv(os.path.join(args.out, "families", "train_680k.csv"), rendered)
    print(f"family renders kept: {len(rendered)}/"
          f"{len(family_rows) * args.variants}", flush=True)

    # held-out same-family molecules (never trained)
    held = build_heldout(rng)
    held_rendered = render_rows(held, os.path.join(args.out, "heldout_motifs"),
                                seed=args.seed + 1,
                                variants=args.n_heldout_variants,
                                workers=2)
    write_csv(os.path.join(args.out, "heldout_motifs", "heldout_680k.csv"),
              held_rendered)
    print(f"held-out renders kept: {len(held_rendered)}/{len(held) * args.n_heldout_variants}",
          flush=True)

    # replay domains
    wild = sample_wildmol_replay(args.wildmol_replay, args.seed)
    write_csv(os.path.join(args.out, "wildmol_replay.csv"), wild)
    mk_replay, mk_eval = sample_markush(args.markush_replay, args.markush_eval_n,
                                        args.seed)
    write_csv(os.path.join(args.out, "markush_replay.csv"), mk_replay)
    ev_dir = os.path.join(args.out, "markush_eval")
    os.makedirs(ev_dir, exist_ok=True)
    with open(os.path.join(ev_dir, "pairs_markush200.json"), "w") as fh:
        json.dump([{"name": f"mk{i:03d}", "image": r["file_path"],
                    "gold": r["SMILES"]}
                   for i, r in enumerate(mk_eval)], fh, indent=1)
    v10lin = sample_v10_linear(args.seed)
    lin_gen = generate_linear_pool(args.linear_gen, args.seed + 2)
    lin_rendered = render_rows(lin_gen, os.path.join(args.out, "linear_gen"),
                               seed=args.seed + 3, variants=args.variants,
                               workers=args.workers)
    lin_all = v10lin + [{"file_path": r["file_path"], "SMILES": r["smiles"],
                         "source": "linear_gen"} for r in lin_rendered]
    write_csv(os.path.join(args.out, "linear_replay.csv"), lin_all)
    print(f"replay: wildmol={len(wild)} markush={len(mk_replay)} "
          f"markush_eval={len(mk_eval)} v10_linear={len(v10lin)} "
          f"linear_gen_rendered={len(lin_rendered)}", flush=True)

    manifest = {
        "purpose": "round-4 motif-family expansion + anti-forgetting replay",
        "families": {
            "Q_aryl": [r["tags"] for r in family_rows if r["family"] == "Q"],
            "P_sub": [r["tags"] for r in family_rows if r["family"] == "P"],
            "bridge": [r["tags"] for r in family_rows if r["family"] == "B"],
        },
        "heldout": held,
        "counts": {
            "family_molecules": len(family_rows),
            "family_rows_rendered": len(rendered),
            "heldout_molecules": len(held),
            "heldout_rows_rendered": len(held_rendered),
            "wildmol_replay_rows": len(wild),
            "markush_replay_rows": len(mk_replay),
            "markush_eval_rows": len(mk_eval),
            "v10_linear_rows": len(v10lin),
            "linear_gen_rows_rendered": len(lin_rendered),
        },
        "seed": args.seed,
    }
    with open(os.path.join(args.out, "families_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(manifest["counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
