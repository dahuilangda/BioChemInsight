"""Round-3 synthetic-pool vocabulary expansion (real-domain SFT, round 3).

Round-2 diagnosis (FINAL_SUMMARY_R2.json: 6-figure mean 0.9045, gate 0.95
missed) located the bottleneck at content level: the synthetic vocabulary
lacks three motif classes that dominate the chronic failure cases
(page_260_box_1 / page_219_box_3, both fluorinated KRAS crops scoring 0.0):

  1. fluoro side chains      — aryl-F (Cc1ccccc1F), terminal-fluoroalkyl
                               (CCCF family), CF3, 4-fluoroproline
  2. quaternary chiral C     — [C@@](C)(aryl) alpha-methyl aryl acids
                               (methyl mis-attribution) plus the spiro
                               quaternary ester of the KRAS scaffold
  3. 2-substituted proline   — C2[C@@H](C)CC[C@H]2N class (methyl aminocyclo
                               pentane carboxamide / methylprolines) that
                               round-2 decoded as plain proline

Expansion sources
  A. every verified unique KRAS manifest molecule (121, the exact chemistry
     of the failing domain; 55 contain F, 84 contain a quaternary centre)
  B. motif-forced cyclopeptides: the round-2 grammar with 1-3 forced
     fragments per molecule drawn from the three new residue families

Output: <out>/vocab_r3_pool.csv (file_path,SMILES[,tags]) — the file_path
column points at the source manifest so render_aligned_pool.py's existence
filter keeps the rows; rendering uses only the SMILES column.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

from rdkit import Chem, RDLogger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RDLogger.DisableLog("rdApp.*")

MANIFEST = "/tmp/crops_kras/real_crops_manifest.json"

# --------------------------------------------------------------------------
# 1. fluoro side-chain residues (side chain written inside N[...](...)C(=O))
# --------------------------------------------------------------------------
FLUORO_SIDES = {
    "pF-Phe": "Cc1ccc(F)cc1",
    "mF-Phe": "Cc1cccc(F)c1",
    "oF-Phe": "Cc1c(F)cccc1",
    "diF-Phe": "Cc1ccc(F)c(F)c1",
    "CF3-Phe": "Cc1ccc(C(F)(F)F)cc1",
    "F-Tyr": "Cc1ccc(O)c(F)c1",
    "4Fbutyl": "CCCF",          # terminal-fluoroalkyl (diagnosis: CCCF)
    "5Fpentyl": "CCCCF",
    "3Fisoamyl": "CCC(F)C",
    "4Fisoamyl": "CC(F)CC",
    "FCh2cyclopentyl": "CC1CCC(F)C1",
}

# --------------------------------------------------------------------------
# 2. quaternary chiral centres: N[C@@](C)(side)C(=O), alpha-methyl, no C-H
# --------------------------------------------------------------------------
QUAT_UNITS = {
    "aMePhe": "Cc1ccccc1",
    "aMepFPhe": "Cc1ccc(F)cc1",
    "aMeTyr": "Cc1ccc(O)cc1",
    "aMeTrp_bnz": "Cc1c[nH]c2ccccc12",       # benzylic indole
    "aMeTrp_direct": "c1c[nH]c2ccccc12",     # direct aryl (fig_s52 gold class)
    "aMeNle": "CCCC",
    "aMeAsn": "CC(N)=O",
    "aMeMeCys": "CSC",
    "aMeAbg": "c1ccc2ccccc2c1",              # alpha-methyl naphthylalanine
}

# spiro quaternary ester — the KRAS scaffold signature C2C3CC2(...)(C3)
# carried as an O-ester side chain on a Ser-type residue
SPIRO_SIDES = {
    "spiroMe": "COC(=O)C1C2CC1(C)C2",
    "spiroH": "COC(=O)C1C2CC1C2",
    "spiroMeCOmide": "COC(=O)C1C2CC1(CNC(=O)C)C2",
}

# --------------------------------------------------------------------------
# 3. 2-substituted proline analogs (ring: gold C2[C@@H](C)CC[C@H]2N class)
#    written in generator convention so the amide N precedes the ring
# --------------------------------------------------------------------------
PROLINE_ANALOG_UNITS = {
    "MeACPC_1": "N[C@H]1CC[C@@H](C)C1C(=O)",      # gold topology, methyl beta
    "MeACPC_2": "N[C@H]1CC[C@H](C)C1C(=O)",
    "MeACPC_3": "N[C@@H]1CC[C@H](C)C1C(=O)",
    "5MePro": "N1CC[C@H](C)C[C@@H]1C(=O)",
    "4MePro": "N1C[C@@H](C)C[C@H]1C(=O)",
    "3MePro": "N1C[C@H](C)CC[C@H]1C(=O)",
    "4FPro": "N1CC[C@H](F)C[C@@H]1C(=O)",         # fluorinated proline
    "4OHPro": "N1CC[C@H](O)C[C@@H]1C(=O)",
    "2MePro_quat": "N1CCC[C@@]1(C)C(=O)",         # quaternary proline C-alpha
    "5MePro_Nm": "CN1CC[C@H](C)C[C@@H]1C(=O)",    # N-methyl-5-methylproline
}


def _validate(name, frag, n_expected_ring_atoms=None):
    """A residue fragment is legal if it composes into a dipeptide."""
    probe = "CC(=O)" + frag + "NCC(=O)O"
    mol = Chem.MolFromSmiles(probe)
    return mol is not None


def build_synthetic(rng: random.Random, n_target: int) -> list[dict]:
    """Motif-forced cyclopeptides: same backbone grammar as round-2
    (cyclopeptide_generator) but 1-3 positions per molecule are forced to
    the three new residue families, so every molecule carries >=1 of the
    diagnosed patterns."""
    from cyclopeptide_generator import (
        _weighted_letter, _residue_fragment, _ring_size,
        _draw_molecule_motif, _motif_or_residue,
        TRP_VARIANTS, TRP_WEIGHTS, ARYL_PHE, D_STEREO_PROB, STANDARD)

    fluo = [(k, v) for k, v in FLUORO_SIDES.items()
            if _validate(k, f"N[C@H]({v})C(=O)")]
    quat = [(k, v) for k, v in QUAT_UNITS.items()
            if _validate(k, f"N[C@@]({v})(C)C(=O)")]
    spiro = [(k, v) for k, v in SPIRO_SIDES.items()
             if _validate(k, f"N[C@@H]({v})C(=O)")]
    prol = [(k, v) for k, v in PROLINE_ANALOG_UNITS.items()
            if _validate(k, v)]
    print(f"validated fragments: fluoro={len(fluo)} quat={len(quat)} "
          f"spiro={len(spiro)} proline={len(prol)}", flush=True)

    def fluoro_residue(key, side):
        # quaternary F-bearing carbons also live in this class via CF3
        centre = "[C@@H]" if rng.random() < D_STEREO_PROB else "[C@H]"
        return f"N{centre}({side})C(=O)"

    def quat_residue(key, side):
        w = "[C@@]" if rng.random() < 0.5 else "[C@]"
        return f"N{w}({side})(C)C(=O)"

    def spiro_residue(key, side):
        centre = "[C@@H]" if rng.random() < D_STEREO_PROB else "[C@H]"
        return f"N{centre}({side})C(=O)"

    out, seen = [], set()
    tries = 0
    while len(out) < n_target and tries < n_target * 30:
        tries += 1
        size = _ring_size(rng)
        seq = [_weighted_letter(rng) for _ in range(size)]
        if seq[0] in ("P", "HypOCM", "PipOH"):
            seq[0] = "S"

        # pick 1-3 forced motif slots (head slot 0 is never forced — the
        # head grammar only supports a subset of residues)
        n_forced = rng.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
        n_forced = min(n_forced, size - 1)
        slots = rng.sample(range(1, size), n_forced)

        head_spec = seq[0]
        n0 = "N9"
        centre = "[C@@H]" if rng.random() < D_STEREO_PROB else "[C@H]"
        side = STANDARD.get(head_spec, "")
        if head_spec == "Sar":
            head = f"{n0}(C)CC(=O)"
        elif head_spec == "Nle":
            head = f"{n0}[C@@H](CCCC)C(=O)"
        elif head_spec == "ACPC":
            head = f"{n0}[C@H]1CC[C@H](C1)C(=O)"
        elif head_spec == "W":
            head = f"{n0}{centre}({TRP_VARIANTS['W']})C(=O)"
        elif head_spec == "G":
            head = f"{n0}CC(=O)"
        elif head_spec in ARYL_PHE:
            head = f"{n0}{centre}({ARYL_PHE[head_spec]})C(=O)"
        elif head_spec == "N":
            head = f"{n0}[C@@H](CC(N)=O)C(=O)"
        elif not side:
            head = f"{n0}CC(=O)"
        else:
            head = f"{n0}{centre}({side})C(=O)"

        motif_slot, unit = _draw_molecule_motif(rng, len(seq) - 1)
        tags = []
        body = []
        for pos, spec in enumerate(seq[1:]):
            p = pos + 1
            if motif_slot >= 0 and pos == motif_slot:
                body.append(_motif_or_residue(rng, spec, motif_slot, pos, unit))
                continue
            if p in slots:
                klass = rng.choices(["F", "Q", "P"],
                                    weights=[0.34, 0.33, 0.33])[0]
                if klass == "F":
                    k, s = rng.choice(fluo)
                    body.append(fluoro_residue(k, s))
                    tags.append("F:" + k)
                elif klass == "Q":
                    if rng.random() < 0.25 and spiro:
                        k, s = rng.choice(spiro)
                        body.append(spiro_residue(k, s))
                        tags.append("Q:spiro-" + k)
                    else:
                        k, s = rng.choice(quat)
                        body.append(quat_residue(k, s))
                        tags.append("Q:" + k)
                else:
                    k, s = rng.choice(prol)
                    body.append(s)
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
        out.append({"smiles": smi, "source": "motif_forced",
                    "tags": ";".join(tags)})
    return out


def classify_real(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    tags = []
    if any(a.GetSymbol() == "F" for a in mol.GetAtoms()):
        tags.append("F:real")
    for a in mol.GetAtoms():
        if (a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                and a.GetDegree() == 4 and a.GetTotalNumHs() == 0):
            tags.append("Q:real")
            break
    else:
        for a in mol.GetAtoms():
            n_rings = len([r for r in mol.GetRingInfo().AtomRings()
                           if a.GetIdx() in r])
            if a.GetDegree() == 4 and n_rings >= 2:
                tags.append("Q:spiro-real")
                break
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) == 5:
            syms = [mol.GetAtomWithIdx(i).GetSymbol() for i in ring]
            if syms.count("N") >= 1 and any(
                    any(x.GetSymbol() == "C" and x.GetDegree() == 1
                        for x in mol.GetAtomWithIdx(i).GetNeighbors())
                    for i in ring):
                tags.append("P:real")
                break
    return ";".join(tags)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--out", default="/data/molsight_realpatent_sft_r3/vocab_r3_pool.csv")
    ap.add_argument("--n_synth", type=int, default=350)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    real = {}
    for r in man["rows"]:
        if r.get("smiles"):
            real.setdefault(r["smiles"], r["crop"])
    rng = random.Random(args.seed)
    synth = build_synthetic(rng, args.n_synth)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    counts = {"F": 0, "Q": 0, "P": 0}
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file_path", "SMILES", "source", "tags"])
        for smi, crop in real.items():
            tags = classify_real(smi)
            for t in filter(None, tags.split(";")):
                counts[t[0]] += 1
            w.writerow([os.path.join(os.path.dirname(args.manifest), crop),
                        smi, "real_kras", tags])
            n += 1
        for row in synth:
            for t in filter(None, row["tags"].split(";")):
                counts[t[0]] += 1
            # file_path must exist for render_aligned_pool.py's row filter;
            # rendering consumes only the SMILES column
            w.writerow([args.manifest, row["smiles"], row["source"], row["tags"]])
            n += 1
    print(json.dumps({
        "pool_csv": args.out, "pool_rows": n,
        "real_kras_rows": len(real), "motif_forced_rows": len(synth),
        "motif_tag_instances": counts,
    }, indent=2))


if __name__ == "__main__":
    main()
