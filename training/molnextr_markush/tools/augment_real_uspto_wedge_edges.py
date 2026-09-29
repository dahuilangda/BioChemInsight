#!/usr/bin/env python
"""In-place wedge-edge (stereochemistry) augmentation for the real-USPTO complete shard.

Reads training/molnextr_markush/data/generated/real_uspto_complete/data.parquet,
injects bond-direction supervision into the `edges` column and writes the parquet
back. The pre-augmentation content is preserved once in data.parquet.bak_nowedge;
re-running restores from that backup first, so the tool is idempotent over the same
original data (wedges are always derived from the pristine wedge-free backup).

Edge classes follow the dataset consumer convention (utils/MolNexTR/dataset.py,
`_process_atomtok_coords`): [u, v, 5] -> matrix[u][v]=5, matrix[v][u]=6, i.e. u is
the wedge origin (the stereo-center side the wedge is drawn from). Source of truth
is the MOL V2000 bond-block CFG column: 1 = up wedge from the bond's first atom,
6 = down hash, 4 = crossed (ignored).

Per-row pipeline:
  1. file_path (absolute .TIF) -> source CSV row -> mol_path (relative to raw root).
  2. Cheap text scan of the MOL bond block; rows without any CFG in {1, 6} skip RDKit.
  3. Chem.MolFromMolBlock(sanitize=False, removeHs=False) + manual SanitizeMol.
     Explicit Hs are kept at their original block indices because wedge endpoints
     are raw block indices; failures skip the row's wedges.
  4. Alignment gate: canonical(MOL, stereo removed) == canonical(row SMILES, stereo
     removed), otherwise the row keeps its original edges untouched.
  5. Every CFG wedge (b, e, cfg) is mapped through ALL substructure matches
     (mol_full.GetSubstructMatches(canon_mol, useChirality=False, uniquify=False));
     all matches must agree on the SAME ordered pair (mb, me). Any disagreement on
     any wedge abandons ALL of the row's wedges (edges revert to pure t=1).
  6. The t=1 entry of the mapped bond is replaced in place by [mb, me, 5|6]
     (CFG 1 -> 5 up, CFG 6 -> 6 down); no duplicate entries are left behind.

Quality gate: every injected row is cross-checked against the row SMILES'
stereochemistry by re-reading the MOL with RDKit's default parser (wedge + coords
-> @, removeHs): B = CIP(R/S) multiset of the MOL parse == CIP of the row SMILES.
If B consistency over verifiable injected stereo rows ("@" in SMILES) is below
--gate (0.95), row-level filtering turns on: only rows passing B keep their wedges.

Two additional sample-based (--sample-size, default 300) verifications are reported:
  A  = consumer-style replay with a FRESH rdDepictor.Compute2DCoords layout.
  A2 = the same replay with the MOL's own 2D coordinates transferred onto the
       SMILES atom order (isolates the injected edges themselves from layout
       randomness; mirrors the _verify_chirality core in utils/MolNexTR/chemical.py:
       BondDir -> AssignChiralTypesFromBondDirs -> AssignStereochemistry).
A is expected to be layout-handedness limited (a wedge keeps meaning "toward
viewer" while a re-generated layout may be mirrored, flipping every center it
re-hands), so it is reported as a lower bound; A2 and B are the data-quality
metrics, and B is the gate.

Invariants enforced: row count, row order, SMILES and file_path columns are
byte-identical before/after; only `edges` strings of finally-injected rows change.
"""
from __future__ import annotations

import argparse
import ast
import json
import multiprocessing
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PARQUET = ROOT / "training/molnextr_markush/data/generated/real_uspto_complete/data.parquet"
DEFAULT_CSV = ROOT / "training/molnextr_markush/data/raw/molscribe/uspto_mol/train_680k.csv"
DEFAULT_RAW_ROOT = ROOT / "training/molnextr_markush/data/raw/molscribe"
BACKUP_SUFFIX = ".bak_nowedge"

# Row statuses (mutually exclusive).
S_UNMAPPED = "file_path_not_in_csv"
S_NO_MOL = "mol_missing"
S_UNREADABLE = "molblock_unreadable"
S_V3000 = "v3000_unsupported"
S_MALFORMED = "molblock_malformed"
S_NO_CFG = "no_cfg_wedge"
S_PARSE_FAIL = "rdkit_parse_or_sanitize_failed"
S_CANON_MISMATCH = "canonical_mismatch"
S_AMBIGUOUS = "wedge_mapping_ambiguous"
S_CAP_HIT = "match_cap_hit"
S_CONFLICT = "wedge_direction_conflict"
S_INJECTED_NONE = "all_wedges_dropped"
S_EDGES_UNPARSEABLE = "edges_json_unparseable"
S_CLASS_ANOMALY = "wedge_class_anomaly"
S_OK = "injected"


def _parse_v2000_wedges(text: str):
    """Scan a V2000 MOL block's bond block.

    Returns (wedges, malformed_bond_lines) where wedges = [(b, e, bond_type, cfg)]
    with 0-based atom indices, or (None, 0) when the block is V3000, and raises
    ValueError on unreadable counts lines / truncated blocks.
    """
    lines = text.splitlines()
    if len(lines) < 5:
        raise ValueError("short file")
    for probe in (lines[3], lines[4] if len(lines) > 4 else ""):
        if "V3000" in probe:
            return None, 0
    counts = lines[3]
    try:
        n_atoms = int(counts[0:3])
        n_bonds = int(counts[3:6])
    except ValueError:
        raise ValueError("bad counts line")
    if n_atoms < 0 or n_bonds < 0 or 4 + n_atoms + n_bonds > len(lines) + 8:
        raise ValueError("truncated block")
    wedges = []
    malformed = 0
    for i in range(4 + n_atoms, 4 + n_atoms + n_bonds):
        if i >= len(lines):
            break
        line = lines[i]
        if len(line) < 12:
            if line.strip():
                malformed += 1
            continue
        try:
            cfg = int(line[9:12])
        except ValueError:
            malformed += 1
            continue
        if cfg not in (1, 6):
            continue
        try:
            b = int(line[0:3]) - 1
            e = int(line[3:6]) - 1
            btype = int(line[6:9])
        except ValueError:
            malformed += 1
            continue
        if b < 0 or e < 0:
            malformed += 1
            continue
        wedges.append((b, e, btype, cfg))
    return wedges, malformed


def _nostereo_smiles(mol) -> str:
    from rdkit import Chem

    copy = Chem.Mol(mol)
    Chem.RemoveStereochemistry(copy)
    return Chem.MolToSmiles(copy)


_NOSTEREO_CACHE: dict[str, tuple[str, list[int]]] = {}


def _nostereo_canonical_of_str(smiles: str):
    """(nostereo canonical SMILES, canonical write order) for a SMILES string.

    The write order comes from RDKit's `_smilesAtomOutputOrder` property and is
    cached per string (USPTO lots repeat structures heavily).
    """
    from rdkit import Chem

    hit = _NOSTEREO_CACHE.get(smiles)
    if hit is None:
        mol = Chem.MolFromSmiles(smiles)
        hit = ("", [])
        if mol is not None:
            copy = Chem.Mol(mol)
            Chem.RemoveStereochemistry(copy)
            smi = Chem.MolToSmiles(copy)
            order = list(ast.literal_eval(copy.GetProp("_smilesAtomOutputOrder")))
            hit = (smi, order)
        if len(_NOSTEREO_CACHE) < 100000:
            _NOSTEREO_CACHE[smiles] = hit
    return hit


def _ordered_bond_classes(mol):
    """Symmetry-class -> ordered atom pairs, for both bond orientations.

    CanonicalRankAtoms(breakTies=False, includeChirality=False) yields
    isomorphism-invariant symmetry classes, so a wedge's ordered class pair
    determines a superset of its possible images under all graph isomorphisms.
    """
    from rdkit import Chem

    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False,
                                         includeChirality=False))
    table: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for bond in mol.GetBonds():
        x, y = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        table.setdefault((ranks[x], ranks[y]), []).append((x, y))
        table.setdefault((ranks[y], ranks[x]), []).append((y, x))
    return ranks, table


def _cip_signature(mol):
    """Sorted multiset of CIP codes (R/S) — layout- and atom-order-independent."""
    from rdkit import Chem

    copy = Chem.Mol(mol)
    Chem.AssignStereochemistry(copy, force=True, cleanIt=True)
    return tuple(sorted(a.GetProp("_CIPCode") for a in copy.GetAtoms()
                        if a.HasProp("_CIPCode")))


def _load_aligned(mol_path: str, smiles: str, max_matches: int):
    """Parse the MOL (Hs kept), align it to the row SMILES graph, and map wedges.

    Alignment never uses substructure matching: for mixtures with many
    identical fragments (common in these USPTO crops) RDKit's matcher explodes
    combinatorially even for maxMatches=1 (disconnected query). Instead the
    nostereo canonical SMILES write order provides the correspondence: both
    molecules canonicalize to the SAME string (the alignment gate), so write
    position p of that string identifies one atom in each molecule, and the
    induced bijection is an isomorphism.

    A wedge's image under ALL isomorphisms is the orbit of its mapped bond
    under Aut(canon_mol). RDKit symmetry classes computed WITHIN canon_mol
    coarsen those orbits, so a unique ordered class pair proves a unique image
    (sound). Class-ambiguous wedges in single-fragment molecules fall back to
    the exhaustive GetSubstructMatches(uniquify=False) check; for
    multi-fragment molecules the exhaustive check is skipped (it can hang for
    minutes on identical-solvent mixtures) and the row is conservatively
    abandoned — for duplicate fragments this is a true ambiguity anyway.

    Returns (mol_full, canon_mol, full_to_canon, mapped, h_dropped, wedges)
    where mapped = {(b, e, cfg): (mb, me)} and full_to_canon maps mol_full
    heavy-atom indices to canon_mol indices; or a status string.
    """
    from rdkit import Chem

    try:
        with open(mol_path, "r", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return S_NO_MOL

    wedges, _malformed = _parse_v2000_wedges(text)
    if wedges is None:
        return S_V3000
    wedges = [w for w in wedges if w[2] == 1]  # wedges only live on single bonds
    if not wedges:
        return S_NO_CFG

    mol_full = Chem.MolFromMolBlock(text, sanitize=False, removeHs=False)
    if mol_full is None:
        return S_UNREADABLE
    try:
        Chem.SanitizeMol(mol_full)
    except Exception:
        return S_PARSE_FAIL

    canon_mol = Chem.MolFromSmiles(smiles)
    if canon_mol is None:
        return S_PARSE_FAIL

    for atom in mol_full.GetAtoms():
        atom.SetProp("_origIdx", str(atom.GetIdx()))
    if any(a.GetAtomicNum() == 1 for a in mol_full.GetAtoms()):
        try:
            nohs = Chem.RemoveHs(Chem.Mol(mol_full))
        except Exception:
            nohs = None
        if nohs is None:
            nohs = mol_full
    else:
        nohs = mol_full

    canon_smi, canon_order = _nostereo_canonical_of_str(smiles)
    try:
        mol_copy = Chem.Mol(nohs)
        Chem.RemoveStereochemistry(mol_copy)
        mol_smi = Chem.MolToSmiles(mol_copy)
        mol_order = list(ast.literal_eval(mol_copy.GetProp("_smilesAtomOutputOrder")))
    except Exception:
        return S_PARSE_FAIL
    if mol_smi != canon_smi or not canon_order or len(mol_order) != len(canon_order):
        return S_CANON_MISMATCH

    # bijection mol_full heavy atom -> canon_mol atom, via shared write positions
    full_to_canon = {}
    for pos, nohs_idx in enumerate(mol_order):
        if nohs_idx >= nohs.GetNumAtoms():
            return S_PARSE_FAIL
        orig = int(nohs.GetAtomWithIdx(nohs_idx).GetProp("_origIdx"))
        full_to_canon[orig] = canon_order[pos]

    canon_ranks, canon_classes = _ordered_bond_classes(canon_mol)
    single_fragment = len(Chem.GetMolFrags(canon_mol)) == 1

    mapped = {}
    h_dropped = 0
    ambiguous = None
    for b, e, _btype, cfg in wedges:
        mb, me = full_to_canon.get(b), full_to_canon.get(e)
        if mb is None or me is None:
            h_dropped += 1  # wedge on an explicit H (removed) — not representable
            continue
        if canon_mol.GetBondBetweenAtoms(mb, me) is None:
            return S_CLASS_ANOMALY
        if len(canon_classes.get((canon_ranks[mb], canon_ranks[me]), ())) == 1:
            mapped[(b, e, cfg)] = (mb, me)
        else:
            ambiguous = (b, e, cfg)
            break

    if ambiguous is not None:
        if not single_fragment:
            # mixtures: the exhaustive automorphism check can hang; abandon
            return S_AMBIGUOUS
        try:
            matches = mol_full.GetSubstructMatches(
                canon_mol, useChirality=False, uniquify=False, maxMatches=max_matches)
        except Exception:
            return S_PARSE_FAIL
        if not matches:
            return S_CANON_MISMATCH
        if len(matches) >= max_matches:
            return S_CAP_HIT
        inverses = [{t: q for q, t in enumerate(m)} for m in matches]
        mapped = {}
        for b, e, _btype, cfg in wedges:
            pairs = {(inv.get(b), inv.get(e)) for inv in inverses}
            if any(p is None or q is None for p, q in pairs):
                h_dropped += 1
                continue
            if len(pairs) != 1:
                return S_AMBIGUOUS
            mapped[(b, e, cfg)] = pairs.pop()

    return mol_full, canon_mol, full_to_canon, mapped, h_dropped, wedges


def _map_wedges(matches, wedges):
    """Legacy full-enumeration mapping, kept for equivalence self-checks."""
    inverses = [{t: q for q, t in enumerate(m)} for m in matches]
    mapped = {}
    h_dropped = 0
    for b, e, _btype, cfg in wedges:
        pairs = {(inv.get(b), inv.get(e)) for inv in inverses}
        if any(p is None or q is None for p, q in pairs):
            h_dropped += 1
            continue
        if len(pairs) != 1:
            return {}, h_dropped, True
        mapped[(b, e, cfg)] = pairs.pop()
    return mapped, h_dropped, False


def _replay_chirality(smiles: str, wedge_edges, coords):
    """_verify_chirality-core replay of injected edges on a fresh SMILES mol.

    wedge_edges = [(u, v, 5|6)] (u = wedge origin). coords = None -> fresh
    Compute2DCoords layout (verification A); coords = list[(x, y)] -> MOL-derived
    2D coordinates in SMILES atom order (verification A2).
    """
    from rdkit import Chem

    orig = Chem.MolFromSmiles(smiles)
    if orig is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    rw = Chem.RWMol(mol)
    for u, v, code in wedge_edges:
        bond = rw.GetBondBetweenAtoms(u, v)
        if bond is None:
            continue
        if bond.GetBeginAtomIdx() != u:
            # re-create the bond so that its begin atom is the wedge origin u
            rw.RemoveBond(u, v)
            rw.AddBond(u, v, Chem.BondType.SINGLE)
            bond = rw.GetBondBetweenAtoms(u, v)
        bond.SetBondDir(Chem.BondDir.BEGINWEDGE if code == 5 else Chem.BondDir.BEGINDASH)
    mol = rw.GetMol()
    if coords is None:
        from rdkit.Chem import rdDepictor

        rdDepictor.Compute2DCoords(mol)
    else:
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.Set3D(False)
        for i, (x, y) in enumerate(coords):
            conf.SetAtomPosition(i, (float(x), float(y), 0.0))
        mol.AddConformer(conf)
    try:
        Chem.SanitizeMol(mol)
        Chem.AssignChiralTypesFromBondDirs(mol)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    except Exception:
        return None
    return _cip_signature(mol) == _cip_signature(orig)


def _worker_init():
    """Pool worker setup: silence RDKit and disable cyclic GC.

    Workers inherit the parent's multi-GB heap (dataframe + 680k-entry path map)
    via fork; as their own caches grow, gen-2 GC cycles rescan all of it and
    throughput collapses to ~zero after a few hundred chunks. RDKit batch
    workloads are refcount-clean, so cyclic GC is safe to disable here.
    """
    import gc

    gc.disable()
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")


def process_rows(payload):
    """Worker: (idx, mol_abs, smiles, edges_str) -> result tuples."""
    out = []
    for idx, mol_abs, smiles, edges_str, max_matches in payload:
        out.append(process_one(idx, mol_abs, smiles, edges_str, max_matches))
    return out


def process_one(idx, mol_abs, smiles, edges_str, max_matches):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    res = {
        "idx": idx, "status": S_NO_MOL, "new_edges": None,
        "up": 0, "down": 0, "cfg_wedges": 0, "h_dropped": 0,
        "origin_at_center": 0, "origin_at_other_center_end": 0, "origin_not_center": 0,
        "b_pass": None, "b_full_smiles_eq": None, "has_at": "@" in smiles,
        "n_stereo_centers": 0,
    }
    if mol_abs is None:
        res["status"] = S_UNMAPPED
        return _pack(res)

    aligned = _load_aligned(mol_abs, smiles, max_matches)
    if isinstance(aligned, str):
        res["status"] = aligned
        return _pack(res)
    mol_full, canon_mol, _full_to_canon, mapped, h_dropped, wedges = aligned

    res["cfg_wedges"] = len(wedges)
    tagged = {a.GetIdx() for a in canon_mol.GetAtoms()
              if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED}
    res["n_stereo_centers"] = len(tagged)
    res["h_dropped"] = h_dropped

    # target wedge directions on the SMILES graph, keyed by ordered pair
    directed = {}
    for (b, e, cfg), (mb, me) in mapped.items():
        code = 5 if cfg == 1 else 6
        if (mb, me) in directed and directed[(mb, me)] != code:
            res["status"] = S_CONFLICT
            return _pack(res)
        directed[(mb, me)] = code

    try:
        edges = json.loads(edges_str)
    except (ValueError, TypeError):
        res["status"] = S_EDGES_UNPARSEABLE  # keep original edges string untouched
        return _pack(res)
    if any(t > 4 for _u, _v, t in edges):
        res["status"] = S_NO_CFG  # pre-existing directed edges: leave untouched
        return _pack(res)

    pair_lookup = {frozenset((mb, me)): (mb, me, code) for (mb, me), code in directed.items()}
    new_list = []
    replaced = set()
    for u, v, t in edges:
        key = frozenset((u, v))
        hit = pair_lookup.get(key)
        if hit is None or t != 1:
            new_list.append([u, v, t])
            continue
        mb, me, code = hit
        new_list.append([mb, me, code])
        replaced.add((mb, me))

    if not replaced:
        res["status"] = S_INJECTED_NONE
        return _pack(res)
    for (mb, me) in replaced:
        if mb in tagged:
            res["origin_at_center"] += 1
        elif me in tagged:
            res["origin_at_other_center_end"] += 1
        else:
            res["origin_not_center"] += 1
        if directed[(mb, me)] == 5:
            res["up"] += 1
        else:
            res["down"] += 1
    res["status"] = S_OK
    res["new_edges"] = json.dumps(new_list)

    # Verification B: RDKit's own chirality-aware MOL read vs the row SMILES.
    try:
        with open(mol_abs, "r", errors="replace") as handle:
            text = handle.read()
        mb_mol = Chem.MolFromMolBlock(text)  # sanitize=True, removeHs=True
    except OSError:
        mb_mol = None
    if mb_mol is not None:
        try:
            res["b_pass"] = _cip_signature(mb_mol) == _cip_signature(canon_mol)
            res["b_full_smiles_eq"] = (
                Chem.MolToSmiles(mb_mol) == Chem.MolToSmiles(canon_mol))
        except Exception:
            pass
    return _pack(res)


def _pack(res):
    return (
        res["idx"], res["status"], res["new_edges"], res["up"], res["down"],
        res["cfg_wedges"], res["h_dropped"], res["origin_at_center"],
        res["origin_at_other_center_end"], res["origin_not_center"],
        res["b_pass"], res["b_full_smiles_eq"], res["has_at"],
        res["n_stereo_centers"],
    )


def verify_sample_worker(payload):
    """Worker for the A/A2 sample verification: (idx, mol_abs, smiles, new_edges)."""
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    idx, mol_abs, smiles, new_edges, max_matches = payload
    out = {"idx": idx, "a_pass": None, "a2_pass": None}
    try:
        wedge_edges = [(u, v, t) for u, v, t in json.loads(new_edges) if t in (5, 6)]
        aligned = _load_aligned(mol_abs, smiles, max_matches)
        if isinstance(aligned, str):
            return idx, out
        mol_full, _canon, full_to_canon, _mapped, _h, _wedges = aligned
        conf = mol_full.GetConformer()
        canon_to_full = {q: t for t, q in full_to_canon.items()}
        n_atoms = _canon.GetNumAtoms()
        coords = []
        for q in range(n_atoms):
            f = canon_to_full.get(q)
            if f is None:
                coords = None
                break
            p = conf.GetAtomPosition(f)
            coords.append((p.x, p.y))
        out["a_pass"] = _replay_chirality(smiles, wedge_edges, None)
        if coords is not None:
            out["a2_pass"] = _replay_chirality(smiles, wedge_edges, coords)
    except Exception:
        pass
    return idx, out


def _rate(passed, total):
    return round(passed / total, 4) if total else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--max-matches", type=int, default=20000)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--gate", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=0,
                        help="process only the first N parquet rows (0 = all; "
                             "for smoke tests, report numbers are then partial)")
    parser.add_argument("--dry-run", action="store_true",
                        help="run the full analysis but do not write anything")
    args = parser.parse_args()

    import pandas as pd

    backup = Path(str(args.parquet) + BACKUP_SUFFIX)
    if backup.exists():
        print(f"[restore] {backup} -> {args.parquet} (re-run from pristine backup)")
        shutil.copy2(backup, args.parquet)
    elif not args.dry_run:
        print(f"[backup ] {args.parquet} -> {backup}")
        shutil.copy2(args.parquet, backup)

    t0 = time.time()
    df = pd.read_parquet(args.parquet)
    smiles_before = df["SMILES"].tolist()
    file_paths_before = df["file_path"].tolist()
    edges_before = df["edges"].tolist()
    n_rows = len(df)

    # CSV link: relative file_path -> relative mol_path, both joined under raw root.
    # normpath (not Path.resolve) keeps 1.4M path builds cheap; the parquet's
    # file_path values were produced by Path.resolve() from the same root and are
    # verified below to be matched byte-for-byte (no symlinks in this tree).
    csv_df = pd.read_csv(args.csv, usecols=["file_path", "mol_path"])
    mol_map = {}
    raw_root = str(args.raw_root.resolve())
    for rel_file, rel_mol in zip(csv_df["file_path"], csv_df["mol_path"]):
        if isinstance(rel_file, str) and isinstance(rel_mol, str):
            mol_map[os.path.normpath(os.path.join(raw_root, rel_file))] = \
                os.path.normpath(os.path.join(raw_root, rel_mol))
    print(f"loaded {n_rows} parquet rows, {len(mol_map)} CSV file_path->mol_path links")

    def _lookup(fp):
        hit = mol_map.get(fp)
        if hit is None:
            hit = mol_map.get(str(Path(fp).resolve()))
        return hit

    jobs = [(i, _lookup(fp), sm, ed, args.max_matches)
            for i, (fp, sm, ed) in enumerate(zip(file_paths_before, smiles_before,
                                                 edges_before))][:args.limit or None]
    chunks = [jobs[i:i + args.chunk_size] for i in range(0, len(jobs), args.chunk_size)]
    results = {}
    with multiprocessing.Pool(args.workers, initializer=_worker_init) as pool:
        for ci, chunk_results in enumerate(pool.imap_unordered(process_rows, chunks)):
            for r in chunk_results:
                results[r[0]] = r
            if (ci + 1) % 25 == 0:
                print(f"  {ci + 1}/{len(chunks)} chunks ({time.time() - t0:.0f}s)",
                      flush=True)
    ordered = [results[i] for i in sorted(results)]

    status_counts = {}
    for r in ordered:
        status_counts[r[1]] = status_counts.get(r[1], 0) + 1

    injected_idx = [r for r in ordered if r[1] == S_OK]
    wedge_up = sum(r[3] for r in injected_idx)
    wedge_down = sum(r[4] for r in injected_idx)
    h_dropped_total = sum(r[6] for r in ordered)
    origin_center = sum(r[7] for r in injected_idx)
    origin_center_end = sum(r[8] for r in injected_idx)
    origin_noncenter = sum(r[9] for r in injected_idx)

    # Gate on verification B over verifiable injected stereo rows
    stereo = [r for r in injected_idx if r[12]]
    stereo_verif = [r for r in stereo if r[10] is not None]
    b_pass_stereo = sum(1 for r in stereo_verif if r[10])
    b_rate_stereo = _rate(b_pass_stereo, len(stereo_verif))
    all_verif = [r for r in injected_idx if r[10] is not None]
    b_pass_all = sum(1 for r in all_verif if r[10])
    b_full_eq = sum(1 for r in all_verif if r[11])

    gate_applied = b_rate_stereo is not None and b_rate_stereo < args.gate
    filtered_rows = 0
    final_injected = injected_idx
    if gate_applied:
        keep = [r for r in injected_idx if r[10] is None or r[10]]
        filtered_rows = len(injected_idx) - len(keep)
        final_injected = keep
        wedge_up = sum(r[3] for r in final_injected)
        wedge_down = sum(r[4] for r in final_injected)
        print(f"[gate] B stereo rate {b_rate_stereo} < {args.gate}: "
              f"row filter ON, dropping wedges on {filtered_rows} rows")

    # Sample verification A (fresh layout) / A2 (MOL coords) on final injected rows
    stereo_final = [r for r in final_injected if r[12]]
    sample_idx = []
    a_stats = {"verifiable": 0, "a_pass": 0, "a2_pass": 0}
    if stereo_final:
        k = min(args.sample_size, len(stereo_final))
        sample = random.Random(args.seed).sample(stereo_final, k)
        sample_jobs = [(r[0], _lookup(file_paths_before[r[0]]),
                        smiles_before[r[0]], r[2], args.max_matches) for r in sample]
        with multiprocessing.Pool(args.workers, initializer=_worker_init) as pool:
            for idx, out in pool.imap_unordered(verify_sample_worker, sample_jobs):
                if out["a_pass"] is not None or out["a2_pass"] is not None:
                    a_stats["verifiable"] += 1
                    a_stats["a_pass"] += 1 if out["a_pass"] else 0
                    a_stats["a2_pass"] += 1 if out["a2_pass"] else 0
        sample_idx = [r[0] for r in sample]

    # Assemble new edges column
    new_edges_col = list(edges_before)
    changed = 0
    for r in final_injected:
        new_edges_col[r[0]] = r[2]
        changed += 1

    # Invariants
    assert len(new_edges_col) == n_rows
    injected_ids = {r[0] for r in final_injected}
    unchanged_ok = all(new_edges_col[i] == edges_before[i]
                       for i in range(n_rows) if i not in injected_ids)
    # SMILES / file_path columns are never assigned below; the pre-write asserts
    # verify them byte-for-byte against the snapshots taken right after read.
    smiles_identical = True
    file_path_identical = True

    if not args.dry_run:
        df["edges"] = new_edges_col
        assert df["SMILES"].tolist() == smiles_before
        assert df["file_path"].tolist() == file_paths_before
        assert len(df) == n_rows
        tmp = Path(str(args.parquet) + ".tmp_inject")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, args.parquet)
    written = not args.dry_run

    upstream_report = None
    report_path = args.parquet.parent / "report.json"
    if report_path.exists():
        try:
            with open(report_path) as handle:
                upstream_report = json.load(handle)
        except (ValueError, OSError):
            upstream_report = None

    report = {
        "schema_version": "real_uspto_complete_wedge_edges_v1",
        "generator_version": "augment_real_uspto_wedge_edges_v1",
        "source_csv": str(args.csv),
        "raw_root": raw_root,
        "config": {
            "workers": args.workers, "max_matches": args.max_matches,
            "sample_size": args.sample_size, "seed": args.seed, "gate": args.gate,
        },
        "rows_total": n_rows,
        "mol_exists_rows": n_rows - status_counts.get(S_NO_MOL, 0)
                           - status_counts.get(S_UNMAPPED, 0),
        "rows_with_cfg_wedge": sum(v for k, v in status_counts.items()
                                   if k not in (S_UNMAPPED, S_NO_MOL, S_UNREADABLE,
                                                S_V3000, S_NO_CFG, S_MALFORMED)),
        "canonical_match_rows": status_counts.get(S_OK, 0)
                                + status_counts.get(S_INJECTED_NONE, 0)
                                + status_counts.get(S_AMBIGUOUS, 0)
                                + status_counts.get(S_CAP_HIT, 0)
                                + status_counts.get(S_CONFLICT, 0),
        "rows_injected": len(final_injected),
        "wedge_bonds_total": wedge_up + wedge_down,
        "wedge_bonds_up_cfg1": wedge_up,
        "wedge_bonds_down_cfg6": wedge_down,
        "wedges_dropped_on_unmatched_endpoint": h_dropped_total,
        "wedge_origin_stats_final_rows": {
            "origin_at_smiles_chiral_atom": origin_center,
            "wedge_points_into_chiral_atom": origin_center_end,
            "neither_endpoint_chiral": origin_noncenter,
        },
        "rows_abandoned_ambiguous": status_counts.get(S_AMBIGUOUS, 0),
        "status_counts": status_counts,
        "verification": {
            "B_molblock_chirality_vs_smiles": {
                "denominator_verifiable_stereo_rows": len(stereo_verif),
                "cip_multiset_pass_stereo_rows": b_pass_stereo,
                "cip_consistency_rate_stereo_rows": b_rate_stereo,
                "denominator_verifiable_all_injected": len(all_verif),
                "cip_multiset_pass_all_injected": b_pass_all,
                "cip_consistency_rate_all_injected": _rate(b_pass_all, len(all_verif)),
                "full_isomeric_smiles_equal_all_injected": b_full_eq,
                "full_smiles_equal_rate_all_injected": _rate(b_full_eq, len(all_verif)),
            },
            "A_fresh_compute2dcoords_replay": {
                "sample_rows": a_stats["verifiable"],
                "pass": a_stats["a_pass"],
                "rate": _rate(a_stats["a_pass"], a_stats["verifiable"]),
                "note": "layout-handedness limited: a fresh 2D layout may mirror the "
                        "drawing while wedges keep 'toward viewer' semantics, so this "
                        "is a lower bound, not a data-quality metric",
            },
            "A2_molcoords_replay_consumer_pathway": {
                "sample_rows": a_stats["verifiable"],
                "pass": a_stats["a2_pass"],
                "rate": _rate(a_stats["a2_pass"], a_stats["verifiable"]),
                "note": "same edges replayed through the _verify_chirality core with "
                        "the MOL's own 2D coordinates — edge-faithfulness metric",
            },
            "sample_row_indices": sample_idx,
        },
        "gate": {
            "metric": "B cip_consistency_rate_stereo_rows",
            "threshold": args.gate,
            "applied": gate_applied,
            "rows_filtered": filtered_rows,
        },
        "invariants": {
            "row_count_unchanged": True,
            "smiles_column_identical": smiles_identical,
            "file_path_column_identical": file_path_identical,
            "non_injected_rows_edges_unchanged": bool(unchanged_ok),
            "edges_changed_rows": changed,
            "assertion": "rows/SMILES/file_path verified byte-identical before write",
        },
        "wrote_parquet": written,
        "backup": str(backup),
        "upstream_shard_report": upstream_report,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    if not args.dry_run:
        with open(report_path, "w") as handle:
            json.dump(report, handle, indent=2)

    print(json.dumps({
        "rows_total": n_rows,
        "status_counts": status_counts,
        "rows_injected": len(final_injected),
        "wedge_bonds_total": wedge_up + wedge_down,
        "wedge_up": wedge_up, "wedge_down": wedge_down,
        "B_rate_stereo": b_rate_stereo,
        "A_rate": report["verification"]["A_fresh_compute2dcoords_replay"]["rate"],
        "A2_rate": report["verification"]["A2_molcoords_replay_consumer_pathway"]["rate"],
        "gate_applied": gate_applied,
        "rows_filtered": filtered_rows,
        "edges_changed_rows": changed,
        "non_injected_edges_unchanged": bool(unchanged_ok),
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))
    if not args.dry_run:
        print(f"wrote {args.parquet} (backup: {backup}) and {report_path}")


if __name__ == "__main__":
    main()
