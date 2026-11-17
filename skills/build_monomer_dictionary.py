"""Build data/monomer_dictionary.json: the built-in peptide monomer
vocabulary, merged offline from permissively licensed sources. The
runtime never calls the network for dictionary lookups; this script is
the only place sources are read.

Sources:
  pypept  Boehringer-Ingelheim/pyPept details_monomers.csv   MIT
  helm    Pistoia Alliance HELM Core Library                 MIT
  chembl  ChEMBL HELM monomer library                        CC-BY-SA 4.0
  ccd     RCSB Chemical Component Dictionary                 public

A code resolves to entries; each entry carries a backbone form (residue
chaining) and a cap form (N-terminal acyl). Entries are keyed by every
alias the source printed for them ('hC or hCys or Hcy' style aliases come
from the sources' own synonym fields). Conflicting readings for one code
are kept side by side — arbitration at use time decides, the dictionary
never picks a winner.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from utils.peptide_fragment_contract import (backbone_fragment,
                                             cap_fragment)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "monomer_dictionary.json"
_LABEL_RE = re.compile(r"\|\$([^$]*)\$\|")


def _parse(smiles: str):
    """SMILES -> (backbone form, cap form, concept key); either form may
    be None. The concept key is the InChIKey first block of the
    uncharged free form — protonation variants across sources share one
    bucket, stereo variants keep separate entries inside it."""
    if not smiles:
        return None, None, None
    mol = Chem.MolFromSmiles(_strip_atom_labels(smiles))
    if mol is None:
        return None, None, None
    canonical = Chem.MolToSmiles(mol)
    key = _concept_key(canonical)
    return (backbone_fragment(canonical), cap_fragment(canonical), key)


def _concept_key(smiles: str) -> str | None:
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        key = Chem.MolToInchiKey(mol)
        return key[:14] if key else None
    except Exception:
        return None


def _strip_atom_labels(smiles: str) -> str:
    smiles = re.sub(r"\[H:\d+\]", "[H]", smiles)
    smiles = re.sub(r"\[OH:\d+\]", "[OH]", smiles)
    return _LABEL_RE.sub("", smiles).strip()


def _chembl_dummies_to_free(smiles: str) -> str | None:
    """ChEMBL marks attachment dummies with |$..._R1;;_R2;$| labels; R1
    caps as H (N-terminus), R2 caps as OH (C-terminus)."""
    match = _LABEL_RE.search(smiles)
    body = _strip_atom_labels(smiles)
    if not match:
        return None
    named = [tok for tok in match.group(1).split(";")
             if tok.startswith("_R")]
    parts = body.split("[*]")
    if len(parts) - 1 != len(named):
        return None
    caps = {"_R1": "[H]", "_R2": "[OH]"}
    out = [parts[0]]
    for label, tail in zip(named, parts[1:]):
        out.append(caps.get(label, "[H]"))
        out.append(tail)
    return "".join(out)


def _read_pypept(path: str):
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            yield [row["Symbol"]], row.get("Name", ""), row["SMILES"]


def _read_helm(path: str):
    data = json.load(open(path, encoding="utf-8"))
    for entry in data:
        if entry.get("polymerType") != "PEPTIDE":
            continue
        yield [entry["symbol"]], entry.get("name", ""), entry["smiles"]


def _read_chembl(path: str):
    import xml.etree.ElementTree as ET
    ns = {"lmr": "lmr"}
    root = ET.parse(path).getroot()
    for monomer in root.iterfind(".//lmr:Monomer", ns):
        smiles = monomer.findtext("lmr:MonomerSmiles", "", ns)
        free = _chembl_dummies_to_free(smiles)
        if not free:
            continue
        code = monomer.findtext("lmr:MonomerID", "", ns).strip()
        name = monomer.findtext("lmr:MonomerName", "", ns).strip()
        yield [code], name, free


def _read_ccd(path: str):
    import gemmi
    doc = gemmi.cif.read(path)      # gemmi reads .gz transparently
    for block in doc:
        comp_type = block.find_value("_chem_comp.type") or ""
        if "peptide" not in comp_type.lower():
            continue
        code = block.find_value("_chem_comp.id") or ""
        name = block.find_value("_chem_comp.name") or ""
        descriptors = []
        for row in block.find("_pdbx_chem_comp_descriptor.",
                              ["type", "descriptor"]):
            kind = row.str(0)
            if kind in ("SMILES", "SMILES_CANONICAL"):
                descriptors.append(row.str(1))
        if not descriptors:
            continue
        yield [code], name.strip('"'), descriptors[0]


def build(sources: dict[str, str]) -> list[dict]:
    readers = {"pypept": _read_pypept, "helm": _read_helm,
               "chembl": _read_chembl, "ccd": _read_ccd}
    merged: dict[tuple, dict] = {}
    aliases: dict[tuple, set[str]] = defaultdict(set)
    counts = {}
    for source, path in sources.items():
        kept = 0
        for codes, name, smiles in readers[source](path):
            backbone, cap, concept = _parse(smiles)
            if backbone is None and cap is None:
                continue
            key = (concept or f"raw:{backbone or ''}|{cap or ''}",
                   backbone or "", cap or "")
            entry = merged.setdefault(key, {
                "backbone": backbone, "cap": cap, "name": name,
                "concept": concept, "sources": []})
            if name and len(name) > len(entry.get("name", "") or ""):
                entry["name"] = name
            if source not in entry["sources"]:
                entry["sources"].append(source)
            for code in codes:
                code = code.strip()
                if code:
                    aliases[key].add(code)
            kept += 1
        counts[source] = kept
    out = []
    for key, entry in merged.items():
        entry["codes"] = sorted(aliases[key])
        out.append(entry)
    print(f"kept {counts}, entries {len(out)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pypept", default="/tmp/pypept_details.csv")
    parser.add_argument("--helm", default="/tmp/HELMCoreLibrary.json")
    parser.add_argument("--chembl", default="/tmp/chembl_monomers.xml")
    parser.add_argument("--ccd", default="/tmp/ccd_check/components.cif.gz")
    parser.add_argument("--skip", default="", help="comma-separated sources")
    parser.add_argument("--out", default=str(DATA_PATH))
    args = parser.parse_args()
    skip = {s for s in args.skip.split(",") if s}
    sources = {name: getattr(args, name) for name in
               ("pypept", "helm", "chembl", "ccd") if name not in skip}
    entries = build(sources)
    Path(args.out).write_text(json.dumps(entries, indent=1),
                              encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
