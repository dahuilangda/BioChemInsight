"""Structure routing harness (§44.71) — image-level markush-vs-complete judgment.

The user directive: judge from the IMAGE whether a segment is a markush
structure or a complete molecule, then route to the right backend instead of
trusting a single upstream label. Signals (strongest first):

  1. keeper decode dummies — a complete molecule never decodes with ``*``
     atoms; dummies in the keeper decode are a hard markush signal.
  2. declared structure_type — the LLM filter's verdict ('markush',
     'fragment', 'complete_compound').
  3. macrocycle enclosure — a drawn macrocycle encloses a large background
     region. cv2 connected-component analysis of the free space measures the
     largest hole that does not touch the crop border, as a fraction of the
     image area. Benzene-scale rings are ~0.001-0.005; gramicidin-S-scale
     macrocycles are 10-100x that. This catches the failure mode where the
     keeper BREAKS the ring (its decode shows no large ring, so the decode
     alone cannot reveal the macrocycle).

Routing policy (measured, §44.60-§44.71):
  markush   -> MarkushGlyph replace  (identity 63.93% vs keeper 20.1%)
  complete  -> keeper stands         (exact 93.31% vs MolSight 33.98%)
  escalation-> MolSight when the keeper decode is empty, or when the decode
               is a valid amide-heavy chain with NO large ring — the measured
               signature of a broken macrocycle (gramicidin S / cyclosporine:
               keeper returns a linear polyamide, gold holds a 23-ring).

A pixel-enclosure heuristic was calibrated and REJECTED (§44.71): tight crops
of small rings read 0.056 enclosed-area ratio while a real 23-ring with side
chains reads 0.007 — the signal inverts on crop tightness. It is kept as an
audit-only signal in the decision's signals dict.

Arbitration guard: MolSight replaces a non-empty keeper decode only when its
decode is valid, contains a ring >= MOLSIGHT_RING_MIN, AND its atom count is
within COMPOSITION_DRIFT of the keeper decode (a big-ring decode with wildly
different composition is a hallucination, not a recovery). Every branch is
fail-open: any signal/service failure returns the keeper row unchanged. The
harness can only add structures that today would be empty or ring-broken,
never regress a healthy keeper decode.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

ROUTE_KEEPER = "keeper"
ROUTE_GLYPH = "glyph"
ROUTE_MOLSIGHT = "molsight"

# Calibration (§44.71 tests): benzene-scale hole ≈ 0.001-0.005 of crop area;
# gramicidin-S-scale macrocycle ≈ 0.04-0.15. Threshold sits an order of
# magnitude above the ring scale. Override via env for experiments.
MACRO_ENCLOSURE_RATIO = float(os.environ.get("STRUCTURE_MACRO_ENCLOSURE_RATIO", "0.02"))
# A keeper decode whose largest ring is below this while amide-heavy is
# treated as a broken macrocycle (gold 23-rings clear it; drug-like rings
# max out around 8-10).
MACRO_RING_MIN = int(os.environ.get("STRUCTURE_MACRO_RING_MIN", "14"))
# MolSight decode must find at least this ring size to win arbitration.
MOLSIGHT_RING_MIN = int(os.environ.get("STRUCTURE_MOLSIGHT_RING_MIN", "12"))
# Amide-nitrogen count that marks a keeper decode as peptide-like. The
# gramicidin/cyclosporine broken decodes carry 8-11 backbone nitrogens;
# ordinary drug-like completes stay under 4.
PEPTIDE_AMIDE_N_MIN = int(os.environ.get("STRUCTURE_PEPTIDE_AMIDE_N_MIN", "5"))
# MolSight-vs-keeper atom-count drift allowed for the swap (hallucination guard).
COMPOSITION_DRIFT = float(os.environ.get("STRUCTURE_MOLSIGHT_DRIFT", "0.35"))


@dataclass(frozen=True)
class RouteDecision:
    kind: str                      # 'markush' | 'complete'
    backend: str                   # ROUTE_KEEPER | ROUTE_GLYPH | ROUTE_MOLSIGHT
    reason: str
    signals: dict = field(default_factory=dict)
    macro_expected: bool = False   # escalation: molsight must find a big ring


def has_dummy_count(smiles: str) -> int:
    """Number of dummy atoms ('*') in a decoded SMILES, 0 on any failure."""
    text = str(smiles or "")
    if not text.strip():
        return 0
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            # CXSMILES / attachment-star variants may need the loose parse
            mol = Chem.MolFromSmiles(text.replace("*", "[*]"))
        if mol is None:
            return 0
        return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
    except Exception:
        return 0


def max_ring_size(smiles: str) -> int:
    """Largest SSSR ring size in the decoded SMILES, 0 on any failure."""
    text = str(smiles or "")
    if not text.strip():
        return 0
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return 0
        ri = mol.GetRingInfo()
        return max((len(r) for r in ri.AtomRings()), default=0)
    except Exception:
        return 0


def enclosure_ratio(image_path) -> float | None:
    """Largest border-detached background hole, as a fraction of image area.

    Pure OpenCV/numpy — no GPU, ~1ms at 512px. None on unreadable images.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    if not image_path:
        return None
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            return None
        _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        free = (bw == 0).astype(np.uint8)
        num, _, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
        h, w = bw.shape
        best = 0
        for i in range(1, num):
            x, y, cw, ch, area = stats[i]
            # A hole is a free-space component fully inside the crop; anything
            # touching the border is outside-the-structure background.
            if x == 0 or y == 0 or x + cw >= w or y + ch >= h:
                continue
            best = max(best, int(area))
        return best / float(h * w)
    except Exception:
        return None


def declared_kind(structure_type: str) -> str:
    text = str(structure_type or "").lower()
    if "markush" in text:
        return "markush"
    if "fragment" in text:
        return "fragment"
    return "complete"


def amide_nitrogen_count(smiles: str) -> int:
    """Backbone-like nitrogens (aliphatic N with >=2 heavy neighbors), which
    in a ring-broken peptide decode equals the residue count."""
    text = str(smiles or "")
    if not text.strip():
        return 0
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return 0
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "N" or atom.GetIsAromatic():
                continue
            if atom.GetDegree() >= 2:
                count += 1
        return count
    except Exception:
        return 0


def atom_count(smiles: str) -> int:
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(str(smiles or ""))
        return mol.GetNumAtoms() if mol is not None else 0
    except Exception:
        return 0


def judge(structure_type: str, keeper_smiles: str, image_path) -> RouteDecision:
    """Route decision from declared type + keeper decode + image audit signal."""
    kind = declared_kind(structure_type)
    dummies = has_dummy_count(keeper_smiles)
    keeper_ok = bool(str(keeper_smiles or "").strip())
    keeper_ring = max_ring_size(keeper_smiles) if keeper_ok else 0
    amides = amide_nitrogen_count(keeper_smiles) if keeper_ok else 0
    ratio = enclosure_ratio(image_path)
    signals = {
        "declared": kind,
        "keeper_ok": keeper_ok,
        "keeper_dummies": dummies,
        "keeper_max_ring": keeper_ring,
        "keeper_amide_n": amides,
        "keeper_atoms": atom_count(keeper_smiles) if keeper_ok else 0,
        "enclosure_ratio": round(ratio, 5) if ratio is not None else None,
    }

    if kind == "markush":
        return RouteDecision("markush", ROUTE_GLYPH, "declared_markush", signals)

    # complete side only — fragments legitimately carry attachment dummies
    # (the fragment contract), so a dummy in their decode is not a markush
    # misdeclaration.
    if kind == "complete" and dummies > 0:
        # Hard signal: complete molecules have no dummy atoms. A declared
        # complete whose keeper decode contains dummies is a mis-declared
        # markush (keeper drops R-labels often; emitting them is definitive).
        return RouteDecision("markush", ROUTE_GLYPH, "keeper_decode_has_dummies",
                             signals)

    if not keeper_ok:
        return RouteDecision("complete", ROUTE_MOLSIGHT,
                             "keeper_empty_escalation", signals)

    # Ring-broken macrocycle signature: peptide-amide-heavy decode, no large
    # ring. This is exactly what the keeper returns for gramicidin S and
    # cyclosporine (valid linear polyamide; gold holds a 23-ring).
    if (amides >= PEPTIDE_AMIDE_N_MIN
            and keeper_ring < MACRO_RING_MIN):
        return RouteDecision("complete", ROUTE_MOLSIGHT,
                             "amide_heavy_ring_broken_macrocycle_suspect",
                             signals, macro_expected=True)

    return RouteDecision(kind, ROUTE_KEEPER, "keeper_stands", signals)


def arbitrate_macrocycle(keeper_smiles: str, molsight_smiles: str) -> bool:
    """Final guard before MolSight replaces a non-empty keeper decode.

    MolSight wins only when its decode is valid, actually contains the large
    ring the keeper decode lacks, and stays compositionally close to the
    keeper decode (a big-ring decode with wildly different composition is a
    hallucination, not a recovery). Otherwise the keeper row stands.
    """
    text = str(molsight_smiles or "").strip()
    if not text:
        return False
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return False
        ri = mol.GetRingInfo()
        if max((len(r) for r in ri.AtomRings()), default=0) < MOLSIGHT_RING_MIN:
            return False
        keeper_atoms = atom_count(keeper_smiles)
        molsight_atoms = mol.GetNumAtoms()
        if keeper_atoms <= 0:
            return True
        return abs(molsight_atoms - keeper_atoms) <= COMPOSITION_DRIFT * keeper_atoms
    except Exception:
        return False
