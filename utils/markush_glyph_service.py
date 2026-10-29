"""MarkushGlyph integration service.

Production client for the MarkushGlyph image→CXSMILES model:
  - persistent worker subprocess inside the isolated glyph venv
    (transformers 5.x never touches the production llm environment)
  - lazy start, batch `predict_many`, death-detection + restart
  - `aligned_identity` numbering adapter: MarkushGlyph emits variable-name
    labels; the deployed gold convention numbers dummies (numeric label
    keeps its number, non-numeric labels take sequential placeholders in
    dummy order, skipping used numbers).
  - `numbered_molblock`: CoordGen depiction → coordinate-bearing molblock
    for the assembly chain (MarkushGlyph output is topology-only)

Environment:
  MARKUSHGLYPH_VENV_PYTHON  glyph venv python (default /data/glyph_venv/bin/python)
  MARKUSHGLYPH_WORKER       worker script path (default repo utils/markushglyph_worker.py)
  MARKUSHGLYPH_ENABLED      set "1" to enable (default off — flip explicitly)
"""
from __future__ import annotations

import json
import os
import subprocess
import threading

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

_WORKER_SCRIPT = os.environ.get(
    "MARKUSHGLYPH_WORKER",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "markushglyph_worker.py"))
_VENV_PYTHON = os.environ.get(
    "MARKUSHGLYPH_VENV_PYTHON", "/data/glyph_venv/bin/python")


class MarkushGlyphService:
    """Lazy persistent MarkushGlyph worker client (fork-safe via lock)."""

    def __init__(self):
        self._process = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return os.environ.get("MARKUSHGLYPH_ENABLED", "1") != "0" and os.path.exists(_WORKER_SCRIPT)

    def _start(self):
        if not os.path.exists(_VENV_PYTHON):
            raise RuntimeError(f"glyph venv python missing: {_VENV_PYTHON}")
        self._process = subprocess.Popen(
            [_VENV_PYTHON, _WORKER_SCRIPT],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def _ensure_started(self):
        if self._process is None or self._process.poll() is not None:
            if self._process is not None:
                try:
                    self._process.kill()
                except OSError:
                    pass
            self._start()
            line = self._process.stdout.readline()
            ready = json.loads(line) if line else {}
            if ready.get("event") != "ready":
                raise RuntimeError(f"markushglyph worker failed to start: {line[:200]}")

    def predict_many(self, image_paths: list[str], max_retries: int = 1) -> list[dict]:
        """Batch prediction; returns one result dict per input, order-stable."""
        with self._lock:
            results: list[dict | None] = [None] * len(image_paths)
            for attempt in range(max_retries + 1):
                try:
                    self._ensure_started()
                    outstanding = [i for i, r in enumerate(results) if r is None]
                    # INTERLEAVED write/read (request i → response i → …):
                    # write-all-then-read-all deadlocks once the OS pipe
                    # buffer fills (~64KB) with both sides waiting.
                    for i in outstanding:
                        self._process.stdin.write(
                            json.dumps({"image_path": image_paths[i]}) + "\n")
                        self._process.stdin.flush()
                        line = self._process.stdout.readline()
                        if not line:
                            raise RuntimeError("worker died (empty response)")
                        results[i] = json.loads(line)
                    break
                except (RuntimeError, json.JSONDecodeError, OSError):
                    if attempt >= max_retries:
                        break
                    try:
                        if self._process is not None:
                            self._process.kill()
                    except OSError:
                        pass
                    self._process = None
            return [r if r is not None else {"ok": False, "error": "worker_unavailable"}
                    for r in results]

    def predict(self, image_path: str) -> dict:
        return self.predict_many([image_path])[0]

    def shutdown(self):
        with self._lock:
            if self._process is not None:
                try:
                    self._process.stdin.close()
                    self._process.terminate()
                except OSError:
                    pass
                self._process = None


_SERVICE: MarkushGlyphService | None = None
_SERVICE_LOCK = threading.Lock()


def markush_glyph_service() -> MarkushGlyphService:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = MarkushGlyphService()
        return _SERVICE


# ---------------------------------------------------------------- adapter ---

def numeric_value(label: str):
    text = str(label or '').strip()
    compact = text.replace('R', '').replace('r', '').strip()
    return int(compact) if compact.isdigit() else None


def apply_numbering(mol: Chem.Mol) -> Chem.Mol:
    """Numbering rule: numeric variable labels keep their number (R4 -> 4);
    non-numeric labels take sequential placeholders in dummy order, skipping
    used numbers."""
    used: set[int] = set()
    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0 and a.GetDegree() >= 1]
    for atom in dummies:
        label = atom.GetProp('atomLabel') if atom.HasProp('atomLabel') else ''
        number = numeric_value(label)
        if number is not None and number not in used:
            atom.SetIsotope(number)
            used.add(number)
    sequence = 0
    for atom in dummies:
        if atom.GetIsotope() > 0:
            continue
        sequence += 1
        while sequence in used:
            sequence += 1
        atom.SetIsotope(sequence)
        used.add(sequence)
    return mol


def numbered_mol_from_cxsmiles(cxsmiles: str) -> Chem.Mol | None:
    """MarkushGlyph CXSMILES (|$label...$| annotations) → numbered RDKit mol."""
    params = Chem.SmilesParserParams()
    params.allowCXSMILES = True
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(cxsmiles or ''), params)
    if mol is None:
        return None
    return apply_numbering(mol)


def numbered_molblock(cxsmiles: str, *, coordgen: bool = True) -> str | None:
    """Topology-only MarkushGlyph output → coordinate-bearing molblock.

    CoordGen produces the clean 2D layout the assembly chain and the output
    CSVs expect (the product contract is correct structures, not pixel-matched
    poses).
    """
    mol = numbered_mol_from_cxsmiles(cxsmiles)
    if mol is None:
        return None
    try:
        if coordgen:
            from rdkit.Chem import rdDepictor
            rdDepictor.SetPreferCoordGen(True)
            rdDepictor.Compute2DCoords(mol)
        else:
            from rdkit.Chem import AllChem
            AllChem.Compute2DCoords(mol)
        return Chem.MolToMolBlock(mol)
    except Exception:
        return None


def aligned_identity(prediction_smiles: str, gold_smiles: str) -> bool:
    """Canonical isotope-sequence equality (order-invariant identity)."""
    pred = numbered_mol_from_cxsmiles(prediction_smiles)
    gold = Chem.MolFromSmiles(str(gold_smiles or ''))
    if pred is None or gold is None:
        return False

    def canon_iso(mol):
        can = Chem.MolToSmiles(mol)
        mol2 = Chem.MolFromSmiles(can)
        if mol2 is None:
            return None
        return tuple(a.GetIsotope() for a in mol2.GetAtoms() if a.GetAtomicNum() == 0)

    pi, gi = canon_iso(pred), canon_iso(gold)
    if not gi:
        return False
    return pi == gi
