"""Model-native arbitration (§45.13) — VLM render-and-compare judge.

Replaces every rule-based decision in the routing harness (amide-nitrogen
counts, ring-size thresholds, composition-drift windows) with the published
evidence-grounded pattern (MLLM-as-a-Judge, ICML 2024; ChemExplain's negative
result on programmatic round-trips):

  1. same_molecule(crop, smiles): render the candidate SMILES with RDKit,
     ask the VLM for a forced verdict {"same_molecule": "yes|no|partial",
     "differences": [...], "confidence": "high|medium|low"}. Pairwise,
     never absolute scoring (judge bias findings).
  2. macrocycle_check(crop, smiles): checklist-style — the judge first
     states what the ORIGINAL shows (ring count / largest ring), then
     verifies the rendered candidate against that list (judging against a
     transcribed list beats raw perception, GTR-CoT lesson).

No regex, no thresholds — the model decides; callers get structured
verdicts. Every call is fail-open to {"verdict": "unavailable"}.
"""
from __future__ import annotations

import base64
import json
import os
import tempfile

_RENDER_CACHE: dict[str, str] = {}


def render_candidate(smiles: str, highlight_ring: bool = False) -> str | None:
    """RDKit 2D depiction of a candidate decode -> PNG path (cached)."""
    key = f"{smiles}|{highlight_ring}"
    if key in _RENDER_CACHE and os.path.exists(_RENDER_CACHE[key]):
        return _RENDER_CACHE[key]
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Draw
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if highlight_ring:
            ri = mol.GetRingInfo()
            largest = max(ri.AtomRings(), key=len) if ri.NumRings() else ()
            if largest:
                for idx in largest:
                    atom = mol.GetAtomWithIdx(idx)
                    atom.SetProp("molAtomMapNumber", str(idx))
        path = tempfile.mktemp(prefix="cand_", suffix=".png", dir="/tmp")
        Draw.MolToFile(mol, path, imageSize=(512, 512))
        _RENDER_CACHE[key] = path
        return path
    except Exception:
        return None


class ModelArbiter:
    def __init__(self):
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import constants
            from openai import OpenAI
            self._client = OpenAI(
                base_url=constants.LLM_OPENAI_COMPATIBLE_MODEL_URL,
                api_key=constants.LLM_OPENAI_COMPATIBLE_MODEL_KEY)
            self._model = constants.VISUAL_MODEL_NAME
        return self._client

    def _judge(self, crop_path: str, render_path: str, question: str) -> dict:
        img_b64 = base64.b64encode(open(crop_path, 'rb').read()).decode()
        ren_b64 = base64.b64encode(open(render_path, 'rb').read()).decode()
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "text", "text": "IMAGE 1 is the original structure crop from a patent page. IMAGE 2 is a candidate decoded structure rendered as 2D."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ren_b64}"}}]}],
            temperature=0, max_tokens=1200, timeout=120)
        text = resp.choices[0].message.content or ''
        start = text.find('{')
        depth = 0
        for i in range(start, len(text)) if start >= 0 else []:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
        return {"verdict": "unavailable", "raw": text[:300]}

    def same_molecule(self, crop_path: str, smiles: str) -> dict:
        """Forced pairwise verdict: does the rendered candidate depict the
        original crop's molecule?"""
        render_path = render_candidate(smiles)
        if not render_path:
            return {"verdict": "invalid_candidate"}
        question = (
            "Compare IMAGE 1 (original) and IMAGE 2 (candidate decode).\n"
            "Decide by counting substructures: atoms by element, ring count, "
            "largest ring size, side chains, amide groups. Ignore drawing "
            "style, coordinates, and stereo-bond rendering.\n"
            "Respond with ONLY this JSON object — no prose, no analysis, "
            "the output must start with '{':\n"
            "{\"same_molecule\": \"yes|no|partial\", \"differences\": [\"...\"], "
            "\"confidence\": \"high|medium|low\"}\n"
            "A macrocycle closed in one image and open in the other is \"no\". "
            "Same molecule with small drawing differences is \"yes\".")
        return self._judge(crop_path, render_path, question)

    def macrocycle_check(self, crop_path: str, smiles: str) -> dict:
        """Checklist probe: does the ORIGINAL image show a large ring that the
        candidate decode lacks? Model-native broken-macrocycle detection."""
        render_path = render_candidate(smiles)
        if not render_path:
            return {"verdict": "invalid_candidate"}
        question = (
            "IMAGE 1 is the original patent crop; IMAGE 2 is the candidate "
            "decode. Look at the largest closed loop in each.\n"
            "Respond with ONLY this JSON object — no prose, the output must "
            "start with '{':\n"
            "{\"original_largest_ring\": <int>, \"candidate_largest_ring\": <int>, "
            "\"candidate_broke_ring\": <true|false>, \"confidence\": "
            "\"high|medium|low\"}\n"
            "candidate_broke_ring is true only when the original clearly shows "
            "a large closed ring (>= 10 atoms) that the candidate opens or "
            "fragments.")
        return self._judge(crop_path, render_path, question)


_ARBITER: ModelArbiter | None = None


def model_arbiter() -> ModelArbiter:
    global _ARBITER
    if _ARBITER is None:
        _ARBITER = ModelArbiter()
    return _ARBITER
