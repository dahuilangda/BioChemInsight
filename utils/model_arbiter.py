"""VLM render-and-compare judge for structure routing.

same_molecule(): forced pairwise verdict of a rendered candidate against
the original crop.
macrocycle_check(): checklist-style ring comparison (the judge states the
original's largest ring, then verifies the candidate against it).

Every call fails open to {"verdict": "unavailable"}.
"""
from __future__ import annotations

import base64
import json
import os
import tempfile

_RENDER_CACHE: dict[str, str] = {}


def render_candidate(smiles: str) -> str | None:
    """RDKit 2D depiction of a candidate decode -> PNG path (cached)."""
    key = smiles
    if key in _RENDER_CACHE and os.path.exists(_RENDER_CACHE[key]):
        return _RENDER_CACHE[key]
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Draw
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        path = tempfile.NamedTemporaryFile(prefix="cand_", suffix=".png", dir="/tmp", delete=False).name
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
        with open(crop_path, 'rb') as cfh:
            img_b64 = base64.b64encode(cfh.read()).decode()
        with open(render_path, 'rb') as rfh:
            ren_b64 = base64.b64encode(rfh.read()).decode()
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
        print(f"model arbiter: unparseable LLM response: {text[:200]!r}")
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
        """Checklist probe: does the original image show a large ring that
        the candidate decode lacks?"""
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
