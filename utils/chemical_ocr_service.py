"""ChemicalOCR standalone service (): text + normalized bboxes for
every label glyph on a structure crop — the SOTA-aligned position+text
source that replaces the heatmap detector in the label rewrite chain.

Wraps docling-project/ChemicalOCR (Idefics3-256M finetune, apache-2.0) with
MarkushGrapher's side-loaded transformers 4.46.3 pattern (the llm env ships
4.43 without idefics3). Validated: load 0.64 s, ~1.45 s/image, deterministic,
reads R2a/R3/R4 correctly on eval siblings.

Side-load prerequisites (no env pollution):
  pip install --target /tmp/hf_tf_4463 --no-deps \
      transformers==4.46.3 tokenizers==0.22.2
  blank /tmp/hf_tf_4463/transformers/dependency_versions_check.py
  patch idefics3 resize for max_image_size (copied from MG chemical_ocr.py)
  sys.path.insert(0, '/tmp/hf_tf_4463') BEFORE importing transformers
"""
from __future__ import annotations

import re


class ChemicalOCRService:
    def __init__(self, model_path: str, side_load_dir: str = "/tmp/hf_tf_4463",
                 device: str = "cuda", longest_edge: int = 1820,
                 max_new_tokens: int = 1024):
        import sys
        if side_load_dir not in sys.path:
            sys.path.insert(0, side_load_dir)
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor
        self.device = device
        # 1820 validated on dense real page crops; synthetic structure renders
        # (~400-700px) run ~4x faster (same read quality) at 768.
        self.longest_edge = int(longest_edge)
        # Structure crops emit <20 label cells; 1024 (docling default) lets
        # degenerate repetition loops burn 15x the decode time.
        self.max_new_tokens = int(max_new_tokens)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path, torch_dtype=torch.bfloat16).to(device).eval()
        self.prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text",
                 "text": "Perform OCR on this chemical structure image."}]}],
            tokenize=False, add_generation_prompt=True)
        self._torch = torch

    @staticmethod
    def _clean_ocr_text(text: str) -> str:
        text = re.sub(r"^.*?(<ocr>)", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"(</ocr>).*?$", r"\1", text, flags=re.DOTALL)
        return text

    @staticmethod
    def _parse_ocr_string(ocr_string: str):
        cleaned = re.sub(r"</?ocr>", "", ocr_string).strip()
        words, boxes = [], []
        if "<loc_" in cleaned:
            cleaned = re.sub(
                r"^<loc_0><loc_0><loc_500><loc_500>", "", cleaned, count=1).strip()
            for line in cleaned.splitlines():
                locs = list(map(int, re.findall(r"<loc_(\d+)>", line)))
                t = re.sub(r"(?:<loc_\d+>){4}", "", line).strip()
                if len(locs) >= 4 and t:
                    words.append(t)
                    boxes.append([x / 500 for x in locs[-4:]])
        else:
            for line in cleaned.splitlines():
                m = re.match(
                    r"^(?:\d+>)*(\d+)>(\d+)>(\d+)>(\d+)>(.+)$", line.strip())
                if m:
                    x1, y1, x2, y2 = map(int, m.groups()[:4])
                    t = m.group(5).strip()
                    if t:
                        words.append(t)
                        boxes.append([x1 / 500, y1 / 500, x2 / 500, y2 / 500])
        return words, boxes

    def __call__(self, pil_image):
        torch = self._torch
        inputs = self.processor(
            text=self.prompt, images=[pil_image], return_tensors="pt",
            size={"longest_edge": self.longest_edge}).to(self.device)
        gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                  do_sample=False)
        n_in = inputs["input_ids"].shape[-1]
        text = self.processor.batch_decode(
            gen[:, n_in:], skip_special_tokens=True)[0]
        words, boxes = self._parse_ocr_string(self._clean_ocr_text(text))
        import PIL.Image
        w, h = pil_image.size
        return [{"bbox": [round(b[0], 4), round(b[1], 4),
                          round(b[2], 4), round(b[3], 4)],
                 "text": w_}
                for w_, b in zip(words, boxes)]
