from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from utils.markush_labels import is_markush_label, normalize_label


COUNT_BUCKETS = ["0", "1", "2", "3-4", "5-8", "9+"]


def parse_render_quality(row: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("render_quality") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def markush_payload(row: dict[str, Any]) -> dict[str, Any]:
    quality = parse_render_quality(row)
    markush = quality.get("markush")
    return markush if isinstance(markush, dict) else {}


def annotation_r_labels(annotation: str) -> list[str]:
    return [
        normalize_label(match)
        for match in re.findall(r"<r>(.*?)</r>", str(annotation or ""), flags=re.IGNORECASE | re.DOTALL)
        if normalize_label(match)
    ]


def _cxsmiles_base(cxsmiles: str) -> str:
    return str(cxsmiles or "").split(" |", 1)[0].strip()


def _fallback_smiles_atom_tokens(smiles: str) -> list[str]:
    tokens: list[str] = []
    text = str(smiles or "")
    index = 0
    while index < len(text):
        char = text[index]
        if char == "[":
            end = text.find("]", index + 1)
            if end < 0:
                index += 1
                continue
            body = text[index + 1 : end]
            match = re.search(r"\*", body) or re.search(r"[A-Z][a-z]?|[cnopsb]", body)
            tokens.append(match.group(0) if match else body)
            index = end + 1
            continue
        if char == "*":
            tokens.append("*")
            index += 1
            continue
        if text.startswith(("Cl", "Br"), index):
            tokens.append(text[index : index + 2])
            index += 2
            continue
        if char in "BCNOPSFIPSH" or char in "cnopsb":
            tokens.append(char)
        index += 1
    return tokens


def _fallback_cxsmiles_atom_labels(cxsmiles: str) -> dict[int, str]:
    text = str(cxsmiles or "")
    labels: dict[int, str] = {}
    block_match = re.search(r"\|[^|]*\$([^$]*)\$", text)
    if block_match:
        for index, raw_label in enumerate(block_match.group(1).split(";")):
            label = normalize_label(raw_label)
            if label and label != "*":
                labels[index] = label
    for match in re.finditer(
        r"(?:^|[:|,])(?P<index>\d+)\.(?P<prop>dummyLabel|atomLabel|molFileAlias)\.(?P<label>[^:|,]+)",
        text,
    ):
        label = normalize_label(match.group("label"))
        if label and label != "*":
            labels[int(match.group("index"))] = label
    return labels


def cxsmiles_dummy_labels(cxsmiles: str) -> dict[int, str]:
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        params = Chem.SmilesParserParams()
        params.allowCXSMILES = True
        params.strictCXSMILES = False
        params.removeHs = False
        molecule = Chem.MolFromSmiles(str(cxsmiles or ""), params)
    except Exception:
        molecule = None
    if molecule is not None:
        parsed: dict[int, str] = {}
        for atom in molecule.GetAtoms():
            if atom.GetAtomicNum() != 0:
                continue
            label = ""
            for prop_name in ["dummyLabel", "atomLabel", "molFileAlias"]:
                if atom.HasProp(prop_name):
                    label = atom.GetProp(prop_name)
                    break
            if not label and atom.GetSymbol() != "*":
                label = atom.GetSymbol()
            label = normalize_label(label or "*")
            if label and label != "*":
                parsed[int(atom.GetIdx())] = label
        if parsed:
            return parsed

    labels: dict[int, str] = {}
    for match in re.finditer(r"(?:^|[:|,])(?P<index>\d+)\.dummyLabel\.(?P<label>[^:|,]+)", str(cxsmiles or "")):
        label = normalize_label(match.group("label"))
        if not label:
            continue
        labels[int(match.group("index"))] = label
    atom_tokens = _fallback_smiles_atom_tokens(_cxsmiles_base(cxsmiles))
    if atom_tokens:
        atom_labels = _fallback_cxsmiles_atom_labels(cxsmiles)
        for index, token in enumerate(atom_tokens):
            if token != "*":
                continue
            label = normalize_label(atom_labels.get(index, ""))
            if label and label != "*":
                labels[index] = label
    return labels


def annotation_stable_labels(annotation: str) -> list[str]:
    labels: list[str] = []
    for body in re.findall(r"<stable>(.*?)</stable>", str(annotation or ""), flags=re.IGNORECASE | re.DOTALL):
        for match in re.finditer(r"(?:^|<ns>)([^:<>\n]+):", str(body)):
            label = normalize_label(match.group(1))
            if label and is_markush_label(label):
                labels.append(label)
    return labels


def markush_annotation_label_counts(row: dict[str, Any]) -> Counter[str]:
    markush = markush_payload(row)
    labels = [
        label
        for label in cxsmiles_dummy_labels(
            str(markush.get("cxsmiles") or row.get("SMILES") or row.get("smiles") or "")
        ).values()
        if is_markush_label(label)
    ]
    if not labels:
        labels = [
            label
            for label in annotation_r_labels(str(markush.get("annotation") or ""))
            if is_markush_label(label)
        ]
    if not labels:
        labels = annotation_stable_labels(str(markush.get("annotation") or ""))
    return Counter(labels)


def normalized_box(cell: dict[str, Any]) -> list[float] | None:
    bbox = cell.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x1 <= x2 <= 1.0 and 0.0 <= y1 <= y2 <= 1.0):
        return None
    return [(x1 + x2) * 0.5, (y1 + y2) * 0.5, max(1e-4, x2 - x1), max(1e-4, y2 - y1)]


def valid_normalized_box(cell: dict[str, Any]) -> bool:
    return normalized_box(cell) is not None


def markush_layout_cells(row: dict[str, Any]) -> list[dict[str, Any]]:
    markush = markush_payload(row)
    cells = markush.get("ocr_cells")
    if not isinstance(cells, list):
        return []
    label_counts = markush_annotation_label_counts(row)
    if not label_counts:
        return []

    selected: list[dict[str, Any]] = []
    remaining = Counter(label_counts)
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        box = normalized_box(cell)
        if box is None:
            continue
        text = normalize_label(str(cell.get("text") or ""))
        if not text or remaining[text] <= 0:
            continue
        remaining[text] -= 1
        selected.append(
            {
                "text": text,
                "box": box,
                "source": str(cell.get("source") or ""),
                "atom_index": cell.get("atom_index"),
            }
        )
    selected.sort(key=lambda item: (item["box"][1], item["box"][0], item["text"]))
    return selected


def markush_variable_count(row: dict[str, Any]) -> int:
    return len(markush_layout_cells(row))


def count_bucket_name(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 4:
        return "3-4"
    if count <= 8:
        return "5-8"
    return "9+"


def count_bucket_index(count: int) -> int:
    return COUNT_BUCKETS.index(count_bucket_name(count))
