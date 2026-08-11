from __future__ import annotations

import re
from typing import Any

from SmilesPE.pretokenizer import atomwise_tokenizer

from utils.markush_labels import (
    SINGLE_LETTER_LABELS,
    is_generic_markush_class_label,
    is_markush_label,
    normalize_label,
)


def is_atom_token(token: str) -> bool:
    return token.isalpha() or token.startswith("[") or token == "*"


def atom_tokens(smiles: str) -> list[str]:
    return [token for token in atomwise_tokenizer(str(smiles or "")) if is_atom_token(token)]


def visible_atom_label(token: str) -> str:
    text = normalize_label(token)
    return "*" if text == "*" else text


def label_family(label: str) -> str:
    text = normalize_label(label)
    if is_generic_markush_class_label(text):
        return "generic"
    if text in SINGLE_LETTER_LABELS:
        return "single_letter"
    if re.fullmatch(r"R\d+", text):
        return "r_digit"
    if re.fullmatch(r"R[a-z]", text):
        return "r_letter"
    if re.fullmatch(r"R(?:'+|[A-Za-z0-9]*'+)", text):
        return "r_prime"
    if text == "R":
        return "r_plain"
    if re.fullmatch(r"[A-Z][A-Za-z]?\d+(?:'+)?", text):
        return "letter_digit"
    if re.fullmatch(r"[A-Z]'+", text):
        return "single_letter_prime"
    return "other"


def label_features(labels: list[str]) -> set[str]:
    values = {normalize_label(label) for label in labels}
    features = set()
    count = len(values)
    if count:
        features.add("has_markush_label")
    features.add(f"label_count_{count}" if count < 4 else "label_count_4plus")
    if count >= 3:
        features.add("label_count_3plus")
    for label in values:
        family = label_family(label)
        features.add(f"family_{family}")
        if "'" in label:
            features.add("has_prime")
        if re.search(r"\d", label):
            features.add("has_digit")
        if re.search(r"\d{2,}", label):
            features.add("has_digit_10plus")
        if family == "r_letter":
            features.add("has_r_letter")
        if family == "generic":
            features.add("has_generic_word")
        if family == "single_letter":
            features.add("has_single_letter")
    return features


def expected_labels(smiles: str) -> list[str]:
    labels = []
    for token in atom_tokens(smiles):
        label = visible_atom_label(token)
        if is_markush_label(label):
            labels.append(normalize_label(label))
    return sorted(set(labels))


def label_count_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count >= 4:
        return "4plus"
    return str(count)


def label_difficulty(labels: list[str]) -> str:
    count = len(set(labels))
    if count >= 4:
        return "hard"
    if count == 3:
        return "medium"
    if any(label_family(label) in {"r_prime", "letter_digit", "single_letter_prime"} for label in labels):
        return "medium"
    if any(re.search(r"\d{2,}", normalize_label(label)) for label in labels):
        return "medium"
    return "easy" if count else "none"


def label_instances_from_smiles(
    smiles: str,
    centers: list[list[float]] | None = None,
    boxes: dict[int, list[float]] | None = None,
) -> list[dict[str, Any]]:
    instances = []
    for atom_index, token in enumerate(atom_tokens(smiles)):
        text = visible_atom_label(token)
        if not is_markush_label(text):
            continue
        item: dict[str, Any] = {
            "text": normalize_label(text),
            "atom_index": atom_index,
            "family": label_family(text),
        }
        if centers is not None and atom_index < len(centers):
            item["center"] = [float(centers[atom_index][0]), float(centers[atom_index][1])]
        if boxes and atom_index in boxes:
            item["bbox"] = [float(value) for value in boxes[atom_index]]
        instances.append(item)
    return instances
