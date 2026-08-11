from __future__ import annotations

import importlib.util
import re
from pathlib import Path


BRACKET_TOKEN_RE = re.compile(r"\[([^]\[]+)\]")
CHEMICAL_SYMBOLS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
    "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
    "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
    "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl",
    "Mc", "Lv", "Ts", "Og",
}
GENERIC_WORD_LABELS = {
    "Ar",
    "Aryl",
    "Het",
    "HetAr",
    "Hetaryl",
    "Heteroaryl",
    "Hal",
    "EWG",
    "Nu",
}
SINGLE_LETTER_LABELS = {"A", "D", "E", "G", "J", "K", "L", "M", "Q", "T", "U", "V", "W", "X", "Y", "Z"}
ELEMENT_SYMBOL_MARKUSH_LABEL_EXCEPTIONS = {
    "Ar",
    "Y",
}


def _load_molnextr_symbols() -> tuple[set[str], set[str]]:
    path = Path(__file__).resolve().parent / "MolNexTR" / "abbrs.py"
    spec = importlib.util.spec_from_file_location("_molnextr_abbrs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MolNexTR abbreviations from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return set(getattr(module, "ABBREVIATIONS", [])), set(getattr(module, "RGROUP_SYMBOLS", []))


ABBREVIATIONS, RGROUP_SYMBOLS = _load_molnextr_symbols()


def markush_label_category(label: str) -> str:
    """Classify text in a dummy/pseudo-atom label context.

    This is intentionally label-context only. A real atom parsed by RDKit/CDK
    as atomic number 18 is argon even though the text "Ar" can be a Markush
    aryl placeholder when it appears as a dummy/pseudo atom label.
    """
    text = normalize_label(label)
    if not text:
        return "empty"
    if is_generic_markush_class_label(text):
        return "markush_variable_label"
    if text in CHEMICAL_SYMBOLS and text not in ELEMENT_SYMBOL_MARKUSH_LABEL_EXCEPTIONS:
        return "element_symbol_or_nonvariable"
    if text in RGROUP_SYMBOLS:
        return "markush_variable_label"
    if text in ABBREVIATIONS:
        return "fixed_substituent_abbreviation"
    if re.fullmatch(r"R[A-Za-z0-9]*(?:'+)?", text):
        return "markush_variable_label"
    if re.fullmatch(r"R'+", text):
        return "markush_variable_label"
    if re.fullmatch(r"[A-Z][A-Za-z]?\d+(?:'+)?", text):
        return "markush_variable_label"
    if re.fullmatch(r"[A-Z]'+", text):
        return "markush_variable_label"
    if re.fullmatch(r"[A-Z]", text) and (text in SINGLE_LETTER_LABELS or text not in CHEMICAL_SYMBOLS):
        return "markush_variable_label"
    if text in CHEMICAL_SYMBOLS:
        return "element_symbol_or_nonvariable"
    return "nonvariable_pseudo_label"


def normalize_label(label: str) -> str:
    text = str(label or "").strip().replace("′", "'")
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    return text


def is_markush_label(label: str) -> bool:
    return markush_label_category(label) == "markush_variable_label"


def is_fixed_substituent_label(label: str) -> bool:
    return markush_label_category(label) == "fixed_substituent_abbreviation"


def is_generic_markush_class_label(label: str) -> bool:
    text = normalize_label(label)
    if text in GENERIC_WORD_LABELS:
        return True
    if re.fullmatch(r"Ar(?:\d+|[a-z]|[A-Z][A-Za-z0-9']*)", text):
        return True
    if re.fullmatch(r"Het(?:Ar|aryl)?(?:\d+|[a-z]|[A-Z][A-Za-z0-9']*)?", text):
        return True
    return False


def expected_labels(smiles: str) -> list[str]:
    return sorted({normalize_label(label) for label in BRACKET_TOKEN_RE.findall(str(smiles)) if is_markush_label(label)})


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
        if "'" in label:
            features.add("has_prime")
        if re.search(r"\d", label):
            features.add("has_digit")
        if re.search(r"\d{2,}", label):
            features.add("has_digit_10plus")
        if re.fullmatch(r"R[a-z]", label):
            features.add("has_r_letter")
        if is_generic_markush_class_label(label):
            features.add("has_generic_word")
        if label in SINGLE_LETTER_LABELS:
            features.add("has_single_letter")
    return features


def label_weight(smiles: str, feature_weights: dict[str, float], default_weight: float = 1.0) -> tuple[float, set[str]]:
    labels = expected_labels(smiles)
    features = label_features(labels)
    weight = float(default_weight)
    for feature in features:
        if feature in feature_weights:
            weight *= float(feature_weights[feature])
    return weight, features
