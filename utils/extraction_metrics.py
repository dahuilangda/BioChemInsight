"""Offline extraction scoring against annotated references."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from rdkit import Chem

STATUSES = {"passed", "failed", "inconclusive"}


def combine_evaluation_statuses(statuses: Iterable[str]) -> str:
    values = tuple(statuses)
    if any(value not in STATUSES for value in values):
        raise ValueError("Unknown evaluation status")
    if "failed" in values:
        return "failed"
    if not values or "inconclusive" in values:
        return "inconclusive"
    return "passed"


def canonicalize_smiles(smiles: Any) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    params = Chem.SmilesParserParams()
    params.parseName = False
    molecule = Chem.MolFromSmiles(smiles, params)
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _expected_ids(identifiers: Iterable[str]) -> tuple[str, ...]:
    if isinstance(identifiers, (str, bytes)):
        raise TypeError("Expected compound IDs must be a collection of strings")
    values = tuple(identifiers)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Expected compound IDs must be non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError("Expected compound IDs must be unique")
    return values


def _index_predictions(predictions, field):
    if isinstance(predictions, Mapping):
        entries = list(predictions.items())
    elif isinstance(predictions, Sequence) and not isinstance(
        predictions, (str, bytes)
    ):
        entries = [
            (row.get("COMPOUND_ID"), row) if isinstance(row, Mapping) else (None, row)
            for row in predictions
        ]
    else:
        raise TypeError("Predictions must be a compound mapping or a record sequence")
    indexed = {}
    invalid = []
    for index, (identifier, value) in enumerate(entries):
        if not isinstance(identifier, str) or not identifier.strip():
            invalid.append({"row": index, "reason": "Missing compound ID"})
            continue
        if isinstance(value, Mapping):
            declared = value.get("COMPOUND_ID", identifier)
            if declared != identifier:
                invalid.append({"id": identifier, "reason": "Conflicting compound IDs"})
                continue
            value = value.get(field)
        indexed.setdefault(identifier, []).append(value)
    duplicates = [
        identifier for identifier, values in indexed.items() if len(values) > 1
    ]
    return indexed, invalid, duplicates, len(entries)


def _coverage(correct, expected):
    return round(correct / expected, 6) if expected else None


def _grade_status(expected, errors, reference_errors):
    if not expected:
        return "inconclusive"
    if errors:
        return "failed"
    if reference_errors:
        return "inconclusive"
    return "passed"


def _scalar_text(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    text = str(value)
    return "".join(text.split()) if text.strip() else None


def score_assay_values(
    predictions, expected_members: Mapping[str, Any]
) -> dict[str, Any]:
    expected = _expected_ids(expected_members)
    indexed, invalid, duplicates, emitted = _index_predictions(predictions, "value")
    reference_errors, correct, wrong = [], [], []
    valid_values = {}
    for identifier, values in indexed.items():
        if len(values) != 1:
            continue
        value = _scalar_text(values[0])
        if value is None:
            invalid.append(
                {"id": identifier, "reason": "Missing or invalid assay value"}
            )
        else:
            valid_values[identifier] = value
    for identifier in expected:
        reference = expected_members[identifier]
        value = reference.get("value") if isinstance(reference, Mapping) else reference
        target = _scalar_text(value)
        if target is None:
            reference_errors.append(
                {"id": identifier, "reason": "Missing reference value"}
            )
        elif identifier in valid_values:
            if valid_values[identifier] == target:
                correct.append(identifier)
            else:
                wrong.append(
                    {
                        "id": identifier,
                        "got": str(indexed[identifier][0]),
                        "want": str(value),
                    }
                )
    missing = [identifier for identifier in expected if identifier not in indexed]
    unexpected = [
        identifier for identifier in indexed if identifier not in expected_members
    ]
    errors = bool(missing or wrong or unexpected or invalid or duplicates)
    return {
        "status": _grade_status(expected, errors, reference_errors),
        "expected": len(expected),
        "emitted": emitted,
        "correct": len(correct),
        "missing": missing,
        "wrong": wrong,
        "unexpected_keys": unexpected,
        "invalid": invalid,
        "duplicates": duplicates,
        "reference_errors": reference_errors,
        "coverage": _coverage(len(correct), len(expected)),
    }


def _structure_predictions(predictions, expected):
    indexed, invalid, duplicates, emitted = _index_predictions(predictions, "SMILES")
    molecules = {}
    for identifier, values in indexed.items():
        if len(values) != 1:
            continue
        canonical = canonicalize_smiles(values[0])
        if canonical is None:
            invalid.append({"id": identifier, "reason": "Missing or invalid SMILES"})
        else:
            molecules[identifier] = canonical
    missing = [identifier for identifier in expected if identifier not in indexed]
    expected_set = set(expected)
    unexpected = [identifier for identifier in indexed if identifier not in expected_set]
    return molecules, {
        "expected_members": len(expected),
        "emitted": emitted,
        "missing": missing,
        "unexpected_keys": unexpected,
        "invalid": invalid,
        "duplicates": duplicates,
        "output_coverage": _coverage(
            sum(identifier in molecules for identifier in expected), len(expected)
        ),
    }


def score_structure_matches(
    predictions,
    references: Mapping[str, Any],
    *,
    expected_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    expected = _expected_ids(references if expected_ids is None else expected_ids)
    molecules, result = _structure_predictions(predictions, expected)
    matched, wrong, reference_errors = [], [], []
    available = 0
    for identifier in expected:
        canonical = canonicalize_smiles(references.get(identifier))
        if canonical is None:
            reference_errors.append(
                {"id": identifier, "reason": "Full molecular reference unavailable"}
            )
            continue
        available += 1
        if identifier in molecules:
            if canonical == molecules[identifier]:
                matched.append(identifier)
            else:
                wrong.append(
                    {"id": identifier, "got": molecules[identifier], "want": canonical}
                )
    errors = bool(
        wrong
        or result["missing"]
        or result["unexpected_keys"]
        or result["invalid"]
        or result["duplicates"]
    )
    return {
        **result,
        "status": _grade_status(expected, errors, reference_errors),
        "reference_members": len(expected),
        "available_references": available,
        "reference_errors": reference_errors,
        "matched": len(matched),
        "wrong": wrong,
        "coverage": _coverage(len(matched), len(expected)),
        "full_structure_accuracy": None
        if reference_errors
        else _coverage(len(matched), len(expected)),
    }


def score_structure_core(
    predictions, expected_ids: Iterable[str], core_smarts: str
) -> dict[str, Any]:
    expected = _expected_ids(expected_ids)
    molecules, result = _structure_predictions(predictions, expected)
    core = Chem.MolFromSmarts(core_smarts) if isinstance(core_smarts, str) else None
    embedded, violations = [], []
    if core is not None:
        for identifier in expected:
            if identifier in molecules:
                molecule = Chem.MolFromSmiles(molecules[identifier])
                if molecule.HasSubstructMatch(core, useChirality=True):
                    embedded.append(identifier)
                else:
                    violations.append(
                        {"id": identifier, "smiles": molecules[identifier]}
                    )
    errors = bool(
        violations
        or result["missing"]
        or result["unexpected_keys"]
        or result["invalid"]
        or result["duplicates"]
    )
    reference_errors = ["Full molecular references are not annotated"]
    if core is None:
        reference_errors.append("Invalid core SMARTS reference")
    core_status = _grade_status(
        expected, errors, [] if core is not None else reference_errors
    )
    return {
        **result,
        "status": _grade_status(expected, errors, reference_errors),
        "evaluation_scope": "core",
        "core_status": core_status,
        "embedded_core": len(embedded),
        "violations": violations,
        "reference_errors": reference_errors,
        "coverage": _coverage(len(embedded), len(expected)),
        "full_structure_accuracy": None,
    }
