"""Unassisted recognition metrics with explicit annotation semantics."""

from collections import defaultdict

from .abbreviations import resolve_abbreviations
from .structure import StructureError, parse_structure


def compare(gold_text, predicted_text):
    gold = parse_structure(gold_text)
    try:
        predicted = parse_structure(predicted_text)
    except StructureError:
        return {"parse_success": False, "literal_exact": False, "resolved_exact": False}
    result = {"parse_success": True, "literal_exact": gold.equivalent(predicted)}
    try:
        result["resolved_exact"] = resolve_abbreviations(gold).document.equivalent(
            resolve_abbreviations(predicted).document
        )
    except StructureError:
        result["resolved_exact"] = False
        result["resolution_failed"] = True
    return result


def aggregate(rows, *, group_by="category"):
    categories = defaultdict(list)
    for row in rows:
        value = row[group_by]
        if value is not None:
            categories[value].append(row)
    return {
        name: {
            "count": len(values),
            **{
                key: sum(row[key] for row in values) / len(values)
                for key in (
                    "parse_success",
                    "literal_exact",
                    "resolved_exact",
                    "terminated",
                )
            },
        }
        for name, values in categories.items()
    }
