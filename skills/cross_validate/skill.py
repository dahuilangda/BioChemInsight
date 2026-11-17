"""Peptide graph construction and residue-code comparison."""

from skills.registry import register
from utils.residue_typer import compare_with_anchor
from utils.residue_typer import smiles_to_sequence as decode_residue_codes
from utils.sequence_parser import build_cyclopeptide_smiles

_CODES_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "items": {"type": "string", "minLength": 1},
}
_SMILES_SCHEMA = {"type": "string", "minLength": 1}


def _codes(payload, minimum=1):
    values = payload.get("codes")
    if (
        not isinstance(values, list)
        or len(values) < minimum
        or any(not isinstance(code, str) or not code.strip() for code in values)
    ):
        raise ValueError(
            f"codes must contain at least {minimum} non-empty residue strings"
        )
    return values


def _smiles(payload):
    value = payload.get("smiles")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("smiles must be a non-empty string")
    return value


@register(
    "sequence_to_smiles",
    "chemistry",
    "Build a cyclic peptide from declared residue codes.",
    input_schema={
        "type": "object",
        "required": ["codes"],
        "properties": {"codes": {**_CODES_SCHEMA, "minItems": 3}},
    },
)
async def sequence_to_smiles(payload: dict) -> dict:
    try:
        codes = _codes(payload, minimum=3)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    smiles = build_cyclopeptide_smiles(codes)
    if smiles is None:
        return {"ok": False, "error": "Could not assemble the declared residue codes"}
    return {"ok": True, "smiles": smiles}


@register(
    "smiles_to_sequence",
    "chemistry",
    "Read residue codes from a cyclic peptide graph; stereochemistry is not verified.",
    input_schema={
        "type": "object",
        "required": ["smiles"],
        "properties": {"smiles": _SMILES_SCHEMA},
    },
)
async def smiles_to_sequence(payload: dict) -> dict:
    try:
        smiles = _smiles(payload)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    codes, ok, detail = decode_residue_codes(smiles)
    return {
        "ok": ok,
        "codes": codes,
        "detail": detail,
        "validation_scope": "residue_codes",
        "stereochemistry_verified": False,
        "source_identity_verified": False,
    }


@register(
    "cross_validate",
    "chemistry",
    "Compare residue codes with an anchor; stereochemistry and source identity are not verified.",
    input_schema={
        "type": "object",
        "required": ["smiles", "codes"],
        "properties": {"smiles": _SMILES_SCHEMA, "codes": _CODES_SCHEMA},
    },
)
async def cross_validate(payload: dict) -> dict:
    try:
        smiles, codes = _smiles(payload), _codes(payload)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    result = compare_with_anchor(smiles, codes)
    return {
        **result,
        "validation_scope": "residue_codes",
        "stereochemistry_verified": False,
        "source_identity_verified": False,
    }
