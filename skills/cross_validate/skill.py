"""Cross-validation skill: both directions between a cyclopeptide
sequence and a SMILES, plus the agreement verdict when both exist."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from skills.registry import register
from utils.residue_typer import smiles_to_sequence, compare_with_anchor
from utils.sequence_parser import build_cyclopeptide_smiles


@register("sequence_to_smiles", "chemistry",
          "residue codes -> assembled cyclopeptide SMILES")
async def sequence_to_smiles(payload: dict) -> dict:
    codes = payload.get("codes") or []
    if not codes:
        return {"ok": False, "error": "no codes given"}
    smiles = build_cyclopeptide_smiles(codes)
    if smiles is None:
        return {"ok": False,
                "error": "assembly returned None: unknown residue code, "
                         "unparseable fragment, or fewer than 3 usable codes"}
    return {"ok": True, "smiles": smiles}


@register("smiles_to_sequence", "chemistry",
          "cyclopeptide SMILES -> typed residue codes")
async def smiles_to_sequence(payload: dict) -> dict:
    codes, ok, detail = smiles_to_sequence(payload.get("smiles", ""))
    return {"ok": ok, "codes": codes, "detail": detail}


@register("cross_validate", "chemistry",
          "SMILES vs anchor codes -> per-position agreement")
async def cross_validate(payload: dict) -> dict:
    return compare_with_anchor(payload.get("smiles", ""),
                               payload.get("codes") or [])
