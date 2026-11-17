"""Capability boundary declarations for every unit operation.

Each unit operation declares what it accepts, what it guarantees on
success, what it refuses, and what it requires. The routing policy
matches evidence needs against these declarations instead of
trial-and-error escalation. Boundaries marked `empirical` were measured
on real documents — they are project knowledge, not documentation.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Capability:
    name: str
    accepts: str                  # input contract
    guarantees: str               # what a success verdict means
    refuses: tuple[str, ...]      # named exclusions (never silently handled)
    requires: tuple[str, ...]     # preconditions (server, gpu, network)
    empirical: tuple[str, ...] = field(default_factory=tuple)


REGISTRY: dict[str, Capability] = {
    # ---- structure <-> sequence cluster ----
    "macrocycle_sequence": Capability(
        name="macrocycle_sequence",
        accepts="cyclic peptide SMILES with a large macrocycle ring",
        guarantees="residue codes for ring residues; side chains walked",
        refuses=("linear peptides (empty result + named note)",),
        requires=("rdkit",),
        empirical=("validated on cyclopeptide patents "
                   "(KRAS/GPC3 reference sets)",),
    ),
    "linear_backbone_sequence": Capability(
        name="linear_backbone_sequence",
        accepts="linear peptide SMILES (amide-bond disconnection walk)",
        guarantees=("residue codes for chain residues; N-acyl caps and "
                    "terminal groups reported separately; unknown units "
                    "emitted as their own fragment SMILES"),
        refuses=("cyclic peptides (route to macrocycle_sequence)",
                 "residues with side-chain amides: the amide splits into a "
                 "named side fragment (Asn/Gln rejoin pending)"),
        requires=("rdkit",),
        empirical=("WO2025137159 C1: 16/18 units positionally exact; "
                   "Asn side-chain amide split is the known artifact",),
    ),
    "mass_gate": Capability(
        name="mass_gate",
        accepts=("assembled SMILES + a declared mass printed in the "
                 "document (mono or average convention)"),
        guarantees="accept/reject with delta and convention",
        refuses=("L/D stereo arbitration (same mass)",
                 "asymmetric diacid cap attachment end",
                 "rows with no declared mass (no reading-match fallback "
                 "exists unless the row carries its own parseable SMILES)"),
        requires=("rdkit",),
        empirical=("WO2025137159 MW columns are average convention; "
                   "delta 0.0 achieved on verified rows",),
    ),
    # ---- structure pages ----
    "ocsr_decode": Capability(
        name="ocsr_decode",
        accepts="cropped drawn-structure images (MolDetv2 boxes)",
        guarantees="SMILES candidate via MolNexTR/MolSight with routing "
                   "arbitration",
        refuses=("cyclic peptide macrocycles (not validated end-to-end; "
                 "audit 2026-11-12: training line has P0 defects, "
                 "acceptance not reached)",
                 "structures without clean box isolation"),
        requires=("gpu", "biocheminsight container (ultralytics, models)"),
        empirical=("WO2025137159 Tba/Ogl crops: decoded correctly; "
                   "C813 cyclic crop: linear decode + unmatched ring digit "
                   "(2 attempts, both failed)",),
    ),
    "paddle_table_recovery": Capability(
        name="paddle_table_recovery",
        accepts="page images and PDFs (PaddleOCR-VL-1.6 pipeline, "
                "paddle-ocr:vl16 stack)",
        guarantees=("markdown for the page's first table region; layout "
                    "boxes and structure-crop tags in raw_predictions "
                    "(shape differs from the pre-VL PPStructureV3 "
                    "pipeline)"),
        refuses=("full dense-sheet coverage in one call (blocks beyond "
                 "the first region are unread; the harness's "
                 "block-crop reader provides coverage)"),
        requires=("paddle-ocr service (serial celery worker)",),
        empirical=("the pre-VL PPStructureV3 pipeline collapsed dense "
                   "rotated sheets at ~21% row coverage; VL-1.6 reads the "
                   "first region character-exact (O3S/Ogl codes vs visual "
                   "ground truth)",),
    ),
    "textline_grid_reconstruction": Capability(
        name="textline_grid_reconstruction",
        accepts="text-line OCR boxes (PP-OCR det+rec) over a table page",
        guarantees=("table grid rebuilt geometrically from the page's own "
                    "header anchors; detector box splits rejoin within the "
                    "same column; rows without a printed id merge upward; "
                    "off-table lines are named, never dropped"),
        refuses=("headerless freeform legends (use the definition-page "
                 "miner)",),
        requires=("ppocr venv (cpu) inside the harness container",),
        empirical=("WO2025137159 EXAMPLE sheets: s325 4 rows mined, 2 "
                   "verified delta=0.0 end-to-end; character accuracy "
                   "inherits PP-OCR (already mass-proven on fig1 rows); "
                   "notation/SMILES cells OCR-case noise is rejected by "
                   "the mass contract, never accepted silently; on "
                   "ultra-tight sheets row association glues across "
                   "blocks (the VL block reader covers those pages)",),
    ),
    "native_scan_orientation": Capability(
        name="native_scan_orientation",
        accepts="pdfimages-extracted native page embeds (1-bit 300dpi)",
        guarantees=("page image in reading orientation; direction decided "
                    "by scoring the downstream OCR over {native, cw, ccw} "
                    "and keeping the best (high-confidence lines, count)"),
        refuses=("orienting pdftoppm renders (the PDF /Rotate transform "
                 "already applied there; re-orienting corrupts)",),
        requires=("pdfimages",),
        empirical=("WO2025137159: embeds are stored rotated, direction "
                   "VARIES per figure (s325 cw, s400 the other way); the "
                   "rule-energy probe returns 90 for both and cannot be "
                   "trusted for direction on table pages",),
    ),
    # ---- name resolution ----
    "opsin_names": Capability(
        name="opsin_names",
        accepts="systematic/IUPAC-style chemical names (local, no server)",
        guarantees="SMILES for parseable names",
        refuses=("residue acronyms (Oic, 1-Nal fail)",
                 "fuzzy matching (no such option exists)"),
        requires=("py2opsin + java",),
        empirical=("17/18 patent-style residue names parsed in testing; "
                   "'X or Y' alternatives need splitting first",),
    ),
    "ccd_candidates": Capability(
        name="ccd_candidates",
        accepts="PDB chem_comp 3-letter codes",
        guarantees="candidate SMILES; codes are CANDIDATES ONLY — catalog "
                  "namespaces collide with patent codes",
        refuses=("trust without arbitration (CCD HCY is a steroid, "
                 "TBA is tetrabutylammonium)",),
        requires=("network",),
        empirical=("WO2025137159 codes: 14/15 records exist, only 8/15 "
                   "correct skeletons",),
    ),
    # ---- page routing ----
    "vision_server": Capability(
        name="vision_server",
        accepts="page images with targeted questions (short outputs)",
        guarantees="JSON transcription within the vision timeout",
        refuses=("long verbatim transcriptions (timeout + retry storm "
                 "can deadlock self-hosted servers)",),
        requires=("vision endpoint (serial, concurrency 1)",),
        empirical=("gemma4-31b self-hosted: full-sheet prompts time out at "
                   "90s; targeted per-row queries succeed; WO2025137159 "
                   "EXAMPLE sheets: per-compound transcription timeouts are "
                   "systematic (90s x2 every sheet), not flaky",),
    ),
    "dense_ocr_service": Capability(
        name="dense_ocr_service",
        accepts="page images, one call per page (PaddleOCR-VL-1.6, 0.9B)",
        guarantees=("markdown of the page's first table region; notation "
                    "cells character-exact (O3S/Ogl codes verified against "
                    "visual ground truth); masses exact"),
        refuses=("full dense-sheet coverage in one call (the page-level "
                 "pipeline partitions to the first region; the rest is "
                 "unread) "
                 "loop-free long SMILES without the ngram guard "
                 "(decoder loops on arginine/aspartate motifs; "
                 "no_repeat_ngram_size=20 patched into generation_config)"),
        requires=("paddle-ocr:vl16 stack (redis + api + celery worker, "
                  "serial concurrency 1)",),
        empirical=("WO2025137159 sheet 325: single-pass covers block 231 "
                   "only; harness block-crop reads cover all 5 blocks in "
                   "~6s each; GLM-OCR tested 12x faster but misreads "
                   "O3S->038 and loops earlier; Unlimited-OCR-2 (6.4B "
                   "int8) garbled notation cells entirely and was "
                   "removed",),
    ),
    "definition_page_miner": Capability(
        name="definition_page_miner",
        accepts="R-group legend pages (left code column + right "
                "name/definition region)",
        guarantees=("code -> printed names / R-defs, positional pairing; "
                    "structure-drawing entries flagged for the OCSR route"),
        refuses=("pages whose left region holds prose (no plausible code "
                 "cells -> empty result, named)",),
        requires=("text-line OCR boxes",),
        empirical=("WO2025137159 pages 119/121: Trp_5CONH2, Trp_6CONH2 "
                   "recovered where the earlier pass lost them",),
    ),
}


def declared(name: str) -> Capability | None:
    return REGISTRY.get(name)


def rungs_supporting(need: str, policy) -> list[str]:
    """Rung names whose declared capabilities cover the evidence need.
    The policy's order is preserved; unsupported rungs are dropped before
    any call is made."""
    covers = {
        "memory": "any",
        "lexicon": "any",
        "builtin": "any",
        "document": "any",
        "opsin": "opsin_names",
        "ccd": "ccd_candidates",
        "pubchem": "ccd_candidates",
        "ocsr": "ocsr_decode",
        "vision": "vision_server",
        "llm": "vision_server",
    }
    out = []
    for rung in policy.token_rungs() if hasattr(policy, "token_rungs") else []:
        cap = covers.get(rung)
        if cap == "any" or cap in REGISTRY:
            out.append(rung)
    return out
