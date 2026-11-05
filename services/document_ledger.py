"""Document Ledger: typed state object for cross-capability patent parsing.

Design principles (from Gero et al. self-verification, Mint-Agent evidence
ledger, and the MolParser/HARVEST patent-mining literature):
- Every extracted entity carries provenance (skill, input_ref, confidence)
- Contradictions between skills are first-class rows, never overwritten
- Gaps (missing fields after all skills attempted) enable gap-filling
- Cross-capability inference runs only when the ledger makes both sides
  visible (e.g., a figure whose decode failed but whose sequence anchor
  succeeded — the ledger makes both visible to the repair logic)

Usage:
    ledger = DocumentLedger()
    ledger.add_figure(page=3, bbox=[...], smiles="...", id="SEQ53",
                      evidence=EvidenceRecord(skill="recognize", ...))
    ledger.gap_fill()   # cross-capability inference
    ledger.report()     # terminal states + outstanding gaps
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EntityKind(str, Enum):
    FIGURE = "figure"
    SEQUENCE = "sequence"
    ACTIVITY = "activity"
    MARKUSH = "markush"


class Resolution(str, Enum):
    RESOLVED = "RESOLVED"
    REVIEW_QUEUED = "REVIEW_QUEUED"
    UNSOLVABLE = "UNSOLVABLE"
    PENDING = "PENDING"


class GapReason(str, Enum):
    DECODE_FAILED = "DECODE_FAILED"
    LABEL_UNREADABLE = "LABEL_UNREADABLE"
    ID_CONFLICT = "ID_CONFLICT"
    SEQUENCE_MISMATCH = "SEQUENCE_MISMATCH"
    ACTIVITY_LINK_EMPTY = "ACTIVITY_LINK_EMPTY"
    VOCABULARY_UNKNOWN = "VOCABULARY_UNKNOWN"
    VERIFICATION_CONTRADICTION = "VERIFICATION_CONTRADICTION"


@dataclass
class EvidenceRecord:
    skill: str
    input_ref: str          # crop path / text span / page number
    output_digest: str      # hash or prefix of the output
    confidence: float       # 0.0-1.0
    detail: str = ""
    ts: float = 0.0

    def __post_init__(self):
        if not self.ts:
            self.ts = time.time()

    @property
    def record_id(self) -> str:
        raw = f"{self.skill}:{self.input_ref}:{self.output_digest}:{self.ts}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class Contradiction:
    kind: str               # e.g. "smiles-mismatch", "id-conflict", "seq-mismatch"
    entity_id: str
    source_a: str           # skill or evidence record id
    source_b: str
    detail_a: str
    detail_b: str
    resolution: Optional[str] = None  # set when resolved (e.g. "grammar path wins")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "entity_id": self.entity_id,
                "source_a": self.source_a, "source_b": self.source_b,
                "detail_a": self.detail_a, "detail_b": self.detail_b,
                "resolution": self.resolution}


@dataclass
class Gap:
    entity_id: str
    field: str              # e.g. "smiles", "compound_id", "activity"
    reason_code: str        # GapReason
    attempted: List[str] = field(default_factory=list)  # skills tried
    filled_by: Optional[str] = None  # set when a later skill fills it

    def to_dict(self) -> dict:
        return {"entity_id": self.entity_id, "field": self.field,
                "reason_code": self.reason_code, "attempted": self.attempted,
                "filled_by": self.filled_by}


@dataclass
class FigureEntity:
    entity_id: str
    page: int
    bbox: List[int]
    smiles: Optional[str] = None
    smiles_valid: bool = False
    compound_id: Optional[str] = None
    compound_name: Optional[str] = None
    seq_id: Optional[str] = None
    seq_residues: Optional[List[str]] = None
    route_kind: Optional[str] = None      # markush | complete
    verification: Optional[Dict] = None
    activity_links: List[Dict] = field(default_factory=list)
    resolution: Resolution = Resolution.PENDING
    reason_code: Optional[str] = None
    evidence: List[EvidenceRecord] = field(default_factory=list)
    contradictions: List[Contradiction] = field(default_factory=list)
    gaps: List[Gap] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id, "page": self.page, "bbox": self.bbox,
            "smiles": self.smiles, "smiles_valid": self.smiles_valid,
            "compound_id": self.compound_id, "compound_name": self.compound_name,
            "seq_id": self.seq_id, "seq_residues": self.seq_residues,
            "route_kind": self.route_kind,
            "verification": self.verification,
            "activity_links": self.activity_links,
            "resolution": self.resolution.value if isinstance(self.resolution, Resolution) else self.resolution,
            "reason_code": self.reason_code,
            "evidence": [{"skill": e.skill, "input_ref": e.input_ref,
                          "output_digest": e.output_digest,
                          "confidence": e.confidence} for e in self.evidence],
            "contradictions": [c.to_dict() for c in self.contradictions],
            "gaps": [g.to_dict() for g in self.gaps],
        }


@dataclass
class SequenceEntity:
    entity_id: str
    page: int
    codes: List[str]
    smiles: Optional[str] = None
    seq_id: Optional[str] = None
    source: str = "sequence_table"

    def to_dict(self) -> dict:
        return {"entity_id": self.entity_id, "page": self.page,
                "codes": self.codes, "smiles": self.smiles,
                "seq_id": self.seq_id, "source": self.source}


@dataclass
class ActivityEntity:
    entity_id: str
    page: int
    compound_id: str
    values: Dict[str, Any]
    assay: str
    confidence: float

    def to_dict(self) -> dict:
        return {"entity_id": self.entity_id, "page": self.page,
                "compound_id": self.compound_id, "values": self.values,
                "assay": self.assay, "confidence": self.confidence}


class DocumentLedger:
    """Document-scoped accumulator for all extracted entities, evidence,
    contradictions, and gaps. Enables cross-capability gap-filling:
    a figure's decode failure can be rescued by the sequence path, and an
    activity table's row can be linked to a figure via the shared seq_id —
    only because the ledger makes both visible to the repair logic."""

    def __init__(self, document_id: str):
        self.document_id = document_id
        self.figures: Dict[str, FigureEntity] = {}
        self.sequences: Dict[str, SequenceEntity] = {}
        self.activities: Dict[str, ActivityEntity] = {}
        self.contradictions: List[Contradiction] = []
        self.gaps: List[Gap] = []
        self._seq_counter = 0
        self._act_counter = 0

    def add_figure(self, entity: FigureEntity) -> str:
        if not entity.entity_id:
            entity.entity_id = f"fig_p{entity.page}_b{entity.bbox[0]}_{entity.bbox[1]}"
        self.figures[entity.entity_id] = entity
        return entity.entity_id

    def add_sequence(self, entity: SequenceEntity) -> str:
        self._seq_counter += 1
        entity.entity_id = entity.entity_id or f"seq_{entity.page}_{self._seq_counter}"
        self.sequences[entity.entity_id] = entity
        return entity.entity_id

    def add_activity(self, entity: ActivityEntity) -> str:
        self._act_counter += 1
        entity.entity_id = entity.entity_id or f"act_{entity.page}_{self._act_counter}"
        self.activities[entity.entity_id] = entity
        return entity.entity_id

    def add_contradiction(self, c: Contradiction):
        self.contradictions.append(c)
        fig = self.figures.get(c.entity_id)
        if fig:
            fig.contradictions.append(c)

    def add_gap(self, g: Gap):
        self.gaps.append(g)
        fig = self.figures.get(g.entity_id)
        if fig:
            fig.gaps.append(g)

    def fill_gap(self, entity_id: str, field: str, value: Any,
                 filled_by: str):
        """Fill a known gap — recorded as evidence, never silent."""
        fig = self.figures.get(entity_id)
        if fig is None:
            return
        setattr(fig, field, value)
        for g in fig.gaps:
            if g.field == field and g.filled_by is None:
                g.filled_by = filled_by
                break
        fig.evidence.append(EvidenceRecord(
            skill="gap_fill", input_ref=filled_by,
            output_digest=str(value)[:80], confidence=0.7))

    # cross-capability gap-filling -----------------------------------------

    def gap_fill(self):
        """Cross-capability inference — only runs when the ledger makes
        both sides visible. Never guesses; always records the source."""
        # rule 1: figure without SMILES but with seq_id → sequence lookup
        for fig in self.figures.values():
            if fig.smiles or not fig.seq_id:
                continue
            match = self._find_sequence_by_id(fig.seq_id, fig.page)
            if match:
                from utils.sequence_parser import build_cyclopeptide_smiles, \
                    parse_cyclopeptide_notation
                codes = parse_cyclopeptide_notation(match.codes)
                built = build_cyclopeptide_smiles(codes) if codes else None
                if built:
                    self.fill_gap(fig.entity_id, "smiles", built,
                                  filled_by=f"sequence:{match.entity_id}")

        # rule 2: figure without compound_id but activity table has a match
        for fig in self.figures.values():
            if fig.compound_id or not fig.activity_links:
                continue
            for link in fig.activity_links:
                if link.get("compound_id"):
                    self.fill_gap(fig.entity_id, "compound_id",
                                  link["compound_id"],
                                  filled_by=f"activity:{link.get('source', 'table')}")
                    break

        # rule 3: contradiction between two ID sources → keep both, flag
        for fig in self.figures.values():
            id_sources = [e for e in fig.evidence
                          if "identify" in e.skill or "label" in e.skill]
            if len(id_sources) >= 2:
                digests = {e.output_digest for e in id_sources}
                if len(digests) > 1:
                    self.add_contradiction(Contradiction(
                        kind="id-conflict", entity_id=fig.entity_id,
                        source_a=id_sources[0].skill,
                        source_b=id_sources[-1].skill,
                        detail_a=id_sources[0].output_digest[:60],
                        detail_b=id_sources[-1].output_digest[:60]))

    def _find_sequence_by_id(self, seq_id: str, page: int) -> Optional[SequenceEntity]:
        """Match a figure's SEQ label ("SEQ ID NO: 53" / "SEQ53" / "53")
        against registered sequence entities by their declared seq_id."""
        clean = re.sub(r"[^0-9A-Z]", "", seq_id.strip().upper())
        digits = re.sub(r"[^0-9]", "", clean)
        for seq in self.sequences.values():
            declared = re.sub(r"[^0-9A-Z]", "",
                              (seq.seq_id or "").upper())
            if declared and declared in (clean, digits):
                return seq
        # page-proximity only resolves when it is unambiguous
        candidates = [s for s in self.sequences.values()
                      if abs(s.page - page) <= 2]
        if len(candidates) == 1:
            return candidates[0]
        return None

    # reporting ------------------------------------------------------------

    def report(self) -> dict:
        figures = [f.to_dict() for f in self.figures.values()]
        resolved = sum(1 for f in self.figures.values()
                       if f.resolution == Resolution.RESOLVED)
        review = sum(1 for f in self.figures.values()
                     if f.resolution == Resolution.REVIEW_QUEUED)
        unsolvable = sum(1 for f in self.figures.values()
                         if f.resolution == Resolution.UNSOLVABLE)
        pending = sum(1 for f in self.figures.values()
                      if f.resolution == Resolution.PENDING)
        return {
            "document_id": self.document_id,
            "figures": figures,
            "sequences": [s.to_dict() for s in self.sequences.values()],
            "activities": [a.to_dict() for a in self.activities.values()],
            "summary": {
                "figures_total": len(self.figures),
                "resolved": resolved, "review_queued": review,
                "unsolvable": unsolvable, "pending": pending,
                "sequences_total": len(self.sequences),
                "activities_total": len(self.activities),
                "contradictions": len(self.contradictions),
                "open_gaps": sum(1 for g in self.gaps if not g.filled_by),
            },
        }

    def save(self, path: str):
        data = self.report()
        data["figures"] = [f.to_dict() for f in self.figures.values()]
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "DocumentLedger":
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
        ledger = cls(data.get("document_id", ""))
        for fd in data.get("figures", []):
            e = FigureEntity(entity_id=fd["entity_id"], page=fd["page"],
                             bbox=fd["bbox"])
            for key in ("smiles", "compound_id", "compound_name", "seq_id",
                        "route_kind", "reason_code"):
                setattr(e, key, fd.get(key))
            e.smiles_valid = fd.get("smiles_valid", False)
            e.seq_residues = fd.get("seq_residues")
            e.verification = fd.get("verification")
            e.activity_links = fd.get("activity_links", [])
            e.resolution = Resolution(fd.get("resolution", "PENDING"))
            ledger.figures[e.entity_id] = e
        for sd in data.get("sequences", []):
            ledger.sequences[sd["entity_id"]] = SequenceEntity(
                entity_id=sd["entity_id"], page=sd["page"],
                codes=sd.get("codes", []), smiles=sd.get("smiles"),
                seq_id=sd.get("seq_id"),
                source=sd.get("source", "sequence_table"))
        for ad in data.get("activities", []):
            ledger.activities[ad["entity_id"]] = ActivityEntity(
                entity_id=ad["entity_id"], page=ad["page"],
                compound_id=ad.get("compound_id", ""),
                values=ad.get("values", {}), assay=ad.get("assay", ""),
                confidence=ad.get("confidence", 0.5))
        return ledger
