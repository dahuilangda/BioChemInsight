"""Agent-mode parsing session: the agent supplies perception, the
deployment runs deterministic chemistry.

Coexists with the model-driven pipeline (mode A). Neither mode affects
the other. The deployment never calls a model in agent mode — the agent
is the model.

Session lifecycle:
  start(pdf_path, output_dir)
      -> creates session, returns session_id
  record_structure_decisions(session_id, decisions)
      -> store per-page structure decisions from the agent
  record_measurements(session_id, observations)
      -> store measurement observations from the agent
  record_notation_rows(session_id, rows)
      -> store OCR'd notation rows
  run_chemistry(session_id)
      -> evidence ladder + SAR join on stored data
  result(session_id)
      -> final report and coverage
"""
from __future__ import annotations

import os
import uuid
from typing import Any

_sessions: dict[str, dict[str, Any]] = {}
_SESSION_TTL = 8 * 3600


def create_session(pdf_path: str, output_dir: str) -> dict[str, Any]:
    session_id = uuid.uuid4().hex
    os.makedirs(output_dir, exist_ok=True)
    _sessions[session_id] = {
        "pdf_path": pdf_path,
        "output_dir": output_dir,
        "structure_decisions": {},
        "measurement_observations": [],
        "notation_rows": [],
        "status": "created",
    }
    return {"session_id": session_id}


def get_session(session_id: str) -> dict[str, Any] | None:
    entry = _sessions.get(session_id)
    if entry is None:
        return None
    return entry


def record_structure_decisions(session_id: str,
                               decisions: list[dict[str, Any]]) -> dict:
    session = _sessions.get(session_id)
    if session is None:
        return {"error": "unknown session"}
    for d in decisions:
        page = int(d["page"])
        session["structure_decisions"][page] = d
    session["structure_pages"] = sorted(session["structure_decisions"])
    session["status"] = "structure_pages_recorded"
    return {"structure_pages": session["structure_pages"],
            "count": len(session["structure_pages"])}


def record_measurements(session_id: str,
                        observations: list[dict[str, Any]]) -> dict:
    session = _sessions.get(session_id)
    if session is None:
        return {"error": "unknown session"}
    session["measurement_observations"].extend(observations)
    session["status"] = "measurements_recorded"
    return {"total": len(session["measurement_observations"])}


def record_notation_rows(session_id: str,
                         rows: list[dict[str, Any]]) -> dict:
    session = _sessions.get(session_id)
    if session is None:
        return {"error": "unknown session"}
    session["notation_rows"].extend(rows)
    session["status"] = "notation_rows_recorded"
    return {"total": len(session["notation_rows"])}


def run_chemistry(session_id: str) -> dict[str, Any]:
    """Evidence ladder + SAR join on stored data. Deterministic only."""
    session = _sessions.get(session_id)
    if session is None:
        return {"error": "unknown session"}

    sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if sys_path not in os.sys.path:
        os.sys.path.insert(0, sys_path)

    from utils.peptide_harness import (NotationRow, PeptideResolutionHarness,
                                       write_results_csv)

    harness = PeptideResolutionHarness("AGENT_MODE")
    row_objs = []
    for r in session["notation_rows"]:
        row_objs.append(NotationRow(
            notation=r["notation"], seq_id=str(r.get("seq_id", "")),
            form=r.get("form", ""), declared_mass=r.get("mw"),
            cyclization=r.get("cyclization", {}),
            smiles_reading=r.get("smiles_reading", ""),
            source=f"agent:{r.get('page', '')}"))

    harness.bootstrap(row_objs)

    out_dir = session["output_dir"]
    csv_path = os.path.join(out_dir, "merged.csv")
    write_results_csv(harness, csv_path, row_objs)

    coverage = harness.coverage_report()
    session["coverage"] = coverage
    session["status"] = "chemistry_complete"

    report_path = os.path.join(out_dir, "report.md")
    _write_report(session, report_path)

    return {"status": "complete", "report": report_path,
            "coverage": coverage}


def _write_report(session: dict, report_path: str) -> None:
    coverage = session.get("coverage", {})
    rows = session.get("notation_rows", [])
    measurements = session.get("measurement_observations", [])
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Agent-mode extraction report\n\n")
        f.write(f"- notation rows: {len(rows)}\n")
        f.write(f"- measurement observations: {len(measurements)}\n")
        f.write(f"- verified: {coverage.get('rows_verified', 0)}\n")
        f.write(f"- deferred: {coverage.get('rows_deferred', 0)}\n")
        gaps = coverage.get("gap_reasons", {})
        if gaps:
            f.write("\n## Named gaps\n\n")
            for reason, count in sorted(gaps.items(), key=lambda x: -x[1])[:15]:
                f.write(f"- ({count}x) {reason}\n")


def get_result(session_id: str) -> dict[str, Any]:
    session = _sessions.get(session_id)
    if session is None:
        return {"error": "unknown session"}
    return {"session_id": session_id, "status": session["status"],
            "coverage": session.get("coverage", {}),
            "output_dir": session.get("output_dir", "")}
