from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COUNT_BUCKETS = ["1", "2", "3-4", "5-8", "9+"]
REAL_ATOMS = {"B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si", "Se", "Te", "As"}


def import_rdkit():
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return Chem


def cxsmiles_main(value: str) -> str:
    return str(value or "").split("|", 1)[0].strip()


def row_bucket(row: dict[str, str]) -> str:
    return str(row.get("annotation_r_bucket") or "").strip()


def analyze_cxsmiles(value: str) -> dict[str, Any]:
    Chem = import_rdkit()
    main = cxsmiles_main(value)
    mol = Chem.MolFromSmiles(main)
    if mol is None:
        return {
            "rdkit_parse_ok": False,
            "real_atom_count": 0,
            "dummy_atom_count": 0,
            "fragment_count": 0,
            "largest_fragment_real_atoms": 0,
        }
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    fragment_real_counts = [
        sum(1 for atom in fragment.GetAtoms() if atom.GetSymbol() in REAL_ATOMS)
        for fragment in fragments
    ]
    return {
        "rdkit_parse_ok": True,
        "real_atom_count": sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in REAL_ATOMS),
        "dummy_atom_count": sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0),
        "fragment_count": len(fragments),
        "largest_fragment_real_atoms": max(fragment_real_counts) if fragment_real_counts else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Markush candidate quality features before expensive CDK generation.")
    parser.add_argument("--candidate-plan-csv", required=True)
    parser.add_argument("--generation-readiness", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    with Path(args.candidate_plan_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    readiness = json.loads(Path(args.generation_readiness).read_text(encoding="utf-8")) if args.generation_readiness else {}

    counters: dict[str, Counter[str]] = defaultdict(Counter)
    numeric_by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    risky_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bucket_rows: Counter[str] = Counter()

    for row in rows:
        bucket = row_bucket(row)
        if bucket not in COUNT_BUCKETS:
            bucket = "missing"
        bucket_rows[bucket] += 1
        features = analyze_cxsmiles(str(row.get("cxsmiles") or ""))
        if not features["rdkit_parse_ok"]:
            counters["rdkit_parse_failed"].update([bucket])
            reason = "rdkit_parse_failed"
        elif int(features["real_atom_count"]) < 3:
            counters["real_atom_lt3"].update([bucket])
            reason = "real_atom_lt3"
        elif int(features["largest_fragment_real_atoms"]) < 3:
            counters["largest_fragment_real_atom_lt3"].update([bucket])
            reason = "largest_fragment_real_atom_lt3"
        elif int(features["fragment_count"]) > 1:
            counters["multi_fragment"].update([bucket])
            reason = "multi_fragment"
        else:
            counters["low_risk_by_simple_features"].update([bucket])
            reason = ""

        numeric_by_bucket[bucket].update([f"real_atoms:{int(features['real_atom_count'])}"])
        numeric_by_bucket[bucket].update([f"fragments:{int(features['fragment_count'])}"])
        if reason and len(risky_examples[reason]) < int(args.max_examples):
            risky_examples[reason].append(
                {
                    "plan_index": row.get("plan_index"),
                    "source_id": row.get("source_id"),
                    "bucket": bucket,
                    "features": features,
                    "cxsmiles": str(row.get("cxsmiles") or "")[:240],
                }
            )

    selected_by_bucket = {
        bucket: int((readiness.get("selected_candidate_rows_by_bucket") or {}).get(bucket) or 0)
        for bucket in COUNT_BUCKETS
    }
    available_by_bucket = {
        bucket: int((readiness.get("available_source_documents_by_bucket") or {}).get(bucket) or 0)
        for bucket in COUNT_BUCKETS
    }
    target_by_bucket = {
        bucket: int((readiness.get("bucket_targets") or {}).get(bucket) or 0)
        for bucket in COUNT_BUCKETS
    }
    extra_capacity_by_bucket = {
        bucket: max(0, int(available_by_bucket[bucket]) - int(selected_by_bucket[bucket])) for bucket in COUNT_BUCKETS
    }

    report = {
        "schema_version": "markush_candidate_quality_audit_v1",
        "candidate_plan_csv": str(args.candidate_plan_csv),
        "generation_readiness": str(args.generation_readiness),
        "row_count": len(rows),
        "bucket_rows": {bucket: int(bucket_rows.get(bucket, 0)) for bucket in COUNT_BUCKETS},
        "selected_by_bucket": selected_by_bucket,
        "available_by_bucket": available_by_bucket,
        "target_by_bucket": target_by_bucket,
        "extra_capacity_by_bucket": extra_capacity_by_bucket,
        "risk_counts_by_bucket": {
            name: {bucket: int(counter.get(bucket, 0)) for bucket in COUNT_BUCKETS}
            for name, counter in sorted(counters.items())
        },
        "risky_examples": risky_examples,
        "policy": {
            "audit_only": True,
            "simple_rdkit_features_do_not_replace_cdk_pose_generation": True,
            "prefiltering_requires_followup_generation_evidence": True,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
