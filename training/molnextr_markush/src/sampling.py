from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd

from training.molnextr_markush.src.curriculum import assign_bucket, profile_row
from training.molnextr_markush.src.labels import expected_labels, label_features


@dataclass(frozen=True)
class RowSamplingContext:
    source: str
    label_count: str
    task: str
    features: set[str]
    bucket: str
    has_star: bool


def source_key(row: pd.Series | dict) -> str:
    source_arrow = str(row.get("source_arrow") or "")
    if source_arrow.startswith("pose_factory:"):
        return source_arrow.split(":", 2)[1]
    if source_arrow == "mg2_reverse_graph" or source_arrow.startswith("mg2_reverse_graph:"):
        return "mg2"
    source_id = str(row.get("source_id") or "")
    return source_id.split(":", 1)[0] if ":" in source_id else "official"


def label_count_key(labels: list[str]) -> str:
    count = len(set(labels))
    return "4plus" if count >= 4 else str(count)


def task_key(source: str, bucket: str, labels: list[str], smiles: str) -> str:
    attachment_sources = {
        "fragment_endpoint",
        "wavy_fragment",
        "document_fragment",
    }
    if source in attachment_sources:
        return "attachment_fragment"
    if bucket == "ordinary_structure" and "*" not in str(smiles or "") and not labels:
        return "ordinary_structure"
    count = len(set(labels))
    if count <= 0:
        return "markush_unlabeled_or_query"
    if count >= 4:
        return "markush_label_4plus"
    return f"markush_label_{count}"


def annotate_curriculum_buckets(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    curriculum = config.get("curriculum") or {}
    if not curriculum.get("enabled"):
        return df
    if "structure_type_bucket" in df.columns:
        return df
    if not curriculum.get("profile_if_missing", True):
        raise ValueError("curriculum.enabled requires structure_type_bucket unless profile_if_missing is true")
    annotated = df.copy()
    annotated["structure_type_bucket"] = [assign_bucket(profile_row(row)) for _, row in annotated.iterrows()]
    return annotated


def normalized_counter(counter: Counter, total: float | None = None) -> dict[str, float]:
    denominator = float(total if total is not None else sum(counter.values()))
    if denominator <= 0:
        return {}
    return {
        str(key): float(value) / denominator
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def normalized_mix(mix: dict[str, float]) -> dict[str, float]:
    total = sum(max(float(value), 0.0) for value in mix.values())
    if total <= 0:
        return {}
    return {str(key): max(float(value), 0.0) / total for key, value in mix.items()}


def weighted_counts(keys: list[str], weights: list[float]) -> Counter:
    counter = Counter()
    for key, weight in zip(keys, weights):
        counter[str(key)] += float(weight)
    return counter


def apply_target_mix(weights: list[float], keys: list[str], target_mix: dict[str, float]) -> list[float]:
    target = normalized_mix(target_mix)
    if not target:
        return weights

    total_weight = sum(weights)
    if total_weight <= 0:
        return weights

    current = normalized_counter(weighted_counts(keys, weights), total_weight)
    multipliers = {}
    for key, target_share in target.items():
        current_share = current.get(key, 0.0)
        multipliers[key] = 0.0 if target_share <= 0 else target_share / max(current_share, 1e-12)

    return [
        float(weight) * multipliers[str(key)]
        for key, weight in zip(keys, weights)
    ]


def validate_target_mix_coverage(keys: list[str], target_mix: dict[str, float], name: str) -> None:
    if not target_mix:
        return
    target = normalized_mix(target_mix)
    if not target:
        raise ValueError(f"{name} target mix must contain at least one positive weight")
    observed = {str(key) for key in keys}
    configured = {str(key) for key in target_mix}
    missing = sorted(observed - configured)
    if missing:
        raise ValueError(f"{name} target mix missing observed groups: {missing}")


def apply_target_mixes(
    weights: list[float],
    dimensions: list[tuple[list[str], dict[str, float]]],
    iterations: int,
) -> list[float]:
    balanced = list(weights)
    for _ in range(max(int(iterations), 1)):
        for keys, target_mix in dimensions:
            balanced = apply_target_mix(balanced, keys, target_mix)
    return balanced


def summarize_weighted_dimension(keys: list[str], weights: list[float]) -> dict[str, dict[str, float]]:
    count_values = Counter(str(key) for key in keys)
    weight_values = weighted_counts(keys, weights)
    total_weight = sum(weights)
    return {
        "counts": dict(sorted(count_values.items())),
        "observed_ratio": normalized_counter(count_values, len(keys)),
        "weighted_mass": {key: round(float(value), 6) for key, value in sorted(weight_values.items())},
        "weighted_ratio": {
            key: round(value, 6)
            for key, value in normalized_counter(weight_values, total_weight).items()
        },
    }


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(max(int(round((len(sorted_values) - 1) * q)), 0), len(sorted_values) - 1)
    return float(sorted_values[index])


def weight_distribution_summary(weights: list[float]) -> dict[str, float]:
    if not weights:
        return {}
    ordered = sorted(float(weight) for weight in weights)
    total = sum(ordered)
    squared_total = sum(weight * weight for weight in ordered)
    top_1pct_count = max(int(round(len(ordered) * 0.01)), 1)
    top_5pct_count = max(int(round(len(ordered) * 0.05)), 1)
    return {
        "min": ordered[0],
        "p50": quantile(ordered, 0.50),
        "p90": quantile(ordered, 0.90),
        "p99": quantile(ordered, 0.99),
        "p999": quantile(ordered, 0.999),
        "max": ordered[-1],
        "mean": total / len(ordered),
        "effective_sample_size": (total * total / squared_total) if squared_total > 0 else 0.0,
        "top_1pct_mass_ratio": sum(ordered[-top_1pct_count:]) / total if total > 0 else 0.0,
        "top_5pct_mass_ratio": sum(ordered[-top_5pct_count:]) / total if total > 0 else 0.0,
    }


def compute_curriculum_row_weights(df: pd.DataFrame, config: dict) -> tuple[list[float], dict]:
    curriculum = config.get("curriculum") or {}
    label_curriculum = config.get("label_curriculum") or {}
    if not curriculum.get("enabled") and not label_curriculum.get("enabled"):
        return [], {}
    if curriculum.get("enabled") and "structure_type_bucket" not in df.columns:
        raise ValueError("curriculum sampling requires structure_type_bucket")

    bucket_weights = {str(key): float(value) for key, value in (curriculum.get("bucket_weights") or {}).items()}
    default_weight = float(curriculum.get("default_weight", 1.0))
    source_weights = {
        str(key): float(value)
        for key, value in (label_curriculum.get("source_weights") or {}).items()
    }
    feature_weights = {
        str(key): float(value)
        for key, value in (label_curriculum.get("feature_weights") or {}).items()
    }
    row_boosts = {
        str(key): float(value)
        for key, value in (label_curriculum.get("row_boosts") or {}).items()
    }
    target_source_mix = {
        str(key): float(value)
        for key, value in (label_curriculum.get("target_source_mix") or {}).items()
    }
    target_label_count_mix = {
        str(key): float(value)
        for key, value in (label_curriculum.get("target_label_count_mix") or {}).items()
    }
    target_task_mix = {
        str(key): float(value)
        for key, value in (label_curriculum.get("target_task_mix") or {}).items()
    }
    label_default_weight = float(label_curriculum.get("default_weight", 1.0))
    max_row_weight = float(label_curriculum.get("max_row_weight", 0.0) or 0.0)
    balance_iterations = int(label_curriculum.get("target_balance_iterations", 8))

    contexts: list[RowSamplingContext] = []
    label_feature_counts = Counter()
    base_weights = []

    for _, row in df.iterrows():
        smiles = str(row.get("SMILES") or "")
        labels = expected_labels(smiles)
        features = label_features(labels)
        label_feature_counts.update(features)

        bucket = str(row.get("structure_type_bucket") or "")
        weight = default_weight
        if curriculum.get("enabled"):
            weight = float(bucket_weights.get(bucket, default_weight))

        source = source_key(row)
        weight *= float(source_weights.get(source, 1.0))

        if label_curriculum.get("enabled"):
            multiplier = label_default_weight
            for feature in features:
                if feature in feature_weights:
                    multiplier *= float(feature_weights[feature])
            weight *= multiplier

        contexts.append(
            RowSamplingContext(
                source=source,
                label_count=label_count_key(labels),
                task=task_key(source, bucket, labels, smiles),
                features=features,
                bucket=bucket,
                has_star="*" in smiles,
            )
        )
        base_weights.append(max(float(weight), 0.0))

    for index, context in enumerate(contexts):
        if context.task != "attachment_fragment" and context.has_star:
            base_weights[index] *= float(row_boosts.get("non_attachment_with_star", 1.0))
        if context.task == "attachment_fragment" and context.has_star:
            base_weights[index] *= float(row_boosts.get("attachment_with_star", 1.0))

    source_keys = [context.source for context in contexts]
    label_count_keys = [context.label_count for context in contexts]
    task_keys = [context.task for context in contexts]
    bucket_keys = [context.bucket for context in contexts]

    dimensions = []
    if target_source_mix:
        validate_target_mix_coverage(source_keys, target_source_mix, "source")
        dimensions.append((source_keys, target_source_mix))
    if target_task_mix:
        validate_target_mix_coverage(task_keys, target_task_mix, "task")
        dimensions.append((task_keys, target_task_mix))
    if target_label_count_mix:
        validate_target_mix_coverage(label_count_keys, target_label_count_mix, "label_count")
        dimensions.append((label_count_keys, target_label_count_mix))
    row_weights = apply_target_mixes(base_weights, dimensions, balance_iterations) if dimensions else list(base_weights)

    clipped_rows = 0
    if max_row_weight > 0:
        clipped_rows = sum(1 for weight in row_weights if weight > max_row_weight)
        row_weights = [min(weight, max_row_weight) for weight in row_weights]

    if not any(weight > 0 for weight in row_weights):
        raise ValueError("curriculum row weights are all zero")

    report = {
        "enabled": True,
        "bucket_weights": bucket_weights,
        "default_weight": default_weight,
        "num_rows": int(len(df)),
        "label_curriculum": {
            "enabled": bool(label_curriculum.get("enabled")),
            "feature_counts": dict(label_feature_counts),
            "feature_weights": feature_weights,
            "row_boosts": row_boosts,
            "source_weights": source_weights,
            "target_source_mix": normalized_mix(target_source_mix),
            "target_task_mix": normalized_mix(target_task_mix),
            "target_label_count_mix": normalized_mix(target_label_count_mix),
            "target_balance_iterations": balance_iterations,
            "default_weight": label_default_weight,
            "max_row_weight": max_row_weight,
        },
        "dimensions": {
            "source": summarize_weighted_dimension(source_keys, row_weights),
            "task": summarize_weighted_dimension(task_keys, row_weights),
            "label_count": summarize_weighted_dimension(label_count_keys, row_weights),
            "bucket": summarize_weighted_dimension(bucket_keys, row_weights),
        },
        "base_row_weight_summary": {
            **weight_distribution_summary(base_weights),
        },
        "row_weight_summary": {
            **weight_distribution_summary(row_weights),
            "clipped_rows": clipped_rows,
        },
    }
    return row_weights, report
