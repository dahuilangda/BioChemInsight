"""Sidecar/router threshold calibration utilities.

Extracted from the original ``tools/train_moe.py`` monolith.
"""
from __future__ import annotations

from ._common import *  # noqa: F401,F403

def _progress(label: str, done: int, total: int, every: int) -> None:
    if every and every > 0 and (done == total or done % every == 0):
        print(f"{label}: {done}/{total}", flush=True)


def _sidecar_threshold_vector(sidecar_threshold, num_experts: int) -> np.ndarray:
    """Normalize the deployment threshold contract to one value per expert."""
    num_experts = max(1, int(num_experts))
    if isinstance(sidecar_threshold, dict):
        values = [float("inf")] + [1.01] * (num_experts - 1)
        for key, raw in sidecar_threshold.items():
            idx = int(key)
            if 0 <= idx < num_experts:
                values[idx] = float(raw)
    elif isinstance(sidecar_threshold, (list, tuple, np.ndarray)):
        raw_values = [float(value) for value in list(sidecar_threshold)]
        values = raw_values[:num_experts]
        values.extend([1.01] * (num_experts - len(values)))
    else:
        values = [float("inf")] + [float(sidecar_threshold)] * (num_experts - 1)
    values[0] = float("inf")
    return np.asarray(values, dtype=np.float64)


def _sidecar_acceptance_from_probs(probs: np.ndarray, sidecar_threshold):
    """Apply the exact inference rule: threshold is selected by argmax expert."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] < 1:
        raise ValueError(f"router probabilities must be (N, K), got {probs.shape}")
    thresholds = _sidecar_threshold_vector(sidecar_threshold, probs.shape[1])
    argmax = probs.argmax(axis=1)
    top = probs.max(axis=1)
    accepted = (argmax != 0) & (top >= thresholds[argmax])
    return accepted, argmax, top, thresholds


def calibrate_threshold(router, encoder, cal_df, tokenizer_args, device, target_recall,
                        progress_every: int = 0):
    """Legacy complete-p0 report for backward compatibility."""
    cal_complete = cal_df[cal_df["structure_type_label"] == 0].head(200)
    if len(cal_complete) == 0:
        return 0.9, {"note": "no complete calibration rows", "forced": [0, 0]}
    _labels, probability_matrix = _router_probs_for_df(
        router,
        encoder,
        cal_complete,
        tokenizer_args,
        device,
        progress_every=progress_every,
        scan_label="legacy complete-p0 calibration",
    )
    if probability_matrix.size == 0:
        return 0.9, {"note": "all calibration images unreadable", "forced": [0, 0]}
    probs = np.sort(probability_matrix[:, 0].astype(np.float64))
    keep = max(1, int(np.ceil(target_recall * len(probs))))
    threshold = float(min(max(probs[-keep], 0.5), 0.999))
    forced = int((probs >= threshold).sum())
    return threshold, {"complete_force_routed": [forced, int(len(probs))],
                       "min_complete_p0": float(probs[0]), "median_complete_p0": float(np.median(probs))}


def audit_sidecar_threshold(router, encoder, cal_df, tokenizer_args, device, sidecar_threshold,
                            progress_every: int = 0):
    """Measure complete leakage using the exact per-expert deployment rule."""
    cal_complete = cal_df[cal_df["structure_type_label"] == 0].head(512)
    if len(cal_complete) == 0:
        return {"note": "no complete calibration rows", "complete_rows": 0}
    _labels, probs = _router_probs_for_df(
        router,
        encoder,
        cal_complete,
        tokenizer_args,
        device,
        progress_every=progress_every,
        scan_label="sidecar threshold audit",
    )
    if probs.size == 0:
        return {"note": "all calibration images unreadable", "complete_rows": 0}
    accepted, argmax, _top, thresholds = _sidecar_acceptance_from_probs(
        probs,
        sidecar_threshold,
    )
    sidecar = argmax != 0
    values = probs[:, 1:].max(axis=1) if probs.shape[1] > 1 else np.zeros(len(probs))
    accepted_per_expert = {
        str(expert_idx): int((accepted & (argmax == expert_idx)).sum())
        for expert_idx in range(1, probs.shape[1])
    }
    n = int(len(probs))
    return {
        "complete_rows": int(n),
        "argmax_sidecar": int(sidecar.sum()),
        "accepted_sidecar": int(accepted.sum()),
        "accepted_sidecar_rate": float(accepted.sum() / max(n, 1)),
        "accepted_per_expert": accepted_per_expert,
        "max_sidecar_p99": float(np.quantile(values, 0.99)),
        "max_sidecar_p95": float(np.quantile(values, 0.95)),
        "max_sidecar_max": float(values.max()),
        "threshold": float(max(thresholds[1:], default=float("inf"))),
        "thresholds": [float(value) for value in thresholds],
    }


def _df_for_router_calibration(cal_df, max_per_label: int) -> "pd.DataFrame":
    if max_per_label is None or int(max_per_label) <= 0:
        return cal_df
    pieces = []
    for label in sorted(cal_df["structure_type_label"].dropna().unique()):
        pieces.append(cal_df[cal_df["structure_type_label"] == label].head(int(max_per_label)))
    if not pieces:
        return cal_df.head(0)
    return pd.concat(pieces, ignore_index=True)


def _router_probs_for_df(router, encoder, cal_df, tokenizer_args, device, max_per_label: int = 0,
                         progress_every: int = 0,
                         scan_label: str = "router threshold calibration"):
    """Return (labels, probs) for calibration rows readable by the MolNexTR transform."""
    from utils.MolNexTR.model import resize_small_image_to_long_edge
    _model_args, base = tokenizer_args
    df_eval = _df_for_router_calibration(cal_df, max_per_label)
    labels, probs = [], []
    bs = 8
    print(f"{scan_label}: scanning {len(df_eval)} rows", flush=True)
    router_was_training = bool(router.training)
    encoder_was_training = bool(encoder.training)
    router.eval()
    encoder.eval()
    try:
        for i in range(0, len(df_eval), bs):
            chunk = df_eval.iloc[i:i + bs]
            tensors, kept_labels = [], []
            for _, row in chunk.iterrows():
                import cv2
                img = cv2.imread(str(row["file_path"]))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = resize_small_image_to_long_edge(img, base.preprocess_long_edge)
                tensors.append(base.transform(image=img, keypoints=[])["image"])
                kept_labels.append(int(row["structure_type_label"]))
            if not tensors:
                continue
            stacked = torch.stack(tensors, 0).to(device)
            with torch.inference_mode():
                feats, _ = encoder(stacked)
                p = router(feats).softmax(-1).detach().cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(kept_labels)
            _progress(scan_label, min(i + bs, len(df_eval)),
                      len(df_eval), progress_every)
    finally:
        router.train(router_was_training)
        encoder.train(encoder_was_training)
    if not probs:
        return np.asarray([], dtype=np.int64), np.zeros((0, 3), dtype=np.float32)
    return np.asarray(labels, dtype=np.int64), np.asarray(probs, dtype=np.float32)


def calibrate_sidecar_thresholds(
    router,
    encoder,
    cal_df,
    tokenizer_args,
    device,
    *,
    requested_threshold: float,
    margin: float,
    max_per_label: int,
    progress_every: int = 0,
):
    """Select per-sidecar thresholds with zero observed complete->sidecar accepts."""
    labels, probs = _router_probs_for_df(
        router, encoder, cal_df, tokenizer_args, device, max_per_label=max_per_label,
        progress_every=progress_every,
    )
    if probs.size == 0:
        raise RuntimeError(
            "router threshold calibration has no readable rows; fix the calibration split "
            "or file paths instead of using uncalibrated sidecar thresholds"
        )
    argmax = probs.argmax(axis=1)
    top = probs.max(axis=1)
    num_experts = probs.shape[1]
    thresholds = [1.01] + [float(requested_threshold)] * max(0, num_experts - 1)
    complete = labels == 0
    complete_max_per_expert = {}
    for expert_idx in range(1, num_experts):
        # max prob a COMPLETE molecule assigns to this sidecar when the router's
        # argmax is this sidecar (the only way a complete molecule can leak).
        # threshold > this => zero observed complete->sidecar leakage.
        risky = complete & (argmax == expert_idx)
        max_complete = float(probs[risky, expert_idx].max()) if bool(risky.any()) else 0.0
        complete_max_per_expert[expert_idx] = max_complete
        # Data-driven: just above the calibration split's complete-max, so we
        # accept every sidecar-confident sample the router can reliably identify
        # (the requested floor only protects against an under-trained router).
        calibrated = max(
            float(requested_threshold),
            float(np.nextafter(max_complete, np.inf)) + float(margin),
        )
        thresholds[expert_idx] = 1.01 if calibrated > 1.0 else float(calibrated)

    accepted = (argmax != 0) & (top >= np.asarray(thresholds, dtype=np.float32)[argmax])
    per_expert = {}
    for expert_idx in range(num_experts):
        name = EXPERT_NAMES[expert_idx] if expert_idx < len(EXPERT_NAMES) else f"expert{expert_idx}"
        pred = argmax == expert_idx
        accepted_k = accepted & pred
        true_k = labels == expert_idx
        per_expert[name] = {
            "threshold": float(thresholds[expert_idx]),
            "calibration_rows_true_label": int(true_k.sum()),
            "argmax_rows": int(pred.sum()),
            "accepted_rows": int(accepted_k.sum()),
            "accepted_true_label_rows": int((accepted_k & true_k).sum()),
            "accepted_complete_rows": int((accepted_k & complete).sum()),
            "true_label_accept_rate": float((accepted_k & true_k).sum() / max(int(true_k.sum()), 1)),
            "argmax_precision": float((pred & true_k).sum() / max(int(pred.sum()), 1)),
        }
    report = {
        "schema_version": "moe_router_sidecar_threshold_calibration_v1",
        "policy": {
            "default_route": "expert0",
            "complete_to_sidecar_max_observed": 0,
            "thresholds_selected_on_calibration_split": True,
            "accept_rule": "argmax != expert0 and top_prob >= sidecar_confidence_thresholds[argmax]",
        },
        "rows": int(len(labels)),
        "label_counts": {str(k): int((labels == k).sum()) for k in range(num_experts)},
        "requested_threshold": float(requested_threshold),
        "margin": float(margin),
        "thresholds": [float(x) for x in thresholds],
        "complete_to_sidecar_max_per_expert": {
            int(k): float(v) for k, v in complete_max_per_expert.items()
        },
        "complete_accepted_sidecar_rows": int((accepted & complete).sum()),
        "sidecar_accept_rows": int(accepted.sum()),
        "per_expert": per_expert,
    }
    return thresholds, report


