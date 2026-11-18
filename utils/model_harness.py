from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ModelHarnessError(Exception):
    """Base class for model harness failures."""

    error_type = "model_error"

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = details


class ModelTimeoutError(ModelHarnessError, TimeoutError):
    error_type = "timeout"


class ModelCapacityError(ModelHarnessError, TimeoutError):
    error_type = 'capacity_error'


class ModelContractError(ModelHarnessError, ValueError):
    error_type = "contract_error"


class ModelOutputLimitError(ModelHarnessError):
    error_type = 'output_limit'

    def __init__(self, message, response_text=''):
        super().__init__(message)
        self.response_text = response_text


class ModelEvidenceError(ModelHarnessError):
    """Source evidence is insufficient; identical-input retries cannot repair it."""

    error_type = "evidence_error"


class ModelSourceAmbiguityError(ModelEvidenceError):
    error_type = "source_ambiguity"


class ModelDownstreamVerificationError(ModelHarnessError):
    """Extraction succeeded but a nested verifier failed its own retry budget.

    The nested verifier already retried internally; the outer retry loop
    treats this as terminal (re-running would only multiply cost).
    """

    error_type = "contract_error"


class ModelProviderError(ModelHarnessError):
    error_type = "provider_error"


_thread_local_usage = threading.local()
_thread_local_timing = threading.local()


def _captured_usage(channel):
    """Return usage captured by the current caller thread."""
    return getattr(_thread_local_usage, "value", None) or {}


@dataclass
class ModelCallAudit:
    task_name: str
    channel: str
    ok: bool
    attempts: int
    elapsed_ms: int
    ts: str = ""
    call_id: str = ""
    response_sha256: str = ""
    response_artifact: str = ""
    response_bytes: int = 0
    response_truncated: bool = False
    prompt_fingerprint: str = ""
    error_type: str = ""
    error: str = ""
    raw_response_preview: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def preview_text(value: Any, limit: int = 8192) -> str:
    text = "" if value is None else str(value)
    return text[:limit] + ("..." if len(text) > limit else "")


def is_timeout_exception(exc: BaseException | None) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    text = str(exc or "").lower()
    return "timed out" in text or "timeout" in text


def classify_exception(exc: BaseException | None) -> str:
    if isinstance(exc, ModelHarnessError):
        return exc.error_type
    if is_timeout_exception(exc):
        return "timeout"
    if isinstance(exc, (json.JSONDecodeError, ValueError, TypeError)):
        return "contract_error"
    return "provider_error"


_RUN_JSON_AUDIT_LOCK = threading.Lock()

TASK_CIRCUIT_BREAKER_ENABLED = True
TASK_CIRCUIT_WINDOW = 8
TASK_CIRCUIT_FAILURE_RATIO = 0.5
TASK_CIRCUIT_COOLDOWN_SECONDS = 120.0
_TASK_CIRCUIT_STATE: dict[str, dict[str, Any]] = {}
_TASK_CIRCUIT_LOCK = threading.Lock()


class ModelCircuitOpen(ModelHarnessError, RuntimeError):
    """Raised when a task's circuit is open; no replacement result is returned."""

    error_type = "circuit_open"

    def __init__(self, task_name: str, remaining_seconds: float):
        super().__init__(
            f"circuit open for {task_name} ({remaining_seconds:.0f}s cooldown left)"
        )
        self.task_name = task_name
        self.remaining_seconds = remaining_seconds


def _circuit_gate(task_name: str) -> bool:
    """Admit exactly one recovery probe after cooldown, even under concurrency."""
    if not TASK_CIRCUIT_BREAKER_ENABLED:
        return False
    now = time.monotonic()
    with _TASK_CIRCUIT_LOCK:
        state = _TASK_CIRCUIT_STATE.get(task_name)
        if not state or state.get("opened_at") is None:
            return False
        remaining = TASK_CIRCUIT_COOLDOWN_SECONDS - (now - state["opened_at"])
        if remaining > 0 or state.get("probe_in_flight"):
            raise ModelCircuitOpen(task_name, max(0.0, remaining))
        state["probe_in_flight"] = True
        return True


def _circuit_record(task_name: str, error_type: str, *, is_probe: bool = False) -> None:
    """Update the failure window or complete a recovery probe."""
    if not TASK_CIRCUIT_BREAKER_ENABLED or (
        error_type == "circuit_open" and not is_probe
    ):
        return
    if error_type == 'capacity_error':
        if is_probe:
            _release_circuit_probe(task_name)
        return
    with _TASK_CIRCUIT_LOCK:
        state = _TASK_CIRCUIT_STATE.setdefault(
            task_name, {"outcomes": [], "opened_at": None, "probe_in_flight": False}
        )
        failure = error_type in {"timeout", "provider_error", "circuit_open"}
        if is_probe and state.get("probe_in_flight"):
            state["probe_in_flight"] = False
            state["outcomes"] = []
            state["opened_at"] = time.monotonic() if failure else None
            return
        # Calls admitted before another call opened the circuit cannot close it.
        if state.get("opened_at") is not None:
            return
        outcomes = state.setdefault("outcomes", [])
        outcomes.append(error_type)
        del outcomes[:-TASK_CIRCUIT_WINDOW]
        if len(outcomes) >= TASK_CIRCUIT_WINDOW:
            failures = sum(item in {"timeout", "provider_error"} for item in outcomes)
            if failures / len(outcomes) > TASK_CIRCUIT_FAILURE_RATIO:
                state["opened_at"] = time.monotonic()
                outcomes.clear()


def _release_circuit_probe(task_name: str) -> None:
    with _TASK_CIRCUIT_LOCK:
        state = _TASK_CIRCUIT_STATE.get(task_name)
        if state is not None:
            state['probe_in_flight'] = False


_NON_RETRYABLE_ERROR_PATTERN = None


def _non_retryable_pattern():
    """Error text patterns for calls that identical-input retries cannot fix:
    quota/billing exhaustion and context-window overflow are deterministic."""
    global _NON_RETRYABLE_ERROR_PATTERN
    if _NON_RETRYABLE_ERROR_PATTERN is None:
        import re
        _NON_RETRYABLE_ERROR_PATTERN = re.compile(
            r"insufficient_quota|quota exceeded|billing|usage limit reached"
            r"|ContextWindowExceeded|maximum context length|context.?length"
            r"|invalid_api_key|unauthorized|forbidden|permission denied"
            r"|model_not_found|does not exist",
            re.IGNORECASE,
        )
    return _NON_RETRYABLE_ERROR_PATTERN


def is_retryable_exception(exc: BaseException | None) -> bool:
    if exc is None:
        return True
    if isinstance(
        exc, (ModelDownstreamVerificationError, ModelCircuitOpen, ModelEvidenceError, ModelCapacityError, ModelOutputLimitError)
    ):
        return False
    if isinstance(exc, ModelContractError):
        return True
    # SDK errors expose HTTP status independently of their wording.
    status = getattr(exc, "status_code", None)
    if (
        isinstance(status, int)
        and 400 <= status < 500
        and status not in {408, 409, 429}
    ):
        return False
    return _non_retryable_pattern().search(str(exc)) is None


def _retry_after_seconds(exc: Exception) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {})
    value = headers.get("Retry-After")
    if not value:
        return 0.0
    try:
        delay = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return 0.0
    return max(0.0, delay) if math.isfinite(delay) else 0.0


def append_jsonl(path: str | Path | None, payload: Mapping[str, Any]) -> None:
    if not path:
        return
    try:
        line = json.dumps(dict(payload), ensure_ascii=False, default=str) + "\n"
        with _RUN_JSON_AUDIT_LOCK:
            audit_path = Path(path)
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Cannot write model audit to %s: %s", path, exc)


MODEL_AUDIT_MAX_RESPONSE_BYTES = 1024 * 1024


def _save_response_artifact(audit_path, call_id, attempt, response_text):
    if not audit_path or not response_text:
        return {}
    raw = str(response_text).encode("utf-8", errors="backslashreplace")
    info = {
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "response_bytes": len(raw),
        "response_truncated": len(raw) > MODEL_AUDIT_MAX_RESPONSE_BYTES,
    }
    relative = Path("model_responses") / f"{call_id}-{attempt}.txt"
    path = Path(audit_path).parent / relative
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(
            os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
        ) as handle:
            handle.write(raw[:MODEL_AUDIT_MAX_RESPONSE_BYTES])
        info["response_artifact"] = relative.as_posix()
    except OSError as exc:
        logger.warning("Cannot save model response artifact: %s", exc)
    return info


def extract_json_content(text: str) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()
    if "```json" in text:
        json_match = text.split("```json", 1)
        if len(json_match) > 1:
            return json_match[1].split("```", 1)[0].strip()
    if "```" in text and text.count("```") >= 2:
        parts = text.split("```", 2)
        if len(parts) >= 2:
            return parts[1].strip()
    if "“json" in text:
        json_match = text.split("“json", 1)
        if len(json_match) > 1:
            temp_content = json_match[1].strip()
            if temp_content.endswith("”"):
                temp_content = temp_content[:-1].strip()
            return temp_content

    start_brace = text.find("{")
    end_brace = text.rfind("}")
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        return text[start_brace:end_brace + 1].strip()
    return None


def require_json_object(
    response_text: str, task_name: str, *, allow_wrapped: bool = True
) -> dict[str, Any]:
    json_content = (
        extract_json_content(response_text) if allow_wrapped else response_text
    )
    if not isinstance(json_content, str) or not json_content.strip():
        raise ModelContractError(f"{task_name} returned no JSON object")

    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"duplicate JSON key: {key!r}")
            obj[key] = value
        return obj

    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        payload = json.loads(
            json_content,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except ValueError as exc:
        raise ModelContractError(f"{task_name} returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelContractError(
            f"{task_name} returned {type(payload).__name__}, expected object"
        )
    return payload


def require_required_keys(
    payload: Mapping[str, Any],
    schema: Mapping[str, Any] | None,
    task_name: str,
    context: str = "payload",
) -> None:
    required_keys = schema.get("required_keys", []) if isinstance(schema, Mapping) else []
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ModelContractError(f"{task_name} {context} missing required keys: {', '.join(missing)}")


def _schema_list(schema: Mapping[str, Any], key: str) -> list[Any]:
    value = schema.get(key)
    return value if isinstance(value, list) else []


def _schema_mapping(schema: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = schema.get(key)
    return value if isinstance(value, Mapping) else {}


def _require_boolean_keys(payload: Mapping[str, Any], keys: list[Any], task_name: str, context: str) -> None:
    for key in keys:
        if not isinstance(payload.get(key), bool):
            raise ModelContractError(f"{task_name} {context} has non-boolean {key}")


def _require_integer_keys(payload: Mapping[str, Any], keys: list[Any], task_name: str, context: str) -> None:
    for key in keys:
        if not isinstance(payload.get(key), int) or isinstance(payload.get(key), bool):
            raise ModelContractError(f"{task_name} {context} has non-integer {key}")


def _require_nullable_integer_keys(payload: Mapping[str, Any], keys: list[Any], task_name: str, context: str) -> None:
    for key in keys:
        value = payload.get(key)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
            raise ModelContractError(f"{task_name} {context} has non-integer-or-null {key}")


def _require_list_keys(payload: Mapping[str, Any], keys: list[Any], task_name: str, context: str) -> None:
    for key in keys:
        if not isinstance(payload.get(key), list):
            raise ModelContractError(f"{task_name} {context} has non-list {key}")


def _require_integer_list_keys(payload: Mapping[str, Any], keys: list[Any], task_name: str, context: str) -> None:
    for key in keys:
        value = payload.get(key)
        if not isinstance(value, list):
            raise ModelContractError(f"{task_name} {context} has non-list {key}")
        invalid = [item for item in value if not isinstance(item, int) or isinstance(item, bool)]
        if invalid:
            raise ModelContractError(f"{task_name} {context} has non-integer item in {key}")


def _require_string_list_keys(payload: Mapping[str, Any], keys: list[Any], task_name: str, context: str) -> None:
    for key in keys:
        value = payload.get(key)
        if not isinstance(value, list):
            raise ModelContractError(f"{task_name} {context} has non-list {key}")
        invalid = [item for item in value if not isinstance(item, str)]
        if invalid:
            raise ModelContractError(f"{task_name} {context} has non-string item in {key}")


def _require_non_empty_string_list_keys(
    payload: Mapping[str, Any],
    keys: list[Any],
    task_name: str,
    context: str,
) -> None:
    for key in keys:
        value = payload.get(key)
        if not isinstance(value, list):
            raise ModelContractError(f"{task_name} {context} has non-list {key}")
        invalid = [item for item in value if not isinstance(item, str) or not item.strip()]
        if invalid:
            raise ModelContractError(f"{task_name} {context} has empty or non-string item in {key}")


def _require_string_keys(payload: Mapping[str, Any], keys: list[Any], task_name: str, context: str) -> None:
    for key in keys:
        if not isinstance(payload.get(key), str):
            raise ModelContractError(f"{task_name} {context} has non-string {key}")


def _require_nullable_string_keys(payload: Mapping[str, Any], keys: list[Any], task_name: str, context: str) -> None:
    for key in keys:
        value = payload.get(key)
        if value is not None and not isinstance(value, str):
            raise ModelContractError(f"{task_name} {context} has non-string-or-null {key}")


def _require_non_empty_string_keys(payload: Mapping[str, Any], keys: list[Any], task_name: str, context: str) -> None:
    for key in keys:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ModelContractError(f"{task_name} {context} has empty {key}")


def _require_non_empty_list_keys(payload: Mapping[str, Any], keys: list[Any], task_name: str, context: str) -> None:
    for key in keys:
        if not isinstance(payload.get(key), list) or not payload.get(key):
            raise ModelContractError(f"{task_name} {context} has empty or non-list {key}")


def _require_allowed_values(
    payload: Mapping[str, Any],
    allowed_by_key: Mapping[str, Any],
    task_name: str,
    context: str,
) -> None:
    for key, allowed in allowed_by_key.items():
        if isinstance(allowed, (list, tuple, set)):
            require_enum_value(payload.get(key), allowed, task_name, key=f"{context}.{key}")


def _require_decision_schema_metadata(
    decision: Mapping[str, Any],
    schema: Mapping[str, Any],
    task_name: str,
    context: str,
    prefix: str = "decision",
) -> None:
    require_required_keys(
        decision,
        {"required_keys": _schema_list(schema, f"{prefix}_required_keys")},
        task_name,
        context=context,
    )
    _require_boolean_keys(decision, _schema_list(schema, f"{prefix}_boolean_keys"), task_name, context)
    _require_integer_keys(decision, _schema_list(schema, f"{prefix}_integer_keys"), task_name, context)
    _require_nullable_integer_keys(
        decision,
        _schema_list(schema, f"{prefix}_nullable_integer_keys"),
        task_name,
        context,
    )
    _require_string_keys(decision, _schema_list(schema, f"{prefix}_string_keys"), task_name, context)
    _require_nullable_string_keys(
        decision,
        _schema_list(schema, f"{prefix}_nullable_string_keys"),
        task_name,
        context,
    )
    _require_list_keys(decision, _schema_list(schema, f"{prefix}_list_keys"), task_name, context)
    _require_integer_list_keys(decision, _schema_list(schema, f"{prefix}_integer_list_keys"), task_name, context)
    _require_string_list_keys(decision, _schema_list(schema, f"{prefix}_string_list_keys"), task_name, context)
    _require_non_empty_string_list_keys(
        decision,
        _schema_list(schema, f"{prefix}_non_empty_string_list_keys"),
        task_name,
        context,
    )
    _require_non_empty_string_keys(
        decision,
        _schema_list(schema, f"{prefix}_non_empty_string_keys"),
        task_name,
        context,
    )
    _require_non_empty_list_keys(
        decision,
        _schema_list(schema, f"{prefix}_non_empty_list_keys"),
        task_name,
        context,
    )
    _require_allowed_values(
        decision,
        _schema_mapping(schema, f"{prefix}_allowed_values"),
        task_name,
        context,
    )


def require_decision_contract(
    decision: Mapping[str, Any],
    schema: Mapping[str, Any] | None,
    task_name: str,
    *,
    decision_key: str = "",
    condition_key: str = "keep",
) -> None:
    """Validate a decision object against common decision schema metadata.

    Keys: decision_<type>_keys / decision_non_empty_<type>_keys (nullable
    variants included), decision_allowed_values, and <condition_key>_{true,false}_*.
    """
    if not isinstance(decision, Mapping):
        label = f" decision {decision_key!r}" if decision_key else " decision"
        raise ModelContractError(f"{task_name}{label} must be an object")

    label = f"decision {decision_key!r}" if decision_key else "decision"
    schema = schema or {}
    require_object_contract(decision, schema, task_name, object_key=decision_key, prefix="decision")

    condition_value = decision.get(condition_key)
    if condition_value is True:
        suffix = "true"
    elif condition_value is False:
        suffix = "false"
    else:
        return

    _require_decision_schema_metadata(
        decision,
        schema,
        task_name,
        f"{label} when {condition_key}={suffix}",
        prefix=f"{condition_key}_{suffix}",
    )


def require_object_contract(
    payload: Mapping[str, Any],
    schema: Mapping[str, Any] | None,
    task_name: str,
    *,
    object_key: str = "",
    prefix: str = "decision",
    label: str | None = None,
) -> None:
    if not isinstance(payload, Mapping):
        object_label = label or prefix
        keyed = f" {object_label} {object_key!r}" if object_key else f" {object_label}"
        raise ModelContractError(f"{task_name}{keyed} must be an object")

    object_label = label or prefix
    context = f"{object_label} {object_key!r}" if object_key else object_label
    _require_decision_schema_metadata(payload, schema or {}, task_name, context, prefix=prefix)


def require_enum_value(value: Any, allowed: set[str] | list[str] | tuple[str, ...], task_name: str, key: str) -> str:
    normalized = str(value or "").strip().lower()
    allowed_set = {str(item).strip().lower() for item in allowed}
    if normalized not in allowed_set:
        raise ModelContractError(f"{task_name} payload has invalid {key}: {value!r}")
    return normalized


def require_confidence_value(value: Any, task_name: str, key: str = "confidence") -> str:
    return require_enum_value(value, {"high", "medium", "low"}, task_name, key)


def parse_validated_json_object(response_text: str, schema: Mapping[str, Any] | None, task_name: str) -> dict[str, Any]:
    payload = require_json_object(response_text, task_name)
    require_required_keys(payload, schema or {}, task_name)
    return payload


def run_with_retries(
    *,
    task_name: str,
    channel: str,
    operation: Callable[[], str],
    retry: int,
    retry_delays: list[float] | tuple[float, ...] | None = None,
    audit_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    return run_json_task(
        task_name=task_name,
        channel=channel,
        operation=operation,
        parser=lambda response: response,
        retry=retry,
        retry_delays=retry_delays,
        audit_path=audit_path,
        metadata=metadata,
    )


def raise_classified_error(
    task_name: str, attempts: int, exc: BaseException | None
) -> None:
    if isinstance(exc, (ModelCircuitOpen, ModelEvidenceError, ModelDownstreamVerificationError, ModelCapacityError, ModelOutputLimitError)):
        raise exc
    error_type = classify_exception(exc)
    message = str(exc or "unknown model error")
    if error_type == "timeout":
        raise ModelTimeoutError(
            f"{task_name} timed out after {attempts} attempts: {message}"
        ) from exc
    if error_type == "contract_error":
        raise ModelContractError(
            f"{task_name} failed after {attempts} attempts: {message}"
        ) from exc
    raise ModelProviderError(
        f"{task_name} failed after {attempts} attempts: {message}"
    ) from exc


def run_json_task(
    *,
    task_name: str,
    channel: str,
    operation: Callable[[], str],
    parser: Callable[[str], T],
    retry: int,
    retry_delays: list[float] | tuple[float, ...] | None = None,
    audit_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
    on_retry: Callable[[BaseException, str, int], None] | None = None,
    deadline: float | None = None,
    repair_contract: bool = False,
) -> T:
    """Run and validate a request within one retry budget."""
    attempts = max(1, int(retry or 1))
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError("deadline must be a finite monotonic timestamp")
    delays = list(retry_delays or [])
    started = time.monotonic()
    call_id = uuid.uuid4().hex
    last_exc: BaseException | None = None
    prompt_fingerprint = str((metadata or {}).get("prompt_sha256") or "")
    if not prompt_fingerprint:
        prompt_hint = str((metadata or {}).get("prompt_chars") or "")
        if prompt_hint:
            prompt_fingerprint = (
                "len:" + hashlib.sha256(prompt_hint.encode()).hexdigest()[:12]
            )

    def _write_audit(ok, attempt_number, response_text, exc=None, usage=None, timing=None):
        audit = ModelCallAudit(
            task_name=task_name,
            channel=channel,
            ok=ok,
            attempts=attempt_number,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            call_id=call_id,
            **_save_response_artifact(
                audit_path, call_id, attempt_number, response_text
            ),
            prompt_fingerprint=prompt_fingerprint,
            error_type="" if exc is None else classify_exception(exc),
            error="" if exc is None else str(exc),
            raw_response_preview=preview_text(response_text, 2000),
            usage=dict(usage or {}),
            timing=dict(timing or {}),
            metadata=dict(metadata or {}),
        )
        append_jsonl(audit_path, asdict(audit))

    if deadline is not None and time.monotonic() >= deadline:
        error = ModelTimeoutError(f"{task_name} exceeded its deadline before execution")
        _write_audit(False, 0, '', exc=error)
        raise error

    provider = tuple(str((metadata or {}).get(key) or "") for key in ("url", "model"))
    circuit_key = task_name
    if any(provider):
        identity = json.dumps((channel, *provider), ensure_ascii=False)
        circuit_key += ":" + hashlib.sha256(identity.encode()).hexdigest()[:12]
    try:
        is_probe = bool(_circuit_gate(circuit_key))
    except ModelCircuitOpen as exc:
        _write_audit(False, 0, "", exc=exc)
        raise

    attempt = 0
    try:
        allow_repair = repair_contract and attempts == 1
        for next_attempt in range(1, attempts + 1 + int(allow_repair)):
            if deadline is not None and time.monotonic() >= deadline:
                last_exc = ModelTimeoutError(f"{task_name} exceeded its deadline")
                _write_audit(False, attempt, "", exc=last_exc)
                break
            attempt = next_attempt
            response_text = ""
            operation_usage = {}
            operation_timing = None
            _thread_local_usage.value = {}
            _thread_local_timing.value = {}
            try:
                response_text = operation()
                # Capture usage before nested verifier calls.
                operation_usage = dict(_captured_usage(channel))
                operation_timing = _thread_local_timing.value.copy()
                parsed = parser(response_text)
                if deadline is not None and time.monotonic() >= deadline:
                    raise ModelTimeoutError(f"{task_name} exceeded its deadline")
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, ModelOutputLimitError):
                    response_text = exc.response_text
                    operation_usage = dict(_captured_usage(channel))
                if operation_timing is None:
                    operation_timing = _thread_local_timing.value
                last_exc = exc
                _write_audit(
                    False, attempt, response_text, exc=exc, usage=operation_usage,
                    timing=operation_timing,
                )
                can_repair = allow_repair and attempt == 1 and bool(response_text) and isinstance(exc, ModelContractError)
                if not is_retryable_exception(exc) or (attempt >= attempts and not can_repair):
                    break
                delay = delays[attempt - 1] if attempt - 1 < len(delays) else attempt
                delay = max(0.0, float(delay), _retry_after_seconds(exc))
                remaining = deadline - time.monotonic() if deadline is not None else 60.0
                if delay >= remaining:
                    break
                if on_retry is not None:
                    on_retry(exc, response_text, attempt)
                time.sleep(delay)
            else:
                _write_audit(True, attempt, response_text, usage=operation_usage, timing=operation_timing)
                _circuit_record(circuit_key, "ok", is_probe=is_probe)
                last_exc = None
                break
    except BaseException:
        # Release half-open probes on cancellation.
        _circuit_record(circuit_key, "provider_error", is_probe=is_probe)
        raise

    if last_exc is None:
        if deadline is not None and time.monotonic() >= deadline:
            raise ModelTimeoutError(f"{task_name} exceeded its deadline during completion")
        return parsed
    if attempt:
        _circuit_record(circuit_key, classify_exception(last_exc), is_probe=is_probe)
    elif is_probe:
        _release_circuit_probe(circuit_key)
    raise_classified_error(task_name, attempt, last_exc)
