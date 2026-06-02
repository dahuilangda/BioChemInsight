from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar


T = TypeVar("T")


class ModelHarnessError(Exception):
    """Base class for model harness failures."""

    error_type = "model_error"


class ModelTimeoutError(ModelHarnessError, TimeoutError):
    error_type = "timeout"


class ModelContractError(ModelHarnessError, ValueError):
    error_type = "contract_error"


class ModelProviderError(ModelHarnessError):
    error_type = "provider_error"


@dataclass
class ModelCallAudit:
    task_name: str
    channel: str
    ok: bool
    attempts: int
    elapsed_ms: int
    error_type: str = ""
    error: str = ""
    raw_response_preview: str = ""
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


def append_jsonl(path: str | Path | None, payload: Mapping[str, Any]) -> None:
    if not path:
        return
    audit_path = Path(path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, default=str) + "\n")


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


def require_json_object(response_text: str, task_name: str) -> dict[str, Any]:
    json_content = extract_json_content(response_text)
    if not json_content:
        raise ModelContractError(f"{task_name} returned no JSON object")
    try:
        payload = json.loads(json_content)
    except json.JSONDecodeError as exc:
        raise ModelContractError(f"{task_name} returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelContractError(f"{task_name} returned {type(payload).__name__}, expected object")
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

    Schema fields:
    - decision_required_keys: keys required for every decision.
    - decision_boolean_keys: keys that must be booleans.
    - decision_integer_keys: keys that must be integers.
    - decision_nullable_integer_keys: keys that must be integers or null.
    - decision_string_keys: keys that must be strings.
    - decision_nullable_string_keys: keys that must be strings or null.
    - decision_list_keys: keys that must be lists.
    - decision_integer_list_keys: keys that must be integer lists.
    - decision_string_list_keys: keys that must be string lists.
    - decision_non_empty_string_keys: keys that must be non-empty strings.
    - decision_non_empty_list_keys: keys that must be non-empty lists.
    - decision_allowed_values: mapping of key to allowed enum values.
    - <condition_key>_true_required_keys: keys required when condition_key is True.
    - <condition_key>_true_integer_keys: integer keys when True.
    - <condition_key>_true_list_keys: list keys when True.
    - <condition_key>_true_non_empty_string_keys: non-empty string keys when True.
    - <condition_key>_true_non_empty_list_keys: non-empty list keys when True.
    - <condition_key>_true_allowed_values: mapping of key to allowed enum values when True.
    - <condition_key>_false_required_keys: keys required when condition_key is False.
    - <condition_key>_false_integer_keys: integer keys when False.
    - <condition_key>_false_list_keys: list keys when False.
    - <condition_key>_false_non_empty_string_keys: non-empty string keys when False.
    - <condition_key>_false_non_empty_list_keys: non-empty list keys when False.
    - <condition_key>_false_allowed_values: mapping of key to allowed enum values when False.
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
    attempts = max(1, int(retry or 1))
    delays = list(retry_delays or [])
    started = time.monotonic()
    last_exc: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            response_text = operation()
            audit = ModelCallAudit(
                task_name=task_name,
                channel=channel,
                ok=True,
                attempts=attempt,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                raw_response_preview=preview_text(response_text, 2000),
                metadata=dict(metadata or {}),
            )
            append_jsonl(audit_path, asdict(audit))
            return response_text
        except Exception as exc:
            last_exc = exc
            if attempt < attempts:
                delay = delays[attempt - 1] if attempt - 1 < len(delays) else (1 + attempt - 1)
                time.sleep(max(0.0, float(delay)))
                continue

    error_type = classify_exception(last_exc)
    message = str(last_exc or "unknown model error")
    audit = ModelCallAudit(
        task_name=task_name,
        channel=channel,
        ok=False,
        attempts=attempts,
        elapsed_ms=int((time.monotonic() - started) * 1000),
        error_type=error_type,
        error=message,
        metadata=dict(metadata or {}),
    )
    append_jsonl(audit_path, asdict(audit))
    if error_type == "timeout":
        raise ModelTimeoutError(f"{task_name} timed out after {attempts} attempts: {message}") from last_exc
    if error_type == "contract_error":
        raise ModelContractError(f"{task_name} failed after {attempts} attempts: {message}") from last_exc
    raise ModelProviderError(f"{task_name} failed after {attempts} attempts: {message}") from last_exc


def raise_classified_error(task_name: str, attempts: int, exc: BaseException | None) -> None:
    error_type = classify_exception(exc)
    message = str(exc or "unknown model error")
    if error_type == "timeout":
        raise ModelTimeoutError(f"{task_name} timed out after {attempts} attempts: {message}") from exc
    if error_type == "contract_error":
        raise ModelContractError(f"{task_name} failed after {attempts} attempts: {message}") from exc
    raise ModelProviderError(f"{task_name} failed after {attempts} attempts: {message}") from exc


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
) -> T:
    """Run a model call and validate its JSON contract within the retry loop."""
    attempts = max(1, int(retry or 1))
    delays = list(retry_delays or [])
    started = time.monotonic()
    last_exc: BaseException | None = None
    last_response: str = ""

    for attempt in range(1, attempts + 1):
        try:
            response_text = operation()
            last_response = response_text
            parsed = parser(response_text)
            audit = ModelCallAudit(
                task_name=task_name,
                channel=channel,
                ok=True,
                attempts=attempt,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                raw_response_preview=preview_text(response_text, 2000),
                metadata=dict(metadata or {}),
            )
            append_jsonl(audit_path, asdict(audit))
            return parsed
        except Exception as exc:
            last_exc = exc
            if attempt < attempts:
                delay = delays[attempt - 1] if attempt - 1 < len(delays) else (1 + attempt - 1)
                time.sleep(max(0.0, float(delay)))
                continue

    audit = ModelCallAudit(
        task_name=task_name,
        channel=channel,
        ok=False,
        attempts=attempts,
        elapsed_ms=int((time.monotonic() - started) * 1000),
        error_type=classify_exception(last_exc),
        error=str(last_exc or "unknown model error"),
        raw_response_preview=preview_text(last_response, 2000),
        metadata=dict(metadata or {}),
    )
    append_jsonl(audit_path, asdict(audit))
    raise_classified_error(task_name, attempts, last_exc)
