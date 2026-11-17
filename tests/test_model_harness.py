"""Fault injection for model calls."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from utils import llm_utils as llm
from utils import model_harness as h


@pytest.fixture(autouse=True)
def isolated_circuit_state():
    h._TASK_CIRCUIT_STATE.clear()
    yield
    h._TASK_CIRCUIT_STATE.clear()


def _run_model_task(operation, **extra):
    return h.run_json_task(
        task_name="test",
        channel="text",
        operation=operation,
        parser=lambda text: h.require_json_object(text, "test"),
        retry=3,
        retry_delays=[0, 0],
        **extra,
    )


def test_retry_audit_records_attempt_data(tmp_path):
    responses = iter(["{bad", TimeoutError("transport"), '{"ok":true}'])

    def operation():
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        h._thread_local_usage.value = {"total_tokens": 5}
        return item

    path = tmp_path / "audit.jsonl"
    assert _run_model_task(operation, audit_path=path) == {"ok": True}
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["attempts"] for row in rows] == [1, 2, 3]
    assert rows[1]["raw_response_preview"] == ""
    assert rows[1]["usage"] == {}
    assert rows[1]["error_type"] == "timeout"


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
def test_http_permanent_errors_fail_once_with_actual_count(status):
    class HTTPError(Exception):
        status_code = status

    calls = []

    def operation():
        calls.append(1)
        raise HTTPError("rejected")

    with pytest.raises(h.ModelProviderError, match="after 1 attempts"):
        _run_model_task(operation)
    assert len(calls) == 1


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504])
def test_transient_http_errors_retry_within_budget(status):
    class HTTPError(Exception):
        status_code = status

    calls = []

    def operation():
        calls.append(1)
        if len(calls) < 3:
            raise HTTPError("temporary")
        return "{}"

    assert _run_model_task(operation) == {}
    assert len(calls) == 3


@pytest.mark.parametrize(
    "message",
    [
        "insufficient_quota",
        "invalid_api_key",
        "maximum context length",
        "model_not_found",
    ],
)
def test_deterministic_errors_do_not_repeat(message):
    calls = []

    def operation():
        calls.append(1)
        raise RuntimeError(message)

    with pytest.raises(h.ModelProviderError, match="after 1 attempts"):
        _run_model_task(operation)
    assert len(calls) == 1


def test_audit_write_failure_preserves_result(tmp_path):
    calls = []

    def operation():
        calls.append(1)
        return '{"ok":true}'

    assert _run_model_task(operation, audit_path=tmp_path) == {"ok": True}
    assert len(calls) == 1

    def failure():
        raise h.ModelDownstreamVerificationError("nested verifier exhausted")

    with pytest.raises(
        h.ModelDownstreamVerificationError, match="nested verifier exhausted"
    ):
        _run_model_task(failure, audit_path=tmp_path)


def test_concurrent_audit_records_are_complete(tmp_path):
    path = tmp_path / "parallel.jsonl"
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(
            pool.map(
                lambda i: h.append_jsonl(path, {"i": i, "text": "α" * 3000}), range(512)
            )
        )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert sorted(row["i"] for row in rows) == list(range(512))
    assert all(row["text"] == "α" * 3000 for row in rows)


def _trip_circuit(monkeypatch):
    monkeypatch.setattr(h, "TASK_CIRCUIT_WINDOW", 4)
    for _ in range(4):
        h._circuit_record("test", "timeout")
    assert h._TASK_CIRCUIT_STATE["test"]["opened_at"] is not None


def test_open_circuit_fast_fails_and_is_audited(monkeypatch, tmp_path):
    _trip_circuit(monkeypatch)
    path = tmp_path / "audit.jsonl"

    def forbidden():
        pytest.fail("open circuit made a provider call")

    with pytest.raises(h.ModelCircuitOpen):
        _run_model_task(forbidden, audit_path=path)
    row = json.loads(path.read_text())
    assert row["error_type"] == "circuit_open" and row["attempts"] == 0


@pytest.mark.parametrize(
    "outcome", ["ok", "contract_error", "timeout", "provider_error"]
)
def test_half_open_allows_one_probe_and_recovery(monkeypatch, outcome):
    _trip_circuit(monkeypatch)
    monkeypatch.setattr(h, "TASK_CIRCUIT_COOLDOWN_SECONDS", 0)
    barrier = threading.Barrier(32)

    def gate(_):
        barrier.wait(timeout=5)
        try:
            return h._circuit_gate("test")
        except h.ModelCircuitOpen:
            return False

    with ThreadPoolExecutor(max_workers=32) as pool:
        admitted = list(pool.map(gate, range(32)))
    assert admitted.count(True) == 1
    # Late successes must not close an active recovery probe.
    h._circuit_record("test", "ok")
    with pytest.raises(h.ModelCircuitOpen):
        h._circuit_gate("test")
    h._circuit_record("test", outcome, is_probe=True)
    state = h._TASK_CIRCUIT_STATE["test"]
    assert state["probe_in_flight"] is False
    assert (state["opened_at"] is None) == (outcome in {"ok", "contract_error"})


def test_cancelled_probe_releases_recovery_state(monkeypatch):
    _trip_circuit(monkeypatch)
    monkeypatch.setattr(h, "TASK_CIRCUIT_COOLDOWN_SECONDS", 0)

    def cancelled():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        _run_model_task(cancelled)
    assert not h._TASK_CIRCUIT_STATE["test"]["probe_in_flight"]


def test_breakers_are_provider_isolated(monkeypatch):
    monkeypatch.setattr(h, "TASK_CIRCUIT_WINDOW", 1)

    def outage():
        raise RuntimeError("offline")

    with pytest.raises(h.ModelProviderError):
        _run_model_task(outage, metadata={"model": "a", "url": "https://a"})
    assert (
        _run_model_task(lambda: "{}", metadata={"model": "b", "url": "https://b"}) == {}
    )
    with pytest.raises(h.ModelCircuitOpen):
        _run_model_task(lambda: "{}", metadata={"model": "a", "url": "https://a"})


def test_retry_entry_points_share_policy():
    calls = []

    def operation():
        calls.append(1)
        raise RuntimeError("invalid_api_key")

    with pytest.raises(h.ModelProviderError):
        h.run_with_retries(
            task_name="test", channel="text", operation=operation, retry=5
        )
    assert len(calls) == 1


def test_timeout_retains_worker_slot():
    slot = threading.BoundedSemaphore(1)
    started = threading.Event()
    release = threading.Event()
    exited = threading.Event()

    def hung():
        started.set()
        try:
            release.wait(timeout=5)
        finally:
            exited.set()
        return "{}"

    try:
        with pytest.raises(TimeoutError, match="call exceeded"):
            llm._run_bounded_model_call(
                hung, semaphore=slot, timeout_seconds=0.02, label="test"
            )
        assert started.is_set()
        for _ in range(20):
            with pytest.raises(TimeoutError, match="slot exceeded"):
                llm._run_bounded_model_call(
                    lambda: pytest.fail("extra worker started"),
                    semaphore=slot,
                    timeout_seconds=0.001,
                    label="test",
                )
    finally:
        release.set()
        assert exited.wait(timeout=5)
    assert slot.acquire(timeout=1)
    slot.release()
    assert (
        llm._run_bounded_model_call(
            lambda: "recovered", semaphore=slot, timeout_seconds=1, label="test"
        )
        == "recovered"
    )


def test_bounded_call_usage_is_isolated_under_concurrency():
    slot = threading.BoundedSemaphore(8)

    def invoke(i):
        def operation():
            h._thread_local_usage.value = {"total_tokens": i}
            return i

        assert (
            llm._run_bounded_model_call(
                operation, semaphore=slot, timeout_seconds=5, label="test"
            )
            == i
        )
        return h._thread_local_usage.value

    with ThreadPoolExecutor(max_workers=16) as pool:
        values = list(pool.map(invoke, range(256)))
    assert values == [{"total_tokens": i} for i in range(256)]


def test_text_sdk_retries_disabled_clients_closed_and_usage_captured(monkeypatch):
    calls, closed = [], []

    def client_factory(**kwargs):
        calls.append(kwargs)
        response = SimpleNamespace(
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 7}),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))],
        )
        return SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: response)
            ),
            close=lambda: closed.append(1),
        )

    monkeypatch.setattr(llm, "require_openai", lambda: client_factory)
    monkeypatch.setattr(llm, "LLM_TEXT_MODEL_NAME", "fake")
    monkeypatch.setattr(llm, "LLM_TEXT_MODEL_URL", "https://fake")
    monkeypatch.setattr(llm, "LLM_TEXT_MODEL_KEY", "fake")
    assert llm.run_text_json_task(
        task_name="test", prompt="test", parser=json.loads, retry=3
    ) == {"ok": True}
    assert len(calls) == len(closed) == 1
    assert calls[0]["max_retries"] == 0
    assert h._thread_local_usage.value == {"total_tokens": 7}


@pytest.mark.parametrize(
    "raw",
    [
        '{"a":1,"a":2}',
        '{"a":{"keep":false,"keep":true}}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
    ],
)
def test_ambiguous_or_nonstandard_json_is_rejected(raw):
    with pytest.raises(h.ModelContractError):
        h.require_json_object(raw, "test")


def test_text_contract_retry_receives_bounded_validation_feedback(monkeypatch):
    import copy

    messages, closed = [], []
    responses = iter(['{"a":1,"a":2}', '{"a":2}'])

    def create(**kwargs):
        messages.append(copy.deepcopy(kwargs["messages"]))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))],
            usage=None,
        )

    def factory(**kwargs):
        assert kwargs["max_retries"] == 0
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=lambda: closed.append(1),
        )

    monkeypatch.setattr(llm, "require_openai", lambda: factory)
    monkeypatch.setattr(llm, "LLM_TEXT_MODEL_NAME", "fake")
    monkeypatch.setattr(llm, "LLM_TEXT_MODEL_URL", "https://fake")
    monkeypatch.setattr(llm, "LLM_TEXT_MODEL_KEY", "fake")
    monkeypatch.setattr(llm, "get_retry_delays", lambda *a, **kw: [0])
    assert llm.run_text_json_task(
        task_name="test",
        prompt="extract",
        parser=lambda raw: h.require_json_object(raw, "test"),
        retry=2,
    ) == {"a": 2}
    assert len(messages[0]) == 2 and len(messages[1]) == 4
    assert "duplicate JSON key" in messages[1][-1]["content"]
    assert len(closed) == 2


def test_client_closes_on_provider_failure(monkeypatch):
    closed = []

    def create(**kwargs):
        raise RuntimeError("invalid_api_key")

    def factory(**kwargs):
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=lambda: closed.append(1),
        )

    monkeypatch.setattr(llm, "require_openai", lambda: factory)
    monkeypatch.setattr(llm, "LLM_TEXT_MODEL_NAME", "fake")
    monkeypatch.setattr(llm, "LLM_TEXT_MODEL_URL", "https://fake")
    monkeypatch.setattr(llm, "LLM_TEXT_MODEL_KEY", "fake")
    with pytest.raises(h.ModelProviderError):
        llm.run_text_json_task(
            task_name="test", prompt="test", parser=json.loads, retry=3
        )
    assert len(closed) == 1


def test_complete_responses_are_replayable_with_hash_and_call_id(tmp_path):
    import hashlib

    path = tmp_path / "audit.jsonl"
    raw = json.dumps({"evidence": "α" * 3000}, ensure_ascii=False)
    assert _run_model_task(lambda: raw, audit_path=path)["evidence"] == "α" * 3000
    row = json.loads(path.read_text())
    saved = (path.parent / row["response_artifact"]).read_text()
    assert saved == raw
    assert row["response_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert not row["response_truncated"]
    assert row["call_id"] and len(row["raw_response_preview"]) < len(saved)
    assert ((path.parent / row["response_artifact"]).stat().st_mode & 0o777) == 0o600


def test_response_artifact_size_limit_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setattr(h, "MODEL_AUDIT_MAX_RESPONSE_BYTES", 100)
    path = tmp_path / "audit.jsonl"
    raw = json.dumps({"evidence": "α" * 300}, ensure_ascii=False)
    _run_model_task(lambda: raw, audit_path=path)
    row = json.loads(path.read_text())
    assert row["response_truncated"] and row["response_bytes"] == len(raw.encode())
    assert (path.parent / row["response_artifact"]).stat().st_size == 100


def test_response_artifact_failure_preserves_success_and_audit(tmp_path):
    (tmp_path / "model_responses").write_text("blocked directory")
    path = tmp_path / "audit.jsonl"
    assert _run_model_task(lambda: "{}", audit_path=path) == {}
    row = json.loads(path.read_text())
    assert row["ok"] and not row["response_artifact"] and row["response_sha256"]


def test_nested_circuit_error_releases_probe(monkeypatch):
    _trip_circuit(monkeypatch)
    monkeypatch.setattr(h, "TASK_CIRCUIT_COOLDOWN_SECONDS", 0)

    def nested_failure():
        raise h.ModelCircuitOpen("nested", 120)

    with pytest.raises(h.ModelCircuitOpen):
        _run_model_task(nested_failure)
    assert not h._TASK_CIRCUIT_STATE["test"]["probe_in_flight"]
    assert h._TASK_CIRCUIT_STATE["test"]["opened_at"] is not None


def test_parent_usage_is_not_overwritten_by_nested_parser_call(tmp_path):
    path = tmp_path / "audit.jsonl"

    def operation():
        h._thread_local_usage.value = {"total_tokens": 100}
        return "{}"

    def parser(raw):
        def nested():
            h._thread_local_usage.value = {"total_tokens": 5}
            return "{}"

        _run_model_task(nested, audit_path=path)
        return json.loads(raw)

    h.run_json_task(
        task_name="parent",
        channel="text",
        operation=operation,
        parser=parser,
        retry=1,
        audit_path=path,
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["usage"] == {"total_tokens": 5}
    assert rows[1]["usage"] == {"total_tokens": 100}


def test_evidence_failure_is_terminal_and_does_not_trip_provider_breaker(tmp_path):
    calls = []

    def operation():
        calls.append(1)
        raise h.ModelEvidenceError("original results table missing")

    with pytest.raises(h.ModelEvidenceError):
        _run_model_task(operation, audit_path=tmp_path / "audit.jsonl")
    assert len(calls) == 1
    row = json.loads((tmp_path / "audit.jsonl").read_text())
    assert row["error_type"] == "evidence_error" and not row["ok"]
    assert h._TASK_CIRCUIT_STATE["test"]["opened_at"] is None


def test_schema_rejection_stops_retries(
    monkeypatch,
):
    from utils.assay_reconciliation import AssayReconciliationRequest

    schema = AssayReconciliationRequest.from_pages(
        ["endpoint"], {1: "Measured compound-level result."}
    ).build_response_schema()
    captured = []

    class RejectedSchema(Exception):
        status_code = 400

    def create(**kwargs):
        captured.append(kwargs)
        raise RejectedSchema("strict schema unsupported")

    def factory(**kwargs):
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=lambda: None,
        )

    monkeypatch.setattr(llm, "require_openai", lambda: factory)
    monkeypatch.setattr(llm, "LLM_TEXT_MODEL_NAME", "fake")
    monkeypatch.setattr(llm, "LLM_TEXT_MODEL_URL", "https://fake")
    monkeypatch.setattr(llm, "LLM_TEXT_MODEL_KEY", "fake")
    with pytest.raises(h.ModelProviderError, match="strict schema unsupported"):
        llm.run_text_json_task(
            task_name="test",
            prompt="adjudicate",
            parser=json.loads,
            retry=3,
            response_schema=schema,
        )
    assert len(captured) == 1
    response_format = captured[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == schema


def test_structured_response_content_reaches_parser_unchanged(monkeypatch):
    raw = '{"evidence":"<think>literal source text</think>"}'

    def factory(**kwargs):
        response = SimpleNamespace(
            usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content=raw))]
        )
        return SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: response)
            ),
            close=lambda: None,
        )

    monkeypatch.setattr(llm, "require_openai", lambda: factory)
    monkeypatch.setattr(llm, "LLM_TEXT_MODEL_NAME", "fake")
    monkeypatch.setattr(llm, "LLM_TEXT_MODEL_URL", "https://fake")
    monkeypatch.setattr(llm, "LLM_TEXT_MODEL_KEY", "fake")
    assert (
        llm.run_text_json_task(
            task_name="test",
            prompt="test",
            parser=lambda text: text,
            retry=1,
            response_schema={"type": "object"},
        )
        == raw
    )


def test_contract_errors_retry_independently_of_provider_error_words():
    attempts = []

    def operation():
        attempts.append(1)
        if len(attempts) == 1:
            raise h.ModelContractError(
                "model_not_found field is invalid; source does not exist"
            )
        return "{}"

    assert _run_model_task(operation) == {}
    assert len(attempts) == 2


def test_call_metadata_cannot_override_provider_identity(monkeypatch, tmp_path):
    response = SimpleNamespace(
        usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
    )

    def factory(**kwargs):
        return SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: response)
            ),
            close=lambda: None,
        )

    monkeypatch.setattr(llm, "require_openai", lambda: factory)
    monkeypatch.setattr(llm, "LLM_TEXT_MODEL_NAME", "actual-model")
    monkeypatch.setattr(llm, "LLM_TEXT_MODEL_URL", "https://actual-provider")
    monkeypatch.setattr(llm, "LLM_TEXT_MODEL_KEY", "unused")
    path = tmp_path / "model_calls.jsonl"
    llm.run_text_json_task(
        task_name="test",
        prompt="extract",
        parser=json.loads,
        retry=1,
        audit_path=path,
        metadata={
            "model": "wrong-model",
            "url": "https://wrong-provider",
            "prompt_sha256": "wrong",
        },
    )
    metadata = json.loads(path.read_text())["metadata"]
    assert metadata["model"] == "actual-model"
    assert metadata["url"] == "https://actual-provider"
    assert metadata["prompt_sha256"] != "wrong"


def test_visual_response_content_reaches_parser_unchanged(monkeypatch, tmp_path):
    raw = '{"evidence":"<think>literal source text</think>"}'
    requests, closed = [], []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content=raw))]
        )

    def factory(**kwargs):
        assert kwargs["max_retries"] == 0
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=lambda: closed.append(1),
        )

    monkeypatch.setattr(llm, "require_openai", lambda: factory)
    monkeypatch.setattr(llm, "VISUAL_MODEL_NAME", "visual-model")
    monkeypatch.setattr(llm, "VISUAL_MODEL_URL", "https://visual-provider")
    monkeypatch.setattr(llm, "VISUAL_MODEL_KEY", "unused")
    image = tmp_path / "source.png"
    image.write_bytes(b"image fixture")
    result = llm.run_vision_json_task(
        task_name="test",
        image_file=str(image),
        prompt="read",
        parser=lambda text: h.require_json_object(text, "test", allow_wrapped=False),
        retry=1,
        preserve_response=True,
    )
    assert result["evidence"] == "<think>literal source text</think>"
    assert len(requests) == len(closed) == 1
    assert requests[0]["model"] == "visual-model"


@pytest.mark.parametrize(
    "base_url,expected_proxy",
    [
        ("http://model.test/v1", "http://http-proxy.test"),
        ("https://model.test/v1", "http://https-proxy.test"),
    ],
)
def test_model_transport_uses_explicit_scheme_proxy(
    monkeypatch, base_url, expected_proxy
):
    import os

    import httpx

    clients, options = [], []
    monkeypatch.setenv("HTTP_PROXY", "http://ambient.invalid")
    before = dict(os.environ)
    monkeypatch.setattr(llm, "HTTP_PROXY", "http://http-proxy.test")
    monkeypatch.setattr(llm, "HTTPS_PROXY", "http://https-proxy.test")
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: options.append(kwargs) or object()
    )
    monkeypatch.setattr(
        llm, "OpenAI", lambda **kwargs: clients.append(kwargs) or object()
    )
    llm.require_openai()(base_url=base_url)
    assert options == [{"proxy": expected_proxy, "trust_env": False, "timeout": None}]
    assert clients[0]["http_client"] is not None
    assert dict(os.environ) == before


def test_empty_proxy_configuration_ignores_ambient_proxy(monkeypatch):
    import httpx

    options = []
    monkeypatch.setenv("HTTPS_PROXY", "http://ambient.invalid")
    monkeypatch.setattr(llm, "HTTP_PROXY", "")
    monkeypatch.setattr(llm, "HTTPS_PROXY", "")
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: options.append(kwargs) or object()
    )
    monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: object())
    llm.require_openai()(base_url="https://model.test/v1")
    assert options[0]["proxy"] is None and options[0]["trust_env"] is False


def test_failed_model_client_initialization_closes_transport(monkeypatch):
    import httpx

    closed = []
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: SimpleNamespace(close=lambda: closed.append(1)),
    )

    def fail(**kwargs):
        raise ValueError("invalid client configuration")

    monkeypatch.setattr(llm, "OpenAI", fail)
    with pytest.raises(ValueError):
        llm.require_openai()(base_url="https://model.test/v1")
    assert closed == [1]


def test_expired_callers_cannot_open_provider_circuit(monkeypatch):
    monkeypatch.setattr(h.time, "monotonic", lambda: 100.0)
    for _ in range(16):
        with pytest.raises(h.ModelTimeoutError):
            _run_model_task(lambda: pytest.fail("expired operation ran"), deadline=99)
    assert not h._TASK_CIRCUIT_STATE
    assert _run_model_task(lambda: '{"ok":true}', deadline=101) == {"ok": True}


def test_expired_half_open_caller_does_not_admit_or_reopen_probe(monkeypatch):
    _trip_circuit(monkeypatch)
    before = dict(h._TASK_CIRCUIT_STATE["test"])
    with pytest.raises(h.ModelTimeoutError):
        _run_model_task(lambda: pytest.fail("expired probe ran"), deadline=-1)
    assert h._TASK_CIRCUIT_STATE["test"] == before


def test_retry_delay_cannot_overrun_shared_deadline(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(h.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        h.time, "sleep", lambda delay: pytest.fail("delay exceeded budget")
    )
    calls = []

    def fail():
        calls.append(1)
        clock[0] += 0.5
        raise h.ModelContractError("invalid response")

    with pytest.raises(h.ModelContractError):
        h.run_json_task(
            task_name="test",
            channel="text",
            operation=fail,
            parser=lambda x: x,
            retry=3,
            retry_delays=[1],
            deadline=11,
        )
    assert len(calls) == 1


def test_contract_repair_shares_sdk_and_outer_deadline(monkeypatch):
    clock = [2.0]
    sdk_options, outer_limits, messages = [], [], []
    responses = iter(['{"wrong":true}', '{"ok":true}'])
    monkeypatch.setattr(h.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(h.time, "sleep", lambda delay: None)
    monkeypatch.setattr(llm, "get_retry_delays", lambda *args, **kwargs: [0])

    def bounded(operation, **kwargs):
        outer_limits.append(kwargs["timeout_seconds"])
        return operation()

    monkeypatch.setattr(llm, "_run_bounded_model_call", bounded)

    def create(**kwargs):
        messages.append(kwargs["messages"])
        clock[0] += 3
        return SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))],
        )

    def factory(**kwargs):
        sdk_options.append(kwargs)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=lambda: None,
        )

    monkeypatch.setattr(llm, "require_openai", lambda: factory)

    def parse(raw):
        result = h.require_json_object(raw, "deadline-test", allow_wrapped=False)
        if set(result) != {"ok"}:
            raise h.ModelContractError("Missing ok field")
        return result

    assert llm.run_text_json_task(
        task_name="deadline-test",
        prompt="read",
        parser=parse,
        retry=2,
        timeout_seconds=90,
        deadline=10,
        response_schema={"type": "object"},
    ) == {"ok": True}
    assert outer_limits == [8, 5]
    assert [item["timeout"] for item in sdk_options] == [8, 5]
    assert all(item["max_retries"] == 0 for item in sdk_options)
    assert len(messages[1]) == 4 and "Missing ok field" in messages[1][-1]["content"]


def test_unused_half_open_probe_is_released_on_deadline_race(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(h.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(h, "TASK_CIRCUIT_COOLDOWN_SECONDS", 1)
    h._TASK_CIRCUIT_STATE["test"] = {
        "outcomes": [],
        "opened_at": 0,
        "probe_in_flight": False,
    }
    original_gate = h._circuit_gate

    def gate(name):
        probe = original_gate(name)
        clock[0] = 12
        return probe

    monkeypatch.setattr(h, "_circuit_gate", gate)
    with pytest.raises(h.ModelTimeoutError):
        _run_model_task(lambda: pytest.fail("expired probe ran"), deadline=11)
    assert h._TASK_CIRCUIT_STATE["test"] == {
        "outcomes": [],
        "opened_at": 0,
        "probe_in_flight": False,
    }


def test_nested_verification_failure_preserves_details_without_retry():
    calls = []
    original = h.ModelDownstreamVerificationError(
        "unresolved", details={"issues": ["conflicting source values"]}
    )

    def fail():
        calls.append(1)
        raise original

    with pytest.raises(h.ModelDownstreamVerificationError) as caught:
        _run_model_task(lambda: _run_model_task(fail))
    assert caught.value is original and caught.value.details == original.details
    assert calls == [1]


def test_local_queue_saturation_cannot_poison_provider_health():
    calls = []
    semaphore = SimpleNamespace(acquire=lambda **kwargs: False)

    def operation():
        return llm._run_bounded_model_call(
            lambda: calls.append(1),
            semaphore=semaphore,
            timeout_seconds=1,
            label="test",
        )

    for _ in range(16):
        with pytest.raises(h.ModelCapacityError):
            _run_model_task(operation)
    assert not calls and not h._TASK_CIRCUIT_STATE
    assert _run_model_task(lambda: '{"ok":true}') == {"ok": True}


def test_saturated_recovery_probe_remains_available(monkeypatch):
    _trip_circuit(monkeypatch)
    state = h._TASK_CIRCUIT_STATE["test"]
    state["opened_at"] -= h.TASK_CIRCUIT_COOLDOWN_SECONDS + 1
    opened_at = state["opened_at"]

    def saturated():
        raise h.ModelCapacityError("local slots occupied")

    with pytest.raises(h.ModelCapacityError):
        _run_model_task(saturated)
    assert state["opened_at"] == opened_at and not state["probe_in_flight"]
    assert _run_model_task(lambda: '{"ok":true}') == {"ok": True}


@pytest.mark.parametrize("channel", ["text", "vision"])
def test_truncated_generation_is_terminal_and_retained(monkeypatch, tmp_path, channel):
    calls = []
    raw = '{"ok":true}'

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            usage=SimpleNamespace(model_dump=lambda: {"completion_tokens": 64}),
            choices=[
                SimpleNamespace(
                    finish_reason="length", message=SimpleNamespace(content=raw)
                )
            ],
        )

    def factory(**kwargs):
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=lambda: None,
        )

    monkeypatch.setattr(llm, "require_openai", lambda: factory)
    audit = tmp_path / "calls.jsonl"
    arguments = {
        "task_name": "output-limit-test",
        "prompt": "read",
        "retry": 3,
        "audit_path": audit,
        "parser": lambda raw: pytest.fail("Truncated response reached parser"),
    }
    with pytest.raises(h.ModelOutputLimitError):
        if channel == "text":
            llm.run_text_json_task(
                **arguments, max_tokens=64, response_schema={"type": "object"}
            )
        else:
            image = tmp_path / "source.png"
            image.write_bytes(b"image")
            llm.run_vision_json_task(
                **arguments, image_file=image, preserve_response=True
            )
    assert len(calls) == 1
    if channel == "text":
        assert calls[0]["max_tokens"] == 64
    record = json.loads(audit.read_text())
    assert (
        record["error_type"] == "output_limit"
        and record["usage"]["completion_tokens"] == 64
    )
    assert (tmp_path / record["response_artifact"]).read_text() == raw


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "64"])
def test_invalid_token_limits_fail_before_provider(limit):
    with pytest.raises(ValueError):
        llm.run_text_json_task(
            task_name="invalid-limit",
            prompt="read",
            parser=lambda raw: raw,
            max_tokens=limit,
        )


def test_expired_queue_admission_releases_slot_without_request(monkeypatch):
    clock = [0.0]
    released, calls = [], []
    monkeypatch.setattr(llm.time, "monotonic", lambda: clock[0])

    def acquire(**kwargs):
        clock[0] = 2
        return True

    semaphore = SimpleNamespace(acquire=acquire, release=lambda: released.append(1))
    with pytest.raises(h.ModelCapacityError):
        llm._run_bounded_model_call(
            lambda: calls.append(1),
            semaphore=semaphore,
            timeout_seconds=1,
            label="test",
        )
    assert not calls and released == [1]


def test_completed_but_late_worker_is_rejected(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(llm.time, "monotonic", lambda: clock[0])
    semaphore = threading.BoundedSemaphore(1)

    def operation():
        clock[0] = 2
        return "late"

    with pytest.raises(h.ModelTimeoutError):
        llm._run_bounded_model_call(
            operation, semaphore=semaphore, timeout_seconds=1, label="test"
        )
    assert semaphore.acquire(blocking=False)
    semaphore.release()


def test_request_factory_cannot_weaken_structured_output(monkeypatch):
    monkeypatch.setattr(
        llm, "require_openai", lambda: pytest.fail("Invalid request reached provider")
    )
    with pytest.raises(h.ModelContractError, match="remove required structured output"):
        llm.run_text_json_task(
            task_name="strict-factory",
            prompt="read",
            parser=lambda raw: raw,
            response_schema={"type": "object"},
            request_factory=lambda: ("read", None),
            retry=1,
        )


def test_audit_delay_cannot_return_after_deadline_or_penalize_provider(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(h.time, "monotonic", lambda: clock[0])
    events = []
    monkeypatch.setattr(h, "_circuit_record", lambda key, result, **kwargs: events.append(result))

    def delayed_audit(*args):
        clock[0] = 2.0

    monkeypatch.setattr(h, "append_jsonl", delayed_audit)
    with pytest.raises(h.ModelTimeoutError, match="during completion"):
        _run_model_task(lambda: '{"ok":true}', deadline=1.0)
    assert events == ["ok"]
