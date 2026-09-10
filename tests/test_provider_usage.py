from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import agentic_translation.providers_llm as providers_llm
from agentic_translation.models import ProviderCallRecord
from agentic_translation.providers_llm import _OpenAIJSONProvider, inspect_response_cache


class _FakeCompletions:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.response


class _FakeClient:
    def __init__(self, completions: _FakeCompletions, **kwargs: object) -> None:
        self.chat = SimpleNamespace(completions=completions)
        self.kwargs = kwargs


def _json_response(*, usage: object | None) -> object:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
        usage=usage,
    )


def _native_response(*, usage: object | None) -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(
                                name="finish",
                                arguments='{"summary":"done"}',
                            ),
                        )
                    ]
                )
            )
        ],
        usage=usage,
    )


def _factory(response: object, captured: dict[str, object]):
    completions = _FakeCompletions(response)

    def client_factory(**kwargs: object) -> _FakeClient:
        captured.update(kwargs)
        return _FakeClient(completions, **kwargs)

    return client_factory, completions


def test_provider_call_record_usage_fields_are_optional_and_nonnegative() -> None:
    record = ProviderCallRecord(
        role="judge",
        namespace="judge",
        provider="openai",
        payload_sha256="a" * 64,
        response_sha256="b" * 64,
        cache_file="judge.json",
    )

    assert record.input_tokens is None
    assert record.output_tokens is None
    assert record.cached_input_tokens is None
    assert record.total_tokens is None
    assert record.elapsed_ms is None
    assert record.recorded_elapsed_ms is None

    invalid = record.model_dump()
    invalid["input_tokens"] = -1
    with pytest.raises(ValidationError):
        ProviderCallRecord.model_validate(invalid)


def test_openai_json_usage_is_recorded_in_a_separate_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret-api-key")
    usage = SimpleNamespace(
        prompt_tokens=11,
        completion_tokens=7,
        total_tokens=18,
        prompt_tokens_details=SimpleNamespace(cached_tokens=3),
    )
    captured: dict[str, object] = {}
    client_factory, completions = _factory(_json_response(usage=usage), captured)
    provider = _OpenAIJSONProvider(
        provider_mode="live",
        cache_dir=tmp_path,
        record_cache=True,
        model_name="gpt-test",
        client_factory=client_factory,
        max_output_tokens=99,
        temperature=0.2,
        request_timeout_seconds=12.5,
    )

    assert provider._call_json(namespace="judge", payload={"source": "x"}, messages=[]) == {"ok": True}

    record = provider.call_records[0]
    assert (record.input_tokens, record.output_tokens, record.cached_input_tokens, record.total_tokens) == (
        11,
        7,
        3,
        18,
    )
    assert record.cache_hit is False
    assert record.elapsed_ms is not None and record.elapsed_ms >= 0
    assert captured["api_key"] == "secret-api-key"
    assert captured["timeout"] == 12.5
    assert completions.calls[0]["max_tokens"] == 99
    assert completions.calls[0]["temperature"] == 0.2

    response_file = next(tmp_path.glob("judge_*.json"))
    response = json.loads(response_file.read_text(encoding="utf-8"))
    assert response == {"ok": True}
    receipt_file = next(tmp_path.glob("usage_judge_*.json"))
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    assert receipt["input_tokens"] == 11
    assert receipt["elapsed_ms"] == record.elapsed_ms
    assert "secret-api-key" not in receipt_file.read_text(encoding="utf-8")
    assert inspect_response_cache(tmp_path).integrity_passed is True


def test_deepseek_json_usage_supports_cache_hit_field_and_omits_none_temperature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    usage = {
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "prompt_cache_hit_tokens": 8,
        "total_tokens": 25,
    }
    captured: dict[str, object] = {}
    client_factory, completions = _factory(_json_response(usage=usage), captured)
    provider = _OpenAIJSONProvider(
        provider_mode="live",
        cache_dir=tmp_path,
        record_cache=True,
        provider_name="deepseek",
        client_factory=client_factory,
        max_output_tokens=None,
        temperature=None,
        request_timeout_seconds=9,
    )

    provider._call_json(namespace="translation", payload={"source": "天"}, messages=[])

    record = provider.call_records[0]
    assert record.provider == "deepseek"
    assert (record.input_tokens, record.output_tokens, record.cached_input_tokens, record.total_tokens) == (
        20,
        5,
        8,
        25,
    )
    assert captured == {
        "api_key": "deepseek-key",
        "base_url": "https://api.deepseek.com",
        "timeout": 9,
    }
    assert "temperature" not in completions.calls[0]
    assert "max_tokens" not in completions.calls[0]


def test_openai_native_usage_is_recorded_without_changing_action_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    usage = {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
    captured: dict[str, object] = {}
    client_factory, completions = _factory(_native_response(usage=usage), captured)
    provider = _OpenAIJSONProvider(
        provider_mode="live",
        cache_dir=tmp_path,
        record_cache=True,
        model_name="gpt-native",
        client_factory=client_factory,
    )

    result = provider._call_chat_tool(
        namespace="agent_action",
        payload={"task": "finish"},
        messages=[],
        tools=[{"type": "function", "function": {"name": "finish", "parameters": {}}}],
        normalize_call=lambda call: call.as_action_payload(),
    )

    assert result == {"tool": "finish", "summary": "done"}
    record = provider.call_records[0]
    assert (record.input_tokens, record.output_tokens, record.total_tokens) == (4, 2, 6)
    assert completions.calls[0]["tool_choice"] == "required"
    assert completions.calls[0]["temperature"] == 0
    assert "response_format" not in completions.calls[0]
    assert "timeout" not in captured


def test_deepseek_native_usage_is_recorded_and_legacy_missing_usage_is_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    native_usage = SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=3,
        total_tokens=15,
        prompt_tokens_details=SimpleNamespace(cached_tokens=5),
    )
    captured: dict[str, object] = {}
    client_factory, _ = _factory(_native_response(usage=native_usage), captured)
    provider = _OpenAIJSONProvider(
        provider_mode="live",
        cache_dir=tmp_path,
        record_cache=True,
        provider_name="deepseek",
        model_name="deepseek-chat",
        client_factory=client_factory,
    )
    provider._call_chat_tool(
        namespace="agent_action",
        payload={"task": "finish"},
        messages=[],
        tools=[{"type": "function", "function": {"name": "finish", "parameters": {}}}],
    )

    record = provider.call_records[0]
    assert (record.input_tokens, record.output_tokens, record.cached_input_tokens, record.total_tokens) == (
        12,
        3,
        5,
        15,
    )
    assert captured["base_url"] == "https://api.deepseek.com"

    old_cache = tmp_path / "old-cache"
    old_provider = _OpenAIJSONProvider(
        provider_mode="live",
        cache_dir=old_cache,
        record_cache=True,
        model_name="deepseek-chat",
        provider_name="deepseek",
        client_factory=lambda **kwargs: _factory(_json_response(usage=None), {})[0](**kwargs),
    )
    old_provider._call_json(namespace="legacy", payload={"x": 1}, messages=[])
    old_record = old_provider.call_records[0]
    assert old_record.input_tokens is None
    assert old_record.output_tokens is None
    assert old_record.cached_input_tokens is None
    assert old_record.total_tokens is None


def test_replay_preserves_recorded_usage_but_measures_current_latency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    clock = iter([100.0, 100.25, 200.0, 200.005])
    monkeypatch.setattr(providers_llm.time, "perf_counter", lambda: next(clock))
    usage = {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16}
    live_factory, _ = _factory(_json_response(usage=usage), {})
    live = _OpenAIJSONProvider(
        provider_mode="live",
        cache_dir=tmp_path,
        record_cache=True,
        model_name="test-model",
        client_factory=live_factory,
    )
    payload = {"a": 1}
    live._call_json(namespace="test", payload=payload, messages=[])

    replay = _OpenAIJSONProvider(
        provider_mode="replay",
        cache_dir=tmp_path,
        model_name="test-model",
        client_factory=lambda **kwargs: (_ for _ in ()).throw(AssertionError("network used during replay")),
    )
    replay._call_json(namespace="test", payload=payload, messages=[])

    live_record = live.call_records[0]
    replay_record = replay.call_records[0]
    assert live_record.elapsed_ms == pytest.approx(250.0)
    assert replay_record.elapsed_ms == pytest.approx(5.0)
    assert replay_record.recorded_elapsed_ms == pytest.approx(250.0)
    assert replay_record.cache_hit is True
    assert (replay_record.input_tokens, replay_record.output_tokens, replay_record.total_tokens) == (10, 6, 16)
    assert replay_record.payload_sha256 == live_record.payload_sha256
    assert replay_record.response_sha256 == live_record.response_sha256
