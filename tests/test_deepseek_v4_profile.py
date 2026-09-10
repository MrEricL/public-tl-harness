from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_translation.agent_provider import (
    SHOWCASE_TOOL_SCHEMA_VERSION,
    AgentActionRequest,
    LLMAgentActionProvider,
)
from agentic_translation.agent_tools import (
    SHOWCASE_TOOL_REGISTRY,
    ToolCall,
    ToolCallValidationError,
)
from agentic_translation.providers_llm import ResponseCache, _OpenAIJSONProvider
from agentic_translation.showcase_providers import (
    make_action_provider,
    make_translation_provider,
    resolve_profile,
)


DISABLED_THINKING = {"thinking": {"type": "disabled"}}


class _RecordingCompletions:
    def __init__(self, *, native: bool = False) -> None:
        self.native = native
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.native:
            message = SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        id="call_1",
                        function=SimpleNamespace(
                            name="finish",
                            arguments='{"summary":"Complete."}',
                        ),
                    )
                ]
            )
        else:
            message = SimpleNamespace(content='{"ok":true}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _Client:
    def __init__(self, completions: _RecordingCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def _provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    completions: _RecordingCompletions,
    *,
    extra_body: dict[str, object] | None = None,
) -> _OpenAIJSONProvider:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    return _OpenAIJSONProvider(
        provider_mode="live",
        cache_dir=tmp_path,
        record_cache=True,
        provider_name="deepseek",
        model_name="deepseek-v4-flash",
        client_factory=lambda **kwargs: _Client(completions),
        max_retries=0,
        max_output_tokens=2048,
        temperature=0.0,
        extra_body=extra_body,
    )


@pytest.mark.parametrize(
    "model", ["deepseek-v4-flash", "deepseek-v4-pro", "DeepSeek-V4-preview"]
)
def test_explicit_deepseek_v4_profiles_disable_thinking(model: str) -> None:
    profile = resolve_profile("deepseek", model)

    assert profile["model_source"] == "explicit"
    assert profile["extra_body"] == DISABLED_THINKING
    assert profile["tool_protocol"] == "json_prompt"
    assert profile["temperature"] == 0.0
    assert profile["max_output_tokens"] == 2048


def test_legacy_and_non_deepseek_profiles_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    environment_profile = resolve_profile("deepseek")
    legacy_profile = resolve_profile("deepseek", "deepseek-chat")
    openai_profile = resolve_profile("openai", "deepseek-v4-flash")
    assert "extra_body" not in environment_profile
    assert "extra_body" not in legacy_profile
    assert "extra_body" not in openai_profile
    assert environment_profile["tool_protocol"] == "native_function"
    assert legacy_profile["tool_protocol"] == "native_function"
    assert openai_profile["tool_protocol"] == "native_function"


def test_showcase_factories_copy_and_propagate_v4_extra_body(tmp_path: Path) -> None:
    profile = resolve_profile("deepseek", "deepseek-v4-flash")
    action = make_action_provider(
        mode="replay",
        profile=profile,
        cache_dir=tmp_path / "action",
    )
    translation = make_translation_provider(
        mode="replay",
        profile=profile,
        cache_dir=tmp_path / "translation",
    )

    profile["extra_body"]["thinking"]["type"] = "enabled"
    assert action.extra_body == DISABLED_THINKING
    assert translation.extra_body == DISABLED_THINKING


def test_json_action_adapter_accepts_prompt_contract_argument_shape() -> None:
    nested = ToolCall.from_json_action(
        {
            "tool": "read_paragraphs",
            "arguments": {"document": "source", "start": 0, "count": 3},
        }
    )
    flattened = ToolCall.from_json_action(
        {"tool": "read_paragraphs", "document": "source", "start": 0, "count": 3}
    )

    assert nested == flattened
    assert SHOWCASE_TOOL_REGISTRY.action_from_call(
        nested, visible_names={"read_paragraphs"}
    ).tool == "read_paragraphs"


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"tool": "read_paragraphs", "arguments": []}, "arguments must be an object"),
        (
            {
                "tool": "read_paragraphs",
                "arguments": {"document": "source", "start": 0, "count": 3},
                "count": 3,
            },
            "must not mix",
        ),
    ],
)
def test_json_action_adapter_rejects_invalid_nested_arguments(
    payload: dict, message: str
) -> None:
    with pytest.raises(ToolCallValidationError, match=message):
        ToolCall.from_json_action(payload)


def test_json_request_sends_extra_body_and_binds_it_to_cache_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_completions = _RecordingCompletions()
    first = _provider(
        tmp_path,
        monkeypatch,
        first_completions,
        extra_body=DISABLED_THINKING,
    )
    payload = {"task": "translate"}
    messages = [{"role": "user", "content": "{}"}]

    assert first._call_json(namespace="translation", payload=payload, messages=messages) == {
        "ok": True
    }
    assert first_completions.calls[0]["extra_body"] == DISABLED_THINKING
    assert first_completions.calls[0]["temperature"] == 0.0
    assert first_completions.calls[0]["max_tokens"] == 2048

    second_completions = _RecordingCompletions()
    second = _provider(
        tmp_path,
        monkeypatch,
        second_completions,
        extra_body={"thinking": {"type": "enabled"}},
    )
    second._call_json(namespace="translation", payload=payload, messages=messages)

    report = ResponseCache(tmp_path).inspect()
    assert len(report.entries) == 2
    assert len({entry.payload_sha256 for entry in report.entries}) == 2
    assert second_completions.calls[0]["extra_body"] == {
        "thinking": {"type": "enabled"}
    }


def test_native_action_request_and_replay_use_v4_cache_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    completions = _RecordingCompletions(native=True)
    request = AgentActionRequest(
        episode_id="deepseek-v4-test",
        step_number=1,
        story_slug="showcase",
        chapter="0001",
        remaining_steps=1,
        remaining_patch_attempts=0,
        tool_schema_version=SHOWCASE_TOOL_SCHEMA_VERSION,
        instruction_context={"role": "coordinator"},
    )
    kwargs = {
        "provider_name": "deepseek",
        "model_name": "deepseek-v4-pro",
        "tool_protocol": "native_function",
        "registry": SHOWCASE_TOOL_REGISTRY,
        "max_output_tokens": 2048,
        "temperature": 0.0,
        "extra_body": DISABLED_THINKING,
    }
    live = LLMAgentActionProvider(
        provider_mode="live",
        cache_dir=tmp_path,
        record_cache=True,
        client_factory=lambda **client_kwargs: _Client(completions),
        max_retries=0,
        **kwargs,
    )

    assert live.next_action(request).tool == "finish"
    sent = completions.calls[0]
    assert sent["extra_body"] == DISABLED_THINKING
    assert sent["tool_choice"] == "required"
    assert sent["parallel_tool_calls"] is False
    raw_payload = request.canonical_payload(registry=SHOWCASE_TOOL_REGISTRY)
    raw_payload["tool_protocol"] = "native_function"
    assert live.call_records[0].payload_sha256 != ResponseCache(
        tmp_path
    )._payload_digest(raw_payload)

    replay = LLMAgentActionProvider(
        provider_mode="replay",
        cache_dir=tmp_path,
        client_factory=lambda **client_kwargs: (_ for _ in ()).throw(
            AssertionError("replay must not create a client")
        ),
        **kwargs,
    )
    assert replay.next_action(request).tool == "finish"
    assert replay.call_records[0].cache_hit is True
    assert replay.call_records[0].payload_sha256 == live.call_records[0].payload_sha256


def test_absent_extra_body_preserves_legacy_payload_hash(tmp_path: Path) -> None:
    provider = _OpenAIJSONProvider(cache_dir=tmp_path)
    payload = {"a": 1}
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    assert provider._call_payload(payload) is payload
    assert provider.cache._payload_digest(provider._call_payload(payload)) == expected
    assert expected == (
        "f9d86028c6e0d64e225186f96acb69338b2c59764df79162107f5c4bb34d1310"
    )


@pytest.mark.parametrize(
    "extra_body, message",
    [
        (["not", "an", "object"], "JSON object"),
        ({"authorization": "Bearer secret"}, "credential field"),
        ({"headers": {"x-api-key": "secret"}}, "credential field"),
        ({"thinking": float("nan")}, "finite JSON-compatible"),
        ({"thinking": ("disabled",)}, "JSON-compatible"),
    ],
)
def test_extra_body_rejects_unsafe_or_non_json_values(
    tmp_path: Path, extra_body: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _OpenAIJSONProvider(cache_dir=tmp_path, extra_body=extra_body)  # type: ignore[arg-type]
