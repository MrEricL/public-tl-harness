from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_translation.agent_provider import AgentActionRequest
from agentic_translation.providers_llm import LLMProviderUnavailable
from agentic_translation.structured_transport import (
    ModelSpec,
    StructuredActionProvider,
    invoke_structured,
)
import agentic_translation.structured_transport as transport


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "score": {"type": "integer", "minimum": 0, "maximum": 4},
    },
    "required": ["verdict", "score"],
    "additionalProperties": False,
}


def _codex_spec() -> ModelSpec:
    return ModelSpec("codex", "gpt-5.6-terra", effort="high")


def _install_codex_result(monkeypatch: pytest.MonkeyPatch, result: dict, events: str = "") -> None:
    def fake_run(command, *, prompt, cwd, timeout_seconds):
        last_path = Path(command[command.index("--output-last-message") + 1])
        last_path.write_text(json.dumps(result), encoding="utf-8")
        return events, "", 0

    monkeypatch.setattr(transport, "_run_process", fake_run)


def _assert_openai_strict_schema(value) -> None:
    if isinstance(value, list):
        for child in value:
            _assert_openai_strict_schema(child)
        return
    if not isinstance(value, dict):
        return
    assert "oneOf" not in value
    assert "default" not in value
    assert "discriminator" not in value
    properties = value.get("properties")
    if value.get("type") == "object" or isinstance(properties, dict):
        assert value.get("additionalProperties") is False
        assert set(value.get("required", ())) == set(properties or {})
    for child in value.values():
        _assert_openai_strict_schema(child)


def test_generic_codex_schema_and_pinned_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_codex_result(
        monkeypatch,
        {"verdict": "pass", "score": 4},
        '{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":3}}\n',
    )

    receipt = invoke_structured(
        _codex_spec(), "Judge this output.", JUDGE_SCHEMA, tmp_path / "call"
    )

    assert receipt["status"] == "completed"
    assert receipt["output"] == {"verdict": "pass", "score": 4}
    assert receipt["usage"]["total_tokens"] == 15
    config = json.loads((tmp_path / "call" / "request_config.json").read_text())
    command = config["command"]
    assert command[0].endswith("codex") and command[1] == "exec"
    assert "--ignore-user-config" in command
    assert "--ephemeral" in command
    assert "--output-schema" in command
    assert 'model_reasoning_effort="high"' in command
    assert config["identity"]["schema_sha256"]
    assert config["wire_result_wrapper"] is None
    _assert_openai_strict_schema(config["wire_response_schema"])


def test_codex_wire_schema_normalizes_defaulted_fields_and_root_union() -> None:
    provider = StructuredActionProvider(_codex_spec(), Path("unused"))
    request = AgentActionRequest(
        episode_id="wire-schema",
        step_number=1,
        story_slug="story",
        chapter="0001",
        remaining_steps=2,
        remaining_patch_attempts=1,
    )
    caller_schema = provider._action_schema(provider._declared_tools(request))

    wire_schema, unwrap_key = transport._codex_wire_schema(caller_schema)

    assert unwrap_key == "result"
    assert wire_schema["type"] == "object"
    result_schema = wire_schema["properties"]["result"]
    assert result_schema["type"] == "object"
    assert "finish" in result_schema["properties"]["tool"]["enum"]
    assert "get_qa_findings" in result_schema["properties"]["tool"]["enum"]
    assert "summary" in result_schema["required"]
    assert {"type": "null"} in result_schema["properties"]["summary"]["anyOf"]
    assert "CompleteReviewAction" not in wire_schema.get("$defs", {})
    assert "FinishAction" not in wire_schema.get("$defs", {})
    _assert_openai_strict_schema(wire_schema)


def test_codex_wire_schema_reproduces_and_repairs_missing_findings_requirement() -> None:
    pydantic_style_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "findings": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
            },
        },
        "required": ["summary"],
    }

    wire_schema, unwrap_key = transport._codex_wire_schema(pydantic_style_schema)

    assert unwrap_key is None
    assert wire_schema["required"] == ["summary", "findings"]
    assert "default" not in wire_schema["properties"]["findings"]
    assert wire_schema["additionalProperties"] is False


def test_codex_wire_schema_normalizes_nested_judging_objects() -> None:
    # Authored transport fixture: nested optional objects exercise the wire
    # conversion without depending on any study's evaluator schema.
    nested_schema = {
        "type": "object",
        "properties": {
            "errors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_span": {
                            "oneOf": [
                                {"type": "object", "properties": {
                                    "start": {"type": "integer"},
                                    "end": {"type": "integer"},
                                }},
                                {"type": "null"},
                            ],
                            "default": None,
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["reason"],
                },
                "default": [],
            },
        },
    }

    wire_schema, unwrap_key = transport._codex_wire_schema(nested_schema)

    assert unwrap_key is None
    assert wire_schema["type"] == "object"
    assert "anyOf" in wire_schema["properties"]["errors"]["items"]["properties"]["source_span"]
    _assert_openai_strict_schema(wire_schema)


@pytest.mark.parametrize("items,expected_status", [(["one", "two"], "completed"), (["one", "one"], "failed")])
def test_codex_unique_items_is_removed_only_on_wire_and_enforced_locally(tmp_path, monkeypatch, items, expected_status):
    schema = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "string"},
              "uniqueItems": True}}, "required": ["items"], "additionalProperties": False}
    original = json.loads(json.dumps(schema))
    _install_codex_result(monkeypatch, {"items": items})
    receipt = invoke_structured(_codex_spec(), "Return distinct items.", schema, tmp_path / "unique")
    config = json.loads((tmp_path / "unique" / "request_config.json").read_text())
    assert "uniqueItems" not in config["wire_response_schema"]["properties"]["items"]
    assert config["response_schema"]["properties"]["items"]["uniqueItems"] is True
    assert config["identity"]["schema_sha256"] == transport._digest(original)
    assert schema == original
    assert receipt["status"] == expected_status
    if expected_status == "failed":
        assert "items must be unique" in receipt["error"]


def test_haiku_cli_pin_records_adaptive_thinking_without_changing_legacy_identity(tmp_path, monkeypatch):
    captured = []

    def fake_run(command, *, prompt, cwd, timeout_seconds):
        captured.append(command)
        return json.dumps({"result": '{"verdict":"pass","score":3}',
                           "modelUsage": {"claude-haiku-5-5": {"inputTokens": 15, "outputTokens": 4}}}), "", 0

    monkeypatch.setattr(transport, "_run_process", fake_run)
    spec = ModelSpec("claude", "claude-haiku-5-5", effort="high")
    receipt = invoke_structured(spec, "Judge.", JUDGE_SCHEMA, tmp_path / "haiku")
    assert receipt["status"] == "completed"
    assert receipt["resolved_model"] == "claude-haiku-5-5"
    assert receipt["resolved_models"] == ["claude-haiku-5-5"]
    assert receipt["thinking"] == receipt["identity"]["thinking"] == "adaptive"
    assert receipt["sampling"] == receipt["identity"]["sampling"] == "provider_default"
    command = captured[0]
    assert command[command.index("--model") + 1] == "claude-haiku-5-5"
    assert command[command.index("--effort") + 1] == "high"
    assert "--thinking" not in command
    old = ModelSpec("claude", "claude-sonnet-5", effort="high")
    assert "thinking" not in transport._identity(old, "Judge.", JUDGE_SCHEMA, None)
    assert transport._claude_command(old) == ["claude", *command[1:command.index("--model") + 1], "claude-sonnet-5", *command[command.index("--model") + 2:]]
    with pytest.raises(ValueError, match="requires effort"):
        ModelSpec("claude", "claude-haiku-5-5", effort="medium")
    with pytest.raises(ValueError, match="not pinned"):
        ModelSpec("claude", "haiku", effort="high")


@pytest.mark.parametrize("payload", [
    {"model": "claude-haiku-4-5", "modelUsage": {"claude-haiku-5-5": {}}},
    {"model": "claude-haiku-5-5", "modelUsage": {"claude-haiku-5-5": {}, "claude-sonnet-5-5": {}}},
])
def test_haiku_cli_rejects_resolved_model_or_usage_substitution(tmp_path, monkeypatch, payload):
    monkeypatch.setattr(transport, "_run_process", lambda *_, **__: (json.dumps({**payload, "result": '{"verdict":"pass","score":3}'}), "", 0))
    receipt = invoke_structured(ModelSpec("claude", "claude-haiku-5-5", effort="high"), "Judge.", JUDGE_SCHEMA, tmp_path / "substitute")
    assert receipt["status"] == "failed"
    assert "different model" in receipt["error"]


def test_deepseek_request_is_deterministic_and_generic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                model="deepseek-v4-flash",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"verdict":"fail","score":1}')
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=20, completion_tokens=4, total_tokens=24),
            )

    class Factory:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret-not-real")
    monkeypatch.setattr(transport, "_openai_client", Factory)
    spec = ModelSpec("deepseek", "deepseek-v4-flash", max_output_tokens=12288)

    receipt = invoke_structured(spec, "Judge it.", JUDGE_SCHEMA, tmp_path / "deepseek")

    assert receipt["status"] == "completed"
    assert receipt["output"]["score"] == 1
    assert captured["temperature"] == 0
    assert captured["max_tokens"] == 12288
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    assert captured["response_format"] == {"type": "json_object"}


def test_claude_wrapper_keeps_cli_cost_separate_from_invoice_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    def fake_run(command, *, prompt, cwd, timeout_seconds):
        captured["command"] = command
        return (
            json.dumps(
                {
                    "result": '{"verdict":"pass","score":3}',
                    "model": "claude-sonnet-5",
                    "usage": {"inputTokens": 15, "outputTokens": 4},
                    "total_cost_usd": 0.0123,
                }
            ),
            "",
            0,
        )

    monkeypatch.setattr(transport, "_run_process", fake_run)
    spec = ModelSpec("claude", "claude-sonnet-5", effort="high")

    receipt = invoke_structured(spec, "Judge.", JUDGE_SCHEMA, tmp_path / "claude")

    assert receipt["status"] == "completed"
    assert receipt["output"] == {"verdict": "pass", "score": 3}
    assert receipt["cli_reported_cost_usd"] == pytest.approx(0.0123)
    assert receipt["invoice_cost_usd"] is None
    assert receipt["invoice_cost_status"] == "unknown"
    assert receipt["truncation_status"] == "unknown"
    assert receipt["provider_completion_confirmed"] is False
    assert captured["command"][0] == "claude"
    assert captured["command"][captured["command"].index("--tools") + 1] == ""
    assert "--no-session-persistence" in captured["command"]
    assert "--strict-mcp-config" in captured["command"]


def test_wrong_schema_is_a_recorded_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_codex_result(monkeypatch, {"verdict": "maybe", "score": 99})

    receipt = invoke_structured(
        _codex_spec(), "Judge this output.", JUDGE_SCHEMA, tmp_path / "wrong"
    )

    assert receipt["status"] == "failed"
    assert receipt["output"] is None
    assert receipt["provider_calls"] == 1
    assert "schema" in receipt["error"].lower() or "enum" in receipt["error"].lower()
    assert (tmp_path / "wrong" / "receipt.json").is_file()


def test_unexpected_tool_event_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_codex_result(
        monkeypatch,
        {"verdict": "pass", "score": 4},
        '{"type":"item.completed","item":{"type":"command_execution"}}\n',
    )

    receipt = invoke_structured(
        _codex_spec(), "Judge this output.", JUDGE_SCHEMA, tmp_path / "tool"
    )

    assert receipt["status"] == "failed"
    assert receipt["provider_calls"] == 1
    assert receipt["tool_calls"] == 1
    assert "disabled tool" in receipt["error"]


def test_timeout_records_one_started_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timeout(*args, **kwargs):
        raise transport._ProcessTimeout("process exceeded 1s timeout", "partial", "")

    monkeypatch.setattr(transport, "_run_process", timeout)
    spec = ModelSpec("codex", "gpt-5.6-terra", effort="high", timeout_seconds=1)

    receipt = invoke_structured(spec, "Judge.", JUDGE_SCHEMA, tmp_path / "timeout")

    assert receipt["status"] == "timeout"
    assert receipt["provider_calls"] == 1
    assert receipt["usage"] == {
        "input_tokens": None,
        "output_tokens": None,
        "cached_input_tokens": None,
        "total_tokens": None,
    }


def test_process_interrupt_terminates_child_before_reraising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class InterruptedProcess:
        pid = None
        returncode = None

        def __init__(self):
            self.terminated = False
            self.waited = False

        def communicate(self, **kwargs):
            raise KeyboardInterrupt

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            self.waited = True

    process = InterruptedProcess()
    monkeypatch.setattr(transport.subprocess, "Popen", lambda *args, **kwargs: process)

    with pytest.raises(KeyboardInterrupt):
        transport._run_process(
            ["codex", "exec"],
            prompt="request",
            cwd=tmp_path,
            timeout_seconds=180,
        )

    assert process.terminated is True
    assert process.waited is True


def test_provider_failure_redacts_token_and_preserves_unknown_invoice_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "vck_" + "not-a-real-credential-for-redaction"

    class Completions:
        def create(self, **kwargs):
            raise RuntimeError(f"authorization failed for Bearer {secret}")

    class Factory:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    monkeypatch.setattr(transport, "_openai_client", Factory)
    spec = ModelSpec("deepseek", "deepseek-v4-flash")

    receipt = invoke_structured(spec, "Judge.", JUDGE_SCHEMA, tmp_path / "failed")

    assert receipt["status"] == "failed"
    assert receipt["provider_calls"] == 1
    assert receipt["invoice_cost_usd"] is None
    assert receipt["invoice_cost_status"] == "unknown"
    assert secret not in json.dumps(receipt)
    assert secret not in (tmp_path / "failed" / "receipt.json").read_text()


def test_replay_is_identity_checked_and_never_invokes_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "replay"
    _install_codex_result(monkeypatch, {"verdict": "pass", "score": 4})
    live = invoke_structured(_codex_spec(), "Judge.", JUDGE_SCHEMA, output_dir)
    assert live["provider_calls"] == 1

    def forbidden(*args, **kwargs):  # pragma: no cover - assertion path only.
        raise AssertionError("replay attempted a provider call")

    monkeypatch.setattr(transport, "_run_process", forbidden)
    replay = invoke_structured(
        _codex_spec(),
        "Judge.",
        JUDGE_SCHEMA,
        output_dir,
        provider_mode="replay",
    )

    assert replay["output"] == live["output"]
    assert replay["cache_hit"] is True
    assert replay["provider_calls"] == 0
    with pytest.raises(Exception, match="No completed replay receipt"):
        invoke_structured(
            _codex_spec(),
            "Different prompt.",
            JUDGE_SCHEMA,
            output_dir,
            provider_mode="replay",
        )


def test_luna_effort_is_part_of_replay_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "luna-replay"
    _install_codex_result(monkeypatch, {"verdict": "pass", "score": 4})
    invoke_structured(
        ModelSpec("codex", "gpt-5.6-luna", effort="max"),
        "Judge.",
        JUDGE_SCHEMA,
        output_dir,
    )

    def forbidden(*args, **kwargs):  # pragma: no cover - assertion path only.
        raise AssertionError("replay attempted a provider call")

    monkeypatch.setattr(transport, "_run_process", forbidden)
    with pytest.raises(LLMProviderUnavailable, match="request identity"):
        invoke_structured(
            ModelSpec("codex", "gpt-5.6-luna", effort="high"),
            "Judge.",
            JUDGE_SCHEMA,
            output_dir,
            provider_mode="replay",
        )


def test_structured_action_provider_validates_and_records_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command, *, prompt, cwd, timeout_seconds):
        schema_path = Path(command[command.index("--output-schema") + 1])
        wire_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        properties = wire_schema["properties"]["result"]["properties"]
        result = {name: None for name in properties}
        result.update({"tool": "finish", "summary": "Done."})
        last_path = Path(command[command.index("--output-last-message") + 1])
        last_path.write_text(json.dumps({"result": result}), encoding="utf-8")
        return "", "", 0

    monkeypatch.setattr(transport, "_run_process", fake_run)
    provider = StructuredActionProvider(_codex_spec(), tmp_path / "actions")
    request = AgentActionRequest(
        episode_id="episode-1",
        step_number=1,
        story_slug="story",
        chapter="0001",
        remaining_steps=1,
        remaining_patch_attempts=0,
    )

    action = provider.get_action(request)

    assert action.tool == "finish"
    assert provider.tool_protocol == "json_prompt"
    assert provider.provider_name == "codex"
    assert len(provider.call_records) == 1
    assert provider.call_records[0].cache_hit is False


def test_model_pins_and_chapter_budget_are_enforced() -> None:
    assert ModelSpec("codex", "gpt-5.6-luna").effort == "max"
    assert ModelSpec("codex", "gpt-5.6-luna", effort="max").max_output_tokens == 8192
    assert {
        ModelSpec("codex", "gpt-5.6-luna", effort=effort).effort
        for effort in ("max", "high", "medium", "low")
    } == {"max", "high", "medium", "low"}
    assert ModelSpec("codex", "gpt-6-astra", effort="xhigh").effort == "xhigh"
    assert ModelSpec("codex", "gpt-5.6-sol", effort="high").effort == "high"
    assert ModelSpec("claude", "claude-sonnet-5", effort="high").effort == "high"
    with pytest.raises(ValueError, match="requires effort in"):
        ModelSpec("codex", "gpt-5.6-luna", effort="xhigh")
    with pytest.raises(ValueError, match="requires effort"):
        ModelSpec("codex", "gpt-5.6-terra", effort="max")
    with pytest.raises(ValueError, match="not pinned"):
        ModelSpec("codex", "floating-model", effort="high")
    with pytest.raises(ValueError, match="between 8192"):
        ModelSpec("deepseek", "deepseek-v4-flash", max_output_tokens=2048)


def test_gpt_6_luna_is_pinned_with_max_default_and_lower_effort_fallbacks() -> None:
    assert ModelSpec("codex", "gpt-6-luna").effort == "max"
    assert {
        ModelSpec("codex", "gpt-6-luna", effort=effort).effort
        for effort in ("max", "xhigh", "high", "medium", "low")
    } == {"max", "xhigh", "high", "medium", "low"}
    with pytest.raises(ValueError, match="requires effort in"):
        ModelSpec("codex", "gpt-6-luna", effort="ultra-low")


def test_claude_sonnet_5_allows_high_medium_low_effort() -> None:
    assert {
        ModelSpec("claude", "claude-sonnet-5", effort=effort).effort
        for effort in ("high", "medium", "low")
    } == {"high", "medium", "low"}
    with pytest.raises(ValueError, match="requires effort in"):
        ModelSpec("claude", "claude-sonnet-5", effort="max")


def test_codex_binary_resolution_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # 1) CODEX_BIN override wins outright.
    monkeypatch.setenv("CODEX_BIN", "/opt/custom/codex")
    monkeypatch.setattr(transport.shutil, "which", lambda _name: "/usr/local/bin/codex")
    assert transport._codex_binary() == "/opt/custom/codex"

    # 2) Without an override, PATH resolution (shutil.which) wins.
    monkeypatch.delenv("CODEX_BIN", raising=False)
    assert transport._codex_binary() == "/usr/local/bin/codex"

    # 3) Without PATH or override, the ChatGPT.app bundle path is used if present.
    monkeypatch.setattr(transport.shutil, "which", lambda _name: None)
    fake_app_path = tmp_path / "codex"
    fake_app_path.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(transport, "_CHATGPT_APP_CODEX_PATH", fake_app_path)
    assert transport._codex_binary() == str(fake_app_path)

    # 4) Falls back to a bare "codex" when nothing else resolves.
    monkeypatch.setattr(transport, "_CHATGPT_APP_CODEX_PATH", tmp_path / "missing" / "codex")
    assert transport._codex_binary() == "codex"


def test_codex_command_uses_resolved_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transport, "_codex_binary", lambda: "/opt/custom/codex")
    command = transport._codex_command(_codex_spec(), Path("schema.json"), Path("last.txt"))
    assert command[0] == "/opt/custom/codex"


@pytest.mark.parametrize("backend,confirmed", [("codex", True), ("claude", True), ("claude", False)])
def test_cli_threshold_does_not_reject_completed_output_or_reasoning(tmp_path, monkeypatch, backend, confirmed):
    def fake_run(command, **kwargs):
        result = {"verdict": "pass", "score": 4}
        if backend == "codex":
            Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps(result))
            return json.dumps({"type": "turn.completed", "model": "gpt-6-luna", "usage": {
                "input_tokens": 10, "output_tokens": 17000, "reasoning_output_tokens": 16900}}), "", 0
        return json.dumps({"result": json.dumps(result), "stop_reason": "end_turn" if confirmed else None, "model": "claude-haiku-5-5",
            "usage": {"input_tokens": 10, "output_tokens": 17000, "cache_read_input_tokens": 20,
                "cache_creation_input_tokens": 30, "output_tokens_details": {"thinking_tokens": 16900}}}), "", 0

    monkeypatch.setattr(transport, "_run_process", fake_run)
    spec = ModelSpec(backend, "gpt-6-luna" if backend == "codex" else "claude-haiku-5-5",
                     effort="max" if backend == "codex" else "high", max_output_tokens=16384)
    receipt = invoke_structured(spec, "Judge.", JUDGE_SCHEMA, tmp_path / backend)
    assert receipt["status"] == "completed"
    assert receipt["output_limit_policy"] == "observational_threshold"
    assert receipt["threshold_exceeded"] is True
    assert receipt["reasoning_output_tokens"] == 16900
    assert receipt["nonreasoning_output_tokens"] == 100
    assert receipt["truncation_status"] == ("not_observed" if confirmed else "unknown")
    assert receipt["provider_completion_confirmed"] is confirmed
    assert receipt["usage"]["output_tokens"] == 17000
    if backend == "claude":
        assert receipt["usage"] == {"input_tokens": 60, "output_tokens": 17000,
                                     "cached_input_tokens": 20, "total_tokens": 17060}
        assert receipt["cache_creation_input_tokens"] == 30


@pytest.mark.parametrize("backend,confirmed", [("codex", True), ("claude", True), ("claude", False)])
def test_invalid_inner_json_preserves_usage_and_completion_evidence(tmp_path, monkeypatch, backend, confirmed):
    def fake_run(command, **kwargs):
        if backend == "codex":
            Path(command[command.index("--output-last-message") + 1]).write_text('{"verdict":')
            return json.dumps({"type": "turn.completed", "model": "gpt-6-luna", "usage": {
                "input_tokens": 10, "output_tokens": 100, "reasoning_output_tokens": 80}}), "", 0
        return json.dumps({"result": '{"verdict":', "stop_reason": "end_turn" if confirmed else None,
            "modelUsage": {"claude-haiku-5-5": {"inputTokens": 10, "outputTokens": 100, "thinkingTokens": 80}}}), "", 0

    monkeypatch.setattr(transport, "_run_process", fake_run)
    model = "gpt-6-luna" if backend == "codex" else "claude-haiku-5-5"
    receipt = invoke_structured(ModelSpec(backend, model, effort="high" if backend == "claude" else None), "Judge.", JUDGE_SCHEMA, tmp_path / backend)
    assert receipt["status"] == "failed"
    assert receipt["failure_kind"] == ("invalid_json" if confirmed else "unconfirmed_final_output")
    assert receipt["usage"]["total_tokens"] == 110
    assert receipt["raw_usage"]
    assert receipt["resolved_model"] == model
    assert receipt["reasoning_output_tokens"] == 80
    assert receipt["nonreasoning_output_tokens"] == 20
    assert receipt["provider_completion_confirmed"] is confirmed
    assert receipt["truncation_status"] == ("not_observed" if confirmed else "unknown")


@pytest.mark.parametrize("backend", ["codex", "claude"])
def test_invalid_cli_framing_is_not_an_inner_output_failure(tmp_path, monkeypatch, backend):
    def fake_run(command, **kwargs):
        if backend == "codex":
            Path(command[command.index("--output-last-message") + 1]).write_text('{"verdict":"pass","score":4}')
            return '{"type": broken}\n{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":100}}', "", 0
        return '{"result":"broken",', "", 0

    monkeypatch.setattr(transport, "_run_process", fake_run)
    receipt = invoke_structured(ModelSpec(backend, "gpt-6-luna" if backend == "codex" else "claude-haiku-5-5",
                                          effort="high" if backend == "claude" else None),
                                "Judge.", JUDGE_SCHEMA, tmp_path / backend)
    assert receipt["status"] == "failed"
    assert receipt["failure_kind"] == "invalid_cli_envelope"
    assert receipt["output"] is None
    assert receipt["provider_completion_confirmed"] is False


@pytest.mark.parametrize("event,expected", [
    ({"type": "item.completed", "item": {"type": "error", "message": "Under-development features enabled: skip_host_skill_discovery"}}, "completed"),
    ({"type": "turn.failed", "error": {"message": "Provider failed"}}, "failed"),
    ({"type": "error", "message": "Provider failed"}, "failed"),
])
def test_codex_distinguishes_warning_items_from_top_level_failures(tmp_path, monkeypatch, event, expected):
    events = "\n".join(json.dumps(row) for row in [event, {"type": "turn.completed", "usage": {
        "input_tokens": 10, "output_tokens": 100}}])
    _install_codex_result(monkeypatch, {"verdict": "pass", "score": 4}, events)
    receipt = invoke_structured(_codex_spec(), "Judge.", JUDGE_SCHEMA, tmp_path / "warning")
    assert receipt["status"] == expected
    assert receipt["usage"]["total_tokens"] == 110
    if expected == "completed":
        assert receipt["provider_completion_confirmed"] is True
        assert receipt["truncation_status"] == "not_observed"
    else:
        assert receipt["failure_kind"] == "provider_or_transport"
        assert receipt["output"] is None


def test_claude_explicit_token_limit_retains_metadata_before_failed_result_parsing(tmp_path, monkeypatch):
    payload = {"result": "broken", "stop_reason": "max_tokens", "model": "claude-haiku-5-5",
        "usage": {"input_tokens": 10, "output_tokens": 8192, "cache_read_input_tokens": 20,
                  "cache_creation_input_tokens": 30}, "total_cost_usd": .05}
    monkeypatch.setattr(transport, "_run_process", lambda *args, **kwargs: (json.dumps(payload), "", 0))
    receipt = invoke_structured(ModelSpec("claude", "claude-haiku-5-5", effort="high"), "Judge.", JUDGE_SCHEMA, tmp_path / "limit")
    assert receipt["status"] == "failed"
    assert receipt["failure_kind"] == "output_limit"
    assert receipt["stop_reason"] == "max_tokens"
    assert receipt["truncation_status"] == "provider_limit_reached"
    assert receipt["usage"]["total_tokens"] == 8252
    assert receipt["resolved_model"] == "claude-haiku-5-5"
    assert receipt["raw_usage"] == payload["usage"]
    assert receipt["cli_reported_cost_usd"] == .05


def test_native_limit_retains_usage_and_v3_identity(tmp_path, monkeypatch):
    response = SimpleNamespace(model="deepseek-v4-flash", usage={"prompt_tokens": 10, "completion_tokens": 8192},
        choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content="broken"))])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-test-key")
    monkeypatch.setattr(transport, "_openai_client", lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response))))
    spec = ModelSpec("deepseek", "deepseek-v4-flash")
    receipt = invoke_structured(spec, "Judge.", JUDGE_SCHEMA, tmp_path / "native")
    assert receipt["status"] == "failed"
    assert receipt["truncation_status"] == "provider_limit_reached"
    assert receipt["usage"]["total_tokens"] == 8202
    assert receipt["raw_usage"] == response.usage
    assert receipt["stop_reason"] == "length"
    assert receipt["identity"]["version"] == "structured-transport.v3"


@pytest.mark.parametrize("backend", ["codex", "claude", "deepseek"])
def test_cli_v3_replay_is_stale_while_native_v3_replay_remains_compatible(tmp_path, monkeypatch, backend):
    model = {"codex": "gpt-6-luna", "claude": "claude-haiku-5-5", "deepseek": "deepseek-v4-flash"}[backend]
    spec = ModelSpec(backend, model, effort="high" if backend == "claude" else None)
    identity = transport._identity(spec, "Judge.", JUDGE_SCHEMA, None)
    identity["version"] = "structured-transport.v3"
    identity.pop("identity_sha256")
    identity["identity_sha256"] = transport._digest(identity)
    folder = tmp_path / backend
    folder.mkdir()
    (folder / "receipt.json").write_text(json.dumps({"identity": identity, "status": "completed",
        "output": {"verdict": "pass", "score": 4}}))
    monkeypatch.setattr(transport, "_run_process", lambda *args, **kwargs: pytest.fail("replay made a live call"))
    monkeypatch.setattr(transport, "_openai_client", lambda *args, **kwargs: pytest.fail("replay made a live call"))
    if backend == "deepseek":
        replay = invoke_structured(spec, "Judge.", JUDGE_SCHEMA, folder, provider_mode="replay")
        assert replay["provider_calls"] == 0
        assert replay["output"]["score"] == 4
    else:
        with pytest.raises(LLMProviderUnavailable, match="request identity"):
            invoke_structured(spec, "Judge.", JUDGE_SCHEMA, folder, provider_mode="replay")
