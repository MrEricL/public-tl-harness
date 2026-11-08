"""Offline transport fixtures; these tests provide no model-quality evidence."""
from decimal import Decimal
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from agentic_translation.autonomous_provider import (
    GenerationOutputError, GenerationTransportError, _anthropic_judge_price,
    digest, ledger_for, make_judges,
)
from agentic_translation.costing import BudgetExceeded
from agentic_translation.providers_llm import LLMProviderUnavailable
from agentic_translation.structured_transport import ModelSpec


SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}},
          "required": ["ok"], "additionalProperties": False}


def config(tmp_path, model="claude-haiku-5-5", backend="anthropic"):
    return {"budgets": {"ledger": str(tmp_path / "ledger.json"), "max_experiment_usd": 1,
                        "evaluation_reserve_usd": 0},
            "judging": {"judges": [{"model": model, "backend": backend,
                                     "effort": "max" if backend == "codex" else "high"}]}}


def request(rep="1", protocol="error-count-v1"):
    return {"prompt": "An authored candidate.", "schema": SCHEMA,
            "protocol_version": protocol, "repetition_id": rep}


def install_anthropic(monkeypatch, *, usage=None, stop="end_turn", model="claude-haiku-5-5", content='{"ok":true}'):
    calls = []
    usage = {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 50,
             "cache_creation_input_tokens": 0} if usage is None else usage

    class Fake:
        def __init__(self, **kwargs):
            assert kwargs == {"max_retries": 0, "timeout": 600}
            self.messages = SimpleNamespace(create=self.create)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(_request_id="fixture-request", model_dump=lambda **_:
                {"model": model, "usage": usage, "stop_reason": stop,
                 "content": [{"type": "text", "text": content}]})

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=Fake))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-key")
    return calls


def test_native_independent_repetitions_and_zero_call_replay(tmp_path, monkeypatch):
    calls = install_anthropic(monkeypatch)
    cfg = config(tmp_path)
    judges, _ = make_judges(tmp_path, cfg)
    first, second = [judges[0][2](request(rep)) for rep in ("1", "2")]
    assert len(calls) == 2
    assert first["receipt"]["request_sha256"] != second["receipt"]["request_sha256"]
    assert first["receipt"]["repetition_id"] == "1"
    assert second["receipt"]["repetition_id"] == "2"
    assert first["receipt"]["input_tokens_total"] == 150
    assert first["receipt"]["input_tokens_uncached"] == 100
    assert Decimal(first["receipt"]["estimated_charge"]) == Decimal("0.0000205")
    for wire in calls:
        assert wire["thinking"] == {"type": "disabled"}
        assert wire["output_config"]["effort"] == "high"
        assert wire["output_config"]["format"]["schema"] == SCHEMA
        assert not {"temperature", "top_p", "top_k", "tools"} & wire.keys()
    replay, _ = make_judges(tmp_path, cfg, replay=True)
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    for rep in ("1", "2"):
        result = replay[0][2](request(rep))
        assert result["receipt"]["provider_calls"] == 0
        assert result["receipt"]["physical_charge_usd"] == "0"
    assert len(calls) == 2
    assert len(ledger_for(cfg).snapshot()["calls"]) == 2
    with pytest.raises(LLMProviderUnavailable, match="replay"):
        replay[0][2](request("3"))
    with pytest.raises(LLMProviderUnavailable, match="replay"):
        replay[0][2](request("1", "error-count-v2"))


def test_codex_repetitions_use_separate_physical_calls_and_preserve_live_receipts(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, *, prompt, cwd, timeout_seconds):
        calls.append(command)
        Path(command[command.index("--output-last-message") + 1]).write_text('{"ok":true}')
        return '{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":20}}\n', "", 0

    monkeypatch.setattr("agentic_translation.structured_transport._run_process", fake_run)
    cfg = config(tmp_path, "gpt-6-luna", "codex")
    judges, _ = make_judges(tmp_path, cfg)
    receipts = [judges[0][2](request(rep))["receipt"] for rep in ("1", "2")]
    assert len(calls) == 2
    assert len({r["artifact_dir"] for r in receipts}) == 2
    assert all(r["effort"] == "max" and r["provider_calls"] == 1 for r in receipts)
    assert [r["identity"]["exposure"]["repetition_id"] for r in receipts] == ["1", "2"]
    assert all(r["identity"]["exposure"]["protocol_version"] == "error-count-v1" for r in receipts)
    replay, _ = make_judges(tmp_path, cfg, replay=True)
    for rep in ("1", "2"):
        assert replay[0][2](request(rep))["receipt"]["provider_calls"] == 0
    assert len(calls) == 2
    for receipt in receipts:
        persisted = json.loads((Path(receipt["artifact_dir"]) / "receipt.json").read_text())
        assert persisted["provider_calls"] == 1
        assert persisted["judge_request_identity"]["repetition_id"] in {"1", "2"}
    with pytest.raises(LLMProviderUnavailable, match="replay"):
        replay[0][2](request("3"))


def test_legacy_codex_cache_path_is_unchanged(tmp_path, monkeypatch):
    seen = []

    def fake(spec, prompt, schema, folder, *, provider_mode):
        seen.append(folder)
        return {"status": "completed", "output": {"ok": True}}

    monkeypatch.setattr("agentic_translation.structured_transport.invoke_structured", fake)
    judges, _ = make_judges(tmp_path, config(tmp_path, "gpt-6-luna", "codex"))
    legacy = {"prompt": "An authored candidate.", "schema": SCHEMA}
    judges[0][2](legacy)
    assert seen == [tmp_path / "evaluation" / "cli" / "codex" / "gpt-6-luna" / digest(legacy)]
    with pytest.raises(ValueError, match="repetition_id"):
        judges[0][2]({**legacy, "protocol_version": "error-count-v1"})


def test_structured_sol_adjudicator_is_pinned_and_luna_max_retained(tmp_path):
    cfg = config(tmp_path, "gpt-6-luna", "codex")
    cfg["judging"]["adjudicator"] = {"backend": "codex", "model": "gpt-6.1-sol", "effort": "high"}
    judges, adjudicator = make_judges(tmp_path, cfg)
    assert judges[0][:2] == ("gpt-6-luna", "openai")
    assert adjudicator[:2] == ("gpt-6.1-sol", "openai")
    ModelSpec("codex", "gpt-6.1-sol", effort="high")
    ModelSpec("codex", "gpt-6-luna", effort="max")
    with pytest.raises(ValueError, match="requires effort"):
        ModelSpec("codex", "gpt-6.1-sol", effort="medium")
    for changes in ({"model": "claude-haiku-unpinned"}, {"effort": "max"}, {"family": "openai"}):
        cfg = config(tmp_path)
        cfg["judging"]["judges"][0].update(changes)
        with pytest.raises(ValueError):
            make_judges(tmp_path, cfg)


def test_native_budget_stops_before_call_and_missing_usage_retains_reservation(tmp_path, monkeypatch):
    calls = install_anthropic(monkeypatch, usage={})
    cfg = config(tmp_path)
    cfg["budgets"]["max_experiment_usd"] = 0
    judge = make_judges(tmp_path, cfg)[0][0][2]
    with pytest.raises(BudgetExceeded):
        judge(request())
    assert not calls
    cfg["budgets"].update(ledger=str(tmp_path / "other-ledger.json"), max_experiment_usd=1)
    judge = make_judges(tmp_path, cfg)[0][0][2]
    result = judge(request())
    assert result["receipt"]["estimated_charge"] is None
    state = ledger_for(cfg).snapshot()
    assert state["unknown_usage_calls"] == 1
    row = next(iter(state["calls"].values()))
    assert Decimal(row["budget_charge_usd"]) == Decimal(row["reserved_usd"]) > 0


@pytest.mark.parametrize("stop,model,content", [
    ("max_tokens", "claude-haiku-5-5", '{"ok":true}'),
    ("end_turn", "claude-sonnet-5-5", '{"ok":true}'),
    ("end_turn", "claude-haiku-5-5", '{"unexpected":true}'),
])
def test_native_paid_failures_are_receipted_and_never_cached(tmp_path, monkeypatch, stop, model, content):
    calls = install_anthropic(monkeypatch, stop=stop, model=model, content=content)
    cfg = config(tmp_path)
    judge = make_judges(tmp_path, cfg)[0][0][2]
    with pytest.raises((ValueError, LLMProviderUnavailable)):
        judge(request())
    assert len(calls) == 1
    receipt = json.loads(next((tmp_path / "calls").glob("*.json")).read_text())
    assert receipt["status"] == "failed" and receipt["usage_complete"]
    assert Decimal(receipt["estimated_charge"]) > 0
    assert len(ledger_for(cfg).snapshot()["calls"]) == 1
    with pytest.raises(LLMProviderUnavailable, match="replay"):
        make_judges(tmp_path, cfg, replay=True)[0][0][2](request())


def test_native_truncation_has_typed_failure_with_raw_paid_receipt(tmp_path, monkeypatch):
    calls = install_anthropic(monkeypatch, stop="max_tokens", content='{"ok":')
    cfg = config(tmp_path)
    with pytest.raises(GenerationOutputError) as caught:
        make_judges(tmp_path, cfg)[0][0][2](request())
    receipt = caught.value.receipt
    assert receipt["truncation_status"] == "provider_limit_reached"
    assert receipt["raw_response"]["content"][0]["text"] == '{"ok":'
    assert receipt["status"] == "failed" and receipt["usage_complete"]
    assert receipt["error_type"] == "GenerationOutputError"
    assert Decimal(receipt["estimated_charge"]) > 0
    saved = json.loads((tmp_path / "calls" / (receipt["call_id"] + ".json")).read_text())
    assert saved == receipt
    assert len(calls) == 1


def test_native_refusal_is_fatal_even_with_schema_invalid_text(tmp_path, monkeypatch):
    install_anthropic(monkeypatch, stop="refusal", content="I cannot comply.")
    with pytest.raises(GenerationTransportError):
        make_judges(tmp_path, config(tmp_path))[0][0][2](request())


def test_haiku_cli_judge_receipts_and_identity_disclose_transport_settings(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, *, prompt, cwd, timeout_seconds):
        calls.append(command)
        return json.dumps({"model": "claude-haiku-5-5", "result": '{"ok":true}',
                           "usage": {"inputTokens": 100, "outputTokens": 20}, "total_cost_usd": .001}), "", 0

    monkeypatch.setattr("agentic_translation.structured_transport._run_process", fake_run)
    cfg = config(tmp_path, "claude-haiku-5-5", "claude")
    result = make_judges(tmp_path, cfg)[0][0][2](request())
    receipt = result["receipt"]
    assert receipt["thinking"] == receipt["judge_request_identity"]["thinking"] == "adaptive"
    assert receipt["sampling"] == receipt["judge_request_identity"]["sampling"] == "provider_default"
    assert receipt["cost_basis"] == "subscription_equivalent_usd_not_charged"
    assert receipt["estimated_charge"] is None
    assert receipt["subscription_equivalent_usd"] == pytest.approx(.001)
    replay = make_judges(tmp_path, cfg, replay=True)[0][0][2](request())
    assert replay["receipt"]["provider_calls"] == 0
    assert len(calls) == 1


@pytest.mark.parametrize("receipt,expected", [
    ({"status": "failed", "truncation_status": "provider_limit_reached", "error": "RuntimeError: limit"}, GenerationOutputError),
    ({"status": "failed", "error": "_SchemaError: $.ok: expected boolean"}, GenerationOutputError),
    ({"status": "failed", "failure_kind": "invalid_json", "provider_completion_confirmed": True,
      "error": "JSONDecodeError: malformed final result", "usage": {"output_tokens": 123}}, GenerationOutputError),
    ({"status": "failed", "failure_kind": "invalid_cli_envelope",
      "error": "JSONDecodeError: malformed outer framing"}, GenerationTransportError),
    ({"status": "failed", "failure_kind": "unconfirmed_final_output",
      "error": "JSONDecodeError: malformed final result"}, GenerationTransportError),
    ({"status": "failed", "error": "ValueError: invalid JSON"}, GenerationTransportError),
    ({"status": "timeout", "error": "_ProcessTimeout: deadline"}, GenerationTransportError),
    ({"status": "failed", "error": "RuntimeError: Claude reported is_error=true"}, GenerationTransportError),
])
def test_cli_only_explicit_output_failure_evidence_is_recoverable(tmp_path, monkeypatch, receipt, expected):
    def fake(spec, prompt, schema, folder, *, provider_mode=None, **kwargs):
        return receipt

    monkeypatch.setattr("agentic_translation.structured_transport.invoke_structured", fake)
    with pytest.raises(expected) as caught:
        make_judges(tmp_path, config(tmp_path, "gpt-6-luna", "codex"))[0][0][2](request())
    assert caught.value.receipt["error"] == receipt["error"]
    if expected is GenerationOutputError:
        assert caught.value.receipt["protocol_version"] == "error-count-v1"
        assert caught.value.receipt["repetition_id"] == "1"


@pytest.mark.parametrize("backend,model", [("anthropic", "claude-haiku-5-5"), ("codex", "gpt-6-luna")])
def test_identical_same_repetition_different_evaluation_units_make_separate_calls(tmp_path, monkeypatch, backend, model):
    if backend == "anthropic":
        calls = install_anthropic(monkeypatch)
    else:
        calls = []

        def fake_run(command, *, prompt, cwd, timeout_seconds):
            calls.append(command)
            Path(command[command.index("--output-last-message") + 1]).write_text('{"ok":true}')
            return '{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":20}}\n', "", 0

        monkeypatch.setattr("agentic_translation.structured_transport._run_process", fake_run)
    cfg = config(tmp_path, model, backend)
    judge = make_judges(tmp_path, cfg)[0][0][2]
    receipts = [judge({**request(), "evaluation_unit_id": unit})["receipt"] for unit in ("opaque-unit-a", "opaque-unit-b")]
    assert len(calls) == 2
    assert [r["evaluation_unit_id"] for r in receipts] == ["opaque-unit-a", "opaque-unit-b"]
    if backend == "codex":
        assert [r["identity"]["exposure"]["evaluation_unit_id"] for r in receipts] == ["opaque-unit-a", "opaque-unit-b"]
    else:
        assert receipts[0]["request_sha256"] != receipts[1]["request_sha256"]
        assert all("opaque-unit" not in json.dumps(call["messages"]) for call in calls)
    replay = make_judges(tmp_path, cfg, replay=True)[0][0][2]
    for unit in ("opaque-unit-a", "opaque-unit-b"):
        assert replay({**request(), "evaluation_unit_id": unit})["receipt"]["provider_calls"] == 0
    assert len(calls) == 2


def test_native_long_context_mixed_cache_ttls_and_reasoning_price():
    usage = {"input_tokens": 100_000, "output_tokens": 20, "reasoning_tokens": 4,
             "cache_read_input_tokens": 100, "cache_creation_input_tokens": 30,
             "cache_creation": {"ephemeral_5m_input_tokens": 10, "ephemeral_1h_input_tokens": 20}}
    priced = _anthropic_judge_price(usage, "claude-haiku-5-5")
    assert priced["pricing_period"] == "prompt_over_100k"
    assert priced["input_tokens_total"] == 100_130
    assert priced["output_tokens_billable_ordinary"] == 16
    assert priced["reasoning_tokens_if_separately_reported"] == 4
    assert Decimal(priced["estimated_charge"]) == Decimal("0.05008125")
    assert priced["cache_write_1h_per_million"] == "1.00"
    no_ttl = _anthropic_judge_price({k: v for k, v in usage.items() if k != "cache_creation"}, "claude-haiku-5-5")
    assert no_ttl["estimated_charge"] is None and not no_ttl["usage_complete"]
    with pytest.raises(ValueError):
        _anthropic_judge_price({"input_tokens": -1, "output_tokens": 1}, "claude-haiku-5-5")


@pytest.mark.parametrize("backend,thinking", [("claude", "adaptive"), ("anthropic", "disabled")])
def test_haiku_explicit_thinking_and_sampling_are_validated_before_calls(tmp_path, monkeypatch, backend, thinking):
    def forbidden(*args, **kwargs):
        raise AssertionError("configuration validation must not invoke a provider")

    monkeypatch.setattr("agentic_translation.structured_transport.invoke_structured", forbidden)
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=forbidden))
    cfg = config(tmp_path, "claude-haiku-5-5", backend)
    entry = cfg["judging"]["judges"][0]
    entry.update(thinking=thinking, sampling="provider_default")
    assert make_judges(tmp_path, cfg)[0][0][:2] == ("claude-haiku-5-5", "anthropic")
    for field, value in (("thinking", "disabled" if thinking == "adaptive" else "adaptive"),
                         ("thinking", None), ("sampling", "temperature_zero"), ("sampling", None)):
        entry.update(thinking=thinking, sampling="provider_default")
        entry[field] = value
        with pytest.raises(ValueError, match=field):
            make_judges(tmp_path, cfg)
