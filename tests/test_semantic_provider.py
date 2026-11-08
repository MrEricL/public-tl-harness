from __future__ import annotations

import io
import hashlib
import json
import urllib.error
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

from agentic_translation.semantic_models import JevPolicy, SemanticSnapshot
from agentic_translation.semantic_provider import (
    GatewayJudgmentProvider,
    JudgmentProviderUnavailable,
    ReplayJudgmentProvider,
    ReplayMissError,
    _retry_after_seconds,
    semantic_config_digest,
)
from agentic_translation.semantic_signals import categories_for_set


def _response(question_set: str = "focused") -> dict:
    return {
        "model": "typesafe-ai/jev",
        "answers": {
            category: {"type": "boolean", "probability": index / 20}
            for index, category in enumerate(categories_for_set(question_set))
        },
        "usage": {"inputTokens": 123, "outputTokens": 5},
        "providerMetadata": {
            "gateway": {
                "routing": {
                    "originalModelId": "typesafe-ai/jev",
                    "resolvedProvider": "typesafe-ai",
                    "canonicalSlug": "typesafe-ai/jev",
                    "finalProvider": "typesafe-ai",
                },
                "cost": "0.000005166",
                "marketCost": "0.000005166",
                "surchargeCost": "0",
                "gatewayCost": "0.000005166",
                "generationId": "gen_test",
            }
        },
    }


def test_off_policy_never_calls_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = SemanticSnapshot(source_text="原文", draft_text="draft")
    provider = GatewayJudgmentProvider(JevPolicy(mode="off"))
    monkeypatch.setattr(
        provider,
        "_post_json",
        lambda *_: pytest.fail("off policy must not call the network"),
    )

    report = provider.evaluate(snapshot, "focused")

    assert report.status == "unavailable"
    assert report.requests == []
    assert report.issues[0].code == "disabled"


def test_enabled_policy_requires_gateway_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    provider = GatewayJudgmentProvider(JevPolicy(mode="shadow"))
    with pytest.raises(JudgmentProviderUnavailable, match="AI_GATEWAY_API_KEY"):
        provider.evaluate(SemanticSnapshot(source_text="原", draft_text="draft"), "focused")


def test_gateway_request_uses_native_evaluate_shape_and_parses_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-only-key")
    provider = GatewayJudgmentProvider(JevPolicy(mode="shadow"), record_dir=tmp_path)
    seen: list[dict] = []

    def fake_post(payload: dict, api_key: str) -> dict:
        seen.append(payload)
        assert api_key == "test-only-key"
        return _response()

    monkeypatch.setattr(provider, "_post_json", fake_post)
    snapshot = SemanticSnapshot(
        source_text="他没有走。" * 250,
        draft_text="He did not leave. " * 80,
    )
    report = provider.evaluate(snapshot, "focused")

    assert report.status == "completed"
    assert report.is_fresh(snapshot)
    assert len(report.signals) == 5
    assert len(report.requests) == 1
    request = report.requests[0]
    assert request.input_tokens == 123
    assert str(request.cost_usd) == "0.000005166"
    assert request.routing.final_provider == "typesafe-ai"
    assert request.accounting_complete is True
    assert report.coverage.source_judged_fraction == 1
    assert report.coverage.draft_judged_fraction == 1
    assert seen[0]["model"] == "typesafe-ai/jev"
    assert seen[0]["providerOptions"]["gateway"]["only"] == ["typesafe-ai"]
    assert set(seen[0]["state"]) == {
        "source_window",
        "draft_window",
        "glossary",
        "style_guide",
        "context",
    }
    assert all(question["type"] == "boolean" for question in seen[0]["questions"].values())
    receipt_path = next((tmp_path / "receipts").glob("*.json"))
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    assert receipt["request_payload"] == seen[0]
    assert receipt["response"] == _response()
    assert "Authorization" not in receipt_bytes.decode("utf-8")
    assert request.receipt_sha256s == [hashlib.sha256(receipt_bytes).hexdigest()]


def test_missing_and_malformed_answers_produce_partial_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-only-key")
    provider = GatewayJudgmentProvider(JevPolicy(mode="shadow"))
    response = _response()
    response["answers"].pop("omission")
    response["answers"]["unsupported_addition"] = {
        "type": "boolean",
        "probability": 7,
    }
    monkeypatch.setattr(provider, "_post_json", lambda *_: response)

    report = provider.evaluate(
        SemanticSnapshot(source_text="原文", draft_text="draft"), "focused"
    )

    assert report.status == "partial"
    assert len(report.signals) == 3
    assert {issue.code for issue in report.issues} == {"missing_answer", "malformed_answer"}
    assert report.requests[0].status == "partial"


def test_transient_error_retries_and_honors_bounded_retry_after(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "secret-test-key")
    policy = JevPolicy(mode="shadow", max_retries=2, retry_after_cap_seconds=1)
    provider = GatewayJudgmentProvider(policy, record_dir=tmp_path)
    attempts = 0
    sleeps: list[float] = []

    def flaky(*_: object) -> dict:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise urllib.error.HTTPError(
                policy.endpoint,
                429,
                "retry secret-test-key",
                {"Retry-After": "99"},
                io.BytesIO(b'{"error":"secret-test-key"}'),
            )
        return _response()

    monkeypatch.setattr(provider, "_post_json", flaky)
    monkeypatch.setattr("agentic_translation.semantic_provider.time.sleep", sleeps.append)
    report = provider.evaluate(
        SemanticSnapshot(source_text="原文", draft_text="draft"), "focused"
    )

    assert attempts == 2
    assert sleeps == [1]
    assert report.status == "completed"
    assert report.requests[0].attempts == 2
    assert report.requests[0].retry_count == 1
    assert report.requests[0].accounting_complete is False
    assert report.requests[0].input_tokens is None
    assert report.requests[0].cost_usd is None
    assert report.requests[0].last_response_input_tokens == 123
    assert str(report.requests[0].last_response_cost_usd) == "0.000005166"
    assert "secret-test-key" not in report.model_dump_json()
    receipts = [json.loads(path.read_text()) for path in sorted((tmp_path / "receipts").glob("*.json"))]
    assert [receipt["status"] for receipt in receipts] == ["http_error", "completed"]
    assert [receipt["attempt"] for receipt in receipts] == [1, 2]
    assert "secret-test-key" not in json.dumps(receipts)
    assert report.requests[0].receipt_sha256s == [
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((tmp_path / "receipts").glob("*.json"))
    ]


def test_http_date_retry_after_is_honored_and_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        "agentic_translation.semantic_provider.time.time",
        lambda: now.timestamp(),
    )
    headers = {"Retry-After": format_datetime(now + timedelta(seconds=90), usegmt=True)}
    assert _retry_after_seconds(headers, cap=7.5) == 7.5


def test_permanent_auth_error_is_not_retried_and_secret_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "secret-test-key")
    provider = GatewayJudgmentProvider(
        JevPolicy(mode="shadow", max_retries=5),
        record_dir=tmp_path,
    )
    attempts = 0

    def unauthorized(*_: object) -> dict:
        nonlocal attempts
        attempts += 1
        raise urllib.error.HTTPError(
            "https://ai-gateway.vercel.sh/v1/evaluate",
            401,
            "bad key",
            {},
            io.BytesIO(b'{"error":"Bearer secret-test-key"}'),
        )

    monkeypatch.setattr(provider, "_post_json", unauthorized)
    report = provider.evaluate(
        SemanticSnapshot(source_text="原文", draft_text="draft"), "focused"
    )

    assert attempts == 1
    assert report.status == "unavailable"
    assert report.requests[0].attempts == 1
    assert report.requests[0].accounting_complete is False
    assert report.coverage.source_fraction == 1
    assert report.coverage.source_judged_fraction == 0
    assert "secret-test-key" not in report.model_dump_json()
    assert "[REDACTED]" in report.issues[0].message
    receipt_text = next((tmp_path / "receipts").glob("*.json")).read_text()
    assert "secret-test-key" not in receipt_text
    assert "[REDACTED]" in receipt_text


def test_record_and_replay_are_fresh_cache_only_and_stale_inputs_miss(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-only-key")
    policy = JevPolicy(mode="shadow")
    live = GatewayJudgmentProvider(policy, record_dir=tmp_path)
    monkeypatch.setattr(live, "_post_json", lambda *_: _response())
    snapshot = SemanticSnapshot(source_text="原文", draft_text="draft")
    expected = live.evaluate(snapshot, "focused")

    monkeypatch.delenv("AI_GATEWAY_API_KEY")
    replay = ReplayJudgmentProvider(tmp_path, policy=policy)
    actual = replay.evaluate(snapshot, "focused")

    assert actual == expected
    assert type(expected).model_validate_json(expected.model_dump_json()) == expected
    with pytest.raises(ReplayMissError, match="No fresh"):
        replay.evaluate(
            SemanticSnapshot(source_text="原文 changed", draft_text="draft"),
            "focused",
        )


def test_identical_live_inputs_replay_distinct_results_in_invocation_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-only-key")
    policy = JevPolicy(mode="shadow")
    live = GatewayJudgmentProvider(policy, record_dir=tmp_path)
    invocation = 0

    def changing_response(*_: object) -> dict:
        nonlocal invocation
        response = _response()
        response["answers"]["omission"]["probability"] = 0.1 + invocation * 0.7
        invocation += 1
        return response

    monkeypatch.setattr(live, "_post_json", changing_response)
    snapshot = SemanticSnapshot(source_text="相同原文", draft_text="same draft")
    first_live = live.evaluate(snapshot, "focused")
    second_live = live.evaluate(snapshot, "focused")
    assert sorted(path.name for path in tmp_path.glob("*.json")) == [
        f"{first_live.cache_key}.000000.json",
        f"{first_live.cache_key}.000001.json",
    ]
    receipt_paths = sorted((tmp_path / "receipts").glob("*.json"))
    assert len(receipt_paths) == 2
    assert [
        json.loads(path.read_text())["response"]["answers"]["omission"]["probability"]
        for path in receipt_paths
    ] == pytest.approx([0.1, 0.8])
    assert first_live.requests[0].receipt_sha256s == [
        hashlib.sha256(receipt_paths[0].read_bytes()).hexdigest()
    ]
    assert second_live.requests[0].receipt_sha256s == [
        hashlib.sha256(receipt_paths[1].read_bytes()).hexdigest()
    ]

    replay = ReplayJudgmentProvider(tmp_path, policy=policy)
    first_replay = replay.evaluate(snapshot, "focused")
    second_replay = replay.evaluate(snapshot, "focused")
    assert first_replay.signals[0].probability == 0.1
    assert second_replay.signals[0].probability == pytest.approx(0.8)
    assert first_replay.created_at == first_live.created_at
    assert second_replay.created_at == second_live.created_at
    with pytest.raises(ReplayMissError, match="sequence exhausted"):
        replay.evaluate(snapshot, "focused")


def test_replay_without_policy_checks_all_snapshot_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-only-key")
    policy = JevPolicy(mode="shadow")
    live = GatewayJudgmentProvider(policy, record_dir=tmp_path)
    monkeypatch.setattr(live, "_post_json", lambda *_: _response())
    snapshot = SemanticSnapshot(
        source_text="原文",
        draft_text="draft",
        glossary_text="魔王=Demon King",
        style_guide="terse",
        context="chapter 1",
    )
    live.evaluate(snapshot, "focused")

    replay = ReplayJudgmentProvider(tmp_path)
    assert replay.evaluate(snapshot, "focused").is_fresh(snapshot)
    with pytest.raises(ReplayMissError, match="No fresh"):
        replay.evaluate(snapshot.model_copy(update={"style_guide": "lyrical"}), "focused")


def test_config_digest_includes_questions_and_policy() -> None:
    focused = JevPolicy(question_set="focused")
    dense = JevPolicy(question_set="dense")
    assert semantic_config_digest(focused) != semantic_config_digest(dense)
    assert semantic_config_digest(focused) != semantic_config_digest(
        focused.model_copy(update={"window_chars": 5000})
    )


def test_over_budget_unaligned_pair_returns_no_probabilities_or_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    snapshot = SemanticSnapshot(source_text="甲" * 5000, draft_text="A" * 15_000)
    provider = GatewayJudgmentProvider(
        JevPolicy(mode="shadow", max_request_bytes=4096)
    )
    monkeypatch.setattr(
        provider,
        "_post_json",
        lambda *_: pytest.fail("unaligned over-budget input must not call the network"),
    )

    report = provider.evaluate(snapshot, "focused")

    assert report.status == "unavailable"
    assert report.signals == []
    assert report.requests == []
    assert report.coverage.alignment == "unavailable_unaligned_over_budget"
    assert report.coverage.source_fraction == 0
    assert report.coverage.source_judged_fraction == 0
    assert report.issues[0].code == "unaligned_over_budget"


def test_supporting_text_truncation_is_partial_and_rendered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-only-key")
    policy = JevPolicy(mode="shadow", supporting_text_chars=10)
    provider = GatewayJudgmentProvider(policy)
    monkeypatch.setattr(provider, "_post_json", lambda *_: _response())
    report = provider.evaluate(
        SemanticSnapshot(
            source_text="原文",
            draft_text="draft",
            glossary_text="g" * 11,
            context="c" * 11,
        ),
        "focused",
    )

    assert report.status == "partial"
    assert report.coverage.supporting_text_truncated == ["glossary_text", "context"]
    assert {issue.code for issue in report.issues} == {"supporting_text_truncated"}


def test_question_set_must_match_serialized_policy() -> None:
    provider = GatewayJudgmentProvider(JevPolicy(mode="off", question_set="focused"))
    with pytest.raises(ValueError, match="does not match policy"):
        provider.evaluate(SemanticSnapshot(source_text="原", draft_text="draft"), "dense")
