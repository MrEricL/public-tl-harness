"""Contract checks for the direct TypeSafe decision boundary."""

from __future__ import annotations

import json

import pytest

from agentic_translation.decisions import DecisionRequest, RecordedDecisionProvider, TypeSafeDecisionProvider
from agentic_translation.decisions.questions import question_schema


def _request(**overrides: object) -> DecisionRequest:
    fields = {"segment_id": "chapter-1:segment-1", "source_text": "他没有开门。", "draft_text": "He did not open the door.", "context_text": "The door is locked."}
    fields.update(overrides)
    return DecisionRequest(**fields)


def _response(request: DecisionRequest, **noul_values: float) -> dict:
    answers = {}
    for name, question in question_schema(request).items():
        if question["type"] == "noul":
            answers[name] = {"type": "noul", "noul": noul_values.get(name, 0.02)}
        elif question["type"] == "choice":
            answers[name] = {"type": "choice", "choice": "sufficient", "confidence": 0.9, "probabilities": {"sufficient": 0.9, "ambiguous": 0.05, "incomplete": 0.05}}
        else:
            answers[name] = {"type": "score", "score": 0.1, "confidence": 0.9, "legend": {str(i): level for i, level in enumerate(question["criteria"])}, "probabilities": {"0": 0.9, "1": 0.1, "2": 0.0, "3": 0.0}}
    return {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 100, "output_tokens": 25}}


class _Client:
    def __init__(self, responses: list[object]):
        self.responses = responses
        self.calls = 0

    def system_one(self, **payload: object) -> dict:
        self.calls += 1
        answer = self.responses.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_live_mapping_and_exact_replay_invalidation(tmp_path) -> None:
    request = _request()
    client = _Client([_response(request, negation_condition_error=0.78)])
    result = TypeSafeDecisionProvider(client=client, record_dir=tmp_path).evaluate(request)
    assert result.status == "ok"
    assert result.route_level == "material"
    assert result.material_flags == ["negation_condition_error"]
    assert result.receipt.requested_model == "jev-1.13.0"
    assert result.receipt.served_model == "jev-1.13.0"
    assert result.receipt.input_tokens == 100
    assert result.receipt.output_tokens == 25
    assert str(result.receipt.estimated_usd) == "0.0000042"
    assert result.receipt.cache_status == "recorded"
    saved = next(tmp_path.glob("*.json")).read_text(encoding="utf-8")
    assert request.source_text not in saved and request.draft_text not in saved
    replay = RecordedDecisionProvider(tmp_path)
    assert replay.evaluate(request).receipt.cache_status == "replayed"
    for changed in (_request(draft_text="He opened the door."), _request(context_text="A different neighbor."), _request(requested_model="jev-latest")):
        missed = replay.evaluate(changed)
        assert missed.status == "unavailable" and missed.route_level == "fallback"
        assert missed.receipt.cache_status == "miss"


def test_malformed_or_failed_responses_never_route_clean() -> None:
    request = _request()
    broken = _response(request)
    del broken["answers"]["omitted_material"]
    invalid = TypeSafeDecisionProvider(client=_Client([broken])).evaluate(request)
    assert invalid.status == "unavailable"
    assert invalid.route_level == "fallback"
    assert invalid.receipt.transport_status == "invalid_response"

    class Unauthorized(Exception):
        status = 401

    client = _Client([Unauthorized("secret response")])
    failed = TypeSafeDecisionProvider(client=client).evaluate(request)
    assert failed.status == "unavailable" and client.calls == 1
    assert "secret response" not in failed.model_dump_json()


def test_transient_retry_is_bounded_and_counted() -> None:
    request = _request()

    class RateLimited(Exception):
        status = 429

    client = _Client([RateLimited(), _response(request)])
    result = TypeSafeDecisionProvider(client=client, max_retries=1, backoff_seconds=0).evaluate(request)
    assert result.status == "ok" and result.receipt.attempt_count == 2
    assert client.calls == 2

    exhausted = _Client([RateLimited(), RateLimited()])
    result = TypeSafeDecisionProvider(client=exhausted, max_retries=1, backoff_seconds=0).evaluate(request)
    assert result.status == "unavailable" and result.receipt.attempt_count == 2


def test_patch_question_and_replay_corruption(tmp_path) -> None:
    request = _request(candidate_text="He opened the door.")
    client = _Client([_response(request, patch_unsupported_claim=0.92)])
    result = TypeSafeDecisionProvider(client=client, record_dir=tmp_path).evaluate(request)
    assert result.material_flags == ["patch_unsupported_claim"]
    path = next(tmp_path.glob("*.json"))
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["result"]["receipt"]["source_sha256"] = "0" * 64
    path.write_text(json.dumps(saved), encoding="utf-8")
    try:
        RecordedDecisionProvider(tmp_path).evaluate(request)
    except ValueError as exc:
        assert "corrupt decision replay receipt" in str(exc)
    else:
        raise AssertionError("corrupt replay was accepted")


def test_installed_sdk_serializes_questions_and_decodes_real_response_type() -> None:
    """Exercise SDK 0.6's msgspec response through a local HTTP transport."""
    httpx2 = pytest.importorskip("httpx2")
    sdk = pytest.importorskip("typesafe_sdk")
    request = _request()
    seen = []

    def handler(http_request):
        assert str(http_request.url) == "https://api.typesafe.ai/v1/systemone"
        payload = json.loads(http_request.content)
        seen.append(payload)
        return httpx2.Response(200, json=_response(request, omitted_material=0.66))

    with sdk.TypeSafeClient(
        api_key="dummy-test-key", transport=httpx2.MockTransport(handler),
        retry=sdk.RetryPolicy(max_retries=0),
    ) as client:
        result = TypeSafeDecisionProvider(client=client).evaluate(request)

    assert len(seen) == 1
    assert seen[0]["state"]["source"] == request.source_text
    assert seen[0]["questions"]["omitted_material"]["type"] == "noul"
    assert seen[0]["questions"]["readability_problem"]["type"] == "score"
    assert result.status == "ok"
    assert result.material_flags == ["omitted_material"]
    assert result.receipt.raw_answers["readability_problem"]["probabilities"] == {"0": 0.9, "1": 0.1, "2": 0.0, "3": 0.0}


def test_installed_sdk_402_is_recorded_without_retry_or_error_body() -> None:
    httpx2 = pytest.importorskip("httpx2")
    sdk = pytest.importorskip("typesafe_sdk")
    seen = []

    def handler(http_request):
        seen.append(http_request)
        return httpx2.Response(402, json={"error": {"message": "private account detail"}})

    with sdk.TypeSafeClient(
        api_key="dummy-test-key", transport=httpx2.MockTransport(handler),
        retry=sdk.RetryPolicy(max_retries=0),
    ) as client:
        result = TypeSafeDecisionProvider(client=client, max_retries=2, backoff_seconds=0).evaluate(_request())

    assert len(seen) == 1
    assert result.status == "unavailable" and result.route_level == "fallback"
    assert result.receipt.http_status == 402
    assert result.receipt.attempt_count == 1
    assert "private account detail" not in result.model_dump_json()


def test_pinned_model_mismatch_is_unavailable() -> None:
    request = _request()
    response = _response(request)
    response["model"] = "jev-1.14.0"
    result = TypeSafeDecisionProvider(client=_Client([response])).evaluate(request)
    assert result.status == "unavailable"
    assert result.receipt.transport_status == "invalid_response"
    assert result.route_level == "fallback"

    alias_request = _request(requested_model="jev-latest")
    alias_result = TypeSafeDecisionProvider(client=_Client([response])).evaluate(alias_request)
    assert alias_result.status == "ok"
    assert alias_result.receipt.requested_model == "jev-latest"
    assert alias_result.receipt.served_model == "jev-1.14.0"
