"""Direct TypeSafe System One adapter with explicit bounded retries.

The SDK's own retries are disabled, so attempt_count is observable and the
outer retry bound is the only one. Credentials are read from the environment
and never serialized into receipts.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from .models import DecisionReceipt, DecisionRequest, DecisionResult, QUESTION_SCHEMA_VERSION
from .questions import QUESTIONS, question_schema, question_schema_digest, request_id, request_payload


ERROR_QUESTIONS = tuple(name for name, question in QUESTIONS.items() if question["type"] == "noul")
INPUT_USD_PER_MILLION = Decimal("0.042")  # TypeSafe Jev 1.13 list rate, checked 2026-09-22.


class _MissingKeyError(RuntimeError):
    pass


def _as_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "__struct_fields__"):
        # typesafe-sdk 0.6 uses msgspec.Struct, not Pydantic response models.
        import msgspec

        return msgspec.to_builtins(response)
    raise ValueError("response is not a TypeSafe object or mapping")


def _valid_probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value <= 1:
        raise ValueError("invalid probability")
    return float(value)


def _validated_answers(request: DecisionRequest, response: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected = question_schema(request)
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(expected):
        raise ValueError("missing or unexpected answer keys")
    checked: dict[str, dict[str, Any]] = {}
    for key, question in expected.items():
        answer = answers[key]
        if not isinstance(answer, dict) or answer.get("type") != question["type"]:
            raise ValueError(f"wrong answer type: {key}")
        if question["type"] == "noul":
            checked[key] = {"type": "noul", "noul": _valid_probability(answer.get("noul"))}
        elif question["type"] == "choice":
            options = set(question["criteria"])
            choice = answer.get("choice")
            probabilities = answer.get("probabilities")
            if choice not in options or not isinstance(probabilities, dict) or set(probabilities) != options:
                raise ValueError(f"invalid choice: {key}")
            checked[key] = {"type": "choice", "choice": choice, "confidence": _valid_probability(answer.get("confidence")), "probabilities": {label: _valid_probability(probabilities[label]) for label in options}}
        else:
            score = answer.get("score")
            probabilities = answer.get("probabilities")
            levels = {str(index) for index in range(len(question["criteria"]))}
            if isinstance(score, bool) or not isinstance(score, (float, int)) or not 0 <= score <= len(levels) - 1:
                raise ValueError(f"invalid score: {key}")
            if not isinstance(probabilities, dict):
                raise ValueError(f"invalid score probabilities: {key}")
            # The SDK's typed ScoreAnswer uses integer keys; raw JSON uses strings.
            probabilities = {str(level): value for level, value in probabilities.items()}
            if set(probabilities) != levels:
                raise ValueError(f"invalid score probabilities: {key}")
            checked[key] = {"type": "score", "score": float(score), "confidence": _valid_probability(answer.get("confidence")), "probabilities": {level: _valid_probability(probabilities[level]) for level in levels}, "legend": answer.get("legend")}
    return checked


def _derived_flags(request: DecisionRequest, answers: dict[str, dict[str, Any]]) -> dict[str, bool]:
    material = request.routing_thresholds["material"]
    diagnostic = request.routing_thresholds["diagnostic"]
    flags = {name: answers[name]["noul"] >= material for name in ERROR_QUESTIONS}
    if request.candidate_text is not None:
        flags["patch_unsupported_claim"] = answers["patch_unsupported_claim"]["noul"] >= material
    flags["diagnostic_error"] = any(diagnostic <= answers[name]["noul"] < material for name in ERROR_QUESTIONS)
    flags["severe_readability"] = answers["readability_problem"]["score"] >= request.routing_thresholds["readability"]
    flags["source_incomplete"] = answers["source_sufficiency"]["choice"] == "incomplete"
    flags["source_ambiguous"] = answers["source_sufficiency"]["choice"] == "ambiguous"
    return flags


def _transient(exc: Exception) -> bool:
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status in {408, 429} or 500 <= status <= 599
    return isinstance(exc, (ConnectionError, TimeoutError))


def _http_status(exc: Exception) -> int | None:
    status = getattr(exc, "status", None)
    return status if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599 else None


def _pinned_model_mismatch(requested: str, served: str) -> bool:
    """Aliases may resolve; explicit Jev version IDs must stay pinned."""
    return bool(re.fullmatch(r"jev-\d+(?:\.\d+)+", requested)) and requested != served


class TypeSafeDecisionProvider:
    """Evaluate one passage with TypeSafe and optionally record a replay receipt."""

    def __init__(
        self, *, record_dir: str | Path | None = None, client: Any | None = None,
        max_retries: int = 2, timeout_seconds: float = 30.0, backoff_seconds: float = 0.5,
    ) -> None:
        if not 0 <= max_retries <= 5 or timeout_seconds <= 0 or backoff_seconds < 0:
            raise ValueError("invalid TypeSafe retry/timeout settings")
        self.record_dir = Path(record_dir) if record_dir is not None else None
        self.client = client
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.backoff_seconds = backoff_seconds

    def _base_receipt(self, request: DecisionRequest) -> dict[str, Any]:
        return {
            "request_id": request_id(request), "segment_id": request.segment_id,
            "requested_model": request.requested_model, "source_sha256": request.source_sha256,
            "draft_sha256": request.draft_sha256, "context_sha256": request.context_sha256,
            "candidate_sha256": hashlib.sha256(request.candidate_text.encode("utf-8")).hexdigest() if request.candidate_text is not None else None,
            "question_schema_version": QUESTION_SCHEMA_VERSION,
            "question_schema_sha256": question_schema_digest(request),
            "routing_policy_version": request.routing_policy_version,
            "routing_thresholds": request.routing_thresholds,
        }

    def evaluate(self, request: DecisionRequest) -> DecisionResult:
        started = time.perf_counter()
        base = self._base_receipt(request)
        owned_client = None
        try:
            if self.client is None:
                if not os.environ.get("TYPESAFE_API_KEY", "").strip():
                    raise _MissingKeyError
                from typesafe_sdk import RetryPolicy, TypeSafeClient

                owned_client = TypeSafeClient(retry=RetryPolicy(max_retries=0), timeout=self.timeout_seconds)
            client = self.client or owned_client
            assert client is not None
            for attempt in range(1, self.max_retries + 2):
                try:
                    payload = request_payload(request)
                    response = _as_dict(client.system_one(**payload))
                    answers = _validated_answers(request, response)
                    served_model = response.get("model")
                    if not isinstance(served_model, str) or not served_model:
                        raise ValueError("response missing served model")
                    if _pinned_model_mismatch(request.requested_model, served_model):
                        raise ValueError("served model differs from pinned requested model")
                    usage = response.get("usage")
                    if not isinstance(usage, dict):
                        raise ValueError("response missing usage")
                    input_tokens = usage.get("input_tokens")
                    output_tokens = usage.get("output_tokens")
                    for value in (input_tokens, output_tokens):
                        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                            raise ValueError("invalid token usage")
                    receipt = DecisionReceipt(
                        **base, served_model=served_model, raw_answers=answers,
                        derived_flags=_derived_flags(request, answers), input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        estimated_usd=(Decimal(input_tokens) * INPUT_USD_PER_MILLION / Decimal(1_000_000)) if input_tokens is not None and served_model == "jev-1.13.0" else None,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        attempt_count=attempt, transport_status="ok", cache_status="live",
                    )
                    result = DecisionResult(status="ok", receipt=receipt)
                    if self.record_dir is not None:
                        from .replay import record_decision

                        record_decision(self.record_dir, result)
                        result.receipt.cache_status = "recorded"
                    return result
                except Exception as exc:  # SDK errors have version-dependent subclasses.
                    if _transient(exc) and attempt <= self.max_retries:
                        time.sleep(min(self.backoff_seconds * 2 ** (attempt - 1), 2.0))
                        continue
                    status = "invalid_response" if isinstance(exc, ValueError) else "unavailable"
                    return DecisionResult(status="unavailable", receipt=DecisionReceipt(
                        **base, latency_ms=(time.perf_counter() - started) * 1000,
                        attempt_count=attempt, transport_status=status, cache_status="live",
                        http_status=_http_status(exc),
                        error_code=type(exc).__name__,
                    ))
        except Exception as exc:
            return DecisionResult(status="unavailable", receipt=DecisionReceipt(
                **base, latency_ms=(time.perf_counter() - started) * 1000,
                attempt_count=0, transport_status="unavailable", cache_status="live",
                http_status=_http_status(exc),
                error_code="missing_typesafe_api_key" if isinstance(exc, _MissingKeyError) else type(exc).__name__,
            ))
        finally:
            if owned_client is not None:
                owned_client.close()
