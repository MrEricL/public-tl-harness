"""Native HTTP and replay providers for Vercel AI Gateway evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timezone
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import ValidationError

from .semantic_models import (
    CoverageSummary,
    GatewayRouting,
    JevPolicy,
    QuestionSetName,
    SemanticIssue,
    SemanticRequestRecord,
    SemanticSignal,
    SemanticSignalReport,
    SemanticSnapshot,
)
from .semantic_signals import (
    QUESTION_VERSION,
    RENDER_VERSION,
    WINDOWING_VERSION,
    SemanticWindow,
    build_windows,
    categories_for_set,
    question_definitions,
)


class JudgmentProviderUnavailable(RuntimeError):
    """Raised when a configured live provider cannot be started safely."""


class ReplayMissError(RuntimeError):
    """Raised when cache-only replay has no unique fresh matching record."""


@runtime_checkable
class JudgmentProvider(Protocol):
    def evaluate(
        self,
        snapshot: SemanticSnapshot,
        question_set: QuestionSetName,
    ) -> SemanticSignalReport: ...


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def semantic_config_digest(
    policy: JevPolicy,
    question_set: QuestionSetName | None = None,
) -> str:
    effective = question_set or policy.question_set
    payload = {
        "policy": policy.model_dump(mode="json"),
        "question_version": QUESTION_VERSION,
        "windowing_version": WINDOWING_VERSION,
        "render_version": RENDER_VERSION,
        "questions": question_definitions(effective),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def semantic_snapshot_digest(snapshot: SemanticSnapshot) -> str:
    payload = {
        "source_text": snapshot.source_text,
        "draft_text": snapshot.draft_text,
        "glossary_text": snapshot.glossary_text,
        "style_guide": snapshot.style_guide,
        "context": snapshot.context,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _cache_key(
    snapshot: SemanticSnapshot,
    policy: JevPolicy,
    question_set: QuestionSetName,
) -> str:
    payload = {
        "snapshot_digest": semantic_snapshot_digest(snapshot),
        "config_digest": semantic_config_digest(policy, question_set),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _coverage_for(snapshot: SemanticSnapshot, policy: JevPolicy) -> CoverageSummary:
    coverage = build_windows(
        snapshot,
        window_chars=policy.window_chars,
        overlap=policy.window_overlap,
        max_windows=policy.max_windows,
        supporting_text_chars=policy.supporting_text_chars,
        max_request_bytes=policy.max_request_bytes,
    )[1]
    truncated = [
        field
        for field in ("glossary_text", "style_guide", "context")
        if len(getattr(snapshot, field)) > policy.supporting_text_chars
    ]
    return coverage.model_copy(update={"supporting_text_truncated": truncated})


def unavailable_report(
    snapshot: SemanticSnapshot,
    policy: JevPolicy,
    question_set: QuestionSetName,
    issue: str,
    *,
    code: str = "unavailable",
) -> SemanticSignalReport:
    return SemanticSignalReport(
        source_sha256=snapshot.source_sha256,
        draft_sha256=snapshot.draft_sha256,
        question_set=question_set,
        question_version=QUESTION_VERSION,
        windowing_version=WINDOWING_VERSION,
        render_version=RENDER_VERSION,
        requested_model=policy.model,
        snapshot_digest=semantic_snapshot_digest(snapshot),
        policy_digest=semantic_config_digest(policy, question_set),
        cache_key=_cache_key(snapshot, policy, question_set),
        advisory_threshold=policy.advisory_threshold,
        status="unavailable",
        coverage=_coverage_for(snapshot, policy),
        issues=[SemanticIssue(code=code, message=issue)],
    )


def _redact(message: str, secret: str | None = None) -> str:
    return _redact_unbounded(message, secret)[:1000]


def _redact_unbounded(message: str, secret: str | None = None) -> str:
    redacted = message.replace(secret, "[REDACTED]") if secret else message
    redacted = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", redacted)
    redacted = re.sub(r"\bvck_[A-Za-z0-9_-]+", "[REDACTED]", redacted)
    return redacted


def _sanitize_json(value: Any, secret: str | None = None) -> Any:
    """Recursively remove credential-shaped values from receipt content."""

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _redact_unbounded(value, secret)
    if isinstance(value, list):
        return [_sanitize_json(item, secret) for item in value]
    if isinstance(value, dict):
        return {
            _redact_unbounded(str(key), secret): _sanitize_json(item, secret)
            for key, item in value.items()
        }
    return _redact_unbounded(repr(value), secret)


def _attempt_receipt(
    *,
    payload: dict[str, Any],
    window_index: int,
    attempt: int,
    status: str,
    latency_ms: float,
    secret: str,
    response: dict[str, Any] | None = None,
    http_status: int | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    request_payload = _sanitize_json(payload, secret)
    response_payload = _sanitize_json(response, secret) if response is not None else None
    return {
        "schema_version": "semantic-gateway-receipt.v1",
        "endpoint": "https://ai-gateway.vercel.sh/v1/evaluate",
        "method": "POST",
        "window_index": window_index,
        "attempt": attempt,
        "status": status,
        "latency_ms": latency_ms,
        "request_payload": request_payload,
        "request_sha256": hashlib.sha256(
            _canonical_json(request_payload).encode("utf-8")
        ).hexdigest(),
        "response": response_payload,
        "response_sha256": (
            hashlib.sha256(_canonical_json(response_payload).encode("utf-8")).hexdigest()
            if response_payload is not None
            else None
        ),
        "http_status": http_status,
        "error_code": error_code,
        "error_message": _redact(error_message, secret) if error_message else None,
    }


def _receipt_sha256(receipt: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(receipt).encode("utf-8")).hexdigest()


def _optional_nonnegative_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() and parsed >= 0 else None


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _retry_after_seconds(headers: Any, cap: float) -> float:
    if headers is None:
        return 0.0
    raw = headers.get("Retry-After")
    if raw is None:
        return 0.0
    try:
        return min(max(float(raw), 0.0), cap)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(raw))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return min(max(retry_at.timestamp() - time.time(), 0.0), cap)
        except (TypeError, ValueError, OverflowError):
            return 0.0


@dataclass(frozen=True)
class _WindowResult:
    window_index: int
    signals: list[SemanticSignal]
    record: SemanticRequestRecord
    issues: list[SemanticIssue]
    resolved_model: str | None
    receipts: list[dict[str, Any]]


class GatewayJudgmentProvider:
    """Evaluate bounded text windows through Vercel's native `/v1/evaluate`."""

    def __init__(self, policy: JevPolicy, record_dir: str | Path | None = None) -> None:
        self.policy = policy
        self.record_dir = Path(record_dir) if record_dir is not None else None

    def _post_json(self, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        request = urllib.request.Request(
            self.policy.endpoint,
            data=_canonical_json(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.policy.timeout_seconds) as response:
            raw = response.read()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Gateway response must be a JSON object")
        return value

    def _window_state(self, snapshot: SemanticSnapshot, window: SemanticWindow) -> dict[str, str]:
        cap = self.policy.supporting_text_chars
        return {
            "source_window": window.source_span.text,
            "draft_window": window.draft_span.text,
            "glossary": snapshot.glossary_text[:cap],
            "style_guide": snapshot.style_guide[:cap],
            "context": snapshot.context[:cap],
        }

    def _evaluate_window(
        self,
        snapshot: SemanticSnapshot,
        window: SemanticWindow,
        question_set: QuestionSetName,
        api_key: str,
    ) -> _WindowResult:
        payload = {
            "model": self.policy.model,
            "state": self._window_state(snapshot, window),
            "questions": question_definitions(question_set),
            "providerOptions": {
                "gateway": {"only": ["typesafe-ai"]},
            },
        }
        started = time.monotonic()
        attempts = 0
        response: dict[str, Any] | None = None
        error_code: str | None = None
        error_message: str | None = None
        receipts: list[dict[str, Any]] = []
        while attempts <= self.policy.max_retries:
            attempts += 1
            attempt_started = time.monotonic()
            try:
                response = self._post_json(payload, api_key)
                receipts.append(
                    _attempt_receipt(
                        payload=payload,
                        window_index=window.index,
                        attempt=attempts,
                        status="completed",
                        latency_ms=(time.monotonic() - attempt_started) * 1000,
                        secret=api_key,
                        response=response,
                        http_status=200,
                    )
                )
                break
            except urllib.error.HTTPError as exc:
                error_code = f"http_{exc.code}"
                body = exc.read(2048).decode("utf-8", errors="replace")
                error_message = _redact(f"Gateway HTTP {exc.code}: {body}", api_key)
                try:
                    parsed_body = json.loads(body)
                    parsed_response = parsed_body if isinstance(parsed_body, dict) else {"body": parsed_body}
                except json.JSONDecodeError:
                    parsed_response = {"body": body}
                receipts.append(
                    _attempt_receipt(
                        payload=payload,
                        window_index=window.index,
                        attempt=attempts,
                        status="http_error",
                        latency_ms=(time.monotonic() - attempt_started) * 1000,
                        secret=api_key,
                        response=parsed_response,
                        http_status=exc.code,
                        error_code=error_code,
                        error_message=error_message,
                    )
                )
                transient = exc.code == 429 or 500 <= exc.code < 600
                if not transient or attempts > self.policy.max_retries:
                    break
                delay = _retry_after_seconds(exc.headers, self.policy.retry_after_cap_seconds)
                if delay:
                    time.sleep(delay)
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                error_code = "transport_error"
                error_message = _redact(f"Gateway transport error: {exc}", api_key)
                receipts.append(
                    _attempt_receipt(
                        payload=payload,
                        window_index=window.index,
                        attempt=attempts,
                        status="transport_error",
                        latency_ms=(time.monotonic() - attempt_started) * 1000,
                        secret=api_key,
                        error_code=error_code,
                        error_message=error_message,
                    )
                )
                if attempts > self.policy.max_retries:
                    break
            except (json.JSONDecodeError, ValueError) as exc:
                error_code = "malformed_response"
                error_message = _redact(f"Malformed Gateway response: {exc}", api_key)
                receipts.append(
                    _attempt_receipt(
                        payload=payload,
                        window_index=window.index,
                        attempt=attempts,
                        status="malformed_response",
                        latency_ms=(time.monotonic() - attempt_started) * 1000,
                        secret=api_key,
                        error_code=error_code,
                        error_message=error_message,
                    )
                )
                break

        latency_ms = (time.monotonic() - started) * 1000
        if response is None:
            issue = SemanticIssue(
                code=error_code or "unavailable",
                message=error_message or "Gateway request failed",
                window_index=window.index,
            )
            record = SemanticRequestRecord(
                window_index=window.index,
                status="unavailable",
                attempts=attempts,
                retry_count=max(0, attempts - 1),
                accounting_complete=False,
                latency_ms=latency_ms,
                error_code=issue.code,
                error_message=issue.message,
                receipt_sha256s=[_receipt_sha256(receipt) for receipt in receipts],
            )
            return _WindowResult(window.index, [], record, [issue], None, receipts)

        answers = response.get("answers")
        issues: list[SemanticIssue] = []
        signals: list[SemanticSignal] = []
        if not isinstance(answers, dict):
            issues.append(
                SemanticIssue(
                    code="malformed_answers",
                    message="Gateway response answers must be an object",
                    window_index=window.index,
                )
            )
            answers = {}
        for category in categories_for_set(question_set):
            answer = answers.get(category)
            if not isinstance(answer, dict):
                issues.append(
                    SemanticIssue(
                        code="missing_answer",
                        message="Gateway response omitted a required answer",
                        window_index=window.index,
                        question_id=category,
                    )
                )
                continue
            probability = answer.get("probability")
            if (
                answer.get("type") != "boolean"
                or isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not 0 <= probability <= 1
            ):
                issues.append(
                    SemanticIssue(
                        code="malformed_answer",
                        message="Boolean answer requires probability in [0, 1]",
                        window_index=window.index,
                        question_id=category,
                    )
                )
                continue
            signals.append(
                SemanticSignal(
                    category=category,
                    probability=float(probability),
                    impact_domain="style" if category == "register_style" else "fidelity",
                    source_span=window.source_span,
                    draft_span=window.draft_span,
                    window_index=window.index,
                    localization=(
                        "full_pair"
                        if window.source_span.start == 0
                        and window.source_span.end == len(snapshot.source_text)
                        and window.draft_span.start == 0
                        and window.draft_span.end == len(snapshot.draft_text)
                        else "heuristic_unaligned"
                    ),
                )
            )

        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        provider_metadata = (
            response.get("providerMetadata")
            if isinstance(response.get("providerMetadata"), dict)
            else {}
        )
        gateway = (
            provider_metadata.get("gateway")
            if isinstance(provider_metadata.get("gateway"), dict)
            else {}
        )
        routing_raw = gateway.get("routing") if isinstance(gateway.get("routing"), dict) else {}
        routing = GatewayRouting(
            original_model_id=routing_raw.get("originalModelId")
            if isinstance(routing_raw.get("originalModelId"), str)
            else None,
            resolved_provider=routing_raw.get("resolvedProvider")
            if isinstance(routing_raw.get("resolvedProvider"), str)
            else None,
            canonical_slug=routing_raw.get("canonicalSlug")
            if isinstance(routing_raw.get("canonicalSlug"), str)
            else None,
            final_provider=routing_raw.get("finalProvider")
            if isinstance(routing_raw.get("finalProvider"), str)
            else None,
            generation_id=gateway.get("generationId")
            if isinstance(gateway.get("generationId"), str)
            else None,
        )
        expected = len(categories_for_set(question_set))
        record_status = "completed" if len(signals) == expected else "partial"
        last_input_tokens = _optional_nonnegative_int(
            usage.get("inputTokens", usage.get("input_tokens"))
        )
        last_output_tokens = _optional_nonnegative_int(
            usage.get("outputTokens", usage.get("output_tokens"))
        )
        last_cost = _optional_nonnegative_decimal(gateway.get("cost"))
        last_market_cost = _optional_nonnegative_decimal(gateway.get("marketCost"))
        last_surcharge_cost = _optional_nonnegative_decimal(gateway.get("surchargeCost"))
        last_gateway_cost = _optional_nonnegative_decimal(gateway.get("gatewayCost"))
        accounting_complete = attempts == 1
        record = SemanticRequestRecord(
            window_index=window.index,
            status=record_status,
            attempts=attempts,
            retry_count=max(0, attempts - 1),
            accounting_complete=accounting_complete,
            latency_ms=latency_ms,
            input_tokens=last_input_tokens if accounting_complete else None,
            output_tokens=last_output_tokens if accounting_complete else None,
            cost_usd=last_cost if accounting_complete else None,
            market_cost_usd=last_market_cost if accounting_complete else None,
            surcharge_cost_usd=last_surcharge_cost if accounting_complete else None,
            gateway_cost_usd=last_gateway_cost if accounting_complete else None,
            last_response_input_tokens=last_input_tokens,
            last_response_output_tokens=last_output_tokens,
            last_response_cost_usd=last_cost,
            last_response_market_cost_usd=last_market_cost,
            last_response_surcharge_cost_usd=last_surcharge_cost,
            last_response_gateway_cost_usd=last_gateway_cost,
            routing=routing,
            receipt_sha256s=[_receipt_sha256(receipt) for receipt in receipts],
        )
        resolved_model = response.get("model") if isinstance(response.get("model"), str) else None
        return _WindowResult(window.index, signals, record, issues, resolved_model, receipts)

    def evaluate(
        self,
        snapshot: SemanticSnapshot,
        question_set: QuestionSetName,
    ) -> SemanticSignalReport:
        if question_set != self.policy.question_set:
            raise ValueError(
                f"question_set {question_set!r} does not match policy {self.policy.question_set!r}"
            )
        if self.policy.mode == "off":
            return unavailable_report(
                snapshot,
                self.policy,
                question_set,
                "Jev policy is off; no request was made",
                code="disabled",
            )
        windows, coverage = build_windows(
            snapshot,
            window_chars=self.policy.window_chars,
            overlap=self.policy.window_overlap,
            max_windows=self.policy.max_windows,
            supporting_text_chars=self.policy.supporting_text_chars,
            max_request_bytes=self.policy.max_request_bytes,
        )
        coverage = _coverage_for(snapshot, self.policy)
        if not windows:
            report = unavailable_report(
                snapshot,
                self.policy,
                question_set,
                "Complete source/draft pair exceeds the dense request byte budget; "
                "no verified alignment is available, so Jev defect questions were not run",
                code="unaligned_over_budget",
            )
            self._record(report)
            return report
        api_key = os.environ.get("AI_GATEWAY_API_KEY")
        if not api_key:
            raise JudgmentProviderUnavailable(
                "AI_GATEWAY_API_KEY is required when Jev policy mode is shadow or advisory"
            )
        with ThreadPoolExecutor(max_workers=min(self.policy.max_concurrency, len(windows))) as pool:
            futures = [
                pool.submit(self._evaluate_window, snapshot, window, question_set, api_key)
                for window in windows
            ]
            results = [future.result() for future in futures]
        results.sort(key=lambda item: item.window_index)
        signals = [signal for result in results for signal in result.signals]
        requests = [result.record for result in results]
        issues = [issue for result in results for issue in result.issues]
        if any(result.signals for result in results):
            coverage = coverage.model_copy(
                update={
                    "source_judged_codepoints": coverage.source_covered_codepoints,
                    "draft_judged_codepoints": coverage.draft_covered_codepoints,
                }
            )
        if coverage.supporting_text_truncated:
            issues.append(
                SemanticIssue(
                    code="supporting_text_truncated",
                    message=(
                        "Supporting inputs were truncated by policy: "
                        + ", ".join(coverage.supporting_text_truncated)
                    ),
                )
            )
        completed = sum(record.status == "completed" for record in requests)
        if completed == len(requests) and not issues:
            status = "completed"
        elif signals:
            status = "partial"
        else:
            status = "unavailable"
        resolved_models = {result.resolved_model for result in results if result.resolved_model}
        resolved_model = next(iter(resolved_models)) if len(resolved_models) == 1 else None
        if len(resolved_models) > 1:
            issues.append(
                SemanticIssue(
                    code="mixed_resolved_models",
                    message="Gateway returned multiple resolved model identifiers",
                )
            )
            status = "partial"
        report = SemanticSignalReport(
            source_sha256=snapshot.source_sha256,
            draft_sha256=snapshot.draft_sha256,
            question_set=question_set,
            question_version=QUESTION_VERSION,
            windowing_version=WINDOWING_VERSION,
            render_version=RENDER_VERSION,
            requested_model=self.policy.model,
            resolved_model=resolved_model,
            snapshot_digest=semantic_snapshot_digest(snapshot),
            policy_digest=semantic_config_digest(self.policy, question_set),
            cache_key=_cache_key(snapshot, self.policy, question_set),
            advisory_threshold=self.policy.advisory_threshold,
            status=status,
            coverage=coverage,
            signals=signals,
            requests=requests,
            issues=issues,
        )
        self._record(report, [receipt for result in results for receipt in result.receipts])
        return report

    def _record(
        self,
        report: SemanticSignalReport,
        receipts: list[dict[str, Any]] | None = None,
    ) -> None:
        if self.record_dir is None:
            return
        self.record_dir.mkdir(parents=True, exist_ok=True)
        serialized = report.model_dump_json(indent=2)
        # Repeated identical inputs are independent physical judgments.  Keep
        # every invocation immutable and ordered rather than overwriting a
        # content-addressed "latest" record.
        for sequence in range(1_000_000):
            target = self.record_dir / f"{report.cache_key}.{sequence:06d}.json"
            try:
                with target.open("x", encoding="utf-8") as handle:
                    handle.write(serialized)
                receipt_dir = self.record_dir / "receipts"
                for receipt in receipts or []:
                    receipt_dir.mkdir(parents=True, exist_ok=True)
                    receipt_target = receipt_dir / (
                        f"{report.cache_key}.{sequence:06d}."
                        f"w{receipt['window_index']:04d}.a{receipt['attempt']:04d}.json"
                    )
                    with receipt_target.open("x", encoding="utf-8") as receipt_handle:
                        receipt_handle.write(_canonical_json(receipt))
                return
            except FileExistsError:
                continue
        raise RuntimeError("Semantic record sequence exhausted")


class ReplayJudgmentProvider:
    """Cache-only semantic provider.  It never owns a network transport."""

    def __init__(self, record_dir: str | Path, policy: JevPolicy | None = None) -> None:
        self.record_dir = Path(record_dir)
        self.policy = policy
        self._cursors: dict[str, int] = {}

    def evaluate(
        self,
        snapshot: SemanticSnapshot,
        question_set: QuestionSetName,
    ) -> SemanticSignalReport:
        if self.policy is not None:
            if question_set != self.policy.question_set:
                raise ReplayMissError("Replay question set does not match the supplied policy")
            cache_key = _cache_key(snapshot, self.policy, question_set)
            legacy = self.record_dir / f"{cache_key}.json"
            candidates = ([legacy] if legacy.exists() else []) + sorted(
                self.record_dir.glob(f"{cache_key}.[0-9][0-9][0-9][0-9][0-9][0-9].json")
            )
        else:
            candidates = sorted(self.record_dir.glob("*.json")) if self.record_dir.exists() else []
        matches: list[SemanticSignalReport] = []
        for path in candidates:
            if not path.is_file():
                continue
            try:
                report = SemanticSignalReport.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValidationError, ValueError):
                continue
            if (
                report.question_set == question_set
                and report.question_version == QUESTION_VERSION
                and report.windowing_version == WINDOWING_VERSION
                and report.render_version == RENDER_VERSION
                and report.is_fresh(snapshot)
                and (
                    report.snapshot_digest == semantic_snapshot_digest(snapshot)
                    or (
                        report.snapshot_digest is None
                        and not snapshot.glossary_text
                        and not snapshot.style_guide
                        and not snapshot.context
                    )
                )
            ):
                if self.policy is None or report.policy_digest == semantic_config_digest(
                    self.policy, question_set
                ):
                    matches.append(report)
        if not matches:
            raise ReplayMissError("No fresh semantic signal report exists for this snapshot and policy")
        matching_keys = {report.cache_key for report in matches}
        if len(matching_keys) > 1:
            raise ReplayMissError(
                "Multiple semantic signal configurations match; supply the original JevPolicy"
            )
        cache_key = next(iter(matching_keys))
        cursor = self._cursors.get(cache_key, 0)
        if cursor >= len(matches):
            raise ReplayMissError(
                "Semantic replay sequence exhausted for this snapshot and policy"
            )
        self._cursors[cache_key] = cursor + 1
        return matches[cursor]
