"""Cache-only replay of source-bound successful decision receipts."""

from __future__ import annotations

import json
import hashlib
import os
import uuid
from pathlib import Path

from .models import QUESTION_SCHEMA_VERSION, DecisionReceipt, DecisionRequest, DecisionResult, digest
from .questions import question_schema_digest, request_id


def record_decision(record_dir: str | Path, result: DecisionResult) -> Path:
    """Write one source-free receipt atomically, keyed by exact request identity."""
    if result.status != "ok":
        raise ValueError("only successful decisions can be replayed")
    directory = Path(record_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{result.receipt.request_id}.json"
    saved = result.model_copy(deep=True)
    saved.receipt.cache_status = "recorded"
    temp = directory / f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        body = saved.model_dump(mode="json")
        temp.write_text(json.dumps({"result": body, "record_sha256": digest(body)}, indent=2), encoding="utf-8")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    return target


class RecordedDecisionProvider:
    def __init__(self, record_dir: str | Path) -> None:
        self.record_dir = Path(record_dir)

    def evaluate(self, request: DecisionRequest) -> DecisionResult:
        key = request_id(request)
        path = self.record_dir / f"{key}.json"
        if not path.is_file():
            return DecisionResult(status="unavailable", receipt=DecisionReceipt(
                request_id=key, segment_id=request.segment_id, requested_model=request.requested_model,
                source_sha256=request.source_sha256, draft_sha256=request.draft_sha256,
                context_sha256=request.context_sha256,
                candidate_sha256=hashlib.sha256(request.candidate_text.encode("utf-8")).hexdigest() if request.candidate_text is not None else None,
                question_schema_version=QUESTION_SCHEMA_VERSION,
                question_schema_sha256=question_schema_digest(request),
                routing_policy_version=request.routing_policy_version,
                routing_thresholds=request.routing_thresholds,
                latency_ms=0, attempt_count=0, transport_status="unavailable", cache_status="miss", error_code="replay_miss",
            ))
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            data = envelope["result"]
            if envelope["record_sha256"] != digest(data):
                raise ValueError("record checksum mismatch")
            result = DecisionResult.model_validate(data)
            if (result.status != "ok" or result.receipt.request_id != key or result.receipt.cache_status != "recorded"
                or result.receipt.source_sha256 != request.source_sha256
                or result.receipt.draft_sha256 != request.draft_sha256
                or result.receipt.context_sha256 != request.context_sha256
                or result.receipt.question_schema_sha256 != question_schema_digest(request)
                or result.receipt.requested_model != request.requested_model):
                raise ValueError("stale or invalid recorded decision")
            result.receipt.cache_status = "replayed"
            return result
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"corrupt decision replay receipt: {path.name}") from exc
