from __future__ import annotations

import json

import pytest

from agentic_translation.adaptive import run_adaptive_chapter
from agentic_translation.decisions.models import DecisionReceipt, DecisionResult, QUESTION_SCHEMA_VERSION
from agentic_translation.decisions.questions import question_schema_digest, request_id
from agentic_translation.glossary import parse_glossary_text
from agentic_translation.segmentation import DraftSegment, segment_source
from agentic_translation.unattended import _verification_policy, run_chapter_arms


def test_configured_jev_thresholds_reach_decision_receipt_and_checkpoint(tmp_path) -> None:
    config = {"jev": {"model": "jev-1.13.0", "low_risk_threshold": 0.27,
                       "repair_threshold": 0.61, "readability_threshold": 1.75}}
    policy = _verification_policy(config, "jev_adaptive")
    source = segment_source("w", "c1", "他守住山门。\n")
    draft = (DraftSegment.from_text(source[0].segment_id, "Chapter 1\n\nHe guarded the gate."),)

    class CleanDecision:
        requests = []

        def evaluate(self, request):
            self.requests.append(request)
            return DecisionResult(status="ok", receipt=DecisionReceipt(
                request_id=request_id(request), segment_id=request.segment_id,
                requested_model=request.requested_model, source_sha256=request.source_sha256,
                draft_sha256=request.draft_sha256, context_sha256=request.context_sha256,
                question_schema_version=QUESTION_SCHEMA_VERSION,
                question_schema_sha256=question_schema_digest(request),
                routing_policy_version=request.routing_policy_version,
                routing_thresholds=request.routing_thresholds,
                latency_ms=0, attempt_count=1, transport_status="ok", cache_status="recorded",
            ))

    provider = CleanDecision()
    result = run_adaptive_chapter(
        source, draft, glossary=parse_glossary_text(""), decision_provider=provider,
        review_provider=None, verification_policy=policy, session_dir=tmp_path / "session",
    )
    expected = {"diagnostic": 0.27, "material": 0.61, "readability": 1.75}
    assert result.outcome == "delivered"
    assert provider.requests[0].routing_thresholds == expected
    assert provider.requests[0].requested_model == "jev-1.13.0"
    checkpoint = json.loads((tmp_path / "session" / "adaptive_checkpoint.json").read_text())
    assert checkpoint["verification_policy"]["routing_thresholds"] == expected


def test_invalid_threshold_order_is_rejected_before_provider_calls(tmp_path) -> None:
    bad = {"jev": {"low_risk_threshold": 0.7, "repair_threshold": 0.5}}
    with pytest.raises(ValueError, match="thresholds"):
        _verification_policy(bad, "jev_adaptive")
    with pytest.raises(ValueError, match="thresholds"):
        run_chapter_arms(
            work_id="w", chapter_id="c1", source_text="source", segments=(), tape=None,
            run_dir=tmp_path, config=bad, memory_cost_usd=None,
            generator_factory=lambda *args, **kwargs: pytest.fail("provider should not start"),
        )
