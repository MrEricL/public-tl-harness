from __future__ import annotations

from agentic_translation.adaptive import (
    AdaptiveReviewFinding,
    AdaptiveReviewResult,
    BudgetPolicy,
    VerificationPolicy,
    _current,
    run_adaptive_chapter,
)
from agentic_translation.agent_models import SegmentTextEdit, SubmitSegmentPatchAction
from agentic_translation.agent_repair import RepairToolExecutor
from agentic_translation.decisions.models import DecisionReceipt, DecisionResult, DecisionRequest, QUESTION_SCHEMA_VERSION
from agentic_translation.decisions.questions import question_schema_digest, request_id
from agentic_translation.glossary import parse_glossary_text
from agentic_translation.segmentation import DraftSegment, segment_source, sha256_text


SOURCE = "他守住山门。\n她走进院子。\n"
SOURCES = segment_source("w", "c1", SOURCE, target_chars=8, max_chars=12)
DRAFTS = (
    DraftSegment.from_text(SOURCES[0].segment_id, "Chapter 1\n\nShe guarded the gate."),
    DraftSegment.from_text(SOURCES[1].segment_id, "She entered the courtyard."),
)
GLOSSARY = parse_glossary_text("")


def _decision(request: DecisionRequest, *, flags=None, status="ok") -> DecisionResult:
    flags = flags or {}
    return DecisionResult(
        status=status,
        receipt=DecisionReceipt(
            request_id=request_id(request),
            segment_id=request.segment_id,
            requested_model=request.requested_model,
            source_sha256=request.source_sha256,
            draft_sha256=request.draft_sha256,
            context_sha256=request.context_sha256,
            candidate_sha256=sha256_text(request.candidate_text) if request.candidate_text else None,
            question_schema_version=QUESTION_SCHEMA_VERSION, question_schema_sha256=question_schema_digest(request),
            derived_flags=flags,
            routing_policy_version=request.routing_policy_version,
            routing_thresholds=request.routing_thresholds,
            latency_ms=1, attempt_count=1 if status == "ok" else 0,
            transport_status="ok" if status == "ok" else "unavailable",
            cache_status="recorded",
        ),
    )


class Screening:
    def __init__(self, *, unavailable=False):
        self.requests = []
        self.unavailable = unavailable

    def evaluate(self, request):
        self.requests.append(request)
        if self.unavailable:
            return _decision(request, status="unavailable")
        flags = {}
        if request.candidate_text is None and request.segment_id == SOURCES[0].segment_id and "She guarded" in request.draft_text:
            flags = {"actor_relation_error": True}
        return _decision(request, flags=flags)


class Reviewer:
    def __init__(self, *, patch=True):
        self.requests = []
        self.patch = patch

    def review(self, request):
        self.requests.append(request)
        if not self.patch or "She guarded" not in request.draft_text:
            return AdaptiveReviewResult(status="completed", summary="No edit needed")
        return AdaptiveReviewResult(
            status="completed", summary="Fix the actor",
            findings=(AdaptiveReviewFinding("actor_relation_error", "material", "Wrong actor"),),
            patches=(SubmitSegmentPatchAction(
                segment_id=request.segment_id,
                expected_segment_sha256=sha256_text(request.draft_text),
                edits=[SegmentTextEdit(old_text="She guarded", new_text="He guarded")],
                issue_ids=["actor_relation_error"], rationale="The source actor is male.",
            ),),
        )


def test_adaptive_repairs_flag_and_rescreens_final_segments() -> None:
    screen = Screening()
    reviewer = Reviewer()
    result = run_adaptive_chapter(SOURCES, DRAFTS, glossary=GLOSSARY,
                                  decision_provider=screen, review_provider=reviewer)
    assert result.outcome == "delivered"
    assert result.final_text == "Chapter 1\n\nHe guarded the gate.\n\nShe entered the courtyard."
    assert result.patch_actions == 1
    assert len([request for request in screen.requests if request.segment_id == SOURCES[0].segment_id and request.candidate_text is None]) == 2
    assert len([request for request in screen.requests if request.segment_id == SOURCES[1].segment_id]) == 2
    assert len([receipt for receipt in result.evidence if receipt.level == "jev_screen"]) == 2
    assert len(reviewer.requests) == 1


def test_clean_adaptive_chapter_does_not_call_review() -> None:
    drafts = (DRAFTS[0].with_text("Chapter 1\n\nHe guarded the gate."), DRAFTS[1])
    screen = Screening()
    reviewer = Reviewer()
    result = run_adaptive_chapter(SOURCES, drafts, glossary=GLOSSARY,
                                  decision_provider=screen, review_provider=reviewer)
    assert result.outcome == "delivered"
    assert result.screen_calls == 2
    assert result.review_calls == 0


def test_policy_pins_requested_model_and_context_change_invalidates_evidence() -> None:
    screen = Screening()
    policy = VerificationPolicy(decision_model="jev-frozen-test")
    result = run_adaptive_chapter(SOURCES, DRAFTS, glossary=GLOSSARY,
                                  decision_provider=screen, review_provider=Reviewer(patch=False),
                                  verification_policy=policy, context_by_segment={SOURCES[0].segment_id: "old memory"})
    assert {request.requested_model for request in screen.requests} == {"jev-frozen-test"}
    first_screen = next(e for e in result.evidence if e.level == "jev_screen" and e.context.segment_id == SOURCES[0].segment_id)
    assert _current(first_screen, 0, SOURCES, DRAFTS, result.memory_snapshot_sha256,
                    policy, {SOURCES[0].segment_id: "old memory"})
    assert not _current(first_screen, 0, SOURCES, DRAFTS, result.memory_snapshot_sha256,
                        policy, {SOURCES[0].segment_id: "changed memory"})


def test_rejects_stale_draft_hash_and_invalid_policy() -> None:
    stale = (DraftSegment(SOURCES[0].segment_id, DRAFTS[0].translated_text, "0" * 64), DRAFTS[1])
    import pytest
    with pytest.raises(ValueError, match="stale draft segment"):
        run_adaptive_chapter(SOURCES, stale, glossary=GLOSSARY,
                             decision_provider=Screening(), review_provider=Reviewer())
    with pytest.raises(ValueError, match="thresholds"):
        VerificationPolicy(routing_thresholds={"material": 0.1, "diagnostic": 0.5, "readability": 1.5})


def test_unavailable_jev_routes_review_and_marks_fallback() -> None:
    result = run_adaptive_chapter(SOURCES, DRAFTS, glossary=GLOSSARY,
                                  decision_provider=Screening(unavailable=True), review_provider=Reviewer(patch=False))
    assert result.review_calls == 2
    assert result.outcome == "delivered_with_warnings"
    assert {warning.code for warning in result.warnings} == {"router_fallback"}
    assert all(receipt.status == "unavailable" for receipt in result.evidence if receipt.level == "jev_screen")


def test_stale_decision_receipt_cannot_count_as_clean_screen() -> None:
    class StaleScreen:
        def evaluate(self, request):
            good = _decision(request)
            bad = good.receipt.model_copy(update={"context_sha256": "0" * 64})
            return good.model_copy(update={"receipt": bad})

    result = run_adaptive_chapter(SOURCES, DRAFTS, glossary=GLOSSARY,
                                  decision_provider=StaleScreen(), review_provider=Reviewer(patch=False))
    assert result.review_calls == 2
    assert result.outcome == "delivered_with_warnings"
    assert any(event["kind"] == "screen_stale" for event in result.events)
    assert "router_fallback" in {warning.code for warning in result.warnings}


def test_full_review_requires_each_segment_and_fails_closed_on_budget() -> None:
    reviewer = Reviewer(patch=False)
    result = run_adaptive_chapter(SOURCES, DRAFTS, glossary=GLOSSARY,
                                  decision_provider=None, review_provider=reviewer,
                                  verification_policy=VerificationPolicy(mode="full_segment_review"),
                                  budget_policy=BudgetPolicy(max_coordinator_steps=1))
    assert result.review_calls == 1
    assert result.outcome == "delivered_with_warnings"
    assert "missing_review_coverage" in {warning.code for warning in result.warnings}


def test_full_review_rechecks_edited_segment() -> None:
    reviewer = Reviewer()
    result = run_adaptive_chapter(SOURCES, DRAFTS, glossary=GLOSSARY,
                                  decision_provider=None, review_provider=reviewer,
                                  verification_policy=VerificationPolicy(mode="full_segment_review"))
    assert result.outcome == "delivered"
    assert result.review_calls == 3
    assert [request.draft_text for request in reviewer.requests if request.segment_id == SOURCES[0].segment_id] == [
        DRAFTS[0].translated_text,
        "Chapter 1\n\nHe guarded the gate.",
    ]
    assert len([e for e in result.evidence if e.level == "generative_review"]) == 2


def test_fatal_generative_provider_error_propagates(tmp_path) -> None:
    class FatalProviderError(RuntimeError):
        fatal_provider_error = True

    class FatalReview:
        def review(self, request):
            raise FatalProviderError("provider budget exhausted")

    import pytest
    with pytest.raises(FatalProviderError, match="budget exhausted"):
        run_adaptive_chapter(
            SOURCES, DRAFTS, glossary=GLOSSARY, decision_provider=Screening(),
            review_provider=FatalReview(), session_dir=tmp_path / "fatal",
        )
    import json
    saved = json.loads((tmp_path / "fatal" / "adaptive_checkpoint.json").read_text())
    assert saved["terminal"] is False


def test_terminal_checkpoint_recovers_edited_translation_without_provider_calls(tmp_path) -> None:
    session = tmp_path / "edited"
    original = run_adaptive_chapter(
        SOURCES, DRAFTS, glossary=GLOSSARY, decision_provider=Screening(),
        review_provider=Reviewer(), session_dir=session,
    )

    class NoCalls:
        def evaluate(self, request):
            raise AssertionError("Jev should not be called for a terminal checkpoint")

        def review(self, request):
            raise AssertionError("Review should not be called for a terminal checkpoint")

    recovered = run_adaptive_chapter(
        SOURCES, DRAFTS, glossary=GLOSSARY, decision_provider=NoCalls(),
        review_provider=NoCalls(), session_dir=session,
    )
    assert recovered == original
    assert recovered.final_text == "Chapter 1\n\nHe guarded the gate.\n\nShe entered the courtyard."


def test_jev_provider_error_still_routes_generative_fallback() -> None:
    class BrokenDecision:
        def evaluate(self, request):
            raise RuntimeError("Jev unavailable")

    result = run_adaptive_chapter(SOURCES, DRAFTS, glossary=GLOSSARY,
                                  decision_provider=BrokenDecision(), review_provider=Reviewer(patch=False))
    assert result.review_calls == len(SOURCES)
    assert result.outcome == "delivered_with_warnings"
    assert "router_fallback" in {warning.code for warning in result.warnings}


def test_fatal_decision_budget_error_stops_initial_screening() -> None:
    from agentic_translation.costing import BudgetExceeded
    import pytest

    class ExhaustedDecision:
        def evaluate(self, request):
            raise BudgetExceeded("decision budget exhausted")

    reviewer = Reviewer()
    with pytest.raises(BudgetExceeded, match="decision budget exhausted"):
        run_adaptive_chapter(SOURCES, DRAFTS, glossary=GLOSSARY,
                             decision_provider=ExhaustedDecision(), review_provider=reviewer)
    assert reviewer.requests == []


def test_fatal_decision_budget_error_stops_candidate_check() -> None:
    from agentic_translation.costing import BudgetExceeded
    import pytest

    class ExhaustedOnCandidate(Screening):
        def evaluate(self, request):
            if request.candidate_text is not None:
                raise BudgetExceeded("candidate budget exhausted")
            return super().evaluate(request)

    reviewer = Reviewer()
    with pytest.raises(BudgetExceeded, match="candidate budget exhausted"):
        run_adaptive_chapter(SOURCES, DRAFTS, glossary=GLOSSARY,
                             decision_provider=ExhaustedOnCandidate(), review_provider=reviewer)
    assert len(reviewer.requests) == 1


def test_ordinary_candidate_check_error_keeps_generative_fallback() -> None:
    class UnavailableOnCandidate(Screening):
        def evaluate(self, request):
            if request.candidate_text is not None:
                raise RuntimeError("temporary decision service failure")
            return super().evaluate(request)

    result = run_adaptive_chapter(SOURCES, DRAFTS, glossary=GLOSSARY,
                                  decision_provider=UnavailableOnCandidate(), review_provider=Reviewer())
    assert result.final_text == "Chapter 1\n\nHe guarded the gate.\n\nShe entered the courtyard."
    assert any(event["kind"] == "candidate_check_failed" for event in result.events)


def test_segment_patch_is_atomic_on_second_ambiguous_edit() -> None:
    executor = RepairToolExecutor(SOURCE, "abc abc\n\nother", GLOSSARY,
                                  allow_nonregressing_patches=True)
    action = SubmitSegmentPatchAction(
        segment_id="s1", expected_segment_sha256=sha256_text("abc abc"),
        edits=[SegmentTextEdit(old_text="abc abc", new_text="abc"),
               SegmentTextEdit(old_text="x", new_text="z")],
        issue_ids=["x"], rationale="Test atomicity",
    )
    outcome = executor.submit_segment_patch(action, [("s1", "abc abc"), ("s2", "other")])
    assert not outcome.observation.ok
    assert outcome.observation.data["reason"] == "ambiguous_target"
    assert executor.current_text == "abc abc\n\nother"


def test_segment_patch_allows_repeated_phrase_in_other_segment() -> None:
    executor = RepairToolExecutor(SOURCE, "wrong\n\nwrong", GLOSSARY,
                                  allow_nonregressing_patches=True)
    action = SubmitSegmentPatchAction(
        segment_id="s1", expected_segment_sha256=sha256_text("wrong"),
        edits=[SegmentTextEdit(old_text="wrong", new_text="right")],
        issue_ids=["meaning"], rationale="Target one segment",
    )
    outcome = executor.submit_segment_patch(action, [("s1", "wrong"), ("s2", "wrong")])
    assert outcome.observation.ok
    assert executor.current_text == "right\n\nwrong"
