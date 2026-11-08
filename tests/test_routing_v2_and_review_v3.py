"""Routing v2 selectivity, review v3 policy/schema, and QA warning hygiene.

Covers work from the harness-v4 repair/routing pass:
- adaptive-routing.v2's material/diagnostic-band/source_incomplete table,
  contrasted against adaptive-routing.v1's unchanged (more trigger-happy)
  semantics.
- natural-review-v3's prompt content, polish severity, multi-edit and
  whole-segment-rewrite patches, and the deterministic content-loss guard.
- qa-warnings.v2 treating cosmetic/legacy QA finding categories as
  informational rather than blocking.
"""

from __future__ import annotations

from agentic_translation.adaptive import (
    AdaptiveReviewFinding,
    AdaptiveReviewRequest,
    AdaptiveReviewResult,
    VerificationPolicy,
    _content_signals,
    _polish_guard_violation,
    _route_and_issues,
    _segment_deterministic_findings,
    run_adaptive_chapter,
)
from agentic_translation.agent_models import SegmentTextEdit, SubmitSegmentPatchAction
from agentic_translation.decisions.models import (
    DEFAULT_THRESHOLDS,
    QUESTION_SCHEMA_VERSION,
    DecisionReceipt,
    DecisionRequest,
    DecisionResult,
)
from agentic_translation.decisions.questions import question_schema_digest, request_id
from agentic_translation.glossary import parse_glossary_text
from agentic_translation.segmentation import DraftSegment, segment_source, sha256_text
from agentic_translation.unattended import REVIEW_SCHEMA, GenerativeReviewAdapter


SOURCE = "他守住山门。\n"
SOURCES = segment_source("w", "c1", SOURCE, target_chars=8, max_chars=12)
DRAFTS = (DraftSegment.from_text(SOURCES[0].segment_id, "Chapter 1\n\nHe guarded the gate."),)
GLOSSARY = parse_glossary_text("")


def _glossary(*pairs: tuple[str, str]):
    from agentic_translation.models import GlossaryEntry, GlossaryParseResult

    return GlossaryParseResult(entries=[GlossaryEntry(source=source, target=target) for source, target in pairs])


def _decision(request: DecisionRequest | None = None, *, noul_by_question=None, derived_flags=None,
             routing_policy_version="adaptive-routing.v2", status="ok") -> DecisionResult:
    if request is None:
        request = DecisionRequest(
            segment_id=SOURCES[0].segment_id, source_text=SOURCES[0].text,
            draft_text=DRAFTS[0].translated_text, routing_policy_version=routing_policy_version,
        )
    raw_answers = {}
    for name, value in (noul_by_question or {}).items():
        raw_answers[name] = {"type": "noul", "noul": value}
    return DecisionResult(
        status=status,
        receipt=DecisionReceipt(
            request_id=request_id(request), segment_id=request.segment_id,
            requested_model=request.requested_model, source_sha256=request.source_sha256,
            draft_sha256=request.draft_sha256, context_sha256=request.context_sha256,
            candidate_sha256=(sha256_text(request.candidate_text) if request.candidate_text is not None else None),
            question_schema_version=QUESTION_SCHEMA_VERSION,
            question_schema_sha256=question_schema_digest(request),
            raw_answers=raw_answers, derived_flags=derived_flags or {},
            routing_policy_version=request.routing_policy_version,
            routing_thresholds=request.routing_thresholds,
            latency_ms=1, attempt_count=1, transport_status="ok", cache_status="recorded",
        ),
    )


def test_routing_v2_material_flag_always_routes() -> None:
    policy = VerificationPolicy(routing_policy_version="adaptive-routing.v2")
    decision = _decision(noul_by_question={"actor_relation_error": 0.9},
                        derived_flags={"actor_relation_error": True})
    route, issue_ids, notes = _route_and_issues(decision, policy)
    assert route == "material"
    assert issue_ids == ("actor_relation_error",)
    assert notes == ()


def test_routing_v2_single_diagnostic_flag_is_screened_clean() -> None:
    policy = VerificationPolicy(routing_policy_version="adaptive-routing.v2", diagnostic_min_flags=2)
    decision = _decision(noul_by_question={"omitted_material": 0.2})
    route, issue_ids, notes = _route_and_issues(decision, policy)
    assert route == "clean"
    assert issue_ids == ()
    assert notes == ("omitted_material",)


def test_routing_v2_two_diagnostic_flags_route_to_review() -> None:
    policy = VerificationPolicy(routing_policy_version="adaptive-routing.v2", diagnostic_min_flags=2)
    decision = _decision(noul_by_question={"omitted_material": 0.2, "terminology_conflict": 0.3})
    route, issue_ids, notes = _route_and_issues(decision, policy)
    assert route == "diagnostic"
    assert set(issue_ids) == {"omitted_material", "terminology_conflict"}


def test_routing_v2_severe_readability_alone_routes_to_review() -> None:
    policy = VerificationPolicy(routing_policy_version="adaptive-routing.v2", diagnostic_min_flags=2)
    decision = _decision(derived_flags={"severe_readability": True})
    route, issue_ids, notes = _route_and_issues(decision, policy)
    assert route == "diagnostic"
    assert issue_ids == ("severe_readability",)


def test_routing_v2_source_incomplete_alone_is_recorded_not_routed_by_default() -> None:
    policy = VerificationPolicy(routing_policy_version="adaptive-routing.v2")
    decision = _decision(derived_flags={"source_incomplete": True})
    route, issue_ids, notes = _route_and_issues(decision, policy)
    assert route == "clean"
    assert issue_ids == ()
    assert notes == ("source_incomplete",)


def test_routing_v2_source_incomplete_routes_when_configured() -> None:
    policy = VerificationPolicy(routing_policy_version="adaptive-routing.v2", route_source_incomplete=True)
    decision = _decision(derived_flags={"source_incomplete": True})
    route, issue_ids, notes = _route_and_issues(decision, policy)
    assert route == "material"
    assert issue_ids == ("source_incomplete",)


def test_routing_v1_semantics_unchanged_for_single_diagnostic_flag() -> None:
    """v1 has no diagnostic_min_flags gate: any nonempty issue set routes to review."""
    policy = VerificationPolicy(routing_policy_version="adaptive-routing.v1")
    decision = _decision(
        noul_by_question={"omitted_material": 0.2}, routing_policy_version="adaptive-routing.v1",
        derived_flags={"diagnostic_error": True},
    )
    route, issue_ids, notes = _route_and_issues(decision, policy)
    assert route == "diagnostic"
    assert issue_ids == ("diagnostic_error",)
    assert notes == ()


def test_routing_v1_source_incomplete_alone_still_forces_material_and_warns() -> None:
    policy = VerificationPolicy(routing_policy_version="adaptive-routing.v1")
    decision = _decision(derived_flags={"source_incomplete": True}, routing_policy_version="adaptive-routing.v1")
    route, issue_ids, notes = _route_and_issues(decision, policy)
    assert route == "material"
    assert issue_ids == ("source_incomplete",)


def test_fallback_and_none_decision_both_route_review_v1_and_v2() -> None:
    for version in ("adaptive-routing.v1", "adaptive-routing.v2"):
        policy = VerificationPolicy(routing_policy_version=version)
        route, issue_ids, notes = _route_and_issues(None, policy)
        assert (route, issue_ids) == ("fallback", ("router_fallback",))
        unavailable = _decision(status="unavailable", routing_policy_version=version)
        route, issue_ids, notes = _route_and_issues(unavailable, policy)
        assert (route, issue_ids) == ("fallback", ("router_fallback",))


def test_review_v3_prompt_states_new_policy_and_polish_severity() -> None:
    class Recording:
        def __init__(self):
            self.prompts = []

        def call(self, operation, prompt, schema, *, max_output_tokens):
            self.prompts.append(prompt)
            return {"summary": "ok", "findings": [], "patches": []}

    generator = Recording()
    adapter = GenerativeReviewAdapter(generator)
    request = AdaptiveReviewRequest(
        segment_id=SOURCES[0].segment_id, source_text=SOURCES[0].text,
        draft_text=DRAFTS[0].translated_text, context_text="", issue_ids=(),
        mode="adaptive_segment_screen", review_round=1, instruction_version="natural-review-v3",
    )
    result = adapter.review(request)
    assert result.status == "completed"
    prompt = generator.prompts[0]
    assert "awkward or unnatural English" in prompt
    assert "polish" in prompt
    assert "whole-segment rewrite" in prompt
    assert "never add facts" in prompt


def test_review_schema_allows_polish_severity_and_multi_edit_patch() -> None:
    severities = REVIEW_SCHEMA["properties"]["findings"]["items"]["properties"]["severity"]["enum"]
    assert set(severities) == {"material", "diagnostic", "polish"}
    patch = SubmitSegmentPatchAction(
        segment_id="s1", expected_segment_sha256=sha256_text("abc"),
        edits=[SegmentTextEdit(old_text=f"a{i}", new_text=f"b{i}") for i in range(8)],
        issue_ids=["polish"], rationale="Multiple small polish edits interacting.",
    )
    assert len(patch.edits) == 8


def test_whole_segment_rewrite_accepted_through_executor() -> None:
    from agentic_translation.agent_repair import RepairToolExecutor

    original_segment = "He guarded the gate stiffly and without much grace."
    other_segment = "She entered the courtyard quietly."
    executor = RepairToolExecutor("source", f"{original_segment}\n\n{other_segment}", GLOSSARY,
                                  allow_nonregressing_patches=True)
    action = SubmitSegmentPatchAction(
        segment_id="s1", expected_segment_sha256=sha256_text(original_segment),
        edits=[SegmentTextEdit(old_text=original_segment,
                               new_text="He held the gate with quiet, easy confidence.")],
        issue_ids=["polish"], rationale="Whole-segment rewrite for natural phrasing.",
    )
    outcome = executor.submit_segment_patch(action, [("s1", original_segment), ("s2", other_segment)])
    assert outcome.observation.ok
    assert outcome.observation.data["candidate_segment_text"] == "He held the gate with quiet, easy confidence."


def test_polish_guard_rejects_truncating_rewrite() -> None:
    old_text = "He said, \"I will guard the gate with all three hundred men.\""
    truncated = "He guarded it."
    assert _polish_guard_violation(old_text, truncated) is not None
    natural = "He said he would guard the gate with all three hundred men."
    # Same digit/quote content roughly preserved and not drastically shorter.
    assert _polish_guard_violation(old_text, "He said, \"I will guard the gate with all three hundred men, resolutely.\"") is None


def test_polish_guard_vetoes_only_unambiguous_content_loss() -> None:
    # A number printed in the Chinese source must survive the rewrite.
    assert _polish_guard_violation("He has 300 men.", "He has many men.", "他有300人。") is not None
    # "3" -> "three" is meaning-preserving when the source spells the number in Chinese.
    assert _polish_guard_violation("She had 3 swords.", "She had three swords.", "她有三把剑。") is None
    # Direct -> indirect speech is a signal, not a veto.
    assert _polish_guard_violation('She said, "Run."', "She told him to run.", "她说：跑。") is None
    assert _content_signals('She said, "Run 3."', "She told him to run.") == ["fewer_digits", "fewer_quote_marks"]


def test_segment_deterministic_findings_name_unused_story_memory_terms() -> None:
    glossary = _glossary(("符匠", "Talisman Artisan"))
    findings = _segment_deterministic_findings("他想成為符匠。", "He wanted to become a talisman maker.", glossary)
    assert len(findings) == 1 and findings[0].startswith("glossary_required: source term 符匠")
    assert _segment_deterministic_findings("他想成為符匠。", "He wanted to become a Talisman Artisan.", glossary) == ()
    residual = _segment_deterministic_findings("他。", "He said 你好.", parse_glossary_text(""))
    assert residual and residual[0].startswith("residual_chinese")


def test_deterministic_findings_route_clean_segments_to_review_with_the_finding() -> None:
    glossary = _glossary(("山门", "Mountain Gate"))

    class CleanScreen:
        def evaluate(self, request):
            return _decision(request)

    class Recorder:
        def __init__(self):
            self.requests = []

        def review(self, request):
            self.requests.append(request)
            return AdaptiveReviewResult(status="completed", summary="ok")

    reviewer = Recorder()
    policy = VerificationPolicy(routing_policy_version="adaptive-routing.v2", route_deterministic_findings=True,
                                instruction_version="natural-review-v4")
    run_adaptive_chapter(SOURCES, DRAFTS, glossary=glossary, decision_provider=CleanScreen(),
                         review_provider=reviewer, verification_policy=policy)
    assert reviewer.requests and reviewer.requests[0].deterministic_findings
    assert "Mountain Gate" in reviewer.requests[0].deterministic_findings[0]
    # Without the knob a clean Jev screen still skips review.
    quiet = Recorder()
    run_adaptive_chapter(SOURCES, DRAFTS, glossary=glossary, decision_provider=CleanScreen(),
                         review_provider=quiet,
                         verification_policy=VerificationPolicy(routing_policy_version="adaptive-routing.v2"))
    assert quiet.requests == []


def test_review_v4_prompt_includes_findings_instruction_only_when_present() -> None:
    class Generator:
        def __init__(self):
            self.prompts = []

        def call(self, operation, prompt, schema, max_output_tokens):
            self.prompts.append(prompt)
            return {"summary": "ok", "findings": [], "patches": []}

    generator = Generator()
    adapter = GenerativeReviewAdapter(generator)
    base = dict(segment_id="s0001", source_text="他。", draft_text="He.", context_text="", issue_ids=(),
                mode="adaptive_segment_screen", review_round=1, instruction_version="natural-review-v4")
    adapter.review(AdaptiveReviewRequest(**base, deterministic_findings=("glossary_required: x",)))
    adapter.review(AdaptiveReviewRequest(**base))
    assert "Deterministic checks flagged" in generator.prompts[0]
    assert "Deterministic checks flagged" not in generator.prompts[1]
    assert "deterministic_findings" not in generator.prompts[1]


def test_candidate_rescreen_rejects_new_material_flag() -> None:
    class Screening:
        def evaluate(self, request):
            if request.candidate_text is None and "gate" not in request.draft_text:
                return _decision(request, noul_by_question={"omitted_material": 0.9},
                                 derived_flags={"omitted_material": True})
            return _decision(request)

    class Rewriter:
        def __init__(self):
            self.calls = 0

        def review(self, request):
            self.calls += 1
            if self.calls > 1:
                return AdaptiveReviewResult(status="completed", summary="done")
            return AdaptiveReviewResult(
                status="completed", summary="Rewrite",
                findings=(AdaptiveReviewFinding("polish", "polish", "Smoother", blocking=False),),
                patches=(SubmitSegmentPatchAction(
                    segment_id=request.segment_id, expected_segment_sha256=sha256_text(request.draft_text),
                    edits=[SegmentTextEdit(old_text=request.draft_text,
                                           new_text="Chapter 1\n\nHe stood watch, unmoving and silent.")],
                    issue_ids=["polish"], rationale="Flow.",
                ),),
            )

    policy = VerificationPolicy(routing_policy_version="adaptive-routing.v2", route_deterministic_findings=True,
                                candidate_rescreen=True, instruction_version="natural-review-v4")
    glossary = _glossary(("山门", "Mountain Gate"))
    two_sources = segment_source("w", "c1", "他守住山门。\n她走进院子。\n", target_chars=8, max_chars=12)
    two_drafts = (DraftSegment.from_text(two_sources[0].segment_id, "Chapter 1\n\nHe guarded the gate."),
                  DraftSegment.from_text(two_sources[1].segment_id, "She entered the courtyard of the gate."))
    result = run_adaptive_chapter(two_sources, two_drafts, glossary=glossary, decision_provider=Screening(),
                                  review_provider=Rewriter(), verification_policy=policy)
    assert "He guarded the gate." in result.final_text
    rescreens = [e for e in result.events if e["kind"] == "candidate_rescreened"]
    assert rescreens and rescreens[0]["introduced"] == ["omitted_material"]


def test_informational_checks_do_not_block_patches_under_qa_warnings_v2() -> None:
    from agentic_translation.agent_repair import RepairToolExecutor

    source = "他守住山门。"
    base = "Chapter 1\n\nHe guarded the gate."
    candidate = "Chapter 1\n\nHe said, “I guard the gate.”"
    action = SubmitSegmentPatchAction(
        segment_id="s0001", expected_segment_sha256=sha256_text(base),
        edits=[SegmentTextEdit(old_text="He guarded the gate.", new_text="He said, “I guard the gate.”")],
        issue_ids=["polish"], rationale="Voice.",
    )
    strict = RepairToolExecutor(source_text=source, translated_text=base, glossary=GLOSSARY,
                                allow_nonregressing_patches=True)
    lenient = RepairToolExecutor(source_text=source, translated_text=base, glossary=GLOSSARY,
                                 allow_nonregressing_patches=True,
                                 ignored_qa_checks={"chinese_punctuation", "heading_format"})
    assert not strict.submit_segment_patch(action, [("s0001", base)]).observation.ok
    assert lenient.submit_segment_patch(action, [("s0001", base)]).observation.ok
    assert lenient.current_text == candidate


def test_new_policy_fields_at_legacy_defaults_keep_session_identity() -> None:
    from dataclasses import asdict
    from agentic_translation.adaptive import BudgetPolicy, InteractionPolicy, _identity_hash, _identity_verification

    policy = VerificationPolicy()
    legacy = {key: value for key, value in asdict(policy).items()
              if key not in {"diagnostic_min_flags", "route_source_incomplete", "qa_warning_policy_version",
                             "route_deterministic_findings", "candidate_rescreen", "term_alignment_pass",
                             "candidate_rescreen_threshold", "gloss_guard", "duplication_guard"}}
    assert _identity_verification(policy) == legacy
    changed = VerificationPolicy(route_deterministic_findings=True)
    assert _identity_hash("s", "d", policy, InteractionPolicy(), BudgetPolicy(), "m", {}) != \
        _identity_hash("s", "d", changed, InteractionPolicy(), BudgetPolicy(), "m", {})


def test_adaptive_chapter_rejects_patch_that_would_truncate_content() -> None:
    class Screening:
        def evaluate(self, request):
            return _decision(request)

    class TruncatingReviewer:
        def __init__(self):
            self.calls = 0

        def review(self, request):
            self.calls += 1
            if self.calls > 1:
                return AdaptiveReviewResult(status="completed", summary="No further edit")
            return AdaptiveReviewResult(
                status="completed", summary="Rewrite",
                findings=(AdaptiveReviewFinding("polish", "polish", "Tighten prose", blocking=False),),
                patches=(SubmitSegmentPatchAction(
                    segment_id=request.segment_id,
                    expected_segment_sha256=sha256_text(request.draft_text),
                    edits=[SegmentTextEdit(old_text=request.draft_text, new_text="He guarded it.")],
                    issue_ids=["polish"], rationale="Shorten for pace.",
                ),),
            )

    two_segment_source = "他守住山门。\n她走进院子。\n"
    two_sources = segment_source("w", "c1", two_segment_source, target_chars=8, max_chars=12)
    long_draft = (
        DraftSegment.from_text(
            two_sources[0].segment_id,
            "Chapter 1\n\nHe stood at the gate for three hundred long days, resolute and unmoving.",
        ),
        DraftSegment.from_text(two_sources[1].segment_id, "She entered the courtyard."),
    )
    reviewer = TruncatingReviewer()
    result = run_adaptive_chapter(
        two_sources, long_draft, glossary=GLOSSARY, decision_provider=Screening(),
        review_provider=reviewer,
        verification_policy=VerificationPolicy(mode="full_segment_review"),
    )
    assert "He stood at the gate for three hundred long days" in result.final_text
    assert any(event["kind"] == "candidate_guard_rejected" for event in result.events)


def test_qa_warning_hygiene_treats_legacy_categories_as_informational() -> None:
    policy_v1 = VerificationPolicy(qa_warning_policy_version="qa-warnings.v1")
    policy_v2 = VerificationPolicy(qa_warning_policy_version="qa-warnings.v2")

    class CleanScreen:
        def evaluate(self, request):
            return _decision(request)

    # A draft with a curly-quote (English typographic quote) and a spelled-out
    # chapter heading trips only the two legacy/cosmetic QA categories.
    draft = (DraftSegment.from_text(
        SOURCES[0].segment_id,
        "Chapter One: The Gate\n\nHe said, “I will guard it,” and did not move.",
    ),)
    result_v1 = run_adaptive_chapter(SOURCES, draft, glossary=GLOSSARY, decision_provider=CleanScreen(),
                                     review_provider=None, verification_policy=policy_v1)
    result_v2 = run_adaptive_chapter(SOURCES, draft, glossary=GLOSSARY, decision_provider=CleanScreen(),
                                     review_provider=None, verification_policy=policy_v2)
    assert result_v1.outcome == "delivered_with_warnings"
    assert "deterministic_qa_findings" in {w.code for w in result_v1.warnings}
    assert result_v2.outcome == "delivered"
    assert "deterministic_qa_findings" not in {w.code for w in result_v2.warnings}
    assert any(event["kind"] == "qa_findings_informational" for event in result_v2.events)


def test_term_alignment_pass_applies_batched_patches_before_screening() -> None:
    from agentic_translation.adaptive import TermAlignmentItem

    glossary = _glossary(("山门", "Mountain Gate"))
    two_sources = segment_source("w", "c1", "他守住山门。\n她走进院子。\n", target_chars=8, max_chars=12)
    two_drafts = (DraftSegment.from_text(two_sources[0].segment_id, "Chapter 1\n\nHe guarded the mountain entrance."),
                  DraftSegment.from_text(two_sources[1].segment_id, "She entered the courtyard."))

    class Aligner:
        def __init__(self):
            self.items = []

        def align(self, items):
            self.items = list(items)
            item = items[0]
            return [SubmitSegmentPatchAction(
                segment_id=item.segment_id, expected_segment_sha256=item.expected_segment_sha256,
                edits=[SegmentTextEdit(old_text="the mountain entrance", new_text="the Mountain Gate")],
                issue_ids=["term_alignment"], rationale="Established rendering.",
            )]

    class CleanScreen:
        def evaluate(self, request):
            return _decision(request)

    aligner = Aligner()
    policy = VerificationPolicy(routing_policy_version="adaptive-routing.v2", term_alignment_pass=True)
    result = run_adaptive_chapter(two_sources, two_drafts, glossary=glossary, decision_provider=CleanScreen(),
                                  review_provider=None, verification_policy=policy, alignment_provider=aligner)
    assert [item.segment_id for item in aligner.items] == [two_sources[0].segment_id]
    assert aligner.items[0].terms == (("山门", "Mountain Gate"),)
    assert "He guarded the Mountain Gate." in result.final_text
    kinds = [event["kind"] for event in result.events]
    assert kinds.index("term_alignment_applied") < kinds.index("segment_screened")


def test_gloss_guard_rejects_added_notes_and_alternatives_only() -> None:
    from agentic_translation.adaptive import _gloss_guard_violation

    assert _gloss_guard_violation("He drew a talisman.", "He drew a talisman (an artisan's charm).", "他画符。")
    assert _gloss_guard_violation("A river wraith rose.", "A river wraith / drowned shade rose.", "河魅出现。")
    assert _gloss_guard_violation("He drew a talisman.", "He drew a talisman (see below).", "他画符（见下）。") is None
    assert _gloss_guard_violation("It cost 3/4 of his qi.", "It cost three quarters of his qi.", "") is None


def test_terms_are_not_enforced_where_the_source_supplies_english() -> None:
    from agentic_translation.adaptive import _unused_story_terms

    glossary = _glossary(("雾魇", "mist wraith"), ("山门", "Mountain Gate"))
    source = "第三类是雾魇(Mistwraith)。他守住山门。"
    draft = "The third category is Mistwraith. He guarded the gate."
    assert _unused_story_terms(source, draft, glossary) == (("山门", "Mountain Gate"),)
    assert all("雾魇" not in item for item in _segment_deterministic_findings(source, draft, glossary))


def test_duplication_guard_rejects_copied_paragraphs() -> None:
    from agentic_translation.adaptive import _duplication_violation

    other = ["The silence stretched on, broken only by the distant starlight over the ridge."]
    old = "He waited."
    assert _duplication_violation(old, "He waited.\n" + other[0], other)
    repeated = "A long paragraph that should appear exactly once in this segment's text.\n" * 2
    assert _duplication_violation(old, repeated, [])
    assert _duplication_violation(old, "He waited for a long time before speaking again.", other) is None


def test_review_v5_prompt_limits_style_edits() -> None:
    class Generator:
        def __init__(self):
            self.prompts = []

        def call(self, operation, prompt, schema, max_output_tokens):
            self.prompts.append(prompt)
            return {"summary": "ok", "findings": [], "patches": []}

    generator = Generator()
    base = dict(segment_id="s0001", source_text="他。", draft_text="He.", context_text="", issue_ids=(),
                mode="adaptive_segment_screen", review_round=1)
    GenerativeReviewAdapter(generator).review(AdaptiveReviewRequest(**base, instruction_version="natural-review-v5"))
    GenerativeReviewAdapter(generator).review(AdaptiveReviewRequest(**base, instruction_version="natural-review-v4"))
    assert "do not rephrase acceptable sentences" in generator.prompts[0]
    assert "do not rephrase acceptable sentences" not in generator.prompts[1]
