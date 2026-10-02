from __future__ import annotations

from pathlib import Path

import pytest

from agentic_translation.adaptive import InteractionPolicy, run_adaptive_chapter
from agentic_translation.agent_models import PromoteGlossaryTermAction
from agentic_translation.agent_policy import RunLocalGlossaryAuthority, UnattendedToolPolicy
from agentic_translation.agent_tools import SHOWCASE_TOOL_REGISTRY
from agentic_translation.glossary import parse_glossary_text
from agentic_translation.segmentation import DraftSegment, segment_source


def test_unattended_policy_never_returns_human_approval() -> None:
    action = PromoteGlossaryTermAction(term="山门", rationale="Automated preference")
    spec = SHOWCASE_TOOL_REGISTRY.spec(action.tool)
    result = UnattendedToolPolicy().before_tool(action, spec)
    assert result.outcome == "reject"
    assert "unattended" in result.rule_id
    with pytest.raises(ValueError, match="human approval"):
        InteractionPolicy(human_approval=True)


def test_run_local_authority_stays_inside_own_glossary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    master = tmp_path / "master.txt"
    authority = RunLocalGlossaryAuthority(run_dir, master)
    assert authority.authorize(run_dir / "glossary" / "terms.json").outcome == "allow"
    assert authority.authorize(master).outcome == "reject"
    assert authority.authorize(tmp_path / "other-run" / "glossary" / "terms.json").outcome == "reject"
    assert authority.authorize(run_dir / "glossary" / ".." / "outside.json").outcome == "reject"


def test_unattended_session_persists_terminal_policy_identity(tmp_path: Path) -> None:
    source = "他守住山门。\n"
    sources = segment_source("w", "c1", source)
    drafts = (DraftSegment.from_text(sources[0].segment_id, "Chapter 1\n\nHe guarded the gate."),)

    class CleanDecision:
        def evaluate(self, request):
            from agentic_translation.decisions.models import DecisionReceipt, DecisionResult, QUESTION_SCHEMA_VERSION
            from agentic_translation.decisions.questions import question_schema_digest, request_id
            return DecisionResult(
                status="ok",
                receipt=DecisionReceipt(
                    request_id=request_id(request),
                    segment_id=request.segment_id, requested_model=request.requested_model,
                    source_sha256=request.source_sha256, draft_sha256=request.draft_sha256,
                    context_sha256=request.context_sha256,
                    question_schema_version=QUESTION_SCHEMA_VERSION,
                    question_schema_sha256=question_schema_digest(request),
                    routing_policy_version=request.routing_policy_version,
                    routing_thresholds=request.routing_thresholds,
                    latency_ms=1, attempt_count=1, transport_status="ok", cache_status="recorded",
                ),
            )

    result = run_adaptive_chapter(
        sources, drafts, glossary=parse_glossary_text(""),
        decision_provider=CleanDecision(), review_provider=None,
        session_dir=tmp_path / "session",
    )
    assert result.outcome == "delivered"
    import json
    saved = json.loads((tmp_path / "session" / "adaptive_checkpoint.json").read_text())
    assert saved["terminal"] is True
    assert saved["identity_sha256"] == result.identity_sha256
    assert saved["result"]["interaction_policy"]["mode"] == "unattended"

    class NoCalls:
        def evaluate(self, request):
            raise AssertionError("Terminal checkpoint must not call Jev again")

    recovered = run_adaptive_chapter(
        sources, drafts, glossary=parse_glossary_text(""),
        decision_provider=NoCalls(), review_provider=None,
        session_dir=tmp_path / "session",
    )
    assert recovered == result

    changed = (drafts[0].with_text("Chapter 1\n\nShe guarded the gate."),)
    with pytest.raises(ValueError, match="identity differs"):
        run_adaptive_chapter(
            sources, changed, glossary=parse_glossary_text(""),
            decision_provider=NoCalls(), review_provider=None,
            session_dir=tmp_path / "session",
        )


def test_configured_arms_runs_only_requested_subset_in_frozen_order():
    from agentic_translation.unattended import configured_arms

    assert configured_arms({"arms": ["jev_adaptive", "naive", "contextual"]}) == (
        "naive", "contextual", "jev_adaptive")
    assert configured_arms({}) == ("naive", "contextual", "simple_revise", "always_review", "jev_adaptive")
    import pytest
    with pytest.raises(ValueError):
        configured_arms({"arms": ["naive", "jev_adaptive"]})
    with pytest.raises(ValueError):
        configured_arms({"arms": ["naive", "bogus"]})


def test_clean_term_target_strips_glosses_and_alternatives():
    from agentic_translation.unattended import clean_term_target

    assert clean_term_target("Inner Breath (Neixi)") == "Inner Breath"
    assert clean_term_target("third-rank (martial grade)") == "third-rank"
    assert clean_term_target("Jade Lotus Sword Saint / Lotus Saint") == "Jade Lotus Sword Saint"
    assert clean_term_target("Azure Cloud Sect（青云宗）") == "Azure Cloud Sect"
    assert clean_term_target("Su Ming") == "Su Ming"
    assert clean_term_target("Heaven or Hell Sect") == "Heaven or Hell Sect"


def test_translator_terms_join_context_but_only_where_they_occur():
    from agentic_translation.segmentation import segment_source
    from agentic_translation.unattended import _with_translator_terms, translator_terms_for_chapter

    source = "他拜入青云宗。\n师兄笑了。\n"
    segments = segment_source("w", "c1", source, target_chars=8, max_chars=12)
    terms = translator_terms_for_chapter({"青云宗": "Azure Cloud Sect (sect)", "天剑": "Heaven Sword", "宗": "x"}, source)
    assert terms == {"青云宗": "Azure Cloud Sect"}
    merged = _with_translator_terms({segments[0].segment_id: "TERM 师兄 -> Senior Brother"}, segments, terms)
    assert merged[segments[0].segment_id].startswith("TERM 青云宗 -> Azure Cloud Sect [translator glossary]\nTERM 师兄")
    assert "青云宗" not in merged[segments[1].segment_id]


def test_translator_terms_replace_overlapping_memory_terms_in_context():
    from agentic_translation.segmentation import segment_source
    from agentic_translation.unattended import _with_translator_terms

    source = "他是铸霞修士。\n"
    segments = segment_source("w", "c1", source, target_chars=20, max_chars=40)
    memory = {segments[0].segment_id: "TERM 铸霞修士 -> Dawn-Forging cultivator\nTERM 師兄 -> Senior Brother"}
    merged = _with_translator_terms(memory, segments, {"铸霞": "Dawn Forging"},
                                    drop_overlapping_memory_terms=True)
    text = merged[segments[0].segment_id]
    assert "Dawn Forging [translator glossary]" in text
    assert "Dawn-Forging cultivator" not in text and "Senior Brother" in text
