from __future__ import annotations

from pathlib import Path

import pytest

from agentic_translation.glossary import load_glossary, parse_glossary_text
from agentic_translation.models import QAFinding, QALocation, RepairPatch
from agentic_translation.pipeline import _apply_repair_queue
from agentic_translation.providers_offline import OfflineJudgeProvider, OfflineRepairProvider
from agentic_translation.qa import run_translation_qa
from agentic_translation.repair import (
    apply_patch,
    prioritized_repairable_findings,
    route_repair_strategy,
    validate_patch_improves_qa,
)


class MissingSpanRepairProvider:
    provider_name = "missing-span"
    model_name = "test"

    def propose_patch(self, **kwargs):  # noqa: ANN003
        return RepairPatch(
            patch_id="missing_span",
            patch_type="replace_span",
            chapter=kwargs["chapter"],
            old_text="Law Lord",
            new_text="Lord of Laws",
            reason="Bad singular patch for plural text.",
            source_finding_check_id=kwargs["finding"].check_id,
        )


def test_replace_span_patch_replaces_all_occurrences() -> None:
    text = "Way of Heaven above. Way of Heaven below."
    patch = RepairPatch(
        patch_id="p1",
        patch_type="replace_span",
        chapter="0001",
        old_text="Way of Heaven",
        new_text="Heavenly Dao",
        reason="Use glossary canon.",
    )

    assert apply_patch(text, patch) == "Heavenly Dao above. Heavenly Dao below."


def test_replace_span_patch_does_not_replace_inside_larger_english_token() -> None:
    patch = RepairPatch(
        patch_id="p_plural",
        patch_type="replace_span",
        chapter="0001",
        old_text="Law Lord",
        new_text="Lord of Laws",
        reason="Use glossary canon.",
    )

    result = apply_patch("A Law Lord arrived. The Law Lords waited.", patch)

    assert result == "A Lord of Laws arrived. The Law Lords waited."
    assert "Lord of Lawss" not in result


def test_replace_span_rejects_empty_old_text() -> None:
    patch = RepairPatch(
        patch_id="bad",
        patch_type="replace_span",
        chapter="0001",
        old_text="",
        new_text="oops",
        reason="Invalid empty span.",
    )

    with pytest.raises(ValueError, match="non-empty old_text"):
        apply_patch("Chapter 1\n\nBody", patch)


def test_replace_paragraph_patch_replaces_target_paragraph() -> None:
    text = "Chapter 1\n\nOld paragraph.\n\nLast paragraph."
    patch = RepairPatch(
        patch_id="p2",
        patch_type="replace_paragraph",
        chapter="0001",
        paragraph_index=1,
        old_text="Old paragraph.",
        new_text="New paragraph.",
        reason="Panel repair.",
    )

    assert apply_patch(text, patch) == "Chapter 1\n\nNew paragraph.\n\nLast paragraph."


def test_prioritized_findings_puts_panel_before_glossary() -> None:
    source = Path("samples/public_demo/source/0001.txt").read_text(encoding="utf-8")
    dirty = Path("samples/public_demo/expected/dirty_translation.txt").read_text(encoding="utf-8")
    glossary = load_glossary("samples/public_demo/terms/master_glossary.txt")
    report = run_translation_qa(
        run_id="test",
        story_slug="public_demo",
        chapter="0001",
        source_text=source,
        translated_text=dirty,
        glossary=glossary,
    )

    ordered = prioritized_repairable_findings(report.findings)

    assert ordered[0].check_id == "residual_chinese"
    assert "system_panel_count" in [finding.check_id for finding in ordered]


def test_patch_validation_requires_score_improvement() -> None:
    source = Path("samples/public_demo/source/0001.txt").read_text(encoding="utf-8")
    dirty = Path("samples/public_demo/expected/dirty_translation.txt").read_text(encoding="utf-8")
    glossary = load_glossary("samples/public_demo/terms/master_glossary.txt")
    before = run_translation_qa(
        run_id="test",
        story_slug="public_demo",
        chapter="0001",
        source_text=source,
        translated_text=dirty,
        glossary=glossary,
    )
    after = run_translation_qa(
        run_id="test",
        story_slug="public_demo",
        chapter="0001",
        source_text=source,
        translated_text=dirty,
        glossary=glossary,
    )

    assert validate_patch_improves_qa(before_report=before, after_report=after) is False


def test_heading_repair_never_emits_placeholder_text() -> None:
    patch = OfflineRepairProvider().propose_patch(
        chapter="0042",
        source_text="第一章 模拟器启动\n\n正文",
        translation_text="Chapter One: Deduction Begins\n\nBody.",
        finding=QAFinding(
            check_id="heading_format",
            severity="warning",
            message="Bad heading.",
            location=QALocation(chapter="0042", paragraph_index=0),
            found="Chapter One: Deduction Begins",
            expected="Chapter N or Chapter N: Subtitle",
            auto_repairable=True,
        ),
        glossary=load_glossary("samples/public_demo/terms/master_glossary.txt"),
    )

    assert patch is not None
    assert patch.new_text.startswith("Chapter 42")
    assert "Chapter N" not in patch.new_text


def test_chinese_punctuation_repair_uses_character_map() -> None:
    patch = OfflineRepairProvider().propose_patch(
        chapter="0001",
        source_text="",
        translation_text='Chapter 1\n\nHe said：“Begin simulation！”',
        finding=QAFinding(
            check_id="chinese_punctuation",
            severity="warning",
            message="Chinese punctuation.",
            location=QALocation(chapter="0001", paragraph_index=1, snippet='He said：“Begin simulation！”'),
            found="：“！”",
            expected="English punctuation",
            auto_repairable=True,
        ),
        glossary=load_glossary("samples/public_demo/terms/master_glossary.txt"),
    )

    assert patch is not None
    assert patch.new_text == 'He said: "Begin simulation!"'
    assert "English punctuation" not in patch.new_text


def test_residual_chinese_without_expected_is_not_routed_to_offline_auto_repair() -> None:
    finding = QAFinding(
        check_id="residual_chinese",
        severity="warning",
        message="Residual Chinese.",
        location=QALocation(chapter="0001", paragraph_index=1, snippet="未知词"),
        found="未知词",
        expected=None,
        auto_repairable=True,
    )

    assert route_repair_strategy(finding, provider_mode="offline") == "human_review"


def test_router_sends_mechanical_and_ambiguous_findings_to_distinct_strategies() -> None:
    mechanical = QAFinding(
        check_id="blocked_glossary_variant",
        severity="warning",
        message="Blocked term.",
        location=QALocation(chapter="0001", paragraph_index=1),
        found="Way of Heaven",
        expected="Heavenly Dao",
        auto_repairable=True,
    )
    ambiguous = QAFinding(
        check_id="residual_chinese",
        severity="warning",
        message="Residual Chinese.",
        location=QALocation(chapter="0001", paragraph_index=1),
        found="天道",
        expected="Heavenly Dao",
        auto_repairable=True,
    )

    assert route_repair_strategy(mechanical, provider_mode="offline") == "rule"
    assert route_repair_strategy(ambiguous, provider_mode="live") == "candidate_selection"


def test_panel_count_routes_to_candidate_selection() -> None:
    finding = QAFinding(
        check_id="system_panel_count",
        severity="warning",
        message="Panel mismatch.",
        location=QALocation(chapter="0001"),
        found="1",
        expected="2",
        auto_repairable=True,
    )

    assert route_repair_strategy(finding, provider_mode="offline") == "candidate_selection"


def test_glossary_required_known_alias_routes_to_rule_repair() -> None:
    glossary = parse_glossary_text("仙盟: Immortal Alliance, Fairy Alliance\n")
    source = "第一章\n\n仙盟集结。"
    translated = "Chapter 1\n\nThe Fairy Alliance mobilized."
    before = run_translation_qa(
        run_id="alias",
        story_slug="story",
        chapter="0001",
        source_text=source,
        translated_text=translated,
        glossary=glossary,
    )
    finding = next(finding for finding in before.findings if finding.check_id == "glossary_required")

    assert finding.auto_repairable is True
    assert finding.found == "Fairy Alliance"
    assert finding.expected == "Immortal Alliance"
    assert route_repair_strategy(finding, provider_mode="offline") == "rule"

    patch = OfflineRepairProvider().propose_patch(
        chapter="0001",
        source_text=source,
        translation_text=translated,
        finding=finding,
        glossary=glossary,
    )

    assert patch is not None
    assert patch.old_text == "Fairy Alliance"
    assert patch.new_text == "Immortal Alliance"
    after_text = apply_patch(translated, patch)
    after = run_translation_qa(
        run_id="alias",
        story_slug="story",
        chapter="0001",
        source_text=source,
        translated_text=after_text,
        glossary=glossary,
    )
    assert after.summary.total_findings == 0


def test_glossary_required_alias_uses_observed_translation_casing_for_patch() -> None:
    glossary = parse_glossary_text("仙盟: Immortal Alliance, Fairy Alliance\n")
    report = run_translation_qa(
        run_id="alias-case",
        story_slug="story",
        chapter="0001",
        source_text="第一章\n\n仙盟集结。",
        translated_text="Chapter 1\n\nThe fairy alliance mobilized.",
        glossary=glossary,
    )
    finding = next(finding for finding in report.findings if finding.check_id == "glossary_required")

    assert finding.auto_repairable is True
    assert finding.found == "fairy alliance"
    patch = OfflineRepairProvider().propose_patch(
        chapter="0001",
        source_text="第一章\n\n仙盟集结。",
        translation_text="Chapter 1\n\nThe fairy alliance mobilized.",
        finding=finding,
        glossary=glossary,
    )

    assert patch is not None
    assert apply_patch("Chapter 1\n\nThe fairy alliance mobilized.", patch) == "Chapter 1\n\nThe Immortal Alliance mobilized."


def test_glossary_required_detects_cross_glossary_alias_when_source_term_absent() -> None:
    glossary = parse_glossary_text("法則之主: Lord of Laws\n法主: Law Lord\n")
    source = "第一章\n\n法則之主降临。"
    translated = "Chapter 1\n\nThe Law Lord descended."

    report = run_translation_qa(
        run_id="cross-alias",
        story_slug="story",
        chapter="0001",
        source_text=source,
        translated_text=translated,
        glossary=glossary,
    )
    finding = next(finding for finding in report.findings if finding.check_id == "glossary_required")

    assert finding.auto_repairable is True
    assert finding.found == "Law Lord"
    assert finding.expected == "Lord of Laws"
    assert route_repair_strategy(finding, provider_mode="offline") == "rule"


def test_glossary_required_does_not_use_cross_alias_when_alias_source_term_is_present() -> None:
    glossary = parse_glossary_text("法則之主: Lord of Laws\n法主: Law Lord\n")
    source = "第一章\n\n法則之主和法主都在场。"
    translated = "Chapter 1\n\nThe Law Lord was also present."

    report = run_translation_qa(
        run_id="cross-alias-blocked",
        story_slug="story",
        chapter="0001",
        source_text=source,
        translated_text=translated,
        glossary=glossary,
    )
    findings = [finding for finding in report.findings if finding.check_id == "glossary_required"]
    target_finding = next(finding for finding in findings if finding.expected == "Lord of Laws")

    assert target_finding.auto_repairable is False
    assert target_finding.found == "法則之主"


def test_glossary_required_alias_conflicting_with_present_canonical_term_requires_review() -> None:
    glossary = parse_glossary_text("煞氣: baleful qi, Black Baleful Stone\n黑煞石: Black Baleful Stone\n")
    source = "第一章\n\n煞氣和黑煞石都出现了。"
    translated = "Chapter 1\n\nThe Black Baleful Stone appeared."

    report = run_translation_qa(
        run_id="alias-conflict",
        story_slug="story",
        chapter="0001",
        source_text=source,
        translated_text=translated,
        glossary=glossary,
    )
    finding = next(finding for finding in report.findings if finding.expected == "baleful qi")

    assert finding.found == "Black Baleful Stone"
    assert finding.auto_repairable is False
    assert route_repair_strategy(finding, provider_mode="offline") == "human_review"


def test_repair_queue_skips_conflicting_alias_and_repairs_independent_alias() -> None:
    glossary = parse_glossary_text("甲: Alpha, SharedAlias\n乙: Beta, AliasY\n丙: SharedAlias\n")

    result = _apply_repair_queue(
        run_id="conflict_then_continue",
        story_slug="story",
        chapter="0001",
        source_text="第1章\n\n甲乙丙",
        initial_text="Chapter 1\n\nSharedAlias and AliasY",
        glossary=glossary,
        repair_provider=OfflineRepairProvider(),
        judge_provider=OfflineJudgeProvider(),
        provider_mode="offline",
        allow_live_provider_fallback=False,
        live_provider_fallback_state=None,
        seed=7,
        max_repairs=3,
    )

    rejected = [attempt for attempt in result.attempts if attempt.accepted is False]
    accepted = [attempt for attempt in result.attempts if attempt.accepted is True]
    assert rejected == []
    assert accepted
    assert accepted[0].patch is not None
    assert accepted[0].patch.old_text == "AliasY"
    assert "Beta" in result.text
    assert "AliasY" not in result.text
    assert "SharedAlias" in result.text


def test_repair_queue_rejects_missing_patch_span_instead_of_raising() -> None:
    glossary = parse_glossary_text("法主: Lord of Laws, Law Lord\n")

    result = _apply_repair_queue(
        run_id="missing_span",
        story_slug="story",
        chapter="0001",
        source_text="第1章\n\n法主",
        initial_text="Chapter 1\n\nThose Law Lords monopolized everything.",
        glossary=glossary,
        repair_provider=MissingSpanRepairProvider(),
        judge_provider=OfflineJudgeProvider(),
        provider_mode="offline",
        allow_live_provider_fallback=False,
        live_provider_fallback_state=None,
        seed=7,
        max_repairs=2,
    )

    assert result.text == "Chapter 1\n\nThose Law Lords monopolized everything."
    assert result.qa_report.summary.total_findings == 1
    assert result.attempts
    assert result.attempts[0].accepted is False
    assert result.attempts[0].patch is not None
    assert result.attempts[0].patch.accepted is False
    assert "Patch old_text not found" in result.attempts[0].reason


def test_glossary_required_without_observed_alias_stays_human_review() -> None:
    glossary = parse_glossary_text("仙盟: Immortal Alliance, Fairy Alliance\n")
    report = run_translation_qa(
        run_id="missing",
        story_slug="story",
        chapter="0001",
        source_text="第一章\n\n仙盟集结。",
        translated_text="Chapter 1\n\nThe sect mobilized.",
        glossary=glossary,
    )
    finding = next(finding for finding in report.findings if finding.check_id == "glossary_required")

    assert finding.auto_repairable is False
    assert finding.found == "仙盟"
    assert route_repair_strategy(finding, provider_mode="offline") == "human_review"
