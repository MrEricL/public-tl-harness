from __future__ import annotations

from pathlib import Path

from agentic_translation.glossary import load_glossary, parse_glossary_text
from agentic_translation.models import QAFinding
from agentic_translation.qa import panel_count, run_source_qa, run_translation_qa, weighted_score


def test_source_qa_accepts_clean_public_demo_source() -> None:
    source = Path("samples/public_demo/source/0001.txt").read_text(encoding="utf-8")

    report = run_source_qa(
        run_id="test",
        story_slug="public_demo",
        chapter="0001",
        source_text=source,
    )

    assert report.summary.total_findings == 0
    assert report.panel_count == 2


def test_translation_qa_flags_dirty_baseline_failures() -> None:
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

    check_ids = {finding.check_id for finding in report.findings}
    assert "heading_format" in check_ids
    assert "residual_chinese" in check_ids
    assert "chinese_punctuation" in check_ids
    assert "blocked_glossary_variant" in check_ids
    assert "glossary_required" in check_ids
    assert "system_panel_count" in check_ids


def test_panel_count_handles_suffix_and_multiline_source_panels() -> None:
    # Authored technical fixture: brackets and paragraph layout are the inputs.
    text = """第1章

【溫度感測器（室內）】：讀數20攝氏度。

【註：1，每次測試只記錄一次溫度。

2，電源中斷時，測試立即停止。

3，測試期間，採樣間隔保持不變。】
"""

    assert panel_count(text) == 2


def test_panel_count_ignores_suffix_false_positive() -> None:
    glossary = load_glossary("samples/public_demo/terms/master_glossary.txt")
    source = "第1章\n\n【溫度感測器（室內）】：讀數20攝氏度。\n\n【電池電壓（待機）】：讀數3伏特。"
    translated = "Chapter 1\n\n[Temperature sensor (indoors): Reading, 20 degrees Celsius.]\n\n[Battery voltage (standby): Reading, 3 volts.]"

    report = run_translation_qa(
        run_id="panel-suffix",
        story_slug="story",
        chapter="0001",
        source_text=source,
        translated_text=translated,
        glossary=glossary,
    )

    assert "system_panel_count" not in {finding.check_id for finding in report.findings}


def test_glossary_translation_improves_but_still_needs_repair() -> None:
    source = Path("samples/public_demo/source/0001.txt").read_text(encoding="utf-8")
    dirty = Path("samples/public_demo/expected/dirty_translation.txt").read_text(encoding="utf-8")
    glossary_text = Path("samples/public_demo/expected/glossary_translation.txt").read_text(encoding="utf-8")
    glossary = load_glossary("samples/public_demo/terms/master_glossary.txt")

    baseline = run_translation_qa(
        run_id="test",
        story_slug="public_demo",
        chapter="0001",
        source_text=source,
        translated_text=dirty,
        glossary=glossary,
    )
    improved = run_translation_qa(
        run_id="test",
        story_slug="public_demo",
        chapter="0001",
        source_text=source,
        translated_text=glossary_text,
        glossary=glossary,
    )

    assert weighted_score(improved) > weighted_score(baseline)
    assert {finding.check_id for finding in improved.findings} == {"heading_format", "system_panel_count"}


def test_source_only_glossary_required_keeps_source_paragraph_index_for_triage() -> None:
    glossary = load_glossary("samples/public_demo/terms/master_glossary.txt")
    source = "第1章\n\n前文。\n\n天道在这里出现。\n\n后文。"
    translated = "Chapter 1\n\nEarlier.\n\nDao appears here.\n\nLater."

    report = run_translation_qa(
        run_id="triage-index",
        story_slug="story",
        chapter="0001",
        source_text=source,
        translated_text=translated,
        glossary=glossary,
    )
    finding = next(finding for finding in report.findings if finding.check_id == "glossary_required")

    assert finding.found == "天道"
    assert finding.location.paragraph_index == 2


def test_translation_qa_preserves_first_contiguous_chinese_residue() -> None:
    glossary = load_glossary("samples/public_demo/terms/master_glossary.txt")

    report = run_translation_qa(
        run_id="test",
        story_slug="public_demo",
        chapter="0001",
        source_text="Chapter 1\n\n道心 and 天道.",
        translated_text="Chapter 1\n\nThe 道心 and 天道 remain.",
        glossary=glossary,
    )

    residual = next(finding for finding in report.findings if finding.check_id == "residual_chinese")
    assert residual.found == "道心"


def _glossary_findings(
    *,
    source: str,
    translated: str,
    glossary_text: str,
) -> list[QAFinding]:
    report = run_translation_qa(
        run_id="nested-glossary",
        story_slug="invented-story",
        chapter="0001",
        source_text=source,
        translated_text=translated,
        glossary=parse_glossary_text(glossary_text),
    )
    return [finding for finding in report.findings if finding.check_id == "glossary_required"]


def test_nested_glossary_source_does_not_require_short_term_target() -> None:
    findings = _glossary_findings(
        source="第1章\n\n星河仙宗召集了弟子。",
        translated="Chapter 1\n\nThe Astral Immortal Sect summoned its disciples.",
        glossary_text="星河仙宗 -> Astral Immortal Sect\n河仙 -> River Immortal",
    )

    assert findings == []


def test_nested_glossary_source_still_requires_separate_short_term_occurrence() -> None:
    findings = _glossary_findings(
        source="第1章\n\n星河仙宗召集了弟子。\n\n河仙随后到场。",
        translated=(
            "Chapter 1\n\nThe Astral Immortal Sect summoned its disciples."
            "\n\nAnother cultivator arrived later."
        ),
        glossary_text="星河仙宗 -> Astral Immortal Sect\n河仙 -> River Immortal",
    )

    assert [(finding.found, finding.expected) for finding in findings] == [
        ("河仙", "River Immortal")
    ]
    assert findings[0].location.paragraph_index == 2


def test_nested_glossary_source_still_requires_missing_long_term() -> None:
    findings = _glossary_findings(
        source="第1章\n\n星河仙宗召集了弟子。",
        translated="Chapter 1\n\nThe order summoned its disciples.",
        glossary_text="星河仙宗 -> Astral Immortal Sect\n河仙 -> River Immortal",
    )

    assert [(finding.found, finding.expected) for finding in findings] == [
        ("星河仙宗", "Astral Immortal Sect")
    ]


def test_nested_glossary_suppression_is_independent_of_entry_order() -> None:
    source = "第1章\n\n星河仙宗召集了弟子。"
    translated = "Chapter 1\n\nThe order summoned its disciples."
    glossary_lines = [
        "星河仙宗 -> Astral Immortal Sect",
        "河仙 -> River Immortal",
    ]

    findings_by_order = [
        _glossary_findings(
            source=source,
            translated=translated,
            glossary_text="\n".join(lines),
        )
        for lines in (glossary_lines, list(reversed(glossary_lines)))
    ]

    assert [
        {(finding.found, finding.expected) for finding in findings}
        for findings in findings_by_order
    ] == [
        {('星河仙宗', "Astral Immortal Sect")},
        {('星河仙宗', "Astral Immortal Sect")},
    ]


def test_suppressed_short_term_is_not_cross_glossary_repair_alias() -> None:
    findings = _glossary_findings(
        source="第1章\n\n星河仙宗召集了弟子。",
        translated="Chapter 1\n\nThe River Immortal Sect summoned its disciples.",
        glossary_text=(
            "星河仙宗 -> Star River Immortal Sect\n"
            "河仙 -> River Immortal Sect"
        ),
    )

    assert len(findings) == 1
    assert findings[0].found == "星河仙宗"
    assert findings[0].auto_repairable is False


def test_suppressed_short_term_does_not_block_safe_long_term_alias_repair() -> None:
    findings = _glossary_findings(
        source="第1章\n\n星河仙宗召集了弟子。",
        translated="Chapter 1\n\nThe River Immortal Sect summoned its disciples.",
        glossary_text=(
            "星河仙宗: Star River Immortal Sect, River Immortal Sect\n"
            "河仙 -> River Immortal Sect"
        ),
    )

    assert len(findings) == 1
    assert findings[0].found == "River Immortal Sect"
    assert findings[0].expected == "Star River Immortal Sect"
    assert findings[0].auto_repairable is True


def test_partially_overlapping_glossary_sources_remain_independent() -> None:
    findings = _glossary_findings(
        source="第1章\n\n星河仙出现了。",
        translated="Chapter 1\n\nSomeone appeared.",
        glossary_text="星河 -> Star River\n河仙 -> River Immortal",
    )

    assert {(finding.found, finding.expected) for finding in findings} == {
        ("星河", "Star River"),
        ("河仙", "River Immortal"),
    }
