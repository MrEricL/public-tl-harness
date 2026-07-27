from __future__ import annotations

from pathlib import Path

from agentic_translation.glossary import load_glossary
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
    text = """第1章

【說服者（白色天賦）】：售價10能量值。

【註：1，宿主每次模擬僅可進行一次深度模擬。

2，深度模擬狀態下宿主死亡，深度模擬將直接結束。

3，深度模擬狀態下，時間流速保持一致。】
"""

    assert panel_count(text) == 2


def test_panel_count_ignores_suffix_false_positive() -> None:
    glossary = load_glossary("samples/public_demo/terms/master_glossary.txt")
    source = "第1章\n\n【說服者（白色天賦）】：售價10能量值。\n\n【團隊管理經驗（十個月）】：售價1能量值。"
    translated = "Chapter 1\n\n[Persuader (white talent): Price, 10 energy value.]\n\n[Team management experience (ten months): Price, 1 energy value.]"

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
