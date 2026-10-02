"""Tests for the story-bible export tool, built on a synthetic StoryMemoryTape.

Snapshots are written the way the unattended web-novel runner persists
them: ``<run_dir>/story_memory/<work_id>/<chapter_id>.json``, one
file per chapter, each holding the tape's cumulative ``MemorySnapshot``.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from agentic_translation.segmentation import segment_source
from agentic_translation.story_bible import render_bible
from agentic_translation.story_memory import (
    MemoryEntry,
    SourceEvidence,
    StoryMemoryTape,
    TerminologyCandidate,
)


def _write_snapshot(run_dir, work_id, snapshot):
    path = run_dir / "story_memory" / work_id / f"{snapshot.chapter_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(snapshot), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _entry(entry_id, chapter_id, segment_id, excerpt, *, kind="entity", subject="阿蘭",
          value="Alan is the traveler", scope="narrator", parent=None, aliases=()):
    return MemoryEntry(
        entry_id=entry_id, kind=kind, subject=subject, value=value,
        source_evidence=(SourceEvidence(chapter_id, segment_id, excerpt),),
        known_from_chapter=chapter_id, known_from_segment=segment_id,
        scope=scope, status="active", created_by_model="test-extractor",
        policy_version="source-memory-v1", parent_entry_id=parent, source_aliases=aliases,
    )


def _build_two_chapter_work(tmp_path, work_id="work_a"):
    """Chapter 1 introduces a character, a location, and a term; chapter 2
    changes the term's rendering and adds an organization."""
    ch1 = segment_source(work_id, "0001", "阿蘭走進城。\n星河步出现了。", target_chars=7, max_chars=7)
    ch2 = segment_source(work_id, "0002", "阿蘭加入了天音门。\n星河步再次施展。", target_chars=9, max_chars=9)

    tape = StoryMemoryTape()
    person = _entry("work_a_0001_e0", "0001", ch1[0].segment_id, "阿蘭", kind="entity",
                    subject="阿蘭", value="A traveler entering the city")
    location = _entry("work_a_0001_e1", "0001", ch1[0].segment_id, "阿蘭走進城", kind="location",
                      subject="城", value="A walled city")
    term1 = TerminologyCandidate("星河步", "Starstream Step",
                                 (SourceEvidence("0001", ch1[1].segment_id, "星河步"),))
    snap1 = tape.add_chapter("0001", ch1, (person, location), (term1,))
    _write_snapshot(tmp_path, work_id, snap1)

    org = _entry("work_a_0002_e0", "0002", ch2[0].segment_id, "天音门", kind="organization",
                subject="天音门", value="A sect Alan joins")
    term2 = TerminologyCandidate(
        "星河步", "Star River Step", (SourceEvidence("0002", ch2[1].segment_id, "星河步"),),
        adjudication_reason="scope_change", adjudication_detail="Source reveals a distinct variant here.",
    )
    snap2 = tape.add_chapter("0002", ch2, (org,), (term2,))
    _write_snapshot(tmp_path, work_id, snap2)
    return tape


def test_render_bible_writes_all_four_files(tmp_path):
    _build_two_chapter_work(tmp_path)
    result = render_bible(tmp_path, "work_a")
    out = tmp_path / "bible" / "work_a"
    assert (out / "glossary.md").exists()
    assert (out / "entities.md").exists()
    assert (out / "changes.md").exists()
    assert (out / "bible.json").exists()
    assert result["chapters"] == 2
    assert result["terms"] == 1
    assert result["entities"] == 3


def test_glossary_reports_change_with_reason_and_chapter_coverage(tmp_path):
    _build_two_chapter_work(tmp_path)
    render_bible(tmp_path, "work_a")
    data = json.loads((tmp_path / "bible" / "work_a" / "bible.json").read_text(encoding="utf-8"))
    term = data["glossary"][0]
    assert term["source"] == "星河步"
    assert term["chosen_target"] == "Star River Step"
    assert term["first_seen_chapter"] == "0001"
    assert term["chapters_seen"] == 2
    assert len(term["changes"]) == 1
    change = term["changes"][0]
    assert change["from_target"] == "Starstream Step"
    assert change["to_target"] == "Star River Step"
    assert change["reason"] == "scope_change"
    glossary_md = (tmp_path / "bible" / "work_a" / "glossary.md").read_text(encoding="utf-8")
    assert "Starstream Step" in glossary_md and "Star River Step" in glossary_md
    assert "scope_change" in glossary_md


def test_entities_are_grouped_by_kind_with_scope_labeled(tmp_path):
    _build_two_chapter_work(tmp_path)
    render_bible(tmp_path, "work_a")
    data = json.loads((tmp_path / "bible" / "work_a" / "bible.json").read_text(encoding="utf-8"))
    entities = data["entities"]
    assert any(row["subject"] == "阿蘭" for row in entities["characters_entities"])
    assert any(row["subject"] == "城" for row in entities["locations"])
    assert any(row["subject"] == "天音门" for row in entities["organizations"])
    assert entities["techniques"] == []
    entities_md = (tmp_path / "bible" / "work_a" / "entities.md").read_text(encoding="utf-8")
    assert "Characters / Entities" in entities_md
    assert "narrator (established fact)" in entities_md


def test_character_belief_scope_is_labeled_distinctly(tmp_path):
    work_id = "work_b"
    ch1 = segment_source(work_id, "0001", "他说自己是王。")
    tape = StoryMemoryTape()
    belief = _entry("work_b_0001_e0", "0001", ch1[0].segment_id, "他说自己是王",
                    kind="identity", subject="他", value="Claims to be a king", scope="dialogue_claim")
    snap = tape.add_chapter("0001", ch1, (belief,))
    _write_snapshot(tmp_path, work_id, snap)
    render_bible(tmp_path, work_id)
    entities_md = (tmp_path / "bible" / work_id / "entities.md").read_text(encoding="utf-8")
    assert "dialogue_claim" in entities_md


def test_changes_log_is_chapter_by_chapter(tmp_path):
    _build_two_chapter_work(tmp_path)
    render_bible(tmp_path, "work_a")
    data = json.loads((tmp_path / "bible" / "work_a" / "bible.json").read_text(encoding="utf-8"))
    changes = {row["chapter_id"]: row for row in data["changes"]}
    assert set(changes) == {"0001", "0002"}
    assert len(changes["0001"]["added_entries"]) == 2
    assert changes["0001"]["terminology_events"][0]["outcome"] == "accepted"
    assert len(changes["0002"]["added_entries"]) == 1
    assert changes["0002"]["terminology_events"][0]["outcome"] == "changed"
    changes_md = (tmp_path / "bible" / "work_a" / "changes.md").read_text(encoding="utf-8")
    assert "Chapter `0001`" in changes_md
    assert "Chapter `0002`" in changes_md


def test_evidence_excerpts_over_thirty_chars_are_omitted(tmp_path):
    work_id = "work_c"
    long_excerpt = "这是一段用于测试的很长很长很长很长很长的源文摘录超过三十个字符"
    assert len(long_excerpt) > 30
    ch1 = segment_source(work_id, "0001", long_excerpt)
    tape = StoryMemoryTape()
    entry = _entry("work_c_0001_e0", "0001", ch1[0].segment_id, long_excerpt,
                   kind="entity", subject="测试", value="A long excerpt test subject")
    snap = tape.add_chapter("0001", ch1, (entry,))
    _write_snapshot(tmp_path, work_id, snap)
    render_bible(tmp_path, work_id)
    data = json.loads((tmp_path / "bible" / work_id / "bible.json").read_text(encoding="utf-8"))
    row = data["entities"]["characters_entities"][0]
    assert row["evidence_excerpts"] == []


def test_missing_work_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        render_bible(tmp_path, "nonexistent_work")


def test_custom_out_dir_is_respected(tmp_path):
    _build_two_chapter_work(tmp_path)
    out_dir = tmp_path / "custom_bible_output"
    render_bible(tmp_path, "work_a", out_dir=out_dir)
    assert (out_dir / "glossary.md").exists()
    assert not (tmp_path / "bible").exists()
