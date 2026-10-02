from __future__ import annotations

import json

import pytest

from agentic_translation.segmentation import segment_source
from agentic_translation.story_memory import (
    MemoryEntry,
    SourceEvidence,
    StoryMemoryTape,
    TerminologyCandidate,
    build_source_memory,
)


def _entry(entry_id: str, chapter_id: str, segment_id: str, excerpt: str, *, scope: str = "narrator", parent: str | None = None) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id,
        kind="identity",
        subject="阿蘭",
        value="Alan is the traveler",
        source_evidence=(SourceEvidence(chapter_id, segment_id, excerpt),),
        known_from_chapter=chapter_id,
        known_from_segment=segment_id,
        scope=scope,
        status="active",
        created_by_model="test-extractor",
        policy_version="source-memory-v1",
        parent_entry_id=parent,
        source_aliases=("阿蘭",),
    )


def test_memory_rejects_future_and_inexact_evidence() -> None:
    chapter = segment_source("work", "0001", "阿蘭走進城。\n他看見玉牌。", target_chars=7, max_chars=7)
    tape = StoryMemoryTape()
    future = _entry("e1", "0001", chapter[0].segment_id, "玉牌")
    with pytest.raises(ValueError, match="exactly"):
        tape.add_chapter("0001", chapter, (future,))
    assert tape.chapter_ids == ()
    future_chapter = _entry("e2", "0002", "s0001", "阿蘭")
    with pytest.raises(ValueError, match="current chapter"):
        tape.add_chapter("0001", chapter, (future_chapter,))
    later_segment = MemoryEntry(
        **{**_entry("e3", "0001", chapter[0].segment_id, "阿蘭").__dict__,
           "source_evidence": (SourceEvidence("0001", chapter[1].segment_id, "玉牌"),)}
    )
    with pytest.raises(ValueError, match="after known_from"):
        tape.add_chapter("0001", chapter, (later_segment,))


def test_memory_preserves_belief_scope_and_causal_visibility(tmp_path) -> None:
    chapter = segment_source("work", "0001", "阿蘭說自己是王。\n阿蘭後來承認他說謊。", target_chars=9, max_chars=13)
    tape = StoryMemoryTape(run_dir=tmp_path)
    belief = _entry("belief", "0001", chapter[0].segment_id, "阿蘭說自己是王", scope="dialogue_claim")
    later = _entry("later", "0001", chapter[1].segment_id, "阿蘭後來承認")
    bad_promotion = _entry("fact", "0001", chapter[0].segment_id, "阿蘭說自己是王", parent="belief")
    with pytest.raises(ValueError, match="without new source evidence"):
        tape.add_chapter("0001", chapter, (belief, bad_promotion))
    snapshot = tape.add_chapter("0001", chapter, (belief, later))
    early_pack = tape.context_for_segment("0001", chapter[0].segment_id)
    assert "belief" in early_pack.entry_ids
    assert "later" not in early_pack.entry_ids
    assert "dialogue_claim" in early_pack.text
    late_pack = tape.context_for_segment("0001", chapter[1].segment_id)
    assert "later" in late_pack.entry_ids
    assert snapshot.snapshot_sha256 == json.loads((tmp_path / "story_memory" / "snapshots" / "0001.json").read_text())["snapshot_sha256"]
    assert (tmp_path / "story_memory" / "updates.jsonl").read_text().count("memory_entry") == 2


def test_run_local_terms_keep_existing_choice_without_adjudication() -> None:
    first = segment_source("work", "0001", "阿蘭使出星河步。")
    second = segment_source("work", "0002", "星河步再次出現。")
    tape = StoryMemoryTape()
    evidence_1 = SourceEvidence("0001", first[0].segment_id, "星河步")
    evidence_2 = SourceEvidence("0002", second[0].segment_id, "星河步")
    tape.add_chapter("0001", first, terminology_candidates=(TerminologyCandidate("星河步", "Starstream Step", (evidence_1,)),))
    snapshot = tape.add_chapter("0002", second, terminology_candidates=(TerminologyCandidate("星河步", "Galaxy Step", (evidence_2,)),))
    assert tape.glossary_for_segment("0002", second[0].segment_id) == {"星河步": "Starstream Step"}
    assert snapshot.terminology[-1].outcome == "kept_existing"
    assert snapshot.terminology[-1].decision_actor == "automation"
    assert snapshot.terminology[-1].decision_basis == "source_grounded_policy"
    changed = tape.context_for_segment("0002", second[0].segment_id)
    assert "Starstream Step" in changed.text
    assert changed.budget_kind == "characters_not_tokens"


def test_term_change_requires_concrete_reason_and_is_visible_from_its_segment() -> None:
    first = segment_source("work", "0001", "星河步出現。")
    second = segment_source("work", "0002", "他說話。\n星河步是另一種招式。", target_chars=5, max_chars=15)
    tape = StoryMemoryTape()
    tape.add_chapter("0001", first, terminology_candidates=(TerminologyCandidate("星河步", "Starstream Step", (SourceEvidence("0001", "s0001", "星河步"),)),))
    updated = TerminologyCandidate(
        "星河步", "Star River Step", (SourceEvidence("0002", second[1].segment_id, "星河步"),),
        adjudication_reason="scope_change", adjudication_detail="Source identifies a different technique here.",
    )
    tape.add_chapter("0002", second, terminology_candidates=(updated,))
    assert tape.glossary_for_segment("0002", second[0].segment_id)["星河步"] == "Starstream Step"
    assert tape.glossary_for_segment("0002", second[1].segment_id)["星河步"] == "Star River Step"
    assert tape.snapshot_for_chapter("0002").terminology[-1].outcome == "changed"


def test_context_pack_is_bounded_and_source_extractor_runs_in_order() -> None:
    chapters = [(f"{number:04d}", segment_source("work", f"{number:04d}", "阿蘭走進城。")) for number in (1, 2)]
    seen: list[tuple[str, str | None]] = []

    def extract(chapter_id, segments, previous):
        seen.append((chapter_id, previous.chapter_id if previous else None))
        return (_entry(chapter_id, chapter_id, segments[0].segment_id, "阿蘭"),), ()

    tape = build_source_memory(chapters, extract)
    assert seen == [("0001", None), ("0002", "0001")]
    pack = tape.context_for_segment("0002", "s0001", max_context_chars=50)
    assert len(pack.text) <= 50
    assert pack.truncated
    assert tape.snapshot_for_chapter("0002").preceding_snapshot_sha256 == tape.snapshot_for_chapter("0001").snapshot_sha256
