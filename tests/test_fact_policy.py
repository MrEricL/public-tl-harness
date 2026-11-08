"""Memory pack policy regressions using authored data only."""
from dataclasses import replace

import pytest

from agentic_translation.segmentation import segment_source
from agentic_translation.story_memory import (FACT_POLICIES, MemoryEntry, SourceEvidence, StoryMemoryTape,
                                              TerminologyCandidate, order_facts)


def _fact(entry_id, kind, value, chapter, excerpt, subject="阿蘭"):
    return MemoryEntry(entry_id=entry_id, kind=kind, subject=subject, value=value,
                       source_evidence=(SourceEvidence(chapter, "s0001", excerpt),),
                       known_from_chapter=chapter, known_from_segment="s0001", scope="narrator", status="active",
                       created_by_model="test", policy_version="source-memory-v2", source_aliases=(subject,))


def test_order_facts():
    old_state = _fact("e1", "state", "a guard", "0001", "x")
    event = _fact("e2", "event", "left town", "0001", "x")
    identity = _fact("e3", "identity", "the prince", "0002", "x")
    new_state = _fact("e4", "state", "a general", "0003", "x")
    visible = [old_state, event, identity, new_state]
    assert order_facts(visible, visible, "legacy") == visible
    assert order_facts(visible, visible, "terms_only") == []
    # Only the newest state per subject; stable kinds, then states, then events; newest first in a group.
    assert [e.entry_id for e in order_facts(visible, visible, "current_first")] == ["e3", "e4", "e2"]
    with pytest.raises(ValueError):
        order_facts(visible, visible, "newest")


def _tape():
    tape = StoryMemoryTape(policy_version="source-memory-v2")
    chapters = [segment_source("work", f"000{i}", text) for i, text in
                enumerate(["阿蘭當了守衛。星河步出現。", "阿蘭離開了城。", "阿蘭成了將軍。星河步再現。"], 1)]
    tape.add_chapter("0001", chapters[0], (_fact("e1", "state", "a guard", "0001", "阿蘭當了守衛"),),
                     terminology_candidates=(TerminologyCandidate("星河步", "Starstream Step",
                                                                  (SourceEvidence("0001", "s0001", "星河步"),)),))
    tape.add_chapter("0002", chapters[1], (_fact("e2", "event", "left town", "0002", "阿蘭離開了城"),))
    tape.add_chapter("0003", chapters[2], (_fact("e3", "state", "a general", "0003", "阿蘭成了將軍"),))
    return tape


def test_pack_by_policy_and_legacy_hash_unchanged():
    tape = _tape()
    default = tape.context_for_segment("0003", "s0001")
    legacy = tape.context_for_segment("0003", "s0001", fact_policy="legacy")
    current = tape.context_for_segment("0003", "s0001", fact_policy="current_first")
    names = tape.context_for_segment("0003", "s0001", fact_policy="terms_only")
    assert default == legacy                                   # same text and same dependency hash
    assert legacy.text.index("a guard") < legacy.text.index("a general")      # legacy: oldest first
    assert "a guard" not in current.text and "a general" in current.text      # stale state dropped
    assert names.text.splitlines() == ["TERM 星河步 -> Starstream Step"]
    assert current.dependency_sha256 != legacy.dependency_sha256
    assert set(FACT_POLICIES) == {"legacy", "current_first", "terms_only"}
