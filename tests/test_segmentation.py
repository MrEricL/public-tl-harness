from __future__ import annotations

import json

import pytest

from agentic_translation.segmentation import (
    DraftSegment,
    make_context_receipt,
    parse_draft_envelope,
    receipt_is_current,
    render_draft,
    segment_source,
    validate_draft_segments,
    validate_source_segments,
)


def test_source_segments_cover_chapter_without_changing_paragraphs() -> None:
    source = ("天地玄黃，宇宙洪荒。" * 50) + "\n\n" + ("日月盈昃，辰宿列張。" * 55) + "\n"
    segments = segment_source("work", "0001", source)
    assert len(segments) > 1
    assert "".join(segment.text for segment in segments) == source
    assert all(segment.source_end - segment.source_start <= 900 for segment in segments)
    assert [segment.segment_id for segment in segments] == [f"s{number:04d}" for number in range(1, len(segments) + 1)]
    validate_source_segments(source, segments)
    with pytest.raises(ValueError, match="gap, overlap"):
        validate_source_segments(source, segments[1:])
    with pytest.raises(ValueError, match="Duplicate source"):
        validate_source_segments(source, [segments[0], segments[0]])


def test_boundary_and_trailing_unicode_blanks_remain_with_content() -> None:
    # The first line closes exactly at the target, so the following blanks
    # would otherwise become an untranslatable source segment.
    source = "甲" * 399 + "\n\u3000\n\u00a0\t  "
    segments = segment_source("work", "0001", source, target_chars=400, max_chars=900)
    assert len(segments) == 1
    assert segments[0].text == source
    assert segments[0].source_start == 0
    assert segments[0].source_end == len(source)
    assert segments[0].paragraph_ids == ("p0001", "p0002", "p0003")
    assert segments[0].source_sha256


def test_leading_blanks_attach_to_first_content_even_past_soft_limit() -> None:
    source = "\u3000\n" + "乙" * 400
    segments = segment_source("work", "0001", source, target_chars=400, max_chars=400)
    assert len(segments) == 1
    assert segments[0].text == source
    assert segments[0].source_end == len(source)
    with pytest.raises(ValueError, match="source text with content"):
        segment_source("work", "0001", "\u3000\n\u00a0\t")


def test_common_draft_envelope_requires_exact_coverage_and_keeps_prose() -> None:
    source = segment_source("work", "0001", "甲。\n乙。", target_chars=3, max_chars=3)
    payload = {"segments": [
        {"segment_id": source[1].segment_id, "translated_text": "Second.  "},
        {"segment_id": source[0].segment_id, "translated_text": "First!"},
    ]}
    drafts = parse_draft_envelope("```json\n" + json.dumps(payload) + "\n```", source)
    assert [segment.segment_id for segment in drafts] == [segment.segment_id for segment in source]
    assert render_draft(drafts) == "First!\n\nSecond.  "
    validate_draft_segments(drafts, source)
    with pytest.raises(ValueError, match="order/coverage"):
        render_draft(drafts[::-1], source_segments=source)
    with pytest.raises(ValueError, match="Duplicate draft"):
        parse_draft_envelope({"segments": [payload["segments"][0]] * 2}, source)
    with pytest.raises(ValueError, match="coverage mismatch"):
        parse_draft_envelope({"segments": payload["segments"][:1]}, source)


def test_context_receipt_tracks_neighbors_memory_and_policy() -> None:
    source = segment_source("work", "0001", "甲。\n乙。", target_chars=3, max_chars=3)
    drafts = [DraftSegment.from_text(segment.segment_id, text) for segment, text in zip(source, ["First.", "Second."])]
    kwargs = {
        "neighbors": ((source[1], drafts[1]),),
        "memory_snapshot_sha256": "memory-a",
        "memory_entry_hashes": ("entry-a",),
        "instruction_version": "v1",
        "review_policy": "jev-adaptive-v1",
        "evidence_level": "jev-screen",
    }
    receipt = make_context_receipt(source[0], drafts[0], **kwargs)
    assert receipt_is_current(receipt, source[0], drafts[0], **{key: value for key, value in kwargs.items() if key != "evidence_level"})
    stale_neighbor = ((source[1], drafts[1].with_text("Changed second.")),)
    assert not receipt_is_current(receipt, source[0], drafts[0], neighbors=stale_neighbor, memory_snapshot_sha256="memory-a", memory_entry_hashes=("entry-a",), instruction_version="v1", review_policy="jev-adaptive-v1")
    assert not receipt_is_current(receipt, source[0], drafts[0], neighbors=kwargs["neighbors"], memory_snapshot_sha256="memory-b", memory_entry_hashes=("entry-a",), instruction_version="v1", review_policy="jev-adaptive-v1")
    assert not receipt_is_current(receipt, source[0], drafts[0], neighbors=kwargs["neighbors"], memory_snapshot_sha256="memory-a", memory_entry_hashes=("entry-a",), instruction_version="v2", review_policy="jev-adaptive-v1")
