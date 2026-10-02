"""Stable source windows and a common chapter draft envelope.

Offsets are Python character offsets into the unmodified source. Segment IDs
belong to the source and survive changes to translated text.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceSegment:
    work_id: str
    chapter_id: str
    segment_id: str
    source_start: int
    source_end: int
    source_sha256: str
    paragraph_ids: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class DraftSegment:
    segment_id: str
    translated_text: str
    draft_sha256: str

    @classmethod
    def from_text(cls, segment_id: str, text: str) -> DraftSegment:
        if not segment_id or not text.strip():
            raise ValueError("Draft segment needs an ID and non-empty text")
        return cls(segment_id, text, sha256_text(text))

    def with_text(self, text: str) -> DraftSegment:
        return self.from_text(self.segment_id, text)


@dataclass(frozen=True)
class ContextReceipt:
    segment_id: str
    source_sha256: str
    draft_sha256: str
    included_neighbor_ids: tuple[str, ...]
    neighbor_hashes: tuple[tuple[str, str, str], ...]
    memory_snapshot_sha256: str
    memory_entry_hashes: tuple[str, ...]
    instruction_version: str
    review_policy: str
    evidence_level: str
    tool_contract_version: str
    truncated: bool
    dependency_sha256: str


def _source_spans(text: str, max_chars: int) -> list[tuple[int, int, str]]:
    """Return paragraph-shaped spans, splitting only oversized paragraphs."""
    spans: list[tuple[int, int, str]] = []
    offset = 0
    for paragraph_number, line in enumerate(text.splitlines(keepends=True), 1):
        paragraph_id = f"p{paragraph_number:04d}"
        start = offset
        offset += len(line)
        while start < offset:
            end = min(start + max_chars, offset)
            if end < offset:
                # Prefer a sentence boundary near the end of an oversized line.
                boundary = max((text.rfind(mark, start + max_chars // 2, end) for mark in "。！？.!?；;"), default=-1)
                if boundary >= start + max_chars // 2:
                    end = boundary + 1
            spans.append((start, end, paragraph_id))
            start = end
    return spans


def segment_source(
    work_id: str,
    chapter_id: str,
    text: str,
    *,
    target_chars: int = 650,
    max_chars: int = 900,
) -> tuple[SourceSegment, ...]:
    """Group source paragraphs into ordered, lossless screening windows."""
    if not work_id or not chapter_id or not text.strip():
        raise ValueError("work_id, chapter_id, and source text with content are required")
    if not 0 < target_chars <= max_chars:
        raise ValueError("Require 0 < target_chars <= max_chars")
    spans = _source_spans(text, max_chars)
    if not spans:
        raise ValueError("Source text has no paragraphs")
    groups: list[list[tuple[int, int, str]]] = []
    current: list[tuple[int, int, str]] = []
    for span in spans:
        if current and span[1] - current[0][0] > max_chars:
            groups.append(current)
            current = []
        current.append(span)
        if current[-1][1] - current[0][0] >= target_chars:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    # A source boundary can fall exactly before blank lines. Keep their bytes
    # and offsets, but never ask the writer for a translation of blank space.
    # The size limit is soft only for whitespace attached to a real segment.
    content_groups: list[list[tuple[int, int, str]]] = []
    leading_whitespace: list[tuple[int, int, str]] = []
    for group in groups:
        if not text[group[0][0]:group[-1][1]].strip():
            if content_groups:
                content_groups[-1].extend(group)
            else:
                leading_whitespace.extend(group)
            continue
        if leading_whitespace:
            group = leading_whitespace + group
            leading_whitespace = []
        content_groups.append(group)
    groups = content_groups
    segments: list[SourceSegment] = []
    for index, group in enumerate(groups, 1):
        start, end = group[0][0], group[-1][1]
        segment_text = text[start:end]
        segments.append(
            SourceSegment(
                work_id=work_id,
                chapter_id=chapter_id,
                segment_id=f"s{index:04d}",
                source_start=start,
                source_end=end,
                source_sha256=sha256_text(segment_text),
                paragraph_ids=tuple(dict.fromkeys(span[2] for span in group)),
                text=segment_text,
            )
        )
    validate_source_segments(text, segments)
    return tuple(segments)


def validate_source_segments(text: str, segments: Sequence[SourceSegment]) -> None:
    if not segments:
        raise ValueError("At least one source segment is required")
    expected_start = 0
    ids: set[str] = set()
    identity = (segments[0].work_id, segments[0].chapter_id)
    for segment in segments:
        if (segment.work_id, segment.chapter_id) != identity:
            raise ValueError("Source segments belong to different chapters")
        if segment.segment_id in ids:
            raise ValueError(f"Duplicate source segment ID: {segment.segment_id}")
        ids.add(segment.segment_id)
        if segment.source_start != expected_start or segment.source_end <= segment.source_start:
            raise ValueError("Source segments have a gap, overlap, or invalid offset")
        if text[segment.source_start:segment.source_end] != segment.text:
            raise ValueError(f"Source segment text differs at {segment.segment_id}")
        if sha256_text(segment.text) != segment.source_sha256:
            raise ValueError(f"Source segment hash differs at {segment.segment_id}")
        expected_start = segment.source_end
    if expected_start != len(text):
        raise ValueError("Source segments do not cover the chapter")


def _decode_envelope(payload: str | Mapping[str, object] | Sequence[object]) -> object:
    if not isinstance(payload, str):
        return payload
    raw = payload.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```" or lines[0].strip() not in {"```", "```json"}:
            raise ValueError("Invalid JSON code fence")
        raw = "\n".join(lines[1:-1])
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Draft envelope is not valid JSON") from exc


def parse_draft_envelope(
    payload: str | Mapping[str, object] | Sequence[object],
    source_segments: Sequence[SourceSegment],
) -> tuple[DraftSegment, ...]:
    """Validate one shared N/C generation envelope without repairing prose."""
    decoded = _decode_envelope(payload)
    if isinstance(decoded, Mapping):
        decoded = decoded.get("segments")
    if not isinstance(decoded, list):
        raise ValueError("Draft envelope must contain a segments list")
    expected = [segment.segment_id for segment in source_segments]
    if len(expected) != len(set(expected)):
        raise ValueError("Source segment IDs must be unique")
    seen: dict[str, DraftSegment] = {}
    for item in decoded:
        if not isinstance(item, Mapping):
            raise ValueError("Each draft segment must be an object")
        segment_id = item.get("segment_id")
        translated_text = item.get("translated_text")
        if not isinstance(segment_id, str) or not isinstance(translated_text, str) or not translated_text.strip():
            raise ValueError("Each draft segment needs an ID and non-empty text")
        if segment_id in seen:
            raise ValueError(f"Duplicate draft segment ID: {segment_id}")
        seen[segment_id] = DraftSegment.from_text(segment_id, translated_text)
    missing = set(expected) - seen.keys()
    extra = seen.keys() - set(expected)
    if missing or extra:
        raise ValueError(f"Draft segment coverage mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    return tuple(seen[segment_id] for segment_id in expected)


def validate_draft_segments(
    draft_segments: Sequence[DraftSegment],
    source_segments: Sequence[SourceSegment],
) -> None:
    """Check final coverage after edits without changing source identities."""
    expected = [segment.segment_id for segment in source_segments]
    actual = [segment.segment_id for segment in draft_segments]
    if actual != expected:
        raise ValueError(f"Draft order/coverage differs from source: expected={expected}, actual={actual}")
    for segment in draft_segments:
        if not segment.translated_text.strip() or sha256_text(segment.translated_text) != segment.draft_sha256:
            raise ValueError(f"Empty or stale draft segment: {segment.segment_id}")


def render_draft(
    draft_segments: Sequence[DraftSegment],
    *,
    source_segments: Sequence[SourceSegment] | None = None,
) -> str:
    """Render caller-ordered segments; only the inter-segment separator is added."""
    if source_segments is not None:
        validate_draft_segments(draft_segments, source_segments)
    if len({segment.segment_id for segment in draft_segments}) != len(draft_segments):
        raise ValueError("Duplicate draft segment IDs")
    for segment in draft_segments:
        if not segment.translated_text.strip() or sha256_text(segment.translated_text) != segment.draft_sha256:
            raise ValueError(f"Empty or stale draft segment for {segment.segment_id}")
    return "\n\n".join(segment.translated_text for segment in draft_segments)


def _dependency_hash(payload: Mapping[str, object]) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def make_context_receipt(
    source_segment: SourceSegment,
    draft_segment: DraftSegment,
    *,
    neighbors: Iterable[tuple[SourceSegment, DraftSegment]] = (),
    memory_snapshot_sha256: str,
    instruction_version: str,
    review_policy: str,
    evidence_level: str,
    memory_entry_hashes: Iterable[str] = (),
    truncated: bool = False,
    tool_contract_version: str = "",
) -> ContextReceipt:
    if source_segment.segment_id != draft_segment.segment_id:
        raise ValueError("Source and draft segment IDs differ")
    if sha256_text(source_segment.text) != source_segment.source_sha256:
        raise ValueError("Stale source segment hash")
    if sha256_text(draft_segment.translated_text) != draft_segment.draft_sha256:
        raise ValueError("Stale draft segment hash")
    neighbor_hashes: list[tuple[str, str, str]] = []
    for source_neighbor, draft_neighbor in neighbors:
        if source_neighbor.segment_id != draft_neighbor.segment_id:
            raise ValueError("Neighbor source and draft segment IDs differ")
        if source_neighbor.segment_id == source_segment.segment_id:
            raise ValueError("Target segment cannot be its own neighbor")
        if sha256_text(source_neighbor.text) != source_neighbor.source_sha256 or sha256_text(draft_neighbor.translated_text) != draft_neighbor.draft_sha256:
            raise ValueError("Stale neighbor hash")
        neighbor_hashes.append((source_neighbor.segment_id, source_neighbor.source_sha256, draft_neighbor.draft_sha256))
    if len({row[0] for row in neighbor_hashes}) != len(neighbor_hashes):
        raise ValueError("Duplicate neighbor ID")
    entry_hashes = tuple(sorted(set(memory_entry_hashes)))
    payload: dict[str, object] = {
        "segment_id": source_segment.segment_id,
        "source_sha256": source_segment.source_sha256,
        "draft_sha256": draft_segment.draft_sha256,
        "neighbor_hashes": neighbor_hashes,
        "memory_snapshot_sha256": memory_snapshot_sha256,
        "memory_entry_hashes": entry_hashes,
        "instruction_version": instruction_version,
        "review_policy": review_policy,
        "evidence_level": evidence_level,
        "tool_contract_version": tool_contract_version,
        "truncated": truncated,
    }
    return ContextReceipt(
        segment_id=source_segment.segment_id,
        source_sha256=source_segment.source_sha256,
        draft_sha256=draft_segment.draft_sha256,
        included_neighbor_ids=tuple(row[0] for row in neighbor_hashes),
        neighbor_hashes=tuple(neighbor_hashes),
        memory_snapshot_sha256=memory_snapshot_sha256,
        memory_entry_hashes=entry_hashes,
        instruction_version=instruction_version,
        review_policy=review_policy,
        evidence_level=evidence_level,
        tool_contract_version=tool_contract_version,
        truncated=truncated,
        dependency_sha256=_dependency_hash(payload),
    )


def receipt_is_current(
    receipt: ContextReceipt,
    source_segment: SourceSegment,
    draft_segment: DraftSegment,
    *,
    neighbors: Iterable[tuple[SourceSegment, DraftSegment]] = (),
    memory_snapshot_sha256: str,
    instruction_version: str,
    review_policy: str,
    memory_entry_hashes: Iterable[str] = (),
    tool_contract_version: str = "",
) -> bool:
    try:
        current = make_context_receipt(
            source_segment,
            draft_segment,
            neighbors=neighbors,
            memory_snapshot_sha256=memory_snapshot_sha256,
            instruction_version=instruction_version,
            review_policy=review_policy,
            evidence_level=receipt.evidence_level,
            memory_entry_hashes=memory_entry_hashes,
            truncated=receipt.truncated,
            tool_contract_version=tool_contract_version,
        )
    except ValueError:
        return False
    return receipt == current
