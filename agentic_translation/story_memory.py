"""Source-grounded, causal story context for unattended translation runs.

The tape accepts entries extracted from source by a caller, validates their
evidence, and writes immutable JSON snapshots inside a selected run directory.
It never reads candidate translations or edits a master glossary.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .segmentation import SourceSegment, sha256_text, validate_source_segments


SCOPES = frozenset({"narrator", "character_belief", "dialogue_claim", "editorial"})
STATUSES = frozenset({"provisional", "active", "disputed", "superseded"})
ENTRY_KINDS = frozenset({"entity", "alias", "relationship", "identity", "location", "organization", "technique", "artifact", "rank", "event", "state", "editorial_convention", "ambiguity"})
CONFLICT_REASONS = frozenset({"source_correction", "scope_change", "new_source_evidence"})


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _hash(value: object) -> str:
    return sha256_text(_canonical(value))


@dataclass(frozen=True)
class SourceEvidence:
    chapter_id: str
    segment_id: str
    exact_excerpt: str


@dataclass(frozen=True)
class MemoryEntry:
    entry_id: str
    kind: str
    subject: str
    value: str
    source_evidence: tuple[SourceEvidence, ...]
    known_from_chapter: str
    known_from_segment: str
    scope: str
    status: str
    created_by_model: str
    policy_version: str
    parent_entry_id: str | None = None
    source_aliases: tuple[str, ...] = ()

    @property
    def sha256(self) -> str:
        return _hash(asdict(self))


@dataclass(frozen=True)
class TerminologyCandidate:
    source: str
    target: str
    source_evidence: tuple[SourceEvidence, ...]
    provisional: bool = False
    adjudication_reason: str | None = None
    adjudication_detail: str | None = None


@dataclass(frozen=True)
class TerminologyDecision:
    source: str
    target: str
    chapter_id: str
    segment_id: str
    source_evidence: tuple[SourceEvidence, ...]
    status: str
    outcome: str
    decision_actor: str = "automation"
    decision_basis: str = "source_grounded_policy"
    adjudication_reason: str | None = None
    adjudication_detail: str | None = None


@dataclass(frozen=True)
class MemorySnapshot:
    chapter_id: str
    chapter_source_sha256: str
    preceding_snapshot_sha256: str
    entries: tuple[MemoryEntry, ...]
    terminology: tuple[TerminologyDecision, ...]
    policy_version: str
    snapshot_sha256: str


@dataclass(frozen=True)
class ContextPack:
    chapter_id: str
    segment_id: str
    text: str
    entry_ids: tuple[str, ...]
    entry_hashes: tuple[str, ...]
    terminology_sources: tuple[str, ...]
    snapshot_sha256: str
    dependency_sha256: str
    truncated: bool
    budget_kind: str = "characters_not_tokens"


# How a passage's pack chooses and orders memory facts (names always come first).
#   legacy        facts that name something in the passage, in the order they were
#                 recorded (oldest first), then the five newest; the original pack.
#   current_first only the newest "state" fact per subject; stable facts (identity,
#                 alias, rank, relationship, entities) before states, then events;
#                 newest first within each group.
#   terms_only    no facts, names only.
FACT_POLICIES = ("legacy", "current_first", "terms_only")
_STABLE_KINDS = frozenset({"identity", "alias", "relationship", "rank", "entity", "organization", "location",
                           "artifact", "technique", "editorial_convention", "ambiguity"})


def order_facts(selected: Sequence["MemoryEntry"], visible: Sequence["MemoryEntry"], policy: str) -> list["MemoryEntry"]:
    """Apply a fact policy to the selected entries (visible is in recorded order)."""
    if policy not in FACT_POLICIES:
        raise ValueError(f"Unknown fact policy: {policy}")
    if policy == "legacy":
        return list(selected)
    if policy == "terms_only":
        return []
    position = {entry.entry_id: i for i, entry in enumerate(visible)}
    newest_state = {}
    for entry in visible:
        if entry.kind == "state":
            newest_state[entry.subject] = entry.entry_id
    kept = [entry for entry in selected if entry.kind != "state" or newest_state.get(entry.subject) == entry.entry_id]
    group = lambda entry: 0 if entry.kind in _STABLE_KINDS else 1 if entry.kind == "state" else 2  # noqa: E731
    return sorted(kept, key=lambda entry: (group(entry), -position[entry.entry_id]))


def _saved_snapshot_chain(run_dir: str | Path) -> list[dict]:
    """Read one complete snapshot chain, including its original empty predecessor."""
    folder = Path(run_dir) / "story_memory" / "snapshots"
    raw = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("*.json"))]
    hashes, chapters, by_previous = set(), set(), {}
    for item in raw:
        if not isinstance(item, dict) or any(not isinstance(item.get(key), str)
                                             for key in ("chapter_id", "snapshot_sha256", "preceding_snapshot_sha256")):
            raise ValueError("Story memory snapshot has invalid chain metadata")
        if not item["snapshot_sha256"] or item["snapshot_sha256"] in hashes:
            raise ValueError("Story memory snapshots have a missing or duplicate hash")
        if not item["chapter_id"] or item["chapter_id"] in chapters:
            raise ValueError("Story memory snapshots have a missing or duplicate chapter")
        if item["preceding_snapshot_sha256"] in by_previous:
            raise ValueError("Story memory snapshots fork: two chapters share a predecessor")
        hashes.add(item["snapshot_sha256"])
        chapters.add(item["chapter_id"])
        by_previous[item["preceding_snapshot_sha256"]] = item
    if "" not in by_previous:
        raise ValueError("Story memory snapshots are missing the original chain root")
    chain, visited, current = [], set(), by_previous[""]
    while current is not None:
        if current["snapshot_sha256"] in visited:
            raise ValueError("Story memory snapshot chain contains a cycle")
        visited.add(current["snapshot_sha256"])
        chain.append(current)
        current = by_previous.get(current["snapshot_sha256"])
    if len(chain) != len(raw):
        raise ValueError("Story memory snapshot chain is broken")
    for item in chain:
        try:
            payload = {key: item[key] for key in ("chapter_id", "chapter_source_sha256", "preceding_snapshot_sha256",
                                                  "entries", "terminology", "policy_version")}
        except KeyError as exc:
            raise ValueError("Story memory snapshot has incomplete contents") from exc
        if _hash(payload) != item["snapshot_sha256"]:
            raise ValueError(f"Snapshot {item['chapter_id']} does not match its hash")
    return chain


def saved_chapter_ids(run_dir: str | Path) -> list[str]:
    """Chapter ids of a run's complete, hash-checked snapshot chain, in order."""
    return [item["chapter_id"] for item in _saved_snapshot_chain(run_dir)]


def _evidence(items) -> tuple[SourceEvidence, ...]:
    return tuple(SourceEvidence(**item) for item in items)


def _entry_from_dict(item: dict) -> "MemoryEntry":
    return MemoryEntry(**{**item, "source_evidence": _evidence(item["source_evidence"]),
                          "source_aliases": tuple(item.get("source_aliases") or ())})


def _decision_from_dict(item: dict) -> "TerminologyDecision":
    return TerminologyDecision(**{**item, "source_evidence": _evidence(item["source_evidence"])})


class StoryMemoryTape:
    """Append-only source tape; chapters must be added in narrative order."""

    def __init__(self, *, policy_version: str = "source-memory-v1", run_dir: str | Path | None = None) -> None:
        if not policy_version:
            raise ValueError("policy_version is required")
        self.policy_version = policy_version
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self._chapters: list[str] = []
        self._segments: dict[tuple[str, str], SourceSegment] = {}
        self._snapshots: dict[str, MemorySnapshot] = {}
        self._entries: list[MemoryEntry] = []
        self._decisions: list[TerminologyDecision] = []

    @classmethod
    def load(cls, run_dir: str | Path, sources: "dict[str, Sequence[SourceSegment]]", *,
             policy_version: str, chapter_ids: Sequence[str] | None = None) -> "StoryMemoryTape":
        """Rebuild a tape from the snapshots a run saved, to continue a series in a later run.

        ``sources`` maps each chapter_id to its segments (rebuilt from the saved Chinese text).
        Every snapshot is checked before use: its chapter source hash against the supplied
        text, its own hash against its contents, and the preceding-hash chain (one start, no
        gaps). ``chapter_ids`` keeps only a leading part of the chain (a carried prefix).
        New chapters can then be added; they chain onto the last loaded snapshot.
        """
        chain = _saved_snapshot_chain(run_dir)
        if chapter_ids is not None:
            wanted = list(chapter_ids)
            if [item["chapter_id"] for item in chain[:len(wanted)]] != wanted:
                raise ValueError("Requested chapters are not the leading part of the saved chain")
            chain = chain[:len(wanted)]
        tape = cls(policy_version=policy_version, run_dir=run_dir)
        for item in chain:
            chapter_id = item["chapter_id"]
            if item["policy_version"] != policy_version:
                raise ValueError(f"Snapshot {chapter_id} uses policy {item['policy_version']}, not {policy_version}")
            segments = tuple(sources.get(chapter_id) or ())
            if not segments or sha256_text("".join(segment.text for segment in segments)) != item["chapter_source_sha256"]:
                raise ValueError(f"Source text for {chapter_id} is missing or differs from the saved memory")
            if any(segment.chapter_id != chapter_id for segment in segments):
                raise ValueError(f"Source segments do not belong to chapter {chapter_id}")
            validate_source_segments("".join(segment.text for segment in segments), segments)
            snapshot = MemorySnapshot(
                chapter_id=chapter_id, chapter_source_sha256=item["chapter_source_sha256"],
                preceding_snapshot_sha256=item["preceding_snapshot_sha256"],
                entries=tuple(_entry_from_dict(entry) for entry in item["entries"]),
                terminology=tuple(_decision_from_dict(decision) for decision in item["terminology"]),
                policy_version=item["policy_version"], snapshot_sha256=item["snapshot_sha256"])
            if (snapshot.entries[:len(tape._entries)] != tuple(tape._entries)
                    or snapshot.terminology[:len(tape._decisions)] != tuple(tape._decisions)):
                raise ValueError(f"Snapshot {chapter_id} does not extend the preceding memory")
            existing = {entry.entry_id: entry for entry in tape._entries}
            for entry in snapshot.entries[len(tape._entries):]:
                tape._validate_entry(entry, current=chapter_id, current_segments=segments, existing=existing)
                existing[entry.entry_id] = entry
            lookup = {**tape._segments, **{(segment.chapter_id, segment.segment_id): segment for segment in segments}}
            for decision in snapshot.terminology[len(tape._decisions):]:
                if decision.chapter_id != chapter_id or not decision.source or not decision.target:
                    raise ValueError("Saved terminology has an invalid source, target, or chapter")
                known_from = tape._position(decision.chapter_id, decision.segment_id,
                                            current=chapter_id, current_segments=segments)
                tape._validate_evidence(decision.source_evidence, current=chapter_id,
                                        current_segments=segments, known_from=known_from)
                if not any(decision.source in lookup[(e.chapter_id, e.segment_id)].text
                           for e in decision.source_evidence):
                    raise ValueError("Saved terminology source does not occur in referenced segment")
            tape._chapters.append(chapter_id)
            tape._segments.update({(segment.chapter_id, segment.segment_id): segment for segment in segments})
            tape._snapshots[chapter_id] = snapshot
            tape._entries, tape._decisions = list(snapshot.entries), list(snapshot.terminology)
        return tape

    @property
    def chapter_ids(self) -> tuple[str, ...]:
        return tuple(self._chapters)

    def snapshot_for_chapter(self, chapter_id: str) -> MemorySnapshot:
        return self._snapshots[chapter_id]

    def snapshot_before_chapter(self, chapter_id: str) -> MemorySnapshot | None:
        index = self._chapters.index(chapter_id)
        return None if index == 0 else self._snapshots[self._chapters[index - 1]]

    def _position(self, chapter_id: str, segment_id: str, *, current: str, current_segments: Sequence[SourceSegment]) -> tuple[int, int]:
        if chapter_id == current:
            ids = [segment.segment_id for segment in current_segments]
            if segment_id not in ids:
                raise ValueError(f"Unknown evidence segment {chapter_id}/{segment_id}")
            return (len(self._chapters), ids.index(segment_id))
        if chapter_id not in self._chapters or (chapter_id, segment_id) not in self._segments:
            raise ValueError(f"Future or unknown evidence segment {chapter_id}/{segment_id}")
        chapter_index = self._chapters.index(chapter_id)
        ids = [segment_id_ for chapter_id_, segment_id_ in self._segments if chapter_id_ == chapter_id]
        return (chapter_index, ids.index(segment_id))

    def _validate_evidence(
        self,
        evidence: Sequence[SourceEvidence],
        *,
        current: str,
        current_segments: Sequence[SourceSegment],
        known_from: tuple[int, int],
    ) -> None:
        if not evidence:
            raise ValueError("Source evidence is required")
        new_segments = {(segment.chapter_id, segment.segment_id): segment for segment in current_segments}
        for item in evidence:
            position = self._position(item.chapter_id, item.segment_id, current=current, current_segments=current_segments)
            if position > known_from:
                raise ValueError("Evidence occurs after known_from point")
            segment = new_segments.get((item.chapter_id, item.segment_id)) or self._segments.get((item.chapter_id, item.segment_id))
            if not item.exact_excerpt or segment is None or item.exact_excerpt not in segment.text:
                raise ValueError("Evidence excerpt must occur exactly in referenced source segment")

    def _validate_entry(self, entry: MemoryEntry, *, current: str, current_segments: Sequence[SourceSegment], existing: dict[str, MemoryEntry]) -> None:
        if not entry.entry_id or entry.entry_id in existing:
            raise ValueError(f"Missing or duplicate memory entry ID: {entry.entry_id}")
        if entry.kind not in ENTRY_KINDS or entry.scope not in SCOPES or entry.status not in STATUSES:
            raise ValueError("Invalid entry kind, scope, or status")
        if (entry.kind == "editorial_convention") != (entry.scope == "editorial"):
            raise ValueError("Editorial conventions must be separate from factual claims")
        if not entry.subject or not entry.value or not entry.created_by_model or not entry.policy_version:
            raise ValueError("Entry subject, value, model, and policy version are required")
        if entry.policy_version != self.policy_version:
            raise ValueError("Entry policy version differs from tape")
        if entry.known_from_chapter != current:
            raise ValueError("New memory entries become known only in the current chapter")
        known_from = self._position(current, entry.known_from_segment, current=current, current_segments=current_segments)
        self._validate_evidence(entry.source_evidence, current=current, current_segments=current_segments, known_from=known_from)
        evidence_segments = [self._segments.get((item.chapter_id, item.segment_id)) or next((segment for segment in current_segments if segment.chapter_id == item.chapter_id and segment.segment_id == item.segment_id), None) for item in entry.source_evidence]
        if any(not any(alias in segment.text for segment in evidence_segments if segment is not None) for alias in entry.source_aliases):
            raise ValueError("Source alias must occur in cited source evidence")
        if entry.parent_entry_id is not None:
            parent = existing.get(entry.parent_entry_id)
            if parent is None:
                raise ValueError("Unknown parent entry")
            if parent.scope in {"character_belief", "dialogue_claim"} and entry.scope == "narrator":
                old_evidence = set(parent.source_evidence)
                if not any(item not in old_evidence for item in entry.source_evidence):
                    raise ValueError("Belief or dialogue cannot become narrator fact without new source evidence")

    def add_chapter(
        self,
        chapter_id: str,
        source_segments: Sequence[SourceSegment],
        proposed_entries: Iterable[MemoryEntry] = (),
        terminology_candidates: Iterable[TerminologyCandidate] = (),
    ) -> MemorySnapshot:
        """Validate and append a chapter, including automatic run-local terms."""
        if chapter_id in self._snapshots:
            raise ValueError(f"Chapter already appended: {chapter_id}")
        if not chapter_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in chapter_id):
            raise ValueError("chapter_id must be a safe single filename component")
        segments = tuple(source_segments)
        if not segments or any(segment.chapter_id != chapter_id for segment in segments):
            raise ValueError("Segments must all belong to chapter_id")
        validate_source_segments("".join(segment.text for segment in segments), segments)
        existing = {entry.entry_id: entry for entry in self._entries}
        additions: list[MemoryEntry] = []
        for entry in proposed_entries:
            self._validate_entry(entry, current=chapter_id, current_segments=segments, existing=existing)
            existing[entry.entry_id] = entry
            additions.append(entry)
        decisions: list[TerminologyDecision] = []
        active_terms = {item.source: item for item in self._decisions if item.outcome in {"accepted", "changed"}}
        for candidate in terminology_candidates:
            if not candidate.source or not candidate.target:
                raise ValueError("Term source and target are required")
            # The term itself must occur in referenced source, not merely nearby.
            if not candidate.source_evidence:
                raise ValueError("Terminology evidence is required")
            current_evidence = [item for item in candidate.source_evidence if item.chapter_id == chapter_id]
            known_segment = max(current_evidence, key=lambda item: self._position(item.chapter_id, item.segment_id, current=chapter_id, current_segments=segments)).segment_id if current_evidence else segments[0].segment_id
            known_from = self._position(chapter_id, known_segment, current=chapter_id, current_segments=segments)
            self._validate_evidence(candidate.source_evidence, current=chapter_id, current_segments=segments, known_from=known_from)
            lookup = {(segment.chapter_id, segment.segment_id): segment for segment in segments}
            lookup.update(self._segments)
            if not any(candidate.source in lookup[(item.chapter_id, item.segment_id)].text for item in candidate.source_evidence):
                raise ValueError("Terminology source does not occur in referenced segment")
            old = active_terms.get(candidate.source)
            if old is None:
                target, status, outcome = candidate.target, "provisional" if candidate.provisional else "active", "accepted"
            elif old.target == candidate.target:
                # A repeated candidate does not need another write receipt.
                continue
            elif candidate.adjudication_reason in CONFLICT_REASONS and candidate.adjudication_detail and candidate.adjudication_detail.strip():
                target, status, outcome = candidate.target, "provisional" if candidate.provisional else "active", "changed"
            else:
                target, status, outcome = old.target, old.status, "kept_existing"
            decision = TerminologyDecision(
                source=candidate.source,
                target=target,
                chapter_id=chapter_id,
                segment_id=known_segment,
                source_evidence=candidate.source_evidence,
                status=status,
                outcome=outcome,
                adjudication_reason=candidate.adjudication_reason if outcome == "changed" else None,
                adjudication_detail=candidate.adjudication_detail if outcome == "changed" else None,
            )
            decisions.append(decision)
            if outcome in {"accepted", "changed"}:
                active_terms[candidate.source] = decision
        all_entries = tuple(self._entries + additions)
        all_decisions = tuple(self._decisions + decisions)
        chapter_source_sha256 = sha256_text("".join(segment.text for segment in segments))
        previous_sha = self._snapshots[self._chapters[-1]].snapshot_sha256 if self._chapters else ""
        snapshot_payload = {
            "chapter_id": chapter_id,
            "chapter_source_sha256": chapter_source_sha256,
            "preceding_snapshot_sha256": previous_sha,
            "entries": [asdict(entry) for entry in all_entries],
            "terminology": [asdict(decision) for decision in all_decisions],
            "policy_version": self.policy_version,
        }
        snapshot = MemorySnapshot(
            chapter_id=chapter_id,
            chapter_source_sha256=chapter_source_sha256,
            preceding_snapshot_sha256=previous_sha,
            entries=all_entries,
            terminology=all_decisions,
            policy_version=self.policy_version,
            snapshot_sha256=_hash(snapshot_payload),
        )
        if self.run_dir is not None:
            self._persist(snapshot, additions, decisions)
        self._chapters.append(chapter_id)
        self._segments.update({(segment.chapter_id, segment.segment_id): segment for segment in segments})
        self._entries.extend(additions)
        self._decisions.extend(decisions)
        self._snapshots[chapter_id] = snapshot
        return snapshot

    def _persist(self, snapshot: MemorySnapshot, additions: Sequence[MemoryEntry], decisions: Sequence[TerminologyDecision]) -> None:
        assert self.run_dir is not None
        snapshots_dir = self.run_dir / "story_memory" / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        path = snapshots_dir / f"{snapshot.chapter_id}.json"
        if path.exists():
            raise FileExistsError(f"Immutable story memory snapshot already exists: {path}")
        # All or nothing: a snapshot file on disk means its chapter's log lines are
        # complete. Write the snapshot to a temporary file, append the log in one
        # write, then rename the snapshot into place. On any failure, truncate the
        # log back and remove the temporary file, so a failed chapter never leaves
        # an orphan snapshot that later readers would take as a second chain root.
        tmp = path.with_name(path.name + ".tmp")
        log = self.run_dir / "story_memory" / "updates.jsonl"
        lines = [_canonical({"type": "memory_entry", "chapter_id": snapshot.chapter_id, "entry": asdict(entry)})
                 for entry in additions]
        lines += [_canonical({"type": "terminology_decision", "chapter_id": snapshot.chapter_id,
                              "decision": asdict(decision)}) for decision in decisions]
        prior_size = log.stat().st_size if log.exists() else 0
        try:
            tmp.write_text(json.dumps(asdict(snapshot), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
            with log.open("a", encoding="utf-8") as stream:
                stream.write("".join(line + "\n" for line in lines))
            os.replace(tmp, path)
        except BaseException:
            if log.exists() and log.stat().st_size > prior_size:
                with log.open("r+b") as stream:
                    stream.truncate(prior_size)
            tmp.unlink(missing_ok=True)
            raise

    def glossary_for_segment(self, chapter_id: str, segment_id: str) -> dict[str, str]:
        cutoff = self._known_position(chapter_id, segment_id)
        choices: dict[str, str] = {}
        for decision in self._snapshots[chapter_id].terminology:
            if decision.outcome not in {"accepted", "changed"}:
                continue
            if self._known_position(decision.chapter_id, decision.segment_id) <= cutoff:
                choices[decision.source] = decision.target
        return choices

    def _known_position(self, chapter_id: str, segment_id: str) -> tuple[int, int]:
        if chapter_id not in self._chapters or (chapter_id, segment_id) not in self._segments:
            raise ValueError(f"Unknown chapter/segment: {chapter_id}/{segment_id}")
        chapter_index = self._chapters.index(chapter_id)
        ids = [key[1] for key in self._segments if key[0] == chapter_id]
        return chapter_index, ids.index(segment_id)

    def context_for_segment(self, chapter_id: str, segment_id: str, *, max_context_chars: int = 3000, recent_entries: int = 5,
                            fact_policy: str = "legacy") -> ContextPack:
        """Retrieve a small causal pack; max_context_chars is not token usage."""
        if max_context_chars < 0 or recent_entries < 0:
            raise ValueError("Context limits cannot be negative")
        if fact_policy not in FACT_POLICIES:
            raise ValueError(f"Unknown fact policy: {fact_policy}")
        segment = self._segments[(chapter_id, segment_id)]
        snapshot = self._snapshots[chapter_id]
        cutoff = self._known_position(chapter_id, segment_id)
        visible = [entry for entry in snapshot.entries if self._known_position(entry.known_from_chapter, entry.known_from_segment) <= cutoff and entry.status in {"active", "provisional", "disputed"}]
        matched = [entry for entry in visible if any(alias in segment.text for alias in (entry.subject, *entry.source_aliases) if alias)]
        recent = visible[-recent_entries:] if recent_entries else []
        selected = order_facts(list(dict.fromkeys([*matched, *recent])), visible, fact_policy)
        terms = self.glossary_for_segment(chapter_id, segment_id)
        matching_terms = [(source, target) for source, target in terms.items() if source in segment.text]
        lines: list[tuple[str, str | None, str | None]] = []
        for source, target in matching_terms:
            lines.append((f"TERM {source} -> {target}", None, source))
        for entry in selected:
            marker = f" [{entry.scope}; {entry.status}; from {entry.known_from_chapter}/{entry.known_from_segment}]"
            lines.append((f"{entry.kind} {entry.subject}: {entry.value}{marker}", entry.entry_id, None))
        kept: list[str] = []
        ids: list[str] = []
        term_sources: list[str] = []
        used = 0
        for line, entry_id, term_source in lines:
            extra = len(line) + (1 if kept else 0)
            if used + extra > max_context_chars:
                continue
            kept.append(line)
            used += extra
            if entry_id is not None:
                ids.append(entry_id)
            if term_source is not None:
                term_sources.append(term_source)
        entry_hashes = tuple(entry.sha256 for entry in selected if entry.entry_id in ids)
        text = "\n".join(kept)
        dependencies = {
            "chapter_id": chapter_id,
            "segment_id": segment_id,
            "source_sha256": segment.source_sha256,
            "entry_hashes": entry_hashes,
            "terminology": [(source, terms[source]) for source in term_sources],
            "policy_version": self.policy_version,
            "max_context_chars": max_context_chars,
        }
        if fact_policy != "legacy":  # legacy packs keep their original dependency hash
            dependencies["fact_policy"] = fact_policy
        return ContextPack(
            chapter_id=chapter_id,
            segment_id=segment_id,
            text=text,
            entry_ids=tuple(ids),
            entry_hashes=entry_hashes,
            terminology_sources=tuple(term_sources),
            snapshot_sha256=snapshot.snapshot_sha256,
            dependency_sha256=_hash(dependencies),
            truncated=len(kept) < len(lines),
        )


def build_source_memory(
    chapters: Iterable[tuple[str, Sequence[SourceSegment]]],
    extractor: Callable[[str, Sequence[SourceSegment], MemorySnapshot | None], tuple[Iterable[MemoryEntry], Iterable[TerminologyCandidate]]],
    *,
    policy_version: str = "source-memory-v1",
    run_dir: str | Path | None = None,
) -> StoryMemoryTape:
    """Call a source-only extractor in chapter order and validate each result."""
    tape = StoryMemoryTape(policy_version=policy_version, run_dir=run_dir)
    for chapter_id, source_segments in chapters:
        previous = tape._snapshots[tape._chapters[-1]] if tape._chapters else None
        entries, terms = extractor(chapter_id, source_segments, previous)
        tape.add_chapter(chapter_id, source_segments, entries, terms)
    return tape
