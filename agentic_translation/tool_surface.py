"""A small, general tool surface for the "harness lab".

This module defines exactly one thing: a bounded set of tools that let *any*
harness (this package's own minimal agent loop, Claude Code, Codex, or a bare
CLI) repair one chapter's flagged segments. The same :class:`ToolSurface` is
driven by every condition in the lab so that any measured difference in
outcome is attributable to the harness, not to a different tool contract.

State lives entirely on disk in a "task directory" as plain JSON, so a
subprocess-based harness (``claude -p``, ``codex exec``) can drive it through
:mod:`agentic_translation.tools_cli` or :mod:`agentic_translation.mcp_server`
without any in-process Python object surviving between calls. Each call:

1. loads :class:`TaskState` from ``task.json``,
2. checks the task is not finished and budgets are not exhausted,
3. executes the tool (reusing :class:`agentic_translation.agent_repair.RepairToolExecutor`
   for any segment mutation, so the same exact-match/atomic/QA-gate rules that
   guard the production adaptive pipeline guard this lab too),
4. persists the updated state and appends one entry to the action log
   (``actions.jsonl``), and
5. returns a small JSON-safe observation.

Errors are actionable: a caller (human or model) should be able to fix its
next call from the message alone, without guessing.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .agent_models import SegmentTextEdit, SubmitSegmentPatchAction
from .agent_repair import RepairToolExecutor
from .models import GlossaryEntry, GlossaryParseResult

try:  # pragma: no cover - defensive only; adaptive.py always ships with this repo
    from .adaptive import (
        INFORMATIONAL_QA_CHECKS,
        _content_signals,
        _polish_guard_violation,
        _segment_deterministic_findings,
    )
except ImportError:  # pragma: no cover
    INFORMATIONAL_QA_CHECKS = frozenset()  # type: ignore[assignment]
    _polish_guard_violation = None  # type: ignore[assignment]
    _content_signals = None  # type: ignore[assignment]
    _segment_deterministic_findings = None  # type: ignore[assignment]


TASK_STATE_FILENAME = "task.json"
ACTION_LOG_FILENAME = "actions.jsonl"
SEGMENT_JOIN = "\n\n"

DEFAULT_MAX_TOOL_CALLS = 40
DEFAULT_MAX_EDITS = 12


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


class ToolSurfaceError(Exception):
    """Raised for a caller-facing, actionable tool failure."""


@dataclass
class SegmentRecord:
    segment_id: str
    source_text: str
    draft_text: str
    context_text: str = ""
    issue_ids: list[str] = field(default_factory=list)

    @property
    def draft_sha256(self) -> str:
        return sha256_text(self.draft_text)

    def to_json(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "source_text": self.source_text,
            "draft_text": self.draft_text,
            "context_text": self.context_text,
            "issue_ids": list(self.issue_ids),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "SegmentRecord":
        return cls(
            segment_id=str(data["segment_id"]),
            source_text=str(data["source_text"]),
            draft_text=str(data["draft_text"]),
            context_text=str(data.get("context_text", "")),
            issue_ids=[str(x) for x in data.get("issue_ids", [])],
        )


@dataclass
class TaskState:
    """The persisted state of one chapter-repair task."""

    task_id: str
    work_id: str
    chapter_id: str
    segments: list[SegmentRecord]
    glossary: list[GlossaryEntry]
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_edits: int = DEFAULT_MAX_EDITS
    tool_calls: int = 0
    edits: int = 0
    finished: bool = False
    summary: str | None = None
    last_action_id: str | None = None
    created_at: float = field(default_factory=time.time)

    # -- persistence -----------------------------------------------------

    @property
    def source_text(self) -> str:
        return SEGMENT_JOIN.join(s.source_text for s in self.segments)

    @property
    def draft_text(self) -> str:
        return SEGMENT_JOIN.join(s.draft_text for s in self.segments)

    @property
    def glossary_result(self) -> GlossaryParseResult:
        return GlossaryParseResult(entries=list(self.glossary))

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "work_id": self.work_id,
            "chapter_id": self.chapter_id,
            "segments": [s.to_json() for s in self.segments],
            "glossary": [
                {"source": g.source, "target": g.target, "candidates": list(g.candidates),
                 "blocked_variants": list(g.blocked_variants)}
                for g in self.glossary
            ],
            "max_tool_calls": self.max_tool_calls,
            "max_edits": self.max_edits,
            "tool_calls": self.tool_calls,
            "edits": self.edits,
            "finished": self.finished,
            "summary": self.summary,
            "last_action_id": self.last_action_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "TaskState":
        return cls(
            task_id=str(data["task_id"]),
            work_id=str(data["work_id"]),
            chapter_id=str(data["chapter_id"]),
            segments=[SegmentRecord.from_json(s) for s in data["segments"]],
            glossary=[GlossaryEntry(source=g["source"], target=g["target"],
                                     candidates=list(g.get("candidates", [])),
                                     blocked_variants=list(g.get("blocked_variants", [])))
                      for g in data.get("glossary", [])],
            max_tool_calls=int(data.get("max_tool_calls", DEFAULT_MAX_TOOL_CALLS)),
            max_edits=int(data.get("max_edits", DEFAULT_MAX_EDITS)),
            tool_calls=int(data.get("tool_calls", 0)),
            edits=int(data.get("edits", 0)),
            finished=bool(data.get("finished", False)),
            summary=data.get("summary"),
            last_action_id=data.get("last_action_id"),
            created_at=float(data.get("created_at", time.time())),
        )

    @classmethod
    def load(cls, task_dir: str | Path) -> "TaskState":
        path = Path(task_dir) / TASK_STATE_FILENAME
        if not path.exists():
            raise ToolSurfaceError(
                f"No task state at {path}; call ToolSurface.create_task(...) "
                "or the harness lab's prepare step first."
            )
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))

    def save(self, task_dir: str | Path) -> None:
        _atomic_write_json(Path(task_dir) / TASK_STATE_FILENAME, self.to_json())

    # -- lookups -----------------------------------------------------------

    def index_of(self, segment_id: str) -> int | None:
        for index, segment in enumerate(self.segments):
            if segment.segment_id == segment_id:
                return index
        return None

    def ordered_pairs(self) -> list[tuple[str, str]]:
        return [(s.segment_id, s.draft_text) for s in self.segments]


def append_action_log(
    task_dir: str | Path,
    *,
    parent_id: str | None,
    tool: str,
    args: Mapping[str, Any],
    ok: bool,
    message: str,
    data: Mapping[str, Any],
    elapsed_ms: float,
) -> str:
    """Append one JSONL entry and return its id.

    Entries carry ``parent_id`` so a caller (or the lab runner) can replay or
    branch an episode: forking a task directory before some action id and
    re-running from there produces a comparable alternate attempt without
    losing the original log.
    """

    action_id = uuid.uuid4().hex
    entry = {
        "id": action_id,
        "parent_id": parent_id,
        "ts": time.time(),
        "tool": tool,
        "args": _bounded(args),
        "ok": ok,
        "message": message,
        "data": _bounded(data),
        "elapsed_ms": elapsed_ms,
    }
    path = Path(task_dir) / ACTION_LOG_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return action_id


def read_action_log(task_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(task_dir) / ACTION_LOG_FILENAME
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def _bounded(value: Any, *, depth: int = 0, limit: int = 4000) -> Any:
    """Keep logged args/data finite and JSON-safe (mirrors agent_repair's guard)."""

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= limit else value[: limit - 20] + " …[truncated]"
    if depth >= 4:
        return "<truncated>"
    if isinstance(value, Mapping):
        return {str(k): _bounded(v, depth=depth + 1, limit=limit) for k, v in list(value.items())[:64]}
    if isinstance(value, (list, tuple)):
        return [_bounded(v, depth=depth + 1, limit=limit) for v in list(value)[:64]]
    return f"<{type(value).__name__}>"


def _observation(ok: bool, message: str, data: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"ok": ok, "message": message, "data": dict(data or {})}


class ToolSurface:
    """Loads/saves :class:`TaskState` around each tool call.

    Every public ``tool_*`` method is a complete unit of work: load state,
    validate, act, persist, log, return. This makes the surface safe to drive
    from a fresh subprocess per call (the CLI, or a harness that shells out)
    as well as from a single long-lived process (the MCP server, or the
    in-process ``own_loop`` condition).
    """

    def __init__(self, task_dir: str | Path):
        self.task_dir = Path(task_dir)

    # -- task lifecycle ------------------------------------------------

    @staticmethod
    def create_task(
        task_dir: str | Path,
        *,
        task_id: str,
        work_id: str,
        chapter_id: str,
        segments: Sequence[Mapping[str, Any]],
        glossary: Sequence[Mapping[str, str]] = (),
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        max_edits: int = DEFAULT_MAX_EDITS,
    ) -> "ToolSurface":
        records = [SegmentRecord.from_json(s) for s in segments]
        if len({r.segment_id for r in records}) != len(records):
            raise ToolSurfaceError("Duplicate segment_id while creating a task")
        entries = [GlossaryEntry(source=g["source"], target=g["target"]) for g in glossary]
        state = TaskState(
            task_id=task_id,
            work_id=work_id,
            chapter_id=chapter_id,
            segments=records,
            glossary=entries,
            max_tool_calls=max_tool_calls,
            max_edits=max_edits,
        )
        state.save(task_dir)
        return ToolSurface(task_dir)

    TOOL_NAMES = (
        "list_segments",
        "read_segment",
        "search_chapter",
        "lookup_glossary",
        "edit_segment",
        "rewrite_segment",
        "done",
    )

    def call(self, tool: str, args: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Dispatch one named tool call. Used by the CLI and MCP server alike."""

        args = dict(args or {})
        if tool not in self.TOOL_NAMES:
            return _observation(False, f"Unknown tool '{tool}'. Known tools: {', '.join(self.TOOL_NAMES)}.")
        handler = getattr(self, f"tool_{tool}")
        started = time.perf_counter()
        state = TaskState.load(self.task_dir)

        if state.finished and tool != "done":
            observation = _observation(
                False,
                "Task is already finished (done() was already called). "
                "Start a new task directory to continue working.",
            )
            self._log(state, tool, args, observation, started)
            return observation

        if state.tool_calls >= state.max_tool_calls:
            observation = _observation(
                False,
                f"Tool-call budget exhausted (max_tool_calls={state.max_tool_calls}); "
                "call done() to finish this episode.",
                {"tool_calls": state.tool_calls},
            )
            state.tool_calls += 1
            state.save(self.task_dir)
            self._log(state, tool, args, observation, started)
            return observation

        if tool in {"edit_segment", "rewrite_segment"} and state.edits >= state.max_edits:
            observation = _observation(
                False,
                f"Edit budget exhausted (max_edits={state.max_edits}); call done() to finish, "
                "or use read_segment/search_chapter/lookup_glossary which do not consume it.",
                {"edits": state.edits},
            )
            state.tool_calls += 1
            state.save(self.task_dir)
            self._log(state, tool, args, observation, started)
            return observation

        try:
            observation, state = handler(state, args)
        except ToolSurfaceError as exc:
            observation = _observation(False, str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            observation = _observation(False, f"Internal tool error: {type(exc).__name__}: {exc}")

        state.tool_calls += 1
        if tool in {"edit_segment", "rewrite_segment"}:
            state.edits += 1
        state.save(self.task_dir)
        self._log(state, tool, args, observation, started)
        return observation

    def _log(
        self,
        state: TaskState,
        tool: str,
        args: Mapping[str, Any],
        observation: Mapping[str, Any],
        started: float,
    ) -> None:
        elapsed_ms = (time.perf_counter() - started) * 1000
        action_id = append_action_log(
            self.task_dir,
            parent_id=state.last_action_id,
            tool=tool,
            args=args,
            ok=bool(observation.get("ok")),
            message=str(observation.get("message", "")),
            data=observation.get("data", {}),
            elapsed_ms=elapsed_ms,
        )
        state.last_action_id = action_id
        state.save(self.task_dir)

    # -- read-only tools -------------------------------------------------

    def tool_list_segments(self, state: TaskState, args: Mapping[str, Any]) -> tuple[dict, TaskState]:
        rows = [
            {
                "segment_id": s.segment_id,
                "source_len": len(s.source_text),
                "draft_len": len(s.draft_text),
                "issue_ids": list(s.issue_ids),
                "sha256": s.draft_sha256,
            }
            for s in state.segments
        ]
        return _observation(True, f"{len(rows)} segment(s).", {"segments": rows}), state

    def tool_read_segment(self, state: TaskState, args: Mapping[str, Any]) -> tuple[dict, TaskState]:
        segment_id = str(args.get("segment_id", ""))
        index = state.index_of(segment_id)
        if index is None:
            return _observation(
                False,
                f"Unknown segment_id '{segment_id}'. Call list_segments to see valid ids.",
                {"known_segment_ids": [s.segment_id for s in state.segments]},
            ), state
        segment = state.segments[index]
        data: dict[str, Any] = {
            "segment_id": segment.segment_id,
            "source_text": segment.source_text,
            "draft_text": segment.draft_text,
            "context_text": segment.context_text,
            "issue_ids": list(segment.issue_ids),
            "sha256": segment.draft_sha256,
        }
        if _segment_deterministic_findings is not None:
            data["deterministic_findings"] = list(
                _segment_deterministic_findings(segment.source_text, segment.draft_text, state.glossary_result)
            )
        return _observation(True, f"Segment {segment_id}.", data), state

    def tool_search_chapter(self, state: TaskState, args: Mapping[str, Any]) -> tuple[dict, TaskState]:
        query = str(args.get("query", ""))
        if not query:
            raise ToolSurfaceError("search_chapter requires a non-empty 'query'.")
        matches: list[dict[str, Any]] = []
        needle = query.lower()
        for segment in state.segments:
            for field_name, text in (("source", segment.source_text), ("draft", segment.draft_text)):
                lowered = text.lower()
                start = 0
                while True:
                    position = lowered.find(needle, start)
                    if position == -1:
                        break
                    left = max(0, position - 40)
                    right = min(len(text), position + len(query) + 40)
                    matches.append({
                        "segment_id": segment.segment_id,
                        "field": field_name,
                        "snippet": text[left:right],
                    })
                    start = position + max(len(needle), 1)
                    if len(matches) >= 50:
                        break
                if len(matches) >= 50:
                    break
            if len(matches) >= 50:
                break
        message = f"{len(matches)} match(es) for '{query}'." if matches else (
            f"No matches for '{query}'. Try a shorter substring or check list_segments/read_segment."
        )
        return _observation(True, message, {"matches": matches}), state

    def tool_lookup_glossary(self, state: TaskState, args: Mapping[str, Any]) -> tuple[dict, TaskState]:
        term = str(args.get("term", ""))
        if not term:
            raise ToolSurfaceError("lookup_glossary requires a non-empty 'term'.")
        for entry in state.glossary:
            if entry.source == term:
                return _observation(True, f"Glossary hit for '{term}'.", {
                    "term": term, "target": entry.target,
                }), state
        candidates = [g.source for g in state.glossary if term in g.source or g.source in term]
        return _observation(
            False,
            f"No glossary entry for '{term}'. Use search_chapter to find how it is rendered in context.",
            {"term": term, "near_misses": candidates[:10]},
        ), state

    # -- mutating tools ----------------------------------------------------

    def _build_executor(self, state: TaskState) -> RepairToolExecutor:
        # The lab's flagged issues are Jev naturalness/fidelity/consistency
        # findings, which the deterministic QA checks in qa.py do not score.
        # allow_nonregressing_patches=True (an existing, documented executor
        # knob) accepts any exact-once edit that does not regress deterministic
        # QA or introduce a new finding identity, rather than requiring a
        # deterministic-QA-score improvement that most polish edits cannot
        # produce. The content-loss guard (_polish_guard_violation) still
        # applies to every edit regardless of this setting.
        return RepairToolExecutor(
            source_text=state.source_text,
            translated_text=state.draft_text,
            glossary=state.glossary_result,
            run_id="harness-lab",
            story_slug=state.work_id,
            chapter=state.chapter_id,
            allow_nonregressing_patches=True,
            ignored_qa_checks=INFORMATIONAL_QA_CHECKS,
        )

    @staticmethod
    def _segment_guard(
        original_segment_text: str,
        source_segment_text: str,
        signals_sink: list[str],
    ) -> Callable[[str], tuple[bool, str]]:
        def guard(candidate_text: str) -> tuple[bool, str]:
            if _content_signals is not None:
                signals_sink.extend(_content_signals(original_segment_text, candidate_text))
            if _polish_guard_violation is None:
                return True, ""
            violation = _polish_guard_violation(original_segment_text, candidate_text, source_segment_text)
            if violation is not None:
                return False, violation
            return True, ""
        return guard

    def _apply_segment_edits(
        self,
        state: TaskState,
        *,
        segment_id: str,
        expected_sha256: str,
        edits: Sequence[Mapping[str, str]],
    ) -> tuple[dict, TaskState]:
        index = state.index_of(segment_id)
        if index is None:
            return _observation(
                False,
                f"Unknown segment_id '{segment_id}'. Call list_segments to see valid ids.",
                {"known_segment_ids": [s.segment_id for s in state.segments]},
            ), state
        segment = state.segments[index]
        current_sha = segment.draft_sha256
        if expected_sha256 != current_sha:
            return _observation(
                False,
                f"Segment {segment_id} has changed since it was last read "
                f"(expected sha256={expected_sha256}, current sha256={current_sha}); "
                "call read_segment to get the latest draft and sha256, then retry.",
                {"segment_id": segment_id, "expected_segment_sha256": expected_sha256,
                 "current_segment_sha256": current_sha},
            ), state
        if not edits:
            raise ToolSurfaceError("At least one edit is required.")
        try:
            segment_edits = [SegmentTextEdit(old_text=e["old_text"], new_text=e.get("new_text", "")) for e in edits]
        except KeyError as exc:
            raise ToolSurfaceError(f"Each edit requires 'old_text' (missing key {exc}).") from exc

        action = SubmitSegmentPatchAction(
            segment_id=segment_id,
            expected_segment_sha256=expected_sha256,
            edits=segment_edits,
            issue_ids=list(segment.issue_ids) or ["manual"],
            rationale="harness_lab tool_surface edit",
        )
        executor = self._build_executor(state)
        content_signals: list[str] = []
        result = executor.submit_segment_patch(
            action,
            state.ordered_pairs(),
            candidate_guard=self._segment_guard(segment.draft_text, segment.source_text, content_signals),
        )
        observation = result.observation
        if not observation.ok:
            reason = observation.data.get("reason")
            occurrences = observation.data.get("occurrences")
            if reason == "ambiguous_target" and occurrences == 0:
                message = (
                    f"old_text matched 0 times in {segment_id}; re-read with read_segment "
                    "and copy an exact span from draft_text."
                )
            elif reason == "ambiguous_target":
                message = (
                    f"old_text matched {occurrences} times in {segment_id}; expand old_text "
                    "with more surrounding context so it matches exactly once."
                )
            elif reason == "whole_chapter_target":
                message = (
                    f"old_text spans the whole chapter, not just segment {segment_id}; "
                    "use rewrite_segment for a whole-segment replacement instead."
                )
            elif reason == "source_guard":
                message = observation.message
            elif reason == "structural_regression":
                message = (
                    f"Edit would remove all translated content for segment {segment_id}; "
                    "provide non-empty new_text or leave the segment unchanged."
                )
            else:
                message = observation.message
            data = dict(observation.data)
            if content_signals:
                data["content_signals"] = content_signals
            return _observation(False, message, data), state

        new_text = str(observation.data["candidate_segment_text"])
        segment.draft_text = new_text
        state.segments[index] = segment
        data = {
            "segment_id": segment_id,
            "sha256": segment.draft_sha256,
            "draft_text": new_text,
        }
        if content_signals:
            data["content_signals"] = content_signals
        qa_after = result.qa_after
        if qa_after is not None:
            data["remaining_findings"] = qa_after.summary.total_findings
        return _observation(True, f"Segment {segment_id} updated.", data), state

    def tool_edit_segment(self, state: TaskState, args: Mapping[str, Any]) -> tuple[dict, TaskState]:
        segment_id = str(args.get("segment_id", ""))
        expected_sha256 = str(args.get("expected_sha256", ""))
        edits = args.get("edits")
        if not isinstance(edits, list) or not edits:
            raise ToolSurfaceError("edit_segment requires a non-empty 'edits' list of {old_text,new_text}.")
        return self._apply_segment_edits(state, segment_id=segment_id, expected_sha256=expected_sha256, edits=edits)

    def tool_rewrite_segment(self, state: TaskState, args: Mapping[str, Any]) -> tuple[dict, TaskState]:
        segment_id = str(args.get("segment_id", ""))
        expected_sha256 = str(args.get("expected_sha256", ""))
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ToolSurfaceError("rewrite_segment requires non-empty 'text'.")
        index = state.index_of(segment_id)
        if index is None:
            return _observation(
                False,
                f"Unknown segment_id '{segment_id}'. Call list_segments to see valid ids.",
                {"known_segment_ids": [s.segment_id for s in state.segments]},
            ), state
        current_text = state.segments[index].draft_text
        edits = [{"old_text": current_text, "new_text": text}]
        return self._apply_segment_edits(state, segment_id=segment_id, expected_sha256=expected_sha256, edits=edits)

    def tool_done(self, state: TaskState, args: Mapping[str, Any]) -> tuple[dict, TaskState]:
        summary = str(args.get("summary", ""))
        if not summary.strip():
            raise ToolSurfaceError("done requires a non-empty 'summary'.")
        state.finished = True
        state.summary = summary
        return _observation(True, "Task marked done.", {
            "summary": summary, "edits": state.edits, "tool_calls": state.tool_calls,
        }), state


# -- JSON Schemas for MCP tools/list and for the own_loop native function path --

TOOL_JSON_SCHEMAS: dict[str, dict[str, Any]] = {
    "list_segments": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    "read_segment": {
        "type": "object",
        "properties": {"segment_id": {"type": "string"}},
        "required": ["segment_id"],
        "additionalProperties": False,
    },
    "search_chapter": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
    "lookup_glossary": {
        "type": "object",
        "properties": {"term": {"type": "string"}},
        "required": ["term"],
        "additionalProperties": False,
    },
    "edit_segment": {
        "type": "object",
        "properties": {
            "segment_id": {"type": "string"},
            "expected_sha256": {"type": "string"},
            "edits": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                    },
                    "required": ["old_text", "new_text"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["segment_id", "expected_sha256", "edits"],
        "additionalProperties": False,
    },
    "rewrite_segment": {
        "type": "object",
        "properties": {
            "segment_id": {"type": "string"},
            "expected_sha256": {"type": "string"},
            "text": {"type": "string"},
        },
        "required": ["segment_id", "expected_sha256", "text"],
        "additionalProperties": False,
    },
    "done": {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
        "additionalProperties": False,
    },
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    "list_segments": "List every segment in this chapter with id, lengths, flags, and current sha256.",
    "read_segment": "Read one segment's source text, current draft, context pack, flags, and sha256.",
    "search_chapter": "Case-insensitive substring search across all segments' source and draft text.",
    "lookup_glossary": "Look up the fixed English rendering of a source-language term.",
    "edit_segment": "Apply one or more exact-once find/replace edits inside one segment (atomic, QA-gated).",
    "rewrite_segment": "Replace an entire segment's draft text (guarded whole-segment rewrite).",
    "done": "Finish the episode with a short summary. No-change is a valid outcome.",
}
