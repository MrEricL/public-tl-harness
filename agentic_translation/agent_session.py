"""MVP durable records for a process-independent Harness v3 session.

The event log and projections here intentionally stay simple.  Events are
sequence-numbered and flushed, while snapshots and episode artifacts use a
temporary file followed by ``replace``.  Integrity chains, fsync barriers, and
corruption recovery belong to a later hardening pass.
"""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any, Mapping

from pydantic import BaseModel, TypeAdapter, ValidationError

from .agent_models import (
    AgentAction,
    AgentEpisode,
    AgentObservation,
    AgentStep,
    AgentSessionIdentity,
    AgentSessionSnapshot,
    ApprovalDecision,
    ApprovalRequest,
    CompleteReviewAction,
    DelegateReviewAction,
    GlossaryPromotionProposal,
    FinishAction,
    NormalizePunctuationAction,
    PromoteGlossaryTermAction,
    ReadParagraphsAction,
    ReviewFinding,
    RepairExecutorSnapshot,
    PolicyDecision,
    SearchToolsAction,
    SelectTermAction,
    SpecialistReview,
    SubmitPatchAction,
    TermSuggestion,
    SessionEvent,
)
from .agent_policy import DefaultToolPolicy
from .agent_provider import (
    AgentActionProvider,
    AgentActionRequest,
    AgentActionValidationError,
    PriorObservableStep,
    REGISTRY_TOOL_SCHEMA_VERSION,
    SHOWCASE_TOOL_SCHEMA_VERSION,
    _bounded_instruction_context,
)
from .agent_repair import (
    RepairToolExecutor,
    ToolExecutionResult,
    _action_payload_is_bounded,
    _bounded_action_payload,
    _bounded_observation,
    _bounded_typed_action,
    _episode_findings,
    _episode_identifier,
    _latest_provider_call,
    _provider_call_records,
)
from .agent_tools import (
    AGENT_TOOL_REGISTRY,
    SHOWCASE_TOOL_REGISTRY,
    ToolCallValidationError,
    ToolRegistry,
)
from .models import GlossaryEntry, GlossaryParseResult, ProviderCallRecord, QAReport
from .terminology import TerminologyResolver
from .text import split_paragraphs


class SessionStore:
    """Persist a session's append-only events and resumable projections."""

    EVENTS_FILENAME = "session_events.jsonl"
    SNAPSHOT_FILENAME = "session_snapshot.json"
    EPISODE_FILENAME = "agent_episode.json"

    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.session_dir / self.EVENTS_FILENAME
        self.snapshot_path = self.session_dir / self.SNAPSHOT_FILENAME
        self.episode_path = self.session_dir / self.EPISODE_FILENAME
        # Explicit names are convenient to callers that mirror the artifact
        # filenames; the shorter aliases above keep the common path API tidy.
        self.session_events_path = self.events_path
        self.session_snapshot_path = self.snapshot_path
        self.agent_episode_path = self.episode_path

    def append(
        self,
        event_type: str,
        payload: Mapping[str, Any] | BaseModel,
    ) -> SessionEvent:
        """Append one event and assign the next monotonically increasing sequence."""

        existing = self.read_events()
        sequence = existing[-1].sequence + 1 if existing else 1
        if isinstance(payload, BaseModel):
            payload_data = payload.model_dump(mode="json")
        else:
            payload_data = dict(payload)
        event = SessionEvent(
            sequence=sequence,
            event_type=event_type,
            payload=payload_data,
        )
        encoded = json.dumps(
            event.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
        return event

    def read_events(self) -> list[SessionEvent]:
        """Read and validate all persisted events in sequence order."""

        if not self.events_path.exists():
            return []
        events: list[SessionEvent] = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = SessionEvent.model_validate(json.loads(line))
            expected = len(events) + 1
            if event.sequence != expected:
                raise ValueError(
                    f"session event sequence must be {expected}, got {event.sequence}"
                )
            events.append(event)
        return events

    def write_snapshot(self, snapshot: AgentSessionSnapshot) -> None:
        """Atomically replace the resumable session snapshot."""

        value = snapshot if isinstance(snapshot, AgentSessionSnapshot) else AgentSessionSnapshot.model_validate(snapshot)
        self._atomic_write_json(self.snapshot_path, value.model_dump(mode="json"))

    def load_snapshot(self) -> AgentSessionSnapshot | None:
        """Load the snapshot, returning ``None`` when it has not been written."""

        if not self.snapshot_path.exists():
            return None
        return AgentSessionSnapshot.model_validate_json(
            self.snapshot_path.read_text(encoding="utf-8")
        )

    def write_episode(self, episode: AgentEpisode) -> None:
        """Atomically write the legacy-compatible episode projection."""

        value = episode if isinstance(episode, AgentEpisode) else AgentEpisode.model_validate(episode)
        self._atomic_write_json(self.episode_path, value.model_dump(mode="json"))

    def load_episode(self) -> AgentEpisode | None:
        """Load the materialized episode projection when present."""

        if not self.episode_path.exists():
            return None
        return AgentEpisode.model_validate_json(
            self.episode_path.read_text(encoding="utf-8")
        )

    @staticmethod
    def _atomic_write_json(path: Path, value: Any) -> None:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                handle.write(encoded)
                handle.write("\n")
                handle.flush()
            os.replace(temporary_name, path)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass


class SessionIdentityMismatchError(ValueError):
    """Raised when resume inputs do not match the persisted session identity."""


@dataclass
class AgentSessionResult:
    """Public result returned by a Harness v3 session run or resume."""

    snapshot: AgentSessionSnapshot
    session_dir: Path
    final_text: str
    final_qa: QAReport
    episode: AgentEpisode
    events: list[SessionEvent]
    canonical_glossary_path: Path | None = None

    @property
    def final_status(self) -> str:
        """Return the durable session status."""

        return self.snapshot.status


_AGENT_ACTION_ADAPTER = TypeAdapter(AgentAction)
_BOOTSTRAP_TOOLS = ("escalate", "finish", "get_qa_findings", "tools.search")
_SHOWCASE_BOOTSTRAP_TOOLS = _BOOTSTRAP_TOOLS + (
    "read_paragraphs",
    "delegate_review",
    "select_term",
)
MAX_PARAGRAPH_RESULT_CHARS = 4000
MAX_SPECIALIST_REVIEWS = 16
_SHOWCASE_OBSERVATION_MAX_DEPTH = 8

ReviewHandler = Callable[..., list[SpecialistReview]]


def _registry_for_version(version: str) -> ToolRegistry:
    """Return the opt-in registry while keeping the v3 default object intact."""

    if version == SHOWCASE_TOOL_SCHEMA_VERSION:
        return SHOWCASE_TOOL_REGISTRY
    if version in {
        REGISTRY_TOOL_SCHEMA_VERSION,
        "agent-tools.v1",
        "agent-tools.v2",
    }:
        return AGENT_TOOL_REGISTRY
    raise ValueError(f"Unsupported agent tool schema version: {version}")


def _master_glossary(
    glossary: GlossaryParseResult | None,
    master_glossary: GlossaryParseResult | None,
) -> GlossaryParseResult:
    if glossary is None and master_glossary is None:
        raise TypeError("glossary or master_glossary is required")
    if glossary is not None and master_glossary is not None:
        # Treat the explicit master value as authoritative while retaining a
        # convenient ``glossary=`` alias for callers of the legacy API.
        return master_glossary
    return glossary if glossary is not None else master_glossary  # type: ignore[return-value]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bounded_event_payload(value: Any) -> Any:
    """Keep event payloads JSON-safe and modest without hardening machinery."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, BaseModel):
        return _bounded_event_payload(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {
            str(key): _bounded_event_payload(item)
            for key, item in list(value.items())[:100]
        }
    if isinstance(value, (list, tuple)):
        return [_bounded_event_payload(item) for item in list(value)[:100]]
    return f"<{type(value).__name__}>"


def _append_event(store: SessionStore, event_type: str, payload: Mapping[str, Any]) -> SessionEvent:
    return store.append(event_type, _bounded_event_payload(payload))


def _write_checkpoint(
    store: SessionStore,
    snapshot: AgentSessionSnapshot,
) -> SessionEvent:
    """Persist the projection and record its ordered checkpoint event."""

    event = _append_event(
        store,
        "checkpoint_written",
        {
            "status": snapshot.status,
            "step_count": len(snapshot.episode.steps),
            "patch_attempts": snapshot.patch_attempts,
        },
    )
    snapshot.last_event_sequence = event.sequence
    store.write_snapshot(snapshot)
    store.write_episode(snapshot.episode)
    return event


def _session_result(
    store: SessionStore,
    snapshot: AgentSessionSnapshot,
    *,
    canonical_glossary_path: Path | None = None,
) -> AgentSessionResult:
    events = store.read_events()
    episode = snapshot.episode
    final_qa = snapshot.executor.current_qa
    return AgentSessionResult(
        snapshot=snapshot,
        session_dir=store.session_dir,
        final_text=snapshot.executor.current_text,
        final_qa=final_qa,
        episode=episode,
        events=events,
        canonical_glossary_path=canonical_glossary_path,
    )


def _provider_metadata(provider: AgentActionProvider) -> tuple[str, str]:
    return str(getattr(provider, "provider_name", "")), str(getattr(provider, "model_name", ""))


def _normalized_provider_protocol(provider: AgentActionProvider) -> str:
    """Normalize provider transport exactly as the continuation request path does."""

    configured_protocol = getattr(provider, "tool_protocol", "json_prompt")
    return (
        configured_protocol
        if isinstance(configured_protocol, str)
        and configured_protocol in {"json_prompt", "native_function"}
        else "json_prompt"
    )


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _master_glossary_sha256(master_glossary: GlossaryParseResult) -> str:
    return _canonical_json_sha256(master_glossary.model_dump(mode="json"))


def _session_identity_candidate(
    *,
    provider: AgentActionProvider,
    source_text: str,
    master_glossary: GlossaryParseResult,
    run_id: str,
    story_slug: str,
    chapter: str,
    provider_mode: str,
    tool_schema_version: str = REGISTRY_TOOL_SCHEMA_VERSION,
    registry: ToolRegistry | None = None,
) -> AgentSessionIdentity:
    provider_name, model_name = _provider_metadata(provider)
    active_registry = registry or _registry_for_version(tool_schema_version)
    return AgentSessionIdentity(
        run_id=run_id,
        story_slug=story_slug,
        chapter=chapter,
        provider_mode=provider_mode,
        provider=provider_name,
        model=model_name,
        tool_schema_version=tool_schema_version,
        tool_protocol=_normalized_provider_protocol(provider),
        source_sha256=_sha256_text(source_text),
        master_glossary_sha256=_master_glossary_sha256(master_glossary),
        registry_sha256=active_registry.contract_sha256(),
    )


def _validate_session_identity(
    snapshot: AgentSessionSnapshot,
    candidate: AgentSessionIdentity,
) -> None:
    persisted = snapshot.identity
    if persisted is None:
        raise SessionIdentityMismatchError(
            "Session snapshot has no persisted identity; restart with "
            "run_repair_session in a new session directory."
        )
    persisted_data = persisted.model_dump(mode="json")
    candidate_data = candidate.model_dump(mode="json")
    mismatched_fields = sorted(
        field
        for field in persisted_data
        if persisted_data.get(field) != candidate_data.get(field)
    )
    if mismatched_fields:
        raise SessionIdentityMismatchError(
            "Session identity mismatch for fields: " + ", ".join(mismatched_fields)
        )


def _initial_snapshot(
    *,
    provider: AgentActionProvider,
    executor: RepairToolExecutor,
    run_id: str,
    story_slug: str,
    chapter: str,
    provider_mode: str,
    max_steps: int,
    max_patch_attempts: int,
    exposed_tool_names: list[str],
    identity: AgentSessionIdentity | None = None,
    instruction_context: dict[str, Any] | None = None,
    require_fidelity_review: bool = False,
    allow_nonregressing_patches: bool = False,
    max_delegation_rounds: int = 2,
) -> AgentSessionSnapshot:
    initial_qa = RepairToolExecutor._bounded_report(executor.current_qa)
    assert initial_qa is not None
    provider_name, model_name = _provider_metadata(provider)
    episode = AgentEpisode(
        episode_id=_episode_identifier(run_id, story_slug, chapter),
        run_id=run_id,
        story_slug=story_slug,
        chapter=chapter,
        provider_mode=provider_mode,
        provider=provider_name,
        model=model_name,
        max_steps=max_steps,
        max_patch_attempts=max_patch_attempts,
        initial_qa=initial_qa,
    )
    return AgentSessionSnapshot(
        status="running",
        episode=episode,
        executor=executor.snapshot(),
        identity=identity,
        exposed_tool_names=list(exposed_tool_names),
        instruction_context=instruction_context,
        require_fidelity_review=require_fidelity_review,
        allow_nonregressing_patches=allow_nonregressing_patches,
        max_delegation_rounds=max_delegation_rounds,
    )


def _action_from_provider(value: Any) -> AgentAction:
    try:
        return _AGENT_ACTION_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise AgentActionValidationError(
            "Agent action failed schema validation.",
            response=value.model_dump(mode="json") if isinstance(value, BaseModel) else value,
        ) from exc


def _invalid_v3_step(
    *,
    sequence: int,
    error: AgentActionValidationError,
    provider_call: ProviderCallRecord | None,
) -> AgentStep:
    return AgentStep(
        sequence=sequence,
        action={"tool": "invalid_action"},
        observation=AgentObservation(
            ok=False,
            kind="invalid_action",
            message="Provider response was not a valid repair action.",
            data={"response": error.parsed_response},
        ),
        provider_call=provider_call,
    )


def _proposal_text(before_text: str, term: str, target: str) -> tuple[str, str | None]:
    """Replace one existing ``source -> target`` line or append one."""

    lines = before_text.splitlines(keepends=True)
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line or line.startswith("#") or "->" not in line:
            continue
        source = line.split("->", 1)[0].strip()
        if source.casefold() != term.casefold():
            continue
        newline = "\n" if raw_line.endswith("\n") else ""
        prefix = raw_line[: len(raw_line) - len(raw_line.lstrip())]
        replacement = f"{prefix}{source} -> {target}{newline}"
        return "".join(lines[:index] + [replacement] + lines[index + 1 :]), line.split("->", 1)[1].strip()
    separator = "" if not before_text or before_text.endswith(("\n", "\r")) else "\n"
    return before_text + separator + f"{term} -> {target}\n", None


def _promotion_proposal(
    *,
    executor: RepairToolExecutor,
    action: PromoteGlossaryTermAction,
    canonical_glossary_path: Path | None,
    proposal_prefix: str,
) -> tuple[GlossaryPromotionProposal | None, AgentObservation]:
    if canonical_glossary_path is None:
        return None, AgentObservation(
            ok=False,
            kind="glossary_promotion_rejected",
            message="Canonical glossary path is required for promotion.",
            data={"term": action.term},
        )
    canonical_glossary_path = canonical_glossary_path.resolve()
    if not canonical_glossary_path.exists() or not canonical_glossary_path.is_file():
        return None, AgentObservation(
            ok=False,
            kind="glossary_promotion_rejected",
            message="Canonical glossary file does not exist.",
            data={"path": str(canonical_glossary_path)},
        )
    entry = executor._find_glossary_entry(action.term)
    if entry is None:
        return None, AgentObservation(
            ok=False,
            kind="glossary_promotion_rejected",
            message="No episode glossary entry is available for promotion.",
            data={"term": action.term},
        )
    before_text = canonical_glossary_path.read_text(encoding="utf-8")
    after_text, canonical_previous = _proposal_text(before_text, entry.source, entry.target)
    proposal_id = hashlib.sha256(
        f"{proposal_prefix}:{entry.source}:{_sha256_text(before_text)}:{_sha256_text(after_text)}".encode(
            "utf-8"
        )
    ).hexdigest()[:32]
    proposal = GlossaryPromotionProposal(
        proposal_id=proposal_id,
        canonical_glossary_path=str(canonical_glossary_path),
        term=entry.source,
        previous_target=canonical_previous or entry.target,
        proposed_target=entry.target,
        before_text=before_text,
        after_text=after_text,
        before_sha256=_sha256_text(before_text),
        after_sha256=_sha256_text(after_text),
    )
    return proposal, AgentObservation(
        ok=True,
        kind="glossary_promotion_pending",
        message="Canonical glossary promotion is awaiting reviewer approval.",
        data={
            "proposal_id": proposal.proposal_id,
            "term": proposal.term,
            "proposed_target": proposal.proposed_target,
            "before_sha256": proposal.before_sha256,
            "after_sha256": proposal.after_sha256,
        },
    )


def _bounded_paragraphs(
    *,
    executor: RepairToolExecutor,
    action: ReadParagraphsAction,
) -> AgentObservation:
    """Return at most six indexed paragraphs and 4000 result characters."""

    text = executor.source_text if action.document == "source" else executor.current_text
    paragraphs = split_paragraphs(text)
    selected: list[dict[str, Any]] = []
    used_chars = 0
    for index in range(action.start, min(action.start + action.count, len(paragraphs))):
        paragraph = paragraphs[index]
        separator_chars = 2 if selected else 0
        remaining = MAX_PARAGRAPH_RESULT_CHARS - used_chars - separator_chars
        if remaining <= 0:
            break
        if len(paragraph) > remaining:
            marker = "...[truncated]"
            paragraph = (
                marker[:remaining]
                if remaining <= len(marker)
                else paragraph[: remaining - len(marker)].rstrip() + marker
            )
        selected.append({"index": index, "text": paragraph})
        used_chars += separator_chars + len(paragraph)
        if len(paragraph) < len(paragraphs[index]):
            break
    return AgentObservation(
        ok=True,
        kind="paragraphs_read",
        message=f"Returned {len(selected)} bounded {action.document} paragraph(s).",
        data={
            "document": action.document,
            "start": action.start,
            "count": action.count,
            "paragraphs": selected,
        },
    )


def _bounded_showcase_json_value(value: Any, *, depth: int = 0) -> Any:
    """Bound v4 observations while retaining typed nested review evidence.

    The legacy repair projection intentionally stops at depth four.  v4
    coordinator observations contain one additional layer for paragraphs,
    findings, edits, and term suggestions, so use a deeper bound only for the
    opt-in showcase projection.  The item and string limits remain the same.
    """

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return RepairToolExecutor._bounded_snippet(value) or ""
    if isinstance(value, float):
        return value if value == value and value not in {float("inf"), float("-inf")} else None
    if depth >= _SHOWCASE_OBSERVATION_MAX_DEPTH:
        return "...[truncated]"
    if isinstance(value, Mapping):
        return {
            key: _bounded_showcase_json_value(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))[:32]
            if isinstance(key, str)
        }
    if isinstance(value, (list, tuple)):
        return [
            _bounded_showcase_json_value(item, depth=depth + 1)
            for item in list(value)[:32]
        ]
    return f"<{type(value).__name__}>"


def _bounded_showcase_observation(observation: AgentObservation) -> AgentObservation:
    """Copy a v4 observation without collapsing review evidence at depth four."""

    return observation.model_copy(
        update={
            "message": RepairToolExecutor._bounded_snippet(observation.message) or "",
            "data": _bounded_showcase_json_value(observation.data),
        },
        deep=True,
    )


def _review_observation_payload(review: SpecialistReview) -> dict[str, Any]:
    """Project a child result into safe coordinator-visible evidence.

    Provider call metadata and child artifact paths are durable diagnostics but
    are intentionally absent from the next model context.  They are retained
    in the typed snapshot instead.
    """

    return {
        "role": review.role,
        "status": review.status,
        "summary": review.summary,
        "findings": [finding.model_dump(mode="json") for finding in review.findings],
        "proposed_edits": [edit.model_dump(mode="json") for edit in review.proposed_edits],
        "term_suggestions": [
            suggestion.model_dump(mode="json") for suggestion in review.term_suggestions
        ],
        "draft_sha256": review.draft_sha256,
    }


def _coerce_specialist_reviews(
    value: Any,
    *,
    requested_roles: list[str],
) -> list[SpecialistReview]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("review_handler must return a list of SpecialistReview values")
    reviews: list[SpecialistReview] = []
    seen_roles: set[str] = set()
    for raw_review in value:
        review = (
            raw_review
            if isinstance(raw_review, SpecialistReview)
            else SpecialistReview.model_validate(raw_review)
        )
        if review.role not in requested_roles:
            raise ValueError(f"review returned unrequested specialist role: {review.role}")
        if review.role in seen_roles:
            raise ValueError(f"review returned duplicate specialist role: {review.role}")
        seen_roles.add(review.role)
        reviews.append(review)
    if not reviews:
        raise ValueError("review_handler must return at least one SpecialistReview")
    missing_roles = set(requested_roles) - seen_roles
    if missing_roles:
        raise ValueError(
            "review_handler omitted requested specialist role(s): "
            + ", ".join(sorted(missing_roles))
        )
    return reviews


def _select_term_from_reviews(
    *,
    executor: RepairToolExecutor,
    snapshot: AgentSessionSnapshot,
    action: SelectTermAction,
) -> tuple[AgentObservation, QAReport | None, QAReport | None]:
    current_digest = _sha256_text(executor.current_text)
    suggestion: TermSuggestion | None = None
    for review in reversed(snapshot.specialist_reviews):
        if review.role != "terminology" or review.status != "completed":
            continue
        if review.draft_sha256 != current_digest:
            continue
        for candidate in review.term_suggestions:
            if candidate.term == action.term and candidate.target == action.target:
                suggestion = candidate
                break
        if suggestion is not None:
            break
    if suggestion is None:
        return (
            AgentObservation(
                ok=False,
                kind="term_selection_rejected",
                message="Term selection must exactly match a current completed terminology suggestion.",
                data={"term": action.term, "target": action.target, "draft_sha256": current_digest},
            ),
            None,
            None,
        )

    matching_index = next(
        (
            index
            for index, entry in enumerate(executor.episode_glossary.entries)
            if entry.source == action.term or entry.source.casefold() == action.term.casefold()
        ),
        None,
    )
    if matching_index is None:
        executor.episode_glossary.entries.append(
            GlossaryEntry(source=action.term, target=action.target, candidates=[action.target])
        )
    else:
        entry = executor.episode_glossary.entries[matching_index]
        candidates = list(entry.candidates)
        if action.target not in candidates:
            candidates.insert(0, action.target)
        executor.episode_glossary.entries[matching_index] = entry.model_copy(
            update={"target": action.target, "candidates": candidates}
        )
    executor.glossary = executor.episode_glossary
    before_qa = executor.current_qa
    executor.current_qa = executor._run_qa(executor.current_text)
    return (
        AgentObservation(
            ok=True,
            kind="term_selected",
            message="Terminology suggestion selected as an episode-local glossary override.",
            data={
                "term": action.term,
                "target": action.target,
                "rationale": action.rationale,
                "draft_sha256": current_digest,
                "persistent_write": False,
            },
        ),
        RepairToolExecutor._bounded_report(before_qa),
        RepairToolExecutor._bounded_report(executor.current_qa),
    )


def _fidelity_gate_observation(
    *,
    snapshot: AgentSessionSnapshot,
    current_text: str,
) -> AgentObservation | None:
    """Return a blocking finish observation when a fresh fidelity review is absent or unsafe."""

    current_digest = _sha256_text(current_text)
    current_reviews = [
        review
        for review in snapshot.specialist_reviews
        if review.role == "fidelity"
        and review.status == "completed"
        and review.draft_sha256 == current_digest
    ]
    if not current_reviews:
        return AgentObservation(
            ok=False,
            kind="fidelity_review_required",
            message=(
                "Cannot finish until a completed fidelity review covers the current draft; "
                "delegate a fresh review or escalate if the issue is unrepairable."
            ),
            data={"draft_sha256": current_digest, "action": "delegate_review_or_escalate"},
        )
    blocking = [
        finding.model_dump(mode="json")
        for review in current_reviews
        for finding in review.findings
        if finding.category == "fidelity" and finding.blocking
    ]
    if blocking:
        return AgentObservation(
            ok=False,
            kind="fidelity_findings_blocking",
            message=(
                f"Cannot finish while {len(blocking)} blocking fidelity finding(s) remain; "
                "repair them or escalate for human review."
            ),
            data={
                "draft_sha256": current_digest,
                "blocking_findings": blocking,
                "action": "repair_or_escalate",
            },
        )
    return None


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
            handle.flush()
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _mark_terminal(
    snapshot: AgentSessionSnapshot,
    *,
    session_status: str,
    final_status: str,
    summary: str,
) -> None:
    snapshot.status = session_status  # type: ignore[assignment]
    snapshot.episode.final_status = final_status  # type: ignore[assignment]
    snapshot.episode.summary = summary
    snapshot.executor = snapshot.executor.model_copy(
        update={"escalated": snapshot.executor.escalated, "finished": snapshot.executor.finished},
        deep=True,
    )


def _continue_v3_session(
    *,
    store: SessionStore,
    snapshot: AgentSessionSnapshot,
    provider: AgentActionProvider,
    source_text: str,
    master_glossary: GlossaryParseResult,
    provider_mode: str,
    terminology_resolver: TerminologyResolver | None,
    canonical_glossary_path: Path | None,
    terminology_source_context_chars: int,
    terminology_translation_context_chars: int,
    tool_schema_version: str = REGISTRY_TOOL_SCHEMA_VERSION,
    registry: ToolRegistry | None = None,
    instruction_context: dict[str, Any] | None = None,
    review_handler: ReviewHandler | None = None,
    require_fidelity_review: bool = False,
    allow_nonregressing_patches: bool = False,
    max_delegation_rounds: int = 2,
) -> AgentSessionResult:
    """Continue a bounded session until terminal, budget, interruption, or approval pause."""

    active_registry = registry or _registry_for_version(tool_schema_version)

    episode = snapshot.episode
    # Persisted showcase policy is authoritative on continuation.  The
    # arguments remain useful for fresh sessions and for callers that invoke
    # this private continuation helper directly.
    require_fidelity_review = snapshot.require_fidelity_review
    allow_nonregressing_patches = snapshot.allow_nonregressing_patches
    max_delegation_rounds = snapshot.max_delegation_rounds
    if allow_nonregressing_patches and tool_schema_version != SHOWCASE_TOOL_SCHEMA_VERSION:
        raise ValueError(
            "allow_nonregressing_patches is supported only for showcase sessions"
        )
    if allow_nonregressing_patches and not require_fidelity_review:
        raise ValueError(
            "allow_nonregressing_patches requires require_fidelity_review=True"
        )
    executor = RepairToolExecutor.from_snapshot(
        snapshot.executor,
        source_text=source_text,
        glossary=master_glossary,
        run_id=episode.run_id,
        story_slug=episode.story_slug,
        chapter=episode.chapter,
        terminology_resolver=terminology_resolver,
        terminology_source_context_chars=terminology_source_context_chars,
        terminology_translation_context_chars=terminology_translation_context_chars,
        allow_nonregressing_patches=allow_nonregressing_patches,
    )
    policy = DefaultToolPolicy()
    prior_steps: list[PriorObservableStep] = []
    for raw in snapshot.prior_steps[-24:]:
        try:
            prior_steps.append(PriorObservableStep.model_validate(raw))
        except (TypeError, ValueError):
            continue

    tool_protocol = _normalized_provider_protocol(provider)

    while len(episode.steps) < episode.max_steps:
        step_number = len(episode.steps) + 1
        request = AgentActionRequest(
            episode_id=episode.episode_id,
            step_number=step_number,
            story_slug=episode.story_slug,
            chapter=episode.chapter,
            current_findings=_episode_findings(executor),
            remaining_steps=episode.max_steps - len(episode.steps),
            remaining_patch_attempts=max(episode.max_patch_attempts - snapshot.patch_attempts, 0),
            prior_steps=list(prior_steps[-24:]),
            tool_schema_version=tool_schema_version,
            tool_protocol=tool_protocol,
            exposed_tool_names=tuple(snapshot.exposed_tool_names),
            instruction_context=(
                snapshot.instruction_context
                if tool_schema_version == SHOWCASE_TOOL_SCHEMA_VERSION
                else None
            ),
        )
        _append_event(
            store,
            "model_requested",
            {
                "step": step_number,
                "tool_protocol": request.tool_protocol,
                "exposed_tool_names": list(snapshot.exposed_tool_names),
            },
        )
        before_call_count = len(_provider_call_records(provider))
        try:
            proposed = provider.next_action(request)
            if proposed is None:
                _append_event(
                    store,
                    "session_interrupted",
                    {"step": step_number, "reason": "provider_returned_no_decision"},
                )
                snapshot.executor = executor.snapshot()
                _write_checkpoint(store, snapshot)
                return _session_result(
                    store,
                    snapshot,
                    canonical_glossary_path=canonical_glossary_path,
                )
            action = _action_from_provider(proposed)
        except KeyboardInterrupt:
            # The previous checkpoint contains all completed actions.  Record
            # the current projection before allowing the interruption to
            # propagate so a later resume can continue from that boundary.
            snapshot.executor = executor.snapshot()
            _write_checkpoint(store, snapshot)
            raise
        except AgentActionValidationError as exc:
            call = _latest_provider_call(provider, before_call_count)
            _append_event(store, "tool_proposed", {"step": step_number, "tool": "invalid_action"})
            _append_event(store, "tool_rejected", {"step": step_number, "reason": str(exc)})
            step = _invalid_v3_step(sequence=step_number, error=exc, provider_call=call)
            episode.steps.append(step)
            episode.final_qa = RepairToolExecutor._bounded_report(executor.current_qa)
            snapshot.executor = executor.snapshot()
            _write_checkpoint(store, snapshot)
            continue
        except Exception as exc:  # noqa: BLE001 - provider is an explicit session boundary.
            _append_event(store, "tool_rejected", {"step": step_number, "reason": type(exc).__name__})
            snapshot.executor = executor.snapshot()
            _mark_terminal(
                snapshot,
                session_status="failed",
                final_status="failed",
                summary=f"Provider failure ({type(exc).__name__})",
            )
            finished_event = _append_event(store, "run_finished", {"status": snapshot.status})
            snapshot.last_event_sequence = finished_event.sequence
            store.write_snapshot(snapshot)
            store.write_episode(episode)
            return _session_result(store, snapshot, canonical_glossary_path=canonical_glossary_path)

        persisted_action = _bounded_action_payload(action)
        _append_event(store, "tool_proposed", {"step": step_number, "action": persisted_action})
        try:
            spec = active_registry.spec(action.tool)
        except ToolCallValidationError:
            spec = None
        exposed = action.tool in set(snapshot.exposed_tool_names)
        if spec is None or not exposed:
            reason = "Unknown registered tool." if spec is None else "Tool was not exposed."
            _append_event(
                store,
                "tool_rejected",
                {"step": step_number, "tool": action.tool, "reason": reason},
            )
            observation = AgentObservation(
                ok=False,
                kind="tool_rejected",
                message=reason,
                data={"tool": action.tool, "exposed_tool_names": list(snapshot.exposed_tool_names)},
            )
            step = AgentStep(
                sequence=step_number,
                action=persisted_action,
                observation=observation,
                provider_call=_latest_provider_call(provider, before_call_count),
            )
            episode.steps.append(step)
            prior_steps.append(
                PriorObservableStep(
                    sequence=step.sequence,
                    action=_bounded_typed_action(action),
                    observation=observation,
                )
            )
            snapshot.prior_steps = [item.model_dump(mode="json") for item in prior_steps[-24:]]
            snapshot.executor = executor.snapshot()
            _write_checkpoint(store, snapshot)
            continue

        _append_event(store, "tool_validated", {"step": step_number, "tool": action.tool})
        decision = policy.before_tool(action, spec)
        _append_event(store, "policy_decided", {"step": step_number, "tool": action.tool, "decision": decision})
        if decision.outcome == "reject" or decision.outcome == "abort":
            observation = AgentObservation(ok=False, kind="tool_rejected", message=decision.reason, data={"rule_id": decision.rule_id})
            _append_event(store, "tool_rejected", {"step": step_number, "tool": action.tool, "reason": decision.reason})
            execution = ToolExecutionResult(observation=observation)
        elif isinstance(action, SearchToolsAction):
            matches = active_registry.search(action.query, limit=action.limit)
            names = sorted({*snapshot.exposed_tool_names, *(spec.name for spec in matches)})
            if "tools.search" not in names:
                names.append("tools.search")
            snapshot.exposed_tool_names = names
            observation = AgentObservation(
                ok=True,
                kind="tools_searched",
                message=f"Discovered {len(matches)} trusted tool(s).",
                data={"query": action.query, "matches": [item.name for item in matches], "exposed_tool_names": names},
            )
            execution = ToolExecutionResult(observation=observation)
            _append_event(store, "tool_executed", {"step": step_number, "tool": action.tool, "matches": [item.name for item in matches]})
        elif isinstance(action, ReadParagraphsAction):
            execution = ToolExecutionResult(observation=_bounded_paragraphs(executor=executor, action=action))
            _append_event(
                store,
                "tool_executed",
                {"step": step_number, "tool": action.tool, "kind": execution.observation.kind},
            )
        elif isinstance(action, DelegateReviewAction):
            if snapshot.delegation_rounds >= snapshot.max_delegation_rounds:
                execution = ToolExecutionResult(
                    observation=AgentObservation(
                        ok=False,
                        kind="delegation_budget_exhausted",
                        message="Delegation round budget is exhausted; no specialist was invoked.",
                        data={
                            "max_delegation_rounds": snapshot.max_delegation_rounds,
                            "delegation_rounds": snapshot.delegation_rounds,
                        },
                    )
                )
            else:
                snapshot.delegation_rounds += 1
                if review_handler is None:
                    execution = ToolExecutionResult(
                        observation=AgentObservation(
                            ok=False,
                            kind="review_unavailable",
                            message="No review handler is configured for specialist delegation.",
                            data={"specialists": list(action.specialists)},
                        )
                    )
                else:
                    try:
                        raw_reviews = review_handler(
                            action,
                            source_text=source_text,
                            translated_text=executor.current_text,
                            glossary=executor.episode_glossary.model_copy(deep=True),
                            chapter=episode.chapter,
                            session_dir=store.session_dir,
                            step_number=step_number,
                        )
                        reviews = _coerce_specialist_reviews(
                            raw_reviews,
                            requested_roles=list(action.specialists),
                        )
                    except Exception as exc:  # noqa: BLE001 - child boundary is fail-closed.
                        execution = ToolExecutionResult(
                            observation=AgentObservation(
                                ok=False,
                                kind="review_failed",
                                message="Specialist review failed; no review result was accepted.",
                                data={
                                    "specialists": list(action.specialists),
                                    "error_type": type(exc).__name__,
                                },
                            )
                        )
                    else:
                        snapshot.specialist_reviews.extend(reviews)
                        snapshot.specialist_reviews = snapshot.specialist_reviews[-MAX_SPECIALIST_REVIEWS:]
                        public_reviews = [_review_observation_payload(review) for review in reviews]
                        execution = ToolExecutionResult(
                            observation=AgentObservation(
                                ok=True,
                                kind="specialist_reviews_received",
                                message=f"Received {len(reviews)} bounded specialist review(s).",
                                data={
                                    "round": snapshot.delegation_rounds,
                                    "objective": action.objective,
                                    "reviews": public_reviews,
                                },
                            )
                        )
                        _append_event(
                            store,
                            "specialist_reviews_received",
                            {
                                "step": step_number,
                                "round": snapshot.delegation_rounds,
                                "reviews": public_reviews,
                            },
                        )
            _append_event(
                store,
                "tool_executed",
                {
                    "step": step_number,
                    "tool": action.tool,
                    "kind": execution.observation.kind,
                    "round": snapshot.delegation_rounds,
                },
            )
        elif isinstance(action, SelectTermAction):
            observation, qa_before, qa_after = _select_term_from_reviews(
                executor=executor,
                snapshot=snapshot,
                action=action,
            )
            execution = ToolExecutionResult(
                observation=observation,
                qa_before=qa_before,
                qa_after=qa_after,
            )
            _append_event(
                store,
                "tool_executed",
                {"step": step_number, "tool": action.tool, "kind": observation.kind},
            )
        elif decision.outcome == "require_approval" and isinstance(action, PromoteGlossaryTermAction):
            proposal, observation = _promotion_proposal(
                executor=executor,
                action=action,
                canonical_glossary_path=canonical_glossary_path,
                proposal_prefix=episode.episode_id,
            )
            execution = ToolExecutionResult(observation=observation)
            if proposal is not None:
                request_receipt = ApprovalRequest(
                    proposal_id=proposal.proposal_id,
                    tool=action.tool,
                    arguments=action.model_dump(mode="json"),
                    reason=decision.reason,
                )
                snapshot.pending_approval = request_receipt
                snapshot.pending_proposal = proposal
                snapshot.pending_decision = None
                _append_event(store, "approval_requested", request_receipt)
            else:
                _append_event(store, "glossary_promotion_rejected", observation.data)
        elif isinstance(action, (SubmitPatchAction, NormalizePunctuationAction)) and snapshot.patch_attempts >= episode.max_patch_attempts:
            execution = ToolExecutionResult(
                observation=AgentObservation(
                    ok=False,
                    kind="patch_budget_exhausted",
                    message="Patch attempt budget is exhausted; no additional patch was executed.",
                    data={"max_patch_attempts": episode.max_patch_attempts},
                ),
                qa_before=RepairToolExecutor._bounded_report(executor.current_qa),
                qa_after=RepairToolExecutor._bounded_report(executor.current_qa),
            )
            _append_event(
                store,
                "tool_executed",
                {"step": step_number, "tool": action.tool, "kind": execution.observation.kind},
            )
        else:
            fidelity_gate = (
                _fidelity_gate_observation(
                    snapshot=snapshot,
                    current_text=executor.current_text,
                )
                if isinstance(action, FinishAction) and require_fidelity_review
                else None
            )
            if fidelity_gate is not None:
                execution = ToolExecutionResult(
                    observation=fidelity_gate,
                    qa_before=RepairToolExecutor._bounded_report(executor.current_qa),
                    qa_after=RepairToolExecutor._bounded_report(executor.current_qa),
                )
                _append_event(
                    store,
                    "tool_executed",
                    {"step": step_number, "tool": action.tool, "kind": fidelity_gate.kind},
                )
            else:
                execution = executor.execute(action)
            _append_event(
                store,
                "tool_executed",
                {"step": step_number, "tool": action.tool, "kind": execution.observation.kind},
            )

        bounded_observation = (
            _bounded_showcase_observation(execution.observation)
            if tool_schema_version == SHOWCASE_TOOL_SCHEMA_VERSION
            else _bounded_observation(execution.observation)
        )
        step = AgentStep(
            sequence=step_number,
            action=persisted_action,
            observation=bounded_observation,
            provider_call=_latest_provider_call(provider, before_call_count),
            auxiliary_provider_calls=list(execution.auxiliary_provider_calls),
            qa_before=execution.qa_before,
            qa_after=execution.qa_after,
        )
        episode.steps.append(step)
        episode.final_qa = RepairToolExecutor._bounded_report(executor.current_qa)
        if execution.resolution is not None:
            episode.terminology_resolutions.append(execution.resolution)
        if _action_payload_is_bounded(action):
            prior_steps.append(
                PriorObservableStep(
                    sequence=step.sequence,
                    action=_bounded_typed_action(action),
                    observation=bounded_observation,
                )
            )
        snapshot.prior_steps = [item.model_dump(mode="json") for item in prior_steps[-24:]]
        if isinstance(action, PromoteGlossaryTermAction) and snapshot.pending_proposal is not None:
            snapshot.executor = executor.snapshot()
            snapshot.status = "awaiting_approval"
            _write_checkpoint(store, snapshot)
            return _session_result(store, snapshot, canonical_glossary_path=canonical_glossary_path)

        if isinstance(action, (SubmitPatchAction, NormalizePunctuationAction)):
            snapshot.patch_attempts += 1
        snapshot.executor = executor.snapshot()
        if bounded_observation.kind == "finished" and bounded_observation.ok:
            _mark_terminal(snapshot, session_status="completed", final_status="verified", summary=bounded_observation.message)
        elif bounded_observation.kind == "escalated" and bounded_observation.ok:
            _mark_terminal(snapshot, session_status="completed", final_status="escalated", summary=bounded_observation.message)
        elif bounded_observation.kind == "patch_budget_exhausted":
            _mark_terminal(snapshot, session_status="completed", final_status="budget_exhausted", summary=bounded_observation.message)
        _write_checkpoint(store, snapshot)
        if snapshot.status != "running":
            finished_event = _append_event(store, "run_finished", {"status": snapshot.status, "final_status": snapshot.episode.final_status})
            snapshot.last_event_sequence = finished_event.sequence
            store.write_snapshot(snapshot)
            store.write_episode(episode)
            return _session_result(store, snapshot, canonical_glossary_path=canonical_glossary_path)

    snapshot.executor = executor.snapshot()
    _mark_terminal(snapshot, session_status="completed", final_status="budget_exhausted", summary="Repair episode action budget is exhausted.")
    finished_event = _append_event(store, "run_finished", {"status": snapshot.status, "final_status": snapshot.episode.final_status})
    snapshot.last_event_sequence = finished_event.sequence
    store.write_snapshot(snapshot)
    store.write_episode(episode)
    return _session_result(store, snapshot, canonical_glossary_path=canonical_glossary_path)


def run_repair_session(
    *,
    provider: AgentActionProvider,
    session_dir: str | Path,
    source_text: str,
    translated_text: str,
    glossary: GlossaryParseResult | None = None,
    master_glossary: GlossaryParseResult | None = None,
    canonical_glossary_path: str | Path | None = None,
    run_id: str,
    story_slug: str,
    chapter: str,
    provider_mode: str,
    max_steps: int = 5,
    max_patch_attempts: int = 2,
    dynamic_tools: bool = True,
    terminology_resolver: TerminologyResolver | None = None,
    terminology_source_context_chars: int = 800,
    terminology_translation_context_chars: int = 800,
    tool_schema_version: str = REGISTRY_TOOL_SCHEMA_VERSION,
    instruction_context: dict[str, Any] | None = None,
    review_handler: ReviewHandler | None = None,
    require_fidelity_review: bool = False,
    allow_nonregressing_patches: bool = False,
    max_delegation_rounds: int = 2,
) -> AgentSessionResult:
    """Run a durable session with registry/policy enforcement."""

    if max_steps < 1 or max_patch_attempts < 1:
        raise ValueError("max_steps and max_patch_attempts must be at least 1")
    if max_delegation_rounds < 0 or max_delegation_rounds > 8:
        raise ValueError("max_delegation_rounds must be between 0 and 8")
    if allow_nonregressing_patches and tool_schema_version != SHOWCASE_TOOL_SCHEMA_VERSION:
        raise ValueError(
            "allow_nonregressing_patches is supported only for showcase sessions"
        )
    if allow_nonregressing_patches and not require_fidelity_review:
        raise ValueError(
            "allow_nonregressing_patches requires require_fidelity_review=True"
        )
    registry = _registry_for_version(tool_schema_version)
    if tool_schema_version != SHOWCASE_TOOL_SCHEMA_VERSION:
        # Showcase instructions and specialist policy are deliberately opt-in.
        instruction_context = None
        review_handler = None
        require_fidelity_review = False
    else:
        instruction_context = _bounded_instruction_context(instruction_context)
    master = _master_glossary(glossary, master_glossary)
    canonical_path = Path(canonical_glossary_path) if canonical_glossary_path is not None else None
    store = SessionStore(session_dir)
    existing_artifacts = tuple(
        path
        for path in (store.events_path, store.snapshot_path, store.episode_path)
        if path.exists() or path.is_symlink()
    )
    if existing_artifacts:
        names = ", ".join(path.name for path in existing_artifacts)
        raise FileExistsError(
            f"Session artifacts already exist in {store.session_dir} ({names}); "
            "use resume_repair_session to continue or choose a new directory."
        )
    executor = RepairToolExecutor(
        source_text=source_text,
        translated_text=translated_text,
        glossary=master,
        run_id=run_id,
        story_slug=story_slug,
        chapter=chapter,
        terminology_resolver=terminology_resolver,
        terminology_source_context_chars=terminology_source_context_chars,
        terminology_translation_context_chars=terminology_translation_context_chars,
        allow_nonregressing_patches=allow_nonregressing_patches,
    )
    identity = _session_identity_candidate(
        provider=provider,
        source_text=source_text,
        master_glossary=master,
        run_id=run_id,
        story_slug=story_slug,
        chapter=chapter,
        provider_mode=provider_mode,
        tool_schema_version=tool_schema_version,
        registry=registry,
    )
    if not dynamic_tools:
        exposed = list(registry.visible_specs())
    else:
        exposed = list(
            _SHOWCASE_BOOTSTRAP_TOOLS
            if tool_schema_version == SHOWCASE_TOOL_SCHEMA_VERSION
            else _BOOTSTRAP_TOOLS
        )
        exposed = [name for name in exposed if name in {spec.name for spec in registry.visible_specs()}]
    # Store logical names, not ToolSpec instances, in the public projection.
    exposed_names = [item.name if hasattr(item, "name") else str(item) for item in exposed]
    snapshot = _initial_snapshot(
        provider=provider,
        executor=executor,
        run_id=run_id,
        story_slug=story_slug,
        chapter=chapter,
        provider_mode=provider_mode,
        max_steps=max_steps,
        max_patch_attempts=max_patch_attempts,
        exposed_tool_names=sorted(set(exposed_names)),
        identity=identity,
        instruction_context=instruction_context,
        require_fidelity_review=require_fidelity_review,
        allow_nonregressing_patches=allow_nonregressing_patches,
        max_delegation_rounds=max_delegation_rounds,
    )
    _append_event(store, "run_started", {"run_id": run_id, "episode_id": snapshot.episode.episode_id})
    _append_event(store, "tools_exposed", {"dynamic_tools": dynamic_tools, "tool_names": snapshot.exposed_tool_names})
    _write_checkpoint(store, snapshot)
    return _continue_v3_session(
        store=store,
        snapshot=snapshot,
        provider=provider,
        source_text=source_text,
        master_glossary=master,
        provider_mode=provider_mode,
        terminology_resolver=terminology_resolver,
        canonical_glossary_path=canonical_path,
        terminology_source_context_chars=terminology_source_context_chars,
        terminology_translation_context_chars=terminology_translation_context_chars,
        tool_schema_version=tool_schema_version,
        registry=registry,
        instruction_context=instruction_context,
        review_handler=review_handler,
        require_fidelity_review=require_fidelity_review,
        allow_nonregressing_patches=allow_nonregressing_patches,
        max_delegation_rounds=max_delegation_rounds,
    )


def resume_repair_session(
    *,
    session_dir: str | Path,
    provider: AgentActionProvider,
    source_text: str,
    glossary: GlossaryParseResult | None = None,
    master_glossary: GlossaryParseResult | None = None,
    canonical_glossary_path: str | Path | None = None,
    terminology_resolver: TerminologyResolver | None = None,
    run_id: str | None = None,
    story_slug: str | None = None,
    chapter: str | None = None,
    provider_mode: str | None = None,
    decision: str | None = None,
    reviewer: str = "operator",
    note: str = "",
    terminology_source_context_chars: int = 800,
    terminology_translation_context_chars: int = 800,
    tool_schema_version: str | None = None,
    instruction_context: dict[str, Any] | None = None,
    review_handler: ReviewHandler | None = None,
    require_fidelity_review: bool | None = None,
    allow_nonregressing_patches: bool | None = None,
    max_delegation_rounds: int | None = None,
) -> AgentSessionResult:
    """Continue a session, applying a decision when approval is pending."""

    store = SessionStore(session_dir)
    snapshot = store.load_snapshot()
    if snapshot is None:
        raise FileNotFoundError(f"No session snapshot found in {store.session_dir}")
    master = _master_glossary(glossary, master_glossary)
    effective_run_id = snapshot.episode.run_id if run_id is None else run_id
    effective_story_slug = snapshot.episode.story_slug if story_slug is None else story_slug
    effective_chapter = snapshot.episode.chapter if chapter is None else chapter
    effective_provider_mode = (
        snapshot.episode.provider_mode if provider_mode is None else provider_mode
    )
    # The legacy resume API implicitly means v3.  Callers resuming an opt-in
    # showcase run pass v4 explicitly; keeping the implicit default at v3 is
    # what lets us detect a tampered persisted identity instead of trusting it.
    effective_tool_schema_version = (
        REGISTRY_TOOL_SCHEMA_VERSION if tool_schema_version is None else tool_schema_version
    )
    registry = _registry_for_version(effective_tool_schema_version)
    candidate = _session_identity_candidate(
        provider=provider,
        source_text=source_text,
        master_glossary=master,
        run_id=effective_run_id,
        story_slug=effective_story_slug,
        chapter=effective_chapter,
        provider_mode=effective_provider_mode,
        tool_schema_version=effective_tool_schema_version,
        registry=registry,
    )
    _validate_session_identity(snapshot, candidate)
    if effective_tool_schema_version == SHOWCASE_TOOL_SCHEMA_VERSION:
        if (
            instruction_context is not None
            and _bounded_instruction_context(instruction_context) != snapshot.instruction_context
        ):
            raise SessionIdentityMismatchError("Session instruction context mismatch")
        if (
            require_fidelity_review is not None
            and require_fidelity_review != snapshot.require_fidelity_review
        ):
            raise SessionIdentityMismatchError("Session fidelity-review policy mismatch")
        if (
            allow_nonregressing_patches is not None
            and allow_nonregressing_patches != snapshot.allow_nonregressing_patches
        ):
            raise SessionIdentityMismatchError("Session nonregressing-patch policy mismatch")
        if snapshot.allow_nonregressing_patches and not snapshot.require_fidelity_review:
            raise SessionIdentityMismatchError(
                "Session nonregressing-patch policy requires fidelity review"
            )
        if (
            max_delegation_rounds is not None
            and max_delegation_rounds != snapshot.max_delegation_rounds
        ):
            raise SessionIdentityMismatchError("Session delegation budget mismatch")
        effective_instruction_context = snapshot.instruction_context
        effective_review_handler = review_handler
        effective_require_fidelity_review = snapshot.require_fidelity_review
        effective_allow_nonregressing_patches = snapshot.allow_nonregressing_patches
        effective_max_delegation_rounds = snapshot.max_delegation_rounds
    else:
        if snapshot.allow_nonregressing_patches:
            raise SessionIdentityMismatchError(
                "Session nonregressing-patch policy is unsupported outside showcase sessions"
            )
        if allow_nonregressing_patches:
            raise ValueError(
                "allow_nonregressing_patches is supported only for showcase sessions"
            )
        effective_instruction_context = None
        effective_review_handler = None
        effective_require_fidelity_review = False
        effective_allow_nonregressing_patches = False
        effective_max_delegation_rounds = snapshot.max_delegation_rounds
    canonical_path = (
        Path(canonical_glossary_path).resolve()
        if canonical_glossary_path is not None
        else None
    )
    # Terminal sessions are deliberately idempotent: no event, provider call,
    # or filesystem write is performed for a repeated decision.
    if snapshot.status == "running" and decision is None:
        return _continue_v3_session(
            store=store,
            snapshot=snapshot,
            provider=provider,
            source_text=source_text,
            master_glossary=master,
            provider_mode=effective_provider_mode,
            terminology_resolver=terminology_resolver,
            canonical_glossary_path=canonical_path,
            terminology_source_context_chars=terminology_source_context_chars,
            terminology_translation_context_chars=terminology_translation_context_chars,
            tool_schema_version=effective_tool_schema_version,
            registry=registry,
            instruction_context=effective_instruction_context,
            review_handler=effective_review_handler,
            require_fidelity_review=effective_require_fidelity_review,
            allow_nonregressing_patches=effective_allow_nonregressing_patches,
            max_delegation_rounds=effective_max_delegation_rounds,
        )
    if snapshot.status != "awaiting_approval":
        return _session_result(store, snapshot, canonical_glossary_path=canonical_path)
    if decision is None:
        raise ValueError("An explicit approved/rejected decision is required for pending approval")
    if decision not in {"approved", "rejected"}:
        raise ValueError("decision must be 'approved' or 'rejected'")
    if snapshot.pending_approval is None or snapshot.pending_proposal is None:
        raise ValueError("session is awaiting approval without a pending proposal")
    if decision == "approved" and canonical_path is None:
        raise ValueError("canonical_glossary_path is required to approve a pending promotion")
    receipt = ApprovalDecision(
        proposal_id=snapshot.pending_approval.proposal_id,
        decision=decision,
        reviewer=reviewer,
        note=note,
    )
    snapshot.pending_decision = receipt
    _append_event(store, "approval_decided", receipt)
    if decision == "rejected":
        snapshot.executor = snapshot.executor.model_copy(update={"escalated": True}, deep=True)
        snapshot.pending_approval = None
        snapshot.pending_proposal = None
        snapshot.status = "rejected"
        snapshot.episode.final_status = "escalated"
        snapshot.episode.summary = receipt.note
        _append_event(store, "glossary_promotion_rejected", {"proposal_id": receipt.proposal_id, "decision": receipt.decision})
        _write_checkpoint(store, snapshot)
        finished = _append_event(store, "run_finished", {"status": snapshot.status, "final_status": snapshot.episode.final_status})
        snapshot.last_event_sequence = finished.sequence
        store.write_snapshot(snapshot)
        store.write_episode(snapshot.episode)
        return _session_result(store, snapshot, canonical_glossary_path=canonical_path)

    proposal = snapshot.pending_proposal
    if str(canonical_path) != proposal.canonical_glossary_path:
        snapshot.executor = snapshot.executor.model_copy(update={"escalated": True}, deep=True)
        snapshot.pending_approval = None
        snapshot.pending_proposal = None
        snapshot.status = "blocked"
        snapshot.episode.final_status = "escalated"
        snapshot.episode.summary = "Approval target does not match the reviewed canonical glossary path."
        _append_event(
            store,
            "glossary_promotion_rejected",
            {"proposal_id": proposal.proposal_id, "reason": "path_mismatch"},
        )
        _write_checkpoint(store, snapshot)
        finished = _append_event(
            store,
            "run_finished",
            {"status": snapshot.status, "final_status": snapshot.episode.final_status},
        )
        snapshot.last_event_sequence = finished.sequence
        store.write_snapshot(snapshot)
        store.write_episode(snapshot.episode)
        return _session_result(store, snapshot, canonical_glossary_path=canonical_path)
    if not canonical_path.exists() or not canonical_path.is_file():
        current_text = ""
    else:
        current_text = canonical_path.read_text(encoding="utf-8")
    current_digest = _sha256_text(current_text)
    if current_digest == proposal.before_sha256 and current_text == proposal.before_text:
        _atomic_write_text(canonical_path, proposal.after_text)
        wrote = True
    elif current_digest == proposal.after_sha256 and current_text == proposal.after_text:
        wrote = False
    else:
        snapshot.executor = snapshot.executor.model_copy(update={"escalated": True}, deep=True)
        snapshot.pending_approval = None
        snapshot.pending_proposal = None
        snapshot.status = "blocked"
        snapshot.episode.final_status = "escalated"
        snapshot.episode.summary = "Canonical glossary changed since the proposal was prepared."
        _append_event(store, "glossary_promotion_rejected", {"proposal_id": proposal.proposal_id, "reason": "content_changed"})
        _write_checkpoint(store, snapshot)
        finished = _append_event(store, "run_finished", {"status": snapshot.status, "final_status": snapshot.episode.final_status})
        snapshot.last_event_sequence = finished.sequence
        store.write_snapshot(snapshot)
        store.write_episode(snapshot.episode)
        return _session_result(store, snapshot, canonical_glossary_path=canonical_path)

    _append_event(store, "glossary_promotion_applied", {"proposal_id": proposal.proposal_id, "wrote": wrote, "after_sha256": proposal.after_sha256})
    # Retain the approval as an observable prior step before requesting the
    # next model action.  It is not a second provider action or patch budget
    # attempt.
    try:
        approved_action = _AGENT_ACTION_ADAPTER.validate_python(snapshot.pending_approval.arguments)
    except ValidationError:
        approved_action = PromoteGlossaryTermAction(term=proposal.term, rationale="approved promotion")
    approved_observation = AgentObservation(
        ok=True,
        kind="glossary_promotion_approved",
        message="Reviewer approved the canonical glossary promotion.",
        data={"proposal_id": proposal.proposal_id, "wrote": wrote, "after_sha256": proposal.after_sha256},
    )
    prior_steps = [PriorObservableStep.model_validate(raw) for raw in snapshot.prior_steps[-24:]]
    prior_steps.append(
        PriorObservableStep(
            sequence=max(len(snapshot.episode.steps), 1),
            action=approved_action,
            observation=approved_observation,
        )
    )
    snapshot.prior_steps = [item.model_dump(mode="json") for item in prior_steps[-24:]]
    snapshot.pending_approval = None
    snapshot.pending_proposal = None
    snapshot.status = "running"
    _write_checkpoint(store, snapshot)
    result = _continue_v3_session(
        store=store,
        snapshot=snapshot,
        provider=provider,
        source_text=source_text,
        master_glossary=master,
        provider_mode=effective_provider_mode,
        terminology_resolver=terminology_resolver,
        canonical_glossary_path=canonical_path,
        terminology_source_context_chars=terminology_source_context_chars,
        terminology_translation_context_chars=terminology_translation_context_chars,
        tool_schema_version=effective_tool_schema_version,
        registry=registry,
        instruction_context=effective_instruction_context,
        review_handler=effective_review_handler,
        require_fidelity_review=effective_require_fidelity_review,
        allow_nonregressing_patches=effective_allow_nonregressing_patches,
        max_delegation_rounds=effective_max_delegation_rounds,
    )
    return result


__all__ = [
    "AgentSessionSnapshot",
    "ApprovalDecision",
    "ApprovalRequest",
    "GlossaryPromotionProposal",
    "PolicyDecision",
    "RepairExecutorSnapshot",
    "SessionEvent",
    "SessionStore",
    "SessionIdentityMismatchError",
    "AgentSessionResult",
    "ReviewHandler",
    "run_repair_session",
    "resume_repair_session",
]
