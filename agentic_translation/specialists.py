"""Bounded specialist reviews for the showcase harness.

Specialists are deliberately less capable than the coordinator.  They receive
one immutable source/draft snapshot, can inspect small context windows and the
glossary, and return evidence plus proposed changes.  They never mutate the
translation or glossary themselves; the coordinator and verifier own those
effects.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from .agent_models import (
    CompleteReviewAction,
    DelegateReviewAction,
    LookupGlossaryAction,
    ReadParagraphsAction,
    ReviewFinding,
    SpecialistReview,
    TermSuggestion,
    TextEdit,
)
from .agent_provider import (
    AgentActionProvider,
    AgentActionRequest,
    AgentActionValidationError,
    PriorObservableStep,
    SHOWCASE_TOOL_SCHEMA_VERSION,
)
from .agent_tools import (
    SPECIALIST_TOOL_REGISTRY,
    ToolCall,
    ToolCallValidationError,
)
from .models import GlossaryParseResult, ProviderCallRecord
from .text import split_paragraphs


_MAX_EXCERPT_CHARS = 4000
_MAX_CHILD_STEPS = 4
_MAX_STEP_CONTEXT = 24
_ROLE_INSTRUCTIONS: dict[str, str] = {
    "terminology": (
        "Review terminology consistency between the source, translation, and trusted glossary. "
        "Use the read-only tools to inspect evidence before reporting any finding. "
        "Return exact evidence and bounded proposals; do not edit files or promote terms."
    ),
    "fidelity": (
        "Review source fidelity, including omissions, negation, relationships, and meaning shifts. "
        "Use the read-only tools to inspect evidence before reporting any finding. "
        "Return exact evidence and bounded proposals; do not edit files or promote terms."
    ),
}
_SPECIALIST_TOOL_NAMES = tuple(
    spec.name for spec in SPECIALIST_TOOL_REGISTRY.visible_specs()
)


def _clip(value: str, limit: int = _MAX_EXCERPT_CHARS) -> str:
    """Return a deterministic excerpt whose total length never exceeds limit."""

    if len(value) <= limit:
        return value
    marker = "\n[excerpt truncated]"
    if limit <= len(marker):
        return marker[:limit]
    return value[: limit - len(marker)].rstrip() + marker


def _bounded_json(value: Any, *, limit: int = _MAX_EXCERPT_CHARS) -> Any:
    """Bound diagnostic values without invoking arbitrary object encoders."""

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _clip(value, limit)
    if isinstance(value, Mapping):
        return {
            str(key): _bounded_json(item, limit=limit)
            for key, item in list(value.items())[:32]
            if isinstance(key, str)
        }
    if isinstance(value, (list, tuple)):
        return [_bounded_json(item, limit=limit) for item in list(value)[:32]]
    return type(value).__name__


def _record_model(value: Any) -> ProviderCallRecord | None:
    try:
        return value if isinstance(value, ProviderCallRecord) else ProviderCallRecord.model_validate(value)
    except (TypeError, ValueError, ValidationError):
        return None


def _provider_records(provider: Any) -> list[ProviderCallRecord]:
    raw = getattr(provider, "call_records", ())
    if not isinstance(raw, (list, tuple)):
        return []
    result: list[ProviderCallRecord] = []
    for item in raw:
        record = _record_model(item)
        if record is not None:
            result.append(record)
    return result


def _latest_provider_record(provider: Any, before_count: int) -> ProviderCallRecord | None:
    records = _provider_records(provider)
    # A provider may omit a receipt for a failed/fixture call.  Do not attach
    # the previous step's receipt to the current action in that case.
    return records[-1] if len(records) > before_count else None


def _provider_protocol(provider: Any) -> Literal["json_prompt", "native_function"]:
    protocol = getattr(provider, "tool_protocol", "json_prompt")
    if protocol not in {"json_prompt", "native_function"}:
        return "json_prompt"
    return protocol  # type: ignore[return-value]


def _raw_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, ToolCall):
        try:
            return value.as_action_payload()
        except ToolCallValidationError:
            return {"tool": value.name}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {"_response_type": "str"}
        if isinstance(parsed, (Mapping, BaseModel, ToolCall)):
            return _raw_payload(parsed)
        return {"_response_type": "str"}
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="python")
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        return {"_response_type": type(value).__name__}
    return {str(key): _bounded_json(item) for key, item in payload.items() if isinstance(key, str)}


def _rejection_payload(error: Exception) -> dict[str, Any]:
    response = getattr(error, "response", None)
    if response is None:
        response = getattr(error, "parsed_response", None)
    return _raw_payload(response) if response is not None else {"_error": type(error).__name__}


def _step_payload(
    *,
    sequence: int,
    action: Mapping[str, Any],
    observation: Mapping[str, Any],
    provider_call: ProviderCallRecord | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sequence": sequence,
        "action": dict(action),
        "observation": dict(observation),
    }
    if provider_call is not None:
        payload["provider_call"] = provider_call.model_dump(mode="json")
    return payload


def _observation(
    *,
    ok: bool,
    kind: str,
    message: str,
    data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "kind": kind,
        "message": message,
        "data": _bounded_json(dict(data or {})),
    }


def _make_review(
    *,
    role: str,
    status: Literal["completed", "failed"],
    summary: str,
    findings: list[ReviewFinding] | None = None,
    proposed_edits: list[TextEdit] | None = None,
    term_suggestions: list[TermSuggestion] | None = None,
    steps: list[dict[str, Any]] | None = None,
    provider_calls: list[ProviderCallRecord] | None = None,
    draft_sha256: str,
) -> SpecialistReview:
    return SpecialistReview.model_validate(
        {
            "role": role,
            "status": status,
            "summary": _clip(summary, 1500),
            "findings": findings or [],
            "proposed_edits": proposed_edits or [],
            "term_suggestions": term_suggestions or [],
            "steps": steps or [],
            "provider_calls": provider_calls or [],
            "draft_sha256": draft_sha256,
        }
    )


class SpecialistRunner:
    """Run a bounded, isolated child review for each selected specialist role."""

    def __init__(
        self,
        provider_factory: Callable[[str, str], AgentActionProvider],
        *,
        style_guide: str = "",
        max_steps: int = 4,
        max_workers: int = 2,
        review_instructions: str = "",
    ) -> None:
        if not callable(provider_factory):
            raise TypeError("provider_factory must be callable")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        self.provider_factory = provider_factory
        self.style_guide = _clip(style_guide)
        self.review_instructions = _clip(review_instructions)
        # Keep the child budget a hard harness invariant even if a caller
        # supplies a larger convenience value.
        self.max_steps = min(max_steps, _MAX_CHILD_STEPS)
        self.max_workers = min(max_workers, 2)

    def __call__(
        self,
        action: DelegateReviewAction,
        *,
        source_text: str,
        translated_text: str,
        glossary: GlossaryParseResult,
        chapter: str,
        session_dir: str | Path,
        step_number: int,
    ) -> list[SpecialistReview]:
        if not isinstance(action, DelegateReviewAction):
            action = DelegateReviewAction.model_validate(action)
        if isinstance(step_number, bool) or not isinstance(step_number, int) or step_number < 1:
            raise ValueError("step_number must be a positive integer")
        # Materialize all input-derived values before starting threads.  Child
        # workers then close over immutable snapshots and cannot observe a
        # coordinator mutation half way through a batch.
        source_snapshot = str(source_text)
        translation_snapshot = str(translated_text)
        glossary_snapshot = glossary.model_copy(deep=True)
        roles = tuple(dict.fromkeys(action.specialists))
        child_args = {
            "source_text": source_snapshot,
            "translated_text": translation_snapshot,
            "glossary": glossary_snapshot,
            "chapter": chapter,
            "step_number": step_number,
            "objective": action.objective,
            "session_dir": session_dir,
        }
        worker_count = min(self.max_workers, len(roles)) if roles else 1
        results: dict[str, SpecialistReview] = {}
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="specialist") as executor:
            futures = {
                executor.submit(self._run_child, role, **child_args): role
                for role in roles
            }
            for future in as_completed(futures):
                role = futures[future]
                try:
                    results[role] = future.result()
                except Exception as exc:  # noqa: BLE001 - child failure is a review result.
                    child_id = f"{chapter}-{step_number}-{role}"
                    review = _make_review(
                        role=role,
                        status="failed",
                        summary=f"Specialist failed ({type(exc).__name__}).",
                        draft_sha256=_sha256(translation_snapshot),
                    )
                    results[role] = review
                    self._write_artifact(
                        session_dir=session_dir,
                        child_id=child_id,
                        chapter=chapter,
                        step_number=step_number,
                        review=review,
                    )
        return [results[role] for role in roles]

    def _run_child(
        self,
        role: str,
        *,
        source_text: str,
        translated_text: str,
        glossary: GlossaryParseResult,
        chapter: str,
        step_number: int,
        objective: str,
        session_dir: str | Path | None = None,
    ) -> SpecialistReview:
        child_id = f"{chapter}-{step_number}-{role}"
        draft_sha256 = _sha256(translated_text)
        provider: Any = None
        steps: list[dict[str, Any]] = []
        prior_steps: list[PriorObservableStep] = []
        try:
            # Keep each child context isolated even when a provider or test
            # fixture attempts an in-process mutation of its glossary view.
            glossary = glossary.model_copy(deep=True)
            provider = self.provider_factory(role, child_id)
            for sequence in range(1, self.max_steps + 1):
                request = self._request(
                    provider=provider,
                    child_id=child_id,
                    role=role,
                    chapter=chapter,
                    sequence=sequence,
                    objective=objective,
                    source_text=source_text,
                    translated_text=translated_text,
                    glossary=glossary,
                    prior_steps=prior_steps,
                )
                before_calls = len(_provider_records(provider))
                try:
                    raw_action = provider.next_action(request)
                except (AgentActionValidationError, ToolCallValidationError) as exc:
                    provider_call = _latest_provider_record(provider, before_calls)
                    action_payload = _rejection_payload(exc)
                    observation = _observation(
                        ok=False,
                        kind="tool_rejected",
                        message="Specialist action was not exposed or failed validation.",
                        data={"error": str(exc), "tool": action_payload.get("tool", "")},
                    )
                    steps.append(
                        _step_payload(
                            sequence=sequence,
                            action=action_payload,
                            observation=observation,
                            provider_call=provider_call,
                        )
                    )
                    # A malformed/unknown action consumes this step.  If it
                    # was a valid AgentAction, preserve it as observable
                    # history; unknown actions cannot be represented safely in
                    # PriorObservableStep and are therefore omitted there.
                    continue
                except Exception as exc:  # noqa: BLE001 - provider is a child boundary.
                    provider_calls = _provider_records(provider)
                    review = _make_review(
                        role=role,
                        status="failed",
                        summary=f"Specialist provider failed ({type(exc).__name__}).",
                        steps=steps,
                        provider_calls=provider_calls,
                        draft_sha256=draft_sha256,
                    )
                    self._write_artifact(
                        session_dir=session_dir,
                        child_id=child_id,
                        chapter=chapter,
                        step_number=step_number,
                        review=review,
                    )
                    return review

                provider_call = _latest_provider_record(provider, before_calls)
                try:
                    child_action = self._validate_action(raw_action)
                except (ToolCallValidationError, ValidationError, ValueError) as exc:
                    action_payload = _raw_payload(raw_action)
                    observation = _observation(
                        ok=False,
                        kind="tool_rejected",
                        message="Specialist action was not exposed or failed validation.",
                        data={"error": str(exc), "tool": action_payload.get("tool", "")},
                    )
                    steps.append(
                        _step_payload(
                            sequence=sequence,
                            action=action_payload,
                            observation=observation,
                            provider_call=provider_call,
                        )
                    )
                    continue

                action_payload = child_action.model_dump(mode="json")
                if isinstance(child_action, ReadParagraphsAction):
                    observation = self._read_paragraphs(
                        child_action,
                        source_text=source_text,
                        translated_text=translated_text,
                    )
                elif isinstance(child_action, LookupGlossaryAction):
                    observation = self._lookup_glossary(child_action, glossary=glossary)
                elif isinstance(child_action, CompleteReviewAction):
                    observation = _observation(
                        ok=True,
                        kind="review_completed",
                        message="Specialist review accepted as a structured result.",
                        data={
                            "finding_count": len(child_action.findings),
                            "proposed_edit_count": len(child_action.proposed_edits),
                            "term_suggestion_count": len(child_action.term_suggestions),
                        },
                    )
                    steps.append(
                        _step_payload(
                            sequence=sequence,
                            action=action_payload,
                            observation=observation,
                            provider_call=provider_call,
                        )
                    )
                    review = _make_review(
                        role=role,
                        status="completed",
                        summary=child_action.summary,
                        findings=list(child_action.findings),
                        proposed_edits=list(child_action.proposed_edits),
                        term_suggestions=list(child_action.term_suggestions),
                        steps=steps,
                        provider_calls=_provider_records(provider),
                        draft_sha256=draft_sha256,
                    )
                    self._write_artifact(
                        session_dir=session_dir,
                        child_id=child_id,
                        chapter=chapter,
                        step_number=step_number,
                        review=review,
                    )
                    return review
                else:  # pragma: no cover - registry exposure prevents this.
                    observation = _observation(
                        ok=False,
                        kind="tool_rejected",
                        message="Specialist action is not executable by this child.",
                        data={"tool": action_payload.get("tool", "")},
                    )

                steps.append(
                    _step_payload(
                        sequence=sequence,
                        action=action_payload,
                        observation=observation,
                        provider_call=provider_call,
                    )
                )
                self._append_prior_step(
                    prior_steps,
                    sequence=sequence,
                    action=child_action,
                    observation=observation,
                )

            review = _make_review(
                role=role,
                status="failed",
                summary=f"Specialist exhausted its {self.max_steps}-step budget without complete_review.",
                steps=steps,
                provider_calls=_provider_records(provider),
                draft_sha256=draft_sha256,
            )
            self._write_artifact(
                session_dir=session_dir,
                child_id=child_id,
                chapter=chapter,
                step_number=step_number,
                review=review,
            )
            return review
        except Exception as exc:  # noqa: BLE001 - child failures never escape the batch.
            review = _make_review(
                role=role,
                status="failed",
                summary=f"Specialist failed ({type(exc).__name__}).",
                steps=steps,
                provider_calls=_provider_records(provider) if provider is not None else [],
                draft_sha256=draft_sha256,
            )
            self._write_artifact(
                session_dir=session_dir,
                child_id=child_id,
                chapter=chapter,
                step_number=step_number,
                review=review,
            )
            return review

    def _request(
        self,
        *,
        provider: Any,
        child_id: str,
        role: str,
        chapter: str,
        sequence: int,
        objective: str,
        source_text: str,
        translated_text: str,
        glossary: GlossaryParseResult,
        prior_steps: list[PriorObservableStep],
    ) -> AgentActionRequest:
        source_paragraphs = split_paragraphs(source_text)
        translation_paragraphs = split_paragraphs(translated_text)
        instruction_context = {
            "role": role,
            "role_instructions": _ROLE_INSTRUCTIONS.get(role, "Review the assigned material and report evidence.") + (" " + self.review_instructions if self.review_instructions else ""),
            "style_guide": self.style_guide,
            "objective": _clip(objective, 1000),
            "initial_evidence": {
                "source_paragraph_count": len(source_paragraphs),
                "translation_paragraph_count": len(translation_paragraphs),
                "glossary_entry_count": len(glossary.entries),
            },
        }
        return AgentActionRequest(
            episode_id=child_id,
            step_number=sequence,
            story_slug="specialist-review",
            chapter=chapter,
            current_findings=[],
            remaining_steps=self.max_steps - sequence + 1,
            remaining_patch_attempts=0,
            prior_steps=list(prior_steps[-_MAX_STEP_CONTEXT:]),
            tool_schema_version=SHOWCASE_TOOL_SCHEMA_VERSION,
            tool_protocol=_provider_protocol(provider),
            exposed_tool_names=_SPECIALIST_TOOL_NAMES,
            instruction_context=instruction_context,
        )

    @staticmethod
    def _validate_action(raw_action: Any) -> Any:
        if isinstance(raw_action, ToolCall):
            call = raw_action
        elif isinstance(raw_action, BaseModel):
            call = ToolCall.from_json_action(raw_action)
        elif isinstance(raw_action, (Mapping, str)):
            call = ToolCall.from_json_action(raw_action)
        else:
            raise ToolCallValidationError("Specialist action must be a tool call or action object")
        return SPECIALIST_TOOL_REGISTRY.action_from_call(
            call,
            visible_names=_SPECIALIST_TOOL_NAMES,
        )

    @staticmethod
    def _append_prior_step(
        prior_steps: list[PriorObservableStep],
        *,
        sequence: int,
        action: Any,
        observation: Mapping[str, Any],
    ) -> None:
        # Only typed AgentAction instances are appended.  Unknown rejected
        # actions remain in the persisted step trace but never enter a typed
        # request context.
        try:
            prior_steps.append(
                PriorObservableStep(
                    sequence=sequence,
                    action=action,
                    observation=observation,
                )
            )
        except (TypeError, ValueError, ValidationError):
            return

    @staticmethod
    def _read_paragraphs(
        action: ReadParagraphsAction,
        *,
        source_text: str,
        translated_text: str,
    ) -> dict[str, Any]:
        paragraphs = split_paragraphs(source_text if action.document == "source" else translated_text)
        selected = paragraphs[action.start : action.start + action.count]
        bounded: list[dict[str, Any]] = []
        remaining = _MAX_EXCERPT_CHARS
        for index, paragraph in enumerate(selected, start=action.start):
            separator_cost = 2 if bounded else 0
            if remaining <= separator_cost:
                break
            excerpt = _clip(paragraph, remaining - separator_cost)
            bounded.append({"index": index, "text": excerpt})
            remaining -= separator_cost + len(excerpt)
            if len(excerpt) < len(paragraph):
                break
        return _observation(
            ok=True,
            kind="paragraphs_read",
            message=f"Read {len(bounded)} {action.document} paragraph(s).",
            data={
                "document": action.document,
                "start": action.start,
                "requested_count": action.count,
                "count": action.count,
                "returned_count": len(bounded),
                "paragraph_count": len(bounded),
                "total_count": len(paragraphs),
                "paragraphs": bounded,
            },
        )

    @staticmethod
    def _lookup_glossary(
        action: LookupGlossaryAction,
        *,
        glossary: GlossaryParseResult,
    ) -> dict[str, Any]:
        needle = action.term.casefold()
        matches = []
        for entry in glossary.entries:
            candidates = [entry.source, entry.target, *entry.candidates, *entry.blocked_variants]
            if any(isinstance(value, str) and value.casefold() == needle for value in candidates):
                matches.append(entry.model_dump(mode="json"))
        return _observation(
            ok=True,
            kind="glossary_lookup",
            message=f"Found {len(matches)} glossary match(es).",
            data={"term": action.term, "count": len(matches), "matches": matches[:16]},
        )

    @staticmethod
    def _write_artifact(
        *,
        session_dir: str | Path | None,
        child_id: str,
        chapter: str,
        step_number: int,
        review: SpecialistReview,
    ) -> None:
        if session_dir is None:
            return
        children_dir = Path(session_dir) / "children"
        children_dir.mkdir(parents=True, exist_ok=True)
        # Chapter ids are normally numeric, but avoid allowing an accidental
        # path separator from an operator supplied id to escape children/.
        safe_child_id = child_id.replace("/", "_").replace("\\", "_")
        payload = {
            "schema_version": "specialist-review.v1",
            "child_id": child_id,
            "chapter": chapter,
            "step_number": step_number,
            **review.model_dump(mode="json"),
        }
        (children_dir / f"{safe_child_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "SPECIALIST_TOOL_REGISTRY",
    "SpecialistReview",
    "SpecialistRunner",
]
