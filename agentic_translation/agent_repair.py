"""Deterministic, bounded tools for the repair-agent prototype.

The executor deliberately owns only an in-memory working translation.  Every
candidate patch is scored by the existing translation QA before it can mutate
that working copy; no filesystem or shell operations are exposed here.
"""

from __future__ import annotations

import hashlib
import errno
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .agent_models import (
    AgentAction,
    AgentEpisode,
    AgentObservation,
    AgentStep,
    EscalateAction,
    FinishAction,
    GetQAFindingsAction,
    LookupGlossaryAction,
    ResolveTerminologyAction,
    ReadSourceContextAction,
    ReadTranslationContextAction,
    SubmitPatchAction,
)
from .agent_provider import (
    AgentActionProvider,
    AgentActionRequest,
    AgentActionValidationError,
    AgentToolSchemaVersion,
    BASE_TOOL_SCHEMA_VERSION,
    PriorObservableStep,
)
from .models import GlossaryEntry, GlossaryParseResult, ProviderCallRecord, QAFinding, QAReport
from .qa import run_translation_qa
from .repair import validate_patch_improves_qa
from .terminology import TerminologyResolutionError, TerminologyResolver
from .terminology_models import TerminologyRequest, TerminologyResolution
from .text import split_paragraphs


# Keep context useful but bounded even if a model asks for an unreasonably
# large radius.  Paragraphs themselves are split by the existing text helper.
MAX_CONTEXT_RADIUS = 3
MAX_CONTEXT_CHARS = 2000
TRUNCATION_MARKER = "...[truncated]"


@dataclass
class ToolExecutionResult:
    """Observable result of one typed repair tool invocation."""

    observation: AgentObservation
    qa_before: QAReport | None = None
    qa_after: QAReport | None = None
    resolution: TerminologyResolution | None = None
    auxiliary_provider_calls: list[ProviderCallRecord] = field(default_factory=list)


def finding_identity(finding: QAFinding) -> tuple[str, str, int | None, int | None, str | None]:
    """Return a stable identity for a deterministic QA finding.

    Snippets and ``found`` values are intentionally excluded: those values can
    shrink or change as a patch fixes an existing location.  The check,
    chapter, location, and expected canonical value are stable enough to catch
    genuinely new findings without treating an edited snippet as a new one.
    """

    location = finding.location
    return (
        finding.check_id,
        location.chapter,
        location.paragraph_index,
        location.line_index,
        finding.expected,
    )


def finding_identities(
    report_or_findings: QAReport | list[QAFinding],
) -> set[tuple[str, str, int | None, int | None, str | None]]:
    """Return stable identities for all findings in a QA report or list."""

    findings = (
        report_or_findings.findings
        if isinstance(report_or_findings, QAReport)
        else report_or_findings
    )
    return {finding_identity(finding) for finding in findings}


class RepairToolExecutor:
    """Execute the seven bounded repair tools against an in-memory translation."""

    def __init__(
        self,
        source_text: str,
        translated_text: str,
        glossary: GlossaryParseResult,
        *,
        run_id: str = "agent-repair",
        story_slug: str = "demo",
        chapter: str = "0001",
        terminology_resolver: TerminologyResolver | None = None,
        terminology_source_context_chars: int = 800,
        terminology_translation_context_chars: int = 800,
    ) -> None:
        self.source_text = source_text
        # Each episode receives an isolated glossary copy.  Resolution can
        # update only this copy, so a provisional term never mutates the
        # story's master glossary or another chapter's QA.
        self.master_glossary = glossary
        self.episode_glossary = glossary.model_copy(deep=True)
        self.glossary = self.episode_glossary
        self.terminology_resolver = terminology_resolver
        self.terminology_source_context_chars = max(
            100, min(int(terminology_source_context_chars), 4000)
        )
        self.terminology_translation_context_chars = max(
            100, min(int(terminology_translation_context_chars), 4000)
        )
        self.run_id = run_id
        self.story_slug = story_slug
        self.chapter = chapter
        self.current_text = translated_text
        self.current_qa = self._run_qa(translated_text)
        self.escalated = False
        self.finished = False

    def _run_qa(self, translated_text: str, *, glossary: GlossaryParseResult | None = None) -> QAReport:
        return run_translation_qa(
            run_id=self.run_id,
            story_slug=self.story_slug,
            chapter=self.chapter,
            source_text=self.source_text,
            translated_text=translated_text,
            glossary=glossary or self.episode_glossary,
        )

    @staticmethod
    def _bounded_snippet(snippet: str | None, limit: int = MAX_CONTEXT_CHARS) -> str | None:
        if snippet is None or len(snippet) <= limit:
            return snippet
        prefix_length = max(limit - len(TRUNCATION_MARKER), 0)
        return snippet[:prefix_length].rstrip() + TRUNCATION_MARKER

    @classmethod
    def _bounded_report(cls, report: QAReport | None) -> QAReport | None:
        if report is None:
            return None
        findings: list[QAFinding] = []
        for finding in report.findings:
            snippet = cls._bounded_snippet(finding.location.snippet)
            location = finding.location.model_copy(update={"snippet": snippet})
            findings.append(
                finding.model_copy(
                    update={
                        "location": location,
                        "found": cls._bounded_snippet(finding.found),
                        "expected": cls._bounded_snippet(finding.expected),
                        "message": cls._bounded_snippet(finding.message),
                        "suggested_action": cls._bounded_snippet(finding.suggested_action),
                    }
                )
            )
        return report.model_copy(update={"findings": findings}, deep=True)

    @classmethod
    def _result(
        cls,
        observation: AgentObservation,
        *,
        qa_before: QAReport | None = None,
        qa_after: QAReport | None = None,
        resolution: TerminologyResolution | None = None,
        auxiliary_provider_calls: list[ProviderCallRecord] | None = None,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            observation=observation,
            qa_before=cls._bounded_report(qa_before),
            qa_after=cls._bounded_report(qa_after),
            resolution=resolution,
            auxiliary_provider_calls=list(auxiliary_provider_calls or []),
        )

    @staticmethod
    def _observation(
        *,
        ok: bool,
        kind: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> AgentObservation:
        return AgentObservation(
            ok=ok,
            kind=kind,
            message=message,
            data=data or {},
        )

    def execute(self, action: AgentAction) -> ToolExecutionResult:
        """Execute one already schema-validated action.

        AgentAction is a discriminated Pydantic union.  The explicit dispatch
        below keeps the allowed tool surface auditable and gives a safe
        rejected observation if a caller bypasses type validation at runtime.
        """

        if self.escalated or self.finished:
            return self._result(
                self._observation(
                    ok=False,
                    kind="terminal_rejected",
                    message="Repair executor is already in a terminal state.",
                    data={"escalated": self.escalated, "finished": self.finished},
                )
            )

        if isinstance(action, GetQAFindingsAction):
            return self._get_qa_findings()
        if isinstance(action, ReadSourceContextAction):
            return self._read_context(action, source=True)
        if isinstance(action, ReadTranslationContextAction):
            return self._read_context(action, source=False)
        if isinstance(action, LookupGlossaryAction):
            return self._lookup_glossary(action)
        if isinstance(action, ResolveTerminologyAction):
            return self._resolve_terminology(action)
        if isinstance(action, SubmitPatchAction):
            return self._submit_patch(action)
        if isinstance(action, EscalateAction):
            return self._escalate(action)
        if isinstance(action, FinishAction):
            return self._finish(action)

        # This is defensive only; a valid AgentAction cannot reach this branch.
        return self._result(
            self._observation(
                ok=False,
                kind="tool_rejected",
                message="Unsupported repair tool action.",
            )
        )

    def _get_qa_findings(self) -> ToolExecutionResult:
        bounded_report = self._bounded_report(self.current_qa)
        assert bounded_report is not None
        findings = [finding.model_dump() for finding in bounded_report.findings]
        observation = self._observation(
            ok=True,
            kind="qa_findings",
            message=f"Current translation QA has {len(findings)} finding(s).",
            data={
                "count": len(findings),
                "findings": findings,
                "score": self.current_qa.score,
            },
        )
        return self._result(observation, qa_before=self.current_qa, qa_after=self.current_qa)

    def _read_context(
        self,
        action: ReadSourceContextAction | ReadTranslationContextAction,
        *,
        source: bool,
    ) -> ToolExecutionResult:
        finding_index = action.finding_index
        if finding_index < 0 or finding_index >= len(self.current_qa.findings):
            return self._result(
                self._observation(
                    ok=False,
                    kind="context_rejected",
                    message=f"finding_index is out of range: {finding_index}.",
                    data={"finding_index": finding_index},
                )
            )

        finding = self.current_qa.findings[finding_index]
        paragraph_index = finding.location.paragraph_index
        text = self.source_text if source else self.current_text
        paragraphs = split_paragraphs(text)
        if paragraph_index is None or paragraph_index < 0 or paragraph_index >= len(paragraphs):
            return self._result(
                self._observation(
                    ok=False,
                    kind="context_rejected",
                    message="Finding has no valid paragraph context in the requested text.",
                    data={
                        "finding_index": finding_index,
                        "paragraph_index": paragraph_index,
                    },
                )
            )

        radius = min(action.radius, MAX_CONTEXT_RADIUS)
        start = max(0, paragraph_index - radius)
        end = min(len(paragraphs), paragraph_index + radius + 1)
        context = "\n\n".join(paragraphs[start:end])
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS].rstrip()

        kind = "source_context" if source else "translation_context"
        observation = self._observation(
            ok=True,
            kind=kind,
            message=f"Returned bounded {kind.replace('_', ' ')}.",
            data={
                "finding_index": finding_index,
                "paragraph_index": paragraph_index,
                "radius": radius,
                "context": context,
            },
        )
        return self._result(observation)

    def _lookup_glossary(self, action: LookupGlossaryAction) -> ToolExecutionResult:
        entry = self._find_glossary_entry(action.term)
        if entry is None:
            return self._result(
                self._observation(
                    ok=False,
                    kind="glossary_not_found",
                    message=f"No glossary entry found for {action.term!r}.",
                    data={"term": action.term},
                )
            )

        observation = self._observation(
            ok=True,
            kind="glossary_lookup",
            message=f"Found glossary entry for {entry.source!r}.",
            data={
                "term": entry.source,
                "canonical": entry.target,
                "canonical_term": entry.target,
                "target": entry.target,
                "candidates": list(entry.candidates),
                "blocked_variants": list(entry.blocked_variants),
            },
        )
        return self._result(observation)

    def _find_glossary_entry(self, term: str) -> GlossaryEntry | None:
        lowered = term.casefold()
        for entry in self.glossary.entries:
            if entry.source == term or entry.source.casefold() == lowered:
                return entry
        return None

    def _terminology_request(self, action: ResolveTerminologyAction) -> TerminologyRequest:
        """Build a bounded resolver request from the selected finding/context."""

        source_context = self.source_text
        translation_context = self.current_text
        finding_id: str | None = None
        if action.finding_index is not None and 0 <= action.finding_index < len(self.current_qa.findings):
            finding = self.current_qa.findings[action.finding_index]
            paragraph_index = finding.location.paragraph_index
            source_paragraphs = split_paragraphs(self.source_text)
            translation_paragraphs = split_paragraphs(self.current_text)
            if paragraph_index is not None:
                start = max(0, paragraph_index - MAX_CONTEXT_RADIUS)
                end = paragraph_index + MAX_CONTEXT_RADIUS + 1
                source_context = "\n\n".join(source_paragraphs[start:end])
                translation_context = "\n\n".join(translation_paragraphs[start:end])
            finding_id = ":".join(
                filter(
                    None,
                    [
                        finding.check_id,
                        str(finding.location.paragraph_index),
                        str(finding.location.line_index),
                    ],
                )
            )
        source_context = self._bounded_snippet(source_context, self.terminology_source_context_chars) or ""
        translation_context = self._bounded_snippet(
            translation_context, self.terminology_translation_context_chars
        ) or ""

        relevant: list[GlossaryEntry] = []
        seen: set[str] = set()
        for entry in self.episode_glossary.entries:
            if (
                entry.source == action.term
                or entry.source.casefold() == action.term.casefold()
                or entry.source in source_context
                or entry.source in translation_context
            ) and entry.source.casefold() not in seen:
                relevant.append(entry.model_copy(deep=True))
                seen.add(entry.source.casefold())
            if len(relevant) >= 100:
                break
        blocked: list[str] = []
        for value in self.episode_glossary.blocked_variants:
            if value not in blocked:
                blocked.append(value)
        for entry in relevant:
            for value in entry.blocked_variants:
                if value not in blocked:
                    blocked.append(value)
        return TerminologyRequest(
            story_slug=self.story_slug,
            chapter=self.chapter,
            source_term=action.term,
            source_context=source_context,
            translation_context=translation_context,
            glossary_entries=relevant,
            blocked_variants=blocked[:100],
            finding_id=finding_id,
        )

    def _resolve_terminology(self, action: ResolveTerminologyAction) -> ToolExecutionResult:
        if self.terminology_resolver is None:
            return self._result(
                self._observation(
                    ok=False,
                    kind="terminology_consensus_unavailable",
                    message="Terminology consensus is not configured for this episode.",
                    data={"term": action.term},
                )
            )
        request = self._terminology_request(action)
        try:
            resolution = self.terminology_resolver.resolve(request)
        except TerminologyResolutionError as exc:
            calls = self._valid_provider_calls(exc.call_records)
            return self._result(
                self._observation(
                    ok=False,
                    kind="terminology_resolution_failed",
                    message="Terminology resolver failed; no glossary change was applied.",
                    data={"term": action.term, "error_type": type(exc).__name__},
                ),
                auxiliary_provider_calls=calls,
            )
        except Exception as exc:  # noqa: BLE001 - resolver boundary is fail-closed.
            return self._result(
                self._observation(
                    ok=False,
                    kind="terminology_resolution_failed",
                    message="Terminology resolver failed; no glossary change was applied.",
                    data={"term": action.term, "error_type": type(exc).__name__},
                )
            )

        calls = self._valid_provider_calls(resolution.provider_calls)
        summary = {
            "term": action.term,
            "selected_translation": resolution.selected_translation,
            "agreement": resolution.agreement,
            "evaluator_used": resolution.evaluator_used,
            "escalated": resolution.escalated,
            "vote_count": len(resolution.votes),
        }
        if resolution.escalated or not resolution.selected_translation:
            return self._result(
                self._observation(
                    ok=False,
                    kind="terminology_escalated",
                    message="Terminology resolution requires human review; no glossary change was applied.",
                    data=summary,
                ),
                resolution=resolution,
                auxiliary_provider_calls=calls,
            )

        selected = resolution.selected_translation
        matching_index = next(
            (
                index
                for index, entry in enumerate(self.episode_glossary.entries)
                if entry.source == action.term or entry.source.casefold() == action.term.casefold()
            ),
            None,
        )
        if matching_index is None:
            self.episode_glossary.entries.append(
                GlossaryEntry(source=action.term, target=selected, candidates=[selected])
            )
        else:
            entry = self.episode_glossary.entries[matching_index]
            candidates = list(entry.candidates)
            if selected not in candidates:
                candidates.insert(0, selected)
            self.episode_glossary.entries[matching_index] = entry.model_copy(
                update={"target": selected, "candidates": candidates}
            )
        self.glossary = self.episode_glossary
        before_qa = self.current_qa
        self.current_qa = self._run_qa(self.current_text)
        return self._result(
            self._observation(
                ok=True,
                kind="terminology_resolved",
                message="Terminology resolution applied as an episode-local glossary override.",
                data=summary,
            ),
            qa_before=before_qa,
            qa_after=self.current_qa,
            resolution=resolution,
            auxiliary_provider_calls=calls,
        )

    @staticmethod
    def _valid_provider_calls(values: object) -> list[ProviderCallRecord]:
        calls: list[ProviderCallRecord] = []
        if not isinstance(values, (list, tuple)):
            return calls
        for value in values:
            try:
                record = ProviderCallRecord.model_validate(value)
            except (TypeError, ValueError):
                continue
            if record not in calls:
                calls.append(record)
        return calls

    def _submit_patch(self, action: SubmitPatchAction) -> ToolExecutionResult:
        before_report = self.current_qa
        occurrence_count = self.current_text.count(action.old_text)
        if occurrence_count != 1:
            return self._result(
                self._observation(
                    ok=False,
                    kind="patch_rejected",
                    message=(
                        "old_text must occur exactly once in the current translation "
                        f"(found {occurrence_count})."
                    ),
                    data={"occurrences": occurrence_count},
                ),
                qa_before=before_report,
            )

        candidate_text = self.current_text.replace(action.old_text, action.new_text, 1)
        candidate_report = self._run_qa(candidate_text)
        before_keys = finding_identities(before_report)
        after_keys = finding_identities(candidate_report)
        new_keys = after_keys - before_keys
        improves = validate_patch_improves_qa(
            before_report=before_report,
            after_report=candidate_report,
        )
        accepted = improves and not new_keys

        evidence = {
            "before_score": before_report.score,
            "after_score": candidate_report.score,
            "before_findings": before_report.summary.total_findings,
            "after_findings": candidate_report.summary.total_findings,
            "new_finding_identities": [list(identity) for identity in sorted(new_keys, key=str)],
        }
        if not accepted:
            reason = "Patch did not strictly improve weighted QA."
            if new_keys:
                reason = "Patch introduced a new finding identity."
            observation = self._observation(
                ok=False,
                kind="patch_rejected",
                message=reason,
                data={**evidence, "accepted": False},
            )
            return self._result(
                observation,
                qa_before=before_report,
                qa_after=candidate_report,
            )

        self.current_text = candidate_text
        self.current_qa = candidate_report
        observation = self._observation(
            ok=True,
            kind="patch_accepted",
            message="Patch accepted after deterministic QA verification.",
            data={**evidence, "accepted": True},
        )
        return self._result(
            observation,
            qa_before=before_report,
            qa_after=candidate_report,
        )

    def _escalate(self, action: EscalateAction) -> ToolExecutionResult:
        self.escalated = True
        observation = self._observation(
            ok=True,
            kind="escalated",
            message="Repair episode escalated for human review.",
            data={"reason": action.reason},
        )
        return self._result(observation)

    def _finish(self, action: FinishAction) -> ToolExecutionResult:
        finding_count = self.current_qa.summary.total_findings
        if finding_count:
            observation = self._observation(
                ok=False,
                kind="finish_rejected",
                message=f"Cannot finish while {finding_count} QA finding(s) remain.",
                data={"count": finding_count, "summary": action.summary},
            )
            return self._result(observation, qa_before=self.current_qa, qa_after=self.current_qa)

        self.finished = True
        observation = self._observation(
            ok=True,
            kind="finished",
            message="Repair verified: deterministic QA has no findings.",
            data={"count": 0, "summary": action.summary},
        )
        return self._result(observation, qa_before=self.current_qa, qa_after=self.current_qa)


@dataclass
class AgentRepairResult:
    """Final observable result of one bounded repair episode."""

    episode: AgentEpisode
    final_text: str
    final_qa: QAReport


_AGENT_ACTION_ADAPTER = TypeAdapter(AgentAction)


def _bounded_json_value(value: Any, *, depth: int = 0) -> Any:
    """Keep persisted observations and actions finite and JSON-safe.

    The executor already bounds QA evidence.  Provider-selected strings and
    action observations can still be arbitrarily large, so the episode trace
    applies the same conservative limits before writing any model payload.
    Unsupported values are represented by their type name rather than calling
    an arbitrary ``__str__`` implementation.
    """

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in {float("inf"), float("-inf")} else None
    if isinstance(value, str):
        return RepairToolExecutor._bounded_snippet(value)
    if depth >= 4:
        return TRUNCATION_MARKER
    if isinstance(value, dict):
        items = sorted(
            ((key, item) for key, item in value.items() if isinstance(key, str)),
            key=lambda item: item[0],
        )[:32]
        return {key: _bounded_json_value(item, depth=depth + 1) for key, item in items}
    if isinstance(value, (list, tuple)):
        return [_bounded_json_value(item, depth=depth + 1) for item in list(value)[:32]]
    return f"<{type(value).__name__}>"


def _bounded_observation(observation: AgentObservation) -> AgentObservation:
    """Copy an observation without retaining unbounded provider/tool data."""

    return observation.model_copy(
        update={
            "message": RepairToolExecutor._bounded_snippet(observation.message) or "",
            "data": _bounded_json_value(observation.data),
        },
        deep=True,
    )


def _bounded_action_payload(action: AgentAction) -> dict[str, Any]:
    return _bounded_json_value(action.model_dump(mode="json"))


def _action_payload_is_bounded(action: AgentAction) -> bool:
    """Return whether the exact typed action survives episode bounding."""

    return action.model_dump(mode="json") == _bounded_action_payload(action)


def _bounded_typed_action(action: AgentAction) -> AgentAction:
    """Return the bounded typed action used in subsequent prior-step context."""

    return _AGENT_ACTION_ADAPTER.validate_python(_bounded_action_payload(action))


def _episode_identifier(run_id: str, story_slug: str, chapter: str) -> str:
    raw = f"{run_id}:{story_slug}:{chapter}"
    if len(raw) <= 200:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return raw[:167] + ":" + digest


_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = {
    errno.EINVAL,
    errno.ENOTSUP,
    errno.EOPNOTSUPP,
}


def _fsync_directory(directory: Path) -> None:
    """Flush a directory entry update when the platform supports it."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError as exc:
        if exc.errno in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
                return
            raise
    finally:
        os.close(descriptor)


def _persist_episode(episode_path: Path, episode: AgentEpisode) -> None:
    """Persist an episode through an adjacent temporary file and replace."""

    episode_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{episode_path.name}.",
            suffix=".tmp",
            dir=episode_path.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(episode.model_dump_json(indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, episode_path)
        _fsync_directory(episode_path.parent)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _provider_call_records(provider: AgentActionProvider) -> list[ProviderCallRecord]:
    records = getattr(provider, "call_records", [])
    if not isinstance(records, list):
        return []
    normalized: list[ProviderCallRecord] = []
    for record in records:
        try:
            normalized.append(ProviderCallRecord.model_validate(record))
        except (TypeError, ValueError):
            continue
    return normalized


def _latest_provider_call(provider: AgentActionProvider, before_count: int) -> ProviderCallRecord | None:
    records = _provider_call_records(provider)
    if len(records) <= before_count:
        return None
    return records[-1]


def _episode_findings(executor: RepairToolExecutor) -> list[dict[str, Any]]:
    bounded_report = RepairToolExecutor._bounded_report(executor.current_qa)
    assert bounded_report is not None
    return [finding.model_dump(mode="json") for finding in bounded_report.findings[:32]]


def _invalid_action_step(
    *,
    sequence: int,
    error: AgentActionValidationError,
    provider_call: ProviderCallRecord | None,
) -> AgentStep:
    observation = AgentObservation(
        ok=False,
        kind="invalid_action",
        message="Provider response was not a valid approved repair action.",
        data={"response": _bounded_json_value(error.parsed_response)},
    )
    return AgentStep(
        sequence=sequence,
        action={"tool": "invalid_action"},
        observation=observation,
        provider_call=provider_call,
    )


def run_repair_episode(
    *,
    provider: AgentActionProvider,
    episode_path: Path,
    source_text: str,
    translated_text: str,
    glossary: GlossaryParseResult,
    run_id: str,
    story_slug: str,
    chapter: str,
    provider_mode: str,
    max_steps: int = 5,
    max_patch_attempts: int = 2,
    terminology_resolver: TerminologyResolver | None = None,
    terminology_source_context_chars: int = 800,
    terminology_translation_context_chars: int = 800,
    tool_schema_version: AgentToolSchemaVersion = BASE_TOOL_SCHEMA_VERSION,
) -> AgentRepairResult:
    """Run one bounded, persistent model-directed repair episode.

    The provider selects exactly one typed action per step.  All mutations are
    delegated to :class:`RepairToolExecutor`; this function only manages
    budgets, observable chronology, provider evidence, and durable state.
    """

    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if max_patch_attempts < 1:
        raise ValueError("max_patch_attempts must be at least 1")

    episode_path = Path(episode_path)
    executor = RepairToolExecutor(
        source_text=source_text,
        translated_text=translated_text,
        glossary=glossary,
        run_id=run_id,
        story_slug=story_slug,
        chapter=chapter,
        terminology_resolver=terminology_resolver,
        terminology_source_context_chars=terminology_source_context_chars,
        terminology_translation_context_chars=terminology_translation_context_chars,
    )
    initial_qa = RepairToolExecutor._bounded_report(executor.current_qa)
    assert initial_qa is not None
    episode = AgentEpisode(
        episode_id=_episode_identifier(run_id, story_slug, chapter),
        run_id=run_id,
        story_slug=story_slug,
        chapter=chapter,
        provider_mode=provider_mode,
        provider=str(getattr(provider, "provider_name", "")),
        model=str(getattr(provider, "model_name", "")),
        max_steps=max_steps,
        max_patch_attempts=max_patch_attempts,
        initial_qa=initial_qa,
    )
    # Persist the initialized episode before any provider call so an outage or
    # process interruption still leaves a valid, inspectable artifact.
    _persist_episode(episode_path, episode)

    prior_steps: list[PriorObservableStep] = []
    patch_attempts = 0

    while len(episode.steps) < max_steps:
        request = AgentActionRequest(
            episode_id=episode.episode_id,
            step_number=len(episode.steps) + 1,
            story_slug=story_slug,
            chapter=chapter,
            current_findings=_episode_findings(executor),
            remaining_steps=max_steps - len(episode.steps),
            remaining_patch_attempts=max(max_patch_attempts - patch_attempts, 0),
            prior_steps=list(prior_steps[-24:]),
            tool_schema_version=tool_schema_version,
        )
        prior_call_count = len(_provider_call_records(provider))
        try:
            provided_action = provider.next_action(request)
            try:
                action = _AGENT_ACTION_ADAPTER.validate_python(provided_action)
                if tool_schema_version == BASE_TOOL_SCHEMA_VERSION and isinstance(
                    action, ResolveTerminologyAction
                ):
                    raise AgentActionValidationError(
                        "resolve_terminology is unavailable under agent-tools.v1.",
                        response=provided_action.model_dump(mode="json")
                        if isinstance(provided_action, ResolveTerminologyAction)
                        else provided_action,
                    )
            except ValidationError as exc:
                raise AgentActionValidationError(
                    "Agent action failed schema validation.",
                    response=provided_action,
                ) from exc
        except AgentActionValidationError as exc:
            step = _invalid_action_step(
                sequence=len(episode.steps) + 1,
                error=exc,
                provider_call=_latest_provider_call(provider, prior_call_count),
            )
            episode.steps.append(step)
            episode.final_qa = RepairToolExecutor._bounded_report(executor.current_qa)
            _persist_episode(episode_path, episode)
            continue
        except Exception as exc:
            episode.final_qa = RepairToolExecutor._bounded_report(executor.current_qa)
            episode.final_status = "failed"
            episode.summary = f"Provider failure ({type(exc).__name__})"
            _persist_episode(episode_path, episode)
            raise

        persisted_action = _bounded_action_payload(action)
        if isinstance(action, SubmitPatchAction):
            patch_attempts += 1
        if isinstance(action, SubmitPatchAction) and patch_attempts > max_patch_attempts:
            execution = ToolExecutionResult(
                observation=AgentObservation(
                    ok=False,
                    kind="patch_budget_exhausted",
                    message="Patch attempt budget is exhausted; no third patch was executed.",
                    data={"max_patch_attempts": max_patch_attempts},
                ),
                qa_before=RepairToolExecutor._bounded_report(executor.current_qa),
                qa_after=RepairToolExecutor._bounded_report(executor.current_qa),
            )
        elif not _action_payload_is_bounded(action):
            execution = ToolExecutionResult(
                observation=AgentObservation(
                    ok=False,
                    kind="action_too_large",
                    message="Action exceeds the bounded episode trace and was not executed.",
                    data={"tool": action.tool},
                ),
                qa_before=RepairToolExecutor._bounded_report(executor.current_qa),
                qa_after=RepairToolExecutor._bounded_report(executor.current_qa),
            )
        elif isinstance(action, SubmitPatchAction):
            execution = executor.execute(action)
        else:
            execution = executor.execute(action)

        bounded_observation = _bounded_observation(execution.observation)
        step = AgentStep(
            sequence=len(episode.steps) + 1,
            action=persisted_action,
            observation=bounded_observation,
            provider_call=_latest_provider_call(provider, prior_call_count),
            auxiliary_provider_calls=list(execution.auxiliary_provider_calls),
            qa_before=execution.qa_before,
            qa_after=execution.qa_after,
        )
        episode.steps.append(step)
        episode.final_qa = RepairToolExecutor._bounded_report(executor.current_qa)
        if execution.resolution is not None:
            episode.terminology_resolutions.append(execution.resolution)

        # Only valid typed actions are exposed to the next provider request;
        # invalid responses are recorded but never placed in prior_steps.
        prior_steps.append(
            PriorObservableStep(
                sequence=step.sequence,
                action=_bounded_typed_action(action),
                observation=bounded_observation,
            )
        )

        if execution.observation.kind == "finished" and execution.observation.ok:
            episode.final_status = "verified"
            episode.summary = bounded_observation.message
        elif execution.observation.kind == "escalated" and execution.observation.ok:
            episode.final_status = "escalated"
            episode.summary = bounded_observation.message
        elif execution.observation.kind == "patch_budget_exhausted":
            episode.final_status = "budget_exhausted"
            episode.summary = bounded_observation.message

        _persist_episode(episode_path, episode)
        if episode.final_status is not None:
            break

    if episode.final_status is None:
        episode.final_status = "budget_exhausted"
        episode.summary = "Repair episode action budget is exhausted."
        episode.final_qa = RepairToolExecutor._bounded_report(executor.current_qa)
        _persist_episode(episode_path, episode)

    final_qa = RepairToolExecutor._bounded_report(executor.current_qa)
    assert final_qa is not None
    return AgentRepairResult(episode=episode, final_text=executor.current_text, final_qa=final_qa)


__all__ = [
    "AgentRepairResult",
    "RepairToolExecutor",
    "ToolExecutionResult",
    "finding_identity",
    "finding_identities",
    "run_repair_episode",
]
