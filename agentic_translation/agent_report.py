"""Chronological Markdown and standalone HTML reports for repair episodes."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from jinja2 import Environment, FileSystemLoader

from .agent_models import AgentEpisode, AgentStep
from .models import ProviderCallRecord, QAReport


DEFAULT_CONTEXT_CHARS = 1200
MAX_REPORT_FINDINGS = 50
_TEMPLATE_NAME = "agent_report.html.j2"
_MARKDOWN_META = frozenset(r"\`*_{}[]()#+-.!|>")


def _bounded(text: str | None, limit: int = DEFAULT_CONTEXT_CHARS) -> str:
    """Return bounded report context without allowing a report to grow unbounded."""

    value = text or ""
    if len(value) <= limit:
        return value
    marker = "...[truncated]"
    return value[: max(limit - len(marker), 0)].rstrip() + marker


def _qa_count(report: QAReport | None) -> int:
    if report is None:
        return 0
    return report.summary.total_findings


def _finding_rows(report: QAReport | None) -> tuple[list[dict[str, Any]], int]:
    if report is None:
        return [], 0
    rows = [
        {
            "check_id": finding.check_id,
            "severity": finding.severity,
            "message": finding.message,
            "location": finding.location.model_dump(mode="json"),
            "found": finding.found,
            "expected": finding.expected,
        }
        for finding in report.findings[:MAX_REPORT_FINDINGS]
    ]
    return rows, max(len(report.findings) - len(rows), 0)


def _step_status(step: AgentStep) -> str:
    kind = step.observation.kind
    if kind in {"glossary_promotion_pending", "approval_pending"}:
        return "PENDING"
    if kind == "patch_rejected":
        return "REJECTED"
    if kind == "patch_accepted":
        return "ACCEPTED"
    if step.observation.ok:
        return "OK"
    return "REJECTED"


def _repair_synopsis(
    *,
    accepted_patch_count: int,
    accepted_patch_qa_before: int | None,
    accepted_patch_qa_after: int | None,
    rejected_patch_count: int,
    final_qa_available: bool,
    initial_count: int,
    final_count: int,
    summary: str,
) -> str:
    """Return a deterministic, evidence-based repair synopsis.

    Patch status is taken from the bounded step rows rather than model-written
    rationale.  A findings transition is only meaningful when the episode has
    a persisted final QA report; otherwise the episode's own summary is the
    only available synopsis.
    """

    patch_parts: list[str] = []
    if rejected_patch_count:
        rejected_noun = "patch proposal" if rejected_patch_count == 1 else "patch proposals"
        rejected_verb = "was" if rejected_patch_count == 1 else "were"
        patch_parts.append(
            f"{rejected_patch_count} {rejected_noun} {rejected_verb} rejected before mutation"
        )
    if accepted_patch_count:
        accepted_noun = "patch" if accepted_patch_count == 1 else "patches"
        accepted_verb = "was" if accepted_patch_count == 1 else "were"
        accepted_part = (
            f"{accepted_patch_count} {accepted_noun} {accepted_verb} accepted after deterministic QA"
        )
        if (
            final_qa_available
            and accepted_patch_count == 1
            and accepted_patch_qa_before is not None
            and accepted_patch_qa_after is not None
        ):
            accepted_part += (
                ", reducing active findings from "
                f"{accepted_patch_qa_before} to {accepted_patch_qa_after}"
            )
        patch_parts.append(accepted_part)

    if not patch_parts:
        return summary
    if final_qa_available and not (
        accepted_patch_count == 1
        and accepted_patch_qa_before is not None
        and accepted_patch_qa_after is not None
    ):
        finding_noun = "finding" if initial_count == 1 else "findings"
        patch_parts.append(
            f"the run began with {initial_count} {finding_noun} and ended with {final_count}"
        )
    return "; ".join(patch_parts) + "."


def _call_records(
    episode: AgentEpisode,
    call_records: Sequence[ProviderCallRecord] | None,
) -> list[ProviderCallRecord]:
    source_records = list(call_records or [])
    if call_records is None:
        source_records = []
    for step in episode.steps:
        if step.provider_call is not None:
            source_records.append(step.provider_call)
        source_records.extend(step.auxiliary_provider_calls)
    records: list[ProviderCallRecord] = []
    seen: set[tuple[str, str, str, str]] = set()
    for value in source_records:
        try:
            record = ProviderCallRecord.model_validate(value)
        except (TypeError, ValueError):
            continue
        key = (record.namespace, record.payload_sha256, record.response_sha256, record.cache_file)
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
    return records


def build_agent_report_context(
    episode: AgentEpisode,
    *,
    story_title: str | None = None,
    source_text: str | None = None,
    translation_text: str | None = None,
    translated_text: str | None = None,
    final_text: str | None = None,
    artifact_paths: Mapping[str, str | Path] | None = None,
    call_records: Sequence[ProviderCallRecord] | None = None,
    provenance_note: str | None = None,
    session_snapshot: Any | None = None,
    session_events: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Build the bounded data context shared by Markdown and HTML reports."""

    records = _call_records(episode, call_records)
    if translation_text is None:
        translation_text = translated_text
    cache_hits = sum(1 for record in records if record.cache_hit)
    initial_count = _qa_count(episode.initial_qa)
    final_count = _qa_count(episode.final_qa)
    initial_findings, initial_omitted = _finding_rows(episode.initial_qa)
    final_findings, final_omitted = _finding_rows(episode.final_qa)
    steps: list[dict[str, Any]] = []
    for step in episode.steps:
        action = dict(step.action)
        raw_edits = action.get("edits")
        edits: list[dict[str, str]] = []
        if isinstance(raw_edits, list):
            for raw_edit in raw_edits[:8]:
                if not isinstance(raw_edit, Mapping):
                    continue
                edits.append(
                    {
                        "old_text": _bounded(str(raw_edit.get("old_text", "")), 1000),
                        "new_text": _bounded(str(raw_edit.get("new_text", "")), 1000),
                    }
                )
        elif action.get("old_text") is not None or action.get("new_text") is not None:
            # Legacy v1/v2 episode traces may still contain a single edit.
            edits.append(
                {
                    "old_text": _bounded(str(action.get("old_text", "")), 1000),
                    "new_text": _bounded(str(action.get("new_text", "")), 1000),
                }
            )
        provider_call = step.provider_call
        qa_before = _qa_count(step.qa_before) if step.qa_before is not None else None
        qa_after = _qa_count(step.qa_after) if step.qa_after is not None else None
        if qa_before is not None and qa_after is not None:
            qa_note = f"QA before/after: {qa_before} → {qa_after}"
        elif qa_before is not None:
            qa_note = f"QA before: {qa_before}; candidate not evaluated"
        elif qa_after is not None:
            qa_note = f"QA after: {qa_after}"
        else:
            qa_note = ""
        steps.append(
            {
                "sequence": step.sequence,
                "tool": str(action.get("tool", "unknown")),
                "status": _step_status(step),
                "ok": step.observation.ok,
                "message": _bounded(step.observation.message),
                "data": step.observation.data,
                "action": action,
                "rationale": _bounded(str(action.get("rationale", ""))) if action.get("rationale") else "",
                "old_text": _bounded(str(action.get("old_text", ""))) if action.get("old_text") else "",
                "new_text": _bounded(str(action.get("new_text", ""))) if action.get("new_text") else "",
                "edits": edits,
                "qa_before": qa_before,
                "qa_after": qa_after,
                "qa_note": qa_note,
                "cache_hit": provider_call.cache_hit if provider_call is not None else None,
                "provider_call": provider_call.model_dump(mode="json") if provider_call else None,
                "auxiliary_calls": [
                    {
                        "namespace": call.namespace,
                        "provider": call.provider,
                        "model": call.model,
                        "cache_hit": call.cache_hit,
                    }
                    for call in step.auxiliary_provider_calls
                ],
            }
        )

    final_qa_available = episode.final_qa is not None
    accepted_patch_count = sum(
        step["tool"] == "submit_patch" and step["status"] == "ACCEPTED"
        for step in steps
    )
    rejected_patch_count = sum(
        step["tool"] == "submit_patch" and step["status"] == "REJECTED"
        for step in steps
    )
    accepted_patch_step = next(
        (
            step
            for step in steps
            if step["tool"] == "submit_patch" and step["status"] == "ACCEPTED"
        ),
        None,
    )
    repair_synopsis = _repair_synopsis(
        accepted_patch_count=accepted_patch_count,
        accepted_patch_qa_before=(
            accepted_patch_step["qa_before"] if accepted_patch_step is not None else None
        ),
        accepted_patch_qa_after=(
            accepted_patch_step["qa_after"] if accepted_patch_step is not None else None
        ),
        rejected_patch_count=rejected_patch_count,
        final_qa_available=final_qa_available,
        initial_count=initial_count,
        final_count=final_count,
        summary=episode.summary,
    )

    terminology_summaries: list[dict[str, Any]] = []
    for resolution in episode.terminology_resolutions:
        source_term = resolution.votes[0].source_term if resolution.votes else ""
        terminology_summaries.append(
            {
                "source_term": _bounded(source_term, 200),
                "selected_translation": _bounded(resolution.selected_translation, 300),
                "agreement": resolution.agreement,
                "evaluator_used": resolution.evaluator_used,
                "escalated": resolution.escalated,
                "vote_count": len(resolution.votes),
            }
        )

    paths = {str(key): str(value) for key, value in (artifact_paths or {}).items()}

    def _model_dump(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if hasattr(value, "model_dump"):
            try:
                dumped = value.model_dump(mode="json")
            except (TypeError, ValueError):
                dumped = value.model_dump()
            return dumped if isinstance(dumped, dict) else {}
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    snapshot_data = _model_dump(session_snapshot)
    event_rows: list[dict[str, Any]] = []
    for value in session_events or ():
        row = _model_dump(value)
        if row:
            event_rows.append(row)
    tool_protocol: str | None = None
    for event in event_rows:
        if event.get("event_type") != "model_requested":
            continue
        payload = event.get("payload")
        candidate = payload.get("tool_protocol") if isinstance(payload, Mapping) else None
        if isinstance(candidate, str) and candidate in {"json_prompt", "native_function"}:
            tool_protocol = candidate
            break
    pending_approval = snapshot_data.get("pending_approval")
    pending_proposal = snapshot_data.get("pending_proposal")
    pending_decision = snapshot_data.get("pending_decision")
    session_identity = snapshot_data.get("identity")
    approval_decided = any(
        event.get("event_type") == "approval_decided"
        for event in event_rows
    )
    glossary_effect: dict[str, Any] = {}
    for step in steps:
        if step["tool"] != "promote_glossary_term" or not isinstance(step["data"], Mapping):
            continue
        for key in ("term", "proposed_target", "before_sha256", "after_sha256", "proposal_id"):
            if key in step["data"]:
                glossary_effect[key] = step["data"][key]
    if isinstance(pending_proposal, Mapping):
        for key in ("term", "proposed_target", "before_sha256", "after_sha256", "proposal_id"):
            if key in pending_proposal:
                glossary_effect[key] = pending_proposal[key]
    glossary_effect_applied = any(
        event.get("event_type") == "glossary_promotion_applied"
        for event in event_rows
    )
    exposed_tool_names = list(snapshot_data.get("exposed_tool_names") or ())
    used_tool_names = list(dict.fromkeys(step["tool"] for step in steps))
    exposed_tool_name_set = set(exposed_tool_names)
    used_tool_name_set = set(used_tool_names)
    unused_tool_names = [
        name for name in exposed_tool_names if name not in used_tool_name_set
    ]
    used_exposed_tool_names = [
        name for name in used_tool_names if name in exposed_tool_name_set
    ]
    used_unrecorded_tool_names = [
        name for name in used_tool_names if name not in exposed_tool_name_set
    ]
    return {
        "story": {"title": story_title or episode.story_slug, "slug": episode.story_slug},
        "episode": episode,
        "run_id": episode.run_id,
        "chapter": episode.chapter,
        "provider_mode": episode.provider_mode,
        "provider": episode.provider,
        "model": episode.model,
        "status": snapshot_data.get("status") or episode.final_status or "in_progress",
        "summary": episode.summary,
        "initial_qa": episode.initial_qa,
        "final_qa": episode.final_qa,
        "initial_findings": initial_findings,
        "initial_omitted": initial_omitted,
        "final_findings": final_findings,
        "final_omitted": final_omitted,
        "initial_count": initial_count,
        "final_count": final_count,
        "final_qa_available": final_qa_available,
        "accepted_patch_count": accepted_patch_count,
        "rejected_patch_count": rejected_patch_count,
        "repair_synopsis": repair_synopsis,
        "source_context": _bounded(source_text),
        "translation_context": _bounded(translation_text),
        "final_context": _bounded(final_text),
        "steps": steps,
        "terminology_resolutions": terminology_summaries,
        "call_records": [record.model_dump(mode="json") for record in records],
        "cache_total": len(records),
        "cache_hits": cache_hits,
        "artifact_paths": paths,
        "provenance_note": provenance_note,
        "session_snapshot": snapshot_data,
        "session_events": event_rows,
        "session_status": snapshot_data.get("status") or episode.final_status or "in_progress",
        "pending_approval": pending_approval if isinstance(pending_approval, Mapping) else None,
        "pending_proposal": pending_proposal if isinstance(pending_proposal, Mapping) else None,
        "pending_decision": pending_decision if isinstance(pending_decision, Mapping) else None,
        "session_identity": session_identity if isinstance(session_identity, Mapping) else None,
        "resume_identity_matched": bool(
            isinstance(session_identity, Mapping)
            and isinstance(pending_decision, Mapping)
            and approval_decided
        ),
        "event_count": len(event_rows),
        "tool_protocol": tool_protocol,
        "exposed_tool_names": exposed_tool_names,
        "used_tool_names": used_tool_names,
        "unused_tool_names": unused_tool_names,
        "used_exposed_tool_names": used_exposed_tool_names,
        "used_unrecorded_tool_names": used_unrecorded_tool_names,
        "glossary_effect": glossary_effect,
        "glossary_effect_applied": glossary_effect_applied,
    }


def _markdown_inline(value: Any) -> str:
    """Escape and flatten model-controlled values used inline in Markdown."""

    flattened = str(value).replace("\r", " ").replace("\n", " ")
    escaped = html.escape(flattened, quote=False)
    return "".join(f"\\{character}" if character in _MARKDOWN_META else character for character in escaped)


def _markdown_trusted_inline(value: Any) -> str:
    """Flatten values constrained by typed/deterministic schemas for display."""

    return html.escape(str(value).replace("\r", " ").replace("\n", " "), quote=False)


def _markdown_inline_code(value: Any) -> str:
    """Wrap untrusted inline text with a fence longer than any backtick run."""

    escaped = _markdown_inline(value)
    run = 0
    exact_max = 0
    for character in escaped:
        if character == "`":
            run += 1
            exact_max = max(exact_max, run)
        else:
            run = 0
    fence = "`" * max(1, exact_max + 1)
    return f"{fence}{escaped}{fence}"


def _markdown_code_block(value: Any) -> str:
    """Render bounded multiline text with a fence longer than its backticks."""

    escaped = html.escape(str(value), quote=False)
    max_run = 0
    run = 0
    for character in escaped:
        if character == "`":
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    fence = "`" * max(3, max_run + 1)
    return f"{fence}text\n{escaped}\n{fence}"


def render_agent_episode_markdown(
    episode: AgentEpisode,
    *,
    story_title: str | None = None,
    source_text: str | None = None,
    translation_text: str | None = None,
    translated_text: str | None = None,
    final_text: str | None = None,
    artifact_paths: Mapping[str, str | Path] | None = None,
    call_records: Sequence[ProviderCallRecord] | None = None,
    provenance_note: str | None = None,
    session_snapshot: Any | None = None,
    session_events: Sequence[Any] | None = None,
) -> str:
    """Render a compact, vertical chronology suitable for a run artifact."""

    context = build_agent_report_context(
        episode,
        story_title=story_title,
        source_text=source_text,
        translation_text=translation_text,
        translated_text=translated_text,
        final_text=final_text,
        artifact_paths=artifact_paths,
        call_records=call_records,
        provenance_note=provenance_note,
        session_snapshot=session_snapshot,
        session_events=session_events,
    )
    lines = [
        "# Agent Repair Timeline",
        "",
        f"**Story:** {_markdown_inline(context['story']['title'])}  ",
        f"**Chapter:** {_markdown_inline(context['chapter'])}  ",
        f"**Run:** {_markdown_inline_code(context['run_id'])}  ",
        f"**Provider:** {_markdown_inline(context['provider'])} / {_markdown_inline(context['model'])} ({_markdown_inline(context['provider_mode'])})  ",
        f"**Status:** **{_markdown_inline(context['status'])}**  ",
        f"QA findings: {context['initial_count']} → {context['final_count']}",
        f"Replay cache: {context['cache_hits']}/{context['cache_total']} hits",
    ]
    if context["provenance_note"]:
        lines.append(
            f"**Provenance:** {_markdown_trusted_inline(context['provenance_note'])}"
        )
    if context["session_snapshot"]:
        lines.extend(
            [
                "",
                "## Persisted session receipt",
                "",
                f"- Session: **{_markdown_inline(context['session_status'])}**",
                f"- Persisted events: {context['event_count']}",
                f"- Exposed tools: {_markdown_inline(', '.join(context['exposed_tool_names']))}",
                f"- Used tools: {_markdown_inline(', '.join(context['used_tool_names']))}",
            ]
        )
        if context["tool_protocol"]:
            protocol_label = (
                "native function"
                if context["tool_protocol"] == "native_function"
                else "prompt JSON"
            )
            lines.insert(
                len(lines) - 3,
                f"- Transport: **Harness v3 / {protocol_label}**",
            )
        if context["pending_approval"]:
            approval = context["pending_approval"]
            lines.append(
                f"- Approval receipt: pending for {_markdown_inline(approval.get('tool', 'persistent action'))} "
                f"({_markdown_inline_code(approval.get('proposal_id', 'unknown'))})"
            )
        if context["pending_decision"]:
            decision = context["pending_decision"]
            lines.append(
                f"- Reviewer decision: {_markdown_inline(decision.get('decision', 'unknown'))} by "
                f"{_markdown_inline(decision.get('reviewer', 'unknown'))} — "
                f"{_markdown_inline(decision.get('note', ''))}"
            )
        if context["glossary_effect"]:
            effect = context["glossary_effect"]
            glossary_label = (
                "Applied glossary delta"
                if context["glossary_effect_applied"]
                else "Proposed glossary delta"
            )
            lines.extend(
                [
                    f"- {glossary_label}: {_markdown_inline(effect.get('term', ''))} → "
                    f"{_markdown_inline(effect.get('proposed_target', ''))}",
                    f"- Glossary before/after: {_markdown_inline_code(effect.get('before_sha256', ''))} → "
                    f"{_markdown_inline_code(effect.get('after_sha256', ''))}",
                ]
            )
    lines.extend([
        "",
        "## Source and translation context",
        "",
        "### Source",
        "",
        _markdown_code_block(context["source_context"]),
        "",
        "### Initial translation",
        "",
        _markdown_code_block(context["translation_context"]),
    ])
    if context["final_context"]:
        lines.extend(["", "### Final translation", "", _markdown_code_block(context["final_context"])])

    lines.extend(["", "## Initial QA findings", ""])
    if context["initial_findings"]:
        for finding in context["initial_findings"]:
            lines.append(
                f"- **{_markdown_trusted_inline(finding['severity']).upper()}** {_markdown_trusted_inline(finding['check_id'])}: "
                f"{_markdown_inline(finding['message'])}"
            )
        if context["initial_omitted"]:
            lines.append(f"- {context['initial_omitted']} additional finding(s) omitted for bounded reporting.")
    else:
        lines.append("- None")

    lines.extend(["", "## Chronology", ""])
    for step in context["steps"]:
        label = f"{step['tool']} — {step['status']}"
        lines.extend([
            f"### {step['sequence']}. {_markdown_trusted_inline(label)}",
            "",
            f"- Observation: {_markdown_inline(step['message'])}",
        ])
        if step["rationale"]:
            lines.append(f"- Rationale: {_markdown_inline(step['rationale'])}")
        if step["edits"]:
            lines.append("- Edits:")
            for index, edit in enumerate(step["edits"], start=1):
                lines.append(
                    f"  {index}. {_markdown_inline_code(edit['old_text'])} → "
                    f"{_markdown_inline_code(edit['new_text'])}"
                )
        if step["qa_note"]:
            lines.append(f"- {step['qa_note']}")
        if step["cache_hit"] is not None:
            lines.append(f"- Cache hit: {'yes' if step['cache_hit'] else 'no'}")
        if step["auxiliary_calls"]:
            labels = ", ".join(
                f"{call['namespace']} ({call['provider']}/{call['model']})"
                for call in step["auxiliary_calls"]
            )
            lines.append(f"- Auxiliary calls: {_markdown_inline(labels)}")
        lines.append("")

    if context["session_events"]:
        lines.extend(["## Session event rail", ""])
        for event in context["session_events"]:
            lines.append(
                f"{event.get('sequence', '?')}. **{_markdown_inline(event.get('event_type', 'event'))}** — "
                f"{_markdown_inline(json.dumps(event.get('payload', {}), ensure_ascii=False, sort_keys=True))}"
            )
        lines.append("")

    if context["terminology_resolutions"]:
        lines.extend(["## Terminology resolutions", ""])
        for resolution in context["terminology_resolutions"]:
            selected = resolution["selected_translation"] or "(escalated)"
            lines.append(
                f"- {_markdown_inline(resolution['source_term'])} → {_markdown_inline(selected)} "
                f"(votes: {resolution['vote_count']}, agreement: "
                f"{'yes' if resolution['agreement'] else 'no'}, evaluator: "
                f"{'yes' if resolution['evaluator_used'] else 'no'}, escalated: "
                f"{'yes' if resolution['escalated'] else 'no'})"
            )

    lines.extend(["## Final QA", "", f"- Findings: {context['final_count']}"])
    if context["final_findings"]:
        for finding in context["final_findings"]:
            lines.append(f"- {_markdown_trusted_inline(finding['check_id'])}: {_markdown_inline(finding['message'])}")
        if context["final_omitted"]:
            lines.append(f"- {context['final_omitted']} additional finding(s) omitted for bounded reporting.")
    else:
        lines.append("- Verified clean.")

    lines.extend(["", "## Artifacts", ""])
    if context["artifact_paths"]:
        for name, path in context["artifact_paths"].items():
            lines.append(f"- **{_markdown_inline(name)}:** {_markdown_inline_code(path)}")
    else:
        lines.append("- None recorded.")
    lines.append("")
    return "\n".join(lines)


def render_agent_episode_html(
    output_path: Path,
    episode: AgentEpisode,
    *,
    story_title: str | None = None,
    source_text: str | None = None,
    translation_text: str | None = None,
    translated_text: str | None = None,
    final_text: str | None = None,
    artifact_paths: Mapping[str, str | Path] | None = None,
    call_records: Sequence[ProviderCallRecord] | None = None,
    template_dir: Path | None = None,
    provenance_note: str | None = None,
    session_snapshot: Any | None = None,
    session_events: Sequence[Any] | None = None,
) -> Path:
    """Render a standalone escaped HTML report using the agent template."""

    context = build_agent_report_context(
        episode,
        story_title=story_title,
        source_text=source_text,
        translation_text=translation_text,
        translated_text=translated_text,
        final_text=final_text,
        artifact_paths=artifact_paths,
        call_records=call_records,
        provenance_note=provenance_note,
        session_snapshot=session_snapshot,
        session_events=session_events,
    )
    selected_template_dir = template_dir or Path(__file__).resolve().parent / "templates"
    environment = Environment(
        loader=FileSystemLoader(str(selected_template_dir)),
        # ``.html.j2`` does not end in ``.html``; enable escaping explicitly
        # so untrusted translation/model text cannot become markup.
        autoescape=True,
    )
    rendered = environment.get_template(_TEMPLATE_NAME).render(**context)
    # Curly braces are not HTML-significant, but keeping a literal ``{{`` in
    # an artifact makes it look like an unrendered Jinja expression.  Encode
    # brace pairs after template expansion, including braces originating in
    # untrusted source/translation excerpts.
    rendered = rendered.replace("{{", "&#123;&#123;").replace("}}", "&#125;&#125;")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return output_path


__all__ = [
    "build_agent_report_context",
    "render_agent_episode_html",
    "render_agent_episode_markdown",
]
