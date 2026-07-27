"""Chronological Markdown and standalone HTML reports for repair episodes."""

from __future__ import annotations

import html
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
    if kind == "patch_rejected":
        return "REJECTED"
    if kind == "patch_accepted":
        return "ACCEPTED"
    if step.observation.ok:
        return "OK"
    return "REJECTED"


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
        provider_call = step.provider_call
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
                "qa_before": _qa_count(step.qa_before),
                "qa_after": _qa_count(step.qa_after),
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
    return {
        "story": {"title": story_title or episode.story_slug, "slug": episode.story_slug},
        "episode": episode,
        "run_id": episode.run_id,
        "chapter": episode.chapter,
        "provider_mode": episode.provider_mode,
        "provider": episode.provider,
        "model": episode.model,
        "status": episode.final_status or "in_progress",
        "summary": episode.summary,
        "initial_qa": episode.initial_qa,
        "final_qa": episode.final_qa,
        "initial_findings": initial_findings,
        "initial_omitted": initial_omitted,
        "final_findings": final_findings,
        "final_omitted": final_omitted,
        "initial_count": initial_count,
        "final_count": final_count,
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
        lines.append(f"**Provenance:** {context['provenance_note']}")
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
        if step["old_text"] or step["new_text"]:
            lines.append(
                f"- Patch: {_markdown_inline_code(step['old_text'])} → {_markdown_inline_code(step['new_text'])}"
            )
        if step["qa_before"] or step["qa_after"]:
            lines.append(f"- QA before/after: {step['qa_before']} → {step['qa_after']}")
        if step["cache_hit"] is not None:
            lines.append(f"- Cache hit: {'yes' if step['cache_hit'] else 'no'}")
        if step["auxiliary_calls"]:
            labels = ", ".join(
                f"{call['namespace']} ({call['provider']}/{call['model']})"
                for call in step["auxiliary_calls"]
            )
            lines.append(f"- Auxiliary calls: {_markdown_inline(labels)}")
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
