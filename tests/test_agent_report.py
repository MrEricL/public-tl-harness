from __future__ import annotations

import json
from pathlib import Path
import re

from agentic_translation.agent_provider import LLMAgentActionProvider
from agentic_translation.agent_repair import run_repair_episode
from agentic_translation.agent_report import (
    render_agent_episode_html,
    render_agent_episode_markdown,
)
from agentic_translation.glossary import load_glossary
from agentic_translation.story import load_story_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORY = PROJECT_ROOT / "samples/agentic_repair_demo/story.yaml"


def _golden_episode(tmp_path: Path):
    config = load_story_config(STORY)
    source = (config.paths.source_dir / "0001.txt").read_text(encoding="utf-8")
    dirty = (config.paths.expected_dir / "dirty_translation.txt").read_text(encoding="utf-8")
    glossary = load_glossary(config.paths.glossary_path)
    provider = LLMAgentActionProvider(
        provider_mode="replay",
        provider_name="openai",
        model_name="fixture-agent-v1",
        cache_dir=STORY.parent / "replay_cache",
    )
    result = run_repair_episode(
        provider=provider,
        episode_path=tmp_path / "agent_episode.json",
        source_text=source,
        translated_text=dirty,
        glossary=glossary,
        run_id="agentic_repair_demo_replay",
        story_slug=config.slug,
        chapter="0001",
        provider_mode="replay",
        max_steps=5,
        max_patch_attempts=2,
    )
    return config, source, dirty, result, provider


def test_markdown_report_contains_chronological_agent_evidence(tmp_path: Path) -> None:
    config, source, dirty, result, provider = _golden_episode(tmp_path)
    report = render_agent_episode_markdown(
        result.episode,
        story_title=config.title,
        source_text=source,
        translation_text=dirty,
        final_text=result.final_text,
        artifact_paths={"final_text": "translated_final/0001.txt", "report": "report.html"},
        call_records=provider.call_records,
    )

    for text in [
        "Agent Repair Timeline",
        "lookup_glossary",
        "REJECTED",
        "read_source_context",
        "ACCEPTED",
        "QA findings: 3 → 0",
        "verified",
        "Replay cache: 5/5 hits",
    ]:
        assert text in report


def test_html_report_is_standalone_escaped_and_rendered(tmp_path: Path) -> None:
    config, source, dirty, result, provider = _golden_episode(tmp_path)
    html_path = tmp_path / "report.html"
    render_agent_episode_html(
        html_path,
        result.episode,
        story_title=config.title,
        source_text=source + "\n<script>alert('{{bad}}')</script>",
        translation_text=dirty,
        final_text=result.final_text,
        artifact_paths={"report": "report.html"},
        call_records=provider.call_records,
    )
    rendered = html_path.read_text(encoding="utf-8")
    assert "<!doctype html>" in rendered.lower()
    assert "Agent Repair Timeline" in rendered
    assert "&lt;script&gt;" in rendered
    assert "{{bad}}" not in rendered
    assert "{{" not in rendered


def test_html_report_uses_replay_provider_mode_copy(tmp_path: Path) -> None:
    config, source, dirty, result, provider = _golden_episode(tmp_path)
    html_path = tmp_path / "replay-mode.html"
    render_agent_episode_html(
        html_path,
        result.episode,
        story_title=config.title,
        source_text=source,
        translation_text=dirty,
        final_text=result.final_text,
        call_records=provider.call_records,
    )
    rendered = html_path.read_text(encoding="utf-8")

    assert "Synthetic contract fixture. No API key or network call is used." not in rendered
    assert "Replay mode; cache-only evidence. No live provider call is used." in rendered
    assert "Live provider mode; inspect persisted receipts for provider evidence." not in rendered
    assert "Deterministic synthetic fixture, not live provider output" not in rendered


def test_html_report_uses_live_provider_mode_copy(tmp_path: Path) -> None:
    config, source, dirty, result, _provider = _golden_episode(tmp_path)
    live_episode = result.episode.model_copy(
        update={
            "provider_mode": "live",
            "provider": "openai",
            "model": "gpt-live",
            "steps": [],
        }
    )
    html_path = tmp_path / "live-mode.html"
    render_agent_episode_html(
        html_path,
        live_episode,
        story_title=config.title,
        source_text=source,
        translation_text=dirty,
        final_text=result.final_text,
    )
    rendered = html_path.read_text(encoding="utf-8")

    assert "Synthetic contract fixture. No API key or network call is used." not in rendered
    assert "Replay mode; cache-only evidence. No live provider call is used." not in rendered
    assert "Live provider mode; inspect persisted receipts for provider evidence." in rendered
    assert "Live provider output; inspect persisted receipts and call metadata for evidence." in rendered


def test_default_html_template_is_owned_by_the_package(tmp_path: Path) -> None:
    package_template = PROJECT_ROOT / "agentic_translation" / "templates" / "agent_report.html.j2"
    assert package_template.exists()

    config, source, dirty, result, provider = _golden_episode(tmp_path)
    html_path = tmp_path / "default.html"
    render_agent_episode_html(
        html_path,
        result.episode,
        story_title=config.title,
        source_text=source,
        translation_text=dirty,
        final_text=result.final_text,
        call_records=provider.call_records,
    )
    rendered = html_path.read_text(encoding="utf-8")
    assert "Agent Repair Timeline" in rendered


def test_html_report_frames_the_run_as_a_translation_proof_record(tmp_path: Path) -> None:
    config, source, dirty, result, provider = _golden_episode(tmp_path)
    used_tools = list(
        dict.fromkeys(str(step.action.get("tool", "unknown")) for step in result.episode.steps)
    )
    accepted_patch = next(
        step
        for step in result.episode.steps
        if step.action.get("tool") == "submit_patch" and step.observation.kind == "patch_accepted"
    )
    assert accepted_patch.qa_before is not None
    assert accepted_patch.qa_after is not None
    accepted_before = accepted_patch.qa_before.summary.total_findings
    accepted_after = accepted_patch.qa_after.summary.total_findings
    exposed_tools = [*used_tools, "unused_fixture_tool"]
    html_path = tmp_path / "proof-record.html"
    render_agent_episode_html(
        html_path,
        result.episode,
        story_title=config.title,
        source_text=source,
        translation_text=dirty,
        final_text=result.final_text,
        call_records=provider.call_records,
        session_snapshot={
            "status": "completed",
            "exposed_tool_names": exposed_tools,
            "identity": {
                "schema_version": "agent-session-identity.v1",
                "run_id": result.episode.run_id,
                "story_slug": result.episode.story_slug,
                "chapter": result.episode.chapter,
                "provider_mode": result.episode.provider_mode,
                "provider": result.episode.provider,
                "model": result.episode.model,
                "tool_protocol": "native_function",
                "tool_schema_version": "agent-tools.v3",
                "source_sha256": "1" * 64,
                "master_glossary_sha256": "2" * 64,
                "registry_sha256": "3" * 64,
            },
            "pending_decision": {
                "decision": "approved",
                "reviewer": "demo-reviewer",
                "note": "Approved in the Harness v3 review flow.",
            },
        },
        session_events=[
            {
                "sequence": 1,
                "event_type": "model_requested",
                "payload": {"tool_protocol": "native_function"},
            },
            {
                "sequence": 2,
                "event_type": "approval_decided",
                "payload": {"decision": "approved"},
            },
        ],
    )
    rendered = html_path.read_text(encoding="utf-8")

    for text in (
        "Translation repair record · completed",
        "Translation verified",
        "Verified galley",
        "Audit record",
        "Execution trace",
        "Evidence receipts",
        "Execution context",
        "Decision receipt",
        "Reviewed by demo-reviewer",
        f"1 patch proposal was rejected before mutation; 1 patch was accepted after deterministic QA, reducing active findings from {accepted_before} to {accepted_after}.",
        "Rejected proposal · not applied",
        "Applied mutation · QA verified",
        f"{len(used_tools)} of {len(exposed_tools)} exposed tools invoked",
        "Not invoked",
        "unused_fixture_tool",
        "Model fixture-agent-v1",
        "Resume identity matched",
        "prefers-contrast: more",
        "prefers-reduced-transparency: reduce",
    ):
        assert text in rendered
    for retired_pattern in (
        'class="state-ledger"',
        'class="run-index"',
        'class="provenance"',
        "A source-aware account of model decisions",
        "box-shadow:",
    ):
        assert retired_pattern not in rendered
    assert 'href="#galley"' in rendered
    assert 'class="fact qa-clean"' in rendered
    assert "Final QA" in rendered
    assert "3 at intake" in rendered
    assert 'class="fact approval-approved"' in rendered
    assert "111111111111…11111111" in rendered
    assert "333333333333…33333333" in rendered
    assert "1" * 64 not in rendered
    assert "3" * 64 not in rendered
    assert "recruiter" not in rendered.lower()
    assert "portfolio" not in rendered.lower()


def test_html_report_uses_balanced_editorial_typography_scale() -> None:
    template = (
        PROJECT_ROOT / "agentic_translation" / "templates" / "agent_report.html.j2"
    ).read_text(encoding="utf-8")

    for declaration in (
        "font-size: clamp(2.25rem, 3.8vw, 3rem)",
        "line-height: .98",
        "letter-spacing: -.03em",
        "font-weight: 600",
        "font: 600 .76rem/1.25 var(--display)",
        "letter-spacing: .05em",
        "font: 400 1.0625rem/1.5 var(--body)",
        "font: 400 1.1rem/1.5 var(--reading)",
        "font-size: clamp(1.15rem, 1.55vw, 1.34rem)",
        "padding-top: clamp(1.75rem, 3vw, 2.75rem)",
        "grid-template-columns: minmax(0, 1.45fr) minmax(22rem, .95fr)",
        ".synopsis-copy > .utility { display: block; margin-bottom: .5rem; }",
        ".audit-column-heading { min-height: 2.75rem",
        ".audit-trace .audit-column-heading { margin-left: -1.05rem; }",
        ".audit-trace::before { content: \"\"; position: absolute; top: .75rem; bottom: 0; left: 0; width: 2px; background: var(--proof-blue); }",
        "font: 600 .90rem/1.25 var(--display)",
        "letter-spacing: .045em",
        ".change-pair { display: grid; grid-template-columns: 4.8rem minmax(0, 1fr)",
        ".tool-breakdown { display: grid; grid-template-columns: 5.4rem minmax(0, 1fr)",
        ".fact.qa-clean { border-left-color: var(--verified); }",
        ".fact.qa-open { border-left-color: var(--rejected); }",
        ".fact.approval-approved { border-left-color: var(--verified); }",
    ):
        assert declaration in template


def test_html_report_does_not_claim_clean_qa_without_a_final_report(tmp_path: Path) -> None:
    config, source, dirty, result, provider = _golden_episode(tmp_path)
    incomplete_episode = result.episode.model_copy(update={"final_qa": None})
    html_path = tmp_path / "qa-not-run.html"

    render_agent_episode_html(
        html_path,
        incomplete_episode,
        story_title=config.title,
        source_text=source,
        translation_text=dirty,
        final_text=result.final_text,
        call_records=provider.call_records,
    )
    rendered = html_path.read_text(encoding="utf-8")

    assert "Translation not yet verified" in rendered
    assert 'class="fact qa-unknown"' in rendered
    assert "Final QA not recorded" in rendered
    assert "Clean QA." not in rendered


def test_html_report_marks_remaining_findings_as_open(tmp_path: Path) -> None:
    config, source, dirty, result, provider = _golden_episode(tmp_path)
    open_episode = result.episode.model_copy(update={"final_qa": result.episode.initial_qa})
    html_path = tmp_path / "qa-open.html"

    render_agent_episode_html(
        html_path,
        open_episode,
        story_title=config.title,
        source_text=source,
        translation_text=dirty,
        final_text=result.final_text,
        call_records=provider.call_records,
    )
    rendered = html_path.read_text(encoding="utf-8")

    assert "Translation requires review" in rendered
    assert 'class="fact qa-open"' in rendered
    assert f"{result.episode.initial_qa.summary.total_findings} open" in rendered


def test_reports_call_the_session_receipt_persisted(tmp_path: Path) -> None:
    config, source, dirty, result, provider = _golden_episode(tmp_path)
    session_snapshot = {"status": "completed"}
    session_events = [
        {
            "sequence": 1,
            "event_type": "session_started",
            "payload": {"run_id": result.episode.run_id},
        }
    ]

    markdown = render_agent_episode_markdown(
        result.episode,
        story_title=config.title,
        source_text=source,
        translation_text=dirty,
        final_text=result.final_text,
        call_records=provider.call_records,
        session_snapshot=session_snapshot,
        session_events=session_events,
    )
    assert "## Persisted session receipt" in markdown
    assert "- Persisted events: 1" in markdown
    assert "Durable" not in markdown

    html_path = tmp_path / "persisted-receipt.html"
    render_agent_episode_html(
        html_path,
        result.episode,
        story_title=config.title,
        source_text=source,
        translation_text=dirty,
        final_text=result.final_text,
        call_records=provider.call_records,
        session_snapshot=session_snapshot,
        session_events=session_events,
    )
    rendered = html_path.read_text(encoding="utf-8")
    assert "Persisted session receipt" in rendered
    assert "Show persisted receipt" in rendered
    assert "Durable session receipt" not in rendered


def test_markdown_report_contains_adversarial_context_inside_dynamic_fences(tmp_path: Path) -> None:
    config, _source, dirty, result, provider = _golden_episode(tmp_path)
    malicious = "```\n# injected heading\n[unsafe link](javascript:alert(1))\n```"
    report = render_agent_episode_markdown(
        result.episode,
        story_title="title\n# injected",
        source_text=malicious,
        translation_text=dirty,
        final_text=result.final_text,
        call_records=provider.call_records,
    )

    assert re.search(r"(?m)^`{4,}text$", report)
    assert "**Story:** title \\# injected" in report
    assert "\ntitle\n# injected" not in report
    assert "\n# injected heading\n[unsafe link]" in report


def test_reports_cap_qa_rows_and_show_omitted_count(tmp_path: Path) -> None:
    config, source, dirty, result, provider = _golden_episode(tmp_path)
    initial = result.episode.initial_qa
    findings = [initial.findings[index % len(initial.findings)] for index in range(60)]
    initial = initial.model_copy(
        update={"findings": findings, "summary": initial.summary.model_copy(update={"total_findings": 60})}
    )
    episode = result.episode.model_copy(update={"initial_qa": initial})
    markdown = render_agent_episode_markdown(
        episode,
        story_title=config.title,
        source_text=source,
        translation_text=dirty,
        final_text=result.final_text,
        call_records=provider.call_records,
    )
    html_path = tmp_path / "bounded.html"
    render_agent_episode_html(
        html_path,
        episode,
        story_title=config.title,
        source_text=source,
        translation_text=dirty,
        final_text=result.final_text,
        call_records=provider.call_records,
    )
    assert "10 additional finding(s) omitted" in markdown
    assert "10 additional finding(s) omitted" in html_path.read_text(encoding="utf-8")


def test_markdown_report_escapes_inline_images_links_and_metacharacters(tmp_path: Path) -> None:
    config, source, dirty, result, provider = _golden_episode(tmp_path)
    malicious = "![tracking](https://evil.invalid)\n# injected\n*emphasis* [link](javascript:alert(1))"
    step = result.episode.steps[0]
    step = step.model_copy(
        update={
            "observation": step.observation.model_copy(update={"message": malicious}),
            "action": {"tool": "lookup_glossary", "term": malicious},
        }
    )
    finding = result.episode.initial_qa.findings[0].model_copy(update={"message": malicious})
    initial = result.episode.initial_qa.model_copy(update={"findings": [finding]})
    episode = result.episode.model_copy(update={"steps": [step, *result.episode.steps[1:],], "initial_qa": initial})
    markdown = render_agent_episode_markdown(
        episode,
        story_title=config.title,
        source_text=source,
        translation_text=dirty,
        final_text=result.final_text,
        call_records=provider.call_records,
    )

    assert "![tracking](" not in markdown
    assert "[link](javascript:" not in markdown
    assert "\n# injected\n" not in markdown
    assert "lookup_glossary" in markdown
