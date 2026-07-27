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
