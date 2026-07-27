from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .batch import parse_chapter_selection
from .glossary import load_glossary
from .models import TerminologyConsensusConfig
from .providers_llm import inspect_response_cache, is_openai_compatible_provider, openai_compatible_provider_names, required_live_provider_config
from .story import load_story_config


PreflightStatus = Literal["ok", "warn", "fail"]


class PreflightCheck(BaseModel):
    name: str
    status: PreflightStatus
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class PreflightReport(BaseModel):
    passed: bool
    story_yaml: str
    provider_mode: str
    chapters: list[str] = Field(default_factory=list)
    checks: list[PreflightCheck] = Field(default_factory=list)
    status_counts: dict[str, int] = Field(default_factory=dict)


def _check(
    checks: list[PreflightCheck],
    name: str,
    status: PreflightStatus,
    message: str,
    **details: Any,
) -> None:
    checks.append(PreflightCheck(name=name, status=status, message=message, details=details))


def _finish(
    *,
    story_yaml: Path,
    provider_mode: str,
    chapters: list[str],
    checks: list[PreflightCheck],
) -> PreflightReport:
    counts = {
        "ok": sum(1 for check in checks if check.status == "ok"),
        "warn": sum(1 for check in checks if check.status == "warn"),
        "fail": sum(1 for check in checks if check.status == "fail"),
    }
    return PreflightReport(
        passed=counts["fail"] == 0,
        story_yaml=str(story_yaml),
        provider_mode=provider_mode,
        chapters=chapters,
        checks=checks,
        status_counts=counts,
    )


def _selected_chapters(default_chapters: list[str], selection: str | list[str] | None) -> list[str]:
    if selection is None:
        return default_chapters
    if isinstance(selection, str):
        return parse_chapter_selection(selection)
    return selection


def _required_replay_namespaces(
    *,
    translation_provider_name: str,
    judge_provider_name: str,
    repair_provider_name: str,
) -> list[str]:
    namespaces: list[str] = []
    if translation_provider_name != "offline":
        namespaces.append("translation")
    if judge_provider_name != "offline":
        namespaces.append("judge")
    if repair_provider_name != "offline":
        namespaces.append("repair")
    return namespaces


def run_preflight(
    story_yaml: Path,
    *,
    chapters: str | list[str] | None = None,
    provider_mode: str = "offline",
    translation_provider_name: str = "offline",
    judge_provider_name: str = "offline",
    repair_provider_name: str = "offline",
    record_cache: bool = False,
    cache_dir: Path | None = None,
    model_name: str | None = None,
    tool_agent_enabled: bool = False,
    terminology_consensus: TerminologyConsensusConfig | None = None,
) -> PreflightReport:
    checks: list[PreflightCheck] = []
    try:
        story = load_story_config(story_yaml)
    except Exception as exc:  # noqa: BLE001 - preflight should report config failures.
        _check(checks, "story_config", "fail", f"Story config could not be loaded: {exc}")
        return _finish(story_yaml=story_yaml, provider_mode=provider_mode, chapters=[], checks=checks)
    _check(checks, "story_config", "ok", f"Loaded story '{story.slug}'.")
    effective_terminology = (
        story.agent.terminology_consensus
        if terminology_consensus is None
        else terminology_consensus
    )

    try:
        selected_chapters = _selected_chapters(story.chapter_ids, chapters)
    except ValueError as exc:
        _check(checks, "chapter_selection", "fail", str(exc))
        selected_chapters = []
    if selected_chapters:
        _check(checks, "chapter_selection", "ok", f"Selected {len(selected_chapters)} chapter(s).")
    else:
        _check(checks, "chapter_selection", "fail", "No chapters selected.")

    if story.paths.source_dir.exists() and story.paths.source_dir.is_dir():
        missing_sources = [
            chapter
            for chapter in selected_chapters
            if not (story.paths.source_dir / f"{chapter}.txt").exists()
        ]
        if missing_sources:
            _check(
                checks,
                "source_chapters",
                "fail",
                f"Missing source chapter file(s): {', '.join(missing_sources)}.",
                missing=missing_sources,
            )
        else:
            _check(checks, "source_chapters", "ok", f"Found {len(selected_chapters)} source chapter file(s).")
    else:
        _check(checks, "source_chapters", "fail", f"Source directory does not exist: {story.paths.source_dir}")

    if story.paths.glossary_path.exists():
        glossary = load_glossary(story.paths.glossary_path)
        if glossary.entries:
            _check(checks, "glossary", "ok", f"Loaded {len(glossary.entries)} glossary entries.")
        else:
            _check(checks, "glossary", "fail", "Glossary exists but contains no parsed entries.")
        for warning in glossary.warnings:
            _check(checks, "glossary_warning", "warn", warning)
    else:
        _check(checks, "glossary", "fail", f"Glossary file does not exist: {story.paths.glossary_path}")

    if story.paths.prompt_path:
        if story.paths.prompt_path.exists():
            _check(checks, "prompt", "ok", f"Prompt file exists: {story.paths.prompt_path}")
        else:
            _check(checks, "prompt", "fail", f"Prompt file does not exist: {story.paths.prompt_path}")

    if story.paths.baseline_dir:
        if not story.paths.baseline_dir.exists():
            _check(checks, "baseline_dir", "fail", f"Baseline directory does not exist: {story.paths.baseline_dir}")
        else:
            missing_baselines = [
                chapter
                for chapter in selected_chapters
                if not (story.paths.baseline_dir / f"{chapter}.txt").exists()
            ]
            if missing_baselines:
                _check(
                    checks,
                    "baseline_dir",
                    "warn",
                    f"Missing baseline chapter file(s): {', '.join(missing_baselines)}.",
                    missing=missing_baselines,
                )
            else:
                _check(checks, "baseline_dir", "ok", "Baseline files are present for selected chapters.")

    provider_names = {translation_provider_name, judge_provider_name, repair_provider_name}
    non_offline_provider_names = {name for name in provider_names if name != "offline"}
    live_provider_names = openai_compatible_provider_names(provider_names)
    if tool_agent_enabled:
        tool_agent_errors: list[str] = []
        if provider_mode not in {"live", "replay"}:
            tool_agent_errors.append("Tool-agent mode requires provider mode live or replay.")
        if repair_provider_name == "offline":
            tool_agent_errors.append("Tool-agent mode requires a non-offline repair provider.")
        if not model_name:
            tool_agent_errors.append("Tool-agent mode requires an explicit model_name.")
        if tool_agent_errors:
            for error in tool_agent_errors:
                _check(checks, "tool_agent", "fail", error)
        else:
            _check(
                checks,
                "tool_agent",
                "ok",
                "Tool-agent mode is configured with a live/replay repair provider and explicit model_name.",
            )
    if effective_terminology.enabled:
        term_errors: list[str] = []
        if not tool_agent_enabled:
            term_errors.append("Terminology consensus requires tool-agent mode.")
        if provider_mode not in {"live", "replay"}:
            term_errors.append("Terminology consensus requires provider mode live or replay.")
        if repair_provider_name == "offline":
            term_errors.append("Terminology consensus requires a non-offline repair provider.")
        if not effective_terminology.openai_model:
            term_errors.append("Terminology consensus requires an explicit OpenAI model.")
        if not effective_terminology.deepseek_model:
            term_errors.append("Terminology consensus requires an explicit DeepSeek model.")
        if cache_dir is None:
            term_errors.append("Terminology consensus requires --cache-dir.")
        if provider_mode == "live":
            if not record_cache:
                term_errors.append("Live terminology consensus requires --record-cache.")
            for provider_name, term_model in (
                ("openai", effective_terminology.openai_model),
                ("deepseek", effective_terminology.deepseek_model),
            ):
                if term_model:
                    term_errors.extend(required_live_provider_config(provider_name, model_name=term_model))
        if term_errors:
            for error in dict.fromkeys(term_errors):
                _check(checks, "terminology_consensus", "fail", error)
        else:
            _check(
                checks,
                "terminology_consensus",
                "ok",
                "Dual-model terminology consensus is configured for OpenAI and DeepSeek.",
            )
    if provider_mode not in {"offline", "replay", "live"}:
        _check(checks, "providers", "fail", f"Unsupported provider mode: {provider_mode}")
    elif provider_mode == "offline":
        _check(checks, "providers", "ok", "Offline provider mode requires no live providers.")
    else:
        if not non_offline_provider_names:
            _check(
                checks,
                "providers",
                "fail",
                f"{provider_mode} mode requires at least one non-offline provider.",
            )
        elif provider_mode == "live" and not live_provider_names:
            _check(checks, "providers", "fail", "Live mode requires at least one live provider such as openai or deepseek.")
        else:
            _check(checks, "providers", "ok", f"{provider_mode} provider selection is runnable.")

    if provider_mode == "live":
        if not record_cache or cache_dir is None:
            _check(
                checks,
                "cache",
                "fail",
                "Live batch/corpus runs require --record-cache and --cache-dir so runs can be replayed.",
            )
        else:
            _check(checks, "cache", "ok", f"Live responses will be recorded under {cache_dir}.")
    elif provider_mode == "replay":
        if cache_dir is None:
            _check(checks, "cache", "fail", "Replay mode requires --cache-dir.")
        elif not cache_dir.exists():
            _check(checks, "cache", "fail", f"Replay cache directory does not exist: {cache_dir}")
        elif not cache_dir.is_dir():
            _check(checks, "cache", "fail", f"Replay cache path is not a directory: {cache_dir}")
        else:
            cache_report = inspect_response_cache(cache_dir)
            if cache_report.total_entries == 0:
                _check(checks, "cache", "fail", f"Replay cache has no indexed entries: {cache_dir}")
            elif not cache_report.integrity_passed:
                _check(
                    checks,
                    "cache",
                    "fail",
                    f"Replay cache integrity failed with {cache_report.invalid_entries} invalid indexed entrie(s).",
                    issues=[issue.model_dump() for issue in cache_report.integrity_issues],
                )
            else:
                required_namespaces = _required_replay_namespaces(
                    translation_provider_name=translation_provider_name,
                    judge_provider_name=judge_provider_name,
                    repair_provider_name=repair_provider_name,
                )
                missing_namespaces = [
                    namespace
                    for namespace in required_namespaces
                    if cache_report.by_namespace.get(namespace, 0) == 0
                ]
                if missing_namespaces:
                    _check(
                        checks,
                        "cache",
                        "fail",
                        f"Replay cache is missing indexed namespace(s): {', '.join(missing_namespaces)}.",
                        missing=missing_namespaces,
                    )
                else:
                    namespaces = ", ".join(sorted(cache_report.by_namespace))
                    _check(
                        checks,
                        "cache",
                        "ok",
                        f"Replay cache has {cache_report.total_entries} indexed response(s): {namespaces}.",
                        namespaces=cache_report.by_namespace,
                    )
    else:
        _check(checks, "cache", "ok", "Offline mode does not require a response cache.")

    if provider_mode == "live" and live_provider_names:
        missing_env: list[str] = []
        for provider_name in live_provider_names:
            if is_openai_compatible_provider(provider_name):
                missing_env.extend(required_live_provider_config(provider_name, model_name=model_name))
        if effective_terminology.enabled:
            for provider_name, term_model in (
                ("openai", effective_terminology.openai_model),
                ("deepseek", effective_terminology.deepseek_model),
            ):
                if term_model:
                    missing_env.extend(
                        required_live_provider_config(provider_name, model_name=term_model)
                    )
        missing_env = sorted(set(missing_env))
        if missing_env:
            _check(
                checks,
                "env",
                "fail",
                f"Missing required live provider environment/config value(s): {', '.join(missing_env)}.",
                missing=missing_env,
            )
        else:
            _check(checks, "env", "ok", "Live provider credentials/config are present.")
    else:
        _check(checks, "env", "ok", "No live OpenAI-compatible environment variables required for this mode.")

    return _finish(story_yaml=story_yaml, provider_mode=provider_mode, chapters=selected_chapters, checks=checks)
