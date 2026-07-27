from __future__ import annotations

import contextlib
import io
import json
import re
import shlex
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console
from rich.table import Table

from .batch import (
    accept_reviewed_chapters,
    apply_manual_text_replacement,
    apply_glossary_update_plan,
    build_agent_work_order,
    build_batch_inspection_report,
    build_batch_proof_report,
    build_glossary_gap_report,
    build_glossary_update_plan,
    build_manual_edit_plan,
    build_panel_report,
    collect_review_queue,
    execute_agent_work_order,
    last_attempt_label,
    load_batch_manifest,
    normalize_panel_splits,
    parse_chapter_selection,
    preview_agent_work_order_execution,
    refresh_batch_pipeline,
    render_agent_work_order_markdown,
    render_glossary_gap_report_markdown,
    render_glossary_update_application_markdown,
    render_glossary_update_pass_markdown,
    render_glossary_update_plan_markdown,
    render_manual_edit_plan_markdown,
    render_panel_report_markdown,
    render_review_queue_markdown,
    replay_batch_pipeline,
    resume_batch_pipeline,
    run_batch_pipeline,
    run_glossary_update_pass,
    run_live_proof_pipeline,
    write_agent_work_order_execution_preview,
    write_batch_proof_artifacts,
    write_batch_triage_artifacts,
)
from .models import TerminologyConsensusConfig
from .env_config import load_cli_env
from .agent_provider import LLMAgentActionProvider
from .agent_provider import TERMINOLOGY_TOOL_SCHEMA_VERSION
from .agent_repair import run_repair_episode
from .agent_report import render_agent_episode_html, render_agent_episode_markdown
from .glossary import load_glossary
from .preflight import PreflightReport, run_preflight
from .providers_llm import LLMProviderUnavailable
from .providers_llm import inspect_response_cache, is_openai_compatible_provider, probe_live_provider
from .pipeline import import_local_fixture, run_demo_pipeline
from .qa import run_translation_qa
from .story import load_story_config, prepare_run_dir
from .tool_agent_pipeline import build_terminology_resolver


app = typer.Typer(help="Agentic long-form translation production system prototype.")
demo_app = typer.Typer(help="Demo commands.")
batch_app = typer.Typer(help="Batch corpus-production commands.")
cache_app = typer.Typer(help="Live/replay cache commands.")
app.add_typer(demo_app, name="demo")
app.add_typer(batch_app, name="batch")
app.add_typer(cache_app, name="cache")
console = Console()

TERMINAL_BATCH_STATUSES = {"packaged", "review_required", "failed", "skipped"}
DEFAULT_PANEL_REVIEWER = "human"
DEFAULT_PANEL_NOTE_PREFIX = "Merged split numbered note panels."
PRACTICAL_REVIEWER = "codex"
PRACTICAL_PANEL_NOTE_PREFIX = "Merged split corpus panel."
DEFAULT_AGENT_REPLAY_RUN_ID = "agentic_repair_demo_replay"


def _apply_term_consensus_repair_defaults(
    *,
    enabled: bool,
    repair_provider: str,
    model_name: str | None,
    openai_term_model: str | None,
    deepseek_term_model: str | None,
) -> tuple[str, str | None]:
    """Choose a terminology-compatible repair model without changing old defaults."""

    if not enabled:
        return repair_provider, model_name
    if repair_provider == "offline":
        repair_provider = "openai"
    if model_name is not None:
        return repair_provider, model_name
    compatible_model = {
        "openai": openai_term_model,
        "deepseek": deepseek_term_model,
    }.get(repair_provider)
    if compatible_model:
        return repair_provider, compatible_model
    raise ValueError(
        "--model is required when --term-consensus uses repair provider "
        f"{repair_provider!r}; no compatible terminology model is configured."
    )


@app.callback()
def main(
    env_file: Path | None = typer.Option(
        None,
        "--env-file",
        help="Load provider credentials/config from a dotenv-style file before running the command.",
    ),
) -> None:
    try:
        load_cli_env(env_file)
    except OSError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc


def _print_preflight_report(report: PreflightReport) -> None:
    table = Table(title=f"Preflight {Path(report.story_yaml).name}")
    table.add_column("Status")
    table.add_column("Check")
    table.add_column("Message")
    for check in report.checks:
        color = {"ok": "green", "warn": "yellow", "fail": "red"}.get(check.status, "white")
        table.add_row(f"[{color}]{check.status.upper()}[/{color}]", check.name, check.message)
    console.print(table)
    console.print(
        f"Summary: {report.status_counts.get('ok', 0)} ok, "
        f"{report.status_counts.get('warn', 0)} warn, {report.status_counts.get('fail', 0)} fail"
    )


@app.command("doctor")
def doctor(
    story_yaml: Path,
    chapters: str | None = typer.Option(None, "--chapters", help="Chapter range/list such as 0001-0010 or 1,3,5."),
    provider_mode: Literal["offline", "replay", "live"] = typer.Option("offline", "--provider-mode", help="Provider execution mode to preflight."),
    translation_provider: str = typer.Option("offline", help="Translation provider name."),
    judge_provider: str = typer.Option("offline", help="Judge provider name."),
    repair_provider: str = typer.Option("offline", help="Repair provider name."),
    record_cache: bool = typer.Option(False, help="Whether the intended live run will record provider responses."),
    cache_dir: Path | None = typer.Option(None, help="Response cache directory for live/replay providers."),
    model_name: str | None = typer.Option(None, "--model", help="Explicit OpenAI-compatible model name. Alternative to AGENTIC_TRANSLATION_MODEL."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    report = run_preflight(
        story_yaml,
        chapters=parse_chapter_selection(chapters) if chapters else None,
        provider_mode=provider_mode,
        translation_provider_name=translation_provider,
        judge_provider_name=judge_provider,
        repair_provider_name=repair_provider,
        record_cache=record_cache,
        cache_dir=cache_dir,
        model_name=model_name,
    )
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        _print_preflight_report(report)
    if not report.passed:
        raise typer.Exit(1)


@app.command("provider-probe")
def provider_probe(
    provider_name: str = typer.Argument(..., help="OpenAI-compatible provider name, such as openai or deepseek."),
    provider_mode: Literal["live", "replay"] = typer.Option("live", "--provider-mode", help="Probe mode: live spends one tiny call; replay uses cache only."),
    cache_dir: Path = typer.Option(Path(".agentic_cache/provider_probe"), help="Response cache directory."),
    record_cache: bool = typer.Option(True, "--record-cache/--no-record-cache", help="Record the live probe response for replay."),
    model_name: str | None = typer.Option(None, "--model", help="Explicit model name. DeepSeek defaults to deepseek-chat for this probe."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    effective_model = model_name or ("deepseek-chat" if provider_name == "deepseek" else None)
    try:
        result = probe_live_provider(
            provider_name=provider_name,
            provider_mode=provider_mode,
            cache_dir=cache_dir,
            record_cache=record_cache,
            model_name=effective_model,
        )
    except (ValueError, LLMProviderUnavailable) as exc:
        console.print(f"[red]Provider probe failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return
    table = Table(title="Provider Probe")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Provider", result.provider)
    table.add_row("Mode", result.mode)
    table.add_row("Model", result.model)
    table.add_row("Cache Dir", result.cache_dir)
    table.add_row("Cache Hit", str(result.cache_hit))
    table.add_row("Cache File", result.cache_file)
    table.add_row("Response", json.dumps(result.response, ensure_ascii=False))
    console.print(table)


@cache_app.command("inspect")
def cache_inspect(
    cache_dir: Path,
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    report = inspect_response_cache(cache_dir)
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
        return
    table = Table(title=f"Cache {cache_dir}")
    table.add_column("Namespace")
    table.add_column("Model")
    table.add_column("Cache File")
    for entry in report.entries:
        table.add_row(entry.namespace, entry.model or "", entry.cache_file)
    console.print(table)
    console.print(f"Summary: {report.total_entries} cached response(s)")
    integrity = "passed" if report.integrity_passed else "failed"
    color = "green" if report.integrity_passed else "red"
    console.print(
        f"Integrity: [{color}]{integrity}[/{color}] "
        f"({report.valid_entries} valid, {report.invalid_entries} invalid indexed entrie(s))"
    )
    if report.integrity_issues:
        issue_table = Table(title="Cache Integrity Issues")
        issue_table.add_column("Namespace")
        issue_table.add_column("Issue")
        issue_table.add_column("Cache File")
        issue_table.add_column("Message")
        for issue in report.integrity_issues:
            issue_table.add_row(issue.namespace or "", issue.issue_type, issue.cache_file, issue.message)
        console.print(issue_table)
        for issue in report.integrity_issues:
            console.print(f"Integrity issue: {issue.issue_type} - {issue.cache_file}")


_AGENT_CHAPTER_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_agent_chapter(chapter: str) -> str:
    if not chapter or not _AGENT_CHAPTER_RE.fullmatch(chapter):
        raise ValueError(
            "chapter must be one canonical filename component containing only letters, digits, '_' or '-'."
        )
    return chapter


def _require_path_under(path: Path, root: Path, *, label: str) -> Path:
    resolved_root = root.expanduser().resolve()
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"Resolved {label} escapes its selected root: {resolved_path}")
    return resolved_path


def _safe_agent_report_label(path: Path, *, story_root: Path) -> str:
    """Return a report-safe story-relative path without exposing workspace roots."""

    try:
        return path.relative_to(story_root).as_posix()
    except ValueError:
        return path.name


@app.command("demo-repair")
def demo_repair(
    story: Path = typer.Option(..., "--story", help="Story YAML fixture."),
    chapter: str = typer.Option("0001", "--chapter", help="Chapter id to repair."),
    provider_mode: Literal["replay", "live"] = typer.Option(
        "replay", "--provider-mode", help="Provider execution mode. Replay never calls a live provider."
    ),
    provider: str | None = typer.Option(None, "--provider", help="OpenAI-compatible action provider."),
    model: str | None = typer.Option(None, "--model", help="Action-provider model label."),
    cache_dir: Path | None = typer.Option(None, "--cache-dir", help="Replay/live response cache directory."),
    record_cache: bool = typer.Option(False, "--record-cache", help="Record live responses for later replay."),
    run_id: str | None = typer.Option(None, "--run-id", help="Run directory name."),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Output runs root; defaults to story config."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace exactly the selected existing run directory."),
    term_consensus: bool = typer.Option(
        False,
        "--term-consensus",
        help="Enable two-model OpenAI/DeepSeek terminology arbitration for this standalone demo.",
    ),
    openai_term_model: str | None = typer.Option(
        None, "--openai-term-model", help="Explicit OpenAI terminology voter model."
    ),
    deepseek_term_model: str | None = typer.Option(
        None, "--deepseek-term-model", help="Explicit DeepSeek terminology voter model."
    ),
    term_evaluator: Literal["openai", "deepseek"] = typer.Option(
        "openai", "--term-evaluator", help="Provider used to arbitrate terminology disagreements."
    ),
    term_confidence: float = typer.Option(
        0.65,
        "--term-confidence",
        min=0.0,
        max=1.0,
        help="Minimum terminology confidence required for automatic acceptance.",
    ),
) -> None:
    """Run the bounded replay-first repair-agent demonstration."""

    try:
        story_path = Path(story).expanduser().resolve()
        config = load_story_config(story_path)
        chapter = _validate_agent_chapter(chapter)
        effective_provider = provider
        effective_model = model
        if provider_mode == "replay":
            effective_provider = effective_provider or "openai"
            effective_model = effective_model or (
                "fixture-agent-v2" if term_consensus else "fixture-agent-v1"
            )
        elif not effective_provider or not effective_model:
            raise ValueError(
                "Live agent repair requires explicit --provider and --model, "
                "as well as --cache-dir (cache_dir) and --record-cache (record_cache)."
            )
        if not is_openai_compatible_provider(effective_provider):
            raise ValueError(f"Unsupported action provider {effective_provider!r}.")

        if config.paths.expected_dir is None:
            raise ValueError("Story config must define paths.expected_dir for the dirty translation fixture.")
        source_root = config.paths.source_dir.resolve()
        expected_root = config.paths.expected_dir.resolve()
        source_path = _require_path_under(source_root / f"{chapter}.txt", source_root, label="source chapter")
        dirty_path = _require_path_under(expected_root / "dirty_translation.txt", expected_root, label="dirty translation")
        source_text = source_path.read_text(encoding="utf-8")
        translation_text = dirty_path.read_text(encoding="utf-8")
        glossary = load_glossary(config.paths.glossary_path)

        effective_cache_dir = cache_dir
        if provider_mode == "replay":
            effective_cache_dir = effective_cache_dir or (story_path.parent / "replay_cache")
            if not effective_cache_dir.exists() or not effective_cache_dir.is_dir():
                raise ValueError(f"Replay cache directory does not exist: {effective_cache_dir}")
        elif effective_cache_dir is None or not record_cache:
            raise ValueError(
                "Live agent repair requires explicit --cache-dir (cache_dir) and --record-cache (record_cache)."
            )

        default_replay_run_id = (
            f"{config.slug}_replay" if term_consensus else DEFAULT_AGENT_REPLAY_RUN_ID
        )
        effective_run_id = run_id or (
            default_replay_run_id if provider_mode == "replay" else f"{config.slug}_{chapter}_live"
        )
        # The committed bundled replay cache is keyed to one reviewed action
        # trajectory. A custom run id selects only the output directory for
        # that cache; explicitly supplied cache directories retain the prior
        # caller-controlled identity semantics for recording/replay.
        trajectory_run_id = (
            default_replay_run_id
            if provider_mode == "replay" and cache_dir is None
            else effective_run_id
        )
        selected_runs_dir = runs_dir or config.paths.runs_dir
        if selected_runs_dir is None:
            raise ValueError("Story config must define paths.runs_dir or pass --runs-dir.")
        run_dir = prepare_run_dir(selected_runs_dir, effective_run_id, overwrite=overwrite)
        episode_path = run_dir / "agent_episode.json"
        initial_qa_path = run_dir / "qa_initial.json"
        final_qa_path = run_dir / "qa_final.json"
        final_dir = run_dir / "translated_final"
        final_path = _require_path_under(final_dir / f"{chapter}.txt", run_dir, label="final translation")
        markdown_path = run_dir / "repair_report.md"
        html_path = run_dir / "report.html"

        initial_qa = run_translation_qa(
            run_id=trajectory_run_id,
            story_slug=config.slug,
            chapter=chapter,
            source_text=source_text,
            translated_text=translation_text,
            glossary=glossary,
        )
        initial_qa_path.write_text(initial_qa.model_dump_json(indent=2), encoding="utf-8")

        action_provider = LLMAgentActionProvider(
            provider_mode=provider_mode,
            provider_name=effective_provider,
            model_name=effective_model,
            cache_dir=effective_cache_dir,
            record_cache=record_cache,
        )
        terminology_config = (
            TerminologyConsensusConfig(
                enabled=True,
                openai_model=openai_term_model,
                deepseek_model=deepseek_term_model,
                evaluator_provider=term_evaluator,
                confidence_threshold=term_confidence,
            )
            if term_consensus
            else None
        )
        terminology_resolver = build_terminology_resolver(
            terminology_config,
            provider_mode=provider_mode,
            cache_dir=effective_cache_dir,
            record_cache=record_cache,
        )
        result = run_repair_episode(
            provider=action_provider,
            episode_path=episode_path,
            source_text=source_text,
            translated_text=translation_text,
            glossary=glossary,
            run_id=trajectory_run_id,
            story_slug=config.slug,
            chapter=chapter,
            provider_mode=provider_mode,
            max_steps=5,
            max_patch_attempts=2,
            terminology_resolver=terminology_resolver,
            terminology_source_context_chars=(
                terminology_config.source_context_chars if terminology_config else 800
            ),
            terminology_translation_context_chars=(
                terminology_config.translation_context_chars if terminology_config else 800
            ),
            tool_schema_version=(
                TERMINOLOGY_TOOL_SCHEMA_VERSION
                if term_consensus
                else "agent-tools.v1"
            ),
        )
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path.write_text(result.final_text, encoding="utf-8")
        final_qa_path.write_text(result.final_qa.model_dump_json(indent=2), encoding="utf-8")
        artifact_paths = {
            "source": _safe_agent_report_label(source_path, story_root=story_path.parent),
            "translation_initial": _safe_agent_report_label(dirty_path, story_root=story_path.parent),
            "translated_final": str(final_path.relative_to(run_dir)),
            "qa_initial": str(initial_qa_path.relative_to(run_dir)),
            "qa_final": str(final_qa_path.relative_to(run_dir)),
            "agent_episode": str(episode_path.relative_to(run_dir)),
            "repair_report": str(markdown_path.relative_to(run_dir)),
            "report_html": str(html_path.relative_to(run_dir)),
        }
        provenance_note = None
        bundled_replay_cache = (story_path.parent / "replay_cache").resolve()
        if (
            term_consensus
            and provider_mode == "replay"
            and config.slug == "agentic_terminology_demo"
            and effective_cache_dir is not None
            and effective_cache_dir.resolve() == bundled_replay_cache
        ):
            provenance_note = (
                "Synthetic replay fixture; not evidence of a funded live-provider run."
            )
        markdown_path.write_text(
            render_agent_episode_markdown(
                result.episode,
                story_title=config.title,
                source_text=source_text,
                translation_text=translation_text,
                final_text=result.final_text,
                artifact_paths=artifact_paths,
                call_records=action_provider.call_records,
                provenance_note=provenance_note,
            ),
            encoding="utf-8",
        )
        render_agent_episode_html(
            html_path,
            result.episode,
            story_title=config.title,
            source_text=source_text,
            translation_text=translation_text,
            final_text=result.final_text,
            artifact_paths=artifact_paths,
            call_records=action_provider.call_records,
            provenance_note=provenance_note,
        )

        console.print(f"Initial QA findings: {initial_qa.summary.total_findings}")
        for step in result.episode.steps:
            tool = str(step.action.get("tool", "unknown"))
            console.print(f"Step {step.sequence}: {tool}")
            if step.observation.kind == "patch_rejected":
                console.print(f"[red]PATCH REJECTED[/red]: {step.observation.message}")
            elif step.observation.kind == "patch_accepted":
                console.print(f"[green]PATCH ACCEPTED[/green]: {step.observation.message}")
            elif step.observation.kind in {"finished", "escalated"}:
                console.print(f"{step.observation.kind.upper()}: {step.observation.message}")
            for call in step.auxiliary_provider_calls:
                console.print(
                    f"  {call.namespace}: {call.provider}/{call.model} "
                    f"({'cache hit' if call.cache_hit else 'live'})"
                )
        console.print(f"Final QA findings: {result.final_qa.summary.total_findings}")
        if result.episode.terminology_resolutions:
            resolution = result.episode.terminology_resolutions[0]
            console.print(
                "Terminology consensus: "
                f"{resolution.votes[0].source_term} -> {resolution.selected_translation} "
                f"(evaluator_used={str(resolution.evaluator_used).lower()}, "
                f"escalated={str(resolution.escalated).lower()})"
            )
        console.print(f"Status: {result.episode.final_status}")
        console.print(f"Episode: {episode_path}")
        console.print(f"Report: {html_path}")
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, json.JSONDecodeError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

@demo_app.command("run")
def demo_run(
    story_yaml: Path,
    provider_mode: Literal["offline", "replay", "live"] = typer.Option("offline", "--provider-mode", help="Provider execution mode."),
    offline: bool | None = typer.Option(None, "--offline/--live", help="Compatibility alias for --provider-mode offline/live."),
    translation_provider: str = typer.Option("offline", help="Translation provider name."),
    judge_provider: str = typer.Option("offline", help="Judge provider name."),
    repair_provider: str = typer.Option("offline", help="Repair provider name."),
    record_cache: bool = typer.Option(False, help="Record live provider JSON responses for replay mode."),
    cache_dir: Path | None = typer.Option(None, help="Response cache directory for live/replay providers."),
    model_name: str | None = typer.Option(None, "--model", help="Explicit OpenAI-compatible model name. Alternative to provider model env vars."),
    run_id: str | None = typer.Option(None, help="Optional deterministic run id."),
    seed: int = typer.Option(7, help="Deterministic candidate shuffle seed."),
    overwrite: bool = typer.Option(False, help="Overwrite an existing run directory."),
    skip_epub: bool = typer.Option(False, help="Skip EPUB generation. Default demo requires EPUB."),
    allow_source_qa_fail: bool = typer.Option(False, help="Continue after source QA errors. Intended only for local debugging."),
    allow_live_provider_fallback: bool = typer.Option(False, help="If a live translation/judge/repair call fails, continue with offline fallback and record the fallback reason."),
    report_mode: str | None = typer.Option(None, help="Override report mode: full, excerpt, or redacted."),
) -> None:
    try:
        result = run_demo_pipeline(
            story_yaml,
            provider_mode=provider_mode,
            offline=offline,
            translation_provider_name=translation_provider,
            judge_provider_name=judge_provider,
            repair_provider_name=repair_provider,
            record_cache=record_cache,
            cache_dir=cache_dir,
            model_name=model_name,
            run_id=run_id,
            seed=seed,
            overwrite=overwrite,
            skip_epub=skip_epub,
            allow_source_qa_fail=allow_source_qa_fail,
            allow_live_provider_fallback=allow_live_provider_fallback,
            report_mode=report_mode,
        )
    except (ValueError, RuntimeError, LLMProviderUnavailable) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    trace_path = result.run_dir / "trace.jsonl"
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            status = str(record.get("status", "ok"))
            color = {"ok": "green", "warn": "yellow", "fail": "red", "skipped": "cyan"}.get(status, "white")
            detail = ""
            if "findings" in record:
                detail = f", findings {record['findings']}"
            if "score" in record:
                detail += f", score {record['score']}"
            if record.get("stage") == "repair":
                detail = f", patches {record.get('patches', 0)}, accepted {record.get('accepted', 0)}"
            if record.get("stage") == "artifact_qa":
                detail = f", failures {record.get('failures', 0)}"
            console.print(f"[{color}]{status.upper()}[/{color}] {record.get('stage')}{detail}")
    table = Table(title="Final Metrics")
    table.add_column("Metric")
    table.add_column("Baseline")
    table.add_column("Glossary")
    table.add_column("Final")
    table.add_row("Residual Chinese", str(result.baseline_metrics.residual_chinese), str(result.glossary_metrics.residual_chinese), str(result.final_metrics.residual_chinese))
    table.add_row("Glossary Violations", str(result.baseline_metrics.glossary_violations), str(result.glossary_metrics.glossary_violations), str(result.final_metrics.glossary_violations))
    table.add_row("Panel Mismatches", str(result.baseline_metrics.panel_mismatches), str(result.glossary_metrics.panel_mismatches), str(result.final_metrics.panel_mismatches))
    table.add_row("Score", str(result.baseline_metrics.score), str(result.glossary_metrics.score), str(result.final_metrics.score))
    console.print(table)
    console.print(f"Report: [bold]{result.report_path}[/bold]")


def _print_batch_summary(manifest_path: Path) -> None:
    manifest = load_batch_manifest(manifest_path)
    inspection = build_batch_inspection_report(manifest)
    table = Table(title=f"Batch {manifest.run_id}")
    table.add_column("Chapter")
    table.add_column("Status")
    table.add_column("Score")
    table.add_column("Findings")
    table.add_column("Attempts")
    table.add_column("Last Attempt")
    table.add_column("Repairs")
    table.add_column("Accepted")
    for chapter, chapter_run in manifest.chapters.items():
        accepted_repairs = sum(1 for attempt in chapter_run.patch_attempts if attempt.accepted)
        table.add_row(
            chapter,
            chapter_run.status,
            "" if chapter_run.final_score is None else str(chapter_run.final_score),
            "" if chapter_run.final_findings is None else str(chapter_run.final_findings),
            str(len(chapter_run.attempts)),
            last_attempt_label(chapter_run),
            str(len(chapter_run.patch_attempts)),
            str(accepted_repairs),
        )
    console.print(table)
    console.print(
        f"Summary: {manifest.summary.packaged}/{manifest.summary.total_chapters} packaged, "
        f"{manifest.summary.review_required} review_required, {manifest.summary.failed} failed, "
        f"{manifest.summary.incomplete} incomplete"
    )
    evidence_label = "supported" if inspection.agentic_evidence.agentic_claim_supported else "not supported"
    console.print(f"Agentic evidence: {evidence_label} - {inspection.agentic_evidence.reason}")
    if inspection.provider_failures:
        console.print("Provider failures:")
        for failure in inspection.provider_failures:
            provider_label = "/".join(piece for piece in [failure.provider, failure.model] if piece) or "unknown"
            fallback_label = "fallback used" if failure.fallback_used else "no fallback"
            console.print(f"- {failure.chapter} {failure.role}/{provider_label}: {fallback_label} - {failure.reason}")
    if manifest.artifacts:
        for name, rel_path in manifest.artifacts.items():
            console.print(f"{name}: [bold]{Path(manifest.run_dir) / rel_path}[/bold]")


def _write_triage_if_requested(run_dir: Path, *, write_triage: bool) -> None:
    if not write_triage:
        return
    artifacts = write_batch_triage_artifacts(run_dir)
    console.print(f"Wrote {len(artifacts)} triage artifact(s):")
    for name, rel_path in artifacts.items():
        console.print(f"{name}: [bold]{run_dir / rel_path}[/bold]")


def _batch_has_blockers(manifest_path: Path, *, allow_review_required: bool = False) -> bool:
    manifest = load_batch_manifest(manifest_path)
    if _incomplete_chapters(manifest):
        return True
    if manifest.summary.failed:
        return True
    if manifest.artifact_qa and not manifest.artifact_qa.passed:
        return True
    return bool(manifest.summary.review_required and not allow_review_required)


def _exit_if_agentic_required(manifest, *, quiet: bool = False) -> None:  # noqa: ANN001 - keep CLI helper import surface small.
    report = build_batch_inspection_report(manifest)
    if report.agentic_evidence.agentic_claim_supported:
        return
    if not quiet:
        console.print("[red]Agentic evidence required:[/red] " + report.agentic_evidence.reason)
    raise typer.Exit(1)


def _exit_if_replayable_required(manifest, *, quiet: bool = False) -> None:  # noqa: ANN001 - keep CLI helper import surface small.
    report = build_batch_inspection_report(manifest)
    if report.agentic_evidence.replay_cache_ready:
        return
    if not quiet:
        missing = ", ".join(report.agentic_evidence.cache_missing_namespaces) or "indexed model-provider cache entries"
        console.print("[red]Replayable cache required:[/red] missing " + missing)
    raise typer.Exit(1)


def _incomplete_chapters(manifest) -> list[tuple[str, str]]:  # noqa: ANN001 - keep import surface small for Typer CLI helpers.
    return [
        (chapter, chapter_run.status)
        for chapter, chapter_run in manifest.chapters.items()
        if chapter_run.status not in TERMINAL_BATCH_STATUSES
    ]


def _exit_if_batch_has_blockers(manifest_path: Path, *, allow_review_required: bool = False) -> None:
    manifest = load_batch_manifest(manifest_path)
    incomplete = _incomplete_chapters(manifest)
    if incomplete:
        preview = ", ".join(f"{chapter} ({status})" for chapter, status in incomplete[:5])
        suffix = "" if len(incomplete) <= 5 else f", and {len(incomplete) - 5} more"
        console.print(
            "[red]Error:[/red] "
            f"Batch has {len(incomplete)} incomplete chapter(s): {preview}{suffix}. "
            "Run batch resume or use --force when replacing partial chapter runs."
        )
        raise typer.Exit(1)
    if manifest.summary.failed:
        console.print(f"[red]Error:[/red] Batch finished with {manifest.summary.failed} failed chapter(s).")
        raise typer.Exit(1)
    if manifest.artifact_qa and not manifest.artifact_qa.passed:
        console.print("[red]Error:[/red] Batch artifact QA failed.")
        for failure in manifest.artifact_qa.failures:
            console.print(f"- {failure}")
        raise typer.Exit(1)
    if manifest.summary.review_required and not allow_review_required:
        console.print(
            "[yellow]Review required:[/yellow] "
            f"{manifest.summary.review_required} chapter(s) still have QA findings. "
            "Pass --allow-review-required only for triage/debug runs."
        )
        raise typer.Exit(1)


def _print_review_queue(run_dir: Path) -> None:
    queue = collect_review_queue(run_dir)
    table = Table(title=f"Review Queue {queue.run_id}")
    table.add_column("Chapter")
    table.add_column("Severity")
    table.add_column("Check")
    table.add_column("Expected")
    table.add_column("Message")
    for item in queue.items:
        color = {"error": "red", "warning": "yellow", "info": "cyan"}.get(item.severity, "white")
        table.add_row(
            item.chapter,
            f"[{color}]{item.severity}[/{color}]",
            item.check_id,
            item.expected or "",
            item.message,
        )
    console.print(table)
    console.print(
        f"Summary: {queue.summary.total_items} item(s), "
        f"{len(queue.summary.by_chapter)} chapter(s), {len(queue.summary.by_check)} check type(s)"
    )
    if queue.summary.chapter_selection:
        console.print(f"Chapter selector: [bold]{queue.summary.chapter_selection}[/bold]")


def _print_glossary_gap_report(run_dir: Path) -> None:
    report = build_glossary_gap_report(run_dir)
    table = Table(title=f"Glossary Gaps {report.run_id}")
    table.add_column("Found")
    table.add_column("Expected")
    table.add_column("Occurrences")
    table.add_column("Chapters")
    table.add_column("Suggested Action")
    for gap in report.gaps:
        table.add_row(
            gap.found or "",
            gap.expected or "",
            str(gap.count),
            gap.chapter_selection,
            gap.suggested_action,
        )
    console.print(table)
    console.print(
        f"Summary: {report.summary.term_count} term(s), "
        f"{report.summary.total_occurrences} occurrence(s), {len(report.summary.by_chapter)} chapter(s)"
    )
    if report.summary.chapter_selection:
        console.print(f"Chapter selector: [bold]{report.summary.chapter_selection}[/bold]")


def _print_agent_work_order(run_dir: Path) -> None:
    work_order = build_agent_work_order(run_dir)
    table = Table(title=f"Agent Work Order {work_order.run_id}")
    table.add_column("Chapter")
    table.add_column("Action")
    table.add_column("Check")
    table.add_column("Reason")
    for item in work_order.items:
        table.add_row(item.chapter, item.action, item.check_id, item.reason)
    console.print(table)
    console.print(
        f"Summary: {work_order.summary.total_items} item(s), "
        f"{len(work_order.summary.by_chapter)} chapter(s), {len(work_order.summary.by_action)} action type(s)"
    )
    if work_order.summary.live_retry_selection:
        console.print(f"Live retry selector: [bold]{work_order.summary.live_retry_selection}[/bold]")
    if work_order.summary.glossary_selection:
        console.print(f"Glossary selector: [bold]{work_order.summary.glossary_selection}[/bold]")
    if work_order.summary.manual_review_selection:
        console.print(f"Manual review selector: [bold]{work_order.summary.manual_review_selection}[/bold]")


def _print_work_order_execution_preview(preview) -> None:  # noqa: ANN001 - keep CLI helper decoupled from model import.
    table = Table(title=f"Work-Order Dry Run {preview.run_id}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Action", preview.action)
    table.add_row("Provider Mode", preview.provider_mode)
    table.add_row("Chapters", ",".join(preview.chapters))
    table.add_row("Would Mutate", str(preview.would_mutate).lower())
    table.add_row("Preflight", "passed" if preview.preflight_passed else "failed")
    table.add_row("Recommended Next Action", preview.recommended_next_action)
    console.print(table)
    console.print(f"Recommended command: [bold]{preview.recommended_command}[/bold]")
    if preview.preflight_blockers:
        console.print("[red]Preflight blockers:[/red]")
        for blocker in preview.preflight_blockers:
            console.print(f"- {blocker}")
    if preview.preflight_checks:
        check_table = Table(title="Preflight Checks")
        check_table.add_column("Status")
        check_table.add_column("Check")
        check_table.add_column("Message")
        for check in preview.preflight_checks:
            status = str(check.get("status", ""))
            color = {"ok": "green", "warn": "yellow", "fail": "red"}.get(status, "white")
            check_table.add_row(f"[{color}]{status.upper()}[/{color}]", str(check.get("name", "")), str(check.get("message", "")))
        console.print(check_table)


def _print_batch_proof(report) -> None:  # noqa: ANN001 - keep CLI helper import surface small.
    table = Table(title=f"Agentic Proof {report.run_id}")
    table.add_column("Gate")
    table.add_column("Status")
    for name, passed in report.gates.items():
        color = "green" if passed else "red"
        table.add_row(name, f"[{color}]{'pass' if passed else 'fail'}[/{color}]")
    console.print(table)
    console.print(f"Proof passed: [bold]{str(report.proof_passed).lower()}[/bold]")
    if report.blockers:
        console.print("Blockers:")
        for blocker in report.blockers:
            console.print(f"- {blocker}")


def _print_live_proof_result(result) -> None:  # noqa: ANN001 - keep CLI helper import surface small.
    table = Table(title="Live Proof")
    table.add_column("Stage")
    table.add_column("Run")
    table.add_column("Proof")
    table.add_column("Path")
    table.add_row(
        "live",
        result.live_result.manifest.run_id,
        str(result.live_proof.proof_passed).lower(),
        str(result.live_result.run_dir),
    )
    table.add_row(
        "replay",
        result.replay_result.manifest.run_id,
        str(result.replay_proof.proof_passed).lower(),
        str(result.replay_result.run_dir),
    )
    console.print(table)
    console.print(f"Live proof passed: [bold]{str(result.proof_passed).lower()}[/bold]")


@batch_app.command("run")
def batch_run(
    story_yaml: Path,
    chapters: str = typer.Option(..., "--chapters", help="Chapter selection such as 0001-0010 or 1,3,5."),
    provider_mode: Literal["offline", "replay", "live"] = typer.Option("offline", "--provider-mode", help="Provider execution mode."),
    translation_provider: str = typer.Option("offline", help="Translation provider name."),
    judge_provider: str = typer.Option("offline", help="Judge provider name."),
    repair_provider: str = typer.Option("offline", help="Repair provider name."),
    record_cache: bool = typer.Option(False, help="Record live provider JSON responses for replay mode."),
    cache_dir: Path | None = typer.Option(None, help="Response cache directory for live/replay providers."),
    model_name: str | None = typer.Option(None, "--model", help="Explicit OpenAI-compatible model name. Alternative to AGENTIC_TRANSLATION_MODEL."),
    run_id: str | None = typer.Option(None, help="Optional deterministic run id."),
    seed: int = typer.Option(7, help="Deterministic candidate shuffle seed."),
    overwrite: bool = typer.Option(False, help="Overwrite an existing batch run directory."),
    force: bool = typer.Option(False, help="Re-run completed chapter subruns."),
    skip_epub: bool = typer.Option(False, help="Skip EPUB generation."),
    allow_source_qa_fail: bool = typer.Option(False, help="Continue after source QA errors. Intended only for local debugging."),
    allow_live_provider_fallback: bool = typer.Option(False, help="If a live translation/judge/repair call fails, continue with offline fallback and record the fallback reason."),
    tool_agent: bool = typer.Option(False, "--tool-agent", help="Route remaining QA findings through the bounded repair agent."),
    term_consensus: bool = typer.Option(False, "--term-consensus", help="Enable two-model OpenAI/DeepSeek terminology consensus (also enables the tool agent)."),
    openai_term_model: str | None = typer.Option(None, "--openai-term-model", help="Explicit OpenAI terminology voter model."),
    deepseek_term_model: str | None = typer.Option(None, "--deepseek-term-model", help="Explicit DeepSeek terminology voter model."),
    term_evaluator: Literal["openai", "deepseek"] = typer.Option("openai", "--term-evaluator", help="Provider used to arbitrate terminology disagreements."),
    term_confidence: float = typer.Option(0.65, "--term-confidence", min=0.0, max=1.0, help="Minimum confidence required for automatic terminology acceptance."),
    allow_review_required: bool = typer.Option(False, help="Return exit code 0 when chapters need human review. Intended only for triage/debug runs."),
    report_mode: str | None = typer.Option("excerpt", help="Override chapter report mode: full, excerpt, or redacted."),
    write_proof: bool = typer.Option(False, "--write-proof", help="Write non-gating agentic_proof.json and agentic_proof.md after the run."),
    write_triage: bool = typer.Option(False, "--write-triage", help="Write review queue, glossary gap report, and agent work order after the run."),
) -> None:
    try:
        if term_consensus:
            repair_provider, model_name = _apply_term_consensus_repair_defaults(
                enabled=True,
                repair_provider=repair_provider,
                model_name=model_name,
                openai_term_model=openai_term_model,
                deepseek_term_model=deepseek_term_model,
            )
        terminology_consensus = (
            TerminologyConsensusConfig(
                enabled=True,
                openai_model=openai_term_model,
                deepseek_model=deepseek_term_model,
                evaluator_provider=term_evaluator,
                confidence_threshold=term_confidence,
            )
            if term_consensus
            else None
        )
        result = run_batch_pipeline(
            story_yaml,
            chapters=parse_chapter_selection(chapters),
            provider_mode=provider_mode,
            translation_provider_name=translation_provider,
            judge_provider_name=judge_provider,
            repair_provider_name=repair_provider,
            record_cache=record_cache,
            cache_dir=cache_dir,
            model_name=model_name,
            run_id=run_id,
            seed=seed,
            overwrite=overwrite,
            force=force,
            skip_epub=skip_epub,
            allow_source_qa_fail=allow_source_qa_fail,
            allow_live_provider_fallback=allow_live_provider_fallback,
            tool_agent_enabled=tool_agent or term_consensus,
            terminology_consensus=terminology_consensus,
            report_mode=report_mode,
            write_proof=write_proof,
        )
    except (ValueError, RuntimeError, LLMProviderUnavailable) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_batch_summary(result.manifest_path)
    _write_triage_if_requested(result.run_dir, write_triage=write_triage)
    _exit_if_batch_has_blockers(result.manifest_path, allow_review_required=allow_review_required)


@batch_app.command("inspect")
def batch_inspect(
    run_dir: Path,
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable batch manifest JSON."),
    status_json: bool = typer.Option(False, "--status-json", help="Emit derived delivery status and blocker JSON."),
    strict: bool = typer.Option(False, "--strict", help="Return exit code 1 when the batch has incomplete, failed, review-required, or artifact-QA blockers."),
    require_agentic: bool = typer.Option(False, "--require-agentic", help="Return exit code 1 unless model-backed judge/repair evidence was observed."),
    require_replayable: bool = typer.Option(False, "--require-replayable", help="Return exit code 1 unless required model-provider cache namespaces are indexed."),
) -> None:
    manifest_path = run_dir / "batch_manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]Error:[/red] No batch manifest found at {manifest_path}")
        raise typer.Exit(1)
    if json_output and status_json:
        console.print("[red]Error:[/red] Choose only one of --json or --status-json.")
        raise typer.Exit(1)
    manifest = load_batch_manifest(manifest_path)
    if status_json:
        report = build_batch_inspection_report(manifest)
        typer.echo(report.model_dump_json(indent=2))
        if strict and not report.ready_for_delivery:
            raise typer.Exit(1)
        if require_agentic and not report.agentic_evidence.agentic_claim_supported:
            raise typer.Exit(1)
        if require_replayable and not report.agentic_evidence.replay_cache_ready:
            raise typer.Exit(1)
        return
    if json_output:
        typer.echo(manifest.model_dump_json(indent=2))
        if strict and _batch_has_blockers(manifest_path):
            raise typer.Exit(1)
        if require_agentic:
            _exit_if_agentic_required(manifest, quiet=True)
        if require_replayable:
            _exit_if_replayable_required(manifest, quiet=True)
        return
    _print_batch_summary(manifest_path)
    if strict:
        _exit_if_batch_has_blockers(manifest_path)
    if require_agentic:
        _exit_if_agentic_required(manifest)
    if require_replayable:
        _exit_if_replayable_required(manifest)


@batch_app.command("prove")
def batch_prove(
    run_dir: Path,
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable proof JSON."),
    write: bool = typer.Option(False, "--write", help="Write agentic_proof.json and agentic_proof.md into the batch run directory."),
) -> None:
    manifest_path = run_dir / "batch_manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]Error:[/red] No batch manifest found at {manifest_path}")
        raise typer.Exit(1)
    manifest = load_batch_manifest(manifest_path)
    if write:
        report = write_batch_proof_artifacts(run_dir, manifest)
        if not json_output:
            console.print(f"Wrote proof JSON: [bold]{run_dir / 'agentic_proof.json'}[/bold]")
            console.print(f"Wrote proof Markdown: [bold]{run_dir / 'agentic_proof.md'}[/bold]")
    else:
        report = build_batch_proof_report(manifest)
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        _print_batch_proof(report)
    if not report.proof_passed:
        raise typer.Exit(1)


@batch_app.command("live-proof")
def batch_live_proof(
    story_yaml: Path,
    chapters: str = typer.Option(..., "--chapters", help="Chapter selection such as 0001 or 0001-0003."),
    translation_provider: str = typer.Option("offline", help="Translation provider name."),
    judge_provider: str = typer.Option("openai", help="Judge provider name."),
    repair_provider: str = typer.Option("offline", help="Repair provider name."),
    cache_dir: Path | None = typer.Option(Path(".agentic_cache"), help="Response cache directory for recording and replay."),
    model_name: str | None = typer.Option(None, "--model", help="Explicit OpenAI-compatible model name. Alternative to AGENTIC_TRANSLATION_MODEL."),
    run_id: str | None = typer.Option(None, help="Optional live run id."),
    replay_run_id: str | None = typer.Option(None, help="Optional replay run id. Defaults to <live_run_id>_replay."),
    seed: int = typer.Option(7, help="Deterministic candidate shuffle seed."),
    overwrite: bool = typer.Option(False, help="Overwrite existing live/replay run directories."),
    skip_epub: bool = typer.Option(False, help="Skip EPUB generation."),
    allow_source_qa_fail: bool = typer.Option(False, help="Continue after source QA errors. Intended only for local debugging."),
    tool_agent: bool = typer.Option(False, "--tool-agent", help="Require the bounded repair agent during live proof."),
    term_consensus: bool = typer.Option(False, "--term-consensus", help="Enable two-model OpenAI/DeepSeek terminology consensus (also enables the tool agent)."),
    openai_term_model: str | None = typer.Option(None, "--openai-term-model", help="Explicit OpenAI terminology voter model."),
    deepseek_term_model: str | None = typer.Option(None, "--deepseek-term-model", help="Explicit DeepSeek terminology voter model."),
    term_evaluator: Literal["openai", "deepseek"] = typer.Option("openai", "--term-evaluator", help="Provider used to arbitrate terminology disagreements."),
    term_confidence: float = typer.Option(0.65, "--term-confidence", min=0.0, max=1.0, help="Minimum confidence required for automatic terminology acceptance."),
    report_mode: str | None = typer.Option("excerpt", help="Override chapter report mode: full, excerpt, or redacted."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        if term_consensus:
            repair_provider, model_name = _apply_term_consensus_repair_defaults(
                enabled=True,
                repair_provider=repair_provider,
                model_name=model_name,
                openai_term_model=openai_term_model,
                deepseek_term_model=deepseek_term_model,
            )
        terminology_consensus = (
            TerminologyConsensusConfig(
                enabled=True,
                openai_model=openai_term_model,
                deepseek_model=deepseek_term_model,
                evaluator_provider=term_evaluator,
                confidence_threshold=term_confidence,
            )
            if term_consensus
            else None
        )
        result = run_live_proof_pipeline(
            story_yaml,
            chapters=parse_chapter_selection(chapters),
            translation_provider_name=translation_provider,
            judge_provider_name=judge_provider,
            repair_provider_name=repair_provider,
            cache_dir=cache_dir,
            model_name=model_name,
            run_id=run_id,
            replay_run_id=replay_run_id,
            seed=seed,
            overwrite=overwrite,
            skip_epub=skip_epub,
            allow_source_qa_fail=allow_source_qa_fail,
            tool_agent_enabled=tool_agent or term_consensus,
            terminology_consensus=terminology_consensus,
            report_mode=report_mode,
        )
    except (ValueError, RuntimeError, LLMProviderUnavailable) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return
    _print_live_proof_result(result)


@batch_app.command("review")
def batch_review(
    run_dir: Path,
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    markdown_output: bool = typer.Option(False, "--markdown", help="Emit a human-readable Markdown review packet."),
    chapters_only: bool = typer.Option(False, "--chapters-only", help="Print only the comma-separated chapter selector for targeted resume."),
    write: bool = typer.Option(False, "--write", help="Write review_queue.json into the batch run directory."),
    write_markdown: bool = typer.Option(False, "--write-markdown", help="Write review_queue.md into the batch run directory."),
) -> None:
    manifest_path = run_dir / "batch_manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]Error:[/red] No batch manifest found at {manifest_path}")
        raise typer.Exit(1)
    queue = collect_review_queue(run_dir)
    if write:
        output_path = run_dir / "review_queue.json"
        output_path.write_text(queue.model_dump_json(indent=2), encoding="utf-8")
        chapter_selection_path = run_dir / "review_chapters.txt"
        chapter_selection_path.write_text(queue.summary.chapter_selection + "\n", encoding="utf-8")
        if not json_output and not chapters_only:
            console.print(f"Wrote review queue: [bold]{output_path}[/bold]")
            console.print(f"Wrote chapter selector: [bold]{chapter_selection_path}[/bold]")
    if write_markdown:
        markdown_path = run_dir / "review_queue.md"
        markdown_path.write_text(render_review_queue_markdown(queue), encoding="utf-8")
        if not json_output and not chapters_only and not markdown_output:
            console.print(f"Wrote Markdown review queue: [bold]{markdown_path}[/bold]")
    if chapters_only:
        typer.echo(queue.summary.chapter_selection)
        return
    if json_output:
        typer.echo(queue.model_dump_json(indent=2))
        return
    if markdown_output:
        typer.echo(render_review_queue_markdown(queue), nl=False)
        return
    _print_review_queue(run_dir)


@batch_app.command("panel-report")
def batch_panel_report(
    run_dir: Path,
    chapters: str | None = typer.Option(None, "--chapters", help="Optional chapter subset, such as 0003,0007 or 1-3."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable panel report JSON."),
    markdown_output: bool = typer.Option(False, "--markdown", help="Emit a human-readable Markdown panel report."),
    write: bool = typer.Option(False, "--write", help="Write panel_report.json and panel_report.md into the batch run directory."),
) -> None:
    manifest_path = run_dir / "batch_manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]Error:[/red] No batch manifest found at {manifest_path}")
        raise typer.Exit(1)
    try:
        report = build_panel_report(run_dir, chapters=parse_chapter_selection(chapters) if chapters else None)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    if write:
        json_path = run_dir / "panel_report.json"
        markdown_path = run_dir / "panel_report.md"
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        markdown_path.write_text(render_panel_report_markdown(report), encoding="utf-8")
        if not json_output and not markdown_output:
            console.print(f"Wrote panel report: [bold]{json_path}[/bold]")
            console.print(f"Wrote panel Markdown: [bold]{markdown_path}[/bold]")
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
        return
    if markdown_output:
        typer.echo(render_panel_report_markdown(report), nl=False)
        return
    typer.echo(render_panel_report_markdown(report), nl=False)


@batch_app.command("glossary-report")
def batch_glossary_report(
    run_dir: Path,
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable glossary gap JSON."),
    markdown_output: bool = typer.Option(False, "--markdown", help="Emit a human-readable Markdown glossary gap report."),
    write: bool = typer.Option(False, "--write", help="Write glossary_gap_report.json and glossary_gap_report.md into the batch run directory."),
) -> None:
    manifest_path = run_dir / "batch_manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]Error:[/red] No batch manifest found at {manifest_path}")
        raise typer.Exit(1)
    report = build_glossary_gap_report(run_dir)
    if write:
        json_path = run_dir / "glossary_gap_report.json"
        markdown_path = run_dir / "glossary_gap_report.md"
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        markdown_path.write_text(render_glossary_gap_report_markdown(report), encoding="utf-8")
        if not json_output and not markdown_output:
            console.print(f"Wrote glossary gap report: [bold]{json_path}[/bold]")
            console.print(f"Wrote glossary gap Markdown: [bold]{markdown_path}[/bold]")
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
        return
    if markdown_output:
        typer.echo(render_glossary_gap_report_markdown(report), nl=False)
        return
    _print_glossary_gap_report(run_dir)


@batch_app.command("glossary-update-plan")
def batch_glossary_update_plan(
    run_dir: Path,
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable glossary update plan JSON."),
    markdown_output: bool = typer.Option(False, "--markdown", help="Emit a human-readable Markdown glossary update plan."),
    write: bool = typer.Option(False, "--write", help="Write glossary_update_plan.json and glossary_update_plan.md into the batch run directory."),
) -> None:
    manifest_path = run_dir / "batch_manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]Error:[/red] No batch manifest found at {manifest_path}")
        raise typer.Exit(1)
    plan = build_glossary_update_plan(run_dir)
    if write:
        json_path = run_dir / "glossary_update_plan.json"
        markdown_path = run_dir / "glossary_update_plan.md"
        json_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        markdown_path.write_text(render_glossary_update_plan_markdown(plan), encoding="utf-8")
        if not json_output and not markdown_output:
            console.print(f"Wrote glossary update plan: [bold]{json_path}[/bold]")
            console.print(f"Wrote glossary update Markdown: [bold]{markdown_path}[/bold]")
    if json_output:
        typer.echo(plan.model_dump_json(indent=2))
        return
    if markdown_output:
        typer.echo(render_glossary_update_plan_markdown(plan), nl=False)
        return
    typer.echo(render_glossary_update_plan_markdown(plan), nl=False)


@batch_app.command("apply-glossary-update-plan")
def batch_apply_glossary_update_plan(
    run_dir: Path,
    glossary_path: Path | None = typer.Option(
        None,
        "--glossary",
        help="Glossary file to update. Defaults to the batch story config glossary path.",
    ),
    write: bool = typer.Option(False, "--write", help="Actually update the glossary. Without this, print a dry-run plan."),
    no_backup: bool = typer.Option(False, "--no-backup", help="Do not write <glossary>.bak before applying changes."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable application JSON."),
    markdown_output: bool = typer.Option(False, "--markdown", help="Emit a human-readable Markdown application report."),
) -> None:
    manifest_path = run_dir / "batch_manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]Error:[/red] No batch manifest found at {manifest_path}")
        raise typer.Exit(1)
    try:
        application = apply_glossary_update_plan(
            run_dir,
            glossary_path=glossary_path,
            write=write,
            create_backup=not no_backup,
        )
    except OSError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    if json_output:
        typer.echo(application.model_dump_json(indent=2))
        return
    if markdown_output:
        typer.echo(render_glossary_update_application_markdown(application), nl=False)
        return
    if application.dry_run:
        console.print("[yellow]Dry run:[/yellow] pass --write to update the glossary.")
    elif application.summary.changed_count:
        console.print(f"Updated glossary: [bold]{application.glossary_path}[/bold]")
        if application.backup_path:
            console.print(f"Backup: [bold]{application.backup_path}[/bold]")
    else:
        console.print("No glossary changes were needed.")
    typer.echo(render_glossary_update_application_markdown(application), nl=False)


@batch_app.command("glossary-pass")
def batch_glossary_pass(
    run_dir: Path,
    glossary_path: Path | None = typer.Option(
        None,
        "--glossary",
        help="Glossary file to update. Defaults to the batch story config glossary path.",
    ),
    chapters: str | None = typer.Option(None, "--chapters", help="Optional chapter subset to rerun after applying glossary updates."),
    write: bool = typer.Option(False, "--write", help="Apply glossary updates and rerun affected review chapters. Without this, dry-run only."),
    no_backup: bool = typer.Option(False, "--no-backup", help="Do not write <glossary>.bak before applying changes."),
    seed: int = typer.Option(7, help="Deterministic candidate shuffle seed for the rerun."),
    skip_epub: bool = typer.Option(False, help="Skip EPUB generation during rerun."),
    allow_source_qa_fail: bool = typer.Option(False, help="Continue after source QA errors during rerun."),
    report_mode: str | None = typer.Option("excerpt", help="Override chapter report mode during rerun."),
    write_proof: bool = typer.Option(False, "--write-proof", help="Write non-gating proof artifacts after rerun."),
    write_triage: bool = typer.Option(True, "--write-triage/--no-write-triage", help="Refresh triage artifacts after a write rerun."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable pass JSON."),
    markdown_output: bool = typer.Option(False, "--markdown", help="Emit a human-readable Markdown pass report."),
) -> None:
    manifest_path = run_dir / "batch_manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]Error:[/red] No batch manifest found at {manifest_path}")
        raise typer.Exit(1)
    try:
        result = run_glossary_update_pass(
            run_dir,
            glossary_path=glossary_path,
            write=write,
            create_backup=not no_backup,
            chapters=parse_chapter_selection(chapters) if chapters else None,
            seed=seed,
            skip_epub=skip_epub,
            allow_source_qa_fail=allow_source_qa_fail,
            report_mode=report_mode,
            write_proof=write_proof,
            write_triage=write_triage,
        )
    except (ValueError, RuntimeError, LLMProviderUnavailable, OSError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return
    if markdown_output:
        typer.echo(render_glossary_update_pass_markdown(result), nl=False)
        return
    if result.dry_run:
        console.print("[yellow]Dry run:[/yellow] pass --write to update the glossary and rerun affected review chapters.")
    elif result.rerun_started:
        console.print(f"Applied glossary updates and reran: [bold]{','.join(result.chapters)}[/bold]")
    else:
        console.print(result.message)
    typer.echo(render_glossary_update_pass_markdown(result), nl=False)


@batch_app.command("manual-edit-plan")
def batch_manual_edit_plan(
    run_dir: Path,
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable manual edit plan JSON."),
    markdown_output: bool = typer.Option(False, "--markdown", help="Emit a human-readable Markdown manual edit plan."),
    write: bool = typer.Option(False, "--write", help="Write manual_edit_plan.json and manual_edit_plan.md into the batch run directory."),
) -> None:
    manifest_path = run_dir / "batch_manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]Error:[/red] No batch manifest found at {manifest_path}")
        raise typer.Exit(1)
    plan = build_manual_edit_plan(run_dir)
    if write:
        json_path = run_dir / "manual_edit_plan.json"
        markdown_path = run_dir / "manual_edit_plan.md"
        json_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        markdown_path.write_text(render_manual_edit_plan_markdown(plan), encoding="utf-8")
        if not json_output and not markdown_output:
            console.print(f"Wrote manual edit plan: [bold]{json_path}[/bold]")
            console.print(f"Wrote manual edit Markdown: [bold]{markdown_path}[/bold]")
    if json_output:
        typer.echo(plan.model_dump_json(indent=2))
        return
    if markdown_output:
        typer.echo(render_manual_edit_plan_markdown(plan), nl=False)
        return
    typer.echo(render_manual_edit_plan_markdown(plan), nl=False)


@batch_app.command("work-order")
def batch_work_order(
    run_dir: Path,
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable agent work-order JSON."),
    markdown_output: bool = typer.Option(False, "--markdown", help="Emit a human-readable Markdown work order."),
    write: bool = typer.Option(False, "--write", help="Write agentic_work_order.json and agentic_work_order.md into the batch run directory."),
) -> None:
    manifest_path = run_dir / "batch_manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]Error:[/red] No batch manifest found at {manifest_path}")
        raise typer.Exit(1)
    work_order = build_agent_work_order(run_dir)
    if write:
        json_path = run_dir / "agentic_work_order.json"
        markdown_path = run_dir / "agentic_work_order.md"
        json_path.write_text(work_order.model_dump_json(indent=2), encoding="utf-8")
        markdown_path.write_text(render_agent_work_order_markdown(work_order), encoding="utf-8")
        if not json_output and not markdown_output:
            console.print(f"Wrote agent work order: [bold]{json_path}[/bold]")
            console.print(f"Wrote agent work-order Markdown: [bold]{markdown_path}[/bold]")
    if json_output:
        typer.echo(work_order.model_dump_json(indent=2))
        return
    if markdown_output:
        typer.echo(render_agent_work_order_markdown(work_order), nl=False)
        return
    _print_agent_work_order(run_dir)


@batch_app.command("execute-work-order")
def batch_execute_work_order(
    run_dir: Path,
    action: Literal["live-retry"] = typer.Option("live-retry", "--action", help="Work-order action to execute."),
    provider_mode: Literal["replay", "live"] = typer.Option("live", "--provider-mode", help="Provider execution mode for selected work-order chapters."),
    translation_provider: str = typer.Option("offline", help="Translation provider name."),
    judge_provider: str = typer.Option("openai", help="Judge provider name."),
    repair_provider: str = typer.Option("openai", help="Repair provider name."),
    record_cache: bool = typer.Option(True, "--record-cache/--no-record-cache", help="Record live provider JSON responses for replay mode."),
    cache_dir: Path | None = typer.Option(Path(".agentic_cache"), help="Response cache directory for live/replay providers."),
    model_name: str | None = typer.Option(None, "--model", help="Explicit OpenAI-compatible model name. Alternative to AGENTIC_TRANSLATION_MODEL."),
    seed: int = typer.Option(7, help="Deterministic candidate shuffle seed."),
    skip_epub: bool = typer.Option(False, help="Skip EPUB generation."),
    allow_source_qa_fail: bool = typer.Option(False, help="Continue after source QA errors. Intended only for local debugging."),
    allow_live_provider_fallback: bool = typer.Option(False, help="If a live translation/judge/repair call fails, continue with offline fallback and record the fallback reason."),
    allow_review_required: bool = typer.Option(False, help="Return exit code 0 when chapters still need review. Intended only for triage/debug runs."),
    report_mode: str | None = typer.Option("excerpt", help="Override chapter report mode: full, excerpt, or redacted."),
    tool_agent: bool | None = typer.Option(None, "--tool-agent/--no-tool-agent", help="Override the source batch's bounded repair-agent setting."),
    write_proof: bool = typer.Option(True, "--write-proof/--no-write-proof", help="Write non-gating agentic_proof.json and agentic_proof.md after execution."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview selected chapters and preflight checks without mutating the batch."),
    write_preview: bool = typer.Option(False, "--write-preview", help="With --dry-run, write agentic_execution_preview.json and .md without mutating the batch manifest."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON. Most useful with --dry-run."),
) -> None:
    manifest_path = run_dir / "batch_manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]Error:[/red] No batch manifest found at {manifest_path}")
        raise typer.Exit(1)
    if dry_run:
        try:
            preview = preview_agent_work_order_execution(
                run_dir,
                action=action,
                provider_mode=provider_mode,
                translation_provider_name=translation_provider,
                judge_provider_name=judge_provider,
                repair_provider_name=repair_provider,
                record_cache=record_cache,
                cache_dir=cache_dir,
                model_name=model_name,
                tool_agent_enabled=tool_agent,
            )
        except (ValueError, RuntimeError, LLMProviderUnavailable) as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc
        if write_preview:
            json_path, markdown_path = write_agent_work_order_execution_preview(run_dir, preview)
            if not json_output:
                console.print(f"Wrote execution preview: [bold]{json_path}[/bold]")
                console.print(f"Wrote execution preview Markdown: [bold]{markdown_path}[/bold]")
        if json_output:
            typer.echo(preview.model_dump_json(indent=2))
        else:
            _print_work_order_execution_preview(preview)
        if not preview.preflight_passed:
            raise typer.Exit(1)
        return
    try:
        result = execute_agent_work_order(
            run_dir,
            action=action,
            provider_mode=provider_mode,
            translation_provider_name=translation_provider,
            judge_provider_name=judge_provider,
            repair_provider_name=repair_provider,
            record_cache=record_cache,
            cache_dir=cache_dir,
            model_name=model_name,
            tool_agent_enabled=tool_agent,
            seed=seed,
            skip_epub=skip_epub,
            allow_source_qa_fail=allow_source_qa_fail,
            allow_live_provider_fallback=allow_live_provider_fallback,
            report_mode=report_mode,
            retry_review_required=True,
            write_proof=write_proof,
        )
    except (ValueError, RuntimeError, LLMProviderUnavailable) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    if json_output:
        typer.echo(result.manifest.model_dump_json(indent=2))
        return
    _print_batch_summary(result.manifest_path)
    _exit_if_batch_has_blockers(result.manifest_path, allow_review_required=allow_review_required)


@batch_app.command("replay")
def batch_replay(
    source_run_dir: Path,
    chapters: str | None = typer.Option(None, "--chapters", help="Optional chapter subset to replay, such as 0003,0007 or 1-3."),
    run_id: str | None = typer.Option(None, help="Optional replay run id. Defaults to <source_run_id>_replay."),
    seed: int = typer.Option(7, help="Deterministic candidate shuffle seed."),
    overwrite: bool = typer.Option(False, help="Overwrite an existing replay run directory."),
    skip_epub: bool = typer.Option(False, help="Skip EPUB generation."),
    allow_source_qa_fail: bool = typer.Option(False, help="Continue after source QA errors. Intended only for local debugging."),
    allow_review_required: bool = typer.Option(False, help="Return exit code 0 when chapters need human review. Intended only for triage/debug runs."),
    report_mode: str | None = typer.Option("excerpt", help="Override chapter report mode: full, excerpt, or redacted."),
    write_proof: bool = typer.Option(True, "--write-proof/--no-write-proof", help="Write non-gating agentic_proof.json and agentic_proof.md after replay."),
) -> None:
    manifest_path = source_run_dir / "batch_manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]Error:[/red] No batch manifest found at {manifest_path}")
        raise typer.Exit(1)
    try:
        result = replay_batch_pipeline(
            source_run_dir,
            chapters=parse_chapter_selection(chapters) if chapters else None,
            run_id=run_id,
            seed=seed,
            overwrite=overwrite,
            skip_epub=skip_epub,
            allow_source_qa_fail=allow_source_qa_fail,
            report_mode=report_mode,
            write_proof=write_proof,
        )
    except (ValueError, RuntimeError, LLMProviderUnavailable) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_batch_summary(result.manifest_path)
    _exit_if_batch_has_blockers(result.manifest_path, allow_review_required=allow_review_required)


@batch_app.command("resume")
def batch_resume(
    run_dir: Path,
    chapters: str | None = typer.Option(None, "--chapters", help="Optional chapter subset to resume, such as 0003,0007 or 1-3."),
    provider_mode: Literal["offline", "replay", "live"] | None = typer.Option(None, "--provider-mode", help="Override provider execution mode."),
    translation_provider: str | None = typer.Option(None, help="Override translation provider name."),
    judge_provider: str | None = typer.Option(None, help="Override judge provider name."),
    repair_provider: str | None = typer.Option(None, help="Override repair provider name."),
    record_cache: bool = typer.Option(False, help="Record live provider JSON responses for replay mode."),
    cache_dir: Path | None = typer.Option(None, help="Response cache directory for live/replay providers."),
    model_name: str | None = typer.Option(None, "--model", help="Explicit OpenAI-compatible model name. Alternative to AGENTIC_TRANSLATION_MODEL."),
    seed: int = typer.Option(7, help="Deterministic candidate shuffle seed."),
    force: bool = typer.Option(False, help="Re-run completed chapter subruns."),
    retry_review_required: bool = typer.Option(False, help="Rerun only review_required chapters while leaving packaged chapters untouched."),
    skip_epub: bool = typer.Option(False, help="Skip EPUB generation."),
    allow_source_qa_fail: bool = typer.Option(False, help="Continue after source QA errors. Intended only for local debugging."),
    allow_live_provider_fallback: bool = typer.Option(False, help="If a live translation/judge/repair call fails, continue with offline fallback and record the fallback reason."),
    tool_agent: bool | None = typer.Option(None, "--tool-agent/--no-tool-agent", help="Override the saved batch repair-agent setting."),
    allow_review_required: bool = typer.Option(False, help="Return exit code 0 when chapters need human review. Intended only for triage/debug runs."),
    report_mode: str | None = typer.Option("excerpt", help="Override chapter report mode: full, excerpt, or redacted."),
    write_proof: bool = typer.Option(False, "--write-proof", help="Write non-gating agentic_proof.json and agentic_proof.md after resume."),
    write_triage: bool = typer.Option(False, "--write-triage", help="Write review queue, glossary gap report, and agent work order after resume."),
) -> None:
    manifest_path = run_dir / "batch_manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]Error:[/red] No batch manifest found at {manifest_path}")
        raise typer.Exit(1)
    try:
        result = resume_batch_pipeline(
            run_dir,
            chapters=parse_chapter_selection(chapters) if chapters else None,
            provider_mode=provider_mode,
            translation_provider_name=translation_provider,
            judge_provider_name=judge_provider,
            repair_provider_name=repair_provider,
            record_cache=record_cache,
            cache_dir=cache_dir,
            model_name=model_name,
            seed=seed,
            force=force,
            retry_review_required=retry_review_required,
            skip_epub=skip_epub,
            allow_source_qa_fail=allow_source_qa_fail,
            allow_live_provider_fallback=allow_live_provider_fallback,
            tool_agent_enabled=tool_agent,
            report_mode=report_mode,
            write_proof=write_proof,
        )
    except (ValueError, RuntimeError, LLMProviderUnavailable) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_batch_summary(result.manifest_path)
    _write_triage_if_requested(result.run_dir, write_triage=write_triage)
    _exit_if_batch_has_blockers(result.manifest_path, allow_review_required=allow_review_required)


@batch_app.command("refresh")
def batch_refresh(
    run_dir: Path,
    chapters: str | None = typer.Option(None, "--chapters", help="Optional chapter subset to re-QA, such as 0003,0007 or 1-3."),
    skip_epub: bool = typer.Option(False, help="Skip EPUB generation."),
    allow_review_required: bool = typer.Option(False, help="Return exit code 0 when chapters need human review. Intended only for triage/debug runs."),
    write_proof: bool = typer.Option(False, "--write-proof", help="Write non-gating agentic_proof.json and agentic_proof.md after refresh."),
) -> None:
    manifest_path = run_dir / "batch_manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]Error:[/red] No batch manifest found at {manifest_path}")
        raise typer.Exit(1)
    try:
        result = refresh_batch_pipeline(
            run_dir,
            chapters=parse_chapter_selection(chapters) if chapters else None,
            skip_epub=skip_epub,
            write_proof=write_proof,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_batch_summary(result.manifest_path)
    _exit_if_batch_has_blockers(result.manifest_path, allow_review_required=allow_review_required)


@batch_app.command("accept")
def batch_accept(
    run_dir: Path,
    chapters: str | None = typer.Option(None, "--chapters", help="Optional chapter subset to accept, such as 0003,0007 or 1-3."),
    reviewer: str = typer.Option("human", "--reviewer", help="Name or handle for the human reviewer."),
    note: str = typer.Option(..., "--note", help="Required note explaining the human review decision."),
    skip_epub: bool = typer.Option(False, help="Skip EPUB generation."),
    allow_review_required: bool = typer.Option(False, help="Return exit code 0 when chapters still need review. Intended only for accepted triage edge cases."),
    write_proof: bool = typer.Option(False, "--write-proof", help="Write non-gating agentic_proof.json and agentic_proof.md after acceptance."),
) -> None:
    manifest_path = run_dir / "batch_manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]Error:[/red] No batch manifest found at {manifest_path}")
        raise typer.Exit(1)
    try:
        result = accept_reviewed_chapters(
            run_dir,
            chapters=parse_chapter_selection(chapters) if chapters else None,
            reviewer=reviewer,
            note=note,
            skip_epub=skip_epub,
            write_proof=write_proof,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_batch_summary(result.manifest_path)
    console.print(f"Manual review ledger: [bold]{run_dir / 'manual_review.jsonl'}[/bold]")
    _exit_if_batch_has_blockers(result.manifest_path, allow_review_required=allow_review_required)


def _parse_single_chapter(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("A chapter id is required.")
    try:
        parsed = parse_chapter_selection(stripped)
    except ValueError:
        return stripped
    if len(parsed) != 1:
        raise ValueError("Choose exactly one chapter for text replacement.")
    return parsed[0]


@batch_app.command("replace-text")
def batch_replace_text(
    run_dir: Path,
    chapter: str = typer.Option(..., "--chapter", help="Single chapter id, such as 0001 or 1."),
    old_text: str = typer.Option(..., "--old", help="Exact text to replace in the final chapter file."),
    new_text: str = typer.Option(..., "--new", help="Replacement text to write into the final chapter file."),
    reviewer: str = typer.Option("human", "--reviewer", help="Name or handle for the human reviewer."),
    note: str | None = typer.Option(None, "--note", help="Manual review note. Defaults to a generated replacement note."),
    refresh_only: bool = typer.Option(False, "--refresh-only", help="Only refresh QA/packaging; do not write a manual-review ledger entry."),
    skip_epub: bool = typer.Option(False, help="Skip EPUB generation."),
    allow_review_required: bool = typer.Option(False, help="Return exit code 0 when chapters still need review after replacement."),
    write_proof: bool = typer.Option(False, "--write-proof", help="Write non-gating agentic_proof.json and agentic_proof.md after replacement."),
    write_triage: bool = typer.Option(True, "--write-triage/--no-write-triage", help="Refresh triage artifacts after replacement."),
) -> None:
    manifest_path = run_dir / "batch_manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]Error:[/red] No batch manifest found at {manifest_path}")
        raise typer.Exit(1)
    try:
        result = apply_manual_text_replacement(
            run_dir,
            chapter=_parse_single_chapter(chapter),
            old_text=old_text,
            new_text=new_text,
            reviewer=reviewer,
            note=note,
            refresh_only=refresh_only,
            skip_epub=skip_epub,
            write_proof=write_proof,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(
        f"Replaced {result.occurrence_count} occurrence(s) in [bold]{result.chapter}[/bold]: "
        f"{result.final_path}"
    )
    console.print(
        f"After refresh: status [bold]{result.status_after}[/bold], "
        f"score {result.final_score_after}, findings {result.final_findings_after}"
    )
    if not result.refresh_only:
        console.print(f"Manual review ledger: [bold]{run_dir / 'manual_review.jsonl'}[/bold]")
    _write_triage_if_requested(run_dir, write_triage=write_triage)
    _exit_if_batch_has_blockers(manifest_path, allow_review_required=allow_review_required)


@batch_app.command("normalize-panels")
def batch_normalize_panels(
    run_dir: Path,
    chapters: str | None = typer.Option(None, "--chapters", help="Optional chapter subset, such as 0003,0007 or 1-3."),
    reviewer: str = typer.Option("human", "--reviewer", help="Name or handle for the reviewer recorded in the ledger."),
    note_prefix: str = typer.Option("Merged split numbered note panels.", "--note-prefix", help="Prefix for generated manual-review notes."),
    skip_epub: bool = typer.Option(False, help="Skip EPUB generation."),
    allow_review_required: bool = typer.Option(False, help="Return exit code 0 when chapters still need review after panel normalization."),
    write_proof: bool = typer.Option(False, "--write-proof", help="Write non-gating agentic_proof.json and agentic_proof.md after normalization."),
    write_triage: bool = typer.Option(True, "--write-triage/--no-write-triage", help="Refresh triage artifacts after normalization."),
) -> None:
    manifest_path = run_dir / "batch_manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]Error:[/red] No batch manifest found at {manifest_path}")
        raise typer.Exit(1)
    try:
        result = normalize_panel_splits(
            run_dir,
            chapters=parse_chapter_selection(chapters) if chapters else None,
            reviewer=reviewer,
            note_prefix=note_prefix,
            skip_epub=skip_epub,
            write_proof=write_proof,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(
        f"Normalized {result.normalized_count} panel split(s); "
        f"skipped {result.skipped_count}."
    )
    if result.items:
        table = Table(title="Panel Normalization")
        table.add_column("Chapter")
        table.add_column("Status")
        table.add_column("Merges", justify="right")
        table.add_column("After")
        table.add_column("Reason")
        for item in result.items:
            after = item.status_after or "-"
            if item.final_findings_after is not None:
                after = f"{after}, findings {item.final_findings_after}"
            table.add_row(item.chapter, item.status, str(item.replacement_count), after, item.reason)
        console.print(table)
    console.print(f"Manual review ledger: [bold]{run_dir / 'manual_review.jsonl'}[/bold]")
    _write_triage_if_requested(run_dir, write_triage=write_triage)
    _exit_if_batch_has_blockers(manifest_path, allow_review_required=allow_review_required)


def _smoke_label(chapters: list[str]) -> str:
    if not chapters:
        return "empty"
    if len(chapters) == 1:
        return chapters[0]
    return f"{chapters[0]}_{chapters[-1]}"


def _default_smoke_out(source_dir: Path, chapters: list[str]) -> Path:
    source_root = source_dir.parent.name or source_dir.name
    return Path("local_fixtures") / f"{source_root}_smoke_{_smoke_label(chapters)}"


def _default_smoke_run_id(project_dir: Path, chapters: list[str], *, practical: bool, deepseek: bool) -> str:
    label = _smoke_label(chapters)
    if deepseek:
        mode = "deepseek"
    elif practical:
        mode = "practical"
    else:
        mode = "smoke"
    return f"{project_dir.name}_{mode}_{label}"


def _project_translated_dir(project_dir: Path, chapters: list[str]) -> Path | None:
    candidates = sorted(
        path
        for path in project_dir.glob("translated*")
        if path.is_dir() and all((path / f"{chapter}.txt").exists() for chapter in chapters)
    )
    return candidates[-1] if candidates else None


def _require_smoke_project_layout(project_dir: Path) -> tuple[Path, Path]:
    source_dir = project_dir / "scraped"
    glossary = project_dir / "terms" / "master_glossary.txt"
    missing = []
    if not source_dir.is_dir():
        missing.append("scraped/")
    if not glossary.is_file():
        missing.append("terms/master_glossary.txt")
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Expected project layout under {project_dir}: missing {joined}")
    return source_dir, glossary


def _project_source_chapters(source_dir: Path) -> list[str]:
    return sorted(path.stem for path in source_dir.glob("*.txt") if path.stem.isdigit())


def _normalize_chapter_id(chapter: str | None) -> str | None:
    return chapter.zfill(4) if chapter and chapter.isdigit() else chapter


def _project_first_chapters(
    source_dir: Path,
    *,
    first: int,
    start: str | None = None,
    until: str | None = None,
) -> list[str]:
    if first < 1:
        raise ValueError("--first must be at least 1")
    available = _project_source_chapters(source_dir)
    if not available:
        raise ValueError(f"No numeric chapter .txt files found in {source_dir}")
    start_id = _normalize_chapter_id(start)
    until_id = _normalize_chapter_id(until)
    if start_id:
        available = [chapter for chapter in available if chapter >= start_id]
    if until_id:
        available = [chapter for chapter in available if chapter <= until_id]
    selected = available[:first]
    if not selected:
        if until_id and start_id:
            raise ValueError(f"No chapter files found from {start_id} through {until_id}")
        raise ValueError(f"No chapter files found at or after {start_id}")
    return selected


def _last_manifest_chapter(run_dir: Path) -> str:
    manifest_path = run_dir / "batch_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Previous run has no batch_manifest.json: {run_dir}")
    manifest = load_batch_manifest(manifest_path)
    chapters = sorted(chapter for chapter in manifest.chapters if chapter.isdigit())
    if not chapters:
        raise ValueError(f"Previous run has no numeric chapter ids: {run_dir}")
    return chapters[-1]


def _latest_project_run(project_dir: Path, *, runs_dir: Path = Path("runs")) -> Path:
    story_prefix = f"{project_dir.name}_smoke_"
    best: tuple[str, Path] | None = None
    for manifest_path in sorted(runs_dir.glob("*/batch_manifest.json")):
        try:
            manifest = load_batch_manifest(manifest_path)
        except (OSError, ValueError):
            continue
        if not manifest.story_slug.startswith(story_prefix):
            continue
        chapters = sorted(chapter for chapter in manifest.chapters if chapter.isdigit())
        if not chapters:
            continue
        candidate = (chapters[-1], manifest_path.parent)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        raise ValueError(f"No previous smoke-project runs found for {project_dir.name} under {runs_dir}")
    return best[1]


def _project_run_manifests(project_dir: Path, *, runs_dir: Path = Path("runs")) -> list[tuple[Path, object]]:
    story_prefix = f"{project_dir.name}_smoke_"
    manifests = []
    for manifest_path in sorted(runs_dir.glob("*/batch_manifest.json")):
        try:
            manifest = load_batch_manifest(manifest_path)
        except (OSError, ValueError):
            continue
        if manifest.story_slug.startswith(story_prefix):
            manifests.append((manifest_path.parent, manifest))
    return manifests


def _project_status_payload(project_dir: Path, *, runs_dir: Path = Path("runs")) -> dict[str, object]:
    source_dir, _glossary = _require_smoke_project_layout(project_dir)
    source_chapters = _project_source_chapters(source_dir)
    manifests = _project_run_manifests(project_dir, runs_dir=runs_dir)
    latest_by_chapter: dict[str, tuple[float, str, str]] = {}
    recent_runs: list[dict[str, object]] = []
    for run_dir, manifest in manifests:
        numeric_chapters = sorted(chapter for chapter in manifest.chapters if chapter.isdigit())
        if not numeric_chapters:
            continue
        mtime = (run_dir / "batch_manifest.json").stat().st_mtime
        latest_chapter = numeric_chapters[-1]
        recent_runs.append(
            {
                "run_id": manifest.run_id,
                "run_dir": str(run_dir),
                "latest_chapter": latest_chapter,
                "summary": manifest.summary.model_dump(),
            }
        )
        for chapter in numeric_chapters:
            chapter_run = manifest.chapters[chapter]
            current = latest_by_chapter.get(chapter)
            if current is None or mtime >= current[0]:
                latest_by_chapter[chapter] = (mtime, chapter_run.status, manifest.run_id)
    recent_runs.sort(key=lambda item: str(item["latest_chapter"]))
    processed_chapters = sorted(latest_by_chapter)
    status_counts: dict[str, int] = {}
    for _mtime, status, _run_id in latest_by_chapter.values():
        status_counts[status] = status_counts.get(status, 0) + 1
    latest_chapter = processed_chapters[-1] if processed_chapters else None
    next_chapter = None
    if latest_chapter is not None:
        for chapter in source_chapters:
            if chapter > latest_chapter:
                next_chapter = chapter
                break
    return {
        "project": project_dir.name,
        "runs_dir": str(runs_dir),
        "run_count": len(manifests),
        "total_source_chapters": len(source_chapters),
        "processed_unique_chapters": len(processed_chapters),
        "latest_chapter": latest_chapter,
        "next_chapter": next_chapter,
        "status_counts": dict(sorted(status_counts.items())),
        "recent_runs": recent_runs[-10:],
    }


def _next_chapter_id(chapter: str) -> str:
    return str(int(chapter) + 1).zfill(len(chapter))


def _chunked_chapters(chapters: list[str], count: int) -> list[list[str]]:
    return [chapters[index : index + count] for index in range(0, len(chapters), count)]


def _produce_follow_up_command(
    project_dir: Path,
    *,
    chunk: list[str],
    provider: str,
    cheap: int | None,
    cache_dir: Path | None,
) -> str:
    parts = [
        "agentic-translation",
        "produce",
        str(project_dir),
        "--chapters",
        ",".join(chunk),
    ]
    if provider != "offline":
        parts.extend(["--provider", provider])
    if cheap is not None:
        parts.extend(["--cheap", str(cheap)])
    if cache_dir is not None:
        parts.extend(["--cache-dir", str(cache_dir)])
    parts.append("--overwrite")
    return shlex.join(parts)


def _produce_chapter_chunks(
    project_dir: Path,
    *,
    count: int,
    chapters: str | None,
    start: str | None,
    until: str | None,
) -> tuple[list[list[str]], Path | None]:
    if count < 1:
        raise ValueError("--count must be at least 1")
    source_dir, _glossary = _require_smoke_project_layout(project_dir)
    if chapters:
        return [parse_chapter_selection(chapters)], None

    start_id = _normalize_chapter_id(start)
    latest_run: Path | None = None
    if start_id is None:
        try:
            latest_run = _latest_project_run(project_dir)
        except ValueError:
            latest_run = None
        else:
            start_id = _next_chapter_id(_last_manifest_chapter(latest_run))
    until_id = _normalize_chapter_id(until)
    available = _project_source_chapters(source_dir)
    if start_id:
        available = [chapter for chapter in available if chapter >= start_id]
    if until_id:
        available = [chapter for chapter in available if chapter <= until_id]
    selected = available if until_id else available[:count]
    if not selected:
        if start_id and until_id:
            raise ValueError(f"No chapter files found from {start_id} through {until_id}")
        if start_id:
            raise ValueError(f"No chapter files found at or after {start_id}")
        raise ValueError(f"No numeric chapter .txt files found in {source_dir}")
    return _chunked_chapters(selected, count), latest_run


def _empty_produce_status_summary() -> dict[str, int]:
    return {
        "total_chapters": 0,
        "pending": 0,
        "packaged": 0,
        "review_required": 0,
        "failed": 0,
        "skipped": 0,
        "incomplete": 0,
    }


def _produce_manifest_artifacts(run_dir: Path, manifest) -> dict[str, str]:  # noqa: ANN001 - manifest is a Pydantic model from load_batch_manifest.
    artifacts: dict[str, str] = {}
    for artifact_type in ("txt", "epub"):
        relative_path = manifest.artifacts.get(artifact_type)
        if not relative_path:
            continue
        artifact_path = run_dir / relative_path
        if artifact_path.exists():
            artifacts[artifact_type] = str(artifact_path)
    return artifacts


def _produce_run_summaries(run_ids: list[str]) -> tuple[list[dict[str, object]], dict[str, int]]:
    run_summaries: list[dict[str, object]] = []
    aggregate = _empty_produce_status_summary()
    for run_id in run_ids:
        run_dir = Path("runs") / run_id
        summary: dict[str, int] | None = None
        artifacts: dict[str, str] = {}
        manifest_path = run_dir / "batch_manifest.json"
        if manifest_path.exists():
            manifest = load_batch_manifest(manifest_path)
            summary = manifest.summary.model_dump()
            artifacts = _produce_manifest_artifacts(run_dir, manifest)
            for key in aggregate:
                aggregate[key] += int(summary.get(key, 0))
        run_summaries.append(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "summary": summary,
                "artifacts": artifacts,
            }
        )
    return run_summaries, aggregate


def _produce_artifact_summary(run_summaries: list[dict[str, object]]) -> dict[str, list[str]]:
    artifacts = {"txt": [], "epub": []}
    for run_summary in run_summaries:
        run_artifacts = run_summary.get("artifacts")
        if not isinstance(run_artifacts, dict):
            continue
        for artifact_type in artifacts:
            artifact_path = run_artifacts.get(artifact_type)
            if artifact_path:
                artifacts[artifact_type].append(str(artifact_path))
    return artifacts


def _produce_next_dry_run_command(
    project_dir: Path,
    *,
    count: int,
    provider: str,
    cheap: int | None,
    cache_dir: Path | None,
) -> str:
    parts = ["agentic-translation", "produce", str(project_dir), "--count", str(count), "--dry-run"]
    if provider != "offline":
        parts.extend(["--provider", provider])
    if cheap is not None:
        parts.extend(["--cheap", str(cheap)])
    if cache_dir is not None:
        parts.extend(["--cache-dir", str(cache_dir)])
    return shlex.join(parts)


def _produce_recommended_next_actions(
    project_dir: Path,
    *,
    count: int,
    provider: str,
    cheap: int | None,
    cache_dir: Path | None,
    run_summaries: list[dict[str, object]],
    status_summary: dict[str, int],
) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    for run_summary in run_summaries:
        run_id = str(run_summary["run_id"])
        run_dir = str(run_summary["run_dir"])
        summary = run_summary.get("summary")
        if not isinstance(summary, dict):
            actions.append(
                {
                    "action": "inspect",
                    "run_id": run_id,
                    "run_dir": run_dir,
                    "reason": "batch manifest was not found.",
                    "command": shlex.join(["agentic-translation", "batch", "inspect", run_dir]),
                }
            )
            continue
        failed = int(summary.get("failed", 0))
        incomplete = int(summary.get("incomplete", 0))
        if failed or incomplete:
            actions.append(
                {
                    "action": "inspect",
                    "run_id": run_id,
                    "run_dir": run_dir,
                    "reason": f"{failed} failed and {incomplete} incomplete chapter(s).",
                    "command": shlex.join(["agentic-translation", "batch", "inspect", run_dir]),
                }
            )
            continue
        review_required = int(summary.get("review_required", 0))
        if review_required:
            actions.append(
                {
                    "action": "review",
                    "run_id": run_id,
                    "run_dir": run_dir,
                    "reason": f"{review_required} chapter(s) require review.",
                    "command": shlex.join(["agentic-translation", "batch", "review", run_dir, "--write", "--write-markdown"]),
                }
            )
    if not actions and status_summary["total_chapters"] and status_summary["packaged"] == status_summary["total_chapters"]:
        actions.append(
            {
                "action": "plan_next_chunk",
                "reason": "produced run(s) packaged cleanly.",
                "command": _produce_next_dry_run_command(
                    project_dir,
                    count=count,
                    provider=provider,
                    cheap=cheap,
                    cache_dir=cache_dir,
                ),
            }
        )
    return actions


@app.command("project-status")
def project_status(
    project_dir: Path = typer.Argument(..., help="Corpus project directory containing scraped/ and terms/master_glossary.txt."),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", help="Directory containing batch/smoke run folders."),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    try:
        payload = _project_status_payload(project_dir, runs_dir=runs_dir)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    table = Table(title=f"Project Status: {payload['project']}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Runs", str(payload["run_count"]))
    table.add_row("Source chapters", str(payload["total_source_chapters"]))
    table.add_row("Processed unique chapters", str(payload["processed_unique_chapters"]))
    table.add_row("Latest chapter", str(payload["latest_chapter"] or "-"))
    table.add_row("Next chapter", str(payload["next_chapter"] or "-"))
    status_counts = payload["status_counts"]
    if isinstance(status_counts, dict):
        table.add_row("Statuses", ", ".join(f"{key}={value}" for key, value in status_counts.items()) or "-")
    console.print(table)
    recent_runs = payload["recent_runs"]
    if isinstance(recent_runs, list) and recent_runs:
        recent = Table(title="Recent Matching Runs")
        recent.add_column("Run")
        recent.add_column("Latest")
        recent.add_column("Summary")
        for item in recent_runs[-5:]:
            if not isinstance(item, dict):
                continue
            summary = item.get("summary", {})
            summary_text = ""
            if isinstance(summary, dict):
                summary_text = (
                    f"packaged={summary.get('packaged', 0)}, "
                    f"review={summary.get('review_required', 0)}, "
                    f"failed={summary.get('failed', 0)}"
                )
            recent.add_row(str(item.get("run_id", "-")), str(item.get("latest_chapter", "-")), summary_text)
        console.print(recent)


@app.command("smoke-local")
def smoke_local(
    source_dir: Path = typer.Option(..., help="Existing source chapter directory."),
    glossary: Path = typer.Option(..., help="Existing master glossary path."),
    chapters: str = typer.Option("0001", "--chapters", help="Chapter range/list for the smoke run."),
    out: Path | None = typer.Option(None, help="Output fixture directory. Defaults to local_fixtures/<source>_smoke_<range>."),
    translated_dir: Path | None = typer.Option(None, help="Optional existing translated chapter directory to use as cheap baseline."),
    deepseek: bool = typer.Option(False, "--deepseek", help="Shortcut for live DeepSeek translation with cache recording and offline fallback."),
    provider_mode: Literal["offline", "replay", "live"] = typer.Option("offline", "--provider-mode", help="Provider execution mode when --deepseek is not used."),
    translation_provider: str = typer.Option("offline", help="Translation provider name when --deepseek is not used."),
    judge_provider: str = typer.Option("offline", help="Judge provider name when --deepseek is not used."),
    repair_provider: str = typer.Option("offline", help="Repair provider name when --deepseek is not used."),
    record_cache: bool = typer.Option(False, help="Record live provider JSON responses."),
    cache_dir: Path | None = typer.Option(None, help="Response cache directory."),
    model_name: str | None = typer.Option(None, "--model", help="Explicit OpenAI-compatible model name."),
    source_char_limit: int | None = typer.Option(None, "--source-char-limit", help="Copy only the first N source/baseline characters into the smoke fixture for cheap live probes."),
    run_id: str | None = typer.Option(None, help="Optional batch run id."),
    overwrite: bool = typer.Option(False, help="Overwrite an existing batch run directory."),
    skip_epub: bool = typer.Option(False, help="Skip EPUB generation."),
    allow_source_qa_fail: bool = typer.Option(True, "--allow-source-qa-fail/--strict-source-qa", help="Continue after source QA errors by default for local smoke runs."),
    allow_review_required: bool = typer.Option(True, "--allow-review-required/--strict-review", help="Exit 0 when chapters need review by default for local smoke runs."),
    allow_live_provider_fallback: bool = typer.Option(True, "--allow-live-provider-fallback/--no-live-provider-fallback", help="Use offline fallback for live provider failures by default."),
    report_mode: str | None = typer.Option("excerpt", help="Override report mode for the smoke run."),
    practical: bool = typer.Option(False, "--practical", help="Shortcut for the useful corpus loop: safe glossary updates and panel normalization."),
    glossary_pass: bool = typer.Option(False, "--glossary-pass", help="Apply safe glossary updates and rerun affected chapters after the smoke batch."),
    normalize_panels: bool = typer.Option(False, "--normalize-panels", help="After the smoke batch, merge split numbered note panels through the manual-review path."),
    panel_reviewer: str = typer.Option(DEFAULT_PANEL_REVIEWER, "--panel-reviewer", help="Reviewer name for normalize-panels manual-review records."),
    panel_note_prefix: str = typer.Option(DEFAULT_PANEL_NOTE_PREFIX, "--panel-note-prefix", help="Manual-review note prefix for normalize-panels records."),
    write_proof: bool = typer.Option(True, "--write-proof/--no-write-proof", help="Write proof artifacts by default."),
    write_triage: bool = typer.Option(True, "--write-triage/--no-write-triage", help="Write triage artifacts by default."),
) -> None:
    selected_chapters = parse_chapter_selection(chapters)
    fixture_out = out or _default_smoke_out(source_dir, selected_chapters)
    if practical:
        glossary_pass = True
        normalize_panels = True
        if panel_reviewer == DEFAULT_PANEL_REVIEWER:
            panel_reviewer = PRACTICAL_REVIEWER
        if panel_note_prefix == DEFAULT_PANEL_NOTE_PREFIX:
            panel_note_prefix = PRACTICAL_PANEL_NOTE_PREFIX
    if deepseek:
        provider_mode = "live"
        translation_provider = "deepseek"
        judge_provider = "offline"
        repair_provider = "offline"
        record_cache = True
        cache_dir = cache_dir or Path(".agentic_cache")
        model_name = model_name or "deepseek-chat"
        allow_live_provider_fallback = True
    story_yaml = import_local_fixture(
        source_dir=source_dir,
        glossary=glossary,
        chapter=selected_chapters[0],
        out=fixture_out,
        translated_dir=translated_dir,
        chapters=selected_chapters,
        source_char_limit=source_char_limit,
    )
    console.print(f"Wrote local fixture story config: {story_yaml}")
    try:
        result = run_batch_pipeline(
            story_yaml,
            chapters=selected_chapters,
            provider_mode=provider_mode,
            translation_provider_name=translation_provider,
            judge_provider_name=judge_provider,
            repair_provider_name=repair_provider,
            record_cache=record_cache,
            cache_dir=cache_dir,
            model_name=model_name,
            run_id=run_id,
            overwrite=overwrite,
            force=False,
            skip_epub=skip_epub,
            allow_source_qa_fail=allow_source_qa_fail,
            allow_live_provider_fallback=allow_live_provider_fallback,
            report_mode=report_mode,
            write_proof=write_proof,
        )
    except (ValueError, RuntimeError, LLMProviderUnavailable) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_batch_summary(result.manifest_path)
    post_passes_ran = False
    if glossary_pass:
        post_passes_ran = True
        try:
            pass_result = run_glossary_update_pass(
                result.run_dir,
                write=True,
                chapters=selected_chapters,
                seed=7,
                skip_epub=skip_epub,
                allow_source_qa_fail=allow_source_qa_fail,
                report_mode=report_mode,
                write_proof=write_proof,
                write_triage=write_triage,
            )
        except (ValueError, RuntimeError, LLMProviderUnavailable, OSError) as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc
        if pass_result.rerun_started:
            console.print(f"Applied glossary updates and reran: [bold]{','.join(pass_result.chapters)}[/bold]")
        else:
            console.print(pass_result.message)
        typer.echo(render_glossary_update_pass_markdown(pass_result), nl=False)
    if normalize_panels:
        post_passes_ran = True
        try:
            panel_result = normalize_panel_splits(
                result.run_dir,
                chapters=selected_chapters,
                reviewer=panel_reviewer,
                note_prefix=panel_note_prefix,
                skip_epub=skip_epub,
                write_proof=write_proof,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc
        console.print(
            f"Normalized {panel_result.normalized_count} panel split(s); "
            f"skipped {panel_result.skipped_count}."
        )
    if normalize_panels or not glossary_pass:
        _write_triage_if_requested(result.run_dir, write_triage=write_triage)
    if post_passes_ran:
        _print_batch_summary(result.manifest_path)
    _exit_if_batch_has_blockers(result.manifest_path, allow_review_required=allow_review_required)


@app.command("smoke-project")
def smoke_project(
    project_dir: Path = typer.Argument(..., help="Corpus project directory containing scraped/ and terms/master_glossary.txt."),
    chapters: str = typer.Option("0001", "--chapters", help="Chapter range/list for the smoke run."),
    first: int | None = typer.Option(None, "--first", help="Select the first N chapter files from scraped/ instead of --chapters."),
    chunks: int = typer.Option(1, "--chunks", help="Run this many consecutive --first-sized chunks."),
    start: str | None = typer.Option(None, "--start", help="When --first is set, start at this chapter id."),
    until: str | None = typer.Option(None, "--until", help="With --first, keep running chunks until this chapter id is included."),
    after_run: Path | None = typer.Option(None, "--after-run", help="With --first, continue after the highest chapter in a previous batch run."),
    continue_latest: bool = typer.Option(False, "--continue-latest", help="With --first, continue after the highest chapter in the latest matching project smoke run."),
    out: Path | None = typer.Option(None, help="Output fixture directory. Defaults to local_fixtures/<project>_smoke_<range>."),
    translated_dir: Path | None = typer.Option(None, help="Optional translated chapter directory. Defaults to the latest translated* directory that covers the selected chapters."),
    deepseek: bool = typer.Option(False, "--deepseek", help="Shortcut for live DeepSeek translation with cache recording and offline fallback."),
    provider_mode: Literal["offline", "replay", "live"] = typer.Option("offline", "--provider-mode", help="Provider execution mode when --deepseek is not used."),
    translation_provider: str = typer.Option("offline", help="Translation provider name when --deepseek is not used."),
    judge_provider: str = typer.Option("offline", help="Judge provider name when --deepseek is not used."),
    repair_provider: str = typer.Option("offline", help="Repair provider name when --deepseek is not used."),
    record_cache: bool = typer.Option(False, help="Record live provider JSON responses."),
    cache_dir: Path | None = typer.Option(None, help="Response cache directory."),
    model_name: str | None = typer.Option(None, "--model", help="Explicit OpenAI-compatible model name."),
    source_char_limit: int | None = typer.Option(None, "--source-char-limit", help="Copy only the first N source/baseline characters into the smoke fixture for cheap live probes."),
    run_id: str | None = typer.Option(None, help="Optional batch run id."),
    overwrite: bool = typer.Option(False, help="Overwrite an existing batch run directory."),
    skip_epub: bool = typer.Option(False, help="Skip EPUB generation."),
    allow_source_qa_fail: bool = typer.Option(True, "--allow-source-qa-fail/--strict-source-qa", help="Continue after source QA errors by default for local smoke runs."),
    allow_review_required: bool = typer.Option(True, "--allow-review-required/--strict-review", help="Exit 0 when chapters need review by default for local smoke runs."),
    allow_live_provider_fallback: bool = typer.Option(True, "--allow-live-provider-fallback/--no-live-provider-fallback", help="Use offline fallback for live provider failures by default."),
    report_mode: str | None = typer.Option("excerpt", help="Override report mode for the smoke run."),
    practical: bool = typer.Option(False, "--practical", help="Shortcut for the useful corpus loop: safe glossary updates and panel normalization."),
    glossary_pass: bool = typer.Option(False, "--glossary-pass", help="Apply safe glossary updates and rerun affected chapters after the smoke batch."),
    normalize_panels: bool = typer.Option(False, "--normalize-panels", help="After the smoke batch, merge split numbered note panels through the manual-review path."),
    panel_reviewer: str = typer.Option(DEFAULT_PANEL_REVIEWER, "--panel-reviewer", help="Reviewer name for normalize-panels manual-review records."),
    panel_note_prefix: str = typer.Option(DEFAULT_PANEL_NOTE_PREFIX, "--panel-note-prefix", help="Manual-review note prefix for normalize-panels records."),
    write_proof: bool = typer.Option(True, "--write-proof/--no-write-proof", help="Write proof artifacts by default."),
    write_triage: bool = typer.Option(True, "--write-triage/--no-write-triage", help="Write triage artifacts by default."),
) -> None:
    if chunks < 1:
        console.print("[red]Error:[/red] --chunks must be at least 1")
        raise typer.Exit(1)
    if until is not None and chunks != 1:
        console.print("[red]Error:[/red] --until cannot be combined with --chunks")
        raise typer.Exit(1)
    loop_until = _normalize_chapter_id(until)
    if chunks > 1 or loop_until is not None:
        if first is None:
            console.print("[red]Error:[/red] --chunks/--until requires --first")
            raise typer.Exit(1)
        if run_id is not None:
            console.print("[red]Error:[/red] --chunks/--until cannot be combined with --run-id")
            raise typer.Exit(1)
        if out is not None:
            console.print("[red]Error:[/red] --chunks/--until cannot be combined with --out")
            raise typer.Exit(1)
        next_after_run = after_run
        next_continue_latest = continue_latest
        next_start = start
        chunk_index = 0
        while True:
            chunk_index += 1
            chunk_chapters: list[str] | None = None
            if loop_until is None:
                console.print(f"[bold]Chunk {chunk_index}/{chunks}[/bold]")
            else:
                console.print(f"[bold]Chunk {chunk_index}[/bold]")
                console.print(f"Until: {loop_until}")
                try:
                    source_dir, _glossary = _require_smoke_project_layout(project_dir)
                    chunk_after_run = next_after_run
                    chunk_start = next_start
                    if next_continue_latest:
                        chunk_after_run = _latest_project_run(project_dir)
                    if chunk_after_run is not None:
                        latest = _last_manifest_chapter(chunk_after_run)
                        chunk_start = _next_chapter_id(latest)
                    chunk_chapters = _project_first_chapters(
                        source_dir,
                        first=first,
                        start=chunk_start,
                        until=loop_until,
                    )
                    if chunk_after_run is not None:
                        console.print(f"Continuing after run: {chunk_after_run} (last chapter {_last_manifest_chapter(chunk_after_run)})")
                except ValueError as exc:
                    console.print(f"[red]Error:[/red] {exc}")
                    raise typer.Exit(1) from exc
            smoke_project(
                project_dir=project_dir,
                chapters=",".join(chunk_chapters) if chunk_chapters is not None else chapters,
                first=None if chunk_chapters is not None else first,
                chunks=1,
                start=None if chunk_chapters is not None else next_start,
                until=None,
                after_run=None if chunk_chapters is not None else next_after_run,
                continue_latest=False if chunk_chapters is not None else next_continue_latest,
                out=None,
                translated_dir=translated_dir,
                deepseek=deepseek,
                provider_mode=provider_mode,
                translation_provider=translation_provider,
                judge_provider=judge_provider,
                repair_provider=repair_provider,
                record_cache=record_cache,
                cache_dir=cache_dir,
                model_name=model_name,
                source_char_limit=source_char_limit,
                run_id=None,
                overwrite=overwrite,
                skip_epub=skip_epub,
                allow_source_qa_fail=allow_source_qa_fail,
                allow_review_required=allow_review_required,
                allow_live_provider_fallback=allow_live_provider_fallback,
                report_mode=report_mode,
                practical=practical,
                glossary_pass=glossary_pass,
                normalize_panels=normalize_panels,
                panel_reviewer=panel_reviewer,
                panel_note_prefix=panel_note_prefix,
                write_proof=write_proof,
                write_triage=write_triage,
            )
            if loop_until is None and chunk_index >= chunks:
                break
            if loop_until is not None:
                latest_chapter = chunk_chapters[-1] if chunk_chapters else ""
                if latest_chapter >= loop_until:
                    break
            next_after_run = None
            next_continue_latest = True
            next_start = None
        return
    try:
        source_dir, glossary = _require_smoke_project_layout(project_dir)
        if continue_latest:
            if after_run is not None:
                raise ValueError("--continue-latest cannot be combined with --after-run")
            if first is None:
                raise ValueError("--continue-latest requires --first")
            if start is not None:
                raise ValueError("--continue-latest cannot be combined with --start")
            after_run = _latest_project_run(project_dir)
        if after_run is not None:
            if first is None:
                raise ValueError("--after-run requires --first")
            if start is not None:
                raise ValueError("--after-run cannot be combined with --start")
            last_chapter = _last_manifest_chapter(after_run)
            start = _next_chapter_id(last_chapter)
        else:
            last_chapter = None
        selected_chapters = (
            _project_first_chapters(source_dir, first=first, start=start, until=until)
            if first is not None
            else parse_chapter_selection(chapters)
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    effective_run_id = run_id or _default_smoke_run_id(project_dir, selected_chapters, practical=practical, deepseek=deepseek)
    effective_translated_dir = translated_dir or _project_translated_dir(project_dir, selected_chapters)
    if translated_dir is not None and not translated_dir.is_dir():
        console.print(f"[red]Error:[/red] translated directory does not exist: {translated_dir}")
        raise typer.Exit(1)
    console.print(f"Using source directory: {source_dir}")
    console.print(f"Using glossary: {glossary}")
    if effective_translated_dir is not None:
        console.print(f"Using translated baseline: {effective_translated_dir}")
    else:
        console.print("[yellow]No translated baseline found; using offline cheap-pass baseline.[/yellow]")
    if after_run is not None:
        if continue_latest:
            console.print(f"Continuing after latest matching run: {after_run} (last chapter {last_chapter})")
        else:
            console.print(f"Continuing after run: {after_run} (last chapter {last_chapter})")
    console.print(f"Selected chapters: {','.join(selected_chapters)}")
    console.print(f"Run id: {effective_run_id}")
    smoke_local(
        source_dir=source_dir,
        glossary=glossary,
        chapters=",".join(selected_chapters),
        out=out,
        translated_dir=effective_translated_dir,
        deepseek=deepseek,
        provider_mode=provider_mode,
        translation_provider=translation_provider,
        judge_provider=judge_provider,
        repair_provider=repair_provider,
        record_cache=record_cache,
        cache_dir=cache_dir,
        model_name=model_name,
        source_char_limit=source_char_limit,
        run_id=effective_run_id,
        overwrite=overwrite,
        skip_epub=skip_epub,
        allow_source_qa_fail=allow_source_qa_fail,
        allow_review_required=allow_review_required,
        allow_live_provider_fallback=allow_live_provider_fallback,
        report_mode=report_mode,
        practical=practical,
        glossary_pass=glossary_pass,
        normalize_panels=normalize_panels,
        panel_reviewer=panel_reviewer,
        panel_note_prefix=panel_note_prefix,
        write_proof=write_proof,
        write_triage=write_triage,
    )


@app.command("produce")
def produce(
    project_dir: Path = typer.Argument(..., help="Corpus project directory containing scraped/ and terms/master_glossary.txt."),
    count: int = typer.Option(2, "--count", "-n", help="Number of chapters per production chunk."),
    chapters: str | None = typer.Option(None, "--chapters", help="Explicit chapter range/list, such as 0029,0030 or 29-30."),
    start: str | None = typer.Option(None, "--start", help="Start chapter when selecting by count."),
    until: str | None = typer.Option(None, "--until", help="Keep selecting count-sized chunks through this chapter."),
    provider: Literal["offline", "deepseek"] = typer.Option("offline", "--provider", help="Provider shortcut for the production lane."),
    cheap: int | None = typer.Option(None, "--cheap", help="Limit copied source/baseline chars for cheap live provider probes."),
    cache_dir: Path | None = typer.Option(None, "--cache-dir", help="Live provider cache directory. Defaults to .agentic_cache/produce_deepseek for DeepSeek."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing inferred run directories."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan selected chunks without importing fixtures or running batches."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable production plan for dry runs."),
) -> None:
    try:
        source_dir, glossary = _require_smoke_project_layout(project_dir)
        chunks, latest_run = _produce_chapter_chunks(
            project_dir,
            count=count,
            chapters=chapters,
            start=start,
            until=until,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    deepseek_enabled = provider == "deepseek"
    effective_cache_dir = cache_dir or (Path(".agentic_cache/produce_deepseek") if deepseek_enabled else None)
    chunk_payloads = [
        {
            "chapters": chunk,
            "run_id": _default_smoke_run_id(project_dir, chunk, practical=True, deepseek=deepseek_enabled),
            "translated_dir": str(translated_dir) if (translated_dir := _project_translated_dir(project_dir, chunk)) else None,
            "follow_up_command": _produce_follow_up_command(
                project_dir,
                chunk=chunk,
                provider=provider,
                cheap=cheap,
                cache_dir=effective_cache_dir,
            ),
        }
        for chunk in chunks
    ]
    follow_up_commands = [str(chunk_payload["follow_up_command"]) for chunk_payload in chunk_payloads]
    first_translated_dir = _project_translated_dir(project_dir, chunks[0])
    project_status_payload = _project_status_payload(project_dir)
    payload = {
        "project": project_dir.name,
        "project_dir": str(project_dir),
        "source_dir": str(source_dir),
        "glossary": str(glossary),
        "translated_dir": str(first_translated_dir) if first_translated_dir else None,
        "provider": provider,
        "cache_dir": str(effective_cache_dir) if effective_cache_dir else None,
        "source_char_limit": cheap,
        "latest_run": str(latest_run) if latest_run else None,
        "project_status": project_status_payload,
        "would_mutate": not dry_run,
        "next_chunk": chunk_payloads[0]["chapters"],
        "next_command": follow_up_commands[0],
        "chunks": chunk_payloads,
        "follow_up_commands": follow_up_commands,
    }
    if dry_run:
        if json_output:
            typer.echo(json.dumps(payload, indent=2))
            return
        table = Table(title="Produce Plan")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("Project", payload["project"])
        table.add_row("Provider", provider)
        table.add_row("Source Dir", str(source_dir))
        table.add_row("Glossary", str(glossary))
        table.add_row("Latest Run", payload["latest_run"] or "-")
        table.add_row("Source Chapters", str(project_status_payload["total_source_chapters"]))
        table.add_row("Processed Chapters", str(project_status_payload["processed_unique_chapters"]))
        table.add_row("Latest Chapter", str(project_status_payload["latest_chapter"] or "-"))
        table.add_row("Next Chapter", str(project_status_payload["next_chapter"] or "-"))
        table.add_row("Chunks", "; ".join(",".join(chunk["chapters"]) for chunk in chunk_payloads))
        table.add_row("Run IDs", "; ".join(str(chunk["run_id"]) for chunk in chunk_payloads))
        translated_dirs = [str(chunk["translated_dir"]) for chunk in chunk_payloads if chunk["translated_dir"]]
        table.add_row("Translated Dir", "; ".join(translated_dirs) if translated_dirs else "-")
        table.add_row("Would Mutate", "false")
        console.print(table)
        typer.echo("Source dir(s): " + str(source_dir))
        typer.echo("Glossary: " + str(glossary))
        typer.echo("Run ID(s): " + "; ".join(str(chunk["run_id"]) for chunk in chunk_payloads))
        typer.echo("Translated dir(s): " + ("; ".join(translated_dirs) if translated_dirs else "-"))
        typer.echo("Next command(s):")
        for command in follow_up_commands:
            typer.echo(command)
        return

    if deepseek_enabled:
        try:
            probe = probe_live_provider(
                provider_name="deepseek",
                cache_dir=effective_cache_dir,
                record_cache=True,
                model_name="deepseek-chat",
            )
        except LLMProviderUnavailable as exc:
            if not json_output:
                console.print(f"[yellow]DeepSeek probe failed; production will use fallback if needed:[/yellow] {exc}")
        else:
            if not json_output:
                console.print(f"DeepSeek probe ok: {probe.model}, cache file {probe.cache_file}")

    completed_run_ids: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_payload = chunk_payloads[index - 1]
        translated_dir = _project_translated_dir(project_dir, chunk)
        if not json_output:
            console.print(f"[bold]Produce chunk {index}/{len(chunks)}:[/bold] {','.join(chunk)}")
        smoke_kwargs = {
            "project_dir": project_dir,
            "chapters": ",".join(chunk),
            "first": None,
            "chunks": 1,
            "start": None,
            "until": None,
            "after_run": None,
            "continue_latest": False,
            "out": None,
            "translated_dir": translated_dir,
            "deepseek": deepseek_enabled,
            "provider_mode": "offline",
            "translation_provider": "offline",
            "judge_provider": "offline",
            "repair_provider": "offline",
            "record_cache": False,
            "cache_dir": effective_cache_dir,
            "model_name": "deepseek-chat" if deepseek_enabled else None,
            "source_char_limit": cheap,
            "run_id": str(chunk_payload["run_id"]),
            "overwrite": overwrite,
            "skip_epub": False,
            "allow_source_qa_fail": True,
            "allow_review_required": True,
            "allow_live_provider_fallback": True,
            "report_mode": "excerpt",
            "practical": True,
            "glossary_pass": False,
            "normalize_panels": False,
            "panel_reviewer": DEFAULT_PANEL_REVIEWER,
            "panel_note_prefix": DEFAULT_PANEL_NOTE_PREFIX,
            "write_proof": False,
            "write_triage": True,
        }
        if json_output:
            with contextlib.redirect_stdout(io.StringIO()):
                smoke_project(**smoke_kwargs)
        else:
            smoke_project(**smoke_kwargs)
        completed_run_ids.append(str(chunk_payload["run_id"]))
    review_runs = [str(Path("runs") / run_id) for run_id in completed_run_ids]
    run_summaries, status_summary = _produce_run_summaries(completed_run_ids)
    artifact_summary = _produce_artifact_summary(run_summaries)
    project_status_after = _project_status_payload(project_dir)
    recommended_next_actions = _produce_recommended_next_actions(
        project_dir,
        count=count,
        provider=provider,
        cheap=cheap,
        cache_dir=effective_cache_dir,
        run_summaries=run_summaries,
        status_summary=status_summary,
    )
    if json_output:
        receipt = payload | {
            "would_mutate": True,
            "run_ids": completed_run_ids,
            "review_runs": review_runs,
            "run_summaries": run_summaries,
            "status_summary": status_summary,
            "artifacts": artifact_summary,
            "project_status_after": project_status_after,
            "recommended_next_actions": recommended_next_actions,
        }
        typer.echo(json.dumps(receipt, indent=2))
        return
    typer.echo("Produced run(s): " + "; ".join(completed_run_ids))
    typer.echo("Review run(s): " + "; ".join(review_runs))
    typer.echo(
        "Status: "
        f"{status_summary['packaged']}/{status_summary['total_chapters']} packaged, "
        f"{status_summary['review_required']} review_required, "
        f"{status_summary['failed']} failed, "
        f"{status_summary['incomplete']} incomplete"
    )
    if artifact_summary["txt"]:
        typer.echo("TXT artifact(s): " + "; ".join(artifact_summary["txt"]))
    if artifact_summary["epub"]:
        typer.echo("EPUB artifact(s): " + "; ".join(artifact_summary["epub"]))
    typer.echo(
        "Project progress: "
        f"{project_status_after['processed_unique_chapters']}/{project_status_after['total_source_chapters']} processed, "
        f"latest {project_status_after['latest_chapter'] or '-'}, "
        f"next {project_status_after['next_chapter'] or '-'}"
    )
    if recommended_next_actions:
        typer.echo("Next action(s):")
        for action in recommended_next_actions:
            typer.echo(str(action["command"]))


@app.command("import-local")
def import_local(
    source_dir: Path = typer.Option(..., help="Existing source chapter directory."),
    glossary: Path = typer.Option(..., help="Existing master glossary path."),
    chapter: str = typer.Option("0001", help="Four-digit chapter id."),
    chapters: str | None = typer.Option(None, "--chapters", help="Chapter range/list for batch fixture creation."),
    out: Path = typer.Option(..., help="Output fixture directory under local_fixtures/."),
    translated_dir: Path | None = typer.Option(None, help="Optional existing translated chapter directory to use as cheap baseline."),
    run_batch: bool = typer.Option(False, "--run-batch", help="Immediately run the imported fixture through batch run."),
    provider_mode: Literal["offline", "replay", "live"] = typer.Option("offline", "--provider-mode", help="Provider execution mode when --run-batch is used."),
    translation_provider: str = typer.Option("offline", help="Translation provider name when --run-batch is used."),
    judge_provider: str = typer.Option("offline", help="Judge provider name when --run-batch is used."),
    repair_provider: str = typer.Option("offline", help="Repair provider name when --run-batch is used."),
    record_cache: bool = typer.Option(False, help="Record live provider JSON responses when --run-batch is used."),
    cache_dir: Path | None = typer.Option(None, help="Response cache directory when --run-batch is used."),
    model_name: str | None = typer.Option(None, "--model", help="Explicit OpenAI-compatible model name when --run-batch is used."),
    source_char_limit: int | None = typer.Option(None, "--source-char-limit", help="Copy only the first N source/baseline characters into the fixture."),
    run_id: str | None = typer.Option(None, help="Optional batch run id when --run-batch is used."),
    overwrite: bool = typer.Option(False, help="Overwrite an existing batch run directory when --run-batch is used."),
    force: bool = typer.Option(False, help="Rerun already complete chapters when --run-batch is used."),
    skip_epub: bool = typer.Option(False, help="Skip EPUB generation when --run-batch is used."),
    allow_source_qa_fail: bool = typer.Option(False, help="Continue after source QA errors when --run-batch is used."),
    allow_review_required: bool = typer.Option(False, help="Exit 0 even if the immediate batch run needs review."),
    allow_live_provider_fallback: bool = typer.Option(False, help="If a live translation/judge/repair call fails during --run-batch, continue with offline fallback."),
    report_mode: str | None = typer.Option(None, help="Override report mode for the immediate batch run."),
    write_proof: bool = typer.Option(False, help="Write proof artifacts for the immediate batch run."),
    write_triage: bool = typer.Option(False, "--write-triage", help="Write review queue, glossary gap report, and agent work order for the immediate batch run."),
) -> None:
    selected_chapters = parse_chapter_selection(chapters) if chapters else None
    story_yaml = import_local_fixture(
        source_dir=source_dir,
        glossary=glossary,
        chapter=chapter,
        out=out,
        translated_dir=translated_dir,
        chapters=selected_chapters,
        source_char_limit=source_char_limit,
    )
    console.print(f"Wrote local fixture story config: {story_yaml}")
    if not run_batch:
        return
    try:
        result = run_batch_pipeline(
            story_yaml,
            chapters=selected_chapters or [chapter],
            provider_mode=provider_mode,
            translation_provider_name=translation_provider,
            judge_provider_name=judge_provider,
            repair_provider_name=repair_provider,
            record_cache=record_cache,
            cache_dir=cache_dir,
            model_name=model_name,
            run_id=run_id,
            overwrite=overwrite,
            force=force,
            skip_epub=skip_epub,
            allow_source_qa_fail=allow_source_qa_fail,
            allow_live_provider_fallback=allow_live_provider_fallback,
            report_mode=report_mode,
            write_proof=write_proof,
        )
    except (ValueError, RuntimeError, LLMProviderUnavailable) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_batch_summary(result.manifest_path)
    _write_triage_if_requested(result.run_dir, write_triage=write_triage)
    _exit_if_batch_has_blockers(result.manifest_path, allow_review_required=allow_review_required)


@app.command("open-latest")
def open_latest(runs_dir: Path = Path("runs")) -> None:
    if not runs_dir.exists():
        console.print("[red]No runs found.[/red]")
        raise typer.Exit(code=1)
    runs = sorted([path for path in runs_dir.iterdir() if path.is_dir()])
    if not runs:
        console.print("[red]No runs found.[/red]")
        raise typer.Exit(code=1)
    console.print(runs[-1] / "report.html")


if __name__ == "__main__":
    app()
