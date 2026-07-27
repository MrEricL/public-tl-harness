from __future__ import annotations

from dataclasses import dataclass, field
import json
import shutil
from pathlib import Path

from .agent_provider import (
    AgentActionProvider,
    BASE_TOOL_SCHEMA_VERSION,
    LLMAgentActionProvider,
    TERMINOLOGY_TOOL_SCHEMA_VERSION,
)
from .ensemble import blind_and_shuffle, generate_repair_candidates
from .evals import metrics_from_qa
from .glossary import load_glossary
from .manifest import build_manifest
from .models import (
    ArtifactQAReport,
    BenchAblationReport,
    BenchAblationStep,
    BenchAblationSummary,
    EnsembleDecision,
    GlossaryParseResult,
    PatchAttempt,
    PipelineResult,
    ProviderCallRecord,
    ProviderLabel,
    QAReport,
    RepairDecision,
    RepairPatch,
    StageRecord,
    TerminologyConsensusConfig,
    TranslationCandidate,
)
from .package import build_epub, build_txt, verify_epub_artifact, verify_txt_artifact
from .providers import JudgeProvider, RepairProvider
from .providers_llm import LLMProviderUnavailable, openai_compatible_provider_names
from .providers_offline import OfflineJudgeProvider, OfflineRepairProvider, OfflineTranslationProvider
from .qa import run_source_qa, run_translation_qa
from .repair import apply_patch, prioritized_repairable_findings, route_repair_strategy, validate_patch_improves_qa
from .report import render_report
from .story import load_story_config, make_run_id, prepare_run_dir, read_chapter
from .text import trim_for_report
from .trace import TraceWriter
from .translate import get_judge_provider, get_repair_provider, get_translation_provider
from .tool_agent_pipeline import run_tool_agent_phase
from .utils import sha256_file


def write_json(path: Path, model: object) -> None:
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")  # type: ignore[attr-defined]


def _write_run_notes(
    path: Path,
    *,
    story_slug: str,
    chapter: str,
    mode: str,
    patches: list[RepairPatch],
    artifact_qa: ArtifactQAReport,
    final_qa: QAReport,
) -> None:
    patch_lines = "\n".join(
        f"- {patch.patch_type}: {patch.reason} ({'accepted' if patch.accepted else 'rejected'})"
        for patch in patches
    ) or "- No patches applied."
    path.write_text(
        "\n".join(
            [
                "# Run Notes",
                "",
                "## Story",
                f"{story_slug} / chapter {chapter}",
                "",
                "## Mode",
                mode,
                "",
                "## Key Decisions",
                "- Ran source QA before translation.",
                "- Compared baseline and glossary-controlled translation.",
                "- Used prioritized repair queue with QA veto.",
                patch_lines,
                "",
                "## Final QA",
                f"- Findings: {final_qa.summary.total_findings}",
                f"- Score: {final_qa.score}",
                "",
                "## Artifact QA",
                f"- Passed: {artifact_qa.passed}",
                f"- TXT Chinese residue: {artifact_qa.txt.contains_chinese}",
                f"- TXT prompt leakage: {artifact_qa.txt.contains_prompt_leakage}",
                f"- EPUB Chinese residue: {artifact_qa.epub.contains_chinese if artifact_qa.epub else 'skipped'}",
                f"- EPUB prompt leakage: {artifact_qa.epub.contains_prompt_leakage if artifact_qa.epub else 'skipped'}",
                f"- Failures: {', '.join(artifact_qa.failures) if artifact_qa.failures else 'none'}",
                "",
                "## Notes For Future Agents",
                "- Run notes are not bureaucracy; they preserve source quirks, term decisions, and provider behavior.",
                "- Offline providers are deterministic. Live providers should use the same typed provider contracts.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_run_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _limit_fixture_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    if max_chars < 1:
        raise ValueError("source_char_limit must be at least 1")
    limited = text[:max_chars].rstrip()
    paragraph_break = limited.rfind("\n\n")
    if paragraph_break > max_chars // 2:
        limited = limited[:paragraph_break].rstrip()
    return limited + "\n"


def _trace_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _qa_status(report: QAReport) -> str:
    if report.summary.error_count:
        return "fail"
    if report.summary.total_findings:
        return "warn"
    return "ok"


def _validate_provider_mode(
    *,
    provider_mode: str,
    translation_provider_name: str,
    judge_provider_name: str,
    repair_provider_name: str,
) -> None:
    if provider_mode not in {"offline", "replay", "live"}:
        raise ValueError(f"Unsupported provider mode: {provider_mode}")
    if provider_mode == "offline":
        return
    provider_names = {translation_provider_name, judge_provider_name, repair_provider_name}
    if not any(name != "offline" for name in provider_names):
        raise ValueError(f"{provider_mode} provider mode requires at least one non-offline provider.")
    if provider_mode == "live" and not openai_compatible_provider_names(provider_names):
        raise ValueError("live provider mode requires at least one live provider such as openai or deepseek.")


def _artifact_qa_status(artifact_qa: ArtifactQAReport) -> str:
    return "ok" if artifact_qa.passed else "fail"


def _provider_call_records(*providers: object) -> list[ProviderCallRecord]:
    records: list[ProviderCallRecord] = []
    for provider in providers:
        for record in getattr(provider, "call_records", []):
            records.append(ProviderCallRecord.model_validate(record))
    return records


def _build_bench_ablation(
    *,
    baseline_metrics,
    glossary_metrics,
    final_metrics,
    qa_baseline: QAReport,
    qa_glossary: QAReport,
    qa_final: QAReport,
    patch_attempts: list[PatchAttempt],
    artifact_qa: ArtifactQAReport,
) -> BenchAblationReport:
    steps = [
        BenchAblationStep(
            step_id="cheap_baseline",
            label="cheap baseline",
            pattern="single cheap pass",
            compliance_score=baseline_metrics.score,
            finding_count=qa_baseline.summary.total_findings,
            note="Untrusted first output before glossary canon or repair.",
        ),
        BenchAblationStep(
            step_id="glossary_canon",
            label="glossary canon",
            pattern="prompt chaining",
            compliance_score=glossary_metrics.score,
            finding_count=qa_glossary.summary.total_findings,
            note="Code-produced canon/block transform; no hidden polished fixture.",
        ),
        BenchAblationStep(
            step_id="router_patch_loop",
            label="router + patch loop",
            pattern="routing + evaluator-optimizer",
            compliance_score=final_metrics.score,
            finding_count=qa_final.summary.total_findings,
            note=f"{len(patch_attempts)} patch attempt(s), accepted only after QA re-check.",
        ),
        BenchAblationStep(
            step_id="artifact_gate",
            label="artifact gate",
            pattern="packaging scorer",
            compliance_score=final_metrics.score,
            finding_count=len(artifact_qa.failures),
            artifact_passed=artifact_qa.passed,
            note="TXT/EPUB checked after packaging.",
        ),
    ]
    return BenchAblationReport(
        steps=steps,
        summary=BenchAblationSummary(
            score_gain=final_metrics.score - baseline_metrics.score,
            finding_reduction=(
                qa_baseline.summary.total_findings - qa_final.summary.total_findings
            ),
        ),
    )


@dataclass
class RepairQueueResult:
    text: str
    qa_report: QAReport
    patches: list[RepairPatch] = field(default_factory=list)
    ensemble_decision: EnsembleDecision | None = None
    candidates: list[TranslationCandidate] = field(default_factory=list)
    decisions: list[RepairDecision] = field(default_factory=list)
    attempts: list[PatchAttempt] = field(default_factory=list)


@dataclass
class LiveProviderFallbackState:
    translation_reason: str | None = None
    judge_reason: str | None = None
    repair_reason: str | None = None


def _build_artifact_qa(
    *,
    txt_path: Path,
    epub_path: Path | None,
    expected_chapters: int,
) -> ArtifactQAReport:
    txt_audit = verify_txt_artifact(txt_path)
    epub_audit = verify_epub_artifact(epub_path) if epub_path else None
    failures: list[str] = []
    if txt_audit["contains_chinese"]:
        failures.append("TXT contains Chinese residue.")
    if txt_audit["contains_prompt_leakage"]:
        failures.append("TXT contains prompt leakage.")
    if txt_audit["chapter_markers"] != expected_chapters:
        failures.append("TXT chapter marker count does not match expected chapter count.")
    if epub_audit:
        if epub_audit["contains_chinese"]:
            failures.append("EPUB contains Chinese residue.")
        if epub_audit["contains_prompt_leakage"]:
            failures.append("EPUB contains prompt leakage.")
        if epub_audit["xhtml_chapters"] != expected_chapters:
            failures.append("EPUB XHTML chapter count does not match expected chapter count.")
    return ArtifactQAReport(
        expected_chapters=expected_chapters,
        txt=txt_audit,
        epub=epub_audit,
        passed=not failures,
        failures=failures,
    )


def _apply_repair_queue(
    *,
    run_id: str,
    story_slug: str,
    chapter: str,
    source_text: str,
    initial_text: str,
    glossary: GlossaryParseResult,
    repair_provider: RepairProvider,
    judge_provider: JudgeProvider,
    provider_mode: str,
    allow_live_provider_fallback: bool,
    live_provider_fallback_state: LiveProviderFallbackState | None,
    seed: int,
    max_repairs: int,
) -> RepairQueueResult:
    current_text = initial_text
    patches: list[RepairPatch] = []
    decisions: list[RepairDecision] = []
    attempts: list[PatchAttempt] = []
    last_decision: EnsembleDecision | None = None
    last_candidates: list[TranslationCandidate] = []
    rule_repair_provider = OfflineRepairProvider()
    rejected_finding_signatures: set[tuple[str, str | None, str | None]] = set()

    for _ in range(max_repairs):
        before = run_translation_qa(
            run_id=run_id,
            story_slug=story_slug,
            chapter=chapter,
            source_text=source_text,
            translated_text=current_text,
            glossary=glossary,
        )
        findings = [
            finding
            for finding in prioritized_repairable_findings(before.findings)
            if (finding.check_id, finding.found, finding.expected) not in rejected_finding_signatures
        ]
        if not findings:
            return RepairQueueResult(current_text, before, patches, last_decision, last_candidates, decisions, attempts)
        finding = findings[0]
        strategy = route_repair_strategy(finding, provider_mode=provider_mode)
        repair_decision = RepairDecision(
            finding_check_id=finding.check_id,
            strategy=strategy,
            reason=f"Router selected {strategy} for {finding.check_id}.",
            requires_human_review=strategy == "human_review",
        )
        decisions.append(repair_decision)
        attempt = PatchAttempt(
            finding_check_id=finding.check_id,
            strategy=strategy,
            before_score=before.score,
            before_findings=before.summary.total_findings,
            reason=repair_decision.reason,
        )
        if strategy == "human_review":
            attempts.append(attempt)
            return RepairQueueResult(current_text, before, patches, last_decision, last_candidates, decisions, attempts)

        blinded = None
        decision = None
        fallback_note = ""
        if strategy == "candidate_selection":
            candidates = generate_repair_candidates(
                translated_text=current_text,
                finding=finding,
                glossary=glossary,
            )
            blinded = blind_and_shuffle(candidates, seed + len(patches))
            if allow_live_provider_fallback and live_provider_fallback_state and live_provider_fallback_state.judge_reason:
                decision = OfflineJudgeProvider().judge(
                    source_text=source_text,
                    candidates=blinded,
                    glossary=glossary,
                    seed=seed,
                )
                fallback_note = (
                    "Skipped live judge because of previous live judge provider failure "
                    f"({live_provider_fallback_state.judge_reason}); fell back to offline judge."
                )
                repair_decision.reason += " " + fallback_note
                attempt.reason = repair_decision.reason
            else:
                try:
                    decision = judge_provider.judge(
                        source_text=source_text,
                        candidates=blinded,
                        glossary=glossary,
                        seed=seed,
                    )
                except LLMProviderUnavailable as exc:
                    if not allow_live_provider_fallback:
                        raise
                    if live_provider_fallback_state is not None:
                        live_provider_fallback_state.judge_reason = str(exc)
                    decision = OfflineJudgeProvider().judge(
                        source_text=source_text,
                        candidates=blinded,
                        glossary=glossary,
                        seed=seed,
                    )
                    fallback_note = f"Live judge provider failed ({exc}); fell back to offline judge."
                    repair_decision.reason += " " + fallback_note
                    attempt.reason = repair_decision.reason
            repair_decision.selected_candidate_id = decision.selected_candidate_id
            patch_provider = repair_provider
        else:
            patch_provider = rule_repair_provider

        if (
            strategy == "candidate_selection"
            and allow_live_provider_fallback
            and live_provider_fallback_state
            and live_provider_fallback_state.repair_reason
        ):
            patch = rule_repair_provider.propose_patch(
                chapter=chapter,
                source_text=source_text,
                translation_text=current_text,
                finding=finding,
                glossary=glossary,
                ensemble_decision=decision,
                candidates=blinded,
            )
            extra_note = (
                "Skipped live repair because of previous live repair provider failure "
                f"({live_provider_fallback_state.repair_reason}); fell back to offline patcher."
            )
            fallback_note = f"{fallback_note} {extra_note}".strip()
            repair_decision.reason += " " + extra_note
            attempt.reason = repair_decision.reason
        else:
            try:
                patch = patch_provider.propose_patch(
                    chapter=chapter,
                    source_text=source_text,
                    translation_text=current_text,
                    finding=finding,
                    glossary=glossary,
                    ensemble_decision=decision,
                    candidates=blinded,
                )
            except LLMProviderUnavailable as exc:
                if not allow_live_provider_fallback:
                    raise
                if live_provider_fallback_state is not None:
                    live_provider_fallback_state.repair_reason = str(exc)
                patch = rule_repair_provider.propose_patch(
                    chapter=chapter,
                    source_text=source_text,
                    translation_text=current_text,
                    finding=finding,
                    glossary=glossary,
                    ensemble_decision=decision,
                    candidates=blinded,
                )
                extra_note = f"Live repair provider failed ({exc}); fell back to offline patcher."
                fallback_note = f"{fallback_note} {extra_note}".strip()
                repair_decision.reason += " " + extra_note
                attempt.reason = repair_decision.reason
        if patch is None:
            attempt.reason = "No patch proposed; human review required."
            attempts.append(attempt)
            return RepairQueueResult(current_text, before, patches, decision, blinded or last_candidates, decisions, attempts)
        try:
            candidate_text = apply_patch(current_text, patch)
        except ValueError as exc:
            patch.accepted = False
            patches.append(patch)
            attempt.patch = patch
            attempt.after_score = before.score
            attempt.after_findings = before.summary.total_findings
            attempt.accepted = False
            attempt.reason = f"Rejected because patch could not be applied: {exc}"
            if fallback_note:
                attempt.reason += " Fallback: " + fallback_note
            attempts.append(attempt)
            rejected_finding_signatures.add((finding.check_id, finding.found, finding.expected))
            continue
        after = run_translation_qa(
            run_id=run_id,
            story_slug=story_slug,
            chapter=chapter,
            source_text=source_text,
            translated_text=candidate_text,
            glossary=glossary,
        )
        patch.accepted = validate_patch_improves_qa(before_report=before, after_report=after)
        patches.append(patch)
        attempt.patch = patch
        attempt.after_score = after.score
        attempt.after_findings = after.summary.total_findings
        attempt.accepted = patch.accepted
        attempt.reason = "Accepted because compliance QA improved." if patch.accepted else "Rejected because compliance QA did not improve."
        if fallback_note:
            attempt.reason += " Fallback: " + fallback_note
        attempts.append(attempt)
        if decision is not None:
            last_decision = decision
        last_candidates = blinded or last_candidates
        if not patch.accepted:
            rejected_finding_signatures.add((finding.check_id, finding.found, finding.expected))
            continue
        current_text = candidate_text

    final_report = run_translation_qa(
        run_id=run_id,
        story_slug=story_slug,
        chapter=chapter,
        source_text=source_text,
        translated_text=current_text,
        glossary=glossary,
    )
    return RepairQueueResult(current_text, final_report, patches, last_decision, last_candidates, decisions, attempts)


def run_demo_pipeline(
    story_yaml: Path,
    *,
    chapter_override: str | None = None,
    provider_mode: str = "offline",
    offline: bool | None = None,
    translation_provider_name: str = "offline",
    judge_provider_name: str = "offline",
    repair_provider_name: str = "offline",
    record_cache: bool = False,
    cache_dir: Path | None = None,
    model_name: str | None = None,
    allow_live_provider_fallback: bool = False,
    live_provider_fallback_state: LiveProviderFallbackState | None = None,
    run_id: str | None = None,
    seed: int = 7,
    overwrite: bool = False,
    skip_epub: bool = False,
    allow_source_qa_fail: bool = False,
    runs_dir: Path | None = None,
    report_mode: str | None = None,
    tool_agent_enabled: bool = False,
    tool_agent_provider: AgentActionProvider | None = None,
    terminology_consensus: TerminologyConsensusConfig | None = None,
) -> PipelineResult:
    story = load_story_config(story_yaml)
    if chapter_override is not None:
        story = story.model_copy(update={"chapter_ids": [chapter_override]})
    chapter = story.chapter_ids[0]
    effective_terminology_consensus = (
        story.agent.terminology_consensus
        if terminology_consensus is None
        else terminology_consensus
    )
    if offline is not None:
        provider_mode = "offline" if offline else "live"
    if provider_mode == "offline":
        translation_provider_name = judge_provider_name = repair_provider_name = "offline"
    _validate_provider_mode(
        provider_mode=provider_mode,
        translation_provider_name=translation_provider_name,
        judge_provider_name=judge_provider_name,
        repair_provider_name=repair_provider_name,
    )
    if report_mode:
        story = story.model_copy(update={"report": story.report.model_copy(update={"mode": report_mode})})

    run_id = run_id or make_run_id(story.slug)
    target_runs_dir = runs_dir or story.paths.runs_dir
    run_dir = prepare_run_dir(target_runs_dir, run_id, overwrite=overwrite)
    trace = TraceWriter(run_dir / "trace.jsonl")

    source_run_dir = run_dir / "source"
    baseline_dir = run_dir / "translated_baseline"
    glossary_dir = run_dir / "translated_glossary"
    final_dir = run_dir / "translated_final"
    review_dir = run_dir / "review"

    with trace.stage("load_story", story=story.slug):
        source_text = read_chapter(story.paths.source_dir, chapter)
        source_run_dir.mkdir(parents=True, exist_ok=True)
        _write_run_text(source_run_dir / f"{chapter}.txt", source_text)
        glossary = load_glossary(story.paths.glossary_path)

    with trace.stage("source_qa", chapter=chapter) as stage:
        qa_source = run_source_qa(
            run_id=run_id,
            story_slug=story.slug,
            chapter=chapter,
            source_text=source_text,
        )
        write_json(run_dir / "qa_source.json", qa_source)
        stage.finish(status=_qa_status(qa_source), findings=qa_source.summary.total_findings, score=qa_source.score)
        if qa_source.summary.error_count and not allow_source_qa_fail:
            raise RuntimeError(
                f"Source QA failed with {qa_source.summary.error_count} error(s). "
                "Use --allow-source-qa-fail only for local debugging."
            )

    translation_provider = get_translation_provider(
        translation_provider_name,
        provider_mode=provider_mode,
        cache_dir=cache_dir,
        record_cache=record_cache,
        model_name=model_name,
    )
    judge_provider = get_judge_provider(
        judge_provider_name,
        provider_mode=provider_mode,
        cache_dir=cache_dir,
        record_cache=record_cache,
        model_name=model_name,
    )
    repair_provider = get_repair_provider(
        repair_provider_name,
        provider_mode=provider_mode,
        cache_dir=cache_dir,
        record_cache=record_cache,
        model_name=model_name,
    )
    provider_failure_messages: list[str] = []

    live_translation_provider = translation_provider.provider_name != "offline"
    with trace.stage("translate_baseline", provider=translation_provider.provider_name) as stage:
        translation_mode = "glossary" if live_translation_provider else "baseline"
        fallback_reason = None
        if live_translation_provider and allow_live_provider_fallback and live_provider_fallback_state and live_provider_fallback_state.translation_reason:
            fallback_reason = (
                "Skipped live translation because of previous live translation provider failure "
                f"({live_provider_fallback_state.translation_reason}); used offline translation."
            )
            baseline_text = OfflineTranslationProvider().translate(
                source_text,
                story=story,
                glossary=glossary,
                mode="glossary",
            )
        else:
            try:
                baseline_text = translation_provider.translate(
                    source_text,
                    story=story,
                    glossary=glossary,
                    mode=translation_mode,
                )
            except LLMProviderUnavailable as exc:
                if not (live_translation_provider and allow_live_provider_fallback):
                    raise
                if live_provider_fallback_state is not None:
                    live_provider_fallback_state.translation_reason = str(exc)
                fallback_reason = f"Live translation provider failed ({exc}); fell back to offline translation."
                baseline_text = OfflineTranslationProvider().translate(
                    source_text,
                    story=story,
                    glossary=glossary,
                    mode="glossary",
                )
        _write_run_text(baseline_dir / f"{chapter}.txt", baseline_text)
        if fallback_reason:
            provider_failure_messages.append(fallback_reason)
            stage.finish(
                status="warn",
                mode="glossary",
                fallback_reason=fallback_reason,
                fallback_provider="offline",
                reused_for="translate_glossary",
            )
        elif live_translation_provider:
            stage.finish(mode="glossary", reused_for="translate_glossary")

    with trace.stage("translate_glossary", provider=translation_provider.provider_name) as stage:
        if live_translation_provider:
            glossary_text = baseline_text
            stage.finish(status="skipped", reused_from="translate_baseline")
        else:
            glossary_text = translation_provider.translate(
                source_text,
                story=story,
                glossary=glossary,
                mode="glossary",
            )
        _write_run_text(glossary_dir / f"{chapter}.txt", glossary_text)

    with trace.stage("qa_baseline", chapter=chapter) as stage:
        qa_baseline = run_translation_qa(
            run_id=run_id,
            story_slug=story.slug,
            chapter=chapter,
            source_text=source_text,
            translated_text=baseline_text,
            glossary=glossary,
        )
        write_json(run_dir / "qa_baseline.json", qa_baseline)
        stage.finish(status=_qa_status(qa_baseline), findings=qa_baseline.summary.total_findings, score=qa_baseline.score)

    with trace.stage("qa_glossary", chapter=chapter) as stage:
        qa_glossary = run_translation_qa(
            run_id=run_id,
            story_slug=story.slug,
            chapter=chapter,
            source_text=source_text,
            translated_text=glossary_text,
            glossary=glossary,
        )
        write_json(run_dir / "qa_glossary.json", qa_glossary)
        stage.finish(status=_qa_status(qa_glossary), findings=qa_glossary.summary.total_findings, score=qa_glossary.score)

    with trace.stage("repair", max_repairs=story.qa.max_repairs) as stage:
        repair_result = _apply_repair_queue(
            run_id=run_id,
            story_slug=story.slug,
            chapter=chapter,
            source_text=source_text,
            initial_text=glossary_text,
            glossary=glossary,
            repair_provider=repair_provider,
            judge_provider=judge_provider,
            provider_mode=provider_mode,
            allow_live_provider_fallback=allow_live_provider_fallback,
            live_provider_fallback_state=live_provider_fallback_state,
            seed=seed,
            max_repairs=story.qa.max_repairs,
        )
        final_text = repair_result.text
        qa_final = repair_result.qa_report
        patches = repair_result.patches
        ensemble_decision = repair_result.ensemble_decision
        candidates = repair_result.candidates
        repair_decisions = repair_result.decisions
        patch_attempts = repair_result.attempts
        stage.finish(status="ok", patches=len(patches), accepted=sum(1 for patch in patches if patch.accepted))

    tool_agent_result = None
    if tool_agent_enabled:
        with trace.stage("tool_agent", chapter=chapter) as stage:
            if not qa_final.summary.total_findings:
                stage.finish(status="skipped", reason="Fixed-repair QA is clean.")
            else:
                action_provider = tool_agent_provider or LLMAgentActionProvider(
                    provider_mode=provider_mode,
                    provider_name=repair_provider_name,
                    model_name=model_name,
                    cache_dir=cache_dir,
                    record_cache=record_cache,
                )
                tool_agent_result = run_tool_agent_phase(
                    provider=action_provider,
                    run_dir=run_dir,
                    source_text=source_text,
                    translated_text=final_text,
                    glossary=glossary,
                    run_id=run_id,
                    story_slug=story.slug,
                    chapter=chapter,
                    provider_mode=provider_mode,
                    cache_dir=cache_dir,
                    record_cache=record_cache,
                    story_title=story.title,
                    terminology_config=effective_terminology_consensus,
                    terminology_source_context_chars=effective_terminology_consensus.source_context_chars,
                    terminology_translation_context_chars=effective_terminology_consensus.translation_context_chars,
                    tool_schema_version=(
                        TERMINOLOGY_TOOL_SCHEMA_VERSION
                        if effective_terminology_consensus.enabled
                        else BASE_TOOL_SCHEMA_VERSION
                    ),
                )
                final_text = tool_agent_result.final_text
                qa_final = tool_agent_result.final_qa
                stage.finish(
                    status=_qa_status(qa_final),
                    final_status=tool_agent_result.final_status,
                    findings=qa_final.summary.total_findings,
                    steps=tool_agent_result.step_count,
                )

    # Persist the text selected by fixed repairs or the optional tool agent
    # before any downstream QA, metrics, package, or report artifact is built.
    _write_run_text(final_dir / f"{chapter}.txt", final_text)

    with trace.stage("qa_final", chapter=chapter) as stage:
        write_json(run_dir / "qa_final.json", qa_final)
        stage.finish(status=_qa_status(qa_final), findings=qa_final.summary.total_findings, score=qa_final.score)

    baseline_metrics = metrics_from_qa("baseline", qa_baseline)
    glossary_metrics = metrics_from_qa("glossary", qa_glossary)
    final_metrics = metrics_from_qa("final", qa_final)

    with trace.stage("package_txt"):
        txt_path = build_txt(
            output_path=review_dir / f"{story.slug}_{chapter}.txt",
            chapter=chapter,
            translated_text=final_text,
        )
    epub_path = review_dir / f"{story.slug}_{chapter}.epub"
    if not skip_epub:
        with trace.stage("package_epub"):
            build_epub(
                output_path=epub_path,
                story_title=story.title,
                chapter=chapter,
                translated_text=final_text,
            )
    else:
        epub_path = None  # type: ignore[assignment]

    with trace.stage("artifact_qa", chapter=chapter) as stage:
        artifact_qa = _build_artifact_qa(
            txt_path=txt_path,
            epub_path=epub_path if not skip_epub else None,
            expected_chapters=1,
        )
        write_json(run_dir / "artifact_qa.json", artifact_qa)
        stage.finish(status=_artifact_qa_status(artifact_qa), failures=len(artifact_qa.failures))
        known_unresolved_failures = {
            "TXT contains Chinese residue.",
            "TXT contains prompt leakage.",
            "EPUB contains Chinese residue.",
            "EPUB contains prompt leakage.",
        }
        unresolved_agent_review = (
            tool_agent_result is not None
            and qa_final.summary.total_findings > 0
            and tool_agent_result.final_status in {"escalated", "budget_exhausted", "verified"}
            and bool(artifact_qa.failures)
            and all(failure in known_unresolved_failures for failure in artifact_qa.failures)
        )
        if not artifact_qa.passed and not unresolved_agent_review:
            raise RuntimeError("Artifact QA failed: " + "; ".join(artifact_qa.failures))

    with trace.stage("bench_ablation", chapter=chapter) as stage:
        bench_ablation = _build_bench_ablation(
            baseline_metrics=baseline_metrics,
            glossary_metrics=glossary_metrics,
            final_metrics=final_metrics,
            qa_baseline=qa_baseline,
            qa_glossary=qa_glossary,
            qa_final=qa_final,
            patch_attempts=patch_attempts,
            artifact_qa=artifact_qa,
        )
        write_json(run_dir / "bench_ablation.json", bench_ablation)
        stage.finish(
            score_gain=bench_ablation.summary.score_gain,
            finding_reduction=bench_ablation.summary.finding_reduction,
        )

    artifacts = {
        "report_html": "report.html",
        "txt": f"review/{txt_path.name}",
        "trace": "trace.jsonl",
        "run_notes": "run_notes.md",
        "bench_ablation": "bench_ablation.json",
        "qa_source": "qa_source.json",
        "qa_baseline": "qa_baseline.json",
        "qa_glossary": "qa_glossary.json",
        "qa_final": "qa_final.json",
    }
    if not skip_epub:
        artifacts["epub"] = f"review/{epub_path.name}"
    if tool_agent_result is not None:
        run_root = run_dir.resolve()
        artifacts.update(
            {
                "agent_episode": str(tool_agent_result.episode_path.relative_to(run_root)),
                "agent_report": str(tool_agent_result.markdown_report_path.relative_to(run_root)),
                "agent_report_html": str(tool_agent_result.html_report_path.relative_to(run_root)),
            }
        )

    inputs = {
        "story_yaml": str(story_yaml),
        "source_sha256": sha256_file(story.paths.source_dir / f"{chapter}.txt"),
        "glossary_sha256": sha256_file(story.paths.glossary_path),
    }
    if story.paths.prompt_path and story.paths.prompt_path.exists():
        inputs["prompt_sha256"] = sha256_file(story.paths.prompt_path)

    trace_records = _trace_records(run_dir / "trace.jsonl")
    provider_calls = _provider_call_records(translation_provider, judge_provider, repair_provider)
    if tool_agent_result is not None:
        provider_calls.extend(tool_agent_result.provider_calls)
    stages = [
        StageRecord(
            name=str(record["stage"]),
            status=str(record["status"]),  # type: ignore[arg-type]
            message=f"{record.get('duration_ms', 0)} ms",
        )
        for record in trace_records
    ]
    providers = {
        "translation": ProviderLabel(provider=translation_provider.provider_name, model=translation_provider.model_name),
        "judge": ProviderLabel(provider=judge_provider.provider_name, model=judge_provider.model_name),
        "repair": ProviderLabel(provider=repair_provider.provider_name, model=repair_provider.model_name),
    }
    if tool_agent_result is not None:
        providers["agent"] = ProviderLabel(
            provider=tool_agent_result.provider_calls[0].provider
            if tool_agent_result.provider_calls
            else tool_agent_result.episode.provider,
            model=tool_agent_result.provider_calls[0].model
            if tool_agent_result.provider_calls
            else tool_agent_result.episode.model,
        )
    manifest = build_manifest(
        run_id=run_id,
        story_slug=story.slug,
        title=story.title,
        chapter_ids=story.chapter_ids,
        mode=provider_mode,
        public_safe=story.public_safe,
        inputs=inputs,
        providers=providers,
        provider_calls=provider_calls,
        qa={
            "source_findings": qa_source.summary.total_findings,
            "baseline_findings": qa_baseline.summary.total_findings,
            "glossary_findings": qa_glossary.summary.total_findings,
            "final_findings": qa_final.summary.total_findings,
        },
        artifact_qa=artifact_qa,
        chapters={
            chapter: {
                "source_findings": qa_source.summary.total_findings,
                "baseline_findings": qa_baseline.summary.total_findings,
                "glossary_findings": qa_glossary.summary.total_findings,
                "final_findings": qa_final.summary.total_findings,
                "final_score": qa_final.score,
            }
        },
        artifacts=artifacts,
        stages=stages,
        eval_metrics=[baseline_metrics, glossary_metrics, final_metrics],
        bench_ablation=bench_ablation,
    )
    write_json(run_dir / "manifest.json", manifest)
    _write_run_notes(
        run_dir / "run_notes.md",
        story_slug=story.slug,
        chapter=chapter,
        mode=manifest.mode,
        patches=patches,
        artifact_qa=artifact_qa,
        final_qa=qa_final,
    )

    if ensemble_decision is None:
        from .models import EnsembleDecision

        ensemble_decision = EnsembleDecision(
            selected_candidate_id="none",
            votes=[],
            aggregate_scores={},
            disagreement=0.0,
            requires_human_review=False,
        )

    context = {
        "manifest": manifest,
        "story": story,
        "chapter": chapter,
        "source_text": trim_for_report(source_text, mode=story.report.mode, max_chars=story.report.max_source_chars),
        "baseline_text": trim_for_report(baseline_text, mode=story.report.mode, max_chars=story.report.max_translation_chars),
        "glossary_text": trim_for_report(glossary_text, mode=story.report.mode, max_chars=story.report.max_translation_chars),
        "final_text": trim_for_report(final_text, mode=story.report.mode, max_chars=story.report.max_translation_chars),
        "qa_reports": {
            "source": qa_source,
            "baseline": qa_baseline,
            "glossary": qa_glossary,
            "final": qa_final,
        },
        "ensemble": ensemble_decision,
        "candidates": candidates,
        "repair_decisions": repair_decisions,
        "patches": patches,
        "patch_attempts": patch_attempts,
        "artifact_qa": artifact_qa,
        "glossary_entries": glossary.entries,
        "eval_metrics": [baseline_metrics, glossary_metrics, final_metrics],
        "bench_ablation": bench_ablation,
        "baseline": baseline_metrics,
        "glossary": glossary_metrics,
        "final": final_metrics,
        "trace": trace_records,
        "tool_agent": tool_agent_result.run_record if tool_agent_result is not None else None,
    }
    report_path = render_report(
        output_path=run_dir / "report.html",
        context=context,
    )

    return PipelineResult(
        run_dir=run_dir,
        report_path=report_path,
        qa_source=qa_source,
        qa_baseline=qa_baseline,
        qa_glossary=qa_glossary,
        qa_final=qa_final,
        tool_agent=tool_agent_result.run_record if tool_agent_result is not None else None,
        artifact_qa=artifact_qa,
        repair_decisions=repair_decisions,
        patch_attempts=patch_attempts,
        provider_calls=provider_calls,
        provider_failure_messages=provider_failure_messages,
        baseline_metrics=baseline_metrics,
        glossary_metrics=glossary_metrics,
        final_metrics=final_metrics,
        bench_ablation=bench_ablation,
    )


def import_local_fixture(
    *,
    source_dir: Path,
    glossary: Path,
    chapter: str,
    out: Path,
    translated_dir: Path | None = None,
    chapters: list[str] | None = None,
    source_char_limit: int | None = None,
) -> Path:
    if source_char_limit is not None and source_char_limit < 1:
        raise ValueError("source_char_limit must be at least 1")
    selected_chapters = chapters or [chapter]
    out.mkdir(parents=True, exist_ok=True)
    (out / "source").mkdir(exist_ok=True)
    (out / "terms").mkdir(exist_ok=True)
    for selected_chapter in selected_chapters:
        source_text = (source_dir / f"{selected_chapter}.txt").read_text(encoding="utf-8")
        (out / "source" / f"{selected_chapter}.txt").write_text(
            _limit_fixture_text(source_text, source_char_limit),
            encoding="utf-8",
        )
    shutil.copy2(glossary, out / "terms" / "master_glossary.txt")
    if translated_dir is None:
        candidates = sorted(
            path
            for path in source_dir.parent.glob("translated*")
            if path.is_dir() and all((path / f"{selected_chapter}.txt").exists() for selected_chapter in selected_chapters)
        )
        translated_dir = candidates[-1] if candidates else None
    expected_dir_line = ""
    baseline_dir_line = ""
    if translated_dir and len(selected_chapters) == 1 and (translated_dir / f"{selected_chapters[0]}.txt").exists():
        (out / "expected").mkdir(exist_ok=True)
        translated_text = (translated_dir / f"{selected_chapters[0]}.txt").read_text(encoding="utf-8")
        (out / "expected" / "dirty_translation.txt").write_text(
            _limit_fixture_text(translated_text, source_char_limit),
            encoding="utf-8",
        )
        expected_dir_line = f'  expected_dir: "{out / "expected"}"\n'
    elif translated_dir:
        (out / "baseline").mkdir(exist_ok=True)
        for selected_chapter in selected_chapters:
            source_translation = translated_dir / f"{selected_chapter}.txt"
            if source_translation.exists():
                translated_text = source_translation.read_text(encoding="utf-8")
                (out / "baseline" / f"{selected_chapter}.txt").write_text(
                    _limit_fixture_text(translated_text, source_char_limit),
                    encoding="utf-8",
                )
        baseline_dir_line = f'  baseline_dir: "{out / "baseline"}"\n'
    chapter_lines = "\n".join(f'  - "{selected_chapter}"' for selected_chapter in selected_chapters)
    story_yaml = out / "story.yaml"
    story_yaml.write_text(
        f"""slug: "{out.name}"
title: "{out.name}"
language: zh
public_safe: false
chapter_ids:
{chapter_lines}
paths:
  source_dir: "{out / "source"}"
  glossary_path: "{out / "terms" / "master_glossary.txt"}"
{expected_dir_line.rstrip()}
{baseline_dir_line.rstrip()}
  runs_dir: runs
translation:
  provider: offline
  model: offline-fixture-v1
report:
  mode: excerpt
  max_source_chars: 1200
  max_translation_chars: 1200
""",
        encoding="utf-8",
    )
    return story_yaml
