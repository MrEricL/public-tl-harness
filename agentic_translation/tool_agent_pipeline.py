"""Reusable chapter-level tool-agent repair phase."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from .agent_models import AgentEpisode
from .agent_provider import (
    AgentActionProvider,
    AgentToolSchemaVersion,
    BASE_TOOL_SCHEMA_VERSION,
    TERMINOLOGY_TOOL_SCHEMA_VERSION,
)
from .agent_repair import run_repair_episode
from .agent_report import render_agent_episode_html, render_agent_episode_markdown
from .models import (
    GlossaryParseResult,
    ProviderCallRecord,
    QAReport,
    TerminologyConsensusConfig,
    ToolAgentRunRecord,
)
from .terminology import TerminologyResolver
from .terminology_provider import LLMTerminologyEvaluatorProvider, LLMTerminologyVoterProvider


def build_terminology_resolver(
    config: TerminologyConsensusConfig | None,
    *,
    provider_mode: str,
    cache_dir: Path | None,
    record_cache: bool,
) -> TerminologyResolver | None:
    """Build the exact two-model terminology resolver for one agent episode.

    The factory is intentionally small and cache-backed: one independently
    configured OpenAI voter, one DeepSeek voter, and an evaluator using the
    configured provider's model.  Disabled/absent configuration returns
    ``None`` so the historical v1 action contract remains untouched.
    """

    if config is None or not config.enabled:
        return None
    if not config.openai_model or not config.deepseek_model:
        raise ValueError("enabled terminology consensus requires explicit OpenAI and DeepSeek models")
    voters = [
        LLMTerminologyVoterProvider(
            voter_id="openai",
            provider_name="openai",
            provider_mode=provider_mode,
            cache_dir=cache_dir,
            record_cache=record_cache,
            model_name=config.openai_model,
        ),
        LLMTerminologyVoterProvider(
            voter_id="deepseek",
            provider_name="deepseek",
            provider_mode=provider_mode,
            cache_dir=cache_dir,
            record_cache=record_cache,
            model_name=config.deepseek_model,
        ),
    ]
    evaluator_model = (
        config.openai_model
        if config.evaluator_provider == "openai"
        else config.deepseek_model
    )
    evaluator = LLMTerminologyEvaluatorProvider(
        provider_name=config.evaluator_provider,
        provider_mode=provider_mode,
        cache_dir=cache_dir,
        record_cache=record_cache,
        model_name=evaluator_model,
    )
    return TerminologyResolver(
        voters=voters,
        evaluator=evaluator,
        confidence_threshold=config.confidence_threshold,
    )


@dataclass
class ToolAgentPhaseResult:
    """Full in-memory result plus compact persisted metadata for one phase."""

    episode: AgentEpisode
    final_text: str
    final_qa: QAReport
    run_record: ToolAgentRunRecord

    @property
    def episode_path(self) -> Path:
        return self.run_record.episode_path

    @property
    def markdown_report_path(self) -> Path:
        return self.run_record.report_path

    @property
    def html_report_path(self) -> Path:
        return self.run_record.html_path

    @property
    def final_status(self) -> str:
        return self.run_record.final_status

    @property
    def step_count(self) -> int:
        return self.run_record.step_count

    @property
    def initial_findings(self) -> int:
        return self.run_record.initial_findings

    @property
    def final_findings(self) -> int:
        return self.run_record.final_findings

    @property
    def accepted_patch_count(self) -> int:
        return self.run_record.accepted_patch_count

    @property
    def rejected_patch_count(self) -> int:
        return self.run_record.rejected_patch_count

    @property
    def provider_calls(self) -> list[ProviderCallRecord]:
        return self.run_record.provider_calls


def _provider_calls(
    provider: AgentActionProvider,
    *,
    start_index: int = 0,
    auxiliary_calls: list[ProviderCallRecord] | None = None,
    episode: AgentEpisode | None = None,
) -> list[ProviderCallRecord]:
    """Copy valid provider records without retaining provider-owned state."""

    records = getattr(provider, "call_records", [])
    if not isinstance(records, list):
        return []
    if episode is not None:
        ordered_records: list[ProviderCallRecord] = []
        for step in episode.steps:
            if step.provider_call is not None:
                ordered_records.append(step.provider_call)
            ordered_records.extend(step.auxiliary_provider_calls)
        # Preserve any valid provider record not attached to a step (for
        # provider implementations that append evidence asynchronously), but
        # keep it after the persisted chronological episode evidence.
        ordered_records.extend(records[start_index:])
    else:
        ordered_records = list(records[start_index:]) + list(auxiliary_calls or [])
    calls: list[ProviderCallRecord] = []
    for record in ordered_records:
        try:
            parsed = ProviderCallRecord.model_validate(record)
        except (TypeError, ValueError):
            continue
        if parsed not in calls:
            calls.append(parsed)
    return calls


def run_tool_agent_phase(
    provider: AgentActionProvider,
    run_dir: Path,
    source_text: str,
    translated_text: str,
    glossary: GlossaryParseResult,
    run_id: str,
    story_slug: str,
    chapter: str,
    provider_mode: str,
    cache_dir: Path | None = None,
    record_cache: bool = False,
    story_title: str | None = None,
    terminology_resolver: TerminologyResolver | None = None,
    terminology_config: TerminologyConsensusConfig | None = None,
    terminology_source_context_chars: int = 800,
    terminology_translation_context_chars: int = 800,
    tool_schema_version: AgentToolSchemaVersion = BASE_TOOL_SCHEMA_VERSION,
) -> ToolAgentPhaseResult:
    """Run and persist the bounded repair agent for one chapter.

    The helper owns only the chapter's ``agent_repair`` artifacts.  It delegates
    action execution, persistence, and QA mutation to ``run_repair_episode``
    and delegates report formatting to the existing report renderers.
    """

    run_root = Path(run_dir).expanduser()
    if run_root.is_symlink():
        raise ValueError(f"Refusing symlinked tool-agent run directory: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    run_root = run_root.resolve()
    agent_dir = run_root / "agent_repair"
    if agent_dir.is_symlink():
        raise ValueError(f"Refusing symlinked tool-agent artifact directory: {agent_dir}")
    if agent_dir.exists() and not agent_dir.is_dir():
        raise ValueError(f"Tool-agent artifact path is not a directory: {agent_dir}")
    resolved_agent_dir = agent_dir.resolve()
    if not resolved_agent_dir.is_relative_to(run_root):
        raise ValueError(f"Tool-agent artifacts escape run directory: {resolved_agent_dir}")
    agent_dir.mkdir(parents=True, exist_ok=True)
    provider_records = getattr(provider, "call_records", [])
    provider_call_start = len(provider_records) if isinstance(provider_records, list) else 0
    episode_path = agent_dir / "agent_episode.json"
    markdown_path = agent_dir / "report.md"
    html_path = agent_dir / "report.html"

    if terminology_config is not None and terminology_config.enabled:
        # Direct phase callers get the versioned terminology tool contract as
        # well; legacy callers that provide a resolver explicitly can still
        # opt into v2 by passing the schema version themselves.
        if tool_schema_version == BASE_TOOL_SCHEMA_VERSION:
            tool_schema_version = TERMINOLOGY_TOOL_SCHEMA_VERSION
    if terminology_resolver is None:
        terminology_resolver = build_terminology_resolver(
            terminology_config,
            provider_mode=provider_mode,
            cache_dir=cache_dir,
            record_cache=record_cache,
        )

    result = run_repair_episode(
        provider=provider,
        episode_path=episode_path,
        source_text=source_text,
        translated_text=translated_text,
        glossary=glossary,
        run_id=run_id,
        story_slug=story_slug,
        chapter=chapter,
        provider_mode=provider_mode,
        max_steps=5,
        max_patch_attempts=2,
        terminology_resolver=terminology_resolver,
        terminology_source_context_chars=terminology_source_context_chars,
        terminology_translation_context_chars=terminology_translation_context_chars,
        tool_schema_version=tool_schema_version,
    )
    calls = _provider_calls(
        provider,
        start_index=provider_call_start,
        episode=result.episode,
    )
    artifact_paths = {
        "agent_episode": str(episode_path.relative_to(run_root)),
        "repair_report": str(markdown_path.relative_to(run_root)),
        "report_html": str(html_path.relative_to(run_root)),
    }
    markdown_path.write_text(
        render_agent_episode_markdown(
            result.episode,
            story_title=story_title,
            source_text=source_text,
            translation_text=translated_text,
            final_text=result.final_text,
            artifact_paths=artifact_paths,
            call_records=calls,
        ),
        encoding="utf-8",
    )
    render_agent_episode_html(
        html_path,
        result.episode,
        story_title=story_title,
        source_text=source_text,
        translation_text=translated_text,
        final_text=result.final_text,
        artifact_paths=artifact_paths,
        call_records=calls,
    )

    run_record = ToolAgentRunRecord(
        episode_path=episode_path,
        report_path=markdown_path,
        html_path=html_path,
        final_status=result.episode.final_status or "failed",
        step_count=len(result.episode.steps),
        initial_findings=result.episode.initial_qa.summary.total_findings,
        final_findings=result.final_qa.summary.total_findings,
        accepted_patch_count=sum(
            step.observation.kind == "patch_accepted" for step in result.episode.steps
        ),
        rejected_patch_count=sum(
            step.observation.kind == "patch_rejected" for step in result.episode.steps
        ),
        final_text_sha256=hashlib.sha256(result.final_text.encode("utf-8")).hexdigest(),
        provider_calls=calls,
    )
    return ToolAgentPhaseResult(
        episode=result.episode,
        final_text=result.final_text,
        final_qa=result.final_qa,
        run_record=run_record,
    )


__all__ = ["ToolAgentPhaseResult", "run_tool_agent_phase"]
