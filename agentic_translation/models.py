from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Severity = Literal["info", "warning", "error"]
StageStatus = Literal["pending", "ok", "warn", "fail", "skipped"]


class StoryPaths(BaseModel):
    source_dir: Path
    glossary_path: Path
    prompt_path: Path | None = None
    expected_dir: Path | None = None
    baseline_dir: Path | None = None
    runs_dir: Path = Path("runs")


class TranslationConfig(BaseModel):
    provider: str = "offline"
    model: str = "offline-fixture-v1"
    max_glossary_entries: int = 120
    max_chunk_chars: int = 1800


class QAConfig(BaseModel):
    max_repairs: int = 3


class ReportConfig(BaseModel):
    mode: Literal["full", "excerpt", "redacted"] = "full"
    max_source_chars: int = 1200
    max_translation_chars: int = 1200


class PackageConfig(BaseModel):
    formats: list[str] = Field(default_factory=lambda: ["txt", "epub"])


class BatchConfig(BaseModel):
    default_chapters: str | None = None
    concurrency: int = 1
    skip_existing: bool = True


class TerminologyModelConfig(BaseModel):
    """One explicitly identified terminology voter/evaluator model."""

    model_config = ConfigDict(extra="forbid", strict=True)

    voter_id: Literal["openai", "deepseek"]
    provider: Literal["openai", "deepseek"]
    model: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def _require_matching_identity(self) -> "TerminologyModelConfig":
        if self.voter_id != self.provider:
            raise ValueError("voter_id and provider must identify the same model family")
        if not self.model.strip():
            raise ValueError("model cannot be blank")
        return self


class TerminologyConsensusConfig(BaseModel):
    """Optional dual-model Chinese terminology arbitration settings."""

    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool = False
    openai_model: str | None = None
    deepseek_model: str | None = None
    evaluator_provider: Literal["openai", "deepseek"] = "openai"
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    source_context_chars: int = Field(default=800, ge=100, le=4000)
    translation_context_chars: int = Field(default=800, ge=100, le=4000)

    @model_validator(mode="after")
    def _validate_enabled_models(self) -> "TerminologyConsensusConfig":
        for field_name in ("openai_model", "deepseek_model"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} cannot be blank")
        if self.enabled and (not self.openai_model or not self.deepseek_model):
            raise ValueError(
                "enabled terminology consensus requires explicit openai_model and deepseek_model"
            )
        return self

    @property
    def voters(self) -> list[TerminologyModelConfig]:
        if not self.enabled:
            return []
        # The pair is deliberately fixed: terminology consensus means exactly
        # one OpenAI voter and one DeepSeek voter, never two aliases for one
        # provider or an accidental duplicate.
        return [
            TerminologyModelConfig(
                voter_id="openai", provider="openai", model=self.openai_model or ""
            ),
            TerminologyModelConfig(
                voter_id="deepseek", provider="deepseek", model=self.deepseek_model or ""
            ),
        ]


class AgentConfig(BaseModel):
    max_attempts: int = 3
    cache_required: bool = True
    review_on_unfixed: bool = True
    terminology_consensus: TerminologyConsensusConfig = Field(
        default_factory=TerminologyConsensusConfig
    )


class StoryConfig(BaseModel):
    slug: str
    title: str
    language: str = "zh"
    public_safe: bool = True
    chapter_ids: list[str] = Field(default_factory=lambda: ["0001"])
    paths: StoryPaths
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    qa: QAConfig = Field(default_factory=QAConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    package: PackageConfig = Field(default_factory=PackageConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)


class GlossaryEntry(BaseModel):
    source: str
    target: str
    candidates: list[str] = Field(default_factory=list)
    blocked_variants: list[str] = Field(default_factory=list)


class GlossaryParseResult(BaseModel):
    entries: list[GlossaryEntry]
    warnings: list[str] = Field(default_factory=list)
    blocked_variants: list[str] = Field(default_factory=list)


class QALocation(BaseModel):
    chapter: str
    paragraph_index: int | None = None
    line_index: int | None = None
    snippet: str | None = None


class QAFinding(BaseModel):
    check_id: str
    severity: Severity
    message: str
    location: QALocation
    found: str | None = None
    expected: str | None = None
    suggested_action: str | None = None
    status: Literal["open", "fixed", "ignored"] = "open"
    auto_repairable: bool = False


class QASummary(BaseModel):
    total_findings: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    by_check: dict[str, int] = Field(default_factory=dict)


class QAReport(BaseModel):
    run_id: str
    story_slug: str
    chapter: str
    report_type: Literal["source", "translation"] = "translation"
    findings: list[QAFinding]
    summary: QASummary
    panel_count: int = 0
    score: int = 100


class TranslationCandidate(BaseModel):
    candidate_id: str
    text: str
    source: str = "unknown"
    notes: str | None = None


class CandidateScore(BaseModel):
    compliance: "ComplianceCandidateScore"
    quality: "QualityCandidateScore | None" = None
    aggregate: float = 0.0
    notes: str = ""


class ComplianceCandidateScore(BaseModel):
    residue_free: int = Field(ge=1, le=10)
    glossary_consistency: int = Field(ge=1, le=10)
    panel_preservation: int = Field(ge=1, le=10)
    prompt_safety: int = Field(ge=1, le=10)
    readability: int = Field(ge=1, le=10)
    notes: str = ""


class QualityCandidateScore(BaseModel):
    faithfulness: int = Field(ge=1, le=10)
    fluency: int = Field(ge=1, le=10)
    rationale: str = ""


class JudgeVote(BaseModel):
    judge_id: str
    winner: str
    scores: dict[str, CandidateScore]
    rationale: str = ""


class EnsembleDecision(BaseModel):
    selected_candidate_id: str
    votes: list[JudgeVote]
    aggregate_scores: dict[str, float]
    disagreement: float
    requires_human_review: bool


class RepairPatch(BaseModel):
    patch_id: str
    patch_type: Literal["replace_span", "replace_paragraph"]
    chapter: str
    paragraph_index: int | None = None
    old_text: str
    new_text: str
    reason: str
    source_finding_check_id: str | None = None
    accepted: bool = False


class RepairDecision(BaseModel):
    finding_check_id: str
    strategy: Literal["rule", "candidate_selection", "human_review", "none"]
    selected_candidate_id: str | None = None
    reason: str
    requires_human_review: bool = False


class PatchAttempt(BaseModel):
    finding_check_id: str
    strategy: Literal["rule", "candidate_selection", "human_review", "none"]
    before_score: int
    after_score: int | None = None
    before_findings: int
    after_findings: int | None = None
    accepted: bool = False
    reason: str
    patch: RepairPatch | None = None


class TxtArtifactAudit(BaseModel):
    chapter_markers: int = 0
    contains_chinese: bool = False
    contains_prompt_leakage: bool = False


class EpubArtifactAudit(BaseModel):
    xhtml_chapters: int = 0
    contains_chinese: bool = False
    contains_prompt_leakage: bool = False


class ArtifactQAReport(BaseModel):
    expected_chapters: int
    txt: TxtArtifactAudit = Field(default_factory=TxtArtifactAudit)
    epub: EpubArtifactAudit | None = None
    passed: bool = False
    failures: list[str] = Field(default_factory=list)


class StageRecord(BaseModel):
    name: str
    status: StageStatus
    started_at: datetime | None = None
    ended_at: datetime | None = None
    message: str | None = None
    artifact_paths: list[str] = Field(default_factory=list)


class EvalMetrics(BaseModel):
    mode: str
    residual_chinese: int = 0
    chinese_punctuation: int = 0
    glossary_violations: int = 0
    panel_mismatches: int = 0
    prompt_leakage: int = 0
    total_findings: int = 0
    score: int = 100


class BenchAblationStep(BaseModel):
    step_id: str
    label: str
    pattern: str
    compliance_score: int | None = None
    finding_count: int | None = None
    artifact_passed: bool | None = None
    note: str = ""


class BenchAblationSummary(BaseModel):
    score_gain: int
    finding_reduction: int


class BenchAblationReport(BaseModel):
    note: str = "Compliance score is not semantic translation quality."
    steps: list[BenchAblationStep]
    summary: BenchAblationSummary


class ProviderLabel(BaseModel):
    provider: str
    model: str


class ProviderCallRecord(BaseModel):
    role: str
    namespace: str
    provider: str
    model: str | None = None
    payload_sha256: str
    response_sha256: str
    cache_file: str
    cache_hit: bool = False


class ToolAgentRunRecord(BaseModel):
    """Compact, batch-safe evidence for one chapter tool-agent episode.

    This model intentionally stores paths and scalar episode metadata rather
    than an ``AgentEpisode``.  ``agent_models`` imports this module, so keeping
    the compact record independent avoids a circular import while still
    allowing callers to retain the full episode JSON artifact on disk.
    """

    episode_path: Path
    report_path: Path
    html_path: Path
    final_status: str
    step_count: int = Field(default=0, ge=0)
    initial_findings: int = Field(default=0, ge=0)
    final_findings: int = Field(default=0, ge=0)
    accepted_patch_count: int = Field(default=0, ge=0)
    rejected_patch_count: int = Field(default=0, ge=0)
    final_text_sha256: str | None = None
    provider_calls: list[ProviderCallRecord] = Field(default_factory=list)

    @property
    def markdown_report_path(self) -> Path:
        """Compatibility label for the Markdown report artifact."""

        return self.report_path

    @property
    def html_report_path(self) -> Path:
        """Compatibility label for the HTML report artifact."""

        return self.html_path


class RunManifest(BaseModel):
    schema_version: str = "0.1"
    run_id: str
    story_slug: str
    title: str
    chapter_ids: list[str]
    mode: str = "offline"
    public_safe: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    inputs: dict[str, str] = Field(default_factory=dict)
    providers: dict[str, ProviderLabel] = Field(default_factory=dict)
    provider_calls: list[ProviderCallRecord] = Field(default_factory=list)
    qa: dict[str, int] = Field(default_factory=dict)
    artifact_qa: ArtifactQAReport | None = None
    chapters: dict[str, dict[str, int | str]] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    stages: list[StageRecord] = Field(default_factory=list)
    eval_metrics: list[EvalMetrics] = Field(default_factory=list)
    bench_ablation: BenchAblationReport | None = None


class PipelineResult(BaseModel):
    run_dir: Path
    report_path: Path
    qa_source: QAReport
    qa_baseline: QAReport
    qa_glossary: QAReport
    qa_final: QAReport
    tool_agent: ToolAgentRunRecord | None = None
    artifact_qa: ArtifactQAReport | None = None
    repair_decisions: list[RepairDecision] = Field(default_factory=list)
    patch_attempts: list[PatchAttempt] = Field(default_factory=list)
    provider_calls: list[ProviderCallRecord] = Field(default_factory=list)
    provider_failure_messages: list[str] = Field(default_factory=list)
    baseline_metrics: EvalMetrics
    glossary_metrics: EvalMetrics
    final_metrics: EvalMetrics
    bench_ablation: BenchAblationReport | None = None


BatchChapterStatus = Literal[
    "pending",
    "running",
    "translated",
    "qa_warn",
    "repaired",
    "review_required",
    "packaged",
    "failed",
    "skipped",
]


class AgentAttempt(BaseModel):
    attempt_id: str
    chapter: str
    provider: str
    model: str
    action: str
    status: StageStatus
    message: str | None = None


class BaselineComparison(BaseModel):
    baseline_path: str
    baseline_sha256: str
    final_sha256: str
    changed: bool


class ManualReviewRecord(BaseModel):
    chapter: str
    reviewer: str = "human"
    note: str
    status_before: BatchChapterStatus
    status_after: BatchChapterStatus
    qa_score_after: int | None = None
    qa_findings_after: int | None = None
    artifact_qa_passed: bool | None = None
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BatchChapterRun(BaseModel):
    chapter: str
    status: BatchChapterStatus = "pending"
    source_path: str | None = None
    chapter_run_dir: str | None = None
    final_path: str | None = None
    report_path: str | None = None
    final_score: int | None = None
    final_findings: int | None = None
    error: str | None = None
    baseline_comparison: BaselineComparison | None = None
    attempts: list[AgentAttempt] = Field(default_factory=list)
    manual_reviews: list[ManualReviewRecord] = Field(default_factory=list)
    repair_decisions: list[RepairDecision] = Field(default_factory=list)
    patch_attempts: list[PatchAttempt] = Field(default_factory=list)
    provider_calls: list[ProviderCallRecord] = Field(default_factory=list)
    tool_agent_episode_path: str | None = None
    tool_agent_report_path: str | None = None
    tool_agent_html_report_path: str | None = None
    tool_agent_final_status: str | None = None
    tool_agent_steps: int = 0
    tool_agent_initial_findings: int = 0
    tool_agent_final_findings: int = 0
    tool_agent_accepted_patches: int = 0
    tool_agent_rejected_patches: int = 0
    tool_agent_final_text_sha256: str | None = None


class BatchSummary(BaseModel):
    total_chapters: int = 0
    pending: int = 0
    packaged: int = 0
    review_required: int = 0
    failed: int = 0
    skipped: int = 0
    incomplete: int = 0


class BatchRunConfig(BaseModel):
    provider_mode: str = "offline"
    translation_provider: str = "offline"
    judge_provider: str = "offline"
    repair_provider: str = "offline"
    record_cache: bool = False
    cache_dir: str | None = None
    model_name: str | None = None
    allow_live_provider_fallback: bool = False
    tool_agent_enabled: bool = False
    terminology_consensus: TerminologyConsensusConfig = Field(
        default_factory=TerminologyConsensusConfig
    )


class BatchManifest(BaseModel):
    schema_version: str = "0.1"
    run_id: str
    story_slug: str
    title: str
    story_yaml: str
    run_dir: str
    replay_source_run_dir: str | None = None
    mode: str = "offline"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    providers: dict[str, ProviderLabel] = Field(default_factory=dict)
    run_config: BatchRunConfig | None = None
    chapters: dict[str, BatchChapterRun] = Field(default_factory=dict)
    summary: BatchSummary = Field(default_factory=BatchSummary)
    artifacts: dict[str, str] = Field(default_factory=dict)
    artifact_qa: ArtifactQAReport | None = None

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        story_slug: str,
        title: str,
        story_yaml: Path,
        chapters: list[str],
        mode: str,
        providers: dict[str, ProviderLabel],
        run_dir: Path,
        replay_source_run_dir: str | None = None,
        run_config: BatchRunConfig | None = None,
    ) -> "BatchManifest":
        manifest = cls(
            run_id=run_id,
            story_slug=story_slug,
            title=title,
            story_yaml=str(story_yaml),
            run_dir=str(run_dir),
            mode=mode,
            replay_source_run_dir=replay_source_run_dir,
            providers=providers,
            run_config=run_config,
            chapters={chapter: BatchChapterRun(chapter=chapter) for chapter in chapters},
        )
        manifest.refresh_summary()
        return manifest

    def refresh_summary(self) -> None:
        counts = {status: 0 for status in BatchChapterStatus.__args__}  # type: ignore[attr-defined]
        for chapter in self.chapters.values():
            counts[chapter.status] = counts.get(chapter.status, 0) + 1
        terminal_statuses = {"packaged", "review_required", "failed", "skipped"}
        incomplete = sum(1 for chapter in self.chapters.values() if chapter.status not in terminal_statuses)
        self.summary = BatchSummary(
            total_chapters=len(self.chapters),
            pending=counts.get("pending", 0),
            packaged=counts.get("packaged", 0),
            review_required=counts.get("review_required", 0),
            failed=counts.get("failed", 0),
            skipped=counts.get("skipped", 0),
            incomplete=incomplete,
        )


class BatchPipelineResult(BaseModel):
    run_dir: Path
    manifest_path: Path
    manifest: BatchManifest
    artifact_qa: ArtifactQAReport | None = None


class ManualTextReplacementResult(BaseModel):
    run_id: str
    run_dir: str
    chapter: str
    final_path: str
    old_text: str
    new_text: str
    occurrence_count: int
    refresh_only: bool = False
    reviewer: str | None = None
    note: str | None = None
    status_after: BatchChapterStatus
    final_score_after: int | None = None
    final_findings_after: int | None = None
    summary_after: BatchSummary


class PanelNormalizationItem(BaseModel):
    chapter: str
    status: Literal["normalized", "skipped"]
    reason: str = ""
    replacement_count: int = 0
    status_after: BatchChapterStatus | None = None
    final_score_after: int | None = None
    final_findings_after: int | None = None


class PanelNormalizationResult(BaseModel):
    run_id: str
    run_dir: str
    items: list[PanelNormalizationItem] = Field(default_factory=list)
    normalized_count: int = 0
    skipped_count: int = 0
    summary_after: BatchSummary


class BatchInspectionBlocker(BaseModel):
    blocker_type: Literal["incomplete", "failed", "review_required", "artifact_qa"]
    message: str
    chapter: str | None = None
    status: str | None = None


class AgenticEvidence(BaseModel):
    mode: str
    configured_model_roles: list[str] = Field(default_factory=list)
    observed_agentic_roles: list[str] = Field(default_factory=list)
    candidate_selection_repairs: int = 0
    model_backed_patch_attempts: int = 0
    cache_dir: str | None = None
    cache_available: bool = False
    cache_entries: int = 0
    cache_namespaces: dict[str, int] = Field(default_factory=dict)
    cache_required_namespaces: list[str] = Field(default_factory=list)
    cache_missing_namespaces: list[str] = Field(default_factory=list)
    cache_integrity_passed: bool = False
    cache_valid_entries: int = 0
    cache_invalid_entries: int = 0
    cache_integrity_issues: list[str] = Field(default_factory=list)
    provider_call_records: int = 0
    cache_verified_call_records: int = 0
    cache_missing_call_records: list[str] = Field(default_factory=list)
    cache_metadata_mismatches: list[str] = Field(default_factory=list)
    verified_candidate_selection_records: int = 0
    candidate_selection_mismatches: list[str] = Field(default_factory=list)
    verified_repair_patch_records: int = 0
    repair_patch_mismatches: list[str] = Field(default_factory=list)
    replay_cache_ready: bool = False
    agentic_claim_supported: bool = False
    reason: str


class ToolAgentEvidence(BaseModel):
    """Proof data for the model-directed tool-agent episode.

    This is intentionally separate from :class:`AgenticEvidence`, which
    describes the older candidate-selection workflow.  The tool-agent proof
    validates the persisted episode, typed action trajectory, deterministic
    patch observations, and cache-backed replay independently.
    """

    applicable: bool = False
    episodes_observed: int = 0
    verified_episodes: int = 0
    observed_actions: int = 0
    verified_actions: int = 0
    verified_patch_records: int = 0
    verified_cache_records: int = 0
    replay_cache_ready: bool = False
    action_sequence_matches: bool | None = None
    patch_decisions_match: bool | None = None
    final_text_matches: bool | None = None
    final_qa_matches: bool | None = None
    final_status_matches: bool | None = None
    final_text_sha256: dict[str, str] = Field(default_factory=dict)
    final_qa_sha256: dict[str, str] = Field(default_factory=dict)
    mismatches: list[str] = Field(default_factory=list)
    proof_ready: bool = False


class ProviderFailureSummary(BaseModel):
    chapter: str
    role: Literal["translation", "judge", "repair", "unknown"] = "unknown"
    provider: str | None = None
    model: str | None = None
    reason: str
    fallback_used: bool = False


class BatchInspectionReport(BaseModel):
    run_id: str
    story_slug: str
    run_dir: str
    ready_for_delivery: bool
    blocker_count: int
    blockers: list[BatchInspectionBlocker] = Field(default_factory=list)
    summary: BatchSummary
    artifact_qa: ArtifactQAReport | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    agentic_evidence: AgenticEvidence
    provider_failures: list[ProviderFailureSummary] = Field(default_factory=list)
    run_config: BatchRunConfig | None = None


class BatchProofReport(BaseModel):
    run_id: str
    story_slug: str
    run_dir: str
    proof_passed: bool
    gates: dict[str, bool] = Field(default_factory=dict)
    blockers: list[str] = Field(default_factory=list)
    inspection: BatchInspectionReport
    tool_agent_evidence: ToolAgentEvidence = Field(default_factory=ToolAgentEvidence)


class BatchLiveProofResult(BaseModel):
    story_yaml: str
    chapters: list[str]
    proof_passed: bool
    live_result: BatchPipelineResult
    live_proof: BatchProofReport
    replay_result: BatchPipelineResult | None = None
    replay_proof: BatchProofReport | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)


class CacheIndexEntry(BaseModel):
    namespace: str
    cache_file: str
    payload_sha256: str
    response_sha256: str
    provider: str | None = None
    model: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CacheIntegrityIssue(BaseModel):
    namespace: str | None = None
    cache_file: str
    issue_type: Literal[
        "missing_file",
        "invalid_json",
        "response_digest_mismatch",
        "cache_file_mismatch",
    ]
    message: str


class CacheIndexReport(BaseModel):
    cache_dir: str
    total_entries: int = 0
    by_namespace: dict[str, int] = Field(default_factory=dict)
    entries: list[CacheIndexEntry] = Field(default_factory=list)
    valid_entries: int = 0
    invalid_entries: int = 0
    integrity_passed: bool = True
    integrity_issues: list[CacheIntegrityIssue] = Field(default_factory=list)


class ReviewQueueItem(BaseModel):
    chapter: str
    chapter_status: BatchChapterStatus
    check_id: str
    severity: Severity
    message: str
    found: str | None = None
    expected: str | None = None
    paragraph_index: int | None = None
    line_index: int | None = None
    snippet: str | None = None
    source_context: str | None = None
    final_context: str | None = None
    report_path: str | None = None
    final_path: str | None = None


class ReviewQueueSummary(BaseModel):
    total_items: int = 0
    by_check: dict[str, int] = Field(default_factory=dict)
    by_chapter: dict[str, int] = Field(default_factory=dict)
    chapters: list[str] = Field(default_factory=list)
    chapter_selection: str = ""


class ReviewQueue(BaseModel):
    run_id: str
    story_slug: str
    run_dir: str
    items: list[ReviewQueueItem] = Field(default_factory=list)
    summary: ReviewQueueSummary = Field(default_factory=ReviewQueueSummary)


class PanelLine(BaseModel):
    index: int
    line_number: int
    text: str


class PanelComparisonRow(BaseModel):
    index: int
    source: PanelLine | None = None
    final: PanelLine | None = None
    status: Literal["paired", "missing_final", "extra_final"]


class PanelChapterReport(BaseModel):
    chapter: str
    chapter_status: BatchChapterStatus
    source_path: str | None = None
    final_path: str | None = None
    source_count: int = 0
    final_count: int = 0
    count_delta: int = 0
    rows: list[PanelComparisonRow] = Field(default_factory=list)


class PanelReportSummary(BaseModel):
    total_chapters: int = 0
    mismatch_chapters: int = 0
    total_source_panels: int = 0
    total_final_panels: int = 0
    chapter_selection: str = ""


class PanelReport(BaseModel):
    run_id: str
    story_slug: str
    run_dir: str
    chapters: list[PanelChapterReport] = Field(default_factory=list)
    summary: PanelReportSummary = Field(default_factory=PanelReportSummary)


class GlossaryGapOccurrence(BaseModel):
    chapter: str
    chapter_status: BatchChapterStatus
    severity: Severity
    message: str
    paragraph_index: int | None = None
    line_index: int | None = None
    snippet: str | None = None
    source_context: str | None = None
    final_context: str | None = None
    report_path: str | None = None
    final_path: str | None = None


class GlossaryGapItem(BaseModel):
    found: str | None = None
    expected: str | None = None
    count: int = 0
    chapters: list[str] = Field(default_factory=list)
    chapter_selection: str = ""
    suggested_action: str
    suggested_aliases: list[str] = Field(default_factory=list)
    occurrences: list[GlossaryGapOccurrence] = Field(default_factory=list)


class GlossaryGapSummary(BaseModel):
    total_occurrences: int = 0
    term_count: int = 0
    by_chapter: dict[str, int] = Field(default_factory=dict)
    chapters: list[str] = Field(default_factory=list)
    chapter_selection: str = ""


class GlossaryGapReport(BaseModel):
    run_id: str
    story_slug: str
    run_dir: str
    gaps: list[GlossaryGapItem] = Field(default_factory=list)
    summary: GlossaryGapSummary = Field(default_factory=GlossaryGapSummary)


GlossaryUpdateAction = Literal["add_candidates", "manual_review"]


class GlossaryUpdatePlanItem(BaseModel):
    found: str | None = None
    expected: str | None = None
    action: GlossaryUpdateAction
    count: int = 0
    chapters: list[str] = Field(default_factory=list)
    chapter_selection: str = ""
    suggested_aliases: list[str] = Field(default_factory=list)
    suggested_line: str | None = None
    note: str = ""
    occurrences: list[GlossaryGapOccurrence] = Field(default_factory=list)


class GlossaryUpdatePlanSummary(BaseModel):
    total_items: int = 0
    add_candidates_count: int = 0
    manual_review_count: int = 0
    chapters: list[str] = Field(default_factory=list)
    chapter_selection: str = ""


class GlossaryUpdatePlan(BaseModel):
    run_id: str
    story_slug: str
    run_dir: str
    items: list[GlossaryUpdatePlanItem] = Field(default_factory=list)
    summary: GlossaryUpdatePlanSummary = Field(default_factory=GlossaryUpdatePlanSummary)


GlossaryUpdateApplicationStatus = Literal["updated", "appended", "skipped", "manual_review"]


class GlossaryUpdateApplicationItem(BaseModel):
    found: str | None = None
    expected: str | None = None
    aliases: list[str] = Field(default_factory=list)
    status: GlossaryUpdateApplicationStatus
    line_number: int | None = None
    before_line: str | None = None
    after_line: str | None = None
    reason: str = ""


class GlossaryUpdateApplicationSummary(BaseModel):
    total_items: int = 0
    changed_count: int = 0
    updated_count: int = 0
    appended_count: int = 0
    skipped_count: int = 0
    manual_review_count: int = 0


class GlossaryUpdateApplication(BaseModel):
    run_id: str
    story_slug: str
    run_dir: str
    glossary_path: str
    backup_path: str | None = None
    dry_run: bool = True
    items: list[GlossaryUpdateApplicationItem] = Field(default_factory=list)
    summary: GlossaryUpdateApplicationSummary = Field(default_factory=GlossaryUpdateApplicationSummary)


class GlossaryUpdatePassResult(BaseModel):
    run_id: str
    run_dir: str
    dry_run: bool = True
    chapters: list[str] = Field(default_factory=list)
    rerun_started: bool = False
    message: str = ""
    application: GlossaryUpdateApplication
    before_summary: "BatchSummary"
    after_summary: "BatchSummary | None" = None


WorkOrderAction = Literal[
    "failed_chapter_retry",
    "glossary_triage",
    "live_candidate_selection",
    "manual_review",
]


class AgentWorkOrderItem(BaseModel):
    chapter: str
    action: WorkOrderAction
    check_id: str
    severity: Severity
    reason: str
    found: str | None = None
    expected: str | None = None
    final_path: str | None = None
    report_path: str | None = None
    source_context: str | None = None
    final_context: str | None = None


class AgentWorkOrderSummary(BaseModel):
    total_items: int = 0
    by_action: dict[str, int] = Field(default_factory=dict)
    by_chapter: dict[str, int] = Field(default_factory=dict)
    chapters: list[str] = Field(default_factory=list)
    chapter_selection: str = ""
    live_retry_chapters: list[str] = Field(default_factory=list)
    live_retry_selection: str = ""
    glossary_chapters: list[str] = Field(default_factory=list)
    glossary_selection: str = ""
    manual_review_chapters: list[str] = Field(default_factory=list)
    manual_review_selection: str = ""


class AgentWorkOrder(BaseModel):
    run_id: str
    story_slug: str
    run_dir: str
    items: list[AgentWorkOrderItem] = Field(default_factory=list)
    summary: AgentWorkOrderSummary = Field(default_factory=AgentWorkOrderSummary)
    commands: dict[str, str] = Field(default_factory=dict)


class ManualEditPlanItem(BaseModel):
    chapter: str
    action: WorkOrderAction
    check_id: str
    final_path: str
    report_path: str | None = None
    found: str | None = None
    expected: str | None = None
    instruction: str
    source_context: str | None = None
    final_context: str | None = None


class ManualEditPlanSummary(BaseModel):
    total_items: int = 0
    by_file: dict[str, int] = Field(default_factory=dict)
    by_action: dict[str, int] = Field(default_factory=dict)
    chapters: list[str] = Field(default_factory=list)
    chapter_selection: str = ""


class ManualEditPlan(BaseModel):
    run_id: str
    story_slug: str
    run_dir: str
    items: list[ManualEditPlanItem] = Field(default_factory=list)
    summary: ManualEditPlanSummary = Field(default_factory=ManualEditPlanSummary)


class AgentWorkOrderExecutionPreview(BaseModel):
    run_id: str
    story_slug: str
    run_dir: str
    action: str
    provider_mode: str
    translation_provider: str
    judge_provider: str
    repair_provider: str
    record_cache: bool = False
    cache_dir: str | None = None
    model_name: str | None = None
    tool_agent_enabled: bool = False
    chapters: list[str] = Field(default_factory=list)
    dry_run: bool = True
    would_mutate: bool = False
    command: str = ""
    dry_run_command: str = ""
    execution_command: str = ""
    recommended_next_action: Literal["execute_live_retry", "fix_preflight"] = "fix_preflight"
    recommended_command: str = ""
    preflight_blockers: list[str] = Field(default_factory=list)
    preflight_passed: bool = False
    preflight_status_counts: dict[str, int] = Field(default_factory=dict)
    preflight_checks: list[dict[str, object]] = Field(default_factory=list)
