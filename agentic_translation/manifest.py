from __future__ import annotations

from .models import (
    ArtifactQAReport,
    BenchAblationReport,
    EvalMetrics,
    ProviderCallRecord,
    ProviderLabel,
    RunManifest,
    StageRecord,
)


def build_manifest(
    *,
    run_id: str,
    story_slug: str,
    title: str,
    chapter_ids: list[str],
    mode: str,
    public_safe: bool,
    inputs: dict[str, str],
    providers: dict[str, ProviderLabel],
    provider_calls: list[ProviderCallRecord] | None = None,
    qa: dict[str, int],
    artifact_qa: ArtifactQAReport | None = None,
    chapters: dict[str, dict[str, int | str]] | None = None,
    artifacts: dict[str, str],
    stages: list[StageRecord] | None = None,
    eval_metrics: list[EvalMetrics],
    bench_ablation: BenchAblationReport | None = None,
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        story_slug=story_slug,
        title=title,
        chapter_ids=chapter_ids,
        mode=mode,
        public_safe=public_safe,
        inputs=inputs,
        providers=providers,
        provider_calls=provider_calls or [],
        qa=qa,
        artifact_qa=artifact_qa,
        chapters=chapters or {},
        artifacts=artifacts,
        stages=stages or [],
        eval_metrics=eval_metrics,
        bench_ablation=bench_ablation,
    )
