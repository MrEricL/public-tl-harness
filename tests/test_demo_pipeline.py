from __future__ import annotations

from pathlib import Path
import json

import pytest

from agentic_translation.models import EnsembleDecision, GlossaryParseResult, QAFinding, RepairPatch, StoryConfig, TranslationCandidate
from agentic_translation.providers_llm import LLMProviderUnavailable
from agentic_translation.pipeline import run_demo_pipeline


def test_offline_demo_pipeline_writes_expected_artifacts(tmp_path: Path) -> None:
    result = run_demo_pipeline(
        Path("samples/public_demo/story.yaml"),
        offline=True,
        run_id="demo_test",
        seed=7,
        overwrite=True,
        runs_dir=tmp_path,
    )
    run_dir = result.run_dir

    for rel in [
        "qa_source.json",
        "qa_baseline.json",
        "qa_glossary.json",
        "qa_final.json",
        "artifact_qa.json",
        "bench_ablation.json",
        "manifest.json",
        "trace.jsonl",
        "run_notes.md",
        "source/0001.txt",
        "translated_baseline/0001.txt",
        "translated_glossary/0001.txt",
        "translated_final/0001.txt",
        "review/public_demo_0001.txt",
        "review/public_demo_0001.epub",
        "report.html",
    ]:
        assert (run_dir / rel).exists(), rel

    assert result.final_metrics.score > result.baseline_metrics.score
    assert result.qa_final.summary.total_findings < result.qa_baseline.summary.total_findings
    assert result.qa_final.summary.total_findings == 0
    assert result.final_metrics.panel_mismatches == 0

    artifact_qa = json.loads((run_dir / "artifact_qa.json").read_text(encoding="utf-8"))
    assert artifact_qa["passed"] is True
    assert artifact_qa["txt"]["contains_chinese"] is False
    assert artifact_qa["epub"]["contains_prompt_leakage"] is False

    trace = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_stage = {record["stage"]: record for record in trace}
    assert by_stage["qa_baseline"]["status"] == "warn"
    assert by_stage["qa_glossary"]["status"] == "warn"
    assert by_stage["qa_final"]["status"] == "ok"
    assert by_stage["bench_ablation"]["status"] == "ok"
    assert by_stage["bench_ablation"]["score_gain"] == 81

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_qa"]["passed"] is True
    assert manifest["artifacts"]["bench_ablation"] == "bench_ablation.json"

    bench = json.loads((run_dir / "bench_ablation.json").read_text(encoding="utf-8"))
    assert bench["note"] == "Compliance score is not semantic translation quality."
    assert [step["step_id"] for step in bench["steps"]] == [
        "cheap_baseline",
        "glossary_canon",
        "router_patch_loop",
        "artifact_gate",
    ]
    assert bench["steps"][0]["compliance_score"] == result.baseline_metrics.score
    assert bench["steps"][1]["compliance_score"] == result.glossary_metrics.score
    assert bench["steps"][2]["compliance_score"] == result.final_metrics.score
    assert bench["steps"][3]["artifact_passed"] is True
    assert bench["summary"]["score_gain"] == result.final_metrics.score - result.baseline_metrics.score
    assert bench["summary"]["finding_reduction"] == (
        result.qa_baseline.summary.total_findings - result.qa_final.summary.total_findings
    )
    assert "estimated_cost_usd" not in json.dumps(bench)


def test_public_demo_exercises_candidate_selection_and_polishes_title(tmp_path: Path) -> None:
    result = run_demo_pipeline(
        Path("samples/public_demo/story.yaml"),
        provider_mode="offline",
        run_id="candidate_selection_demo",
        seed=7,
        overwrite=True,
        runs_dir=tmp_path,
    )
    final_text = (result.run_dir / "translated_final/0001.txt").read_text(encoding="utf-8")
    report = result.report_path.read_text(encoding="utf-8")

    assert any(decision.strategy == "candidate_selection" for decision in result.repair_decisions)
    assert final_text.splitlines()[0] == "Chapter 1: The Simulator Starts"
    assert "panel_repair" in report
    assert "[Remaining uses: 3]" in report
    assert "candidate_" in report
    assert result.qa_final.summary.total_findings == 0


def test_offline_demo_pipeline_is_not_seed_lucky(tmp_path: Path) -> None:
    for seed in range(20):
        result = run_demo_pipeline(
            Path("samples/public_demo/story.yaml"),
            provider_mode="offline",
            run_id=f"demo_seed_{seed}",
            seed=seed,
            overwrite=True,
            runs_dir=tmp_path,
        )

        assert result.qa_final.summary.total_findings == 0
        assert result.final_metrics.score == 100


def test_review_artifacts_use_actual_chapter_id(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    source_dir = fixture / "source"
    terms_dir = fixture / "terms"
    expected_dir = fixture / "expected"
    source_dir.mkdir(parents=True)
    terms_dir.mkdir()
    expected_dir.mkdir()
    (source_dir / "0042.txt").write_text(Path("samples/public_demo/source/0001.txt").read_text(encoding="utf-8"), encoding="utf-8")
    (terms_dir / "master_glossary.txt").write_text(Path("samples/public_demo/terms/master_glossary.txt").read_text(encoding="utf-8"), encoding="utf-8")
    (expected_dir / "dirty_translation.txt").write_text(Path("samples/public_demo/expected/dirty_translation.txt").read_text(encoding="utf-8"), encoding="utf-8")
    story_yaml = fixture / "story.yaml"
    story_yaml.write_text(
        f"""
slug: chapter_id_demo
title: Chapter Id Demo
public_safe: true
chapter_ids:
  - "0042"
paths:
  source_dir: "{source_dir}"
  glossary_path: "{terms_dir / "master_glossary.txt"}"
  expected_dir: "{expected_dir}"
  runs_dir: "{tmp_path / "runs"}"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = run_demo_pipeline(story_yaml, provider_mode="offline", run_id="chapter_id", overwrite=True)

    assert (result.run_dir / "review/chapter_id_demo_0042.txt").exists()
    assert (result.run_dir / "review/chapter_id_demo_0042.epub").exists()


def test_rule_strategy_uses_rule_patcher_not_configured_live_repair_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class RaisingRepairProvider:
        provider_name = "openai"
        model_name = "test-raising-provider"

        def propose_patch(
            self,
            *,
            chapter: str,
            source_text: str,
            translation_text: str,
            finding: QAFinding,
            glossary: GlossaryParseResult,
            ensemble_decision: EnsembleDecision | None = None,
            candidates: list[TranslationCandidate] | None = None,
        ) -> RepairPatch | None:
            raise AssertionError("rule strategy should not call configured live repair provider")

    import agentic_translation.pipeline as pipeline

    monkeypatch.setattr(pipeline, "get_repair_provider", lambda *args, **kwargs: RaisingRepairProvider())

    fixture = tmp_path / "fixture"
    source_dir = fixture / "source"
    terms_dir = fixture / "terms"
    expected_dir = fixture / "expected"
    source_dir.mkdir(parents=True)
    terms_dir.mkdir()
    expected_dir.mkdir()
    (source_dir / "0001.txt").write_text("第一章 模拟器启动\n\n天道展开。", encoding="utf-8")
    (terms_dir / "master_glossary.txt").write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    (expected_dir / "dirty_translation.txt").write_text(
        "Chapter One: The Simulator Starts\n\nThe Heavenly Dao opened.\n",
        encoding="utf-8",
    )
    story_yaml = fixture / "story.yaml"
    story_yaml.write_text(
        f"""
slug: rule_provider_demo
title: Rule Provider Demo
public_safe: true
chapter_ids:
  - "0001"
paths:
  source_dir: "{source_dir}"
  glossary_path: "{terms_dir / "master_glossary.txt"}"
  expected_dir: "{expected_dir}"
  runs_dir: "{tmp_path / "runs"}"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = run_demo_pipeline(
        story_yaml,
        provider_mode="live",
        translation_provider_name="offline",
        judge_provider_name="offline",
        repair_provider_name="openai",
        run_id="rule_provider_demo",
        overwrite=True,
    )

    assert [decision.strategy for decision in result.repair_decisions] == ["rule"]
    assert result.qa_final.summary.total_findings == 0


def test_live_translation_provider_is_called_once_per_chapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class CountingTranslationProvider:
        provider_name = "deepseek"
        model_name = "deepseek-chat"

        def __init__(self) -> None:
            self.modes: list[str] = []

        def translate(self, source_text: str, *, story: StoryConfig, glossary: GlossaryParseResult, mode: str) -> str:
            self.modes.append(mode)
            return (
                "Chapter 1: The Simulator Starts\n\n"
                "[Simulator Started]\n\n"
                "The Heavenly Dao split open above the city.\n\n"
                "Lin Che looked at the panel and whispered, \"Begin simulation.\"\n\n"
                "remaining uses: 3"
            )

    import agentic_translation.pipeline as pipeline

    provider = CountingTranslationProvider()
    monkeypatch.setattr(pipeline, "get_translation_provider", lambda *args, **kwargs: provider)

    result = run_demo_pipeline(
        Path("samples/public_demo/story.yaml"),
        provider_mode="live",
        translation_provider_name="deepseek",
        judge_provider_name="offline",
        repair_provider_name="offline",
        run_id="live_translation_once",
        overwrite=True,
        runs_dir=tmp_path,
    )

    assert provider.modes == ["glossary"]
    assert result.qa_final.summary.total_findings == 0
    assert (result.run_dir / "translated_baseline/0001.txt").read_text(encoding="utf-8") == (
        result.run_dir / "translated_glossary/0001.txt"
    ).read_text(encoding="utf-8")


def test_live_candidate_selection_can_fallback_to_offline_judge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class BalanceErrorJudgeProvider:
        provider_name = "deepseek"
        model_name = "deepseek-chat"
        call_records: list[object] = []

        def judge(
            self,
            *,
            source_text: str,
            candidates: list[TranslationCandidate],
            glossary: GlossaryParseResult,
            seed: int,
        ) -> EnsembleDecision:
            raise LLMProviderUnavailable("402 Insufficient Balance")

    import agentic_translation.pipeline as pipeline

    monkeypatch.setattr(pipeline, "get_judge_provider", lambda *args, **kwargs: BalanceErrorJudgeProvider())

    result = run_demo_pipeline(
        Path("samples/public_demo/story.yaml"),
        provider_mode="live",
        translation_provider_name="offline",
        judge_provider_name="deepseek",
        repair_provider_name="offline",
        allow_live_provider_fallback=True,
        run_id="fallback_demo",
        seed=7,
        overwrite=True,
        runs_dir=tmp_path,
    )

    assert result.qa_final.summary.total_findings == 0
    assert any("fallback" in attempt.reason.lower() for attempt in result.patch_attempts)
    assert any(decision.selected_candidate_id for decision in result.repair_decisions)


def test_live_translation_can_fallback_to_offline_translation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class BalanceErrorTranslationProvider:
        provider_name = "deepseek"
        model_name = "deepseek-chat"
        call_records: list[object] = []

        def translate(self, source_text: str, *, story: StoryConfig, glossary: GlossaryParseResult, mode: str) -> str:
            raise LLMProviderUnavailable("402 Insufficient Balance")

    import agentic_translation.pipeline as pipeline

    monkeypatch.setattr(pipeline, "get_translation_provider", lambda *args, **kwargs: BalanceErrorTranslationProvider())

    result = run_demo_pipeline(
        Path("samples/public_demo/story.yaml"),
        provider_mode="live",
        translation_provider_name="deepseek",
        judge_provider_name="offline",
        repair_provider_name="offline",
        allow_live_provider_fallback=True,
        run_id="translation_fallback_demo",
        seed=7,
        overwrite=True,
        runs_dir=tmp_path,
    )

    assert result.qa_final.summary.total_findings == 0
    assert result.provider_failure_messages == [
        "Live translation provider failed (402 Insufficient Balance); fell back to offline translation."
    ]
    trace = [
        json.loads(line)
        for line in (result.run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    translate_stage = next(record for record in trace if record["stage"] == "translate_baseline")
    assert translate_stage["status"] == "warn"
    assert "Insufficient Balance" in translate_stage["fallback_reason"]


def test_source_qa_errors_block_by_default_and_can_be_allowed(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    source_dir = fixture / "source"
    terms_dir = fixture / "terms"
    expected_dir = fixture / "expected"
    source_dir.mkdir(parents=True)
    terms_dir.mkdir()
    expected_dir.mkdir()
    (source_dir / "0001.txt").write_text("第一章 模拟器启动\n\nCloudflare 天道", encoding="utf-8")
    (terms_dir / "master_glossary.txt").write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    (expected_dir / "dirty_translation.txt").write_text("Chapter 1\n\nHeavenly Dao\n", encoding="utf-8")
    story_yaml = fixture / "story.yaml"
    story_yaml.write_text(
        f"""
slug: bad_source_demo
title: Bad Source Demo
public_safe: true
chapter_ids:
  - "0001"
paths:
  source_dir: "{source_dir}"
  glossary_path: "{terms_dir / "master_glossary.txt"}"
  expected_dir: "{expected_dir}"
  runs_dir: "{tmp_path / "runs"}"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Source QA failed"):
        run_demo_pipeline(story_yaml, provider_mode="offline", run_id="blocked", overwrite=True)

    result = run_demo_pipeline(
        story_yaml,
        provider_mode="offline",
        run_id="allowed",
        overwrite=True,
        allow_source_qa_fail=True,
    )
    assert result.qa_source.summary.error_count == 1
