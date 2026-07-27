from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agentic_translation.agent_models import AgentEpisode, AgentObservation, AgentStep
from agentic_translation.agent_provider import AgentActionRequest
from agentic_translation.batch import accept_reviewed_chapters, apply_glossary_update_plan, apply_manual_text_replacement, build_agent_work_order, build_batch_inspection_report, build_batch_proof_report, build_batch_run_config, build_glossary_gap_report, build_glossary_update_plan, build_manual_edit_plan, build_panel_report, build_tool_agent_evidence, collect_review_queue, execute_agent_work_order, last_attempt_label, load_batch_manifest, normalize_panel_splits, parse_chapter_selection, preview_agent_work_order_execution, refresh_batch_pipeline, render_agent_work_order_markdown, render_agent_work_order_execution_preview_markdown, render_batch_proof_markdown, render_glossary_gap_report_markdown, render_glossary_update_application_markdown, render_glossary_update_pass_markdown, render_glossary_update_plan_markdown, render_manual_edit_plan_markdown, render_panel_report_markdown, replay_batch_pipeline, resume_batch_pipeline, run_batch_pipeline, run_glossary_update_pass, run_live_proof_pipeline, write_agent_work_order_execution_preview, write_batch_manifest, write_batch_triage_artifacts
from agentic_translation.models import AgentAttempt, AgenticEvidence, ArtifactQAReport, BatchChapterRun, BatchInspectionReport, BatchManifest, BatchPipelineResult, BatchProofReport, BatchRunConfig, BatchSummary, EvalMetrics, PatchAttempt, PipelineResult, ProviderCallRecord, ProviderLabel, QAFinding, QALocation, QAReport, QASummary, RepairDecision, RepairPatch, StoryConfig, StoryPaths, TerminologyConsensusConfig, ToolAgentRunRecord
from agentic_translation.preflight import PreflightCheck, PreflightReport
from agentic_translation.providers_llm import LLMProviderUnavailable, ResponseCache


def _write_public_batch_fixture(tmp_path: Path, chapters: list[str] | None = None) -> Path:
    chapters = chapters or ["0001", "0002"]
    fixture = tmp_path / "story"
    source_dir = fixture / "source"
    terms_dir = fixture / "terms"
    expected_dir = fixture / "expected"
    baseline_dir = fixture / "baseline"
    source_dir.mkdir(parents=True)
    terms_dir.mkdir()
    expected_dir.mkdir()
    baseline_dir.mkdir()
    public_source = Path("samples/public_demo/source/0001.txt").read_text(encoding="utf-8")
    for chapter in chapters:
        (source_dir / f"{chapter}.txt").write_text(public_source, encoding="utf-8")
        (baseline_dir / f"{chapter}.txt").write_text(f"Existing baseline {chapter}\n", encoding="utf-8")
    (terms_dir / "master_glossary.txt").write_text(
        Path("samples/public_demo/terms/master_glossary.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (expected_dir / "dirty_translation.txt").write_text(
        Path("samples/public_demo/expected/dirty_translation.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    story_yaml = fixture / "story.yaml"
    story_yaml.write_text(
        f"""
slug: public_batch
title: Public Batch
public_safe: true
chapter_ids:
{chr(10).join(f'  - "{chapter}"' for chapter in chapters)}
paths:
  source_dir: "{source_dir}"
  glossary_path: "{terms_dir / "master_glossary.txt"}"
  expected_dir: "{expected_dir}"
  baseline_dir: "{baseline_dir}"
  runs_dir: "{tmp_path / "runs"}"
qa:
  max_repairs: 3
report:
  mode: excerpt
  max_source_chars: 1200
  max_translation_chars: 1200
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return story_yaml


def _write_manual_review_batch(tmp_path: Path, *, final_text: str = "Chapter 1\n\nDao remains vague.") -> Path:
    fixture = tmp_path / "manual_story"
    source_dir = fixture / "source"
    terms_dir = fixture / "terms"
    runs_dir = tmp_path / "runs"
    source_dir.mkdir(parents=True)
    terms_dir.mkdir()
    (source_dir / "0001.txt").write_text("第1章\n\n天道。", encoding="utf-8")
    (terms_dir / "master_glossary.txt").write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    story_yaml = fixture / "story.yaml"
    story_yaml.write_text(
        f"""
slug: manual_story
title: Manual Story
chapter_ids:
  - "0001"
paths:
  source_dir: "{source_dir}"
  glossary_path: "{terms_dir / "master_glossary.txt"}"
  runs_dir: "{runs_dir}"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    run_dir = runs_dir / "manual_review"
    chapter_dir = run_dir / "chapters" / "0001"
    final_path = chapter_dir / "translated_final" / "0001.txt"
    final_path.parent.mkdir(parents=True)
    final_path.write_text(final_text, encoding="utf-8")
    source_path = chapter_dir / "source" / "0001.txt"
    source_path.parent.mkdir(parents=True)
    source_path.write_text((source_dir / "0001.txt").read_text(encoding="utf-8"), encoding="utf-8")
    qa = QAReport(
        run_id="0001",
        story_slug="manual_story",
        chapter="0001",
        findings=[
            QAFinding(
                check_id="glossary_required",
                severity="warning",
                message="Canonical glossary term is missing.",
                location=QALocation(chapter="0001"),
                found="天道",
                expected="Heavenly Dao",
            )
        ],
        summary=QASummary(total_findings=1, warning_count=1, by_check={"glossary_required": 1}),
        score=94,
    )
    (chapter_dir / "qa_final.json").write_text(qa.model_dump_json(indent=2), encoding="utf-8")
    manifest = BatchManifest.create(
        run_id="manual_review",
        story_slug="manual_story",
        title="Manual Story",
        story_yaml=story_yaml,
        chapters=["0001"],
        mode="offline",
        providers={},
        run_dir=run_dir,
    )
    manifest.chapters["0001"].status = "review_required"
    manifest.chapters["0001"].source_path = str(source_path)
    manifest.chapters["0001"].chapter_run_dir = str(chapter_dir)
    manifest.chapters["0001"].final_path = str(final_path)
    manifest.chapters["0001"].final_findings = 1
    manifest.chapters["0001"].final_score = 94
    write_batch_manifest(run_dir / "batch_manifest.json", manifest)
    return run_dir


def _write_panel_split_batch(tmp_path: Path) -> Path:
    fixture = tmp_path / "panel_story"
    source_dir = fixture / "source"
    terms_dir = fixture / "terms"
    runs_dir = tmp_path / "runs"
    source_dir.mkdir(parents=True)
    terms_dir.mkdir()
    source_text = (
        "第1章\n\n"
        "【註：1，宿主每次模擬僅可進行一次深度模擬。 "
        "2，深度模擬狀態下宿主死亡，深度模擬將直接結束。 "
        "3，深度模擬狀態下時間流速保持一致。】\n\n"
        "【叮，正在進入深度模擬......】"
    )
    final_text = (
        "Chapter 1\n\n"
        "[Note: 1. Host may conduct deep simulation only once per simulation.]\n\n"
        "[2. If host dies in deep simulation state, deep simulation will end directly.]\n\n"
        "[3. In deep simulation state, time flow remains consistent.]\n\n"
        "[Ding. Entering deep simulation...]"
    )
    (source_dir / "0001.txt").write_text(source_text, encoding="utf-8")
    (terms_dir / "master_glossary.txt").write_text("", encoding="utf-8")
    story_yaml = fixture / "story.yaml"
    story_yaml.write_text(
        f"""
slug: panel_story
title: Panel Story
chapter_ids:
  - "0001"
paths:
  source_dir: "{source_dir}"
  glossary_path: "{terms_dir / "master_glossary.txt"}"
  runs_dir: "{runs_dir}"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    run_dir = runs_dir / "panel_split"
    chapter_dir = run_dir / "chapters" / "0001"
    final_path = chapter_dir / "translated_final" / "0001.txt"
    final_path.parent.mkdir(parents=True)
    final_path.write_text(final_text, encoding="utf-8")
    run_source_path = chapter_dir / "source" / "0001.txt"
    run_source_path.parent.mkdir(parents=True)
    run_source_path.write_text(source_text, encoding="utf-8")
    qa = QAReport(
        run_id="0001",
        story_slug="panel_story",
        chapter="0001",
        findings=[
            QAFinding(
                check_id="system_panel_count",
                severity="warning",
                message="System/panel count differs.",
                location=QALocation(chapter="0001"),
                found="4",
                expected="2",
            )
        ],
        summary=QASummary(total_findings=1, warning_count=1, by_check={"system_panel_count": 1}),
        panel_count=4,
        score=88,
    )
    (chapter_dir / "qa_final.json").write_text(qa.model_dump_json(indent=2), encoding="utf-8")
    manifest = BatchManifest.create(
        run_id="panel_split",
        story_slug="panel_story",
        title="Panel Story",
        story_yaml=story_yaml,
        chapters=["0001"],
        mode="offline",
        providers={},
        run_dir=run_dir,
    )
    chapter_run = manifest.chapters["0001"]
    chapter_run.status = "review_required"
    chapter_run.source_path = str(run_source_path)
    chapter_run.chapter_run_dir = str(chapter_dir)
    chapter_run.final_path = str(final_path)
    chapter_run.final_findings = 1
    chapter_run.final_score = 88
    write_batch_manifest(run_dir / "batch_manifest.json", manifest)
    return run_dir


def _write_single_extra_panel_split_batch(tmp_path: Path) -> Path:
    fixture = tmp_path / "panel_story"
    source_dir = fixture / "source"
    terms_dir = fixture / "terms"
    runs_dir = tmp_path / "runs"
    source_dir.mkdir(parents=True)
    terms_dir.mkdir()
    source_text = (
        "第1章\n\n"
        "【你沉默片刻，在心底默念：等哪天你也犯傻栽進坑裡，吃盡苦頭，自然就長記性了。"
        "那些血淚換來的教訓，可比旁人說一萬遍都管用。】\n\n"
        "【畢竟，沒有模擬器的話，我早就死了。】"
    )
    final_text = (
        "Chapter 1\n\n"
        "[You fall silent for a moment, then silently recite in your heart: When one day you also do something stupid and fall into a pit, suffer all kinds of hardship, you will naturally learn your lesson.]\n\n"
        "[After all, those lessons bought with blood and tears are far more useful than hearing others say something ten thousand times.]\n\n"
        "[After all, without the Simulator, I would already be dead.]"
    )
    (source_dir / "0001.txt").write_text(source_text, encoding="utf-8")
    (terms_dir / "master_glossary.txt").write_text("", encoding="utf-8")
    story_yaml = fixture / "story.yaml"
    story_yaml.write_text(
        f"""
slug: panel_story
title: Panel Story
chapter_ids:
  - "0001"
paths:
  source_dir: "{source_dir}"
  glossary_path: "{terms_dir / "master_glossary.txt"}"
  runs_dir: "{runs_dir}"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    run_dir = runs_dir / "panel_split"
    chapter_dir = run_dir / "chapters" / "0001"
    final_path = chapter_dir / "translated_final" / "0001.txt"
    final_path.parent.mkdir(parents=True)
    final_path.write_text(final_text, encoding="utf-8")
    run_source_path = chapter_dir / "source" / "0001.txt"
    run_source_path.parent.mkdir(parents=True)
    run_source_path.write_text(source_text, encoding="utf-8")
    qa = QAReport(
        run_id="0001",
        story_slug="panel_story",
        chapter="0001",
        findings=[
            QAFinding(
                check_id="system_panel_count",
                severity="warning",
                message="System/panel count differs.",
                location=QALocation(chapter="0001"),
                found="3",
                expected="2",
            )
        ],
        summary=QASummary(total_findings=1, warning_count=1, by_check={"system_panel_count": 1}),
        panel_count=3,
        score=88,
    )
    (chapter_dir / "qa_final.json").write_text(qa.model_dump_json(indent=2), encoding="utf-8")
    manifest = BatchManifest.create(
        run_id="panel_split",
        story_slug="panel_story",
        title="Panel Story",
        story_yaml=story_yaml,
        chapters=["0001"],
        mode="offline",
        providers={},
        run_dir=run_dir,
    )
    chapter_run = manifest.chapters["0001"]
    chapter_run.status = "review_required"
    chapter_run.source_path = str(run_source_path)
    chapter_run.chapter_run_dir = str(chapter_dir)
    chapter_run.final_path = str(final_path)
    chapter_run.final_findings = 1
    chapter_run.final_score = 88
    write_batch_manifest(run_dir / "batch_manifest.json", manifest)
    return run_dir


def _write_glossary_gap_batch(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "glossary_gap"
    manifest = BatchManifest.create(
        run_id="glossary_gap",
        story_slug="manual_story",
        title="Manual Story",
        story_yaml=Path("story.yaml"),
        chapters=["0001", "0002"],
        mode="offline",
        providers={},
        run_dir=run_dir,
    )
    fixtures = {
        "0001": [
            ("天道", "Heavenly Dao", "The Dao stayed vague."),
            ("煞氣", "baleful qi", "The yin-baleful aura stayed vague."),
        ],
        "0002": [
            ("天道", "Heavenly Dao", "The Dao stayed vague again."),
        ],
    }
    for chapter, findings in fixtures.items():
        chapter_dir = run_dir / "chapters" / chapter
        source_path = chapter_dir / "source" / f"{chapter}.txt"
        final_path = chapter_dir / "translated_final" / f"{chapter}.txt"
        source_path.parent.mkdir(parents=True)
        final_path.parent.mkdir(parents=True)
        source_path.write_text(
            f"第{int(chapter)}章\n\n" + "\n\n".join(f"{found}在这里出现。" for found, _, _ in findings),
            encoding="utf-8",
        )
        final_path.write_text(
            f"Chapter {int(chapter)}\n\n" + "\n\n".join(snippet for _, _, snippet in findings),
            encoding="utf-8",
        )
        qa = QAReport(
            run_id=chapter,
            story_slug="manual_story",
            chapter=chapter,
            findings=[
                QAFinding(
                    check_id="glossary_required",
                    severity="warning",
                    message="Canonical glossary term is missing.",
                    location=QALocation(chapter=chapter, paragraph_index=index + 1, snippet=snippet),
                    found=found,
                    expected=expected,
                )
                for index, (found, expected, snippet) in enumerate(findings)
            ],
            summary=QASummary(
                total_findings=len(findings),
                warning_count=len(findings),
                by_check={"glossary_required": len(findings)},
            ),
            score=100 - (6 * len(findings)),
        )
        (chapter_dir / "qa_final.json").write_text(qa.model_dump_json(indent=2), encoding="utf-8")
        manifest.chapters[chapter].status = "review_required"
        manifest.chapters[chapter].source_path = str(source_path)
        manifest.chapters[chapter].chapter_run_dir = str(chapter_dir)
        manifest.chapters[chapter].final_path = str(final_path)
        manifest.chapters[chapter].final_findings = len(findings)
        manifest.chapters[chapter].final_score = qa.score
    write_batch_manifest(run_dir / "batch_manifest.json", manifest)
    return run_dir


def _write_mixed_review_batch(tmp_path: Path) -> Path:
    fixture = tmp_path / "mixed_story"
    story_source_dir = fixture / "source"
    terms_dir = fixture / "terms"
    story_source_dir.mkdir(parents=True)
    terms_dir.mkdir()
    (terms_dir / "master_glossary.txt").write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    for chapter in ["0001", "0002", "0003"]:
        (story_source_dir / f"{chapter}.txt").write_text(f"第{int(chapter)}章\n\n天道。\n\n【面板】", encoding="utf-8")
    story_yaml = fixture / "story.yaml"
    story_yaml.write_text(
        f"""
slug: manual_story
title: Manual Story
chapter_ids:
  - "0001"
  - "0002"
  - "0003"
paths:
  source_dir: "{story_source_dir}"
  glossary_path: "{terms_dir / "master_glossary.txt"}"
  runs_dir: "{tmp_path / "runs"}"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "runs" / "mixed_review"
    manifest = BatchManifest.create(
        run_id="mixed_review",
        story_slug="manual_story",
        title="Manual Story",
        story_yaml=story_yaml,
        chapters=["0001", "0002", "0003"],
        mode="offline",
        providers={},
        run_dir=run_dir,
    )
    chapter_findings = {
        "0001": [
            QAFinding(
                check_id="glossary_required",
                severity="warning",
                message="Canonical glossary term is missing.",
                location=QALocation(chapter="0001", paragraph_index=0, snippet="The Dao stayed vague."),
                found="天道",
                expected="Heavenly Dao",
            )
        ],
        "0002": [
            QAFinding(
                check_id="system_panel_count",
                severity="warning",
                message="System panel count changed from source to translation.",
                location=QALocation(chapter="0002", paragraph_index=1, snippet="Panel missing."),
                found="1",
                expected="2",
            )
        ],
    }
    for chapter, findings in chapter_findings.items():
        chapter_dir = run_dir / "chapters" / chapter
        source_path = chapter_dir / "source" / f"{chapter}.txt"
        final_path = chapter_dir / "translated_final" / f"{chapter}.txt"
        report_path = chapter_dir / "report.html"
        source_path.parent.mkdir(parents=True)
        final_path.parent.mkdir(parents=True)
        source_path.write_text(f"第{int(chapter)}章\n\n天道。\n\n【面板】", encoding="utf-8")
        final_path.write_text(f"Chapter {int(chapter)}\n\nNeeds review.", encoding="utf-8")
        report_path.write_text("<html></html>", encoding="utf-8")
        qa = QAReport(
            run_id=chapter,
            story_slug="manual_story",
            chapter=chapter,
            findings=findings,
            summary=QASummary(
                total_findings=len(findings),
                warning_count=len(findings),
                by_check={finding.check_id: 1 for finding in findings},
            ),
            score=94,
        )
        (chapter_dir / "qa_final.json").write_text(qa.model_dump_json(indent=2), encoding="utf-8")
        manifest.chapters[chapter].status = "review_required"
        manifest.chapters[chapter].source_path = str(source_path)
        manifest.chapters[chapter].chapter_run_dir = str(chapter_dir)
        manifest.chapters[chapter].final_path = str(final_path)
        manifest.chapters[chapter].report_path = str(report_path)
        manifest.chapters[chapter].final_findings = len(findings)
        manifest.chapters[chapter].final_score = qa.score
    manifest.chapters["0003"].status = "failed"
    manifest.chapters["0003"].error = "Provider timed out."
    write_batch_manifest(run_dir / "batch_manifest.json", manifest)
    return run_dir


def _write_live_source_batch(tmp_path: Path, *, cache_dir: Path | None = None) -> Path:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001", "0002"])
    run_dir = tmp_path / "runs" / "live_source"
    cache_dir = cache_dir or tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = BatchManifest.create(
        run_id="live_source",
        story_slug="public_batch",
        title="Public Batch",
        story_yaml=story_yaml,
        chapters=["0001", "0002"],
        mode="live",
        providers={
            "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
            "judge": ProviderLabel(provider="openai", model="gpt-test"),
            "repair": ProviderLabel(provider="offline", model="offline-patch-v1"),
        },
        run_dir=run_dir,
        run_config=BatchRunConfig(
            provider_mode="live",
            translation_provider="offline",
            judge_provider="openai",
            repair_provider="offline",
            record_cache=True,
            cache_dir=str(cache_dir),
            model_name="gpt-test",
        ),
    )
    write_batch_manifest(run_dir / "batch_manifest.json", manifest)
    return run_dir


def _proof_report_for_manifest(manifest: BatchManifest, *, passed: bool = True) -> BatchProofReport:
    evidence = AgenticEvidence(
        mode=manifest.mode,
        configured_model_roles=["judge"],
        observed_agentic_roles=["judge"] if passed else [],
        candidate_selection_repairs=1,
        cache_available=passed,
        cache_entries=1 if passed else 0,
        cache_required_namespaces=["judge"],
        cache_integrity_passed=passed,
        provider_call_records=1 if passed else 0,
        cache_verified_call_records=1 if passed else 0,
        verified_candidate_selection_records=1 if passed else 0,
        replay_cache_ready=passed,
        agentic_claim_supported=passed,
        reason="verified" if passed else "not verified",
    )
    inspection = BatchInspectionReport(
        run_id=manifest.run_id,
        story_slug=manifest.story_slug,
        run_dir=manifest.run_dir,
        ready_for_delivery=passed,
        blocker_count=0 if passed else 1,
        summary=manifest.summary,
        agentic_evidence=evidence,
        run_config=manifest.run_config,
    )
    return BatchProofReport(
        run_id=manifest.run_id,
        story_slug=manifest.story_slug,
        run_dir=manifest.run_dir,
        proof_passed=passed,
        gates={"delivery": passed, "agentic": passed, "replayable": passed},
        blockers=[] if passed else ["agentic:not verified"],
        inspection=inspection,
    )


def _write_tool_agent_proof_fixture(tmp_path: Path) -> BatchManifest:
    cache_dir = tmp_path / "cache"
    run_dir = tmp_path / "run"
    agent_dir = run_dir / "chapters" / "0001" / "agent_repair"
    final_dir = run_dir / "chapters" / "0001" / "translated_final"
    agent_dir.mkdir(parents=True)
    final_dir.mkdir(parents=True)
    request = AgentActionRequest(
        episode_id="0001:demo:0001",
        step_number=1,
        story_slug="demo",
        chapter="0001",
        current_findings=[],
        remaining_steps=5,
        remaining_patch_attempts=2,
    )
    response = {"tool": "finish", "summary": "QA is clean."}
    entry = ResponseCache(cache_dir).save(
        "agent_action",
        request.canonical_payload(),
        response,
        metadata={"provider": "openai", "model": "fixture-agent"},
    )
    call = ProviderCallRecord(
        role="agent_action",
        namespace="agent_action",
        provider="openai",
        model="fixture-agent",
        payload_sha256=entry.payload_sha256,
        response_sha256=entry.response_sha256,
        cache_file=entry.cache_file,
        cache_hit=True,
    )
    qa = QAReport(
        run_id="0001",
        story_slug="demo",
        chapter="0001",
        findings=[],
        summary=QASummary(),
        score=100,
    )
    episode = AgentEpisode(
        episode_id="0001:demo:0001",
        run_id="0001",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        provider="openai",
        model="fixture-agent",
        initial_qa=qa,
        final_qa=qa,
        final_status="verified",
        steps=[
            AgentStep(
                sequence=1,
                action=response,
                observation=AgentObservation(ok=True, kind="finished", message="QA is clean."),
                provider_call=call,
                qa_before=qa,
                qa_after=qa,
            )
        ],
    )
    episode_path = agent_dir / "agent_episode.json"
    episode_path.write_text(episode.model_dump_json(indent=2), encoding="utf-8")
    final_path = final_dir / "0001.txt"
    final_path.write_text("Chapter 1\n\nDao Heart.", encoding="utf-8")
    (run_dir / "chapters" / "0001" / "qa_final.json").write_text(
        qa.model_dump_json(indent=2), encoding="utf-8"
    )
    manifest = BatchManifest.create(
        run_id="proof",
        story_slug="demo",
        title="Demo",
        story_yaml=tmp_path / "story.yaml",
        chapters=["0001"],
        mode="replay",
        providers={"repair": ProviderLabel(provider="openai", model="fixture-agent")},
        run_dir=run_dir,
        run_config=BatchRunConfig(
            provider_mode="replay",
            repair_provider="openai",
            model_name="fixture-agent",
            cache_dir=str(cache_dir),
            tool_agent_enabled=True,
        ),
    )
    chapter = manifest.chapters["0001"]
    chapter.status = "packaged"
    chapter.chapter_run_dir = str(run_dir / "chapters" / "0001")
    chapter.final_path = str(final_path)
    chapter.final_score = 100
    chapter.final_findings = 0
    chapter.tool_agent_episode_path = str(episode_path)
    chapter.tool_agent_final_status = "verified"
    chapter.tool_agent_steps = 1
    chapter.tool_agent_final_text_sha256 = hashlib.sha256(final_path.read_bytes()).hexdigest()
    chapter.provider_calls = [call]
    return manifest


def test_parse_chapter_selection_supports_ranges_and_lists() -> None:
    assert parse_chapter_selection("1-3,0005,42") == ["0001", "0002", "0003", "0005", "0042"]


def test_parse_chapter_selection_rejects_empty_and_backwards_ranges() -> None:
    with pytest.raises(ValueError, match="chapter selection"):
        parse_chapter_selection("")
    with pytest.raises(ValueError, match="backwards"):
        parse_chapter_selection("0010-0001")


def test_batch_manifest_round_trips_with_chapter_state(tmp_path: Path) -> None:
    manifest = BatchManifest.create(
        run_id="batch_test",
        story_slug="story",
        title="Story",
        story_yaml=Path("story.yaml"),
        chapters=["0001", "0002"],
        mode="offline",
        providers={},
        run_dir=tmp_path / "runs" / "batch_test",
    )

    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    loaded = BatchManifest.model_validate_json(path.read_text(encoding="utf-8"))

    assert loaded.chapters["0001"].status == "pending"
    assert loaded.summary.total_chapters == 2


def test_batch_summary_counts_incomplete_chapters(tmp_path: Path) -> None:
    manifest = BatchManifest.create(
        run_id="batch_incomplete",
        story_slug="story",
        title="Story",
        story_yaml=Path("story.yaml"),
        chapters=["0001", "0002", "0003"],
        mode="offline",
        providers={},
        run_dir=tmp_path / "runs" / "batch_incomplete",
    )
    manifest.chapters["0001"].status = "running"
    manifest.chapters["0002"].status = "packaged"
    manifest.chapters["0003"].status = "qa_warn"
    manifest.refresh_summary()

    assert manifest.summary.incomplete == 2


def test_batch_inspection_report_lists_delivery_blockers(tmp_path: Path) -> None:
    manifest = BatchManifest.create(
        run_id="blocked_batch",
        story_slug="story",
        title="Story",
        story_yaml=Path("story.yaml"),
        chapters=["0001", "0002", "0003"],
        mode="offline",
        providers={},
        run_dir=tmp_path / "runs" / "blocked_batch",
    )
    manifest.chapters["0001"].status = "running"
    manifest.chapters["0002"].status = "failed"
    manifest.chapters["0002"].error = "provider timed out"
    manifest.chapters["0003"].status = "review_required"
    manifest.chapters["0003"].final_findings = 2
    manifest.artifact_qa = ArtifactQAReport(
        expected_chapters=3,
        passed=False,
        failures=["TXT contains Chinese residue."],
    )
    manifest.refresh_summary()

    report = build_batch_inspection_report(manifest)

    assert report.ready_for_delivery is False
    assert report.summary.incomplete == 1
    assert [blocker.blocker_type for blocker in report.blockers] == [
        "incomplete",
        "failed",
        "review_required",
        "artifact_qa",
    ]
    assert report.blockers[0].chapter == "0001"
    assert report.blockers[1].message == "provider timed out"
    assert report.blockers[2].message == "2 final QA finding(s) remain."
    assert report.blockers[3].message == "TXT contains Chinese residue."


def test_batch_inspection_report_separates_delivery_from_agentic_evidence(tmp_path: Path) -> None:
    manifest = BatchManifest.create(
        run_id="offline_packaged",
        story_slug="story",
        title="Story",
        story_yaml=Path("story.yaml"),
        chapters=["0001"],
        mode="offline",
        providers={
            "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
            "judge": ProviderLabel(provider="offline", model="offline-fixture-v1"),
            "repair": ProviderLabel(provider="offline", model="offline-fixture-v1"),
        },
        run_dir=tmp_path / "runs" / "offline_packaged",
    )
    manifest.chapters["0001"].status = "packaged"
    manifest.refresh_summary()

    report = build_batch_inspection_report(manifest)

    assert report.ready_for_delivery is True
    assert report.agentic_evidence.agentic_claim_supported is False
    assert report.agentic_evidence.mode == "offline"
    assert report.agentic_evidence.configured_model_roles == []
    assert report.agentic_evidence.observed_agentic_roles == []
    assert "offline mode" in report.agentic_evidence.reason


def test_batch_inspection_report_detects_model_backed_candidate_selection(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_entry = ResponseCache(cache_dir).save(
        "judge",
        {"payload": "judge"},
        {"selected_candidate_id": "candidate_b"},
        metadata={"provider": "openai", "model": "gpt-test"},
    )
    manifest = BatchManifest.create(
        run_id="live_judge",
        story_slug="story",
        title="Story",
        story_yaml=Path("story.yaml"),
        chapters=["0001"],
        mode="live",
        providers={
            "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
            "judge": ProviderLabel(provider="openai", model="gpt-test"),
            "repair": ProviderLabel(provider="offline", model="offline-fixture-v1"),
        },
        run_dir=tmp_path / "runs" / "live_judge",
    )
    manifest.run_config = BatchRunConfig(
        provider_mode="live",
        translation_provider="offline",
        judge_provider="openai",
        repair_provider="offline",
        record_cache=True,
        cache_dir=str(cache_dir),
        model_name="gpt-test",
    )
    manifest.chapters["0001"].status = "packaged"
    manifest.chapters["0001"].repair_decisions = [
        RepairDecision(
            finding_check_id="system_panel_count",
            strategy="candidate_selection",
            selected_candidate_id="candidate_b",
            reason="Router selected candidate_selection for system_panel_count.",
        )
    ]
    manifest.chapters["0001"].patch_attempts = [
        PatchAttempt(
            finding_check_id="system_panel_count",
            strategy="candidate_selection",
            before_score=84,
            after_score=100,
            before_findings=1,
            after_findings=0,
            accepted=True,
            reason="Accepted because compliance QA improved.",
        )
    ]
    manifest.chapters["0001"].provider_calls = [
        ProviderCallRecord(
            role="judge",
            namespace="judge",
            provider="openai",
            model="gpt-test",
            payload_sha256=cache_entry.payload_sha256,
            response_sha256=cache_entry.response_sha256,
            cache_file=cache_entry.cache_file,
            cache_hit=False,
        )
    ]
    manifest.refresh_summary()

    report = build_batch_inspection_report(manifest)

    assert report.agentic_evidence.agentic_claim_supported is True
    assert report.agentic_evidence.configured_model_roles == ["judge"]
    assert report.agentic_evidence.observed_agentic_roles == ["judge"]
    assert report.agentic_evidence.candidate_selection_repairs == 1
    assert report.agentic_evidence.verified_candidate_selection_records == 1
    assert report.agentic_evidence.candidate_selection_mismatches == []
    assert report.agentic_evidence.model_backed_patch_attempts == 0
    assert "model-backed judge" in report.agentic_evidence.reason


def test_batch_inspection_report_does_not_claim_agentic_without_provider_call_record(tmp_path: Path) -> None:
    manifest = BatchManifest.create(
        run_id="fake_live_judge",
        story_slug="story",
        title="Story",
        story_yaml=Path("story.yaml"),
        chapters=["0001"],
        mode="live",
        providers={
            "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
            "judge": ProviderLabel(provider="openai", model="gpt-test"),
            "repair": ProviderLabel(provider="offline", model="offline-fixture-v1"),
        },
        run_dir=tmp_path / "runs" / "fake_live_judge",
    )
    manifest.chapters["0001"].status = "packaged"
    manifest.chapters["0001"].repair_decisions = [
        RepairDecision(
            finding_check_id="system_panel_count",
            strategy="candidate_selection",
            selected_candidate_id="candidate_b",
            reason="Router selected candidate_selection for system_panel_count.",
        )
    ]
    manifest.chapters["0001"].patch_attempts = [
        PatchAttempt(
            finding_check_id="system_panel_count",
            strategy="candidate_selection",
            before_score=84,
            after_score=100,
            before_findings=1,
            after_findings=0,
            accepted=True,
            reason="Accepted because compliance QA improved.",
        )
    ]
    manifest.refresh_summary()

    report = build_batch_inspection_report(manifest)

    assert report.agentic_evidence.agentic_claim_supported is False
    assert report.agentic_evidence.configured_model_roles == ["judge"]
    assert report.agentic_evidence.observed_agentic_roles == []
    assert report.agentic_evidence.candidate_selection_repairs == 1
    assert "no recorded model-backed provider calls" in report.agentic_evidence.reason


def test_batch_inspection_report_surfaces_live_provider_fallback_failures(tmp_path: Path) -> None:
    manifest = BatchManifest.create(
        run_id="deepseek_fallback",
        story_slug="story",
        title="Story",
        story_yaml=Path("story.yaml"),
        chapters=["0001", "0002"],
        mode="live",
        providers={
            "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
            "judge": ProviderLabel(provider="deepseek", model="deepseek-chat"),
            "repair": ProviderLabel(provider="offline", model="offline-fixture-v1"),
        },
        run_dir=tmp_path / "runs" / "deepseek_fallback",
        run_config=BatchRunConfig(
            provider_mode="live",
            translation_provider="offline",
            judge_provider="deepseek",
            repair_provider="offline",
            model_name="deepseek-chat",
            allow_live_provider_fallback=True,
        ),
    )
    for chapter, reason in {
        "0001": "Live judge provider failed (402 Insufficient Balance); fell back to offline judge.",
        "0002": (
            "Skipped live judge because of previous live judge provider failure "
            "(402 Insufficient Balance); used offline judge."
        ),
    }.items():
        chapter_run = manifest.chapters[chapter]
        chapter_run.status = "packaged"
        chapter_run.patch_attempts = [
            PatchAttempt(
                finding_check_id="system_panel_count",
                strategy="candidate_selection",
                before_score=84,
                after_score=100,
                before_findings=1,
                after_findings=0,
                accepted=True,
                reason=reason,
            )
        ]
    manifest.refresh_summary()

    report = build_batch_inspection_report(manifest)

    assert len(report.provider_failures) == 2
    assert report.provider_failures[0].chapter == "0001"
    assert report.provider_failures[0].role == "judge"
    assert report.provider_failures[0].provider == "deepseek"
    assert report.provider_failures[0].model == "deepseek-chat"
    assert report.provider_failures[0].fallback_used is True
    assert "Insufficient Balance" in report.provider_failures[0].reason
    assert report.provider_failures[1].chapter == "0002"
    assert "previous live judge provider failure" in report.provider_failures[1].reason


def test_batch_inspection_report_surfaces_failed_live_provider_errors(tmp_path: Path) -> None:
    manifest = BatchManifest.create(
        run_id="deepseek_failed",
        story_slug="story",
        title="Story",
        story_yaml=Path("story.yaml"),
        chapters=["0001"],
        mode="live",
        providers={
            "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
            "judge": ProviderLabel(provider="deepseek", model="deepseek-chat"),
            "repair": ProviderLabel(provider="offline", model="offline-fixture-v1"),
        },
        run_dir=tmp_path / "runs" / "deepseek_failed",
        run_config=BatchRunConfig(
            provider_mode="live",
            translation_provider="offline",
            judge_provider="deepseek",
            repair_provider="offline",
            model_name="deepseek-chat",
        ),
    )
    chapter_run = manifest.chapters["0001"]
    chapter_run.status = "failed"
    chapter_run.error = "Live provider call failed after retries: 402 Insufficient Balance"
    chapter_run.attempts = [
        AgentAttempt(
            attempt_id="0001-attempt-001",
            chapter="0001",
            provider="translation=offline;judge=deepseek;repair=offline",
            model="deepseek-chat",
            action="run_chapter",
            status="fail",
            message=chapter_run.error,
        )
    ]
    manifest.refresh_summary()

    report = build_batch_inspection_report(manifest)

    assert len(report.provider_failures) == 1
    assert report.provider_failures[0].chapter == "0001"
    assert report.provider_failures[0].role == "unknown"
    assert report.provider_failures[0].provider == "deepseek"
    assert report.provider_failures[0].model == "deepseek-chat"
    assert report.provider_failures[0].fallback_used is False
    assert "Insufficient Balance" in report.provider_failures[0].reason


def test_batch_inspection_report_includes_cache_index_evidence(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    ResponseCache(cache_dir).save(
        "judge",
        {"payload": "judge-request"},
        {"selected_candidate_id": "candidate_b"},
        metadata={"provider": "openai", "model": "gpt-test"},
    )
    cache_entry = ResponseCache(cache_dir).inspect().entries[0]
    manifest = BatchManifest.create(
        run_id="replay_judge",
        story_slug="story",
        title="Story",
        story_yaml=Path("story.yaml"),
        chapters=["0001"],
        mode="replay",
        providers={
            "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
            "judge": ProviderLabel(provider="openai", model="gpt-test"),
            "repair": ProviderLabel(provider="offline", model="offline-fixture-v1"),
        },
        run_dir=tmp_path / "runs" / "replay_judge",
        run_config=BatchRunConfig(
            provider_mode="replay",
            translation_provider="offline",
            judge_provider="openai",
            repair_provider="offline",
            cache_dir=str(cache_dir),
            model_name="gpt-test",
        ),
    )
    manifest.chapters["0001"].status = "packaged"
    manifest.chapters["0001"].provider_calls = [
        ProviderCallRecord(
            role="judge",
            namespace="judge",
            provider="openai",
            model="gpt-test",
            payload_sha256=cache_entry.payload_sha256,
            response_sha256=cache_entry.response_sha256,
            cache_file=cache_entry.cache_file,
            cache_hit=True,
        )
    ]
    manifest.refresh_summary()

    report = build_batch_inspection_report(manifest)

    assert report.agentic_evidence.cache_dir == str(cache_dir)
    assert report.agentic_evidence.cache_available is True
    assert report.agentic_evidence.cache_entries == 1
    assert report.agentic_evidence.cache_namespaces == {"judge": 1}
    assert report.agentic_evidence.cache_required_namespaces == ["judge"]
    assert report.agentic_evidence.cache_missing_namespaces == []
    assert report.agentic_evidence.provider_call_records == 1
    assert report.agentic_evidence.cache_verified_call_records == 1
    assert report.agentic_evidence.cache_missing_call_records == []
    assert report.agentic_evidence.replay_cache_ready is True


def test_batch_inspection_report_does_not_accept_unrelated_cache_namespace_as_replay_proof(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    ResponseCache(cache_dir).save(
        "judge",
        {"payload": "unrelated-judge-request"},
        {"selected_candidate_id": "candidate_b"},
        metadata={"provider": "openai", "model": "gpt-test"},
    )
    manifest = BatchManifest.create(
        run_id="replay_judge",
        story_slug="story",
        title="Story",
        story_yaml=Path("story.yaml"),
        chapters=["0001"],
        mode="replay",
        providers={
            "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
            "judge": ProviderLabel(provider="openai", model="gpt-test"),
            "repair": ProviderLabel(provider="offline", model="offline-fixture-v1"),
        },
        run_dir=tmp_path / "runs" / "replay_judge",
        run_config=BatchRunConfig(
            provider_mode="replay",
            translation_provider="offline",
            judge_provider="openai",
            repair_provider="offline",
            cache_dir=str(cache_dir),
            model_name="gpt-test",
        ),
    )
    manifest.chapters["0001"].status = "packaged"
    manifest.chapters["0001"].provider_calls = [
        ProviderCallRecord(
            role="judge",
            namespace="judge",
            provider="openai",
            model="gpt-test",
            payload_sha256="a" * 64,
            response_sha256="b" * 64,
            cache_file="judge_" + ("a" * 64) + ".json",
            cache_hit=True,
        )
    ]
    manifest.refresh_summary()

    report = build_batch_inspection_report(manifest)

    assert report.agentic_evidence.cache_namespaces == {"judge": 1}
    assert report.agentic_evidence.cache_missing_namespaces == []
    assert report.agentic_evidence.provider_call_records == 1
    assert report.agentic_evidence.cache_verified_call_records == 0
    assert report.agentic_evidence.cache_missing_call_records == ["0001:judge:" + ("a" * 64)]
    assert report.agentic_evidence.replay_cache_ready is False


def test_collect_review_queue_aggregates_findings_and_failed_chapters(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "review_queue"
    chapter_dir = run_dir / "chapters" / "0001"
    chapter_dir.mkdir(parents=True)
    source_path = chapter_dir / "source" / "0001.txt"
    source_path.parent.mkdir()
    source_path.write_text(
        "第1章\n\n前文。\n\n天道在这里出现。\n\n后文。",
        encoding="utf-8",
    )
    final_path = chapter_dir / "translated_final" / "0001.txt"
    final_path.parent.mkdir()
    final_path.write_text(
        "Chapter 1\n\nEarlier paragraph.\n\nDao stayed untranslated.\n\nLater paragraph.",
        encoding="utf-8",
    )
    report_path = chapter_dir / "report.html"
    report_path.write_text("<html></html>", encoding="utf-8")
    finding = QAFinding(
        check_id="glossary_required",
        severity="warning",
        message="Source term appears in source but canonical translation is missing.",
        location=QALocation(chapter="0001", paragraph_index=3, snippet="Dao stayed untranslated."),
        found="天道",
        expected="Heavenly Dao",
    )
    qa = QAReport(
        run_id="0001",
        story_slug="story",
        chapter="0001",
        findings=[finding],
        summary=QASummary(total_findings=1, warning_count=1, by_check={"glossary_required": 1}),
        score=94,
    )
    (chapter_dir / "qa_final.json").write_text(qa.model_dump_json(indent=2), encoding="utf-8")
    manifest = BatchManifest.create(
        run_id="review_queue",
        story_slug="story",
        title="Story",
        story_yaml=Path("story.yaml"),
        chapters=["0001", "0002"],
        mode="offline",
        providers={},
        run_dir=run_dir,
    )
    manifest.chapters["0001"].status = "review_required"
    manifest.chapters["0001"].source_path = str(source_path)
    manifest.chapters["0001"].chapter_run_dir = str(chapter_dir)
    manifest.chapters["0001"].final_path = str(final_path)
    manifest.chapters["0001"].report_path = str(report_path)
    manifest.chapters["0001"].final_score = 94
    manifest.chapters["0001"].final_findings = 1
    manifest.chapters["0002"].status = "failed"
    manifest.chapters["0002"].error = "Artifact QA failed."
    write_batch_manifest(run_dir / "batch_manifest.json", manifest)

    queue = collect_review_queue(run_dir)

    assert queue.summary.total_items == 2
    assert queue.summary.by_check == {"chapter_failed": 1, "glossary_required": 1}
    assert queue.summary.by_chapter == {"0001": 1, "0002": 1}
    assert queue.summary.chapters == ["0001", "0002"]
    assert queue.summary.chapter_selection == "0001,0002"
    glossary_item = next(item for item in queue.items if item.check_id == "glossary_required")
    assert glossary_item.chapter == "0001"
    assert glossary_item.expected == "Heavenly Dao"
    assert glossary_item.paragraph_index == 3
    assert glossary_item.report_path == str(report_path)
    assert "天道在这里出现" in (glossary_item.source_context or "")
    assert "Dao stayed untranslated" in (glossary_item.final_context or "")
    failed_item = next(item for item in queue.items if item.check_id == "chapter_failed")
    assert failed_item.severity == "error"
    assert "Artifact QA failed" in failed_item.message


def test_collect_review_queue_aligns_source_only_gap_context_by_position(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "aligned_review_queue"
    chapter_dir = run_dir / "chapters" / "0001"
    source_path = chapter_dir / "source" / "0001.txt"
    final_path = chapter_dir / "translated_final" / "0001.txt"
    source_path.parent.mkdir(parents=True)
    final_path.parent.mkdir(parents=True)
    source_path.write_text("第1章\n\n" + ("前文。" * 300) + "煞氣在这里出现。" + ("后文。" * 20), encoding="utf-8")
    final_path.write_text(
        "Chapter 1\n\n" + ("Opening filler. " * 300) + "The baleful aura stayed vague." + (" Ending filler." * 20),
        encoding="utf-8",
    )
    qa = QAReport(
        run_id="0001",
        story_slug="story",
        chapter="0001",
        findings=[
            QAFinding(
                check_id="glossary_required",
                severity="warning",
                message="Source term appears in source but canonical translation is missing.",
                location=QALocation(chapter="0001", snippet="煞氣"),
                found="煞氣",
                expected="baleful qi",
            )
        ],
        summary=QASummary(total_findings=1, warning_count=1, by_check={"glossary_required": 1}),
        score=94,
    )
    (chapter_dir / "qa_final.json").write_text(qa.model_dump_json(indent=2), encoding="utf-8")
    manifest = BatchManifest.create(
        run_id="aligned_review_queue",
        story_slug="story",
        title="Story",
        story_yaml=Path("story.yaml"),
        chapters=["0001"],
        mode="offline",
        providers={},
        run_dir=run_dir,
    )
    manifest.chapters["0001"].status = "review_required"
    manifest.chapters["0001"].source_path = str(source_path)
    manifest.chapters["0001"].chapter_run_dir = str(chapter_dir)
    manifest.chapters["0001"].final_path = str(final_path)
    write_batch_manifest(run_dir / "batch_manifest.json", manifest)

    queue = collect_review_queue(run_dir)

    assert "baleful aura" in (queue.items[0].final_context or "")


def test_write_batch_triage_artifacts_creates_review_packet_and_updates_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "triage_packet"
    chapter_dir = run_dir / "chapters" / "0001"
    source_path = chapter_dir / "source" / "0001.txt"
    final_path = chapter_dir / "translated_final" / "0001.txt"
    source_path.parent.mkdir(parents=True)
    final_path.parent.mkdir(parents=True)
    source_path.write_text("第1章\n\n天道在这里出现。", encoding="utf-8")
    final_path.write_text("Chapter 1\n\nDao stayed untranslated.", encoding="utf-8")
    report_path = chapter_dir / "report.html"
    report_path.write_text("<html></html>", encoding="utf-8")
    qa = QAReport(
        run_id="0001",
        story_slug="story",
        chapter="0001",
        findings=[
            QAFinding(
                check_id="glossary_required",
                severity="warning",
                message="Source term appears in source but canonical translation is missing.",
                location=QALocation(chapter="0001", paragraph_index=1, snippet="Dao stayed untranslated."),
                found="天道",
                expected="Heavenly Dao",
            )
        ],
        summary=QASummary(total_findings=1, warning_count=1, by_check={"glossary_required": 1}),
        score=94,
    )
    (chapter_dir / "qa_final.json").write_text(qa.model_dump_json(indent=2), encoding="utf-8")
    manifest = BatchManifest.create(
        run_id="triage_packet",
        story_slug="story",
        title="Story",
        story_yaml=Path("story.yaml"),
        chapters=["0001"],
        mode="offline",
        providers={},
        run_dir=run_dir,
    )
    manifest.chapters["0001"].status = "review_required"
    manifest.chapters["0001"].source_path = str(source_path)
    manifest.chapters["0001"].chapter_run_dir = str(chapter_dir)
    manifest.chapters["0001"].final_path = str(final_path)
    manifest.chapters["0001"].report_path = str(report_path)
    manifest.chapters["0001"].final_score = 94
    manifest.chapters["0001"].final_findings = 1
    write_batch_manifest(run_dir / "batch_manifest.json", manifest)

    artifacts = write_batch_triage_artifacts(run_dir)

    expected_paths = {
        "review_queue": "review_queue.json",
        "review_chapters": "review_chapters.txt",
        "review_queue_markdown": "review_queue.md",
        "glossary_gap_report": "glossary_gap_report.json",
        "glossary_gap_report_markdown": "glossary_gap_report.md",
        "agentic_work_order": "agentic_work_order.json",
        "agentic_work_order_markdown": "agentic_work_order.md",
        "manual_edit_plan": "manual_edit_plan.json",
        "manual_edit_plan_markdown": "manual_edit_plan.md",
        "glossary_update_plan": "glossary_update_plan.json",
        "glossary_update_plan_markdown": "glossary_update_plan.md",
    }
    assert artifacts == expected_paths
    for relative_path in expected_paths.values():
        assert (run_dir / relative_path).exists()
    assert (run_dir / "review_chapters.txt").read_text(encoding="utf-8").strip() == "0001"
    manifest = load_batch_manifest(run_dir / "batch_manifest.json")
    for name, relative_path in expected_paths.items():
        assert manifest.artifacts[name] == relative_path
    status = json.loads((run_dir / "batch_status.json").read_text(encoding="utf-8"))
    assert status["artifacts"]["review_queue"] == "review_queue.json"


def test_build_glossary_gap_report_groups_unresolved_terms(tmp_path: Path) -> None:
    run_dir = _write_glossary_gap_batch(tmp_path)

    report = build_glossary_gap_report(run_dir)

    assert report.run_id == "glossary_gap"
    assert report.summary.total_occurrences == 3
    assert report.summary.term_count == 2
    assert report.summary.chapter_selection == "0001,0002"
    assert [gap.found for gap in report.gaps] == ["天道", "煞氣"]
    first_gap = report.gaps[0]
    assert first_gap.expected == "Heavenly Dao"
    assert first_gap.count == 2
    assert first_gap.chapters == ["0001", "0002"]
    assert "Dao stayed" in first_gap.suggested_aliases
    assert first_gap.occurrences[0].chapter == "0001"
    assert "天道在这里出现" in (first_gap.occurrences[0].source_context or "")
    assert "The Dao stayed vague" in (first_gap.occurrences[0].final_context or "")
    assert "do not auto-patch from source-only evidence" in first_gap.suggested_action
    second_gap = report.gaps[1]
    assert "yin-baleful aura" in second_gap.suggested_aliases


def test_render_glossary_gap_report_markdown_includes_context_and_actions(tmp_path: Path) -> None:
    run_dir = _write_glossary_gap_batch(tmp_path)
    report = build_glossary_gap_report(run_dir)

    markdown = render_glossary_gap_report_markdown(report)

    assert "# Glossary Gap Report: glossary_gap" in markdown
    assert "- Chapter selector: `0001,0002`" in markdown
    assert "## `天道` -> `Heavenly Dao`" in markdown
    assert "- Occurrences: 2" in markdown
    assert "- Suggested aliases: `Dao stayed`" in markdown
    assert "The Dao stayed vague again." in markdown
    assert "`yin-baleful aura`" in markdown
    assert "do not auto-patch from source-only evidence" in markdown


def test_build_glossary_update_plan_suggests_candidate_lines(tmp_path: Path) -> None:
    run_dir = _write_glossary_gap_batch(tmp_path)

    plan = build_glossary_update_plan(run_dir)

    assert plan.run_id == "glossary_gap"
    assert plan.summary.total_items == 2
    assert plan.summary.add_candidates_count == 2
    assert plan.summary.manual_review_count == 0
    assert plan.summary.chapter_selection == "0001,0002"
    first = plan.items[0]
    assert first.found == "天道"
    assert first.expected == "Heavenly Dao"
    assert first.action == "add_candidates"
    assert first.suggested_aliases == ["Dao stayed"]
    assert first.suggested_line == "天道: Heavenly Dao, Dao stayed"
    assert first.chapters == ["0001", "0002"]
    second = plan.items[1]
    assert second.suggested_line == "煞氣: baleful qi, yin-baleful aura"


def test_render_glossary_update_plan_markdown_lists_candidate_lines(tmp_path: Path) -> None:
    run_dir = _write_glossary_gap_batch(tmp_path)
    plan = build_glossary_update_plan(run_dir)

    markdown = render_glossary_update_plan_markdown(plan)

    assert "# Glossary Update Plan: glossary_gap" in markdown
    assert "- Add candidate lines: 2" in markdown
    assert "## `天道` -> `Heavenly Dao`" in markdown
    assert "- Suggested glossary line: `天道: Heavenly Dao, Dao stayed`" in markdown
    assert "- Suggested glossary line: `煞氣: baleful qi, yin-baleful aura`" in markdown
    assert "Edit your private glossary; this file does not mutate it." in markdown


def test_glossary_update_plan_does_not_add_english_observed_alias_as_source(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "english_alias_gap"
    chapter_dir = run_dir / "chapters" / "0001"
    source_path = chapter_dir / "source" / "0001.txt"
    final_path = chapter_dir / "translated_final" / "0001.txt"
    source_path.parent.mkdir(parents=True)
    final_path.parent.mkdir(parents=True)
    source_path.write_text("第1章\n\n煞氣在这里出现。", encoding="utf-8")
    final_path.write_text("Chapter 1\n\nThe Black Baleful Stone appeared.", encoding="utf-8")
    qa = QAReport(
        run_id="0001",
        story_slug="story",
        chapter="0001",
        findings=[
            QAFinding(
                check_id="glossary_required",
                severity="warning",
                message="Source term appears in source but canonical translation is missing.",
                location=QALocation(chapter="0001", paragraph_index=1, snippet="Black Baleful Stone"),
                found="Black Baleful Stone",
                expected="baleful qi",
            )
        ],
        summary=QASummary(total_findings=1, warning_count=1, by_check={"glossary_required": 1}),
        score=94,
    )
    (chapter_dir / "qa_final.json").write_text(qa.model_dump_json(indent=2), encoding="utf-8")
    manifest = BatchManifest.create(
        run_id="english_alias_gap",
        story_slug="story",
        title="Story",
        story_yaml=Path("story.yaml"),
        chapters=["0001"],
        mode="offline",
        providers={},
        run_dir=run_dir,
    )
    manifest.chapters["0001"].status = "review_required"
    manifest.chapters["0001"].source_path = str(source_path)
    manifest.chapters["0001"].chapter_run_dir = str(chapter_dir)
    manifest.chapters["0001"].final_path = str(final_path)
    write_batch_manifest(run_dir / "batch_manifest.json", manifest)

    plan = build_glossary_update_plan(run_dir)

    assert plan.items[0].found == "Black Baleful Stone"
    assert plan.items[0].action == "manual_review"
    assert plan.items[0].suggested_line is None


def test_apply_glossary_update_plan_dry_run_does_not_change_glossary(tmp_path: Path) -> None:
    run_dir = _write_glossary_gap_batch(tmp_path)
    glossary_path = tmp_path / "master_glossary.txt"
    original = "天道 -> Heavenly Dao\n煞氣: baleful qi\n"
    glossary_path.write_text(original, encoding="utf-8")

    result = apply_glossary_update_plan(run_dir, glossary_path=glossary_path, write=False)
    markdown = render_glossary_update_application_markdown(result)

    assert result.dry_run is True
    assert result.summary.changed_count == 2
    assert result.summary.updated_count == 2
    assert result.backup_path is None
    assert glossary_path.read_text(encoding="utf-8") == original
    assert "Dry run: true" in markdown
    assert "天道: Heavenly Dao, Dao stayed" in markdown


def test_apply_glossary_update_plan_writes_candidates_and_backup(tmp_path: Path) -> None:
    run_dir = _write_glossary_gap_batch(tmp_path)
    glossary_path = tmp_path / "master_glossary.txt"
    glossary_path.write_text("天道 -> Heavenly Dao\n煞氣: baleful qi\n其他 -> Other\n", encoding="utf-8")

    result = apply_glossary_update_plan(run_dir, glossary_path=glossary_path, write=True)

    updated = glossary_path.read_text(encoding="utf-8")
    assert result.dry_run is False
    assert result.summary.changed_count == 2
    assert result.summary.updated_count == 2
    assert result.backup_path is not None
    assert Path(result.backup_path).exists()
    assert "天道: Heavenly Dao, Dao stayed" in updated
    assert "煞氣: baleful qi, yin-baleful aura" in updated
    assert "其他 -> Other" in updated


def test_glossary_update_pass_dry_run_does_not_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = _write_glossary_gap_batch(tmp_path)
    glossary_path = tmp_path / "master_glossary.txt"
    original = "天道 -> Heavenly Dao\n煞氣: baleful qi\n"
    glossary_path.write_text(original, encoding="utf-8")

    def fail_resume(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("dry-run glossary pass must not resume the batch")

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(batch_module, "resume_batch_pipeline", fail_resume)

    result = run_glossary_update_pass(run_dir, glossary_path=glossary_path, write=False)
    markdown = render_glossary_update_pass_markdown(result)

    assert result.dry_run is True
    assert result.rerun_started is False
    assert result.chapters == ["0001", "0002"]
    assert result.application.summary.changed_count == 2
    assert glossary_path.read_text(encoding="utf-8") == original
    assert "Rerun started: false" in markdown


def test_glossary_update_pass_writes_and_resumes_changed_chapters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = _write_glossary_gap_batch(tmp_path)
    glossary_path = tmp_path / "master_glossary.txt"
    glossary_path.write_text("天道 -> Heavenly Dao\n煞氣: baleful qi\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_resume_batch_pipeline(run_dir_arg: Path, **kwargs):  # noqa: ANN202, ANN003
        captured["run_dir"] = run_dir_arg
        captured["kwargs"] = kwargs
        manifest = load_batch_manifest(run_dir / "batch_manifest.json")
        for chapter in kwargs["chapters"]:
            manifest.chapters[chapter].status = "packaged"
            manifest.chapters[chapter].final_findings = 0
            manifest.chapters[chapter].final_score = 100
        manifest.refresh_summary()
        return BatchPipelineResult(run_dir=run_dir, manifest_path=run_dir / "batch_manifest.json", manifest=manifest)

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(batch_module, "resume_batch_pipeline", fake_resume_batch_pipeline)

    result = run_glossary_update_pass(run_dir, glossary_path=glossary_path, write=True)

    assert result.dry_run is False
    assert result.rerun_started is True
    assert result.application.summary.changed_count == 2
    assert result.chapters == ["0001", "0002"]
    assert captured["run_dir"] == run_dir
    assert captured["kwargs"]["chapters"] == ["0001", "0002"]
    assert captured["kwargs"]["retry_review_required"] is True
    assert "天道: Heavenly Dao, Dao stayed" in glossary_path.read_text(encoding="utf-8")
    assert result.after_summary is not None
    assert result.after_summary.packaged == 2


def test_manual_edit_plan_groups_unresolved_items_by_final_file(tmp_path: Path) -> None:
    run_dir = _write_glossary_gap_batch(tmp_path)

    plan = build_manual_edit_plan(run_dir)
    markdown = render_manual_edit_plan_markdown(plan)

    assert plan.run_id == "glossary_gap"
    assert plan.summary.total_items == 3
    assert plan.summary.chapter_selection == "0001,0002"
    assert sorted(plan.summary.by_file.values()) == [1, 2]
    first = plan.items[0]
    assert first.chapter == "0001"
    assert first.final_path is not None
    assert first.expected == "Heavenly Dao"
    assert "source-only evidence cannot be patched automatically" in first.instruction
    assert "The Dao stayed vague" in (first.final_context or "")
    assert "# Manual Edit Plan: glossary_gap" in markdown
    assert "Use canonical term `Heavenly Dao`" in markdown
    assert "The Dao stayed vague again." in markdown


def test_glossary_gap_report_global_chapter_selector_is_sorted(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "sorted_glossary_gap"
    manifest = BatchManifest.create(
        run_id="sorted_glossary_gap",
        story_slug="manual_story",
        title="Manual Story",
        story_yaml=Path("story.yaml"),
        chapters=["0001", "0002", "0003"],
        mode="offline",
        providers={},
        run_dir=run_dir,
    )
    chapter_findings = {
        "0001": ("天道", "Heavenly Dao"),
        "0002": ("煞氣", "baleful qi"),
        "0003": ("天道", "Heavenly Dao"),
    }
    for chapter, (found, expected) in chapter_findings.items():
        chapter_dir = run_dir / "chapters" / chapter
        source_path = chapter_dir / "source" / f"{chapter}.txt"
        final_path = chapter_dir / "translated_final" / f"{chapter}.txt"
        source_path.parent.mkdir(parents=True)
        final_path.parent.mkdir(parents=True)
        source_path.write_text(f"{found}在这里出现。", encoding="utf-8")
        final_path.write_text("The term stayed vague.", encoding="utf-8")
        qa = QAReport(
            run_id=chapter,
            story_slug="manual_story",
            chapter=chapter,
            findings=[
                QAFinding(
                    check_id="glossary_required",
                    severity="warning",
                    message="Canonical glossary term is missing.",
                    location=QALocation(chapter=chapter, paragraph_index=0, snippet="The term stayed vague."),
                    found=found,
                    expected=expected,
                )
            ],
            summary=QASummary(total_findings=1, warning_count=1, by_check={"glossary_required": 1}),
            score=94,
        )
        (chapter_dir / "qa_final.json").write_text(qa.model_dump_json(indent=2), encoding="utf-8")
        manifest.chapters[chapter].status = "review_required"
        manifest.chapters[chapter].source_path = str(source_path)
        manifest.chapters[chapter].chapter_run_dir = str(chapter_dir)
        manifest.chapters[chapter].final_path = str(final_path)
        manifest.chapters[chapter].final_findings = 1
        manifest.chapters[chapter].final_score = 94
    write_batch_manifest(run_dir / "batch_manifest.json", manifest)

    report = build_glossary_gap_report(run_dir)

    assert [gap.found for gap in report.gaps] == ["天道", "煞氣"]
    assert report.summary.chapters == ["0001", "0002", "0003"]
    assert report.summary.chapter_selection == "0001,0002,0003"


def test_build_agent_work_order_classifies_unresolved_batch_work(tmp_path: Path) -> None:
    run_dir = _write_mixed_review_batch(tmp_path)

    work_order = build_agent_work_order(run_dir)

    assert work_order.run_id == "mixed_review"
    assert work_order.summary.total_items == 3
    assert work_order.summary.chapter_selection == "0001,0002,0003"
    assert work_order.summary.by_action == {
        "failed_chapter_retry": 1,
        "glossary_triage": 1,
        "live_candidate_selection": 1,
    }
    assert work_order.summary.live_retry_selection == "0002,0003"
    assert work_order.summary.glossary_selection == "0001"
    assert work_order.summary.manual_review_selection == ""
    by_chapter = {item.chapter: item for item in work_order.items}
    assert by_chapter["0001"].action == "glossary_triage"
    assert "source-only evidence" in by_chapter["0001"].reason
    assert by_chapter["0002"].action == "live_candidate_selection"
    assert "candidate selection" in by_chapter["0002"].reason
    assert by_chapter["0003"].action == "failed_chapter_retry"
    assert "Provider timed out" in by_chapter["0003"].reason
    assert "batch execute-work-order" in work_order.commands["live_retry"]
    assert "--action live-retry" in work_order.commands["live_retry"]
    assert "batch execute-work-order" in work_order.commands["live_retry_dry_run"]
    assert "--dry-run" in work_order.commands["live_retry_dry_run"]
    assert "--write-preview" in work_order.commands["live_retry_dry_run"]
    assert "batch glossary-report" in work_order.commands["glossary_triage"]


def test_build_agent_work_order_inherits_tool_agent_commands(tmp_path: Path) -> None:
    run_dir = _write_mixed_review_batch(tmp_path)
    manifest = load_batch_manifest(run_dir / "batch_manifest.json")
    manifest.run_config = BatchRunConfig(
        provider_mode="live",
        translation_provider="offline",
        judge_provider="openai",
        repair_provider="openai",
        model_name="gpt-test",
        tool_agent_enabled=True,
    )
    write_batch_manifest(run_dir / "batch_manifest.json", manifest)

    work_order = build_agent_work_order(run_dir)

    assert "--tool-agent" in work_order.commands["live_retry"]
    assert "--tool-agent" in work_order.commands["live_retry_dry_run"]


def test_render_agent_work_order_markdown_includes_commands(tmp_path: Path) -> None:
    run_dir = _write_mixed_review_batch(tmp_path)
    work_order = build_agent_work_order(run_dir)

    markdown = render_agent_work_order_markdown(work_order)

    assert "# Agent Work Order: mixed_review" in markdown
    assert "- Live retry chapters: `0002,0003`" in markdown
    assert "## 0002 - live_candidate_selection" in markdown
    assert "agentic-translation batch execute-work-order" in markdown
    assert "agentic-translation batch glossary-report" in markdown


def test_execute_agent_work_order_live_retry_uses_selected_chapters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)
    run_dir = _write_mixed_review_batch(tmp_path)
    captured: dict[str, object] = {}

    def fake_resume_batch_pipeline(run_dir_arg: Path, **kwargs):  # noqa: ANN202, ANN003
        captured["run_dir"] = run_dir_arg
        captured.update(kwargs)
        manifest = load_batch_manifest(run_dir_arg / "batch_manifest.json")
        return BatchPipelineResult(
            run_dir=run_dir_arg,
            manifest_path=run_dir_arg / "batch_manifest.json",
            manifest=manifest,
        )

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(batch_module, "resume_batch_pipeline", fake_resume_batch_pipeline)

    result = execute_agent_work_order(
        run_dir,
        provider_mode="live",
        translation_provider_name="offline",
        judge_provider_name="openai",
        repair_provider_name="openai",
        record_cache=True,
        cache_dir=tmp_path / "cache",
        model_name="test-model",
        allow_live_provider_fallback=True,
        write_proof=True,
    )

    assert result.run_dir == run_dir
    assert captured["run_dir"] == run_dir
    assert captured["chapters"] == ["0002", "0003"]
    assert captured["retry_review_required"] is True
    assert captured["provider_mode"] == "live"
    assert captured["translation_provider_name"] == "offline"
    assert captured["judge_provider_name"] == "openai"
    assert captured["repair_provider_name"] == "openai"
    assert captured["record_cache"] is True
    assert captured["cache_dir"] == tmp_path / "cache"
    assert captured["model_name"] == "test-model"
    assert captured["allow_live_provider_fallback"] is True
    assert captured["write_proof"] is True


def test_execute_agent_work_order_forwards_effective_tool_agent_setting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    run_dir = _write_mixed_review_batch(tmp_path)
    manifest = load_batch_manifest(run_dir / "batch_manifest.json")
    manifest.run_config = BatchRunConfig(
        provider_mode="live",
        translation_provider="offline",
        judge_provider="openai",
        repair_provider="openai",
        model_name="gpt-test",
        tool_agent_enabled=True,
    )
    write_batch_manifest(run_dir / "batch_manifest.json", manifest)
    captured: dict[str, object] = {}

    def fake_resume_batch_pipeline(run_dir_arg: Path, **kwargs):  # noqa: ANN202, ANN003
        captured["run_dir"] = run_dir_arg
        captured.update(kwargs)
        return BatchPipelineResult(
            run_dir=run_dir_arg,
            manifest_path=run_dir_arg / "batch_manifest.json",
            manifest=load_batch_manifest(run_dir_arg / "batch_manifest.json"),
        )

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(batch_module, "resume_batch_pipeline", fake_resume_batch_pipeline)

    execute_agent_work_order(
        run_dir,
        provider_mode="live",
        translation_provider_name="offline",
        judge_provider_name="openai",
        repair_provider_name="openai",
        record_cache=True,
        cache_dir=tmp_path / "cache",
        model_name="gpt-test",
    )

    assert captured["tool_agent_enabled"] is True


def test_execute_agent_work_order_live_retry_fails_when_no_selected_chapters(tmp_path: Path) -> None:
    run_dir = _write_manual_review_batch(tmp_path)

    with pytest.raises(ValueError, match="No live-retry chapters"):
        execute_agent_work_order(
            run_dir,
            provider_mode="live",
            translation_provider_name="offline",
            judge_provider_name="openai",
            repair_provider_name="openai",
            record_cache=True,
            cache_dir=tmp_path / "cache",
            model_name="test-model",
        )


def test_execute_agent_work_order_preflights_before_resume_mutates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)
    run_dir = _write_mixed_review_batch(tmp_path)
    manifest_path = run_dir / "batch_manifest.json"
    before = manifest_path.read_text(encoding="utf-8")
    (tmp_path / "mixed_story" / "source" / "0002.txt").unlink()

    import agentic_translation.batch as batch_module

    def fail_resume(*args, **kwargs):  # noqa: ANN202, ANN002, ANN003
        raise AssertionError("resume_batch_pipeline must not run when preflight fails")

    monkeypatch.setattr(batch_module, "resume_batch_pipeline", fail_resume)

    with pytest.raises(ValueError, match="Preflight failed"):
        execute_agent_work_order(
            run_dir,
            provider_mode="live",
            translation_provider_name="offline",
            judge_provider_name="openai",
            repair_provider_name="openai",
            record_cache=True,
            cache_dir=tmp_path / "cache",
            model_name="test-model",
        )

    assert manifest_path.read_text(encoding="utf-8") == before


def test_preview_agent_work_order_execution_preflights_without_mutating(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)
    run_dir = _write_mixed_review_batch(tmp_path)
    manifest_path = run_dir / "batch_manifest.json"
    before = manifest_path.read_text(encoding="utf-8")

    preview = preview_agent_work_order_execution(
        run_dir,
        provider_mode="live",
        translation_provider_name="offline",
        judge_provider_name="openai",
        repair_provider_name="openai",
        record_cache=True,
        cache_dir=tmp_path / "cache",
        model_name="test-model",
    )

    assert preview.dry_run is True
    assert preview.would_mutate is False
    assert preview.chapters == ["0002", "0003"]
    assert preview.preflight_passed is True
    assert preview.preflight_status_counts["fail"] == 0
    assert "batch execute-work-order" in preview.command
    assert preview.recommended_next_action == "execute_live_retry"
    assert preview.preflight_blockers == []
    assert "--dry-run" in preview.dry_run_command
    assert "--write-preview" in preview.dry_run_command
    assert "--json" in preview.dry_run_command
    assert "--dry-run" not in preview.execution_command
    assert preview.recommended_command == preview.execution_command
    assert manifest_path.read_text(encoding="utf-8") == before


def test_preview_agent_work_order_execution_inherits_tool_agent_setting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    run_dir = _write_mixed_review_batch(tmp_path)
    manifest = load_batch_manifest(run_dir / "batch_manifest.json")
    manifest.run_config = BatchRunConfig(
        provider_mode="live",
        translation_provider="offline",
        judge_provider="openai",
        repair_provider="openai",
        model_name="gpt-test",
        tool_agent_enabled=True,
    )
    write_batch_manifest(run_dir / "batch_manifest.json", manifest)

    preview = preview_agent_work_order_execution(
        run_dir,
        provider_mode="live",
        translation_provider_name="offline",
        judge_provider_name="openai",
        repair_provider_name="openai",
        record_cache=True,
        cache_dir=tmp_path / "cache",
        model_name="gpt-test",
    )

    assert preview.tool_agent_enabled is True
    assert "--tool-agent" in preview.execution_command
    assert "--tool-agent" in preview.dry_run_command


def test_preview_explicitly_disables_inherited_tool_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    run_dir = _write_mixed_review_batch(tmp_path)
    manifest = load_batch_manifest(run_dir / "batch_manifest.json")
    manifest.run_config = BatchRunConfig(
        provider_mode="live",
        translation_provider="offline",
        judge_provider="openai",
        repair_provider="openai",
        model_name="gpt-test",
        tool_agent_enabled=True,
    )
    write_batch_manifest(run_dir / "batch_manifest.json", manifest)

    preview = preview_agent_work_order_execution(
        run_dir,
        provider_mode="live",
        translation_provider_name="offline",
        judge_provider_name="openai",
        repair_provider_name="openai",
        record_cache=True,
        cache_dir=tmp_path / "cache",
        model_name="gpt-test",
        tool_agent_enabled=False,
    )

    assert preview.tool_agent_enabled is False
    assert "--no-tool-agent" in preview.execution_command
    assert "--no-tool-agent" in preview.dry_run_command


def test_execute_explicitly_disables_inherited_tool_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    run_dir = _write_mixed_review_batch(tmp_path)
    manifest = load_batch_manifest(run_dir / "batch_manifest.json")
    manifest.run_config = BatchRunConfig(
        provider_mode="live",
        translation_provider="offline",
        judge_provider="openai",
        repair_provider="openai",
        model_name="gpt-test",
        tool_agent_enabled=True,
    )
    write_batch_manifest(run_dir / "batch_manifest.json", manifest)
    captured: dict[str, object] = {}

    def fake_resume_batch_pipeline(run_dir_arg: Path, **kwargs):  # noqa: ANN202, ANN003
        captured.update(kwargs)
        return BatchPipelineResult(
            run_dir=run_dir_arg,
            manifest_path=run_dir_arg / "batch_manifest.json",
            manifest=load_batch_manifest(run_dir_arg / "batch_manifest.json"),
        )

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(batch_module, "resume_batch_pipeline", fake_resume_batch_pipeline)

    execute_agent_work_order(
        run_dir,
        provider_mode="live",
        translation_provider_name="offline",
        judge_provider_name="openai",
        repair_provider_name="openai",
        record_cache=True,
        cache_dir=tmp_path / "cache",
        model_name="gpt-test",
        tool_agent_enabled=False,
    )

    assert captured["tool_agent_enabled"] is False


def test_write_agent_work_order_execution_preview_artifacts_without_manifest_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)
    run_dir = _write_mixed_review_batch(tmp_path)
    manifest_path = run_dir / "batch_manifest.json"
    before = manifest_path.read_text(encoding="utf-8")
    preview = preview_agent_work_order_execution(
        run_dir,
        provider_mode="live",
        translation_provider_name="offline",
        judge_provider_name="openai",
        repair_provider_name="openai",
        record_cache=True,
        cache_dir=tmp_path / "cache",
        model_name="test-model",
    )

    json_path, markdown_path = write_agent_work_order_execution_preview(run_dir, preview)
    markdown = render_agent_work_order_execution_preview_markdown(preview)

    assert json_path == run_dir / "agentic_execution_preview.json"
    assert markdown_path == run_dir / "agentic_execution_preview.md"
    assert json.loads(json_path.read_text(encoding="utf-8"))["would_mutate"] is False
    assert "# Agent Work-Order Execution Preview: mixed_review" in markdown_path.read_text(encoding="utf-8")
    assert "Preflight: `passed`" in markdown
    assert "Recommended next action: `execute_live_retry`" in markdown
    assert "## Execution Command" in markdown
    assert "## Dry-Run Command" in markdown
    assert "0002,0003" in markdown
    assert manifest_path.read_text(encoding="utf-8") == before


def test_preview_agent_work_order_execution_reports_preflight_blockers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)
    run_dir = _write_mixed_review_batch(tmp_path)

    preview = preview_agent_work_order_execution(
        run_dir,
        provider_mode="live",
        translation_provider_name="offline",
        judge_provider_name="openai",
        repair_provider_name="openai",
        record_cache=True,
        cache_dir=tmp_path / "cache",
    )

    assert preview.preflight_passed is False
    assert preview.recommended_next_action == "fix_preflight"
    assert preview.recommended_command == preview.dry_run_command
    assert any("env" in blocker and "OPENAI_API_KEY" in blocker for blocker in preview.preflight_blockers)


def test_refresh_batch_pipeline_reqa_manual_final_edits_without_rerun(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = _write_manual_review_batch(tmp_path)
    final_path = run_dir / "chapters" / "0001" / "translated_final" / "0001.txt"
    final_path.write_text("Chapter 1\n\nHeavenly Dao.", encoding="utf-8")

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(
        batch_module,
        "run_demo_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("refresh must not rerun translation")),
    )

    result = refresh_batch_pipeline(run_dir, chapters=["0001"], skip_epub=True)

    chapter = result.manifest.chapters["0001"]
    assert chapter.status == "packaged"
    assert chapter.final_findings == 0
    assert chapter.final_score == 100
    assert result.artifact_qa is not None
    assert result.artifact_qa.passed is True
    refreshed_qa = QAReport.model_validate_json((run_dir / "chapters" / "0001" / "qa_final.json").read_text(encoding="utf-8"))
    assert refreshed_qa.summary.total_findings == 0
    assert "Heavenly Dao" in (run_dir / "review" / "manual_story_0001_0001.txt").read_text(encoding="utf-8")


def test_refresh_batch_pipeline_marks_manual_edit_review_required_when_qa_fails(tmp_path: Path) -> None:
    run_dir = _write_manual_review_batch(tmp_path, final_text="Chapter 1\n\n天道 remains.")

    result = refresh_batch_pipeline(run_dir, chapters=["0001"], skip_epub=True)

    chapter = result.manifest.chapters["0001"]
    assert chapter.status == "review_required"
    assert chapter.final_findings and chapter.final_findings > 0
    assert result.artifact_qa is not None
    assert result.artifact_qa.passed is False


def test_refresh_batch_pipeline_prefers_run_local_source_copy(tmp_path: Path) -> None:
    run_dir = _write_manual_review_batch(tmp_path, final_text="Chapter 1\n\n[Panel]\n\nHeavenly Dao.")
    chapter_source = run_dir / "chapters" / "0001" / "source" / "0001.txt"
    chapter_source.write_text("第1章\n\n【面板】\n\n天道。", encoding="utf-8")
    stale_source = tmp_path / "stale_source.txt"
    stale_source.write_text("第1章\n\n天道。", encoding="utf-8")
    manifest_path = run_dir / "batch_manifest.json"
    manifest = load_batch_manifest(manifest_path)
    manifest.chapters["0001"].source_path = str(stale_source)
    write_batch_manifest(manifest_path, manifest)

    result = refresh_batch_pipeline(run_dir, chapters=["0001"], skip_epub=True)

    chapter = result.manifest.chapters["0001"]
    assert chapter.status == "packaged"
    refreshed_qa = QAReport.model_validate_json((run_dir / "chapters" / "0001" / "qa_final.json").read_text(encoding="utf-8"))
    assert "system_panel_count" not in {finding.check_id for finding in refreshed_qa.findings}
    assert result.manifest.chapters["0001"].source_path == str(chapter_source)


def test_normalize_panel_splits_merges_numbered_note_panels_and_accepts(tmp_path: Path) -> None:
    run_dir = _write_panel_split_batch(tmp_path)

    result = normalize_panel_splits(
        run_dir,
        chapters=["0001"],
        reviewer="codex",
        note_prefix="Merged split note panels.",
        skip_epub=True,
    )

    assert result.normalized_count == 1
    assert result.skipped_count == 0
    assert result.items[0].replacement_count == 1
    final_text = (run_dir / "chapters" / "0001" / "translated_final" / "0001.txt").read_text(encoding="utf-8")
    assert "[Note: 1. Host may conduct deep simulation only once per simulation. 2. If host dies" in final_text
    assert "\n\n[2. If host dies" not in final_text
    manifest = load_batch_manifest(run_dir / "batch_manifest.json")
    chapter = manifest.chapters["0001"]
    assert chapter.status == "packaged"
    assert chapter.final_findings == 0
    assert chapter.manual_reviews[-1].reviewer == "codex"
    assert "Merged split note panels." in chapter.manual_reviews[-1].note
    queue = collect_review_queue(run_dir)
    assert queue.summary.total_items == 0


def test_normalize_panel_splits_merges_single_extra_adjacent_panel_by_length(tmp_path: Path) -> None:
    run_dir = _write_single_extra_panel_split_batch(tmp_path)

    result = normalize_panel_splits(
        run_dir,
        chapters=["0001"],
        reviewer="codex",
        note_prefix="Merged split panel.",
        skip_epub=True,
    )

    assert result.normalized_count == 1
    final_text = (run_dir / "chapters" / "0001" / "translated_final" / "0001.txt").read_text(encoding="utf-8")
    assert "learn your lesson. After all, those lessons bought with blood and tears" in final_text
    assert "\n\n[After all, those lessons bought with blood and tears" not in final_text
    assert "\n\n[After all, without the Simulator, I would already be dead.]" in final_text
    manifest = load_batch_manifest(run_dir / "batch_manifest.json")
    chapter = manifest.chapters["0001"]
    assert chapter.status == "packaged"
    assert chapter.final_findings == 0
    report = build_panel_report(run_dir, chapters=["0001"])
    assert report.chapters[0].source_count == 2
    assert report.chapters[0].final_count == 2


def test_accept_reviewed_chapters_records_manual_review_in_manifest_and_ledger(tmp_path: Path) -> None:
    run_dir = _write_manual_review_batch(tmp_path)
    final_path = run_dir / "chapters" / "0001" / "translated_final" / "0001.txt"
    final_path.write_text("Chapter 1\n\nHeavenly Dao.", encoding="utf-8")

    result = accept_reviewed_chapters(
        run_dir,
        chapters=["0001"],
        reviewer="eric",
        note="Fixed glossary drift after human review.",
        skip_epub=True,
    )

    chapter = result.manifest.chapters["0001"]
    assert chapter.status == "packaged"
    assert chapter.final_score == 100
    assert chapter.final_findings == 0
    assert len(chapter.manual_reviews) == 1
    review = chapter.manual_reviews[0]
    assert review.reviewer == "eric"
    assert review.note == "Fixed glossary drift after human review."
    assert review.status_before == "review_required"
    assert review.status_after == "packaged"
    assert review.qa_findings_after == 0
    ledger_path = run_dir / "manual_review.jsonl"
    ledger_records = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert ledger_records == [review.model_dump(mode="json")]


def test_accept_reviewed_chapters_rejects_empty_notes(tmp_path: Path) -> None:
    run_dir = _write_manual_review_batch(tmp_path)

    with pytest.raises(ValueError, match="review note"):
        accept_reviewed_chapters(run_dir, chapters=["0001"], reviewer="eric", note="  ")


def test_apply_manual_text_replacement_edits_final_and_records_acceptance(tmp_path: Path) -> None:
    run_dir = _write_manual_review_batch(tmp_path)

    result = apply_manual_text_replacement(
        run_dir,
        chapter="0001",
        old_text="Dao remains vague.",
        new_text="Heavenly Dao.",
        reviewer="eric",
        note=None,
        skip_epub=True,
    )

    final_path = run_dir / "chapters" / "0001" / "translated_final" / "0001.txt"
    assert final_path.read_text(encoding="utf-8") == "Chapter 1\n\nHeavenly Dao."
    assert result.occurrence_count == 1
    assert result.status_after == "packaged"
    assert result.final_findings_after == 0
    assert result.note is not None
    assert "Manual text replacement" in result.note
    manifest = load_batch_manifest(run_dir / "batch_manifest.json")
    chapter = manifest.chapters["0001"]
    assert chapter.status == "packaged"
    assert len(chapter.manual_reviews) == 1
    assert chapter.manual_reviews[0].reviewer == "eric"
    assert "Manual text replacement" in (run_dir / "manual_review.jsonl").read_text(encoding="utf-8")


def test_apply_manual_text_replacement_rejects_empty_or_missing_old_text(tmp_path: Path) -> None:
    run_dir = _write_manual_review_batch(tmp_path)

    with pytest.raises(ValueError, match="old text"):
        apply_manual_text_replacement(run_dir, chapter="0001", old_text="", new_text="Heavenly Dao")

    with pytest.raises(ValueError, match="not found"):
        apply_manual_text_replacement(run_dir, chapter="0001", old_text="missing text", new_text="Heavenly Dao")


def test_batch_pipeline_runs_two_chapters_and_writes_aggregate_artifacts(tmp_path: Path) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path)

    result = run_batch_pipeline(
        story_yaml,
        chapters=parse_chapter_selection("0001-0002"),
        provider_mode="offline",
        run_id="batch_demo",
        overwrite=True,
    )

    manifest_path = result.run_dir / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["summary"]["total_chapters"] == 2
    assert manifest["summary"]["packaged"] == 2
    assert manifest["chapters"]["0001"]["status"] == "packaged"
    assert manifest["chapters"]["0001"]["baseline_comparison"]["baseline_path"].endswith("0001.txt")
    assert manifest["run_config"]["provider_mode"] == "offline"
    assert manifest["run_config"]["translation_provider"] == "offline"
    assert manifest["run_config"]["judge_provider"] == "offline"
    assert manifest["run_config"]["repair_provider"] == "offline"
    assert manifest["run_config"]["record_cache"] is False
    assert manifest["run_config"]["cache_dir"] is None
    assert manifest["artifacts"]["status_json"] == "batch_status.json"
    assert (result.run_dir / "review/public_batch_0001_0002.txt").exists()
    assert (result.run_dir / "review/public_batch_0001_0002.epub").exists()
    status_payload = json.loads((result.run_dir / "batch_status.json").read_text(encoding="utf-8"))
    assert status_payload["ready_for_delivery"] is True
    assert status_payload["blocker_count"] == 0
    assert status_payload["run_config"]["provider_mode"] == "offline"
    assert status_payload["artifacts"]["status_json"] == "batch_status.json"
    assert status_payload["agentic_evidence"]["agentic_claim_supported"] is False
    assert "offline mode" in status_payload["agentic_evidence"]["reason"]
    assert status_payload["agentic_evidence"]["cache_available"] is False
    assert status_payload["agentic_evidence"]["cache_entries"] == 0
    assert status_payload["agentic_evidence"]["cache_namespaces"] == {}
    assert status_payload["agentic_evidence"]["replay_cache_ready"] is False
    batch_report = (result.run_dir / "batch_report.md").read_text(encoding="utf-8")
    assert "- Agentic evidence: not supported - offline mode is deterministic; no model-backed agentic claim." in batch_report
    assert result.artifact_qa is not None
    assert result.artifact_qa.passed is True


def test_batch_pipeline_can_write_nonblocking_proof_artifacts(tmp_path: Path) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])

    result = run_batch_pipeline(
        story_yaml,
        chapters=["0001"],
        provider_mode="offline",
        run_id="batch_proof",
        overwrite=True,
        skip_epub=True,
        write_proof=True,
    )

    proof_json = result.run_dir / "agentic_proof.json"
    proof_markdown = result.run_dir / "agentic_proof.md"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    proof_payload = json.loads(proof_json.read_text(encoding="utf-8"))

    assert proof_json.exists()
    assert proof_markdown.exists()
    assert manifest["artifacts"]["agentic_proof_json"] == "agentic_proof.json"
    assert manifest["artifacts"]["agentic_proof_markdown"] == "agentic_proof.md"
    assert proof_payload["gates"]["delivery"] is True
    assert proof_payload["gates"]["agentic"] is False
    assert proof_payload["gates"]["replayable"] is False
    assert proof_payload["proof_passed"] is False
    assert "Agentic Proof" in proof_markdown.read_text(encoding="utf-8")


def test_render_batch_proof_markdown_includes_model_evidence_details() -> None:
    evidence = AgenticEvidence(
        mode="live",
        configured_model_roles=["judge", "repair"],
        observed_agentic_roles=[],
        candidate_selection_repairs=2,
        model_backed_patch_attempts=1,
        cache_dir=".agentic_cache",
        cache_available=True,
        cache_entries=2,
        cache_namespaces={"judge": 1, "repair": 1},
        cache_required_namespaces=["judge", "repair"],
        cache_integrity_passed=True,
        provider_call_records=2,
        cache_verified_call_records=2,
        verified_candidate_selection_records=1,
        candidate_selection_mismatches=[
            "0002:cached judge selected candidate did not match decision candidate_b (observed candidate_a)"
        ],
        verified_repair_patch_records=0,
        repair_patch_mismatches=[
            "0002:cached repair patch did not match accepted patch replace_span:'Dao'->'Heavenly Dao'@None"
        ],
        replay_cache_ready=False,
        reason="candidate_selection repair patch did not match verified cached repair response.",
    )
    report = BatchProofReport(
        run_id="proof_mismatch",
        story_slug="story",
        run_dir="runs/proof_mismatch",
        proof_passed=False,
        gates={"delivery": True, "agentic": False, "replayable": False},
        blockers=["agentic:candidate_selection repair patch did not match verified cached repair response."],
        inspection=BatchInspectionReport(
            run_id="proof_mismatch",
            story_slug="story",
            run_dir="runs/proof_mismatch",
            ready_for_delivery=True,
            blocker_count=0,
            summary=BatchSummary(total_chapters=1, packaged=1),
            agentic_evidence=evidence,
        ),
    )

    markdown = render_batch_proof_markdown(report)

    assert "- Verified candidate selections: 1/2" in markdown
    assert "- Verified repair patches: 0/1" in markdown
    assert "## Candidate Selection Mismatches" in markdown
    assert "candidate_a" in markdown
    assert "## Repair Patch Mismatches" in markdown
    assert "Heavenly Dao" in markdown


def test_tool_agent_proof_verifies_episode_and_cache_action(tmp_path: Path) -> None:
    manifest = _write_tool_agent_proof_fixture(tmp_path)

    report = build_batch_proof_report(manifest)

    assert report.tool_agent_evidence.applicable is True
    assert report.tool_agent_evidence.verified_episodes == 1
    assert report.tool_agent_evidence.verified_actions == 1
    assert report.tool_agent_evidence.proof_ready is True
    assert report.gates["tool_agent"] is True
    assert report.gates["agentic"] is True
    assert report.tool_agent_evidence.final_text_sha256["0001"]
    assert report.tool_agent_evidence.final_qa_sha256["0001"]


def test_tool_agent_proof_rejects_tampered_persisted_action(tmp_path: Path) -> None:
    manifest = _write_tool_agent_proof_fixture(tmp_path)
    episode_path = Path(manifest.chapters["0001"].tool_agent_episode_path)
    payload = json.loads(episode_path.read_text(encoding="utf-8"))
    payload["steps"][0]["action"] = {"tool": "escalate", "reason": "tampered"}
    episode_path.write_text(json.dumps(payload), encoding="utf-8")

    evidence = build_tool_agent_evidence(manifest)

    assert evidence.proof_ready is False
    assert any("cached action differs" in mismatch for mismatch in evidence.mismatches)


def test_tool_agent_proof_rejects_orphaned_or_duplicate_manifest_calls(tmp_path: Path) -> None:
    manifest = _write_tool_agent_proof_fixture(tmp_path)
    call = manifest.chapters["0001"].provider_calls[0]
    manifest.chapters["0001"].provider_calls = [call, call]

    evidence = build_tool_agent_evidence(manifest)

    assert evidence.proof_ready is False
    assert any("duplicate" in mismatch for mismatch in evidence.mismatches)
    assert any("missing agent_action episode call" in mismatch for mismatch in evidence.mismatches)


def test_tool_agent_proof_rejects_cache_metadata_mismatch(tmp_path: Path) -> None:
    manifest = _write_tool_agent_proof_fixture(tmp_path)
    manifest.chapters["0001"].provider_calls[0].model = "different-model"

    evidence = build_tool_agent_evidence(manifest)

    assert evidence.proof_ready is False
    assert evidence.mismatches


def test_tool_agent_proof_rejects_replay_cache_miss_and_mode_mismatch(tmp_path: Path) -> None:
    manifest = _write_tool_agent_proof_fixture(tmp_path)
    manifest.chapters["0001"].provider_calls[0].cache_hit = False
    evidence = build_tool_agent_evidence(manifest)
    assert evidence.proof_ready is False
    assert any("cache_hit" in mismatch for mismatch in evidence.mismatches)

    manifest = _write_tool_agent_proof_fixture(tmp_path / "mode")
    episode_path = Path(manifest.chapters["0001"].tool_agent_episode_path)
    payload = json.loads(episode_path.read_text(encoding="utf-8"))
    payload["provider_mode"] = "live"
    episode_path.write_text(json.dumps(payload), encoding="utf-8")
    evidence = build_tool_agent_evidence(manifest)
    assert evidence.proof_ready is False
    assert any("provider_mode" in mismatch for mismatch in evidence.mismatches)


def test_tool_agent_proof_rejects_tampered_final_qa_artifact(tmp_path: Path) -> None:
    manifest = _write_tool_agent_proof_fixture(tmp_path)
    qa_path = Path(manifest.chapters["0001"].chapter_run_dir) / "qa_final.json"
    qa_path.write_text(
        QAReport(
            run_id="0001",
            story_slug="demo",
            chapter="0001",
            findings=[],
            summary=QASummary(total_findings=1),
            score=99,
        ).model_dump_json(),
        encoding="utf-8",
    )

    evidence = build_tool_agent_evidence(manifest)

    assert evidence.proof_ready is False
    assert any("final QA" in mismatch for mismatch in evidence.mismatches)


def test_tool_agent_proof_rejects_tampered_manifest_episode_counters(tmp_path: Path) -> None:
    manifest = _write_tool_agent_proof_fixture(tmp_path)
    chapter = manifest.chapters["0001"]
    chapter.tool_agent_final_status = "escalated"
    chapter.tool_agent_steps = 99
    chapter.tool_agent_initial_findings = 4
    chapter.tool_agent_final_findings = 2
    chapter.tool_agent_accepted_patches = 3
    chapter.tool_agent_rejected_patches = 4

    evidence = build_tool_agent_evidence(manifest)

    assert evidence.proof_ready is False
    assert any("manifest" in mismatch for mismatch in evidence.mismatches)


def test_tool_agent_proof_requires_safe_chapter_dir_and_typed_final_qa(tmp_path: Path) -> None:
    manifest = _write_tool_agent_proof_fixture(tmp_path)
    manifest.chapters["0001"].chapter_run_dir = None
    evidence = build_tool_agent_evidence(manifest)
    assert evidence.proof_ready is False
    assert any("chapter_run_dir" in mismatch for mismatch in evidence.mismatches)

    manifest = _write_tool_agent_proof_fixture(tmp_path / "unsafe")
    chapter_dir = Path(manifest.chapters["0001"].chapter_run_dir)
    qa_path = chapter_dir / "qa_final.json"
    qa_path.unlink()
    evidence = build_tool_agent_evidence(manifest)
    assert evidence.proof_ready is False
    assert any("final QA artifact" in mismatch for mismatch in evidence.mismatches)


def test_tool_agent_proof_rejects_final_text_mutation_after_agent_persistence(tmp_path: Path) -> None:
    manifest = _write_tool_agent_proof_fixture(tmp_path)
    final_path = Path(manifest.chapters["0001"].final_path)
    final_path.write_text("mutated after agent", encoding="utf-8")

    evidence = build_tool_agent_evidence(manifest)

    assert evidence.proof_ready is False
    assert any("final text" in mismatch for mismatch in evidence.mismatches)


def test_tool_agent_proof_requires_persisted_final_text_digest_when_enabled(tmp_path: Path) -> None:
    manifest = _write_tool_agent_proof_fixture(tmp_path)
    manifest.chapters["0001"].tool_agent_final_text_sha256 = None

    evidence = build_tool_agent_evidence(manifest)

    assert evidence.proof_ready is False
    assert any("persisted tool-agent final text digest is missing" in mismatch for mismatch in evidence.mismatches)


def test_tool_agent_proof_rejects_escalated_episode_marked_packaged_with_clean_qa(tmp_path: Path) -> None:
    manifest = _write_tool_agent_proof_fixture(tmp_path)
    episode_path = Path(manifest.chapters["0001"].tool_agent_episode_path)
    payload = json.loads(episode_path.read_text(encoding="utf-8"))
    payload["final_status"] = "escalated"
    episode_path.write_text(json.dumps(payload), encoding="utf-8")

    evidence = build_tool_agent_evidence(manifest)

    assert evidence.proof_ready is False
    assert any("status" in mismatch for mismatch in evidence.mismatches)


def test_tool_agent_proof_rejects_unsafe_episode_and_malformed_episode(tmp_path: Path) -> None:
    manifest = _write_tool_agent_proof_fixture(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    manifest.chapters["0001"].tool_agent_episode_path = str(outside)
    evidence = build_tool_agent_evidence(manifest)
    assert any("unsafe tool-agent episode path" in mismatch for mismatch in evidence.mismatches)

    manifest = _write_tool_agent_proof_fixture(tmp_path / "malformed")
    episode_path = Path(manifest.chapters["0001"].tool_agent_episode_path)
    episode_path.write_text("not json", encoding="utf-8")
    evidence = build_tool_agent_evidence(manifest)
    assert any("malformed tool-agent episode" in mismatch for mismatch in evidence.mismatches)


def test_tool_agent_proof_requires_patch_qa_and_acceptance_evidence(tmp_path: Path) -> None:
    manifest = _write_tool_agent_proof_fixture(tmp_path)
    episode_path = Path(manifest.chapters["0001"].tool_agent_episode_path)
    payload = json.loads(episode_path.read_text(encoding="utf-8"))
    payload["steps"][0]["action"] = {
        "tool": "submit_patch",
        "old_text": "Dao",
        "new_text": "Dao Heart",
        "rationale": "fix",
    }
    payload["steps"][0]["observation"] = {
        "ok": True,
        "kind": "patch_accepted",
        "message": "accepted",
        "data": {},
    }
    episode_path.write_text(json.dumps(payload), encoding="utf-8")

    evidence = build_tool_agent_evidence(manifest)

    assert evidence.proof_ready is False
    assert any("patch" in mismatch for mismatch in evidence.mismatches)


def test_tool_agent_proof_is_n_a_without_agent_calls(tmp_path: Path) -> None:
    manifest = BatchManifest.create(
        run_id="offline",
        story_slug="demo",
        title="Demo",
        story_yaml=tmp_path / "story.yaml",
        chapters=["0001"],
        mode="offline",
        providers={},
        run_dir=tmp_path / "run",
    )

    evidence = build_tool_agent_evidence(manifest)

    assert evidence.applicable is False
    assert evidence.proof_ready is True


def test_tool_agent_proof_records_source_provenance_comparisons(tmp_path: Path) -> None:
    manifest = _write_tool_agent_proof_fixture(tmp_path)
    write_batch_manifest(Path(manifest.run_dir) / "batch_manifest.json", manifest)
    manifest.replay_source_run_dir = manifest.run_dir

    evidence = build_tool_agent_evidence(manifest)

    assert evidence.action_sequence_matches is True
    assert evidence.patch_decisions_match is True
    assert evidence.final_text_matches is True
    assert evidence.final_qa_matches is True
    assert evidence.final_status_matches is True


def test_replay_batch_pipeline_reuses_source_manifest_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_run_dir = _write_live_source_batch(tmp_path)
    cache_entry = ResponseCache(tmp_path / "cache").save(
        "judge",
        {"payload": "judge"},
        {"selected_candidate_id": "candidate_b"},
        metadata={"provider": "openai", "model": "gpt-test"},
    )
    source_manifest = load_batch_manifest(source_run_dir / "batch_manifest.json")
    source_manifest.chapters["0002"].provider_calls = [
        ProviderCallRecord(
            role="judge",
            namespace="judge",
            provider="openai",
            model="gpt-test",
            payload_sha256=cache_entry.payload_sha256,
            response_sha256=cache_entry.response_sha256,
            cache_file=cache_entry.cache_file,
            cache_hit=False,
        )
    ]
    write_batch_manifest(source_run_dir / "batch_manifest.json", source_manifest)
    captured: dict[str, object] = {}

    def fake_run_batch_pipeline(story_yaml: Path, **kwargs):  # noqa: ANN003
        captured["story_yaml"] = story_yaml
        captured.update(kwargs)
        replay_run_dir = tmp_path / "runs" / str(kwargs["run_id"])
        manifest = BatchManifest.create(
            run_id=str(kwargs["run_id"]),
            story_slug="public_batch",
            title="Public Batch",
            story_yaml=story_yaml,
            chapters=list(kwargs["chapters"]),
            mode="replay",
            providers={},
            run_dir=replay_run_dir,
        )
        return BatchPipelineResult(
            run_dir=replay_run_dir,
            manifest_path=replay_run_dir / "batch_manifest.json",
            manifest=manifest,
        )

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(batch_module, "run_batch_pipeline", fake_run_batch_pipeline)

    result = replay_batch_pipeline(
        source_run_dir,
        chapters=["0002"],
        run_id="live_source_replay",
        overwrite=True,
        skip_epub=True,
    )

    assert result.run_dir == tmp_path / "runs" / "live_source_replay"
    assert captured["chapters"] == ["0002"]
    assert captured["provider_mode"] == "replay"
    assert captured["translation_provider_name"] == "offline"
    assert captured["judge_provider_name"] == "openai"
    assert captured["repair_provider_name"] == "offline"
    assert captured["record_cache"] is False
    assert captured["cache_dir"] == tmp_path / "cache"
    assert captured["model_name"] == "gpt-test"
    assert captured["run_id"] == "live_source_replay"
    assert captured["overwrite"] is True
    assert captured["skip_epub"] is True
    assert captured["write_proof"] is True


def test_resume_batch_pipeline_preserves_manifest_tool_agent_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])
    run_dir = tmp_path / "runs" / "tool_agent_resume"
    manifest = BatchManifest.create(
        run_id="tool_agent_resume",
        story_slug="public_batch",
        title="Public Batch",
        story_yaml=story_yaml,
        chapters=["0001"],
        mode="replay",
        providers={
            "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
            "repair": ProviderLabel(provider="openai", model="fixture-agent"),
        },
        run_dir=run_dir,
        run_config=BatchRunConfig(
            provider_mode="replay",
            translation_provider="offline",
            repair_provider="openai",
            cache_dir=str(tmp_path / "cache"),
            model_name="fixture-agent",
            tool_agent_enabled=True,
        ),
    )
    write_batch_manifest(run_dir / "batch_manifest.json", manifest)
    captured: dict[str, object] = {}

    def fake_run_demo_pipeline(*args, **kwargs):  # noqa: ANN001, ANN002
        captured.update(kwargs)
        return _fake_batch_pipeline_result(story_yaml=story_yaml, runs_dir=Path(kwargs["runs_dir"]))

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(batch_module, "run_demo_pipeline", fake_run_demo_pipeline)
    resumed = resume_batch_pipeline(
        run_dir,
        provider_mode="replay",
        cache_dir=tmp_path / "cache",
        force=True,
        skip_epub=True,
    )

    assert captured["tool_agent_enabled"] is True
    assert resumed.manifest.run_config is not None
    assert resumed.manifest.run_config.tool_agent_enabled is True


def test_replay_batch_pipeline_preserves_manifest_tool_agent_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_run_dir = _write_live_source_batch(tmp_path)
    cache_entry = ResponseCache(tmp_path / "cache").save(
        "judge",
        {"payload": "judge"},
        {"selected_candidate_id": "candidate_b"},
        metadata={"provider": "openai", "model": "gpt-test"},
    )
    source_manifest = load_batch_manifest(source_run_dir / "batch_manifest.json")
    source_manifest.run_config = source_manifest.run_config.model_copy(update={"tool_agent_enabled": True})
    source_manifest.run_config = source_manifest.run_config.model_copy(
        update={"repair_provider": "openai", "model_name": "gpt-test"}
    )
    ResponseCache(tmp_path / "cache").save(
        "repair",
        {"payload": "repair"},
        {"patch_type": "replace_span"},
        metadata={"provider": "openai", "model": "gpt-test"},
    )
    source_manifest.chapters["0001"].provider_calls = [
        ProviderCallRecord(
            role="judge",
            namespace="judge",
            provider="openai",
            model="gpt-test",
            payload_sha256=cache_entry.payload_sha256,
            response_sha256=cache_entry.response_sha256,
            cache_file=cache_entry.cache_file,
        )
    ]
    write_batch_manifest(source_run_dir / "batch_manifest.json", source_manifest)
    captured: dict[str, object] = {}

    def fake_run_batch_pipeline(story_yaml: Path, **kwargs):  # noqa: ANN001
        captured.update(kwargs)
        run_dir = tmp_path / "runs" / str(kwargs["run_id"])
        manifest = BatchManifest.create(
            run_id=str(kwargs["run_id"]),
            story_slug="public_batch",
            title="Public Batch",
            story_yaml=story_yaml,
            chapters=list(kwargs["chapters"]),
            mode="replay",
            providers={},
            run_dir=run_dir,
        )
        return BatchPipelineResult(run_dir=run_dir, manifest_path=run_dir / "batch_manifest.json", manifest=manifest)

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(batch_module, "run_batch_pipeline", fake_run_batch_pipeline)
    replay_batch_pipeline(source_run_dir, chapters=["0001"], run_id="tool_agent_replay")

    assert captured["tool_agent_enabled"] is True


def test_replay_batch_pipeline_requires_cached_model_provider_configuration(tmp_path: Path) -> None:
    source_run_dir = _write_live_source_batch(tmp_path)
    manifest = load_batch_manifest(source_run_dir / "batch_manifest.json")
    manifest.run_config = BatchRunConfig(
        provider_mode="live",
        translation_provider="offline",
        judge_provider="offline",
        repair_provider="offline",
        record_cache=False,
        cache_dir=None,
    )
    write_batch_manifest(source_run_dir / "batch_manifest.json", manifest)

    with pytest.raises(ValueError, match="non-offline provider"):
        replay_batch_pipeline(source_run_dir)


def test_replay_batch_pipeline_fails_before_run_when_cache_dir_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_run_dir = _write_live_source_batch(tmp_path)
    (tmp_path / "cache").rmdir()

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(
        batch_module,
        "run_batch_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("replay should fail before starting run")),
    )

    with pytest.raises(ValueError, match="Replay cache directory does not exist"):
        replay_batch_pipeline(source_run_dir)


def test_replay_batch_pipeline_fails_before_run_when_cache_namespace_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_run_dir = _write_live_source_batch(tmp_path)
    ResponseCache(tmp_path / "cache").save(
        "translation",
        {"payload": "translation"},
        {"text": "cached translation"},
        metadata={"provider": "openai", "model": "gpt-test"},
    )

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(
        batch_module,
        "run_batch_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("replay should fail before starting run")),
    )

    with pytest.raises(ValueError, match="missing indexed namespace"):
        replay_batch_pipeline(source_run_dir)


def test_replay_batch_pipeline_fails_before_run_when_cache_integrity_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_run_dir = _write_live_source_batch(tmp_path)
    ResponseCache(tmp_path / "cache").save(
        "judge",
        {"payload": "judge"},
        {"selected_candidate_id": "candidate_b"},
        metadata={"provider": "openai", "model": "gpt-test"},
    )
    cache_file = next((tmp_path / "cache").glob("judge_*.json"))
    cache_file.write_text('{"selected_candidate_id": "candidate_a"}', encoding="utf-8")

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(
        batch_module,
        "run_batch_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("replay should fail before starting run")),
    )

    with pytest.raises(ValueError, match="Replay cache integrity failed"):
        replay_batch_pipeline(source_run_dir)


def test_replay_batch_pipeline_fails_before_run_when_source_has_no_recorded_provider_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_run_dir = _write_live_source_batch(tmp_path)
    ResponseCache(tmp_path / "cache").save(
        "judge",
        {"payload": "judge"},
        {"selected_candidate_id": "candidate_b"},
        metadata={"provider": "openai", "model": "gpt-test"},
    )

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(
        batch_module,
        "run_batch_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("replay should fail before starting run")),
    )

    with pytest.raises(ValueError, match="no recorded provider calls"):
        replay_batch_pipeline(source_run_dir)


def test_replay_batch_pipeline_fails_before_run_when_source_provider_call_metadata_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_run_dir = _write_live_source_batch(tmp_path)
    cache_entry = ResponseCache(tmp_path / "cache").save(
        "judge",
        {"payload": "judge"},
        {"selected_candidate_id": "candidate_b"},
        metadata={"provider": "openai", "model": "different-model"},
    )
    source_manifest = load_batch_manifest(source_run_dir / "batch_manifest.json")
    source_manifest.chapters["0001"].provider_calls = [
        ProviderCallRecord(
            role="judge",
            namespace="judge",
            provider="openai",
            model="gpt-test",
            payload_sha256=cache_entry.payload_sha256,
            response_sha256=cache_entry.response_sha256,
            cache_file=cache_entry.cache_file,
            cache_hit=False,
        )
    ]
    write_batch_manifest(source_run_dir / "batch_manifest.json", source_manifest)

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(
        batch_module,
        "run_batch_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("replay should fail before starting run")),
    )

    with pytest.raises(ValueError, match="metadata mismatch"):
        replay_batch_pipeline(source_run_dir, chapters=["0001"])


def test_replay_batch_pipeline_validates_selected_chapters_before_replay_namespace_lookup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_run_dir = _write_live_source_batch(tmp_path)
    monkeypatch.setattr(
        "agentic_translation.batch.run_batch_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("replay should fail before starting run")),
    )

    with pytest.raises(ValueError, match="not in the source batch manifest"):
        replay_batch_pipeline(source_run_dir, chapters=["9999"])


def test_run_live_proof_pipeline_preflight_failure_happens_before_live_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])
    report = PreflightReport(
        passed=False,
        story_yaml=str(story_yaml),
        provider_mode="live",
        chapters=["0001"],
        checks=[
            PreflightCheck(
                name="credentials",
                status="fail",
                message="Missing OPENAI_API_KEY.",
            )
        ],
        status_counts={"ok": 0, "warn": 0, "fail": 1},
    )
    monkeypatch.setattr("agentic_translation.preflight.run_preflight", lambda *args, **kwargs: report)

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(
        batch_module,
        "run_batch_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("live run should not start")),
    )

    with pytest.raises(ValueError, match="Live proof preflight failed"):
        run_live_proof_pipeline(
            story_yaml,
            chapters=["0001"],
            cache_dir=tmp_path / "cache",
            model_name="gpt-test",
            run_id="live_proof",
        )


def test_run_live_proof_pipeline_runs_live_then_replay_and_requires_both_proofs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])
    preflight = PreflightReport(
        passed=True,
        story_yaml=str(story_yaml),
        provider_mode="live",
        chapters=["0001"],
        checks=[],
        status_counts={"ok": 1, "warn": 0, "fail": 0},
    )
    monkeypatch.setattr("agentic_translation.preflight.run_preflight", lambda *args, **kwargs: preflight)

    calls: list[tuple[str, dict[str, object]]] = []

    def make_result(run_id: str, mode: str) -> BatchPipelineResult:
        run_dir = tmp_path / "runs" / run_id
        manifest = BatchManifest.create(
            run_id=run_id,
            story_slug="public_batch",
            title="Public Batch",
            story_yaml=story_yaml,
            chapters=["0001"],
            mode=mode,
            providers={"judge": ProviderLabel(provider="openai", model="gpt-test")},
            run_dir=run_dir,
            run_config=BatchRunConfig(
                provider_mode=mode,
                translation_provider="offline",
                judge_provider="openai",
                repair_provider="offline",
                record_cache=mode == "live",
                cache_dir=str(tmp_path / "cache"),
                model_name="gpt-test",
            ),
        )
        manifest.chapters["0001"].status = "packaged"
        manifest.refresh_summary()
        return BatchPipelineResult(run_dir=run_dir, manifest_path=run_dir / "batch_manifest.json", manifest=manifest)

    def fake_run_batch_pipeline(story_path: Path, **kwargs):  # noqa: ANN003
        calls.append(("run", {"story_yaml": story_path, **kwargs}))
        return make_result(str(kwargs["run_id"]), "live")

    def fake_replay_batch_pipeline(source_run_dir: Path, **kwargs):  # noqa: ANN003
        calls.append(("replay", {"source_run_dir": source_run_dir, **kwargs}))
        return make_result(str(kwargs["run_id"]), "replay")

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(batch_module, "run_batch_pipeline", fake_run_batch_pipeline)
    monkeypatch.setattr(batch_module, "replay_batch_pipeline", fake_replay_batch_pipeline)
    monkeypatch.setattr(batch_module, "build_batch_proof_report", lambda manifest: _proof_report_for_manifest(manifest))

    result = run_live_proof_pipeline(
        story_yaml,
        chapters=["0001"],
        translation_provider_name="offline",
        judge_provider_name="openai",
        repair_provider_name="offline",
        cache_dir=tmp_path / "cache",
        model_name="gpt-test",
        run_id="live_proof",
        replay_run_id="live_proof_replay",
        overwrite=True,
        skip_epub=True,
        report_mode="excerpt",
    )

    assert result.proof_passed is True
    assert result.live_result.manifest.mode == "live"
    assert result.replay_result.manifest.mode == "replay"
    assert result.live_proof.proof_passed is True
    assert result.replay_proof.proof_passed is True
    assert [call[0] for call in calls] == ["run", "replay"]
    assert calls[0][1]["provider_mode"] == "live"
    assert calls[0][1]["record_cache"] is True
    assert calls[0][1]["cache_dir"] == tmp_path / "cache"
    assert calls[0][1]["write_proof"] is True
    assert calls[1][1]["source_run_dir"] == tmp_path / "runs" / "live_proof"
    assert calls[1][1]["chapters"] == ["0001"]
    assert calls[1][1]["run_id"] == "live_proof_replay"
    assert calls[1][1]["write_proof"] is True


def test_run_live_proof_pipeline_writes_combined_summary_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])
    preflight = PreflightReport(
        passed=True,
        story_yaml=str(story_yaml),
        provider_mode="live",
        chapters=["0001"],
        checks=[],
        status_counts={"ok": 1, "warn": 0, "fail": 0},
    )
    monkeypatch.setattr("agentic_translation.preflight.run_preflight", lambda *args, **kwargs: preflight)

    def make_result(run_id: str, mode: str) -> BatchPipelineResult:
        run_dir = tmp_path / "runs" / run_id
        manifest = BatchManifest.create(
            run_id=run_id,
            story_slug="public_batch",
            title="Public Batch",
            story_yaml=story_yaml,
            chapters=["0001"],
            mode=mode,
            providers={"judge": ProviderLabel(provider="openai", model="gpt-test")},
            run_dir=run_dir,
            run_config=BatchRunConfig(
                provider_mode=mode,
                translation_provider="offline",
                judge_provider="openai",
                repair_provider="offline",
                record_cache=mode == "live",
                cache_dir=str(tmp_path / "cache"),
                model_name="gpt-test",
            ),
        )
        manifest.chapters["0001"].status = "packaged"
        manifest.refresh_summary()
        return BatchPipelineResult(run_dir=run_dir, manifest_path=run_dir / "batch_manifest.json", manifest=manifest)

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(batch_module, "run_batch_pipeline", lambda *args, **kwargs: make_result(str(kwargs["run_id"]), "live"))
    monkeypatch.setattr(batch_module, "replay_batch_pipeline", lambda *args, **kwargs: make_result(str(kwargs["run_id"]), "replay"))
    monkeypatch.setattr(batch_module, "build_batch_proof_report", lambda manifest: _proof_report_for_manifest(manifest))

    result = run_live_proof_pipeline(
        story_yaml,
        chapters=["0001"],
        cache_dir=tmp_path / "cache",
        model_name="gpt-test",
        run_id="live_proof",
        replay_run_id="live_proof_replay",
        overwrite=True,
    )

    json_path = tmp_path / "runs" / "live_proof" / "live_proof_summary.json"
    markdown_path = tmp_path / "runs" / "live_proof" / "live_proof_summary.md"
    assert json_path.exists()
    assert markdown_path.exists()
    assert result.artifacts["live_proof_summary_json"] == "live_proof_summary.json"
    assert result.artifacts["live_proof_summary_markdown"] == "live_proof_summary.md"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["proof_passed"] is True
    assert payload["replay_result"]["manifest"]["run_id"] == "live_proof_replay"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Live Proof Summary" in markdown
    assert "live_proof_replay" in markdown


def test_run_live_proof_pipeline_writes_failure_summary_when_replay_proof_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])
    preflight = PreflightReport(
        passed=True,
        story_yaml=str(story_yaml),
        provider_mode="live",
        chapters=["0001"],
        checks=[],
        status_counts={"ok": 1, "warn": 0, "fail": 0},
    )
    monkeypatch.setattr("agentic_translation.preflight.run_preflight", lambda *args, **kwargs: preflight)

    def make_result(run_id: str, mode: str) -> BatchPipelineResult:
        run_dir = tmp_path / "runs" / run_id
        manifest = BatchManifest.create(
            run_id=run_id,
            story_slug="public_batch",
            title="Public Batch",
            story_yaml=story_yaml,
            chapters=["0001"],
            mode=mode,
            providers={"judge": ProviderLabel(provider="openai", model="gpt-test")},
            run_dir=run_dir,
            run_config=BatchRunConfig(
                provider_mode=mode,
                translation_provider="offline",
                judge_provider="openai",
                repair_provider="offline",
                record_cache=mode == "live",
                cache_dir=str(tmp_path / "cache"),
                model_name="gpt-test",
            ),
        )
        manifest.chapters["0001"].status = "packaged"
        manifest.refresh_summary()
        return BatchPipelineResult(run_dir=run_dir, manifest_path=run_dir / "batch_manifest.json", manifest=manifest)

    def proof_for(manifest: BatchManifest) -> BatchProofReport:
        return _proof_report_for_manifest(manifest, passed=manifest.mode == "live")

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(batch_module, "run_batch_pipeline", lambda *args, **kwargs: make_result(str(kwargs["run_id"]), "live"))
    monkeypatch.setattr(batch_module, "replay_batch_pipeline", lambda *args, **kwargs: make_result(str(kwargs["run_id"]), "replay"))
    monkeypatch.setattr(batch_module, "build_batch_proof_report", proof_for)

    with pytest.raises(RuntimeError, match="Replay proof failed"):
        run_live_proof_pipeline(
            story_yaml,
            chapters=["0001"],
            cache_dir=tmp_path / "cache",
            model_name="gpt-test",
            run_id="live_proof",
            replay_run_id="live_proof_replay",
            overwrite=True,
        )

    json_path = tmp_path / "runs" / "live_proof" / "live_proof_summary.json"
    markdown_path = tmp_path / "runs" / "live_proof" / "live_proof_summary.md"
    assert json_path.exists()
    assert markdown_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["proof_passed"] is False
    assert payload["replay_proof"]["blockers"] == ["agentic:not verified"]
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Replay Proof" in markdown
    assert "agentic:not verified" in markdown


def test_run_live_proof_pipeline_writes_failure_summary_when_live_proof_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])
    preflight = PreflightReport(
        passed=True,
        story_yaml=str(story_yaml),
        provider_mode="live",
        chapters=["0001"],
        checks=[],
        status_counts={"ok": 1, "warn": 0, "fail": 0},
    )
    monkeypatch.setattr("agentic_translation.preflight.run_preflight", lambda *args, **kwargs: preflight)

    run_dir = tmp_path / "runs" / "live_proof"
    manifest = BatchManifest.create(
        run_id="live_proof",
        story_slug="public_batch",
        title="Public Batch",
        story_yaml=story_yaml,
        chapters=["0001"],
        mode="live",
        providers={"judge": ProviderLabel(provider="openai", model="gpt-test")},
        run_dir=run_dir,
        run_config=BatchRunConfig(
            provider_mode="live",
            translation_provider="offline",
            judge_provider="openai",
            repair_provider="offline",
            record_cache=True,
            cache_dir=str(tmp_path / "cache"),
            model_name="gpt-test",
        ),
    )
    manifest.chapters["0001"].status = "packaged"
    manifest.refresh_summary()

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(
        batch_module,
        "run_batch_pipeline",
        lambda *args, **kwargs: BatchPipelineResult(
            run_dir=run_dir,
            manifest_path=run_dir / "batch_manifest.json",
            manifest=manifest,
        ),
    )
    monkeypatch.setattr(
        batch_module,
        "replay_batch_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("replay should not start")),
    )
    monkeypatch.setattr(batch_module, "build_batch_proof_report", lambda proof_manifest: _proof_report_for_manifest(proof_manifest, passed=False))

    with pytest.raises(RuntimeError, match="Live proof failed"):
        run_live_proof_pipeline(
            story_yaml,
            chapters=["0001"],
            cache_dir=tmp_path / "cache",
            model_name="gpt-test",
            run_id="live_proof",
            replay_run_id="live_proof_replay",
            overwrite=True,
        )

    json_path = run_dir / "live_proof_summary.json"
    markdown_path = run_dir / "live_proof_summary.md"
    assert json_path.exists()
    assert markdown_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["proof_passed"] is False
    assert payload["replay_result"] is None
    assert payload["replay_proof"] is None
    assert payload["live_proof"]["blockers"] == ["agentic:not verified"]
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Replay run: `not attempted`" in markdown
    assert "agentic:not verified" in markdown


def test_batch_pipeline_records_chapter_attempt_audit(tmp_path: Path) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])

    result = run_batch_pipeline(
        story_yaml,
        chapters=["0001"],
        provider_mode="offline",
        run_id="attempt_audit",
        overwrite=True,
        skip_epub=True,
    )

    chapter = result.manifest.chapters["0001"]
    assert len(chapter.attempts) == 1
    attempt = chapter.attempts[0]
    assert attempt.attempt_id == "0001-attempt-001"
    assert attempt.chapter == "0001"
    assert attempt.action == "run_chapter"
    assert attempt.provider == "translation=offline;judge=offline;repair=offline"
    assert attempt.model == "offline-fixture-v1"
    assert attempt.status == "ok"
    assert attempt.message == "Chapter packaged with 0 final QA findings."


def test_batch_pipeline_records_failed_attempt_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])

    def fail_run_demo_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("provider timed out")

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(batch_module, "run_demo_pipeline", fail_run_demo_pipeline)

    result = run_batch_pipeline(
        story_yaml,
        chapters=["0001"],
        provider_mode="offline",
        run_id="attempt_failure",
        overwrite=True,
        skip_epub=True,
    )

    chapter = result.manifest.chapters["0001"]
    assert chapter.status == "failed"
    assert chapter.error == "provider timed out"
    assert len(chapter.attempts) == 1
    assert chapter.attempts[0].status == "fail"
    assert chapter.attempts[0].message == "provider timed out"
    batch_report = (result.run_dir / "batch_report.md").read_text(encoding="utf-8")
    assert "| Chapter | Status | Score | Findings | Attempts | Last Attempt | Repairs | Accepted | Report |" in batch_report
    assert "fail: provider timed out" in batch_report
    status_payload = json.loads((result.run_dir / "batch_status.json").read_text(encoding="utf-8"))
    assert status_payload["ready_for_delivery"] is False
    assert status_payload["blockers"][0]["blocker_type"] == "failed"
    assert status_payload["blockers"][0]["message"] == "provider timed out"


def _fake_batch_pipeline_result(
    *,
    story_yaml: Path,
    runs_dir: Path,
    chapter: str = "0001",
    final_findings: int = 0,
    final_status: str | None = "verified",
    with_tool_agent: bool = True,
) -> PipelineResult:
    chapter_run_dir = runs_dir / chapter
    final_path = chapter_run_dir / "translated_final" / f"{chapter}.txt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text("Chapter 1\n\nHeavenly Dao.", encoding="utf-8")
    report_path = chapter_run_dir / "report.html"
    report_path.write_text("<html></html>", encoding="utf-8")
    qa = QAReport(
        run_id=chapter,
        story_slug="public_batch",
        chapter=chapter,
        findings=[],
        summary=QASummary(total_findings=final_findings),
        score=100 - final_findings,
    )
    tool_agent = None
    provider_calls: list[ProviderCallRecord] = []
    if with_tool_agent and final_status is not None:
        agent_dir = chapter_run_dir / "agent_repair"
        agent_dir.mkdir(parents=True, exist_ok=True)
        episode_path = agent_dir / "agent_episode.json"
        markdown_path = agent_dir / "report.md"
        html_path = agent_dir / "report.html"
        episode_path.write_text("{}", encoding="utf-8")
        markdown_path.write_text("# agent", encoding="utf-8")
        html_path.write_text("<html></html>", encoding="utf-8")
        agent_call = ProviderCallRecord(
            role="agent",
            namespace="agent_action",
            provider="openai",
            model="fixture-agent",
            payload_sha256="payload-agent",
            response_sha256="response-agent",
            cache_file="agent_action_fixture.json",
        )
        provider_calls = [agent_call]
        tool_agent = ToolAgentRunRecord(
            episode_path=episode_path,
            report_path=markdown_path,
            html_path=html_path,
            final_status=final_status,
            step_count=5,
            initial_findings=max(final_findings, 1),
            final_findings=final_findings,
            accepted_patch_count=2,
            rejected_patch_count=1,
            provider_calls=[agent_call],
        )
    metrics = EvalMetrics(mode="final", score=qa.score)
    return PipelineResult(
        run_dir=chapter_run_dir,
        report_path=report_path,
        qa_source=qa,
        qa_baseline=qa,
        qa_glossary=qa,
        qa_final=qa,
        tool_agent=tool_agent,
        provider_calls=provider_calls,
        baseline_metrics=metrics,
        glossary_metrics=metrics,
        final_metrics=metrics,
    )


def test_batch_persists_tool_agent_config_and_episode_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(
        batch_module,
        "run_demo_pipeline",
        lambda *args, **kwargs: _fake_batch_pipeline_result(
            story_yaml=story_yaml,
            runs_dir=Path(kwargs["runs_dir"]),
        ),
    )

    result = run_batch_pipeline(
        story_yaml,
        chapters=["0001"],
        provider_mode="replay",
        translation_provider_name="offline",
        judge_provider_name="offline",
        repair_provider_name="openai",
        model_name="fixture-agent",
        cache_dir=tmp_path / "cache",
        tool_agent_enabled=True,
        run_id="tool_agent_config",
        overwrite=True,
        skip_epub=True,
    )

    chapter = result.manifest.chapters["0001"]
    assert result.manifest.run_config is not None
    assert result.manifest.run_config.tool_agent_enabled is True
    assert chapter.tool_agent_final_status == "verified"
    assert chapter.tool_agent_steps == 5
    assert chapter.tool_agent_initial_findings == 1
    assert chapter.tool_agent_final_findings == 0
    assert chapter.tool_agent_accepted_patches == 2
    assert chapter.tool_agent_rejected_patches == 1
    assert chapter.status == "packaged"
    assert len(chapter.provider_calls) == 1
    assert chapter.provider_calls[0].namespace == "agent_action"


@pytest.mark.parametrize("final_status", ["escalated", "budget_exhausted"])
def test_batch_maps_unresolved_agent_episode_to_review_required(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    final_status: str,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(
        batch_module,
        "run_demo_pipeline",
        lambda *args, **kwargs: _fake_batch_pipeline_result(
            story_yaml=story_yaml,
            runs_dir=Path(kwargs["runs_dir"]),
            final_findings=1,
            final_status=final_status,
        ),
    )

    result = run_batch_pipeline(
        story_yaml,
        chapters=["0001"],
        provider_mode="replay",
        repair_provider_name="openai",
        model_name="fixture-agent",
        cache_dir=tmp_path / "cache",
        tool_agent_enabled=True,
        run_id=f"tool_agent_{final_status}",
        overwrite=True,
        skip_epub=True,
    )

    assert result.manifest.chapters["0001"].status == "review_required"


def test_batch_maps_clean_escalated_agent_episode_to_review_required(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(
        batch_module,
        "run_demo_pipeline",
        lambda *args, **kwargs: _fake_batch_pipeline_result(
            story_yaml=story_yaml,
            runs_dir=Path(kwargs["runs_dir"]),
            final_findings=0,
            final_status="escalated",
        ),
    )

    result = run_batch_pipeline(
        story_yaml,
        chapters=["0001"],
        provider_mode="replay",
        repair_provider_name="openai",
        model_name="fixture-agent",
        cache_dir=tmp_path / "cache",
        tool_agent_enabled=True,
        run_id="tool_agent_clean_escalated",
        overwrite=True,
        skip_epub=True,
    )

    assert result.manifest.chapters["0001"].status == "review_required"


def test_batch_maps_failed_agent_episode_to_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(
        batch_module,
        "run_demo_pipeline",
        lambda *args, **kwargs: _fake_batch_pipeline_result(
            story_yaml=story_yaml,
            runs_dir=Path(kwargs["runs_dir"]),
            final_findings=0,
            final_status="failed",
        ),
    )

    result = run_batch_pipeline(
        story_yaml,
        chapters=["0001"],
        provider_mode="replay",
        repair_provider_name="openai",
        model_name="fixture-agent",
        cache_dir=tmp_path / "cache",
        tool_agent_enabled=True,
        run_id="tool_agent_failed",
        overwrite=True,
        skip_epub=True,
    )

    assert result.manifest.chapters["0001"].status == "failed"


def test_batch_clean_result_without_tool_agent_metadata_stays_packaged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(
        batch_module,
        "run_demo_pipeline",
        lambda *args, **kwargs: _fake_batch_pipeline_result(
            story_yaml=story_yaml,
            runs_dir=Path(kwargs["runs_dir"]),
            with_tool_agent=False,
            final_status=None,
        ),
    )

    result = run_batch_pipeline(
        story_yaml,
        chapters=["0001"],
        provider_mode="replay",
        repair_provider_name="openai",
        model_name="fixture-agent",
        cache_dir=tmp_path / "cache",
        tool_agent_enabled=True,
        run_id="tool_agent_clean",
        overwrite=True,
        skip_epub=True,
    )

    chapter = result.manifest.chapters["0001"]
    assert chapter.status == "packaged"
    assert chapter.tool_agent_final_status is None
    assert chapter.tool_agent_episode_path is None
    assert chapter.tool_agent_steps == 0


def test_run_batch_tool_agent_contract_rejects_offline_before_run_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])
    import agentic_translation.batch as batch_module

    monkeypatch.setattr(
        batch_module,
        "run_demo_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("chapter pipeline must not run")),
    )

    with pytest.raises(ValueError, match="provider mode live or replay"):
        run_batch_pipeline(
            story_yaml,
            chapters=["0001"],
            provider_mode="offline",
            tool_agent_enabled=True,
            run_id="invalid_tool_agent_offline",
            overwrite=True,
        )

    assert not (tmp_path / "runs" / "invalid_tool_agent_offline").exists()


def test_resume_batch_tool_agent_contract_rejects_offline_before_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])
    run_dir = tmp_path / "runs" / "invalid_tool_agent_resume"
    manifest = BatchManifest.create(
        run_id="invalid_tool_agent_resume",
        story_slug="public_batch",
        title="Public Batch",
        story_yaml=story_yaml,
        chapters=["0001"],
        mode="offline",
        providers={},
        run_dir=run_dir,
        run_config=BatchRunConfig(tool_agent_enabled=True),
    )
    write_batch_manifest(run_dir / "batch_manifest.json", manifest)
    import agentic_translation.batch as batch_module

    monkeypatch.setattr(
        batch_module,
        "run_demo_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("chapter pipeline must not run")),
    )

    with pytest.raises(ValueError, match="provider mode live or replay"):
        resume_batch_pipeline(run_dir, provider_mode="offline")


def test_replay_batch_tool_agent_contract_requires_explicit_repair_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_run_dir = _write_live_source_batch(tmp_path)
    source_manifest = load_batch_manifest(source_run_dir / "batch_manifest.json")
    source_manifest.run_config = source_manifest.run_config.model_copy(
        update={"repair_provider": "openai", "model_name": None, "tool_agent_enabled": True}
    )
    write_batch_manifest(source_run_dir / "batch_manifest.json", source_manifest)
    import agentic_translation.batch as batch_module

    monkeypatch.setattr(
        batch_module,
        "run_batch_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("replay run must not start")),
    )

    with pytest.raises(ValueError, match="explicit repair model"):
        replay_batch_pipeline(source_run_dir)


def test_batch_provider_failure_preserves_tool_agent_episode_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])

    import agentic_translation.batch as batch_module

    def fail_with_episode(*args, **kwargs):  # noqa: ANN001, ANN002
        episode_path = Path(kwargs["runs_dir"]) / "0001" / "agent_repair" / "agent_episode.json"
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        episode_path.write_text('{"final_status":"failed"}', encoding="utf-8")
        raise RuntimeError("agent provider cache miss")

    monkeypatch.setattr(batch_module, "run_demo_pipeline", fail_with_episode)
    result = run_batch_pipeline(
        story_yaml,
        chapters=["0001"],
        provider_mode="replay",
        repair_provider_name="openai",
        model_name="fixture-agent",
        cache_dir=tmp_path / "cache",
        tool_agent_enabled=True,
        run_id="tool_agent_failure",
        overwrite=True,
        skip_epub=True,
    )

    chapter = result.manifest.chapters["0001"]
    assert chapter.status == "failed"
    assert chapter.error == "agent provider cache miss"
    assert chapter.tool_agent_episode_path is not None
    assert Path(chapter.tool_agent_episode_path).exists()


def test_batch_provider_failure_recovers_durable_tool_agent_episode_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])
    import agentic_translation.batch as batch_module

    initial_qa = QAReport(
        run_id="0001",
        story_slug="public_batch",
        chapter="0001",
        findings=[],
        summary=QASummary(total_findings=3),
        score=70,
    )
    final_qa = initial_qa.model_copy(update={"summary": QASummary(total_findings=2), "score": 80})
    agent_call = ProviderCallRecord(
        role="agent",
        namespace="agent_action",
        provider="openai",
        model="fixture-agent",
        payload_sha256="payload-failure",
        response_sha256="response-failure",
        cache_file="agent_action_failure.json",
    )

    def fail_with_durable_episode(*args, **kwargs):  # noqa: ANN001, ANN002
        episode_path = Path(kwargs["runs_dir"]) / "0001" / "agent_repair" / "agent_episode.json"
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        episode = AgentEpisode(
            episode_id="episode-failure",
            run_id="0001",
            story_slug="public_batch",
            chapter="0001",
            provider_mode="replay",
            provider="openai",
            model="fixture-agent",
            initial_qa=initial_qa,
            final_qa=final_qa,
            final_status="failed",
            steps=[
                AgentStep(
                    sequence=1,
                    action={"tool": "get_qa_findings"},
                    observation=AgentObservation(ok=True, kind="patch_accepted", message="accepted"),
                    provider_call=agent_call,
                )
            ],
        )
        episode_path.write_text(episode.model_dump_json(indent=2), encoding="utf-8")
        raise RuntimeError("agent provider cache miss")

    monkeypatch.setattr(batch_module, "run_demo_pipeline", fail_with_durable_episode)
    result = run_batch_pipeline(
        story_yaml,
        chapters=["0001"],
        provider_mode="replay",
        repair_provider_name="openai",
        model_name="fixture-agent",
        cache_dir=tmp_path / "cache",
        tool_agent_enabled=True,
        run_id="tool_agent_failure_metadata",
        overwrite=True,
        skip_epub=True,
    )

    chapter = result.manifest.chapters["0001"]
    assert chapter.status == "failed"
    assert chapter.error == "agent provider cache miss"
    assert chapter.tool_agent_final_status == "failed"
    assert chapter.tool_agent_steps == 1
    assert chapter.tool_agent_initial_findings == 3
    assert chapter.tool_agent_final_findings == 2
    assert chapter.tool_agent_accepted_patches == 1
    assert chapter.tool_agent_rejected_patches == 0
    assert chapter.provider_calls == [agent_call]


def test_build_batch_run_config_persists_tool_agent_and_term_consensus(tmp_path: Path) -> None:
    term_config = TerminologyConsensusConfig(
        enabled=True,
        openai_model="gpt-term",
        deepseek_model="deepseek-term",
        confidence_threshold=0.8,
    )
    config = build_batch_run_config(
        provider_mode="replay",
        translation_provider_name="offline",
        judge_provider_name="offline",
        repair_provider_name="openai",
        record_cache=False,
        cache_dir=tmp_path,
        model_name="fixture-agent",
        tool_agent_enabled=True,
        terminology_consensus=term_config,
    )

    assert config.tool_agent_enabled is True
    round_trip = BatchRunConfig.model_validate_json(config.model_dump_json())
    assert round_trip.tool_agent_enabled is True
    assert round_trip.terminology_consensus == term_config


def test_last_attempt_label_prefers_current_packaged_state_after_post_pass() -> None:
    chapter = BatchChapterRun(
        chapter="0029",
        status="packaged",
        final_score=100,
        final_findings=0,
        attempts=[
            AgentAttempt(
                attempt_id="0029-attempt-001",
                chapter="0029",
                provider="translation=offline",
                model="offline-fixture-v1",
                action="run_chapter",
                status="warn",
                message="Chapter review_required with 3 final QA findings.",
            )
        ],
    )

    assert last_attempt_label(chapter) == "final: packaged with 0 final QA findings"


def test_batch_pipeline_persists_per_chapter_repair_audit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])

    def fake_run_demo_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003
        chapter = str(kwargs["chapter_override"])
        run_dir = Path(kwargs["runs_dir"]) / chapter
        final_path = run_dir / "translated_final" / f"{chapter}.txt"
        final_path.parent.mkdir(parents=True)
        final_path.write_text("Chapter 1\n\nHeavenly Dao.", encoding="utf-8")
        report_path = run_dir / "report.html"
        report_path.write_text("<html></html>", encoding="utf-8")
        qa = QAReport(
            run_id=chapter,
            story_slug="public_batch",
            chapter=chapter,
            findings=[],
            summary=QASummary(),
            score=100,
        )
        metrics = EvalMetrics(mode="final")
        patch = RepairPatch(
            patch_id="patch_glossary_required",
            patch_type="replace_span",
            chapter=chapter,
            old_text="Fairy Alliance",
            new_text="Immortal Alliance",
            reason="Use glossary canon.",
            source_finding_check_id="glossary_required",
            accepted=True,
        )
        return PipelineResult(
            run_dir=run_dir,
            report_path=report_path,
            qa_source=qa,
            qa_baseline=qa,
            qa_glossary=qa,
            qa_final=qa,
            repair_decisions=[
                RepairDecision(
                    finding_check_id="glossary_required",
                    strategy="rule",
                    reason="Router selected rule for glossary_required.",
                )
            ],
            patch_attempts=[
                PatchAttempt(
                    finding_check_id="glossary_required",
                    strategy="rule",
                    before_score=94,
                    after_score=100,
                    before_findings=1,
                    after_findings=0,
                    accepted=True,
                    reason="Accepted because compliance QA improved.",
                    patch=patch,
                )
            ],
            baseline_metrics=metrics,
            glossary_metrics=metrics,
            final_metrics=metrics,
        )

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(batch_module, "run_demo_pipeline", fake_run_demo_pipeline)

    result = run_batch_pipeline(
        story_yaml,
        chapters=["0001"],
        provider_mode="offline",
        run_id="repair_audit",
        overwrite=True,
        skip_epub=True,
    )

    chapter_run = result.manifest.chapters["0001"]
    assert chapter_run.repair_decisions[0].strategy == "rule"
    assert chapter_run.patch_attempts[0].patch is not None
    assert chapter_run.patch_attempts[0].patch.patch_id == "patch_glossary_required"
    manifest = load_batch_manifest(result.manifest_path)
    assert manifest.chapters["0001"].repair_decisions[0].finding_check_id == "glossary_required"
    assert manifest.chapters["0001"].patch_attempts[0].accepted is True
    batch_report = (result.run_dir / "batch_report.md").read_text(encoding="utf-8")
    assert "| Chapter | Status | Score | Findings | Attempts | Last Attempt | Repairs | Accepted | Report |" in batch_report
    assert "| 0001 | packaged | 100 | 0 | 1 | final: packaged with 0 final QA findings | 1 | 1 |" in batch_report


def test_resume_batch_pipeline_processes_pending_chapters_without_rerunning_completed(tmp_path: Path) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path)
    result = run_batch_pipeline(
        story_yaml,
        chapters=parse_chapter_selection("0001"),
        provider_mode="offline",
        run_id="resume_demo",
        overwrite=True,
    )
    manifest = load_batch_manifest(result.manifest_path)
    first_run_dir = manifest.chapters["0001"].chapter_run_dir
    manifest.chapters["0002"] = BatchChapterRun(chapter="0002")
    write_batch_manifest(result.manifest_path, manifest)

    resumed = resume_batch_pipeline(result.run_dir, provider_mode="offline")

    assert resumed.manifest.chapters["0001"].status == "packaged"
    assert resumed.manifest.chapters["0001"].chapter_run_dir == first_run_dir
    assert resumed.manifest.chapters["0002"].status == "packaged"
    assert resumed.manifest.summary.packaged == 2
    assert (resumed.run_dir / "review/public_batch_0001_0002.txt").exists()


def test_resume_batch_pipeline_recovers_running_chapter_with_partial_run_dir(tmp_path: Path) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path)
    result = run_batch_pipeline(
        story_yaml,
        chapters=parse_chapter_selection("0001"),
        provider_mode="offline",
        run_id="resume_partial",
        overwrite=True,
    )
    manifest = load_batch_manifest(result.manifest_path)
    chapter_dir = result.run_dir / "chapters" / "0001"
    sentinel = chapter_dir / "partial.tmp"
    sentinel.write_text("interrupted", encoding="utf-8")
    manifest.chapters["0001"].status = "running"
    manifest.chapters["0001"].error = "interrupted during previous run"
    manifest.chapters["0001"].chapter_run_dir = str(chapter_dir)
    write_batch_manifest(result.manifest_path, manifest)

    resumed = resume_batch_pipeline(result.run_dir, provider_mode="offline")

    assert resumed.manifest.chapters["0001"].status == "packaged"
    assert resumed.manifest.chapters["0001"].error is None
    assert resumed.manifest.summary.incomplete == 0
    assert not sentinel.exists()


def test_resume_batch_pipeline_retry_review_required_replaces_existing_chapter_run(tmp_path: Path) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path)
    result = run_batch_pipeline(
        story_yaml,
        chapters=parse_chapter_selection("0001"),
        provider_mode="offline",
        run_id="resume_review_replace",
        overwrite=True,
    )
    manifest = load_batch_manifest(result.manifest_path)
    chapter_dir = result.run_dir / "chapters" / "0001"
    sentinel = chapter_dir / "old_review.tmp"
    sentinel.write_text("old review run", encoding="utf-8")
    manifest.chapters["0001"].status = "review_required"
    manifest.chapters["0001"].final_findings = 1
    manifest.chapters["0001"].chapter_run_dir = str(chapter_dir)
    write_batch_manifest(result.manifest_path, manifest)

    resumed = resume_batch_pipeline(result.run_dir, provider_mode="offline", retry_review_required=True)

    assert resumed.manifest.chapters["0001"].status == "packaged"
    assert resumed.manifest.chapters["0001"].final_findings == 0
    assert not sentinel.exists()


def test_resume_batch_pipeline_skips_review_required_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path)
    result = run_batch_pipeline(
        story_yaml,
        chapters=parse_chapter_selection("0001-0002"),
        provider_mode="offline",
        run_id="review_skip",
        overwrite=True,
    )
    manifest = load_batch_manifest(result.manifest_path)
    manifest.chapters["0002"].status = "review_required"
    manifest.chapters["0002"].final_findings = 2
    write_batch_manifest(result.manifest_path, manifest)
    calls: list[str] = []

    def fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(str(kwargs.get("chapter_override")))
        raise AssertionError("review_required chapters should not rerun by default")

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(batch_module, "run_demo_pipeline", fail_if_called)

    resumed = resume_batch_pipeline(result.run_dir, provider_mode="offline")

    assert calls == []
    assert resumed.manifest.chapters["0002"].status == "review_required"
    assert resumed.manifest.summary.review_required == 1


def test_resume_batch_pipeline_can_retry_only_review_required_chapters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path)
    result = run_batch_pipeline(
        story_yaml,
        chapters=parse_chapter_selection("0001-0002"),
        provider_mode="offline",
        run_id="review_retry",
        overwrite=True,
    )
    manifest = load_batch_manifest(result.manifest_path)
    manifest.chapters["0002"].status = "review_required"
    manifest.chapters["0002"].final_findings = 2
    manifest.chapters["0002"].error = "old review failure"
    write_batch_manifest(result.manifest_path, manifest)
    calls: list[str] = []

    def fake_run_demo_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003
        chapter = str(kwargs["chapter_override"])
        calls.append(chapter)
        run_dir = Path(kwargs["runs_dir"]) / chapter
        final_dir = run_dir / "translated_final"
        final_dir.mkdir(parents=True, exist_ok=True)
        (final_dir / f"{chapter}.txt").write_text(
            f"Chapter {int(chapter)}\n\nRetried clean translation.",
            encoding="utf-8",
        )
        report_path = run_dir / "report.html"
        report_path.write_text("<html></html>", encoding="utf-8")
        summary = QASummary(total_findings=0)
        qa = QAReport(
            run_id=chapter,
            story_slug="public_batch",
            chapter=chapter,
            findings=[],
            summary=summary,
            score=100,
        )
        metrics = EvalMetrics(mode="final", score=100)
        return PipelineResult(
            run_dir=run_dir,
            report_path=report_path,
            qa_source=qa,
            qa_baseline=qa,
            qa_glossary=qa,
            qa_final=qa,
            baseline_metrics=metrics,
            glossary_metrics=metrics,
            final_metrics=metrics,
        )

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(batch_module, "run_demo_pipeline", fake_run_demo_pipeline)

    resumed = resume_batch_pipeline(result.run_dir, provider_mode="offline", retry_review_required=True)

    assert calls == ["0002"]
    assert resumed.manifest.chapters["0001"].status == "packaged"
    assert resumed.manifest.chapters["0002"].status == "packaged"
    assert resumed.manifest.chapters["0002"].error is None
    assert resumed.manifest.summary.packaged == 2
    assert resumed.artifact_qa is not None
    assert resumed.artifact_qa.passed is True


def test_live_provider_fallback_circuit_breaker_skips_repeated_failed_judge_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001", "0002"])
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    calls = {"judge": 0}

    class FailingJudgeProvider:
        provider_name = "openai"
        model_name = "test-model"
        call_records: list[object] = []

        def judge(
            self,
            *,
            source_text: str,
            candidates: list[object],
            glossary: object,
            seed: int,
        ) -> object:
            calls["judge"] += 1
            raise LLMProviderUnavailable("402 Insufficient Balance")

    import agentic_translation.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "get_judge_provider", lambda *args, **kwargs: FailingJudgeProvider())

    result = run_batch_pipeline(
        story_yaml,
        chapters=["0001", "0002"],
        provider_mode="live",
        translation_provider_name="offline",
        judge_provider_name="openai",
        repair_provider_name="offline",
        record_cache=True,
        cache_dir=tmp_path / "cache",
        model_name="test-model",
        allow_live_provider_fallback=True,
        run_id="fallback_circuit",
        overwrite=True,
    )

    assert calls["judge"] == 1
    assert result.manifest.summary.packaged == 2
    first_reason = result.manifest.chapters["0001"].patch_attempts[0].reason
    second_reason = result.manifest.chapters["0002"].patch_attempts[0].reason
    assert "Live judge provider failed" in first_reason
    assert "previous live judge provider failure" in second_reason


def test_resume_batch_pipeline_can_limit_retry_to_selected_chapters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001", "0002", "0003"])
    result = run_batch_pipeline(
        story_yaml,
        chapters=parse_chapter_selection("0001-0003"),
        provider_mode="offline",
        run_id="review_retry_selected",
        overwrite=True,
    )
    manifest = load_batch_manifest(result.manifest_path)
    for chapter in ["0002", "0003"]:
        manifest.chapters[chapter].status = "review_required"
        manifest.chapters[chapter].final_findings = 2
    write_batch_manifest(result.manifest_path, manifest)
    calls: list[str] = []

    def fake_run_demo_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003
        chapter = str(kwargs["chapter_override"])
        calls.append(chapter)
        run_dir = Path(kwargs["runs_dir"]) / chapter
        final_dir = run_dir / "translated_final"
        final_dir.mkdir(parents=True, exist_ok=True)
        (final_dir / f"{chapter}.txt").write_text(
            f"Chapter {int(chapter)}\n\nSelected retry clean translation.",
            encoding="utf-8",
        )
        report_path = run_dir / "report.html"
        report_path.write_text("<html></html>", encoding="utf-8")
        summary = QASummary(total_findings=0)
        qa = QAReport(
            run_id=chapter,
            story_slug="public_batch",
            chapter=chapter,
            findings=[],
            summary=summary,
            score=100,
        )
        metrics = EvalMetrics(mode="final", score=100)
        return PipelineResult(
            run_dir=run_dir,
            report_path=report_path,
            qa_source=qa,
            qa_baseline=qa,
            qa_glossary=qa,
            qa_final=qa,
            baseline_metrics=metrics,
            glossary_metrics=metrics,
            final_metrics=metrics,
        )

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(batch_module, "run_demo_pipeline", fake_run_demo_pipeline)

    resumed = resume_batch_pipeline(
        result.run_dir,
        provider_mode="offline",
        retry_review_required=True,
        chapters=["0003"],
    )

    assert calls == ["0003"]
    assert resumed.manifest.chapters["0001"].status == "packaged"
    assert resumed.manifest.chapters["0002"].status == "review_required"
    assert resumed.manifest.chapters["0003"].status == "packaged"
    assert resumed.manifest.summary.packaged == 2
    assert resumed.manifest.summary.review_required == 1


def test_resume_batch_pipeline_rejects_selected_chapter_not_in_manifest(tmp_path: Path) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path)
    result = run_batch_pipeline(
        story_yaml,
        chapters=parse_chapter_selection("0001"),
        provider_mode="offline",
        run_id="review_retry_missing",
        overwrite=True,
    )

    with pytest.raises(ValueError, match="not in the batch manifest"):
        resume_batch_pipeline(result.run_dir, provider_mode="offline", chapters=["9999"])


def test_resume_batch_pipeline_reuses_manifest_model_when_model_option_omitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])
    run_dir = tmp_path / "runs" / "stored_model_resume"
    manifest = BatchManifest.create(
        run_id="stored_model_resume",
        story_slug="public_batch",
        title="Public Batch",
        story_yaml=story_yaml,
        chapters=["0001"],
        mode="live",
        providers={
            "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
            "judge": ProviderLabel(provider="openai", model="explicit-model"),
            "repair": ProviderLabel(provider="offline", model="offline-patch-v1"),
        },
        run_dir=run_dir,
    )
    write_batch_manifest(run_dir / "batch_manifest.json", manifest)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)
    captured: dict[str, object] = {}

    def fake_run_demo_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003
        captured.update(kwargs)
        chapter_run_dir = run_dir / "chapters" / "0001"
        final_path = chapter_run_dir / "translated_final" / "0001.txt"
        final_path.parent.mkdir(parents=True)
        final_path.write_text("Chapter 1\n\nHeavenly Dao.", encoding="utf-8")
        qa = QAReport(
            run_id="0001",
            story_slug="public_batch",
            chapter="0001",
            findings=[],
            summary=QASummary(),
            score=100,
        )
        metrics = EvalMetrics(mode="final")
        return PipelineResult(
            run_dir=chapter_run_dir,
            report_path=chapter_run_dir / "report.html",
            qa_source=qa,
            qa_baseline=qa,
            qa_glossary=qa,
            qa_final=qa,
            baseline_metrics=metrics,
            glossary_metrics=metrics,
            final_metrics=metrics,
        )

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(batch_module, "run_demo_pipeline", fake_run_demo_pipeline)

    resumed = resume_batch_pipeline(run_dir, record_cache=True, cache_dir=tmp_path / "cache", force=True)

    assert captured["model_name"] == "explicit-model"
    assert resumed.manifest.run_config is not None
    assert resumed.manifest.run_config.provider_mode == "live"
    assert resumed.manifest.run_config.judge_provider == "openai"
    assert resumed.manifest.run_config.record_cache is True
    assert resumed.manifest.run_config.cache_dir == str(tmp_path / "cache")
    assert resumed.manifest.run_config.model_name == "explicit-model"


def test_resume_batch_pipeline_updates_run_config_when_promoting_offline_review_to_live(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])
    result = run_batch_pipeline(
        story_yaml,
        chapters=["0001"],
        provider_mode="offline",
        run_id="promote_to_live",
        overwrite=True,
        skip_epub=True,
    )
    manifest = load_batch_manifest(result.manifest_path)
    manifest.chapters["0001"].status = "review_required"
    write_batch_manifest(result.manifest_path, manifest)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("AGENTIC_TRANSLATION_MODEL", "live-model")

    def fake_run_demo_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003
        chapter = str(kwargs["chapter_override"])
        chapter_run_dir = result.run_dir / "chapters" / chapter
        final_path = chapter_run_dir / "translated_final" / f"{chapter}.txt"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_text("Chapter 1\n\nHeavenly Dao.", encoding="utf-8")
        qa = QAReport(
            run_id=chapter,
            story_slug="public_batch",
            chapter=chapter,
            findings=[],
            summary=QASummary(),
            score=100,
        )
        metrics = EvalMetrics(mode="final")
        return PipelineResult(
            run_dir=chapter_run_dir,
            report_path=chapter_run_dir / "report.html",
            qa_source=qa,
            qa_baseline=qa,
            qa_glossary=qa,
            qa_final=qa,
            baseline_metrics=metrics,
            glossary_metrics=metrics,
            final_metrics=metrics,
        )

    import agentic_translation.batch as batch_module

    monkeypatch.setattr(batch_module, "run_demo_pipeline", fake_run_demo_pipeline)

    promoted = resume_batch_pipeline(
        result.run_dir,
        provider_mode="live",
        judge_provider_name="openai",
        record_cache=True,
        cache_dir=tmp_path / "cache",
        retry_review_required=True,
    )

    assert promoted.manifest.mode == "live"
    assert promoted.manifest.providers["judge"].provider == "openai"
    assert promoted.manifest.run_config is not None
    assert promoted.manifest.run_config.provider_mode == "live"
    assert promoted.manifest.run_config.translation_provider == "offline"
    assert promoted.manifest.run_config.judge_provider == "openai"
    assert promoted.manifest.run_config.repair_provider == "offline"
    assert promoted.manifest.run_config.record_cache is True
    assert promoted.manifest.run_config.cache_dir == str(tmp_path / "cache")
    assert promoted.manifest.run_config.model_name == "live-model"


def test_fake_live_batch_runs_three_chapters_through_model_backed_repair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from agentic_translation.models import StoryConfig
    from agentic_translation.providers_offline import OfflineJudgeProvider, OfflineRepairProvider

    class FakeLiveTranslationProvider:
        provider_name = "openai"
        model_name = "fake-live-translation"

        def translate(self, source_text: str, *, story: StoryConfig, glossary, mode: str) -> str:
            chapter = int(story.chapter_ids[0])
            return (
                f"Chapter {chapter}: The Simulator Starts\n\n"
                "[Simulator Started]\n\n"
                "The Heavenly Dao split open above the city.\n\n"
                "Lin Che looked at the panel and whispered, \"Begin simulation.\"\n\n"
                "remaining uses: 3"
            )

    class FakeLiveJudgeProvider(OfflineJudgeProvider):
        provider_name = "openai"
        model_name = "fake-live-judge"

    class FakeLiveRepairProvider(OfflineRepairProvider):
        provider_name = "openai"
        model_name = "fake-live-repair"

    import agentic_translation.batch as batch_module
    import agentic_translation.pipeline as pipeline_module

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("AGENTIC_TRANSLATION_MODEL", "fake-model")
    monkeypatch.setattr(batch_module, "get_translation_provider", lambda *args, **kwargs: FakeLiveTranslationProvider())
    monkeypatch.setattr(batch_module, "get_judge_provider", lambda *args, **kwargs: FakeLiveJudgeProvider())
    monkeypatch.setattr(batch_module, "get_repair_provider", lambda *args, **kwargs: FakeLiveRepairProvider())
    monkeypatch.setattr(pipeline_module, "get_translation_provider", lambda *args, **kwargs: FakeLiveTranslationProvider())
    monkeypatch.setattr(pipeline_module, "get_judge_provider", lambda *args, **kwargs: FakeLiveJudgeProvider())
    monkeypatch.setattr(pipeline_module, "get_repair_provider", lambda *args, **kwargs: FakeLiveRepairProvider())
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001", "0002", "0003"])

    result = run_batch_pipeline(
        story_yaml,
        chapters=parse_chapter_selection("0001-0003"),
        provider_mode="live",
        translation_provider_name="openai",
        judge_provider_name="openai",
        repair_provider_name="openai",
        record_cache=True,
        cache_dir=tmp_path / "cache",
        run_id="fake_live_batch",
        overwrite=True,
    )

    assert result.manifest.summary.packaged == 3
    assert result.manifest.run_config is not None
    assert result.manifest.run_config.provider_mode == "live"
    assert result.manifest.run_config.translation_provider == "openai"
    assert result.manifest.run_config.judge_provider == "openai"
    assert result.manifest.run_config.repair_provider == "openai"
    assert result.manifest.run_config.record_cache is True
    assert result.manifest.run_config.cache_dir == str(tmp_path / "cache")
    assert result.manifest.run_config.model_name == "fake-model"
    assert result.manifest.providers["translation"].provider == "openai"
    assert result.artifact_qa is not None
    assert result.artifact_qa.passed is True
    for chapter_run in result.manifest.chapters.values():
        assert chapter_run.final_score == 100
        assert chapter_run.status == "packaged"


def test_fake_live_batch_allows_translation_only_without_agentic_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from agentic_translation.models import ProviderCallRecord, StoryConfig

    class FakeLiveTranslationProvider:
        provider_name = "deepseek"
        model_name = "deepseek-chat"

        def __init__(self) -> None:
            self.call_records: list[ProviderCallRecord] = []

        def translate(self, source_text: str, *, story: StoryConfig, glossary, mode: str) -> str:
            self.call_records.append(
                ProviderCallRecord(
                    role="translation",
                    namespace="translation",
                    provider=self.provider_name,
                    model=self.model_name,
                    payload_sha256=f"payload-{mode}-{story.chapter_ids[0]}",
                    response_sha256=f"response-{mode}-{story.chapter_ids[0]}",
                    cache_file=f"translation_{mode}_{story.chapter_ids[0]}.json",
                    cache_hit=False,
                )
            )
            chapter = int(story.chapter_ids[0])
            return (
                f"Chapter {chapter}: The Simulator Starts\n\n"
                "[Simulator Started]\n\n"
                "The Heavenly Dao split open above the city.\n\n"
                "Lin Che looked at the panel and whispered, \"Begin simulation.\"\n\n"
                "remaining uses: 3"
            )

    import agentic_translation.batch as batch_module
    import agentic_translation.pipeline as pipeline_module

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setattr(batch_module, "get_translation_provider", lambda *args, **kwargs: FakeLiveTranslationProvider())
    monkeypatch.setattr(pipeline_module, "get_translation_provider", lambda *args, **kwargs: FakeLiveTranslationProvider())
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])

    result = run_batch_pipeline(
        story_yaml,
        chapters=["0001"],
        provider_mode="live",
        translation_provider_name="deepseek",
        judge_provider_name="offline",
        repair_provider_name="offline",
        record_cache=True,
        cache_dir=tmp_path / "cache",
        model_name="deepseek-chat",
        run_id="translation_only",
        overwrite=True,
    )
    inspection = build_batch_inspection_report(result.manifest)

    assert result.manifest.summary.packaged == 1
    assert result.manifest.run_config is not None
    assert result.manifest.run_config.translation_provider == "deepseek"
    assert result.manifest.run_config.judge_provider == "offline"
    assert result.manifest.run_config.repair_provider == "offline"
    assert result.manifest.chapters["0001"].provider_calls
    assert inspection.agentic_evidence.agentic_claim_supported is False
    assert inspection.agentic_evidence.configured_model_roles == ["translation"]
    assert inspection.agentic_evidence.observed_agentic_roles == []
    assert "translation provider calls were recorded" in inspection.agentic_evidence.reason


def test_live_translation_fallback_is_reported_as_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from agentic_translation.providers_llm import LLMProviderUnavailable

    class BalanceErrorTranslationProvider:
        provider_name = "deepseek"
        model_name = "deepseek-chat"
        call_records: list[object] = []

        def translate(self, source_text: str, *, story: StoryConfig, glossary, mode: str) -> str:
            raise LLMProviderUnavailable("402 Insufficient Balance")

    import agentic_translation.batch as batch_module
    import agentic_translation.pipeline as pipeline_module

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setattr(batch_module, "get_translation_provider", lambda *args, **kwargs: BalanceErrorTranslationProvider())
    monkeypatch.setattr(pipeline_module, "get_translation_provider", lambda *args, **kwargs: BalanceErrorTranslationProvider())
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001"])

    result = run_batch_pipeline(
        story_yaml,
        chapters=["0001"],
        provider_mode="live",
        translation_provider_name="deepseek",
        judge_provider_name="offline",
        repair_provider_name="offline",
        record_cache=True,
        cache_dir=tmp_path / "cache",
        model_name="deepseek-chat",
        allow_live_provider_fallback=True,
        run_id="translation_fallback",
        overwrite=True,
    )
    inspection = build_batch_inspection_report(result.manifest)

    assert result.manifest.summary.packaged == 1
    assert len(inspection.provider_failures) == 1
    assert inspection.provider_failures[0].chapter == "0001"
    assert inspection.provider_failures[0].role == "translation"
    assert inspection.provider_failures[0].provider == "deepseek"
    assert inspection.provider_failures[0].model == "deepseek-chat"
    assert inspection.provider_failures[0].fallback_used is True
    assert "Insufficient Balance" in inspection.provider_failures[0].reason


def test_live_translation_fallback_circuit_breaker_skips_later_translation_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from agentic_translation.providers_llm import LLMProviderUnavailable

    call_count = 0

    class BalanceErrorTranslationProvider:
        provider_name = "deepseek"
        model_name = "deepseek-chat"
        call_records: list[object] = []

        def translate(self, source_text: str, *, story: StoryConfig, glossary, mode: str) -> str:
            nonlocal call_count
            call_count += 1
            raise LLMProviderUnavailable("402 Insufficient Balance")

    import agentic_translation.batch as batch_module
    import agentic_translation.pipeline as pipeline_module

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setattr(batch_module, "get_translation_provider", lambda *args, **kwargs: BalanceErrorTranslationProvider())
    monkeypatch.setattr(pipeline_module, "get_translation_provider", lambda *args, **kwargs: BalanceErrorTranslationProvider())
    story_yaml = _write_public_batch_fixture(tmp_path, chapters=["0001", "0002"])

    result = run_batch_pipeline(
        story_yaml,
        chapters=["0001", "0002"],
        provider_mode="live",
        translation_provider_name="deepseek",
        judge_provider_name="offline",
        repair_provider_name="offline",
        record_cache=True,
        cache_dir=tmp_path / "cache",
        model_name="deepseek-chat",
        allow_live_provider_fallback=True,
        run_id="translation_fallback_circuit",
        overwrite=True,
    )
    inspection = build_batch_inspection_report(result.manifest)

    assert call_count == 1
    assert result.manifest.summary.packaged == 2
    assert len(inspection.provider_failures) == 2
    assert all(failure.role == "translation" for failure in inspection.provider_failures)
    assert "previous live translation provider failure" in inspection.provider_failures[1].reason


def test_build_panel_report_shows_extra_final_panels(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "panel_probe"
    chapter_dir = run_dir / "chapters" / "0001"
    source_path = chapter_dir / "source" / "0001.txt"
    final_path = chapter_dir / "translated_final" / "0001.txt"
    source_path.parent.mkdir(parents=True)
    final_path.parent.mkdir(parents=True)
    source_path.write_text("第1章\n\n【一】\n\n【二】\n", encoding="utf-8")
    final_path.write_text("Chapter 1\n\n[One]\n\n[Two]\n\n[Extra]\n", encoding="utf-8")
    qa = QAReport(
        run_id="0001",
        story_slug="panel_probe",
        chapter="0001",
        findings=[
            QAFinding(
                check_id="system_panel_count",
                severity="warning",
                message="System/panel count differs.",
                location=QALocation(chapter="0001"),
                found="3",
                expected="2",
                auto_repairable=True,
            )
        ],
        summary=QASummary(total_findings=1, warning_count=1, by_check={"system_panel_count": 1}),
        panel_count=3,
        score=88,
    )
    (chapter_dir / "qa_final.json").write_text(qa.model_dump_json(indent=2), encoding="utf-8")
    manifest = BatchManifest.create(
        run_id="panel_probe",
        story_slug="panel_probe",
        title="Panel Probe",
        story_yaml=tmp_path / "story.yaml",
        chapters=["0001"],
        mode="offline",
        providers={},
        run_dir=run_dir,
    )
    chapter_run = manifest.chapters["0001"]
    chapter_run.status = "review_required"
    stale_source_path = tmp_path / "stale_source.txt"
    stale_source_path.write_text("第0001章\n\n天道\n", encoding="utf-8")
    chapter_run.source_path = str(stale_source_path)
    chapter_run.final_path = str(final_path)
    chapter_run.chapter_run_dir = str(chapter_dir)
    manifest.refresh_summary()
    write_batch_manifest(run_dir / "batch_manifest.json", manifest)

    report = build_panel_report(run_dir)
    markdown = render_panel_report_markdown(report)

    assert report.summary.total_chapters == 1
    assert report.summary.mismatch_chapters == 1
    assert report.chapters[0].source_count == 2
    assert report.chapters[0].final_count == 3
    assert report.chapters[0].rows[-1].status == "extra_final"
    assert "[Extra]" in markdown
    assert "Extra Final Panels" in markdown
