from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner
import yaml

from agentic_translation.cli import app
from agentic_translation.models import AgentAttempt, AgenticEvidence, ArtifactQAReport, BatchInspectionReport, BatchLiveProofResult, BatchManifest, BatchPipelineResult, BatchProofReport, BatchRunConfig, BatchSummary, GlossaryUpdateApplication, GlossaryUpdateApplicationSummary, GlossaryUpdatePassResult, ManualReviewRecord, ManualTextReplacementResult, PanelNormalizationItem, PanelNormalizationResult, PatchAttempt, ProviderCallRecord, ProviderLabel, QAFinding, QALocation, QAReport, QASummary, RepairDecision, RepairPatch
from agentic_translation.providers_llm import LiveProviderProbeResult, ResponseCache


def test_batch_cli_does_not_expose_glossary_bridge() -> None:
    result = CliRunner().invoke(app, ["batch", "--help"])

    assert result.exit_code == 0
    assert "bridge-glossary" not in result.output


def test_live_provider_mode_rejects_all_offline_providers() -> None:
    result = CliRunner().invoke(
        app,
        [
            "demo",
            "run",
            "samples/public_demo/story.yaml",
            "--provider-mode",
            "live",
            "--translation-provider",
            "offline",
            "--judge-provider",
            "offline",
            "--repair-provider",
            "offline",
            "--run-id",
            "bad_live",
            "--overwrite",
        ],
    )

    assert result.exit_code != 0
    assert "live provider mode requires" in result.output.lower()


def test_replay_cache_is_gitignored_and_documented() -> None:
    assert ".agentic_cache/" in Path(".gitignore").read_text(encoding="utf-8")
    assert ".agentic_cache" in Path("README.md").read_text(encoding="utf-8")
    assert "should not be committed" in Path("README.md").read_text(encoding="utf-8").lower()


def test_provider_env_files_are_gitignored_and_documented() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert ".env" in gitignore
    assert ".env.local" in gitignore
    assert "agentic.env" in gitignore
    assert "global_env" in gitignore
    assert "--env-file" in Path("README.md").read_text(encoding="utf-8")


def test_cache_inspect_human_output_reports_integrity_failure(tmp_path: Path) -> None:
    ResponseCache(tmp_path).save(
        "judge",
        {"payload": "judge"},
        {"selected_candidate_id": "candidate_b"},
        metadata={"provider": "openai", "model": "gpt-test"},
    )
    cache_file = next(tmp_path.glob("judge_*.json"))
    cache_file.write_text('{"selected_candidate_id": "candidate_a"}', encoding="utf-8")

    cli_result = CliRunner().invoke(app, ["cache", "inspect", str(tmp_path)])

    assert cli_result.exit_code == 0
    assert "Integrity: failed" in cli_result.output
    assert "response_digest_mismatch" in cli_result.output


def _batch_cli_result_with_status(
    tmp_path: Path,
    status: str,
    *,
    artifact_qa_passed: bool = True,
) -> BatchPipelineResult:
    run_dir = tmp_path / f"batch_{status}"
    run_dir.mkdir()
    manifest_path = run_dir / "batch_manifest.json"
    manifest = BatchManifest.create(
        run_id=f"batch_{status}",
        story_slug="story",
        title="Story",
        story_yaml=Path("story.yaml"),
        chapters=["0001"],
        mode="offline",
        providers={},
        run_dir=run_dir,
    )
    manifest.chapters["0001"].status = status
    artifact_qa = ArtifactQAReport(
        expected_chapters=1,
        passed=artifact_qa_passed,
        failures=[] if artifact_qa_passed else ["TXT contains Chinese residue."],
    )
    manifest.artifact_qa = artifact_qa
    manifest.refresh_summary()
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return BatchPipelineResult(
        run_dir=run_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        artifact_qa=artifact_qa,
    )


def _batch_live_proof_cli_result(tmp_path: Path) -> BatchLiveProofResult:
    story_yaml = Path("samples/public_demo/story.yaml")

    def make_result(run_id: str, mode: str) -> BatchPipelineResult:
        run_dir = tmp_path / run_id
        manifest = BatchManifest.create(
            run_id=run_id,
            story_slug="public_demo",
            title="Public Demo",
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

    live_result = make_result("live_probe", "live")
    replay_result = make_result("live_probe_replay", "replay")

    def proof_for(result: BatchPipelineResult) -> BatchProofReport:
        evidence = AgenticEvidence(
            mode=result.manifest.mode,
            configured_model_roles=["judge"],
            observed_agentic_roles=["judge"],
            candidate_selection_repairs=1,
            cache_available=True,
            cache_entries=1,
            cache_required_namespaces=["judge"],
            cache_integrity_passed=True,
            provider_call_records=1,
            cache_verified_call_records=1,
            verified_candidate_selection_records=1,
            replay_cache_ready=True,
            agentic_claim_supported=True,
            reason="verified",
        )
        inspection = BatchInspectionReport(
            run_id=result.manifest.run_id,
            story_slug=result.manifest.story_slug,
            run_dir=result.manifest.run_dir,
            ready_for_delivery=True,
            blocker_count=0,
            summary=result.manifest.summary,
            agentic_evidence=evidence,
            run_config=result.manifest.run_config,
        )
        return BatchProofReport(
            run_id=result.manifest.run_id,
            story_slug=result.manifest.story_slug,
            run_dir=result.manifest.run_dir,
            proof_passed=True,
            gates={"delivery": True, "agentic": True, "replayable": True},
            inspection=inspection,
        )

    return BatchLiveProofResult(
        story_yaml=str(story_yaml),
        chapters=["0001"],
        proof_passed=True,
        live_result=live_result,
        live_proof=proof_for(live_result),
        replay_result=replay_result,
        replay_proof=proof_for(replay_result),
    )


def _write_review_queue_cli_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "review_queue_run"
    chapter_dir = run_dir / "chapters" / "0001"
    chapter_dir.mkdir(parents=True)
    source_path = chapter_dir / "source" / "0001.txt"
    source_path.parent.mkdir()
    source_path.write_text("第1章\n\n天道 appeared.", encoding="utf-8")
    final_path = chapter_dir / "translated_final" / "0001.txt"
    final_path.parent.mkdir()
    final_path.write_text("Chapter 1\n\nNeeds term.", encoding="utf-8")
    finding = QAFinding(
        check_id="glossary_required",
        severity="warning",
        message="Canonical glossary term is missing.",
        location=QALocation(chapter="0001", paragraph_index=1, snippet="Needs term."),
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
        run_id="review_queue_run",
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
    manifest.chapters["0001"].final_score = 94
    manifest.chapters["0001"].final_findings = 1
    (run_dir / "batch_manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return run_dir


def test_open_latest_missing_runs_dir_prints_clean_error(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["open-latest", "--runs-dir", str(tmp_path / "missing")])

    assert result.exit_code == 1
    assert "No runs found." in result.output
    assert "Traceback" not in result.output

def test_doctor_public_offline_passes() -> None:
    result = CliRunner().invoke(
        app,
        [
            "doctor",
            "samples/public_demo/story.yaml",
            "--chapters",
            "0001",
            "--provider-mode",
            "offline",
        ],
    )

    assert result.exit_code == 0
    assert "Preflight" in result.output
    assert "source_chapters" in result.output


def test_doctor_loads_live_provider_env_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    env_file = tmp_path / ".env.deepseek"
    env_file.write_text(
        "DEEPSEEK_API_KEY" + "=secret-do-not-print\nDEEPSEEK_MODEL=deepseek-chat\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "--env-file",
            str(env_file),
            "doctor",
            "samples/public_demo/story.yaml",
            "--chapters",
            "0001",
            "--provider-mode",
            "live",
            "--translation-provider",
            "offline",
            "--judge-provider",
            "deepseek",
            "--repair-provider",
            "offline",
            "--record-cache",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 0
    assert "Live provider credentials/config are present" in result.output
    assert "secret-do-not-print" not in result.output


def test_doctor_loads_env_file_from_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    env_file = tmp_path / "agentic.env"
    env_file.write_text(
        "DEEPSEEK_API_KEY" + "=secret-do-not-print\nDEEPSEEK_MODEL=deepseek-chat\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "doctor",
            "samples/public_demo/story.yaml",
            "--chapters",
            "0001",
            "--provider-mode",
            "live",
            "--translation-provider",
            "offline",
            "--judge-provider",
            "deepseek",
            "--repair-provider",
            "offline",
            "--record-cache",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        env={"AGENTIC_TRANSLATION_ENV_FILE": str(env_file)},
    )

    assert result.exit_code == 0
    assert "Live provider credentials/config are present" in result.output
    assert "secret-do-not-print" not in result.output


def test_provider_probe_deepseek_uses_cheap_defaults(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    captured: dict[str, object] = {}

    def fake_probe_live_provider(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return LiveProviderProbeResult(
            provider="deepseek",
            mode="live",
            model="deepseek-chat",
            cache_dir=str(kwargs["cache_dir"]),
            cache_hit=False,
            cache_file="probe_abc.json",
            response={"ok": True, "message": "pong"},
        )

    monkeypatch.setattr("agentic_translation.cli.probe_live_provider", fake_probe_live_provider)

    result = CliRunner().invoke(
        app,
        [
            "provider-probe",
            "deepseek",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 0
    assert captured["provider_name"] == "deepseek"
    assert captured["provider_mode"] == "live"
    assert captured["model_name"] == "deepseek-chat"
    assert captured["record_cache"] is True
    assert captured["cache_dir"] == tmp_path / "cache"
    assert "deepseek" in result.output
    assert "deepseek-chat" in result.output
    assert "probe_abc.json" in result.output


def test_provider_probe_missing_key_prints_clean_error(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)

    result = CliRunner().invoke(
        app,
        [
            "provider-probe",
            "deepseek",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 1
    assert "DEEPSEEK_API_KEY is required" in result.output
    assert "sk-" not in result.output
    assert "Traceback" not in result.output


def test_cache_inspect_outputs_json_summary(tmp_path: Path) -> None:
    from agentic_translation.providers_llm import ResponseCache

    ResponseCache(tmp_path).save(
        "judge",
        {"payload": "do not show me"},
        {"response": "do not show me either"},
        metadata={"provider": "openai", "model": "test-model"},
    )

    result = CliRunner().invoke(app, ["cache", "inspect", str(tmp_path), "--json"])
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["total_entries"] == 1
    assert payload["by_namespace"] == {"judge": 1}
    assert payload["entries"][0]["model"] == "test-model"
    assert "do not show me" not in result.output


def test_batch_review_outputs_json_queue(tmp_path: Path) -> None:
    run_dir = _write_review_queue_cli_run(tmp_path)

    result = CliRunner().invoke(app, ["batch", "review", str(run_dir), "--json"])
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["summary"]["total_items"] == 1
    assert payload["items"][0]["check_id"] == "glossary_required"
    assert payload["items"][0]["expected"] == "Heavenly Dao"
    assert payload["summary"]["chapters"] == ["0001"]
    assert payload["summary"]["chapter_selection"] == "0001"
    assert "天道 appeared" in payload["items"][0]["source_context"]
    assert "Needs term" in payload["items"][0]["final_context"]


def test_batch_review_can_print_chapter_selection_only(tmp_path: Path) -> None:
    run_dir = _write_review_queue_cli_run(tmp_path)

    result = CliRunner().invoke(app, ["batch", "review", str(run_dir), "--chapters-only"])

    assert result.exit_code == 0
    assert result.output == "0001\n"


def test_batch_review_chapters_only_stays_shell_safe_with_write(tmp_path: Path) -> None:
    run_dir = _write_review_queue_cli_run(tmp_path)

    result = CliRunner().invoke(app, ["batch", "review", str(run_dir), "--write", "--chapters-only"])

    assert result.exit_code == 0
    assert result.output == "0001\n"
    assert (run_dir / "review_queue.json").exists()
    assert (run_dir / "review_chapters.txt").read_text(encoding="utf-8") == "0001\n"


def test_batch_review_write_creates_review_queue_json(tmp_path: Path) -> None:
    run_dir = _write_review_queue_cli_run(tmp_path)

    result = CliRunner().invoke(app, ["batch", "review", str(run_dir), "--write"])

    assert result.exit_code == 0
    output_path = run_dir / "review_queue.json"
    assert output_path.exists()
    chapter_selection_path = run_dir / "review_chapters.txt"
    assert chapter_selection_path.read_text(encoding="utf-8") == "0001\n"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total_items"] == 1


def test_batch_review_outputs_markdown_queue(tmp_path: Path) -> None:
    run_dir = _write_review_queue_cli_run(tmp_path)

    result = CliRunner().invoke(app, ["batch", "review", str(run_dir), "--markdown"])

    assert result.exit_code == 0
    assert "# Review Queue: review_queue_run" in result.output
    assert "Chapter selector: `0001`" in result.output
    assert "## 0001 - glossary_required" in result.output
    assert "- Expected: `Heavenly Dao`" in result.output
    assert "天道 appeared" in result.output
    assert "Needs term." in result.output
    assert "{\n" not in result.output


def test_batch_review_write_markdown_creates_human_review_file(tmp_path: Path) -> None:
    run_dir = _write_review_queue_cli_run(tmp_path)

    result = CliRunner().invoke(app, ["batch", "review", str(run_dir), "--write-markdown"])

    assert result.exit_code == 0
    output_path = run_dir / "review_queue.md"
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "# Review Queue: review_queue_run" in content
    assert "## 0001 - glossary_required" in content
    assert "Heavenly Dao" in content
    assert "Source Context" in content
    assert "Final Context" in content
    assert "Wrote Markdown review queue" in result.output


def test_batch_glossary_report_outputs_json_gap_summary(tmp_path: Path) -> None:
    run_dir = _write_review_queue_cli_run(tmp_path)

    result = CliRunner().invoke(app, ["batch", "glossary-report", str(run_dir), "--json"])
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["summary"]["total_occurrences"] == 1
    assert payload["summary"]["term_count"] == 1
    assert payload["summary"]["chapter_selection"] == "0001"
    assert payload["gaps"][0]["found"] == "天道"
    assert payload["gaps"][0]["expected"] == "Heavenly Dao"
    assert "do not auto-patch from source-only evidence" in payload["gaps"][0]["suggested_action"]


def test_batch_glossary_report_write_creates_json_and_markdown(tmp_path: Path) -> None:
    run_dir = _write_review_queue_cli_run(tmp_path)

    result = CliRunner().invoke(app, ["batch", "glossary-report", str(run_dir), "--write"])

    assert result.exit_code == 0
    json_path = run_dir / "glossary_gap_report.json"
    markdown_path = run_dir / "glossary_gap_report.md"
    assert json_path.exists()
    assert markdown_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["term_count"] == 1
    assert "## `天道` -> `Heavenly Dao`" in markdown_path.read_text(encoding="utf-8")
    assert "Wrote glossary gap report" in result.output


def test_batch_glossary_update_plan_write_creates_json_and_markdown(tmp_path: Path) -> None:
    run_dir = _write_review_queue_cli_run(tmp_path)

    result = CliRunner().invoke(app, ["batch", "glossary-update-plan", str(run_dir), "--write"])

    assert result.exit_code == 0
    json_path = run_dir / "glossary_update_plan.json"
    markdown_path = run_dir / "glossary_update_plan.md"
    assert json_path.exists()
    assert markdown_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total_items"] == 1
    assert payload["items"][0]["suggested_line"].startswith("天道: Heavenly Dao")
    assert "# Glossary Update Plan: review_queue_run" in markdown_path.read_text(encoding="utf-8")
    assert "Wrote glossary update plan" in result.output


def test_batch_apply_glossary_update_plan_writes_json_result(tmp_path: Path) -> None:
    run_dir = _write_review_queue_cli_run(tmp_path)
    chapter_dir = run_dir / "chapters" / "0001"
    (chapter_dir / "translated_final" / "0001.txt").write_text("Chapter 1\n\nThe Dao stayed vague.", encoding="utf-8")
    qa = QAReport(
        run_id="0001",
        story_slug="story",
        chapter="0001",
        findings=[
            QAFinding(
                check_id="glossary_required",
                severity="warning",
                message="Canonical glossary term is missing.",
                location=QALocation(chapter="0001", paragraph_index=1, snippet="The Dao stayed vague."),
                found="天道",
                expected="Heavenly Dao",
            )
        ],
        summary=QASummary(total_findings=1, warning_count=1, by_check={"glossary_required": 1}),
        score=94,
    )
    (chapter_dir / "qa_final.json").write_text(qa.model_dump_json(indent=2), encoding="utf-8")
    glossary_path = tmp_path / "master_glossary.txt"
    glossary_path.write_text("天道 -> Heavenly Dao\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "batch",
            "apply-glossary-update-plan",
            str(run_dir),
            "--glossary",
            str(glossary_path),
            "--write",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dry_run"] is False
    assert payload["summary"]["changed_count"] == 1
    assert payload["items"][0]["status"] == "updated"
    assert "天道: Heavenly Dao" in glossary_path.read_text(encoding="utf-8")
    assert payload["backup_path"] is not None


def test_batch_glossary_pass_dry_run_outputs_json_without_writing(tmp_path: Path) -> None:
    run_dir = _write_review_queue_cli_run(tmp_path)
    chapter_dir = run_dir / "chapters" / "0001"
    (chapter_dir / "translated_final" / "0001.txt").write_text("Chapter 1\n\nThe Dao stayed vague.", encoding="utf-8")
    qa = QAReport(
        run_id="0001",
        story_slug="story",
        chapter="0001",
        findings=[
            QAFinding(
                check_id="glossary_required",
                severity="warning",
                message="Canonical glossary term is missing.",
                location=QALocation(chapter="0001", paragraph_index=1, snippet="The Dao stayed vague."),
                found="天道",
                expected="Heavenly Dao",
            )
        ],
        summary=QASummary(total_findings=1, warning_count=1, by_check={"glossary_required": 1}),
        score=94,
    )
    (chapter_dir / "qa_final.json").write_text(qa.model_dump_json(indent=2), encoding="utf-8")
    glossary_path = tmp_path / "master_glossary.txt"
    original = "天道 -> Heavenly Dao\n"
    glossary_path.write_text(original, encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "batch",
            "glossary-pass",
            str(run_dir),
            "--glossary",
            str(glossary_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["rerun_started"] is False
    assert payload["application"]["summary"]["changed_count"] == 1
    assert glossary_path.read_text(encoding="utf-8") == original


def test_batch_manual_edit_plan_write_creates_json_and_markdown(tmp_path: Path) -> None:
    run_dir = _write_review_queue_cli_run(tmp_path)

    result = CliRunner().invoke(app, ["batch", "manual-edit-plan", str(run_dir), "--write"])

    assert result.exit_code == 0
    json_path = run_dir / "manual_edit_plan.json"
    markdown_path = run_dir / "manual_edit_plan.md"
    assert json_path.exists()
    assert markdown_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["total_items"] == 1
    assert "# Manual Edit Plan: review_queue_run" in markdown_path.read_text(encoding="utf-8")
    assert "Wrote manual edit plan" in result.output


def test_batch_work_order_outputs_json_actions(tmp_path: Path) -> None:
    run_dir = _write_review_queue_cli_run(tmp_path)

    result = CliRunner().invoke(app, ["batch", "work-order", str(run_dir), "--json"])
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["summary"]["total_items"] == 1
    assert payload["summary"]["glossary_selection"] == "0001"
    assert payload["items"][0]["action"] == "glossary_triage"
    assert "batch glossary-report" in payload["commands"]["glossary_triage"]


def test_batch_work_order_write_creates_json_and_markdown(tmp_path: Path) -> None:
    run_dir = _write_review_queue_cli_run(tmp_path)

    result = CliRunner().invoke(app, ["batch", "work-order", str(run_dir), "--write"])

    assert result.exit_code == 0
    json_path = run_dir / "agentic_work_order.json"
    markdown_path = run_dir / "agentic_work_order.md"
    assert json_path.exists()
    assert markdown_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["items"][0]["action"] == "glossary_triage"
    assert "# Agent Work Order: review_queue_run" in markdown_path.read_text(encoding="utf-8")
    assert "Wrote agent work order" in result.output


def test_batch_execute_work_order_live_retry_passes_selected_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = _write_review_queue_cli_run(tmp_path)
    captured: dict[str, object] = {}

    def fake_execute_agent_work_order(run_dir_arg: Path, **kwargs):  # noqa: ANN202, ANN003
        captured["run_dir"] = run_dir_arg
        captured.update(kwargs)
        manifest = BatchManifest.create(
            run_id="executed",
            story_slug="story",
            title="Story",
            story_yaml=Path("story.yaml"),
            chapters=["0001"],
            mode="live",
            providers={},
            run_dir=run_dir_arg,
        )
        return BatchPipelineResult(
            run_dir=run_dir_arg,
            manifest_path=run_dir_arg / "batch_manifest.json",
            manifest=manifest,
        )

    import agentic_translation.cli as cli_module

    monkeypatch.setattr(cli_module, "execute_agent_work_order", fake_execute_agent_work_order)

    result = CliRunner().invoke(
        app,
        [
            "batch",
            "execute-work-order",
            str(run_dir),
            "--provider-mode",
            "live",
            "--translation-provider",
            "offline",
            "--judge-provider",
            "openai",
            "--repair-provider",
            "openai",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--model",
            "test-model",
            "--allow-live-provider-fallback",
            "--allow-review-required",
        ],
    )

    assert result.exit_code == 0
    assert captured["run_dir"] == run_dir
    assert captured["action"] == "live-retry"
    assert captured["provider_mode"] == "live"
    assert captured["translation_provider_name"] == "offline"
    assert captured["judge_provider_name"] == "openai"
    assert captured["repair_provider_name"] == "openai"
    assert captured["record_cache"] is True
    assert captured["cache_dir"] == tmp_path / "cache"
    assert captured["model_name"] == "test-model"
    assert captured["allow_live_provider_fallback"] is True
    assert captured["retry_review_required"] is True
    assert captured["write_proof"] is True


def test_batch_execute_work_order_reports_empty_live_retry(tmp_path: Path) -> None:
    run_dir = _write_review_queue_cli_run(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "batch",
            "execute-work-order",
            str(run_dir),
            "--provider-mode",
            "live",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--model",
            "test-model",
        ],
    )

    assert result.exit_code == 1
    assert "No live-retry chapters" in result.output


def test_batch_execute_work_order_dry_run_outputs_json_without_mutating(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)
    from tests.test_batch import _write_mixed_review_batch

    run_dir = _write_mixed_review_batch(tmp_path)
    manifest_path = run_dir / "batch_manifest.json"
    before = manifest_path.read_text(encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "batch",
            "execute-work-order",
            str(run_dir),
            "--dry-run",
            "--json",
            "--provider-mode",
            "live",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--model",
            "test-model",
        ],
    )
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["dry_run"] is True
    assert payload["would_mutate"] is False
    assert payload["chapters"] == ["0002", "0003"]
    assert payload["preflight_passed"] is True
    assert "batch execute-work-order" in payload["command"]
    assert payload["recommended_next_action"] == "execute_live_retry"
    assert payload["recommended_command"] == payload["execution_command"]
    assert "--dry-run" in payload["dry_run_command"]
    assert manifest_path.read_text(encoding="utf-8") == before


def test_batch_execute_work_order_dry_run_json_exits_nonzero_when_preflight_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)
    from tests.test_batch import _write_mixed_review_batch

    run_dir = _write_mixed_review_batch(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "batch",
            "execute-work-order",
            str(run_dir),
            "--dry-run",
            "--json",
            "--provider-mode",
            "live",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )
    payload = json.loads(result.output)

    assert result.exit_code == 1
    assert payload["dry_run"] is True
    assert payload["preflight_passed"] is False
    assert payload["recommended_next_action"] == "fix_preflight"
    assert payload["recommended_command"] == payload["dry_run_command"]
    assert any("OPENAI_API_KEY" in blocker for blocker in payload["preflight_blockers"])
    assert any(check["name"] == "env" and check["status"] == "fail" for check in payload["preflight_checks"])


def test_batch_execute_work_order_dry_run_cli_prints_recommendation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)
    from tests.test_batch import _write_mixed_review_batch

    run_dir = _write_mixed_review_batch(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "batch",
            "execute-work-order",
            str(run_dir),
            "--dry-run",
            "--provider-mode",
            "live",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--model",
            "test-model",
        ],
    )

    assert result.exit_code == 0
    assert "Recommended Next Action" in result.output
    assert "execute_live_retry" in result.output
    assert "Recommended command:" in result.output


def test_batch_execute_work_order_dry_run_write_preview_creates_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)
    from tests.test_batch import _write_mixed_review_batch

    run_dir = _write_mixed_review_batch(tmp_path)
    manifest_path = run_dir / "batch_manifest.json"
    before = manifest_path.read_text(encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "batch",
            "execute-work-order",
            str(run_dir),
            "--dry-run",
            "--write-preview",
            "--provider-mode",
            "live",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--model",
            "test-model",
        ],
    )

    assert result.exit_code == 0
    assert (run_dir / "agentic_execution_preview.json").exists()
    assert (run_dir / "agentic_execution_preview.md").exists()
    assert "Wrote execution preview" in result.output
    assert manifest_path.read_text(encoding="utf-8") == before


def test_batch_execute_work_order_preflights_before_non_dry_run_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)
    from tests.test_batch import _write_mixed_review_batch

    run_dir = _write_mixed_review_batch(tmp_path)
    manifest_path = run_dir / "batch_manifest.json"
    before = manifest_path.read_text(encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "batch",
            "execute-work-order",
            str(run_dir),
            "--provider-mode",
            "live",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--model",
            "test-model",
        ],
    )

    assert result.exit_code == 1
    assert "Preflight failed for work-order execution" in result.output
    assert "OPENAI_API_KEY" in result.output
    assert manifest_path.read_text(encoding="utf-8") == before


def test_doctor_json_exits_nonzero_for_live_missing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)

    result = CliRunner().invoke(
        app,
        [
            "doctor",
            "samples/public_demo/story.yaml",
            "--chapters",
            "0001",
            "--provider-mode",
            "live",
            "--translation-provider",
            "openai",
            "--judge-provider",
            "openai",
            "--repair-provider",
            "openai",
            "--json",
        ],
    )
    payload = json.loads(result.output)

    assert result.exit_code == 1
    assert payload["passed"] is False
    assert any(check["name"] == "env" and check["status"] == "fail" for check in payload["checks"])


def test_doctor_live_accepts_explicit_model_option(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)

    result = CliRunner().invoke(
        app,
        [
            "doctor",
            "samples/public_demo/story.yaml",
            "--chapters",
            "0001",
            "--provider-mode",
            "live",
            "--judge-provider",
            "openai",
            "--repair-provider",
            "offline",
            "--record-cache",
            "--cache-dir",
            str(tmp_path),
            "--model",
            "explicit-model",
            "--json",
        ],
    )
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["passed"] is True


def test_batch_run_requires_chapter_selection() -> None:
    result = CliRunner().invoke(app, ["batch", "run", "samples/public_demo/story.yaml"])

    assert result.exit_code != 0
    assert "Missing option" in result.output


def test_batch_run_forwards_tool_agent_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_batch_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003
        captured.update(kwargs)
        return _batch_cli_result_with_status(tmp_path, "packaged")

    monkeypatch.setattr("agentic_translation.cli.run_batch_pipeline", fake_run_batch_pipeline)

    result = CliRunner().invoke(
        app,
        [
            "batch",
            "run",
            "samples/public_demo/story.yaml",
            "--chapters",
            "0001",
            "--tool-agent",
        ],
    )

    assert result.exit_code == 0
    assert captured["tool_agent_enabled"] is True


def test_batch_run_term_consensus_implies_tool_agent_and_forwards_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_batch_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003
        captured.update(kwargs)
        return _batch_cli_result_with_status(tmp_path, "packaged")

    monkeypatch.setattr("agentic_translation.cli.run_batch_pipeline", fake_run_batch_pipeline)
    result = CliRunner().invoke(
        app,
        [
            "batch",
            "run",
            "samples/public_demo/story.yaml",
            "--chapters",
            "0001",
            "--term-consensus",
            "--openai-term-model",
            "gpt-term",
            "--deepseek-term-model",
            "deepseek-term",
            "--term-evaluator",
            "deepseek",
            "--term-confidence",
            "0.8",
        ],
    )

    assert result.exit_code == 0
    assert captured["tool_agent_enabled"] is True
    assert captured["repair_provider_name"] == "openai"
    assert captured["model_name"] == "gpt-term"
    config = captured["terminology_consensus"]
    assert config.enabled is True
    assert config.openai_model == "gpt-term"
    assert config.deepseek_model == "deepseek-term"
    assert config.evaluator_provider == "deepseek"
    assert config.confidence_threshold == 0.8


def test_batch_run_term_consensus_aligns_deepseek_repair_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_batch_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003
        captured.update(kwargs)
        return _batch_cli_result_with_status(tmp_path, "packaged")

    monkeypatch.setattr("agentic_translation.cli.run_batch_pipeline", fake_run_batch_pipeline)
    result = CliRunner().invoke(
        app,
        [
            "batch",
            "run",
            "samples/public_demo/story.yaml",
            "--chapters",
            "0001",
            "--term-consensus",
            "--repair-provider",
            "deepseek",
            "--openai-term-model",
            "gpt-term",
            "--deepseek-term-model",
            "deepseek-term",
        ],
    )

    assert result.exit_code == 0
    assert captured["repair_provider_name"] == "deepseek"
    assert captured["model_name"] == "deepseek-term"


def test_batch_resume_tool_agent_option_is_tri_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _batch_cli_result_with_status(tmp_path, "packaged")
    captured: dict[str, object] = {}

    def fake_resume_batch_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003
        captured.update(kwargs)
        return result

    monkeypatch.setattr("agentic_translation.cli.resume_batch_pipeline", fake_resume_batch_pipeline)

    enabled = CliRunner().invoke(app, ["batch", "resume", str(result.run_dir), "--tool-agent"])
    assert enabled.exit_code == 0
    assert captured["tool_agent_enabled"] is True

    captured.clear()
    disabled = CliRunner().invoke(app, ["batch", "resume", str(result.run_dir), "--no-tool-agent"])
    assert disabled.exit_code == 0
    assert captured["tool_agent_enabled"] is False

    captured.clear()
    inherited = CliRunner().invoke(app, ["batch", "resume", str(result.run_dir)])
    assert inherited.exit_code == 0
    assert captured["tool_agent_enabled"] is None


def test_batch_live_proof_forwards_tool_agent_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    proof_result = _batch_live_proof_cli_result(tmp_path)

    def fake_run_live_proof_pipeline(story_yaml: Path, **kwargs):  # noqa: ANN003
        captured["story_yaml"] = story_yaml
        captured.update(kwargs)
        return proof_result

    monkeypatch.setattr("agentic_translation.cli.run_live_proof_pipeline", fake_run_live_proof_pipeline)

    result = CliRunner().invoke(
        app,
        [
            "batch",
            "live-proof",
            "samples/public_demo/story.yaml",
            "--chapters",
            "0001",
            "--tool-agent",
        ],
    )

    assert result.exit_code == 0
    assert captured["tool_agent_enabled"] is True


def test_batch_live_proof_term_consensus_aligns_deepseek_repair_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    proof_result = _batch_live_proof_cli_result(tmp_path)

    def fake_run_live_proof_pipeline(story_yaml: Path, **kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return proof_result

    monkeypatch.setattr("agentic_translation.cli.run_live_proof_pipeline", fake_run_live_proof_pipeline)
    result = CliRunner().invoke(
        app,
        [
            "batch",
            "live-proof",
            "samples/public_demo/story.yaml",
            "--chapters",
            "0001",
            "--term-consensus",
            "--repair-provider",
            "deepseek",
            "--openai-term-model",
            "gpt-term",
            "--deepseek-term-model",
            "deepseek-term",
        ],
    )

    assert result.exit_code == 0
    assert captured["repair_provider_name"] == "deepseek"
    assert captured["model_name"] == "deepseek-term"


def test_batch_execute_work_order_forwards_tool_agent_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _batch_cli_result_with_status(tmp_path, "packaged")
    captured: dict[str, object] = {}

    def fake_execute_agent_work_order(run_dir: Path, **kwargs):  # noqa: ANN003
        captured["run_dir"] = run_dir
        captured.update(kwargs)
        return result

    monkeypatch.setattr("agentic_translation.cli.execute_agent_work_order", fake_execute_agent_work_order)

    cli_result = CliRunner().invoke(
        app,
        ["batch", "execute-work-order", str(result.run_dir), "--tool-agent"],
    )

    assert cli_result.exit_code == 0
    assert captured["tool_agent_enabled"] is True


def test_batch_run_exits_nonzero_when_review_required(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "agentic_translation.cli.run_batch_pipeline",
        lambda *args, **kwargs: _batch_cli_result_with_status(tmp_path, "review_required"),
    )

    result = CliRunner().invoke(app, ["batch", "run", "samples/public_demo/story.yaml", "--chapters", "0001"])

    assert result.exit_code == 1
    assert "review required" in result.output.lower()


def test_batch_run_can_allow_review_required_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "agentic_translation.cli.run_batch_pipeline",
        lambda *args, **kwargs: _batch_cli_result_with_status(tmp_path, "review_required"),
    )

    result = CliRunner().invoke(
        app,
        [
            "batch",
            "run",
            "samples/public_demo/story.yaml",
            "--chapters",
            "0001",
            "--allow-review-required",
        ],
    )

    assert result.exit_code == 0
    assert "review_required" in result.output


def test_batch_run_write_triage_writes_review_packet(
    monkeypatch,
    tmp_path: Path,
) -> None:  # noqa: ANN001
    result = _batch_cli_result_with_status(tmp_path, "review_required")
    triage_calls: list[Path] = []

    monkeypatch.setattr("agentic_translation.cli.run_batch_pipeline", lambda *args, **kwargs: result)
    monkeypatch.setattr(
        "agentic_translation.cli.write_batch_triage_artifacts",
        lambda run_dir: triage_calls.append(run_dir) or {"review_queue": "review_queue.json"},
    )

    cli_result = CliRunner().invoke(
        app,
        [
            "batch",
            "run",
            "samples/public_demo/story.yaml",
            "--chapters",
            "0001",
            "--allow-review-required",
            "--write-triage",
        ],
    )

    assert cli_result.exit_code == 0
    assert triage_calls == [result.run_dir]
    assert "triage artifact" in cli_result.output.lower()


def test_batch_run_exits_nonzero_when_chapter_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "agentic_translation.cli.run_batch_pipeline",
        lambda *args, **kwargs: _batch_cli_result_with_status(tmp_path, "failed"),
    )

    result = CliRunner().invoke(
        app,
        [
            "batch",
            "run",
            "samples/public_demo/story.yaml",
            "--chapters",
            "0001",
            "--allow-review-required",
        ],
    )

    assert result.exit_code == 1
    assert "failed" in result.output.lower()


def test_batch_run_exits_nonzero_when_aggregate_artifact_qa_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "agentic_translation.cli.run_batch_pipeline",
        lambda *args, **kwargs: _batch_cli_result_with_status(tmp_path, "packaged", artifact_qa_passed=False),
    )

    result = CliRunner().invoke(
        app,
        [
            "batch",
            "run",
            "samples/public_demo/story.yaml",
            "--chapters",
            "0001",
            "--allow-review-required",
        ],
    )

    assert result.exit_code == 1
    assert "artifact qa" in result.output.lower()


def test_batch_live_requires_record_cache_and_cache_dir() -> None:
    result = CliRunner().invoke(
        app,
        [
            "batch",
            "run",
            "samples/public_demo/story.yaml",
            "--chapters",
            "0001",
            "--provider-mode",
            "live",
            "--translation-provider",
            "openai",
            "--judge-provider",
            "openai",
            "--repair-provider",
            "openai",
        ],
    )

    assert result.exit_code == 1
    assert "record-cache" in result.output


def test_batch_live_requires_openai_env_when_cache_configured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)
    result = CliRunner().invoke(
        app,
        [
            "batch",
            "run",
            "samples/public_demo/story.yaml",
            "--chapters",
            "0001",
            "--provider-mode",
            "live",
            "--translation-provider",
            "openai",
            "--judge-provider",
            "openai",
            "--repair-provider",
            "openai",
            "--record-cache",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--run-id",
            "missing_env",
            "--overwrite",
        ],
    )

    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output


def test_batch_run_passes_explicit_model_option(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_batch_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003
        captured.update(kwargs)
        return _batch_cli_result_with_status(tmp_path, "packaged")

    monkeypatch.setattr("agentic_translation.cli.run_batch_pipeline", fake_run_batch_pipeline)

    result = CliRunner().invoke(
        app,
        [
            "batch",
            "run",
            "samples/public_demo/story.yaml",
            "--chapters",
            "0001",
            "--provider-mode",
            "live",
            "--translation-provider",
            "offline",
            "--judge-provider",
            "openai",
            "--repair-provider",
            "offline",
            "--record-cache",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--model",
            "explicit-model",
        ],
    )

    assert result.exit_code == 0
    assert captured["model_name"] == "explicit-model"


def test_batch_run_passes_write_proof_option(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_batch_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003
        captured.update(kwargs)
        return _batch_cli_result_with_status(tmp_path, "packaged")

    monkeypatch.setattr("agentic_translation.cli.run_batch_pipeline", fake_run_batch_pipeline)

    result = CliRunner().invoke(
        app,
        [
            "batch",
            "run",
            "samples/public_demo/story.yaml",
            "--chapters",
            "0001",
            "--write-proof",
        ],
    )

    assert result.exit_code == 0
    assert captured["write_proof"] is True


def test_batch_run_passes_live_provider_fallback_option(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_batch_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003
        captured.update(kwargs)
        return _batch_cli_result_with_status(tmp_path, "packaged")

    monkeypatch.setattr("agentic_translation.cli.run_batch_pipeline", fake_run_batch_pipeline)

    result = CliRunner().invoke(
        app,
        [
            "batch",
            "run",
            "samples/public_demo/story.yaml",
            "--chapters",
            "0001",
            "--allow-live-provider-fallback",
        ],
    )

    assert result.exit_code == 0
    assert captured["allow_live_provider_fallback"] is True


def test_batch_live_proof_passes_options_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    proof_result = _batch_live_proof_cli_result(tmp_path)

    def fake_run_live_proof_pipeline(story_yaml: Path, **kwargs):  # noqa: ANN003
        captured["story_yaml"] = story_yaml
        captured.update(kwargs)
        return proof_result

    monkeypatch.setattr("agentic_translation.cli.run_live_proof_pipeline", fake_run_live_proof_pipeline)

    result = CliRunner().invoke(
        app,
        [
            "batch",
            "live-proof",
            "samples/public_demo/story.yaml",
            "--chapters",
            "0001",
            "--translation-provider",
            "offline",
            "--judge-provider",
            "openai",
            "--repair-provider",
            "offline",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--model",
            "gpt-test",
            "--run-id",
            "live_probe",
            "--replay-run-id",
            "live_probe_replay",
            "--overwrite",
            "--skip-epub",
        ],
    )

    assert result.exit_code == 0
    assert captured["story_yaml"] == Path("samples/public_demo/story.yaml")
    assert captured["chapters"] == ["0001"]
    assert captured["translation_provider_name"] == "offline"
    assert captured["judge_provider_name"] == "openai"
    assert captured["repair_provider_name"] == "offline"
    assert captured["cache_dir"] == tmp_path / "cache"
    assert captured["model_name"] == "gpt-test"
    assert captured["run_id"] == "live_probe"
    assert captured["replay_run_id"] == "live_probe_replay"
    assert captured["overwrite"] is True
    assert captured["skip_epub"] is True
    assert "Live proof passed: true" in result.output
    assert "live_probe_replay" in result.output


def test_batch_replay_uses_manifest_defaults_and_writes_proof_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _batch_cli_result_with_status(tmp_path, "packaged")
    run_dir = result.run_dir
    result.manifest.run_config = BatchRunConfig(
        provider_mode="live",
        translation_provider="offline",
        judge_provider="openai",
        repair_provider="offline",
        record_cache=True,
        cache_dir=str(tmp_path / "cache"),
        model_name="gpt-test",
    )
    result.manifest_path.write_text(result.manifest.model_dump_json(indent=2), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_replay_batch_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003
        captured["source_run_dir"] = args[0]
        captured.update(kwargs)
        return result

    monkeypatch.setattr("agentic_translation.cli.replay_batch_pipeline", fake_replay_batch_pipeline)

    cli_result = CliRunner().invoke(
        app,
        [
            "batch",
            "replay",
            str(run_dir),
            "--chapters",
            "0003,0007",
            "--run-id",
            "replay_cli",
            "--overwrite",
            "--skip-epub",
        ],
    )

    assert cli_result.exit_code == 0
    assert captured["source_run_dir"] == run_dir
    assert captured["chapters"] == ["0003", "0007"]
    assert captured["run_id"] == "replay_cli"
    assert captured["overwrite"] is True
    assert captured["skip_epub"] is True
    assert captured["write_proof"] is True


def test_batch_replay_supports_disabling_proof_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _batch_cli_result_with_status(tmp_path, "packaged")
    captured: dict[str, object] = {}

    def fake_replay_batch_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003
        captured.update(kwargs)
        return result

    monkeypatch.setattr("agentic_translation.cli.replay_batch_pipeline", fake_replay_batch_pipeline)

    cli_result = CliRunner().invoke(
        app,
        [
            "batch",
            "replay",
            str(result.run_dir),
            "--no-write-proof",
        ],
    )

    assert cli_result.exit_code == 0
    assert captured["write_proof"] is False


def test_batch_inspect_prints_summary(tmp_path: Path) -> None:
    from agentic_translation.batch import parse_chapter_selection, run_batch_pipeline

    story_yaml = Path("samples/public_demo/story.yaml")
    result = run_batch_pipeline(
        story_yaml,
        chapters=parse_chapter_selection("0001"),
        provider_mode="offline",
        run_id="cli_batch",
        overwrite=True,
    )

    cli_result = CliRunner().invoke(app, ["batch", "inspect", str(result.run_dir)])

    assert cli_result.exit_code == 0
    assert "cli_batch" in cli_result.output
    assert "packaged" in cli_result.output
    assert "Attemp" in cli_result.output
    assert "Repairs" in cli_result.output
    assert "Accept" in cli_result.output


def test_batch_inspect_json_outputs_manifest_for_scripts(tmp_path: Path) -> None:
    from agentic_translation.batch import parse_chapter_selection, run_batch_pipeline

    story_yaml = Path("samples/public_demo/story.yaml")
    result = run_batch_pipeline(
        story_yaml,
        chapters=parse_chapter_selection("0001"),
        provider_mode="offline",
        run_id="cli_batch_json",
        overwrite=True,
    )

    cli_result = CliRunner().invoke(app, ["batch", "inspect", str(result.run_dir), "--json"])
    payload = json.loads(cli_result.output)

    assert cli_result.exit_code == 0
    assert payload["run_id"] == "cli_batch_json"
    assert payload["summary"]["packaged"] == 1
    assert payload["chapters"]["0001"]["status"] == "packaged"
    assert payload["chapters"]["0001"]["repair_decisions"]
    assert payload["chapters"]["0001"]["patch_attempts"]
    assert "Batch cli_batch_json" not in cli_result.output


def test_batch_inspect_strict_exits_nonzero_for_blockers(tmp_path: Path) -> None:
    result = _batch_cli_result_with_status(tmp_path, "review_required")

    cli_result = CliRunner().invoke(app, ["batch", "inspect", str(result.run_dir), "--strict"])

    assert cli_result.exit_code == 1
    assert "review required" in cli_result.output.lower()


def test_batch_inspect_strict_exits_nonzero_for_incomplete_chapters(tmp_path: Path) -> None:
    result = _batch_cli_result_with_status(tmp_path, "running")

    cli_result = CliRunner().invoke(app, ["batch", "inspect", str(result.run_dir), "--strict"])

    assert cli_result.exit_code == 1
    assert "incomplete" in cli_result.output.lower()


def test_batch_inspect_shows_last_attempt_message_for_failed_chapter(tmp_path: Path) -> None:
    result = _batch_cli_result_with_status(tmp_path, "failed")
    manifest = result.manifest
    manifest.chapters["0001"].attempts.append(
        AgentAttempt(
            attempt_id="0001-attempt-001",
            chapter="0001",
            provider="translation=openai;judge=openai;repair=offline",
            model="gpt-test",
            action="run_chapter",
            status="fail",
            message="provider timed out",
        )
    )
    result.manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    cli_result = CliRunner().invoke(app, ["batch", "inspect", str(result.run_dir)])

    assert cli_result.exit_code == 0
    assert "Last" in cli_result.output
    assert "Attempt" in cli_result.output
    assert "fail:" in cli_result.output
    assert "provider" in cli_result.output
    assert "timed" in cli_result.output
    assert "out" in cli_result.output


def test_batch_inspect_shows_provider_failure_summary(tmp_path: Path) -> None:
    result = _batch_cli_result_with_status(tmp_path, "packaged")
    manifest = result.manifest
    manifest.mode = "live"
    manifest.providers = {
        "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
        "judge": ProviderLabel(provider="deepseek", model="deepseek-chat"),
        "repair": ProviderLabel(provider="offline", model="offline-fixture-v1"),
    }
    manifest.run_config = BatchRunConfig(
        provider_mode="live",
        translation_provider="offline",
        judge_provider="deepseek",
        repair_provider="offline",
        model_name="deepseek-chat",
        allow_live_provider_fallback=True,
    )
    manifest.chapters["0001"].patch_attempts = [
        PatchAttempt(
            finding_check_id="system_panel_count",
            strategy="candidate_selection",
            before_score=84,
            after_score=100,
            before_findings=1,
            after_findings=0,
            accepted=True,
            reason="Live judge provider failed (402 Insufficient Balance); fell back to offline judge.",
        )
    ]
    result.manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    cli_result = CliRunner().invoke(app, ["batch", "inspect", str(result.run_dir)])

    assert cli_result.exit_code == 0
    assert "Provider failures" in cli_result.output
    assert "judge/deepseek/deepseek-chat" in cli_result.output
    assert "fallback used" in cli_result.output
    assert "Insufficient Balance" in cli_result.output


def test_batch_inspect_json_strict_keeps_json_parseable_when_exiting_nonzero(tmp_path: Path) -> None:
    result = _batch_cli_result_with_status(tmp_path, "review_required")

    cli_result = CliRunner().invoke(app, ["batch", "inspect", str(result.run_dir), "--json", "--strict"])
    payload = json.loads(cli_result.output)

    assert cli_result.exit_code == 1
    assert payload["summary"]["review_required"] == 1
    assert "review required" not in cli_result.output.lower()


def test_batch_inspect_status_json_reports_blockers_and_stays_parseable_with_strict(tmp_path: Path) -> None:
    result = _batch_cli_result_with_status(tmp_path, "failed", artifact_qa_passed=False)
    result.manifest.chapters["0001"].error = "provider timed out"
    result.manifest_path.write_text(result.manifest.model_dump_json(indent=2), encoding="utf-8")

    cli_result = CliRunner().invoke(app, ["batch", "inspect", str(result.run_dir), "--status-json", "--strict"])
    payload = json.loads(cli_result.output)

    assert cli_result.exit_code == 1
    assert payload["ready_for_delivery"] is False
    assert payload["blocker_count"] == 2
    assert payload["blockers"][0]["blocker_type"] == "failed"
    assert payload["blockers"][0]["message"] == "provider timed out"
    assert payload["blockers"][1]["blocker_type"] == "artifact_qa"
    assert "Batch artifact QA failed" not in cli_result.output


def test_batch_inspect_status_json_require_agentic_exits_nonzero_but_keeps_json_parseable(tmp_path: Path) -> None:
    result = _batch_cli_result_with_status(tmp_path, "packaged")

    cli_result = CliRunner().invoke(app, ["batch", "inspect", str(result.run_dir), "--status-json", "--require-agentic"])
    payload = json.loads(cli_result.output)

    assert cli_result.exit_code == 1
    assert payload["ready_for_delivery"] is True
    assert payload["agentic_evidence"]["agentic_claim_supported"] is False
    assert "agentic evidence" not in cli_result.output.lower()


def test_batch_inspect_status_json_require_agentic_passes_when_model_backed_selection_observed(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_entry = ResponseCache(cache_dir).save(
        "judge",
        {"payload": "judge"},
        {"selected_candidate_id": "candidate_b"},
        metadata={"provider": "openai", "model": "gpt-test"},
    )
    result = _batch_cli_result_with_status(tmp_path, "packaged")
    result.manifest.mode = "live"
    result.manifest.providers = {
        "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
        "judge": ProviderLabel(provider="openai", model="gpt-test"),
        "repair": ProviderLabel(provider="offline", model="offline-fixture-v1"),
    }
    result.manifest.run_config = BatchRunConfig(
        provider_mode="live",
        translation_provider="offline",
        judge_provider="openai",
        repair_provider="offline",
        record_cache=True,
        cache_dir=str(cache_dir),
        model_name="gpt-test",
    )
    result.manifest.chapters["0001"].repair_decisions = [
        RepairDecision(
            finding_check_id="system_panel_count",
            strategy="candidate_selection",
            selected_candidate_id="candidate_b",
            reason="Router selected candidate_selection for system_panel_count.",
        )
    ]
    result.manifest.chapters["0001"].patch_attempts = [
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
    result.manifest.chapters["0001"].provider_calls = [
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
    result.manifest_path.write_text(result.manifest.model_dump_json(indent=2), encoding="utf-8")

    cli_result = CliRunner().invoke(app, ["batch", "inspect", str(result.run_dir), "--status-json", "--require-agentic"])
    payload = json.loads(cli_result.output)

    assert cli_result.exit_code == 0
    assert payload["agentic_evidence"]["agentic_claim_supported"] is True
    assert payload["agentic_evidence"]["observed_agentic_roles"] == ["judge"]
    assert payload["agentic_evidence"]["verified_candidate_selection_records"] == 1
    assert payload["agentic_evidence"]["candidate_selection_mismatches"] == []


def test_batch_inspect_status_json_require_agentic_rejects_configured_provider_without_provider_call(tmp_path: Path) -> None:
    result = _batch_cli_result_with_status(tmp_path, "packaged")
    result.manifest.mode = "live"
    result.manifest.providers = {
        "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
        "judge": ProviderLabel(provider="openai", model="gpt-test"),
        "repair": ProviderLabel(provider="offline", model="offline-fixture-v1"),
    }
    result.manifest.chapters["0001"].repair_decisions = [
        RepairDecision(
            finding_check_id="system_panel_count",
            strategy="candidate_selection",
            selected_candidate_id="candidate_b",
            reason="Router selected candidate_selection for system_panel_count.",
        )
    ]
    result.manifest.chapters["0001"].patch_attempts = [
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
    result.manifest_path.write_text(result.manifest.model_dump_json(indent=2), encoding="utf-8")

    cli_result = CliRunner().invoke(app, ["batch", "inspect", str(result.run_dir), "--status-json", "--require-agentic"])
    payload = json.loads(cli_result.output)

    assert cli_result.exit_code == 1
    assert payload["agentic_evidence"]["agentic_claim_supported"] is False
    assert payload["agentic_evidence"]["observed_agentic_roles"] == []
    assert "no recorded model-backed provider calls" in payload["agentic_evidence"]["reason"]


def test_batch_inspect_status_json_require_agentic_rejects_judge_selection_mismatch(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_entry = ResponseCache(cache_dir).save(
        "judge",
        {"payload": "judge"},
        {"selected_candidate_id": "candidate_a"},
        metadata={"provider": "openai", "model": "gpt-test"},
    )
    result = _batch_cli_result_with_status(tmp_path, "packaged")
    result.manifest.mode = "live"
    result.manifest.providers = {
        "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
        "judge": ProviderLabel(provider="openai", model="gpt-test"),
        "repair": ProviderLabel(provider="offline", model="offline-fixture-v1"),
    }
    result.manifest.run_config = BatchRunConfig(
        provider_mode="live",
        translation_provider="offline",
        judge_provider="openai",
        repair_provider="offline",
        record_cache=True,
        cache_dir=str(cache_dir),
        model_name="gpt-test",
    )
    result.manifest.chapters["0001"].repair_decisions = [
        RepairDecision(
            finding_check_id="system_panel_count",
            strategy="candidate_selection",
            selected_candidate_id="candidate_b",
            reason="Router selected candidate_selection for system_panel_count.",
        )
    ]
    result.manifest.chapters["0001"].provider_calls = [
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
    result.manifest_path.write_text(result.manifest.model_dump_json(indent=2), encoding="utf-8")

    cli_result = CliRunner().invoke(app, ["batch", "inspect", str(result.run_dir), "--status-json", "--require-agentic"])
    payload = json.loads(cli_result.output)

    assert cli_result.exit_code == 1
    assert payload["agentic_evidence"]["agentic_claim_supported"] is False
    assert payload["agentic_evidence"]["observed_agentic_roles"] == []
    assert "selected candidate did not match" in payload["agentic_evidence"]["reason"]


def test_batch_inspect_status_json_require_agentic_rejects_repair_patch_mismatch(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_entry = ResponseCache(cache_dir).save(
        "repair",
        {"payload": "repair"},
        {
            "patch_type": "replace_span",
            "old_text": "Dao",
            "new_text": "Celestial Way",
            "paragraph_index": None,
            "reason": "Live repair proposed a different canon.",
        },
        metadata={"provider": "openai", "model": "gpt-test"},
    )
    result = _batch_cli_result_with_status(tmp_path, "packaged")
    result.manifest.mode = "live"
    result.manifest.providers = {
        "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
        "judge": ProviderLabel(provider="offline", model="offline-fixture-v1"),
        "repair": ProviderLabel(provider="openai", model="gpt-test"),
    }
    result.manifest.run_config = BatchRunConfig(
        provider_mode="live",
        translation_provider="offline",
        judge_provider="offline",
        repair_provider="openai",
        record_cache=True,
        cache_dir=str(cache_dir),
        model_name="gpt-test",
    )
    result.manifest.chapters["0001"].repair_decisions = [
        RepairDecision(
            finding_check_id="glossary_required",
            strategy="candidate_selection",
            selected_candidate_id="raw_b",
            reason="Repair provider supplied a targeted patch.",
        )
    ]
    result.manifest.chapters["0001"].patch_attempts = [
        PatchAttempt(
            finding_check_id="glossary_required",
            strategy="candidate_selection",
            before_score=94,
            after_score=100,
            before_findings=1,
            after_findings=0,
            accepted=True,
            reason="Accepted because compliance QA improved.",
            patch=RepairPatch(
                patch_id="patch_live_glossary_required",
                patch_type="replace_span",
                chapter="0001",
                old_text="Dao",
                new_text="Heavenly Dao",
                reason="Live provider proposed minimal patch.",
                source_finding_check_id="glossary_required",
            ),
        )
    ]
    result.manifest.chapters["0001"].provider_calls = [
        ProviderCallRecord(
            role="repair",
            namespace="repair",
            provider="openai",
            model="gpt-test",
            payload_sha256=cache_entry.payload_sha256,
            response_sha256=cache_entry.response_sha256,
            cache_file=cache_entry.cache_file,
            cache_hit=False,
        )
    ]
    result.manifest_path.write_text(result.manifest.model_dump_json(indent=2), encoding="utf-8")

    cli_result = CliRunner().invoke(app, ["batch", "inspect", str(result.run_dir), "--status-json", "--require-agentic"])
    payload = json.loads(cli_result.output)

    assert cli_result.exit_code == 1
    assert payload["agentic_evidence"]["agentic_claim_supported"] is False
    assert payload["agentic_evidence"]["observed_agentic_roles"] == []
    assert payload["agentic_evidence"]["verified_repair_patch_records"] == 0
    assert len(payload["agentic_evidence"]["repair_patch_mismatches"]) == 1
    assert "repair patch did not match" in payload["agentic_evidence"]["reason"]


def test_batch_inspect_status_json_require_replayable_exits_nonzero_when_cache_namespace_missing(tmp_path: Path) -> None:
    result = _batch_cli_result_with_status(tmp_path, "packaged")
    result.manifest.mode = "live"
    result.manifest.providers = {
        "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
        "judge": ProviderLabel(provider="openai", model="gpt-test"),
        "repair": ProviderLabel(provider="offline", model="offline-fixture-v1"),
    }
    result.manifest.run_config = BatchRunConfig(
        provider_mode="live",
        translation_provider="offline",
        judge_provider="openai",
        repair_provider="offline",
        record_cache=True,
        cache_dir=str(tmp_path / "cache"),
        model_name="gpt-test",
    )
    result.manifest_path.write_text(result.manifest.model_dump_json(indent=2), encoding="utf-8")

    cli_result = CliRunner().invoke(app, ["batch", "inspect", str(result.run_dir), "--status-json", "--require-replayable"])
    payload = json.loads(cli_result.output)

    assert cli_result.exit_code == 1
    assert payload["agentic_evidence"]["cache_required_namespaces"] == ["judge"]
    assert payload["agentic_evidence"]["cache_missing_namespaces"] == ["judge"]
    assert payload["agentic_evidence"]["replay_cache_ready"] is False
    assert "Usage:" not in cli_result.output


def test_batch_inspect_status_json_require_replayable_passes_when_cache_namespace_present(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    ResponseCache(cache_dir).save(
        "judge",
        {"payload": "judge"},
        {"selected_candidate_id": "candidate_b"},
        metadata={"provider": "openai", "model": "gpt-test"},
    )
    cache_entry = ResponseCache(cache_dir).inspect().entries[0]
    result = _batch_cli_result_with_status(tmp_path, "packaged")
    result.manifest.mode = "replay"
    result.manifest.providers = {
        "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
        "judge": ProviderLabel(provider="openai", model="gpt-test"),
        "repair": ProviderLabel(provider="offline", model="offline-fixture-v1"),
    }
    result.manifest.run_config = BatchRunConfig(
        provider_mode="replay",
        translation_provider="offline",
        judge_provider="openai",
        repair_provider="offline",
        cache_dir=str(cache_dir),
        model_name="gpt-test",
    )
    result.manifest.chapters["0001"].provider_calls = [
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
    result.manifest_path.write_text(result.manifest.model_dump_json(indent=2), encoding="utf-8")

    cli_result = CliRunner().invoke(app, ["batch", "inspect", str(result.run_dir), "--status-json", "--require-replayable"])
    payload = json.loads(cli_result.output)

    assert cli_result.exit_code == 0
    assert payload["agentic_evidence"]["cache_required_namespaces"] == ["judge"]
    assert payload["agentic_evidence"]["cache_missing_namespaces"] == []
    assert payload["agentic_evidence"]["provider_call_records"] == 1
    assert payload["agentic_evidence"]["cache_verified_call_records"] == 1
    assert payload["agentic_evidence"]["replay_cache_ready"] is True


def test_batch_inspect_status_json_require_replayable_rejects_unrelated_cache_namespace(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    ResponseCache(cache_dir).save(
        "judge",
        {"payload": "unrelated"},
        {"selected_candidate_id": "candidate_b"},
        metadata={"provider": "openai", "model": "gpt-test"},
    )
    result = _batch_cli_result_with_status(tmp_path, "packaged")
    result.manifest.mode = "replay"
    result.manifest.providers = {
        "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
        "judge": ProviderLabel(provider="openai", model="gpt-test"),
        "repair": ProviderLabel(provider="offline", model="offline-fixture-v1"),
    }
    result.manifest.run_config = BatchRunConfig(
        provider_mode="replay",
        translation_provider="offline",
        judge_provider="openai",
        repair_provider="offline",
        cache_dir=str(cache_dir),
        model_name="gpt-test",
    )
    result.manifest.chapters["0001"].provider_calls = [
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
    result.manifest_path.write_text(result.manifest.model_dump_json(indent=2), encoding="utf-8")

    cli_result = CliRunner().invoke(app, ["batch", "inspect", str(result.run_dir), "--status-json", "--require-replayable"])
    payload = json.loads(cli_result.output)

    assert cli_result.exit_code == 1
    assert payload["agentic_evidence"]["cache_required_namespaces"] == ["judge"]
    assert payload["agentic_evidence"]["cache_missing_namespaces"] == []
    assert payload["agentic_evidence"]["provider_call_records"] == 1
    assert payload["agentic_evidence"]["cache_verified_call_records"] == 0
    assert payload["agentic_evidence"]["replay_cache_ready"] is False


def test_batch_inspect_status_json_require_replayable_rejects_cache_model_mismatch(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_entry = ResponseCache(cache_dir).save(
        "judge",
        {"payload": "judge"},
        {"selected_candidate_id": "candidate_b"},
        metadata={"provider": "openai", "model": "different-model"},
    )
    result = _batch_cli_result_with_status(tmp_path, "packaged")
    result.manifest.mode = "replay"
    result.manifest.providers = {
        "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
        "judge": ProviderLabel(provider="openai", model="gpt-test"),
        "repair": ProviderLabel(provider="offline", model="offline-fixture-v1"),
    }
    result.manifest.run_config = BatchRunConfig(
        provider_mode="replay",
        translation_provider="offline",
        judge_provider="openai",
        repair_provider="offline",
        cache_dir=str(cache_dir),
        model_name="gpt-test",
    )
    result.manifest.chapters["0001"].provider_calls = [
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
    result.manifest_path.write_text(result.manifest.model_dump_json(indent=2), encoding="utf-8")

    cli_result = CliRunner().invoke(app, ["batch", "inspect", str(result.run_dir), "--status-json", "--require-replayable"])
    payload = json.loads(cli_result.output)

    assert cli_result.exit_code == 1
    assert payload["agentic_evidence"]["cache_verified_call_records"] == 0
    assert payload["agentic_evidence"]["replay_cache_ready"] is False
    assert payload["agentic_evidence"]["cache_metadata_mismatches"] == [
        "0001:judge:gpt-test!=different-model"
    ]


def test_batch_prove_json_reports_cache_metadata_mismatch(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_entry = ResponseCache(cache_dir).save(
        "judge",
        {"payload": "judge"},
        {"selected_candidate_id": "candidate_b"},
        metadata={"provider": "openai", "model": "different-model"},
    )
    result = _batch_cli_result_with_status(tmp_path, "packaged")
    result.manifest.mode = "replay"
    result.manifest.providers = {
        "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
        "judge": ProviderLabel(provider="openai", model="gpt-test"),
        "repair": ProviderLabel(provider="offline", model="offline-fixture-v1"),
    }
    result.manifest.run_config = BatchRunConfig(
        provider_mode="replay",
        translation_provider="offline",
        judge_provider="openai",
        repair_provider="offline",
        cache_dir=str(cache_dir),
        model_name="gpt-test",
    )
    result.manifest.chapters["0001"].provider_calls = [
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
    result.manifest_path.write_text(result.manifest.model_dump_json(indent=2), encoding="utf-8")

    cli_result = CliRunner().invoke(app, ["batch", "prove", str(result.run_dir), "--json"])
    payload = json.loads(cli_result.output)

    assert cli_result.exit_code == 1
    assert payload["gates"]["replayable"] is False
    assert any("metadata mismatch" in blocker for blocker in payload["blockers"])


def test_batch_prove_json_exits_nonzero_but_keeps_json_parseable_when_not_proven(tmp_path: Path) -> None:
    result = _batch_cli_result_with_status(tmp_path, "packaged")

    cli_result = CliRunner().invoke(app, ["batch", "prove", str(result.run_dir), "--json"])
    payload = json.loads(cli_result.output)

    assert cli_result.exit_code == 1
    assert payload["proof_passed"] is False
    assert payload["gates"] == {
        "delivery": True,
        "agentic": False,
        "replayable": False,
    }
    assert any("agentic" in blocker for blocker in payload["blockers"])
    assert any("replay" in blocker for blocker in payload["blockers"])


def test_batch_prove_json_passes_when_delivery_agentic_and_replayable(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_entry = ResponseCache(cache_dir).save(
        "judge",
        {"payload": "judge"},
        {"selected_candidate_id": "candidate_b"},
        metadata={"provider": "openai", "model": "gpt-test"},
    )
    result = _batch_cli_result_with_status(tmp_path, "packaged")
    result.manifest.mode = "replay"
    result.manifest.providers = {
        "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
        "judge": ProviderLabel(provider="openai", model="gpt-test"),
        "repair": ProviderLabel(provider="offline", model="offline-fixture-v1"),
    }
    result.manifest.run_config = BatchRunConfig(
        provider_mode="replay",
        translation_provider="offline",
        judge_provider="openai",
        repair_provider="offline",
        cache_dir=str(cache_dir),
        model_name="gpt-test",
    )
    result.manifest.chapters["0001"].repair_decisions = [
        RepairDecision(
            finding_check_id="system_panel_count",
            strategy="candidate_selection",
            selected_candidate_id="candidate_b",
            reason="Router selected candidate_selection for system_panel_count.",
        )
    ]
    result.manifest.chapters["0001"].patch_attempts = [
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
    result.manifest.chapters["0001"].provider_calls = [
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
    result.manifest_path.write_text(result.manifest.model_dump_json(indent=2), encoding="utf-8")

    cli_result = CliRunner().invoke(app, ["batch", "prove", str(result.run_dir), "--json"])
    payload = json.loads(cli_result.output)

    assert cli_result.exit_code == 0
    assert payload["proof_passed"] is True
    assert payload["gates"] == {
        "delivery": True,
        "agentic": True,
        "replayable": True,
    }
    assert payload["inspection"]["agentic_evidence"]["cache_verified_call_records"] == 1


def test_batch_prove_write_persists_json_markdown_and_manifest_links(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_entry = ResponseCache(cache_dir).save(
        "judge",
        {"payload": "judge"},
        {"selected_candidate_id": "candidate_b"},
        metadata={"provider": "openai", "model": "gpt-test"},
    )
    result = _batch_cli_result_with_status(tmp_path, "packaged")
    result.manifest.mode = "replay"
    result.manifest.providers = {
        "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
        "judge": ProviderLabel(provider="openai", model="gpt-test"),
        "repair": ProviderLabel(provider="offline", model="offline-fixture-v1"),
    }
    result.manifest.run_config = BatchRunConfig(
        provider_mode="replay",
        translation_provider="offline",
        judge_provider="openai",
        repair_provider="offline",
        cache_dir=str(cache_dir),
        model_name="gpt-test",
    )
    result.manifest.chapters["0001"].repair_decisions = [
        RepairDecision(
            finding_check_id="system_panel_count",
            strategy="candidate_selection",
            selected_candidate_id="candidate_b",
            reason="Router selected candidate_selection for system_panel_count.",
        )
    ]
    result.manifest.chapters["0001"].patch_attempts = [
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
    result.manifest.chapters["0001"].provider_calls = [
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
    result.manifest_path.write_text(result.manifest.model_dump_json(indent=2), encoding="utf-8")

    cli_result = CliRunner().invoke(app, ["batch", "prove", str(result.run_dir), "--write"])
    manifest = BatchManifest.model_validate_json(result.manifest_path.read_text(encoding="utf-8"))
    proof_payload = json.loads((result.run_dir / "agentic_proof.json").read_text(encoding="utf-8"))
    proof_markdown = (result.run_dir / "agentic_proof.md").read_text(encoding="utf-8")

    assert cli_result.exit_code == 0
    assert proof_payload["proof_passed"] is True
    assert "Agentic Proof" in proof_markdown
    assert manifest.artifacts["agentic_proof_json"] == "agentic_proof.json"
    assert manifest.artifacts["agentic_proof_markdown"] == "agentic_proof.md"


def test_batch_inspect_status_json_require_replayable_fails_when_cache_integrity_fails(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    ResponseCache(cache_dir).save(
        "judge",
        {"payload": "judge"},
        {"selected_candidate_id": "candidate_b"},
        metadata={"provider": "openai", "model": "gpt-test"},
    )
    cache_file = next(cache_dir.glob("judge_*.json"))
    cache_file.write_text('{"selected_candidate_id": "candidate_a"}', encoding="utf-8")
    result = _batch_cli_result_with_status(tmp_path, "packaged")
    result.manifest.mode = "replay"
    result.manifest.providers = {
        "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
        "judge": ProviderLabel(provider="openai", model="gpt-test"),
        "repair": ProviderLabel(provider="offline", model="offline-fixture-v1"),
    }
    result.manifest.run_config = BatchRunConfig(
        provider_mode="replay",
        translation_provider="offline",
        judge_provider="openai",
        repair_provider="offline",
        cache_dir=str(cache_dir),
        model_name="gpt-test",
    )
    result.manifest_path.write_text(result.manifest.model_dump_json(indent=2), encoding="utf-8")

    cli_result = CliRunner().invoke(app, ["batch", "inspect", str(result.run_dir), "--status-json", "--require-replayable"])
    payload = json.loads(cli_result.output)

    assert cli_result.exit_code == 1
    assert payload["agentic_evidence"]["cache_required_namespaces"] == ["judge"]
    assert payload["agentic_evidence"]["cache_missing_namespaces"] == []
    assert payload["agentic_evidence"]["replay_cache_ready"] is False
    assert payload["agentic_evidence"]["cache_integrity_passed"] is False


def test_batch_inspect_rejects_two_json_output_shapes(tmp_path: Path) -> None:
    result = _batch_cli_result_with_status(tmp_path, "packaged")

    cli_result = CliRunner().invoke(app, ["batch", "inspect", str(result.run_dir), "--json", "--status-json"])

    assert cli_result.exit_code == 1
    assert "only one" in cli_result.output


def test_batch_resume_processes_pending_chapter(tmp_path: Path) -> None:
    from tests.test_batch import _write_public_batch_fixture
    from agentic_translation.batch import load_batch_manifest, parse_chapter_selection, run_batch_pipeline, write_batch_manifest
    from agentic_translation.models import BatchChapterRun

    story_yaml = _write_public_batch_fixture(tmp_path)
    result = run_batch_pipeline(
        story_yaml,
        chapters=parse_chapter_selection("0001"),
        provider_mode="offline",
        run_id="cli_resume",
        overwrite=True,
    )
    manifest = load_batch_manifest(result.manifest_path)
    manifest.chapters["0002"] = BatchChapterRun(chapter="0002")
    write_batch_manifest(result.manifest_path, manifest)

    cli_result = CliRunner().invoke(app, ["batch", "resume", str(result.run_dir)])

    assert cli_result.exit_code == 0
    assert "0002" in cli_result.output
    assert "2/2 packaged" in cli_result.output


def test_batch_resume_write_triage_writes_review_packet(
    monkeypatch,
    tmp_path: Path,
) -> None:  # noqa: ANN001
    result = _batch_cli_result_with_status(tmp_path, "review_required")
    triage_calls: list[Path] = []

    monkeypatch.setattr("agentic_translation.cli.resume_batch_pipeline", lambda *args, **kwargs: result)
    monkeypatch.setattr(
        "agentic_translation.cli.write_batch_triage_artifacts",
        lambda run_dir: triage_calls.append(run_dir) or {"review_queue": "review_queue.json"},
    )

    cli_result = CliRunner().invoke(
        app,
        [
            "batch",
            "resume",
            str(result.run_dir),
            "--retry-review-required",
            "--allow-review-required",
            "--write-triage",
        ],
    )

    assert cli_result.exit_code == 0
    assert triage_calls == [result.run_dir]
    assert "triage artifact" in cli_result.output.lower()


def test_batch_resume_passes_retry_review_required(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "batch_resume_retry"
    run_dir.mkdir()
    result = _batch_cli_result_with_status(tmp_path, "packaged")
    manifest_path = run_dir / "batch_manifest.json"
    manifest_path.write_text(result.manifest.model_dump_json(indent=2), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_resume_batch_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003
        captured.update(kwargs)
        return BatchPipelineResult(
            run_dir=run_dir,
            manifest_path=manifest_path,
            manifest=result.manifest,
            artifact_qa=result.artifact_qa,
        )

    monkeypatch.setattr("agentic_translation.cli.resume_batch_pipeline", fake_resume_batch_pipeline)

    cli_result = CliRunner().invoke(
        app,
        [
            "batch",
            "resume",
            str(run_dir),
            "--retry-review-required",
            "--chapters",
            "0003,0007",
            "--allow-live-provider-fallback",
            "--write-proof",
        ],
    )

    assert cli_result.exit_code == 0
    assert captured["retry_review_required"] is True
    assert captured["chapters"] == ["0003", "0007"]
    assert captured["allow_live_provider_fallback"] is True
    assert captured["write_proof"] is True


def test_batch_refresh_passes_chapter_selection_and_skip_epub(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _batch_cli_result_with_status(tmp_path, "packaged")
    run_dir = result.run_dir
    captured: dict[str, object] = {}

    def fake_refresh_batch_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003
        captured["run_dir"] = args[0]
        captured.update(kwargs)
        return result

    monkeypatch.setattr("agentic_translation.cli.refresh_batch_pipeline", fake_refresh_batch_pipeline)

    cli_result = CliRunner().invoke(
        app,
        [
            "batch",
            "refresh",
            str(run_dir),
            "--chapters",
            "0003,0007",
            "--skip-epub",
            "--write-proof",
        ],
    )

    assert cli_result.exit_code == 0
    assert captured["run_dir"] == run_dir
    assert captured["chapters"] == ["0003", "0007"]
    assert captured["skip_epub"] is True
    assert captured["write_proof"] is True


def test_batch_accept_passes_review_metadata_and_allows_review_required(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _batch_cli_result_with_status(tmp_path, "review_required")
    run_dir = result.run_dir
    result.manifest.chapters["0001"].manual_reviews.append(
        ManualReviewRecord(
            chapter="0001",
            reviewer="eric",
            note="Human accepted remaining style issue.",
            status_before="review_required",
            status_after="review_required",
            qa_score_after=94,
            qa_findings_after=1,
        )
    )
    captured: dict[str, object] = {}

    def fake_accept_reviewed_chapters(*args, **kwargs):  # noqa: ANN002, ANN003
        captured["run_dir"] = args[0]
        captured.update(kwargs)
        return result

    monkeypatch.setattr("agentic_translation.cli.accept_reviewed_chapters", fake_accept_reviewed_chapters)

    cli_result = CliRunner().invoke(
        app,
        [
            "batch",
            "accept",
            str(run_dir),
            "--chapters",
            "0003,0007",
            "--reviewer",
            "eric",
            "--note",
            "Human accepted remaining style issue.",
            "--skip-epub",
            "--write-proof",
            "--allow-review-required",
        ],
    )

    assert cli_result.exit_code == 0
    assert captured["run_dir"] == run_dir
    assert captured["chapters"] == ["0003", "0007"]
    assert captured["reviewer"] == "eric"
    assert captured["note"] == "Human accepted remaining style issue."
    assert captured["skip_epub"] is True
    assert captured["write_proof"] is True


def test_batch_replace_text_passes_manual_replacement_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "manual"
    run_dir.mkdir(parents=True)
    manifest = BatchManifest.create(
        run_id="manual",
        story_slug="manual",
        title="Manual",
        story_yaml=tmp_path / "story.yaml",
        chapters=["0001"],
        mode="offline",
        providers={},
        run_dir=run_dir,
    )
    (run_dir / "batch_manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    captured: dict[str, object] = {}
    triage_calls: list[Path] = []

    def fake_apply_manual_text_replacement(*args, **kwargs):  # noqa: ANN002, ANN003
        captured["run_dir"] = args[0]
        captured.update(kwargs)
        return ManualTextReplacementResult(
            run_id="manual",
            run_dir=str(run_dir),
            chapter="0001",
            final_path=str(run_dir / "chapters" / "0001" / "translated_final" / "0001.txt"),
            old_text="Dao",
            new_text="Heavenly Dao",
            occurrence_count=2,
            refresh_only=False,
            reviewer="eric",
            note="Fixed term.",
            status_after="packaged",
            final_score_after=100,
            final_findings_after=0,
            summary_after=BatchSummary(total_chapters=1, packaged=1),
        )

    monkeypatch.setattr("agentic_translation.cli.apply_manual_text_replacement", fake_apply_manual_text_replacement)
    monkeypatch.setattr("agentic_translation.cli._exit_if_batch_has_blockers", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "agentic_translation.cli.write_batch_triage_artifacts",
        lambda path: triage_calls.append(path) or {"review_queue": "review_queue.json"},
    )

    cli_result = CliRunner().invoke(
        app,
        [
            "batch",
            "replace-text",
            str(run_dir),
            "--chapter",
            "1",
            "--old",
            "Dao",
            "--new",
            "Heavenly Dao",
            "--reviewer",
            "eric",
            "--note",
            "Fixed term.",
            "--skip-epub",
            "--write-proof",
            "--allow-review-required",
        ],
    )

    assert cli_result.exit_code == 0
    assert captured["run_dir"] == run_dir
    assert captured["chapter"] == "0001"
    assert captured["old_text"] == "Dao"
    assert captured["new_text"] == "Heavenly Dao"
    assert captured["reviewer"] == "eric"
    assert captured["note"] == "Fixed term."
    assert captured["refresh_only"] is False
    assert captured["skip_epub"] is True
    assert captured["write_proof"] is True
    assert triage_calls == [run_dir]
    assert "Replaced 2 occurrence(s)" in cli_result.output


def test_batch_normalize_panels_passes_options_and_refreshes_triage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "manual"
    run_dir.mkdir(parents=True)
    manifest = BatchManifest.create(
        run_id="manual",
        story_slug="manual",
        title="Manual",
        story_yaml=tmp_path / "story.yaml",
        chapters=["0001"],
        mode="offline",
        providers={},
        run_dir=run_dir,
    )
    (run_dir / "batch_manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    captured: dict[str, object] = {}
    triage_calls: list[Path] = []

    def fake_normalize_panel_splits(*args, **kwargs):  # noqa: ANN002, ANN003
        captured["run_dir"] = args[0]
        captured.update(kwargs)
        return PanelNormalizationResult(
            run_id="manual",
            run_dir=str(run_dir),
            items=[
                PanelNormalizationItem(
                    chapter="0001",
                    status="normalized",
                    reason="Merged adjacent numbered note panels.",
                    replacement_count=1,
                    status_after="packaged",
                    final_score_after=100,
                    final_findings_after=0,
                )
            ],
            normalized_count=1,
            skipped_count=0,
            summary_after=BatchSummary(total_chapters=1, packaged=1),
        )

    monkeypatch.setattr("agentic_translation.cli.normalize_panel_splits", fake_normalize_panel_splits)
    monkeypatch.setattr(
        "agentic_translation.cli.write_batch_triage_artifacts",
        lambda path: triage_calls.append(path) or {"review_queue": "review_queue.json"},
    )
    monkeypatch.setattr("agentic_translation.cli._exit_if_batch_has_blockers", lambda *args, **kwargs: None)

    cli_result = CliRunner().invoke(
        app,
        [
            "batch",
            "normalize-panels",
            str(run_dir),
            "--chapters",
            "1",
            "--reviewer",
            "codex",
            "--note-prefix",
            "Merged split note panels.",
            "--skip-epub",
            "--write-proof",
            "--allow-review-required",
        ],
    )

    assert cli_result.exit_code == 0
    assert captured["run_dir"] == run_dir
    assert captured["chapters"] == ["0001"]
    assert captured["reviewer"] == "codex"
    assert captured["note_prefix"] == "Merged split note panels."
    assert captured["skip_epub"] is True
    assert captured["write_proof"] is True
    assert triage_calls == [run_dir]
    assert "Normalized 1 panel split(s)" in cli_result.output
    assert "0001" in cli_result.output


def test_batch_panel_report_writes_markdown_and_json(tmp_path: Path) -> None:
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
    chapter_run.source_path = str(source_path)
    chapter_run.final_path = str(final_path)
    chapter_run.chapter_run_dir = str(chapter_dir)
    manifest.refresh_summary()
    (run_dir / "batch_manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    result = CliRunner().invoke(app, ["batch", "panel-report", str(run_dir), "--write"])

    assert result.exit_code == 0
    assert "Panel Report" in result.output
    assert "Extra Final Panels" in result.output
    report_json = json.loads((run_dir / "panel_report.json").read_text(encoding="utf-8"))
    assert report_json["summary"]["mismatch_chapters"] == 1
    assert (run_dir / "panel_report.md").exists()


def test_import_local_supports_batch_chapter_range(tmp_path: Path) -> None:
    source_dir = tmp_path / "scraped"
    translated_dir = tmp_path / "translated_001_002"
    terms_dir = tmp_path / "terms"
    out = tmp_path / "local_fixture"
    source_dir.mkdir()
    translated_dir.mkdir()
    terms_dir.mkdir()
    for chapter in ["0001", "0002"]:
        (source_dir / f"{chapter}.txt").write_text(f"第{chapter}章\n\n天道", encoding="utf-8")
        (translated_dir / f"{chapter}.txt").write_text(f"Chapter {int(chapter)}\n\nHeavenly Dao", encoding="utf-8")
    glossary = terms_dir / "master_glossary.txt"
    glossary.write_text("天道 -> Heavenly Dao\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "import-local",
            "--source-dir",
            str(source_dir),
            "--glossary",
            str(glossary),
            "--chapters",
            "0001-0002",
            "--translated-dir",
            str(translated_dir),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    story = yaml.safe_load((out / "story.yaml").read_text(encoding="utf-8"))
    assert story["chapter_ids"] == ["0001", "0002"]
    assert "baseline_dir" in story["paths"]
    assert (out / "baseline" / "0001.txt").exists()
    assert (out / "baseline" / "0002.txt").exists()


def test_import_local_can_run_batch_immediately(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    source_dir = tmp_path / "scraped"
    translated_dir = tmp_path / "translated_001_002"
    terms_dir = tmp_path / "terms"
    out = tmp_path / "local_fixture"
    cache_dir = tmp_path / "cache"
    source_dir.mkdir()
    translated_dir.mkdir()
    terms_dir.mkdir()
    for chapter in ["0001", "0002"]:
        (source_dir / f"{chapter}.txt").write_text(f"第{chapter}章\n\n天道", encoding="utf-8")
        (translated_dir / f"{chapter}.txt").write_text(f"Chapter {int(chapter)}\n\nHeavenly Dao", encoding="utf-8")
    glossary = terms_dir / "master_glossary.txt"
    glossary.write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    captured: dict[str, object] = {}
    triage_calls: list[Path] = []

    def fake_run_batch_pipeline(story_yaml: Path, **kwargs):  # noqa: ANN003
        captured["story_yaml"] = story_yaml
        captured.update(kwargs)
        run_dir = tmp_path / "runs" / "local_probe"
        run_dir.mkdir(parents=True)
        manifest = BatchManifest.create(
            run_id="local_probe",
            story_slug="local_fixture",
            title="local_fixture",
            story_yaml=story_yaml,
            chapters=kwargs["chapters"],
            mode=kwargs["provider_mode"],
            providers={
                "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
                "judge": ProviderLabel(provider="deepseek", model="deepseek-chat"),
                "repair": ProviderLabel(provider="offline", model="offline-fixture-v1"),
            },
            run_dir=run_dir,
        )
        for chapter_run in manifest.chapters.values():
            chapter_run.status = "packaged"
        manifest.refresh_summary()
        manifest_path = run_dir / "batch_manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return BatchPipelineResult(run_dir=run_dir, manifest_path=manifest_path, manifest=manifest)

    monkeypatch.setattr("agentic_translation.cli.run_batch_pipeline", fake_run_batch_pipeline)
    monkeypatch.setattr(
        "agentic_translation.cli.write_batch_triage_artifacts",
        lambda run_dir: triage_calls.append(run_dir) or {"review_queue": "review_queue.json"},
    )

    result = CliRunner().invoke(
        app,
        [
            "import-local",
            "--source-dir",
            str(source_dir),
            "--glossary",
            str(glossary),
            "--chapters",
            "0001-0002",
            "--translated-dir",
            str(translated_dir),
            "--out",
            str(out),
            "--run-batch",
            "--provider-mode",
            "live",
            "--translation-provider",
            "offline",
            "--judge-provider",
            "deepseek",
            "--repair-provider",
            "offline",
            "--record-cache",
            "--cache-dir",
            str(cache_dir),
            "--model",
            "deepseek-chat",
            "--run-id",
            "local_probe",
            "--overwrite",
            "--force",
            "--allow-source-qa-fail",
            "--allow-live-provider-fallback",
            "--allow-review-required",
            "--skip-epub",
            "--write-proof",
            "--write-triage",
            "--report-mode",
            "excerpt",
        ],
    )

    assert result.exit_code == 0
    assert captured["story_yaml"] == out / "story.yaml"
    assert captured["chapters"] == ["0001", "0002"]
    assert captured["provider_mode"] == "live"
    assert captured["translation_provider_name"] == "offline"
    assert captured["judge_provider_name"] == "deepseek"
    assert captured["repair_provider_name"] == "offline"
    assert captured["record_cache"] is True
    assert captured["cache_dir"] == cache_dir
    assert captured["model_name"] == "deepseek-chat"
    assert captured["run_id"] == "local_probe"
    assert captured["overwrite"] is True
    assert captured["force"] is True
    assert captured["skip_epub"] is True
    assert captured["allow_source_qa_fail"] is True
    assert captured["allow_live_provider_fallback"] is True
    assert captured["report_mode"] == "excerpt"
    assert captured["write_proof"] is True
    assert triage_calls == [tmp_path / "runs" / "local_probe"]


def test_smoke_local_deepseek_shorthand_runs_imported_batch(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    source_dir = tmp_path / "simulator" / "scraped"
    translated_dir = tmp_path / "simulator" / "translated_001_002"
    terms_dir = tmp_path / "simulator" / "terms"
    cache_dir = tmp_path / "cache"
    source_dir.mkdir(parents=True)
    translated_dir.mkdir()
    terms_dir.mkdir()
    for chapter in ["0001", "0002"]:
        (source_dir / f"{chapter}.txt").write_text(f"第{chapter}章\n\n天道", encoding="utf-8")
        (translated_dir / f"{chapter}.txt").write_text(f"Chapter {int(chapter)}\n\nHeavenly Dao", encoding="utf-8")
    glossary = terms_dir / "master_glossary.txt"
    glossary.write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    captured: dict[str, object] = {}
    triage_calls: list[Path] = []

    def fake_run_batch_pipeline(story_yaml: Path, **kwargs):  # noqa: ANN003
        captured["story_yaml"] = story_yaml
        captured.update(kwargs)
        run_dir = tmp_path / "runs" / "smoke_probe"
        run_dir.mkdir(parents=True)
        manifest = BatchManifest.create(
            run_id="smoke_probe",
            story_slug=story_yaml.parent.name,
            title=story_yaml.parent.name,
            story_yaml=story_yaml,
            chapters=kwargs["chapters"],
            mode=kwargs["provider_mode"],
            providers={
                "translation": ProviderLabel(provider="deepseek", model="deepseek-chat"),
                "judge": ProviderLabel(provider="offline", model="offline-fixture-v1"),
                "repair": ProviderLabel(provider="offline", model="offline-fixture-v1"),
            },
            run_dir=run_dir,
        )
        for chapter_run in manifest.chapters.values():
            chapter_run.status = "packaged"
        manifest.refresh_summary()
        manifest_path = run_dir / "batch_manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return BatchPipelineResult(run_dir=run_dir, manifest_path=manifest_path, manifest=manifest)

    monkeypatch.setattr("agentic_translation.cli.run_batch_pipeline", fake_run_batch_pipeline)
    monkeypatch.setattr(
        "agentic_translation.cli.write_batch_triage_artifacts",
        lambda run_dir: triage_calls.append(run_dir) or {"review_queue": "review_queue.json"},
    )

    result = CliRunner().invoke(
        app,
        [
            "smoke-local",
            "--source-dir",
            str(source_dir),
            "--glossary",
            str(glossary),
            "--chapters",
            "0001-0002",
            "--translated-dir",
            str(translated_dir),
            "--deepseek",
            "--cache-dir",
            str(cache_dir),
            "--run-id",
            "smoke_probe",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0
    assert captured["story_yaml"] == Path("local_fixtures") / "simulator_smoke_0001_0002" / "story.yaml"
    assert captured["chapters"] == ["0001", "0002"]
    assert captured["provider_mode"] == "live"
    assert captured["translation_provider_name"] == "deepseek"
    assert captured["judge_provider_name"] == "offline"
    assert captured["repair_provider_name"] == "offline"
    assert captured["record_cache"] is True
    assert captured["cache_dir"] == cache_dir
    assert captured["model_name"] == "deepseek-chat"
    assert captured["allow_live_provider_fallback"] is True
    assert captured["allow_source_qa_fail"] is True
    assert captured["report_mode"] == "excerpt"
    assert captured["write_proof"] is True
    assert triage_calls == [tmp_path / "runs" / "smoke_probe"]


def test_smoke_local_source_char_limit_truncates_imported_fixture(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    source_dir = tmp_path / "simulator" / "scraped"
    translated_dir = tmp_path / "simulator" / "translated_001_001"
    terms_dir = tmp_path / "simulator" / "terms"
    source_dir.mkdir(parents=True)
    translated_dir.mkdir()
    terms_dir.mkdir()
    (source_dir / "0001.txt").write_text("第1章\n\n" + ("天道降临。" * 50), encoding="utf-8")
    (translated_dir / "0001.txt").write_text("Chapter 1\n\n" + ("The Heavenly Dao descends. " * 50), encoding="utf-8")
    glossary = terms_dir / "master_glossary.txt"
    glossary.write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    fixture = tmp_path / "fixture"

    def fake_run_batch_pipeline(story_yaml: Path, **kwargs):  # noqa: ANN003
        run_dir = tmp_path / "runs" / "limited_probe"
        run_dir.mkdir(parents=True)
        manifest = BatchManifest.create(
            run_id="limited_probe",
            story_slug=story_yaml.parent.name,
            title=story_yaml.parent.name,
            story_yaml=story_yaml,
            chapters=kwargs["chapters"],
            mode=kwargs["provider_mode"],
            providers={},
            run_dir=run_dir,
        )
        for chapter_run in manifest.chapters.values():
            chapter_run.status = "packaged"
        manifest.refresh_summary()
        manifest_path = run_dir / "batch_manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return BatchPipelineResult(run_dir=run_dir, manifest_path=manifest_path, manifest=manifest)

    monkeypatch.setattr("agentic_translation.cli.run_batch_pipeline", fake_run_batch_pipeline)
    monkeypatch.setattr("agentic_translation.cli.write_batch_triage_artifacts", lambda run_dir: {})

    result = CliRunner().invoke(
        app,
        [
            "smoke-local",
            "--source-dir",
            str(source_dir),
            "--glossary",
            str(glossary),
            "--chapters",
            "0001",
            "--translated-dir",
            str(translated_dir),
            "--out",
            str(fixture),
            "--source-char-limit",
            "40",
            "--run-id",
            "limited_probe",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0
    assert len((fixture / "source" / "0001.txt").read_text(encoding="utf-8")) <= 41
    assert len((fixture / "expected" / "dirty_translation.txt").read_text(encoding="utf-8")) <= 41
    assert (fixture / "source" / "0001.txt").read_text(encoding="utf-8").startswith("第1章")
    assert (fixture / "expected" / "dirty_translation.txt").read_text(encoding="utf-8").startswith("Chapter 1")


def test_smoke_local_can_apply_glossary_pass_after_batch(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    source_dir = tmp_path / "simulator" / "scraped"
    translated_dir = tmp_path / "simulator" / "translated_001_001"
    terms_dir = tmp_path / "simulator" / "terms"
    source_dir.mkdir(parents=True)
    translated_dir.mkdir()
    terms_dir.mkdir()
    (source_dir / "0001.txt").write_text("第1章\n\n天道", encoding="utf-8")
    (translated_dir / "0001.txt").write_text("Chapter 1\n\nHeavenly Dao", encoding="utf-8")
    glossary = terms_dir / "master_glossary.txt"
    glossary.write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run_batch_pipeline(story_yaml: Path, **kwargs):  # noqa: ANN003
        run_dir = tmp_path / "runs" / "smoke_probe"
        run_dir.mkdir(parents=True)
        manifest = BatchManifest.create(
            run_id="smoke_probe",
            story_slug=story_yaml.parent.name,
            title=story_yaml.parent.name,
            story_yaml=story_yaml,
            chapters=kwargs["chapters"],
            mode=kwargs["provider_mode"],
            providers={},
            run_dir=run_dir,
        )
        manifest.chapters["0001"].status = "review_required"
        manifest.refresh_summary()
        manifest_path = run_dir / "batch_manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return BatchPipelineResult(run_dir=run_dir, manifest_path=manifest_path, manifest=manifest)

    def fake_run_glossary_update_pass(run_dir: Path, **kwargs):  # noqa: ANN003
        captured["run_dir"] = run_dir
        captured.update(kwargs)
        return GlossaryUpdatePassResult(
            run_id="smoke_probe",
            run_dir=str(run_dir),
            dry_run=False,
            chapters=["0001"],
            rerun_started=True,
            message="Applied glossary updates and reran affected review chapters.",
            application=GlossaryUpdateApplication(
                run_id="smoke_probe",
                story_slug="simulator_smoke_0001",
                run_dir=str(run_dir),
                glossary_path=str(glossary),
                dry_run=False,
                summary=GlossaryUpdateApplicationSummary(changed_count=1, updated_count=1, total_items=1),
            ),
            before_summary=BatchSummary(total_chapters=1, review_required=1),
            after_summary=BatchSummary(total_chapters=1, packaged=1),
        )

    monkeypatch.setattr("agentic_translation.cli.run_batch_pipeline", fake_run_batch_pipeline)
    monkeypatch.setattr("agentic_translation.cli.run_glossary_update_pass", fake_run_glossary_update_pass)
    monkeypatch.setattr("agentic_translation.cli.write_batch_triage_artifacts", lambda run_dir: {})

    result = CliRunner().invoke(
        app,
        [
            "smoke-local",
            "--source-dir",
            str(source_dir),
            "--glossary",
            str(glossary),
            "--chapters",
            "0001",
            "--translated-dir",
            str(translated_dir),
            "--out",
            str(tmp_path / "fixture"),
            "--run-id",
            "smoke_probe",
            "--overwrite",
            "--glossary-pass",
        ],
    )

    assert result.exit_code == 0
    assert captured["run_dir"] == tmp_path / "runs" / "smoke_probe"
    assert captured["write"] is True
    assert captured["chapters"] == ["0001"]
    assert captured["allow_source_qa_fail"] is True
    assert captured["write_triage"] is True
    assert "Applied glossary updates and reran" in result.output


def test_smoke_local_can_run_glossary_pass_and_normalize_panels(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    source_dir = tmp_path / "simulator" / "scraped"
    translated_dir = tmp_path / "simulator" / "translated_001_001"
    terms_dir = tmp_path / "simulator" / "terms"
    source_dir.mkdir(parents=True)
    translated_dir.mkdir()
    terms_dir.mkdir()
    (source_dir / "0001.txt").write_text("第1章\n\n天道", encoding="utf-8")
    (translated_dir / "0001.txt").write_text("Chapter 1\n\nHeavenly Dao", encoding="utf-8")
    glossary = terms_dir / "master_glossary.txt"
    glossary.write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    calls: list[str] = []
    normalize_captured: dict[str, object] = {}
    triage_calls: list[Path] = []
    summary_calls: list[Path] = []

    def fake_run_batch_pipeline(story_yaml: Path, **kwargs):  # noqa: ANN003
        run_dir = tmp_path / "runs" / "smoke_probe"
        run_dir.mkdir(parents=True)
        manifest = BatchManifest.create(
            run_id="smoke_probe",
            story_slug=story_yaml.parent.name,
            title=story_yaml.parent.name,
            story_yaml=story_yaml,
            chapters=kwargs["chapters"],
            mode=kwargs["provider_mode"],
            providers={},
            run_dir=run_dir,
        )
        manifest.chapters["0001"].status = "review_required"
        manifest.refresh_summary()
        manifest_path = run_dir / "batch_manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return BatchPipelineResult(run_dir=run_dir, manifest_path=manifest_path, manifest=manifest)

    def fake_run_glossary_update_pass(run_dir: Path, **kwargs):  # noqa: ANN003
        calls.append("glossary_pass")
        return GlossaryUpdatePassResult(
            run_id="smoke_probe",
            run_dir=str(run_dir),
            dry_run=False,
            chapters=["0001"],
            rerun_started=True,
            message="Applied glossary updates and reran affected review chapters.",
            application=GlossaryUpdateApplication(
                run_id="smoke_probe",
                story_slug="simulator_smoke_0001",
                run_dir=str(run_dir),
                glossary_path=str(glossary),
                dry_run=False,
                summary=GlossaryUpdateApplicationSummary(changed_count=1, updated_count=1, total_items=1),
            ),
            before_summary=BatchSummary(total_chapters=1, review_required=1),
            after_summary=BatchSummary(total_chapters=1, review_required=1),
        )

    def fake_normalize_panel_splits(run_dir: Path, **kwargs):  # noqa: ANN003
        calls.append("normalize")
        normalize_captured["run_dir"] = run_dir
        normalize_captured.update(kwargs)
        return PanelNormalizationResult(
            run_id="smoke_probe",
            run_dir=str(run_dir),
            normalized_count=1,
            skipped_count=0,
            summary_after=BatchSummary(total_chapters=1, packaged=1),
        )

    monkeypatch.setattr("agentic_translation.cli.run_batch_pipeline", fake_run_batch_pipeline)
    monkeypatch.setattr("agentic_translation.cli.run_glossary_update_pass", fake_run_glossary_update_pass)
    monkeypatch.setattr("agentic_translation.cli.normalize_panel_splits", fake_normalize_panel_splits)
    monkeypatch.setattr(
        "agentic_translation.cli.write_batch_triage_artifacts",
        lambda run_dir: triage_calls.append(run_dir) or {"review_queue": "review_queue.json"},
    )
    monkeypatch.setattr("agentic_translation.cli._print_batch_summary", lambda path: summary_calls.append(path))
    monkeypatch.setattr("agentic_translation.cli._exit_if_batch_has_blockers", lambda *args, **kwargs: None)

    result = CliRunner().invoke(
        app,
        [
            "smoke-local",
            "--source-dir",
            str(source_dir),
            "--glossary",
            str(glossary),
            "--chapters",
            "0001",
            "--translated-dir",
            str(translated_dir),
            "--out",
            str(tmp_path / "fixture"),
            "--run-id",
            "smoke_probe",
            "--overwrite",
            "--glossary-pass",
            "--normalize-panels",
            "--panel-reviewer",
            "codex",
            "--panel-note-prefix",
            "Merge split panels.",
        ],
    )

    assert result.exit_code == 0
    assert calls == ["glossary_pass", "normalize"]
    assert normalize_captured["run_dir"] == tmp_path / "runs" / "smoke_probe"
    assert normalize_captured["chapters"] == ["0001"]
    assert normalize_captured["reviewer"] == "codex"
    assert normalize_captured["note_prefix"] == "Merge split panels."
    assert normalize_captured["write_proof"] is True
    assert triage_calls[-1] == tmp_path / "runs" / "smoke_probe"
    assert summary_calls == [
        tmp_path / "runs" / "smoke_probe" / "batch_manifest.json",
        tmp_path / "runs" / "smoke_probe" / "batch_manifest.json",
    ]
    assert "Normalized 1 panel split(s)" in result.output


def test_smoke_local_practical_enables_useful_corpus_loop(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    source_dir = tmp_path / "simulator" / "scraped"
    translated_dir = tmp_path / "simulator" / "translated_001_001"
    terms_dir = tmp_path / "simulator" / "terms"
    source_dir.mkdir(parents=True)
    translated_dir.mkdir()
    terms_dir.mkdir()
    (source_dir / "0001.txt").write_text("第1章\n\n天道", encoding="utf-8")
    (translated_dir / "0001.txt").write_text("Chapter 1\n\nHeavenly Dao", encoding="utf-8")
    glossary = terms_dir / "master_glossary.txt"
    glossary.write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    calls: list[str] = []
    normalize_captured: dict[str, object] = {}

    def fake_run_batch_pipeline(story_yaml: Path, **kwargs):  # noqa: ANN003
        run_dir = tmp_path / "runs" / "practical_probe"
        run_dir.mkdir(parents=True)
        manifest = BatchManifest.create(
            run_id="practical_probe",
            story_slug=story_yaml.parent.name,
            title=story_yaml.parent.name,
            story_yaml=story_yaml,
            chapters=kwargs["chapters"],
            mode=kwargs["provider_mode"],
            providers={},
            run_dir=run_dir,
        )
        manifest.chapters["0001"].status = "review_required"
        manifest.refresh_summary()
        manifest_path = run_dir / "batch_manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return BatchPipelineResult(run_dir=run_dir, manifest_path=manifest_path, manifest=manifest)

    def fake_run_glossary_update_pass(run_dir: Path, **kwargs):  # noqa: ANN003
        calls.append("glossary_pass")
        return GlossaryUpdatePassResult(
            run_id="practical_probe",
            run_dir=str(run_dir),
            dry_run=False,
            chapters=["0001"],
            rerun_started=True,
            message="Applied glossary updates and reran affected review chapters.",
            application=GlossaryUpdateApplication(
                run_id="practical_probe",
                story_slug="simulator_smoke_0001",
                run_dir=str(run_dir),
                glossary_path=str(glossary),
                dry_run=False,
                summary=GlossaryUpdateApplicationSummary(changed_count=1, updated_count=1, total_items=1),
            ),
            before_summary=BatchSummary(total_chapters=1, review_required=1),
            after_summary=BatchSummary(total_chapters=1, review_required=1),
        )

    def fake_normalize_panel_splits(run_dir: Path, **kwargs):  # noqa: ANN003
        calls.append("normalize")
        normalize_captured.update(kwargs)
        return PanelNormalizationResult(
            run_id="practical_probe",
            run_dir=str(run_dir),
            normalized_count=1,
            skipped_count=0,
            summary_after=BatchSummary(total_chapters=1, packaged=1),
        )

    monkeypatch.setattr("agentic_translation.cli.run_batch_pipeline", fake_run_batch_pipeline)
    monkeypatch.setattr("agentic_translation.cli.run_glossary_update_pass", fake_run_glossary_update_pass)
    monkeypatch.setattr("agentic_translation.cli.normalize_panel_splits", fake_normalize_panel_splits)
    monkeypatch.setattr("agentic_translation.cli.write_batch_triage_artifacts", lambda run_dir: {})
    monkeypatch.setattr("agentic_translation.cli._exit_if_batch_has_blockers", lambda *args, **kwargs: None)

    result = CliRunner().invoke(
        app,
        [
            "smoke-local",
            "--source-dir",
            str(source_dir),
            "--glossary",
            str(glossary),
            "--chapters",
            "0001",
            "--translated-dir",
            str(translated_dir),
            "--out",
            str(tmp_path / "fixture"),
            "--run-id",
            "practical_probe",
            "--overwrite",
            "--practical",
        ],
    )

    assert result.exit_code == 0
    assert calls == ["glossary_pass", "normalize"]
    assert normalize_captured["reviewer"] == "codex"
    assert normalize_captured["note_prefix"] == "Merged split corpus panel."


def test_smoke_project_infers_project_layout_and_runs_smoke_passes(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    project_dir = tmp_path / "simulator_alliance"
    source_dir = project_dir / "scraped"
    translated_dir = project_dir / "translated_001_002"
    terms_dir = project_dir / "terms"
    source_dir.mkdir(parents=True)
    translated_dir.mkdir()
    terms_dir.mkdir()
    for chapter in ["0001", "0002"]:
        (source_dir / f"{chapter}.txt").write_text(f"第{chapter}章\n\n天道", encoding="utf-8")
        (translated_dir / f"{chapter}.txt").write_text(f"Chapter {int(chapter)}\n\nHeavenly Dao", encoding="utf-8")
    glossary = terms_dir / "master_glossary.txt"
    glossary.write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    captured: dict[str, object] = {}
    calls: list[str] = []
    triage_calls: list[Path] = []

    def fake_run_batch_pipeline(story_yaml: Path, **kwargs):  # noqa: ANN003
        captured["story_yaml"] = story_yaml
        captured.update(kwargs)
        run_dir = tmp_path / "runs" / "project_smoke"
        run_dir.mkdir(parents=True)
        manifest = BatchManifest.create(
            run_id="project_smoke",
            story_slug=story_yaml.parent.name,
            title=story_yaml.parent.name,
            story_yaml=story_yaml,
            chapters=kwargs["chapters"],
            mode=kwargs["provider_mode"],
            providers={},
            run_dir=run_dir,
        )
        for chapter_run in manifest.chapters.values():
            chapter_run.status = "review_required"
        manifest.refresh_summary()
        manifest_path = run_dir / "batch_manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return BatchPipelineResult(run_dir=run_dir, manifest_path=manifest_path, manifest=manifest)

    def fake_run_glossary_update_pass(run_dir: Path, **kwargs):  # noqa: ANN003
        calls.append("glossary_pass")
        return GlossaryUpdatePassResult(
            run_id="project_smoke",
            run_dir=str(run_dir),
            dry_run=False,
            chapters=["0001", "0002"],
            rerun_started=True,
            message="Applied glossary updates and reran affected review chapters.",
            application=GlossaryUpdateApplication(
                run_id="project_smoke",
                story_slug="simulator_alliance_smoke_0001_0002",
                run_dir=str(run_dir),
                glossary_path=str(glossary),
                dry_run=False,
                summary=GlossaryUpdateApplicationSummary(changed_count=1, updated_count=1, total_items=1),
            ),
            before_summary=BatchSummary(total_chapters=2, review_required=2),
            after_summary=BatchSummary(total_chapters=2, review_required=2),
        )

    def fake_normalize_panel_splits(run_dir: Path, **kwargs):  # noqa: ANN003
        calls.append("normalize")
        captured["normalize_run_dir"] = run_dir
        captured["normalize_kwargs"] = kwargs
        return PanelNormalizationResult(
            run_id="project_smoke",
            run_dir=str(run_dir),
            normalized_count=1,
            skipped_count=0,
            summary_after=BatchSummary(total_chapters=2, packaged=2),
        )

    monkeypatch.setattr("agentic_translation.cli.run_batch_pipeline", fake_run_batch_pipeline)
    monkeypatch.setattr("agentic_translation.cli.run_glossary_update_pass", fake_run_glossary_update_pass)
    monkeypatch.setattr("agentic_translation.cli.normalize_panel_splits", fake_normalize_panel_splits)
    monkeypatch.setattr(
        "agentic_translation.cli.write_batch_triage_artifacts",
        lambda run_dir: triage_calls.append(run_dir) or {"review_queue": "review_queue.json"},
    )
    monkeypatch.setattr("agentic_translation.cli._exit_if_batch_has_blockers", lambda *args, **kwargs: None)

    result = CliRunner().invoke(
        app,
        [
            "smoke-project",
            str(project_dir),
            "--chapters",
            "0001-0002",
            "--run-id",
            "project_smoke",
            "--overwrite",
            "--practical",
        ],
    )

    fixture_out = Path("local_fixtures") / "simulator_alliance_smoke_0001_0002"
    assert result.exit_code == 0
    assert captured["story_yaml"] == fixture_out / "story.yaml"
    assert captured["chapters"] == ["0001", "0002"]
    assert captured["allow_source_qa_fail"] is True
    assert captured["allow_live_provider_fallback"] is True
    assert captured["report_mode"] == "excerpt"
    story = yaml.safe_load((fixture_out / "story.yaml").read_text(encoding="utf-8"))
    assert story["paths"]["source_dir"] == str(fixture_out / "source")
    assert story["paths"]["glossary_path"] == str(fixture_out / "terms" / "master_glossary.txt")
    assert story["paths"]["baseline_dir"] == str(fixture_out / "baseline")
    assert (fixture_out / "baseline" / "0001.txt").read_text(encoding="utf-8") == "Chapter 1\n\nHeavenly Dao"
    assert calls == ["glossary_pass", "normalize"]
    assert captured["normalize_run_dir"] == tmp_path / "runs" / "project_smoke"
    assert captured["normalize_kwargs"]["chapters"] == ["0001", "0002"]
    assert captured["normalize_kwargs"]["reviewer"] == "codex"
    assert captured["normalize_kwargs"]["note_prefix"] == "Merged split corpus panel."
    assert triage_calls[-1] == tmp_path / "runs" / "project_smoke"
    assert "Using source directory" in result.output
    assert "Using translated baseline" in result.output


def test_smoke_project_requires_expected_project_layout(tmp_path: Path) -> None:
    project_dir = tmp_path / "broken_project"
    project_dir.mkdir()

    result = CliRunner().invoke(app, ["smoke-project", str(project_dir), "--chapters", "0001"])

    assert result.exit_code != 0
    assert "scraped" in result.output
    assert "terms/master_glossary.txt" in result.output


def test_smoke_project_can_select_first_chapters_from_project(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    project_dir = tmp_path / "simulator_alliance"
    source_dir = project_dir / "scraped"
    translated_dir = project_dir / "translated_001_003"
    terms_dir = project_dir / "terms"
    source_dir.mkdir(parents=True)
    translated_dir.mkdir()
    terms_dir.mkdir()
    for chapter in ["0001", "0002", "0003"]:
        (source_dir / f"{chapter}.txt").write_text(f"第{chapter}章\n\n天道", encoding="utf-8")
        (translated_dir / f"{chapter}.txt").write_text(f"Chapter {int(chapter)}\n\nHeavenly Dao", encoding="utf-8")
    (terms_dir / "master_glossary.txt").write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run_batch_pipeline(story_yaml: Path, **kwargs):  # noqa: ANN003
        captured["story_yaml"] = story_yaml
        captured.update(kwargs)
        run_dir = tmp_path / "runs" / "project_first"
        run_dir.mkdir(parents=True)
        manifest = BatchManifest.create(
            run_id="project_first",
            story_slug=story_yaml.parent.name,
            title=story_yaml.parent.name,
            story_yaml=story_yaml,
            chapters=kwargs["chapters"],
            mode=kwargs["provider_mode"],
            providers={},
            run_dir=run_dir,
        )
        for chapter_run in manifest.chapters.values():
            chapter_run.status = "packaged"
        manifest.refresh_summary()
        manifest_path = run_dir / "batch_manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return BatchPipelineResult(run_dir=run_dir, manifest_path=manifest_path, manifest=manifest)

    monkeypatch.setattr("agentic_translation.cli.run_batch_pipeline", fake_run_batch_pipeline)
    monkeypatch.setattr("agentic_translation.cli.write_batch_triage_artifacts", lambda run_dir: {})

    result = CliRunner().invoke(
        app,
        [
            "smoke-project",
            str(project_dir),
            "--first",
            "2",
            "--start",
            "0002",
            "--run-id",
            "project_first",
            "--overwrite",
        ],
    )

    fixture_out = Path("local_fixtures") / "simulator_alliance_smoke_0002_0003"
    assert result.exit_code == 0
    assert captured["chapters"] == ["0002", "0003"]
    assert captured["story_yaml"] == fixture_out / "story.yaml"
    assert (fixture_out / "baseline" / "0002.txt").exists()
    assert (fixture_out / "baseline" / "0003.txt").exists()
    assert "Selected chapters: 0002,0003" in result.output


def test_smoke_project_can_continue_after_previous_run(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    project_dir = tmp_path / "simulator_alliance"
    source_dir = project_dir / "scraped"
    translated_dir = project_dir / "translated_001_004"
    terms_dir = project_dir / "terms"
    source_dir.mkdir(parents=True)
    translated_dir.mkdir()
    terms_dir.mkdir()
    for chapter in ["0001", "0002", "0003", "0004"]:
        (source_dir / f"{chapter}.txt").write_text(f"第{chapter}章\n\n天道", encoding="utf-8")
        (translated_dir / f"{chapter}.txt").write_text(f"Chapter {int(chapter)}\n\nHeavenly Dao", encoding="utf-8")
    (terms_dir / "master_glossary.txt").write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    previous_run_dir = tmp_path / "runs" / "previous"
    previous_run_dir.mkdir(parents=True)
    previous_manifest = BatchManifest.create(
        run_id="previous",
        story_slug="simulator_alliance_smoke_0001_0002",
        title="simulator_alliance_smoke_0001_0002",
        story_yaml=tmp_path / "previous_story.yaml",
        chapters=["0001", "0002"],
        mode="offline",
        providers={},
        run_dir=previous_run_dir,
    )
    (previous_run_dir / "batch_manifest.json").write_text(previous_manifest.model_dump_json(indent=2), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run_batch_pipeline(story_yaml: Path, **kwargs):  # noqa: ANN003
        captured["story_yaml"] = story_yaml
        captured.update(kwargs)
        run_dir = tmp_path / "runs" / "continued"
        run_dir.mkdir(parents=True)
        manifest = BatchManifest.create(
            run_id="continued",
            story_slug=story_yaml.parent.name,
            title=story_yaml.parent.name,
            story_yaml=story_yaml,
            chapters=kwargs["chapters"],
            mode=kwargs["provider_mode"],
            providers={},
            run_dir=run_dir,
        )
        for chapter_run in manifest.chapters.values():
            chapter_run.status = "packaged"
        manifest.refresh_summary()
        manifest_path = run_dir / "batch_manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return BatchPipelineResult(run_dir=run_dir, manifest_path=manifest_path, manifest=manifest)

    monkeypatch.setattr("agentic_translation.cli.run_batch_pipeline", fake_run_batch_pipeline)
    monkeypatch.setattr("agentic_translation.cli.write_batch_triage_artifacts", lambda run_dir: {})

    result = CliRunner().invoke(
        app,
        [
            "smoke-project",
            str(project_dir),
            "--after-run",
            str(previous_run_dir),
            "--first",
            "2",
            "--run-id",
            "continued",
            "--overwrite",
        ],
    )

    fixture_out = Path("local_fixtures") / "simulator_alliance_smoke_0003_0004"
    assert result.exit_code == 0
    assert captured["chapters"] == ["0003", "0004"]
    assert captured["story_yaml"] == fixture_out / "story.yaml"
    assert (fixture_out / "baseline" / "0003.txt").exists()
    assert (fixture_out / "baseline" / "0004.txt").exists()
    assert "Continuing after run" in result.output


def test_smoke_project_infers_run_id_for_selected_chapters(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    project_dir = tmp_path / "simulator_alliance"
    source_dir = project_dir / "scraped"
    translated_dir = project_dir / "translated_001_004"
    terms_dir = project_dir / "terms"
    source_dir.mkdir(parents=True)
    translated_dir.mkdir()
    terms_dir.mkdir()
    for chapter in ["0001", "0002", "0003", "0004"]:
        (source_dir / f"{chapter}.txt").write_text(f"第{chapter}章\n\n天道", encoding="utf-8")
        (translated_dir / f"{chapter}.txt").write_text(f"Chapter {int(chapter)}\n\nHeavenly Dao", encoding="utf-8")
    (terms_dir / "master_glossary.txt").write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    previous_run_dir = tmp_path / "runs" / "previous"
    previous_run_dir.mkdir(parents=True)
    previous_manifest = BatchManifest.create(
        run_id="previous",
        story_slug="simulator_alliance_smoke_0001_0002",
        title="simulator_alliance_smoke_0001_0002",
        story_yaml=tmp_path / "previous_story.yaml",
        chapters=["0001", "0002"],
        mode="offline",
        providers={},
        run_dir=previous_run_dir,
    )
    (previous_run_dir / "batch_manifest.json").write_text(previous_manifest.model_dump_json(indent=2), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run_batch_pipeline(story_yaml: Path, **kwargs):  # noqa: ANN003
        captured.update(kwargs)
        run_dir = tmp_path / "runs" / str(kwargs["run_id"])
        run_dir.mkdir(parents=True)
        manifest = BatchManifest.create(
            run_id=str(kwargs["run_id"]),
            story_slug=story_yaml.parent.name,
            title=story_yaml.parent.name,
            story_yaml=story_yaml,
            chapters=kwargs["chapters"],
            mode=kwargs["provider_mode"],
            providers={},
            run_dir=run_dir,
        )
        for chapter_run in manifest.chapters.values():
            chapter_run.status = "packaged"
        manifest.refresh_summary()
        manifest_path = run_dir / "batch_manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return BatchPipelineResult(run_dir=run_dir, manifest_path=manifest_path, manifest=manifest)

    monkeypatch.setattr("agentic_translation.cli.run_batch_pipeline", fake_run_batch_pipeline)
    monkeypatch.setattr("agentic_translation.cli.write_batch_triage_artifacts", lambda run_dir: {})

    result = CliRunner().invoke(
        app,
        [
            "smoke-project",
            str(project_dir),
            "--after-run",
            str(previous_run_dir),
            "--first",
            "2",
            "--practical",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0
    assert captured["run_id"] == "simulator_alliance_practical_0003_0004"
    assert "Run id: simulator_alliance_practical_0003_0004" in result.output
    assert "Selected chapters: 0003,0004" in result.output


def test_smoke_project_can_continue_after_latest_matching_project_run(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    project_dir = tmp_path / "simulator_alliance"
    source_dir = project_dir / "scraped"
    translated_dir = project_dir / "translated_001_008"
    terms_dir = project_dir / "terms"
    source_dir.mkdir(parents=True)
    translated_dir.mkdir()
    terms_dir.mkdir()
    for chapter in [f"{number:04d}" for number in range(1, 9)]:
        (source_dir / f"{chapter}.txt").write_text(f"第{chapter}章\n\n天道", encoding="utf-8")
        (translated_dir / f"{chapter}.txt").write_text(f"Chapter {int(chapter)}\n\nHeavenly Dao", encoding="utf-8")
    (terms_dir / "master_glossary.txt").write_text("天道 -> Heavenly Dao\n", encoding="utf-8")

    def write_previous_run(run_id: str, story_slug: str, chapters: list[str]) -> Path:
        run_dir = tmp_path / "runs" / run_id
        run_dir.mkdir(parents=True)
        manifest = BatchManifest.create(
            run_id=run_id,
            story_slug=story_slug,
            title=story_slug,
            story_yaml=tmp_path / f"{run_id}.yaml",
            chapters=chapters,
            mode="offline",
            providers={},
            run_dir=run_dir,
        )
        (run_dir / "batch_manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return run_dir

    write_previous_run("lower", "simulator_alliance_smoke_0001_0002", ["0001", "0002"])
    latest_run_dir = write_previous_run("higher", "simulator_alliance_smoke_0005_0006", ["0005", "0006"])
    write_previous_run("irrelevant", "other_project_smoke_0099", ["0099"])
    captured: dict[str, object] = {}

    def fake_run_batch_pipeline(story_yaml: Path, **kwargs):  # noqa: ANN003
        captured["story_yaml"] = story_yaml
        captured.update(kwargs)
        run_dir = tmp_path / "runs" / str(kwargs["run_id"])
        run_dir.mkdir(parents=True)
        manifest = BatchManifest.create(
            run_id=str(kwargs["run_id"]),
            story_slug=story_yaml.parent.name,
            title=story_yaml.parent.name,
            story_yaml=story_yaml,
            chapters=kwargs["chapters"],
            mode=kwargs["provider_mode"],
            providers={},
            run_dir=run_dir,
        )
        for chapter_run in manifest.chapters.values():
            chapter_run.status = "packaged"
        manifest.refresh_summary()
        manifest_path = run_dir / "batch_manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return BatchPipelineResult(run_dir=run_dir, manifest_path=manifest_path, manifest=manifest)

    monkeypatch.setattr("agentic_translation.cli.run_batch_pipeline", fake_run_batch_pipeline)
    monkeypatch.setattr("agentic_translation.cli.write_batch_triage_artifacts", lambda run_dir: {})

    result = CliRunner().invoke(
        app,
        [
            "smoke-project",
            str(project_dir),
            "--continue-latest",
            "--first",
            "2",
            "--practical",
            "--overwrite",
        ],
    )

    fixture_out = Path("local_fixtures") / "simulator_alliance_smoke_0007_0008"
    assert result.exit_code == 0
    assert captured["chapters"] == ["0007", "0008"]
    assert captured["run_id"] == "simulator_alliance_practical_0007_0008"
    assert captured["story_yaml"] == fixture_out / "story.yaml"
    assert "Continuing after latest matching run" in result.output
    assert latest_run_dir.name in result.output
    assert "Selected chapters: 0007,0008" in result.output


def test_smoke_project_can_run_multiple_continue_latest_chunks(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    project_dir = tmp_path / "simulator_alliance"
    source_dir = project_dir / "scraped"
    translated_dir = project_dir / "translated_001_006"
    terms_dir = project_dir / "terms"
    source_dir.mkdir(parents=True)
    translated_dir.mkdir()
    terms_dir.mkdir()
    for chapter in [f"{number:04d}" for number in range(1, 7)]:
        (source_dir / f"{chapter}.txt").write_text(f"第{chapter}章\n\n天道", encoding="utf-8")
        (translated_dir / f"{chapter}.txt").write_text(f"Chapter {int(chapter)}\n\nHeavenly Dao", encoding="utf-8")
    (terms_dir / "master_glossary.txt").write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    previous_run_dir = tmp_path / "runs" / "previous"
    previous_run_dir.mkdir(parents=True)
    previous_manifest = BatchManifest.create(
        run_id="previous",
        story_slug="simulator_alliance_smoke_0001_0002",
        title="simulator_alliance_smoke_0001_0002",
        story_yaml=tmp_path / "previous.yaml",
        chapters=["0001", "0002"],
        mode="offline",
        providers={},
        run_dir=previous_run_dir,
    )
    (previous_run_dir / "batch_manifest.json").write_text(previous_manifest.model_dump_json(indent=2), encoding="utf-8")
    calls: list[tuple[list[str], str]] = []

    def fake_run_batch_pipeline(story_yaml: Path, **kwargs):  # noqa: ANN003
        chapters = list(kwargs["chapters"])
        run_id = str(kwargs["run_id"])
        calls.append((chapters, run_id))
        run_dir = tmp_path / "runs" / run_id
        run_dir.mkdir(parents=True)
        manifest = BatchManifest.create(
            run_id=run_id,
            story_slug=story_yaml.parent.name,
            title=story_yaml.parent.name,
            story_yaml=story_yaml,
            chapters=chapters,
            mode=kwargs["provider_mode"],
            providers={},
            run_dir=run_dir,
        )
        for chapter_run in manifest.chapters.values():
            chapter_run.status = "packaged"
        manifest.refresh_summary()
        manifest_path = run_dir / "batch_manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return BatchPipelineResult(run_dir=run_dir, manifest_path=manifest_path, manifest=manifest)

    monkeypatch.setattr("agentic_translation.cli.run_batch_pipeline", fake_run_batch_pipeline)
    monkeypatch.setattr("agentic_translation.cli.write_batch_triage_artifacts", lambda run_dir: {})

    result = CliRunner().invoke(
        app,
        [
            "smoke-project",
            str(project_dir),
            "--continue-latest",
            "--first",
            "2",
            "--chunks",
            "2",
            "--practical",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        (["0003", "0004"], "simulator_alliance_practical_0003_0004"),
        (["0005", "0006"], "simulator_alliance_practical_0005_0006"),
    ]
    assert "Chunk 1/2" in result.output
    assert "Chunk 2/2" in result.output


def test_smoke_project_can_continue_until_target_chapter(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    project_dir = tmp_path / "simulator_alliance"
    source_dir = project_dir / "scraped"
    translated_dir = project_dir / "translated_001_006"
    terms_dir = project_dir / "terms"
    source_dir.mkdir(parents=True)
    translated_dir.mkdir()
    terms_dir.mkdir()
    for chapter in [f"{number:04d}" for number in range(1, 7)]:
        (source_dir / f"{chapter}.txt").write_text(f"第{chapter}章\n\n天道", encoding="utf-8")
        (translated_dir / f"{chapter}.txt").write_text(f"Chapter {int(chapter)}\n\nHeavenly Dao", encoding="utf-8")
    (terms_dir / "master_glossary.txt").write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    previous_run_dir = tmp_path / "runs" / "previous"
    previous_run_dir.mkdir(parents=True)
    previous_manifest = BatchManifest.create(
        run_id="previous",
        story_slug="simulator_alliance_smoke_0001_0002",
        title="simulator_alliance_smoke_0001_0002",
        story_yaml=tmp_path / "previous.yaml",
        chapters=["0001", "0002"],
        mode="offline",
        providers={},
        run_dir=previous_run_dir,
    )
    (previous_run_dir / "batch_manifest.json").write_text(previous_manifest.model_dump_json(indent=2), encoding="utf-8")
    calls: list[tuple[list[str], str]] = []

    def fake_run_batch_pipeline(story_yaml: Path, **kwargs):  # noqa: ANN003
        chapters = list(kwargs["chapters"])
        run_id = str(kwargs["run_id"])
        calls.append((chapters, run_id))
        run_dir = tmp_path / "runs" / run_id
        run_dir.mkdir(parents=True)
        manifest = BatchManifest.create(
            run_id=run_id,
            story_slug=story_yaml.parent.name,
            title=story_yaml.parent.name,
            story_yaml=story_yaml,
            chapters=chapters,
            mode=kwargs["provider_mode"],
            providers={},
            run_dir=run_dir,
        )
        for chapter_run in manifest.chapters.values():
            chapter_run.status = "packaged"
        manifest.refresh_summary()
        manifest_path = run_dir / "batch_manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return BatchPipelineResult(run_dir=run_dir, manifest_path=manifest_path, manifest=manifest)

    monkeypatch.setattr("agentic_translation.cli.run_batch_pipeline", fake_run_batch_pipeline)
    monkeypatch.setattr("agentic_translation.cli.write_batch_triage_artifacts", lambda run_dir: {})

    result = CliRunner().invoke(
        app,
        [
            "smoke-project",
            str(project_dir),
            "--continue-latest",
            "--first",
            "2",
            "--until",
            "0005",
            "--practical",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        (["0003", "0004"], "simulator_alliance_practical_0003_0004"),
        (["0005"], "simulator_alliance_practical_0005"),
    ]
    assert "Until: 0005" in result.output


def test_produce_dry_run_selects_next_chapters_without_mutation(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    project_dir = tmp_path / "simulator_alliance"
    source_dir = project_dir / "scraped"
    translated_dir = project_dir / "translated_001_004"
    terms_dir = project_dir / "terms"
    source_dir.mkdir(parents=True)
    translated_dir.mkdir()
    terms_dir.mkdir()
    for chapter in ["0001", "0002", "0003", "0004"]:
        (source_dir / f"{chapter}.txt").write_text(f"第{chapter}章\n\n天道", encoding="utf-8")
        (translated_dir / f"{chapter}.txt").write_text(f"Chapter {int(chapter)}\n\nHeavenly Dao", encoding="utf-8")
    (terms_dir / "master_glossary.txt").write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    previous_run_dir = tmp_path / "runs" / "previous"
    previous_run_dir.mkdir(parents=True)
    previous_manifest = BatchManifest.create(
        run_id="previous",
        story_slug="simulator_alliance_smoke_0001_0002",
        title="simulator_alliance_smoke_0001_0002",
        story_yaml=tmp_path / "previous.yaml",
        chapters=["0001", "0002"],
        mode="offline",
        providers={},
        run_dir=previous_run_dir,
    )
    (previous_run_dir / "batch_manifest.json").write_text(previous_manifest.model_dump_json(indent=2), encoding="utf-8")
    monkeypatch.setattr("agentic_translation.cli.smoke_project", lambda **kwargs: (_ for _ in ()).throw(AssertionError("dry-run should not run")))

    result = CliRunner().invoke(
        app,
        [
            "produce",
            str(project_dir),
            "--count",
            "2",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["project"] == "simulator_alliance"
    assert payload["provider"] == "offline"
    assert payload["would_mutate"] is False
    assert payload["project_status"]["total_source_chapters"] == 4
    assert payload["project_status"]["processed_unique_chapters"] == 2
    assert payload["project_status"]["latest_chapter"] == "0002"
    assert payload["project_status"]["next_chapter"] == "0003"
    assert payload["next_chunk"] == ["0003", "0004"]
    assert payload["next_command"] == f"agentic-translation produce {project_dir} --chapters 0003,0004 --overwrite"
    assert payload["chunks"] == [
        {
            "chapters": ["0003", "0004"],
            "run_id": "simulator_alliance_practical_0003_0004",
            "translated_dir": str(translated_dir),
            "follow_up_command": f"agentic-translation produce {project_dir} --chapters 0003,0004 --overwrite",
        }
    ]
    assert payload["follow_up_commands"] == [
        f"agentic-translation produce {project_dir} --chapters 0003,0004 --overwrite"
    ]
    assert payload["translated_dir"] == str(translated_dir)
    assert not (tmp_path / "local_fixtures").exists()


def test_produce_dry_run_prints_follow_up_command(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    project_dir = tmp_path / "simulator_alliance"
    source_dir = project_dir / "scraped"
    translated_dir = project_dir / "translated_001_002"
    terms_dir = project_dir / "terms"
    source_dir.mkdir(parents=True)
    translated_dir.mkdir()
    terms_dir.mkdir()
    for chapter in ["0001", "0002"]:
        (source_dir / f"{chapter}.txt").write_text(f"第{chapter}章\n\n天道", encoding="utf-8")
        (translated_dir / f"{chapter}.txt").write_text(f"Chapter {int(chapter)}\n\nHeavenly Dao", encoding="utf-8")
    (terms_dir / "master_glossary.txt").write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    monkeypatch.setattr("agentic_translation.cli.smoke_project", lambda **kwargs: (_ for _ in ()).throw(AssertionError("dry-run should not run")))

    result = CliRunner().invoke(
        app,
        [
            "produce",
            str(project_dir),
            "--count",
            "2",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "Next command" in result.output
    assert "agentic-translation produce" in result.output
    assert str(project_dir) in result.output
    assert "--chapters 0001,0002" in result.output
    assert "--overwrite" in result.output
    assert not (tmp_path / "local_fixtures").exists()


def test_produce_dry_run_prints_project_status(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    project_dir = tmp_path / "simulator_alliance"
    source_dir = project_dir / "scraped"
    translated_dir = project_dir / "translated_001_004"
    terms_dir = project_dir / "terms"
    source_dir.mkdir(parents=True)
    translated_dir.mkdir()
    terms_dir.mkdir()
    for chapter in ["0001", "0002", "0003", "0004"]:
        (source_dir / f"{chapter}.txt").write_text(f"第{chapter}章\n\n天道", encoding="utf-8")
        (translated_dir / f"{chapter}.txt").write_text(f"Chapter {int(chapter)}\n\nHeavenly Dao", encoding="utf-8")
    (terms_dir / "master_glossary.txt").write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    previous_run_dir = tmp_path / "runs" / "previous"
    previous_run_dir.mkdir(parents=True)
    previous_manifest = BatchManifest.create(
        run_id="previous",
        story_slug="simulator_alliance_smoke_0001_0002",
        title="simulator_alliance_smoke_0001_0002",
        story_yaml=tmp_path / "previous.yaml",
        chapters=["0001", "0002"],
        mode="offline",
        providers={},
        run_dir=previous_run_dir,
    )
    (previous_run_dir / "batch_manifest.json").write_text(previous_manifest.model_dump_json(indent=2), encoding="utf-8")
    monkeypatch.setattr("agentic_translation.cli.smoke_project", lambda **kwargs: (_ for _ in ()).throw(AssertionError("dry-run should not run")))

    result = CliRunner().invoke(
        app,
        [
            "produce",
            str(project_dir),
            "--count",
            "2",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "Source Chapters" in result.output
    assert "Processed Chapters" in result.output
    assert "Latest Chapter" in result.output
    assert "Next Chapter" in result.output
    assert "Run IDs" in result.output
    assert "Source Dir" in result.output
    assert "Glossary" in result.output
    assert "Translated Dir" in result.output
    assert "4" in result.output
    assert "2" in result.output
    assert "0002" in result.output
    assert "0003" in result.output
    assert "simulator_alliance_practical_0003_0004" in result.output
    assert f"Source dir(s): {source_dir}" in result.output
    assert f"Glossary: {terms_dir / 'master_glossary.txt'}" in result.output
    assert str(translated_dir) in result.output
    assert not (tmp_path / "local_fixtures").exists()


def test_produce_json_execution_records_every_chunk(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    project_dir = tmp_path / "simulator_alliance"
    source_dir = project_dir / "scraped"
    translated_dir = project_dir / "translated_001_004"
    terms_dir = project_dir / "terms"
    source_dir.mkdir(parents=True)
    translated_dir.mkdir()
    terms_dir.mkdir()
    for chapter in ["0001", "0002", "0003", "0004"]:
        (source_dir / f"{chapter}.txt").write_text(f"第{chapter}章\n\n天道", encoding="utf-8")
        (translated_dir / f"{chapter}.txt").write_text(f"Chapter {int(chapter)}\n\nHeavenly Dao", encoding="utf-8")
    (terms_dir / "master_glossary.txt").write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    calls: list[str] = []

    def fake_smoke_project(**kwargs):  # noqa: ANN003
        chapters = str(kwargs["chapters"]).split(",")
        calls.append(str(kwargs["chapters"]))
        run_id = str(kwargs["run_id"])
        run_dir = Path("runs") / run_id
        run_dir.mkdir(parents=True)
        manifest = BatchManifest.create(
            run_id=run_id,
            story_slug=f"simulator_alliance_smoke_{chapters[0]}_{chapters[-1]}",
            title=f"simulator_alliance_smoke_{chapters[0]}_{chapters[-1]}",
            story_yaml=Path("story.yaml"),
            chapters=chapters,
            mode="offline",
            providers={},
            run_dir=run_dir,
        )
        for chapter in chapters:
            manifest.chapters[chapter].status = "packaged"
        manifest.refresh_summary()
        (run_dir / "batch_manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    monkeypatch.setattr("agentic_translation.cli.smoke_project", fake_smoke_project)

    result = CliRunner().invoke(
        app,
        [
            "produce",
            str(project_dir),
            "--count",
            "2",
            "--until",
            "0004",
            "--overwrite",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert calls == ["0001,0002", "0003,0004"]
    assert payload["run_ids"] == [
        "simulator_alliance_practical_0001_0002",
        "simulator_alliance_practical_0003_0004",
    ]
    assert payload["status_summary"]["total_chapters"] == 4
    assert payload["status_summary"]["packaged"] == 4


def test_produce_offline_delegates_to_practical_smoke_project(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    project_dir = tmp_path / "simulator_alliance"
    source_dir = project_dir / "scraped"
    translated_dir = project_dir / "translated_001_002"
    terms_dir = project_dir / "terms"
    source_dir.mkdir(parents=True)
    translated_dir.mkdir()
    terms_dir.mkdir()
    for chapter in ["0001", "0002"]:
        (source_dir / f"{chapter}.txt").write_text(f"第{chapter}章\n\n天道", encoding="utf-8")
        (translated_dir / f"{chapter}.txt").write_text(f"Chapter {int(chapter)}\n\nHeavenly Dao", encoding="utf-8")
    (terms_dir / "master_glossary.txt").write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_smoke_project(**kwargs):  # noqa: ANN003
        calls.append(kwargs)
        run_id = str(kwargs["run_id"])
        run_dir = Path("runs") / run_id
        run_dir.mkdir(parents=True)
        manifest = BatchManifest.create(
            run_id=run_id,
            story_slug="simulator_alliance_smoke_0001_0002",
            title="simulator_alliance_smoke_0001_0002",
            story_yaml=Path("story.yaml"),
            chapters=["0001", "0002"],
            mode="offline",
            providers={},
            run_dir=run_dir,
        )
        manifest.chapters["0001"].status = "packaged"
        manifest.chapters["0002"].status = "review_required"
        review_dir = run_dir / "review"
        review_dir.mkdir()
        txt_path = review_dir / "simulator_alliance_smoke_0001_0002.txt"
        epub_path = review_dir / "simulator_alliance_smoke_0001_0002.epub"
        txt_path.write_text("Chapter 1\n\nChapter 2\n", encoding="utf-8")
        epub_path.write_text("fake epub", encoding="utf-8")
        manifest.artifacts["txt"] = str(txt_path.relative_to(run_dir))
        manifest.artifacts["epub"] = str(epub_path.relative_to(run_dir))
        manifest.refresh_summary()
        (run_dir / "batch_manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    monkeypatch.setattr("agentic_translation.cli.smoke_project", fake_smoke_project)

    result = CliRunner().invoke(
        app,
        [
            "produce",
            str(project_dir),
            "--chapters",
            "0001,0002",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    call = calls[0]
    assert call["project_dir"] == project_dir
    assert call["chapters"] == "0001,0002"
    assert call["translated_dir"] == translated_dir
    assert call["provider_mode"] == "offline"
    assert call["deepseek"] is False
    assert call["practical"] is True
    assert call["run_id"] == "simulator_alliance_practical_0001_0002"
    assert call["write_proof"] is False
    assert call["write_triage"] is True
    assert call["overwrite"] is True
    assert "Produce chunk 1/1" in result.output
    assert "Produced run(s): simulator_alliance_practical_0001_0002" in result.output
    assert "Review run(s): runs/simulator_alliance_practical_0001_0002" in result.output
    assert "Status: 1/2 packaged, 1 review_required, 0 failed, 0 incomplete" in result.output
    assert "TXT artifact(s): runs/simulator_alliance_practical_0001_0002/review/simulator_alliance_smoke_0001_0002.txt" in result.output
    assert "EPUB artifact(s): runs/simulator_alliance_practical_0001_0002/review/simulator_alliance_smoke_0001_0002.epub" in result.output
    assert "Project progress: 2/2 processed, latest 0002, next -" in result.output
    assert "Next action(s):" in result.output
    assert "agentic-translation batch review runs/simulator_alliance_practical_0001_0002 --write --write-markdown" in result.output


def test_produce_json_execution_emits_receipt_without_progress_noise(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    project_dir = tmp_path / "simulator_alliance"
    source_dir = project_dir / "scraped"
    translated_dir = project_dir / "translated_001_002"
    terms_dir = project_dir / "terms"
    source_dir.mkdir(parents=True)
    translated_dir.mkdir()
    terms_dir.mkdir()
    for chapter in ["0001", "0002"]:
        (source_dir / f"{chapter}.txt").write_text(f"第{chapter}章\n\n天道", encoding="utf-8")
        (translated_dir / f"{chapter}.txt").write_text(f"Chapter {int(chapter)}\n\nHeavenly Dao", encoding="utf-8")
    (terms_dir / "master_glossary.txt").write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_smoke_project(**kwargs):  # noqa: ANN003
        print("internal smoke output")
        calls.append(kwargs)
        run_id = str(kwargs["run_id"])
        run_dir = Path("runs") / run_id
        run_dir.mkdir(parents=True)
        manifest = BatchManifest.create(
            run_id=run_id,
            story_slug="simulator_alliance_smoke_0001_0002",
            title="simulator_alliance_smoke_0001_0002",
            story_yaml=Path("story.yaml"),
            chapters=["0001", "0002"],
            mode="offline",
            providers={},
            run_dir=run_dir,
        )
        manifest.chapters["0001"].status = "packaged"
        manifest.chapters["0002"].status = "review_required"
        review_dir = run_dir / "review"
        review_dir.mkdir()
        txt_path = review_dir / "simulator_alliance_smoke_0001_0002.txt"
        epub_path = review_dir / "simulator_alliance_smoke_0001_0002.epub"
        txt_path.write_text("Chapter 1\n\nChapter 2\n", encoding="utf-8")
        epub_path.write_text("fake epub", encoding="utf-8")
        manifest.artifacts["txt"] = str(txt_path.relative_to(run_dir))
        manifest.artifacts["epub"] = str(epub_path.relative_to(run_dir))
        manifest.refresh_summary()
        (run_dir / "batch_manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    monkeypatch.setattr("agentic_translation.cli.smoke_project", fake_smoke_project)

    result = CliRunner().invoke(
        app,
        [
            "produce",
            str(project_dir),
            "--chapters",
            "0001,0002",
            "--overwrite",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["project"] == "simulator_alliance"
    assert payload["would_mutate"] is True
    assert payload["run_ids"] == ["simulator_alliance_practical_0001_0002"]
    assert payload["review_runs"] == ["runs/simulator_alliance_practical_0001_0002"]
    assert payload["project_status"]["processed_unique_chapters"] == 0
    assert payload["project_status_after"]["processed_unique_chapters"] == 2
    assert payload["project_status_after"]["latest_chapter"] == "0002"
    assert payload["project_status_after"]["next_chapter"] is None
    assert payload["project_status_after"]["status_counts"] == {
        "packaged": 1,
        "review_required": 1,
    }
    assert payload["status_summary"] == {
        "total_chapters": 2,
        "pending": 0,
        "packaged": 1,
        "review_required": 1,
        "failed": 0,
        "skipped": 0,
        "incomplete": 0,
    }
    assert payload["artifacts"] == {
        "txt": ["runs/simulator_alliance_practical_0001_0002/review/simulator_alliance_smoke_0001_0002.txt"],
        "epub": ["runs/simulator_alliance_practical_0001_0002/review/simulator_alliance_smoke_0001_0002.epub"],
    }
    assert payload["run_summaries"] == [
        {
            "run_id": "simulator_alliance_practical_0001_0002",
            "run_dir": "runs/simulator_alliance_practical_0001_0002",
            "summary": payload["status_summary"],
            "artifacts": {
                "txt": "runs/simulator_alliance_practical_0001_0002/review/simulator_alliance_smoke_0001_0002.txt",
                "epub": "runs/simulator_alliance_practical_0001_0002/review/simulator_alliance_smoke_0001_0002.epub",
            },
        }
    ]
    assert payload["recommended_next_actions"] == [
        {
            "action": "review",
            "run_id": "simulator_alliance_practical_0001_0002",
            "run_dir": "runs/simulator_alliance_practical_0001_0002",
            "reason": "1 chapter(s) require review.",
            "command": "agentic-translation batch review runs/simulator_alliance_practical_0001_0002 --write --write-markdown",
        }
    ]
    assert payload["chunks"] == [
        {
            "chapters": ["0001", "0002"],
            "run_id": "simulator_alliance_practical_0001_0002",
            "translated_dir": str(translated_dir),
            "follow_up_command": f"agentic-translation produce {project_dir} --chapters 0001,0002 --overwrite",
        }
    ]
    assert calls[0]["run_id"] == "simulator_alliance_practical_0001_0002"
    assert "internal smoke output" not in result.output
    assert "Produce chunk" not in result.output


def test_produce_deepseek_uses_probe_and_cheap_defaults(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    project_dir = tmp_path / "simulator_alliance"
    source_dir = project_dir / "scraped"
    translated_dir = project_dir / "translated_001_001"
    terms_dir = project_dir / "terms"
    source_dir.mkdir(parents=True)
    translated_dir.mkdir()
    terms_dir.mkdir()
    (source_dir / "0001.txt").write_text("第1章\n\n天道", encoding="utf-8")
    (translated_dir / "0001.txt").write_text("Chapter 1\n\nHeavenly Dao", encoding="utf-8")
    (terms_dir / "master_glossary.txt").write_text("天道 -> Heavenly Dao\n", encoding="utf-8")
    probe_calls: list[dict[str, object]] = []
    smoke_calls: list[dict[str, object]] = []

    def fake_probe_live_provider(**kwargs):  # noqa: ANN003
        probe_calls.append(kwargs)
        return LiveProviderProbeResult(
            provider="deepseek",
            mode="live",
            model="deepseek-chat",
            cache_dir=str(kwargs["cache_dir"]),
            cache_hit=False,
            cache_file="probe.json",
            response={"ok": True},
        )

    def fake_smoke_project(**kwargs):  # noqa: ANN003
        smoke_calls.append(kwargs)

    monkeypatch.setattr("agentic_translation.cli.probe_live_provider", fake_probe_live_provider)
    monkeypatch.setattr("agentic_translation.cli.smoke_project", fake_smoke_project)

    result = CliRunner().invoke(
        app,
        [
            "produce",
            str(project_dir),
            "--chapters",
            "0001",
            "--provider",
            "deepseek",
            "--cheap",
            "500",
        ],
    )

    assert result.exit_code == 0
    assert probe_calls[0]["provider_name"] == "deepseek"
    assert probe_calls[0]["model_name"] == "deepseek-chat"
    assert probe_calls[0]["cache_dir"] == Path(".agentic_cache/produce_deepseek")
    assert smoke_calls[0]["deepseek"] is True
    assert smoke_calls[0]["cache_dir"] == Path(".agentic_cache/produce_deepseek")
    assert smoke_calls[0]["source_char_limit"] == 500
    assert smoke_calls[0]["allow_live_provider_fallback"] is True
    assert "DeepSeek probe ok" in result.output


def test_project_status_summarizes_matching_smoke_runs(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    project_dir = tmp_path / "simulator_alliance"
    source_dir = project_dir / "scraped"
    terms_dir = project_dir / "terms"
    source_dir.mkdir(parents=True)
    terms_dir.mkdir()
    for chapter in [f"{number:04d}" for number in range(1, 7)]:
        (source_dir / f"{chapter}.txt").write_text(f"第{chapter}章\n\n天道", encoding="utf-8")
    (terms_dir / "master_glossary.txt").write_text("天道 -> Heavenly Dao\n", encoding="utf-8")

    def write_run(run_id: str, story_slug: str, statuses: dict[str, str]) -> None:
        run_dir = tmp_path / "runs" / run_id
        run_dir.mkdir(parents=True)
        manifest = BatchManifest.create(
            run_id=run_id,
            story_slug=story_slug,
            title=story_slug,
            story_yaml=tmp_path / f"{run_id}.yaml",
            chapters=list(statuses),
            mode="offline",
            providers={},
            run_dir=run_dir,
        )
        for chapter, status in statuses.items():
            manifest.chapters[chapter].status = status  # type: ignore[assignment]
        manifest.refresh_summary()
        (run_dir / "batch_manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    write_run("old", "simulator_alliance_smoke_0001_0002", {"0001": "packaged", "0002": "review_required"})
    write_run("new", "simulator_alliance_smoke_0003_0004", {"0003": "packaged", "0004": "packaged"})
    write_run("irrelevant", "other_project_smoke_0005", {"0005": "failed"})

    result = CliRunner().invoke(app, ["project-status", str(project_dir), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["project"] == "simulator_alliance"
    assert payload["run_count"] == 2
    assert payload["processed_unique_chapters"] == 4
    assert payload["total_source_chapters"] == 6
    assert payload["latest_chapter"] == "0004"
    assert payload["next_chapter"] == "0005"
    assert payload["status_counts"] == {"packaged": 3, "review_required": 1}
    assert payload["recent_runs"][-1]["run_id"] == "new"


def test_smoke_local_deepseek_falls_back_when_key_missing(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)
    source_dir = tmp_path / "simulator" / "scraped"
    translated_dir = tmp_path / "simulator" / "translated_001_001"
    terms_dir = tmp_path / "simulator" / "terms"
    source_dir.mkdir(parents=True)
    translated_dir.mkdir()
    terms_dir.mkdir()
    (source_dir / "0001.txt").write_text("第1章\n\n天道", encoding="utf-8")
    (translated_dir / "0001.txt").write_text("Chapter 1\n\nHeavenly Dao", encoding="utf-8")
    glossary = terms_dir / "master_glossary.txt"
    glossary.write_text("天道 -> Heavenly Dao\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "smoke-local",
            "--source-dir",
            str(source_dir),
            "--glossary",
            str(glossary),
            "--chapters",
            "0001",
            "--translated-dir",
            str(translated_dir),
            "--out",
            str(tmp_path / "fixture"),
            "--deepseek",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--run-id",
            "missing_key_fallback",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0
    status = json.loads((tmp_path / "fixture" / "runs" / "missing_key_fallback" / "batch_status.json").read_text(encoding="utf-8"))
    assert status["run_config"]["provider_mode"] == "live"
    assert status["run_config"]["translation_provider"] == "deepseek"
    assert status["provider_failures"]
    assert status["provider_failures"][0]["role"] == "translation"
    assert status["provider_failures"][0]["fallback_used"] is True
    assert "DEEPSEEK_API_KEY is required" in status["provider_failures"][0]["reason"]
