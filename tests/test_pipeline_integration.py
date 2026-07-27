from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_translation.agent_models import EscalateAction, FinishAction, SubmitPatchAction
from agentic_translation.models import ProviderCallRecord
from agentic_translation.pipeline import run_demo_pipeline


SOURCE_TEXT = "第一章\n\n道心守住了山门。"
DIRTY_TEXT = "Chapter 1\n\nHeart of Dao guarded 道心."
CLEAN_TEXT = "Chapter 1\n\nDao Heart guarded the mountain gate."


class DeterministicAgentProvider:
    provider_name = "fixture-agent"
    model_name = "fixture-agent-v1"

    def __init__(self, actions) -> None:
        self.actions = list(actions)
        self.call_records: list[ProviderCallRecord] = []
        self.calls = 0

    def next_action(self, request):
        self.calls += 1
        call_number = len(self.call_records) + 1
        self.call_records.append(
            ProviderCallRecord(
                role="agent",
                namespace="agent_action",
                provider=self.provider_name,
                model=self.model_name,
                payload_sha256=f"{call_number:064d}",
                response_sha256=f"{call_number + 1:064d}",
                cache_file=f"agent_action_{call_number}.json",
                cache_hit=True,
            )
        )
        return self.actions.pop(0)


class ExplodingAgentProvider:
    provider_name = "fixture-agent"
    model_name = "fixture-agent-v1"

    def __init__(self) -> None:
        self.calls = 0
        self.call_records: list[ProviderCallRecord] = []

    def next_action(self, request):
        self.calls += 1
        raise AssertionError("clean fixed QA must skip the tool agent")


def _story(tmp_path: Path, *, clean: bool = False) -> Path:
    root = tmp_path / ("clean_story" if clean else "dirty_story")
    source_dir = root / "source"
    expected_dir = root / "expected"
    terms_dir = root / "terms"
    source_dir.mkdir(parents=True)
    expected_dir.mkdir()
    terms_dir.mkdir()
    (source_dir / "0001.txt").write_text(SOURCE_TEXT, encoding="utf-8")
    (expected_dir / "dirty_translation.txt").write_text(CLEAN_TEXT if clean else DIRTY_TEXT, encoding="utf-8")
    (terms_dir / "master_glossary.txt").write_text("", encoding="utf-8")
    story_yaml = root / "story.yaml"
    story_yaml.write_text(
        f"""
slug: pipeline_integration
title: Pipeline Integration
language: zh
chapter_ids: ["0001"]
paths:
  source_dir: "{source_dir}"
  glossary_path: "{terms_dir / 'master_glossary.txt'}"
  expected_dir: "{expected_dir}"
  runs_dir: "{tmp_path / 'runs'}"
qa:
  max_repairs: 0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return story_yaml


def _accepting_provider() -> DeterministicAgentProvider:
    return DeterministicAgentProvider(
        [
            SubmitPatchAction(
                old_text="missing target",
                new_text="never applied",
                rationale="exercise rejected patch evidence",
            ),
            SubmitPatchAction(
                old_text="Heart of Dao guarded 道心.",
                new_text="Dao Heart guarded the mountain gate.",
                rationale="remove the remaining residue",
            ),
            FinishAction(summary="deterministic QA is clean"),
        ]
    )


def test_pipeline_runs_tool_agent_before_packaging_and_records_post_agent_state(tmp_path: Path) -> None:
    provider = _accepting_provider()
    result = run_demo_pipeline(
        _story(tmp_path),
        provider_mode="replay",
        repair_provider_name="openai",
        model_name="fixture-agent-v1",
        tool_agent_enabled=True,
        tool_agent_provider=provider,
        runs_dir=tmp_path / "runs",
        skip_epub=True,
        run_id="accepted",
    )

    assert result.tool_agent is not None
    assert result.tool_agent.final_status == "verified"
    assert result.qa_final.summary.total_findings == 0
    assert result.final_metrics.total_findings == 0
    assert result.tool_agent.provider_calls == provider.call_records
    assert result.provider_calls[-len(provider.call_records) :] == provider.call_records

    final_text = (result.run_dir / "translated_final/0001.txt").read_text(encoding="utf-8")
    packaged_text = (result.run_dir / "review/pipeline_integration_0001.txt").read_text(encoding="utf-8")
    assert CLEAN_TEXT in final_text
    assert CLEAN_TEXT in packaged_text

    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["providers"]["agent"] == {
        "provider": provider.provider_name,
        "model": provider.model_name,
    }
    assert manifest["artifacts"]["agent_episode"] == "agent_repair/agent_episode.json"
    assert manifest["artifacts"]["agent_report"] == "agent_repair/report.md"
    assert manifest["artifacts"]["agent_report_html"] == "agent_repair/report.html"
    assert manifest["qa"]["final_findings"] == 0
    assert manifest["eval_metrics"][-1]["total_findings"] == 0

    trace = [
        json.loads(line)
        for line in (result.run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stages = [record["stage"] for record in trace]
    assert stages.index("tool_agent") < stages.index("qa_final")
    assert stages.index("tool_agent") < stages.index("package_txt")
    assert (result.run_dir / "report.html").read_text(encoding="utf-8").find(CLEAN_TEXT) >= 0


def test_pipeline_skips_tool_agent_when_fixed_qa_is_clean(tmp_path: Path) -> None:
    provider = ExplodingAgentProvider()
    result = run_demo_pipeline(
        _story(tmp_path, clean=True),
        provider_mode="replay",
        repair_provider_name="openai",
        model_name="fixture-agent-v1",
        tool_agent_enabled=True,
        tool_agent_provider=provider,
        runs_dir=tmp_path / "runs",
        skip_epub=True,
        run_id="clean",
    )

    assert result.tool_agent is None
    assert provider.calls == 0
    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "agent" not in manifest["providers"]
    assert "agent_episode" not in manifest["artifacts"]
    assert all(call.namespace != "agent_action" for call in result.provider_calls)


def test_pipeline_returns_escalated_episode_with_failed_artifact_qa_for_review(tmp_path: Path) -> None:
    provider = DeterministicAgentProvider([EscalateAction(reason="Needs human review.")])
    result = run_demo_pipeline(
        _story(tmp_path),
        provider_mode="replay",
        repair_provider_name="openai",
        model_name="fixture-agent-v1",
        tool_agent_enabled=True,
        tool_agent_provider=provider,
        runs_dir=tmp_path / "runs",
        skip_epub=True,
        run_id="escalated",
    )

    assert result.tool_agent is not None
    assert result.tool_agent.final_status == "escalated"
    assert result.qa_final.summary.total_findings > 0
    assert result.artifact_qa is not None
    assert result.artifact_qa.passed is False
    assert result.artifact_qa.failures == ["TXT contains Chinese residue."]
    assert result.tool_agent.episode_path.exists()
    assert result.tool_agent.report_path.exists()
    assert result.tool_agent.html_path.exists()
    assert (result.run_dir / "translated_final/0001.txt").read_text(encoding="utf-8") == DIRTY_TEXT + "\n"
    assert (result.run_dir / "qa_final.json").exists()
    assert (result.run_dir / "report.html").exists()


def test_pipeline_still_raises_artifact_qa_failure_without_tool_agent(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Artifact QA failed"):
        run_demo_pipeline(
            _story(tmp_path),
            provider_mode="offline",
            tool_agent_enabled=False,
            runs_dir=tmp_path / "runs",
            skip_epub=True,
            run_id="no_agent",
        )
