from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORY = PROJECT_ROOT / "samples/agentic_terminology_demo/story.yaml"


def test_terminology_consensus_replay_demo_from_outside_project(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentic_translation",
            "demo-repair",
            "--story",
            str(STORY),
            "--chapter",
            "0001",
            "--provider-mode",
            "replay",
            "--term-consensus",
            "--openai-term-model",
            "fixture-openai-term",
            "--deepseek-term-model",
            "fixture-deepseek-term",
            "--term-evaluator",
            "openai",
            "--runs-dir",
            str(runs_dir),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "resolve_terminology" in result.stdout
    assert "Final QA findings: 0" in result.stdout
    assert "Terminology consensus" in result.stdout

    run_dir = runs_dir / "agentic_terminology_demo_replay"
    episode = json.loads((run_dir / "agent_episode.json").read_text(encoding="utf-8"))
    assert [step["action"]["tool"] for step in episode["steps"]] == [
        "resolve_terminology",
        "submit_patch",
        "finish",
    ]
    calls = []
    for step in episode["steps"]:
        if step.get("provider_call"):
            calls.append(step["provider_call"])
        calls.extend(step.get("auxiliary_provider_calls", []))
    assert [call["namespace"] for call in calls] == [
        "agent_action",
        "terminology_vote",
        "terminology_vote",
        "terminology_evaluate",
        "agent_action",
        "agent_action",
    ]
    assert all(call["cache_hit"] is True for call in calls)
    resolution = episode["terminology_resolutions"][0]
    assert resolution["evaluator_used"] is True
    assert resolution["selected_translation"] == "Dao Heart"
    assert resolution["escalated"] is False
    assert episode["final_qa"]["summary"]["total_findings"] == 0

    final_text = (run_dir / "translated_final/0001.txt").read_text(encoding="utf-8")
    assert "Dao Heart" in final_text
    assert "Heart of Dao" not in final_text
    assert "道心" not in final_text

    report = (run_dir / "repair_report.md").read_text(encoding="utf-8")
    assert "Dao Heart" in report
    assert "Synthetic replay fixture; not evidence of a funded live-provider run." in report
    assert "evaluator: yes" in report
    assert "terminology\\_vote" in report
    assert "openai/fixture\\-openai\\-term" in report
    assert "deepseek/fixture\\-deepseek\\-term" in report
    assert "terminology\\_evaluate" in report

    master_glossary = (
        STORY.parent / "terms/master_glossary.txt"
    ).read_text(encoding="utf-8")
    assert "道心" not in master_glossary
