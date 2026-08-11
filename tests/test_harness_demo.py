from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "agentic_translation", *args],
        cwd=cwd or PROJECT_ROOT,
        env={"PYTHONPATH": str(PROJECT_ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )


def _artifacts(run_dir: Path) -> dict[str, object]:
    return {
        "events": json.loads("[" + ",".join(
            line for line in (run_dir / "session_events.jsonl").read_text(encoding="utf-8").splitlines() if line
        ) + "]"),
        "snapshot": json.loads((run_dir / "session_snapshot.json").read_text(encoding="utf-8")),
        "episode": json.loads((run_dir / "agent_episode.json").read_text(encoding="utf-8")),
    }


def _assert_neutral_public_artifacts(run_dir: Path, snapshot: dict[str, object]) -> None:
    forbidden_terms = ("recruit" + "er", "port" + "folio")
    payloads = [
        str(run_dir).lower(),
        json.dumps(snapshot).lower(),
        (run_dir / "report.md").read_text(encoding="utf-8").lower(),
        (run_dir / "report.html").read_text(encoding="utf-8").lower(),
    ]
    assert all(
        not any(term in payload for term in forbidden_terms)
        for payload in payloads
    )


def test_golden_pause_writes_review_artifacts_and_preserves_master_glossary(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    result = _run(
        "harness",
        "golden",
        "--runs-dir",
        str(runs_dir),
        "--pause-for-approval",
        "--overwrite",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    run_dir = runs_dir / "agentic_harness_v3_demo"
    for name in (
        "session_events.jsonl",
        "session_snapshot.json",
        "agent_episode.json",
        "glossary.txt",
        "report.html",
        "report.md",
    ):
        assert (run_dir / name).exists(), name
    snapshot = json.loads((run_dir / "session_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["status"] == "awaiting_approval"
    assert snapshot["pending_approval"]["tool"] == "promote_glossary_term"
    _assert_neutral_public_artifacts(run_dir, snapshot)
    report = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "Dao Heart" in report
    assert "synthetic" in report.lower()
    assert "Synthetic contract fixture. No API key or network call is used." in report
    assert "native function" in report.lower()
    assert "PENDING" in report
    assert "Proposed glossary delta" in report
    assert "Applied glossary delta" not in report
    events = [
        json.loads(line)
        for line in (run_dir / "session_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    model_requests = [event for event in events if event["event_type"] == "model_requested"]
    assert model_requests
    assert all(event["payload"]["tool_protocol"] == "native_function" for event in model_requests)
    master = PROJECT_ROOT / "samples/agentic_harness_v3_demo/terms/master_glossary.txt"
    assert master.read_text(encoding="utf-8") == "道心 -> Heart of Dao\n"
    assert (run_dir / "glossary.txt").read_text(encoding="utf-8") == master.read_text(encoding="utf-8")


def test_golden_approve_applies_run_glossary_once_and_finishes_verified(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    paused = _run(
        "harness",
        "golden",
        "--runs-dir",
        str(runs_dir),
        "--pause-for-approval",
    )
    assert paused.returncode == 0, paused.stdout + paused.stderr
    run_dir = runs_dir / "agentic_harness_v3_demo"
    resumed = _run(
        "harness",
        "resume",
        str(run_dir),
        "--approve",
        "--reviewer",
        "demo-reviewer",
        "--note",
        "Approved in the Harness v3 review flow.",
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    snapshot = json.loads((run_dir / "session_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["status"] == "completed"
    assert snapshot["episode"]["final_status"] == "verified"
    _assert_neutral_public_artifacts(run_dir, snapshot)
    assert snapshot["episode"]["final_qa"]["summary"]["total_findings"] == 0
    final_text = (run_dir / "translated_final.txt").read_text(encoding="utf-8")
    assert final_text.strip().endswith("Dao Heart guarded the mountain gate.")
    glossary = (run_dir / "glossary.txt").read_text(encoding="utf-8")
    assert glossary.count("道心 -> Dao Heart") == 1
    assert "Dao Heart" in (run_dir / "report.md").read_text(encoding="utf-8")
    report = (run_dir / "report.html").read_text(encoding="utf-8")
    for phrase in ("tools.search", "PATCH REJECTED", "PATCH ACCEPTED", "QA", "approval"):
        assert phrase.lower() in report.lower()
    assert "QA before: 2; candidate not evaluated" in report
    assert "Applied glossary delta" in report
    assert "Proposed glossary delta" not in report
    assert "not production evidence" in report.lower()


def test_golden_auto_approve_finishes_without_resume(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    result = _run(
        "harness",
        "golden",
        "--runs-dir",
        str(runs_dir),
        "--auto-approve",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    run_dir = runs_dir / "agentic_harness_v3_demo"
    snapshot = json.loads((run_dir / "session_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["status"] == "completed"
    _assert_neutral_public_artifacts(run_dir, snapshot)
    assert (run_dir / "glossary.txt").read_text(encoding="utf-8").count("Dao Heart") == 1


def test_golden_reject_is_terminal_without_glossary_write(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    paused = _run(
        "harness",
        "golden",
        "--runs-dir",
        str(runs_dir),
        "--pause-for-approval",
    )
    assert paused.returncode == 0, paused.stdout + paused.stderr
    run_dir = runs_dir / "agentic_harness_v3_demo"
    before = (run_dir / "glossary.txt").read_text(encoding="utf-8")
    rejected = _run(
        "harness",
        "resume",
        str(run_dir),
        "--reject",
        "--reviewer",
        "demo-reviewer",
        "--note",
        "Keep the glossary unchanged after review.",
    )
    assert rejected.returncode == 0, rejected.stdout + rejected.stderr
    snapshot = json.loads((run_dir / "session_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["status"] == "rejected"
    assert snapshot["episode"]["final_status"] == "escalated"
    _assert_neutral_public_artifacts(run_dir, snapshot)
    assert (run_dir / "glossary.txt").read_text(encoding="utf-8") == before
    report = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "Proposed glossary delta" in report
    assert "Applied glossary delta" not in report
    events = [
        json.loads(line)
        for line in (run_dir / "session_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert not any(event["event_type"] == "glossary_promotion_applied" for event in events)
