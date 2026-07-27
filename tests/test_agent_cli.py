from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
from typer.testing import CliRunner

from agentic_translation.cli import app
from agentic_translation.providers_llm import LLMProviderUnavailable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORY = PROJECT_ROOT / "samples/agentic_repair_demo/story.yaml"


def test_demo_repair_golden_replay_writes_all_artifacts(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "demo-repair",
            "--story",
            str(STORY),
            "--chapter",
            "0001",
            "--provider-mode",
            "replay",
            "--runs-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "PATCH REJECTED" in result.output
    assert "PATCH ACCEPTED" in result.output
    assert "Final QA findings: 0" in result.output
    run_dir = tmp_path / "agentic_repair_demo_replay"
    for relative in [
        "translated_final/0001.txt",
        "qa_initial.json",
        "qa_final.json",
        "agent_episode.json",
        "repair_report.md",
        "report.html",
    ]:
        assert (run_dir / relative).exists(), relative
    assert json.loads((run_dir / "qa_final.json").read_text(encoding="utf-8"))["summary"]["total_findings"] == 0
    assert "{{" not in (run_dir / "report.html").read_text(encoding="utf-8")


def test_demo_repair_custom_run_id_replays_default_bundled_cache(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "demo-repair",
            "--story",
            str(STORY),
            "--chapter",
            "0001",
            "--provider-mode",
            "replay",
            "--run-id",
            "alternate",
            "--runs-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    run_dir = tmp_path / "alternate"
    episode = json.loads((run_dir / "agent_episode.json").read_text(encoding="utf-8"))
    assert episode["run_id"] == "agentic_repair_demo_replay"
    assert episode["episode_id"].startswith("agentic_repair_demo_replay:")
    assert len(episode["steps"]) == 5
    assert all(step["provider_call"]["cache_hit"] for step in episode["steps"])
    assert json.loads((run_dir / "qa_final.json").read_text(encoding="utf-8"))["summary"]["total_findings"] == 0


def test_demo_repair_reports_do_not_leak_absolute_input_or_output_paths(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "demo-repair",
            "--story",
            str(STORY),
            "--provider-mode",
            "replay",
            "--run-id",
            "privacy",
            "--runs-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    run_dir = tmp_path / "privacy"
    for report_name in ("repair_report.md", "report.html"):
        report = (run_dir / report_name).read_text(encoding="utf-8")
        assert str(PROJECT_ROOT) not in report
        assert str(tmp_path) not in report
        assert str(STORY) not in report


def test_demo_repair_module_entrypoint_runs_from_project_root(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentic_translation",
            "demo-repair",
            "--story",
            str(STORY),
            "--provider-mode",
            "replay",
            "--runs-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Final QA findings: 0" in result.stdout


def test_demo_repair_module_entrypoint_resolves_absolute_story_outside_cwd(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentic_translation",
            "demo-repair",
            "--story",
            str(STORY),
            "--provider-mode",
            "replay",
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
    assert "Final QA findings: 0" in result.stdout
    assert (runs_dir / "agentic_repair_demo_replay" / "report.html").exists()


def test_demo_repair_replay_cache_miss_fails_without_live_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_client(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("replay must not construct a live client")

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=fail_client))
    empty_cache = tmp_path / "empty-cache"
    empty_cache.mkdir()
    result = CliRunner().invoke(
        app,
        [
            "demo-repair",
            "--story",
            str(STORY),
            "--cache-dir",
            str(empty_cache),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 1
    assert "Error" in result.output
    assert "replay" in result.output.lower()


def test_demo_repair_replay_cache_miss_never_imports_live_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_import(name, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if name == "openai":
            raise AssertionError("replay must not import or construct a live client")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fail_import)
    empty_cache = tmp_path / "empty-cache"
    empty_cache.mkdir()
    result = CliRunner().invoke(
        app,
        [
            "demo-repair",
            "--story",
            str(STORY),
            "--cache-dir",
            str(empty_cache),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 1
    assert "No replay cache entry" in result.output or "Replay cache" in result.output


def test_demo_repair_existing_run_requires_overwrite(tmp_path: Path) -> None:
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    (run_dir / "keep.txt").write_text("sentinel", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "demo-repair",
            "--story",
            str(STORY),
            "--runs-dir",
            str(tmp_path),
            "--run-id",
            "existing",
        ],
    )
    assert result.exit_code == 1
    assert "already exists" in result.output
    assert (run_dir / "keep.txt").read_text(encoding="utf-8") == "sentinel"


def test_demo_repair_overwrite_replaces_only_selected_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "agentic_repair_demo_replay"
    run_dir.mkdir()
    (run_dir / "sentinel.txt").write_text("remove me", encoding="utf-8")
    sibling = tmp_path / "keep-me.txt"
    sibling.write_text("keep me", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "demo-repair",
            "--story",
            str(STORY),
            "--runs-dir",
            str(tmp_path),
            "--run-id",
            "agentic_repair_demo_replay",
            "--overwrite",
        ],
    )
    assert result.exit_code == 0, result.output
    assert not (run_dir / "sentinel.txt").exists()
    assert (run_dir / "report.html").exists()
    assert sibling.read_text(encoding="utf-8") == "keep me"


def test_demo_repair_overwrite_rejects_selected_run_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "sentinel.txt").write_text("keep target", encoding="utf-8")
    selected = tmp_path / "agentic_repair_demo_replay"
    selected.symlink_to(target, target_is_directory=True)
    result = CliRunner().invoke(
        app,
        [
            "demo-repair",
            "--story",
            str(STORY),
            "--runs-dir",
            str(tmp_path),
            "--overwrite",
        ],
    )
    assert result.exit_code == 1
    assert "symlink" in result.output.lower()
    assert selected.is_symlink()
    assert (target / "sentinel.txt").read_text(encoding="utf-8") == "keep target"


@pytest.mark.parametrize("chapter", ["../../victim", "bad/victim", ".."])
def test_demo_repair_rejects_chapter_path_traversal(tmp_path: Path, chapter: str) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("must remain untouched", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "demo-repair",
            "--story",
            str(STORY),
            "--chapter",
            chapter,
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 1
    assert "chapter" in result.output.lower()
    assert victim.read_text(encoding="utf-8") == "must remain untouched"
    assert not (tmp_path / "runs").exists()


def test_demo_repair_live_requires_explicit_recording_cache(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "demo-repair",
            "--story",
            str(STORY),
            "--provider-mode",
            "live",
            "--runs-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "cache_dir" in result.output.lower()
    assert "record_cache" in result.output.lower()


def test_demo_repair_live_requires_explicit_provider_and_model(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    result = CliRunner().invoke(
        app,
        [
            "demo-repair",
            "--story",
            str(STORY),
            "--provider-mode",
            "live",
            "--cache-dir",
            str(cache_dir),
            "--record-cache",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 1
    assert "provider" in result.output.lower()
    assert "model" in result.output.lower()


def test_demo_repair_rejects_unsupported_provider_cleanly(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "demo-repair",
            "--story",
            str(STORY),
            "--provider",
            "not-a-provider",
            "--runs-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert result.output.startswith("Error:")
    assert "Traceback" not in result.output
