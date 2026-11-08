"""Public command coverage without model requests."""
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentic_translation import autonomous_book
from agentic_translation.cli import app


def test_public_help_exposes_book_run_only():
    runner = CliRunner()
    root = runner.invoke(app, ["--help"])
    assert root.exit_code == 0
    assert "harness" in root.output
    assert "batch" not in root.output
    commands = runner.invoke(app, ["harness", "--help"])
    assert commands.exit_code == 0
    assert "run" in commands.output
    assert "golden" not in commands.output
    assert "resume" not in commands.output


@pytest.mark.parametrize("strategy", ["contextual", "portfolio-review"])
def test_public_book_command_uses_the_shared_runner(tmp_path, monkeypatch, strategy):
    seen = []

    def run(story, out, **kwargs):
        seen.append((story, out, kwargs))
        return {"status": "delivered", "delivery": "delivery/book.txt"}

    monkeypatch.setattr(autonomous_book, "run_unattended_book", run)
    story, out = tmp_path / "story.yaml", tmp_path / "run"
    result = CliRunner().invoke(app, [
        "harness", "run", "--story", str(story), "--out", str(out),
        "--strategy", strategy, "--interaction", "unattended",
        "--memory", "source-grounded", "--glossary-policy", "run-local-auto",
    ])
    assert result.exit_code == 0, result.output
    assert seen == [(story, out, {"model": "deepseek-flash", "decision_model": "jev-1.13.0",
                                  "max_usd": 10.0, "strategy": strategy})]


@pytest.mark.parametrize("extra", [["--decision-provider", "jev"], ["--profile", "openai"]])
def test_invalid_book_settings_are_rejected_before_generation(tmp_path, monkeypatch, extra):
    def unexpected(*args, **kwargs):
        raise AssertionError("The runner must not start for an invalid policy")

    monkeypatch.setattr(autonomous_book, "run_unattended_book", unexpected)
    result = CliRunner().invoke(app, [
        "harness", "run", "--out", str(tmp_path / "run"),
        "--strategy", "contextual", "--interaction", "unattended",
        "--memory", "source-grounded", "--glossary-policy", "run-local-auto", *extra,
    ])
    assert result.exit_code == 1, result.output


def test_a_failed_book_run_explains_itself(tmp_path, monkeypatch):
    def failed(story, out, **kwargs):
        return {"status": "failed_no_output", "error": {"type": "GenerationTransportError"}}

    monkeypatch.setattr(autonomous_book, "run_unattended_book", failed)
    result = CliRunner().invoke(app, ["harness", "run", "--out", str(tmp_path / "run"), "--interaction", "unattended", "--memory", "source-grounded", "--glossary-policy", "run-local-auto"])
    assert result.exit_code == 1
    assert "GenerationTransportError" in result.output
    assert "DEEPSEEK_API_KEY" in result.output
    assert "Error: 1" not in result.output
