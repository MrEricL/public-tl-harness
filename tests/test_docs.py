from __future__ import annotations

from pathlib import Path


def test_readme_frontloads_the_bounded_replay_thesis() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    for text in [
        "Agentic Translation Reliability Harness",
        "Why I built this",
        "Moon-Shadow Step",
        "What the harness changes",
        "Run the main demo",
        "resolve_terminology → submit_patch → finish",
        "What “agentic” means here",
        "What the harness does",
        "Two-model terminology arbitration",
        "Deterministic patch promotion",
        "the model proposes; the verifier disposes",
        "Advanced corpus workflows",
    ]:
        assert text in readme

    assert "## What This Is Not" not in readme
    assert readme.count("## Run the main demo") == 1


def test_demo_script_captures_the_public_demo_story() -> None:
    script = Path("DEMO_SCRIPT.md").read_text(encoding="utf-8")

    for text in [
        "90-second agentic replay demo",
        "resolve_terminology",
        "bounded consensus",
        "verifier-controlled patch",
        "durable evidence",
        "prove literary translation quality",
        "samples/agentic_terminology_demo/story.yaml",
    ]:
        assert text in script
