from pathlib import Path


def test_public_readme_describes_the_released_automatic_surface() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())

    for text in (
        "Agentic Translation Reliability Harness",
        "source-grounded agent harness",
        "Automatic repair is the default",
        "QA does not regress",
        "SHA-256 matches the current draft",
        "Bounded, read-only terminology and fidelity specialists",
        "samples/synthetic_repair_demo/story.yaml",
        "harness resume runs/synthetic-repair",
        "harness replay runs/synthetic-repair",
        "DeepSeek V4 Flash",
        "strict JSON actions",
        "2048",
        "The model proposes; the verifier disposes.",
        "cache-only terminology replay",
        "batch translation workflow",
        "clean model review is evidence",
    ):
        assert text in readme or text in normalized

    assert "harness compare" not in readme
    assert "samples/showcase" not in readme
    assert readme.count("```mermaid") == 1
    assert "actions/workflows/harness-v3.yml/badge.svg" in readme
    assert "python-3.11%2B" in readme
    assert "license-MIT" in readme


def test_public_docs_and_fixture_are_present() -> None:
    for path in (
        "assets/readme-banner.png",
        "DATA_NOTICE.md",
        "CHANGELOG.md",
        "docs/AUTOMATIC_REPAIR.md",
        "docs/SESSION_IDENTITY.md",
        "samples/synthetic_repair_demo/story.yaml",
        "samples/synthetic_repair_demo/scenario.json",
    ):
        assert Path(path).is_file()


def test_demo_script_uses_only_the_neutral_front_door() -> None:
    script = Path("DEMO_SCRIPT.md").read_text(encoding="utf-8")
    assert "samples/synthetic_repair_demo/story.yaml" in script
    assert "controller" in script
    assert "valve" in script
    assert "awaiting_approval" in script
    assert "harness replay" in script
    assert "samples/showcase" not in script
