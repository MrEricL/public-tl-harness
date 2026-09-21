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
        "read-only specialists review terminology and fidelity",
        "samples/synthetic_repair_demo/story.yaml",
        "harness resume runs/synthetic-repair",
        "harness replay runs/synthetic-repair",
        "DeepSeek",
        "cache-only replay",
        "batch workflow",
        "clean model review does not guarantee",
        "optional Jev semantic checks",
        "Jev is off by default",
        "## Contents",
        "[Optional Jev semantic checks](#optional-jev-semantic-checks)",
        "docs/JEV_EXTENSION.md#aggregate-evidence",
        "five focused questions or eighteen dense questions",
        "scripted model decisions",
    ):
        assert text in readme or text in normalized

    assert "harness compare" not in readme
    assert "samples/showcase" not in readme
    assert readme.count("```mermaid") == 1
    assert "actions/workflows/harness-v3.yml/badge.svg" in readme
    assert "python-3.11%2B" in readme
    assert "license-MIT" in readme

    opening, remainder = readme.split("## Contents\n", 1)
    contents = remainder.split("\n## Quickstart\n", 1)[0]
    assert "Jev" in opening
    assert "#optional-jev-semantic-checks" in contents
    assert "Vercel" not in readme
    assert "Gateway" not in readme
    assert "AI_GATEWAY_API_KEY" not in readme
    assert "96 assignments" not in readme
    assert "Execution failure |" not in readme


def test_public_docs_and_fixture_are_present() -> None:
    for path in (
        "assets/readme-banner.png",
        "DATA_NOTICE.md",
        "CHANGELOG.md",
        "docs/AUTOMATIC_REPAIR.md",
        "docs/SESSION_IDENTITY.md",
        "docs/JEV_EXTENSION.md",
        "samples/jev/policy.example.json",
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
