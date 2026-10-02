from pathlib import Path
import re


def _local_links(markdown: str) -> list[str]:
    links = re.findall(r"\]\(([^)\s]+)\)", markdown)
    return [link.split("#", 1)[0] for link in links
            if not link.startswith(("http://", "https://", "#", "mailto:"))]


def test_public_readme_leads_with_results_and_describes_the_released_surface() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())

    for text in (
        "Agentic Translation Reliability Harness",
        "source-grounded agent harness",
        "2.7× fewer inconsistent",
        "65.9%",
        "87.2%",
        "4.5× fewer glossary misses",
        "pre-registered test of 40 held-out chapters",
        "not overall literary quality",
        "14 wins, 9 ties, 17 losses",
        "docs/RESULTS.md",
        "python demo.py --offline",
        "samples/showcase/terms/demo_glossary.json",
        "Automatic repair is the default",
        "QA does not regress",
        "SHA-256 matches the current draft",
        "read-only specialists review terminology and fidelity",
        "samples/synthetic_repair_demo/story.yaml",
        "harness resume runs/synthetic-repair",
        "harness replay runs/synthetic-repair",
        "agentic_translation.mcp_server",
        "not a ranking",
        "DeepSeek",
        "Cache-only replay",
        "clean model review does not guarantee",
        "Optional Jev semantic checks",
        "Jev is off by default",
        "## Contents",
        "[Optional Jev semantic checks](#optional-jev-semantic-checks)",
        "docs/JEV_EXTENSION.md#aggregate-evidence",
        "five focused questions or eighteen dense questions",
        "scripted model decisions",
    ):
        assert text in readme or text in normalized

    assert "harness compare" not in readme
    assert readme.count("```mermaid") == 1
    assert "actions/workflows/harness-v3.yml/badge.svg" in readme
    assert "python-3.11%2B" in readme
    assert "license-MIT" in readme

    opening, remainder = readme.split("## Contents\n", 1)
    contents = remainder.split("\n## Try it\n", 1)[0]
    assert "Jev" in opening
    assert "87.2%" in opening
    assert "#optional-jev-semantic-checks" in contents
    assert "Vercel" not in readme
    assert "Gateway" not in readme
    assert "AI_GATEWAY_API_KEY" not in readme
    assert "96 assignments" not in readme
    assert "Execution failure |" not in readme


def test_public_front_door_links_resolve() -> None:
    for doc in (Path("README.md"), Path("docs/RESULTS.md")):
        for link in _local_links(doc.read_text(encoding="utf-8")):
            assert (doc.parent / link).exists(), f"{doc}: broken link {link}"


def test_public_docs_and_fixture_are_present() -> None:
    for path in (
        "assets/readme-banner.png",
        "DATA_NOTICE.md",
        "CHANGELOG.md",
        "demo.py",
        "docs/AUTOMATIC_REPAIR.md",
        "docs/RESULTS.md",
        "docs/SESSION_IDENTITY.md",
        "docs/JEV_EXTENSION.md",
        "docs/TRY_IT.md",
        "experiments/portfolio_demo/README.md",
        "samples/jev/policy.example.json",
        "samples/showcase/terms/demo_glossary.json",
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
