from pathlib import Path
import re


def _local_links(markdown: str) -> list[str]:
    links = re.findall(r"\]\(([^)\s]+)\)", markdown)
    return [link.split("#", 1)[0] for link in links
            if not link.startswith(("http://", "https://", "#", "mailto:"))]


def test_public_front_door_links_resolve() -> None:
    # Check that the published front door is usable, not that its prose keeps
    # matching an old marketing snapshot. Numeric evidence has its own tests.
    for doc in (
        Path("README.md"),
        Path("docs/RESULTS.md"),
        Path("docs/TRY_IT.md"),
        Path("experiments/heldout_evidence/README.md"),
    ):
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
        "experiments/heldout_evidence/RESULTS.json",
        "experiments/heldout_evidence/verify.py",
        "experiments/heldout_evidence/README.md",
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
