from __future__ import annotations

from pathlib import Path


def test_readme_frontloads_the_bounded_replay_thesis() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    for text in [
        "Agentic Translation Reliability Harness",
        "Why I built this",
        "Moon-Shadow Step",
        "Quickstart and evidence",
        "Run the two-minute Harness v3 walkthrough",
        "Architecture and operations",
        "Agent harness checklist",
        "Supplementary terminology replay",
        "resolve_terminology → submit_patch → finish",
        "Control boundaries",
        "Two-model terminology arbitration",
        "QA-gated patch acceptance",
        "the model proposes; the verifier disposes",
        "Advanced corpus workflows",
    ]:
        assert text in readme

    assert "## What This Is Not" not in readme
    assert readme.count("### Supplementary terminology replay") == 1


def test_readme_story_order_and_public_benchmark_links_are_stable() -> None:
    readme_path = Path("README.md")
    text = readme_path.read_text(encoding="utf-8")
    assert text.count("## Why I built this\n") == 1
    assert text.index("## Why I built this\n") < text.index(
        "## Quickstart and evidence"
    )

    for relative_path in (
        "experiments/mid_corpus_harness_benchmark/README.md",
        "experiments/mid_corpus_harness_benchmark/REPORT.md",
        "experiments/mid_corpus_harness_benchmark/PUBLIC_EXAMPLES.md",
    ):
        assert relative_path in text
        assert (readme_path.parent / relative_path).is_file()


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
        "The provider-native code path is covered by injected-client tests. No checked-in artifact claims a real network provider-native call.",
    ]:
        assert text in script
