from __future__ import annotations

from pathlib import Path


def test_harness_v3_front_door_docs_contract() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())
    for text in [
        "The LLM decides what repair to try.",
        "The model never writes to the final translation directly.",
        "Jump to:",
        "harness checklist",
        "### Agent harness checklist",
        "https://rubriclabs.com/blog/what-is-an-agent-harness",
        "#### Included",
        "**Looping.**",
        "**Context.**",
        "**Orchestration.**",
        "**Tool execution.**",
        "**Verification.**",
        "**Permissions.**",
        "**Observability.**",
        "#### Domain-specific",
        "**Memory.**",
        "**Dispatch.**",
        "#### Out of scope",
        "**Context compaction.**",
        "**Separate planning.**",
        "**MCP.**",
        "Run the two-minute Harness v3 walkthrough",
        "This walkthrough demonstrates the local control loop",
        "python -m agentic_translation harness golden --runs-dir runs --pause-for-approval --overwrite",
        "python -m agentic_translation harness resume runs/agentic_harness_v3_demo --approve --reviewer demo-reviewer --note \"Approve the reviewed run-local glossary promotion.\"",
        "python -m agentic_translation harness golden --runs-dir runs --auto-approve --overwrite",
        "runs/agentic_harness_v3_demo/report.html",
        "participant M as Model",
        "participant P as Python runtime",
        "participant R as Human reviewer",
        "'mirrorActors': false",
        "'actorFontWeight': 600",
        "'messageMargin': 40",
        "loop Until verified, escalated, or budget exhausted",
        "Validate, test on working copy, run QA",
        "Accept and return updated findings",
        "Reject and return findings",
        "Verify saved session identity",
        "Resume with decision",
        "Only a human reviewer can approve writing it to the persistent glossary.",
        "The persistent glossary remains unchanged.",
        "3 cases × 3 variants",
        "9/9 passed",
        "20 decisions: 4 judges × 5 chapters, with answer order counterbalanced",
        "3 Codex evaluator runs + 1 Claude evaluator run",
        "The five-chapter corpus and its full-text derivatives are not distributed for copyright reasons.",
    ]:
        assert text in readme or text in normalized_readme

    assert "recruiter-facing" not in readme
    assert "does **not** claim production translation quality" not in readme
    assert "## Shipped and intentionally deferred" not in readme
    assert "## Technical highlights" not in readme
    assert "## Harness v3 control flow" not in readme
    assert "## Control model" not in readme
    assert readme.count("```mermaid") == 1
    assert "assets/harness-v3-cockpit.png" not in readme
    assert "run-glossary write held at **PENDING** until a reviewer decides" in readme
    assert "exact one-shot baseline draft" not in readme
    assert "same `core_worker` class" not in readme

    story_index = readme.index("## Why I built this")
    assert story_index < readme.index("### Run the two-minute Harness v3 walkthrough")
    assert "actions/workflows/harness-v3.yml/badge.svg" in readme
    assert "python-3.11%2B" in readme
    assert "license-MIT" in readme

    assert Path("assets/readme-banner.png").is_file()
    assert "[USER_GUIDE.md](USER_GUIDE.md)" in readme
    assert "experiments/mid_corpus_harness_benchmark/README.md" in readme
    assert "experiments/mid_corpus_harness_benchmark/REPORT.md" in readme
    assert "experiments/mid_corpus_harness_benchmark/PUBLIC_EXAMPLES.md" in readme
    assert "| Deterministic rule hits | 68 → 0 |" in readme
    assert "| Overall | 5 | 7 | 8 |" in readme
    assert "| Formatting | 4 | 2 | 14 |" in readme
    assert "| Terminology | 1 | 5 | 14 |" in readme
    assert "| Codex evaluators | 3 | 15 | 3 | 6 | 6 |" in readme
    assert "| Claude evaluator | 1 | 5 | 2 | 1 | 2 |" in readme
    assert "| **Total** | **4** | **20** | **5** | **7** | **8** |" in readme
    assert "encoded rule hits fell from 68 to 0" in readme
    assert "Blind overall preference favored the baseline 7–5, with 8 ties." in normalized_readme
    assert "68:56/11/1" not in readme

    guide = Path("USER_GUIDE.md").read_text(encoding="utf-8")
    assert "After installation (see [Standard Install](#standard-install))" in guide
    assert "approval-gated run-local glossary promotion/write" in guide
    assert "runs/agentic_harness_v3_demo/report.md" in guide
    assert "runs/agentic_harness_v3_demo/translated_final.txt" in guide
    v3_section = guide.split("## 1. Core Concepts", 1)[0]
    assert "runs/agentic_harness_v3_demo/repair_report.md" not in v3_section
    assert "runs/agentic_harness_v3_demo/translated_final/0001.txt" not in v3_section

    script = Path("DEMO_SCRIPT.md").read_text(encoding="utf-8")
    for text in [
        "Two-minute Harness v3 demo",
        "After installation (see [README.md](README.md) or [USER_GUIDE.md](USER_GUIDE.md))",
        "tools_search",
        "dynamic",
        "02 · Persisted session receipt",
        "primary numbered action rail",
        "escalate`, `finish`, `get_qa_findings`, and",
        "Rejected",
        "Accepted",
        "policy decision is",
        "**PENDING**",
        "python -m agentic_translation harness resume runs/agentic_harness_v3_demo --approve --reviewer demo-reviewer --note \"Approve the reviewed run-local glossary promotion.\"",
        "python -m agentic_translation harness bench --suite samples/harness_eval/v3_cases.json --out runs/harness_eval --overwrite",
        "golden artifacts are inspectable",
        "paused session is resumable",
        "not replay-cache evidence",
        "prove literary translation quality",
    ]:
        assert text in script


def test_public_readme_keeps_personal_story_before_quickstart() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    story = readme.split("## Why I built this\n", 1)[1].split(
        "\n---\n\n## Quickstart and evidence\n", 1
    )[0]

    assert "Moon-Shadow Step" in story
    assert "The vision is" in story
    assert readme.index("## Why I built this") < readme.index(
        "## Quickstart and evidence"
    )


def test_public_docs_use_portable_commands_and_omit_local_billing_anecdotes() -> None:
    public_docs = [
        Path("README.md"),
        Path("USER_GUIDE.md"),
        Path("DEMO_SCRIPT.md"),
        Path("docs/SESSION_IDENTITY.md"),
        Path("experiments/mid_corpus_harness_benchmark/README.md"),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in public_docs)

    assert "zsh -ic 'global_env" not in combined
    assert "redacted old local project key" not in combined
    assert "simulator_alliance" not in combined
    assert "recruiter" not in combined.lower()
    assert "portfolio" not in combined.lower()
    assert " mvp" not in combined.lower()
    for anecdotal_phrase in (
        "In this shell",
        "account is out of balance",
        "spend pennies",
        "funded key",
    ):
        assert anecdotal_phrase not in combined
