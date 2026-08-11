"""Portable tests for the public, copyright-safe benchmark methodology.

All text fixtures in this module are synthetic and are written under pytest's
temporary directory.  The checked-in benchmark directory intentionally does
not contain the source corpus, translations, packets, scorecards, or session
traces used by the private run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark import (
    DIMENSIONS,
    _session_evidence,
    aggregate_scorecards,
    build_blind_packets,
    load_scorecard,
    render_report,
    run_scripted_session,
    select_window,
)


JUDGES = ("codex_judge_1", "codex_judge_2", "codex_judge_3", "claude_sonnet_5")
ACTION_PROVENANCE = (
    "Codex-originated repair trajectories migrated to the final bounded Harness v3 "
    "schema and executed through the public runtime."
)
PLAIN_PROVENANCE = (
    "The repair actions originated in Codex runs, were migrated to the Harness v3 "
    "schema, and re-executed through this codebase"
)
README_PROVENANCE = (
    "Codex runs produced the original repair actions.",
    "The actions were migrated to the Harness v3 schema and run again through "
    "this repository.",
    "No single end-to-end run produced the published result.",
)
ROSTER = [
    {
        "id": judge,
        "family": "claude" if judge == "claude_sonnet_5" else "codex",
        "model": "Sonnet 5" if judge == "claude_sonnet_5" else "gpt-5.6-terra",
        "worker": "all_purpose_worker",
        "reasoning": "high",
    }
    for judge in JUDGES
]


def make_corpus(tmp_path: Path, *, count: int) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for index in range(1, count + 1):
        (corpus / f"{index:04d}.txt").write_text(
            f"Synthetic chapter {index}\nPortable fixture {index}", encoding="utf-8"
        )
    return corpus


def one_scorecard(
    *,
    chapter: str = "0208",
    judge: str = "codex_judge_1",
    a_fidelity: int = 3,
    b_fidelity: int = 5,
    preference: str = "B",
) -> dict[str, object]:
    a_scores = {dimension: 3 for dimension in DIMENSIONS}
    b_scores = {dimension: 3 for dimension in DIMENSIONS}
    a_scores["fidelity"] = a_fidelity
    b_scores["fidelity"] = b_fidelity
    return {
        "judge": judge,
        "family": "claude" if judge == "claude_sonnet_5" else "codex",
        "chapters": [
            {
                "chapter": chapter,
                "scores": {"A": a_scores, "B": b_scores},
                "preference": preference,
                "pairwise_preferences": {
                    dimension: preference for dimension in DIMENSIONS
                }
                | {"overall": preference},
                "reason": "Candidate B preserves a synthetic term more clearly.",
                "issues": {"A": [], "B": []},
                "serious_errors": {"A": [], "B": []},
            }
        ],
    }


def test_select_window_is_seeded_and_stays_inside_middle_band(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path, count=20)
    selection = select_window(corpus, seed=20260810, count=5, lower=0.30, upper=0.70)
    assert len(selection.files) == 5
    assert selection.files == sorted(selection.files)
    assert selection.start_index >= 6
    assert selection.start_index + 4 < 14
    assert selection.to_dict() == select_window(
        corpus, seed=20260810, count=5, lower=0.30, upper=0.70
    ).to_dict()


def test_scripted_actions_run_through_public_session_with_synthetic_text(
    tmp_path: Path,
) -> None:
    result = run_scripted_session(
        source_text="Synthetic source with a placeholder.",
        baseline_text="Chapter 1\nDraft: 残留。",
        glossary_text="",
        actions=[
            {"tool": "tools.search", "query": "submit patch", "limit": 8},
            {
                "tool": "submit_patch",
                "edits": [{"old_text": "残留。", "new_text": "canonical term."}],
                "rationale": "Use the synthetic glossary fixture.",
            },
            {"tool": "finish", "summary": "Verified synthetic fixture."},
        ],
        session_dir=tmp_path / "session",
        chapter="synthetic-001",
    )
    assert result.final_text == "Chapter 1\nDraft: canonical term."
    assert result.episode.steps[1].action["tool"] == "submit_patch"
    assert (tmp_path / "session" / "session_events.jsonl").is_file()


def test_session_evidence_counts_mutations_without_checked_in_traces(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "sessions" / "synthetic-001"
    session_dir.mkdir(parents=True)
    (session_dir / "agent_episode.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "action": {"tool": "normalize_punctuation"},
                        "observation": {"data": {"accepted": True}},
                    },
                    {
                        "action": {"tool": "submit_patch"},
                        "observation": {"data": {"accepted": False}},
                    },
                    {
                        "action": {"tool": "normalize_punctuation"},
                        "observation": {"data": {"accepted": False}},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (session_dir / "session_snapshot.json").write_text(
        json.dumps({"patch_attempts": 3, "status": "completed"}), encoding="utf-8"
    )

    _, actions = _session_evidence(
        chapters=["synthetic-001"],
        sessions_root=tmp_path / "sessions",
        config={"limits": {"max_steps": 6, "max_patch_attempts": 2}},
    )

    assert actions["by_chapter"]["synthetic-001"]["accepted"] == 1
    assert actions["by_chapter"]["synthetic-001"]["rejected"] == 2
    assert actions["totals"]["accepted"] == 1
    assert actions["totals"]["rejected"] == 2


def test_blind_mapping_is_counterbalanced_and_packets_are_synthetic(tmp_path: Path) -> None:
    (tmp_path / "source").mkdir()
    (tmp_path / "baseline").mkdir()
    (tmp_path / "harness" / "outputs").mkdir(parents=True)
    chapters = ["0208", "0209", "0210", "0211", "0212"]
    for chapter in chapters:
        (tmp_path / "source" / f"{chapter}.txt").write_text(
            f"Synthetic source {chapter}", encoding="utf-8"
        )
        (tmp_path / "baseline" / f"{chapter}.txt").write_text(
            f"Synthetic draft one {chapter}", encoding="utf-8"
        )
        (tmp_path / "harness" / "outputs" / f"{chapter}.txt").write_text(
            f"Synthetic draft two {chapter}", encoding="utf-8"
        )
    mapping = build_blind_packets(
        chapters=chapters, root=tmp_path, seed=77, judge_ids=list(JUDGES)
    )
    assert set(mapping) == set(JUDGES)
    for chapter in chapters:
        baseline_a = sum(mapping[judge][chapter]["A"] == "baseline" for judge in JUDGES)
        assert baseline_a == 2
    for judge in JUDGES:
        assert json.loads(
            (tmp_path / "blind" / "private_mapping" / f"{judge}.json").read_text()
        ) == mapping[judge]
        for chapter in chapters:
            packet = (tmp_path / "blind" / "packets" / judge / f"{chapter}.md").read_text()
            assert "Synthetic source" in packet
            assert "Candidate A" in packet
            assert "Candidate B" in packet
            assert "baseline" not in packet.casefold()
            assert "harness" not in packet.casefold()
    assert mapping == build_blind_packets(
        chapters=chapters, root=tmp_path, seed=77, judge_ids=list(JUDGES)
    )


def test_aggregate_preserves_mapped_metrics_and_renders_notice(tmp_path: Path) -> None:
    session_dir = tmp_path / "harness" / "sessions" / "0208"
    session_dir.mkdir(parents=True)
    (session_dir / "agent_episode.json").write_text(
        json.dumps(
            {
                "chapter": "0208",
                "initial_qa": {
                    "summary": {"total_findings": 2, "by_check": {"glossary_required": 2}},
                    "findings": [{}, {}],
                    "score": 80,
                },
                "final_qa": {
                    "summary": {"total_findings": 0, "by_check": {}},
                    "findings": [],
                    "score": 100,
                },
                "steps": [
                    {"action": {"tool": "tools.search"}},
                    {
                        "action": {"tool": "submit_patch"},
                        "observation": {"data": {"accepted": True}},
                    },
                    {"action": {"tool": "finish"}},
                ],
                "final_status": "verified",
            }
        ),
        encoding="utf-8",
    )
    (session_dir / "session_snapshot.json").write_text(
        json.dumps({"patch_attempts": 1, "status": "completed"}), encoding="utf-8"
    )
    config = {
        "blind": {"roster": ROSTER},
        "selection": {"files": ["0208.txt"], "seed": 7},
        "optional": {"deepseek": {"status": "skipped"}},
        "data_availability": {"full_text_artifacts_distributed": False},
    }
    results = aggregate_scorecards(
        mapping={judge: {"0208": {"A": "baseline", "B": "harness"}} for judge in JUDGES},
        scorecards=[one_scorecard(a_fidelity=3, b_fidelity=5, preference="B")],
        root=tmp_path,
        config=config,
    )
    assert results["systems"]["baseline"]["dimensions"]["fidelity"] == 3.0
    assert results["systems"]["harness"]["dimensions"]["fidelity"] == 5.0
    assert results["pairwise_preferences"]["overall"]["harness"] == 1
    assert results["deterministic_qa"]["totals"]["after_findings"] == 0
    report = render_report(results, config=config)
    assert "real five-chapter corpus" in report
    assert "intentionally not distributed" in report


def test_public_tree_has_no_full_text_benchmark_artifacts() -> None:
    root = Path(__file__).resolve().parent
    prohibited = (
        root / "source",
        root / "baseline",
        root / "blind" / "packets",
        root / "blind" / "private_mapping",
        root / "blind" / "scorecards",
        root / "harness" / "outputs",
        root / "harness" / "sessions",
        root / "harness" / "actions",
    )
    assert all(not path.exists() for path in prohibited)


def test_public_aggregate_and_docs_preserve_sanitized_provenance_contract() -> None:
    root = Path(__file__).resolve().parent
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    readme = (root / "README.md").read_text(encoding="utf-8")
    report = (root / "REPORT.md").read_text(encoding="utf-8")
    examples = (root / "PUBLIC_EXAMPLES.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())
    normalized_report = " ".join(report.split())

    assert results["provenance"]["action_origin"] == ACTION_PROVENANCE
    assert "chapter_evaluations" not in results
    assert ACTION_PROVENANCE not in readme
    assert ACTION_PROVENANCE not in report
    for statement in README_PROVENANCE:
        assert statement in normalized_readme
    assert PLAIN_PROVENANCE in normalized_report
    assert "not produced in a single end-to-end run" in normalized_report
    for internal_label in (
        "gpt-5.6-luna",
        "gpt-5.6-terra",
        "core_worker",
        "all_purpose_worker",
    ):
        assert internal_label not in readme
        assert internal_label not in report
    assert "weak directional formatting preference in this sample" in report
    assert examples.count("**") == 4
