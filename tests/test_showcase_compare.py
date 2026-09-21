from pathlib import Path

from agentic_translation.glossary import load_glossary
from agentic_translation.showcase_compare import (
    FIXTURE_DIR,
    _comparison_report,
    _record_case,
    _summarize_records,
    run_comparison,
)


def test_offline_comparison_runs_every_case_with_equal_inputs(tmp_path: Path) -> None:
    result = run_comparison(tmp_path / "comparison")

    assert result["trials"] == 1
    assert len(result["records"]) == 36
    assert result["comparison_scope"] == "repair_same_supplied_draft"
    assert result["evidence_kind"] == "scripted_contract"
    assert all(row["status"] != "failed" for row in result["records"])
    assert all(row["fixture_expected_match"] for row in result["records"])
    assert all(row["evidence_kind"] == "scripted_contract" for row in result["records"])
    assert all(row["usage"]["total_tokens"] is None for row in result["records"])
    for chapter in result["case_ids"]:
        rows = [row for row in result["records"] if row["case_id"] == chapter]
        assert {row["strategy"] for row in rows} == {"deterministic", "single", "specialists"}
        drafts = [(Path(row["run_dir"]) / f"inputs/drafts/{chapter}.txt").read_text() for row in rows]
        assert len(set(drafts)) == 1
        for row in rows:
            root = Path(row["run_dir"])
            assert not list((root / "inputs").rglob("evaluator.json"))
            if row["category"] == "semantic" and row["strategy"] == "specialists":
                assert row["semantic_handling_success"]
                assert not (root / "delivery").exists()
            if row["strategy"] == "deterministic":
                assert row["usage"]["provider_calls"] == 0
            elif row["strategy"] == "single":
                assert row["child_count"] == 0
    assert (tmp_path / "comparison/report.md").exists()
    assert (tmp_path / "comparison/blind_packet.md").exists()


def test_live_observation_does_not_score_authored_fixture_labels(tmp_path: Path) -> None:
    run_dir = tmp_path / "live-run"
    row = _record_case(
        manifest={
            "run_id": "live-observation-test",
            "chapters": {
                "semantic-01": {
                    "status": "review_required",
                    "final_status": "review_required",
                    "steps": 1,
                    "child_calls": 0,
                    "provider_calls": [],
                }
            },
        },
        run_dir=run_dir,
        chapter="semantic-01",
        case={"id": "semantic-01", "category": "semantic", "expected": {"specialists": "review_required"}},
        source_text="Source text",
        glossary=load_glossary(FIXTURE_DIR / "terms/master_glossary.txt"),
        strategy="specialists",
        trial=1,
        provider_mode="live",
    )

    assert row["expected_outcome"] is None
    assert row["fixture_expected_match"] is None
    assert row["outcome_match"] is None
    assert row["comparison_scope"] == "repair_same_supplied_draft"
    assert row["evidence_kind"] == "live_observation"

    summary = _summarize_records([row])
    assert summary["by_strategy"]["specialists"]["fixture_expected_match_rate"] is None
    assert summary["by_category"]["semantic"]["fixture_expected_match_rate"] is None
    mixed_summary = _summarize_records([row, {**row, "fixture_expected_match": True}])
    assert mixed_summary["by_strategy"]["specialists"]["fixture_expected_match_rate"] == 1.0

    report = _comparison_report(
        {
            "provider_mode": "live",
            "comparison_scope": "repair_same_supplied_draft",
            "evidence_kind": "live_observation",
            "case_count": 1,
            "trials": 1,
            "summary": summary,
        }
    )
    assert "Scope: repair_same_supplied_draft | Evidence: live_observation" in report
    assert "Semantic review fields record whether a review was observed" in report
    assert "expected to surface" not in report
