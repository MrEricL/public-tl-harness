from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentic_translation.cli import app
from agentic_translation.harness_evals import load_suite, run_harness_eval


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUITE = PROJECT_ROOT / "samples/harness_eval/v3_cases.json"


def test_v3_suite_runs_three_cases_across_three_protocol_variants(tmp_path: Path) -> None:
    report = run_harness_eval(load_suite(SUITE), tmp_path / "bench")

    assert len(report.results) == 9
    assert {item.variant for item in report.results} == {
        "prompt_json_all_tools",
        "native_all_tools",
        "native_dynamic_tools",
    }
    assert all(item.passed for item in report.results)
    assert {summary.passed for summary in report.summaries} == {3}
    assert {summary.total for summary in report.summaries} == {3}
    assert {summary.rejected for summary in report.summaries} == {1}
    assert (tmp_path / "bench" / "harness_eval.json").exists()
    for item in report.results:
        artifact = tmp_path / "bench" / item.artifact_path
        assert artifact.is_dir()
        assert (artifact / "session_events.jsonl").exists()
        assert (artifact / "session_snapshot.json").exists()
        assert (artifact / "agent_episode.json").exists()


def test_v3_eval_exposes_real_protocol_and_dynamic_schema_delta(tmp_path: Path) -> None:
    report = run_harness_eval(load_suite(SUITE), tmp_path / "bench")
    by_key = {(item.variant, item.case): item for item in report.results}

    assert all(
        item.tool_protocol == "json_prompt"
        for item in report.results
        if item.variant == "prompt_json_all_tools"
    )
    assert all(
        item.tool_protocol == "native_function"
        for item in report.results
        if item.variant != "prompt_json_all_tools"
    )
    for case in ("repair_verified", "promotion_approved", "promotion_rejected"):
        native = by_key[("native_all_tools", case)]
        dynamic = by_key[("native_dynamic_tools", case)]
        assert dynamic.first_exposed_tool_count < native.first_exposed_tool_count
        assert dynamic.first_request_schema_bytes < native.first_request_schema_bytes


def test_v3_eval_approval_effect_is_exactly_once_or_write_free(tmp_path: Path) -> None:
    report = run_harness_eval(load_suite(SUITE), tmp_path / "bench")
    by_key = {(item.variant, item.case): item for item in report.results}

    for variant in (
        "prompt_json_all_tools",
        "native_all_tools",
        "native_dynamic_tools",
    ):
        approved = by_key[(variant, "promotion_approved")]
        rejected = by_key[(variant, "promotion_rejected")]
        approved_glossary = tmp_path / "bench" / approved.artifact_path / "glossary.txt"
        rejected_glossary = tmp_path / "bench" / rejected.artifact_path / "glossary.txt"
        assert approved.approval_outcome == "approved"
        assert approved.glossary_write_count == 1
        assert approved_glossary.read_text(encoding="utf-8").count("道心 -> Dao Heart") == 1
        assert rejected.approval_outcome == "rejected"
        assert rejected.glossary_write_count == 0
        assert rejected_glossary.read_text(encoding="utf-8") == "道心 -> Heart of Dao\n"
        assert rejected.session_status == "rejected"
        assert rejected.final_status == "escalated"


def test_harness_bench_cli_writes_and_prints_canonical_json(tmp_path: Path) -> None:
    out = tmp_path / "cli-bench"
    result = CliRunner().invoke(
        app,
        [
            "harness",
            "bench",
            "--suite",
            str(SUITE),
            "--out",
            str(out),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    saved = json.loads((out / "harness_eval.json").read_text(encoding="utf-8"))
    printed = json.loads(result.stdout)
    assert printed == saved
    assert len(printed["results"]) == 9


@pytest.mark.parametrize("json_mode", [False, True])
def test_harness_bench_cli_fails_if_any_report_result_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    json_mode: bool,
) -> None:
    report = run_harness_eval(load_suite(SUITE), tmp_path / "fixture")
    failed = report.model_copy(
        update={
            "results": [
                report.results[0].model_copy(update={"passed": False}),
                *report.results[1:],
            ]
        }
    )
    monkeypatch.setattr("agentic_translation.cli.run_harness_eval", lambda *_args, **_kwargs: failed)

    args = [
        "harness",
        "bench",
        "--suite",
        str(SUITE),
        "--out",
        str(tmp_path / "cli-bench"),
    ]
    if json_mode:
        args.append("--json")
    result = CliRunner().invoke(app, args)

    assert result.exit_code == 1, result.output
    if json_mode:
        printed = json.loads(result.stdout)
        assert printed["results"][0]["pass"] is False
