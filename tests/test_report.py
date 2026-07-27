from __future__ import annotations

from pathlib import Path

from agentic_translation.pipeline import run_demo_pipeline
from agentic_translation.report import PACKAGE_TEMPLATE_DIR


def test_main_report_template_is_packaged_with_the_module() -> None:
    assert PACKAGE_TEMPLATE_DIR.parent.name == "agentic_translation"
    assert (PACKAGE_TEMPLATE_DIR / "report.html.j2").is_file()


def test_report_contains_required_demo_sections(tmp_path: Path) -> None:
    result = run_demo_pipeline(
        Path("samples/public_demo/story.yaml"),
        offline=True,
        run_id="report_test",
        seed=7,
        overwrite=True,
        runs_dir=tmp_path,
    )
    report = result.report_path.read_text(encoding="utf-8")

    for text in [
        "Pipeline Timeline",
        "Compliance vs Quality",
        "Glossary Canon",
        "QA Findings",
        "Router / Candidate Selection",
        "Patch Acceptance",
        "Artifact QA",
        "Eval Deltas",
        "Artifacts",
        "Heavenly Dao",
    ]:
        assert text in report
    assert "{{" not in report
    assert "}}" not in report


def test_report_frames_demo_as_cockpit_and_harness(tmp_path: Path) -> None:
    result = run_demo_pipeline(
        Path("samples/public_demo/story.yaml"),
        offline=True,
        run_id="cockpit_report_test",
        seed=7,
        overwrite=True,
        runs_dir=tmp_path,
    )
    report = result.report_path.read_text(encoding="utf-8")

    for text in [
        "Run Cockpit",
        "Productive distrust",
        "Harness trace",
        "Evaluator-optimizer loop",
        "Router decision",
        "Patch accepted after QA",
        "Offline replayable harness",
    ]:
        assert text in report


def test_report_shows_decision_timeline_cards(tmp_path: Path) -> None:
    result = run_demo_pipeline(
        Path("samples/public_demo/story.yaml"),
        offline=True,
        run_id="decision_timeline_report_test",
        seed=7,
        overwrite=True,
        runs_dir=tmp_path,
    )
    report = result.report_path.read_text(encoding="utf-8")

    for text in [
        "Decision Timeline",
        "1. Translate Cheap",
        "2. QA Gauntlet",
        "3. Route Findings",
        "4. Editor's Room",
        "5. Patch Acceptance",
        "6. Re-QA",
        "7. Artifact Gate",
        "8. Outcome Receipt",
        "The run is a sequence of auditable decisions",
    ]:
        assert text in report
    assert "Cost Receipt" not in report
    assert "cost saved" not in report.lower()


def test_report_shows_bench_ablation_strip(tmp_path: Path) -> None:
    result = run_demo_pipeline(
        Path("samples/public_demo/story.yaml"),
        offline=True,
        run_id="bench_strip_report_test",
        seed=7,
        overwrite=True,
        runs_dir=tmp_path,
    )
    report = result.report_path.read_text(encoding="utf-8")

    for text in [
        "Bench Ablation Strip",
        "cheap baseline",
        "glossary canon",
        "router + patch loop",
        "artifact gate",
        "Findings",
        "Score gain",
        "This is a bench view, not a semantic quality claim.",
    ]:
        assert text in report
    assert "frontier everywhere estimate" not in report
    assert "Estimated Cost" not in report


def test_offline_report_is_labeled_as_a_baseline_not_an_agent(tmp_path: Path) -> None:
    result = run_demo_pipeline(
        Path("samples/public_demo/story.yaml"),
        offline=True,
        run_id="offline_label_report_test",
        seed=7,
        overwrite=True,
        runs_dir=tmp_path,
    )
    report = result.report_path.read_text(encoding="utf-8")

    assert "Offline Translation Harness Baseline" in report
    assert "Agentic Long-Form Translation Demo" not in report


def test_report_shows_editors_room_orchestration(tmp_path: Path) -> None:
    result = run_demo_pipeline(
        Path("samples/public_demo/story.yaml"),
        offline=True,
        run_id="editors_room_report_test",
        seed=7,
        overwrite=True,
        runs_dir=tmp_path,
    )
    report = result.report_path.read_text(encoding="utf-8")

    for text in [
        "Editor's Room",
        "orchestrator-workers",
        "blind judge",
        "literal worker / unchanged",
        "canon-strict worker / panel_repair",
        "fluent worker / style_repair",
    ]:
        assert text in report
