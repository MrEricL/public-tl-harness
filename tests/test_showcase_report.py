from __future__ import annotations

import json
from pathlib import Path

from agentic_translation.showcase_report import (
    build_showcase_report_context,
    render_showcase_report,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_fixture(tmp_path: Path, *, status: str = "completed") -> tuple[Path, dict]:
    run_dir = tmp_path / "showcase-run"
    (run_dir / "inputs").mkdir(parents=True)
    (run_dir / "chapters/0001").mkdir(parents=True)
    (run_dir / "chapters/0001/children").mkdir(parents=True)
    (run_dir / "delivery").mkdir()
    (run_dir / "inputs/master_glossary.txt").write_text(
        "校准模式 -> calibration mode\n", encoding="utf-8"
    )
    (run_dir / "glossary.txt").write_text(
        "校准模式 -> calibration mode\n采样周期 -> sampling cycle\n", encoding="utf-8"
    )
    (run_dir / "chapters/0001/chapter_glossary.txt").write_text(
        "校准模式 -> calibration mode\n采样周期 -> sampling cycle\n", encoding="utf-8"
    )
    (run_dir / "chapters/0001/source.txt").write_text(
        "控制器使用校准模式检查阀门。", encoding="utf-8"
    )
    (run_dir / "chapters/0001/draft.txt").write_text(
        "The controller used calibration state。 <script>alert('{{bad}}')</script>",
        encoding="utf-8",
    )
    (run_dir / "chapters/0001/translated_final.txt").write_text(
        "The controller used calibration mode.", encoding="utf-8"
    )
    (run_dir / "delivery/book.txt").write_text("Chapter: 0001\n", encoding="utf-8")
    (run_dir / "delivery/book.epub").write_bytes(b"fixture")

    episode = {
        "steps": [
            {
                "sequence": 1,
                "action": {"tool": "delegate_review", "specialists": ["terminology", "fidelity"]},
                "observation": {
                    "ok": True,
                    "kind": "specialist_reviews_received",
                    "message": "Received bounded reviews.",
                    "data": {},
                },
            },
            {
                "sequence": 2,
                "action": {
                    "tool": "submit_patch",
                    "edits": [
                        {"old_text": "calibration state", "new_text": "calibration mode"}
                    ],
                },
                "observation": {
                    "ok": True,
                    "kind": "patch_accepted",
                    "message": "Patch accepted after deterministic QA verification.",
                    "data": {},
                },
            },
        ]
    }
    _write_json(run_dir / "chapters/0001/agent_episode.json", episode)
    snapshot = {
        "status": status,
        "specialist_reviews": [
            {
                "child_id": "0001-1-terminology",
                "role": "terminology",
                "status": "completed",
                "summary": "Choose one stable target for the technical term.",
                "findings": [
                    {
                        "category": "terminology",
                        "message": "The technical term is inconsistent in the draft.",
                        "source_excerpt": "采样周期",
                        "translation_excerpt": "the 采样周期",
                        "blocking": True,
                    }
                ],
                "proposed_edits": [
                    {"old_text": "采样周期", "new_text": "sampling cycle"}
                ],
                "term_suggestions": [
                    {
                        "term": "采样周期",
                        "target": "sampling cycle",
                        "rationale": "Keep the named technical term stable across chapters.",
                    }
                ],
                "steps": [{"action": {"tool": "read_paragraphs"}}],
                "provider_calls": [],
                "draft_sha256": "a" * 64,
            },
            {
                "child_id": "0001-1-fidelity",
                "role": "fidelity",
                "status": "completed",
                "summary": "The statement remains faithful after the bounded repair.",
                "findings": [],
                "proposed_edits": [
                    {"old_text": "calibration state", "new_text": "calibration mode"}
                ],
                "term_suggestions": [],
                "overlaps": ["Both specialists touched the opening mode name."],
                "steps": [{"action": {"tool": "read_paragraphs"}}],
                "provider_calls": [],
                "draft_sha256": "a" * 64,
            },
        ],
        "pending_proposal": (
            {
                "term": "采样周期",
                "proposed_target": "sampling cycle",
                "proposal_id": "term-0001",
                "decision": "pending",
                "rationale": "The term will be reused by the next chapter.",
            }
            if status == "awaiting_approval"
            else None
        ),
    }
    _write_json(run_dir / "chapters/0001/session_snapshot.json", snapshot)
    (run_dir / "chapters/0001/session_events.jsonl").write_text(
        '{"event_type":"model_requested","payload":{"tool_protocol":"native_function"}}\n',
        encoding="utf-8",
    )
    manifest = {
        "schema": "translation-showcase.v1",
        "run_id": "showcase-run",
        "title": "Synthetic <Control>",
        "slug": "synthetic_control_demo",
        "provider_mode": "offline",
        "execution_mode": "offline",
        "profile": {
            "provider": "fixture",
            "model": "showcase-scripted-v1",
            "tool_protocol": "native_function",
        },
        "strategy": "specialists",
        "status": status,
        "chapter_ids": ["0001"],
        "chapters": {
            "0001": {
                "status": status,
                "initial_findings": 2,
                "final_findings": 0,
                "steps": 2,
                "patch_attempts": 1,
                "accepted_patches": 1,
                "rejected_patches": 0,
                "child_calls": 2,
                "provider_calls": [
                    {
                        "role": "agent_action",
                        "namespace": "agent_action",
                        "provider": "fixture",
                        "model": "showcase-scripted-v1",
                        "cache_hit": False,
                        "elapsed_ms": 0.4,
                    }
                ],
                "elapsed_ms": 14.2,
            }
        },
        "approvals": [],
        "artifacts": {"txt": "delivery/book.txt", "epub": "delivery/book.epub"},
    }
    return run_dir, manifest


def test_showcase_report_renders_workflow_and_child_evidence(tmp_path: Path) -> None:
    run_dir, manifest = _run_fixture(tmp_path)

    context = build_showcase_report_context(run_dir, manifest)
    assert len(context["chapters"][0]["steps"]) == 2
    assert len(context["chapters"][0]["reviews"]) == 2
    assert context["total_initial"] == 2
    assert context["total_final"] == 0

    output = render_showcase_report(run_dir, manifest)
    rendered = output.read_text(encoding="utf-8")
    for phrase in (
        "Source → translation → review → approval → export",
        "Coordinator trace",
        "Specialist evidence",
        "Choose one stable target",
        "sampling cycle",
        "Overlapping evidence",
        "Captured from the master glossary at chapter start.",
        "delivery/book.epub",
        "snapshot",
        "episode",
        "events",
        "Unavailable — offline fixture; provider usage is not measured.",
    ):
        assert phrase in rendered


def test_showcase_report_joins_duplicate_projections_without_merging_children(
    tmp_path: Path,
) -> None:
    run_dir, manifest = _run_fixture(tmp_path)
    snapshot_path = run_dir / "chapters/0001/session_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot_reviews = [dict(review) for review in snapshot["specialist_reviews"]]
    for review in snapshot_reviews:
        # This mirrors the current persisted snapshot, where the Pydantic
        # SpecialistReview projection does not carry the child id.
        review.pop("child_id", None)
    snapshot["specialist_reviews"] = snapshot_reviews
    _write_json(snapshot_path, snapshot)

    # The second and third receipts intentionally contain identical evidence,
    # but their durable child ids identify separate specialist executions.
    for child_id, review in (
        ("0001-1-terminology", snapshot_reviews[0]),
        ("0001-1-fidelity", snapshot_reviews[1]),
        ("0001-2-fidelity", snapshot_reviews[1]),
    ):
        _write_json(
            run_dir / f"chapters/0001/children/{child_id}.json",
            {
                "schema_version": "specialist-review.v1",
                "child_id": child_id,
                "chapter": "0001",
                "step_number": 1,
                **review,
            },
        )

    context = build_showcase_report_context(run_dir, manifest)
    reviews = context["chapters"][0]["reviews"]
    assert len(reviews) == 3
    assert {review["child_id"] for review in reviews} == {
        "0001-1-terminology",
        "0001-1-fidelity",
        "0001-2-fidelity",
    }
    assert context["total_reviews"] == 3


def test_showcase_report_escapes_source_and_model_text(tmp_path: Path) -> None:
    run_dir, manifest = _run_fixture(tmp_path)

    render_showcase_report(run_dir, manifest)
    rendered = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "<script>" not in rendered
    assert "{{bad}}" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&#123;&#123;bad&#125;&#125;" in rendered
    assert "Synthetic &lt;Control&gt;" in rendered


def test_showcase_report_explains_pending_term_and_review_boundary(tmp_path: Path) -> None:
    run_dir, manifest = _run_fixture(tmp_path, status="awaiting_approval")

    render_showcase_report(run_dir, manifest)
    rendered = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "Approval boundary is still open" in rendered
    assert "<code>采样周期</code>" in rendered
    assert "sampling cycle" in rendered
    assert "Current stop: Chapter 0001 is" in rendered


def test_replay_report_separates_recorded_usage_from_current_latency(tmp_path: Path) -> None:
    run_dir, manifest = _run_fixture(tmp_path)
    manifest = {
        **manifest,
        "provider_mode": "offline",
        "execution_mode": "replay",
        "chapters": {
            "0001": {
                **manifest["chapters"]["0001"],
                "provider_calls": [
                    {
                        "namespace": "agent_action",
                        "provider": "openai",
                        "model": "gpt-replay",
                        "cache_hit": True,
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "total_tokens": 120,
                        "recorded_elapsed_ms": 240,
                        "current_elapsed_ms": 3,
                    }
                ],
            }
        },
    }
    render_showcase_report(run_dir, manifest)
    rendered = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "Recorded provider totals" in rendered
    assert "Recorded usage" in rendered
    assert "Current replay latency" in rendered
    assert "240 ms" in rendered
    assert "3 ms" in rendered
    assert "Recorded provider usage and current replay latency are shown separately." in rendered
