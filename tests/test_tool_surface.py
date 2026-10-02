"""Tests for agentic_translation.tool_surface: tool semantics, budgets, and
persistence. No live model/API calls -- everything runs against in-memory
fixtures through the real RepairToolExecutor/QA gate."""

from __future__ import annotations

import json

import pytest

from agentic_translation.tool_surface import (
    ToolSurface,
    ToolSurfaceError,
    read_action_log,
)


def make_task(tmp_path, *, max_tool_calls=40, max_edits=12):
    task_dir = tmp_path / "task"
    return ToolSurface.create_task(
        task_dir,
        task_id="demo/c1",
        work_id="demo",
        chapter_id="c1",
        segments=[
            {
                "segment_id": "s0001",
                "source_text": "你好世界",
                "draft_text": "Hello wrold.",
                "context_text": "TERM 你好 -> Hello",
                "issue_ids": ["typo"],
            },
            {
                "segment_id": "s0002",
                "source_text": "再見",
                "draft_text": "Goodbye.",
                "context_text": "",
                "issue_ids": [],
            },
        ],
        glossary=[{"source": "你好", "target": "Hello"}],
        max_tool_calls=max_tool_calls,
        max_edits=max_edits,
    )


def test_list_segments_reports_lengths_flags_and_sha(tmp_path):
    surface = make_task(tmp_path)
    result = surface.call("list_segments")
    assert result["ok"] is True
    rows = {row["segment_id"]: row for row in result["data"]["segments"]}
    assert rows["s0001"]["issue_ids"] == ["typo"]
    assert rows["s0001"]["draft_len"] == len("Hello wrold.")
    assert rows["s0002"]["issue_ids"] == []
    assert len(rows["s0001"]["sha256"]) == 64


def test_read_segment_returns_source_draft_context_flags_sha(tmp_path):
    surface = make_task(tmp_path)
    result = surface.call("read_segment", {"segment_id": "s0001"})
    assert result["ok"] is True
    data = result["data"]
    assert data["source_text"] == "你好世界"
    assert data["draft_text"] == "Hello wrold."
    assert data["context_text"] == "TERM 你好 -> Hello"
    assert data["issue_ids"] == ["typo"]
    assert len(data["sha256"]) == 64


def test_read_segment_unknown_id_is_actionable(tmp_path):
    surface = make_task(tmp_path)
    result = surface.call("read_segment", {"segment_id": "nope"})
    assert result["ok"] is False
    assert "list_segments" in result["message"]
    assert result["data"]["known_segment_ids"] == ["s0001", "s0002"]


def test_edit_segment_exact_once_match_succeeds(tmp_path):
    surface = make_task(tmp_path)
    sha = surface.call("read_segment", {"segment_id": "s0001"})["data"]["sha256"]
    result = surface.call("edit_segment", {
        "segment_id": "s0001", "expected_sha256": sha,
        "edits": [{"old_text": "wrold", "new_text": "world"}],
    })
    assert result["ok"] is True
    assert result["data"]["draft_text"] == "Hello world."
    # sha rotates after a successful edit
    assert result["data"]["sha256"] != sha


def test_edit_segment_zero_matches_is_actionable(tmp_path):
    surface = make_task(tmp_path)
    sha = surface.call("read_segment", {"segment_id": "s0001"})["data"]["sha256"]
    result = surface.call("edit_segment", {
        "segment_id": "s0001", "expected_sha256": sha,
        "edits": [{"old_text": "not present anywhere", "new_text": "x"}],
    })
    assert result["ok"] is False
    assert "matched 0 times in s0001" in result["message"]
    assert "read_segment" in result["message"]
    assert result["data"]["occurrences"] == 0


def test_edit_segment_ambiguous_multiple_matches_is_actionable(tmp_path):
    task_dir = tmp_path / "task"
    surface = ToolSurface.create_task(
        task_dir, task_id="t", work_id="w", chapter_id="c",
        segments=[{"segment_id": "s1", "source_text": "x", "draft_text": "a a a.", "context_text": "", "issue_ids": []}],
    )
    sha = surface.call("read_segment", {"segment_id": "s1"})["data"]["sha256"]
    result = surface.call("edit_segment", {
        "segment_id": "s1", "expected_sha256": sha,
        "edits": [{"old_text": "a", "new_text": "b"}],
    })
    assert result["ok"] is False
    assert "matched 3 times in s1" in result["message"]


def test_edit_segment_stale_sha_is_rejected_and_actionable(tmp_path):
    surface = make_task(tmp_path)
    sha = surface.call("read_segment", {"segment_id": "s0001"})["data"]["sha256"]
    first = surface.call("edit_segment", {
        "segment_id": "s0001", "expected_sha256": sha,
        "edits": [{"old_text": "wrold", "new_text": "world"}],
    })
    assert first["ok"] is True
    stale = surface.call("edit_segment", {
        "segment_id": "s0001", "expected_sha256": sha,
        "edits": [{"old_text": "Hello", "new_text": "Hi"}],
    })
    assert stale["ok"] is False
    assert "has changed since it was last read" in stale["message"]
    assert "read_segment" in stale["message"]
    assert stale["data"]["current_segment_sha256"] != sha


def test_edit_segment_content_loss_guard_rejects_large_shrink(tmp_path):
    # A second segment keeps old_text from equalling the *whole chapter* text
    # (which the executor rejects for a different reason -- see the
    # whole_chapter_target case below); this isolates the content-loss guard.
    task_dir = tmp_path / "task"
    surface = ToolSurface.create_task(
        task_dir, task_id="t", work_id="w", chapter_id="c",
        segments=[
            {"segment_id": "s1", "source_text": "x",
             "draft_text": "This is a reasonably long sentence with real content in it.",
             "context_text": "", "issue_ids": []},
            {"segment_id": "s2", "source_text": "y", "draft_text": "Untouched segment.",
             "context_text": "", "issue_ids": []},
        ],
    )
    sha = surface.call("read_segment", {"segment_id": "s1"})["data"]["sha256"]
    result = surface.call("edit_segment", {
        "segment_id": "s1", "expected_sha256": sha,
        "edits": [{"old_text": "This is a reasonably long sentence with real content in it.",
                   "new_text": "Short."}],
    })
    assert result["ok"] is False
    assert "shrank" in result["message"]


def test_edit_segment_whole_chapter_target_suggests_rewrite_segment(tmp_path):
    task_dir = tmp_path / "task"
    surface = ToolSurface.create_task(
        task_dir, task_id="t", work_id="w", chapter_id="c",
        segments=[{"segment_id": "s1", "source_text": "x", "draft_text": "Only segment.",
                   "context_text": "", "issue_ids": []}],
    )
    sha = surface.call("read_segment", {"segment_id": "s1"})["data"]["sha256"]
    result = surface.call("edit_segment", {
        "segment_id": "s1", "expected_sha256": sha,
        "edits": [{"old_text": "Only segment.", "new_text": "Different."}],
    })
    assert result["ok"] is False
    assert "rewrite_segment" in result["message"]


def test_rewrite_segment_whole_segment_replacement(tmp_path):
    surface = make_task(tmp_path)
    sha = surface.call("read_segment", {"segment_id": "s0002"})["data"]["sha256"]
    result = surface.call("rewrite_segment", {
        "segment_id": "s0002", "expected_sha256": sha, "text": "Farewell.",
    })
    assert result["ok"] is True
    assert result["data"]["draft_text"] == "Farewell."


def test_search_chapter_finds_source_and_draft_matches(tmp_path):
    surface = make_task(tmp_path)
    result = surface.call("search_chapter", {"query": "hello"})
    assert result["ok"] is True
    hits = result["data"]["matches"]
    assert any(h["segment_id"] == "s0001" and h["field"] == "draft" for h in hits)


def test_search_chapter_no_match_is_actionable(tmp_path):
    surface = make_task(tmp_path)
    result = surface.call("search_chapter", {"query": "zzz_not_there"})
    assert result["ok"] is True
    assert result["data"]["matches"] == []
    assert "No matches" in result["message"]


def test_lookup_glossary_hit_and_miss(tmp_path):
    surface = make_task(tmp_path)
    hit = surface.call("lookup_glossary", {"term": "你好"})
    assert hit["ok"] is True
    assert hit["data"]["target"] == "Hello"
    miss = surface.call("lookup_glossary", {"term": "unknown_term"})
    assert miss["ok"] is False
    assert "search_chapter" in miss["message"]


def test_done_requires_summary_and_finalizes_task(tmp_path):
    surface = make_task(tmp_path)
    empty_summary = surface.call("done", {})
    assert empty_summary["ok"] is False
    assert "non-empty" in empty_summary["message"] or "summary" in empty_summary["message"]
    result = surface.call("done", {"summary": "no changes needed"})
    assert result["ok"] is True
    after = surface.call("list_segments")
    assert after["ok"] is False
    assert "already finished" in after["message"]


def test_tool_call_budget_is_enforced(tmp_path):
    surface = make_task(tmp_path, max_tool_calls=2)
    surface.call("list_segments")
    surface.call("list_segments")
    result = surface.call("list_segments")
    assert result["ok"] is False
    assert "Tool-call budget exhausted" in result["message"]


def test_edit_budget_is_enforced_independent_of_read_calls(tmp_path):
    surface = make_task(tmp_path, max_edits=1)
    sha = surface.call("read_segment", {"segment_id": "s0001"})["data"]["sha256"]
    ok = surface.call("edit_segment", {
        "segment_id": "s0001", "expected_sha256": sha,
        "edits": [{"old_text": "wrold", "new_text": "world"}],
    })
    assert ok["ok"] is True
    sha2 = surface.call("read_segment", {"segment_id": "s0001"})["data"]["sha256"]
    exhausted = surface.call("edit_segment", {
        "segment_id": "s0001", "expected_sha256": sha2,
        "edits": [{"old_text": "Hello", "new_text": "Hi"}],
    })
    assert exhausted["ok"] is False
    assert "Edit budget exhausted" in exhausted["message"]
    # read-only tools still work after the edit budget is exhausted
    still_reads = surface.call("read_segment", {"segment_id": "s0001"})
    assert still_reads["ok"] is True


def test_action_log_is_append_only_with_parent_chain(tmp_path):
    surface = make_task(tmp_path)
    surface.call("list_segments")
    surface.call("read_segment", {"segment_id": "s0001"})
    log = read_action_log(surface.task_dir)
    assert [entry["tool"] for entry in log] == ["list_segments", "read_segment"]
    assert log[0]["parent_id"] is None
    assert log[1]["parent_id"] == log[0]["id"]
    assert all("elapsed_ms" in entry for entry in log)


def test_state_round_trips_through_disk(tmp_path):
    surface = make_task(tmp_path)
    sha = surface.call("read_segment", {"segment_id": "s0001"})["data"]["sha256"]
    surface.call("edit_segment", {
        "segment_id": "s0001", "expected_sha256": sha,
        "edits": [{"old_text": "wrold", "new_text": "world"}],
    })
    # A fresh ToolSurface over the same directory sees the persisted edit.
    reloaded = ToolSurface(surface.task_dir)
    result = reloaded.call("read_segment", {"segment_id": "s0001"})
    assert result["data"]["draft_text"] == "Hello world."


def test_unknown_tool_name_is_rejected(tmp_path):
    surface = make_task(tmp_path)
    result = surface.call("not_a_real_tool")
    assert result["ok"] is False
    assert "Unknown tool" in result["message"]


def test_create_task_rejects_duplicate_segment_ids(tmp_path):
    with pytest.raises(ToolSurfaceError):
        ToolSurface.create_task(
            tmp_path / "task", task_id="t", work_id="w", chapter_id="c",
            segments=[
                {"segment_id": "s1", "source_text": "a", "draft_text": "a", "context_text": "", "issue_ids": []},
                {"segment_id": "s1", "source_text": "b", "draft_text": "b", "context_text": "", "issue_ids": []},
            ],
        )
