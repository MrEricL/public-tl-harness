"""Regression tests for persistence, interrupted runs, and approval states.

Scripted fakes only: no model calls. Each test fails on the code before its fix.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_translation.agent_models import FinishAction, SubmitPatchAction
from agentic_translation.agent_provider import SHOWCASE_TOOL_SCHEMA_VERSION
from agentic_translation.agent_session import SessionStore, resume_repair_session, run_repair_session
from agentic_translation.demo import run_folder
from agentic_translation.demo_provider import DemoGenerator, provider_config
from agentic_translation.glossary import parse_glossary_text
from agentic_translation.semantic_models import JevPolicy
from agentic_translation.semantic_provider import ReplayMissError, unavailable_report
from agentic_translation.workbench import activity
from agentic_translation.workbench_runs import JobBoard

SOURCE_TEXT = "第一章\n\n道心守住了山门。"
TRANSLATED_TEXT = "Chapter 1\n\nHeart of Dao guarded 道心."


class Provider:
    provider_name = "fixture-agent"
    model_name = "fixture-agent-v3"
    tool_protocol = "json_prompt"

    def __init__(self, actions):
        self.actions, self.requests, self.call_records = list(actions), [], []

    def next_action(self, request):
        self.requests.append(request)
        return self.actions.pop(0) if self.actions else None


def _session_kwargs():
    return dict(source_text=SOURCE_TEXT, glossary=parse_glossary_text("道心: Dao Heart\n"), run_id="run",
                story_slug="demo", chapter="0001", provider_mode="replay")


# R1 and R3: a fatal provider error must leave a complete, consistent record for every assigned chapter.
class _Down:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kw):
        raise RuntimeError("unavailable")


def test_fatal_error_writes_every_chapters_result_and_no_arm_stays_not_started(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    for i in (1, 2, 3):
        (source / f"00{i}.txt").write_text(f"林凡走進第{i}章。", encoding="utf-8")
    out = tmp_path / "run"
    gen = DemoGenerator(out, provider_config("openai", "test-model"), client=_Down())
    report = run_folder(source, out, gen.config, generator=gen)
    statuses = [c["status"] for c in report["chapters"]]
    assert statuses == ["failed", "not_started", "not_started"]
    for cid in ("0001", "0002", "0003"):                      # R1: every chapter has its result on disk
        assert (out / "chapters" / cid / "result.json").is_file()
    failed = report["chapters"][0]                            # R3: arms of a failed chapter don't say not_started
    assert all(arm["status"] != "not_started" for arm in failed["arms"].values())


# R2: a run whose process died mid-call must not read back as recorded.
def test_activity_reports_an_unfinished_run_as_interrupted(tmp_path):
    def run_dir(name, kinds, results=False):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "config.json").write_text("{}", encoding="utf-8")
        (folder / "activity.jsonl").write_text("".join(json.dumps({"kind": k}) + "\n" for k in kinds), encoding="utf-8")
        if results:
            (folder / "results.json").write_text("{}", encoding="utf-8")
        return name

    board = JobBoard(tmp_path)
    died = run_dir("died", ["run_started", "chapter_started", "call_started"])
    done = run_dir("done", ["run_started", "chapter_started", "run_finished"], results=True)
    failed = run_dir("failed", ["run_started", "run_failed"])
    assert activity(tmp_path, board, died, 0)["status"] == "interrupted"
    assert activity(tmp_path, board, done, 0)["status"] == "recorded"
    assert activity(tmp_path, board, failed, 0)["status"] == "recorded"


# S1: a failing semantic check after an accepted patch must not leave the resumable state behind the log.
class _FirstCallOnlyJudge:
    def __init__(self):
        self.calls = 0

    def evaluate(self, snapshot, question_set):
        self.calls += 1
        if self.calls == 1:
            policy = JevPolicy(mode="advisory", schedule="after_edit", question_set=question_set)
            return unavailable_report(snapshot, policy, question_set, "fixture: first call ok")
        raise ReplayMissError("fixture: no replay record for the patched draft")


def test_accepted_patch_is_checkpointed_before_the_after_edit_check(tmp_path):
    patch = SubmitPatchAction(old_text="Heart of Dao guarded 道心.", new_text="Dao Heart guarded the mountain gate.",
                              rationale="fix residual Chinese")
    with pytest.raises(ReplayMissError):
        run_repair_session(provider=Provider([patch]), session_dir=tmp_path / "s", translated_text=TRANSLATED_TEXT,
                           max_steps=5, dynamic_tools=False, tool_schema_version=SHOWCASE_TOOL_SCHEMA_VERSION,
                           jev_policy=JevPolicy(mode="advisory", schedule="after_edit"),
                           judgment_provider=_FirstCallOnlyJudge(), **_session_kwargs())
    store = SessionStore(tmp_path / "s")
    logged = [e for e in store.read_events() if e.event_type == "tool_executed" and e.payload.get("tool") == "submit_patch"]
    assert logged and logged[-1].payload.get("kind") == "patch_accepted"
    snapshot, episode = store.load_snapshot(), store.load_episode()
    assert snapshot.executor.current_text != TRANSLATED_TEXT      # the accepted patch is in the durable state
    assert snapshot.patch_attempts == 1 and len(episode.steps) == 1


# S2: a decision given while no approval is pending must raise, not silently do nothing.
def test_decision_on_a_running_session_raises(tmp_path):
    provider = Provider([])                                     # returns None: the session stays running
    result = run_repair_session(provider=provider, session_dir=tmp_path / "s", translated_text=TRANSLATED_TEXT,
                                max_steps=5, dynamic_tools=False, **_session_kwargs())
    assert result.snapshot.status == "running"
    provider.actions.append(FinishAction(summary="done"))
    with pytest.raises(ValueError, match="no approval is pending"):
        resume_repair_session(session_dir=tmp_path / "s", provider=provider, decision="approved",
                              reviewer="tester", note="mistaken", **_session_kwargs())
    asked = len(provider.requests)                              # the raise happened before any provider call
    resume_repair_session(session_dir=tmp_path / "s", provider=provider, **_session_kwargs())
    assert len(provider.requests) > asked                       # a normal resume still advances the session
