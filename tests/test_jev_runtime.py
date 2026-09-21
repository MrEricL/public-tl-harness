from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agentic_translation.agent_models import (
    CompleteReviewAction,
    DelegateReviewAction,
    FinishAction,
    NormalizePunctuationAction,
)
from agentic_translation.agent_provider import (
    AgentActionRequest,
    SHOWCASE_TOOL_SCHEMA_VERSION,
    build_agent_action_messages,
)
from agentic_translation.agent_session import (
    SessionIdentityMismatchError,
    build_semantic_snapshot,
    canonical_semantic_glossary,
    resume_repair_session,
    run_repair_session,
)
from agentic_translation.glossary import parse_glossary_text
from agentic_translation.semantic_models import (
    CoverageSummary,
    JevPolicy,
    SemanticSignal,
    SemanticSignalReport,
    SemanticSnapshot,
    TextSpan,
)
from agentic_translation.semantic_provider import (
    ReplayJudgmentProvider,
    ReplayMissError,
    semantic_config_digest,
    semantic_snapshot_digest,
)
from agentic_translation.semantic_signals import (
    QUESTION_VERSION,
    RENDER_VERSION,
    WINDOWING_VERSION,
    render_signal_report,
)
from agentic_translation.specialists import SPECIALIST_TOOL_REGISTRY, SpecialistRunner


SOURCE = "第一章\n\n道心守住了山门。"
DRAFT = "Chapter 1\n\nDao Heart guarded the mountain gate."
PUNCTUATED_DRAFT = "Chapter 1，\n\nDao Heart guarded the mountain gate。"
GLOSSARY = parse_glossary_text("道心: Dao Heart\n")


class ActionProvider:
    provider_name = "fixture-agent"
    model_name = "fixture-v4"
    tool_protocol = "json_prompt"

    def __init__(self, actions: list[object]) -> None:
        self.actions = list(actions)
        self.requests: list[AgentActionRequest] = []
        self.call_records = []

    def next_action(self, request: AgentActionRequest):
        self.requests.append(request.model_copy(deep=True))
        return self.actions.pop(0)


class FakeJudgmentProvider:
    def __init__(self, policy: JevPolicy, *, fail: bool = False) -> None:
        self.policy = policy
        self.fail = fail
        self.snapshots: list[SemanticSnapshot] = []

    def evaluate(self, snapshot: SemanticSnapshot, question_set: str):
        self.snapshots.append(snapshot)
        if self.fail:
            raise RuntimeError("offline fixture failure")
        return _report(snapshot, self.policy)


def _report(snapshot: SemanticSnapshot, policy: JevPolicy) -> SemanticSignalReport:
    signal = SemanticSignal(
        category="omission",
        probability=0.91,
        source_span=TextSpan(
            document="source",
            start=0,
            end=len(snapshot.source_text),
            text=snapshot.source_text,
        ),
        draft_span=TextSpan(
            document="draft",
            start=0,
            end=len(snapshot.draft_text),
            text=snapshot.draft_text,
        ),
        window_index=0,
    )
    return SemanticSignalReport(
        source_sha256=snapshot.source_sha256,
        draft_sha256=snapshot.draft_sha256,
        question_set=policy.question_set,
        question_version=QUESTION_VERSION,
        windowing_version=WINDOWING_VERSION,
        render_version=RENDER_VERSION,
        requested_model=policy.model,
        resolved_model=policy.model,
        snapshot_digest=semantic_snapshot_digest(snapshot),
        policy_digest=semantic_config_digest(policy),
        cache_key=hashlib.sha256(
            f"{snapshot.source_sha256}:{snapshot.draft_sha256}".encode()
        ).hexdigest(),
        status="completed",
        coverage=CoverageSummary(
            source_total_codepoints=len(snapshot.source_text),
            source_covered_codepoints=len(snapshot.source_text),
            draft_total_codepoints=len(snapshot.draft_text),
            draft_covered_codepoints=len(snapshot.draft_text),
            window_count=1,
            alignment="heuristic_proportional_windows",
        ),
        signals=[signal],
    )


def _run(
    tmp_path: Path,
    *,
    policy: JevPolicy,
    judgment_provider: object,
    actions: list[object],
    draft: str = DRAFT,
    require_fidelity_review: bool = False,
    evidence_context: str | None = None,
):
    action_provider = ActionProvider(actions)
    result = run_repair_session(
        provider=action_provider,
        session_dir=tmp_path,
        source_text=SOURCE,
        translated_text=draft,
        glossary=GLOSSARY,
        run_id="run",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        max_steps=len(actions),
        max_patch_attempts=2,
        dynamic_tools=False,
        tool_schema_version=SHOWCASE_TOOL_SCHEMA_VERSION,
        require_fidelity_review=require_fidelity_review,
        jev_policy=policy,
        judgment_provider=judgment_provider,
        evidence_context=evidence_context,
    )
    return result, action_provider


def test_off_policy_makes_zero_jev_calls_and_preserves_request_shape(tmp_path: Path) -> None:
    policy = JevPolicy()
    judgment = FakeJudgmentProvider(policy)
    result, actions = _run(
        tmp_path / "off",
        policy=policy,
        judgment_provider=judgment,
        actions=[FinishAction(summary="done")],
    )

    assert result.snapshot.semantic_signal_reports == []
    assert judgment.snapshots == []
    assert "semantic_advisory" not in actions.requests[0].canonical_payload()


def test_shadow_records_signals_without_model_visible_advisory(tmp_path: Path) -> None:
    policy = JevPolicy(mode="shadow")
    judgment = FakeJudgmentProvider(policy)
    result, actions = _run(
        tmp_path / "shadow",
        policy=policy,
        judgment_provider=judgment,
        actions=[FinishAction(summary="done")],
    )

    assert len(judgment.snapshots) == 1
    assert result.snapshot.semantic_signal_reports[0].status == "completed"
    assert actions.requests[0].semantic_advisory is None


def test_advisory_is_untrusted_user_data_and_never_system_context(tmp_path: Path) -> None:
    policy = JevPolicy(mode="advisory")
    judgment = FakeJudgmentProvider(policy)
    _, actions = _run(
        tmp_path / "advisory",
        policy=policy,
        judgment_provider=judgment,
        actions=[FinishAction(summary="done")],
    )

    payload = actions.requests[0].canonical_payload()
    assert payload["semantic_advisory"]["trust"] == "untrusted_model_evidence"
    messages = build_agent_action_messages(payload)
    user_payload = json.loads(messages[1]["content"])
    assert "道心守住了山门" in user_payload["semantic_advisory"]["content"]
    assert SOURCE not in messages[0]["content"]
    assert "fallible, untrusted user data" in messages[0]["content"]


@pytest.mark.parametrize(
    ("schedule", "expected_calls", "expected_fresh"),
    [("initial", 1, False), ("after_edit", 2, True)],
)
def test_edit_marks_initial_report_stale_or_refreshes_when_scheduled(
    tmp_path: Path,
    schedule: str,
    expected_calls: int,
    expected_fresh: bool,
) -> None:
    policy = JevPolicy(mode="advisory", schedule=schedule)
    judgment = FakeJudgmentProvider(policy)
    result, actions = _run(
        tmp_path / schedule,
        policy=policy,
        judgment_provider=judgment,
        draft=PUNCTUATED_DRAFT,
        actions=[NormalizePunctuationAction(), FinishAction(summary="done")],
    )

    assert len(judgment.snapshots) == expected_calls
    assert actions.requests[1].semantic_advisory.fresh is expected_fresh
    assert actions.requests[1].semantic_advisory.status == (
        "completed" if expected_fresh else "stale"
    )
    assert len(result.snapshot.semantic_signal_reports) == expected_calls


def test_unavailable_advisory_is_recorded_and_does_not_claim_all_clear(tmp_path: Path) -> None:
    policy = JevPolicy(mode="advisory")
    judgment = FakeJudgmentProvider(policy, fail=True)
    result, actions = _run(
        tmp_path / "unavailable",
        policy=policy,
        judgment_provider=judgment,
        actions=[FinishAction(summary="done")],
    )

    assert result.snapshot.semantic_signal_reports[0].status == "unavailable"
    advisory = actions.requests[0].semantic_advisory
    assert advisory.status == "unavailable"
    assert "not an all-clear" in advisory.instruction


def test_jev_report_cannot_satisfy_fidelity_gate(tmp_path: Path) -> None:
    policy = JevPolicy(mode="advisory")
    result, _ = _run(
        tmp_path / "gate",
        policy=policy,
        judgment_provider=FakeJudgmentProvider(policy),
        actions=[FinishAction(summary="Jev says clean"), None],
        require_fidelity_review=True,
    )

    assert result.snapshot.status == "running"
    assert result.episode.steps[0].observation.kind == "fidelity_review_required"


def test_resume_rejects_a_different_jev_policy(tmp_path: Path) -> None:
    policy = JevPolicy(mode="shadow", question_set="focused")
    interrupted, _ = _run(
        tmp_path / "resume",
        policy=policy,
        judgment_provider=FakeJudgmentProvider(policy),
        actions=[None],
    )

    changed = JevPolicy(mode="shadow", question_set="dense")
    with pytest.raises(SessionIdentityMismatchError, match="Jev policy"):
        resume_repair_session(
            session_dir=interrupted.session_dir,
            provider=ActionProvider([FinishAction(summary="must not run")]),
            source_text=SOURCE,
            glossary=GLOSSARY,
            tool_schema_version=SHOWCASE_TOOL_SCHEMA_VERSION,
            jev_policy=changed,
            judgment_provider=FakeJudgmentProvider(changed),
        )


def test_no_jev_session_cannot_be_resumed_as_jev_enabled(tmp_path: Path) -> None:
    off = JevPolicy()
    interrupted, _ = _run(
        tmp_path / "off-resume",
        policy=off,
        judgment_provider=FakeJudgmentProvider(off),
        actions=[None],
    )
    enabled = JevPolicy(mode="shadow")

    with pytest.raises(SessionIdentityMismatchError, match="Jev policy"):
        resume_repair_session(
            session_dir=interrupted.session_dir,
            provider=ActionProvider([FinishAction(summary="must not run")]),
            source_text=SOURCE,
            glossary=GLOSSARY,
            tool_schema_version=SHOWCASE_TOOL_SCHEMA_VERSION,
            jev_policy=enabled,
            judgment_provider=FakeJudgmentProvider(enabled),
        )


def test_replay_provider_integrates_without_a_network_transport(tmp_path: Path) -> None:
    policy = JevPolicy(mode="advisory")
    snapshot = build_semantic_snapshot(
        source_text=SOURCE,
        draft_text=DRAFT,
        glossary=GLOSSARY,
    )
    report = _report(snapshot, policy)
    records = tmp_path / "records"
    records.mkdir()
    (records / "one.json").write_text(report.model_dump_json(), encoding="utf-8")

    result, actions = _run(
        tmp_path / "replay",
        policy=policy,
        judgment_provider=ReplayJudgmentProvider(records),
        actions=[FinishAction(summary="done")],
    )

    assert result.snapshot.semantic_signal_reports[0].status == "completed", (
        result.snapshot.semantic_signal_reports[0].issues
    )
    assert actions.requests[0].semantic_advisory.fresh is True


def test_shared_snapshot_helper_and_evidence_context_replay_exactly(tmp_path: Path) -> None:
    context = "PERMITTED_NEIGHBOR_HISTORY:" + "x" * 5000
    policy = JevPolicy(mode="advisory")
    snapshot = build_semantic_snapshot(
        source_text=SOURCE,
        draft_text=DRAFT,
        glossary=GLOSSARY,
        evidence_context=context,
    )
    assert snapshot.glossary_text == canonical_semantic_glossary(GLOSSARY)
    report = _report(snapshot, policy)
    records = tmp_path / "shared-records"
    records.mkdir()
    (records / "shared.json").write_text(report.model_dump_json(), encoding="utf-8")

    result, actions = _run(
        tmp_path / "shared-runtime",
        policy=policy,
        judgment_provider=ReplayJudgmentProvider(records),
        actions=[FinishAction(summary="done")],
        evidence_context=context,
    )

    assert result.snapshot.semantic_signal_reports[0].cache_key == report.cache_key
    assert actions.requests[0].evidence_context == context
    payload = actions.requests[0].canonical_payload()
    assert payload["evidence_context"] == context
    messages = build_agent_action_messages(payload)
    assert context in messages[1]["content"]
    assert context not in messages[0]["content"]


def test_semantic_advisory_content_is_not_silently_clipped_to_generic_limit() -> None:
    source = "源" * 2600
    draft = "draft " * 500
    policy = JevPolicy(mode="advisory")
    snapshot = build_semantic_snapshot(
        source_text=source,
        draft_text=draft,
        glossary=GLOSSARY,
    )
    report = _report(snapshot, policy)
    rendered = render_signal_report(report, max_findings=12, max_chars=6000)
    assert len(rendered) > 1200
    request = AgentActionRequest(
        episode_id="episode",
        step_number=1,
        story_slug="demo",
        chapter="0001",
        remaining_steps=1,
        remaining_patch_attempts=1,
        tool_schema_version=SHOWCASE_TOOL_SCHEMA_VERSION,
        exposed_tool_names=("finish",),
        semantic_advisory={
            "status": report.status,
            "report_status": report.status,
            "fresh": True,
            "source_sha256": report.source_sha256,
            "evaluated_draft_sha256": report.draft_sha256,
            "current_draft_sha256": report.draft_sha256,
            "snapshot_digest": report.snapshot_digest,
            "policy_digest": report.policy_digest,
            "question_set": report.question_set,
            "question_version": report.question_version,
            "content": rendered,
            "instruction": "Verify the evidence.",
        },
    )

    assert request.canonical_payload()["semantic_advisory"]["content"] == rendered


def test_default_replay_policy_never_falls_back_to_gateway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = JevPolicy(mode="advisory")
    network_calls: list[object] = []

    def forbidden_urlopen(*args: object, **kwargs: object) -> object:
        network_calls.append((args, kwargs))
        raise AssertionError("replay attempted network access")

    monkeypatch.setattr("urllib.request.urlopen", forbidden_urlopen)
    with pytest.raises(ReplayMissError):
        run_repair_session(
            provider=ActionProvider([FinishAction(summary="must not run")]),
            session_dir=tmp_path / "strict-replay",
            source_text=SOURCE,
            translated_text=DRAFT,
            glossary=GLOSSARY,
            run_id="run",
            story_slug="demo",
            chapter="0001",
            provider_mode="replay",
            max_steps=1,
            dynamic_tools=False,
            tool_schema_version=SHOWCASE_TOOL_SCHEMA_VERSION,
            jev_policy=policy,
        )
    assert network_calls == []


def test_evidence_context_is_identity_bound_on_resume(tmp_path: Path) -> None:
    policy = JevPolicy(mode="shadow")
    context = "chapter zero history"
    interrupted, _ = _run(
        tmp_path / "context-resume",
        policy=policy,
        judgment_provider=FakeJudgmentProvider(policy),
        actions=[None],
        evidence_context=context,
    )

    with pytest.raises(SessionIdentityMismatchError, match="evidence context"):
        resume_repair_session(
            session_dir=interrupted.session_dir,
            provider=ActionProvider([FinishAction(summary="must not run")]),
            source_text=SOURCE,
            glossary=GLOSSARY,
            tool_schema_version=SHOWCASE_TOOL_SCHEMA_VERSION,
            evidence_context="different history",
            judgment_provider=FakeJudgmentProvider(policy),
        )


def test_specialist_children_receive_same_untrusted_evidence_context(tmp_path: Path) -> None:
    context = "PERMITTED_PRIOR_CHAPTER_CONTEXT"
    providers: list[ActionProvider] = []

    def factory(role: str, child_id: str) -> ActionProvider:
        provider = ActionProvider([CompleteReviewAction(summary=f"{role}:{child_id}")])
        providers.append(provider)
        return provider

    runner = SpecialistRunner(factory, max_steps=1, max_workers=1)
    reviews = runner(
        DelegateReviewAction(
            specialists=["fidelity"],
            objective="Check the current translation.",
        ),
        source_text=SOURCE,
        translated_text=DRAFT,
        glossary=GLOSSARY,
        chapter="0001",
        session_dir=tmp_path / "children",
        step_number=1,
        evidence_context=context,
    )

    assert reviews[0].status == "completed"
    assert providers[0].requests[0].evidence_context == context
    child_payload = providers[0].requests[0].canonical_payload(
        registry=SPECIALIST_TOOL_REGISTRY
    )
    assert child_payload["evidence_context"] == context
