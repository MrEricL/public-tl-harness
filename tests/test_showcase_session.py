from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agentic_translation.agent_models import (
    CompleteReviewAction,
    DelegateReviewAction,
    FinishAction,
    GetQAFindingsAction,
    AgentObservation,
    ReadParagraphsAction,
    SelectTermAction,
    SpecialistReview,
    TermSuggestion,
)
from agentic_translation.agent_provider import (
    AgentActionRequest,
    PriorObservableStep,
    REGISTRY_TOOL_SCHEMA_VERSION,
    SHOWCASE_TOOL_SCHEMA_VERSION,
    build_agent_action_messages,
    build_native_agent_action_messages,
)
from agentic_translation.agent_session import resume_repair_session, run_repair_session
from agentic_translation.agent_tools import (
    AGENT_TOOL_REGISTRY,
    SHOWCASE_TOOL_REGISTRY,
    SHOWCASE_CHILD_TOOL_REGISTRY,
    ToolCall,
    ToolCallValidationError,
)
from agentic_translation.glossary import parse_glossary_text
from agentic_translation.models import ProviderCallRecord


SOURCE = "第一章：技术检查\n\n校准模式保持阀门关闭。"
DRAFT = "Chapter 1: Technical Check\n\nCalibration mode keeps the valve closed."
GLOSSARY = parse_glossary_text("校准模式: calibration mode\n")


class ScriptedProvider:
    provider_name = "fixture-agent"
    model_name = "fixture-v4"
    tool_protocol = "json_prompt"

    def __init__(self, actions: list[object]) -> None:
        self.actions = list(actions)
        self.requests: list[AgentActionRequest] = []
        self.call_records: list[ProviderCallRecord] = []

    def next_action(self, request: AgentActionRequest):
        self.requests.append(request.model_copy(deep=True))
        if not self.actions:
            raise RuntimeError("fixture has no action")
        return self.actions.pop(0)


def _review(role: str, translated_text: str = DRAFT) -> SpecialistReview:
    return SpecialistReview(
        role=role,
        status="completed",
        summary=f"{role} review complete.",
        term_suggestions=(
            [TermSuggestion(term="校准模式", target="calibration mode", rationale="Use the episode canon.")]
            if role == "terminology"
            else []
        ),
        steps=[{"child_artifact_path": "/tmp/child.json"}],
        provider_calls=[
            ProviderCallRecord(
                role="specialist",
                namespace="agent_action",
                provider="fixture",
                model="fixture-v4",
                payload_sha256="a" * 64,
                response_sha256="b" * 64,
                cache_file="/tmp/child-cache.json",
            )
        ],
        draft_sha256=hashlib.sha256(translated_text.encode("utf-8")).hexdigest(),
    )


def test_v3_default_contract_is_unchanged_and_v4_is_opt_in() -> None:
    request = AgentActionRequest(
        episode_id="episode",
        step_number=1,
        story_slug="demo",
        chapter="0001",
        remaining_steps=1,
        remaining_patch_attempts=1,
        tool_schema_version=REGISTRY_TOOL_SCHEMA_VERSION,
        exposed_tool_names=("finish",),
        instruction_context={"style": "must not enter v3"},
    )
    payload = request.canonical_payload()
    assert payload["request_schema_version"] == "agent-action-request.v2"
    assert "instruction_context" not in payload
    assert len(AGENT_TOOL_REGISTRY.visible_specs()) == 11

    showcase = request.model_copy(
        update={
            "tool_schema_version": SHOWCASE_TOOL_SCHEMA_VERSION,
            "exposed_tool_names": ("finish", "read_paragraphs"),
        }
    )
    showcase_payload = showcase.canonical_payload()
    assert showcase_payload["request_schema_version"] == "agent-action-request.v3"
    assert showcase_payload["instruction_context"]["style"] == "must not enter v3"
    assert len(SHOWCASE_TOOL_REGISTRY.visible_specs()) == 14

    assert isinstance(
        SHOWCASE_CHILD_TOOL_REGISTRY.action_from_call(
            ToolCall.from_json_action(CompleteReviewAction(summary="child")),
            visible_names=["complete_review"],
        ),
        CompleteReviewAction,
    )
    with pytest.raises(ToolCallValidationError, match="Unknown tool"):
        SHOWCASE_TOOL_REGISTRY.action_from_call(
            ToolCall.from_json_action(CompleteReviewAction(summary="child")),
            visible_names=["complete_review"],
        )


def test_v4_messages_retain_nested_read_and_review_evidence_in_user_context() -> None:
    source_sentinel = "SOURCE_PARAGRAPH_SENTINEL"
    draft_sentinel = "DRAFT_EXCERPT_SENTINEL"
    target_sentinel = "TARGET_TERM_SENTINEL"
    request = AgentActionRequest(
        episode_id="episode",
        step_number=3,
        story_slug="demo",
        chapter="0001",
        remaining_steps=2,
        remaining_patch_attempts=1,
        tool_schema_version=SHOWCASE_TOOL_SCHEMA_VERSION,
        exposed_tool_names=("read_paragraphs", "delegate_review", "finish"),
        instruction_context={"instructions": "Use the trusted operator brief."},
        prior_steps=[
            PriorObservableStep(
                sequence=1,
                action=ReadParagraphsAction(document="source", start=0, count=1),
                observation=AgentObservation(
                    ok=True,
                    kind="paragraphs_read",
                    message="source evidence",
                    data={"paragraphs": [{"index": 0, "text": source_sentinel}]},
                ),
            ),
            PriorObservableStep(
                sequence=2,
                action=DelegateReviewAction(
                    specialists=["terminology"],
                    objective="Check the selected term.",
                ),
                observation=AgentObservation(
                    ok=True,
                    kind="specialist_reviews_received",
                    message="review evidence",
                    data={
                        "reviews": [
                            {
                                "role": "terminology",
                                "status": "completed",
                                "summary": "The term needs a stable target.",
                                "findings": [
                                    {
                                        "category": "terminology",
                                        "message": "Use the reviewed target.",
                                        "source_excerpt": source_sentinel,
                                        "translation_excerpt": draft_sentinel,
                                        "blocking": False,
                                    }
                                ],
                                "proposed_edits": [
                                    {"old_text": "old", "new_text": "new"}
                                ],
                                "term_suggestions": [
                                    {
                                        "term": "校准模式",
                                        "target": target_sentinel,
                                        "rationale": "Keep the episode term consistent.",
                                    }
                                ],
                                "draft_sha256": "a" * 64,
                            }
                        ]
                    },
                ),
            ),
        ],
    )
    payload = request.canonical_payload()
    for messages in (
        build_agent_action_messages(payload),
        build_native_agent_action_messages(payload),
    ):
        system = messages[0]["content"]
        user = messages[1]["content"]
        assert source_sentinel in user
        assert draft_sentinel in user
        assert target_sentinel in user
        assert source_sentinel not in system
        assert draft_sentinel not in system
        assert target_sentinel not in system


def test_v4_oversized_context_drops_oldest_prior_step() -> None:
    prior_steps = []
    for index in range(3):
        marker = f"PRIOR_{index}_SENTINEL"
        prior_steps.append(
            PriorObservableStep(
                sequence=index + 1,
                action=GetQAFindingsAction(),
                observation=AgentObservation(
                    ok=True,
                    kind="large_context",
                    message="large context",
                    data={
                        "evidence": [f"{marker}:{item}:" + "x" * 1200 for item in range(32)]
                    },
                ),
            )
        )
    request = AgentActionRequest(
        episode_id="episode",
        step_number=4,
        story_slug="demo",
        chapter="0001",
        remaining_steps=1,
        remaining_patch_attempts=1,
        tool_schema_version=SHOWCASE_TOOL_SCHEMA_VERSION,
        exposed_tool_names=("finish",),
        prior_steps=prior_steps,
    )
    payload = request.canonical_payload()
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    assert len(encoded) <= 60_000
    assert "PRIOR_0_SENTINEL" not in encoded
    assert "PRIOR_2_SENTINEL" in encoded


def test_showcase_delegation_read_select_and_approval_are_end_to_end(tmp_path: Path) -> None:
    canonical = tmp_path / "glossary.txt"
    canonical.write_text("校准模式 -> calibration state\n", encoding="utf-8")
    provider = ScriptedProvider(
        [
            ReadParagraphsAction(document="source", start=0, count=3),
            DelegateReviewAction(specialists=["terminology"], objective="Review terms."),
            SelectTermAction(term="校准模式", target="calibration mode", rationale="Use the review."),
            {"tool": "promote_glossary_term", "term": "校准模式", "rationale": "Persist canon."},
        ]
    )
    callback_calls: list[dict[str, object]] = []

    def review_handler(action: DelegateReviewAction, **kwargs: object) -> list[SpecialistReview]:
        callback_calls.append({"action": action, **kwargs})
        return [_review("terminology", str(kwargs["translated_text"]))]

    paused = run_repair_session(
        provider=provider,
        session_dir=tmp_path / "session",
        source_text=SOURCE,
        translated_text=DRAFT,
        glossary=GLOSSARY,
        canonical_glossary_path=canonical,
        run_id="run",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        max_steps=4,
        dynamic_tools=False,
        tool_schema_version=SHOWCASE_TOOL_SCHEMA_VERSION,
        instruction_context={"style": "faithful", "role": "coordinator"},
        review_handler=review_handler,
    )

    assert paused.snapshot.status == "awaiting_approval"
    assert [step.observation.kind for step in paused.episode.steps] == [
        "paragraphs_read",
        "specialist_reviews_received",
        "term_selected",
        "glossary_promotion_pending",
    ]
    assert paused.episode.steps[0].observation.data["paragraphs"][0]["index"] == 0
    visible = paused.episode.steps[1].observation.data["reviews"][0]
    assert set(visible) == {
        "role",
        "status",
        "summary",
        "findings",
        "proposed_edits",
        "term_suggestions",
        "draft_sha256",
    }
    assert "provider_calls" not in visible
    assert "child_artifact_path" not in visible
    assert callback_calls[0]["chapter"] == "0001"
    assert paused.snapshot.specialist_reviews[0].provider_calls[0].cache_file == "/tmp/child-cache.json"

    resumed = resume_repair_session(
        session_dir=paused.session_dir,
        provider=ScriptedProvider([FinishAction(summary="approved")]),
        source_text=SOURCE,
        glossary=GLOSSARY,
        canonical_glossary_path=canonical,
        decision="approved",
        reviewer="tester",
        note="approved",
        tool_schema_version=SHOWCASE_TOOL_SCHEMA_VERSION,
        instruction_context={"style": "faithful", "role": "coordinator"},
    )
    assert resumed.snapshot.status == "completed"
    assert canonical.read_text(encoding="utf-8") == "校准模式 -> calibration mode\n"


def test_fidelity_review_must_be_fresh_and_interruption_can_resume(tmp_path: Path) -> None:
    blocked_provider = ScriptedProvider([FinishAction(summary="try"), None])
    blocked = run_repair_session(
        provider=blocked_provider,
        session_dir=tmp_path / "blocked",
        source_text=SOURCE,
        translated_text=DRAFT,
        glossary=GLOSSARY,
        run_id="run",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        max_steps=2,
        dynamic_tools=False,
        tool_schema_version=SHOWCASE_TOOL_SCHEMA_VERSION,
        require_fidelity_review=True,
    )
    assert blocked.snapshot.status == "running"
    assert blocked.episode.steps[0].observation.kind == "fidelity_review_required"

    resumed = resume_repair_session(
        session_dir=blocked.session_dir,
        provider=ScriptedProvider([DelegateReviewAction(specialists=["fidelity"], objective="Check fidelity.")]),
        source_text=SOURCE,
        glossary=GLOSSARY,
        tool_schema_version=SHOWCASE_TOOL_SCHEMA_VERSION,
        review_handler=lambda action, **kwargs: [_review("fidelity", str(kwargs["translated_text"]))],
        require_fidelity_review=True,
        max_delegation_rounds=2,
    )
    assert resumed.snapshot.status == "completed"
    assert resumed.episode.final_status == "budget_exhausted"

    interrupted_provider = ScriptedProvider([None])
    interrupted = run_repair_session(
        provider=interrupted_provider,
        session_dir=tmp_path / "interrupted",
        source_text=SOURCE,
        translated_text=DRAFT,
        glossary=GLOSSARY,
        run_id="run-2",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        max_steps=2,
        dynamic_tools=False,
        tool_schema_version=SHOWCASE_TOOL_SCHEMA_VERSION,
    )
    assert interrupted.snapshot.status == "running"
    continued = resume_repair_session(
        session_dir=interrupted.session_dir,
        provider=ScriptedProvider([FinishAction(summary="continue")]),
        source_text=SOURCE,
        glossary=GLOSSARY,
        tool_schema_version=SHOWCASE_TOOL_SCHEMA_VERSION,
    )
    assert continued.snapshot.status == "completed"
    assert continued.episode.final_status == "verified"
