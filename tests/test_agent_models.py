from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from agentic_translation.agent_models import (
    AgentAction,
    AgentEpisode,
    AgentObservation,
    AgentStep,
    EscalateAction,
    FinishAction,
    ResolveTerminologyAction,
    SubmitPatchAction,
)
from agentic_translation.models import ProviderCallRecord, QAReport


def _qa_payload() -> dict[str, object]:
    return {
        "run_id": "run",
        "story_slug": "demo",
        "chapter": "0001",
        "findings": [],
        "summary": {},
    }


def test_action_union_parses_submit_patch() -> None:
    action = TypeAdapter(AgentAction).validate_python(
        {
            "tool": "submit_patch",
            "old_text": "Heart of Dao",
            "new_text": "Dao Heart",
            "rationale": "Use glossary canon.",
        }
    )

    assert isinstance(action, SubmitPatchAction)
    assert action.old_text == "Heart of Dao"


def test_action_union_parses_context_and_lookup_actions() -> None:
    adapter = TypeAdapter(AgentAction)

    source = adapter.validate_python(
        {"tool": "read_source_context", "finding_index": 0, "radius": 2}
    )
    translation = adapter.validate_python(
        {"tool": "read_translation_context", "finding_index": 1, "radius": 3}
    )
    glossary = adapter.validate_python({"tool": "lookup_glossary", "term": "道心"})

    assert source.tool == "read_source_context"
    assert translation.radius == 3
    assert glossary.term == "道心"


def test_action_union_parses_resolve_terminology_with_optional_finding() -> None:
    action = TypeAdapter(AgentAction).validate_python(
        {"tool": "resolve_terminology", "term": "道心", "finding_index": 0}
    )
    assert isinstance(action, ResolveTerminologyAction)
    assert action.finding_index == 0
    assert isinstance(
        TypeAdapter(AgentAction).validate_python({"tool": "resolve_terminology", "term": "道心"}),
        ResolveTerminologyAction,
    )


def test_resolve_terminology_action_is_bounded_and_strict() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(AgentAction).validate_python(
            {"tool": "resolve_terminology", "term": "x" * 201}
        )
    with pytest.raises(ValidationError):
        TypeAdapter(AgentAction).validate_python(
            {"tool": "resolve_terminology", "term": "道心", "finding_index": True}
        )


def test_action_union_parses_terminal_actions() -> None:
    adapter = TypeAdapter(AgentAction)

    assert adapter.validate_python({"tool": "get_qa_findings"}).tool == "get_qa_findings"
    assert isinstance(
        adapter.validate_python({"tool": "escalate", "reason": "Needs review."}),
        EscalateAction,
    )
    assert isinstance(
        adapter.validate_python({"tool": "finish", "summary": "Verified."}),
        FinishAction,
    )


def test_action_union_rejects_unknown_tool() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(AgentAction).validate_python(
            {"tool": "run_shell", "command": "rm -rf /"}
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "tool": "submit_patch",
            "old_text": "Heart of Dao",
            "new_text": "Dao Heart",
            "rationale": "Use glossary canon.",
            "unexpected": "not allowed",
        },
        {"tool": "read_source_context", "finding_index": True, "radius": 1},
        {"tool": "read_translation_context", "finding_index": 0, "radius": "1"},
        {"tool": "read_translation_context", "finding_index": "0", "radius": 1},
    ],
)
def test_actions_reject_extra_keys_and_coerced_types(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(AgentAction).validate_python(payload)


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"tool": "read_source_context", "finding_index": -1, "radius": 0}, "finding_index"),
        ({"tool": "read_source_context", "finding_index": 0, "radius": -1}, "radius"),
        ({"tool": "lookup_glossary", "term": ""}, "term"),
        (
            {
                "tool": "submit_patch",
                "old_text": "",
                "new_text": "Dao Heart",
                "rationale": "Use glossary canon.",
            },
            "old_text",
        ),
        (
            {
                "tool": "submit_patch",
                "old_text": "Heart of Dao",
                "new_text": "",
                "rationale": "Use glossary canon.",
            },
            "new_text",
        ),
        (
            {
                "tool": "submit_patch",
                "old_text": "Heart of Dao",
                "new_text": "Dao Heart",
                "rationale": "",
            },
            "rationale",
        ),
        ({"tool": "escalate", "reason": ""}, "reason"),
        ({"tool": "finish", "summary": ""}, "summary"),
    ],
)
def test_action_union_rejects_invalid_field_values(payload: dict[str, object], field: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        TypeAdapter(AgentAction).validate_python(payload)

    assert field in str(exc_info.value)


def test_observation_and_step_capture_observable_result() -> None:
    observation = AgentObservation(
        ok=True,
        kind="qa_findings",
        message="No open findings.",
        data={"count": 0},
    )

    step = AgentStep(
        sequence=1,
        action={"tool": "get_qa_findings"},
        observation=observation,
    )

    assert step.observation.data == {"count": 0}


@pytest.mark.parametrize("sequence", [0, -1])
def test_agent_step_sequence_must_be_positive(sequence: int) -> None:
    with pytest.raises(ValidationError):
        AgentStep(
            sequence=sequence,
            action={"tool": "get_qa_findings"},
            observation=AgentObservation(ok=True, kind="qa_findings", message="Done."),
        )


def test_episode_round_trips_observable_steps_and_evidence() -> None:
    qa = QAReport.model_validate(_qa_payload())
    provider_call = ProviderCallRecord(
        role="agent",
        namespace="agent_action",
        provider="openai",
        model="fixture-model",
        payload_sha256="payload",
        response_sha256="response",
        cache_file="cache.json",
        cache_hit=True,
    )
    episode = AgentEpisode(
        episode_id="episode",
        run_id="run",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        provider="openai",
        model="fixture-model",
        initial_qa=qa,
        final_qa=qa,
        steps=[
            AgentStep(
                sequence=1,
                action={"tool": "get_qa_findings"},
                observation=AgentObservation(
                    ok=True,
                    kind="qa_findings",
                    message="No open findings.",
                    data={"count": 0},
                ),
                provider_call=provider_call,
                qa_before=qa,
                qa_after=qa,
            )
        ],
        final_status="verified",
        summary="Verified.",
    )

    restored = AgentEpisode.model_validate_json(episode.model_dump_json())

    assert restored == episode
    assert restored.steps[0].provider_call == provider_call
    assert restored.steps[0].qa_after == qa


def test_episode_backward_parsing_defaults_terminology_fields() -> None:
    qa = QAReport.model_validate(_qa_payload())
    legacy = {
        "episode_id": "episode",
        "run_id": "run",
        "story_slug": "demo",
        "chapter": "0001",
        "provider_mode": "replay",
        "provider": "openai",
        "model": "fixture-model",
        "initial_qa": qa.model_dump(mode="json"),
    }
    restored = AgentEpisode.model_validate(legacy)
    assert restored.terminology_resolutions == []
    assert restored.steps == []


def test_episode_status_is_bounded() -> None:
    with pytest.raises(ValidationError):
        AgentEpisode(
            episode_id="episode",
            run_id="run",
            story_slug="demo",
            chapter="0001",
            provider_mode="replay",
            provider="openai",
            model="fixture-model",
            max_steps=5,
            max_patch_attempts=2,
            initial_qa=_qa_payload(),
            final_status="made_up",
        )
