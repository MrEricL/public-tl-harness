from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import pytest

from agentic_translation.agent_models import (
    CompleteReviewAction,
    DelegateReviewAction,
    LookupGlossaryAction,
    ReadParagraphsAction,
    ReviewFinding,
    SubmitPatchAction,
    TermSuggestion,
    TextEdit,
)
from agentic_translation.agent_tools import (
    SPECIALIST_TOOL_REGISTRY,
    ToolCall,
    ToolCallValidationError,
)
from agentic_translation.agent_provider import build_agent_action_messages
from agentic_translation.models import GlossaryEntry, GlossaryParseResult, ProviderCallRecord
from agentic_translation.specialists import SpecialistRunner


SOURCE = "Technical note\n\n第一段说明阀门保持关闭。\n\n第二段说明读数和操作顺序。"
DRAFT = "Technical note\n\nThe first paragraph says the valve remains closed.\n\nThe second paragraph explains the readings and operation order."
GLOSSARY = GlossaryParseResult(
    entries=[
        GlossaryEntry(
            source="阀门",
            target="valve",
            candidates=["valve", "control valve"],
        )
    ]
)


def _record(index: int) -> ProviderCallRecord:
    digest = hashlib.sha256(f"call-{index}".encode()).hexdigest()
    return ProviderCallRecord(
        role="specialist",
        namespace="agent_action",
        provider="fixture",
        model="fixture-v4",
        payload_sha256=digest,
        response_sha256=digest,
        cache_file=f"fixture-{index}.json",
        cache_hit=False,
    )


class ScriptedProvider:
    provider_name = "fixture"
    model_name = "fixture-v4"
    tool_protocol = "json_prompt"

    def __init__(self, actions: list[Any], *, delay: float = 0.0, error: Exception | None = None):
        self.actions = list(actions)
        self.delay = delay
        self.error = error
        self.requests = []
        self.call_records: list[ProviderCallRecord] = []

    def next_action(self, request):
        if self.delay:
            time.sleep(self.delay)
        self.requests.append(request.model_copy(deep=True))
        self.call_records.append(_record(len(self.call_records) + 1))
        if self.error is not None:
            raise self.error
        index = len(self.requests) - 1
        if index >= len(self.actions):
            raise RuntimeError("fixture has no action")
        return self.actions[index]


def _run(
    tmp_path: Path,
    providers: dict[str, ScriptedProvider],
    *,
    specialists: list[str] | None = None,
    max_steps: int = 4,
    objective: str = "Review the chapter and return evidence.",
):
    factory_calls: list[tuple[str, str]] = []

    def factory(role: str, child_id: str):
        factory_calls.append((role, child_id))
        return providers[role]

    runner = SpecialistRunner(
        factory,
        style_guide="Use concise technical English.",
        max_steps=max_steps,
        max_workers=8,
    )
    action = DelegateReviewAction(
        specialists=specialists or ["fidelity", "terminology"],
        objective=objective,
    )
    reviews = runner(
        action,
        source_text=SOURCE,
        translated_text=DRAFT,
        glossary=GLOSSARY,
        chapter="0001",
        session_dir=tmp_path,
        step_number=7,
    )
    return reviews, factory_calls


def test_specialist_registry_has_only_read_and_report_tools() -> None:
    names = [spec.name for spec in SPECIALIST_TOOL_REGISTRY.visible_specs()]
    assert names == ["complete_review", "lookup_glossary", "read_paragraphs"]
    assert all(spec.side_effect == "none" for spec in SPECIALIST_TOOL_REGISTRY.visible_specs())
    assert all(spec.requires_approval is False for spec in SPECIALIST_TOOL_REGISTRY.visible_specs())

    with pytest.raises(ToolCallValidationError, match="Unknown tool"):
        SPECIALIST_TOOL_REGISTRY.action_from_call(
            ToolCall.from_json_action(
                {
                    "tool": "submit_patch",
                    "edits": [{"old_text": "x", "new_text": "y"}],
                    "rationale": "write",
                }
            ),
            visible_names=names,
        )


def test_runner_returns_structured_results_in_requested_order_and_writes_artifacts(tmp_path: Path) -> None:
    terminology = ScriptedProvider(
        [
            LookupGlossaryAction(term="阀门"),
            CompleteReviewAction(
                summary="The canonical gate term is available.",
                term_suggestions=[
                    TermSuggestion(term="阀门", target="valve", rationale="Matches the glossary.")
                ],
            ),
        ]
    )
    fidelity = ScriptedProvider(
        [
            ReadParagraphsAction(document="translation", start=1, count=3),
            CompleteReviewAction(
                summary="The relationship wording deserves review.",
                findings=[
                    ReviewFinding(
                        category="fidelity",
                        message="Check the relationship clause against the source.",
                        source_excerpt="第二段说明读数和操作顺序。",
                        translation_excerpt="The second paragraph explains the readings and operation order.",
                    )
                ],
                proposed_edits=[
                    TextEdit(old_text="explains the readings", new_text="describes the readings")
                ],
            ),
        ],
        delay=0.03,
    )

    reviews, factory_calls = _run(
        tmp_path,
        {"terminology": terminology, "fidelity": fidelity},
    )

    # Fidelity finishes later, but the parent receives the requested order.
    assert [review.role for review in reviews] == ["fidelity", "terminology"]
    assert all(review.status == "completed" for review in reviews)
    assert reviews[0].findings[0].category == "fidelity"
    assert reviews[1].term_suggestions[0].target == "valve"
    assert all(len(review.steps) == 2 for review in reviews)
    assert all(len(review.provider_calls) == 2 for review in reviews)
    assert all(review.draft_sha256 == hashlib.sha256(DRAFT.encode()).hexdigest() for review in reviews)

    assert sorted(factory_calls) == [
        ("fidelity", "0001-7-fidelity"),
        ("terminology", "0001-7-terminology"),
    ]
    artifact_paths = sorted((tmp_path / "children").glob("*.json"))
    assert [path.name for path in artifact_paths] == [
        "0001-7-fidelity.json",
        "0001-7-terminology.json",
    ]
    artifact = json.loads((tmp_path / "children/0001-7-fidelity.json").read_text())
    assert artifact["child_id"] == "0001-7-fidelity"
    assert artifact["status"] == "completed"
    assert artifact["steps"][0]["action"]["tool"] == "read_paragraphs"

    for provider, role in ((fidelity, "fidelity"), (terminology, "terminology")):
        first = provider.requests[0]
        assert first.episode_id == f"0001-7-{role}"
        assert first.tool_schema_version == "agent-tools.v4"
        assert first.exposed_tool_names == (
            "complete_review",
            "lookup_glossary",
            "read_paragraphs",
        )
        assert first.instruction_context["role"] == role
        assert first.instruction_context["objective"] == "Review the chapter and return evidence."
        assert first.instruction_context["style_guide"] == "Use concise technical English."
        evidence = first.instruction_context["initial_evidence"]
        assert evidence["source_paragraph_count"] == 3
        assert evidence["translation_paragraph_count"] == 3
        assert evidence["glossary_entry_count"] == 1
        assert "source_excerpt" not in evidence
        assert "translation_excerpt" not in evidence


def test_source_and_draft_enter_only_as_read_observations(tmp_path: Path) -> None:
    """The trusted v4 brief contains inventory, while text stays in user observations."""

    provider = ScriptedProvider(
        [
            ReadParagraphsAction(document="translation", start=0, count=1),
            CompleteReviewAction(summary="The draft was inspected."),
        ]
    )
    source_sentinel = "SOURCE_SENTINEL"
    draft_sentinel = "DRAFT_SENTINEL"
    runner = SpecialistRunner(lambda role, child_id: provider)
    runner(
        DelegateReviewAction(specialists=["fidelity"], objective="Inspect the draft."),
        source_text=source_sentinel,
        translated_text=draft_sentinel,
        glossary=GlossaryParseResult(entries=[]),
        chapter="0001",
        session_dir=tmp_path,
        step_number=1,
    )

    first_payload = provider.requests[0].canonical_payload(registry=SPECIALIST_TOOL_REGISTRY)
    second_payload = provider.requests[1].canonical_payload(registry=SPECIALIST_TOOL_REGISTRY)
    first_messages = build_agent_action_messages(first_payload)
    second_messages = build_agent_action_messages(second_payload)

    assert source_sentinel not in first_messages[0]["content"]
    assert draft_sentinel not in first_messages[0]["content"]
    assert source_sentinel not in second_messages[0]["content"]
    assert draft_sentinel not in second_messages[0]["content"]
    read_observation = provider.requests[1].prior_steps[0].observation.data
    assert read_observation["paragraphs"][0]["text"] == draft_sentinel


def test_children_get_fresh_context_and_do_not_mutate_snapshots(tmp_path: Path) -> None:
    providers = {
        role: ScriptedProvider([CompleteReviewAction(summary=f"{role} done")])
        for role in ("fidelity", "terminology")
    }
    source_before = SOURCE
    draft_before = DRAFT
    glossary_before = GLOSSARY.model_dump(mode="json")

    reviews, _ = _run(tmp_path, providers)

    assert SOURCE == source_before
    assert DRAFT == draft_before
    assert GLOSSARY.model_dump(mode="json") == glossary_before
    assert providers["fidelity"].requests[0] is not providers["terminology"].requests[0]
    assert providers["fidelity"].requests[0].instruction_context["role"] != providers["terminology"].requests[0].instruction_context["role"]
    assert [review.status for review in reviews] == ["completed", "completed"]


def test_rejected_write_and_unknown_actions_consume_steps_before_completion(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [
            SubmitPatchAction(
                edits=[TextEdit(old_text="English", new_text="Changed")],
                rationale="attempted child write",
            ),
            '{"tool":"shell","command":"echo unsafe"}',
            CompleteReviewAction(summary="The child completed after rejected actions."),
        ]
    )
    reviews, _ = _run(tmp_path, {"fidelity": provider}, specialists=["fidelity"])

    review = reviews[0]
    assert review.status == "completed"
    assert len(review.steps) == 3
    assert review.steps[0]["action"]["tool"] == "submit_patch"
    assert review.steps[0]["observation"]["kind"] == "tool_rejected"
    assert review.steps[1]["action"]["tool"] == "shell"
    assert review.steps[1]["observation"]["kind"] == "tool_rejected"
    assert review.steps[2]["observation"]["kind"] == "review_completed"
    assert len(provider.requests) == 3


def test_invalid_structured_result_is_rejected_and_budget_is_bounded(tmp_path: Path) -> None:
    invalid = {
        "tool": "complete_review",
        "summary": "invalid finding category",
        "findings": [{"category": "unsupported", "message": "bad"}],
    }
    provider = ScriptedProvider([invalid, CompleteReviewAction(summary="valid result")])
    reviews, _ = _run(tmp_path, {"fidelity": provider}, specialists=["fidelity"], max_steps=2)

    review = reviews[0]
    assert review.status == "completed"
    assert review.steps[0]["observation"]["kind"] == "tool_rejected"
    assert review.steps[1]["observation"]["kind"] == "review_completed"


def test_provider_failure_and_step_exhaustion_are_failed_reviews(tmp_path: Path) -> None:
    provider_failure = ScriptedProvider([], error=RuntimeError("provider unavailable"))
    failed, _ = _run(tmp_path / "provider", {"fidelity": provider_failure}, specialists=["fidelity"])
    assert failed[0].status == "failed"
    assert "RuntimeError" in failed[0].summary
    assert failed[0].steps == []
    assert (tmp_path / "provider/children/0001-7-fidelity.json").is_file()

    exhausted_provider = ScriptedProvider(
        [ReadParagraphsAction(document="source", count=1), ReadParagraphsAction(document="source", count=1)]
    )
    exhausted, _ = _run(
        tmp_path / "exhausted",
        {"fidelity": exhausted_provider},
        specialists=["fidelity"],
        max_steps=2,
    )
    assert exhausted[0].status == "failed"
    assert "2-step budget" in exhausted[0].summary
    assert len(exhausted[0].steps) == 2
    assert all(step["observation"]["kind"] == "paragraphs_read" for step in exhausted[0].steps)


def test_paragraph_tool_keeps_excerpt_bounded(tmp_path: Path) -> None:
    long_source = "Title\n\n" + ("甲" * 6000) + "\n\n尾部"
    provider = ScriptedProvider(
        [
            ReadParagraphsAction(document="source", start=0, count=6),
            CompleteReviewAction(summary="bounded context"),
        ]
    )
    runner = SpecialistRunner(lambda role, child_id: provider)
    result = runner(
        DelegateReviewAction(specialists=["fidelity"], objective="Inspect safely."),
        source_text=long_source,
        translated_text=DRAFT,
        glossary=GLOSSARY,
        chapter="0001",
        session_dir=tmp_path,
        step_number=1,
    )[0]
    data = result.steps[0]["observation"]["data"]
    assert data["returned_count"] == 2
    assert len("\n\n".join(item["text"] for item in data["paragraphs"])) <= 4000
