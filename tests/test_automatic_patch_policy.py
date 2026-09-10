from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agentic_translation.agent_models import (
    DelegateReviewAction,
    FinishAction,
    ReviewFinding,
    SpecialistReview,
    SubmitPatchAction,
)
from agentic_translation.agent_provider import (
    AgentActionRequest,
    REGISTRY_TOOL_SCHEMA_VERSION,
    SHOWCASE_TOOL_SCHEMA_VERSION,
)
from agentic_translation.agent_repair import RepairToolExecutor
from agentic_translation.agent_session import (
    SessionIdentityMismatchError,
    resume_repair_session,
    run_repair_session,
)
from agentic_translation.glossary import parse_glossary_text


SOURCE = "第一章：技术检查\n\n控制器先打开阀门，然后记录三个读数。"
DRAFT = "Chapter 1: Technical Check\n\nThe controller records three readings before opening the valve."
CORRECTED = "Chapter 1: Technical Check\n\nThe controller opens the valve before recording three readings."
GLOSSARY = parse_glossary_text("")


class ScriptedProvider:
    provider_name = "fixture-agent"
    model_name = "fixture-v4"
    tool_protocol = "json_prompt"

    def __init__(self, actions: list[object]) -> None:
        self.actions = list(actions)
        self.requests: list[AgentActionRequest] = []
        self.call_records: list[object] = []

    def next_action(self, request: AgentActionRequest):
        self.requests.append(request.model_copy(deep=True))
        if not self.actions:
            raise RuntimeError("fixture has no action")
        return self.actions.pop(0)


def _executor(*, allow_nonregressing_patches: bool = False) -> RepairToolExecutor:
    return RepairToolExecutor(
        source_text=SOURCE,
        translated_text=DRAFT,
        glossary=GLOSSARY,
        run_id="automatic-patch-policy",
        story_slug="demo",
        chapter="0001",
        allow_nonregressing_patches=allow_nonregressing_patches,
    )


def _meaning_patch() -> SubmitPatchAction:
    return SubmitPatchAction(
        old_text="The controller records three readings before opening the valve.",
        new_text="The controller opens the valve before recording three readings.",
        rationale="Correct the operation order after checking the source.",
    )


def _review(
    translated_text: str,
    *,
    blocking: bool = False,
) -> SpecialistReview:
    findings = (
        [
            ReviewFinding(
                category="fidelity",
                message="The operation order still conflicts with the source.",
                source_excerpt="控制器先打开阀门，然后记录三个读数。",
                translation_excerpt="The controller records three readings before opening the valve.",
                blocking=True,
            )
        ]
        if blocking
        else []
    )
    return SpecialistReview(
        role="fidelity",
        status="completed",
        summary="Fidelity review complete.",
        findings=findings,
        draft_sha256=hashlib.sha256(translated_text.encode("utf-8")).hexdigest(),
    )


def _run_showcase(
    session_dir: Path,
    provider: ScriptedProvider,
    *,
    review_handler=None,
    max_steps: int = 5,
):
    return run_repair_session(
        provider=provider,
        session_dir=session_dir,
        source_text=SOURCE,
        translated_text=DRAFT,
        glossary=GLOSSARY,
        run_id="automatic-patch-policy",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        max_steps=max_steps,
        max_patch_attempts=2,
        dynamic_tools=False,
        tool_schema_version=SHOWCASE_TOOL_SCHEMA_VERSION,
        review_handler=review_handler,
        require_fidelity_review=True,
        allow_nonregressing_patches=True,
    )


def test_neutral_qa_patch_is_legacy_rejected_but_opt_in_accepted() -> None:
    legacy = _executor()
    legacy_result = legacy.execute(_meaning_patch())

    assert legacy.current_qa.score == 100
    assert legacy_result.observation.kind == "patch_rejected"
    assert legacy.current_text == DRAFT

    opted_in = _executor(allow_nonregressing_patches=True)
    opted_in_result = opted_in.execute(_meaning_patch())

    assert opted_in_result.observation.kind == "patch_accepted"
    assert opted_in_result.qa_before is not None
    assert opted_in_result.qa_after is not None
    assert opted_in_result.qa_before.score == opted_in_result.qa_after.score == 100
    assert opted_in.current_text == CORRECTED
    assert opted_in_result.observation.data["acceptance_basis"] == "qa_nonregression"
    assert opted_in_result.observation.data["source_review_required"] is True


def test_opt_in_still_rejects_noop_and_qa_regression() -> None:
    no_op = _executor(allow_nonregressing_patches=True)
    no_op_result = no_op.execute(
        SubmitPatchAction(
            old_text="The controller records three readings before opening the valve.",
            new_text="The controller records three readings before opening the valve.",
            rationale="No effective change.",
        )
    )
    assert no_op_result.observation.kind == "patch_rejected"
    assert no_op.current_text == DRAFT

    regression = _executor(allow_nonregressing_patches=True)
    regression_result = regression.execute(
        SubmitPatchAction(
            old_text="the valve",
            new_text="阀门",
            rationale="Introduce a deterministic regression.",
        )
    )
    assert regression_result.observation.kind == "patch_rejected"
    assert regression_result.observation.data["new_finding_identities"]
    assert regression.current_text == DRAFT


def test_neutral_patch_invalidates_old_review_and_fresh_clean_review_allows_finish(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider(
        [
            DelegateReviewAction(specialists=["fidelity"], objective="Review the initial draft."),
            _meaning_patch(),
            FinishAction(summary="Try the old receipt."),
            DelegateReviewAction(specialists=["fidelity"], objective="Review the corrected draft."),
            FinishAction(summary="Finish with current evidence."),
        ]
    )

    result = _run_showcase(
        tmp_path / "fresh-review",
        provider,
        review_handler=lambda action, **kwargs: [
            _review(str(kwargs["translated_text"]))
        ],
    )

    assert [step.observation.kind for step in result.episode.steps] == [
        "specialist_reviews_received",
        "patch_accepted",
        "fidelity_review_required",
        "specialist_reviews_received",
        "finished",
    ]
    assert result.snapshot.status == "completed"
    assert result.episode.final_status == "verified"
    assert result.final_text == CORRECTED
    assert len({review.draft_sha256 for review in result.snapshot.specialist_reviews}) == 2


def test_current_blocking_fidelity_finding_still_blocks_finish(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [
            DelegateReviewAction(specialists=["fidelity"], objective="Review fidelity."),
            FinishAction(summary="Attempt completion."),
            None,
        ]
    )
    result = _run_showcase(
        tmp_path / "blocking-review",
        provider,
        review_handler=lambda action, **kwargs: [
            _review(str(kwargs["translated_text"]), blocking=True)
        ],
        max_steps=3,
    )

    assert result.snapshot.status == "running"
    assert result.episode.steps[1].observation.kind == "fidelity_findings_blocking"


def test_policy_survives_resume_and_explicit_switch_is_rejected(tmp_path: Path) -> None:
    paused = _run_showcase(
        tmp_path / "resume",
        ScriptedProvider([None]),
        max_steps=2,
    )
    assert paused.snapshot.status == "running"
    assert paused.snapshot.allow_nonregressing_patches is True

    resumed = resume_repair_session(
        session_dir=paused.session_dir,
        provider=ScriptedProvider([_meaning_patch(), None]),
        source_text=SOURCE,
        glossary=GLOSSARY,
        tool_schema_version=SHOWCASE_TOOL_SCHEMA_VERSION,
        require_fidelity_review=True,
    )
    assert resumed.episode.steps[0].observation.kind == "patch_accepted"
    assert resumed.final_text == CORRECTED
    assert resumed.snapshot.allow_nonregressing_patches is True

    with pytest.raises(
        SessionIdentityMismatchError,
        match="nonregressing-patch policy mismatch",
    ):
        resume_repair_session(
            session_dir=resumed.session_dir,
            provider=ScriptedProvider([None]),
            source_text=SOURCE,
            glossary=GLOSSARY,
            tool_schema_version=SHOWCASE_TOOL_SCHEMA_VERSION,
            allow_nonregressing_patches=False,
        )


@pytest.mark.parametrize(
    ("tool_schema_version", "require_fidelity_review", "message"),
    [
        (
            SHOWCASE_TOOL_SCHEMA_VERSION,
            False,
            "requires require_fidelity_review=True",
        ),
        (
            REGISTRY_TOOL_SCHEMA_VERSION,
            True,
            "supported only for showcase sessions",
        ),
    ],
)
def test_session_rejects_unsafe_or_unsupported_opt_in(
    tmp_path: Path,
    tool_schema_version: str,
    require_fidelity_review: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_repair_session(
            provider=ScriptedProvider([None]),
            session_dir=tmp_path / tool_schema_version,
            source_text=SOURCE,
            translated_text=DRAFT,
            glossary=GLOSSARY,
            run_id="invalid-policy",
            story_slug="demo",
            chapter="0001",
            provider_mode="replay",
            tool_schema_version=tool_schema_version,
            require_fidelity_review=require_fidelity_review,
            allow_nonregressing_patches=True,
        )
