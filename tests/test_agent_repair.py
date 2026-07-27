from __future__ import annotations

import pytest

from agentic_translation.agent_models import (
    AgentEpisode,
    EscalateAction,
    FinishAction,
    GetQAFindingsAction,
    LookupGlossaryAction,
    ReadSourceContextAction,
    ReadTranslationContextAction,
    ResolveTerminologyAction,
    SubmitPatchAction,
)
from agentic_translation.agent_provider import (
    AgentActionRequest,
    AgentActionValidationError,
)
from agentic_translation.agent_repair import (
    RepairToolExecutor,
    _persist_episode,
    run_repair_episode,
)
from agentic_translation.models import ProviderCallRecord
from agentic_translation.providers_llm import LLMProviderUnavailable
from agentic_translation.glossary import parse_glossary_text
from agentic_translation.terminology import TerminologyResolutionError, TerminologyResolver
from agentic_translation.terminology_models import TerminologyEvaluation, TerminologyVote


SOURCE_TEXT = "第一章\n\n道心守住了山门。"
TRANSLATED_TEXT = "Chapter 1\n\nHeart of Dao guarded 道心."


@pytest.fixture
def glossary():
    return parse_glossary_text("道心: Dao Heart, Heart of Dao\n# block: Heart of Dao\n")


@pytest.fixture
def executor(glossary):
    return RepairToolExecutor(
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=glossary,
        run_id="test-run",
        story_slug="agentic-demo",
        chapter="0001",
    )


def test_get_qa_findings_returns_current_findings(executor):
    result = executor.execute(GetQAFindingsAction())

    assert result.observation.ok is True
    assert result.observation.kind == "qa_findings"
    assert result.observation.data["count"] == 3
    assert result.observation.data["findings"]


@pytest.mark.parametrize(
    "action_type",
    [ReadSourceContextAction, ReadTranslationContextAction],
)
def test_read_context_tools_return_bounded_finding_context(executor, action_type):
    result = executor.execute(action_type(finding_index=0, radius=1))

    assert result.observation.ok is True
    assert result.observation.kind in {"source_context", "translation_context"}
    assert result.observation.data["context"]
    assert result.observation.data["paragraph_index"] == 1


@pytest.mark.parametrize(
    "action",
    [
        ReadSourceContextAction(finding_index=99, radius=1),
        ReadTranslationContextAction(finding_index=99, radius=1),
    ],
)
def test_read_context_rejects_out_of_range_finding_safely(executor, action):
    result = executor.execute(action)

    assert result.observation.ok is False
    assert result.observation.kind == "context_rejected"


def test_read_context_does_not_expose_unbounded_long_paragraph(glossary):
    long_body = "Heart of Dao guarded 道心. " + ("filler " * 500)
    executor = RepairToolExecutor(
        source_text=SOURCE_TEXT,
        translated_text=f"Chapter 1\n\n{long_body}",
        glossary=glossary,
    )

    result = executor.execute(ReadTranslationContextAction(finding_index=0, radius=0))

    assert result.observation.ok is True
    context = result.observation.data["context"]
    assert len(context) <= 2000
    assert all(
        not isinstance(value, str) or len(value) <= 2000
        for value in result.observation.data.values()
    )


def test_lookup_glossary_returns_canonical_candidates_and_blocked_variants(executor):
    result = executor.execute(LookupGlossaryAction(term="道心"))

    assert result.observation.ok is True
    assert result.observation.kind == "glossary_lookup"
    assert result.observation.data["canonical"] == "Dao Heart"
    assert result.observation.data["candidates"] == ["Dao Heart", "Heart of Dao"]
    assert result.observation.data["blocked_variants"] == ["Heart of Dao"]


def test_lookup_glossary_reports_not_found(executor):
    result = executor.execute(LookupGlossaryAction(term="不存在"))

    assert result.observation.ok is False
    assert result.observation.kind == "glossary_not_found"


class _AgreementVoter:
    def __init__(self, identity: str, recommendation: str = "Dao Heart") -> None:
        self.voter_id = identity
        self.provider_name = identity
        self.model_name = f"{identity}-term"
        self.call_records = [
            ProviderCallRecord(
                role="terminology_vote",
                namespace="terminology_vote",
                provider=identity,
                model=self.model_name,
                payload_sha256=(identity + "p").ljust(64, "0")[:64],
                response_sha256=(identity + "r").ljust(64, "0")[:64],
                cache_file=f"{identity}.json",
                cache_hit=True,
            )
        ]
        self.recommendation = recommendation

    def vote(self, request):
        return TerminologyVote(
            voter_id=self.voter_id,
            provider=self.provider_name,
            model=self.model_name,
            source_term=request.source_term,
            recommendation=self.recommendation,
            confidence=0.9,
            provider_call=self.call_records[0],
        )


class _NoopEvaluator:
    provider_name = "openai"
    model_name = "openai-term"
    call_records = []

    def evaluate(self, request, candidates):
        raise AssertionError("agreement must not invoke evaluator")


def _agreeing_resolver() -> TerminologyResolver:
    return TerminologyResolver(
        voters=[_AgreementVoter("openai"), _AgreementVoter("deepseek")],
        evaluator=_NoopEvaluator(),
        confidence_threshold=0.65,
    )


def test_resolve_terminology_uses_episode_local_override_without_mutating_master(glossary):
    master_dump = glossary.model_dump(mode="json")
    executor = RepairToolExecutor(
        source_text="第一章\n\n道心守住了山门。",
        translated_text="Chapter 1\n\nThe term 道心 remains.",
        glossary=glossary,
        run_id="terminology",
        story_slug="demo",
        chapter="0001",
        terminology_resolver=_agreeing_resolver(),
    )
    result = executor.execute(ResolveTerminologyAction(term="道心", finding_index=0))
    assert result.observation.kind == "terminology_resolved"
    assert result.resolution is not None
    assert executor._find_glossary_entry("道心").target == "Dao Heart"
    assert glossary.model_dump(mode="json") == master_dump


def test_resolve_terminology_replaces_existing_target_preserving_candidates_and_blocked(glossary):
    executor = RepairToolExecutor(
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=glossary,
        terminology_resolver=_agreeing_resolver(),
    )
    result = executor.execute(ResolveTerminologyAction(term="道心"))
    assert result.observation.ok is True
    entry = executor._find_glossary_entry("道心")
    assert entry.target == "Dao Heart"
    assert "Heart of Dao" in entry.candidates
    assert entry.blocked_variants == ["Heart of Dao"]


def test_resolve_terminology_without_resolver_is_typed_unavailable(executor):
    result = executor.execute(ResolveTerminologyAction(term="道心"))
    assert result.observation.ok is False
    assert result.observation.kind == "terminology_consensus_unavailable"
    assert result.resolution is None


def test_resolve_terminology_failure_preserves_partial_provider_calls(glossary):
    call = ProviderCallRecord(
        role="terminology_vote",
        namespace="terminology_vote",
        provider="openai",
        model="oa-term",
        payload_sha256="a" * 64,
        response_sha256="b" * 64,
        cache_file="vote.json",
        cache_hit=True,
    )

    class FailingResolver:
        def resolve(self, request):
            raise TerminologyResolutionError("vote failed", call_records=[call])

    executor = RepairToolExecutor(
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=glossary,
        terminology_resolver=FailingResolver(),
    )
    result = executor.execute(ResolveTerminologyAction(term="道心"))
    assert result.observation.kind == "terminology_resolution_failed"
    assert result.auxiliary_provider_calls == [call]


def test_submit_patch_rejects_missing_target_without_mutating_text(executor):
    result = executor.execute(
        SubmitPatchAction(
            old_text="text that is not present",
            new_text="Dao Heart guarded the gate.",
            rationale="first bad attempt",
        )
    )

    assert result.observation.ok is False
    assert result.observation.kind == "patch_rejected"
    assert executor.current_text == TRANSLATED_TEXT


def test_submit_patch_rejects_duplicate_target(executor):
    executor.current_text = "Chapter 1\n\nbad bad"

    result = executor.execute(
        SubmitPatchAction(old_text="bad", new_text="good", rationale="ambiguous target")
    )

    assert result.observation.ok is False
    assert result.observation.kind == "patch_rejected"
    assert "exactly once" in result.observation.message
    assert executor.current_text == "Chapter 1\n\nbad bad"


def test_submit_patch_rejects_non_improving_or_regressing_patch_without_mutation(executor):
    before_text = executor.current_text
    result = executor.execute(
        SubmitPatchAction(
            old_text="Heart of Dao",
            new_text="Heart of Dao",
            rationale="no-op",
        )
    )

    assert result.observation.ok is False
    assert result.observation.kind == "patch_rejected"
    assert result.qa_before is not None
    assert result.qa_after is not None
    assert executor.current_text == before_text


def test_submit_patch_rejects_patch_that_introduces_new_finding_identity(executor):
    result = executor.execute(
        SubmitPatchAction(
            old_text="Heart of Dao guarded 道心.",
            new_text="Dao Heart guarded 道心，",  # leaves residue and adds Chinese punctuation
            rationale="regressing patch",
        )
    )

    assert result.observation.ok is False
    assert result.observation.kind == "patch_rejected"
    assert executor.current_text == TRANSLATED_TEXT


def test_submit_patch_accepts_qa_improving_working_copy(executor):
    action = SubmitPatchAction(
        old_text="Heart of Dao guarded 道心.",
        new_text="Dao Heart guarded the mountain gate.",
        rationale="resolve both deterministic findings",
    )

    result = executor.execute(action)

    assert result.observation.ok is True
    assert result.observation.kind == "patch_accepted"
    assert executor.current_text.endswith("Dao Heart guarded the mountain gate.")
    assert result.qa_before is not None
    assert result.qa_after is not None
    assert result.qa_after.summary.total_findings == 0
    assert executor.current_qa.summary.total_findings == 0


def test_finish_rejects_while_findings_remain_and_succeeds_after_verified_patch(executor):
    premature = executor.execute(FinishAction(summary="done"))
    assert premature.observation.ok is False
    assert premature.observation.kind == "finish_rejected"
    assert executor.finished is False

    executor.execute(
        SubmitPatchAction(
            old_text="Heart of Dao guarded 道心.",
            new_text="Dao Heart guarded the mountain gate.",
            rationale="resolve findings",
        )
    )
    verified = executor.execute(FinishAction(summary="verified"))

    assert verified.observation.ok is True
    assert verified.observation.kind == "finished"
    assert executor.finished is True


def test_escalate_marks_executor_escalated(executor):
    result = executor.execute(EscalateAction(reason="Needs human review."))

    assert result.observation.ok is True
    assert result.observation.kind == "escalated"
    assert executor.escalated is True


def test_actions_after_escalate_are_rejected_without_mutation(executor):
    executor.execute(EscalateAction(reason="Needs human review."))
    before_text = executor.current_text
    before_qa = executor.current_qa.model_dump()

    result = executor.execute(
        SubmitPatchAction(
            old_text="Heart of Dao guarded 道心.",
            new_text="Dao Heart guarded the mountain gate.",
            rationale="should not run after escalation",
        )
    )

    assert result.observation.ok is False
    assert result.observation.kind == "terminal_rejected"
    assert executor.current_text == before_text
    assert executor.current_qa.model_dump() == before_qa


def test_actions_after_successful_finish_are_rejected_without_mutation(executor):
    executor.execute(
        SubmitPatchAction(
            old_text="Heart of Dao guarded 道心.",
            new_text="Dao Heart guarded the mountain gate.",
            rationale="resolve findings",
        )
    )
    executor.execute(FinishAction(summary="verified"))
    before_text = executor.current_text
    before_qa = executor.current_qa.model_dump()

    result = executor.execute(GetQAFindingsAction())

    assert result.observation.ok is False
    assert result.observation.kind == "terminal_rejected"
    assert executor.current_text == before_text
    assert executor.current_qa.model_dump() == before_qa


def test_findings_and_qa_evidence_bound_long_snippets(glossary):
    long_body = "Heart of Dao guarded 道心. " + ("filler " * 500)
    executor = RepairToolExecutor(
        source_text=SOURCE_TEXT,
        translated_text=f"Chapter 1\n\n{long_body}",
        glossary=glossary,
    )

    findings_result = executor.execute(GetQAFindingsAction())
    finding_snippet = next(
        item["location"]["snippet"]
        for item in findings_result.observation.data["findings"]
        if item["location"]["snippet"]
    )
    assert len(finding_snippet) <= 2000
    assert finding_snippet.endswith("...[truncated]")
    for report in (findings_result.qa_before, findings_result.qa_after):
        assert report is not None
        snippets = [
            finding.location.snippet
            for finding in report.findings
            if finding.location.snippet
        ]
        assert any(snippet.endswith("...[truncated]") for snippet in snippets)
        assert all(len(snippet) <= 2000 for snippet in snippets)

    patch_result = executor.execute(
        SubmitPatchAction(
            old_text="Heart of Dao",
            new_text="Heart of Dao",
            rationale="no-op evidence check",
        )
    )
    for report in (patch_result.qa_before, patch_result.qa_after):
        assert report is not None
        snippets = [
            finding.location.snippet
            for finding in report.findings
            if finding.location.snippet
        ]
        assert any(snippet.endswith("...[truncated]") for snippet in snippets)
        assert all(len(snippet) <= 2000 for snippet in snippets)


def test_findings_and_qa_evidence_bound_long_heading_fields(glossary):
    long_heading = "Not a normalized heading " + ("x" * 3500)
    executor = RepairToolExecutor(
        source_text=SOURCE_TEXT,
        translated_text=f"{long_heading}\n\nHeart of Dao guarded 道心.",
        glossary=glossary,
    )

    findings_result = executor.execute(GetQAFindingsAction())
    heading = next(
        item
        for item in findings_result.observation.data["findings"]
        if item["check_id"] == "heading_format"
    )
    assert len(heading["found"]) <= 2000
    assert heading["found"].endswith("...[truncated]")
    for report in (findings_result.qa_before, findings_result.qa_after):
        assert report is not None
        finding = next(
            finding
            for finding in report.findings
            if finding.check_id == "heading_format"
        )
        assert len(finding.found or "") <= 2000
        assert (finding.found or "").endswith("...[truncated]")

    patch_result = executor.execute(
        SubmitPatchAction(
            old_text="Heart of Dao guarded 道心.",
            new_text="Heart of Dao guarded 道心.",
            rationale="no-op evidence check",
        )
    )
    for report in (patch_result.qa_before, patch_result.qa_after):
        assert report is not None
        finding = next(
            finding
            for finding in report.findings
            if finding.check_id == "heading_format"
        )
        assert len(finding.found or "") <= 2000
        assert (finding.found or "").endswith("...[truncated]")


class ScriptedActionProvider:
    provider_name = "fixture-agent"
    model_name = "fixture-agent-v1"

    def __init__(self, actions):
        self.actions = list(actions)
        self.requests: list[AgentActionRequest] = []
        self.call_records = []

    def next_action(self, request: AgentActionRequest):
        self.requests.append(request)
        return self.actions.pop(0)


def _golden_actions():
    return [
        LookupGlossaryAction(term="道心"),
        SubmitPatchAction(
            old_text="not present",
            new_text="Dao Heart",
            rationale="deliberately rejected demo attempt",
        ),
        ReadSourceContextAction(finding_index=0, radius=1),
        SubmitPatchAction(
            old_text="Heart of Dao guarded 道心.",
            new_text="Dao Heart guarded the mountain gate.",
            rationale="replace the bounded defective paragraph",
        ),
        FinishAction(summary="Deterministic QA is clean."),
    ]


def test_run_repair_episode_follows_bounded_golden_path(tmp_path, glossary):
    provider = ScriptedActionProvider(_golden_actions())
    episode_path = tmp_path / "agent_episode.json"

    result = run_repair_episode(
        provider=provider,
        episode_path=episode_path,
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=glossary,
        run_id="test-run",
        story_slug="agentic-demo",
        chapter="0001",
        provider_mode="replay",
    )

    assert result.episode.final_status == "verified"
    assert [step.action["tool"] for step in result.episode.steps] == [
        "lookup_glossary",
        "submit_patch",
        "read_source_context",
        "submit_patch",
        "finish",
    ]
    assert result.episode.steps[1].observation.ok is False
    assert result.episode.steps[3].observation.ok is True
    assert result.episode.final_qa is not None
    assert result.episode.final_qa.summary.total_findings == 0
    assert result.final_text.endswith("Dao Heart guarded the mountain gate.")
    assert episode_path.exists()


def test_run_repair_episode_persists_terminology_resolution_and_auxiliary_call_order(
    tmp_path, glossary
):
    class Provider(ScriptedActionProvider):
        provider_name = "openai"
        model_name = "fixture-agent-v2"

        def next_action(self, request):
            call_number = len(self.call_records) + 1
            self.call_records.append(
                ProviderCallRecord(
                    role="agent",
                    namespace="agent_action",
                    provider=self.provider_name,
                    model=self.model_name,
                    payload_sha256=f"{call_number:064d}",
                    response_sha256=f"{call_number + 1:064d}",
                    cache_file=f"agent_action_{call_number}.json",
                    cache_hit=True,
                )
            )
            return super().next_action(request)

    provider = Provider(
        [
            ResolveTerminologyAction(term="道心", finding_index=0),
            EscalateAction(reason="stop after terminology demo"),
        ]
    )
    result = run_repair_episode(
        provider=provider,
        episode_path=tmp_path / "episode.json",
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=glossary,
        run_id="terminology",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        terminology_resolver=_agreeing_resolver(),
        tool_schema_version="agent-tools.v2",
        max_steps=2,
    )
    assert result.episode.final_status == "escalated"
    assert len(result.episode.terminology_resolutions) == 1
    step = result.episode.steps[0]
    assert step.action["tool"] == "resolve_terminology"
    assert [call.namespace for call in step.auxiliary_provider_calls] == [
        "terminology_vote",
        "terminology_vote",
    ]
    restored = AgentEpisode.model_validate_json((tmp_path / "episode.json").read_text())
    assert restored.steps[0].auxiliary_provider_calls == step.auxiliary_provider_calls
    assert restored.terminology_resolutions == result.episode.terminology_resolutions


def test_run_repair_episode_attaches_new_provider_call_records(tmp_path, glossary):
    class RecordingProvider(ScriptedActionProvider):
        def next_action(self, request):
            call_number = len(self.call_records) + 1
            self.call_records.append(
                ProviderCallRecord(
                    role="agent",
                    namespace="agent_action",
                    provider=self.provider_name,
                    model=self.model_name,
                    payload_sha256=f"{call_number:064d}",
                    response_sha256=f"{call_number + 1:064d}",
                    cache_file=f"agent_action_{call_number}.json",
                    cache_hit=True,
                )
            )
            return super().next_action(request)

    provider = RecordingProvider([LookupGlossaryAction(term="道心")])
    result = run_repair_episode(
        provider=provider,
        episode_path=tmp_path / "episode.json",
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=glossary,
        run_id="test-run",
        story_slug="agentic-demo",
        chapter="0001",
        provider_mode="replay",
        max_steps=1,
    )

    assert result.episode.steps[0].provider_call == provider.call_records[0]


def test_persist_episode_fsyncs_parent_directory_after_replace(tmp_path, glossary, monkeypatch):
    import agentic_translation.agent_repair as repair_module

    executor = RepairToolExecutor(
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=glossary,
    )
    episode = AgentEpisode(
        episode_id="episode",
        run_id="run",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        provider="fixture",
        model="fixture-v1",
        initial_qa=executor.current_qa,
    )
    episode_path = tmp_path / "episode.json"
    events = []
    real_replace = repair_module.os.replace

    def record_fsync(_fd):
        events.append("file_fsync")

    def record_replace(source, destination):
        events.append("replace")
        real_replace(source, destination)

    monkeypatch.setattr(repair_module.os, "fsync", record_fsync)
    monkeypatch.setattr(repair_module.os, "replace", record_replace)
    monkeypatch.setattr(
        repair_module,
        "_fsync_directory",
        lambda _directory: events.append("directory_fsync"),
    )

    _persist_episode(episode_path, episode)

    assert events == ["file_fsync", "replace", "directory_fsync"]


def test_run_repair_episode_persists_before_first_provider_call(tmp_path, glossary):
    episode_path = tmp_path / "nested" / "agent_episode.json"

    class ObservingProvider(ScriptedActionProvider):
        def next_action(self, request):
            assert episode_path.exists()
            restored = AgentEpisode.model_validate_json(episode_path.read_text(encoding="utf-8"))
            assert restored.steps == []
            assert restored.final_status is None
            return super().next_action(request)

    provider = ObservingProvider([LookupGlossaryAction(term="道心")])
    result = run_repair_episode(
        provider=provider,
        episode_path=episode_path,
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=glossary,
        run_id="test-run",
        story_slug="agentic-demo",
        chapter="0001",
        provider_mode="replay",
        max_steps=1,
    )

    assert result.episode.final_status == "budget_exhausted"
    assert episode_path.read_text(encoding="utf-8").lstrip().startswith("{")


def test_run_repair_episode_exhausts_action_budget(tmp_path, glossary):
    provider = ScriptedActionProvider([GetQAFindingsAction()] * 3)

    result = run_repair_episode(
        provider=provider,
        episode_path=tmp_path / "episode.json",
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=glossary,
        run_id="test-run",
        story_slug="agentic-demo",
        chapter="0001",
        provider_mode="replay",
        max_steps=2,
    )

    assert result.episode.final_status == "budget_exhausted"
    assert len(result.episode.steps) == 2
    assert len(provider.requests) == 2
    assert provider.requests[0].remaining_steps == 2
    assert provider.requests[1].remaining_steps == 1


def test_run_repair_episode_counts_rejected_patches_and_stops_before_third(tmp_path, glossary):
    provider = ScriptedActionProvider(
        [
            SubmitPatchAction(old_text="missing", new_text="one", rationale="one"),
            SubmitPatchAction(old_text="missing", new_text="two", rationale="two"),
            SubmitPatchAction(old_text="missing", new_text="three", rationale="three"),
        ]
    )

    result = run_repair_episode(
        provider=provider,
        episode_path=tmp_path / "episode.json",
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=glossary,
        run_id="test-run",
        story_slug="agentic-demo",
        chapter="0001",
        provider_mode="replay",
        max_steps=5,
        max_patch_attempts=2,
    )

    assert result.episode.final_status == "budget_exhausted"
    assert len(result.episode.steps) == 3
    assert result.episode.steps[-1].observation.kind == "patch_budget_exhausted"
    assert result.final_text == TRANSLATED_TEXT


def test_run_repair_episode_rejects_oversized_action_before_execution(tmp_path, glossary):
    oversized_text = "Dao Heart guarded the mountain gate." + (" filler" * 400)
    provider = ScriptedActionProvider(
        [
            SubmitPatchAction(
                old_text="Heart of Dao guarded 道心.",
                new_text=oversized_text,
                rationale="oversized candidate",
            )
        ]
    )
    episode_path = tmp_path / "episode.json"

    result = run_repair_episode(
        provider=provider,
        episode_path=episode_path,
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=glossary,
        run_id="test-run",
        story_slug="agentic-demo",
        chapter="0001",
        provider_mode="replay",
        max_steps=1,
    )

    assert result.final_text == TRANSLATED_TEXT
    assert result.episode.steps[0].observation.kind == "action_too_large"
    assert result.episode.steps[0].observation.ok is False
    persisted_action = result.episode.steps[0].action
    assert persisted_action["new_text"].endswith("...[truncated]")
    assert result.episode.steps[0].observation.data["tool"] == "submit_patch"
    assert episode_path.read_text(encoding="utf-8").count("patch_accepted") == 0


def test_run_repair_episode_escalates(tmp_path, glossary):
    provider = ScriptedActionProvider([EscalateAction(reason="Needs a human.")])

    result = run_repair_episode(
        provider=provider,
        episode_path=tmp_path / "episode.json",
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=glossary,
        run_id="test-run",
        story_slug="agentic-demo",
        chapter="0001",
        provider_mode="replay",
    )

    assert result.episode.final_status == "escalated"
    assert len(result.episode.steps) == 1


def test_run_repair_episode_records_invalid_action_without_raw_response(tmp_path, glossary):
    marker = "raw-provider-secret-marker"

    class InvalidProvider(ScriptedActionProvider):
        def next_action(self, request):
            raise AgentActionValidationError(
                "Agent action failed schema validation",
                response={"tool": "submit_patch", "old_text": "safe", "secret": marker},
            )

    episode_path = tmp_path / "episode.json"
    result = run_repair_episode(
        provider=InvalidProvider([]),
        episode_path=episode_path,
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=glossary,
        run_id="test-run",
        story_slug="agentic-demo",
        chapter="0001",
        provider_mode="replay",
        max_steps=1,
    )

    serialized = episode_path.read_text(encoding="utf-8")
    assert result.episode.final_status == "budget_exhausted"
    assert result.episode.steps[0].observation.kind == "invalid_action"
    assert marker not in serialized
    assert result.episode.steps[0].action == {"tool": "invalid_action"}


def test_run_repair_episode_persists_provider_failure_and_reraises(tmp_path, glossary):
    marker = "provider-secret-marker"

    class FailingProvider(ScriptedActionProvider):
        def next_action(self, request):
            raise LLMProviderUnavailable(f"temporary provider outage: {marker}")

    episode_path = tmp_path / "episode.json"
    with pytest.raises(LLMProviderUnavailable, match="temporary provider outage"):
        run_repair_episode(
            provider=FailingProvider([]),
            episode_path=episode_path,
            source_text=SOURCE_TEXT,
            translated_text=TRANSLATED_TEXT,
            glossary=glossary,
            run_id="test-run",
            story_slug="agentic-demo",
            chapter="0001",
            provider_mode="replay",
        )

    restored = AgentEpisode.model_validate_json(episode_path.read_text(encoding="utf-8"))
    assert restored.final_status == "failed"
    assert restored.steps == []
    assert restored.summary == "Provider failure (LLMProviderUnavailable)"
    assert marker not in episode_path.read_text(encoding="utf-8")


def test_run_repair_episode_only_shares_valid_prior_steps(tmp_path, glossary):
    class InvalidThenValid(ScriptedActionProvider):
        def __init__(self):
            super().__init__([LookupGlossaryAction(term="道心")])
            self.calls = 0

        def next_action(self, request):
            self.requests.append(request)
            self.calls += 1
            if self.calls == 1:
                raise AgentActionValidationError(
                    "invalid", response={"tool": "run_shell", "command": "secret"}
                )
            return self.actions.pop(0)

    provider = InvalidThenValid()
    run_repair_episode(
        provider=provider,
        episode_path=tmp_path / "episode.json",
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=glossary,
        run_id="test-run",
        story_slug="agentic-demo",
        chapter="0001",
        provider_mode="replay",
        max_steps=2,
    )

    assert provider.requests[1].prior_steps == []
