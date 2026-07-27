from __future__ import annotations

from pathlib import Path

import pytest

from agentic_translation.agent_models import (
    EscalateAction,
    FinishAction,
    LookupGlossaryAction,
    ReadSourceContextAction,
    ResolveTerminologyAction,
    SubmitPatchAction,
)
from agentic_translation.agent_provider import AgentActionRequest
from agentic_translation.models import ProviderCallRecord, TerminologyConsensusConfig
from agentic_translation.glossary import parse_glossary_text
from agentic_translation.terminology import TerminologyResolver
from agentic_translation.terminology_models import TerminologyVote
from agentic_translation.tool_agent_pipeline import build_terminology_resolver, run_tool_agent_phase


SOURCE_TEXT = "第一章\n\n道心守住了山门。"
TRANSLATED_TEXT = "Chapter 1\n\nHeart of Dao guarded 道心."


class GoldenActionProvider:
    provider_name = "fixture-agent"
    model_name = "fixture-agent-v1"

    def __init__(self) -> None:
        self.actions = [
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
        self.requests: list[AgentActionRequest] = []
        self.call_records: list[ProviderCallRecord] = []

    def next_action(self, request: AgentActionRequest):
        self.requests.append(request)
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
        return self.actions.pop(0)


def _run(tmp_path: Path):
    return run_tool_agent_phase(
        provider=GoldenActionProvider(),
        run_dir=tmp_path,
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=parse_glossary_text("道心: Dao Heart, Heart of Dao\n# block: Heart of Dao\n"),
        run_id="0001",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        story_title="Demo Story",
    )


def test_terminology_resolver_factory_uses_exact_openai_deepseek_models(tmp_path: Path) -> None:
    config = TerminologyConsensusConfig(
        enabled=True,
        openai_model="gpt-term",
        deepseek_model="deepseek-term",
        evaluator_provider="deepseek",
        confidence_threshold=0.8,
    )

    resolver = build_terminology_resolver(
        config,
        provider_mode="replay",
        cache_dir=tmp_path / "cache",
        record_cache=False,
    )

    assert resolver is not None
    assert [(v.provider_name, v.model_name) for v in resolver.voters] == [
        ("openai", "gpt-term"),
        ("deepseek", "deepseek-term"),
    ]
    assert resolver.evaluator.provider_name == "deepseek"
    assert resolver.evaluator.model_name == "deepseek-term"
    assert resolver.confidence_threshold == 0.8
    assert build_terminology_resolver(
        TerminologyConsensusConfig(),
        provider_mode="replay",
        cache_dir=tmp_path / "cache",
        record_cache=False,
    ) is None


def test_run_tool_agent_phase_writes_episode_and_reports(tmp_path: Path) -> None:
    result = _run(tmp_path)

    assert result.final_qa.summary.total_findings == 0
    assert result.episode.final_status == "verified"
    assert result.episode_path == tmp_path / "agent_repair" / "agent_episode.json"
    assert result.markdown_report_path == tmp_path / "agent_repair" / "report.md"
    assert result.html_report_path == tmp_path / "agent_repair" / "report.html"
    assert result.episode_path.exists()
    assert result.markdown_report_path.exists()
    assert result.html_report_path.exists()
    report = result.markdown_report_path.read_text(encoding="utf-8")
    assert "agent\\_episode" in report and "agent\\_repair/agent\\_episode\\.json" in report
    assert "repair\\_report" in report and "agent\\_repair/report\\.md" in report
    assert "report\\_html" in report and "agent\\_repair/report\\.html" in report


def test_run_tool_agent_phase_returns_bounded_counts_and_provider_calls(tmp_path: Path) -> None:
    provider = GoldenActionProvider()
    result = run_tool_agent_phase(
        provider=provider,
        run_dir=tmp_path,
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=parse_glossary_text("道心: Dao Heart, Heart of Dao\n# block: Heart of Dao\n"),
        run_id="0001",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
    )

    assert result.step_count == 5
    assert result.initial_findings == 3
    assert result.final_findings == 0
    assert result.accepted_patch_count == 1
    assert result.rejected_patch_count == 1
    assert result.final_status == "verified"
    assert result.provider_calls == provider.call_records
    assert result.run_record.provider_calls == provider.call_records
    assert result.run_record.final_text_sha256
    assert [step.provider_call for step in result.episode.steps] == provider.call_records


def test_run_tool_agent_phase_excludes_preexisting_provider_calls(tmp_path: Path) -> None:
    provider = GoldenActionProvider()
    old_record = ProviderCallRecord(
        role="agent",
        namespace="agent_action",
        provider="old-provider",
        model="old-model",
        payload_sha256="a" * 64,
        response_sha256="b" * 64,
        cache_file="old-call.json",
        cache_hit=True,
    )
    provider.call_records.append(old_record)

    result = run_tool_agent_phase(
        provider=provider,
        run_dir=tmp_path,
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=parse_glossary_text("道心: Dao Heart, Heart of Dao\n# block: Heart of Dao\n"),
        run_id="0001",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
    )

    new_calls = provider.call_records[1:]
    assert result.provider_calls == new_calls
    assert result.run_record.provider_calls == new_calls
    assert [step.provider_call for step in result.episode.steps] == new_calls
    report = result.markdown_report_path.read_text(encoding="utf-8")
    assert "old\\-call\\.json" not in report
    assert "old-provider" not in report


def test_run_tool_agent_phase_rejects_symlinked_agent_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (run_dir / "agent_repair").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        run_tool_agent_phase(
            provider=GoldenActionProvider(),
            run_dir=run_dir,
            source_text=SOURCE_TEXT,
            translated_text=TRANSLATED_TEXT,
            glossary=parse_glossary_text("道心: Dao Heart, Heart of Dao\n# block: Heart of Dao\n"),
            run_id="0001",
            story_slug="demo",
            chapter="0001",
            provider_mode="replay",
        )

    assert (run_dir / "agent_repair").is_symlink()
    assert list(outside.iterdir()) == []


def test_run_tool_agent_phase_reports_budget_safe_escalation(tmp_path: Path) -> None:
    class EscalatingProvider(GoldenActionProvider):
        def __init__(self) -> None:
            super().__init__()
            self.actions = []

        def next_action(self, request: AgentActionRequest):
            self.requests.append(request)
            return EscalateAction(reason="Needs human review.")

    result = run_tool_agent_phase(
        provider=EscalatingProvider(),
        run_dir=tmp_path,
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=parse_glossary_text("道心: Dao Heart, Heart of Dao\n# block: Heart of Dao\n"),
        run_id="0001",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
    )

    assert result.final_status == "escalated"
    assert result.step_count == 1
    assert result.final_findings == result.initial_findings == 3


def test_run_tool_agent_phase_aggregates_action_and_terminology_calls_in_step_order(
    tmp_path: Path,
) -> None:
    class Voter:
        def __init__(self, identity: str) -> None:
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

        def vote(self, request):
            return TerminologyVote(
                voter_id=self.voter_id,
                provider=self.provider_name,
                model=self.model_name,
                source_term=request.source_term,
                recommendation="Dao Heart",
                confidence=0.9,
                provider_call=self.call_records[0],
            )

    class Evaluator:
        provider_name = "openai"
        model_name = "openai-term"
        call_records = []

        def evaluate(self, request, candidates):
            raise AssertionError("agreement must not invoke evaluator")

    class Provider:
        provider_name = "openai"
        model_name = "fixture-agent-v2"

        def __init__(self):
            self.actions = [
                ResolveTerminologyAction(term="道心", finding_index=0),
                EscalateAction(reason="stop after terminology demo"),
            ]
            self.call_records = []

        def next_action(self, request):
            number = len(self.call_records) + 1
            self.call_records.append(
                ProviderCallRecord(
                    role="agent",
                    namespace="agent_action",
                    provider=self.provider_name,
                    model=self.model_name,
                    payload_sha256=f"{number:064d}",
                    response_sha256=f"{number + 1:064d}",
                    cache_file=f"agent_action_{number}.json",
                    cache_hit=True,
                )
            )
            return self.actions.pop(0)

    provider = Provider()
    resolver = TerminologyResolver(
        voters=[Voter("openai"), Voter("deepseek")],
        evaluator=Evaluator(),
    )
    result = run_tool_agent_phase(
        provider=provider,
        run_dir=tmp_path,
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=parse_glossary_text("道心: Dao Heart, Heart of Dao\n# block: Heart of Dao\n"),
        run_id="0001",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        terminology_resolver=resolver,
        tool_schema_version="agent-tools.v2",
    )
    assert [call.namespace for call in result.provider_calls] == [
        "agent_action",
        "terminology_vote",
        "terminology_vote",
        "agent_action",
    ]
