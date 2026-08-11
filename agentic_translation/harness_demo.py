"""Deterministic Harness v3 golden path used by the operational review demo.

This module deliberately keeps the fixture orchestration out of :mod:`cli`.
The provider below is a synthetic native-function contract fixture: it never
opens a network client and it is not a replay cache.  The public v3 session
runtime remains responsible for tool exposure, policy, durable events,
approval binding, and resumable execution.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal

from .agent_models import (
    AgentAction,
)
from .agent_provider import AgentActionRequest, REGISTRY_TOOL_SCHEMA_VERSION
from .agent_report import render_agent_episode_html, render_agent_episode_markdown
from .agent_session import AgentSessionResult, resume_repair_session, run_repair_session
from .agent_tools import AGENT_TOOL_REGISTRY, ToolCall
from .glossary import load_glossary
from .models import GlossaryParseResult
from .terminology import TerminologyResolver
from .terminology_models import (
    TerminologyCandidate,
    TerminologyEvaluation,
    TerminologyRequest,
    TerminologyVote,
)
from .story import prepare_run_dir


DEMO_SLUG = "agentic_harness_v3_demo"
DEMO_TITLE = "Harness v3 Golden Path"
DEMO_CHAPTER = "0001"
DEMO_PROVIDER = "synthetic"
DEMO_MODEL = "harness-v3-native-function-fixture"
DEMO_PROVIDER_MODE = "synthetic_fixture"
DEMO_PROVENANCE = (
    "Synthetic contract fixture — native v3 tool calls are scripted locally; "
    "this is not live-provider or replay evidence. This is not production evidence."
)
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = _PACKAGE_ROOT / "samples" / "agentic_harness_v3_demo"
FIXTURE_SOURCE = FIXTURE_ROOT / "source" / f"{DEMO_CHAPTER}.txt"
FIXTURE_DIRTY_TRANSLATION = FIXTURE_ROOT / "expected" / "dirty_translation.txt"
FIXTURE_MASTER_GLOSSARY = FIXTURE_ROOT / "terms" / "master_glossary.txt"
RUN_GLOSSARY_NAME = "glossary.txt"
RUN_TRANSLATION_NAME = "translated_final.txt"


class SyntheticNativeFixtureProvider:
    """Return one deterministic native-function action for each v3 step.

    The provider uses the same ``ToolCall`` normalization path as a native
    backend, including the ``tools_search`` provider alias.  It intentionally
    has no cache and no network client; ``call_records`` is empty by design.
    """

    provider_name = DEMO_PROVIDER
    model_name = DEMO_MODEL
    tool_protocol = "native_function"

    def __init__(self) -> None:
        self.requests: list[AgentActionRequest] = []
        self.call_records: list[Any] = []

    @staticmethod
    def _wire_action(step_number: int) -> tuple[str, dict[str, Any]]:
        if step_number == 1:
            return "tools_search", {"query": "translation terminology promotion", "limit": 8}
        if step_number == 2:
            return "resolve_terminology", {"term": "道心", "finding_index": 0}
        if step_number == 3:
            # Deliberately rejected: old_text is absent, so the working copy
            # and QA projection cannot be mutated.
            return "submit_patch", {
                "edits": [{
                    "old_text": "This patch target is intentionally absent.",
                    "new_text": "Dao Heart guarded the mountain gate.",
                }],
                "rationale": "Demonstrate fail-closed patch validation before mutation.",
            }
        if step_number == 4:
            return "submit_patch", {
                "edits": [{
                    "old_text": "Heart of Dao guarded 道心.",
                    "new_text": "Dao Heart guarded the mountain gate.",
                }],
                "rationale": "Apply the resolved canonical term and restore natural English.",
            }
        if step_number == 5:
            return "promote_glossary_term", {
                "term": "道心",
                "rationale": "Propose the resolved term for the run-local glossary after QA is clean.",
            }
        # Approval resumes at step six.  A fresh process can therefore replay
        # the same deterministic provider without persisting provider state.
        return "finish", {"summary": "Translation verified after approved glossary promotion."}

    def next_action(self, request: AgentActionRequest) -> AgentAction:
        self.requests.append(request)
        if request.tool_schema_version != REGISTRY_TOOL_SCHEMA_VERSION:
            raise ValueError("Golden fixture requires the Harness v3 registry contract")
        if request.tool_protocol != "native_function":
            raise ValueError("Golden fixture requires the native_function request protocol")
        name, arguments = self._wire_action(request.step_number)
        call = ToolCall.from_native(name=name, arguments=json.dumps(arguments, ensure_ascii=False))
        visible_names = request.exposed_tool_names or tuple(spec.name for spec in AGENT_TOOL_REGISTRY)
        return AGENT_TOOL_REGISTRY.action_from_call(call, visible_names=visible_names)


class _SyntheticTerminologyVoter:
    def __init__(self, voter_id: Literal["openai", "deepseek"]) -> None:
        self.voter_id = voter_id
        self.provider_name = voter_id
        self.model_name = f"synthetic-{voter_id}-terminology"
        self.call_records: list[Any] = []

    def vote(self, request: TerminologyRequest) -> TerminologyVote:
        return TerminologyVote(
            voter_id=self.voter_id,
            provider=self.provider_name,
            model=self.model_name,
            source_term=request.source_term,
            recommendation="Dao Heart",
            confidence=0.99,
            alternatives=["Heart of Dao"],
            rationale="Deterministic contract fixture selects the canonical term for this operational run.",
        )


class _SyntheticTerminologyEvaluator:
    provider_name = "openai"
    model_name = "synthetic-openai-evaluator"
    call_records: list[Any] = []

    def evaluate(
        self,
        request: TerminologyRequest,
        candidates: list[TerminologyCandidate],
    ) -> TerminologyEvaluation:
        selected = next(
            candidate for candidate in candidates if candidate.recommendation == "Dao Heart"
        )
        return TerminologyEvaluation(
            provider=self.provider_name,
            model=self.model_name,
            selected_candidate_id=selected.candidate_id,
            confidence=0.99,
            rationale="Deterministic evaluator fallback; agreement normally avoids arbitration.",
        )


def synthetic_terminology_resolver() -> TerminologyResolver:
    """Return deterministic two-voter terminology agreement for the fixture."""

    return TerminologyResolver(
        voters=[_SyntheticTerminologyVoter("openai"), _SyntheticTerminologyVoter("deepseek")],
        evaluator=_SyntheticTerminologyEvaluator(),
        confidence_threshold=0.65,
    )


def _fixture_inputs() -> tuple[str, str, GlossaryParseResult]:
    return (
        FIXTURE_SOURCE.read_text(encoding="utf-8"),
        FIXTURE_DIRTY_TRANSLATION.read_text(encoding="utf-8"),
        load_glossary(FIXTURE_MASTER_GLOSSARY),
    )


def _copy_run_glossary(run_dir: Path) -> Path:
    target = run_dir / RUN_GLOSSARY_NAME
    shutil.copyfile(FIXTURE_MASTER_GLOSSARY, target)
    return target


def _artifact_paths(run_dir: Path) -> dict[str, str]:
    return {
        "session_events": "session_events.jsonl",
        "session_snapshot": "session_snapshot.json",
        "agent_episode": "agent_episode.json",
        "run_glossary": RUN_GLOSSARY_NAME,
        "translated_final": RUN_TRANSLATION_NAME,
        "report_markdown": "report.md",
        "report_html": "report.html",
    }


def _write_final_and_reports(
    *,
    run_dir: Path,
    result: AgentSessionResult,
    source_text: str,
    initial_text: str,
    provider: SyntheticNativeFixtureProvider,
) -> None:
    (run_dir / RUN_TRANSLATION_NAME).write_text(result.final_text, encoding="utf-8")
    paths = _artifact_paths(run_dir)
    markdown = render_agent_episode_markdown(
        result.episode,
        story_title=DEMO_TITLE,
        source_text=source_text,
        translation_text=initial_text,
        final_text=result.final_text,
        artifact_paths=paths,
        call_records=provider.call_records,
        provenance_note=DEMO_PROVENANCE,
        session_snapshot=result.snapshot,
        session_events=result.events,
    )
    (run_dir / "report.md").write_text(markdown, encoding="utf-8")
    render_agent_episode_html(
        run_dir / "report.html",
        result.episode,
        story_title=DEMO_TITLE,
        source_text=source_text,
        translation_text=initial_text,
        final_text=result.final_text,
        artifact_paths=paths,
        call_records=provider.call_records,
        provenance_note=DEMO_PROVENANCE,
        session_snapshot=result.snapshot,
        session_events=result.events,
    )


def run_golden_demo(
    *,
    runs_dir: str | Path,
    pause_for_approval: bool = True,
    auto_approve: bool = False,
    overwrite: bool = False,
    reviewer: str = "demo-reviewer",
    note: str = "Approved in the Harness v3 review flow.",
) -> AgentSessionResult:
    """Run the golden trajectory and optionally apply its approval receipt."""

    if pause_for_approval and auto_approve:
        raise ValueError("Choose either --pause-for-approval or --auto-approve, not both")
    run_dir = prepare_run_dir(Path(runs_dir), DEMO_SLUG, overwrite=overwrite)
    source_text, initial_text, fixture_glossary = _fixture_inputs()
    run_glossary_path = _copy_run_glossary(run_dir)
    provider = SyntheticNativeFixtureProvider()
    result = run_repair_session(
        provider=provider,
        session_dir=run_dir,
        source_text=source_text,
        translated_text=initial_text,
        glossary=fixture_glossary,
        canonical_glossary_path=run_glossary_path,
        run_id=DEMO_SLUG,
        story_slug=DEMO_SLUG,
        chapter=DEMO_CHAPTER,
        provider_mode=DEMO_PROVIDER_MODE,
        max_steps=8,
        max_patch_attempts=2,
        dynamic_tools=True,
        terminology_resolver=synthetic_terminology_resolver(),
    )
    if auto_approve and result.snapshot.status == "awaiting_approval":
        result = resume_repair_session(
            session_dir=run_dir,
            provider=SyntheticNativeFixtureProvider(),
            source_text=source_text,
            glossary=fixture_glossary,
            canonical_glossary_path=run_glossary_path,
            run_id=DEMO_SLUG,
            story_slug=DEMO_SLUG,
            chapter=DEMO_CHAPTER,
            provider_mode=DEMO_PROVIDER_MODE,
            terminology_resolver=synthetic_terminology_resolver(),
            decision="approved",
            reviewer=reviewer,
            note=note,
        )
        provider = SyntheticNativeFixtureProvider()
    _write_final_and_reports(
        run_dir=run_dir,
        result=result,
        source_text=source_text,
        initial_text=initial_text,
        provider=provider,
    )
    return result


def resume_golden_demo(
    *,
    run_dir: str | Path,
    decision: Literal["approved", "rejected"],
    reviewer: str,
    note: str,
) -> AgentSessionResult:
    """Resume a paused golden run and rewrite the same review artifacts."""

    selected_run = Path(run_dir).expanduser().resolve()
    if not selected_run.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {selected_run}")
    source_text, initial_text, fixture_glossary = _fixture_inputs()
    run_glossary_path = selected_run / RUN_GLOSSARY_NAME
    snapshot_path = selected_run / "session_snapshot.json"
    if not run_glossary_path.is_file() or not snapshot_path.is_file():
        raise FileNotFoundError("Golden run is missing glossary.txt or session_snapshot.json")
    provider = SyntheticNativeFixtureProvider()
    result = resume_repair_session(
        session_dir=selected_run,
        provider=provider,
        source_text=source_text,
        glossary=fixture_glossary,
        canonical_glossary_path=run_glossary_path,
        run_id=DEMO_SLUG,
        story_slug=DEMO_SLUG,
        chapter=DEMO_CHAPTER,
        provider_mode=DEMO_PROVIDER_MODE,
        terminology_resolver=synthetic_terminology_resolver(),
        decision=decision,
        reviewer=reviewer,
        note=note,
    )
    _write_final_and_reports(
        run_dir=selected_run,
        result=result,
        source_text=source_text,
        initial_text=initial_text,
        provider=provider,
    )
    return result


__all__ = [
    "DEMO_MODEL",
    "DEMO_PROVIDER_MODE",
    "DEMO_PROVENANCE",
    "DEMO_SLUG",
    "SyntheticNativeFixtureProvider",
    "resume_golden_demo",
    "run_golden_demo",
    "synthetic_terminology_resolver",
]
