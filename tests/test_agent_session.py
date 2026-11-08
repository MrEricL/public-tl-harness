from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_translation.agent_models import (
    AgentSessionIdentity,
    FinishAction,
    NormalizePunctuationAction,
    PromoteGlossaryTermAction,
    SearchToolsAction,
    SubmitPatchAction,
)
from agentic_translation.agent_models import AgentEpisode
from agentic_translation.agent_provider import AgentActionRequest
from agentic_translation.agent_provider import REGISTRY_TOOL_SCHEMA_VERSION
from agentic_translation.agent_repair import run_repair_episode
from agentic_translation.agent_session import (
    AgentSessionSnapshot,
    RepairExecutorSnapshot,
    SessionIdentityMismatchError,
    SessionStore,
    resume_repair_session,
    run_repair_session,
)
from agentic_translation.agent_tools import AGENT_TOOL_REGISTRY
from agentic_translation.glossary import parse_glossary_text
from agentic_translation.models import GlossaryParseResult, QAReport
from agentic_translation.terminology import TerminologyResolver
from agentic_translation.terminology_models import TerminologyVote


SOURCE_TEXT = "第一章\n\n道心守住了山门。"
TRANSLATED_TEXT = "Chapter 1\n\nHeart of Dao guarded 道心."


class _Provider:
    provider_name = "fixture-agent"
    model_name = "fixture-agent-v3"

    def __init__(self, actions):
        self.actions = list(actions)
        self.requests: list[AgentActionRequest] = []
        self.call_records = []

    def next_action(self, request: AgentActionRequest):
        self.requests.append(request)
        return self.actions.pop(0)


class _Resolver:
    def resolve(self, request):
        return TerminologyResolver(
            voters=[], evaluator=None
        )  # pragma: no cover


class _AgreementVoter:
    def __init__(self, identity: str):
        self.voter_id = identity
        self.provider_name = identity
        self.model_name = f"{identity}-term"
        self.call_records = []

    def vote(self, request):
        return TerminologyVote(
            voter_id=self.voter_id,
            provider=self.provider_name,
            model=self.model_name,
            source_term=request.source_term,
            recommendation="Dao Heart",
            confidence=0.9,
        )


class _NoopEvaluator:
    provider_name = "openai"
    model_name = "openai-term"
    call_records = []

    def evaluate(self, request, candidates):
        raise AssertionError("agreement must not invoke evaluator")


def _resolver():
    return TerminologyResolver(
        voters=[_AgreementVoter("openai"), _AgreementVoter("deepseek")],
        evaluator=_NoopEvaluator(),
    )


def _qa_payload() -> dict[str, object]:
    return {
        "run_id": "run",
        "story_slug": "demo",
        "chapter": "0001",
        "findings": [],
        "summary": {},
    }


def _episode() -> AgentEpisode:
    qa = QAReport.model_validate(_qa_payload())
    return AgentEpisode(
        episode_id="episode",
        run_id="run",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        provider="openai",
        model="fixture-model",
        initial_qa=qa,
    )


def _identity(**overrides: object) -> AgentSessionIdentity:
    payload: dict[str, object] = {
        "run_id": "run",
        "story_slug": "demo",
        "chapter": "0001",
        "provider_mode": "replay",
        "provider": "openai",
        "model": "fixture-model",
        "tool_schema_version": "agent-tools.v3",
        "tool_protocol": "json_prompt",
        "source_sha256": "a" * 64,
        "master_glossary_sha256": "b" * 64,
        "registry_sha256": "c" * 64,
    }
    payload.update(overrides)
    return AgentSessionIdentity.model_validate(payload)


def test_append_events_assigns_sequences_and_reloads(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)

    first = store.append("run_started", {"run_id": "run"})
    second = store.append("checkpoint_written", {"status": "running"})

    assert first.sequence == 1
    assert second.sequence == 2
    assert SessionStore(tmp_path).read_events() == [first, second]


def test_snapshot_and_episode_round_trip_with_legacy_episode_defaults(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path)
    episode = _episode()
    snapshot = AgentSessionSnapshot(
        status="running",
        episode=episode,
        executor=RepairExecutorSnapshot(
            current_text="translated",
            current_qa=episode.initial_qa,
            episode_glossary=GlossaryParseResult(entries=[]),
            escalated=False,
            finished=False,
        ),
        prior_steps=[],
        patch_attempts=0,
        exposed_tool_names=["get_qa_findings"],
        last_event_sequence=0,
    )

    store.write_snapshot(snapshot)
    store.write_episode(episode)

    assert store.load_snapshot() == snapshot
    assert store.load_episode() == episode

    legacy = episode.model_dump(mode="json")
    legacy.pop("steps")
    legacy.pop("terminology_resolutions")
    restored = AgentEpisode.model_validate(legacy)
    assert restored.steps == []
    assert restored.terminology_resolutions == []


def test_session_identity_round_trips_in_snapshot_and_legacy_snapshot_stays_readable(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path)
    episode = _episode()
    snapshot = AgentSessionSnapshot(
        status="running",
        episode=episode,
        executor=RepairExecutorSnapshot(
            current_text="translated",
            current_qa=episode.initial_qa,
            episode_glossary=GlossaryParseResult(entries=[]),
            escalated=False,
            finished=False,
        ),
        identity=_identity(),
        exposed_tool_names=["get_qa_findings"],
    )

    store.write_snapshot(snapshot)

    restored = store.load_snapshot()
    assert restored is not None
    assert restored.identity == snapshot.identity
    assert restored.identity.schema_version == "agent-session-identity.v1"

    legacy_payload = snapshot.model_dump(mode="json")
    legacy_payload.pop("identity")
    legacy = AgentSessionSnapshot.model_validate(legacy_payload)
    assert legacy.identity is None


def test_session_identity_is_strict_and_validates_bounded_lowercase_digests() -> None:
    with pytest.raises(ValidationError):
        AgentSessionIdentity.model_validate({**_identity().model_dump(), "extra": True})
    with pytest.raises(ValidationError):
        AgentSessionIdentity.model_validate({**_identity().model_dump(), "run_id": ""})
    with pytest.raises(ValidationError):
        AgentSessionIdentity.model_validate(
            {**_identity().model_dump(), "source_sha256": "A" * 64}
        )
    with pytest.raises(ValidationError):
        AgentSessionIdentity.model_validate(
            {**_identity().model_dump(), "tool_protocol": "text"}
        )


def test_run_repair_session_rejects_reusing_artifacts_before_provider_call(
    tmp_path: Path,
) -> None:
    glossary = parse_glossary_text("道心: Dao Heart\n")
    session_dir = tmp_path / "session"
    first_provider = _Provider([FinishAction(summary="clean")])

    run_repair_session(
        provider=first_provider,
        session_dir=session_dir,
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=glossary,
        run_id="run",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        max_steps=1,
    )
    assert (session_dir / SessionStore.EVENTS_FILENAME).exists()
    assert (session_dir / SessionStore.SNAPSHOT_FILENAME).exists()
    assert (session_dir / SessionStore.EPISODE_FILENAME).exists()

    second_provider = _Provider([FinishAction(summary="must not run")])
    with pytest.raises((FileExistsError, ValueError)) as excinfo:
        run_repair_session(
            provider=second_provider,
            session_dir=session_dir,
            source_text=SOURCE_TEXT,
            translated_text=TRANSLATED_TEXT,
            glossary=glossary,
            run_id="run",
            story_slug="demo",
            chapter="0001",
            provider_mode="replay",
            max_steps=1,
        )

    message = str(excinfo.value)
    assert "resume_repair_session" in message
    assert "new directory" in message
    assert second_provider.requests == []


@pytest.mark.parametrize(
    "artifact_name",
    [
        SessionStore.EVENTS_FILENAME,
        SessionStore.SNAPSHOT_FILENAME,
        SessionStore.EPISODE_FILENAME,
    ],
)
def test_run_repair_session_rejects_any_partial_artifact_but_ignores_unrelated_files(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    glossary = parse_glossary_text("道心: Dao Heart\n")
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "copied-run-local-glossary.txt").write_text(
        "道心: Dao Heart\n", encoding="utf-8"
    )
    (session_dir / artifact_name).write_text("partial", encoding="utf-8")
    provider = _Provider([FinishAction(summary="must not run")])

    with pytest.raises(FileExistsError, match="resume_repair_session"):
        run_repair_session(
            provider=provider,
            session_dir=session_dir,
            source_text=SOURCE_TEXT,
            translated_text=TRANSLATED_TEXT,
            glossary=glossary,
            run_id="run",
            story_slug="demo",
            chapter="0001",
            provider_mode="replay",
            max_steps=1,
        )

    assert provider.requests == []


def test_run_repair_session_allows_unrelated_files_in_new_directory(
    tmp_path: Path,
) -> None:
    glossary = parse_glossary_text("道心: Dao Heart\n")
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "copied-run-local-glossary.txt").write_text(
        "道心: Dao Heart\n", encoding="utf-8"
    )
    provider = _Provider([FinishAction(summary="clean")])

    result = run_repair_session(
        provider=provider,
        session_dir=session_dir,
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=glossary,
        run_id="run",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        max_steps=1,
    )

    assert result.snapshot.status == "completed"
    assert len(provider.requests) == 1


def test_new_run_persists_bound_identity_with_canonical_fingerprints(tmp_path: Path) -> None:
    glossary = parse_glossary_text("道心: Dao Heart\n")
    provider = _Provider([FinishAction(summary="clean")])

    result = run_repair_session(
        provider=provider,
        session_dir=tmp_path,
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=glossary,
        run_id="run",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        max_steps=1,
    )

    assert result.snapshot.identity is not None
    identity = result.snapshot.identity
    encoded_glossary = json.dumps(
        glossary.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert identity.run_id == "run"
    assert identity.story_slug == "demo"
    assert identity.chapter == "0001"
    assert identity.provider_mode == "replay"
    assert identity.provider == provider.provider_name
    assert identity.model == provider.model_name
    assert identity.tool_protocol == "json_prompt"
    assert identity.tool_schema_version == REGISTRY_TOOL_SCHEMA_VERSION
    assert identity.source_sha256 == hashlib.sha256(SOURCE_TEXT.encode("utf-8")).hexdigest()
    assert identity.master_glossary_sha256 == hashlib.sha256(encoded_glossary).hexdigest()
    assert identity.registry_sha256 == AGENT_TOOL_REGISTRY.contract_sha256()


def test_initial_run_started_append_failure_leaves_no_reserved_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    glossary = parse_glossary_text("道心: Dao Heart\n")
    provider = _Provider([FinishAction(summary="clean")])
    failed = False
    original_append = SessionStore.append

    def fail_once(self, event_type, payload):
        nonlocal failed
        if event_type == "run_started" and not failed:
            failed = True
            raise OSError("injected startup append failure")
        return original_append(self, event_type, payload)

    monkeypatch.setattr(SessionStore, "append", fail_once)
    session_dir = tmp_path / "session"
    run_kwargs = {
        "provider": provider,
        "session_dir": session_dir,
        "source_text": SOURCE_TEXT,
        "translated_text": TRANSLATED_TEXT,
        "glossary": glossary,
        "run_id": "run",
        "story_slug": "demo",
        "chapter": "0001",
        "provider_mode": "replay",
        "max_steps": 1,
    }

    with pytest.raises(OSError, match="injected startup append failure"):
        run_repair_session(**run_kwargs)

    reserved_paths = [
        session_dir / SessionStore.EVENTS_FILENAME,
        session_dir / SessionStore.SNAPSHOT_FILENAME,
        session_dir / SessionStore.EPISODE_FILENAME,
    ]
    assert provider.requests == []
    assert all(not path.exists() and not path.is_symlink() for path in reserved_paths)

    retried = run_repair_session(**run_kwargs)
    assert retried.snapshot.status == "completed"
    assert len(provider.requests) == 1


def _paused_identity_session(tmp_path: Path) -> tuple[Path, Path, _Provider, GlossaryParseResult]:
    glossary = parse_glossary_text("道心: Dao Heart\n")
    canonical = tmp_path / "master_glossary.txt"
    canonical.write_text("道心 -> Heart of Dao\n", encoding="utf-8")
    provider = _Provider(
        [PromoteGlossaryTermAction(term="道心", rationale="consensus")]
    )
    session_dir = tmp_path / "session"
    paused = run_repair_session(
        provider=provider,
        session_dir=session_dir,
        source_text=SOURCE_TEXT,
        translated_text="Chapter 1\n\nDao Heart guarded the mountain gate.",
        glossary=glossary,
        canonical_glossary_path=canonical,
        run_id="run",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        max_steps=2,
        dynamic_tools=False,
    )
    assert paused.snapshot.status == "awaiting_approval"
    return session_dir, canonical, provider, glossary


@pytest.mark.parametrize(
    ("field", "resume_kwargs"),
    [
        ("source_sha256", {"source_text": SOURCE_TEXT + " altered"}),
        ("master_glossary_sha256", {"glossary": parse_glossary_text("道心: Dao Spirit\n")}),
        ("provider", {"provider_name": "other-provider"}),
        ("model", {"model_name": "other-model"}),
        ("provider_mode", {"provider_mode": "live"}),
        ("tool_protocol", {"tool_protocol": "native_function"}),
        ("run_id", {"run_id": "other-run"}),
        ("story_slug", {"story_slug": "other-story"}),
        ("chapter", {"chapter": "0002"}),
    ],
)
def test_resume_rejects_identity_mismatch_without_mutating_artifacts(
    tmp_path: Path,
    field: str,
    resume_kwargs: dict[str, object],
) -> None:
    session_dir, canonical, provider, glossary = _paused_identity_session(tmp_path)
    resume_options = dict(resume_kwargs)
    if "provider_name" in resume_options:
        provider.provider_name = str(resume_options.pop("provider_name"))
    if "model_name" in resume_options:
        provider.model_name = str(resume_options.pop("model_name"))
    if "tool_protocol" in resume_options:
        provider.tool_protocol = str(resume_options.pop("tool_protocol"))
    artifact_paths = [
        session_dir / SessionStore.EVENTS_FILENAME,
        session_dir / SessionStore.SNAPSHOT_FILENAME,
        session_dir / SessionStore.EPISODE_FILENAME,
        canonical,
    ]
    before = {path: path.read_bytes() for path in artifact_paths}
    requests_before = list(provider.requests)
    kwargs: dict[str, object] = {
        "session_dir": session_dir,
        "provider": provider,
        "source_text": SOURCE_TEXT,
        "glossary": glossary,
        "canonical_glossary_path": canonical,
        "decision": "approved",
        "reviewer": "tester",
        "note": "must reject",
    }
    kwargs.update(resume_options)

    with pytest.raises(SessionIdentityMismatchError) as excinfo:
        resume_repair_session(**kwargs)

    assert str(excinfo.value) == f"Session identity mismatch for fields: {field}"
    assert provider.requests == requests_before
    assert {path: path.read_bytes() for path in artifact_paths} == before


@pytest.mark.parametrize("field", ["tool_schema_version", "registry_sha256"])
def test_resume_rejects_tampered_runtime_contract_without_mutating_artifacts(
    tmp_path: Path,
    field: str,
) -> None:
    session_dir, canonical, provider, glossary = _paused_identity_session(tmp_path)
    store = SessionStore(session_dir)
    snapshot = store.load_snapshot()
    assert snapshot is not None and snapshot.identity is not None
    tampered = snapshot.identity.model_copy(
        update={field: "agent-tools.v2" if field == "tool_schema_version" else "d" * 64}
    )
    store.write_snapshot(snapshot.model_copy(update={"identity": tampered}))
    artifact_paths = [
        session_dir / SessionStore.EVENTS_FILENAME,
        session_dir / SessionStore.SNAPSHOT_FILENAME,
        session_dir / SessionStore.EPISODE_FILENAME,
        canonical,
    ]
    before = {path: path.read_bytes() for path in artifact_paths}
    requests_before = list(provider.requests)

    with pytest.raises(SessionIdentityMismatchError) as excinfo:
        resume_repair_session(
            session_dir=session_dir,
            provider=provider,
            source_text=SOURCE_TEXT,
            glossary=glossary,
            canonical_glossary_path=canonical,
            decision="approved",
            reviewer="tester",
            note="must reject",
        )

    assert str(excinfo.value) == f"Session identity mismatch for fields: {field}"
    assert provider.requests == requests_before
    assert {path: path.read_bytes() for path in artifact_paths} == before


def test_resume_rejects_legacy_snapshot_without_identity_without_mutating_artifacts(
    tmp_path: Path,
) -> None:
    session_dir, canonical, provider, glossary = _paused_identity_session(tmp_path)
    store = SessionStore(session_dir)
    snapshot = store.load_snapshot()
    assert snapshot is not None
    store.write_snapshot(snapshot.model_copy(update={"identity": None}))
    artifact_paths = [
        session_dir / SessionStore.EVENTS_FILENAME,
        session_dir / SessionStore.SNAPSHOT_FILENAME,
        session_dir / SessionStore.EPISODE_FILENAME,
        canonical,
    ]
    before = {path: path.read_bytes() for path in artifact_paths}
    requests_before = list(provider.requests)

    with pytest.raises(SessionIdentityMismatchError, match="new session"):
        resume_repair_session(
            session_dir=session_dir,
            provider=provider,
            source_text=SOURCE_TEXT,
            glossary=glossary,
            canonical_glossary_path=canonical,
            decision="approved",
            reviewer="tester",
            note="must reject",
        )

    assert provider.requests == requests_before
    assert {path: path.read_bytes() for path in artifact_paths} == before


@pytest.mark.parametrize(
    "artifact_name",
    [
        SessionStore.EVENTS_FILENAME,
        SessionStore.SNAPSHOT_FILENAME,
        SessionStore.EPISODE_FILENAME,
    ],
)
def test_run_repair_session_rejects_broken_reserved_artifact_symlink_before_provider(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    glossary = parse_glossary_text("道心: Dao Heart\n")
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    external_target = tmp_path / "outside" / artifact_name
    external_target.parent.mkdir()
    reserved_path = session_dir / artifact_name
    reserved_path.symlink_to(external_target)
    assert not reserved_path.exists()
    assert reserved_path.is_symlink()
    provider = _Provider([FinishAction(summary="must not run")])

    with pytest.raises(FileExistsError, match="resume_repair_session"):
        run_repair_session(
            provider=provider,
            session_dir=session_dir,
            source_text=SOURCE_TEXT,
            translated_text=TRANSLATED_TEXT,
            glossary=glossary,
            run_id="run",
            story_slug="demo",
            chapter="0001",
            provider_mode="replay",
            max_steps=1,
        )

    assert provider.requests == []
    assert reserved_path.is_symlink()
    assert not external_target.exists()


def test_v3_dynamic_search_expands_exposure_and_rejects_hidden_action(tmp_path: Path) -> None:
    glossary = parse_glossary_text("道心: Dao Heart\n")
    provider = _Provider(
        [
            SearchToolsAction(query="translation"),
            PromoteGlossaryTermAction(term="道心", rationale="hidden until discovered"),
        ]
    )
    result = run_repair_session(
        provider=provider,
        session_dir=tmp_path,
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=glossary,
        run_id="run",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        max_steps=2,
        dynamic_tools=True,
    )

    assert result.snapshot.exposed_tool_names
    assert "submit_patch" in result.snapshot.exposed_tool_names
    assert result.snapshot.episode.steps[-1].observation.ok is False
    assert result.snapshot.episode.steps[-1].observation.kind == "tool_rejected"
    assert result.snapshot.episode.steps[-1].action["tool"] == "promote_glossary_term"
    assert not list(result.snapshot.episode.steps[-1].observation.data.values()) == []


def test_v3_promotion_pause_approval_resume_writes_once_and_does_not_repeat_provider(
    tmp_path: Path,
) -> None:
    glossary = parse_glossary_text("道心: Dao Heart\n")
    canonical = tmp_path / "master_glossary.txt"
    canonical.write_text("道心 -> Heart of Dao\n", encoding="utf-8")
    provider = _Provider(
        [
            SearchToolsAction(query="terminology"),
            # Search exposes terminology, but the action is still hidden until
            # the exact promotion name is searched.
            SearchToolsAction(query="promotion"),
            PromoteGlossaryTermAction(term="道心", rationale="consensus"),
            FinishAction(summary="clean"),
        ]
    )
    paused = run_repair_session(
        provider=provider,
        session_dir=tmp_path / "session",
        source_text=SOURCE_TEXT,
        translated_text="Chapter 1\n\nDao Heart guarded the mountain gate.",
        glossary=glossary,
        canonical_glossary_path=canonical,
        terminology_resolver=_resolver(),
        run_id="run",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        max_steps=5,
        dynamic_tools=False,
    )

    assert paused.snapshot.status == "awaiting_approval"
    assert canonical.read_text(encoding="utf-8") == "道心 -> Heart of Dao\n"
    calls_before = len(provider.requests)
    resumed = resume_repair_session(
        session_dir=paused.session_dir,
        provider=provider,
        source_text=SOURCE_TEXT,
        glossary=glossary,
        canonical_glossary_path=canonical,
        terminology_resolver=_resolver(),
        run_id="run",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        decision="approved",
        reviewer="tester",
        note="looks right",
    )

    assert resumed.snapshot.status == "completed"
    assert canonical.read_text(encoding="utf-8").count("道心 -> Dao Heart") == 1
    assert len(provider.requests) > calls_before
    again = resume_repair_session(
        session_dir=paused.session_dir,
        provider=provider,
        source_text=SOURCE_TEXT,
        glossary=glossary,
        canonical_glossary_path=canonical,
        terminology_resolver=_resolver(),
        run_id="run",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        decision="approved",
        reviewer="tester",
        note="repeat",
    )
    assert again.snapshot == resumed.snapshot
    assert again.events == resumed.events


def test_v3_patch_budget_rejects_second_patch_before_executor_mutation(tmp_path: Path) -> None:
    glossary = parse_glossary_text("道心: Dao Heart\n")
    provider = _Provider(
        [
            SubmitPatchAction(old_text="missing", new_text="first", rationale="reject"),
            SubmitPatchAction(
                old_text="Heart of Dao guarded 道心.",
                new_text="Dao Heart guarded the mountain gate.",
                rationale="would otherwise fix QA",
            ),
        ]
    )
    result = run_repair_session(
        provider=provider,
        session_dir=tmp_path,
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=glossary,
        run_id="run",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        max_steps=2,
        max_patch_attempts=1,
        dynamic_tools=False,
    )

    assert result.snapshot.status == "completed"
    assert result.snapshot.episode.final_status == "budget_exhausted"
    assert result.final_text == TRANSLATED_TEXT
    assert result.snapshot.episode.steps[-1].observation.kind == "patch_budget_exhausted"


def test_v3_mutation_budget_counts_normalization_and_patch(tmp_path: Path) -> None:
    glossary = parse_glossary_text("道心: Dao Heart\n")
    provider = _Provider(
        [
            NormalizePunctuationAction(),
            SubmitPatchAction(
                edits=[
                    {"old_text": "Heart of Dao guarded 道心.", "new_text": "Dao Heart guarded the mountain gate."}
                ],
                rationale="would otherwise fix QA",
            ),
        ]
    )
    result = run_repair_session(
        provider=provider,
        session_dir=tmp_path,
        source_text=SOURCE_TEXT,
        translated_text=TRANSLATED_TEXT,
        glossary=glossary,
        run_id="run",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        max_steps=2,
        max_patch_attempts=1,
        dynamic_tools=False,
    )

    assert result.snapshot.episode.final_status == "budget_exhausted"
    assert result.snapshot.episode.steps[-1].observation.kind == "patch_budget_exhausted"


def test_v3_approval_binds_to_reviewed_glossary_path(tmp_path: Path) -> None:
    glossary = parse_glossary_text("道心: Dao Heart\n")
    original = tmp_path / "master_glossary.txt"
    substitute = tmp_path / "substitute_glossary.txt"
    before = "道心 -> Heart of Dao\n"
    original.write_text(before, encoding="utf-8")
    substitute.write_text(before, encoding="utf-8")
    provider = _Provider(
        [PromoteGlossaryTermAction(term="道心", rationale="consensus")]
    )
    paused = run_repair_session(
        provider=provider,
        session_dir=tmp_path / "session",
        source_text=SOURCE_TEXT,
        translated_text="Chapter 1\n\nDao Heart guarded the mountain gate.",
        glossary=glossary,
        canonical_glossary_path=original,
        run_id="run",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        max_steps=2,
        dynamic_tools=False,
    )

    assert paused.snapshot.status == "awaiting_approval"
    resumed = resume_repair_session(
        session_dir=paused.session_dir,
        provider=provider,
        source_text=SOURCE_TEXT,
        glossary=glossary,
        canonical_glossary_path=substitute,
        run_id="run",
        story_slug="demo",
        chapter="0001",
        provider_mode="replay",
        decision="approved",
        reviewer="tester",
        note="wrong target must be blocked",
    )

    assert resumed.snapshot.status == "blocked"
    assert original.read_text(encoding="utf-8") == before
    assert substitute.read_text(encoding="utf-8") == before
