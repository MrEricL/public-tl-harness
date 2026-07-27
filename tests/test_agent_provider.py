from __future__ import annotations

import json
import traceback
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_translation.agent_provider import (
    BASE_TOOL_SCHEMA_VERSION,
    CanonicalPayloadError,
    MAX_PAYLOAD_CHARS,
    MAX_NESTED_ITEMS,
    AgentActionRequest,
    AgentActionValidationError,
    LLMAgentActionProvider,
    PriorObservableStep,
    build_agent_action_messages,
)
from agentic_translation.agent_models import ResolveTerminologyAction
from agentic_translation.agent_repair import run_repair_episode
from agentic_translation.glossary import load_glossary
from agentic_translation.qa import run_translation_qa
from agentic_translation.story import load_story_config
from agentic_translation.providers_llm import LLMProviderUnavailable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_STORY = PROJECT_ROOT / "samples/agentic_repair_demo/story.yaml"
GOLDEN_RUN_ID = "agentic_repair_demo_replay"


def test_golden_replay_fixture_runs_the_reviewed_five_action_episode(tmp_path) -> None:
    config = load_story_config(GOLDEN_STORY)
    source_path = config.paths.source_dir / "0001.txt"
    glossary_path = config.paths.glossary_path
    dirty_path = config.paths.expected_dir / "dirty_translation.txt"
    assert config.public_safe is True
    assert config.slug == "agentic_repair_demo"
    assert config.chapter_ids == ["0001"]
    assert config.paths.expected_dir is not None
    assert config.paths.runs_dir.name == "runs"

    source_text = source_path.read_text(encoding="utf-8")
    translated_text = dirty_path.read_text(encoding="utf-8")
    glossary = load_glossary(glossary_path)
    assert glossary.entries[0].target == "Dao Heart"
    assert glossary.entries[0].candidates == ["Dao Heart", "Heart of Dao"]
    assert glossary.entries[0].blocked_variants == ["Heart of Dao"]

    initial_qa = run_translation_qa(
        run_id=GOLDEN_RUN_ID,
        story_slug=config.slug,
        chapter="0001",
        source_text=source_text,
        translated_text=translated_text,
        glossary=glossary,
    )
    assert initial_qa.summary.total_findings == 3
    assert {finding.check_id for finding in initial_qa.findings} == {
        "residual_chinese",
        "blocked_glossary_variant",
        "glossary_required",
    }
    residual = next(finding for finding in initial_qa.findings if finding.check_id == "residual_chinese")
    assert residual.found == "道心"

    provider = LLMAgentActionProvider(
        provider_mode="replay",
        provider_name="openai",
        model_name="fixture-agent-v1",
        cache_dir=GOLDEN_STORY.parent / "replay_cache",
    )
    result = run_repair_episode(
        provider=provider,
        episode_path=tmp_path / "episode.json",
        source_text=source_text,
        translated_text=translated_text,
        glossary=glossary,
        run_id=GOLDEN_RUN_ID,
        story_slug=config.slug,
        chapter="0001",
        provider_mode="replay",
        max_steps=5,
        max_patch_attempts=2,
    )

    assert [step.action["tool"] for step in result.episode.steps] == [
        "lookup_glossary",
        "submit_patch",
        "read_source_context",
        "submit_patch",
        "finish",
    ]
    assert result.episode.steps[1].observation.kind == "patch_rejected"
    assert result.episode.steps[1].observation.ok is False
    assert result.episode.steps[3].observation.kind == "patch_accepted"
    assert result.episode.steps[3].observation.ok is True
    assert result.episode.final_status == "verified"
    assert result.episode.final_qa is not None
    assert result.episode.final_qa.summary.total_findings == 0
    assert result.final_qa.summary.total_findings == 0
    assert result.final_text == "Chapter 1\n\nDao Heart guarded the mountain gate.\n"

    assert len(provider.call_records) == 5
    assert all(record.cache_hit is True for record in provider.call_records)
    cache_report = provider.cache.inspect()
    assert cache_report.integrity_passed is True
    assert cache_report.total_entries == 5
    assert cache_report.valid_entries == 5
    assert cache_report.invalid_entries == 0
    assert cache_report.by_namespace == {"agent_action": 5}


def test_golden_replay_fixture_loads_and_runs_outside_project_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_story_config(GOLDEN_STORY)
    assert config.paths.source_dir == GOLDEN_STORY.parent / "source"
    assert config.paths.glossary_path == GOLDEN_STORY.parent / "terms/master_glossary.txt"
    assert config.paths.expected_dir == GOLDEN_STORY.parent / "expected"
    assert config.paths.runs_dir == GOLDEN_STORY.parent / "runs"

    source_text = (config.paths.source_dir / "0001.txt").read_text(encoding="utf-8")
    translated_text = (config.paths.expected_dir / "dirty_translation.txt").read_text(encoding="utf-8")
    glossary = load_glossary(config.paths.glossary_path)
    provider = LLMAgentActionProvider(
        provider_mode="replay",
        provider_name="openai",
        model_name="fixture-agent-v1",
        cache_dir=GOLDEN_STORY.parent / "replay_cache",
    )

    result = run_repair_episode(
        provider=provider,
        episode_path=tmp_path / "episode.json",
        source_text=source_text,
        translated_text=translated_text,
        glossary=glossary,
        run_id=GOLDEN_RUN_ID,
        story_slug=config.slug,
        chapter="0001",
        provider_mode="replay",
        max_steps=5,
        max_patch_attempts=2,
    )

    assert result.episode.final_status == "verified"
    assert result.final_qa.summary.total_findings == 0
    assert len(provider.call_records) == 5
    assert all(record.cache_hit is True for record in provider.call_records)
    cache_report = provider.cache.inspect()
    assert {
        record.payload_sha256 for record in provider.call_records
    } == {
        entry.payload_sha256 for entry in cache_report.entries
    }


def make_action_request() -> AgentActionRequest:
    return AgentActionRequest(
        episode_id="episode-1",
        step_number=2,
        story_slug="demo-story",
        chapter="0001",
        current_findings=[
            {
                "check_id": "glossary_consistency",
                "severity": "error",
                "location": {"chapter": "0001", "paragraph_index": 1},
                "message": "Use the canonical glossary term.",
            }
        ],
        remaining_steps=3,
        remaining_patch_attempts=1,
        prior_steps=[
            {
                "sequence": 1,
                "action": {"tool": "lookup_glossary", "term": "道心"},
                "observation": {"ok": True, "kind": "glossary_lookup", "message": "looked up"},
            }
        ],
    )


def test_canonical_payload_is_stable_and_json_safe() -> None:
    first = make_action_request()
    second = AgentActionRequest.model_validate(
        {
            **first.model_dump(),
            "current_findings": [
                {
                    "message": "Use the canonical glossary term.",
                    "location": {"paragraph_index": 1, "chapter": "0001"},
                    "severity": "error",
                    "check_id": "glossary_consistency",
                }
            ],
        }
    )

    first_payload = first.canonical_payload()
    second_payload = second.canonical_payload()

    assert first_payload == second_payload
    assert json.dumps(first_payload, ensure_ascii=False, sort_keys=True)
    assert first_payload["tool_schema_version"]
    assert first_payload["current_findings"]
    assert first_payload["prior_steps"]


def test_canonical_payload_bounds_serialized_context() -> None:
    request = AgentActionRequest(
        episode_id="episode-1",
        step_number=1,
        story_slug="demo-story",
        chapter="0001",
        current_findings=[{str(i): "x" * 1200 for i in range(32)} for _ in range(32)],
        remaining_steps=5,
        remaining_patch_attempts=2,
        prior_steps=[
            {
                "sequence": index + 1,
                "action": {"tool": "get_qa_findings"},
                "observation": {"ok": True, "kind": "qa_findings", "message": "ok"},
            }
            for index in range(24)
        ],
    )

    payload = request.canonical_payload()

    assert len(json.dumps(payload, ensure_ascii=False, sort_keys=True)) <= MAX_PAYLOAD_CHARS
    assert payload["context_truncated"] is True


def test_prior_steps_reject_hidden_reasoning_fields() -> None:
    request_data = make_action_request().model_dump()
    request_data["prior_steps"] = [
        {
            "sequence": 1,
            "action": {"tool": "get_qa_findings"},
            "observation": {"ok": True, "kind": "qa_findings", "message": "ok"},
            "hidden_reasoning": "secret chain of thought",
        }
    ]

    with pytest.raises(ValidationError):
        AgentActionRequest.model_validate(request_data)


def test_prior_observable_step_is_typed_and_serializable() -> None:
    step = PriorObservableStep(
        sequence=1,
        action={"tool": "get_qa_findings"},
        observation={"ok": True, "kind": "qa_findings", "message": "ok"},
    )

    request = make_action_request().model_copy(update={"prior_steps": [step]})
    payload = request.canonical_payload()

    assert payload["prior_steps"] == [
        {
            "action": {"tool": "get_qa_findings"},
            "observation": {"data": {}, "kind": "qa_findings", "message": "ok", "ok": True},
            "sequence": 1,
        }
    ]


def test_canonical_payload_rejects_unsupported_values_without_stringifying() -> None:
    class UnstableValue:
        def __str__(self) -> str:
            return f"unstable-{id(self)}"

    request = make_action_request().model_copy(
        update={"current_findings": [{"unstable": UnstableValue()}]}
    )

    with pytest.raises(ValueError, match="Unsupported JSON value") as raised:
        request.canonical_payload()

    assert "unstable-" not in str(raised.value)


def test_canonical_payload_digest_distinguishes_dropped_preview_values() -> None:
    first = make_action_request().model_copy(
        update={
            "current_findings": [
                {"overflow": ["same"] * MAX_NESTED_ITEMS + ["first"]}
            ]
        }
    )
    second = make_action_request().model_copy(
        update={
            "current_findings": [
                {"overflow": ["same"] * MAX_NESTED_ITEMS + ["second"]}
            ]
        }
    )

    first_payload = first.canonical_payload()
    second_payload = second.canonical_payload()

    assert first_payload["current_findings"] == second_payload["current_findings"]
    assert first_payload["current_findings_sha256"] != second_payload["current_findings_sha256"]


def test_canonical_payload_compact_serialization_stays_within_limit() -> None:
    request = AgentActionRequest(
        episode_id="episode-1",
        step_number=1,
        story_slug="demo-story",
        chapter="0001",
        current_findings=[{str(i): "x" * 1200 for i in range(32)} for _ in range(32)],
        remaining_steps=5,
        remaining_patch_attempts=2,
        prior_steps=[
            {
                "sequence": index + 1,
                "action": {"tool": "get_qa_findings"},
                "observation": {"ok": True, "kind": "qa_findings", "message": "ok"},
            }
            for index in range(24)
        ],
    )

    payload = request.canonical_payload()

    assert len(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))) <= MAX_PAYLOAD_CHARS


def test_replay_requires_indexed_agent_action_cache_entry(tmp_path) -> None:
    provider = LLMAgentActionProvider(
        provider_mode="replay",
        provider_name="openai",
        model_name="fixture-agent-v1",
        cache_dir=tmp_path,
    )
    request = make_action_request()
    provider.cache.save(
        "agent_action",
        request.canonical_payload(),
        {"tool": "get_qa_findings"},
        metadata={"provider": "openai", "model": "fixture-agent-v1"},
    )
    (tmp_path / "cache_index.jsonl").unlink()

    with pytest.raises(LLMProviderUnavailable, match="index"):
        provider.next_action(request)


def test_provider_rejects_unknown_mode_before_live_configuration_guard(tmp_path) -> None:
    with pytest.raises(ValueError, match="provider_mode"):
        LLMAgentActionProvider(provider_mode="replaay")


def test_canonical_payload_rejects_excessive_nesting_safely() -> None:
    nested: dict[str, object] = {"leaf": "value"}
    for _ in range(1100):
        nested = {"nested": nested}

    request = make_action_request().model_copy(update={"current_findings": [nested]})

    with pytest.raises(CanonicalPayloadError, match="nesting"):
        request.canonical_payload()


def test_replay_rejects_cache_entry_with_mismatched_provider_metadata(tmp_path) -> None:
    provider = LLMAgentActionProvider(
        provider_mode="replay",
        provider_name="openai",
        model_name="fixture-agent-v1",
        cache_dir=tmp_path,
    )
    request = make_action_request()
    provider.cache.save(
        "agent_action",
        request.canonical_payload(),
        {"tool": "get_qa_findings"},
        metadata={"provider": "deepseek", "model": "other-model"},
    )

    with pytest.raises(LLMProviderUnavailable, match="metadata"):
        provider.next_action(request)


def test_invalid_action_traceback_does_not_include_raw_response(tmp_path) -> None:
    provider = LLMAgentActionProvider(
        provider_mode="replay",
        provider_name="openai",
        model_name="fixture-agent-v1",
        cache_dir=tmp_path,
    )
    request = make_action_request()
    marker = "raw-provider-secret-marker"
    provider.cache.save(
        "agent_action",
        request.canonical_payload(),
        {"tool": "submit_patch", "old_text": marker},
        metadata={"provider": "openai", "model": "fixture-agent-v1"},
    )

    with pytest.raises(AgentActionValidationError) as raised:
        provider.next_action(request)

    formatted = "".join(traceback.format_exception(raised.value))
    assert marker not in formatted


def test_live_agent_provider_requires_explicit_recorded_cache(tmp_path) -> None:
    with pytest.raises(ValueError, match="cache_dir.*record_cache"):
        LLMAgentActionProvider(provider_mode="live", record_cache=True)
    with pytest.raises(ValueError, match="cache_dir.*record_cache"):
        LLMAgentActionProvider(provider_mode="live", cache_dir=tmp_path, record_cache=False)

    provider = LLMAgentActionProvider(provider_mode="live", cache_dir=tmp_path, record_cache=True)
    assert provider.provider_mode == "live"


def test_request_accepts_v2_and_rejects_unknown_tool_schema_version() -> None:
    request_data = make_action_request().model_dump()
    request_data["tool_schema_version"] = "agent-tools.v2"

    assert AgentActionRequest.model_validate(request_data).tool_schema_version == "agent-tools.v2"
    request_data["tool_schema_version"] = "agent-tools.v3"
    with pytest.raises(ValidationError):
        AgentActionRequest.model_validate(request_data)


def test_v1_and_v2_contracts_are_distinct_and_v1_rejects_resolve_response(tmp_path) -> None:
    provider = LLMAgentActionProvider(
        provider_mode="replay",
        provider_name="openai",
        model_name="fixture-agent-v1",
        cache_dir=tmp_path,
    )
    request = make_action_request()
    assert request.tool_schema_version == BASE_TOOL_SCHEMA_VERSION
    assert [item["tool"] for item in request.canonical_payload()["tool_schema"]] == [
        "get_qa_findings",
        "read_source_context",
        "read_translation_context",
        "lookup_glossary",
        "submit_patch",
        "escalate",
        "finish",
    ]
    provider.cache.save(
        "agent_action",
        request.canonical_payload(),
        {"tool": "resolve_terminology", "term": "道心"},
        metadata={"provider": "openai", "model": "fixture-agent-v1"},
    )
    with pytest.raises(AgentActionValidationError, match="agent-tools.v1"):
        provider.next_action(request)


def test_v2_provider_accepts_resolve_response(tmp_path) -> None:
    provider = LLMAgentActionProvider(
        provider_mode="replay",
        provider_name="openai",
        model_name="fixture-agent-v2",
        cache_dir=tmp_path,
    )
    request_data = make_action_request().model_dump()
    request_data["tool_schema_version"] = "agent-tools.v2"
    request = AgentActionRequest.model_validate(request_data)
    provider.cache.save(
        "agent_action",
        request.canonical_payload(),
        {"tool": "resolve_terminology", "term": "道心"},
        metadata={"provider": "openai", "model": "fixture-agent-v2"},
    )
    action = provider.next_action(request)
    assert isinstance(action, ResolveTerminologyAction)


def test_replay_returns_typed_cached_action_without_constructing_client(tmp_path) -> None:
    provider = LLMAgentActionProvider(
        provider_mode="replay",
        provider_name="openai",
        model_name="fixture-agent-v1",
        cache_dir=tmp_path,
        client_factory=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("replay must not construct a live client")
        ),
    )
    request = make_action_request()
    provider.cache.save(
        "agent_action",
        request.canonical_payload(),
        {"tool": "get_qa_findings"},
        metadata={"provider": "openai", "model": "fixture-agent-v1"},
    )

    action = provider.next_action(request)

    assert action.tool == "get_qa_findings"
    assert provider.call_records[-1].cache_hit is True


def test_replay_cache_miss_is_fatal_before_client_construction(tmp_path) -> None:
    constructed = False

    def client_factory(**kwargs):
        nonlocal constructed
        constructed = True
        raise AssertionError("replay must fail before constructing a live client")

    provider = LLMAgentActionProvider(
        provider_mode="replay",
        provider_name="openai",
        model_name="fixture-agent-v1",
        cache_dir=tmp_path,
        client_factory=client_factory,
    )

    with pytest.raises(LLMProviderUnavailable, match="No replay cache entry"):
        provider.next_action(make_action_request())

    assert constructed is False
    assert provider.call_records == []


def test_invalid_cached_action_raises_bounded_validation_error(tmp_path) -> None:
    provider = LLMAgentActionProvider(
        provider_mode="replay",
        provider_name="openai",
        model_name="fixture-agent-v1",
        cache_dir=tmp_path,
    )
    request = make_action_request()
    invalid_response = {
        "tool": "submit_patch",
        "old_text": "old",
        # Missing new_text and rationale; this response also contains a value
        # that must not be copied unboundedly into the raised exception.
        "secret": "do-not-leak" * 1000,
    }
    provider.cache.save(
        "agent_action",
        request.canonical_payload(),
        invalid_response,
        metadata={"provider": "openai", "model": "fixture-agent-v1"},
    )

    with pytest.raises(AgentActionValidationError, match="schema validation") as raised:
        provider.next_action(request)

    assert raised.value.response["tool"] == "submit_patch"
    assert len(json.dumps(raised.value.response)) <= 4096
    assert "do-not-leak" not in str(raised.value)
    assert provider.call_records[-1].namespace == "agent_action"
    assert provider.call_records[-1].cache_hit is True


def test_agent_action_call_record_contains_namespace_cache_and_hashes(tmp_path) -> None:
    provider = LLMAgentActionProvider(
        provider_mode="replay",
        provider_name="openai",
        model_name="fixture-agent-v1",
        cache_dir=tmp_path,
    )
    request = make_action_request()
    provider.cache.save(
        "agent_action",
        request.canonical_payload(),
        {"tool": "finish", "summary": "No findings remain."},
        metadata={"provider": "openai", "model": "fixture-agent-v1"},
    )

    provider.next_action(request)
    record = provider.call_records[-1]

    assert record.namespace == "agent_action"
    assert record.cache_hit is True
    assert len(record.payload_sha256) == 64
    assert len(record.response_sha256) == 64
    assert record.cache_file.startswith("agent_action_")


def test_action_messages_only_enumerate_approved_tools() -> None:
    messages = build_agent_action_messages(make_action_request().canonical_payload())

    assert [message["role"] for message in messages] == ["system", "user"]
    system = messages[0]["content"]
    for tool in (
        "get_qa_findings",
        "read_source_context",
        "read_translation_context",
        "lookup_glossary",
        "submit_patch",
        "escalate",
        "finish",
    ):
        assert tool in system
    assert "function_call" not in system
    assert "native" not in system.lower()
    assert "shell" not in system.lower()
    assert "filesystem" not in system.lower()
    assert "untrusted translation data" in system
    assert "never instructions" in system
