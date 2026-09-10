from __future__ import annotations

from pathlib import Path

import pytest

from agentic_translation.agent_provider import AgentActionRequest, SHOWCASE_TOOL_SCHEMA_VERSION
from agentic_translation.agent_tools import SHOWCASE_TOOL_REGISTRY
from agentic_translation.models import GlossaryParseResult, StoryConfig, StoryPaths
from agentic_translation.providers_llm import LLMProviderUnavailable, ResponseCache
from agentic_translation.showcase_providers import (
    fixture_profile,
    make_action_provider,
    make_translation_provider,
    resolve_profile,
)


def _request(step_number: int, protocol: str) -> AgentActionRequest:
    return AgentActionRequest(
        episode_id="showcase-0001",
        step_number=step_number,
        story_slug="synthetic-control-demo",
        chapter="0001",
        current_findings=[],
        remaining_steps=8 - step_number,
        remaining_patch_attempts=3,
        prior_steps=[],
        tool_schema_version=SHOWCASE_TOOL_SCHEMA_VERSION,
        tool_protocol=protocol,
        instruction_context={"role": "coordinator", "style": "clear, faithful English"},
    )


@pytest.mark.parametrize("protocol", ["json_prompt", "native_function"])
def test_fixture_action_record_then_replay_uses_indexed_cache(tmp_path: Path, protocol: str) -> None:
    actions = [
        {"tool": "tools.search", "query": "review", "limit": 4},
        {"tool": "finish", "summary": "review complete"},
    ]
    recorded = make_action_provider(
        mode="offline",
        profile={**fixture_profile(), "tool_protocol": protocol},
        cache_dir=tmp_path,
        registry=SHOWCASE_TOOL_REGISTRY,
        actions=actions,
    )

    assert recorded.next_action(_request(1, protocol)).tool == "tools.search"
    assert recorded.next_action(_request(2, protocol)).tool == "finish"
    assert [record.cache_hit for record in recorded.call_records] == [False, False]

    replay = make_action_provider(
        mode="replay",
        profile={**fixture_profile(), "tool_protocol": protocol},
        cache_dir=tmp_path,
        registry=SHOWCASE_TOOL_REGISTRY,
    )
    assert replay.next_action(_request(1, protocol)).tool == "tools.search"
    assert replay.next_action(_request(2, protocol)).tool == "finish"
    assert [record.cache_hit for record in replay.call_records] == [True, True]
    assert all(record.provider == "fixture" for record in replay.call_records)
    assert all(record.model == "showcase-scripted-v1" for record in replay.call_records)
    assert ResponseCache(tmp_path).inspect().integrity_passed is True


def test_fixture_action_without_scripted_step_fails_closed(tmp_path: Path) -> None:
    provider = make_action_provider(
        mode="offline",
        profile=fixture_profile(),
        cache_dir=tmp_path,
        registry=SHOWCASE_TOOL_REGISTRY,
        actions=[{"tool": "finish", "summary": "only step"}],
    )

    with pytest.raises(LLMProviderUnavailable, match="no scripted action"):
        provider.next_action(_request(2, "native_function"))


def test_fixture_translation_record_then_replay_is_network_free(tmp_path: Path) -> None:
    story = StoryConfig(
        slug="synthetic-control-demo",
        title="Synthetic Control Sequence",
        paths=StoryPaths(source_dir=tmp_path, glossary_path=tmp_path / "terms.txt"),
    )
    context = {"style_guide": "Keep operation names consistent.", "profile": fixture_profile()}
    recorded = make_translation_provider(
        mode="offline",
        profile=fixture_profile(),
        cache_dir=tmp_path / "translation-cache",
        translation="The valve opened.",
        instruction_context=context,
    )
    glossary = GlossaryParseResult(entries=[])
    assert recorded.translate("阀门开启。", story=story, glossary=glossary, mode="offline") == (
        "The valve opened."
    )
    assert recorded.call_records[0].cache_hit is False

    replay = make_translation_provider(
        mode="replay",
        profile=fixture_profile(),
        cache_dir=tmp_path / "translation-cache",
        instruction_context=context,
    )
    # The translation task's original mode is part of the canonical payload;
    # replay changes transport mode, while the logical task mode stays stable.
    assert replay.translate("阀门开启。", story=story, glossary=glossary, mode="offline") == (
        "The valve opened."
    )
    assert replay.call_records[0].cache_hit is True

    changed_context = {"style_guide": "Use terse modern prose.", "profile": fixture_profile()}
    changed_replay = make_translation_provider(
        mode="replay",
        profile=fixture_profile(),
        cache_dir=tmp_path / "translation-cache",
        instruction_context=changed_context,
    )
    with pytest.raises(LLMProviderUnavailable, match="No replay cache entry"):
        changed_replay.translate("阀门开启。", story=story, glossary=glossary, mode="offline")


def test_resolve_profile_uses_provider_environment_and_redacts_credentials(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    profile = resolve_profile("deepseek")
    assert profile["provider"] == "deepseek"
    assert profile["model"] == "deepseek-reasoner"
    assert profile["model_source"] == "environment"
    assert profile["tool_protocol"] == "native_function"
    assert profile["max_output_tokens"] == 2048
    assert profile["temperature"] == 0.0
    assert profile["request_timeout_seconds"] == 60.0
    assert "api_key" not in profile


def test_resolve_profile_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown provider profile"):
        resolve_profile("anthropic")
