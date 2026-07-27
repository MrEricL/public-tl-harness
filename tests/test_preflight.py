from __future__ import annotations

from pathlib import Path

from agentic_translation.providers_llm import ResponseCache
from agentic_translation.models import TerminologyConsensusConfig
from agentic_translation.preflight import run_preflight


def test_preflight_public_offline_story_passes() -> None:
    report = run_preflight(
        Path("samples/public_demo/story.yaml"),
        chapters=["0001"],
        provider_mode="offline",
    )

    assert report.passed is True
    assert report.chapters == ["0001"]
    assert report.status_counts["fail"] == 0
    assert any(check.name == "source_chapters" and check.status == "ok" for check in report.checks)


def test_preflight_missing_source_chapter_fails() -> None:
    report = run_preflight(
        Path("samples/public_demo/story.yaml"),
        chapters=["9999"],
        provider_mode="offline",
    )

    assert report.passed is False
    assert report.status_counts["fail"] >= 1
    assert any(check.name == "source_chapters" and check.status == "fail" for check in report.checks)


def test_preflight_live_batch_reports_missing_cache_and_openai_env(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)

    report = run_preflight(
        Path("samples/public_demo/story.yaml"),
        chapters=["0001"],
        provider_mode="live",
        translation_provider_name="openai",
        judge_provider_name="openai",
        repair_provider_name="openai",
        record_cache=False,
        cache_dir=None,
    )
    messages = "\n".join(check.message for check in report.checks)

    assert report.passed is False
    assert "--record-cache" in messages
    assert "OPENAI_API_KEY" in messages
    assert "AGENTIC_TRANSLATION_MODEL" in messages


def test_preflight_tool_agent_rejects_offline_mode() -> None:
    report = run_preflight(
        Path("samples/public_demo/story.yaml"),
        chapters=["0001"],
        provider_mode="offline",
        tool_agent_enabled=True,
    )

    assert report.passed is False
    assert any(
        check.name == "tool_agent" and check.status == "fail" and "live or replay" in check.message
        for check in report.checks
    )


def test_preflight_tool_agent_requires_repair_provider_and_explicit_model(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    cache.save(
        "judge",
        {"payload": "x"},
        {"selected_candidate_id": "candidate_a"},
        metadata={"provider": "openai", "model": "fixture-agent"},
    )

    report = run_preflight(
        Path("samples/public_demo/story.yaml"),
        chapters=["0001"],
        provider_mode="replay",
        judge_provider_name="openai",
        repair_provider_name="offline",
        cache_dir=tmp_path,
        tool_agent_enabled=True,
    )
    messages = "\n".join(check.message for check in report.checks if check.name == "tool_agent")

    assert report.passed is False
    assert "non-offline repair provider" in messages
    assert "explicit model_name" in messages


def test_preflight_tool_agent_replay_does_not_require_agent_action_namespace(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    cache.save(
        "repair",
        {"payload": "repair"},
        {"patch_type": "replace_span"},
        metadata={"provider": "openai", "model": "fixture-agent"},
    )

    report = run_preflight(
        Path("samples/public_demo/story.yaml"),
        chapters=["0001"],
        provider_mode="replay",
        translation_provider_name="offline",
        judge_provider_name="offline",
        repair_provider_name="openai",
        cache_dir=tmp_path,
        model_name="fixture-agent",
        tool_agent_enabled=True,
    )

    assert report.passed is True
    cache_check = next(check for check in report.checks if check.name == "cache")
    assert cache_check.status == "ok"
    assert "agent_action" not in cache_check.message


def test_preflight_live_accepts_explicit_model_when_env_model_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)

    report = run_preflight(
        Path("samples/public_demo/story.yaml"),
        chapters=["0001"],
        provider_mode="live",
        judge_provider_name="openai",
        repair_provider_name="offline",
        record_cache=True,
        cache_dir=tmp_path,
        model_name="explicit-model",
    )

    assert report.passed is True
    assert any(check.name == "env" and check.status == "ok" for check in report.checks)


def test_preflight_live_accepts_deepseek_with_deepseek_key_and_explicit_model(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")

    report = run_preflight(
        Path("samples/public_demo/story.yaml"),
        chapters=["0001"],
        provider_mode="live",
        judge_provider_name="deepseek",
        repair_provider_name="offline",
        record_cache=True,
        cache_dir=tmp_path,
        model_name="deepseek-chat",
    )

    assert report.passed is True


def test_preflight_term_consensus_names_both_missing_keys(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)

    report = run_preflight(
        Path("samples/public_demo/story.yaml"),
        chapters=["0001"],
        provider_mode="live",
        repair_provider_name="openai",
        model_name="repair-model",
        record_cache=True,
        cache_dir=tmp_path,
        tool_agent_enabled=True,
        terminology_consensus=TerminologyConsensusConfig(
            enabled=True,
            openai_model="gpt-term",
            deepseek_model="deepseek-term",
        ),
    )
    messages = "\n".join(
        check.message for check in report.checks if check.name == "terminology_consensus"
    )
    assert report.passed is False
    assert "OPENAI_API_KEY" in messages
    assert "DEEPSEEK_API_KEY" in messages
    env_message = next(check.message for check in report.checks if check.name == "env")
    assert "OPENAI_API_KEY" in env_message
    assert "DEEPSEEK_API_KEY" in env_message


def test_preflight_term_consensus_replay_does_not_require_keys(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    cache = ResponseCache(tmp_path)
    cache.save(
        "repair",
        {"payload": "repair"},
        {"patch_type": "replace_span"},
        metadata={"provider": "openai", "model": "repair-model"},
    )
    report = run_preflight(
        Path("samples/public_demo/story.yaml"),
        chapters=["0001"],
        provider_mode="replay",
        repair_provider_name="openai",
        model_name="repair-model",
        cache_dir=tmp_path,
        tool_agent_enabled=True,
        terminology_consensus=TerminologyConsensusConfig(
            enabled=True,
            openai_model="gpt-term",
            deepseek_model="deepseek-term",
        ),
    )
    term_checks = [check for check in report.checks if check.name == "terminology_consensus"]
    assert term_checks and all(check.status == "ok" for check in term_checks)
    assert any(check.name == "providers" and check.status == "ok" for check in report.checks)
    assert any(check.name == "env" and check.status == "ok" for check in report.checks)


def test_preflight_live_accepts_translation_only_provider(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")

    report = run_preflight(
        Path("samples/public_demo/story.yaml"),
        chapters=["0001"],
        provider_mode="live",
        translation_provider_name="deepseek",
        judge_provider_name="offline",
        repair_provider_name="offline",
        record_cache=True,
        cache_dir=tmp_path,
        model_name="deepseek-chat",
    )

    assert report.passed is True
    assert any(check.name == "providers" and check.status == "ok" for check in report.checks)
    assert any(check.name == "env" and check.status == "ok" for check in report.checks)


def test_preflight_live_reports_missing_deepseek_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    report = run_preflight(
        Path("samples/public_demo/story.yaml"),
        chapters=["0001"],
        provider_mode="live",
        judge_provider_name="deepseek",
        repair_provider_name="offline",
        record_cache=True,
        cache_dir=tmp_path,
        model_name="deepseek-chat",
    )
    messages = "\n".join(check.message for check in report.checks)

    assert report.passed is False
    assert "DEEPSEEK_API_KEY" in messages


def test_preflight_replay_requires_existing_cache_dir(tmp_path: Path) -> None:
    missing_cache = tmp_path / "missing_cache"

    report = run_preflight(
        Path("samples/public_demo/story.yaml"),
        chapters=["0001"],
        provider_mode="replay",
        judge_provider_name="openai",
        repair_provider_name="offline",
        cache_dir=missing_cache,
    )

    assert report.passed is False
    assert any(check.name == "cache" and check.status == "fail" for check in report.checks)


def test_preflight_replay_fails_when_cache_index_has_no_entries(tmp_path: Path) -> None:
    cache_dir = tmp_path / "empty_cache"
    cache_dir.mkdir()

    report = run_preflight(
        Path("samples/public_demo/story.yaml"),
        chapters=["0001"],
        provider_mode="replay",
        judge_provider_name="openai",
        repair_provider_name="offline",
        cache_dir=cache_dir,
    )

    assert report.passed is False
    assert any(
        check.name == "cache" and check.status == "fail" and "no indexed entries" in check.message
        for check in report.checks
    )


def test_preflight_replay_requires_indexed_namespaces_for_selected_providers(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    cache.save("judge", {"payload": "x"}, {"selected_candidate_id": "candidate_a"}, metadata={"provider": "openai", "model": "test"})

    report = run_preflight(
        Path("samples/public_demo/story.yaml"),
        chapters=["0001"],
        provider_mode="replay",
        judge_provider_name="openai",
        repair_provider_name="openai",
        cache_dir=tmp_path,
    )

    assert report.passed is False
    assert any(
        check.name == "cache" and check.status == "fail" and "repair" in check.message
        for check in report.checks
    )


def test_preflight_replay_passes_when_required_namespace_is_indexed(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    cache.save("judge", {"payload": "x"}, {"selected_candidate_id": "candidate_a"}, metadata={"provider": "openai", "model": "test"})

    report = run_preflight(
        Path("samples/public_demo/story.yaml"),
        chapters=["0001"],
        provider_mode="replay",
        judge_provider_name="openai",
        repair_provider_name="offline",
        cache_dir=tmp_path,
    )

    assert report.passed is True
    assert any(
        check.name == "cache" and check.status == "ok" and "judge" in check.message
        for check in report.checks
    )


def test_preflight_replay_fails_when_indexed_cache_is_tampered(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    cache.save("judge", {"payload": "x"}, {"selected_candidate_id": "candidate_a"}, metadata={"provider": "openai", "model": "test"})
    cache_file = next(tmp_path.glob("judge_*.json"))
    cache_file.write_text('{"selected_candidate_id": "candidate_b"}', encoding="utf-8")

    report = run_preflight(
        Path("samples/public_demo/story.yaml"),
        chapters=["0001"],
        provider_mode="replay",
        judge_provider_name="openai",
        repair_provider_name="offline",
        cache_dir=tmp_path,
    )

    assert report.passed is False
    assert any(
        check.name == "cache" and check.status == "fail" and "integrity" in check.message
        for check in report.checks
    )
