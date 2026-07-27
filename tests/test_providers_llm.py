from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agentic_translation.models import GlossaryParseResult, TranslationCandidate
from agentic_translation.providers_llm import (
    LLMJudgeProvider,
    LLMProviderUnavailable,
    LiveJudgeResponse,
    LiveRepairResponse,
    ResponseCache,
    _OpenAIJSONProvider,
    inspect_response_cache,
    is_openai_compatible_provider,
    probe_live_provider,
    required_live_provider_config,
)
from agentic_translation.translate import get_judge_provider


def test_live_repair_response_rejects_invalid_patch_type() -> None:
    with pytest.raises(ValidationError):
        LiveRepairResponse.model_validate(
            {
                "patch_type": "rewrite_everything",
                "old_text": "old",
                "new_text": "new",
            }
        )


def test_live_repair_response_rejects_empty_replace_span_old_text() -> None:
    with pytest.raises(ValidationError):
        LiveRepairResponse.model_validate(
            {
                "patch_type": "replace_span",
                "old_text": "",
                "new_text": "new",
            }
        )


def test_live_judge_response_validates_quality_scores() -> None:
    with pytest.raises(ValidationError):
        LiveJudgeResponse.model_validate(
            {
                "selected_candidate_id": "candidate_a",
                "quality_scores": {
                    "candidate_a": {
                        "faithfulness": 11,
                        "fluency": 8,
                        "rationale": "too high",
                    }
                },
            }
        )

def test_live_judge_rejects_unknown_selected_candidate() -> None:
    class BadJudgeProvider(LLMJudgeProvider):
        def _call_json(self, *, namespace, payload, messages):  # noqa: ANN001, ANN202
            return {
                "selected_candidate_id": "missing_candidate",
                "quality_scores": {},
                "rationale": "I picked something outside the candidate set.",
            }

    provider = BadJudgeProvider(provider_mode="replay", cache_dir=None)

    with pytest.raises(LLMProviderUnavailable, match="unknown candidate"):
        provider.judge(
            source_text="source",
            candidates=[TranslationCandidate(candidate_id="candidate_a", text="Candidate A")],
            glossary=GlossaryParseResult(entries=[]),
            seed=7,
        )


def test_response_cache_records_auditable_index_without_payload_body(tmp_path) -> None:
    cache = ResponseCache(tmp_path)
    payload = {"source_text": "天道 secret payload"}
    response = {"translation": "Heavenly Dao secret response"}

    cache.save(
        "judge",
        payload,
        response,
        metadata={"provider": "openai", "model": "test-model"},
    )

    index_path = tmp_path / "cache_index.jsonl"
    assert index_path.exists()
    index_text = index_path.read_text(encoding="utf-8")
    assert "secret payload" not in index_text
    assert "secret response" not in index_text

    report = inspect_response_cache(tmp_path)
    assert report.total_entries == 1
    assert report.by_namespace == {"judge": 1}
    entry = report.entries[0]
    assert entry.namespace == "judge"
    assert entry.provider == "openai"
    assert entry.model == "test-model"
    assert entry.cache_file.startswith("judge_")
    assert len(entry.payload_sha256) == 64
    assert len(entry.response_sha256) == 64
    assert report.integrity_passed is True
    assert report.valid_entries == 1
    assert report.invalid_entries == 0
    assert report.integrity_issues == []


def test_cache_inspection_detects_tampered_cached_response(tmp_path) -> None:
    cache = ResponseCache(tmp_path)
    cache.save(
        "judge",
        {"payload": "x"},
        {"selected_candidate_id": "candidate_a"},
        metadata={"provider": "openai", "model": "test-model"},
    )
    cache_file = next(tmp_path.glob("judge_*.json"))
    cache_file.write_text('{"selected_candidate_id": "candidate_b"}', encoding="utf-8")

    report = inspect_response_cache(tmp_path)

    assert report.integrity_passed is False
    assert report.valid_entries == 0
    assert report.invalid_entries == 1
    assert report.integrity_issues[0].issue_type == "response_digest_mismatch"
    assert report.integrity_issues[0].cache_file == cache_file.name


def test_cache_inspection_detects_missing_indexed_cache_file(tmp_path) -> None:
    cache = ResponseCache(tmp_path)
    cache.save(
        "judge",
        {"payload": "x"},
        {"selected_candidate_id": "candidate_a"},
        metadata={"provider": "openai", "model": "test-model"},
    )
    cache_file = next(tmp_path.glob("judge_*.json"))
    cache_file.unlink()

    report = inspect_response_cache(tmp_path)

    assert report.integrity_passed is False
    assert report.valid_entries == 0
    assert report.invalid_entries == 1
    assert report.integrity_issues[0].issue_type == "missing_file"
    assert report.integrity_issues[0].cache_file == cache_file.name


class _Message:
    content = '{"ok": true}'


class _Choice:
    message = _Message()


class _Response:
    choices = [_Choice()]


class _FakeCompletions:
    def __init__(self, failures_before_success: int) -> None:
        self.failures_before_success = failures_before_success
        self.calls = 0
        self.last_kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> _Response:
        self.calls += 1
        self.last_kwargs = dict(kwargs)
        if self.calls <= self.failures_before_success:
            raise RuntimeError("temporary outage")
        return _Response()


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = _FakeChat(completions)


class _StatusCodeError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class _AlwaysStatusCodeCompletions:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.calls = 0

    def create(self, **kwargs: object) -> _Response:
        self.calls += 1
        raise _StatusCodeError(self.status_code, f"provider status {self.status_code}")


def test_live_json_provider_retries_and_records_cache(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("AGENTIC_TRANSLATION_MODEL", "test-model")
    completions = _FakeCompletions(failures_before_success=1)
    provider = _OpenAIJSONProvider(
        provider_mode="live",
        cache_dir=tmp_path,
        record_cache=True,
        client_factory=lambda **kwargs: _FakeClient(completions),
        sleep=lambda seconds: None,
        max_retries=2,
    )

    result = provider._call_json(namespace="test", payload={"a": 1}, messages=[{"role": "user", "content": "{}"}])

    assert result == {"ok": True}
    assert completions.calls == 2
    assert list(tmp_path.glob("test_*.json"))


def test_live_json_provider_does_not_retry_non_retryable_status(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("AGENTIC_TRANSLATION_MODEL", "test-model")
    completions = _AlwaysStatusCodeCompletions(status_code=402)
    provider = _OpenAIJSONProvider(
        provider_mode="live",
        cache_dir=tmp_path,
        client_factory=lambda **kwargs: _FakeClient(completions),  # type: ignore[arg-type]
        sleep=lambda seconds: (_ for _ in ()).throw(AssertionError("non-retryable errors should not sleep")),
        max_retries=2,
    )

    with pytest.raises(LLMProviderUnavailable, match="without retry.*provider status 402"):
        provider._call_json(namespace="test", payload={"a": 1}, messages=[{"role": "user", "content": "{}"}])

    assert completions.calls == 1


def test_live_json_provider_accepts_explicit_model_without_env_model(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)
    completions = _FakeCompletions(failures_before_success=0)
    provider = _OpenAIJSONProvider(
        provider_mode="live",
        cache_dir=tmp_path,
        record_cache=True,
        model_name="explicit-model",
        client_factory=lambda **kwargs: _FakeClient(completions),
    )

    result = provider._call_json(namespace="test", payload={"a": 1}, messages=[{"role": "user", "content": "{}"}])

    assert result == {"ok": True}
    assert completions.last_kwargs["model"] == "explicit-model"
    report = inspect_response_cache(tmp_path)
    assert report.entries[0].model == "explicit-model"


def test_replay_uses_cache_without_client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("AGENTIC_TRANSLATION_MODEL", "test-model")
    payload = {"a": 1}
    messages = [{"role": "user", "content": "{}"}]
    live = _OpenAIJSONProvider(
        provider_mode="live",
        cache_dir=tmp_path,
        record_cache=True,
        client_factory=lambda **kwargs: _FakeClient(_FakeCompletions(failures_before_success=0)),
    )
    live._call_json(namespace="test", payload=payload, messages=messages)
    replay = _OpenAIJSONProvider(
        provider_mode="replay",
        cache_dir=tmp_path,
        client_factory=lambda **kwargs: (_ for _ in ()).throw(AssertionError("client should not be used")),
    )

    assert replay._call_json(namespace="test", payload=payload, messages=messages) == {"ok": True}
    assert len(replay.call_records) == 1
    assert replay.call_records[0].namespace == "test"
    assert replay.call_records[0].cache_hit is True
    assert replay.call_records[0].payload_sha256 == live.call_records[0].payload_sha256
    assert replay.call_records[0].response_sha256 == live.call_records[0].response_sha256


def test_live_json_provider_records_provider_call_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("AGENTIC_TRANSLATION_MODEL", "test-model")
    provider = _OpenAIJSONProvider(
        provider_mode="live",
        cache_dir=tmp_path,
        record_cache=True,
        client_factory=lambda **kwargs: _FakeClient(_FakeCompletions(failures_before_success=0)),
    )

    provider._call_json(namespace="judge", payload={"a": 1}, messages=[{"role": "user", "content": "{}"}])

    report = inspect_response_cache(tmp_path)
    assert len(provider.call_records) == 1
    record = provider.call_records[0]
    assert record.role == "judge"
    assert record.namespace == "judge"
    assert record.provider == "openai"
    assert record.model == "test-model"
    assert record.cache_hit is False
    assert record.payload_sha256 == report.entries[0].payload_sha256
    assert record.response_sha256 == report.entries[0].response_sha256
    assert record.cache_file == report.entries[0].cache_file


def test_deepseek_json_provider_uses_deepseek_env_and_base_url(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    completions = _FakeCompletions(failures_before_success=0)
    client_kwargs: dict[str, str] = {}

    def client_factory(**kwargs: str) -> _FakeClient:
        client_kwargs.update(kwargs)
        return _FakeClient(completions)

    provider = _OpenAIJSONProvider(
        provider_mode="live",
        cache_dir=tmp_path,
        record_cache=True,
        provider_name="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
        model_env="DEEPSEEK_MODEL",
        base_url_env="DEEPSEEK_BASE_URL",
        default_base_url="https://api.deepseek.com",
        client_factory=client_factory,
    )

    provider._call_json(namespace="judge", payload={"a": 1}, messages=[{"role": "user", "content": "{}"}])

    assert client_kwargs == {"api_key": "deepseek-test-key", "base_url": "https://api.deepseek.com"}
    assert completions.last_kwargs["model"] == "deepseek-chat"
    report = inspect_response_cache(tmp_path)
    assert report.entries[0].provider == "deepseek"
    assert report.entries[0].model == "deepseek-chat"
    assert provider.call_records[0].provider == "deepseek"


def test_probe_live_provider_uses_tiny_probe_payload(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    completions = _FakeCompletions(failures_before_success=0)
    client_kwargs: dict[str, str] = {}

    def client_factory(**kwargs: str) -> _FakeClient:
        client_kwargs.update(kwargs)
        return _FakeClient(completions)

    result = probe_live_provider(
        provider_name="deepseek",
        cache_dir=tmp_path,
        record_cache=True,
        model_name="deepseek-chat",
        client_factory=client_factory,
    )

    assert result.provider == "deepseek"
    assert result.model == "deepseek-chat"
    assert result.cache_hit is False
    assert result.cache_file.startswith("probe_")
    assert result.response == {"ok": True}
    assert client_kwargs == {"api_key": "deepseek-test-key", "base_url": "https://api.deepseek.com"}
    assert completions.last_kwargs["model"] == "deepseek-chat"
    assert "provider_probe" in completions.last_kwargs["messages"][1]["content"]
    report = inspect_response_cache(tmp_path)
    assert report.by_namespace == {"probe": 1}


def test_replay_rejects_tampered_cached_response(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("AGENTIC_TRANSLATION_MODEL", "test-model")
    payload = {"a": 1}
    messages = [{"role": "user", "content": "{}"}]
    live = _OpenAIJSONProvider(
        provider_mode="live",
        cache_dir=tmp_path,
        record_cache=True,
        client_factory=lambda **kwargs: _FakeClient(_FakeCompletions(failures_before_success=0)),
    )
    live._call_json(namespace="test", payload=payload, messages=messages)
    cache_file = next(tmp_path.glob("test_*.json"))
    cache_file.write_text('{"ok": false}', encoding="utf-8")
    replay = _OpenAIJSONProvider(
        provider_mode="replay",
        cache_dir=tmp_path,
        client_factory=lambda **kwargs: (_ for _ in ()).throw(AssertionError("client should not be used")),
    )

    with pytest.raises(LLMProviderUnavailable, match="integrity"):
        replay._call_json(namespace="test", payload=payload, messages=messages)


def test_live_json_provider_raises_after_retry_exhaustion(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("AGENTIC_TRANSLATION_MODEL", "test-model")
    provider = _OpenAIJSONProvider(
        provider_mode="live",
        cache_dir=tmp_path,
        client_factory=lambda **kwargs: _FakeClient(_FakeCompletions(failures_before_success=99)),
        sleep=lambda seconds: None,
        max_retries=1,
    )

    with pytest.raises(LLMProviderUnavailable, match="temporary outage"):
        provider._call_json(namespace="test", payload={"a": 1}, messages=[{"role": "user", "content": "{}"}])


def test_deepseek_is_supported_as_openai_compatible_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)

    assert is_openai_compatible_provider("deepseek") is True
    assert required_live_provider_config("deepseek") == ["DEEPSEEK_API_KEY", "DEEPSEEK_MODEL"]

    provider = get_judge_provider("deepseek", provider_mode="live", model_name="deepseek-chat")

    assert provider.provider_name == "deepseek"
    assert provider.model_name == "deepseek-chat"


def test_deepseek_provider_probe_reports_missing_key(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # noqa: ANN001
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("AGENTIC_TRANSLATION_MODEL", raising=False)

    with pytest.raises(LLMProviderUnavailable, match="DEEPSEEK_API_KEY"):
        probe_live_provider(
            provider_name="deepseek",
            cache_dir=tmp_path / "cache",
            model_name="deepseek-chat",
        )


def test_live_provider_api_errors_are_wrapped_as_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    class FakeStatusError(Exception):
        status_code = 402

        def __str__(self) -> str:
            return "Error code: 402 - Insufficient Balance"

    class FakeCompletions:
        def create(self, **kwargs):  # noqa: ANN003, ANN202
            raise FakeStatusError()

    class FakeClient:
        chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: FakeClient()))

    provider = _OpenAIJSONProvider(
        provider_mode="live",
        cache_dir=tmp_path / "cache",
        provider_name="deepseek",
        model_name="deepseek-chat",
    )

    with pytest.raises(LLMProviderUnavailable, match="Live provider call failed"):
        provider._call_json(namespace="judge", payload={"task": "test"}, messages=[{"role": "user", "content": "{}"}])
