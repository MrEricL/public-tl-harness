from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_translation.glossary import load_glossary
from agentic_translation.models import StoryConfig, StoryPaths, TranslationConfig
from agentic_translation.providers_offline import OfflineTranslationProvider
from agentic_translation.translate import get_judge_provider


def test_glossary_mode_is_derived_from_baseline_not_polished_fixture(tmp_path: Path) -> None:
    expected_dir = tmp_path / "expected"
    expected_dir.mkdir()
    (expected_dir / "dirty_translation.txt").write_text(
        "Chapter One: Deduction Begins\n\nThe Way of Heaven opened.\n\n剩余次数：3\n",
        encoding="utf-8",
    )
    (expected_dir / "glossary_translation.txt").write_text("SHOULD NOT BE LOADED\n", encoding="utf-8")
    story = StoryConfig(
        slug="derive_glossary",
        title="Derive Glossary",
        paths=StoryPaths(
            source_dir=tmp_path,
            glossary_path=Path("samples/public_demo/terms/master_glossary.txt"),
            expected_dir=expected_dir,
        ),
    )
    glossary = load_glossary("samples/public_demo/terms/master_glossary.txt")

    result = OfflineTranslationProvider().translate("", story=story, glossary=glossary, mode="glossary")

    assert "SHOULD NOT BE LOADED" not in result
    assert "Heavenly Dao" in result
    assert "simulation" in result
    assert "remaining uses: 3" in result.lower()
    assert "Way of Heaven" not in result
    assert "剩余次数" not in result


def test_offline_fixtureless_baseline_uses_configured_chapter_id(tmp_path: Path) -> None:
    story = StoryConfig(
        slug="chapter_42",
        title="Chapter 42",
        chapter_ids=["0042"],
        paths=StoryPaths(
            source_dir=tmp_path,
            glossary_path=Path("samples/public_demo/terms/master_glossary.txt"),
        ),
    )
    glossary = load_glossary("samples/public_demo/terms/master_glossary.txt")

    result = OfflineTranslationProvider().translate("正文", story=story, glossary=glossary, mode="baseline")

    assert result.startswith("Chapter 42\n\n")


def test_offline_baseline_uses_imported_baseline_dir_for_chapter(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    (baseline_dir / "0002.txt").write_text(
        "Chapter 2\n\nThe Way of Heaven opened. 剩余次数：3\n",
        encoding="utf-8",
    )
    story = StoryConfig(
        slug="imported",
        title="Imported",
        chapter_ids=["0002"],
        paths=StoryPaths(
            source_dir=tmp_path,
            glossary_path=Path("samples/public_demo/terms/master_glossary.txt"),
            baseline_dir=baseline_dir,
        ),
    )
    glossary = load_glossary("samples/public_demo/terms/master_glossary.txt")

    baseline = OfflineTranslationProvider().translate("天道", story=story, glossary=glossary, mode="baseline")
    glossary_text = OfflineTranslationProvider().translate("天道", story=story, glossary=glossary, mode="glossary")

    assert baseline.startswith("Chapter 2\n\nThe Way of Heaven")
    assert "Heavenly Dao" in glossary_text
    assert "remaining uses: 3" in glossary_text.lower()


def test_live_translation_payload_uses_relevant_glossary_subset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from agentic_translation.providers_llm import LLMTranslationProvider

    captured: dict[str, object] = {}

    class Message:
        content = '{"translation": "Chapter 1\\n\\nDone."}'

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    class Completions:
        def create(self, **kwargs: object) -> Response:
            messages = kwargs["messages"]
            captured["payload"] = messages[1]["content"]  # type: ignore[index]
            return Response()

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("AGENTIC_TRANSLATION_MODEL", "test-model")
    glossary = load_glossary("samples/public_demo/terms/master_glossary.txt")
    story = StoryConfig(
        slug="subset",
        title="Subset",
        paths=StoryPaths(source_dir=tmp_path, glossary_path=Path("samples/public_demo/terms/master_glossary.txt")),
    )
    provider = LLMTranslationProvider(
        provider_mode="live",
        cache_dir=tmp_path,
        client_factory=lambda **kwargs: Client(),
    )

    provider.translate("天道 and nothing else", story=story, glossary=glossary, mode="baseline")

    payload = str(captured["payload"])
    assert "天道" in payload
    assert "模拟器" not in payload


def test_live_translation_includes_story_prompt_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from agentic_translation.providers_llm import LLMTranslationProvider

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Preserve sharp simulator humor.", encoding="utf-8")
    captured: dict[str, object] = {}

    class Message:
        content = '{"translation": "Chapter 1\\n\\nDone."}'

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    class Completions:
        def create(self, **kwargs: object) -> Response:
            captured["messages"] = kwargs["messages"]
            return Response()

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("AGENTIC_TRANSLATION_MODEL", "test-model")
    glossary = load_glossary("samples/public_demo/terms/master_glossary.txt")
    story = StoryConfig(
        slug="prompted",
        title="Prompted",
        paths=StoryPaths(
            source_dir=tmp_path,
            glossary_path=Path("samples/public_demo/terms/master_glossary.txt"),
            prompt_path=prompt_path,
        ),
    )
    provider = LLMTranslationProvider(
        provider_mode="live",
        cache_dir=tmp_path,
        client_factory=lambda **kwargs: Client(),
    )

    provider.translate("天道", story=story, glossary=glossary, mode="baseline")

    messages = captured["messages"]
    assert "Preserve sharp simulator humor." in messages[0]["content"]  # type: ignore[index]


def test_live_translation_chunks_long_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from agentic_translation.providers_llm import LLMTranslationProvider

    captured_payloads: list[dict[str, object]] = []

    class Choice:
        def __init__(self, content: str) -> None:
            self.message = type("Message", (), {"content": content})()

    class Response:
        def __init__(self, content: str) -> None:
            self.choices = [Choice(content)]

    class Completions:
        def create(self, **kwargs: object) -> Response:
            messages = kwargs["messages"]
            payload = json.loads(messages[1]["content"])  # type: ignore[index]
            captured_payloads.append(payload)
            return Response(json.dumps({"translation": f"translated chunk {payload['chunk_index']}"}))

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("AGENTIC_TRANSLATION_MODEL", "test-model")
    glossary = load_glossary("samples/public_demo/terms/master_glossary.txt")
    story = StoryConfig(
        slug="chunked",
        title="Chunked",
        paths=StoryPaths(source_dir=tmp_path, glossary_path=Path("samples/public_demo/terms/master_glossary.txt")),
        translation=TranslationConfig(max_chunk_chars=40),
    )
    provider = LLMTranslationProvider(
        provider_mode="live",
        cache_dir=tmp_path,
        client_factory=lambda **kwargs: Client(),
    )

    result = provider.translate(
        "第一章 长标题\n\n第一段内容很长很长，需要单独翻译。\n\n第二段内容也很长很长，需要继续翻译。",
        story=story,
        glossary=glossary,
        mode="baseline",
    )

    assert len(captured_payloads) > 1
    assert {payload["chunk_count"] for payload in captured_payloads} == {len(captured_payloads)}
    assert result == "\n\n".join(f"translated chunk {index}" for index in range(1, len(captured_payloads) + 1))


def test_get_judge_provider_supports_deepseek(tmp_path: Path) -> None:
    provider = get_judge_provider(
        "deepseek",
        provider_mode="live",
        cache_dir=tmp_path,
        record_cache=True,
        model_name="deepseek-chat",
    )

    assert provider.provider_name == "deepseek"
    assert provider.model_name == "deepseek-chat"
