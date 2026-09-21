from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import agentic_translation.showcase as showcase_module
from agentic_translation.providers_llm import LLMProviderUnavailable
from agentic_translation.showcase import replay_showcase, resume_showcase, run_showcase


ROOT = Path(__file__).resolve().parents[1]
SHOWCASE = ROOT / "samples" / "showcase"


def _copy_showcase(tmp_path: Path) -> Path:
    fixture = tmp_path / "showcase"
    shutil.copytree(SHOWCASE, fixture)
    story = fixture / "story.yaml"
    config = yaml.safe_load(story.read_text(encoding="utf-8"))
    config["paths"].update(
        source_dir=str(fixture / "source"),
        glossary_path=str(fixture / "terms/master_glossary.txt"),
        expected_dir=str(fixture / "expected"),
        runs_dir=str(fixture / "runs"),
    )
    story.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return story


def _run_and_approve(tmp_path: Path):
    story = _copy_showcase(tmp_path)
    paused = run_showcase(story, tmp_path / "run")
    assert paused.manifest["status"] == "awaiting_approval"
    return story, resume_showcase(paused.run_dir, decision="approved", reviewer="test", note="Approved in test.")


def test_showcase_pauses_for_glossary_approval_before_delivery(tmp_path: Path) -> None:
    story = _copy_showcase(tmp_path)

    result = run_showcase(story, tmp_path / "run")

    assert result.manifest["status"] == "awaiting_approval"
    assert result.manifest["chapters"]["0001"]["status"] == "awaiting_approval"
    assert set(result.manifest["chapters"]) == {"0001"}
    assert not (result.run_dir / "delivery").exists()
    assert (result.run_dir / "inputs/source/0001.txt").exists()
    assert (result.run_dir / "inputs/master_glossary.txt").exists()
    assert (result.run_dir / "inputs/style_guide.md").exists()
    assert (result.run_dir / "inputs/scenario.json").exists()


def test_approved_showcase_delivers_three_chapters_and_reuses_starstream_step(tmp_path: Path) -> None:
    _, result = _run_and_approve(tmp_path)

    assert result.manifest["status"] == "completed"
    assert set(result.manifest["chapters"]) == {"0001", "0002", "0003"}
    assert result.manifest["artifacts"] == {
        "report": "report.html",
        "txt": "delivery/book.txt",
        "epub": "delivery/book.epub",
    }
    assert (result.run_dir / "delivery/book.txt").exists()
    assert (result.run_dir / "delivery/book.epub").exists()
    book = (result.run_dir / "delivery/book.txt").read_text(encoding="utf-8")
    assert book.count("Chapter:") == 3
    assert book.count("Starstream Step") >= 3
    assert "星河步" not in book


def test_showcase_resume_uses_saved_inputs_after_original_story_changes(tmp_path: Path) -> None:
    story = _copy_showcase(tmp_path)
    fixture_root = story.parent
    original_source = (fixture_root / "source/0001.txt").read_text(encoding="utf-8")
    original_style = (fixture_root / "style_guide.md").read_text(encoding="utf-8")
    original_glossary = (fixture_root / "terms/master_glossary.txt").read_text(encoding="utf-8")

    paused = run_showcase(story, tmp_path / "run")
    (fixture_root / "source/0001.txt").write_text("MUTATED SOURCE", encoding="utf-8")
    (fixture_root / "style_guide.md").write_text("MUTATED STYLE", encoding="utf-8")
    (fixture_root / "terms/master_glossary.txt").write_text("MUTATED GLOSSARY", encoding="utf-8")

    result = resume_showcase(paused.run_dir, decision="approved")

    assert result.manifest["status"] == "completed"
    assert (result.run_dir / "inputs/source/0001.txt").read_text(encoding="utf-8") == original_source
    assert (result.run_dir / "inputs/style_guide.md").read_text(encoding="utf-8") == original_style
    assert (result.run_dir / "inputs/master_glossary.txt").read_text(encoding="utf-8") == original_glossary
    assert "MUTATED" not in (result.run_dir / "delivery/book.txt").read_text(encoding="utf-8")


def test_replay_rebuilds_from_saved_cache_without_network_and_keeps_logical_run_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, completed = _run_and_approve(tmp_path)
    original_book = (completed.run_dir / "delivery/book.txt").read_text(encoding="utf-8")

    import openai

    def forbidden_openai(**kwargs: object) -> None:
        raise AssertionError("replay attempted to construct a network client")

    monkeypatch.setattr(openai, "OpenAI", forbidden_openai)
    replay = replay_showcase(completed.run_dir, tmp_path / "replay")

    assert replay.manifest["status"] == "completed"
    assert replay.manifest["execution_mode"] == "replay"
    assert replay.manifest["run_id"] == completed.manifest["run_id"]
    assert (replay.run_dir / "delivery/book.txt").read_text(encoding="utf-8") == original_book


def test_replay_rejects_tampered_cached_response(tmp_path: Path) -> None:
    _, completed = _run_and_approve(tmp_path)
    cached_response = next(
        path
        for path in (completed.run_dir / "cache/0001/translation").glob("*.json")
        if not path.name.startswith("usage_")
    )
    cached_response.write_text("{\"tampered\": true}\n", encoding="utf-8")

    with pytest.raises(LLMProviderUnavailable, match="integrity"):
        replay_showcase(completed.run_dir, tmp_path / "replay")


def test_rejected_glossary_review_stops_with_review_required_and_no_delivery(tmp_path: Path) -> None:
    story = _copy_showcase(tmp_path)
    paused = run_showcase(story, tmp_path / "run")

    result = resume_showcase(paused.run_dir, decision="rejected", reviewer="test", note="Needs human review.")

    assert result.manifest["status"] == "review_required"
    assert result.manifest["chapters"]["0001"]["status"] == "review_required"
    assert not (result.run_dir / "delivery").exists()


def test_interrupted_coordinator_persists_checkpoint_and_can_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    story = _copy_showcase(tmp_path)
    original_providers = showcase_module._providers
    interrupted = False

    def providers_with_one_interrupt(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal interrupted
        provider, specialists = original_providers(*args, **kwargs)
        if not interrupted:
            original_next_action = provider.next_action
            def interrupt_once(request):  # noqa: ANN001
                nonlocal interrupted
                if request.step_number == 3:
                    interrupted = True
                    raise KeyboardInterrupt()
                return original_next_action(request)

            provider.next_action = interrupt_once
        return provider, specialists

    monkeypatch.setattr(showcase_module, "_providers", providers_with_one_interrupt)
    interrupted_run = run_showcase(story, tmp_path / "run")

    assert interrupted_run.manifest["status"] == "interrupted"
    assert (interrupted_run.run_dir / "chapters/0001/session_snapshot.json").exists()
    checkpoint = json.loads((interrupted_run.run_dir / "chapters/0001/agent_episode.json").read_text())
    assert len(checkpoint["steps"]) == 2
    resumed = resume_showcase(interrupted_run.run_dir)
    assert resumed.manifest["status"] == "awaiting_approval"
    completed = resume_showcase(resumed.run_dir, decision="approved")
    assert completed.manifest["status"] == "completed"
    assert len(completed.manifest["chapters"]["0001"]["coordinator_calls"]) == 9


def test_supplied_drafts_bypass_translation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    story = _copy_showcase(tmp_path)
    draft_dir = tmp_path / "drafts"
    shutil.copytree(story.parent / "expected", draft_dir)

    def fail_translation(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("supplied drafts must bypass translation")

    import agentic_translation.showcase_providers as showcase_providers

    monkeypatch.setattr(showcase_providers, "make_translation_provider", fail_translation)
    result = run_showcase(story, tmp_path / "run", draft_dir=draft_dir, auto_approve=True)

    assert result.manifest["status"] == "completed"
    assert all(result.manifest["chapters"][chapter]["translation_calls"] == [] for chapter in result.manifest["chapter_ids"])


def test_deterministic_baseline_uses_existing_glossary_rules(tmp_path: Path) -> None:
    story = _copy_showcase(tmp_path)
    drafts = tmp_path / "drafts"
    shutil.copytree(story.parent / "expected", drafts)
    first = drafts / "0001.txt"
    first.write_text(first.read_text().replace("Azure Cloud Sect", "Blue Cloud School", 1))

    result = run_showcase(story, tmp_path / "run", draft_dir=drafts, strategy="deterministic")

    assert result.manifest["status"] == "completed"
    chapter = result.manifest["chapters"]["0001"]
    assert chapter["initial_findings"] > 0
    assert chapter["final_findings"] == 0
    assert chapter["accepted_patches"] == 1
    assert chapter["provider_calls"] == []
    assert "Blue Cloud School" not in (result.run_dir / "delivery/book.txt").read_text()
    assert (result.run_dir / "delivery/book.txt").exists()
    assert not (result.run_dir / "expected").exists()
    assert not (result.run_dir / "inputs/expected").exists()


class _FakeLiveCompletions:
    def __init__(self, translations: list[str]) -> None:
        self.translations = translations
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"translation": self.translations[index]})))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


class _FakeLiveOpenAI:
    instances: list["_FakeLiveOpenAI"] = []
    translations: list[str] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.completions = _FakeLiveCompletions(self.translations)
        self.chat = SimpleNamespace(completions=self.completions)
        self.instances.append(self)


def test_live_runner_uses_mocked_openai_client_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    story = _copy_showcase(tmp_path)
    expected = [
        (story.parent / "expected" / f"{chapter}.txt").read_text(encoding="utf-8")
        for chapter in ("0001", "0002", "0003")
    ]
    _FakeLiveOpenAI.instances = []
    _FakeLiveOpenAI.translations = expected
    import openai

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai, "OpenAI", _FakeLiveOpenAI)

    result = run_showcase(
        story,
        tmp_path / "live-run",
        provider_mode="live",
        profile="openai",
        model="fake-model",
        strategy="deterministic",
    )

    assert result.manifest["status"] == "completed"
    assert len(_FakeLiveOpenAI.instances) == 3
    assert all(instance.kwargs["timeout"] == 60.0 for instance in _FakeLiveOpenAI.instances)
    assert all(call["temperature"] == 0.0 for instance in _FakeLiveOpenAI.instances for call in instance.completions.calls)
    assert all(call["max_tokens"] == 2048 for instance in _FakeLiveOpenAI.instances for call in instance.completions.calls)
    assert "test-key" not in (result.run_dir / "run_manifest.json").read_text(encoding="utf-8")
