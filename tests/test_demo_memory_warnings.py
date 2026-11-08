from agentic_translation.demo import main
from agentic_translation.segmentation import segment_source
from agentic_translation.story_memory import StoryMemoryTape
from agentic_translation.unattended import SourceMemoryExtractor


class EmptyExtractor:
    model = "fixture"

    def call(self, *args, **kwargs):
        return {"entries": [], "terms": []}


def test_empty_memory_warning_only_for_long_chapter():
    extractor = SourceMemoryExtractor(EmptyExtractor(), StoryMemoryTape())
    extractor("short", segment_source("w", "short", "他回来了。"), None)
    assert extractor.warnings == []
    extractor("long", segment_source("w", "long", "他回来了。" * 101), None)
    assert extractor.warnings[0]["code"] == "memory_extract_empty"
    assert extractor.warnings[0]["chapter_id"] == "long"


def test_rejected_citations_do_not_hide_empty_memory():
    class InvalidExtractor(EmptyExtractor):
        def call(self, *args, **kwargs):
            return {"entries": [], "terms": [{"source": "他", "target": "He",
                    "segment_id": "unknown", "exact_excerpt": "他", "provisional": False}]}

    extractor = SourceMemoryExtractor(InvalidExtractor(), StoryMemoryTape())
    assert extractor("c", segment_source("w", "c", "他回来了。" * 101), None) == ([], [])
    assert extractor.rejections and extractor.warnings


def test_cli_resolves_and_records_deepseek_thinking_default(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source"
    source.mkdir()
    (source / "1.txt").write_text("他回来了。")
    seen = []

    def fake_run(source, out, config, **kwargs):
        seen.append(config)
        return {"status": "delivered"}

    monkeypatch.setattr("agentic_translation.demo.run_folder", fake_run)
    assert main(["--source", str(source), "--out", str(tmp_path / "out"),
                 "--provider", "deepseek", "--model", "fixture"]) == 0
    assert seen[0]["thinking_mode"] == "disabled"
    assert "DeepSeek thinking mode: disabled" in capsys.readouterr().out
