"""Demo lead-in and carried memory, end to end with a scripted client."""
import json
from types import SimpleNamespace as NS

import pytest

from agentic_translation.demo import main, run_folder
from agentic_translation.demo_provider import DemoGenerator, provider_config

TERM_LINE = "TERM 月影步 -> Moon-Shadow Step"


class Client:
    """Scripted transport: memory records 月影步 only when reading chapter 0001."""

    def __init__(self):
        self.chat = NS(completions=self)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        user = kw["messages"][1]["content"]
        payload = json.JSONDecoder().raw_decode(user[user.index("\n{") + 1:])[0]
        if user.startswith("Extract"):
            terms = [dict(source="月影步", target="Moon-Shadow Step", segment_id=row["segment_id"], exact_excerpt="月影步",
                          provisional=False)
                     for row in payload["segments"] if payload["chapter_id"] == "0001" and "月影步" in row["source_text"]][:1]
            raw = json.dumps(dict(entries=[], terms=terms))
        elif user.startswith("Translate"):
            raw = "\n\n".join(f"<<<SEGMENT:{row['segment_id']}>>>\nLin Qing used Moon-Shadow Step."
                              for row in payload["source_segments"])
        else:
            raw = json.dumps(dict(summary="No material issue in this fixture", findings=[], patches=[]))
        return NS(model="test-model", usage=NS(prompt_tokens=40, completion_tokens=30),
                  choices=[NS(finish_reason="stop", message=NS(content=raw))])


def _folder(path, texts):
    path.mkdir()
    for i, text in enumerate(texts, 1):
        (path / f"ch{i}.txt").write_text(text, encoding="utf-8")
    return path


def _run(source, out, **kw):
    gen = DemoGenerator(out, provider_config("openai", "test-model"), client=Client())
    return run_folder(source, out, gen.config, generator=gen, limit=0, **kw)


def _contexts(report):
    return "\n".join(v for c in report["chapters"] for v in c["context"].values())


def test_lead_in_reads_memory_without_translating_and_replays(tmp_path, monkeypatch):
    source = _folder(tmp_path / "src", ["林青學會了月影步。", "林青使用月影步。", "月影步再次出現。"])
    out = tmp_path / "run"
    report = _run(source, out, lead_in=1)
    assert [c["id"] for c in report["chapters"]] == ["0002", "0003"]
    assert report["lead_in"] == [dict(id="0001", name="ch1.txt", status="read", names=1, facts=0)]
    assert TERM_LINE in _contexts(report)                       # chapter 2 knew the name from the lead-in
    assert sorted(p.name for p in (out / "inputs").iterdir()) == ["0002.txt", "0003.txt"]
    assert (out / "memory_sources" / "0001.txt").is_file()
    assert len(list((out / "memory" / "story_memory" / "snapshots").glob("*.json"))) == 3
    assert json.loads((out / "config.json").read_text())["lead_in"][0]["id"] == "0001"
    monkeypatch.setattr(DemoGenerator, "_transport", lambda self: pytest.fail("network during replay"))
    replay = tmp_path / "replay"
    assert main(["--replay", str(out), "--out", str(replay)]) == 0
    again = json.loads((replay / "results.json").read_text())
    assert again["resources"]["physical_calls"] == 0
    assert [c["texts"] for c in again["chapters"]] == [c["texts"] for c in report["chapters"]]


def test_memory_from_carries_names_into_a_later_run_and_replays(tmp_path, monkeypatch):
    first = _run(_folder(tmp_path / "a", ["林青學會了月影步。"]), tmp_path / "run1")
    assert first["status"] == "delivered"
    out = tmp_path / "run2"
    report = _run(_folder(tmp_path / "b", ["林青再次使用月影步。"]), out, memory_from=tmp_path / "run1")
    assert [c["id"] for c in report["chapters"]] == ["0002"]
    assert report["carried_memory"]["chapters"] == 1
    assert TERM_LINE in _contexts(report)                       # carried from run 1; run 2's extractor found nothing
    snapshots = out / "memory" / "story_memory" / "snapshots"
    assert sorted(p.name for p in snapshots.iterdir()) == ["0001.json", "0002.json"]
    second = json.loads((snapshots / "0002.json").read_text())
    assert second["preceding_snapshot_sha256"] == json.loads((snapshots / "0001.json").read_text())["snapshot_sha256"]
    monkeypatch.setattr(DemoGenerator, "_transport", lambda self: pytest.fail("network during replay"))
    replay = tmp_path / "replay2"
    assert main(["--replay", str(out), "--out", str(replay)]) == 0
    assert json.loads((replay / "results.json").read_text())["resources"]["physical_calls"] == 0


def test_lead_in_needs_something_left_to_translate_and_replay_refuses_new_options(tmp_path):
    source = _folder(tmp_path / "src", ["林青。"])
    with pytest.raises(ValueError, match="left to translate"):
        _run(source, tmp_path / "run", lead_in=1)
    assert main(["--replay", str(tmp_path), "--out", str(tmp_path / "x"), "--lead-in", "1"]) == 2
