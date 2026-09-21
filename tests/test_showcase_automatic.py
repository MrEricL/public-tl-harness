"""The automatic strategy uses the durable harness and its final-source gate."""
import json
from pathlib import Path

from agentic_translation.showcase import replay_showcase, run_showcase


def test_automatic_showcase_selects_policy_and_replays(tmp_path):
    story = Path(__file__).resolve().parents[1] / "samples/showcase/story.yaml"
    original = tmp_path / "automatic"
    result = run_showcase(story, original, auto_approve=True)
    assert result.manifest["status"] == "completed"
    assert result.manifest["strategy"] == "automatic"
    for chapter in result.manifest["chapter_ids"]:
        directory = original / "chapters" / chapter
        snapshot = json.loads((directory / "session_snapshot.json").read_text())
        assert snapshot["allow_nonregressing_patches"] is True
        assert snapshot["require_fidelity_review"] is True
        events = [json.loads(line) for line in (directory / "session_events.jsonl").read_text().splitlines()]
        exposed = next(e["payload"] for e in events if e["event_type"] == "tools_exposed")
        assert exposed["dynamic_tools"] is False
        assert "submit_patch" in exposed["tool_names"]
    replay = replay_showcase(original, tmp_path / "replay")
    assert replay.manifest["status"] == "completed"
    for chapter in result.manifest["chapter_ids"]:
        path = Path("chapters") / chapter / "translated_final.txt"
        assert (original / path).read_bytes() == (replay.run_dir / path).read_bytes()
