"""New demo runs use the story-memory setup Stage 1 tested; replays keep what their run recorded."""
import json

from agentic_translation.demo import DEMO_MEMORY, LEGACY_DEMO_MEMORY, memory_settings, run_folder
from agentic_translation.demo_provider import DemoGenerator, provider_config


def test_new_runs_use_and_record_the_tested_memory_settings(tmp_path):
    assert DEMO_MEMORY == {"instruction_version": "term-memory-v2", "memory_overlap": "drop_conflicting",
                           "fact_policy": "terms_only", "alignment_names": "translator_only"}
    source = tmp_path / "src"
    source.mkdir()
    (source / "001.txt").write_text("林凡走进青云门。", encoding="utf-8")

    class Down:
        calls = []

        def create(self, **kw):
            raise RuntimeError("unavailable")

    out = tmp_path / "run"
    gen = DemoGenerator(out, provider_config("openai", "test-model"), client=Down())
    run_folder(source, out, gen.config, generator=gen)
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["memory"] == DEMO_MEMORY


def test_replay_uses_recorded_settings_and_legacy_when_unrecorded(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    (old / "config.json").write_text(json.dumps({"schema_version": "portfolio-run.v1"}), encoding="utf-8")
    assert memory_settings(old) == LEGACY_DEMO_MEMORY
    new = tmp_path / "new"
    new.mkdir()
    (new / "config.json").write_text(json.dumps({"memory": DEMO_MEMORY}), encoding="utf-8")
    assert memory_settings(new) == DEMO_MEMORY
    assert memory_settings(None) == DEMO_MEMORY


def test_names_only_pack_keeps_names_and_drops_facts():
    """Stage 2 made names only the default; the receipts follow the same pack."""
    from agentic_translation.demo import build_context_receipts
    from agentic_translation.segmentation import segment_source
    from agentic_translation.story_memory import MemoryEntry, SourceEvidence, StoryMemoryTape, TerminologyCandidate
    from agentic_translation.unattended import _context_payload

    tape = StoryMemoryTape()
    first = segment_source("w", "1", "阿澈就是林澈。他拿着铜钥匙。")
    evidence = SourceEvidence("1", first[0].segment_id, "阿澈就是林澈")
    fact = MemoryEntry("alias", "alias", "林澈", "Alias 阿澈 names Lin Che.", (evidence,),
                       "1", first[0].segment_id, "narrator", "active", "fixture",
                       tape.policy_version, source_aliases=("阿澈",))
    tape.add_chapter("1", first, [fact], [TerminologyCandidate("林澈", "Lin Che", (evidence,))])
    second = segment_source("w", "2", "阿澈叫住林澈。")
    tape.add_chapter("2", second, [], [])
    sid = second[0].segment_id

    legacy = _context_payload(second, tape)[sid]
    names = _context_payload(second, tape, fact_policy=DEMO_MEMORY["fact_policy"])[sid]
    assert "TERM 林澈 -> Lin Che" in legacy and "Alias 阿澈 names Lin Che." in legacy
    assert "TERM 林澈 -> Lin Che" in names and "Alias 阿澈 names Lin Che." not in names
    receipt = build_context_receipts(second, tape, {}, {sid: names}, fact_policy="terms_only")[0]
    assert receipt["selected_facts"] == [] and receipt["memory_terms"][0]["supplied"]
