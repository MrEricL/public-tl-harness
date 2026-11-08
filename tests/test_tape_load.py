"""StoryMemoryTape.load: carry a tape between runs, verified. Authored data only."""
import json
import hashlib

import pytest

from agentic_translation.segmentation import segment_source
from agentic_translation.story_memory import (MemoryEntry, SourceEvidence, StoryMemoryTape, TerminologyCandidate,
                                               saved_chapter_ids)

TEXTS = {"0001": "林青使用月影步。", "0002": "林青離開了城。", "0003": "月影步再次出現。"}


def _segments(cid):
    return segment_source("w", cid, TEXTS[cid])


def _fact(cid):
    return MemoryEntry(entry_id=f"{cid}_e001", kind="state", subject="林青", value=f"state {cid}",
                       source_evidence=(SourceEvidence(cid, "s0001", "林青"),), known_from_chapter=cid,
                       known_from_segment="s0001", scope="narrator", status="active", created_by_model="test",
                       policy_version="source-memory-v2", source_aliases=("林青",))


def _build(run_dir, ids=("0001", "0002")):
    tape = StoryMemoryTape(policy_version="source-memory-v2", run_dir=run_dir)
    for cid in ids:
        terms = (TerminologyCandidate("月影步", "Moon-Shadow Step", (SourceEvidence(cid, "s0001", "月影步"),)),) \
            if "月影步" in TEXTS[cid] else ()
        tape.add_chapter(cid, _segments(cid), (_fact(cid),) if "林青" in TEXTS[cid] else (), terms)
    return tape


def test_load_round_trip_and_continue(tmp_path):
    original = _build(tmp_path)
    loaded = StoryMemoryTape.load(tmp_path, {c: _segments(c) for c in ("0001", "0002")}, policy_version="source-memory-v2")
    assert loaded.chapter_ids == original.chapter_ids
    for cid in loaded.chapter_ids:
        assert loaded.snapshot_for_chapter(cid) == original.snapshot_for_chapter(cid)
    assert loaded.context_for_segment("0002", "s0001") == original.context_for_segment("0002", "s0001")
    third = loaded.add_chapter("0003", _segments("0003"), ())
    assert third.preceding_snapshot_sha256 == original.snapshot_for_chapter("0002").snapshot_sha256
    assert loaded.glossary_for_segment("0003", "s0001") == {"月影步": "Moon-Shadow Step"}   # carried name


def test_load_refuses_changed_source_tampering_and_a_non_leading_part(tmp_path):
    _build(tmp_path)
    sources = {c: _segments(c) for c in ("0001", "0002")}
    with pytest.raises(ValueError, match="differs"):
        StoryMemoryTape.load(tmp_path, {**sources, "0002": segment_source("w", "0002", "林青留在城裡。")},
                             policy_version="source-memory-v2")
    with pytest.raises(ValueError, match="leading part"):
        StoryMemoryTape.load(tmp_path, sources, policy_version="source-memory-v2", chapter_ids=["0002"])
    assert StoryMemoryTape.load(tmp_path, sources, policy_version="source-memory-v2",
                                chapter_ids=["0001"]).chapter_ids == ("0001",)
    path = tmp_path / "story_memory" / "snapshots" / "0002.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["entries"][-1]["value"] = "edited"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        StoryMemoryTape.load(tmp_path, sources, policy_version="source-memory-v2")


def _snapshot_path(run_dir, cid):
    return run_dir / "story_memory" / "snapshots" / f"{cid}.json"


def _rewrite_snapshot(path, change, *, rehash=False):
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    if rehash:
        payload = {key: value for key, value in data.items() if key != "snapshot_sha256"}
        data["snapshot_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


@pytest.mark.parametrize("damage", ["missing_root", "missing_middle", "fork", "duplicate_hash", "cycle"])
def test_saved_chain_and_load_reject_incomplete_or_ambiguous_history(tmp_path, damage):
    _build(tmp_path, ("0001", "0002", "0003"))
    first = json.loads(_snapshot_path(tmp_path, "0001").read_text(encoding="utf-8"))
    third = json.loads(_snapshot_path(tmp_path, "0003").read_text(encoding="utf-8"))
    if damage == "missing_root":
        _snapshot_path(tmp_path, "0001").unlink()
    elif damage == "missing_middle":
        _snapshot_path(tmp_path, "0002").unlink()
    elif damage == "fork":
        _rewrite_snapshot(_snapshot_path(tmp_path, "0003"),
                          lambda data: data.update(preceding_snapshot_sha256=first["snapshot_sha256"]))
    elif damage == "duplicate_hash":
        _rewrite_snapshot(_snapshot_path(tmp_path, "0003"),
                          lambda data: data.update(snapshot_sha256=first["snapshot_sha256"]))
    else:
        _rewrite_snapshot(_snapshot_path(tmp_path, "0002"),
                          lambda data: data.update(preceding_snapshot_sha256=third["snapshot_sha256"]))

    with pytest.raises(ValueError):
        saved_chapter_ids(tmp_path)
    with pytest.raises(ValueError):
        StoryMemoryTape.load(tmp_path, {c: _segments(c) for c in TEXTS}, policy_version="source-memory-v2")


@pytest.mark.parametrize(("record", "field", "value"), [
    ("entries", "known_from_chapter", "missing"),
    ("entries", "known_from_segment", "missing"),
    ("entries", "evidence_chapter", "missing"),
    ("entries", "evidence_segment", "missing"),
    ("terminology", "chapter_id", "missing"),
    ("terminology", "segment_id", "missing"),
    ("terminology", "evidence_chapter", "missing"),
    ("terminology", "evidence_segment", "missing"),
])
def test_load_checks_provenance_references_even_with_a_matching_snapshot_hash(tmp_path, record, field, value):
    _build(tmp_path, ("0001",))

    def change(data):
        row = data[record][0]
        if field.startswith("evidence_"):
            row["source_evidence"][0]["chapter_id" if field == "evidence_chapter" else "segment_id"] = value
        else:
            row[field] = value

    _rewrite_snapshot(_snapshot_path(tmp_path, "0001"), change, rehash=True)
    with pytest.raises(ValueError):
        StoryMemoryTape.load(tmp_path, {"0001": _segments("0001")}, policy_version="source-memory-v2")


def test_loaded_snapshots_must_preserve_the_prior_cumulative_memory(tmp_path):
    _build(tmp_path)
    _rewrite_snapshot(_snapshot_path(tmp_path, "0002"), lambda data: data["entries"].pop(0), rehash=True)
    with pytest.raises(ValueError, match="extend"):
        StoryMemoryTape.load(tmp_path, {c: _segments(c) for c in ("0001", "0002")},
                             policy_version="source-memory-v2")
