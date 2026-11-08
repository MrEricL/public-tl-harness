from agentic_translation.demo import build_context_receipts
from agentic_translation.segmentation import segment_source
from agentic_translation.story_memory import MemoryEntry, SourceEvidence, StoryMemoryTape, TerminologyCandidate
from agentic_translation.unattended import MAX_TRANSLATOR_TERM_LINES, _with_translator_terms


def test_receipts_explain_prior_matches_precedence_and_truncation():
    tape = StoryMemoryTape()
    first = segment_source("w", "1", "阿澈就是林澈。他拿着铜钥匙。")
    evidence = SourceEvidence("1", first[0].segment_id, "阿澈就是林澈")
    fact = MemoryEntry("alias", "alias", "林澈", "Alias 阿澈 names Lin Che.", (evidence,),
                       "1", first[0].segment_id, "narrator", "active", "fixture",
                       tape.policy_version, source_aliases=("阿澈",))
    term = TerminologyCandidate("林澈", "Lin Che", (evidence,))
    tape.add_chapter("1", first, [fact], [term])
    second = segment_source("w", "2", "阿澈叫住林澈。")
    tape.add_chapter("2", second, [], [])
    sid = second[0].segment_id
    full = tape.context_for_segment("2", sid)
    context = _with_translator_terms({sid: full.text}, second, {"林澈": "Lin Che"},
                                     drop_overlapping_memory_terms=True)
    row = build_context_receipts(second, tape, {"林澈": "Lin Che"}, context)[0]
    assert row["selected_facts"][0]["earlier_source"]
    assert row["selected_facts"][0]["matched_strings"] == ["林澈", "阿澈"]
    assert row["selected_facts"][0]["source_evidence"][0]["exact_excerpt"] == "阿澈就是林澈"
    assert row["memory_terms"][0]["omission_reason"] == "translator_glossary_precedence"
    assert not row["memory_terms"][0]["supplied"]
    truncated = build_context_receipts(second, tape, {}, {sid: ""}, max_context_chars=0)[0]
    assert truncated["memory_pack_truncated"]
    assert not truncated["selected_facts"]
    assert truncated["omitted_facts"][0]["entry_id"] == "alias"
    assert truncated["memory_terms"][0]["omission_reason"] == "size_limit"


def test_receipts_only_claim_translator_terms_present_after_context_cap():
    terms = {f"术{index:02d}": f"Technique {index}" for index in range(MAX_TRANSLATOR_TERM_LINES + 1)}
    segments = segment_source("w", "1", "，".join(terms) + "。")
    assert len(segments) == 1
    tape = StoryMemoryTape()
    tape.add_chapter("1", segments, [], [])
    context = _with_translator_terms({}, segments, terms)

    receipt = build_context_receipts(segments, tape, terms, context)[0]

    assert len(receipt["translator_terms"]) == MAX_TRANSLATOR_TERM_LINES
    assert receipt["omitted_translator_terms"] == {"术40": "Technique 40"}
    assert all(f"TERM {source} -> {target} [translator glossary]" in context[segments[0].segment_id]
               for source, target in receipt["translator_terms"].items())


def test_receipts_keep_compatible_compound_names_and_explain_actual_conflicts():
    tape = StoryMemoryTape()
    first = segment_source("w", "1", "青云宗坐落在山中。")
    tape.add_chapter("1", first, [], [TerminologyCandidate(
        "青云宗", "Azure Cloud Sect", (SourceEvidence("1", first[0].segment_id, "青云宗"),))])
    second = segment_source("w", "2", "青云宗开门了。")
    tape.add_chapter("2", second, [], [])
    sid = second[0].segment_id
    pack = {sid: tape.context_for_segment("2", sid).text}

    compatible = {"青云": "Azure Cloud"}
    context = _with_translator_terms(pack, second, compatible, drop_overlapping_memory_terms=True,
                                     memory_overlap="drop_conflicting")
    receipt = build_context_receipts(second, tape, compatible, context)[0]
    assert "TERM 青云宗 -> Azure Cloud Sect" in context[sid].splitlines()
    assert receipt["memory_terms"][0]["supplied"]
    assert receipt["memory_terms"][0]["omission_reason"] is None

    conflicting = {"青云": "Blue Mist"}
    context = _with_translator_terms(pack, second, conflicting, drop_overlapping_memory_terms=True,
                                     memory_overlap="drop_conflicting")
    receipt = build_context_receipts(second, tape, conflicting, context)[0]
    assert "TERM 青云宗 -> Azure Cloud Sect" not in context[sid].splitlines()
    assert not receipt["memory_terms"][0]["supplied"]
    assert receipt["memory_terms"][0]["omission_reason"] == "translator_glossary_precedence"
