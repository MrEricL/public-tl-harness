"""align_chapter_terms: the term-alignment pass on its own (authored data only)."""
from agentic_translation.adaptive import align_chapter_terms
from agentic_translation.agent_models import SubmitSegmentPatchAction
from agentic_translation.demo import demo_policy
from agentic_translation.models import GlossaryEntry, GlossaryParseResult
from agentic_translation.segmentation import DraftSegment, segment_source


class FakeAligner:
    def __init__(self):
        self.items = []

    def align(self, items):
        self.items = list(items)
        return [SubmitSegmentPatchAction(segment_id=item.segment_id, expected_segment_sha256=item.expected_segment_sha256,
                                         edits=[{"old_text": "Lunar Shadow Footwork", "new_text": "Moon-Shadow Step"}],
                                         issue_ids=["term_alignment"], rationale="use the established name")
                for item in items]


def test_alignment_selects_only_passages_missing_the_name_and_patches_them():
    sources = segment_source("w", "c1", "林青使用月影步。\n他再次使用月影步。", target_chars=8, max_chars=10)
    assert len(sources) == 2
    drafts = [DraftSegment.from_text(sources[0].segment_id, "Lin Qing used Lunar Shadow Footwork."),
              DraftSegment.from_text(sources[1].segment_id, "He used Moon-Shadow Step again.")]
    glossary = GlossaryParseResult(entries=[GlossaryEntry(source="月影步", target="Moon-Shadow Step")])
    aligner = FakeAligner()
    aligned, events = align_chapter_terms(sources, drafts, glossary=glossary, alignment_provider=aligner,
                                          verification_policy=demo_policy())
    assert [item.segment_id for item in aligner.items] == [sources[0].segment_id]       # only the passage missing it
    assert aligned[0].translated_text == "Lin Qing used Moon-Shadow Step."
    assert aligned[1].translated_text == drafts[1].translated_text
    assert [e["kind"] for e in events] == ["term_alignment_requested", "term_alignment_applied"]


def test_run_adaptive_chapter_aligns_only_to_the_alignment_glossary():
    """QA keeps the full glossary; alignment enforces only the names it is given (translator_only)."""
    from agentic_translation.adaptive import _identity_hash, run_adaptive_chapter
    sources = segment_source("w", "c1", "林青使用月影步，走进青云门。")
    drafts = [DraftSegment.from_text(sources[0].segment_id, "Lin Qing used Lunar Shadow Footwork to enter Blue Cloud Gate.")]
    memory = GlossaryEntry(source="月影步", target="Moon-Shadow Step")
    translator = GlossaryEntry(source="青云门", target="Azure Cloud Sect")
    full = GlossaryParseResult(entries=[memory, translator])
    only = GlossaryParseResult(entries=[translator])

    aligner = FakeAligner()
    run_adaptive_chapter(sources, drafts, glossary=full, decision_provider=None, review_provider=None,
                         alignment_provider=aligner, verification_policy=demo_policy(), alignment_glossary=only)
    assert [term for item in aligner.items for term in item.terms] == [("青云门", "Azure Cloud Sect")]

    aligner = FakeAligner()
    run_adaptive_chapter(sources, drafts, glossary=full, decision_provider=None, review_provider=None,
                         alignment_provider=aligner, verification_policy=demo_policy())
    assert sorted(term for item in aligner.items for term in item.terms) == [
        ("月影步", "Moon-Shadow Step"), ("青云门", "Azure Cloud Sect")]

    from agentic_translation.adaptive import BudgetPolicy, InteractionPolicy
    args = ("src", "draft", demo_policy(), InteractionPolicy(), BudgetPolicy(), "m", {}, full)
    assert _identity_hash(*args) == _identity_hash(*args, None)      # old sessions keep their identity
    assert _identity_hash(*args) != _identity_hash(*args, only)
