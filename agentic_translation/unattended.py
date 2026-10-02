"""Five paired translation arms over one immutable source-only memory tape.

All prose, prompts, candidate outputs, and receipts produced here belong under
an ignored run directory.  Corpus manifests contain metadata only.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import re
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from agentic_translation.adaptive import (
    AdaptiveReviewFinding, AdaptiveReviewRequest, AdaptiveReviewResult,
    BudgetPolicy, TermAlignmentItem, VerificationPolicy, run_adaptive_chapter,
)
from agentic_translation.costing import BudgetExceeded
from agentic_translation.providers_llm import LLMProviderUnavailable
from agentic_translation.agent_models import SubmitSegmentPatchAction
from agentic_translation.decisions.jev import TypeSafeDecisionProvider
from agentic_translation.models import GlossaryEntry, GlossaryParseResult
from agentic_translation.segmentation import SourceSegment, parse_draft_envelope, render_draft, segment_source
from agentic_translation.story_memory import (
    ENTRY_KINDS, SCOPES, STATUSES, MemoryEntry, SourceEvidence, StoryMemoryTape,
    TerminologyCandidate,
)

from .autonomous_provider import (
    BudgetedDecisions, BudgetedGenerator, GenerationOutputError, GenerationTransportError,
)


ARM_NAMES = ("naive", "contextual", "simple_revise", "always_review", "jev_adaptive")
_NEEDS_CONTEXTUAL = frozenset({"simple_revise", "always_review", "jev_adaptive"})


def configured_arms(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Arms to execute, in frozen order; unconfigured arms are never called or billed."""
    requested = config.get("arms") or ARM_NAMES
    unknown = sorted(set(requested) - set(ARM_NAMES))
    if unknown:
        raise ValueError(f"Unknown arms configured: {', '.join(unknown)}")
    arms = tuple(arm for arm in ARM_NAMES if arm in requested)
    if _NEEDS_CONTEXTUAL.intersection(arms) and "contextual" not in arms:
        raise ValueError("Revision arms reuse the contextual draft; configure contextual as well")
    return arms


def _verification_policy(config: Mapping[str, Any], arm: str) -> VerificationPolicy:
    """Bind the frozen experiment routing settings to every decision request."""
    jev = config.get("jev", {})
    review = config.get("review", {})
    return VerificationPolicy(
        mode="full_segment_review" if arm == "always_review" else "adaptive_segment_screen",
        decision_model=str(jev.get("model", "jev-1.13.0")),
        instruction_version=str(review.get("instruction_version", "natural-review-v2")),
        routing_policy_version=str(jev.get("routing_policy_version", "adaptive-routing.v1")),
        routing_thresholds={
            "diagnostic": float(jev.get("low_risk_threshold", 0.15)),
            "material": float(jev.get("repair_threshold", 0.50)),
            "readability": float(jev.get("readability_threshold", 1.5)),
        },
        diagnostic_min_flags=int(jev.get("diagnostic_min_flags", 2)),
        route_source_incomplete=bool(jev.get("route_source_incomplete", False)),
        qa_warning_policy_version=str(review.get("qa_warning_policy_version", "qa-warnings.v1")),
        route_deterministic_findings=bool(review.get("route_deterministic_findings", False)),
        candidate_rescreen=bool(jev.get("candidate_rescreen", False)),
        term_alignment_pass=bool(review.get("term_alignment_pass", False)),
        gloss_guard=bool(review.get("gloss_guard", False)),
        duplication_guard=bool(review.get("duplication_guard", False)),
        candidate_rescreen_threshold=(float(jev["candidate_rescreen_threshold"])
                                      if jev.get("candidate_rescreen_threshold") is not None else None),
    )
SEGMENT_ITEM_SCHEMA = {
    "type": "object", "required": ["segment_id", "translated_text"],
    "properties": {"segment_id": {"type": "string"}, "translated_text": {"type": "string"}},
    "additionalProperties": False,
}
TRANSLATION_SCHEMA = {
    "type": "object", "required": ["segments"],
    "properties": {"segments": {"type": "array", "items": SEGMENT_ITEM_SCHEMA}},
    "additionalProperties": False,
}
MEMORY_SCHEMA = {
    "type": "object", "required": ["entries", "terms"],
    "properties": {
        "entries": {"type": "array", "items": {
            "type": "object", "required": ["kind", "subject", "value", "segment_id", "exact_excerpt", "scope", "status"],
            "properties": {
                "kind": {"type": "string", "enum": sorted(ENTRY_KINDS)},
                "subject": {"type": "string"}, "value": {"type": "string"},
                "segment_id": {"type": "string"}, "exact_excerpt": {"type": "string"},
                "scope": {"type": "string", "enum": sorted(SCOPES)},
                "status": {"type": "string", "enum": sorted(STATUSES)},
                "source_aliases": {"type": "array", "items": {"type": "string"}},
            }, "additionalProperties": False,
        }},
        "terms": {"type": "array", "items": {
            "type": "object", "required": ["source", "target", "segment_id", "exact_excerpt", "provisional"],
            "properties": {
                "source": {"type": "string"}, "target": {"type": "string"},
                "segment_id": {"type": "string"}, "exact_excerpt": {"type": "string"},
                "provisional": {"type": "boolean"},
            }, "additionalProperties": False,
        }},
    }, "additionalProperties": False,
}
CRITIQUE_SCHEMA = {
    "type": "object", "required": ["issues"],
    "properties": {"issues": {"type": "array", "maxItems": 12,
                              "items": {"type": "string", "maxLength": 240}}},
    "additionalProperties": False,
}
REVIEW_SCHEMA = {
    "type": "object", "required": ["summary", "findings", "patches"],
    "properties": {
        "summary": {"type": "string"},
        "findings": {"type": "array", "items": {
            "type": "object", "required": ["issue_id", "severity", "message", "blocking"],
            "properties": {
                "issue_id": {"type": "string", "minLength": 1},
                "severity": {"type": "string", "enum": ["material", "diagnostic", "polish"]},
                "message": {"type": "string", "minLength": 1},
                "blocking": {"type": "boolean"},
            }, "additionalProperties": True,
        }},
        "patches": {"type": "array", "maxItems": 1,
                    "items": {key: value for key, value in SubmitSegmentPatchAction.model_json_schema().items()
                              if key != "$defs"}},
    }, "additionalProperties": False,
    "$defs": SubmitSegmentPatchAction.model_json_schema().get("$defs", {}),
}


def _receipt_cost(receipts: Sequence[Mapping[str, Any]]) -> float | None:
    """Logical standalone cost, including a reused physical response once."""
    costs: list[float] = []
    for receipt in receipts:
        value = receipt.get("estimated_charge", receipt.get("estimated_usd"))
        if value is None:
            return None
        costs.append(float(value))
    return sum(costs)


def _source_payload(segments: Sequence[SourceSegment]) -> list[dict[str, str]]:
    return [{"segment_id": segment.segment_id, "source_text": segment.text} for segment in segments]


def _context_payload(segments: Sequence[SourceSegment], tape: StoryMemoryTape) -> dict[str, str]:
    return {segment.segment_id: tape.context_for_segment(segment.chapter_id, segment.segment_id).text
            for segment in segments}


TERM_QUALITY_INSTRUCTION = (
    "Each term target is the exact English wording to use in the translated prose: natural, idiomatic "
    "English as a skilled fiction translator would write it. Translate meaningful titles, epithets, sect, "
    "technique, rank, and place names idiomatically rather than as literal calques; keep personal names in "
    "pinyin. A target must contain no parentheses, glosses, alternatives, slashes, or explanations. "
)
_PARENTHETICAL_RE = re.compile(r"\s*[\(（][^\)）]*[\)）]")


def clean_term_target(target: str) -> str:
    """Strip explanatory glosses and alternatives from a proposed story-memory rendering."""
    cleaned = _PARENTHETICAL_RE.sub("", target)
    cleaned = re.split(r"\s*[/;]\s*", cleaned, maxsplit=1)[0]
    return " ".join(cleaned.split()).strip(" ,.:")


class SourceMemoryExtractor:
    """Ask the writer for source-supported facts and auto terms; reject bad citations."""

    def __init__(self, generator: BudgetedGenerator, tape: StoryMemoryTape,
                 *, defer_facts_to_chapter_end: bool = False, instruction_version: str = ""):
        self.generator = generator
        self.tape = tape
        self.defer_facts_to_chapter_end = defer_facts_to_chapter_end
        # "term-quality-v1": ask for bare, idiomatic renderings and strip glosses.
        self.instruction_version = instruction_version
        self.rejections: list[dict[str, str]] = []

    def __call__(self, chapter_id: str, segments: Sequence[SourceSegment], previous: Any) -> tuple[list[MemoryEntry], list[TerminologyCandidate]]:
        prior = [] if previous is None else [
            {"kind": item.kind, "subject": item.subject, "value": item.value,
             "scope": item.scope, "status": item.status}
            for item in previous.entries[-20:]
        ]
        prompt = (
            "Extract up to 12 useful source-grounded facts and 12 Chinese-to-English terms for future "
            "fiction translation. Use only the supplied Chinese source. Every item must cite one exact "
            "Chinese excerpt and segment_id from this chapter. Keep narrator facts separate from "
            "character beliefs or dialogue claims. Do not infer future events or resolve ambiguities. "
            "Entries require kind, subject, value, segment_id, exact_excerpt, scope, status, "
            "source_aliases. Allowed kinds: " + ", ".join(sorted(ENTRY_KINDS)) + ". "
            "Allowed scopes: narrator, character_belief, dialogue_claim, editorial. "
            "Allowed statuses: provisional, active, disputed, superseded. "
            "Terms require source, target, segment_id, exact_excerpt, provisional. "
            + (TERM_QUALITY_INSTRUCTION if self.instruction_version == "term-quality-v1" else "")
            + "Use empty arrays if nothing useful is supported.\n"
            + ("Memory fact availability policy: chapter_end; entries become available only at the "
               "last segment, because this extraction reads the complete current chapter.\n"
               if self.defer_facts_to_chapter_end else "")
            + json.dumps({"chapter_id": chapter_id, "previous_context": prior,
                          "segments": _source_payload(segments)}, ensure_ascii=False)
        )
        payload = self.generator.call("memory_extract", prompt, MEMORY_SCHEMA,
                                      max_output_tokens=4096)
        lookup = {segment.segment_id: segment for segment in segments}
        entries: list[MemoryEntry] = []
        terms: list[TerminologyCandidate] = []
        for index, raw in enumerate(payload["entries"][:12]):
            try:
                sid = raw["segment_id"]
                excerpt = raw["exact_excerpt"]
                if sid not in lookup or not isinstance(excerpt, str) or excerpt not in lookup[sid].text:
                    raise ValueError("citation not in source segment")
                aliases = tuple(value for value in raw.get("source_aliases", [])
                                if isinstance(value, str) and value in lookup[sid].text)
                kind = str(raw["kind"])
                scope = str(raw.get("scope", "narrator"))
                status = str(raw.get("status", "active"))
                if kind not in ENTRY_KINDS or scope not in SCOPES or status not in STATUSES:
                    raise ValueError("invalid memory category")
                if (kind == "editorial_convention") != (scope == "editorial"):
                    raise ValueError("editorial/factual category mismatch")
                entry = MemoryEntry(
                    entry_id=f"{chapter_id}_e{index:03d}", kind=str(raw["kind"]),
                    subject=str(raw["subject"]), value=str(raw["value"]),
                    source_evidence=(SourceEvidence(chapter_id, sid, excerpt),),
                    known_from_chapter=chapter_id,
                    known_from_segment=segments[-1].segment_id if self.defer_facts_to_chapter_end else sid,
                    scope=scope, status=status,
                    created_by_model=self.generator.model, policy_version=self.tape.policy_version,
                    source_aliases=aliases,
                )
                self.tape._validate_entry(entry, current=chapter_id, current_segments=segments,
                                          existing={item.entry_id: item for item in entries})
                entries.append(entry)
            except (KeyError, TypeError, ValueError) as exc:
                self.rejections.append({"chapter_id": chapter_id, "item": f"entry_{index}",
                                        "reason": type(exc).__name__})
        for index, raw in enumerate(payload["terms"][:12]):
            try:
                sid = raw["segment_id"]
                excerpt = raw["exact_excerpt"]
                source = raw["source"]
                if sid not in lookup or not isinstance(excerpt, str) or excerpt not in lookup[sid].text or source not in lookup[sid].text:
                    raise ValueError("term citation not in source segment")
                target = str(raw["target"])
                if self.instruction_version == "term-quality-v1":
                    target = clean_term_target(target)
                if not source or not target:
                    raise ValueError("empty source or target term")
                terms.append(TerminologyCandidate(
                    source=str(source), target=target,
                    source_evidence=(SourceEvidence(chapter_id, sid, excerpt),),
                    provisional=bool(raw.get("provisional", False)),
                ))
            except (KeyError, TypeError, ValueError) as exc:
                self.rejections.append({"chapter_id": chapter_id, "item": f"term_{index}",
                                        "reason": type(exc).__name__})
        return entries, terms


class GenerativeReviewAdapter:
    """One review request proposes bounded exact patches; executor owns mutation."""

    def __init__(self, generator: BudgetedGenerator):
        self.generator = generator

    def review(self, request: AdaptiveReviewRequest) -> AdaptiveReviewResult:
        from agentic_translation.segmentation import sha256_text
        expected_sha = sha256_text(request.draft_text)
        payload = asdict(request)
        if not payload.get("deterministic_findings"):
            payload.pop("deterministic_findings", None)
        if request.instruction_version in {"natural-review-v3", "natural-review-v4", "natural-review-v5", "natural-review-v6"}:
            prompt = (
                "Review this Chinese source window against its English draft. Fix material meaning errors "
                "(changed actors, negation, conditions, quantities, omissions, unsupported additions, or "
                "terminology drift) AND clearly awkward or unnatural English, stiff dialogue, and "
                "inconsistencies with the provided glossary or story context. Preserve correct meaning, "
                "voice, and deliberate ambiguity; never add facts; a no-change answer is valid and expected "
                "for a good segment; make no gratuitous synonym swaps. Return findings and at most one "
                "patch for the current segment. Each finding must contain issue_id, severity (material, "
                "diagnostic, or polish), message, and blocking (boolean; a polish finding is never "
                "blocking). "
                + ("The patch's edits list may contain up to 8 small edits. Each old_text must be a "
                   "nonempty proper substring of the current draft, never the whole segment or chapter, "
                   "even for a short passage. Preserve surrounding text and target only the erroneous "
                   "phrase. Anchor insertions to a short existing phrase. "
                   if request.instruction_version == "natural-review-v6" else
                   "The patch's edits list may contain up to 8 edits, or a single edit whose "
                   "old_text is the entire current segment text (a whole-segment rewrite) when several "
                   "issues interact or the prose needs restructuring. ")
                + "Each patch must contain segment_id, "
                "expected_segment_sha256, edits [{old_text,new_text}], issue_ids, rationale. The exact "
                "expected_segment_sha256 to copy is " + expected_sha + ". "
                "Use an empty patches array if the draft is adequate. old_text must occur exactly once in "
                "draft_text; never replace the whole chapter.\n"
                + (("Restrict style edits to English that is clearly ungrammatical or hard to follow; do not "
                    "rephrase acceptable sentences, and prefer the draft's wording whenever it is correct.\n")
                   if request.instruction_version in {"natural-review-v5", "natural-review-v6"} else "")
                + (("Deterministic checks flagged the items listed in deterministic_findings. Resolve each "
                    "one in your patch: for glossary_required, use the recorded story-memory rendering unless "
                    "the source shows it is wrong in this context (then leave the draft and add a finding "
                    "explaining why); translate any residual Chinese; restore missing bracketed system "
                    "panels. Never delete content to satisfy a check.\n")
                   if request.instruction_version in {"natural-review-v4", "natural-review-v5", "natural-review-v6"}
                   and request.deterministic_findings else "")
                + json.dumps(payload, ensure_ascii=False)
            )
        else:
            prompt = (
                "Review this Chinese source window against its English draft for material meaning errors, "
                "omissions, unsupported additions, actor/negation changes and terminology drift. Preserve "
                "tone and ambiguity. Return findings and at most one exact patch for the current segment. "
                "Each finding must contain issue_id, severity (material or diagnostic), message, and blocking "
                "(boolean). Each patch must contain segment_id, expected_segment_sha256, edits "
                "[{old_text,new_text}], issue_ids, rationale. The exact expected_segment_sha256 to copy is "
                + expected_sha + ". "
                "Use an empty patches array if the draft is adequate. Do not propose "
                "style-only changes. old_text must occur exactly once in draft_text; never replace the whole chapter.\n"
                # Legacy v2 requests predate these fields; omit them so v2 prompt bytes stay stable.
                + json.dumps({key: value for key, value in payload.items() if key != "instruction_version"},
                             ensure_ascii=False)
            )
        try:
            raw = self.generator.call("segment_review", prompt, REVIEW_SCHEMA,
                                      max_output_tokens=4096)
        except BudgetExceeded as exc:
            # The adaptive executor treats ordinary review errors as one failed
            # segment. A spend ceiling is global and must stop new scheduling.
            raise GenerationTransportError("Experiment spend ceiling reached") from exc
        findings = []
        for item in raw["findings"]:
            try:
                message = next((item[key] for key in ("message", "description", "detail", "explanation", "note")
                                if isinstance(item.get(key), str) and item[key].strip()), None)
                if message is None:
                    raise ValueError("Review finding has no explanation")
                severity = item.get("severity") if item.get("severity") in {"material", "diagnostic", "polish"} else "diagnostic"
                findings.append(AdaptiveReviewFinding(
                    issue_id=str(item["issue_id"]),
                    severity=severity,
                    message=message,
                    # A polish finding can never block delivery, regardless of what the model returned.
                    blocking=bool(item.get("blocking", True)) and severity != "polish",
                ))
            except (KeyError, TypeError, ValueError):
                return AdaptiveReviewResult(status="failed", summary="Malformed review finding")
        patches = []
        for item in raw["patches"][:1]:
            try:
                patch = SubmitSegmentPatchAction.model_validate(item)
                if patch.segment_id != request.segment_id or patch.expected_segment_sha256 != expected_sha:
                    raise ValueError("Review patch target or draft hash differs")
                patches.append(patch)
            except (ValueError, TypeError):
                return AdaptiveReviewResult(status="failed", summary="Malformed or stale review patch")
        return AdaptiveReviewResult(status="completed", summary=raw["summary"],
                                    findings=tuple(findings), patches=tuple(patches))


ALIGNMENT_SCHEMA = {
    "type": "object", "required": ["patches"],
    "properties": {"patches": {"type": "array", "items": {
        key: value for key, value in SubmitSegmentPatchAction.model_json_schema().items() if key != "$defs"}}},
    "additionalProperties": False,
    "$defs": SubmitSegmentPatchAction.model_json_schema().get("$defs", {}),
}


class TermAlignmentAdapter:
    """One batched call aligns a chapter's drafts with established story-memory renderings."""

    def __init__(self, generator: BudgetedGenerator):
        self.generator = generator

    def align(self, items: Sequence[TermAlignmentItem]) -> list[SubmitSegmentPatchAction]:
        prompt = (
            "Each item is one segment of a Chinese web-novel chapter with its English draft. The listed "
            "terms are Chinese source terms with this story's established English renderings, chosen "
            "earlier from the source; the draft does not use them yet. For each segment, edit the draft so "
            "every occurrence of each listed source term uses its established rendering, changing only the "
            "words that render that term plus minimal grammar (articles, capitalization, plurals). If the "
            "established rendering is clearly wrong for a particular occurrence (a different sense of the "
            "term), leave that occurrence unchanged. Do not make any other edits. Return at most one patch "
            "per segment with segment_id, expected_segment_sha256 (copy it), edits [{old_text,new_text}] where "
            "old_text occurs exactly once in that segment's draft_text, issue_ids [\"term_alignment\"], and a "
            "short rationale. Return an empty patches array if nothing should change.\n"
            + json.dumps({"segments": [
                {"segment_id": item.segment_id, "source_text": item.source_text, "draft_text": item.draft_text,
                 "expected_segment_sha256": item.expected_segment_sha256,
                 "established_renderings": [{"source": source, "english": target} for source, target in item.terms]}
                for item in items]}, ensure_ascii=False)
        )
        try:
            raw = self.generator.call("term_alignment", prompt, ALIGNMENT_SCHEMA, max_output_tokens=4096)
        except BudgetExceeded as exc:
            raise GenerationTransportError("Experiment spend ceiling reached") from exc
        expected = {item.segment_id: item.expected_segment_sha256 for item in items}
        patches: list[SubmitSegmentPatchAction] = []
        seen: set[str] = set()
        for entry in raw.get("patches", []):
            try:
                patch = SubmitSegmentPatchAction.model_validate(entry)
            except (ValueError, TypeError):
                continue
            if expected.get(patch.segment_id) == patch.expected_segment_sha256 and patch.segment_id not in seen:
                seen.add(patch.segment_id)
                patches.append(patch)
        return patches


def _glossary(tape: StoryMemoryTape, segments: Sequence[SourceSegment],
              translator_terms: Mapping[str, str] | None = None) -> GlossaryParseResult:
    choices: dict[str, str] = {}
    for segment in segments:
        choices.update(tape.glossary_for_segment(segment.chapter_id, segment.segment_id))
    choices.update(translator_terms or {})  # the translator's glossary wins over story memory
    return GlossaryParseResult(entries=[GlossaryEntry(source=source, target=target)
                                        for source, target in choices.items()])


MAX_TRANSLATOR_TERM_LINES = 40


def translator_terms_for_chapter(glossary: Mapping[str, str] | None, source_text: str) -> dict[str, str]:
    """Supplied series-glossary terms that occur in this chapter, with glosses stripped."""
    terms: dict[str, str] = {}
    for source, target in (glossary or {}).items():
        cleaned = clean_term_target(str(target))
        if len(source) >= 2 and source in source_text and 0 < len(cleaned) <= 60:
            terms[source] = cleaned
    return terms


_TAPE_TERM_LINE_RE = re.compile(r"^TERM (\S+) -> ")


def _overlaps_translator_term(source: str, terms: Mapping[str, str]) -> bool:
    return any(source in term or term in source for term in terms)


def _with_translator_terms(context: Mapping[str, str], segments: Sequence[SourceSegment],
                           terms: Mapping[str, str], *, drop_overlapping_memory_terms: bool = False) -> dict[str, str]:
    """Prefix each segment's context pack with the supplied terms that occur in it.

    With ``drop_overlapping_memory_terms``, story-memory TERM lines whose source
    overlaps a translator term (e.g. memory 铸霞修士 vs glossary 铸霞) are removed so
    the prompt never carries two competing renderings.
    """
    merged: dict[str, str] = {}
    for segment in segments:
        lines = [f"TERM {source} -> {target} [translator glossary]"
                 for source, target in sorted(terms.items(), key=lambda item: -len(item[0]))
                 if source in segment.text][:MAX_TRANSLATOR_TERM_LINES]
        base_lines = context.get(segment.segment_id, "").split("\n") if context.get(segment.segment_id) else []
        if drop_overlapping_memory_terms:
            base_lines = [line for line in base_lines
                          if not ((match := _TAPE_TERM_LINE_RE.match(line))
                                  and _overlaps_translator_term(match.group(1), terms))]
        merged[segment.segment_id] = "\n".join([*lines, *base_lines])
    return merged


TERM_ADHERENCE_INSTRUCTION = (
    "Context lines beginning with TERM give this story's established English renderings for Chinese "
    "source terms; use them exactly for those terms unless the current source clearly uses the term in a "
    "different sense. "
)


def _translate(generator: BudgetedGenerator, operation: str, segments: Sequence[SourceSegment],
               context: Mapping[str, str] | None, limit: int,
               *, envelope_mode: str = "json", context_instruction: str = "") -> tuple[str, list[dict[str, str]]]:
    if envelope_mode not in {"json", "segment-text-v1"}:
        raise ValueError("Unknown translation envelope mode")
    output_instruction = (
        "Return exactly one translated_text for each segment_id, in source order. You may merge or split "
        "paragraphs within each segment."
        if envelope_mode == "json" else
        "For each source segment, write a header line exactly <<<SEGMENT:segment_id>>> using its real ID, "
        "then its complete English prose. Include each ID once in source order. Use no JSON or code fences. "
        "You may merge or split paragraphs within each segment."
    )
    prompt = (
        "Translate the supplied Chinese chapter into faithful, natural English. Preserve its events, "
        "relationships, ambiguity, tone, and completeness. Do not add explanatory facts or commentary. "
        + (context_instruction if context is not None else "")
        + output_instruction + "\n"
        + json.dumps({"source_segments": _source_payload(segments),
                      **({"source_grounded_context_by_segment": context} if context is not None else {})},
                     ensure_ascii=False)
    )
    drafts = _generate_covered(generator, operation, prompt, segments, limit,
                               envelope_mode=envelope_mode)
    return render_draft(drafts, source_segments=segments), [
        {"segment_id": item.segment_id, "translated_text": item.translated_text} for item in drafts
    ]


def _generate_covered(generator: BudgetedGenerator, operation: str, prompt: str,
                      segments: Sequence[SourceSegment], limit: int,
                      *, envelope_mode: str = "json"):
    """Permit one identical-rule envelope recovery for N, C, and R."""
    schema = (TRANSLATION_SCHEMA if envelope_mode == "json" else
              {**TRANSLATION_SCHEMA, "x-segment-text-envelope": [item.segment_id for item in segments]})
    payload = generator.call(operation, prompt, schema, max_output_tokens=limit)
    try:
        return parse_draft_envelope(payload, segments)
    except ValueError as exc:
        generator.coverage_recovery_attempted = True
        recovery_instruction = ("Return a complete corrected JSON envelope." if envelope_mode == "json"
                                else "Return complete corrected <<<SEGMENT:segment_id>>> sections, no JSON or fences.")
        recovery_prompt = (
            prompt + "\n\nYour previous segment output did not cover every source segment exactly once: "
            + str(exc) + "\n" + recovery_instruction + " Include all required IDs. "
            "Keep translated text already correct. Previous parsed output:\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        corrected = generator.call(operation, recovery_prompt, schema,
                                   max_output_tokens=limit)
        return parse_draft_envelope(corrected, segments)


def _revise(generator: BudgetedGenerator, segments: Sequence[SourceSegment],
            context: Mapping[str, str], drafts: Sequence[Mapping[str, str]],
            review_limit: int, translation_limit: int,
            *, envelope_mode: str = "json") -> tuple[str, list[dict[str, str]]]:
    packet = {"source_segments": _source_payload(segments), "context_by_segment": context,
              "draft_segments": list(drafts)}
    critique = generator.call(
        "simple_critique",
        "Check the complete Chinese source against the English draft. List at most twelve concrete "
        "material translation errors: changed actors, negation, conditions, quantities, omissions, "
        "unsupported additions, or consequential term conflicts. Each issue must name its segment ID "
        "and fit in one short sentence under 160 characters. State no correct details, confirmations, "
        "quotations, praise, or chapter summary. Return an empty issues array if none.\n"
        + json.dumps(packet, ensure_ascii=False),
        CRITIQUE_SCHEMA, max_output_tokens=review_limit,
    )
    output_instruction = ("Return exactly the original segment IDs in the JSON envelope."
                          if envelope_mode == "json" else
                          "Return one <<<SEGMENT:segment_id>>> header line per original ID, in order, "
                          "with English prose beneath it. Use no JSON or code fences.")
    revision_prompt = (
        "Revise the complete draft once to correct the listed source-grounded issues. Preserve good prose, "
        "structure, and ambiguity; do not invent facts. " + output_instruction + "\n"
        + json.dumps({**packet, "critique": critique}, ensure_ascii=False)
    )
    revised = _generate_covered(generator, "simple_revise", revision_prompt,
                                segments, translation_limit, envelope_mode=envelope_mode)
    return render_draft(revised, source_segments=segments), [
        {"segment_id": item.segment_id, "translated_text": item.translated_text} for item in revised
    ]


def run_chapter_arms(
    *, work_id: str, chapter_id: str, source_text: str,
    segments: Sequence[SourceSegment], tape: StoryMemoryTape,
    run_dir: Path, config: dict[str, Any], memory_cost_usd: float | None,
    memory_elapsed_seconds: float = 0.0,
    existing: dict[str, Any] | None = None,
    generator_factory: Any = BudgetedGenerator,
    decision_provider: Any | None = None,
    save: Any | None = None,
    translator_glossary: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run missing arms only, saving each terminal arm before the next call.

    ``translator_glossary`` is an optional supplied series glossary (source ->
    English). Its terms that occur in the chapter join the contextual prompt and
    the harness's deterministic checks; the naive arm never receives them.
    """
    _verification_policy(config, "jev_adaptive")  # reject invalid routing before any provider call
    row = existing or {"chapter_id": chapter_id, "work_id": work_id, "source_text": source_text,
                       "source_segments": [asdict(item) for item in segments], "arms": {}}
    translator_terms = translator_terms_for_chapter(translator_glossary, source_text)
    context = _context_payload(segments, tape)
    translator_only = translator_terms and config.get("glossary", {}).get("enforce") == "translator_only"
    if translator_terms:
        context = _with_translator_terms(context, segments, translator_terms,
                                         drop_overlapping_memory_terms=bool(translator_only))
    snapshot = tape.snapshot_for_chapter(chapter_id)
    row["memory_snapshot_sha256"] = snapshot.snapshot_sha256
    row["context_by_segment"] = context
    row["preceding_source_context"] = ""  # narrative context is the saved source-derived pack
    writer = config.get("writer", {})
    translation_limit = int(writer.get("translation_max_output_tokens", 16384))
    review_limit = int(writer.get("review_max_output_tokens", 4096))
    envelope_mode = str(config.get("translation_envelope", writer.get("translation_envelope", "json")))

    def generator(phase: str) -> BudgetedGenerator:
        return generator_factory(run_dir, config, phase=phase)

    def finish(arm: str, translation: str, drafts: list[dict[str, str]],
               receipts: list[dict[str, Any]], *, status: str = "delivered",
               warnings: list[Any] | None = None, elapsed: float = 0,
               review_calls: int = 0, patch_calls: int = 0, jev_calls: int = 0,
               evidence: Any = None, cost_unknown: bool = False) -> None:
        direct = _receipt_cost(receipts)
        shared = 0.0 if arm == "naive" else memory_cost_usd
        if arm in {"simple_revise", "always_review", "jev_adaptive"}:
            shared_c = _receipt_cost(row["arms"]["contextual"].get("receipts", []))
            shared = None if shared is None or shared_c is None else shared + shared_c
        cost = None if direct is None or shared is None or cost_unknown else direct + shared
        standalone_elapsed = elapsed + (memory_elapsed_seconds if arm != "naive" else 0.0)
        if arm in {"simple_revise", "always_review", "jev_adaptive"}:
            standalone_elapsed += float(row["arms"]["contextual"].get("physical_arm_elapsed_seconds", 0.0))
        row["arms"][arm] = {
            "translation": translation, "draft_segments": drafts,
            "status": status, "warnings": warnings or [], "receipts": receipts,
            "standalone_cost_usd": cost, "memory_allocated_usd": memory_cost_usd if arm != "naive" else 0,
            "elapsed_seconds": standalone_elapsed,
            "physical_arm_elapsed_seconds": elapsed,
            "memory_allocated_seconds": memory_elapsed_seconds if arm != "naive" else 0.0,
            "review_calls": review_calls,
            "patch_calls": patch_calls, "jev_calls": jev_calls,
            "evidence": evidence,
        }
        if save is not None:
            save(row)

    for arm in configured_arms(config):
        if arm in row["arms"]:
            prior = row["arms"][arm]
            errors = prior.get("warnings", [])
            if (prior.get("status") == "failed_no_output"
                    and any(isinstance(warning, dict) and warning.get("code") == "arm_error"
                            and warning.get("error_type") == "ValueError"
                            and not warning.get("format_recovery_exhausted") for warning in errors)):
                del row["arms"][arm]
            elif (prior.get("status") == "failed_no_output"
                  and "contextual_draft_unavailable" in errors
                  and row["arms"].get("contextual", {}).get("translation")):
                del row["arms"][arm]
            else:
                continue
        started = time.perf_counter()
        gen = generator("system")
        decision_receipts: list[dict[str, Any]] = []
        try:
            if arm in {"naive", "contextual"}:
                text, drafts = _translate(gen, arm, segments, context if arm == "contextual" else None,
                                          translation_limit, envelope_mode=envelope_mode,
                                          context_instruction=TERM_ADHERENCE_INSTRUCTION
                                          if writer.get("context_instruction_version") == "term-adherence-v1" else "")
                finish(arm, text, drafts, gen.receipts, elapsed=time.perf_counter()-started)
            elif not row["arms"].get("contextual", {}).get("translation"):
                finish(arm, "", [], [], status="failed_no_output",
                       warnings=["contextual_draft_unavailable"], elapsed=time.perf_counter()-started)
            elif arm == "simple_revise":
                text, drafts = _revise(gen, segments, context,
                                        row["arms"]["contextual"]["draft_segments"],
                                        review_limit, translation_limit,
                                        envelope_mode=envelope_mode)
                finish(arm, text, drafts, gen.receipts, elapsed=time.perf_counter()-started,
                       review_calls=1)
            else:
                from agentic_translation.segmentation import DraftSegment
                seed = tuple(DraftSegment.from_text(item["segment_id"], item["translated_text"])
                             for item in row["arms"]["contextual"]["draft_segments"])
                decisions = None
                if arm == "jev_adaptive":
                    decisions = BudgetedDecisions(
                        decision_provider or TypeSafeDecisionProvider(record_dir=run_dir / "decisions"),
                        config, run_dir,
                    )
                budgets = config.get("budgets", {})
                budget = BudgetPolicy(
                    max_coordinator_steps=int(budgets.get("max_coordinator_steps", 24)),
                    max_patch_actions=int(budgets.get("max_patch_actions_per_chapter", 12)),
                    max_patch_rounds_per_segment=int(budgets.get("max_patch_rounds_per_segment", 2)),
                    max_generative_review_rounds_per_segment=int(budgets.get("max_review_rounds_per_segment", 2)),
                )
                checkpoint = run_dir / "adaptive_sessions" / f"{work_id}_{chapter_id}_{arm}" / "adaptive_checkpoint.json"
                recovered_terminal = False
                if checkpoint.exists():
                    recovered_terminal = bool(json.loads(checkpoint.read_text(encoding="utf-8")).get("terminal"))
                result = run_adaptive_chapter(
                    segments, seed,
                    glossary=(GlossaryParseResult(entries=[GlossaryEntry(source=source, target=target)
                                                           for source, target in translator_terms.items()])
                              if translator_only else _glossary(tape, segments, translator_terms)),
                    decision_provider=decisions,
                    review_provider=GenerativeReviewAdapter(gen),
                    alignment_provider=TermAlignmentAdapter(gen),
                    verification_policy=_verification_policy(config, arm),
                    budget_policy=budget, context_by_segment=context,
                    memory_snapshot_sha256=snapshot.snapshot_sha256,
                    session_dir=run_dir / "adaptive_sessions" / f"{work_id}_{chapter_id}_{arm}",
                    run_id=f"natural_{arm}", story_slug=work_id,
                )
                decision_receipts = [] if decisions is None else decisions.receipts
                output = result.final_text or ""
                unrecovered_cost = recovered_terminal and bool(result.review_calls or result.screen_calls or result.patch_actions)
                warning_items = [asdict(item) for item in result.warnings]
                if unrecovered_cost:
                    warning_items.append({"code": "cost_receipts_not_rehydrated", "segment_id": None,
                                          "message": "Terminal checkpoint was recovered without original call receipts."})
                evidence = result.to_dict()
                evidence["recovered_terminal_checkpoint"] = recovered_terminal
                finish(arm, output,
                       [{"segment_id": item.segment_id, "translated_text": item.translated_text}
                        for item in result.draft_segments],
                       [*gen.receipts, *decision_receipts],
                       status=result.outcome, warnings=warning_items,
                       elapsed=time.perf_counter()-started, review_calls=result.review_calls,
                       patch_calls=result.patch_actions,
                       jev_calls=(result.screen_calls if recovered_terminal else len(decision_receipts))
                       if decisions is not None else 0,
                       evidence=evidence, cost_unknown=unrecovered_cost)
        except Exception as exc:
            # Keep the assigned arm, all paid attempts, and a scored C fallback for
            # revision arms. The caller decides whether a provider error halts new work.
            fallback = row["arms"].get("contextual", {})
            finish(arm, fallback.get("translation", "") if arm not in {"naive", "contextual"} else "",
                   fallback.get("draft_segments", []) if arm not in {"naive", "contextual"} else [],
                   [*gen.receipts, *decision_receipts],
                   status="delivered_with_warnings" if fallback.get("translation") and arm not in {"naive", "contextual"} else "failed_no_output",
                   warnings=[{"code": "generation_output_exhausted" if isinstance(exc, GenerationOutputError)
                              else "arm_error", "error_type": type(exc).__name__,
                              "message": str(exc)[:200],
                              "format_recovery_exhausted": isinstance(exc, GenerationOutputError)
                              or bool(getattr(gen, "coverage_recovery_attempted", False))}],
                   elapsed=time.perf_counter()-started,
                   review_calls=1 if any(item.get("operation") == "simple_critique"
                                         for item in gen.receipts) else 0)
            if ((isinstance(exc, LLMProviderUnavailable) and not isinstance(exc, GenerationOutputError))
                    or isinstance(exc, BudgetExceeded)
                    or getattr(exc, "http_status", None) in {401, 402, 403}):
                raise
    return row
