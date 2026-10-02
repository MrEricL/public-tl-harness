"""Unattended segment review over the existing atomic repair executor.

The scheduler decides which complete source windows need model attention.
``RepairToolExecutor`` remains the sole owner of candidate mutation and QA.
Every receipt binds source, current draft, adjacent windows, memory, and policy;
an edited neighbor therefore makes a prior contextual judgment stale.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import json
import re
from pathlib import Path
from typing import Literal, Mapping, Protocol, Sequence

from .agent_models import SubmitSegmentPatchAction
from .agent_repair import RepairToolExecutor
from .agent_session import SessionStore
from .decisions.models import DEFAULT_THRESHOLDS, QUESTION_SCHEMA_VERSION, DecisionRequest, DecisionResult
from .decisions.questions import question_schema_digest, request_id as decision_request_id
from .models import GlossaryParseResult
from .qa import _glossary_source_spans
from .text import literal_term_pattern
from .segmentation import (
    ContextReceipt,
    DraftSegment,
    SourceSegment,
    make_context_receipt,
    receipt_is_current,
    render_draft,
    sha256_text,
    validate_source_segments,
)


VerificationMode = Literal["full_segment_review", "adaptive_segment_screen"]
DeliveryOutcome = Literal["delivered", "delivered_with_warnings", "failed_no_output"]


@dataclass(frozen=True)
class VerificationPolicy:
    version: str = "adaptive-verification.v1"
    mode: VerificationMode = "adaptive_segment_screen"
    instruction_version: str = "adaptive-instructions.v1"
    decision_model: str = "jev-1.13.0"
    routing_policy_version: str = "adaptive-routing.v1"
    routing_thresholds: Mapping[str, float] = field(default_factory=lambda: DEFAULT_THRESHOLDS.copy())
    # Routing v2 knobs (ignored under routing_policy_version "adaptive-routing.v1").
    diagnostic_min_flags: int = 2
    route_source_incomplete: bool = False
    # Gates whether cosmetic/legacy-rule QA finding categories block delivery
    # (and, under v2, whether they can reject a patch in the QA gate).
    qa_warning_policy_version: str = "qa-warnings.v1"
    # Route segments with deterministic findings (story-memory glossary term not
    # used, residual Chinese, system-panel mismatch) to review and show them to
    # the reviewer. Off by default so older policies keep their semantics.
    route_deterministic_findings: bool = False
    # Re-screen a candidate segment with Jev and reject new material flags.
    candidate_rescreen: bool = False
    # Before screening, send every unused story-memory term in the chapter to
    # one batched alignment request; patches pass the same executor guards.
    term_alignment_pass: bool = False
    # Material threshold for the candidate re-screen; None reuses the routing one.
    candidate_rescreen_threshold: float | None = None
    # Reject patches that add parenthetical glosses or slash alternatives.
    gloss_guard: bool = False
    # Reject patches that copy a paragraph already present elsewhere in the chapter.
    duplication_guard: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"full_segment_review", "adaptive_segment_screen"}:
            raise ValueError("Unknown adaptive verification mode")
        if (not self.version or not self.instruction_version or not self.decision_model
                or not self.routing_policy_version or not self.qa_warning_policy_version):
            raise ValueError("Adaptive policy versions and decision model are required")
        thresholds = dict(self.routing_thresholds)
        if set(thresholds) != set(DEFAULT_THRESHOLDS):
            raise ValueError("Adaptive routing thresholds must define material, diagnostic, readability")
        material, diagnostic, readability = (thresholds[key] for key in ("material", "diagnostic", "readability"))
        if not (0 <= diagnostic <= material <= 1 and 0 <= readability <= 3):
            raise ValueError("Adaptive routing thresholds are out of range")
        if not isinstance(self.diagnostic_min_flags, int) or isinstance(self.diagnostic_min_flags, bool) or self.diagnostic_min_flags < 1:
            raise ValueError("diagnostic_min_flags must be a positive integer")


@dataclass(frozen=True)
class InteractionPolicy:
    version: str = "interaction.v1"
    mode: Literal["unattended"] = "unattended"
    run_local_glossary_writes: bool = True
    human_approval: bool = False

    def __post_init__(self) -> None:
        if self.mode != "unattended" or self.human_approval:
            raise ValueError("Unattended interaction cannot require human approval")


@dataclass(frozen=True)
class BudgetPolicy:
    version: str = "adaptive-budget.v1"
    max_coordinator_steps: int = 24
    max_patch_actions: int = 12
    max_patch_rounds_per_segment: int = 2
    max_generative_review_rounds_per_segment: int = 2
    max_additional_context_requests_per_segment: int = 1
    max_stronger_fallback_calls_per_chapter: int = 1
    max_transient_transport_retries: int = 2

    def __post_init__(self) -> None:
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
               for name, value in asdict(self).items() if name.startswith("max_")):
            raise ValueError("Adaptive budgets must be nonnegative")


@dataclass(frozen=True)
class AdaptiveReviewRequest:
    segment_id: str
    source_text: str
    draft_text: str
    context_text: str
    issue_ids: tuple[str, ...]
    mode: VerificationMode
    review_round: int
    rejection: str | None = None
    instruction_version: str = "adaptive-instructions.v1"
    deterministic_findings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdaptiveReviewFinding:
    issue_id: str
    severity: Literal["material", "diagnostic", "polish"]
    message: str
    blocking: bool = True


@dataclass(frozen=True)
class AdaptiveReviewResult:
    status: Literal["completed", "failed"]
    summary: str
    findings: tuple[AdaptiveReviewFinding, ...] = ()
    patches: tuple[SubmitSegmentPatchAction, ...] = ()


@dataclass(frozen=True)
class TermAlignmentItem:
    segment_id: str
    source_text: str
    draft_text: str
    expected_segment_sha256: str
    terms: tuple[tuple[str, str], ...]


class TermAlignmentProvider(Protocol):
    def align(self, items: Sequence[TermAlignmentItem]) -> Sequence[SubmitSegmentPatchAction]: ...


def _source_supplies_english(term: str, source_text: str) -> bool:
    """True when the source itself follows the term with Latin-script English, e.g. 雾魇(Mistwraith)."""
    return re.search(re.escape(term) + r"\s*[（(]\s*[A-Za-z]", source_text) is not None


def _unused_story_terms(source_text: str, draft_text: str, glossary: GlossaryParseResult) -> tuple[tuple[str, str], ...]:
    _, independent_spans = _glossary_source_spans(source_text, glossary)
    return tuple((entry.source, entry.target) for entry in glossary.entries
                 if len(entry.source) >= 2 and entry.target and independent_spans.get(entry.source)
                 and not literal_term_pattern(entry.target).search(draft_text)
                 and not _source_supplies_english(entry.source, source_text))


class AdaptiveReviewProvider(Protocol):
    def review(self, request: AdaptiveReviewRequest) -> AdaptiveReviewResult: ...


class AdaptiveDecisionProvider(Protocol):
    def evaluate(self, request: DecisionRequest) -> DecisionResult: ...


def _valid_decision(result: DecisionResult, request: DecisionRequest) -> bool:
    receipt = result.receipt
    return (
        receipt.request_id == decision_request_id(request)
        and receipt.segment_id == request.segment_id
        and receipt.source_sha256 == request.source_sha256
        and receipt.draft_sha256 == request.draft_sha256
        and receipt.context_sha256 == request.context_sha256
        and receipt.candidate_sha256 == (sha256_text(request.candidate_text) if request.candidate_text is not None else None)
        and receipt.question_schema_version == QUESTION_SCHEMA_VERSION
        and receipt.question_schema_sha256 == question_schema_digest(request)
        and receipt.routing_policy_version == request.routing_policy_version
        and receipt.routing_thresholds == request.routing_thresholds
    )


def _route_and_issues(
    decision: DecisionResult | None, policy: VerificationPolicy
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Return (route_level, issue_ids that drive review/warnings, informational notes).

    Under routing v1 this reproduces the exact prior behavior byte-for-byte:
    any nonempty ``issue_ids`` (material or the full diagnostic flag set)
    routes to generative review, and a lone ``source_incomplete`` answer
    forces a material route with a warning.

    Under routing v2 ("adaptive-routing.v2"): a material flag always routes
    to review. A diagnostic-band flag only routes to review if at least
    ``policy.diagnostic_min_flags`` distinct error categories land in the
    band, or readability is severe; otherwise the segment is screened-clean
    and the diagnostic categories are recorded as notes only. A lone
    ``source_incomplete`` answer is recorded as a note (no routing, no
    warning) unless ``policy.route_source_incomplete`` is set.
    """
    if decision is None or decision.status != "ok":
        return "fallback", ("router_fallback",), ()
    if policy.routing_policy_version != "adaptive-routing.v2":
        route = decision.route_level
        issue_ids = tuple(decision.material_flags)
        if route == "diagnostic":
            issue_ids = tuple(key for key, value in decision.receipt.derived_flags.items() if value)
        if decision.receipt.derived_flags.get("source_incomplete"):
            issue_ids = (*issue_ids, "source_incomplete")
        return route, tuple(dict.fromkeys(issue_ids)), ()

    material_flags = tuple(decision.material_flags)
    source_incomplete = bool(decision.receipt.derived_flags.get("source_incomplete"))
    if material_flags:
        return "material", material_flags, ()
    diagnostic_names = decision.diagnostic_flag_names
    severe = bool(decision.receipt.derived_flags.get("severe_readability"))
    if len(diagnostic_names) >= policy.diagnostic_min_flags or severe:
        issue_ids = diagnostic_names + (("severe_readability",) if severe else ())
        return "diagnostic", tuple(dict.fromkeys(issue_ids)), diagnostic_names
    if source_incomplete:
        if policy.route_source_incomplete:
            return "material", ("source_incomplete",), diagnostic_names
        return "clean", (), (*diagnostic_names, "source_incomplete")
    return "clean", (), diagnostic_names


# QA finding categories that are cosmetic/legacy-rule artifacts rather than
# translation-quality signals for natural-style fiction: "chinese_punctuation"
# fires on ordinary English
# typographic quotes (U+201C/U+201D/U+2018/U+2019), and "heading_format"
# expects a bare numeral heading and rejects natural spelled-out or
# subtitle-bearing chapter headings. Gated behind qa_warning_policy_version
# "qa-warnings.v2" so legacy runs keep their exact prior semantics.
INFORMATIONAL_QA_CHECKS = frozenset({"chinese_punctuation", "heading_format"})


_DIGIT_RUN_RE = re.compile(r"\d+")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


def _polish_guard_violation(old_text: str, new_text: str, source_text: str = "") -> str | None:
    """Cheap, deterministic content-loss veto applied to every patch.

    Only unambiguous loss is vetoed: a >30% character shrink, or an Arabic
    number that appears in the Chinese source and the old draft but vanishes
    from the candidate. Surface changes that can be meaning-preserving
    ("3" -> "three", direct -> indirect speech) are only signals; see
    ``_content_signals``.
    """
    if old_text and len(new_text) < 0.7 * len(old_text):
        return "Candidate segment text shrank by more than 30% of its original length."
    source_numbers = set(_DIGIT_RUN_RE.findall(source_text))
    lost = sorted(number for number in set(_DIGIT_RUN_RE.findall(old_text)) & source_numbers
                  if number not in set(_DIGIT_RUN_RE.findall(new_text)))
    if lost:
        return "Candidate segment text dropped source number(s): " + ", ".join(lost[:5]) + "."
    return None


_GLOSS_PAREN_RE = re.compile(r"[\(（][^\)）]{2,80}[\)）]")
_SLASH_ALTERNATIVE_RE = re.compile(r"[^\W\d_]\s*/\s*[^\W\d_]")


def _gloss_guard_violation(old_text: str, new_text: str, source_text: str) -> str | None:
    """Reject a patch that adds translator's-note glosses or unresolved alternatives.

    Development judges repeatedly penalized repairs that inserted parenthetical
    explanations ("(an artisan's charm)") or slash alternatives ("river wraith
    / drowned shade"). A parenthetical is allowed when the source segment has
    at least as many of its own.
    """
    old_parens, new_parens = len(_GLOSS_PAREN_RE.findall(old_text)), len(_GLOSS_PAREN_RE.findall(new_text))
    if new_parens > old_parens and new_parens > len(_GLOSS_PAREN_RE.findall(source_text)):
        return "Candidate adds a parenthetical gloss that the source does not have."
    if len(_SLASH_ALTERNATIVE_RE.findall(new_text)) > len(_SLASH_ALTERNATIVE_RE.findall(old_text)):
        return "Candidate adds a slash-separated alternative rendering."
    return None


def _duplication_violation(old_text: str, new_text: str, other_segments: Sequence[str]) -> str | None:
    """Reject a candidate that repeats a paragraph from another segment or from itself."""
    elsewhere = "\n".join(other_segments)
    paragraphs = [block.strip() for block in new_text.split("\n") if len(block.strip()) >= 40]
    for block in paragraphs:
        if block in elsewhere:
            return "Candidate repeats a paragraph that already appears in another segment."
        if new_text.count(block) > max(1, old_text.count(block)):
            return "Candidate repeats a paragraph within the segment."
    return None


def _content_signals(old_text: str, new_text: str) -> list[str]:
    """Non-vetoing observations recorded with a candidate for later audit."""
    signals = []
    if sum(ch.isdigit() for ch in new_text) < sum(ch.isdigit() for ch in old_text):
        signals.append("fewer_digits")
    quote_chars = "\"“”"
    if sum(new_text.count(ch) for ch in quote_chars) < sum(old_text.count(ch) for ch in quote_chars):
        signals.append("fewer_quote_marks")
    return signals


def _segment_deterministic_findings(
    source_text: str, draft_text: str, glossary: GlossaryParseResult,
) -> tuple[str, ...]:
    """Segment-local checks the reviewer can act on, phrased as instructions-free facts."""
    from .qa import panel_count

    findings: list[str] = []
    for source, target in _unused_story_terms(source_text, draft_text, glossary):
        findings.append(f"glossary_required: source term {source} has the story-memory rendering "
                        f"\"{target}\", which this draft segment does not use")
    residual = _CJK_RUN_RE.findall(draft_text)
    if residual:
        findings.append("residual_chinese: untranslated Chinese remains in the draft: " + ", ".join(residual[:5]))
    source_panels, draft_panels = panel_count(source_text), panel_count(draft_text)
    if source_panels != draft_panels:
        findings.append(f"system_panel_count: source has {source_panels} bracketed system panel(s); "
                        f"draft has {draft_panels}")
    return tuple(findings[:8])


@dataclass(frozen=True)
class SegmentEvidence:
    level: Literal["jev_screen", "generative_review"]
    context: ContextReceipt
    context_sha256: str
    status: Literal["ok", "unavailable", "failed"]
    issue_ids: tuple[str, ...] = ()
    blocking_issue_ids: tuple[str, ...] = ()
    provider_request_id: str | None = None
    # Recorded but non-routing observations (e.g. routing v2's screened-clean
    # diagnostic band, or a lone source_incomplete answer). Never drives a
    # warning or generative review by itself.
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdaptiveWarning:
    code: str
    segment_id: str | None
    message: str


@dataclass
class AdaptiveChapterResult:
    outcome: DeliveryOutcome
    final_text: str | None
    draft_segments: tuple[DraftSegment, ...]
    initial_draft_sha256: str | None
    source_sha256: str
    identity_sha256: str
    verification_policy: VerificationPolicy
    interaction_policy: InteractionPolicy
    budget_policy: BudgetPolicy
    memory_snapshot_sha256: str
    evidence: tuple[SegmentEvidence, ...]
    warnings: tuple[AdaptiveWarning, ...]
    events: tuple[dict[str, object], ...]
    patch_actions: int
    review_calls: int
    screen_calls: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "AdaptiveChapterResult":
        """Restore a terminal checkpoint without rerunning a provider call."""
        drafts = tuple(DraftSegment(**item) for item in data["draft_segments"])
        evidence: list[SegmentEvidence] = []
        for item in data["evidence"]:
            context_data = dict(item["context"])
            context_data["included_neighbor_ids"] = tuple(context_data["included_neighbor_ids"])
            context_data["neighbor_hashes"] = tuple(tuple(row) for row in context_data["neighbor_hashes"])
            context_data["memory_entry_hashes"] = tuple(context_data["memory_entry_hashes"])
            evidence.append(SegmentEvidence(
                level=item["level"], context=ContextReceipt(**context_data),
                context_sha256=item["context_sha256"], status=item["status"],
                issue_ids=tuple(item["issue_ids"]),
                blocking_issue_ids=tuple(item["blocking_issue_ids"]),
                provider_request_id=item["provider_request_id"],
                notes=tuple(item.get("notes", ())),
            ))
        result = cls(
            outcome=data["outcome"], final_text=data["final_text"], draft_segments=drafts,
            initial_draft_sha256=data["initial_draft_sha256"],
            source_sha256=data["source_sha256"], identity_sha256=data["identity_sha256"],
            verification_policy=VerificationPolicy(**data["verification_policy"]),
            interaction_policy=InteractionPolicy(**data["interaction_policy"]),
            budget_policy=BudgetPolicy(**data["budget_policy"]),
            memory_snapshot_sha256=data["memory_snapshot_sha256"],
            evidence=tuple(evidence),
            warnings=tuple(AdaptiveWarning(**item) for item in data["warnings"]),
            events=tuple(dict(item) for item in data["events"]),
            patch_actions=data["patch_actions"], review_calls=data["review_calls"],
            screen_calls=data["screen_calls"],
        )
        if result.final_text is not None and render_draft(result.draft_segments) != result.final_text:
            raise ValueError("Terminal adaptive checkpoint final text does not match its segments")
        return result


def failed_adaptive_chapter(
    source_text: str,
    reason: str,
    *,
    verification_policy: VerificationPolicy | None = None,
    interaction_policy: InteractionPolicy | None = None,
    budget_policy: BudgetPolicy | None = None,
    memory_snapshot_sha256: str | None = None,
    session_dir: str | Path | None = None,
) -> AdaptiveChapterResult:
    """Record a generation/coverage failure before a complete draft exists."""
    if not reason:
        raise ValueError("A failed output needs a concrete reason")
    verification = verification_policy or VerificationPolicy()
    interaction = interaction_policy or InteractionPolicy()
    budget = budget_policy or BudgetPolicy()
    memory_hash = memory_snapshot_sha256 or sha256_text("")
    identity = _identity_hash(source_text, "", verification, interaction, budget, memory_hash, {})
    result = AdaptiveChapterResult(
        outcome="failed_no_output", final_text=None, draft_segments=(),
        initial_draft_sha256=None, source_sha256=sha256_text(source_text),
        identity_sha256=identity, verification_policy=verification,
        interaction_policy=interaction, budget_policy=budget,
        memory_snapshot_sha256=memory_hash, evidence=(),
        warnings=(AdaptiveWarning("failed_no_output", None, reason),),
        events=({"kind": "generation_or_coverage_failed", "reason": reason},),
        patch_actions=0, review_calls=0, screen_calls=0,
    )
    if session_dir is not None:
        store = SessionStore(session_dir)
        store.append("generation_or_coverage_failed", {"reason": reason})
        _write_checkpoint(store, identity, (), result.events, verification, interaction, budget, memory_hash, result)
    return result


# Keep the established policy canonicalization within the versioned identity.
# Unbound v1 checkpoints still require a fresh session; v2 also binds the glossary.
_LEGACY_POLICY_DEFAULTS = {
    "diagnostic_min_flags": 2,
    "route_source_incomplete": False,
    "qa_warning_policy_version": "qa-warnings.v1",
    "route_deterministic_findings": False,
    "candidate_rescreen": False,
    "term_alignment_pass": False,
    "candidate_rescreen_threshold": None,
    "gloss_guard": False,
    "duplication_guard": False,
}


def _identity_verification(verification: VerificationPolicy) -> dict[str, object]:
    payload = asdict(verification)
    for key, default in _LEGACY_POLICY_DEFAULTS.items():
        if payload.get(key) == default:
            payload.pop(key, None)
    return payload


def _identity_hash(
    source_text: str,
    initial_text: str,
    verification: VerificationPolicy,
    interaction: InteractionPolicy,
    budget: BudgetPolicy,
    memory_snapshot_sha256: str,
    context_by_segment: Mapping[str, str],
    glossary: GlossaryParseResult | None = None,
) -> str:
    payload = {
        "schema_version": "adaptive-session.v2",
        # Bind the complete effective terminology contract, including aliases
        # and blocked variants. A changed glossary must not reuse clean receipts.
        "glossary": (glossary or GlossaryParseResult(entries=[])).model_dump(mode="json"),
        "source_sha256": sha256_text(source_text),
        "initial_draft_sha256": sha256_text(initial_text),
        "verification": _identity_verification(verification),
        "interaction": asdict(interaction),
        "budget": asdict(budget),
        "memory_snapshot_sha256": memory_snapshot_sha256,
        "segment_context_sha256": sha256_text(json.dumps(dict(context_by_segment), sort_keys=True)),
    }
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _neighbors(
    index: int, sources: Sequence[SourceSegment], drafts: Sequence[DraftSegment]
) -> tuple[tuple[SourceSegment, DraftSegment], ...]:
    return tuple((sources[n], drafts[n]) for n in (index - 1, index + 1) if 0 <= n < len(sources))


def _context_text(
    index: int,
    sources: Sequence[SourceSegment],
    drafts: Sequence[DraftSegment],
    extra_context: str,
) -> str:
    data = {
        "neighbor_segments": [
            {"segment_id": source.segment_id, "source": source.text, "draft": draft.translated_text}
            for source, draft in _neighbors(index, sources, drafts)
        ],
        "story_context": extra_context,
    }
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _receipt(
    index: int,
    sources: Sequence[SourceSegment],
    drafts: Sequence[DraftSegment],
    memory_snapshot_sha256: str,
    policy: VerificationPolicy,
    level: str,
) -> ContextReceipt:
    return make_context_receipt(
        sources[index], drafts[index], neighbors=_neighbors(index, sources, drafts),
        memory_snapshot_sha256=memory_snapshot_sha256,
        instruction_version=policy.instruction_version,
        review_policy=policy.mode,
        evidence_level=level,
        tool_contract_version="segment-patch.v1",
    )


def _current(
    evidence: SegmentEvidence | None,
    index: int,
    sources: Sequence[SourceSegment],
    drafts: Sequence[DraftSegment],
    memory_snapshot_sha256: str,
    policy: VerificationPolicy,
    context_by_segment: Mapping[str, str],
) -> bool:
    if evidence is None:
        return False
    if evidence.context_sha256 != sha256_text(_context_text(index, sources, drafts, context_by_segment.get(sources[index].segment_id, ""))):
        return False
    return receipt_is_current(
        evidence.context, sources[index], drafts[index],
        neighbors=_neighbors(index, sources, drafts),
        memory_snapshot_sha256=memory_snapshot_sha256,
        instruction_version=policy.instruction_version,
        review_policy=policy.mode,
        tool_contract_version="segment-patch.v1",
    )


def _write_checkpoint(
    store: SessionStore | None,
    identity_sha256: str,
    drafts: Sequence[DraftSegment],
    events: Sequence[dict[str, object]],
    verification_policy: VerificationPolicy,
    interaction_policy: InteractionPolicy,
    budget_policy: BudgetPolicy,
    memory_snapshot_sha256: str,
    result: AdaptiveChapterResult | None = None,
) -> None:
    if store is None:
        return
    payload: dict[str, object] = {
        "schema_version": "adaptive-checkpoint.v2",
        "identity_sha256": identity_sha256,
        "verification_policy": asdict(verification_policy),
        "interaction_policy": asdict(interaction_policy),
        "budget_policy": asdict(budget_policy),
        "memory_snapshot_sha256": memory_snapshot_sha256,
        "draft_segments": [asdict(segment) for segment in drafts],
        "event_count": len(events),
        "terminal": result is not None,
    }
    if result is not None:
        payload["result"] = result.to_dict()
    store._atomic_write_json(store.session_dir / "adaptive_checkpoint.json", payload)


def run_adaptive_chapter(
    source_segments: Sequence[SourceSegment],
    draft_segments: Sequence[DraftSegment],
    *,
    glossary: GlossaryParseResult,
    decision_provider: AdaptiveDecisionProvider | None,
    review_provider: AdaptiveReviewProvider | None,
    verification_policy: VerificationPolicy | None = None,
    interaction_policy: InteractionPolicy | None = None,
    budget_policy: BudgetPolicy | None = None,
    context_by_segment: Mapping[str, str] | None = None,
    memory_snapshot_sha256: str | None = None,
    session_dir: str | Path | None = None,
    run_id: str = "adaptive",
    story_slug: str = "story",
    alignment_provider: TermAlignmentProvider | None = None,
) -> AdaptiveChapterResult:
    """Review a complete chapter without a human approval state.

    A missing/failed decision routes to generative review. A failed review or
    exhausted budget produces a warning output, never a false clean receipt.
    """
    policy = verification_policy or VerificationPolicy()
    interaction = interaction_policy or InteractionPolicy()
    budget = budget_policy or BudgetPolicy()
    context_map = dict(context_by_segment or {})
    memory_hash = memory_snapshot_sha256 or sha256_text("")
    sources = tuple(source_segments)
    drafts = list(draft_segments)
    source_text = "".join(item.text for item in sources)
    validate_source_segments(source_text, sources)
    expected_ids = [item.segment_id for item in sources]
    if [item.segment_id for item in drafts] != expected_ids:
        raise ValueError("Draft segment IDs/order must match complete source coverage")
    initial_text = render_draft(drafts)
    identity = _identity_hash(source_text, initial_text, policy, interaction, budget, memory_hash, context_map, glossary)
    store = SessionStore(session_dir) if session_dir is not None else None
    if store is not None:
        checkpoint = store.session_dir / "adaptive_checkpoint.json"
        if checkpoint.exists():
            previous = json.loads(checkpoint.read_text(encoding="utf-8"))
            if previous.get("schema_version") != "adaptive-checkpoint.v2":
                raise ValueError("Legacy adaptive checkpoint does not bind its glossary; use a fresh session directory")
            if previous.get("identity_sha256") != identity:
                raise ValueError("Adaptive session identity differs from persisted checkpoint")
            if previous.get("terminal"):
                saved = previous.get("result")
                if not isinstance(saved, dict):
                    raise ValueError("Terminal adaptive checkpoint has no result")
                recovered = AdaptiveChapterResult.from_dict(saved)
                if (
                    recovered.identity_sha256 != identity
                    or recovered.source_sha256 != sha256_text(source_text)
                    or recovered.initial_draft_sha256 != sha256_text(initial_text)
                    or recovered.verification_policy != policy
                    or recovered.interaction_policy != interaction
                    or recovered.budget_policy != budget
                    or recovered.memory_snapshot_sha256 != memory_hash
                    or [item.segment_id for item in recovered.draft_segments] != expected_ids
                ):
                    raise ValueError("Terminal adaptive result does not match session identity")
                for evidence in recovered.evidence:
                    index = expected_ids.index(evidence.context.segment_id)
                    if not _current(evidence, index, sources, recovered.draft_segments, memory_hash, policy, context_map):
                        raise ValueError("Terminal adaptive result contains stale coverage evidence")
                return recovered
            raise ValueError("Adaptive session was interrupted; no safe mid-chapter resume is available")
    executor = RepairToolExecutor(
        source_text=source_text,
        translated_text=initial_text,
        glossary=glossary,
        run_id=run_id,
        story_slug=story_slug,
        chapter=sources[0].chapter_id,
        allow_nonregressing_patches=True,
        ignored_qa_checks=INFORMATIONAL_QA_CHECKS if policy.qa_warning_policy_version == "qa-warnings.v2" else (),
    )
    screens: dict[str, SegmentEvidence] = {}
    reviews: dict[str, SegmentEvidence] = {}
    events: list[dict[str, object]] = []
    warnings: list[AdaptiveWarning] = []
    review_rounds: dict[str, int] = {}
    patch_rounds: dict[str, int] = {}
    rejection: dict[str, str] = {}
    patch_actions = review_calls = screen_calls = coordinator_steps = 0
    budget_exhausted_segments: set[str] = set()
    queue = deque(range(len(sources)))
    queued = set(queue)
    initial_screen_pass = policy.mode == "adaptive_segment_screen"

    def record(kind: str, segment_id: str, **details: object) -> None:
        event = {"kind": kind, "segment_id": segment_id, **details}
        events.append(event)
        if store is not None:
            store.append(kind, event)
            _write_checkpoint(store, identity, drafts, events, policy, interaction, budget, memory_hash)

    if policy.term_alignment_pass and alignment_provider is not None:
        items = []
        for index, source in enumerate(sources):
            terms = _unused_story_terms(source.text, drafts[index].translated_text, glossary)
            if terms:
                items.append(TermAlignmentItem(source.segment_id, source.text, drafts[index].translated_text,
                                               drafts[index].draft_sha256, terms))
        if items:
            record("term_alignment_requested", items[0].segment_id,
                   segments=[item.segment_id for item in items], terms=sum(len(item.terms) for item in items))
            review_calls += 1
            try:
                proposed = list(alignment_provider.align(items))
            except Exception as exc:
                if getattr(exc, "fatal_provider_error", False):
                    raise
                proposed = []
                record("term_alignment_failed", items[0].segment_id, error_type=type(exc).__name__)
            wanted = {item.segment_id for item in items}
            for action in proposed:
                if action.segment_id not in wanted or patch_actions >= budget.max_patch_actions:
                    continue
                index = expected_ids.index(action.segment_id)
                source, draft = sources[index], drafts[index]

                def alignment_guard(candidate_text: str, draft: DraftSegment = draft,
                                    source: SourceSegment = source) -> tuple[bool, str]:
                    violation = _polish_guard_violation(draft.translated_text, candidate_text, source.text)
                    if violation is None and policy.gloss_guard:
                        violation = _gloss_guard_violation(draft.translated_text, candidate_text, source.text)
                    if violation is None and policy.duplication_guard:
                        violation = _duplication_violation(
                            draft.translated_text, candidate_text,
                            [item.translated_text for item in drafts if item.segment_id != draft.segment_id])
                    return (False, violation) if violation else (True, "")

                patch_actions += 1
                execution = executor.submit_segment_patch(
                    action, [(item.segment_id, item.translated_text) for item in drafts],
                    candidate_guard=alignment_guard,
                )
                if not execution.observation.ok:
                    record("term_alignment_rejected", action.segment_id,
                           reason=execution.observation.data.get("reason", "qa_gate"),
                           message=execution.observation.message)
                    continue
                drafts[index] = drafts[index].with_text(str(execution.observation.data["candidate_segment_text"]))
                record("term_alignment_applied", action.segment_id, draft_sha256=drafts[index].draft_sha256)

    while queue or initial_screen_pass:
        if not queue:
            initial_screen_pass = False

            def priority(index: int) -> tuple[int, int]:
                evidence = screens[sources[index].segment_id]
                issues = set(evidence.issue_ids)
                if "router_fallback" in issues or "source_incomplete" in issues or issues.intersection({
                    "actor_relation_error", "negation_condition_error", "quantity_time_error", "unsupported_addition"
                }):
                    return (0, index)
                if issues.intersection({"omitted_material", "terminology_conflict"}):
                    return (1, index)
                if issues:
                    return (2, index)
                return (3, index)

            queue.extend(sorted(range(len(sources)), key=priority))
            queued = set(queue)
        index = queue.popleft()
        queued.discard(index)
        source = sources[index]
        draft = drafts[index]
        sid = source.segment_id
        context = _context_text(index, sources, drafts, context_map.get(sid, ""))
        screen = screens.get(sid)
        review = reviews.get(sid)
        if policy.mode == "adaptive_segment_screen" and not _current(screen, index, sources, drafts, memory_hash, policy, context_map):
            if decision_provider is None:
                decision = None
            else:
                try:
                    decision_request = DecisionRequest(
                        segment_id=sid, source_text=source.text, draft_text=draft.translated_text,
                        context_text=context, requested_model=policy.decision_model,
                        routing_policy_version=policy.routing_policy_version,
                        routing_thresholds=dict(policy.routing_thresholds),
                    )
                    decision = decision_provider.evaluate(decision_request)
                    if not _valid_decision(decision, decision_request):
                        record("screen_stale", sid)
                        decision = None
                except Exception as exc:  # provider boundary; retain exact failure class only
                    if getattr(exc, "fatal_provider_error", False):
                        raise
                    decision = None
                    record("screen_failed", sid, error_type=type(exc).__name__)
            screen_calls += 1
            route, issue_ids, notes = _route_and_issues(decision, policy)
            screen = SegmentEvidence(
                level="jev_screen", context=_receipt(index, sources, drafts, memory_hash, policy, "jev_screen"),
                context_sha256=sha256_text(context),
                status="ok" if decision is not None and decision.status == "ok" else "unavailable",
                issue_ids=tuple(dict.fromkeys(issue_ids)),
                notes=tuple(dict.fromkeys(notes)),
                provider_request_id=decision.receipt.request_id if decision is not None else None,
            )
            screens[sid] = screen
            record("segment_screened", sid, route=route, issue_ids=list(screen.issue_ids),
                   notes=list(screen.notes), status=screen.status)
        if initial_screen_pass:
            continue
        deterministic = (_segment_deterministic_findings(source.text, draft.translated_text, glossary)
                         if policy.route_deterministic_findings else ())
        route_review = (policy.mode == "full_segment_review" or (screen is not None and bool(screen.issue_ids))
                        or bool(deterministic))
        if not route_review or _current(review, index, sources, drafts, memory_hash, policy, context_map):
            continue
        if review_provider is None or coordinator_steps >= budget.max_coordinator_steps or review_rounds.get(sid, 0) >= budget.max_generative_review_rounds_per_segment:
            record("review_unavailable", sid, reason="provider_or_budget")
            continue
        review_rounds[sid] = review_rounds.get(sid, 0) + 1
        review_calls += 1
        coordinator_steps += 1
        request = AdaptiveReviewRequest(
            segment_id=sid, source_text=source.text, draft_text=draft.translated_text,
            context_text=context, issue_ids=screen.issue_ids if screen is not None else (),
            mode=policy.mode, review_round=review_rounds[sid], rejection=rejection.pop(sid, None),
            instruction_version=policy.instruction_version,
            deterministic_findings=deterministic,
        )
        if deterministic:
            record("deterministic_findings", sid, findings=list(deterministic))
        try:
            returned = review_provider.review(request)
        except Exception as exc:
            if getattr(exc, "fatal_provider_error", False):
                raise
            returned = AdaptiveReviewResult(status="failed", summary=f"Review provider raised {type(exc).__name__}")
        if returned.status != "completed":
            record("review_failed", sid, summary=returned.summary)
            continue
        review = SegmentEvidence(
            level="generative_review", context=_receipt(index, sources, drafts, memory_hash, policy, "generative_review"),
            context_sha256=sha256_text(context),
            status="ok", issue_ids=tuple(f.issue_id for f in returned.findings),
            blocking_issue_ids=tuple(f.issue_id for f in returned.findings
                                     if f.blocking and f.severity != "polish"),
        )
        reviews[sid] = review
        record("segment_reviewed", sid, issues=list(review.issue_ids), patches=len(returned.patches))
        if not returned.patches:
            continue
        if patch_actions >= budget.max_patch_actions or patch_rounds.get(sid, 0) >= budget.max_patch_rounds_per_segment:
            record("patch_budget_exhausted", sid)
            budget_exhausted_segments.add(sid)
            continue
        action = returned.patches[0]
        if action.segment_id != sid:
            record("patch_rejected", sid, reason="wrong_segment", message="Review patch targeted another segment")
            reviews.pop(sid, None)
            continue
        patch_actions += 1
        patch_rounds[sid] = patch_rounds.get(sid, 0) + 1

        def candidate_guard(candidate_text: str) -> tuple[bool, str]:
            violation = _polish_guard_violation(draft.translated_text, candidate_text, source.text)
            if violation is None and policy.gloss_guard:
                violation = _gloss_guard_violation(draft.translated_text, candidate_text, source.text)
            if violation is None and policy.duplication_guard:
                violation = _duplication_violation(
                    draft.translated_text, candidate_text,
                    [item.translated_text for item in drafts if item.segment_id != sid])
            if violation is not None:
                record("candidate_guard_rejected", sid, reason=violation)
                return False, violation
            signals = _content_signals(draft.translated_text, candidate_text)
            if signals:
                record("candidate_content_signals", sid, signals=signals)
            if policy.mode != "adaptive_segment_screen" or decision_provider is None:
                return True, ""
            # Patch-time checks may use a higher material bar than routing: a low
            # routing threshold widens review, but must not veto good patches on noise.
            guard_thresholds = dict(policy.routing_thresholds)
            if policy.candidate_rescreen_threshold is not None:
                guard_thresholds["material"] = max(guard_thresholds["material"],
                                                   policy.candidate_rescreen_threshold)
            try:
                candidate_request = DecisionRequest(
                    segment_id=sid, source_text=source.text, draft_text=draft.translated_text,
                    candidate_text=candidate_text, context_text=context,
                    requested_model=policy.decision_model,
                    routing_policy_version=policy.routing_policy_version,
                    routing_thresholds=guard_thresholds,
                )
                candidate_decision = decision_provider.evaluate(candidate_request)
                if not _valid_decision(candidate_decision, candidate_request):
                    record("candidate_check_stale", sid)
                    return True, "Generative review supplies the source-grounded fallback."
            except Exception as exc:
                if getattr(exc, "fatal_provider_error", False):
                    raise
                record("candidate_check_failed", sid, error_type=type(exc).__name__)
                return True, "Generative review supplies the source-grounded fallback."
            record("candidate_checked", sid, status=candidate_decision.status,
                   request_id=candidate_decision.receipt.request_id)
            if candidate_decision.status != "ok":
                return True, "Generative review supplies the source-grounded fallback."
            if candidate_decision.receipt.derived_flags.get("patch_unsupported_claim"):
                return False, "Candidate introduces a source-unsupported claim."
            if not policy.candidate_rescreen:
                return True, ""
            # The patch question compares candidate with draft; the error questions
            # above still describe the old draft. Screen the candidate itself and
            # reject material risks the original screen did not already carry.
            try:
                rescreen_request = DecisionRequest(
                    segment_id=sid, source_text=source.text, draft_text=candidate_text,
                    context_text=context, requested_model=policy.decision_model,
                    routing_policy_version=policy.routing_policy_version,
                    routing_thresholds=guard_thresholds,
                )
                rescreen = decision_provider.evaluate(rescreen_request)
                if not _valid_decision(rescreen, rescreen_request) or rescreen.status != "ok":
                    record("candidate_rescreen_unavailable", sid)
                    return True, ""
            except Exception as exc:
                if getattr(exc, "fatal_provider_error", False):
                    raise
                record("candidate_rescreen_failed", sid, error_type=type(exc).__name__)
                return True, ""
            original = set(screen.issue_ids) if screen is not None else set()
            introduced = [flag for flag in rescreen.material_flags
                          if flag != "patch_unsupported_claim" and flag not in original]
            record("candidate_rescreened", sid, introduced=introduced,
                   request_id=rescreen.receipt.request_id)
            if introduced:
                return False, "Candidate introduces new material risk: " + ", ".join(introduced) + "."
            return True, ""

        execution = executor.submit_segment_patch(
            action, [(item.segment_id, item.translated_text) for item in drafts],
            candidate_guard=candidate_guard,
        )
        if not execution.observation.ok:
            rejection[sid] = execution.observation.message
            record("patch_rejected", sid, reason=execution.observation.data.get("reason", "qa_gate"),
                   message=execution.observation.message)
            reviews.pop(sid, None)
            if review_rounds[sid] < budget.max_generative_review_rounds_per_segment and index not in queued:
                queue.append(index)
                queued.add(index)
            continue
        drafts[index] = drafts[index].with_text(str(execution.observation.data["candidate_segment_text"]))
        record("patch_accepted", sid, issue_ids=list(action.issue_ids),
               draft_sha256=drafts[index].draft_sha256)
        for affected in (index - 1, index, index + 1):
            if 0 <= affected < len(sources):
                affected_id = sources[affected].segment_id
                screens.pop(affected_id, None)
                reviews.pop(affected_id, None)
                if affected not in queued:
                    queue.append(affected)
                    queued.add(affected)

    final_text = render_draft(drafts)
    if final_text != executor.current_text:
        raise RuntimeError("Adaptive segment reconstruction diverged from repair executor")
    final_evidence: list[SegmentEvidence] = []
    for index, source in enumerate(sources):
        sid = source.segment_id
        screen = screens.get(sid)
        review = reviews.get(sid)
        current_screen = _current(screen, index, sources, drafts, memory_hash, policy, context_map)
        current_review = _current(review, index, sources, drafts, memory_hash, policy, context_map)
        if current_screen and screen is not None:
            final_evidence.append(screen)
        if current_review and review is not None:
            final_evidence.append(review)
        if policy.mode == "adaptive_segment_screen" and not current_screen:
            warnings.append(AdaptiveWarning("missing_screen_coverage", sid, "No current Jev screening receipt."))
        if policy.mode == "full_segment_review" and not current_review:
            warnings.append(AdaptiveWarning("missing_review_coverage", sid, "No current generative review receipt."))
        if policy.mode == "adaptive_segment_screen" and current_screen and screen is not None:
            if screen.status != "ok":
                warnings.append(AdaptiveWarning("router_fallback", sid, "Jev screening was unavailable; generative fallback was used or attempted."))
            if screen.issue_ids and not current_review:
                warnings.append(AdaptiveWarning("unresolved_screen_flags", sid, ", ".join(screen.issue_ids)))
            if "source_incomplete" in screen.issue_ids:
                warnings.append(AdaptiveWarning("source_incomplete", sid, "The available source is physically incomplete."))
        if current_review and review is not None and review.blocking_issue_ids:
            warnings.append(AdaptiveWarning("unresolved_review_findings", sid, ", ".join(review.blocking_issue_ids)))
        if sid in budget_exhausted_segments:
            warnings.append(AdaptiveWarning("patch_budget_exhausted", sid, "Proposed repair was not attempted because the patch budget was exhausted."))
    if policy.qa_warning_policy_version == "qa-warnings.v2":
        blocking_qa = [f for f in executor.current_qa.findings if f.check_id not in INFORMATIONAL_QA_CHECKS]
        informational_qa = [f for f in executor.current_qa.findings if f.check_id in INFORMATIONAL_QA_CHECKS]
    else:
        blocking_qa = list(executor.current_qa.findings)
        informational_qa = []
    if blocking_qa:
        warnings.append(AdaptiveWarning("deterministic_qa_findings", None,
                                        f"{len(blocking_qa)} deterministic QA finding(s) remain."))
    if informational_qa:
        events.append({
            "kind": "qa_findings_informational",
            "segment_id": None,
            "count": len(informational_qa),
            "check_ids": sorted({finding.check_id for finding in informational_qa}),
        })
    outcome: DeliveryOutcome = "delivered_with_warnings" if warnings else "delivered"
    result = AdaptiveChapterResult(
        outcome=outcome, final_text=final_text, draft_segments=tuple(drafts),
        initial_draft_sha256=sha256_text(initial_text), source_sha256=sha256_text(source_text),
        identity_sha256=identity, verification_policy=policy, interaction_policy=interaction,
        budget_policy=budget, memory_snapshot_sha256=memory_hash,
        evidence=tuple(final_evidence), warnings=tuple(warnings), events=tuple(events),
        patch_actions=patch_actions, review_calls=review_calls, screen_calls=screen_calls,
    )
    _write_checkpoint(store, identity, drafts, events, policy, interaction, budget, memory_hash, result)
    return result
