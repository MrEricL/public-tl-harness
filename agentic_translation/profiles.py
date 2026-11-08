"""Named book policies; experiment configurations retain their own frozen settings."""
from __future__ import annotations

from .adaptive import BudgetPolicy, VerificationPolicy


CONTEXTUAL_PROFILE = "contextual.v1"
PORTFOLIO_REVIEW_PROFILE = "portfolio-review.v1"
LEGACY_ADAPTIVE_PROFILE = "legacy-adaptive.v1"
LEGACY_REVIEW_PROFILE = "legacy-always-review.v1"

BOOK_PROFILES = {
    CONTEXTUAL_PROFILE: "Source-only memory and glossary; deterministic QA, no model review or repair.",
    PORTFOLIO_REVIEW_PROFILE: "Experimental full review and term alignment with bounded edits and deterministic guards.",
    LEGACY_ADAPTIVE_PROFILE: "Legacy v2 Jev routing and review; optional gloss and duplication guards are off.",
    LEGACY_REVIEW_PROFILE: "Legacy v2 full review; optional gloss and duplication guards are off.",
}
BOOK_STRATEGIES = {
    **{name: name for name in BOOK_PROFILES},
    "contextual": CONTEXTUAL_PROFILE,
    "portfolio-review": PORTFOLIO_REVIEW_PROFILE,
    "adaptive": LEGACY_ADAPTIVE_PROFILE,
    "always-review": LEGACY_REVIEW_PROFILE,
}


def resolve_book_profile(strategy: str) -> str:
    try:
        return BOOK_STRATEGIES[strategy]
    except KeyError as exc:
        raise ValueError("Unknown book strategy; use contextual, portfolio-review, adaptive, or always-review") from exc


def portfolio_review_policy() -> VerificationPolicy:
    """The folder demo and explicit full-review book path use these exact guards."""
    return VerificationPolicy(
        version=PORTFOLIO_REVIEW_PROFILE, mode="full_segment_review",
        instruction_version="natural-review-v6", qa_warning_policy_version="qa-warnings.v2",
        route_deterministic_findings=True, term_alignment_pass=True,
        gloss_guard=True, duplication_guard=True,
    )


def book_verification_policy(profile: str, *, decision_model: str = "jev-1.13.0") -> VerificationPolicy | None:
    if profile == CONTEXTUAL_PROFILE:
        return None
    if profile == PORTFOLIO_REVIEW_PROFILE:
        return portfolio_review_policy()
    if profile not in {LEGACY_ADAPTIVE_PROFILE, LEGACY_REVIEW_PROFILE}:
        raise ValueError("Unknown book profile")
    return VerificationPolicy(
        version=profile,
        mode="adaptive_segment_screen" if profile == LEGACY_ADAPTIVE_PROFILE else "full_segment_review",
        instruction_version="natural-review-v2", decision_model=decision_model,
    )


def portfolio_review_budget(segment_count: int) -> BudgetPolicy:
    """Allow the same bounded work per source segment in both entry points."""
    return BudgetPolicy(max_coordinator_steps=max(24, segment_count * 8),
                        max_patch_actions=max(12, segment_count * 2))
