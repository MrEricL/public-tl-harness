"""The two front doors share policy identity; historical review stays explicit."""
from dataclasses import asdict

from agentic_translation.adaptive import VerificationPolicy
from agentic_translation.demo import demo_policy
from agentic_translation.profiles import (
    CONTEXTUAL_PROFILE, LEGACY_ADAPTIVE_PROFILE, LEGACY_REVIEW_PROFILE,
    PORTFOLIO_REVIEW_PROFILE, book_verification_policy, resolve_book_profile,
)


def test_named_book_modes_share_demo_policy_and_make_absent_review_explicit():
    assert resolve_book_profile("contextual") == CONTEXTUAL_PROFILE
    assert book_verification_policy(CONTEXTUAL_PROFILE) is None
    assert resolve_book_profile("portfolio-review") == PORTFOLIO_REVIEW_PROFILE
    assert book_verification_policy(PORTFOLIO_REVIEW_PROFILE) == demo_policy()


def test_explicit_legacy_modes_keep_v2_behavior_under_truthful_names():
    for strategy, name, mode in (("adaptive", LEGACY_ADAPTIVE_PROFILE, "adaptive_segment_screen"),
                                 ("always-review", LEGACY_REVIEW_PROFILE, "full_segment_review")):
        assert resolve_book_profile(strategy) == name
        expected = VerificationPolicy(version=name, mode=mode, instruction_version="natural-review-v2",
                                      decision_model="test-router")
        actual = book_verification_policy(name, decision_model="test-router")
        assert asdict(actual) == asdict(expected)
        assert not actual.gloss_guard and not actual.duplication_guard
