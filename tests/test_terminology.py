from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic_translation.models import (
    AgentConfig,
    GlossaryEntry,
    ProviderCallRecord,
    TerminologyConsensusConfig,
)
from agentic_translation.terminology import (
    TerminologyResolutionError,
    TerminologyResolver,
    blind_terminology_candidates,
    normalize_term,
)
from agentic_translation.terminology_models import (
    TerminologyCandidate,
    TerminologyEvaluation,
    TerminologyModelConfig,
    TerminologyRequest,
    TerminologyResolution,
    TerminologyVote,
)


def _request(*, blocked_variants: list[str] | None = None) -> TerminologyRequest:
    return TerminologyRequest(
        story_slug="demo",
        chapter="0001",
        source_term="道心",
        source_context="道心守住了山门。",
        translation_context="The term remains unresolved.",
        glossary_entries=[],
        blocked_variants=blocked_variants or [],
    )


class FakeVoter:
    def __init__(
        self,
        voter_id: str,
        provider_name: str,
        model_name: str,
        recommendation: str,
        confidence: float,
        *,
        call_records: list[ProviderCallRecord] | None = None,
    ) -> None:
        self.voter_id = voter_id
        self.provider_name = provider_name
        self.model_name = model_name
        self.recommendation = recommendation
        self.confidence = confidence
        self.call_records = list(call_records or [])

    def vote(self, request: TerminologyRequest) -> TerminologyVote:
        return TerminologyVote(
            voter_id=self.voter_id,
            provider=self.provider_name,
            model=self.model_name,
            source_term=request.source_term,
            recommendation=self.recommendation,
            confidence=self.confidence,
            provider_call=self.call_records[-1] if self.call_records else None,
        )


class FakeEvaluator:
    def __init__(
        self,
        choice: str,
        confidence: float,
        *,
        provider_name: str = "openai",
        model_name: str = "oa-evaluator",
        call_records: list[ProviderCallRecord] | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self.choice = choice
        self.confidence = confidence
        self.calls = 0
        self.call_records = list(call_records or [])
        self.seen_candidates: list[TerminologyCandidate] = []

    def evaluate(
        self,
        request: TerminologyRequest,
        candidates: list[TerminologyCandidate],
    ) -> TerminologyEvaluation:
        self.calls += 1
        self.seen_candidates = list(candidates)
        return TerminologyEvaluation(
            provider=self.provider_name,
            model=self.model_name,
            selected_candidate_id=self.choice,
            confidence=self.confidence,
            rationale="Selected the candidate that best fits the supplied context.",
            provider_call=self.call_records[-1] if self.call_records else None,
        )


class FailingEvaluator(FakeEvaluator):
    def evaluate(self, request, candidates):
        raise AssertionError("evaluator must not run")


def _resolver(
    recommendation_a: str = "Dao Heart",
    recommendation_b: str = "Dao Heart",
    confidence_a: float = 0.91,
    confidence_b: float = 0.88,
    evaluator: FakeEvaluator | None = None,
    *,
    voters: list[FakeVoter] | None = None,
) -> tuple[TerminologyResolver, FakeEvaluator]:
    selected_evaluator = evaluator or FakeEvaluator("candidate_b", 0.82)
    selected_voters = voters or [
        FakeVoter("openai", "openai", "oa", recommendation_a, confidence_a),
        FakeVoter("deepseek", "deepseek", "ds", recommendation_b, confidence_b),
    ]
    return (
        TerminologyResolver(
            voters=selected_voters,
            evaluator=selected_evaluator,
            confidence_threshold=0.65,
        ),
        selected_evaluator,
    )


def test_normalize_term_groups_superficial_variants_without_changing_display() -> None:
    assert normalize_term('  “Dao   Heart” ') == "dao heart"
    vote = TerminologyVote(
        voter_id="openai",
        provider="openai",
        model="openai-model",
        source_term="道心",
        recommendation="Dao Heart",
        confidence=0.91,
    )
    assert vote.recommendation == "Dao Heart"
    assert vote.normalized_key == "dao heart"
    assert normalize_term("Sword-Heart") != normalize_term("Sword Heart")


def test_consensus_config_requires_openai_and_deepseek_unique_identities() -> None:
    config = TerminologyConsensusConfig(
        enabled=True,
        openai_model="model-a",
        deepseek_model="deepseek-chat",
        evaluator_provider="openai",
    )
    assert [(v.provider, v.model) for v in config.voters] == [
        ("openai", "model-a"),
        ("deepseek", "deepseek-chat"),
    ]

    with pytest.raises(ValidationError):
        TerminologyConsensusConfig(enabled=True, deepseek_model="deepseek-chat")
    with pytest.raises(ValidationError):
        TerminologyConsensusConfig(
            enabled=True,
            openai_model=" ",
            deepseek_model="deepseek-chat",
        )
    with pytest.raises(ValidationError):
        TerminologyConsensusConfig(
            enabled=True,
            openai_model="oa",
            deepseek_model="ds",
            confidence_threshold=1.1,
        )
    with pytest.raises(ValidationError):
        TerminologyModelConfig(voter_id="openai", provider="deepseek", model="model")


def test_agent_config_backward_parsing_defaults_consensus_disabled() -> None:
    config = AgentConfig.model_validate({"max_attempts": 2})
    assert config.terminology_consensus.enabled is False
    assert config.terminology_consensus.voters == []


def test_high_confidence_agreement_bypasses_evaluator() -> None:
    evaluator = FailingEvaluator("candidate_a", 1.0)
    resolver, _ = _resolver(
        recommendation_a="Dao Heart",
        recommendation_b="dao  heart",
        evaluator=evaluator,
    )
    resolution = resolver.resolve(_request())
    assert resolution.selected_translation == "Dao Heart"
    assert resolution.evaluator_used is False
    assert resolution.escalated is False
    assert resolution.agreement is True


def test_reused_resolver_records_only_calls_from_current_resolution() -> None:
    class DeltaVoter(FakeVoter):
        def __init__(self, voter_id: str, model_name: str) -> None:
            super().__init__(voter_id, voter_id, model_name, "Dao Heart", 0.9)
            self.invocations = 0

        def vote(self, request: TerminologyRequest) -> TerminologyVote:
            self.invocations += 1
            self.call_records.append(
                ProviderCallRecord(
                    role="terminology_vote",
                    namespace="terminology_vote",
                    provider=self.provider_name,
                    model=self.model_name,
                    payload_sha256=f"{self.voter_id}-payload-{self.invocations}",
                    response_sha256=f"{self.voter_id}-response-{self.invocations}",
                    cache_file=f"{self.voter_id}-{self.invocations}.json",
                    cache_hit=True,
                )
            )
            return super().vote(request)

    voters = [DeltaVoter("openai", "oa"), DeltaVoter("deepseek", "ds")]
    resolver = TerminologyResolver(
        voters=voters,
        evaluator=FailingEvaluator("candidate_a", 1.0),
        confidence_threshold=0.65,
    )
    first = resolver.resolve(_request())
    second = resolver.resolve(_request())
    assert len(first.provider_calls) == 2
    assert len(second.provider_calls) == 2
    assert [call.cache_file for call in second.provider_calls] == [
        "openai-2.json",
        "deepseek-2.json",
    ]


def test_disagreement_calls_evaluator_once_and_blinds_candidates() -> None:
    evaluator = FakeEvaluator(choice="candidate_b", confidence=0.82)
    resolver, _ = _resolver(
        recommendation_a="Dao Heart",
        recommendation_b="Heart of Dao",
        evaluator=evaluator,
    )
    resolution = resolver.resolve(_request())
    assert evaluator.calls == 1
    assert resolution.evaluator_used is True
    assert resolution.selected_translation == "Heart of Dao"
    assert [c.candidate_id for c in evaluator.seen_candidates] == ["candidate_a", "candidate_b"]
    assert all(not hasattr(c, "voter_id") for c in evaluator.seen_candidates)


def test_reversed_voter_construction_has_identical_canonical_resolution() -> None:
    evaluator_a = FakeEvaluator(choice="candidate_b", confidence=0.82)
    evaluator_b = FakeEvaluator(choice="candidate_b", confidence=0.82)
    voters = [
        FakeVoter("openai", "openai", "oa", "Dao Heart", 0.9),
        FakeVoter("deepseek", "deepseek", "ds", "Heart of Dao", 0.8),
    ]
    first = TerminologyResolver(
        voters=voters,
        evaluator=evaluator_a,
        confidence_threshold=0.65,
    ).resolve(_request())
    second = TerminologyResolver(
        voters=list(reversed(voters)),
        evaluator=evaluator_b,
        confidence_threshold=0.65,
    ).resolve(_request())
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert [vote.voter_id for vote in first.votes] == ["openai", "deepseek"]


def test_blind_candidate_order_is_deterministic_and_provider_order_independent() -> None:
    first = [
        TerminologyVote(
            voter_id="deepseek",
            provider="deepseek",
            model="ds",
            source_term="道心",
            recommendation="Zeta Heart",
            confidence=0.8,
        ),
        TerminologyVote(
            voter_id="openai",
            provider="openai",
            model="oa",
            source_term="道心",
            recommendation="Alpha Heart",
            confidence=0.8,
        ),
    ]
    second = list(reversed(first))
    assert [c.model_dump() for c in blind_terminology_candidates(first)] == [
        c.model_dump() for c in blind_terminology_candidates(second)
    ]


def test_low_confidence_agreement_escalates_without_override() -> None:
    resolver, evaluator = _resolver(
        confidence_a=0.4,
        confidence_b=0.5,
        evaluator=FailingEvaluator("low-confidence agreement must not evaluate", 1.0),
    )
    resolution = resolver.resolve(_request())
    assert evaluator.calls == 0
    assert resolution.escalated is True
    assert resolution.selected_translation is None
    assert "confidence" in (resolution.escalation_reason or "")


def test_blocked_variant_escalates_even_when_models_agree() -> None:
    resolver, _ = _resolver("Dao Heart", "Dao Heart")
    resolution = resolver.resolve(_request(blocked_variants=["dao heart"]))
    assert resolution.escalated is True
    assert resolution.selected_translation is None
    assert "blocked" in (resolution.escalation_reason or "")


def test_blocked_variant_selected_by_evaluator_escalates() -> None:
    evaluator = FakeEvaluator(choice="candidate_a", confidence=0.9)
    resolver, _ = _resolver("Dao Heart", "Heart of Dao", evaluator=evaluator)
    resolution = resolver.resolve(_request(blocked_variants=["DAO HEART"]))
    assert resolution.evaluator_used is True
    assert resolution.escalated is True
    assert resolution.selected_translation is None


def test_low_confidence_evaluation_escalates() -> None:
    evaluator = FakeEvaluator(choice="candidate_a", confidence=0.4)
    resolver, _ = _resolver("Dao Heart", "Heart of Dao", evaluator=evaluator)
    resolution = resolver.resolve(_request())
    assert resolution.escalated is True
    assert resolution.selected_translation is None
    assert resolution.evaluation is not None


def test_invalid_evaluator_selection_raises_typed_error() -> None:
    evaluator = FakeEvaluator(choice="not-a-candidate", confidence=1.0)
    resolver, _ = _resolver("Dao Heart", "Heart of Dao", evaluator=evaluator)
    with pytest.raises(TerminologyResolutionError, match="unknown candidate"):
        resolver.resolve(_request())


@pytest.mark.parametrize(
    "vote_kwargs",
    [
        {"model": "other-model"},
        {"source_term": "别的词"},
    ],
)
def test_vote_mismatch_to_configured_voter_fails_closed(vote_kwargs) -> None:
    class MismatchedVoter(FakeVoter):
        def vote(self, request: TerminologyRequest) -> TerminologyVote:
            values = {
                "voter_id": self.voter_id,
                "provider": self.provider_name,
                "model": self.model_name,
                "source_term": request.source_term,
                "recommendation": self.recommendation,
                "confidence": self.confidence,
                **vote_kwargs,
            }
            return TerminologyVote.model_validate(values)

    voters = [
        MismatchedVoter("openai", "openai", "oa", "Dao Heart", 0.9),
        FakeVoter("deepseek", "deepseek", "ds", "Dao Heart", 0.9),
    ]
    resolver, _ = _resolver(voters=voters)
    with pytest.raises(TerminologyResolutionError, match="mismatched"):
        resolver.resolve(_request())


def test_evaluator_mismatch_to_configured_model_fails_closed() -> None:
    class MismatchedEvaluator(FakeEvaluator):
        def evaluate(self, request, candidates):
            self.calls += 1
            return TerminologyEvaluation(
                provider="openai",
                model="different-model",
                selected_candidate_id="candidate_a",
                confidence=1.0,
            )

    resolver, _ = _resolver(
        recommendation_a="Dao Heart",
        recommendation_b="Heart of Dao",
        evaluator=MismatchedEvaluator("candidate_a", 1.0),
    )
    with pytest.raises(TerminologyResolutionError, match="evaluator returned mismatched"):
        resolver.resolve(_request())


def test_resolver_requires_exactly_openai_and_deepseek_voters() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        TerminologyResolver(
            voters=[FakeVoter("openai", "openai", "oa", "A", 0.9)],
            evaluator=FakeEvaluator("candidate_a", 1.0),
            confidence_threshold=0.65,
        )
    with pytest.raises(ValueError, match="identities"):
        TerminologyResolver(
            voters=[
                FakeVoter("openai", "openai", "oa", "A", 0.9),
                FakeVoter("openai", "openai", "oa2", "B", 0.9),
            ],
            evaluator=FakeEvaluator("candidate_a", 1.0),
            confidence_threshold=0.65,
        )


def test_partial_voter_failure_preserves_prior_call_evidence() -> None:
    first_call = ProviderCallRecord(
        role="terminology_voter",
        namespace="terminology_vote",
        provider="openai",
        model="oa",
        payload_sha256="a" * 64,
        response_sha256="b" * 64,
        cache_file="oa.json",
        cache_hit=False,
    )

    class BrokenVoter(FakeVoter):
        def vote(self, request: TerminologyRequest) -> TerminologyVote:
            raise RuntimeError("deepseek unavailable")

    resolver, _ = _resolver(
        voters=[
            FakeVoter("openai", "openai", "oa", "A", 0.9, call_records=[first_call]),
            BrokenVoter("deepseek", "deepseek", "ds", "B", 0.9),
        ]
    )
    with pytest.raises(TerminologyResolutionError) as raised:
        resolver.resolve(_request())
    assert raised.value.call_records == [first_call]


def test_terminology_models_reject_extra_fields_and_bound_context() -> None:
    with pytest.raises(ValidationError):
        TerminologyRequest(
            story_slug="demo",
            chapter="0001",
            source_term="道心",
            source_context="context",
            translation_context="translation",
            glossary_entries=[],
            unexpected=True,
        )
    with pytest.raises(ValidationError):
        TerminologyRequest(
            story_slug="demo",
            chapter="0001",
            source_term="道心",
            source_context="x" * 5000,
            translation_context="translation",
            glossary_entries=[],
        )


def test_terminology_response_members_are_individually_bounded() -> None:
    with pytest.raises(ValidationError):
        TerminologyVote(
            voter_id="openai",
            provider="openai",
            model="oa",
            source_term="道心",
            recommendation="Dao Heart",
            confidence=0.9,
            alternatives=["x" * 301],
        )
    with pytest.raises(ValidationError):
        TerminologyRequest(
            story_slug="demo",
            chapter="0001",
            source_term="道心",
            source_context="context",
            translation_context="translation",
            glossary_entries=[],
            blocked_variants=["x" * 301],
        )
    with pytest.raises(ValidationError):
        TerminologyEvaluation(
            provider="openai",
            model="oa",
            selected_candidate_id="x" * 101,
            confidence=1.0,
        )
