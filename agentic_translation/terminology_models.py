"""Strict, bounded data contracts for terminology arbitration.

The terminology models intentionally live apart from the existing translation
candidate models.  They describe a small, auditable two-voter decision and do
not carry prompts, secrets, or unbounded provider responses.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import (
    GlossaryEntry,
    ProviderCallRecord,
    TerminologyConsensusConfig,
    TerminologyModelConfig,
)


_EDGE_QUOTE_RE = re.compile(r"^[\s\"'“”‘’]+|[\s\"'“”‘’]+$")
_BoundedTerm = Annotated[str, Field(min_length=1, max_length=300)]
_BoundedIdentifier = Annotated[str, Field(min_length=1, max_length=100)]


class TerminologyVoterResponse(BaseModel):
    """The bounded JSON object accepted from one terminology voter."""

    model_config = ConfigDict(extra="forbid", strict=True)

    recommendation: str = Field(min_length=1, max_length=300)
    confidence: float = Field(ge=0.0, le=1.0)
    alternatives: list[_BoundedTerm] = Field(max_length=5)
    rationale: str = Field(max_length=600)

    @field_validator("confidence", mode="before")
    @classmethod
    def _require_json_float(cls, value: object) -> object:
        # Pydantic's strict float intentionally accepts integers as a numeric
        # widening conversion.  Provider responses use a JSON number contract
        # and must not silently coerce an integer or boolean confidence.
        if type(value) is not float:
            raise ValueError("confidence must be a JSON float")
        return value


class TerminologyEvaluationResponse(BaseModel):
    """The bounded JSON object accepted from a terminology evaluator."""

    model_config = ConfigDict(extra="forbid", strict=True)

    selected_candidate_id: Literal["candidate_a", "candidate_b"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(max_length=600)

    @field_validator("confidence", mode="before")
    @classmethod
    def _require_json_float(cls, value: object) -> object:
        if type(value) is not float:
            raise ValueError("confidence must be a JSON float")
        return value


def _normalize_term_key(value: str) -> str:
    """Return the deterministic comparison key for a proposed term."""

    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = _EDGE_QUOTE_RE.sub("", normalized).strip()
    normalized = " ".join(normalized.split())
    return normalized.casefold()


class TerminologyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    story_slug: str = Field(min_length=1, max_length=200)
    chapter: str = Field(min_length=1, max_length=200)
    source_term: str = Field(min_length=1, max_length=200)
    source_context: str = Field(default="", max_length=4000)
    translation_context: str = Field(default="", max_length=4000)
    glossary_entries: list[GlossaryEntry] = Field(default_factory=list, max_length=100)
    blocked_variants: list[_BoundedTerm] = Field(default_factory=list, max_length=100)
    finding_id: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _strip_identifiers(self) -> "TerminologyRequest":
        for field_name in ("story_slug", "chapter", "source_term"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} cannot be blank")
        for value in self.blocked_variants:
            if not value.strip():
                raise ValueError("blocked_variants cannot contain blank values")
        for entry in self.glossary_entries:
            for field_name in ("source", "target"):
                value = getattr(entry, field_name)
                if len(value) > 300:
                    raise ValueError(f"glossary entry {field_name} is too long")
            for field_name in ("candidates", "blocked_variants"):
                values = getattr(entry, field_name)
                if len(values) > 5:
                    raise ValueError(f"glossary entry {field_name} has too many values")
                if any(len(value) > 300 for value in values):
                    raise ValueError(f"glossary entry {field_name} contains an overlong value")
        if self.finding_id is not None and not self.finding_id.strip():
            raise ValueError("finding_id cannot be blank")
        return self


class TerminologyVote(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    voter_id: Literal["openai", "deepseek"]
    provider: Literal["openai", "deepseek"]
    model: str = Field(min_length=1, max_length=200)
    source_term: str = Field(min_length=1, max_length=200)
    recommendation: str = Field(min_length=1, max_length=300)
    confidence: float = Field(ge=0.0, le=1.0)
    alternatives: list[_BoundedTerm] = Field(default_factory=list, max_length=5)
    rationale: str = Field(default="", max_length=600)
    normalized_key: str = Field(default="", max_length=300)
    provider_call: ProviderCallRecord | None = None

    @model_validator(mode="after")
    def _validate_identity_and_key(self) -> "TerminologyVote":
        if self.voter_id != self.provider:
            raise ValueError("voter_id and provider must identify the same model family")
        if not self.model.strip() or not self.source_term.strip():
            raise ValueError("model and source_term cannot be blank")
        computed = _normalize_term_key(self.recommendation)
        if not computed:
            raise ValueError("recommendation cannot normalize to blank")
        if self.normalized_key and self.normalized_key != computed:
            raise ValueError("normalized_key must match recommendation")
        self.normalized_key = computed
        return self


class TerminologyCandidate(BaseModel):
    """A voter proposal with identity removed before evaluator arbitration."""

    model_config = ConfigDict(extra="forbid", strict=True)

    candidate_id: Literal["candidate_a", "candidate_b"]
    recommendation: str = Field(min_length=1, max_length=300)
    normalized_key: str = Field(default="", max_length=300)

    @model_validator(mode="after")
    def _validate_key(self) -> "TerminologyCandidate":
        computed = _normalize_term_key(self.recommendation)
        if not computed:
            raise ValueError("recommendation cannot normalize to blank")
        if self.normalized_key and self.normalized_key != computed:
            raise ValueError("normalized_key must match recommendation")
        self.normalized_key = computed
        return self


class TerminologyEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    provider: Literal["openai", "deepseek"]
    model: str = Field(min_length=1, max_length=200)
    selected_candidate_id: _BoundedIdentifier
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=600)
    provider_call: ProviderCallRecord | None = None

    @model_validator(mode="after")
    def _validate_model_name(self) -> "TerminologyEvaluation":
        if not self.model.strip():
            raise ValueError("model cannot be blank")
        return self


class TerminologyResolution(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    votes: list[TerminologyVote] = Field(min_length=2, max_length=2)
    agreement: bool
    candidates: list[TerminologyCandidate] = Field(default_factory=list, max_length=2)
    evaluation: TerminologyEvaluation | None = None
    selected_candidate_id: _BoundedIdentifier | None = None
    selected_translation: str | None = Field(default=None, max_length=300)
    evaluator_used: bool = False
    escalated: bool = False
    escalation_reason: str | None = Field(default=None, max_length=500)
    provider_calls: list[ProviderCallRecord] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def _validate_outcome(self) -> "TerminologyResolution":
        if self.escalated and self.selected_translation is not None:
            raise ValueError("escalated resolution cannot select a translation")
        if self.selected_translation is not None and not self.selected_translation.strip():
            raise ValueError("selected_translation cannot be blank")
        if self.evaluator_used != (self.evaluation is not None):
            raise ValueError("evaluator_used must match presence of evaluation")
        return self


__all__ = [
    "TerminologyCandidate",
    "TerminologyConsensusConfig",
    "TerminologyEvaluation",
    "TerminologyEvaluationResponse",
    "TerminologyModelConfig",
    "TerminologyRequest",
    "TerminologyResolution",
    "TerminologyVote",
    "TerminologyVoterResponse",
]
