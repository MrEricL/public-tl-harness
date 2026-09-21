"""Typed contracts for bounded semantic sensing with an evaluation model.

These models intentionally keep model judgments separate from trusted agent
instructions and from the existing fidelity-review gate.  A semantic signal is
an observation about a code-selected pair of text spans, never an edit or a
free-form model finding.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


QuestionSetName = Literal["focused", "dense"]
SemanticCategory = Literal[
    "omission",
    "unsupported_addition",
    "actor_action_roles",
    "negation_polarity",
    "source_ambiguity",
    "speaker_attribution",
    "identity_coreference",
    "conditions_exceptions",
    "deontic_modality",
    "sequence",
    "causality",
    "purpose",
    "quantity_unit_degree",
    "certainty_evidentiality",
    "tense_aspect",
    "idiom_figurative",
    "term_sense",
    "register_style",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class JevPolicy(StrictModel):
    """Serializable policy and resource bounds for Jev sensing."""

    mode: Literal["off", "shadow", "advisory"] = "off"
    question_set: QuestionSetName = "focused"
    schedule: Literal["initial", "after_edit"] = "initial"
    model: Literal["typesafe-ai/jev"] = "typesafe-ai/jev"
    endpoint: Literal["https://ai-gateway.vercel.sh/v1/evaluate"] = (
        "https://ai-gateway.vercel.sh/v1/evaluate"
    )
    max_concurrency: int = Field(default=8, ge=1, le=8)
    window_chars: int = Field(default=4000, ge=256, le=12_000)
    window_overlap: int = Field(default=400, ge=0, le=4000)
    max_windows: int = Field(default=64, ge=1, le=256)
    supporting_text_chars: int = Field(default=2000, ge=0, le=8000)
    max_request_bytes: int = Field(default=28_000, ge=4096, le=100_000)
    advisory_threshold: float = Field(default=0.5, ge=0, le=1)
    timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    max_retries: int = Field(default=2, ge=0, le=5)
    retry_after_cap_seconds: float = Field(default=30.0, ge=0, le=120)

    @model_validator(mode="after")
    def _overlap_must_advance(self) -> "JevPolicy":
        if self.window_overlap >= self.window_chars:
            raise ValueError("window_overlap must be smaller than window_chars")
        return self


class SemanticSnapshot(StrictModel):
    """Immutable semantic input; hashes use Unicode text exactly as supplied."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    source_text: str = Field(min_length=1)
    draft_text: str = Field(min_length=1)
    glossary_text: str = ""
    style_guide: str = ""
    context: str = ""

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()

    @property
    def draft_sha256(self) -> str:
        return hashlib.sha256(self.draft_text.encode("utf-8")).hexdigest()


class TextSpan(StrictModel):
    document: Literal["source", "draft"]
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def _valid_codepoint_span(self) -> "TextSpan":
        if self.end < self.start:
            raise ValueError("span end must not precede start")
        if len(self.text) != self.end - self.start:
            raise ValueError("span offsets must measure Python Unicode code points")
        return self


class CoverageSummary(StrictModel):
    source_total_codepoints: int = Field(ge=0)
    source_covered_codepoints: int = Field(ge=0)
    source_judged_codepoints: int = Field(default=0, ge=0)
    draft_total_codepoints: int = Field(ge=0)
    draft_covered_codepoints: int = Field(ge=0)
    draft_judged_codepoints: int = Field(default=0, ge=0)
    window_count: int = Field(ge=0)
    alignment: Literal[
        "full_paired_text",
        "heuristic_proportional_windows_unaligned",
        "unavailable_unaligned_over_budget",
        "unavailable",
    ]
    truncated: bool = False
    supporting_text_truncated: list[
        Literal["glossary_text", "style_guide", "context"]
    ] = Field(default_factory=list)
    estimated_dense_request_bytes: int | None = Field(default=None, ge=0)
    request_byte_budget: int | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _read_early_v1_coverage(cls, value: Any) -> Any:
        """Read the first v1 smoke receipt emitted before computed-field cleanup."""

        if not isinstance(value, dict):
            return value
        data = dict(value)
        data.pop("source_fraction", None)
        data.pop("draft_fraction", None)
        if data.get("alignment") == "heuristic_proportional_windows":
            is_one_full_pair = (
                data.get("window_count") == 1
                and data.get("source_covered_codepoints")
                == data.get("source_total_codepoints")
                and data.get("draft_covered_codepoints")
                == data.get("draft_total_codepoints")
            )
            data["alignment"] = (
                "full_paired_text"
                if is_one_full_pair
                else "heuristic_proportional_windows_unaligned"
            )
        return data

    @property
    def source_fraction(self) -> float:
        if self.source_total_codepoints == 0:
            return 1.0
        return self.source_covered_codepoints / self.source_total_codepoints

    @property
    def draft_fraction(self) -> float:
        if self.draft_total_codepoints == 0:
            return 1.0
        return self.draft_covered_codepoints / self.draft_total_codepoints

    @property
    def source_judged_fraction(self) -> float:
        if self.source_total_codepoints == 0:
            return 1.0
        return self.source_judged_codepoints / self.source_total_codepoints

    @property
    def draft_judged_fraction(self) -> float:
        if self.draft_total_codepoints == 0:
            return 1.0
        return self.draft_judged_codepoints / self.draft_total_codepoints


class SemanticSignal(StrictModel):
    category: SemanticCategory
    probability: float = Field(ge=0.0, le=1.0)
    impact_domain: Literal["fidelity", "style"] = "fidelity"
    source_span: TextSpan
    draft_span: TextSpan
    window_index: int = Field(ge=0)
    localization: Literal["full_pair", "heuristic_unaligned"] = "heuristic_unaligned"

    @model_validator(mode="after")
    def _documents_and_domain_match(self) -> "SemanticSignal":
        if self.source_span.document != "source" or self.draft_span.document != "draft":
            raise ValueError("signals require one source span and one draft span")
        if self.category == "register_style" and self.impact_domain != "style":
            raise ValueError("register_style must remain separate from fidelity")
        if self.category != "register_style" and self.impact_domain != "fidelity":
            raise ValueError("only register_style is a stylistic signal")
        return self


class GatewayRouting(StrictModel):
    original_model_id: str | None = None
    resolved_provider: str | None = None
    canonical_slug: str | None = None
    final_provider: str | None = None
    generation_id: str | None = None


class SemanticRequestRecord(StrictModel):
    window_index: int = Field(ge=0)
    status: Literal["completed", "partial", "unavailable"]
    attempts: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    accounting_complete: bool = True
    latency_ms: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: Decimal | None = Field(default=None, ge=0)
    market_cost_usd: Decimal | None = Field(default=None, ge=0)
    surcharge_cost_usd: Decimal | None = Field(default=None, ge=0)
    gateway_cost_usd: Decimal | None = Field(default=None, ge=0)
    last_response_input_tokens: int | None = Field(default=None, ge=0)
    last_response_output_tokens: int | None = Field(default=None, ge=0)
    last_response_cost_usd: Decimal | None = Field(default=None, ge=0)
    last_response_market_cost_usd: Decimal | None = Field(default=None, ge=0)
    last_response_surcharge_cost_usd: Decimal | None = Field(default=None, ge=0)
    last_response_gateway_cost_usd: Decimal | None = Field(default=None, ge=0)
    routing: GatewayRouting = Field(default_factory=GatewayRouting)
    error_code: str | None = None
    error_message: str | None = None
    receipt_sha256s: list[str] = Field(default_factory=list)

    @field_validator("receipt_sha256s")
    @classmethod
    def _valid_receipt_hashes(cls, value: list[str]) -> list[str]:
        if any(len(item) != 64 or any(char not in "0123456789abcdef" for char in item) for item in value):
            raise ValueError("receipt hashes must be lowercase SHA-256 hex")
        return value

    @field_validator(
        "cost_usd",
        "market_cost_usd",
        "surcharge_cost_usd",
        "gateway_cost_usd",
        "last_response_cost_usd",
        "last_response_market_cost_usd",
        "last_response_surcharge_cost_usd",
        "last_response_gateway_cost_usd",
        mode="before",
    )
    @classmethod
    def _read_serialized_decimal(cls, value: Any) -> Any:
        if isinstance(value, str):
            return Decimal(value)
        return value


class SemanticIssue(StrictModel):
    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=1000)
    window_index: int | None = Field(default=None, ge=0)
    question_id: str | None = Field(default=None, max_length=100)


class SemanticSignalReport(StrictModel):
    schema_version: Literal["semantic-signals.v1"] = "semantic-signals.v1"
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    draft_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_set: QuestionSetName
    question_version: str = Field(min_length=1)
    windowing_version: str | None = None
    render_version: str | None = None
    requested_model: str = Field(min_length=1)
    resolved_model: str | None = None
    snapshot_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    policy_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    advisory_threshold: float = Field(default=0.5, ge=0, le=1)
    status: Literal["completed", "partial", "unavailable"]
    coverage: CoverageSummary
    signals: list[SemanticSignal] = Field(default_factory=list)
    requests: list[SemanticRequestRecord] = Field(default_factory=list)
    issues: list[SemanticIssue] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("created_at", mode="before")
    @classmethod
    def _read_serialized_datetime(cls, value: Any) -> Any:
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value

    @model_validator(mode="before")
    @classmethod
    def _read_early_v1_report(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        coverage = data.get("coverage")
        if isinstance(coverage, dict):
            old_alignment = coverage.get("alignment")
            one_full_pair = (
                old_alignment == "heuristic_proportional_windows"
                and coverage.get("window_count") == 1
                and coverage.get("source_covered_codepoints")
                == coverage.get("source_total_codepoints")
                and coverage.get("draft_covered_codepoints")
                == coverage.get("draft_total_codepoints")
            )
            if one_full_pair and isinstance(data.get("signals"), list):
                data["signals"] = [
                    {**signal, "localization": "full_pair"}
                    if isinstance(signal, dict) and "localization" not in signal
                    else signal
                    for signal in data["signals"]
                ]
        return data

    def is_fresh(self, snapshot: SemanticSnapshot) -> bool:
        return (
            self.source_sha256 == snapshot.source_sha256
            and self.draft_sha256 == snapshot.draft_sha256
        )
