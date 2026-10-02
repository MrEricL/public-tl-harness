"""Typed, source-bound contracts for TypeSafe passage decisions.

The request ID covers the exact input text, question schema, model selection, and
routing policy. A changed neighbor/context or draft therefore cannot reuse a
previous decision. Receipts intentionally omit source and draft prose.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


QUESTION_SCHEMA_VERSION = "translation-screen.v1"
ROUTING_POLICY_VERSION = "adaptive-routing.v1"
DEFAULT_THRESHOLDS = {"material": 0.50, "diagnostic": 0.15, "readability": 1.5}


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    segment_id: str = Field(min_length=1)
    source_text: str = Field(min_length=1)
    draft_text: str = Field(min_length=1)
    context_text: str = ""
    candidate_text: str | None = None
    requested_model: str = Field(default="jev-1.13.0", min_length=1)
    routing_policy_version: str = ROUTING_POLICY_VERSION
    routing_thresholds: dict[str, float] = Field(default_factory=lambda: DEFAULT_THRESHOLDS.copy())

    @model_validator(mode="after")
    def _validate_thresholds(self) -> "DecisionRequest":
        if set(self.routing_thresholds) != set(DEFAULT_THRESHOLDS):
            raise ValueError("routing_thresholds must define material, diagnostic, and readability")
        if not 0 <= self.routing_thresholds["diagnostic"] <= self.routing_thresholds["material"] <= 1:
            raise ValueError("diagnostic/material thresholds must satisfy 0 <= diagnostic <= material <= 1")
        if not 0 <= self.routing_thresholds["readability"] <= 3:
            raise ValueError("readability threshold must be between 0 and 3")
        return self

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()

    @property
    def draft_sha256(self) -> str:
        return hashlib.sha256(self.draft_text.encode("utf-8")).hexdigest()

    @property
    def context_sha256(self) -> str:
        return hashlib.sha256(self.context_text.encode("utf-8")).hexdigest()


class DecisionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["decision-receipt.v1"] = "decision-receipt.v1"
    request_id: str
    segment_id: str
    provider: Literal["typesafe"] = "typesafe"
    requested_model: str
    served_model: str | None = None
    source_sha256: str
    draft_sha256: str
    context_sha256: str
    candidate_sha256: str | None = None
    question_schema_version: str
    question_schema_sha256: str
    raw_answers: dict[str, Any] = Field(default_factory=dict)
    derived_flags: dict[str, bool] = Field(default_factory=dict)
    routing_policy_version: str
    routing_thresholds: dict[str, float]
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_usd: Decimal | None = None
    latency_ms: float = Field(ge=0)
    attempt_count: int = Field(ge=0)
    transport_status: Literal["ok", "unavailable", "invalid_response"]
    cache_status: Literal["live", "recorded", "replayed", "miss"]
    http_status: int | None = Field(default=None, ge=100, le=599)
    error_code: str | None = None


class DecisionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "unavailable"]
    receipt: DecisionReceipt

    @property
    def needs_review(self) -> bool:
        return self.status != "ok" or any(self.receipt.derived_flags.values())

    @property
    def material_flags(self) -> list[str]:
        """Individual error categories above the material threshold."""
        return [name for name, flagged in self.receipt.derived_flags.items() if flagged and name in {
            "actor_relation_error", "negation_condition_error", "quantity_time_error",
            "omitted_material", "unsupported_addition", "terminology_conflict",
            "patch_unsupported_claim",
        }]

    @property
    def diagnostic_flag_names(self) -> tuple[str, ...]:
        """Error questions whose probability falls in the diagnostic band (routing v2).

        Reads the raw per-question probabilities directly so callers can count
        how many distinct error categories landed between the diagnostic and
        material thresholds, independent of the single umbrella
        ``diagnostic_error`` flag.
        """
        diagnostic = self.receipt.routing_thresholds["diagnostic"]
        material = self.receipt.routing_thresholds["material"]
        names: list[str] = []
        for name in (
            "actor_relation_error", "negation_condition_error", "quantity_time_error",
            "omitted_material", "unsupported_addition", "terminology_conflict",
        ):
            answer = self.receipt.raw_answers.get(name)
            if isinstance(answer, dict) and isinstance(answer.get("noul"), (int, float)):
                value = float(answer["noul"])
                if diagnostic <= value < material:
                    names.append(name)
        return tuple(names)

    @property
    def source_sufficiency(self) -> str | None:
        answer = self.receipt.raw_answers.get("source_sufficiency", {})
        return answer.get("choice") if isinstance(answer, dict) else None

    @property
    def route_level(self) -> Literal["clean", "diagnostic", "material", "fallback"]:
        if self.status != "ok":
            return "fallback"
        if self.material_flags or self.receipt.derived_flags.get("source_incomplete"):
            return "material"
        if self.needs_review:
            return "diagnostic"
        return "clean"
