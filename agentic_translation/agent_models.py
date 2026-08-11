from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import GlossaryParseResult, ProviderCallRecord, QAReport
from .terminology_models import TerminologyResolution


class AgentActionBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class GetQAFindingsAction(AgentActionBase):
    tool: Literal["get_qa_findings"] = "get_qa_findings"


class ReadSourceContextAction(AgentActionBase):
    tool: Literal["read_source_context"] = "read_source_context"
    finding_index: int = Field(ge=0)
    radius: int = Field(ge=0)


class ReadTranslationContextAction(AgentActionBase):
    tool: Literal["read_translation_context"] = "read_translation_context"
    finding_index: int = Field(ge=0)
    radius: int = Field(ge=0)


class LookupGlossaryAction(AgentActionBase):
    tool: Literal["lookup_glossary"] = "lookup_glossary"
    term: str = Field(min_length=1)


class SearchToolsAction(AgentActionBase):
    """Discover trusted tools by deterministic metadata search."""

    tool: Literal["tools.search"] = "tools.search"
    query: str = Field(min_length=1, max_length=120)
    limit: int = Field(default=8, ge=1, le=16)


class ResolveTerminologyAction(AgentActionBase):
    """Ask the configured terminology resolver to arbitrate one source term."""

    tool: Literal["resolve_terminology"] = "resolve_terminology"
    term: str = Field(min_length=1, max_length=200)
    finding_index: int | None = Field(default=None, ge=0)


class TextEdit(AgentActionBase):
    """One bounded exact text replacement within the working translation."""

    old_text: str = Field(min_length=1, max_length=1000)
    # Empty ``new_text`` is intentional: a bounded edit may remove a span.
    new_text: str = Field(max_length=1000)


class SubmitPatchAction(AgentActionBase):
    tool: Literal["submit_patch"] = "submit_patch"
    edits: list[TextEdit] = Field(min_length=1, max_length=8)
    rationale: str = Field(min_length=1, max_length=1200)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_single_edit(cls, value: Any) -> Any:
        """Normalize v1/v2 ``old_text``/``new_text`` payloads at validation time.

        The normalized model deliberately contains only ``edits``.  This keeps
        the v3/native JSON schema structured while allowing historical replay
        cache responses to continue validating unchanged.
        """

        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        if "edits" not in data and ("old_text" in data or "new_text" in data):
            if "old_text" in data and "new_text" in data:
                data["edits"] = [
                    {"old_text": data.pop("old_text"), "new_text": data.pop("new_text")}
                ]
        return data

    @property
    def old_text(self) -> str:
        """Compatibility accessor for callers that handled one legacy edit."""

        return self.edits[0].old_text

    @property
    def new_text(self) -> str:
        """Compatibility accessor for callers that handled one legacy edit."""

        return self.edits[0].new_text


class NormalizePunctuationAction(AgentActionBase):
    """Normalize Chinese punctuation in the working translation."""

    tool: Literal["normalize_punctuation"] = "normalize_punctuation"


class EscalateAction(AgentActionBase):
    tool: Literal["escalate"] = "escalate"
    reason: str = Field(min_length=1)


class FinishAction(AgentActionBase):
    tool: Literal["finish"] = "finish"
    summary: str = Field(min_length=1)


class PromoteGlossaryTermAction(AgentActionBase):
    """Request promotion of an episode-local term into canonical glossary state."""

    tool: Literal["promote_glossary_term"] = "promote_glossary_term"
    term: str = Field(min_length=1, max_length=200)
    rationale: str = Field(min_length=1, max_length=1200)


AgentAction = Annotated[
    GetQAFindingsAction
    | ReadSourceContextAction
    | ReadTranslationContextAction
    | LookupGlossaryAction
    | SearchToolsAction
    | ResolveTerminologyAction
    | SubmitPatchAction
    | NormalizePunctuationAction
    | EscalateAction
    | FinishAction
    | PromoteGlossaryTermAction,
    Field(discriminator="tool"),
]

AgentFinalStatus = Literal["verified", "escalated", "budget_exhausted", "failed"]


class AgentObservation(BaseModel):
    ok: bool
    kind: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class AgentStep(BaseModel):
    sequence: int = Field(ge=1)
    action: dict[str, Any]
    observation: AgentObservation
    provider_call: ProviderCallRecord | None = None
    auxiliary_provider_calls: list[ProviderCallRecord] = Field(default_factory=list)
    qa_before: QAReport | None = None
    qa_after: QAReport | None = None


class AgentEpisode(BaseModel):
    schema_version: str = "0.1"
    episode_id: str
    run_id: str
    story_slug: str
    chapter: str
    provider_mode: str
    provider: str
    model: str
    max_steps: int = Field(default=5, ge=1)
    max_patch_attempts: int = Field(default=2, ge=1)
    initial_qa: QAReport
    final_qa: QAReport | None = None
    steps: list[AgentStep] = Field(default_factory=list)
    terminology_resolutions: list[TerminologyResolution] = Field(default_factory=list)
    final_status: AgentFinalStatus | None = None
    summary: str = ""


class PolicyDecision(BaseModel):
    """The bounded result of evaluating one proposed tool action."""

    model_config = ConfigDict(extra="forbid", strict=True)

    outcome: Literal["allow", "reject", "require_approval", "abort"]
    rule_id: str = Field(min_length=1, max_length=120)
    reason: str = Field(min_length=1, max_length=2000)


class ApprovalRequest(BaseModel):
    """A request for a reviewer to authorize one proposed persistent effect."""

    model_config = ConfigDict(extra="forbid", strict=True)

    proposal_id: str = Field(min_length=1, max_length=200)
    tool: str = Field(min_length=1, max_length=64)
    arguments: dict[str, Any] = Field(max_length=32)
    reason: str = Field(min_length=1, max_length=2000)


class ApprovalDecision(BaseModel):
    """A reviewer's bounded approval or rejection receipt."""

    model_config = ConfigDict(extra="forbid", strict=True)

    proposal_id: str = Field(min_length=1, max_length=200)
    decision: Literal["approved", "rejected"]
    reviewer: str = Field(min_length=1, max_length=200)
    note: str = Field(min_length=1, max_length=2000)


class GlossaryPromotionProposal(BaseModel):
    """Exact before/after content prepared for canonical glossary promotion."""

    model_config = ConfigDict(extra="forbid", strict=True)

    proposal_id: str = Field(min_length=1, max_length=200)
    canonical_glossary_path: str = Field(min_length=1, max_length=4096)
    term: str = Field(min_length=1, max_length=300)
    previous_target: str | None = Field(default=None, max_length=300)
    proposed_target: str = Field(min_length=1, max_length=300)
    before_text: str = Field(max_length=100_000)
    after_text: str = Field(max_length=100_000)
    before_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    after_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )


class RepairExecutorSnapshot(BaseModel):
    """Serializable working state for the bounded repair executor."""

    model_config = ConfigDict(extra="forbid", strict=True)

    current_text: str = Field(max_length=100_000)
    current_qa: QAReport
    episode_glossary: GlossaryParseResult
    escalated: bool
    finished: bool


class SessionEvent(BaseModel):
    """One append-only, sequence-numbered session record."""

    model_config = ConfigDict(extra="forbid", strict=True)

    sequence: int = Field(ge=1)
    event_type: str = Field(min_length=1, max_length=120)
    payload: dict[str, Any] = Field(max_length=100)


class AgentSessionIdentity(BaseModel):
    """Strict task and runtime identity bound to a durable session snapshot."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["agent-session-identity.v1"] = "agent-session-identity.v1"
    run_id: str = Field(min_length=1, max_length=200)
    story_slug: str = Field(min_length=1, max_length=200)
    chapter: str = Field(min_length=1, max_length=200)
    provider_mode: str = Field(min_length=1, max_length=200)
    provider: str = Field(min_length=1, max_length=200)
    model: str = Field(min_length=1, max_length=200)
    tool_schema_version: str = Field(min_length=1, max_length=200)
    source_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    master_glossary_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    registry_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    tool_protocol: Literal["json_prompt", "native_function"]


class AgentSessionSnapshot(BaseModel):
    """Durable projection used to resume a Harness v3 session."""

    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal[
        "running",
        "awaiting_approval",
        "completed",
        "rejected",
        "blocked",
        "failed",
    ]
    episode: AgentEpisode
    executor: RepairExecutorSnapshot
    identity: AgentSessionIdentity | None = None
    prior_steps: list[dict[str, Any]] = Field(default_factory=list, max_length=24)
    patch_attempts: int = Field(default=0, ge=0, le=100)
    exposed_tool_names: list[str] = Field(default_factory=list, max_length=32)
    pending_approval: ApprovalRequest | None = None
    pending_proposal: GlossaryPromotionProposal | None = None
    pending_decision: ApprovalDecision | None = None
    last_event_sequence: int = Field(default=0, ge=0)
