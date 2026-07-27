from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import ProviderCallRecord, QAReport
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


class ResolveTerminologyAction(AgentActionBase):
    """Ask the configured terminology resolver to arbitrate one source term."""

    tool: Literal["resolve_terminology"] = "resolve_terminology"
    term: str = Field(min_length=1, max_length=200)
    finding_index: int | None = Field(default=None, ge=0)


class SubmitPatchAction(AgentActionBase):
    tool: Literal["submit_patch"] = "submit_patch"
    old_text: str = Field(min_length=1)
    new_text: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class EscalateAction(AgentActionBase):
    tool: Literal["escalate"] = "escalate"
    reason: str = Field(min_length=1)


class FinishAction(AgentActionBase):
    tool: Literal["finish"] = "finish"
    summary: str = Field(min_length=1)


AgentAction = Annotated[
    GetQAFindingsAction
    | ReadSourceContextAction
    | ReadTranslationContextAction
    | LookupGlossaryAction
    | ResolveTerminologyAction
    | SubmitPatchAction
    | EscalateAction
    | FinishAction,
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
