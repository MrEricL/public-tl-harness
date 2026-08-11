from __future__ import annotations

from agentic_translation.agent_models import (
    GetQAFindingsAction,
    PromoteGlossaryTermAction,
)
from agentic_translation.agent_policy import DefaultToolPolicy
from agentic_translation.agent_tools import AGENT_TOOL_REGISTRY


def test_glossary_promotion_requires_approval() -> None:
    action = PromoteGlossaryTermAction(term="道心", rationale="Consensus")
    decision = DefaultToolPolicy().before_tool(
        action,
        AGENT_TOOL_REGISTRY.spec("promote_glossary_term"),
    )

    assert decision.outcome == "require_approval"
    assert decision.rule_id == "persistent-write.v1"


def test_registered_read_tool_is_allowed() -> None:
    action = GetQAFindingsAction()
    decision = DefaultToolPolicy().before_tool(
        action,
        AGENT_TOOL_REGISTRY.spec("get_qa_findings"),
    )

    assert decision.outcome == "allow"
    assert decision.rule_id == "default-allow.v1"


def test_policy_rejects_action_spec_mismatch() -> None:
    action = PromoteGlossaryTermAction(term="道心", rationale="Consensus")
    decision = DefaultToolPolicy().before_tool(
        action,
        AGENT_TOOL_REGISTRY.spec("get_qa_findings"),
    )

    assert decision.outcome == "reject"
    assert decision.rule_id == "tool-spec-mismatch.v1"
