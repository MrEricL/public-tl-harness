"""Small, deterministic policy checks for Harness v3 tool proposals."""

from __future__ import annotations

from .agent_models import (
    AgentAction,
    ApprovalDecision,
    ApprovalRequest,
    GlossaryPromotionProposal,
    PolicyDecision,
)
from .agent_tools import ToolSpec


class DefaultToolPolicy:
    """Allow registered bounded tools, pausing only persistent effects."""

    APPROVAL_RULE_ID = "persistent-write.v1"
    ALLOW_RULE_ID = "default-allow.v1"
    MISMATCH_RULE_ID = "tool-spec-mismatch.v1"

    def before_tool(self, action: AgentAction, spec: ToolSpec) -> PolicyDecision:
        """Evaluate one validated action against its registry metadata."""

        if action.tool != spec.name:
            return PolicyDecision(
                outcome="reject",
                rule_id=self.MISMATCH_RULE_ID,
                reason="Proposed action does not match its registered tool specification.",
            )
        if spec.requires_approval:
            return PolicyDecision(
                outcome="require_approval",
                rule_id=self.APPROVAL_RULE_ID,
                reason="Persistent glossary promotion requires explicit approval.",
            )
        return PolicyDecision(
            outcome="allow",
            rule_id=self.ALLOW_RULE_ID,
            reason="Registered bounded action.",
        )


__all__ = [
    "ApprovalDecision",
    "ApprovalRequest",
    "DefaultToolPolicy",
    "GlossaryPromotionProposal",
    "PolicyDecision",
]
