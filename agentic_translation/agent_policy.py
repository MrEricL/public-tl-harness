"""Small, deterministic policy checks for Harness v3 tool proposals."""

from __future__ import annotations

from pathlib import Path

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


class UnattendedToolPolicy(DefaultToolPolicy):
    """No human pause: allow bounded tools and reject legacy persistent writes."""

    UNATTENDED_DENY_RULE_ID = "unattended-persistent-deny.v1"

    def before_tool(self, action: AgentAction, spec: ToolSpec) -> PolicyDecision:
        decision = super().before_tool(action, spec)
        if decision.outcome == "require_approval":
            return PolicyDecision(
                outcome="reject", rule_id=self.UNATTENDED_DENY_RULE_ID,
                reason="Unattended sessions cannot request a human-approved persistent write.",
            )
        return decision


class RunLocalGlossaryAuthority:
    """Authorize only validated paths inside one run's glossary directory.

    This grants automation authority distinct from the legacy approval tool.
    Callers retain the actual write/validation responsibility and record the
    resulting automated decision receipt; this class never edits a file.
    """

    RULE_ID = "run-local-glossary.v1"

    def __init__(self, run_dir: str | Path, master_path: str | Path | None = None) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.master_path = Path(master_path).resolve() if master_path is not None else None

    def authorize(self, target: str | Path) -> PolicyDecision:
        path = Path(target).resolve()
        glossary_dir = self.run_dir / "glossary"
        if path == self.master_path or not path.is_relative_to(glossary_dir) or path == glossary_dir:
            return PolicyDecision(
                outcome="reject", rule_id=self.RULE_ID,
                reason="Glossary writes are limited to files inside this run's glossary directory.",
            )
        return PolicyDecision(
            outcome="allow", rule_id=self.RULE_ID,
            reason="Automated run-local glossary write is authorized by unattended policy.",
        )


__all__ = [
    "ApprovalDecision",
    "ApprovalRequest",
    "DefaultToolPolicy",
    "UnattendedToolPolicy",
    "RunLocalGlossaryAuthority",
    "GlossaryPromotionProposal",
    "PolicyDecision",
]
