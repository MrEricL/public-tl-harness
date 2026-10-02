"""Provider protocol and offline/live selection for passage decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from .models import DecisionRequest, DecisionResult


@runtime_checkable
class DecisionProvider(Protocol):
    def evaluate(self, request: DecisionRequest) -> DecisionResult: ...


def make_decision_provider(
    mode: Literal["live", "replay"], *, record_dir: str | Path | None = None,
) -> DecisionProvider:
    """Choose a provider explicitly; replay never creates a live client."""
    if mode == "live":
        from .jev import TypeSafeDecisionProvider

        return TypeSafeDecisionProvider(record_dir=record_dir)
    if mode == "replay":
        if record_dir is None:
            raise ValueError("replay requires record_dir")
        from .replay import RecordedDecisionProvider

        return RecordedDecisionProvider(record_dir)
    raise ValueError(f"unknown decision provider mode: {mode}")
