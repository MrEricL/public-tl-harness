from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic_translation.semantic_models import (
    JevPolicy,
    SemanticSignal,
    SemanticSnapshot,
    TextSpan,
)


def test_snapshot_hashes_exact_unicode_text() -> None:
    first = SemanticSnapshot(source_text="魔😀王", draft_text="Demon king")
    second = SemanticSnapshot(source_text="魔😀王 ", draft_text="Demon king")

    assert first.source_sha256 != second.source_sha256
    assert first.draft_sha256 == second.draft_sha256
    assert len(first.source_sha256) == 64


def test_policy_is_strict_and_window_must_advance() -> None:
    with pytest.raises(ValidationError):
        JevPolicy(max_concurrency=9)
    with pytest.raises(ValidationError):
        JevPolicy(window_chars=256, window_overlap=256)
    with pytest.raises(ValidationError):
        JevPolicy(mode="enabled")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        JevPolicy.model_validate({"mode": "off", "unknown": True})


def test_text_spans_use_python_unicode_codepoint_offsets() -> None:
    text = "甲😀乙"
    span = TextSpan(document="source", start=0, end=3, text=text)
    assert span.end - span.start == len(text)

    with pytest.raises(ValidationError):
        TextSpan(document="source", start=0, end=4, text=text)


def test_style_signal_cannot_be_misrepresented_as_fidelity() -> None:
    source = TextSpan(document="source", start=0, end=1, text="甲")
    draft = TextSpan(document="draft", start=0, end=1, text="A")
    with pytest.raises(ValidationError):
        SemanticSignal(
            category="register_style",
            probability=0.8,
            impact_domain="fidelity",
            source_span=source,
            draft_span=draft,
            window_index=0,
            localization="full_pair",
        )
    with pytest.raises(ValidationError):
        SemanticSignal(
            category="omission",
            probability=1.01,
            source_span=source,
            draft_span=draft,
            window_index=0,
            localization="full_pair",
        )
