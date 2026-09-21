from __future__ import annotations

from agentic_translation.semantic_models import (
    JevPolicy,
    SemanticSignal,
    SemanticSnapshot,
)
from agentic_translation.semantic_provider import unavailable_report
from agentic_translation.semantic_signals import (
    build_windows,
    categories_for_set,
    estimate_dense_request_bytes,
    question_definitions,
    render_signal_report,
)


def test_focused_and_dense_share_identical_windows() -> None:
    snapshot = SemanticSnapshot(
        source_text="甲乙丙丁戊己庚辛壬癸" * 80,
        draft_text="abcdefghij" * 77,
    )
    focused_policy = JevPolicy(window_chars=256, window_overlap=40, question_set="focused")
    dense_policy = focused_policy.model_copy(update={"question_set": "dense"})

    focused, focused_coverage = build_windows(
        snapshot,
        window_chars=focused_policy.window_chars,
        overlap=focused_policy.window_overlap,
        max_windows=focused_policy.max_windows,
    )
    dense, dense_coverage = build_windows(
        snapshot,
        window_chars=dense_policy.window_chars,
        overlap=dense_policy.window_overlap,
        max_windows=dense_policy.max_windows,
    )

    assert focused == dense
    assert focused_coverage == dense_coverage
    assert focused_coverage.source_fraction == 1
    assert focused_coverage.draft_fraction == 1
    assert focused_coverage.alignment == "full_paired_text"


def test_long_unequal_texts_use_one_complete_pair_when_dense_request_fits() -> None:
    snapshot = SemanticSnapshot(source_text="甲" * 3000, draft_text="A" * 10_000)
    windows, coverage = build_windows(
        snapshot,
        window_chars=256,
        overlap=20,
        max_windows=1,
    )

    assert len(windows) == 1
    assert windows[0].source_span.text == snapshot.source_text
    assert windows[0].draft_span.text == snapshot.draft_text
    assert coverage.alignment == "full_paired_text"
    assert coverage.truncated is False


def test_dense_utf8_byte_budget_boundary_never_creates_partial_pair() -> None:
    snapshot = SemanticSnapshot(source_text="😀甲" * 1000, draft_text="Aé" * 1000)
    estimated = estimate_dense_request_bytes(snapshot)

    at_limit, at_coverage = build_windows(
        snapshot,
        window_chars=256,
        overlap=20,
        max_windows=2,
        max_request_bytes=estimated,
    )
    over_limit, over_coverage = build_windows(
        snapshot,
        window_chars=256,
        overlap=20,
        max_windows=2,
        max_request_bytes=estimated - 1,
    )

    assert len(at_limit) == 1
    assert at_coverage.source_covered_codepoints == len(snapshot.source_text)
    assert over_limit == []
    assert over_coverage.source_covered_codepoints == 0
    assert over_coverage.draft_covered_codepoints == 0
    assert over_coverage.alignment == "unavailable_unaligned_over_budget"
    assert over_coverage.truncated is True


def test_question_sets_have_five_and_eighteen_narrow_boolean_questions() -> None:
    focused = question_definitions("focused")
    dense = question_definitions("dense")

    assert len(focused) == 5
    assert len(dense) == 18
    assert tuple(focused) == categories_for_set("focused")
    assert tuple(dense) == categories_for_set("dense")
    assert all(question["type"] == "boolean" for question in dense.values())
    assert set(focused).issubset(dense)
    assert "wrong, incomplete, or embellished draft" in dense["source_ambiguity"]["criteria"]["false"]
    assert "no speech" in dense["speaker_attribution"]["criteria"]["false"]
    assert "contains no condition" in dense["conditions_exceptions"]["criteria"]["false"]


def test_render_is_bounded_ranked_and_explicitly_untrusted() -> None:
    snapshot = SemanticSnapshot(source_text="甲乙", draft_text="AB")
    policy = JevPolicy()
    report = unavailable_report(snapshot, policy, "focused", "disabled", code="disabled")
    windows, _ = build_windows(snapshot, window_chars=256, overlap=0, max_windows=1)
    window = windows[0]
    report = report.model_copy(
        update={
            "status": "completed",
            "signals": [
                SemanticSignal(
                    category="omission",
                    probability=0.1,
                    source_span=window.source_span,
                    draft_span=window.draft_span,
                    window_index=0,
                    localization="full_pair",
                ),
                SemanticSignal(
                    category="unsupported_addition",
                    probability=0.9,
                    source_span=window.source_span,
                    draft_span=window.draft_span,
                    window_index=0,
                    localization="full_pair",
                ),
            ],
        }
    )

    rendered = render_signal_report(report, max_findings=1, max_chars=500)
    assert "untrusted model observations" in rendered
    assert "unsupported_addition" in rendered
    assert "omission probability" not in rendered
    assert "summary below=1" in rendered
    assert len(rendered) <= 500
    assert len(render_signal_report(report, max_findings=2, max_chars=40)) <= 40
