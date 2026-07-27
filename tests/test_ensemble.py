from __future__ import annotations

from agentic_translation.ensemble import blind_and_shuffle, generate_repair_candidates
from agentic_translation.glossary import load_glossary
from agentic_translation.models import QAFinding, QALocation
from agentic_translation.providers_offline import OfflineJudgeProvider


def test_panel_candidate_scoring_prefers_bracketed_canonical_text_across_seeds() -> None:
    glossary = load_glossary("samples/public_demo/terms/master_glossary.txt")
    finding = QAFinding(
        check_id="system_panel_count",
        severity="warning",
        message="Panel mismatch.",
        location=QALocation(chapter="0001"),
        found="1",
        expected="2",
        auto_repairable=True,
    )
    candidates = generate_repair_candidates(
        translated_text="Chapter 1\n\n[Simulator Started]\n\nRemaining uses: 3",
        finding=finding,
        glossary=glossary,
    )

    for seed in range(20):
        blinded = blind_and_shuffle(candidates, seed)
        decision = OfflineJudgeProvider().judge(
            source_text="【模拟器启动】\n\n【剩余次数：3】",
            candidates=blinded,
            glossary=glossary,
            seed=seed,
        )
        selected = next(candidate for candidate in blinded if candidate.candidate_id == decision.selected_candidate_id)

        assert selected.text == "[Remaining uses: 3]"


def test_panel_candidates_show_editors_room_worker_identities() -> None:
    glossary = load_glossary("samples/public_demo/terms/master_glossary.txt")
    finding = QAFinding(
        check_id="system_panel_count",
        severity="warning",
        message="Panel mismatch.",
        location=QALocation(chapter="0001"),
        found="1",
        expected="2",
        auto_repairable=True,
    )

    candidates = generate_repair_candidates(
        translated_text="Chapter 1\n\n[Simulator Started]\n\nRemaining uses: 3",
        finding=finding,
        glossary=glossary,
    )

    sources = {candidate.source for candidate in candidates}
    assert "literal worker / unchanged" in sources
    assert "canon-strict worker / panel_repair" in sources
    assert "fluent worker / style_repair" in sources
