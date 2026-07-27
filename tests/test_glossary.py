from __future__ import annotations

from agentic_translation.glossary import load_glossary, parse_glossary_text


def test_parse_arrow_glossary_attaches_block_to_previous_entry() -> None:
    parsed = parse_glossary_text(
        """
        天道 -> Heavenly Dao
        # block: Way of Heaven; Heavenly Way
        灵气 -> spiritual energy
        """
    )

    assert parsed.warnings == []
    assert parsed.entries[0].source == "天道"
    assert parsed.entries[0].target == "Heavenly Dao"
    assert parsed.entries[0].blocked_variants == ["Way of Heaven", "Heavenly Way"]
    assert parsed.entries[1].blocked_variants == []


def test_parse_colon_glossary_uses_first_candidate_as_target() -> None:
    parsed = parse_glossary_text("天道: Heavenly Dao, Way of Heaven\n")

    assert parsed.entries[0].source == "天道"
    assert parsed.entries[0].target == "Heavenly Dao"
    assert parsed.entries[0].candidates == ["Heavenly Dao", "Way of Heaven"]


def test_malformed_leading_block_produces_warning() -> None:
    parsed = parse_glossary_text("# block: orphan\n天道 -> Heavenly Dao\n")

    assert len(parsed.warnings) == 1
    assert "before any glossary entry" in parsed.warnings[0]


def test_load_public_demo_glossary() -> None:
    parsed = load_glossary("samples/public_demo/terms/master_glossary.txt")

    assert [entry.target for entry in parsed.entries] == [
        "Simulator",
        "Heavenly Dao",
        "simulation",
        "remaining uses",
    ]
    assert "Way of Heaven" in parsed.blocked_variants
    assert "deduction" in parsed.blocked_variants

