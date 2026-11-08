"""Keep the public entry points and headline's measurement scope explicit."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_is_a_short_front_door_to_the_shipped_workflows() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert len(readme.splitlines()) <= 400
    assert "python demo.py" in readme
    assert "agentic_translation.workbench" in readme
    for filename in ("RESULTS", "HOW_IT_WORKS", "TRY_IT", "CODE_MAP", "MEMORY"):
        assert (ROOT / "docs" / f"{filename}.md").is_file()
    assert (ROOT / "samples/showcase/source/0001.txt").is_file()


def test_results_keep_replication_and_pack_comparison_distinct() -> None:
    results = " ".join((ROOT / "docs/RESULTS.md").read_text(encoding="utf-8").split()).lower()
    for number in ("59%", "54%", "61%", "392", "2,862"):
        assert number in results, number
    assert "same three novels" in results
    assert "59" in results and "character" in results
    assert "extractor" in results and "human" in results
    assert "reviewer" in results
    assert "correction" in results or "corrected" in results


def test_historical_evidence_verifier_is_shipped() -> None:
    folder = ROOT / "experiments/heldout_evidence"
    assert all((folder / name).is_file() for name in ("README.md", "RESULTS.json", "verify.py"))


def test_readme_headline_numbers_match_the_results_page() -> None:
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    results = " ".join((ROOT / "docs/RESULTS.md").read_text(encoding="utf-8").split())
    for number in ("59%", "54%", "61%", "62.5%", "392", "2,862"):
        assert number in readme, number
        assert number in results, number
