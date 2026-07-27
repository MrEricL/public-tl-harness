from __future__ import annotations

from pathlib import Path

from agentic_translation.models import ArtifactQAReport
from agentic_translation.package import build_epub, build_txt, verify_epub_artifact, verify_txt_artifact


def test_txt_artifact_qa_detects_chinese_and_prompt_leakage(tmp_path: Path) -> None:
    path = build_txt(
        output_path=tmp_path / "review.txt",
        chapter="0001",
        translated_text="Chapter 1\n\nGlossary: 天道",
    )

    audit = verify_txt_artifact(path)

    assert audit["chapter_markers"] == 1
    assert audit["contains_chinese"] is True
    assert audit["contains_prompt_leakage"] is True


def test_epub_artifact_qa_counts_content_chapters_only(tmp_path: Path) -> None:
    path = build_epub(
        output_path=tmp_path / "review.epub",
        story_title="Public Demo",
        chapter="0042",
        translated_text="Chapter 42\n\nClean paragraph.",
    )

    audit = verify_epub_artifact(path)

    assert audit["xhtml_chapters"] == 1
    assert audit["contains_chinese"] is False
    assert audit["contains_prompt_leakage"] is False


def test_epub_build_accepts_nonnumeric_chapter_ids(tmp_path: Path) -> None:
    path = build_epub(
        output_path=tmp_path / "prologue.epub",
        story_title="Public Demo",
        chapter="prologue",
        translated_text="Prologue\n\nClean paragraph.",
    )

    audit = verify_epub_artifact(path)

    assert path.exists()
    assert audit["xhtml_chapters"] == 1


def test_artifact_qa_report_uses_typed_audits() -> None:
    report = ArtifactQAReport(
        expected_chapters=1,
        txt={"chapter_markers": 1, "contains_chinese": False, "contains_prompt_leakage": False},
        epub={"xhtml_chapters": 1, "contains_chinese": False, "contains_prompt_leakage": False},
        passed=True,
    )

    assert report.txt.chapter_markers == 1
    assert report.epub is not None
    assert report.epub.xhtml_chapters == 1
