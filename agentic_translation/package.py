from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path

from ebooklib import epub

from .qa import CHINESE_RE, PROMPT_LEAK_RE
from .text import chapter_display_label


def build_txt(*, output_path: Path, chapter: str, translated_text: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"Chapter: {chapter}\n\n{translated_text.strip()}\n", encoding="utf-8")
    return output_path


def build_epub(*, output_path: Path, story_title: str, chapter: str, translated_text: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    book = epub.EpubBook()
    book.set_identifier(f"{story_title}-{chapter}")
    book.set_title(story_title)
    book.set_language("en")
    chapter_label = chapter_display_label(chapter)
    safe_chapter = re.sub(r"[^A-Za-z0-9_.-]+", "_", chapter.strip()) or "chapter"
    chapter_doc = epub.EpubHtml(title=f"Chapter {chapter_label}", file_name=f"chapter_{safe_chapter}.xhtml", lang="en")
    lines = [line.strip() for line in translated_text.strip().splitlines() if line.strip()]
    heading = lines[0] if lines else f"Chapter {chapter_label}"
    body = lines[1:] if len(lines) > 1 else []
    paragraphs = "".join(f"<p>{html.escape(line)}</p>" for line in body)
    chapter_doc.content = f"<h2>{html.escape(heading)}</h2>{paragraphs}"
    book.add_item(chapter_doc)
    book.toc = (chapter_doc,)
    book.spine = ["nav", chapter_doc]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(output_path), book, {})
    return output_path


def build_txt_collection(*, output_path: Path, chapters: dict[str, str]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    for chapter, translated_text in chapters.items():
        parts.append(f"Chapter: {chapter}\n\n{translated_text.strip()}")
    output_path.write_text("\n\n".join(parts).strip() + "\n", encoding="utf-8")
    return output_path


def build_epub_collection(*, output_path: Path, story_title: str, chapters: dict[str, str]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    book = epub.EpubBook()
    book.set_identifier(f"{story_title}-{next(iter(chapters), 'empty')}-{len(chapters)}")
    book.set_title(story_title)
    book.set_language("en")
    spine: list[object] = ["nav"]
    toc: list[epub.EpubHtml] = []
    for chapter, translated_text in chapters.items():
        chapter_label = chapter_display_label(chapter)
        safe_chapter = re.sub(r"[^A-Za-z0-9_.-]+", "_", chapter.strip()) or "chapter"
        chapter_doc = epub.EpubHtml(title=f"Chapter {chapter_label}", file_name=f"chapter_{safe_chapter}.xhtml", lang="en")
        lines = [line.strip() for line in translated_text.strip().splitlines() if line.strip()]
        heading = lines[0] if lines else f"Chapter {chapter_label}"
        body = lines[1:] if len(lines) > 1 else []
        paragraphs = "".join(f"<p>{html.escape(line)}</p>" for line in body)
        chapter_doc.content = f"<h2>{html.escape(heading)}</h2>{paragraphs}"
        book.add_item(chapter_doc)
        spine.append(chapter_doc)
        toc.append(chapter_doc)
    book.toc = tuple(toc)
    book.spine = spine
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(output_path), book, {})
    return output_path


def verify_txt_artifact(path: Path) -> dict[str, int | bool]:
    text = path.read_text(encoding="utf-8")
    return {
        "chapter_markers": len(re.findall(r"^Chapter:\s+", text, re.MULTILINE)),
        "contains_chinese": bool(CHINESE_RE.search(text)),
        "contains_prompt_leakage": bool(PROMPT_LEAK_RE.search(text)),
    }


def verify_epub_artifact(path: Path) -> dict[str, int | bool]:
    with zipfile.ZipFile(path) as archive:
        xhtml = [
            name
            for name in archive.namelist()
            if name.endswith(".xhtml") and not name.endswith("nav.xhtml")
        ]
        joined = "\n".join(archive.read(name).decode("utf-8", errors="ignore") for name in xhtml)
    return {
        "xhtml_chapters": len(xhtml),
        "contains_chinese": bool(CHINESE_RE.search(joined)),
        "contains_prompt_leakage": bool(PROMPT_LEAK_RE.search(joined)),
    }
