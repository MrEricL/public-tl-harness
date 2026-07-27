from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class PanelSegment:
    index: int
    line_number: int
    text: str


def split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]


def join_paragraphs(paragraphs: list[str]) -> str:
    return "\n\n".join(paragraphs).strip()


def first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def find_paragraph_index(text: str, needle: str) -> int | None:
    for index, paragraph in enumerate(split_paragraphs(text)):
        if needle in paragraph:
            return index
    return None


def chapter_display_label(chapter: str) -> str:
    stripped = chapter.strip()
    if stripped.isdigit():
        return str(int(stripped))
    return stripped or "0"


def extract_panel_segments(text: str) -> list[PanelSegment]:
    panels: list[PanelSegment] = []
    active_start_line: int | None = None
    active_lines: list[str] = []
    active_close = ""
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if active_start_line is not None:
            active_lines.append(stripped)
            if active_close in stripped:
                panels.append(
                    PanelSegment(
                        index=len(panels) + 1,
                        line_number=active_start_line,
                        text=" ".join(active_lines),
                    )
                )
                active_start_line = None
                active_lines = []
                active_close = ""
            continue
        opener = ""
        closer = ""
        if stripped.startswith("【"):
            opener = "【"
            closer = "】"
        elif stripped.startswith("["):
            opener = "["
            closer = "]"
        if not opener:
            continue
        if closer in stripped[1:]:
            panels.append(PanelSegment(index=len(panels) + 1, line_number=line_number, text=stripped))
        else:
            active_start_line = line_number
            active_lines = [stripped]
            active_close = closer
    if active_start_line is not None and active_lines:
        panels.append(
            PanelSegment(
                index=len(panels) + 1,
                line_number=active_start_line,
                text=" ".join(active_lines),
            )
        )
    return panels


def trim_for_report(text: str, *, mode: str, max_chars: int) -> str:
    if mode == "redacted":
        return "[redacted]"
    stripped = text.strip()
    if mode == "excerpt" and len(stripped) > max_chars:
        return stripped[:max_chars].rstrip() + "\n\n[excerpt truncated]"
    return stripped
