"""The shipped documentation must work without any private checkout."""
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def _check_link(document: Path, link: str) -> None:
    parsed = urlsplit(link.strip().strip("<>"))
    if parsed.scheme or parsed.netloc or not parsed.path:
        return
    target = (document.parent / unquote(parsed.path)).resolve()
    assert target.is_relative_to(ROOT), f"{document.relative_to(ROOT)}: link leaves repository: {link}"
    assert target.exists(), f"{document.relative_to(ROOT)}: missing link: {link}"


def test_every_published_markdown_and_svg_link_resolves() -> None:
    excluded = {".git", ".venv", "runs", ".agentic_cache", "__pycache__", ".pytest_cache"}
    for document in ROOT.rglob("*.md"):
        if excluded.intersection(document.relative_to(ROOT).parts):
            continue
        text = document.read_text(encoding="utf-8")
        for link in re.findall(r'\]\(([^\s)]+)(?:\s+"[^"]*")?\)', text):
            _check_link(document, link)
        for attribute, value in re.findall(r'''\b(src|srcset|href)=["']([^"']+)["']''', text):
            links = [value] if attribute != "srcset" else [item.strip().split()[0] for item in value.split(",")]
            for link in links:
                _check_link(document, link)
    for document in (ROOT / "assets").rglob("*.svg"):
        for element in ET.parse(document).iter():
            for attribute, value in element.attrib.items():
                if attribute.rsplit("}", 1)[-1] == "href":
                    _check_link(document, value)


def test_public_docs_have_no_machine_paths_or_missing_experiment_references() -> None:
    docs = [ROOT / name for name in ("README.md", "DATA_NOTICE.md", "CHANGELOG.md")]
    docs.extend((ROOT / "docs").glob("*.md"))
    for document in docs:
        text = document.read_text(encoding="utf-8")
        assert "/Users/" not in text, document
        assert "private-" + "tl-harness" not in text, document
        for cited in set(re.findall(r"experiments/[\w-]+", text)):
            assert (ROOT / cited).is_dir(), f"{document}: missing {cited}"


def test_public_data_notice_describes_the_boundary() -> None:
    notice = (ROOT / "DATA_NOTICE.md").read_text(encoding="utf-8").lower()
    assert "aggregate" in notice
    assert "authored" in notice
    assert "mit license" in notice
    assert "private" in notice
