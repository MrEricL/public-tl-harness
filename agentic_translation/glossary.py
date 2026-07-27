from __future__ import annotations

from pathlib import Path

from .models import GlossaryEntry, GlossaryParseResult


def parse_glossary_text(text: str) -> GlossaryParseResult:
    entries: list[GlossaryEntry] = []
    warnings: list[str] = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# block:"):
            variants = [
                part.strip()
                for part in line.split(":", 1)[1].split(";")
                if part.strip()
            ]
            if not entries:
                warnings.append(f"line {line_number}: # block before any glossary entry")
                continue
            last = entries[-1]
            entries[-1] = last.model_copy(
                update={"blocked_variants": last.blocked_variants + variants}
            )
            continue
        if line.startswith("#"):
            continue
        if "->" in line:
            source, target = [part.strip() for part in line.split("->", 1)]
            if source and target:
                entries.append(
                    GlossaryEntry(source=source, target=target, candidates=[target])
                )
            continue
        if ":" in line or "：" in line:
            normalized = line.replace("：", ":", 1)
            source, candidate_blob = [part.strip() for part in normalized.split(":", 1)]
            candidates = [part.strip() for part in candidate_blob.split(",") if part.strip()]
            if source and candidates:
                entries.append(
                    GlossaryEntry(
                        source=source,
                        target=candidates[0],
                        candidates=candidates,
                    )
                )
            continue
        warnings.append(f"line {line_number}: ignored malformed glossary line")

    blocked: list[str] = []
    for entry in entries:
        for variant in entry.blocked_variants:
            if variant not in blocked:
                blocked.append(variant)

    return GlossaryParseResult(entries=entries, warnings=warnings, blocked_variants=blocked)


def load_glossary(path: str | Path) -> GlossaryParseResult:
    return parse_glossary_text(Path(path).read_text(encoding="utf-8"))


def glossary_map(glossary: GlossaryParseResult) -> dict[str, str]:
    return {entry.source: entry.target for entry in glossary.entries}


def matched_entries(source_text: str, glossary: GlossaryParseResult) -> list[GlossaryEntry]:
    return [entry for entry in glossary.entries if entry.source in source_text]

