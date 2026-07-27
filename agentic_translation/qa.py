from __future__ import annotations

import re
from collections import Counter

from .models import GlossaryParseResult, QAFinding, QALocation, QAReport, QASummary
from .text import extract_panel_segments, find_paragraph_index, first_non_empty_line, split_paragraphs


CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
CHINESE_PUNCT_RE = re.compile(r"[，。！？；：“”‘’、（）《》【】]")
HEADING_RE = re.compile(r"^Chapter\s+\d+(?::\s*.+)?$")
SITE_NOISE_RE = re.compile(
    r"會員免廣告|加入書架|溫馨提示|下一章|上一章|目錄|Cloudflare|challenge|Just a moment",
    re.IGNORECASE,
)
PROMPT_LEAK_RE = re.compile(
    r"translate the following|as an ai language model|system prompt|developer message|glossary:|source text:|translation:",
    re.IGNORECASE,
)
ALIAS_STOPWORDS = {"a", "an", "and", "by", "for", "in", "of", "on", "or", "the", "to", "with"}


SCORE_PENALTIES = {
    "prompt_leakage": 30,
    "residual_chinese": 15,
    "chinese_punctuation": 8,
    "system_panel_count": 12,
    "blocked_glossary_variant": 8,
    "glossary_required": 6,
    "heading_format": 4,
    "source_noise": 12,
    "source_missing_title": 8,
    "source_missing_body": 10,
    "source_no_chinese": 10,
}


def panel_count(text: str) -> int:
    return len(extract_panel_segments(text))


def summarize(findings: list[QAFinding]) -> QASummary:
    counter = Counter(finding.check_id for finding in findings)
    return QASummary(
        total_findings=len(findings),
        error_count=sum(1 for finding in findings if finding.severity == "error"),
        warning_count=sum(1 for finding in findings if finding.severity == "warning"),
        info_count=sum(1 for finding in findings if finding.severity == "info"),
        by_check=dict(counter),
    )


def weighted_score(report_or_findings: QAReport | list[QAFinding]) -> int:
    findings = report_or_findings.findings if isinstance(report_or_findings, QAReport) else report_or_findings
    score = 100
    for finding in findings:
        score -= SCORE_PENALTIES.get(finding.check_id, 3)
    return max(score, 0)


def _finding(
    *,
    check_id: str,
    chapter: str,
    message: str,
    severity: str = "warning",
    found: str | None = None,
    expected: str | None = None,
    suggested_action: str | None = None,
    paragraph_index: int | None = None,
    line_index: int | None = None,
    snippet: str | None = None,
    auto_repairable: bool = False,
) -> QAFinding:
    return QAFinding(
        check_id=check_id,
        severity=severity,  # type: ignore[arg-type]
        message=message,
        location=QALocation(
            chapter=chapter,
            paragraph_index=paragraph_index,
            line_index=line_index,
            snippet=snippet,
        ),
        found=found,
        expected=expected,
        suggested_action=suggested_action,
        auto_repairable=auto_repairable,
    )


def _find_observed_glossary_alias(translated_text: str, entry_source: str, aliases: list[str]) -> tuple[str, int | None] | None:
    for alias in sorted({alias for alias in aliases if alias and alias != entry_source}, key=len, reverse=True):
        match = re.search(re.escape(alias), translated_text, flags=re.IGNORECASE)
        if match:
            observed = match.group(0)
            return observed, find_paragraph_index(translated_text, observed)
    return None


def _observed_alias_conflicts_with_present_source(
    *,
    observed_alias: str,
    source_text: str,
    glossary: GlossaryParseResult,
    entry_source: str,
) -> bool:
    observed_lower = observed_alias.lower()
    for other in glossary.entries:
        if other.source == entry_source or other.source not in source_text:
            continue
        for candidate in [other.target, *other.candidates]:
            if candidate.lower() == observed_lower:
                return True
    return False


def _alias_stems(text: str) -> set[str]:
    stems: set[str] = set()
    for word in re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower()):
        stripped = word.strip("'-")
        if not stripped or stripped in ALIAS_STOPWORDS:
            continue
        if len(stripped) > 3 and stripped.endswith("ies"):
            stripped = stripped[:-3] + "y"
        elif len(stripped) > 3 and stripped.endswith("s"):
            stripped = stripped[:-1]
        stems.add(stripped)
    return stems


def _cross_glossary_aliases(
    *,
    source_text: str,
    glossary: GlossaryParseResult,
    entry_source: str,
    entry_target: str,
) -> list[str]:
    target_stems = _alias_stems(entry_target)
    if len(target_stems) < 2:
        return []
    aliases: list[str] = []
    for other in glossary.entries:
        if other.source == entry_source or other.source in source_text:
            continue
        for alias in [other.target, *other.candidates]:
            if alias.lower() == entry_target.lower():
                continue
            if len(target_stems & _alias_stems(alias)) >= 2 and alias not in aliases:
                aliases.append(alias)
    return aliases


def run_source_qa(
    *,
    run_id: str,
    story_slug: str,
    chapter: str,
    source_text: str,
) -> QAReport:
    findings: list[QAFinding] = []
    paragraphs = split_paragraphs(source_text)
    title = first_non_empty_line(source_text)
    body = "\n\n".join(paragraphs[1:]).strip() if len(paragraphs) > 1 else ""

    if not title:
        findings.append(
            _finding(
                check_id="source_missing_title",
                chapter=chapter,
                severity="error",
                message="Source chapter has no title line.",
            )
        )
    if not body:
        findings.append(
            _finding(
                check_id="source_missing_body",
                chapter=chapter,
                severity="error",
                message="Source chapter has no body text.",
            )
        )
    if not CHINESE_RE.search(source_text):
        findings.append(
            _finding(
                check_id="source_no_chinese",
                chapter=chapter,
                severity="error",
                message="Source chapter does not appear to contain Chinese text.",
            )
        )
    noise_match = SITE_NOISE_RE.search(source_text)
    if noise_match:
        findings.append(
            _finding(
                check_id="source_noise",
                chapter=chapter,
                found=noise_match.group(0),
                severity="error",
                message="Source contains site noise or challenge-page residue.",
            )
        )

    summary = summarize(findings)
    score = weighted_score(findings)
    return QAReport(
        run_id=run_id,
        story_slug=story_slug,
        chapter=chapter,
        report_type="source",
        findings=findings,
        summary=summary,
        panel_count=panel_count(source_text),
        score=score,
    )


def run_translation_qa(
    *,
    run_id: str,
    story_slug: str,
    chapter: str,
    source_text: str,
    translated_text: str,
    glossary: GlossaryParseResult,
) -> QAReport:
    findings: list[QAFinding] = []
    paragraphs = split_paragraphs(translated_text)
    heading = first_non_empty_line(translated_text)

    if not HEADING_RE.match(heading):
        findings.append(
            _finding(
                check_id="heading_format",
                chapter=chapter,
                found=heading,
                expected="Chapter N or Chapter N: Subtitle",
                message="First non-empty line is not a normalized English chapter heading.",
                suggested_action="Normalize the heading to Chapter N or Chapter N: Subtitle.",
                paragraph_index=0,
                auto_repairable=True,
            )
        )

    for index, paragraph in enumerate(paragraphs):
        residue_match = CHINESE_RE.search(paragraph)
        if residue_match:
            found = residue_match.group(0)
            expected = None
            for entry in glossary.entries:
                if entry.source in paragraph:
                    expected = entry.target
                    break
            findings.append(
                _finding(
                    check_id="residual_chinese",
                    chapter=chapter,
                    paragraph_index=index,
                    snippet=paragraph,
                    found=found,
                    expected=expected,
                    message="Translated text contains residual Chinese.",
                    suggested_action="Replace residual Chinese with natural English.",
                    auto_repairable=bool(expected),
                )
            )
        if CHINESE_PUNCT_RE.search(paragraph):
            findings.append(
                _finding(
                    check_id="chinese_punctuation",
                    chapter=chapter,
                    paragraph_index=index,
                    snippet=paragraph,
                    found="".join(sorted(set(CHINESE_PUNCT_RE.findall(paragraph)))),
                    expected="English punctuation",
                    message="Translated text contains Chinese punctuation.",
                    suggested_action="Normalize punctuation to English.",
                    auto_repairable=True,
                )
            )

    translated_lower = translated_text.lower()
    for entry in glossary.entries:
        for variant in entry.blocked_variants:
            if variant and variant.lower() in translated_lower:
                findings.append(
                    _finding(
                        check_id="blocked_glossary_variant",
                        chapter=chapter,
                        found=variant,
                        expected=entry.target,
                        paragraph_index=find_paragraph_index(translated_text, variant),
                        message="Blocked glossary variant appears in translated text.",
                        suggested_action="Replace blocked variant with canonical glossary term.",
                        auto_repairable=True,
                    )
                )

    for entry in glossary.entries:
        if entry.source in source_text and entry.target.lower() not in translated_lower:
            aliases = [candidate for candidate in entry.candidates if candidate != entry.target]
            aliases.extend(
                _cross_glossary_aliases(
                    source_text=source_text,
                    glossary=glossary,
                    entry_source=entry.source,
                    entry_target=entry.target,
                )
            )
            observed_alias = _find_observed_glossary_alias(
                translated_text,
                entry.source,
                aliases,
            )
            alias_conflict = bool(
                observed_alias
                and _observed_alias_conflicts_with_present_source(
                    observed_alias=observed_alias[0],
                    source_text=source_text,
                    glossary=glossary,
                    entry_source=entry.source,
                )
            )
            source_paragraph_index = find_paragraph_index(source_text, entry.source)
            found = observed_alias[0] if observed_alias else entry.source
            findings.append(
                _finding(
                    check_id="glossary_required",
                    chapter=chapter,
                    found=found,
                    expected=entry.target,
                    paragraph_index=observed_alias[1] if observed_alias else source_paragraph_index,
                    snippet=found,
                    message="Source term appears in source but canonical translation is missing.",
                    suggested_action="Use the canonical glossary translation.",
                    auto_repairable=bool(observed_alias) and not alias_conflict,
                )
            )

    source_panels = panel_count(source_text)
    translated_panels = panel_count(translated_text)
    if source_panels != translated_panels:
        findings.append(
            _finding(
                check_id="system_panel_count",
                chapter=chapter,
                found=str(translated_panels),
                expected=str(source_panels),
                message="System/panel count differs between source and translation.",
                suggested_action="Preserve bracketed system or simulator panels.",
                auto_repairable=True,
            )
        )

    leak_match = PROMPT_LEAK_RE.search(translated_text)
    if leak_match:
        findings.append(
            _finding(
                check_id="prompt_leakage",
                chapter=chapter,
                found=leak_match.group(0),
                severity="error",
                message="Translated text contains prompt/model leakage.",
                suggested_action="Remove prompt leakage.",
                auto_repairable=True,
            )
        )

    summary = summarize(findings)
    score = weighted_score(findings)
    return QAReport(
        run_id=run_id,
        story_slug=story_slug,
        chapter=chapter,
        report_type="translation",
        findings=findings,
        summary=summary,
        panel_count=translated_panels,
        score=score,
    )
