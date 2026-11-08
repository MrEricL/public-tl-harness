from __future__ import annotations

import re
from pathlib import Path

from .models import (
    CandidateScore,
    ComplianceCandidateScore,
    EnsembleDecision,
    GlossaryParseResult,
    JudgeVote,
    QAFinding,
    RepairPatch,
    StoryConfig,
    TranslationCandidate,
)
from .qa import CHINESE_RE, PROMPT_LEAK_RE
from .text import chapter_display_label, find_paragraph_index, split_paragraphs


PUNCTUATION_MAP = str.maketrans(
    {
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "；": ";",
        "：": ":",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "、": ",",
        "（": "(",
        "）": ")",
        "《": "<",
        "》": ">",
        "【": "[",
        "】": "]",
    }
)

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

SOURCE_TITLE_OVERRIDES = {
    "模拟器启动": "The Simulator Starts",
}


def normalize_english_punctuation(text: str) -> str:
    normalized = text.translate(PUNCTUATION_MAP)
    return re.sub(r":(?=\S)", ": ", normalized)


def _replace_case_insensitive(text: str, old: str, new: str) -> str:
    return re.sub(re.escape(old), new, text, flags=re.IGNORECASE)


def apply_glossary_transform(text: str, glossary: GlossaryParseResult) -> str:
    transformed = text
    for entry in glossary.entries:
        for variant in sorted(entry.blocked_variants, key=len, reverse=True):
            transformed = _replace_case_insensitive(transformed, variant, entry.target)
        transformed = transformed.replace(entry.source, entry.target)
    return normalize_english_punctuation(transformed)


def _normalize_heading(found: str | None, chapter: str, glossary: GlossaryParseResult) -> str:
    chapter_label = chapter_display_label(chapter)
    heading = found or ""
    subtitle = ""
    match = re.match(r"^\s*Chapter\s+([A-Za-z]+|\d+)(?::\s*(.+))?\s*$", heading, flags=re.IGNORECASE)
    if match and match.group(2):
        subtitle = apply_glossary_transform(match.group(2).strip(), glossary).strip()
    if subtitle:
        return f"Chapter {chapter_label}: {subtitle}"
    return f"Chapter {chapter_label}"


def _source_title_override(source_text: str) -> str | None:
    source_title = next((line.strip() for line in source_text.splitlines() if line.strip()), "")
    for source_phrase, english_title in SOURCE_TITLE_OVERRIDES.items():
        if source_phrase in source_title:
            return english_title
    return None


def _capitalize_panel(text: str) -> str:
    stripped = text.strip()
    return stripped[:1].upper() + stripped[1:] if stripped else stripped


class OfflineTranslationProvider:
    provider_name = "offline"
    model_name = "offline-fixture-v1"

    def translate(
        self,
        source_text: str,
        *,
        story: StoryConfig,
        glossary: GlossaryParseResult,
        mode: str,
    ) -> str:
        chapter = story.chapter_ids[0] if story.chapter_ids else "0001"
        baseline = f"Chapter {chapter_display_label(chapter)}\n\n" + source_text
        if story.paths.expected_dir:
            path = story.paths.expected_dir / "dirty_translation.txt"
            if path.exists():
                baseline = path.read_text(encoding="utf-8")
        elif story.paths.baseline_dir:
            path = story.paths.baseline_dir / f"{chapter}.txt"
            if path.exists():
                baseline = path.read_text(encoding="utf-8")
        if mode == "glossary":
            return apply_glossary_transform(baseline, glossary)
        return baseline


class OfflineJudgeProvider:
    provider_name = "offline"
    model_name = "offline-rubric-v1"

    def judge(
        self,
        *,
        source_text: str,
        candidates: list[TranslationCandidate],
        glossary: GlossaryParseResult,
        seed: int,
    ) -> EnsembleDecision:
        scores: dict[str, CandidateScore] = {}
        aggregate: dict[str, float] = {}
        blocked = [variant.lower() for variant in glossary.blocked_variants]
        for candidate in candidates:
            text = candidate.text
            text_lower = text.lower()
            residue_free = 2 if CHINESE_RE.search(text) else 10
            prompt_safety = 2 if PROMPT_LEAK_RE.search(text) else 10
            glossary_consistency = 10
            for variant in blocked:
                if variant and variant in text_lower:
                    glossary_consistency -= 4
            if "remaining" in text_lower and "remaining uses" not in text_lower:
                glossary_consistency -= 3
            stripped = text.strip()
            panel_score = 10 if stripped.startswith("[") and stripped.endswith("]") else 3
            readability = 8 if 4 <= len(text.split()) <= 40 else 6
            compliance = ComplianceCandidateScore(
                residue_free=max(1, residue_free),
                glossary_consistency=max(1, glossary_consistency),
                panel_preservation=max(1, panel_score),
                prompt_safety=max(1, prompt_safety),
                readability=max(1, readability),
                notes="offline compliance rubric only; no semantic quality claim",
            )
            aggregate_score = (
                0.25 * compliance.residue_free
                + 0.25 * compliance.glossary_consistency
                + 0.25 * compliance.panel_preservation
                + 0.15 * compliance.prompt_safety
                + 0.10 * compliance.readability
            )
            score = CandidateScore(
                compliance=compliance,
                quality=None,
                aggregate=aggregate_score,
                notes="offline compliance score; quality scoring is only available in live/replay judge mode",
            )
            scores[candidate.candidate_id] = score
            aggregate[candidate.candidate_id] = aggregate_score
        winner = max(aggregate, key=aggregate.get)
        votes = [
            JudgeVote(
                judge_id="offline_judge_1",
                winner=winner,
                scores=scores,
                rationale="Selected by deterministic weighted rubric.",
            )
        ]
        return EnsembleDecision(
            selected_candidate_id=winner,
            votes=votes,
            aggregate_scores=aggregate,
            disagreement=0.0,
            requires_human_review=False,
        )


class OfflineRepairProvider:
    provider_name = "offline"
    model_name = "offline-patch-v1"

    def propose_patch(
        self,
        *,
        chapter: str,
        source_text: str,
        translation_text: str,
        finding: QAFinding,
        glossary: GlossaryParseResult,
        ensemble_decision: EnsembleDecision | None = None,
        candidates: list[TranslationCandidate] | None = None,
    ) -> RepairPatch | None:
        if finding.check_id == "system_panel_count":
            selected_text = None
            if ensemble_decision and candidates:
                selected = next(
                    (candidate for candidate in candidates if candidate.candidate_id == ensemble_decision.selected_candidate_id),
                    None,
                )
                selected_text = selected.text if selected else None
            for paragraph in split_paragraphs(translation_text):
                if (
                    ("remaining uses" in paragraph.lower() or "remaining" in paragraph.lower())
                    and not paragraph.strip().startswith("[")
                ):
                    return RepairPatch(
                        patch_id="patch_system_panel_count",
                        patch_type="replace_paragraph",
                        chapter=chapter,
                        paragraph_index=find_paragraph_index(translation_text, paragraph),
                        old_text=paragraph,
                        new_text=selected_text or f"[{_capitalize_panel(paragraph)}]",
                        reason="Restore bracketed system panel shape.",
                        source_finding_check_id=finding.check_id,
                    )
        if finding.check_id == "heading_format":
            paragraphs = split_paragraphs(translation_text)
            index = finding.location.paragraph_index or 0
            if 0 <= index < len(paragraphs):
                source_title = _source_title_override(source_text)
                new_text = (
                    f"Chapter {chapter_display_label(chapter)}: {source_title}"
                    if source_title
                    else _normalize_heading(finding.found or paragraphs[index], chapter, glossary)
                )
                return RepairPatch(
                    patch_id="patch_heading_format",
                    patch_type="replace_paragraph",
                    chapter=chapter,
                    paragraph_index=index,
                    old_text=paragraphs[index],
                    new_text=new_text,
                    reason="Normalize heading with explicit source-title mapping when available.",
                    source_finding_check_id=finding.check_id,
                )
        if finding.check_id == "chinese_punctuation" and finding.location.paragraph_index is not None:
            paragraphs = split_paragraphs(translation_text)
            index = finding.location.paragraph_index
            if 0 <= index < len(paragraphs):
                return RepairPatch(
                    patch_id=f"patch_chinese_punctuation_{index}",
                    patch_type="replace_paragraph",
                    chapter=chapter,
                    paragraph_index=index,
                    old_text=paragraphs[index],
                    new_text=normalize_english_punctuation(paragraphs[index]),
                    reason="Map Chinese punctuation to English punctuation.",
                    source_finding_check_id=finding.check_id,
                )
        if finding.check_id == "residual_chinese" and finding.expected and finding.location.paragraph_index is not None:
            paragraphs = split_paragraphs(translation_text)
            index = finding.location.paragraph_index
            if 0 <= index < len(paragraphs):
                repaired = paragraphs[index]
                for entry in glossary.entries:
                    if entry.source in repaired:
                        repaired = repaired.replace(entry.source, entry.target)
                repaired = normalize_english_punctuation(repaired)
                if repaired != paragraphs[index]:
                    return RepairPatch(
                        patch_id=f"patch_residual_chinese_{index}",
                        patch_type="replace_paragraph",
                        chapter=chapter,
                        paragraph_index=index,
                        old_text=paragraphs[index],
                        new_text=repaired,
                        reason="Replace known residual source term with glossary canon.",
                        source_finding_check_id=finding.check_id,
                    )
        if finding.check_id == "prompt_leakage" and finding.found:
            return RepairPatch(
                patch_id="patch_prompt_leakage",
                patch_type="replace_span",
                chapter=chapter,
                old_text=finding.found,
                new_text="",
                reason="Remove exact prompt leakage span.",
                source_finding_check_id=finding.check_id,
            )
        if finding.check_id == "blocked_glossary_variant" and finding.found and finding.expected:
            return RepairPatch(
                patch_id="patch_blocked_glossary_variant",
                patch_type="replace_span",
                chapter=chapter,
                old_text=finding.found,
                new_text=finding.expected,
                reason="Replace blocked variant with canonical glossary term.",
                source_finding_check_id=finding.check_id,
            )
        if finding.check_id == "glossary_required" and finding.found and finding.expected and not CHINESE_RE.search(finding.found):
            return RepairPatch(
                patch_id="patch_glossary_required",
                patch_type="replace_span",
                chapter=chapter,
                old_text=finding.found,
                new_text=finding.expected,
                reason="Replace known glossary alias with canonical glossary term.",
                source_finding_check_id=finding.check_id,
            )
        if ensemble_decision and candidates:
            selected = next(
                (candidate for candidate in candidates if candidate.candidate_id == ensemble_decision.selected_candidate_id),
                None,
            )
            if selected and finding.location.paragraph_index is not None:
                paragraphs = split_paragraphs(translation_text)
                index = finding.location.paragraph_index
                if 0 <= index < len(paragraphs):
                    return RepairPatch(
                        patch_id=f"patch_{finding.check_id}_{index}",
                        patch_type="replace_paragraph",
                        chapter=chapter,
                        paragraph_index=index,
                        old_text=paragraphs[index],
                        new_text=selected.text,
                        reason=f"Ensemble selected {selected.candidate_id}.",
                        source_finding_check_id=finding.check_id,
                    )
        return None


def expected_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")
