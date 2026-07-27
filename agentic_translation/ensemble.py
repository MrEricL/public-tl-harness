from __future__ import annotations

import random

from .models import EnsembleDecision, GlossaryParseResult, QAFinding, TranslationCandidate
from .providers_offline import OfflineJudgeProvider
from .qa import CHINESE_RE
from .text import split_paragraphs


def _sentence_case(text: str) -> str:
    stripped = text.strip()
    return stripped[:1].upper() + stripped[1:] if stripped else stripped


def blind_and_shuffle(candidates: list[TranslationCandidate], seed: int) -> list[TranslationCandidate]:
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    return [
        candidate.model_copy(update={"candidate_id": f"candidate_{chr(97 + index)}"})
        for index, candidate in enumerate(shuffled)
    ]


def generate_repair_candidates(
    *,
    translated_text: str,
    finding: QAFinding,
    glossary: GlossaryParseResult,
) -> list[TranslationCandidate]:
    paragraphs = split_paragraphs(translated_text)
    if finding.check_id == "system_panel_count":
        target = next((p for p in paragraphs if "remaining" in p.lower() and not p.startswith("[")), "")
        repaired = f"[{_sentence_case(target)}]" if target else "[Remaining uses: 3]"
        alternate = repaired.replace("Remaining uses", "Remaining attempts").replace("remaining uses", "remaining attempts")
        return [
            TranslationCandidate(
                candidate_id="raw_a",
                text=target or translated_text,
                source="literal worker / unchanged",
                notes="Keeps the observed translation unchanged for the blind judge baseline.",
            ),
            TranslationCandidate(
                candidate_id="raw_b",
                text=repaired,
                source="canon-strict worker / panel_repair",
                notes="Restores the bracketed system-panel shape and preserves glossary wording.",
            ),
            TranslationCandidate(
                candidate_id="raw_c",
                text=alternate,
                source="fluent worker / style_repair",
                notes="Offers a more idiomatic panel variant so the judge has a real alternative.",
            ),
        ]
    if finding.found and finding.expected:
        target = paragraphs[finding.location.paragraph_index] if finding.location.paragraph_index is not None else translated_text
        repaired = target.replace(finding.found, finding.expected)
        return [
            TranslationCandidate(candidate_id="raw_a", text=target, source="unchanged"),
            TranslationCandidate(candidate_id="raw_b", text=repaired, source="span_repair"),
            TranslationCandidate(candidate_id="raw_c", text=repaired, source="style_repair"),
        ]
    if finding.check_id == "residual_chinese":
        target = paragraphs[finding.location.paragraph_index or 0]
        replacement = finding.expected or "translated text"
        repaired = target
        if finding.found and finding.found in repaired:
            repaired = repaired.replace(finding.found, replacement)
        else:
            repaired = CHINESE_RE.sub("", repaired).strip() or replacement
        return [
            TranslationCandidate(candidate_id="raw_a", text=target, source="unchanged"),
            TranslationCandidate(candidate_id="raw_b", text=repaired, source="residue_repair"),
        ]
    return [TranslationCandidate(candidate_id="raw_a", text=translated_text, source="unchanged")]


def judge_candidates(
    *,
    candidates: list[TranslationCandidate],
    source_snippet: str,
    finding: QAFinding,
    glossary: GlossaryParseResult,
    seed: int = 7,
) -> EnsembleDecision:
    blinded = blind_and_shuffle(candidates, seed)
    return OfflineJudgeProvider().judge(
        source_text=source_snippet,
        candidates=blinded,
        glossary=glossary,
        seed=seed,
    )
