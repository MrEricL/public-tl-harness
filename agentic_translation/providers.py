from __future__ import annotations

from typing import Literal, Protocol

from .models import EnsembleDecision, GlossaryParseResult, QAFinding, RepairPatch, StoryConfig, TranslationCandidate


class TranslationProvider(Protocol):
    provider_name: str
    model_name: str

    def translate(
        self,
        source_text: str,
        *,
        story: StoryConfig,
        glossary: GlossaryParseResult,
        mode: Literal["baseline", "glossary"],
    ) -> str:
        ...


class JudgeProvider(Protocol):
    provider_name: str
    model_name: str

    def judge(
        self,
        *,
        source_text: str,
        candidates: list[TranslationCandidate],
        glossary: GlossaryParseResult,
        seed: int,
    ) -> EnsembleDecision:
        ...


class RepairProvider(Protocol):
    provider_name: str
    model_name: str

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
        ...

