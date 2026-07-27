from __future__ import annotations

from pathlib import Path

from .providers import JudgeProvider, RepairProvider, TranslationProvider
from .providers_llm import LLMJudgeProvider, LLMRepairProvider, LLMTranslationProvider, is_openai_compatible_provider
from .providers_offline import OfflineJudgeProvider, OfflineRepairProvider, OfflineTranslationProvider


def get_translation_provider(
    name: str,
    *,
    provider_mode: str = "offline",
    cache_dir: Path | None = None,
    record_cache: bool = False,
    model_name: str | None = None,
) -> TranslationProvider:
    if name == "offline":
        return OfflineTranslationProvider()
    if is_openai_compatible_provider(name):
        return LLMTranslationProvider(
            provider_mode=provider_mode,
            cache_dir=cache_dir,
            record_cache=record_cache,
            model_name=model_name,
            provider_name=name,
        )
    raise ValueError(f"Unsupported translation provider: {name}")


def get_judge_provider(
    name: str,
    *,
    provider_mode: str = "offline",
    cache_dir: Path | None = None,
    record_cache: bool = False,
    model_name: str | None = None,
) -> JudgeProvider:
    if name == "offline":
        return OfflineJudgeProvider()
    if is_openai_compatible_provider(name):
        return LLMJudgeProvider(
            provider_mode=provider_mode,
            cache_dir=cache_dir,
            record_cache=record_cache,
            model_name=model_name,
            provider_name=name,
        )
    raise ValueError(f"Unsupported judge provider: {name}")


def get_repair_provider(
    name: str,
    *,
    provider_mode: str = "offline",
    cache_dir: Path | None = None,
    record_cache: bool = False,
    model_name: str | None = None,
) -> RepairProvider:
    if name == "offline":
        return OfflineRepairProvider()
    if is_openai_compatible_provider(name):
        return LLMRepairProvider(
            provider_mode=provider_mode,
            cache_dir=cache_dir,
            record_cache=record_cache,
            model_name=model_name,
            provider_name=name,
        )
    raise ValueError(f"Unsupported repair provider: {name}")
