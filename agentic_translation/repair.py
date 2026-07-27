from __future__ import annotations

import re
from typing import Literal

from .models import QAFinding, QAReport, RepairPatch
from .qa import weighted_score
from .text import join_paragraphs, split_paragraphs


RepairStrategy = Literal["rule", "candidate_selection", "human_review", "none"]

REPAIR_PRIORITY = [
    "prompt_leakage",
    "residual_chinese",
    "system_panel_count",
    "blocked_glossary_variant",
    "glossary_required",
    "heading_format",
    "chinese_punctuation",
]


def route_repair_strategy(finding: QAFinding, *, provider_mode: str) -> RepairStrategy:
    if finding.check_id == "system_panel_count":
        return "candidate_selection"
    if finding.check_id in {"blocked_glossary_variant", "chinese_punctuation", "heading_format", "prompt_leakage"}:
        return "rule"
    if finding.check_id == "glossary_required":
        if finding.auto_repairable and finding.found and finding.expected:
            return "rule"
        return "human_review"
    if finding.check_id == "residual_chinese":
        if provider_mode in {"live", "replay"} and finding.expected:
            return "candidate_selection"
        if finding.expected:
            return "rule"
        return "human_review"
    return "human_review"


def prioritized_repairable_findings(findings: list[QAFinding]) -> list[QAFinding]:
    priority = {check_id: index for index, check_id in enumerate(REPAIR_PRIORITY)}
    return sorted(
        [finding for finding in findings if finding.auto_repairable],
        key=lambda finding: priority.get(finding.check_id, 999),
    )


def apply_patch(text: str, patch: RepairPatch) -> str:
    if patch.patch_type == "replace_span":
        if not patch.old_text:
            raise ValueError("replace_span patch requires non-empty old_text")
        if re.match(r"^[A-Za-z0-9].*[A-Za-z0-9]$", patch.old_text):
            pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(patch.old_text)}(?![A-Za-z0-9])")
            if not pattern.search(text):
                raise ValueError(f"Patch old_text not found: {patch.old_text}")
            return pattern.sub(patch.new_text, text)
        if patch.old_text not in text:
            raise ValueError(f"Patch old_text not found: {patch.old_text}")
        # Demo policy: replace all exact occurrences of the stale span.
        return text.replace(patch.old_text, patch.new_text)
    paragraphs = split_paragraphs(text)
    if patch.paragraph_index is None:
        raise ValueError("replace_paragraph patch requires paragraph_index")
    if patch.paragraph_index < 0 or patch.paragraph_index >= len(paragraphs):
        raise ValueError(f"paragraph_index out of range: {patch.paragraph_index}")
    if paragraphs[patch.paragraph_index] != patch.old_text:
        raise ValueError("Patch old_text does not match target paragraph")
    paragraphs[patch.paragraph_index] = patch.new_text
    return join_paragraphs(paragraphs)


def validate_patch_improves_qa(*, before_report: QAReport, after_report: QAReport) -> bool:
    if weighted_score(after_report) <= weighted_score(before_report):
        return False
    before_checks = before_report.summary.by_check
    after_checks = after_report.summary.by_check
    for forbidden in ["residual_chinese", "prompt_leakage", "system_panel_count"]:
        if after_checks.get(forbidden, 0) > before_checks.get(forbidden, 0):
            return False
    return True
