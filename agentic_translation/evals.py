from __future__ import annotations

from .models import EvalMetrics, QAReport


def metrics_from_qa(mode: str, report: QAReport) -> EvalMetrics:
    by_check = report.summary.by_check
    glossary_violations = by_check.get("blocked_glossary_variant", 0) + by_check.get("glossary_required", 0)
    return EvalMetrics(
        mode=mode,
        residual_chinese=by_check.get("residual_chinese", 0),
        chinese_punctuation=by_check.get("chinese_punctuation", 0),
        glossary_violations=glossary_violations,
        panel_mismatches=by_check.get("system_panel_count", 0),
        prompt_leakage=by_check.get("prompt_leakage", 0),
        total_findings=report.summary.total_findings,
        score=report.score,
    )

