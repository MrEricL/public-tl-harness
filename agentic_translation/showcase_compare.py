"""A small, reproducible comparison harness for the translation showcase.

The comparison deliberately keeps the evaluator outside the model-facing
fixture.  ``scenario.json`` contains the same supplied drafts and scripted
tool calls that the showcase runner may use in offline mode.  The expected
labels live in ``evaluator.json`` and never enter model requests.
This makes the benchmark useful for a resume project without
pretending that a scripted run proves literary quality.

Three arms are measured with the same chapter inputs and limits:

* ``deterministic``: existing deterministic cleanup and QA;
* ``single``: one bounded coordinator with no delegated children;
* ``specialists``: the coordinator plus fixed terminology/fidelity reviews.

The runner is imported lazily because this module is also useful as a fixture
auditor while the main showcase runtime is being developed.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import shutil
import statistics
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .glossary import load_glossary
from .models import GlossaryParseResult
from .qa import run_translation_qa


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "samples" / "showcase_compare"
COMPARISON_SCHEMA_VERSION = "translation-showcase-comparison.v1"
BLIND_PACKET_SCHEMA_VERSION = "translation-showcase-blind.v1"
STRATEGIES: tuple[str, ...] = ("deterministic", "single", "specialists")
SURFACE_LIMITS = {
    "max_steps": 12,
    "max_patch_attempts": 3,
    "max_delegation_rounds": 2,
    "max_child_steps": 4,
    "max_workers": 2,
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _load_runner() -> Any:
    """Import the main showcase runner only when a comparison is requested."""

    return importlib.import_module("agentic_translation.showcase").run_showcase


def _case_ids(story: Mapping[str, Any], evaluator: Mapping[str, Any]) -> list[str]:
    ids = story.get("chapter_ids")
    if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes)):
        return [str(item) for item in ids]
    cases = evaluator.get("cases")
    if isinstance(cases, Sequence) and not isinstance(cases, (str, bytes)):
        result: list[str] = []
        for item in cases:
            if isinstance(item, Mapping) and item.get("id") is not None:
                result.append(str(item["id"]))
        return result
    raise ValueError("showcase comparison fixture must define chapter_ids")


def _case_rows(evaluator: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_cases = evaluator.get("cases", [])
    if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes)):
        raise ValueError("evaluator.json cases must be a list")
    rows: list[dict[str, Any]] = []
    for raw in raw_cases:
        if not isinstance(raw, Mapping):
            raise ValueError("Each evaluator case must be an object")
        row = dict(raw)
        if not row.get("id") or row.get("category") not in {"mechanical", "terminology", "semantic"}:
            raise ValueError("Each evaluator case needs an id and a supported category")
        rows.append(row)
    return rows


def _fixture_inputs() -> tuple[dict[str, Any], dict[str, Any], list[str], list[dict[str, Any]]]:
    story = _read_yaml_like(FIXTURE_DIR / "story.yaml")
    evaluator = _read_json(FIXTURE_DIR / "evaluator.json")
    ids = _case_ids(story, evaluator)
    cases = _case_rows(evaluator)
    case_map = {str(row["id"]): row for row in cases}
    missing = [chapter for chapter in ids if chapter not in case_map]
    if missing:
        raise ValueError(f"evaluator.json is missing cases: {', '.join(missing)}")
    if len(ids) != 12:
        raise ValueError(f"comparison fixture must contain exactly 12 cases, found {len(ids)}")
    return story, evaluator, ids, cases


def _read_yaml_like(path: Path) -> dict[str, Any]:
    """Use the project's YAML dependency without making comparison import it eagerly."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - project dependency
        raise RuntimeError("PyYAML is required to read the comparison story") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML object in {path}")
    return value


def _child_actions(*, role: str, summary: str, findings: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Return a small, evidence-first scripted child review."""

    return [
        {"tool": "read_paragraphs", "document": "source", "start": 0, "count": 3},
        {"tool": "read_paragraphs", "document": "translation", "start": 0, "count": 3},
        {
            "tool": "complete_review",
            "summary": summary,
            "findings": findings or [],
            "proposed_edits": [],
            "term_suggestions": [],
        },
    ]


def _stage_case(
    *,
    root_story: Mapping[str, Any],
    scenario: Mapping[str, Any],
    chapter: str,
    strategy: str,
    root: Path,
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    """Stage exactly one chapter and its draft for an independent run.

    The temporary story contains only source, glossary, style, scenario, and
    one supplied draft.  ``evaluator.json`` is deliberately never copied.
    """

    temp = tempfile.TemporaryDirectory(prefix=f"showcase-{chapter}-{strategy}-", dir=str(root))
    stage = Path(temp.name)
    (stage / "source").mkdir()
    (stage / "drafts").mkdir()
    (stage / "terms").mkdir()
    shutil.copyfile(FIXTURE_DIR / "source" / f"{chapter}.txt", stage / "source" / f"{chapter}.txt")
    shutil.copyfile(FIXTURE_DIR / "drafts" / f"{chapter}.txt", stage / "drafts" / f"{chapter}.txt")
    shutil.copyfile(FIXTURE_DIR / "terms/master_glossary.txt", stage / "terms/master_glossary.txt")
    shutil.copyfile(FIXTURE_DIR / "style_guide.md", stage / "style_guide.md")

    strategy_data = scenario.get("strategies", {}).get(strategy, {})
    if not isinstance(strategy_data, Mapping):
        strategy_data = {}
    actions_by_chapter = strategy_data.get("actions")
    if not isinstance(actions_by_chapter, Mapping):
        actions_by_chapter = scenario.get("actions", {})
    actions = actions_by_chapter.get(chapter, []) if isinstance(actions_by_chapter, Mapping) else []
    if not isinstance(actions, list):
        actions = []
    reviews_by_chapter = strategy_data.get("reviews")
    if not isinstance(reviews_by_chapter, Mapping) or not reviews_by_chapter:
        reviews_by_chapter = scenario.get("reviews", {}).get(chapter, {})
    reviews = dict(reviews_by_chapter) if isinstance(reviews_by_chapter, Mapping) else {}
    # A fresh fidelity review is required after any specialist-arm mutation.
    # Add only clean scripted evidence where the authored scenario has no
    # explicit response.  Semantic cases retain their blocking response.
    for step, action in enumerate(actions, start=1):
        if not isinstance(action, Mapping) or action.get("tool") != "delegate_review":
            continue
        step_key = str(step)
        step_reviews = reviews.get(step_key, {})
        if not isinstance(step_reviews, Mapping):
            step_reviews = {}
        completed = dict(step_reviews)
        roles = action.get("specialists", [])
        if not isinstance(roles, list):
            roles = []
        for role in roles:
            if role in completed:
                continue
            if role == "terminology":
                completed[role] = _child_actions(role=role, summary="No terminology change is required after the reviewed cleanup.")
            elif role == "fidelity":
                completed[role] = _child_actions(role=role, summary="The cleaned draft preserves the source actions and relationships.")
        reviews[step_key] = completed

    staged_story = dict(root_story)
    staged_story["slug"] = f"translation_showcase_compare_{chapter}_{strategy}"
    staged_story["chapter_ids"] = [chapter]
    staged_paths = dict(staged_story.get("paths", {}))
    staged_paths.update(
        source_dir="source",
        glossary_path="terms/master_glossary.txt",
        runs_dir="runs",
    )
    staged_paths.pop("expected_dir", None)
    staged_paths.pop("baseline_dir", None)
    staged_story["paths"] = staged_paths
    staged_story["translation"] = {
        **dict(staged_story.get("translation", {})),
        "provider": "offline",
        "model": "showcase-compare-fixture-v1",
    }
    (stage / "story.yaml").write_text(_dump_yaml(staged_story), encoding="utf-8")
    translations = scenario.get("translations", {})
    translation = translations.get(chapter, "") if isinstance(translations, Mapping) else ""
    staged_scenario = {
        "translations": {chapter: translation},
        "actions": {chapter: actions},
        "reviews": {chapter: reviews},
    }
    (stage / "scenario.json").write_text(
        json.dumps(staged_scenario, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return temp, stage / "story.yaml"


def _dump_yaml(value: Mapping[str, Any]) -> str:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - project dependency
        raise RuntimeError("PyYAML is required to stage the comparison story") from exc
    return yaml.safe_dump(dict(value), allow_unicode=True, sort_keys=False)


def _call_runner(
    runner: Any,
    *,
    story_path: Path,
    draft_dir: Path,
    output_dir: Path,
    strategy: str,
    provider_mode: str,
    profile: str,
    model: str | None,
    auto_approve: bool,
) -> Any:
    """Call the agreed runner API, preserving compatibility with early slices."""

    kwargs: dict[str, Any] = {
        "provider_mode": provider_mode,
        "profile": profile,
        "strategy": strategy,
        "auto_approve": auto_approve,
        "draft_dir": draft_dir,
    }
    if model is not None:
        kwargs["model"] = model
    return runner(story_path, output_dir, **kwargs)


def _result_manifest(result: Any) -> tuple[Path, dict[str, Any]]:
    """Read the fixed ``ShowcaseResult(run_dir, manifest)`` contract."""

    run_dir = Path(result.run_dir)
    raw = result.manifest
    if isinstance(raw, Mapping):
        manifest = dict(raw)
    else:
        dump = getattr(raw, "model_dump", None)
        if not callable(dump):
            raise TypeError("ShowcaseResult.manifest must be a mapping or Pydantic model")
        manifest = dump(mode="json")
    return run_dir, manifest


def _find_chapter(manifest: Mapping[str, Any], chapter: str) -> dict[str, Any]:
    chapters = manifest.get("chapters", {})
    if not isinstance(chapters, Mapping):
        raise ValueError("showcase manifest chapters must be a mapping")
    item = chapters.get(chapter)
    if not isinstance(item, Mapping):
        raise ValueError(f"showcase manifest has no chapter state for {chapter}")
    return dict(item)


def _candidate_text(run_dir: Path, chapter: str, chapter_manifest: Mapping[str, Any]) -> str:
    path = run_dir / "chapters" / chapter / "translated_final.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _first_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float) and math.isfinite(value):
        return max(int(value), 0)
    return None


def _aggregate_usage(value: Any) -> dict[str, float | int | None]:
    calls = value if isinstance(value, list) else []
    totals: dict[str, float | int | None] = {
        "provider_calls": len(calls),
        "input_tokens": None,
        "output_tokens": None,
        "cached_input_tokens": None,
        "total_tokens": None,
        "api_elapsed_ms": None,
    }
    for target, aliases in {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "cached_input_tokens": ("cached_input_tokens", "cache_read_input_tokens"),
        "total_tokens": ("total_tokens", "tokens"),
        "api_elapsed_ms": ("elapsed_ms", "recorded_elapsed_ms"),
    }.items():
        values: list[float] = []
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            for alias in aliases:
                number = call.get(alias)
                if isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(float(number)):
                    values.append(float(number))
                    break
        if values:
            total = sum(values)
            totals[target] = int(total) if all(value.is_integer() for value in values) else total
    return totals


def _qa_for_case(
    *,
    run_id: str,
    chapter: str,
    source_text: str,
    final_text: str,
    glossary: GlossaryParseResult,
) -> dict[str, Any]:
    if not final_text:
        return {
            "total_findings": None,
            "surface_clear": None,
            "by_check": {},
            "score": None,
        }
    report = run_translation_qa(
        run_id=run_id,
        story_slug="translation_showcase_compare",
        chapter=chapter,
        source_text=source_text,
        translated_text=final_text,
        glossary=glossary,
    )
    return {
        "total_findings": report.summary.total_findings,
        "surface_clear": report.summary.total_findings == 0,
        "by_check": dict(report.summary.by_check),
        "score": report.score,
    }


def _status(chapter_manifest: Mapping[str, Any]) -> str:
    value = chapter_manifest.get("status")
    return value if isinstance(value, str) and value else "unknown"


def _explicit_fidelity_finding(run_dir: Path, chapter: str) -> bool:
    path = run_dir / "chapters" / chapter / "session_snapshot.json"
    if not path.exists():
        return False
    try:
        snapshot = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    reviews = snapshot.get("specialist_reviews", [])
    if not isinstance(reviews, list):
        return False
    for review in reviews:
        if not isinstance(review, Mapping) or str(review.get("role", "")).lower() != "fidelity":
            continue
        findings = review.get("findings", [])
        if isinstance(findings, list) and findings:
            return True
    return False


def _reviewed_semantics(*, run_dir: Path, chapter: str, chapter_manifest: Mapping[str, Any]) -> bool:
    final_status = str(chapter_manifest.get("final_status", "")).lower()
    status = _status(chapter_manifest).lower()
    return final_status == "escalated" or (
        status == "review_required" and _explicit_fidelity_finding(run_dir, chapter)
    )


def _step_count(chapter_manifest: Mapping[str, Any]) -> int:
    value = chapter_manifest.get("steps", 0)
    number = _first_int(value)
    return number if number is not None else 0


def _child_count(chapter_manifest: Mapping[str, Any]) -> int:
    value = chapter_manifest.get("child_calls", 0)
    number = _first_int(value)
    return number if number is not None else 0


def _expected_outcome(case: Mapping[str, Any], strategy: str) -> str:
    expected = case.get("expected", {})
    if isinstance(expected, Mapping):
        value = expected.get(strategy)
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping) and isinstance(value.get("outcome"), str):
            return str(value["outcome"])
    category = case.get("category")
    if category == "semantic":
        return "review_required" if strategy == "specialists" else "unreviewed_semantic_risk"
    return "surface_clear"


def _observed_outcome(*, category: str, strategy: str, surface_clear: bool | None, reviewed: bool, status: str) -> str:
    if category == "semantic":
        return "review_required" if reviewed else "unreviewed_semantic_risk"
    return "surface_clear" if surface_clear else "surface_findings_remaining"


def _record_case(
    *,
    manifest: Mapping[str, Any],
    run_dir: Path,
    chapter: str,
    case: Mapping[str, Any],
    source_text: str,
    glossary: GlossaryParseResult,
    strategy: str,
    trial: int,
    provider_mode: str = "offline",
) -> dict[str, Any]:
    chapter_manifest = _find_chapter(manifest, chapter)
    final_text = _candidate_text(run_dir, chapter, chapter_manifest)
    qa = _qa_for_case(
        run_id=str(manifest.get("run_id", f"comparison-{trial}-{strategy}")),
        chapter=chapter,
        source_text=source_text,
        final_text=final_text,
        glossary=glossary,
    )
    status = _status(chapter_manifest)
    reviewed = _reviewed_semantics(run_dir=run_dir, chapter=chapter, chapter_manifest=chapter_manifest)
    usage = _aggregate_usage(chapter_manifest.get("provider_calls", []))
    elapsed_ms = chapter_manifest.get("elapsed_ms")
    if not isinstance(elapsed_ms, (int, float)) or isinstance(elapsed_ms, bool):
        elapsed_ms = None
    category = str(case["category"])
    surface_clear = qa["surface_clear"]
    observed = _observed_outcome(
        category=category,
        strategy=strategy,
        surface_clear=surface_clear,
        reviewed=reviewed,
        status=status,
    )
    # Authored fixture labels are valid contract evidence only for the
    # scripted offline arm.  A live model is free to make a different
    # legitimate decision, so do not turn the fixture expectation into a
    # hidden quality score.
    expected = _expected_outcome(case, strategy) if provider_mode == "offline" else None
    fixture_expected_match = observed == expected if expected is not None else None
    semantic_handling_success = reviewed if category == "semantic" else surface_clear is True
    return {
        "case_id": chapter,
        "category": category,
        "strategy": strategy,
        "trial": trial,
        "run_dir": str(run_dir),
        "status": status,
        "final_status": chapter_manifest.get("final_status"),
        "elapsed_ms": elapsed_ms,
        "surface_clear": surface_clear,
        "qa": qa,
        "semantic_review_observed": reviewed,
        "observed_outcome": observed,
        "expected_outcome": expected,
        "fixture_expected_match": fixture_expected_match,
        "outcome_match": fixture_expected_match,
        "comparison_scope": "repair_same_supplied_draft",
        "evidence_kind": "scripted_contract" if provider_mode == "offline" else "live_observation",
        "semantic_handling_success": semantic_handling_success,
        "step_count": _step_count(chapter_manifest),
        "child_count": _child_count(chapter_manifest),
        "usage": usage,
        "final_text": final_text,
    }


def _fraction(values: Sequence[bool]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _mean(values: Sequence[float | int]) -> float | None:
    return statistics.fmean(values) if values else None


def _summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_strategy: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_strategy[str(record["strategy"])].append(record)
        by_category[str(record["category"])].append(record)

    def summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        surface = [bool(row["surface_clear"]) for row in rows if isinstance(row.get("surface_clear"), bool)]
        semantic_rows = [row for row in rows if row.get("category") == "semantic"]
        reviewed = [bool(row["semantic_review_observed"]) for row in semantic_rows]
        fixture_matches = [
            bool(row["fixture_expected_match"])
            for row in rows
            if isinstance(row.get("fixture_expected_match"), bool)
        ]
        usage = [row.get("usage", {}) for row in rows]
        token_values = [
            float(item["total_tokens"])
            for item in usage
            if isinstance(item, Mapping) and isinstance(item.get("total_tokens"), (int, float))
        ]
        return {
            "cases": len(rows),
            "surface_clear_rate": _fraction(surface),
            "semantic_review_rate": _fraction(reviewed),
            "fixture_expected_match_rate": _fraction(fixture_matches),
            "semantic_handling_rate": _fraction([bool(row.get("semantic_handling_success")) for row in semantic_rows]),
            "mean_steps": _mean([float(row.get("step_count", 0)) for row in rows]),
            "mean_children": _mean([float(row.get("child_count", 0)) for row in rows]),
            "mean_elapsed_ms": _mean([float(row["elapsed_ms"]) for row in rows if isinstance(row.get("elapsed_ms"), (int, float))]),
            "total_tokens": sum(token_values) if token_values else None,
            "provider_calls": sum(int(item.get("provider_calls", 0)) for item in usage if isinstance(item, Mapping)),
        }

    return {
        "by_strategy": {key: summary(rows) for key, rows in sorted(by_strategy.items())},
        "by_category": {key: summary(rows) for key, rows in sorted(by_category.items())},
    }


def _blind_key(case_id: str, trial: int) -> str:
    digest = hashlib.sha256(f"{case_id}:{trial}:translation-showcase".encode("utf-8")).hexdigest()[:10]
    return f"item-{digest}"


def _blind_packet(records: Sequence[Mapping[str, Any]], sources: Mapping[str, str]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["case_id"]), int(record["trial"]))].append(record)
    items: list[dict[str, Any]] = []
    for (case_id, trial), rows in sorted(grouped.items()):
        seed = int(hashlib.sha256(f"{case_id}:{trial}".encode()).hexdigest()[:8], 16)
        ordered = sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{row['strategy']}".encode()).hexdigest())
        candidates: list[dict[str, Any]] = []
        for index, row in enumerate(ordered):
            label = chr(ord("A") + index)
            candidates.append(
                {
                    "candidate_id": f"candidate-{label}",
                    "text": row.get("final_text", ""),
                    "surface_checks": row.get("qa", {}).get("by_check", {}) if isinstance(row.get("qa"), Mapping) else {},
                }
            )
        items.append(
            {
                "item_id": _blind_key(case_id, trial),
                "source_text": sources.get(case_id, ""),
                "candidates": candidates,
            }
        )
    return {
        "schema_version": BLIND_PACKET_SCHEMA_VERSION,
        "instructions": "Compare each candidate with the source for faithfulness and readable English. Record a human judgment separately; no automated quality winner is claimed.",
        "items": items,
    }


def _blind_markdown(packet: Mapping[str, Any]) -> str:
    lines = [
        "# Blind translation comparison",
        "",
        "Candidates are intentionally unlabeled. Compare each candidate with its source and record human judgments separately.",
        "",
    ]
    for item in packet.get("items", []):
        if not isinstance(item, Mapping):
            continue
        lines.extend([f"## {item.get('item_id', 'item')}", "", "### Source", "", str(item.get("source_text", "")), ""])
        for candidate in item.get("candidates", []):
            if not isinstance(candidate, Mapping):
                continue
            lines.extend([f"### Candidate {candidate.get('candidate_id', '?')}", "", str(candidate.get("text", "")), ""])
    return "\n".join(lines).rstrip() + "\n"


def _comparison_report(comparison: Mapping[str, Any]) -> str:
    summary = comparison.get("summary", {})
    by_strategy = summary.get("by_strategy", {}) if isinstance(summary, Mapping) else {}
    mode = str(comparison.get("provider_mode", ""))
    intro = (
        "This is a bounded, scripted comparison of the same supplied drafts. It measures deterministic surface checks and review contracts; it does not claim that a scripted run establishes literary superiority."
        if mode == "offline"
        else "This is a bounded comparison of the same supplied drafts across three arms. It measures deterministic surface checks and review contracts; live trials are observations for one model/profile, not a general quality ranking."
    )
    lines = [
        "# Translation harness comparison",
        "",
        intro,
        "",
        f"Scope: {comparison.get('comparison_scope', 'repair_same_supplied_draft')} | Evidence: {comparison.get('evidence_kind', 'unknown')}",
        "",
        f"Cases: {comparison.get('case_count', 0)} | Trials per strategy: {comparison.get('trials', 0)} | Provider mode: {comparison.get('provider_mode', '')}",
        "",
        "| Arm | Surface clear | Semantic review observed | Mean steps | Mean children | Mean elapsed ms | Tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy in STRATEGIES:
        row = by_strategy.get(strategy, {}) if isinstance(by_strategy, Mapping) else {}
        lines.append(
            "| {strategy} | {surface} | {review} | {steps} | {children} | {elapsed} | {tokens} |".format(
                strategy=strategy,
                surface=_fmt_metric(row.get("surface_clear_rate")),
                review=_fmt_metric(row.get("semantic_review_rate")),
                steps=_fmt_metric(row.get("mean_steps")),
                children=_fmt_metric(row.get("mean_children")),
                elapsed=_fmt_metric(row.get("mean_elapsed_ms")),
                tokens=_fmt_metric(row.get("total_tokens")),
            )
        )
    if mode == "offline":
        interpretation = "Mechanical cases exercise punctuation cleanup. Terminology cases exercise bounded glossary corrections. Semantic cases contain hidden meaning errors that deterministic QA cannot prove; the specialist arm is expected to surface those for human review."
    else:
        interpretation = "Mechanical cases exercise punctuation cleanup. Terminology cases exercise bounded glossary corrections. Semantic review fields record whether a review was observed; they are not a correctness score for the authored semantic cases."
    lines.extend(
        [
            "",
            interpretation,
            "",
            "Offline results are contract evidence from fixed fixtures and cached responses. Live trials, when enabled, should be read as small observations under one model/profile and not as a general model ranking.",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_comparison(
    output_dir: str | Path,
    *,
    provider_mode: str = "offline",
    profile: str = "openai",
    model: str | None = None,
    trials: int | None = None,
) -> dict[str, Any]:
    """Run the 12-case comparison and write machine/human-readable evidence.

    ``offline`` defaults to one trial because its responses are fixed and
    replayable.  ``live`` defaults to three trials.  Passing ``trials`` is
    useful for a quick selected run in tests or for a larger exploratory run.
    """

    if provider_mode not in {"offline", "live"}:
        raise ValueError("provider_mode must be offline or live; compare saved runs with harness replay")
    if trials is None:
        trials = 1 if provider_mode == "offline" else 3
    if trials < 1:
        raise ValueError("trials must be at least 1")
    story, evaluator, chapter_ids, cases = _fixture_inputs()
    scenario = _read_json(FIXTURE_DIR / "scenario.json")
    case_map = {str(case["id"]): case for case in cases}
    glossary_path = FIXTURE_DIR / str(story.get("paths", {}).get("glossary_path", "terms/master_glossary.txt"))
    glossary = load_glossary(glossary_path)
    sources = {
        chapter: (FIXTURE_DIR / "source" / f"{chapter}.txt").read_text(encoding="utf-8")
        for chapter in chapter_ids
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runner = _load_runner()
    records: list[dict[str, Any]] = []
    run_entries: list[dict[str, Any]] = []

    for trial in range(1, trials + 1):
        for strategy in STRATEGIES:
            for chapter in chapter_ids:
                stage, staged_story = _stage_case(
                    root_story=story,
                    scenario=scenario,
                    chapter=chapter,
                    strategy=strategy,
                    root=output,
                )
                try:
                    run_output = output / "runs" / f"trial-{trial:02d}-{strategy}-{chapter}"
                    run_output.parent.mkdir(parents=True, exist_ok=True)
                    started = time.perf_counter()
                    result = _call_runner(
                        runner,
                        story_path=staged_story,
                        draft_dir=staged_story.parent / "drafts",
                        output_dir=run_output,
                        strategy=strategy,
                        provider_mode=provider_mode,
                        profile=profile,
                        model=model,
                        auto_approve=True,
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    run_dir, manifest = _result_manifest(result)
                    manifest.setdefault("comparison_wall_elapsed_ms", elapsed_ms)
                    run_entries.append(
                        {
                            "trial": trial,
                            "strategy": strategy,
                            "case_id": chapter,
                            "run_dir": str(run_dir),
                            "manifest": manifest,
                        }
                    )
                    records.append(
                        _record_case(
                            manifest=manifest,
                            run_dir=run_dir,
                            chapter=chapter,
                            case=case_map[chapter],
                            source_text=sources[chapter],
                            glossary=glossary,
                            strategy=strategy,
                            trial=trial,
                            provider_mode=provider_mode,
                        )
                    )
                finally:
                    stage.cleanup()

    comparison: dict[str, Any] = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "fixture": "samples/showcase_compare",
        "case_count": len(chapter_ids),
        "case_ids": chapter_ids,
        "strategies": list(STRATEGIES),
        "trials": trials,
        "provider_mode": provider_mode,
        "profile": profile,
        "model": model,
        "comparison_scope": "repair_same_supplied_draft",
        "evidence_kind": "scripted_contract" if provider_mode == "offline" else "live_observation",
        "limits": dict(SURFACE_LIMITS),
        "summary": _summarize_records(records),
        "records": records,
        "runs": run_entries,
        "evaluator_note": "Expected labels are evaluator-only metadata and are never included in staged scenarios or provider requests. Fixture matches describe scripted behavior; semantic handling is scored separately.",
    }
    packet = _blind_packet(records, sources)
    _write_json(output / "comparison.json", comparison)
    _write_json(output / "blind_packet.json", packet)
    (output / "blind_packet.md").write_text(_blind_markdown(packet), encoding="utf-8")
    (output / "report.md").write_text(_comparison_report(comparison), encoding="utf-8")
    return comparison


__all__ = [
    "BLIND_PACKET_SCHEMA_VERSION",
    "COMPARISON_SCHEMA_VERSION",
    "FIXTURE_DIR",
    "STRATEGIES",
    "SURFACE_LIMITS",
    "run_comparison",
]
