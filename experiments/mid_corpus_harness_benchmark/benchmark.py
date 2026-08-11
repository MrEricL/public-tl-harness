#!/usr/bin/env python3
"""Small, auditable runner for the five-chapter Harness v3 experiment."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
# The benchmark is checked into the standalone public repository, where the
# Harness v3 runtime lives directly under the repository-root package.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_translation.agent_models import AgentAction  # noqa: E402
from agentic_translation.agent_provider import AgentActionRequest  # noqa: E402
from agentic_translation.agent_session import (  # noqa: E402
    AgentSessionResult,
    run_repair_session,
)
from agentic_translation.agent_tools import (  # noqa: E402
    AGENT_TOOL_REGISTRY,
    ToolCall,
)
from agentic_translation.glossary import load_glossary  # noqa: E402


DIMENSIONS = (
    "fidelity",
    "fluency",
    "terminology",
    "voice",
    "formatting",
    "serious_error_avoidance",
)
PAIRWISE_DIMENSIONS = DIMENSIONS + ("overall",)
DEFAULT_JUDGE_IDS = (
    "codex_judge_1",
    "codex_judge_2",
    "codex_judge_3",
    "claude_sonnet_5",
)
_ISSUE_FIELDS = ("category", "severity", "source_span", "candidate_span", "note")
_MAX_REASON_LENGTH = 1200
_MAX_ISSUE_LENGTH = 1000


@dataclass(frozen=True)
class Selection:
    seed: int
    start_index: int
    files: list[str]
    source_characters: int
    rejected_draws: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RecordedActionProvider:
    """Normalize a migrated Codex repair trajectory through the v3 registry."""

    provider_name = "codex"
    model_name = "gpt-5.6-luna-core-worker"
    tool_protocol = "json_prompt"

    def __init__(self, actions: list[dict[str, Any]]) -> None:
        self.actions = actions
        self.index = 0
        self.requests: list[AgentActionRequest] = []
        self.call_records: list[Any] = []

    def next_action(self, request: AgentActionRequest) -> AgentAction:
        if self.index >= len(self.actions):
            raise ValueError("Recorded action list ended before the session became terminal")
        self.requests.append(request)
        call = ToolCall.from_json_action(self.actions[self.index])
        self.index += 1
        return AGENT_TOOL_REGISTRY.action_from_call(
            call,
            visible_names=request.exposed_tool_names or (),
        )


def select_window(
    corpus_dir: Path,
    *,
    seed: int,
    count: int = 5,
    lower: float = 0.30,
    upper: float = 0.70,
    character_limit: int = 15_000,
) -> Selection:
    files = sorted(corpus_dir.glob("*.txt"))
    first = math.ceil(len(files) * lower)
    stop = math.floor(len(files) * upper)
    starts = list(range(first, stop - count + 1))
    random.Random(seed).shuffle(starts)
    rejected: list[int] = []
    for start in starts:
        window = files[start : start + count]
        texts = [path.read_text(encoding="utf-8") for path in window]
        characters = sum(len(text) for text in texts)
        if all(text.strip() for text in texts) and characters <= character_limit:
            return Selection(
                seed=seed,
                start_index=start,
                files=[path.name for path in window],
                source_characters=characters,
                rejected_draws=rejected,
            )
        rejected.append(start)
    raise ValueError("No eligible chapter window satisfies the experiment limits")


def run_scripted_session(
    *,
    source_text: str,
    baseline_text: str,
    glossary_text: str,
    actions: list[dict[str, Any]],
    session_dir: Path,
    chapter: str,
) -> AgentSessionResult:
    session_dir.mkdir(parents=True, exist_ok=True)
    glossary_path = session_dir / "input_glossary.txt"
    glossary_path.write_text(glossary_text, encoding="utf-8")
    return run_repair_session(
        provider=RecordedActionProvider(actions),
        session_dir=session_dir,
        source_text=source_text,
        translated_text=baseline_text,
        glossary=load_glossary(glossary_path),
        run_id="mid-corpus-harness-benchmark",
        story_slug="simulator-alliance",
        chapter=chapter,
        provider_mode="recorded_codex_actions",
        max_steps=6,
        max_patch_attempts=2,
        dynamic_tools=True,
    )


def _roster_index(
    roster: list[dict[str, Any]] | dict[str, Any] | list[str] | tuple[str, ...] | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Normalize the fixed blind-judge roster for validation and aggregation."""

    if roster is None and config is not None:
        blind = config.get("blind") or {}
        roster = blind.get("roster") or blind.get("judges")
    if roster is None:
        return {}
    if isinstance(roster, dict):
        entries = []
        for judge_id, entry in roster.items():
            if isinstance(entry, dict):
                entries.append({"id": judge_id, **entry})
            else:
                entries.append({"id": judge_id, "family": entry})
    else:
        entries = []
        for entry in roster:
            if isinstance(entry, str):
                entries.append({"id": entry})
            elif isinstance(entry, dict):
                entries.append(entry)
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        judge_id = entry.get("id") or entry.get("judge")
        if not isinstance(judge_id, str) or not judge_id.strip():
            raise ValueError("Blind roster entries require a nonempty id")
        if judge_id in result:
            raise ValueError(f"Duplicate blind roster judge: {judge_id}")
        result[judge_id] = dict(entry)
    return result


def build_blind_packets(
    *,
    chapters: list[str],
    root: Path,
    seed: int,
    judge_ids: list[str] | tuple[str, ...] | None = None,
    judges: list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, dict[str, str]]]:
    """Build deterministic judge-specific packets and private A/B mappings.

    The shuffled pair for each chapter names the two judges who see baseline as
    candidate A. Cycling through five of the six possible pairs guarantees an
    exact two/four split per chapter and a two-or-three A-position count per
    judge across the five-chapter benchmark.
    """

    if judge_ids is not None and judges is not None and tuple(judge_ids) != tuple(judges):
        raise ValueError("judge_ids and judges must agree when both are supplied")
    ids = tuple(judge_ids if judge_ids is not None else judges or DEFAULT_JUDGE_IDS)
    if len(ids) != 4 or len(set(ids)) != 4:
        raise ValueError("Counterbalanced blind packets require exactly four unique judges")
    if not chapters:
        raise ValueError("At least one chapter is required for blind packets")

    blind_root = root / "blind"
    packets_root = blind_root / "packets"
    private_root = blind_root / "private_mapping"
    packets_root.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True)
    pairs = list(combinations(ids, 2))
    random.Random(seed).shuffle(pairs)
    mapping: dict[str, dict[str, dict[str, str]]] = {judge: {} for judge in ids}
    for chapter_index, chapter in enumerate(chapters):
        baseline_as_a = set(pairs[chapter_index % len(pairs)])
        values = {
            "baseline": (root / "baseline" / f"{chapter}.txt").read_text(encoding="utf-8"),
            "harness": (root / "harness" / "outputs" / f"{chapter}.txt").read_text(encoding="utf-8"),
        }
        source = (root / "source" / f"{chapter}.txt").read_text(encoding="utf-8")
        for judge in ids:
            systems = (
                ("baseline", "harness")
                if judge in baseline_as_a
                else ("harness", "baseline")
            )
            mapping[judge][chapter] = {"A": systems[0], "B": systems[1]}
            packet = (
                f"# Blind translation pair: {chapter}\n\n"
                f"## Chinese source\n\n{source}\n\n"
                f"## Candidate A\n\n{values[systems[0]]}\n\n"
                f"## Candidate B\n\n{values[systems[1]]}\n"
            )
            packet = packet.rstrip("\r\n") + "\n"
            judge_packet_root = packets_root / judge
            judge_packet_root.mkdir(parents=True, exist_ok=True)
            (judge_packet_root / f"{chapter}.md").write_text(packet, encoding="utf-8")
    for judge in ids:
        (private_root / f"{judge}.json").write_text(
            json.dumps(mapping[judge], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return mapping


def _bounded_text(value: Any, *, field: str, maximum: int, allow_none: bool = False) -> None:
    if allow_none and value is None:
        return
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"Scorecard {field} must be a nonempty bounded string")


def _validate_issue_container(row: dict[str, Any]) -> None:
    issues = row.get("issues")
    if not isinstance(issues, dict) or set(issues) != {"A", "B"}:
        raise ValueError("Scorecard issues must provide A and B arrays")
    for label in ("A", "B"):
        values = issues[label]
        if not isinstance(values, list):
            raise ValueError("Scorecard issues for A and B must be arrays")
        for issue in values:
            if not isinstance(issue, dict) or set(issue) != set(_ISSUE_FIELDS):
                raise ValueError("Each issue must contain the typed issue fields")
            _bounded_text(issue.get("category"), field="issue category", maximum=_MAX_ISSUE_LENGTH)
            if issue.get("severity") not in {"minor", "major"}:
                raise ValueError("Issue severity must be minor or major")
            _bounded_text(
                issue.get("source_span"),
                field="issue source_span",
                maximum=_MAX_ISSUE_LENGTH,
                allow_none=True,
            )
            _bounded_text(
                issue.get("candidate_span"),
                field="issue candidate_span",
                maximum=_MAX_ISSUE_LENGTH,
                allow_none=True,
            )
            _bounded_text(issue.get("note"), field="issue note", maximum=_MAX_ISSUE_LENGTH)


def load_scorecard(
    path: Path,
    *,
    required_chapters: list[str],
    roster: list[dict[str, Any]] | dict[str, Any] | list[str] | tuple[str, ...] | None = None,
    config: dict[str, Any] | None = None,
    expected_judge: str | None = None,
    expected_family: str | None = None,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Scorecard must be a JSON object")
    judge = value.get("judge")
    if not isinstance(judge, str) or not judge.strip():
        raise ValueError("Scorecard judge must be a nonempty string")
    if expected_judge is not None and judge != expected_judge:
        raise ValueError(f"Scorecard judge mismatch: expected {expected_judge}, got {judge}")
    roster_index = _roster_index(roster, config=config)
    roster_entry = None
    if roster_index:
        if judge not in roster_index:
            raise ValueError(f"Scorecard judge is not registered: {judge}")
        roster_entry = roster_index[judge]
        expected_roster_family = roster_entry.get("family")
        if expected_roster_family is not None and value.get("family") != expected_roster_family:
            raise ValueError(f"Scorecard family mismatch for {judge}")
    if expected_family is not None and value.get("family") != expected_family:
        raise ValueError(f"Scorecard family mismatch: expected {expected_family}")
    if "family" in value:
        _bounded_text(value.get("family"), field="family", maximum=100)
    chapters = value.get("chapters")
    if not isinstance(chapters, list) or any(not isinstance(row, dict) for row in chapters):
        raise ValueError("Scorecard chapters must be an array of objects")
    if {row.get("chapter") for row in chapters} != set(required_chapters):
        raise ValueError("Scorecard must cover every required chapter exactly once")
    if len(chapters) != len(required_chapters):
        raise ValueError("Scorecard has duplicate chapters")
    for row in chapters:
        if row.get("preference") not in {"A", "B", "tie"}:
            raise ValueError("Scorecard preference must be A, B, or tie")
        reason = row.get("reason")
        _bounded_text(reason, field="reason", maximum=_MAX_REASON_LENGTH)
        pairwise = row.get("pairwise_preferences")
        if not isinstance(pairwise, dict) or set(pairwise) != set(PAIRWISE_DIMENSIONS):
            raise ValueError("Scorecard pairwise preferences do not match the rubric")
        if any(pairwise[dimension] not in {"A", "B", "tie"} for dimension in PAIRWISE_DIMENSIONS):
            raise ValueError("Scorecard pairwise preferences must be A, B, or tie")
        if pairwise["overall"] != row["preference"]:
            raise ValueError("Scorecard preference must equal pairwise overall preference")
        scores = row.get("scores", {})
        if not isinstance(scores, dict) or set(scores) != {"A", "B"}:
            raise ValueError("Scorecard scores must provide A and B candidates")
        for label in ("A", "B"):
            candidate = scores[label]
            if not isinstance(candidate, dict) or set(candidate) != set(DIMENSIONS):
                raise ValueError("Scorecard dimensions do not match the rubric")
            if any(
                type(score) is not int or not 1 <= score <= 5
                for score in candidate.values()
            ):
                raise ValueError("Every dimension score must be an integer in 1..5")
        _validate_issue_container(row)
        serious_errors = row.get("serious_errors")
        if not isinstance(serious_errors, dict) or set(serious_errors) != {"A", "B"}:
            raise ValueError("Scorecard serious_errors must provide A and B arrays")
        for label in ("A", "B"):
            errors = serious_errors[label]
            if not isinstance(errors, list):
                raise ValueError("Scorecard serious_errors for A and B must be arrays")
            for error in errors:
                _bounded_text(error, field="serious error", maximum=_MAX_REASON_LENGTH)
            if errors and row["scores"][label]["serious_error_avoidance"] == 5:
                raise ValueError(
                    "Nonempty serious_errors cannot have serious_error_avoidance=5"
                )
    return value


def _qa_summary(qa: dict[str, Any] | None) -> dict[str, Any]:
    """Return compact QA evidence while preserving the recorded check counts."""

    qa = qa or {}
    summary = qa.get("summary") or {}
    findings = qa.get("findings") or []
    by_check = summary.get("by_check")
    if not isinstance(by_check, dict):
        counts: Counter[str] = Counter(
            finding.get("check_id", "unknown")
            for finding in findings
            if isinstance(finding, dict)
        )
        by_check = dict(counts)
    total = summary.get("total_findings")
    if not isinstance(total, int):
        total = len(findings)
    score = qa.get("score")
    return {
        "findings": total,
        "by_check": {str(key): int(value) for key, value in by_check.items()},
        "score": score if isinstance(score, (int, float)) else None,
    }


def _session_evidence(
    *, chapters: list[str], sessions_root: Path | None, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read durable episode/snapshot evidence without re-running any provider."""

    qa_by_chapter: dict[str, Any] = {}
    action_by_chapter: dict[str, Any] = {}
    before_by_check: Counter[str] = Counter()
    after_by_check: Counter[str] = Counter()
    qa_before_scores: list[float] = []
    qa_after_scores: list[float] = []
    action_totals = Counter({"steps": 0, "patch_attempts": 0, "accepted": 0, "rejected": 0})
    final_statuses: Counter[str] = Counter()
    limits = config.get("limits") or {}

    for chapter in chapters:
        episode: dict[str, Any] = {}
        snapshot: dict[str, Any] = {}
        if sessions_root is not None:
            session_dir = sessions_root / chapter
            episode_path = session_dir / "agent_episode.json"
            snapshot_path = session_dir / "session_snapshot.json"
            if episode_path.is_file():
                episode = json.loads(episode_path.read_text(encoding="utf-8"))
            if snapshot_path.is_file():
                snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

        initial = _qa_summary(episode.get("initial_qa"))
        final = _qa_summary(episode.get("final_qa"))
        qa_by_chapter[chapter] = {
            "before_findings": initial["findings"],
            "after_findings": final["findings"],
            "delta_findings": final["findings"] - initial["findings"],
            "before_by_check": initial["by_check"],
            "after_by_check": final["by_check"],
            "before_score": initial["score"],
            "after_score": final["score"],
        }
        before_by_check.update(initial["by_check"])
        after_by_check.update(final["by_check"])
        if isinstance(initial["score"], (int, float)):
            qa_before_scores.append(float(initial["score"]))
        if isinstance(final["score"], (int, float)):
            qa_after_scores.append(float(final["score"]))

        steps = episode.get("steps")
        if not isinstance(steps, list):
            steps = []
        mutation_actions = [
            step
            for step in steps
            if isinstance(step, dict)
            and isinstance(step.get("action"), dict)
            and step["action"].get("tool")
            in {"submit_patch", "normalize_punctuation"}
        ]
        accepted = 0
        rejected = 0
        for step in mutation_actions:
            observation = step.get("observation") or {}
            data = observation.get("data") or {}
            if data.get("accepted") is True:
                accepted += 1
            else:
                rejected += 1
        snapshot_attempts = snapshot.get("patch_attempts")
        patch_attempts = (
            snapshot_attempts
            if isinstance(snapshot_attempts, int)
            else len(mutation_actions)
        )
        final_status = episode.get("final_status") or snapshot.get("status")
        final_status = str(final_status) if final_status is not None else None
        if final_status:
            final_statuses[final_status] += 1
        action_by_chapter[chapter] = {
            "steps": len(steps),
            "patch_attempts": patch_attempts,
            "accepted": accepted,
            "rejected": rejected,
            "final_status": final_status,
            "provider": episode.get("provider"),
            "model": episode.get("model"),
            "provider_mode": episode.get("provider_mode"),
            "max_steps": episode.get("max_steps", limits.get("max_steps")),
            "max_patch_attempts": episode.get(
                "max_patch_attempts", limits.get("max_patch_attempts")
            ),
        }
        action_totals["steps"] += len(steps)
        action_totals["patch_attempts"] += patch_attempts
        action_totals["accepted"] += accepted
        action_totals["rejected"] += rejected

    qa = {
        "by_chapter": qa_by_chapter,
        "totals": {
            "before_findings": sum(
                row["before_findings"] for row in qa_by_chapter.values()
            ),
            "after_findings": sum(
                row["after_findings"] for row in qa_by_chapter.values()
            ),
            "delta_findings": sum(
                row["delta_findings"] for row in qa_by_chapter.values()
            ),
            "before_by_check": dict(before_by_check),
            "after_by_check": dict(after_by_check),
            "before_score_mean": round(mean(qa_before_scores), 3)
            if qa_before_scores
            else None,
            "after_score_mean": round(mean(qa_after_scores), 3)
            if qa_after_scores
            else None,
        },
    }
    actions = {
        "by_chapter": action_by_chapter,
        "totals": {
            "steps": action_totals["steps"],
            "patch_attempts": action_totals["patch_attempts"],
            "accepted": action_totals["accepted"],
            "rejected": action_totals["rejected"],
            "final_statuses": dict(final_statuses),
        },
    }
    return qa, actions


def aggregate_scorecards(
    *,
    mapping: dict[str, dict[str, dict[str, str]]],
    scorecards: list[dict[str, Any]],
    root: Path | None = None,
    session_root: Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    roster = _roster_index(config=config)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Aggregate mapping must contain judge-specific mappings")
    chapters: set[str] = set()
    for judge, judge_mapping in mapping.items():
        if not isinstance(judge_mapping, dict):
            raise ValueError(f"Mapping for {judge} must be chapter-specific")
        for chapter, labels in judge_mapping.items():
            if not isinstance(labels, dict) or set(labels) != {"A", "B"}:
                raise ValueError(f"Mapping for {judge}/{chapter} must provide A and B")
            if set(labels.values()) != {"baseline", "harness"}:
                raise ValueError(f"Mapping for {judge}/{chapter} must name both systems")
            chapters.add(chapter)
    ordered_chapters = sorted(chapters)
    scores: dict[str, dict[str, list[int]]] = {
        system: {dimension: [] for dimension in DIMENSIONS}
        for system in ("baseline", "harness")
    }
    preferences: Counter[str] = Counter({"baseline": 0, "harness": 0, "tie": 0})
    pairwise_preferences: dict[str, Counter[str]] = {
        dimension: Counter({"baseline": 0, "harness": 0, "tie": 0})
        for dimension in PAIRWISE_DIMENSIONS
    }
    preferences_by_judge: dict[str, Counter[str]] = {}
    preferences_by_model_family: dict[str, Counter[str]] = {}
    preferences_by_chapter: dict[str, Counter[str]] = {
        chapter: Counter({"baseline": 0, "harness": 0, "tie": 0})
        for chapter in ordered_chapters
    }
    chapter_evaluations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_judges: set[str] = set()
    for scorecard in scorecards:
        judge = scorecard.get("judge")
        if not isinstance(judge, str) or not judge:
            raise ValueError("Every scorecard must declare a judge")
        if judge in seen_judges:
            raise ValueError(f"Duplicate scorecard judge: {judge}")
        seen_judges.add(judge)
        if judge not in mapping:
            raise ValueError(f"No private mapping is registered for scorecard judge: {judge}")
        roster_entry = roster.get(judge, {})
        family = scorecard.get("family") or roster_entry.get("family")
        if roster and judge not in roster:
            raise ValueError(f"Scorecard judge is not registered: {judge}")
        if not isinstance(family, str) or not family:
            raise ValueError(f"Scorecard family is required for judge: {judge}")
        judge_preferences = preferences_by_judge.setdefault(
            judge, Counter({"baseline": 0, "harness": 0, "tie": 0})
        )
        family_preferences = preferences_by_model_family.setdefault(
            family, Counter({"baseline": 0, "harness": 0, "tie": 0})
        )
        judge_mapping = mapping[judge]
        for row in scorecard["chapters"]:
            chapter = row["chapter"]
            if chapter not in judge_mapping:
                raise ValueError(f"No private mapping for {judge}/{chapter}")
            labels = judge_mapping[chapter]
            for label in ("A", "B"):
                system = labels[label]
                for dimension in DIMENSIONS:
                    scores[system][dimension].append(row["scores"][label][dimension])
            raw_pairwise = row["pairwise_preferences"]
            overall_label = raw_pairwise["overall"]
            system_preference = (
                "tie" if overall_label == "tie" else labels[overall_label]
            )
            for dimension in PAIRWISE_DIMENSIONS:
                pairwise_label = raw_pairwise[dimension]
                mapped = "tie" if pairwise_label == "tie" else labels[pairwise_label]
                pairwise_preferences[dimension][mapped] += 1
            preferences[system_preference] += 1
            judge_preferences[system_preference] += 1
            family_preferences[system_preference] += 1
            preferences_by_chapter[chapter][system_preference] += 1
            mapped_errors: dict[str, list[str]] = {"baseline": [], "harness": []}
            raw_errors = row["serious_errors"]
            for label in ("A", "B"):
                system = labels[label]
                mapped_errors[system] = list(raw_errors[label])
            mapped_issues: dict[str, list[dict[str, Any]]] = {"baseline": [], "harness": []}
            raw_issues = row["issues"]
            for label in ("A", "B"):
                system = labels[label]
                mapped_issues[system] = list(raw_issues[label])
            chapter_evaluations[chapter].append(
                {
                    "judge": judge,
                    "family": family,
                    "preference_label": overall_label,
                    "preference": system_preference,
                    "reason": row["reason"],
                    "issues": mapped_issues,
                    "serious_errors": mapped_errors,
                }
            )
    systems = {
        system: {
            "dimensions": {
                dimension: round(mean(values), 3) if values else None
                for dimension, values in dimensions.items()
            }
        }
        for system, dimensions in scores.items()
    }
    modal_by_chapter: dict[str, float | None] = {}
    exact_by_chapter: dict[str, float | None] = {}
    for chapter in ordered_chapters:
        counts = preferences_by_chapter[chapter]
        n = sum(counts.values())
        modal_by_chapter[chapter] = round(max(counts.values()) / n, 3) if n else None
        if n < 2:
            exact_by_chapter[chapter] = None
        else:
            numerator = sum(count * (count - 1) // 2 for count in counts.values())
            denominator = n * (n - 1) // 2
            exact_by_chapter[chapter] = round(numerator / denominator, 3)
    modal_values = [value for value in modal_by_chapter.values() if value is not None]
    exact_values = [value for value in exact_by_chapter.values() if value is not None]
    modal_mean = round(mean(modal_values), 3) if modal_values else None
    exact_mean = round(mean(exact_values), 3) if exact_values else None
    dimension_deltas = {
        dimension: round(
            systems["harness"]["dimensions"][dimension]
            - systems["baseline"]["dimensions"][dimension],
            3,
        )
        if systems["harness"]["dimensions"][dimension] is not None
        and systems["baseline"]["dimensions"][dimension] is not None
        else None
        for dimension in DIMENSIONS
    }
    sessions_path = session_root
    if sessions_path is None and root is not None:
        sessions_path = Path(root) / "harness" / "sessions"
    deterministic_qa, actions = _session_evidence(
        chapters=ordered_chapters, sessions_root=sessions_path, config=config
    )
    provenance_keys = (
        "selection",
        "conditions",
        "limits",
        "prompts",
        "source_provenance",
        "public_reproducibility",
        "worker",
        "judge",
        "blind",
    )
    provenance = {
        key: config[key]
        for key in provenance_keys
        if key in config
    }
    return {
        "chapters": ordered_chapters,
        "judge_count": len(scorecards),
        "systems": systems,
        "preferences": dict(preferences),
        "preferences_by_judge": {
            judge: dict(counts) for judge, counts in preferences_by_judge.items()
        },
        "preferences_by_chapter": {
            chapter: dict(counts) for chapter, counts in preferences_by_chapter.items()
        },
        "preferences_by_model_family": {
            family: dict(counts)
            for family, counts in sorted(preferences_by_model_family.items())
        },
        "pairwise_preferences": {
            dimension: dict(counts)
            for dimension, counts in pairwise_preferences.items()
        },
        "modal_decision_share": {"by_chapter": modal_by_chapter, "mean": modal_mean},
        "pairwise_exact_agreement": {
            "by_chapter": exact_by_chapter,
            "mean": exact_mean,
        },
        "dimension_deltas": dimension_deltas,
        "deterministic_qa": deterministic_qa,
        "actions": actions,
        "chapter_evaluations": {
            chapter: evaluations
            for chapter, evaluations in chapter_evaluations.items()
        },
        "optional_providers": config.get("optional", {}),
        "provenance": provenance,
    }


def render_report(results: dict[str, Any], *, config: dict[str, Any]) -> str:
    """Render pairwise evidence first, retaining numeric and QA detail as secondary."""

    def cell(value: Any) -> str:
        return "—" if value is None else str(value)

    def text_cell(value: Any) -> str:
        return " ".join(str(value or "").split()).replace("|", "\\|") or "—"

    chapters = results.get("chapters", [])
    qa = results.get("deterministic_qa", {})
    qa_totals = qa.get("totals", {})
    actions = results.get("actions", {})
    action_totals = actions.get("totals", {})
    systems = results.get("systems", {})
    deltas = results.get("dimension_deltas", {})
    overall = results.get("pairwise_preferences", {}).get("overall") or results.get(
        "preferences", {}
    )
    baseline_pref = overall.get("baseline", 0)
    harness_pref = overall.get("harness", 0)
    tie_pref = overall.get("tie", 0)
    total_decisions = baseline_pref + harness_pref + tie_pref
    provenance = results.get("provenance", {})
    selection = provenance.get("selection", config.get("selection", {}))
    selected_files = ", ".join(selection.get("files", chapters))
    optional = results.get("optional_providers", config.get("optional", {}))
    modal = results.get("modal_decision_share", {})
    exact = results.get("pairwise_exact_agreement", {})
    qa_by_check = qa_totals.get("before_by_check") or {}

    def qa_count(*names: str) -> int:
        for name in names:
            value = qa_by_check.get(name)
            if isinstance(value, int):
                return value
        return 0

    lines = [
        "# Mid-Corpus Harness Benchmark Report",
        "",
        "Data availability: the real five-chapter corpus and all derived full-text artifacts (source chapters, baseline translations, Harness outputs, blind packets, private mappings, scorecards, action files, and session evidence) are intentionally not distributed.",
        "",
        "## Bottom line",
        "",
        (
            f"Overall pairwise preferences were baseline {baseline_pref}, harness "
            f"{harness_pref}, and tie {tie_pref} across {total_decisions} chapter-judge "
            "decisions. Deterministic QA findings moved from "
            f"{qa_totals.get('before_findings', 0)} to {qa_totals.get('after_findings', 0)} "
            f"across {len(chapters)} chapters; numeric dimension means are secondary "
            "descriptive metadata."
        ),
        (
            f"QA note: {qa_totals.get('before_findings', 0)} deterministic rule hits "
            f"({qa_count('chinese_punctuation', 'punctuation')} punctuation, "
            f"{qa_count('glossary_required', 'glossary')} glossary, "
            f"{qa_count('system_panel_count', 'panel_count')} "
            "panel-structure rule hits, meaning different numbers of bracketed panels "
            f"between source and translation)—not {qa_totals.get('before_findings', 0)} independent semantic defects; all cleared to "
            f"{qa_totals.get('after_findings', 0)}."
        ),
        "",
        "## Primary pairwise results",
        "",
        "`pairwise_preferences.overall`",
        "",
        "| System | Overall pairwise preference |",
        "| --- | ---: |",
        f"| Baseline | {baseline_pref} |",
        f"| Harness | {harness_pref} |",
        f"| Tie | {tie_pref} |",
        "",
        "`preferences_by_model_family`",
        "",
        "| Model family | Baseline | Harness | Tie |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family, counts in sorted(results.get("preferences_by_model_family", {}).items()):
        lines.append(
            f"| {family} | {counts.get('baseline', 0)} | {counts.get('harness', 0)} | "
            f"{counts.get('tie', 0)} |"
        )
    lines.extend(
        [
            "",
            "### Chapter-level primary metrics",
            "",
            "| Chapter | Baseline | Harness | Tie | modal_decision_share | pairwise_exact_agreement |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    by_chapter = results.get("preferences_by_chapter", {})
    for chapter in chapters:
        counts = by_chapter.get(chapter, {})
        lines.append(
            f"| {chapter} | {counts.get('baseline', 0)} | {counts.get('harness', 0)} | "
            f"{counts.get('tie', 0)} | {cell(modal.get('by_chapter', {}).get(chapter))} | "
            f"{cell(exact.get('by_chapter', {}).get(chapter))} |"
        )
    lines.extend(
        [
            "",
            f"- modal_decision_share mean: {cell(modal.get('mean'))}.",
            f"- pairwise_exact_agreement mean: {cell(exact.get('mean'))}.",
            "",
            "## Secondary numeric means",
            "",
            "The anchored 1–5 scores are secondary metadata; deltas are harness minus baseline.",
            "",
            "| Dimension | Baseline | Harness | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for dimension in DIMENSIONS:
        lines.append(
            f"| {dimension.replace('_', ' ')} | "
            f"{cell(systems.get('baseline', {}).get('dimensions', {}).get(dimension))} | "
            f"{cell(systems.get('harness', {}).get('dimensions', {}).get(dimension))} | "
            f"{cell(deltas.get(dimension))} |"
        )
    lines.extend(
        [
            "",
            "## Judge and chapter detail",
            "",
            "| Judge | Baseline | Harness | Tie |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for judge, counts in sorted(results.get("preferences_by_judge", {}).items()):
        lines.append(
            f"| {judge} | {counts.get('baseline', 0)} | {counts.get('harness', 0)} | "
            f"{counts.get('tie', 0)} |"
        )
    lines.extend(
        [
            "",
            "## QA taxonomy and repair actions",
            "",
            "### QA findings by check (`before_by_check`)",
            "",
            "| Check | Findings before repair |",
            "| --- | ---: |",
        ]
    )
    for check, count in sorted((qa_totals.get("before_by_check") or {}).items()):
        lines.append(f"| {check} | {count} |")
    lines.extend(
        [
            "",
            "| Chapter | QA before → after | Steps | Patches | Accepted | Rejected | Final status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for chapter in chapters:
        qa_row = qa.get("by_chapter", {}).get(chapter, {})
        action_row = actions.get("by_chapter", {}).get(chapter, {})
        lines.append(
            f"| {chapter} | {qa_row.get('before_findings', 0)} → {qa_row.get('after_findings', 0)} | "
            f"{action_row.get('steps', 0)} | {action_row.get('patch_attempts', 0)} | "
            f"{action_row.get('accepted', 0)} | {action_row.get('rejected', 0)} | "
            f"{cell(action_row.get('final_status'))} |"
        )
    lines.append(
        f"| **Total** | **{qa_totals.get('before_findings', 0)} → {qa_totals.get('after_findings', 0)}** | "
        f"**{action_totals.get('steps', 0)}** | **{action_totals.get('patch_attempts', 0)}** | "
        f"**{action_totals.get('accepted', 0)}** | **{action_totals.get('rejected', 0)}** | "
        f"**{json.dumps(action_totals.get('final_statuses', {}), sort_keys=True)}** |"
    )
    lines.extend(
        [
            "",
            "## Sanitized comparison examples",
            "",
            "Only brief, paraphrased examples are retained in the public report; source wording, candidate wording, and long judge rationales are not distributed.",
            "",
            "- One anonymized pair differed mainly in repeated target punctuation normalization.",
            "- Another differed mainly in a source-term romanization versus an established localized rendering.",
            "",
        ]
    )
    lines.extend(
        [
            "",
            "## Reproducibility and provenance",
            "",
            f"- Selected files: `{selected_files}`.",
            f"- Selection seed: `{selection.get('seed', '—')}`; start index: `{selection.get('start_index', '—')}`; source characters: `{selection.get('source_characters', '—')}`.",
            "- Conditions: one-shot baseline versus the same draft after bounded repair.",
            "- The repair actions originated in Codex runs, were migrated to the "
            "Harness v3 schema, and re-executed through this codebase — not produced "
            "in a single end-to-end run.",
            "- Action JSON and episode evidence are intentionally omitted, so this package makes no verbatim replay claim.",
        ]
    )
    if optional:
        lines.append("- Optional provider statuses:")
        for provider, status in sorted(optional.items()):
            if isinstance(status, dict):
                label = status.get("status", "unknown")
                detail = status.get("reason") or status.get("model") or ""
                suffix = f" — {detail}" if detail else ""
                lines.append(f"  - `{provider}`: {label}{suffix}")
            else:
                lines.append(f"  - `{provider}`: {status}")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This is **n=5** consecutive chapters from **one novel**.",
            "- Overall pairwise preferences and anchored dimension scores come from model judges, not human bilingual reviewers.",
            "- The candidates share the same one-shot baseline, and the harness only gets a bounded repair pass; this is not an independent translation-system comparison.",
            "- The selection seed and start index are upstream provenance; without the external full corpus, this public artifact cannot independently verify the mid-corpus/consecutive selection derivation.",
            "- No statistical-significance test is justified or reported here. The evidence is exploratory, not a production-quality claim.",
            "",
            "See [`README.md`](README.md) for exact local commands and the artifact map.",
            "",
        ]
    )
    return "\n".join(lines)


def _chapter_ids(config: dict[str, Any]) -> list[str]:
    return [Path(name).stem for name in config["selection"]["files"]]


def _select(args: argparse.Namespace) -> int:
    root = Path(args.root)
    selection = select_window(Path(args.corpus), seed=args.seed)
    source_dir = root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    for name in selection.files:
        shutil.copyfile(Path(args.corpus) / name, source_dir / name)
    config_path = root / "benchmark_config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else {}
    )
    config.update(
        {
            "selection": selection.to_dict(),
            "conditions": {
                "baseline": "one-shot gpt-5.6-luna core_worker",
                "harness": "Codex-originated repair trajectories migrated to the final bounded Harness v3 schema and executed through the public runtime.",
            },
            "limits": {"chapters": 5, "max_steps": 6, "max_patch_attempts": 2},
        }
    )
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def _run_harness(args: argparse.Namespace) -> int:
    root = Path(args.root)
    config = json.loads((root / "benchmark_config.json").read_text(encoding="utf-8"))
    glossary = Path(args.glossary).read_text(encoding="utf-8")
    for chapter in _chapter_ids(config):
        session_dir = root / "harness" / "sessions" / chapter
        if session_dir.exists():
            shutil.rmtree(session_dir)
        result = run_scripted_session(
            source_text=(root / "source" / f"{chapter}.txt").read_text(encoding="utf-8"),
            baseline_text=(root / "baseline" / f"{chapter}.txt").read_text(encoding="utf-8"),
            glossary_text=glossary,
            actions=json.loads((root / "harness" / "actions" / f"{chapter}.json").read_text(encoding="utf-8")),
            session_dir=session_dir,
            chapter=chapter,
        )
        output = root / "harness" / "outputs" / f"{chapter}.txt"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.final_text, encoding="utf-8")
    return 0


def _blind(args: argparse.Namespace) -> int:
    root = Path(args.root)
    config = json.loads((root / "benchmark_config.json").read_text(encoding="utf-8"))
    roster = _roster_index(config=config)
    judge_ids = tuple(roster) if roster else DEFAULT_JUDGE_IDS
    build_blind_packets(
        chapters=_chapter_ids(config),
        root=root,
        seed=args.seed,
        judge_ids=judge_ids,
    )
    return 0


def _validate_public_results(root: Path) -> None:
    """Validate the aggregate-only artifact retained in the public checkout."""

    result_path = root / "results.json"
    if not result_path.is_file():
        raise ValueError(f"Public aggregate is missing: {result_path}")
    value = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Public aggregate must be a JSON object")
    preferences = value.get("preferences")
    if preferences != {"baseline": 7, "harness": 5, "tie": 8}:
        raise ValueError("Public aggregate does not preserve reconciled 7/5/8 preferences")
    if "chapter_evaluations" in value:
        raise ValueError("Public aggregate must omit text-bearing chapter evaluations")


def _aggregate(args: argparse.Namespace) -> int:
    root = Path(args.root)
    config = json.loads((root / "benchmark_config.json").read_text(encoding="utf-8"))
    chapters = _chapter_ids(config)
    roster = _roster_index(config=config)
    private_root = root / "blind" / "private_mapping"
    scorecard_root = root / "blind" / "scorecards"
    if not private_root.is_dir() or not scorecard_root.is_dir():
        _validate_public_results(root)
        if args.validate_only:
            return 0
        raise ValueError(
            "Full-text private mappings and scorecards are intentionally not distributed; "
            "use a caller-owned fixture root to aggregate."
        )
    mapping: dict[str, dict[str, dict[str, str]]] = {}
    for path in sorted(private_root.glob("*.json")):
        judge = path.stem
        if roster and judge not in roster:
            raise ValueError(f"Private mapping judge is not registered: {judge}")
        mapping[judge] = json.loads(path.read_text(encoding="utf-8"))
    if roster and set(mapping) != set(roster):
        raise ValueError("Private mappings must cover exactly the configured judge roster")
    cards = []
    seen_judges: set[str] = set()
    for path in sorted(scorecard_root.glob("*.json")):
        judge = path.stem
        if judge in seen_judges:
            raise ValueError(f"Duplicate scorecard judge: {judge}")
        seen_judges.add(judge)
        expected_family = roster.get(judge, {}).get("family") if roster else None
        cards.append(
            load_scorecard(
                path,
                required_chapters=chapters,
                config=config,
                expected_judge=judge,
                expected_family=expected_family,
            )
        )
    if roster and seen_judges != set(roster):
        raise ValueError("Scorecards must cover exactly the configured judge roster")
    if args.validate_only:
        return 0
    results = aggregate_scorecards(
        mapping=mapping,
        scorecards=cards,
        root=root,
        config=config,
    )
    (root / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "REPORT.md").write_text(render_report(results, config=config), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select_parser = commands.add_parser("select")
    select_parser.add_argument("--root", default=str(EXPERIMENT_ROOT))
    select_parser.add_argument("--corpus", required=True)
    select_parser.add_argument("--seed", type=int, required=True)
    select_parser.set_defaults(func=_select)

    run_parser = commands.add_parser("run-harness")
    run_parser.add_argument("--root", default=str(EXPERIMENT_ROOT))
    run_parser.add_argument(
        "--glossary",
        default=str(EXPERIMENT_ROOT / "harness" / "sessions" / "0247" / "input_glossary.txt"),
    )
    run_parser.set_defaults(func=_run_harness)

    blind_parser = commands.add_parser("blind")
    blind_parser.add_argument("--root", default=str(EXPERIMENT_ROOT))
    blind_parser.add_argument("--seed", type=int, required=True)
    blind_parser.set_defaults(func=_blind)

    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument("--root", default=str(EXPERIMENT_ROOT))
    aggregate_parser.add_argument("--validate-only", action="store_true")
    aggregate_parser.set_defaults(func=_aggregate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
