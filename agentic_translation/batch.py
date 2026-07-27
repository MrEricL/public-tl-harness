from __future__ import annotations

import json
import hashlib
import math
import os
import re
from collections import Counter, OrderedDict
from pathlib import Path

from pydantic import TypeAdapter

from .agent_models import AgentAction, AgentEpisode
from .models import (
    AgenticEvidence,
    AgentWorkOrder,
    AgentWorkOrderExecutionPreview,
    AgentWorkOrderItem,
    AgentWorkOrderSummary,
    ArtifactQAReport,
    AgentAttempt,
    BaselineComparison,
    BatchInspectionBlocker,
    BatchInspectionReport,
    BatchLiveProofResult,
    BatchManifest,
    BatchPipelineResult,
    BatchProofReport,
    BatchRunConfig,
    ToolAgentEvidence,
    TerminologyConsensusConfig,
    GlossaryGapItem,
    GlossaryGapOccurrence,
    GlossaryGapReport,
    GlossaryGapSummary,
    GlossaryUpdateApplication,
    GlossaryUpdateApplicationItem,
    GlossaryUpdateApplicationSummary,
    GlossaryUpdatePassResult,
    GlossaryUpdatePlan,
    GlossaryUpdatePlanItem,
    GlossaryUpdatePlanSummary,
    ManualEditPlan,
    ManualEditPlanItem,
    ManualEditPlanSummary,
    ManualReviewRecord,
    ManualTextReplacementResult,
    PanelNormalizationItem,
    PanelNormalizationResult,
    PanelChapterReport,
    PanelComparisonRow,
    PanelLine,
    PanelReport,
    PanelReportSummary,
    ProviderLabel,
    ProviderFailureSummary,
    QAReport,
    ReviewQueue,
    ReviewQueueItem,
    ReviewQueueSummary,
    WorkOrderAction,
)
from .glossary import load_glossary
from .package import build_epub_collection, build_txt_collection, verify_epub_artifact, verify_txt_artifact
from .pipeline import LiveProviderFallbackState, run_demo_pipeline
from .providers_llm import inspect_response_cache, openai_compatible_provider_names, required_live_provider_config
from .qa import CHINESE_RE, run_translation_qa
from .story import load_story_config, make_run_id, prepare_run_dir
from .text import extract_panel_segments, split_paragraphs
from .translate import get_judge_provider, get_repair_provider, get_translation_provider
from .utils import sha256_file


COMPLETE_STATUSES = {"packaged", "review_required", "skipped"}
TERMINAL_BATCH_STATUSES = {"packaged", "review_required", "failed", "skipped"}
PROVIDER_ROLE_ORDER = ["translation", "judge", "repair"]
ALIAS_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "that",
    "these",
    "this",
    "those",
    "through",
    "to",
    "with",
}


def normalize_chapter_id(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("Empty chapter id in chapter selection.")
    if not stripped.isdigit():
        raise ValueError(f"Chapter id must be numeric for range selection: {value}")
    return f"{int(stripped):04d}"


def parse_chapter_selection(selection: str) -> list[str]:
    if not selection.strip():
        raise ValueError("A chapter selection is required.")
    chapters: OrderedDict[str, None] = OrderedDict()
    for raw_part in selection.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            raw_start, raw_end = [piece.strip() for piece in part.split("-", 1)]
            start = int(normalize_chapter_id(raw_start))
            end = int(normalize_chapter_id(raw_end))
            if end < start:
                raise ValueError(f"Chapter selection has backwards range: {part}")
            for number in range(start, end + 1):
                chapters[f"{number:04d}"] = None
        else:
            chapters[normalize_chapter_id(part)] = None
    if not chapters:
        raise ValueError("A chapter selection is required.")
    return list(chapters)


def chapter_range_label(chapters: list[str]) -> str:
    if not chapters:
        return "empty"
    return f"{chapters[0]}_{chapters[-1]}"


def _chapter_sort_key(chapter: str) -> tuple[int, int | str]:
    stripped = chapter.strip()
    if stripped.isdigit():
        return (0, int(stripped))
    return (1, stripped)


def _sorted_unique_chapters(chapters: list[str]) -> list[str]:
    return sorted(dict.fromkeys(chapters), key=_chapter_sort_key)


def write_batch_manifest(path: Path, manifest: BatchManifest) -> None:
    manifest.refresh_summary()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def load_batch_manifest(path: Path) -> BatchManifest:
    manifest = BatchManifest.model_validate_json(path.read_text(encoding="utf-8"))
    manifest.refresh_summary()
    return manifest


def build_batch_run_config(
    *,
    provider_mode: str,
    translation_provider_name: str,
    judge_provider_name: str,
    repair_provider_name: str,
    record_cache: bool,
    cache_dir: Path | None,
    model_name: str | None,
    allow_live_provider_fallback: bool = False,
    tool_agent_enabled: bool = False,
    terminology_consensus: TerminologyConsensusConfig | None = None,
) -> BatchRunConfig:
    effective_model = model_name or os.environ.get("AGENTIC_TRANSLATION_MODEL")
    if provider_mode == "offline":
        translation_provider_name = judge_provider_name = repair_provider_name = "offline"
        effective_model = None
    return BatchRunConfig(
        provider_mode=provider_mode,
        translation_provider=translation_provider_name,
        judge_provider=judge_provider_name,
        repair_provider=repair_provider_name,
        record_cache=record_cache,
        cache_dir=str(cache_dir) if cache_dir else None,
        model_name=effective_model,
        allow_live_provider_fallback=allow_live_provider_fallback,
        tool_agent_enabled=tool_agent_enabled,
        terminology_consensus=(
            terminology_consensus
            if terminology_consensus is not None
            else TerminologyConsensusConfig()
        ),
    )


def _required_cache_namespaces(manifest: BatchManifest) -> list[str]:
    if manifest.mode not in {"live", "replay"}:
        return []
    provider_by_role = {
        "translation": manifest.run_config.translation_provider if manifest.run_config else manifest.providers.get("translation", ProviderLabel(provider="offline", model="")).provider,
        "judge": manifest.run_config.judge_provider if manifest.run_config else manifest.providers.get("judge", ProviderLabel(provider="offline", model="")).provider,
        "repair": manifest.run_config.repair_provider if manifest.run_config else manifest.providers.get("repair", ProviderLabel(provider="offline", model="")).provider,
    }
    return [role for role in PROVIDER_ROLE_ORDER if provider_by_role.get(role) != "offline"]


_AGENT_ACTION_ADAPTER = TypeAdapter(AgentAction)


def _canonical_sha256(value: object) -> str:
    """Hash a JSON/Pydantic value using the cache's canonical JSON ordering."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")  # type: ignore[union-attr]
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_artifact_path(root: Path, raw_path: str | Path | None) -> Path | None:
    """Resolve a manifest artifact only when it stays inside ``root``.

    Existing symlink components are rejected even when their resolved target
    would otherwise remain contained.  Proof must describe the exact artifact
    that was recorded, not an artifact that can be swapped through a link.
    """

    if not raw_path:
        return None
    root_input = Path(root)
    if root_input.is_symlink():
        return None
    root_resolved = root_input.resolve(strict=False)
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root_resolved / candidate
    candidate_resolved = candidate.resolve(strict=False)
    try:
        relative = candidate_resolved.relative_to(root_resolved)
    except ValueError:
        return None
    # Reject symlinks in the artifact path, but stop once the candidate's
    # resolved parent reaches the run root so macOS's /var -> /private/var
    # system link does not make every absolute artifact look unsafe.
    ancestor = candidate
    while True:
        if ancestor.is_symlink():
            return None
        if ancestor == root_input or ancestor.resolve(strict=False) == root_resolved:
            break
        parent = ancestor.parent
        if parent == ancestor:
            break
        ancestor = parent
    current = root_resolved
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            return None
    return candidate_resolved


def _safe_external_run_dir(raw_path: str | Path | None) -> Path | None:
    """Resolve a replay source run without following a symlinked run root."""

    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.is_symlink():
        return None
    resolved = candidate.resolve(strict=False)
    if not resolved.is_dir() or (resolved / "batch_manifest.json").is_symlink():
        return None
    if not (resolved / "batch_manifest.json").exists():
        return None
    return resolved


def _tool_episode_path(manifest: BatchManifest, chapter: str) -> Path | None:
    chapter_run = manifest.chapters.get(chapter)
    if chapter_run is None:
        return None
    run_root = Path(manifest.run_dir)
    return _safe_artifact_path(run_root, chapter_run.tool_agent_episode_path)


def _load_tool_episode(
    manifest: BatchManifest,
    chapter: str,
) -> tuple[AgentEpisode | None, list[str], Path | None]:
    """Load and validate one chapter episode, retaining proof diagnostics."""

    chapter_run = manifest.chapters.get(chapter)
    if chapter_run is None:
        return None, [f"{chapter}:chapter is absent from manifest"], None
    raw_path = chapter_run.tool_agent_episode_path
    episode_path = _tool_episode_path(manifest, chapter)
    if raw_path and episode_path is None:
        return None, [f"{chapter}:unsafe tool-agent episode path"], None
    if raw_path and episode_path is not None and not episode_path.exists():
        return None, [f"{chapter}:tool-agent episode is missing: {raw_path}"], episode_path
    if episode_path is None:
        return None, [], None
    try:
        episode = AgentEpisode.model_validate_json(episode_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - malformed evidence is a proof failure, never a crash.
        return None, [f"{chapter}:malformed tool-agent episode ({type(exc).__name__})"], episode_path
    mismatches: list[str] = []
    if episode.story_slug != manifest.story_slug:
        mismatches.append(f"{chapter}:episode story identity {episode.story_slug!r} != {manifest.story_slug!r}")
    if episode.chapter != chapter:
        mismatches.append(f"{chapter}:episode chapter identity {episode.chapter!r} != {chapter!r}")
    if episode.provider_mode != manifest.mode:
        mismatches.append(
            f"{chapter}:episode provider_mode {episode.provider_mode!r} != manifest mode {manifest.mode!r}"
        )
    if episode.final_status is None:
        mismatches.append(f"{chapter}:episode has no final status")
    return episode, mismatches, episode_path


def _load_final_qa_artifact(manifest: BatchManifest, chapter: str) -> QAReport | None:
    chapter_run = manifest.chapters.get(chapter)
    if chapter_run is None:
        return None
    chapter_dir = _safe_chapter_run_dir(manifest, chapter)
    if chapter_dir is None:
        return None
    qa_path = _safe_artifact_path(chapter_dir, "qa_final.json")
    if qa_path is None or not qa_path.exists():
        return None
    try:
        return QAReport.model_validate_json(qa_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_chapter_artifact_path(manifest: BatchManifest, chapter: str, raw_path: str | None) -> Path | None:
    if raw_path:
        return _safe_artifact_path(Path(manifest.run_dir), raw_path)
    chapter_run = manifest.chapters.get(chapter)
    if chapter_run is None:
        return None
    chapter_dir = _safe_artifact_path(Path(manifest.run_dir), chapter_run.chapter_run_dir)
    if chapter_dir is None:
        return None
    return chapter_dir / "translated_final" / f"{chapter}.txt"


def _safe_chapter_run_dir(manifest: BatchManifest, chapter: str) -> Path | None:
    chapter_run = manifest.chapters.get(chapter)
    if chapter_run is None or not chapter_run.chapter_run_dir:
        return None
    chapter_dir = _safe_artifact_path(Path(manifest.run_dir), chapter_run.chapter_run_dir)
    if chapter_dir is None:
        return None
    expected = (Path(manifest.run_dir).resolve(strict=False) / "chapters" / chapter).resolve(strict=False)
    if chapter_dir != expected or not chapter_dir.is_dir():
        return None
    return chapter_dir


def _final_qa_hash(manifest: BatchManifest, chapter: str, episode: AgentEpisode | None) -> str | None:
    chapter_run = manifest.chapters.get(chapter)
    if chapter_run is None:
        return None
    qa_report = _load_final_qa_artifact(manifest, chapter)
    if qa_report is not None:
        return _canonical_sha256(qa_report)
    if episode is not None and episode.final_qa is not None:
        return _canonical_sha256(episode.final_qa)
    if chapter_run.final_findings is None and chapter_run.final_score is None:
        return None
    return _canonical_sha256(
        {
            "total_findings": chapter_run.final_findings,
            "score": chapter_run.final_score,
        }
    )


def _tool_episode_snapshot(
    manifest: BatchManifest,
    chapter: str,
) -> tuple[dict[str, object] | None, list[str]]:
    episode, mismatches, _ = _load_tool_episode(manifest, chapter)
    if episode is None:
        return None, mismatches
    action_sequence: list[dict[str, object]] = []
    patch_decisions: list[dict[str, object]] = []
    sequences: set[int] = set()
    for index, step in enumerate(episode.steps, start=1):
        if step.sequence in sequences or step.sequence != index:
            mismatches.append(f"{chapter}:episode step sequence is not unique/contiguous at {step.sequence}")
        sequences.add(step.sequence)
        try:
            action = _AGENT_ACTION_ADAPTER.validate_python(step.action)
        except Exception as exc:  # noqa: BLE001
            mismatches.append(f"{chapter}:step {step.sequence} persisted action is invalid ({type(exc).__name__})")
            continue
        action_dump = action.model_dump(mode="json")
        action_sequence.append(action_dump)
        if action.tool == "submit_patch":
            if step.qa_before is None or step.qa_after is None:
                mismatches.append(f"{chapter}:step {step.sequence} patch is missing qa_before/qa_after")
            kind = step.observation.kind
            expected_accepted = kind == "patch_accepted"
            if kind not in {"patch_accepted", "patch_rejected"}:
                mismatches.append(f"{chapter}:step {step.sequence} patch has invalid observation kind {kind!r}")
            observed_accepted = step.observation.data.get("accepted")
            if not isinstance(observed_accepted, bool) or observed_accepted is not expected_accepted:
                mismatches.append(f"{chapter}:step {step.sequence} patch decision is internally inconsistent")
            if step.qa_before is not None and step.qa_after is not None:
                expected_values = {
                    "before_findings": step.qa_before.summary.total_findings,
                    "after_findings": step.qa_after.summary.total_findings,
                    "before_score": step.qa_before.score,
                    "after_score": step.qa_after.score,
                }
                for key, expected in expected_values.items():
                    observed = step.observation.data.get(key)
                    if observed != expected:
                        mismatches.append(f"{chapter}:step {step.sequence} patch {key} disagrees with QA")
                if expected_accepted and not (
                    step.qa_after.summary.total_findings < step.qa_before.summary.total_findings
                    or step.qa_after.score > step.qa_before.score
                ):
                    mismatches.append(f"{chapter}:step {step.sequence} accepted patch did not improve QA")
            if not mismatches or not any(item.startswith(f"{chapter}:step {step.sequence}") for item in mismatches):
                patch_decisions.append(
                    {
                        "sequence": step.sequence,
                        "kind": kind,
                        "accepted": expected_accepted,
                        "before_findings": step.qa_before.summary.total_findings if step.qa_before else None,
                        "after_findings": step.qa_after.summary.total_findings if step.qa_after else None,
                    }
                )
    final_path = _safe_chapter_artifact_path(manifest, chapter, manifest.chapters[chapter].final_path)
    if final_path is None or not final_path.exists():
        mismatches.append(f"{chapter}:final translation artifact is missing or unsafe")
        final_text_hash = None
    else:
        final_text_hash = _text_sha256(final_path.read_text(encoding="utf-8"))
    chapter_run = manifest.chapters[chapter]
    chapter_dir = _safe_chapter_run_dir(manifest, chapter)
    if chapter_dir is None:
        mismatches.append(f"{chapter}:chapter_run_dir is missing or unsafe")
    artifact_qa = _load_final_qa_artifact(manifest, chapter)
    qa_path = _safe_artifact_path(chapter_dir, "qa_final.json") if chapter_dir is not None else None
    if chapter_dir is not None and (qa_path is None or not qa_path.exists()):
        mismatches.append(f"{chapter}:final QA artifact is missing or unsafe")
    if qa_path is not None and qa_path.exists() and artifact_qa is None:
        mismatches.append(f"{chapter}:final QA artifact is malformed or not a QAReport")
    if artifact_qa is not None:
        if episode.final_qa is None or _canonical_sha256(artifact_qa) != _canonical_sha256(episode.final_qa):
            mismatches.append(f"{chapter}:final QA artifact does not equal episode final QA")
    if episode.final_qa is not None:
        if chapter_run.final_findings is not None and episode.final_qa.summary.total_findings != chapter_run.final_findings:
            mismatches.append(f"{chapter}:episode final QA finding count disagrees with manifest")
        if chapter_run.final_score is not None and episode.final_qa.score != chapter_run.final_score:
            mismatches.append(f"{chapter}:episode final QA score disagrees with manifest")
    tool_agent_configured = bool(manifest.run_config and manifest.run_config.tool_agent_enabled)
    if tool_agent_configured and chapter_run.tool_agent_final_text_sha256 is None:
        mismatches.append(f"{chapter}:persisted tool-agent final text digest is missing")
    elif final_text_hash is not None and chapter_run.tool_agent_final_text_sha256 is not None:
        if final_text_hash != chapter_run.tool_agent_final_text_sha256:
            mismatches.append(f"{chapter}:final text artifact digest disagrees with persisted agent output digest")
    expected_counters = {
        "tool_agent_final_status": episode.final_status,
        "tool_agent_steps": len(episode.steps),
        "tool_agent_initial_findings": episode.initial_qa.summary.total_findings,
        "tool_agent_final_findings": episode.final_qa.summary.total_findings if episode.final_qa is not None else 0,
        "tool_agent_accepted_patches": sum(step.observation.kind == "patch_accepted" for step in episode.steps),
        "tool_agent_rejected_patches": sum(step.observation.kind == "patch_rejected" for step in episode.steps),
    }
    for field_name, expected in expected_counters.items():
        if getattr(chapter_run, field_name) != expected:
            mismatches.append(
                f"{chapter}:manifest {field_name} {getattr(chapter_run, field_name)!r} != episode {expected!r}"
            )
    final_findings = episode.final_qa.summary.total_findings if episode.final_qa is not None else 0
    expected_status = (
        "packaged"
        if episode.final_status == "verified" and final_findings == 0
        else "failed"
        if episode.final_status == "failed"
        else "review_required"
    )
    if chapter_run.status != expected_status:
        mismatches.append(
            f"{chapter}:manifest chapter status {chapter_run.status!r} != episode-derived status {expected_status!r}"
        )
    snapshot = {
        "actions": action_sequence,
        "patches": patch_decisions,
        "final_text": final_text_hash,
        "final_qa": _final_qa_hash(manifest, chapter, episode),
        "final_status": episode.final_status,
    }
    return snapshot, mismatches


def build_tool_agent_evidence(manifest: BatchManifest) -> ToolAgentEvidence:
    """Verify all persisted tool-agent episodes and their cache records."""

    chapters_with_paths = [chapter for chapter, run in manifest.chapters.items() if run.tool_agent_episode_path]
    manifest_calls = [
        (chapter, call)
        for chapter, run in manifest.chapters.items()
        for call in run.provider_calls
        if call.namespace == "agent_action" and call.provider != "offline"
    ]
    applicable = bool(chapters_with_paths or manifest_calls)
    if not applicable:
        return ToolAgentEvidence(applicable=False, proof_ready=True, replay_cache_ready=True)

    mismatches: list[str] = []
    episode_by_chapter: dict[str, AgentEpisode] = {}
    snapshots: dict[str, dict[str, object]] = {}
    observed_episode_calls: list[tuple[str, object]] = []
    observed_actions = 0
    verified_episodes = 0
    verified_actions = 0
    verified_patch_records = 0

    for chapter in sorted(set(chapters_with_paths) | {chapter for chapter, _ in manifest_calls}):
        episode, episode_errors, _ = _load_tool_episode(manifest, chapter)
        mismatches.extend(episode_errors)
        if episode is None:
            continue
        episode_by_chapter[chapter] = episode
        snapshot, snapshot_errors = _tool_episode_snapshot(manifest, chapter)
        mismatches.extend(snapshot_errors)
        if snapshot is not None:
            snapshots[chapter] = snapshot
        step_calls = []
        for step in episode.steps:
            if step.provider_call is not None:
                step_calls.append((chapter, step.provider_call))
                if step.provider_call.namespace != "agent_action":
                    mismatches.append(f"{chapter}:step {step.sequence} has non-agent provider namespace")
        observed_episode_calls.extend(step_calls)
        observed_actions += len(step_calls)
        if not episode_errors and not snapshot_errors:
            verified_episodes += 1

    manifest_call_keys = Counter(
        (
            chapter,
            call.namespace,
            call.payload_sha256,
            call.response_sha256,
            call.cache_file,
            call.provider,
            call.model,
        )
        for chapter, call in manifest_calls
    )
    if manifest.mode == "replay":
        for chapter, call in manifest_calls:
            if not call.cache_hit:
                mismatches.append(f"{chapter}:replay agent_action manifest call has cache_hit=False")
    episode_call_keys = Counter(
        (
            chapter,
            call.namespace,
            call.payload_sha256,
            call.response_sha256,
            call.cache_file,
            call.provider,
            call.model,
        )
        for chapter, call in observed_episode_calls
    )
    for label, counts in (("manifest", manifest_call_keys), ("episode", episode_call_keys)):
        duplicates = [key for key, count in counts.items() if count > 1]
        if duplicates:
            mismatches.append(f"{label} contains duplicate agent_action provider call(s): {duplicates!r}")
    if manifest_call_keys != episode_call_keys:
        missing = list((episode_call_keys - manifest_call_keys).elements())
        orphaned = list((manifest_call_keys - episode_call_keys).elements())
        if missing:
            mismatches.append("episode has orphaned agent_action provider call(s): " + repr(missing))
        if orphaned:
            mismatches.append("manifest has missing agent_action episode call(s): " + repr(orphaned))

    cache_dir = Path(manifest.run_config.cache_dir) if manifest.run_config and manifest.run_config.cache_dir else None
    cache_report = inspect_response_cache(cache_dir) if cache_dir and cache_dir.exists() and cache_dir.is_dir() else None
    cache_entries: dict[tuple[str, str], list[object]] = {}
    if cache_report:
        for entry in cache_report.entries:
            cache_entries.setdefault((entry.namespace, entry.payload_sha256), []).append(entry)
    for chapter, call in observed_episode_calls:
        if manifest.mode == "replay" and not call.cache_hit:
            mismatches.append(f"{chapter}:replay agent_action call has cache_hit=False")
        entries = cache_entries.get((call.namespace, call.payload_sha256), [])
        if len(entries) != 1:
            mismatches.append(f"{chapter}:step agent_action cache index entry count is {len(entries)}")
            continue
        entry = entries[0]
        if (
            entry.response_sha256 != call.response_sha256
            or entry.cache_file != call.cache_file
            or entry.provider != call.provider
            or entry.model != call.model
        ):
            mismatches.append(f"{chapter}:agent_action cache metadata mismatch for {call.payload_sha256}")
            continue
        if cache_dir is None:
            mismatches.append(f"{chapter}:agent_action cache directory is unavailable")
            continue
        cache_path = _safe_artifact_path(cache_dir, entry.cache_file)
        if cache_path is None or not cache_path.exists():
            mismatches.append(f"{chapter}:agent_action cache file is unsafe or missing: {entry.cache_file}")
            continue
        try:
            cached_response = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_action = _AGENT_ACTION_ADAPTER.validate_python(cached_response)
        except Exception as exc:  # noqa: BLE001
            mismatches.append(f"{chapter}:cached agent_action response is invalid ({type(exc).__name__})")
            continue
        episode = episode_by_chapter.get(chapter)
        if episode is None:
            continue
        matching_steps = [step for step in episode.steps if step.provider_call == call]
        if len(matching_steps) != 1:
            mismatches.append(f"{chapter}:agent_action call does not map to exactly one episode step")
            continue
        step = matching_steps[0]
        try:
            persisted_action = _AGENT_ACTION_ADAPTER.validate_python(step.action)
        except Exception as exc:  # noqa: BLE001
            mismatches.append(f"{chapter}:persisted agent_action is invalid ({type(exc).__name__})")
            continue
        if cached_action.model_dump(mode="json") != persisted_action.model_dump(mode="json"):
            mismatches.append(f"{chapter}:cached action differs from persisted action at step {step.sequence}")
            continue
        verified_actions += 1
        if persisted_action.tool == "submit_patch" and step.qa_before is not None and step.qa_after is not None:
            if any(item.startswith(f"{chapter}:step {step.sequence}") for item in mismatches):
                continue
            verified_patch_records += 1

    replay_cache_ready = (
        cache_report is not None
        and cache_report.integrity_passed
        and bool(observed_episode_calls)
        and verified_actions == len(observed_episode_calls)
        and not mismatches
    )

    final_text_hashes = {
        chapter: str(snapshot["final_text"])
        for chapter, snapshot in snapshots.items()
        if snapshot.get("final_text") is not None
    }
    final_qa_hashes = {
        chapter: str(snapshot["final_qa"])
        for chapter, snapshot in snapshots.items()
        if snapshot.get("final_qa") is not None
    }

    action_sequence_matches: bool | None = None
    patch_decisions_match: bool | None = None
    final_text_matches: bool | None = None
    final_qa_matches: bool | None = None
    final_status_matches: bool | None = None
    source_dir = _safe_external_run_dir(manifest.replay_source_run_dir)
    if manifest.replay_source_run_dir and source_dir is None:
        mismatches.append("unsafe or missing replay source run directory")
    if source_dir is not None:
        try:
            source_manifest = load_batch_manifest(source_dir / "batch_manifest.json")
        except Exception as exc:  # noqa: BLE001
            source_manifest = None
            mismatches.append(f"replay source manifest is malformed ({type(exc).__name__})")
        if source_manifest is not None:
            source_snapshots: dict[str, dict[str, object]] = {}
            for chapter in manifest.chapters:
                source_snapshot, source_errors = _tool_episode_snapshot(source_manifest, chapter)
                mismatches.extend(f"source:{item}" for item in source_errors)
                if source_snapshot is not None:
                    source_snapshots[chapter] = source_snapshot
            compared_chapters = sorted(set(source_snapshots) | set(snapshots))
            action_sequence_matches = all(
                source_snapshots.get(chapter, {}).get("actions") == snapshots.get(chapter, {}).get("actions")
                for chapter in compared_chapters
            )
            patch_decisions_match = all(
                source_snapshots.get(chapter, {}).get("patches") == snapshots.get(chapter, {}).get("patches")
                for chapter in compared_chapters
            )
            final_text_matches = all(
                source_snapshots.get(chapter, {}).get("final_text") == snapshots.get(chapter, {}).get("final_text")
                for chapter in compared_chapters
            )
            final_qa_matches = all(
                source_snapshots.get(chapter, {}).get("final_qa") == snapshots.get(chapter, {}).get("final_qa")
                for chapter in compared_chapters
            )
            final_status_matches = all(
                source_snapshots.get(chapter, {}).get("final_status") == snapshots.get(chapter, {}).get("final_status")
                for chapter in compared_chapters
            )
            if not all(
                value is True
                for value in (
                    action_sequence_matches,
                    patch_decisions_match,
                    final_text_matches,
                    final_qa_matches,
                    final_status_matches,
                )
            ):
                mismatches.append("replay source and replay tool-agent outcomes differ")

    proof_ready = (
        bool(episode_by_chapter)
        and verified_episodes == len(episode_by_chapter)
        and verified_actions == observed_actions == len(manifest_calls)
        and replay_cache_ready
        and not mismatches
        and all(value in {None, True} for value in (action_sequence_matches, patch_decisions_match, final_text_matches, final_qa_matches, final_status_matches))
    )
    return ToolAgentEvidence(
        applicable=True,
        episodes_observed=len(episode_by_chapter),
        verified_episodes=verified_episodes,
        observed_actions=observed_actions,
        verified_actions=verified_actions,
        verified_patch_records=verified_patch_records,
        verified_cache_records=verified_actions,
        replay_cache_ready=replay_cache_ready,
        action_sequence_matches=action_sequence_matches,
        patch_decisions_match=patch_decisions_match,
        final_text_matches=final_text_matches,
        final_qa_matches=final_qa_matches,
        final_status_matches=final_status_matches,
        final_text_sha256=final_text_hashes,
        final_qa_sha256=final_qa_hashes,
        mismatches=mismatches,
        proof_ready=proof_ready,
    )


def build_agentic_evidence(manifest: BatchManifest) -> AgenticEvidence:
    cache_dir = manifest.run_config.cache_dir if manifest.run_config else None
    cache_available = False
    cache_entries = 0
    cache_namespaces: dict[str, int] = {}
    cache_integrity_passed = False
    cache_valid_entries = 0
    cache_invalid_entries = 0
    cache_integrity_issues: list[str] = []
    cache_entries_by_call: dict[tuple[str, str], tuple[str, str, str | None, str | None]] = {}
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_available = cache_path.exists() and cache_path.is_dir()
        if cache_available:
            cache_report = inspect_response_cache(cache_path)
            cache_entries = cache_report.total_entries
            cache_namespaces = cache_report.by_namespace
            cache_integrity_passed = cache_report.integrity_passed
            cache_valid_entries = cache_report.valid_entries
            cache_invalid_entries = cache_report.invalid_entries
            cache_integrity_issues = [issue.message for issue in cache_report.integrity_issues]
            cache_entries_by_call = {
                (entry.namespace, entry.payload_sha256): (
                    entry.response_sha256,
                    entry.cache_file,
                    entry.provider,
                    entry.model,
                )
                for entry in cache_report.entries
            }
    cache_required_namespaces = _required_cache_namespaces(manifest)
    cache_missing_namespaces = [
        namespace for namespace in cache_required_namespaces if cache_namespaces.get(namespace, 0) == 0
    ]
    provider_call_records = [
        (chapter_id, call)
        for chapter_id, chapter_run in manifest.chapters.items()
        for call in chapter_run.provider_calls
        if call.namespace in cache_required_namespaces and call.provider != "offline"
    ]
    cache_missing_call_records: list[str] = []
    cache_metadata_mismatches: list[str] = []
    cache_verified_call_records = 0
    cache_response_by_call: dict[tuple[str, str, str], dict[str, object]] = {}
    for chapter_id, call in provider_call_records:
        cached = cache_entries_by_call.get((call.namespace, call.payload_sha256))
        if cached is None:
            cache_missing_call_records.append(f"{chapter_id}:{call.namespace}:{call.payload_sha256}")
            continue
        cached_response_sha, cached_file, cached_provider, cached_model = cached
        if (cached_response_sha, cached_file) != (call.response_sha256, call.cache_file):
            cache_missing_call_records.append(f"{chapter_id}:{call.namespace}:{call.payload_sha256}")
            continue
        if cached_provider != call.provider:
            cache_metadata_mismatches.append(
                f"{chapter_id}:{call.namespace}:provider {call.provider}!={cached_provider or 'missing'}"
            )
            continue
        if cached_model != call.model:
            cache_metadata_mismatches.append(
                f"{chapter_id}:{call.namespace}:{call.model or 'missing'}!={cached_model or 'missing'}"
            )
            continue
        if cached == (call.response_sha256, call.cache_file, call.provider, call.model):
            cache_verified_call_records += 1
            if cache_dir:
                cache_file_path = Path(cache_dir) / call.cache_file
                if cache_file_path.exists():
                    try:
                        cache_response_by_call[(call.namespace, call.payload_sha256, call.response_sha256)] = json.loads(
                            cache_file_path.read_text(encoding="utf-8")
                        )
                    except json.JSONDecodeError:
                        pass
    replay_cache_ready = (
        bool(cache_required_namespaces)
        and cache_available
        and cache_integrity_passed
        and not cache_missing_namespaces
        and bool(provider_call_records)
        and not cache_missing_call_records
        and not cache_metadata_mismatches
    )
    configured_model_roles = [
        role
        for role in PROVIDER_ROLE_ORDER
        if manifest.mode in {"live", "replay"}
        and role in manifest.providers
        and manifest.providers[role].provider != "offline"
    ]
    configured_provider_by_role = {
        "translation": manifest.run_config.translation_provider if manifest.run_config else manifest.providers.get("translation", ProviderLabel(provider="offline", model="")).provider,
        "judge": manifest.run_config.judge_provider if manifest.run_config else manifest.providers.get("judge", ProviderLabel(provider="offline", model="")).provider,
        "repair": manifest.run_config.repair_provider if manifest.run_config else manifest.providers.get("repair", ProviderLabel(provider="offline", model="")).provider,
    }
    judge_provider_configured = (
        manifest.mode in {"live", "replay"}
        and configured_provider_by_role.get("judge") != "offline"
    )
    candidate_selection_decisions = [
        (chapter_id, decision.selected_candidate_id)
        for chapter_id, chapter_run in manifest.chapters.items()
        for decision in chapter_run.repair_decisions
        if decision.strategy == "candidate_selection"
    ]
    candidate_selection_repairs = len(candidate_selection_decisions)
    verified_candidate_selection_records = 0
    candidate_selection_mismatches: list[str] = []
    provider_calls_by_chapter = {
        chapter_id: [
            call
            for call in chapter_run.provider_calls
            if call.provider != "offline"
        ]
        for chapter_id, chapter_run in manifest.chapters.items()
    }
    for chapter_id, selected_candidate_id in candidate_selection_decisions:
        if not judge_provider_configured:
            continue
        if not selected_candidate_id:
            candidate_selection_mismatches.append(f"{chapter_id}:candidate_selection decision did not record a selected candidate")
            continue
        judge_calls = [
            call
            for call in provider_calls_by_chapter.get(chapter_id, [])
            if call.namespace == "judge" or call.role == "judge"
        ]
        observed_selected_ids: list[str] = []
        matched = False
        for call in judge_calls:
            response = cache_response_by_call.get((call.namespace, call.payload_sha256, call.response_sha256))
            if not response:
                continue
            observed = response.get("selected_candidate_id")
            if isinstance(observed, str):
                observed_selected_ids.append(observed)
                if observed == selected_candidate_id:
                    matched = True
        if matched:
            verified_candidate_selection_records += 1
        elif observed_selected_ids:
            candidate_selection_mismatches.append(
                f"{chapter_id}:cached judge selected candidate did not match decision "
                f"{selected_candidate_id} (observed {', '.join(observed_selected_ids)})"
            )
        elif judge_calls:
            candidate_selection_mismatches.append(
                f"{chapter_id}:judge provider call was recorded but no cached selected_candidate_id verified decision {selected_candidate_id}"
            )
        else:
            candidate_selection_mismatches.append(
                f"{chapter_id}:candidate_selection decision has no recorded judge provider call"
            )
    model_backed_patch_attempts = sum(
        1
        for chapter_run in manifest.chapters.values()
        for attempt in chapter_run.patch_attempts
        if attempt.strategy == "candidate_selection"
        and manifest.mode in {"live", "replay"}
        and manifest.providers.get("repair") is not None
        and manifest.providers["repair"].provider != "offline"
    )
    verified_repair_patch_records = 0
    repair_patch_mismatches: list[str] = []
    model_backed_repair_attempts = [
        (chapter_id, attempt)
        for chapter_id, chapter_run in manifest.chapters.items()
        for attempt in chapter_run.patch_attempts
        if attempt.strategy == "candidate_selection"
        and attempt.patch is not None
        and manifest.mode in {"live", "replay"}
        and manifest.providers.get("repair") is not None
        and manifest.providers["repair"].provider != "offline"
    ]
    for chapter_id, attempt in model_backed_repair_attempts:
        patch = attempt.patch
        if patch is None:
            continue
        repair_calls = [
            call
            for call in provider_calls_by_chapter.get(chapter_id, [])
            if call.namespace == "repair" or call.role == "repair"
        ]
        observed_patch_labels: list[str] = []
        matched = False
        for call in repair_calls:
            response = cache_response_by_call.get((call.namespace, call.payload_sha256, call.response_sha256))
            if not response:
                continue
            response_patch_type = response.get("patch_type")
            response_old_text = response.get("old_text")
            response_new_text = response.get("new_text")
            response_paragraph_index = response.get("paragraph_index")
            observed_patch_labels.append(
                f"{response_patch_type}:{response_old_text!r}->{response_new_text!r}@{response_paragraph_index}"
            )
            if (
                response_patch_type == patch.patch_type
                and response_old_text == patch.old_text
                and response_new_text == patch.new_text
                and response_paragraph_index == patch.paragraph_index
            ):
                matched = True
        if matched:
            verified_repair_patch_records += 1
        elif observed_patch_labels:
            repair_patch_mismatches.append(
                f"{chapter_id}:cached repair patch did not match accepted patch "
                f"{patch.patch_type}:{patch.old_text!r}->{patch.new_text!r}@{patch.paragraph_index} "
                f"(observed {', '.join(observed_patch_labels)})"
            )
        elif repair_calls:
            repair_patch_mismatches.append(
                f"{chapter_id}:repair provider call was recorded but no cached patch verified accepted patch {patch.patch_id}"
            )
        else:
            repair_patch_mismatches.append(
                f"{chapter_id}:candidate_selection patch attempt has no recorded repair provider call"
            )
    observed_agentic_roles: list[str] = []
    all_candidate_selection_verified = (
        bool(candidate_selection_decisions)
        and verified_candidate_selection_records == len(candidate_selection_decisions)
        and not candidate_selection_mismatches
    )
    model_provider_call_roles = {
        role
        for _, call in provider_call_records
        for role in {call.role, call.namespace}
        if role
    }
    if all_candidate_selection_verified and "judge" in model_provider_call_roles:
        observed_agentic_roles.append("judge")
    all_repair_patches_verified = (
        bool(model_backed_repair_attempts)
        and verified_repair_patch_records == len(model_backed_repair_attempts)
        and not repair_patch_mismatches
    )
    if all_repair_patches_verified and "repair" in model_provider_call_roles:
        observed_agentic_roles.append("repair")

    if observed_agentic_roles:
        role_text = "/".join(observed_agentic_roles)
        reason = f"Observed model-backed {role_text} during candidate_selection repair."
    elif manifest.mode == "offline":
        reason = "offline mode is deterministic; no model-backed agentic claim."
    elif not configured_model_roles:
        reason = f"{manifest.mode} mode has no non-offline providers configured."
    elif configured_model_roles == ["translation"]:
        if provider_call_records:
            reason = (
                "Model-backed translation provider calls were recorded, but agentic claim requires "
                "verified model-backed judge or repair work."
            )
        else:
            reason = (
                "Model-backed translation provider configured, but no recorded model-backed provider calls "
                "were observed."
            )
    elif not provider_call_records:
        reason = "Model-backed provider configured, but no recorded model-backed provider calls were observed."
    elif not candidate_selection_repairs:
        reason = "Model-backed provider calls were recorded, but no candidate_selection repair decision was observed."
    elif candidate_selection_mismatches:
        reason = "candidate_selection repair decision selected candidate did not match verified cached judge response."
    elif repair_patch_mismatches:
        reason = "candidate_selection repair patch did not match verified cached repair response."
    else:
        reason = "candidate_selection repairs were observed, but no cached judge/repair response verified the selected work."

    return AgenticEvidence(
        mode=manifest.mode,
        configured_model_roles=configured_model_roles,
        observed_agentic_roles=observed_agentic_roles,
        candidate_selection_repairs=candidate_selection_repairs,
        model_backed_patch_attempts=model_backed_patch_attempts,
        cache_dir=cache_dir,
        cache_available=cache_available,
        cache_entries=cache_entries,
        cache_namespaces=cache_namespaces,
        cache_required_namespaces=cache_required_namespaces,
        cache_missing_namespaces=cache_missing_namespaces,
        cache_integrity_passed=cache_integrity_passed,
        cache_valid_entries=cache_valid_entries,
        cache_invalid_entries=cache_invalid_entries,
        cache_integrity_issues=cache_integrity_issues,
        provider_call_records=len(provider_call_records),
        cache_verified_call_records=cache_verified_call_records,
        cache_missing_call_records=cache_missing_call_records,
        cache_metadata_mismatches=cache_metadata_mismatches,
        verified_candidate_selection_records=verified_candidate_selection_records,
        candidate_selection_mismatches=candidate_selection_mismatches,
        verified_repair_patch_records=verified_repair_patch_records,
        repair_patch_mismatches=repair_patch_mismatches,
        replay_cache_ready=replay_cache_ready,
        agentic_claim_supported=bool(observed_agentic_roles),
        reason=reason,
    )


def _provider_failure_role(message: str) -> str | None:
    lower = message.lower()
    for role in PROVIDER_ROLE_ORDER:
        if f"live {role} provider failed" in lower:
            return role
        if f"previous live {role} provider failure" in lower:
            return role
        if f"skipped live {role}" in lower:
            return role
    if "live provider call failed" in lower:
        return "unknown"
    if "insufficient balance" in lower and "provider" in lower:
        return "unknown"
    return None


def _failure_used_fallback(message: str) -> bool:
    lower = message.lower()
    return (
        "fell back to offline" in lower
        or "used offline" in lower
        or "skipped live" in lower
        or "previous live" in lower
    )


def _parse_attempt_provider_failure_label(provider_text: str) -> tuple[str | None, str | None]:
    if "=" not in provider_text:
        return (provider_text or None, None)
    for part in provider_text.split(";"):
        if "=" not in part:
            continue
        role, provider = [piece.strip() for piece in part.split("=", 1)]
        if role in PROVIDER_ROLE_ORDER and provider and provider != "offline":
            return provider, None
    return None, None


def _provider_label_for_failure(
    manifest: BatchManifest,
    chapter_run: object,
    role: str,
) -> tuple[str | None, str | None]:
    if role in PROVIDER_ROLE_ORDER:
        label = manifest.providers.get(role)
        if label:
            return label.provider, label.model
    for attempt in reversed(getattr(chapter_run, "attempts", [])):
        provider, model = _parse_attempt_provider_failure_label(attempt.provider)
        if provider and provider != "offline":
            return provider, model or attempt.model
    for provider_role in PROVIDER_ROLE_ORDER:
        label = manifest.providers.get(provider_role)
        if label and label.provider != "offline":
            return label.provider, label.model
    return None, None


def _provider_failure_summaries(manifest: BatchManifest) -> list[ProviderFailureSummary]:
    failures: list[ProviderFailureSummary] = []
    seen: set[tuple[str, str, str]] = set()
    for chapter, chapter_run in manifest.chapters.items():
        messages: list[str] = []
        messages.extend(attempt.reason for attempt in chapter_run.patch_attempts)
        if chapter_run.error:
            messages.append(chapter_run.error)
        messages.extend(attempt.message for attempt in chapter_run.attempts if attempt.message)
        for message in messages:
            role = _provider_failure_role(message)
            if role is None:
                continue
            key = (chapter, role, message)
            if key in seen:
                continue
            seen.add(key)
            provider, model = _provider_label_for_failure(manifest, chapter_run, role)
            failures.append(
                ProviderFailureSummary(
                    chapter=chapter,
                    role=role,
                    provider=provider,
                    model=model,
                    reason=message,
                    fallback_used=_failure_used_fallback(message),
                )
            )
    return failures


def build_batch_inspection_report(manifest: BatchManifest, *, allow_review_required: bool = False) -> BatchInspectionReport:
    manifest.refresh_summary()
    blockers: list[BatchInspectionBlocker] = []
    for chapter, chapter_run in manifest.chapters.items():
        if chapter_run.status not in TERMINAL_BATCH_STATUSES:
            blockers.append(
                BatchInspectionBlocker(
                    blocker_type="incomplete",
                    chapter=chapter,
                    status=chapter_run.status,
                    message=f"Chapter is incomplete: {chapter_run.status}.",
                )
            )
        elif chapter_run.status == "failed":
            blockers.append(
                BatchInspectionBlocker(
                    blocker_type="failed",
                    chapter=chapter,
                    status=chapter_run.status,
                    message=chapter_run.error or "Chapter failed.",
                )
            )
        elif chapter_run.status == "review_required" and not allow_review_required:
            findings = chapter_run.final_findings
            message = "Final QA findings remain." if findings is None else f"{findings} final QA finding(s) remain."
            blockers.append(
                BatchInspectionBlocker(
                    blocker_type="review_required",
                    chapter=chapter,
                    status=chapter_run.status,
                    message=message,
                )
            )
    if manifest.artifact_qa and not manifest.artifact_qa.passed:
        for failure in manifest.artifact_qa.failures:
            blockers.append(
                BatchInspectionBlocker(
                    blocker_type="artifact_qa",
                    status="fail",
                    message=failure,
                )
            )
    return BatchInspectionReport(
        run_id=manifest.run_id,
        story_slug=manifest.story_slug,
        run_dir=manifest.run_dir,
        ready_for_delivery=not blockers,
        blocker_count=len(blockers),
        blockers=blockers,
        summary=manifest.summary,
        artifact_qa=manifest.artifact_qa,
        artifacts=manifest.artifacts,
        agentic_evidence=build_agentic_evidence(manifest),
        provider_failures=_provider_failure_summaries(manifest),
        run_config=manifest.run_config,
    )


def build_batch_proof_report(manifest: BatchManifest) -> BatchProofReport:
    inspection = build_batch_inspection_report(manifest)
    tool_agent_evidence = build_tool_agent_evidence(manifest)
    # A clean/non-agent run is not required to produce tool-agent evidence.
    # When episodes are present, their dedicated proof is the authoritative
    # agentic gate even if the older candidate-selection evidence is n/a.
    agentic_gate = inspection.agentic_evidence.agentic_claim_supported or (
        tool_agent_evidence.applicable and tool_agent_evidence.proof_ready
    )
    replayable_gate = inspection.agentic_evidence.replay_cache_ready or (
        tool_agent_evidence.applicable and tool_agent_evidence.replay_cache_ready
    )
    gates = {
        "delivery": inspection.ready_for_delivery,
        "agentic": agentic_gate,
        "replayable": replayable_gate,
    }
    # Preserve the legacy three-key proof JSON for runs that predate the
    # integrated tool-agent flag.  Opted-in runs expose the explicit n/a=true
    # tool-agent gate even when no episode was needed for a clean chapter.
    if tool_agent_evidence.applicable or (
        manifest.run_config is not None and manifest.run_config.tool_agent_enabled
    ):
        gates["tool_agent"] = tool_agent_evidence.proof_ready if tool_agent_evidence.applicable else True
    blockers: list[str] = []
    if not gates["delivery"]:
        blockers.extend(f"delivery:{blocker.blocker_type}:{blocker.message}" for blocker in inspection.blockers)
    if not gates["agentic"]:
        blockers.append(f"agentic:{inspection.agentic_evidence.reason}")
    if not gates["replayable"]:
        replay_details: list[str] = []
        if inspection.agentic_evidence.cache_missing_namespaces:
            replay_details.append("missing namespaces " + ", ".join(inspection.agentic_evidence.cache_missing_namespaces))
        if inspection.agentic_evidence.cache_missing_call_records:
            replay_details.append("missing call records " + ", ".join(inspection.agentic_evidence.cache_missing_call_records))
        if inspection.agentic_evidence.cache_metadata_mismatches:
            replay_details.append("metadata mismatch " + ", ".join(inspection.agentic_evidence.cache_metadata_mismatches))
        if inspection.agentic_evidence.cache_integrity_issues:
            replay_details.append("integrity issues " + "; ".join(inspection.agentic_evidence.cache_integrity_issues))
        if not replay_details:
            replay_details.append("recorded provider calls do not prove replayability")
        blockers.append("replayable:" + "; ".join(replay_details))
    if gates.get("tool_agent") is False:
        details = "; ".join(tool_agent_evidence.mismatches) or "tool-agent evidence did not verify"
        blockers.append("tool_agent:" + details)
    return BatchProofReport(
        run_id=inspection.run_id,
        story_slug=inspection.story_slug,
        run_dir=inspection.run_dir,
        proof_passed=all(gates.values()),
        gates=gates,
        blockers=blockers,
        inspection=inspection,
        tool_agent_evidence=tool_agent_evidence,
    )


def render_batch_proof_markdown(report: BatchProofReport) -> str:
    gate_lines = "\n".join(
        f"- {name}: {'pass' if passed else 'fail'}"
        for name, passed in report.gates.items()
    )
    blocker_lines = "\n".join(f"- {blocker}" for blocker in report.blockers) or "- none"
    evidence = report.inspection.agentic_evidence
    tool_evidence = report.tool_agent_evidence
    candidate_mismatch_lines = "\n".join(f"- {mismatch}" for mismatch in evidence.candidate_selection_mismatches) or "- none"
    repair_mismatch_lines = "\n".join(f"- {mismatch}" for mismatch in evidence.repair_patch_mismatches) or "- none"
    return (
        "\n".join(
            [
                f"# Agentic Proof: {report.run_id}",
                "",
                f"- Story: `{report.story_slug}`",
                f"- Proof passed: `{str(report.proof_passed).lower()}`",
                "",
                "## Gates",
                gate_lines,
                "",
                "## Blockers",
                blocker_lines,
                "",
                "## Evidence",
                f"- Observed agentic roles: {', '.join(evidence.observed_agentic_roles) if evidence.observed_agentic_roles else 'none'}",
                f"- Provider call records: {evidence.provider_call_records}",
                f"- Verified cache call records: {evidence.cache_verified_call_records}",
                f"- Verified candidate selections: {evidence.verified_candidate_selection_records}/{evidence.candidate_selection_repairs}",
                f"- Verified repair patches: {evidence.verified_repair_patch_records}/{evidence.model_backed_patch_attempts}",
                f"- Replay cache ready: `{str(evidence.replay_cache_ready).lower()}`",
                f"- Reason: {evidence.reason}",
                "",
                "## Candidate Selection Mismatches",
                candidate_mismatch_lines,
                "",
                "## Repair Patch Mismatches",
                repair_mismatch_lines,
                "",
                "## Tool-Agent Evidence",
                f"- Applicable: `{str(tool_evidence.applicable).lower()}`",
                f"- Episodes: {tool_evidence.verified_episodes}/{tool_evidence.episodes_observed}",
                f"- Actions: {tool_evidence.verified_actions}/{tool_evidence.observed_actions}",
                f"- Verified patch records: {tool_evidence.verified_patch_records}",
                f"- Replay cache ready: `{str(tool_evidence.replay_cache_ready).lower()}`",
                f"- Action sequence matches: `{str(tool_evidence.action_sequence_matches).lower()}`",
                f"- Patch decisions match: `{str(tool_evidence.patch_decisions_match).lower()}`",
                f"- Final text matches: `{str(tool_evidence.final_text_matches).lower()}`",
                f"- Final QA matches: `{str(tool_evidence.final_qa_matches).lower()}`",
                f"- Final status matches: `{str(tool_evidence.final_status_matches).lower()}`",
                "- Mismatches:",
                "\n".join(f"  - {mismatch}" for mismatch in tool_evidence.mismatches) or "  - none",
            ]
        )
        + "\n"
    )


def write_batch_proof_artifacts(run_dir: Path, manifest: BatchManifest) -> BatchProofReport:
    manifest.artifacts["agentic_proof_json"] = "agentic_proof.json"
    manifest.artifacts["agentic_proof_markdown"] = "agentic_proof.md"
    report = build_batch_proof_report(manifest)
    (run_dir / "agentic_proof.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "agentic_proof.md").write_text(render_batch_proof_markdown(report), encoding="utf-8")
    write_batch_manifest(run_dir / "batch_manifest.json", manifest)
    return report


def _review_summary(items: list[ReviewQueueItem]) -> ReviewQueueSummary:
    by_check: dict[str, int] = {}
    by_chapter: dict[str, int] = {}
    for item in items:
        by_check[item.check_id] = by_check.get(item.check_id, 0) + 1
        by_chapter[item.chapter] = by_chapter.get(item.chapter, 0) + 1
    chapters = list(by_chapter)
    return ReviewQueueSummary(
        total_items=len(items),
        by_check=by_check,
        by_chapter=by_chapter,
        chapters=chapters,
        chapter_selection=",".join(chapters),
    )


def _paragraph_context(text: str, paragraph_index: int | None, *, radius: int = 1) -> str | None:
    if paragraph_index is None:
        return None
    paragraphs = split_paragraphs(text)
    if paragraph_index < 0 or paragraph_index >= len(paragraphs):
        return None
    start = max(0, paragraph_index - radius)
    end = min(len(paragraphs), paragraph_index + radius + 1)
    return "\n\n".join(paragraphs[start:end])


def _window_context(text: str, needles: list[str | None], *, max_chars: int = 800) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    for needle in needles:
        if not needle:
            continue
        index = stripped.find(needle)
        if index == -1:
            continue
        start = max(0, index - max_chars // 2)
        end = min(len(stripped), index + len(needle) + max_chars // 2)
        prefix = "[...]" if start > 0 else ""
        suffix = "[...]" if end < len(stripped) else ""
        return prefix + stripped[start:end].strip() + suffix
    return stripped[:max_chars].rstrip() + ("[...]" if len(stripped) > max_chars else "")


def _aligned_window_context(
    *,
    source_text: str,
    final_text: str,
    source_needle: str | None,
    max_chars: int = 800,
) -> str | None:
    if not source_needle or not source_text.strip() or not final_text.strip():
        return None
    source_index = source_text.find(source_needle)
    if source_index == -1:
        return None
    ratio = source_index / max(len(source_text), 1)
    final_center = int(ratio * len(final_text))
    start = max(0, final_center - max_chars // 2)
    end = min(len(final_text), final_center + max_chars // 2)
    prefix = "[...]" if start > 0 else ""
    suffix = "[...]" if end < len(final_text) else ""
    return prefix + final_text[start:end].strip() + suffix


def _read_optional_text(path: str | None) -> str | None:
    if not path:
        return None
    text_path = Path(path)
    if not text_path.exists():
        return None
    return text_path.read_text(encoding="utf-8")


def _chapter_run_dir(run_dir: Path, chapter_run, chapter: str) -> Path:
    return Path(chapter_run.chapter_run_dir) if chapter_run.chapter_run_dir else run_dir / "chapters" / chapter


def _chapter_source_path(run_dir: Path, chapter_run, chapter: str, fallback_source_dir: Path | None = None) -> Path:
    chapter_run_dir = _chapter_run_dir(run_dir, chapter_run, chapter)
    run_local_source = chapter_run_dir / "source" / f"{chapter}.txt"
    if run_local_source.exists():
        return run_local_source
    if chapter_run.source_path:
        return Path(chapter_run.source_path)
    if fallback_source_dir is not None:
        return fallback_source_dir / f"{chapter}.txt"
    return run_local_source


def collect_review_queue(run_dir: Path) -> ReviewQueue:
    manifest = load_batch_manifest(run_dir / "batch_manifest.json")
    items: list[ReviewQueueItem] = []
    for chapter, chapter_run in manifest.chapters.items():
        if chapter_run.status == "failed":
            items.append(
                ReviewQueueItem(
                    chapter=chapter,
                    chapter_status=chapter_run.status,
                    check_id="chapter_failed",
                    severity="error",
                    message=chapter_run.error or "Chapter failed.",
                    report_path=chapter_run.report_path,
                    final_path=chapter_run.final_path,
                )
            )
            continue
        if chapter_run.status != "review_required":
            continue
        chapter_run_dir = _chapter_run_dir(run_dir, chapter_run, chapter)
        qa_path = chapter_run_dir / "qa_final.json"
        if not qa_path.exists():
            items.append(
                ReviewQueueItem(
                    chapter=chapter,
                    chapter_status=chapter_run.status,
                    check_id="qa_final_missing",
                    severity="error",
                    message=f"Missing final QA report: {qa_path}",
                    report_path=chapter_run.report_path,
                    final_path=chapter_run.final_path,
                )
            )
            continue
        qa_report = QAReport.model_validate_json(qa_path.read_text(encoding="utf-8"))
        source_text = _read_optional_text(str(_chapter_source_path(run_dir, chapter_run, chapter)))
        final_text = _read_optional_text(chapter_run.final_path)
        for finding in qa_report.findings:
            if finding.status == "fixed":
                continue
            source_context = None
            if source_text:
                source_context = _paragraph_context(source_text, finding.location.paragraph_index)
                source_context = source_context or _window_context(source_text, [finding.found, finding.expected])
            final_context = None
            if final_text:
                if finding.found and CHINESE_RE.search(finding.found) and source_text:
                    final_context = _aligned_window_context(
                        source_text=source_text,
                        final_text=final_text,
                        source_needle=finding.found,
                    )
                final_context = final_context or _paragraph_context(final_text, finding.location.paragraph_index)
                final_context = final_context or _window_context(final_text, [finding.expected, finding.found, finding.location.snippet])
            items.append(
                ReviewQueueItem(
                    chapter=chapter,
                    chapter_status=chapter_run.status,
                    check_id=finding.check_id,
                    severity=finding.severity,
                    message=finding.message,
                    found=finding.found,
                    expected=finding.expected,
                    paragraph_index=finding.location.paragraph_index,
                    line_index=finding.location.line_index,
                    snippet=finding.location.snippet,
                    source_context=source_context,
                    final_context=final_context,
                    report_path=chapter_run.report_path,
                    final_path=chapter_run.final_path,
                )
            )
    return ReviewQueue(
        run_id=manifest.run_id,
        story_slug=manifest.story_slug,
        run_dir=str(run_dir),
        items=items,
        summary=_review_summary(items),
    )


def _extract_panel_lines(text: str) -> list[PanelLine]:
    return [
        PanelLine(index=segment.index, line_number=segment.line_number, text=segment.text)
        for segment in extract_panel_segments(text)
    ]


def _panel_rows(source_panels: list[PanelLine], final_panels: list[PanelLine]) -> list[PanelComparisonRow]:
    rows: list[PanelComparisonRow] = []
    max_count = max(len(source_panels), len(final_panels))
    for index in range(max_count):
        source = source_panels[index] if index < len(source_panels) else None
        final = final_panels[index] if index < len(final_panels) else None
        if source is None:
            status = "extra_final"
        elif final is None:
            status = "missing_final"
        else:
            status = "paired"
        rows.append(PanelComparisonRow(index=index + 1, source=source, final=final, status=status))
    return rows


def build_panel_report(run_dir: Path, *, chapters: list[str] | None = None) -> PanelReport:
    manifest = load_batch_manifest(run_dir / "batch_manifest.json")
    if chapters is None:
        queue = collect_review_queue(run_dir)
        selected = [item.chapter for item in queue.items if item.check_id == "system_panel_count"]
        chapters = list(dict.fromkeys(selected)) or list(manifest.chapters)
    reports: list[PanelChapterReport] = []
    for chapter in chapters:
        if chapter not in manifest.chapters:
            raise ValueError(f"Selected chapter is not in the batch manifest: {chapter}")
        chapter_run = manifest.chapters[chapter]
        chapter_run_dir = _chapter_run_dir(run_dir, chapter_run, chapter)
        source_path = _chapter_source_path(run_dir, chapter_run, chapter)
        final_path = Path(chapter_run.final_path) if chapter_run.final_path else chapter_run_dir / "translated_final" / f"{chapter}.txt"
        source_text = source_path.read_text(encoding="utf-8") if source_path.exists() else ""
        final_text = final_path.read_text(encoding="utf-8") if final_path.exists() else ""
        source_panels = _extract_panel_lines(source_text)
        final_panels = _extract_panel_lines(final_text)
        reports.append(
            PanelChapterReport(
                chapter=chapter,
                chapter_status=chapter_run.status,
                source_path=str(source_path) if source_path.exists() else None,
                final_path=str(final_path) if final_path.exists() else None,
                source_count=len(source_panels),
                final_count=len(final_panels),
                count_delta=len(final_panels) - len(source_panels),
                rows=_panel_rows(source_panels, final_panels),
            )
        )
    summary = PanelReportSummary(
        total_chapters=len(reports),
        mismatch_chapters=sum(1 for report in reports if report.source_count != report.final_count),
        total_source_panels=sum(report.source_count for report in reports),
        total_final_panels=sum(report.final_count for report in reports),
        chapter_selection=",".join(report.chapter for report in reports),
    )
    return PanelReport(
        run_id=manifest.run_id,
        story_slug=manifest.story_slug,
        run_dir=str(run_dir),
        chapters=reports,
        summary=summary,
    )


def _panel_line_text(panel: PanelLine | None) -> str:
    if panel is None:
        return "_missing_"
    return f"L{panel.line_number}: {panel.text}"


def render_panel_report_markdown(report: PanelReport) -> str:
    lines = [
        f"# Panel Report: {report.run_id}",
        "",
        f"- Story: {_inline_code(report.story_slug)}",
        f"- Run directory: {_inline_code(report.run_dir)}",
        f"- Chapters: {report.summary.total_chapters}",
        f"- Mismatch chapters: {report.summary.mismatch_chapters}",
        f"- Source/final panels: {report.summary.total_source_panels}/{report.summary.total_final_panels}",
        f"- Chapter selector: {_inline_code(report.summary.chapter_selection)}",
        "",
        "This is an ordinal diagnostic. It compares bracketed source/final panels by position; it does not prove semantic alignment.",
        "",
    ]
    if not report.chapters:
        lines.extend(["No panel chapters were selected.", ""])
        return "\n".join(lines).rstrip() + "\n"
    for chapter in report.chapters:
        lines.extend(
            [
                f"## {chapter.chapter}",
                "",
                f"- Status: {_inline_code(chapter.chapter_status)}",
                f"- Source/final panels: {chapter.source_count}/{chapter.final_count}",
                f"- Count delta: {chapter.count_delta:+d}",
                f"- Source: {_inline_code(chapter.source_path)}",
                f"- Final: {_inline_code(chapter.final_path)}",
                "",
            ]
        )
        extra_rows = [row for row in chapter.rows if row.status == "extra_final"]
        missing_rows = [row for row in chapter.rows if row.status == "missing_final"]
        if extra_rows:
            lines.extend(["### Extra Final Panels", ""])
            for row in extra_rows:
                lines.append(f"- #{row.index}: {_panel_line_text(row.final)}")
            lines.append("")
        if missing_rows:
            lines.extend(["### Missing Final Panels", ""])
            for row in missing_rows:
                lines.append(f"- #{row.index}: source {_panel_line_text(row.source)}")
            lines.append("")
        lines.extend(["### Ordinal Rows", ""])
        for row in chapter.rows:
            lines.append(f"- #{row.index} `{row.status}`")
            lines.append(f"  - Source: {_panel_line_text(row.source)}")
            lines.append(f"  - Final: {_panel_line_text(row.final)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _inline_code(value: str | None) -> str:
    if not value:
        return "_not provided_"
    escaped = value.replace("`", "\\`")
    return f"`{escaped}`"


def _fenced_text(value: str | None) -> str:
    if not value:
        return "_not available_"
    return "```text\n" + value.replace("```", "` ` `") + "\n```"


def render_review_queue_markdown(queue: ReviewQueue) -> str:
    lines = [
        f"# Review Queue: {queue.run_id}",
        "",
        f"- Story: {_inline_code(queue.story_slug)}",
        f"- Run directory: {_inline_code(queue.run_dir)}",
        f"- Items: {queue.summary.total_items}",
        f"- Chapters: {len(queue.summary.by_chapter)}",
        f"- Chapter selector: {_inline_code(queue.summary.chapter_selection)}",
        "",
    ]
    if queue.summary.by_check:
        lines.extend(["## Summary By Check", ""])
        for check_id, count in queue.summary.by_check.items():
            lines.append(f"- {_inline_code(check_id)}: {count}")
        lines.append("")
    if not queue.items:
        lines.extend(["No review items were found.", ""])
        return "\n".join(lines).rstrip() + "\n"
    for item in queue.items:
        lines.extend(
            [
                f"## {item.chapter} - {item.check_id}",
                "",
                f"- Severity: {_inline_code(item.severity)}",
                f"- Status: {_inline_code(item.chapter_status)}",
                f"- Message: {item.message}",
                f"- Found: {_inline_code(item.found)}",
                f"- Expected: {_inline_code(item.expected)}",
                f"- Paragraph: {_inline_code(str(item.paragraph_index) if item.paragraph_index is not None else None)}",
                f"- Report: {_inline_code(item.report_path)}",
                f"- Final: {_inline_code(item.final_path)}",
                "",
                "### Source Context",
                "",
                _fenced_text(item.source_context),
                "",
                "### Final Context",
                "",
                _fenced_text(item.final_context),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _glossary_gap_summary(gaps: list[GlossaryGapItem]) -> GlossaryGapSummary:
    by_chapter: dict[str, int] = {}
    for gap in gaps:
        for occurrence in gap.occurrences:
            by_chapter[occurrence.chapter] = by_chapter.get(occurrence.chapter, 0) + 1
    chapters = _sorted_unique_chapters(list(by_chapter))
    return GlossaryGapSummary(
        total_occurrences=sum(gap.count for gap in gaps),
        term_count=len(gaps),
        by_chapter=by_chapter,
        chapters=chapters,
        chapter_selection=",".join(chapters),
    )


def _glossary_gap_action(found: str | None) -> str:
    if found and CHINESE_RE.search(found):
        return (
            "Offline repair rules do not auto-patch from source-only evidence; "
            "review the contexts, then either edit the affected translation, add explicit glossary "
            "candidates/block variants, or rerun with a live/replay repair provider."
        )
    return (
        "Review the observed non-canonical wording, then add an explicit glossary candidate/block variant "
        "or edit the affected translation before running batch refresh/resume."
    )


def _alias_stems(text: str | None) -> set[str]:
    stems: set[str] = set()
    if not text:
        return stems
    for word in re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower()):
        for stripped in word.strip("'-").replace("'", "-").split("-"):
            if not stripped or stripped in ALIAS_STOPWORDS or len(stripped) < 3:
                continue
            if len(stripped) > 3 and stripped.endswith("ies"):
                stripped = stripped[:-3] + "y"
            elif len(stripped) > 3 and stripped.endswith("s"):
                stripped = stripped[:-1]
            stems.add(stripped)
    return stems


def _trim_alias_phrase(words: list[str]) -> str:
    trimmed = [word.strip(".,;:!?()[]{}\"'") for word in words]
    while trimmed and trimmed[0].lower() in ALIAS_STOPWORDS:
        trimmed.pop(0)
    while trimmed and trimmed[-1].lower() in ALIAS_STOPWORDS:
        trimmed.pop()
    return " ".join(word for word in trimmed if word)


def _suggest_gap_aliases(expected: str | None, items: list[ReviewQueueItem]) -> list[str]:
    expected_stems = _alias_stems(expected)
    if not expected_stems:
        return []
    focused_texts: list[str] = []
    fallback_texts: list[str] = []
    seen_paths: set[str] = set()
    for item in items:
        if item.final_path and item.final_path not in seen_paths:
            seen_paths.add(item.final_path)
            final_text = _read_optional_text(item.final_path)
            if final_text:
                fallback_texts.append(final_text)
        if item.final_context:
            focused_texts.append(item.final_context)

    def collect(texts: list[str]) -> list[str]:
        suggested: OrderedDict[str, None] = OrderedDict()
        for text in texts:
            tokens = list(re.finditer(r"[A-Za-z][A-Za-z'-]*", text))
            for index, token in enumerate(tokens):
                token_stems = _alias_stems(token.group(0))
                if not (expected_stems & token_stems):
                    continue
                start = max(0, index - 1)
                end = min(len(tokens), index + 2)
                phrase = _trim_alias_phrase([candidate.group(0) for candidate in tokens[start:end]])
                if not phrase or (expected and phrase.lower() == expected.lower()):
                    continue
                key = phrase.lower()
                if key not in suggested:
                    suggested[phrase] = None
                if len(suggested) >= 6:
                    return list(suggested)
        return list(suggested)

    return collect(focused_texts) or collect(fallback_texts)


def build_glossary_gap_report(run_dir: Path) -> GlossaryGapReport:
    queue = collect_review_queue(run_dir)
    grouped: OrderedDict[tuple[str | None, str | None], list[ReviewQueueItem]] = OrderedDict()
    for item in queue.items:
        if item.check_id != "glossary_required":
            continue
        grouped.setdefault((item.found, item.expected), []).append(item)

    gaps: list[GlossaryGapItem] = []
    for (found, expected), items in grouped.items():
        chapters = _sorted_unique_chapters([item.chapter for item in items])
        occurrences = [
            GlossaryGapOccurrence(
                chapter=item.chapter,
                chapter_status=item.chapter_status,
                severity=item.severity,
                message=item.message,
                paragraph_index=item.paragraph_index,
                line_index=item.line_index,
                snippet=item.snippet,
                source_context=item.source_context,
                final_context=item.final_context,
                report_path=item.report_path,
                final_path=item.final_path,
            )
            for item in items
        ]
        gaps.append(
            GlossaryGapItem(
                found=found,
                expected=expected,
                count=len(items),
                chapters=chapters,
                chapter_selection=",".join(chapters),
                suggested_action=_glossary_gap_action(found),
                suggested_aliases=_suggest_gap_aliases(expected, items),
                occurrences=occurrences,
            )
        )

    return GlossaryGapReport(
        run_id=queue.run_id,
        story_slug=queue.story_slug,
        run_dir=queue.run_dir,
        gaps=gaps,
        summary=_glossary_gap_summary(gaps),
    )


def render_glossary_gap_report_markdown(report: GlossaryGapReport) -> str:
    lines = [
        f"# Glossary Gap Report: {report.run_id}",
        "",
        f"- Story: {_inline_code(report.story_slug)}",
        f"- Run directory: {_inline_code(report.run_dir)}",
        f"- Terms: {report.summary.term_count}",
        f"- Occurrences: {report.summary.total_occurrences}",
        f"- Chapter selector: {_inline_code(report.summary.chapter_selection)}",
        "",
    ]
    if not report.gaps:
        lines.extend(["No unresolved glossary gaps were found.", ""])
        return "\n".join(lines).rstrip() + "\n"

    for gap in report.gaps:
        lines.extend(
            [
                f"## {_inline_code(gap.found)} -> {_inline_code(gap.expected)}",
                "",
                f"- Occurrences: {gap.count}",
                f"- Chapters: {_inline_code(gap.chapter_selection)}",
                f"- Suggested aliases: {', '.join(_inline_code(alias) for alias in gap.suggested_aliases) if gap.suggested_aliases else 'none'}",
                f"- Suggested action: {gap.suggested_action}",
                "",
            ]
        )
        for occurrence in gap.occurrences:
            lines.extend(
                [
                    f"### {occurrence.chapter}",
                    "",
                    f"- Severity: {_inline_code(occurrence.severity)}",
                    f"- Message: {occurrence.message}",
                    f"- Paragraph: {_inline_code(str(occurrence.paragraph_index) if occurrence.paragraph_index is not None else None)}",
                    f"- Report: {_inline_code(occurrence.report_path)}",
                    f"- Final: {_inline_code(occurrence.final_path)}",
                    "",
                    "#### Source Context",
                    "",
                    _fenced_text(occurrence.source_context),
                    "",
                    "#### Final Context",
                    "",
                    _fenced_text(occurrence.final_context),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _dedupe_candidate_aliases(expected: str | None, aliases: list[str]) -> list[str]:
    seen = {expected.lower()} if expected else set()
    deduped: list[str] = []
    for alias in aliases:
        normalized = alias.strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _glossary_update_line(found: str | None, expected: str | None, aliases: list[str]) -> str | None:
    if not found or not expected:
        return None
    parts = [expected, *aliases]
    return f"{found}: {', '.join(parts)}"


def _glossary_update_summary(items: list[GlossaryUpdatePlanItem]) -> GlossaryUpdatePlanSummary:
    chapters = _sorted_unique_chapters(chapter for item in items for chapter in item.chapters)
    return GlossaryUpdatePlanSummary(
        total_items=len(items),
        add_candidates_count=sum(1 for item in items if item.action == "add_candidates"),
        manual_review_count=sum(1 for item in items if item.action == "manual_review"),
        chapters=chapters,
        chapter_selection=",".join(chapters),
    )


def build_glossary_update_plan(run_dir: Path) -> GlossaryUpdatePlan:
    report = build_glossary_gap_report(run_dir)
    items: list[GlossaryUpdatePlanItem] = []
    for gap in report.gaps:
        aliases = _dedupe_candidate_aliases(gap.expected, gap.suggested_aliases)
        source_like_found = bool(gap.found and CHINESE_RE.search(gap.found))
        suggested_line = _glossary_update_line(gap.found, gap.expected, aliases) if source_like_found else None
        action = "add_candidates" if aliases and suggested_line else "manual_review"
        note = (
            "Add the suggested aliases as glossary candidates, then rerun batch refresh/resume."
            if action == "add_candidates"
            else "No safe source-term candidate line was found; inspect contexts before changing the glossary."
        )
        items.append(
            GlossaryUpdatePlanItem(
                found=gap.found,
                expected=gap.expected,
                action=action,
                count=gap.count,
                chapters=gap.chapters,
                chapter_selection=gap.chapter_selection,
                suggested_aliases=aliases,
                suggested_line=suggested_line,
                note=note,
                occurrences=gap.occurrences,
            )
        )
    return GlossaryUpdatePlan(
        run_id=report.run_id,
        story_slug=report.story_slug,
        run_dir=report.run_dir,
        items=items,
        summary=_glossary_update_summary(items),
    )


def render_glossary_update_plan_markdown(plan: GlossaryUpdatePlan) -> str:
    lines = [
        f"# Glossary Update Plan: {plan.run_id}",
        "",
        f"- Story: {_inline_code(plan.story_slug)}",
        f"- Run directory: {_inline_code(plan.run_dir)}",
        f"- Items: {plan.summary.total_items}",
        f"- Add candidate lines: {plan.summary.add_candidates_count}",
        f"- Manual review lines: {plan.summary.manual_review_count}",
        f"- Chapter selector: {_inline_code(plan.summary.chapter_selection)}",
        "",
        "Edit your private glossary; this file does not mutate it.",
        "",
    ]
    if not plan.items:
        lines.extend(["No glossary update items were planned.", ""])
        return "\n".join(lines).rstrip() + "\n"

    for item in plan.items:
        lines.extend(
            [
                f"## {_inline_code(item.found)} -> {_inline_code(item.expected)}",
                "",
                f"- Action: {_inline_code(item.action)}",
                f"- Occurrences: {item.count}",
                f"- Chapters: {_inline_code(item.chapter_selection)}",
                f"- Suggested aliases: {', '.join(_inline_code(alias) for alias in item.suggested_aliases) if item.suggested_aliases else 'none'}",
                f"- Suggested glossary line: {_inline_code(item.suggested_line)}",
                f"- Note: {item.note}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _parse_glossary_candidate_line(raw_line: str) -> tuple[str, list[str]] | None:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None
    if "->" in line:
        source, target = [part.strip() for part in line.split("->", 1)]
        if source and target:
            return source, [target]
        return None
    if ":" in line or "：" in line:
        normalized = line.replace("：", ":", 1)
        source, candidate_blob = [part.strip() for part in normalized.split(":", 1)]
        candidates = [part.strip() for part in candidate_blob.split(",") if part.strip()]
        if source and candidates:
            return source, candidates
    return None


def _dedupe_glossary_candidates(candidates: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for candidate in candidates:
        normalized = candidate.strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _format_glossary_candidate_line(source: str, candidates: list[str]) -> str:
    return f"{source}: {', '.join(candidates)}"


def _resolve_glossary_update_path(run_dir: Path, glossary_path: Path | None) -> Path:
    if glossary_path is not None:
        return glossary_path
    manifest = load_batch_manifest(run_dir / "batch_manifest.json")
    story = load_story_config(Path(manifest.story_yaml))
    return story.paths.glossary_path


def _glossary_update_application_summary(
    items: list[GlossaryUpdateApplicationItem],
) -> GlossaryUpdateApplicationSummary:
    return GlossaryUpdateApplicationSummary(
        total_items=len(items),
        changed_count=sum(1 for item in items if item.status in {"updated", "appended"}),
        updated_count=sum(1 for item in items if item.status == "updated"),
        appended_count=sum(1 for item in items if item.status == "appended"),
        skipped_count=sum(1 for item in items if item.status == "skipped"),
        manual_review_count=sum(1 for item in items if item.status == "manual_review"),
    )


def apply_glossary_update_plan(
    run_dir: Path,
    *,
    glossary_path: Path | None = None,
    write: bool = False,
    create_backup: bool = True,
) -> GlossaryUpdateApplication:
    plan = build_glossary_update_plan(run_dir)
    target_path = _resolve_glossary_update_path(run_dir, glossary_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Glossary file does not exist: {target_path}")
    original_text = target_path.read_text(encoding="utf-8")
    lines = original_text.splitlines()
    source_to_index: dict[str, int] = {}
    source_to_candidates: dict[str, list[str]] = {}
    for index, raw_line in enumerate(lines):
        parsed = _parse_glossary_candidate_line(raw_line)
        if parsed is None:
            continue
        source, candidates = parsed
        source_to_index.setdefault(source, index)
        source_to_candidates.setdefault(source, candidates)

    application_items: list[GlossaryUpdateApplicationItem] = []
    for plan_item in plan.items:
        if plan_item.action != "add_candidates" or not plan_item.found or not plan_item.expected:
            application_items.append(
                GlossaryUpdateApplicationItem(
                    found=plan_item.found,
                    expected=plan_item.expected,
                    aliases=plan_item.suggested_aliases,
                    status="manual_review",
                    reason="Plan item is not an add-candidates update.",
                )
            )
            continue
        aliases = _dedupe_glossary_candidates(plan_item.suggested_aliases)
        desired_candidates = _dedupe_glossary_candidates([plan_item.expected, *aliases])
        if not aliases or len(desired_candidates) <= 1:
            application_items.append(
                GlossaryUpdateApplicationItem(
                    found=plan_item.found,
                    expected=plan_item.expected,
                    aliases=aliases,
                    status="skipped",
                    reason="No candidate aliases were available.",
                )
            )
            continue
        existing_index = source_to_index.get(plan_item.found)
        if existing_index is None:
            after_line = _format_glossary_candidate_line(plan_item.found, desired_candidates)
            lines.append(after_line)
            source_to_index[plan_item.found] = len(lines) - 1
            source_to_candidates[plan_item.found] = desired_candidates
            application_items.append(
                GlossaryUpdateApplicationItem(
                    found=plan_item.found,
                    expected=plan_item.expected,
                    aliases=aliases,
                    status="appended",
                    line_number=len(lines),
                    before_line=None,
                    after_line=after_line,
                    reason="Source term was not present in the glossary; appended a candidate line.",
                )
            )
            continue
        before_line = lines[existing_index].strip()
        existing_candidates = source_to_candidates.get(plan_item.found, [])
        merged_candidates = _dedupe_glossary_candidates([plan_item.expected, *existing_candidates, *aliases])
        after_line = _format_glossary_candidate_line(plan_item.found, merged_candidates)
        if before_line == after_line:
            status = "skipped"
            reason = "Glossary line already contains the suggested candidates."
        else:
            lines[existing_index] = after_line
            source_to_candidates[plan_item.found] = merged_candidates
            status = "updated"
            reason = "Merged suggested aliases into existing glossary line."
        application_items.append(
            GlossaryUpdateApplicationItem(
                found=plan_item.found,
                expected=plan_item.expected,
                aliases=aliases,
                status=status,
                line_number=existing_index + 1,
                before_line=before_line,
                after_line=after_line,
                reason=reason,
            )
        )

    summary = _glossary_update_application_summary(application_items)
    backup_path: Path | None = None
    if write and summary.changed_count:
        if create_backup:
            backup_path = target_path.with_name(target_path.name + ".bak")
            backup_path.write_text(original_text, encoding="utf-8")
        target_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    return GlossaryUpdateApplication(
        run_id=plan.run_id,
        story_slug=plan.story_slug,
        run_dir=plan.run_dir,
        glossary_path=str(target_path),
        backup_path=str(backup_path) if backup_path else None,
        dry_run=not write,
        items=application_items,
        summary=summary,
    )


def render_glossary_update_application_markdown(application: GlossaryUpdateApplication) -> str:
    lines = [
        f"# Glossary Update Application: {application.run_id}",
        "",
        f"- Story: {_inline_code(application.story_slug)}",
        f"- Run directory: {_inline_code(application.run_dir)}",
        f"- Glossary: {_inline_code(application.glossary_path)}",
        f"- Backup: {_inline_code(application.backup_path)}",
        f"- Dry run: {str(application.dry_run).lower()}",
        f"- Changed: {application.summary.changed_count}",
        f"- Updated: {application.summary.updated_count}",
        f"- Appended: {application.summary.appended_count}",
        f"- Skipped: {application.summary.skipped_count}",
        f"- Manual review: {application.summary.manual_review_count}",
        "",
    ]
    if not application.items:
        lines.extend(["No glossary updates were applied.", ""])
        return "\n".join(lines).rstrip() + "\n"
    for item in application.items:
        lines.extend(
            [
                f"## {_inline_code(item.found)} -> {_inline_code(item.expected)}",
                "",
                f"- Status: {_inline_code(item.status)}",
                f"- Line: {_inline_code(str(item.line_number) if item.line_number is not None else None)}",
                f"- Aliases: {', '.join(_inline_code(alias) for alias in item.aliases) if item.aliases else 'none'}",
                f"- Before: {_inline_code(item.before_line)}",
                f"- After: {_inline_code(item.after_line)}",
                f"- Reason: {item.reason}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _changed_glossary_update_chapters(plan: GlossaryUpdatePlan, application: GlossaryUpdateApplication) -> list[str]:
    changed_keys = {
        (item.found, item.expected)
        for item in application.items
        if item.status in {"updated", "appended"}
    }
    chapters: list[str] = []
    for plan_item in plan.items:
        if (plan_item.found, plan_item.expected) in changed_keys:
            chapters.extend(plan_item.chapters)
    return _sorted_unique_chapters(chapters)


def run_glossary_update_pass(
    run_dir: Path,
    *,
    glossary_path: Path | None = None,
    write: bool = False,
    create_backup: bool = True,
    chapters: list[str] | None = None,
    seed: int = 7,
    skip_epub: bool = False,
    allow_source_qa_fail: bool = False,
    report_mode: str | None = "excerpt",
    write_proof: bool = False,
    write_triage: bool = True,
) -> GlossaryUpdatePassResult:
    manifest = load_batch_manifest(run_dir / "batch_manifest.json")
    before_summary = manifest.summary
    plan = build_glossary_update_plan(run_dir)
    application = apply_glossary_update_plan(
        run_dir,
        glossary_path=glossary_path,
        write=write,
        create_backup=create_backup,
    )
    selected_chapters = chapters or _changed_glossary_update_chapters(plan, application)
    selected_chapters = _sorted_unique_chapters(selected_chapters)
    if not write:
        return GlossaryUpdatePassResult(
            run_id=manifest.run_id,
            run_dir=str(run_dir),
            dry_run=True,
            chapters=selected_chapters,
            rerun_started=False,
            message="Dry run only; pass --write to apply glossary updates and rerun affected review chapters.",
            application=application,
            before_summary=before_summary,
        )
    if application.summary.changed_count == 0:
        return GlossaryUpdatePassResult(
            run_id=manifest.run_id,
            run_dir=str(run_dir),
            dry_run=False,
            chapters=[],
            rerun_started=False,
            message="No safe glossary updates were available; batch was not rerun.",
            application=application,
            before_summary=before_summary,
            after_summary=before_summary,
        )
    if not selected_chapters:
        return GlossaryUpdatePassResult(
            run_id=manifest.run_id,
            run_dir=str(run_dir),
            dry_run=False,
            chapters=[],
            rerun_started=False,
            message="Glossary changed, but no affected chapters were found to rerun.",
            application=application,
            before_summary=before_summary,
            after_summary=before_summary,
        )
    result = resume_batch_pipeline(
        run_dir,
        chapters=selected_chapters,
        provider_mode=manifest.run_config.provider_mode if manifest.run_config else manifest.mode,
        translation_provider_name=manifest.run_config.translation_provider if manifest.run_config else None,
        judge_provider_name=manifest.run_config.judge_provider if manifest.run_config else None,
        repair_provider_name=manifest.run_config.repair_provider if manifest.run_config else None,
        record_cache=manifest.run_config.record_cache if manifest.run_config else False,
        cache_dir=Path(manifest.run_config.cache_dir) if manifest.run_config and manifest.run_config.cache_dir else None,
        model_name=manifest.run_config.model_name if manifest.run_config else None,
        allow_live_provider_fallback=manifest.run_config.allow_live_provider_fallback if manifest.run_config else False,
        seed=seed,
        retry_review_required=True,
        skip_epub=skip_epub,
        allow_source_qa_fail=allow_source_qa_fail,
        report_mode=report_mode,
        write_proof=write_proof,
    )
    if write_triage:
        write_batch_triage_artifacts(run_dir)
    return GlossaryUpdatePassResult(
        run_id=manifest.run_id,
        run_dir=str(run_dir),
        dry_run=False,
        chapters=selected_chapters,
        rerun_started=True,
        message="Applied glossary updates and reran affected review chapters.",
        application=application,
        before_summary=before_summary,
        after_summary=result.manifest.summary,
    )


def render_glossary_update_pass_markdown(result: GlossaryUpdatePassResult) -> str:
    after = result.after_summary
    lines = [
        f"# Glossary Pass: {result.run_id}",
        "",
        f"- Run directory: {_inline_code(result.run_dir)}",
        f"- Dry run: {str(result.dry_run).lower()}",
        f"- Rerun started: {str(result.rerun_started).lower()}",
        f"- Chapters: {_inline_code(','.join(result.chapters))}",
        f"- Message: {result.message}",
        "",
        "## Glossary Updates",
        f"- Changed: {result.application.summary.changed_count}",
        f"- Updated: {result.application.summary.updated_count}",
        f"- Appended: {result.application.summary.appended_count}",
        f"- Manual review: {result.application.summary.manual_review_count}",
        "",
        "## Batch Summary",
        f"- Before packaged/review/failed: {result.before_summary.packaged}/{result.before_summary.review_required}/{result.before_summary.failed}",
        f"- After packaged/review/failed: {after.packaged}/{after.review_required}/{after.failed}" if after else "- After packaged/review/failed: not rerun",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _work_order_action_and_reason(item: ReviewQueueItem) -> tuple[WorkOrderAction, str]:
    if item.check_id == "chapter_failed":
        return (
            "failed_chapter_retry",
            f"Chapter failed before deliverable packaging: {item.message}",
        )
    if item.check_id == "glossary_required":
        if item.found and CHINESE_RE.search(item.found):
            return (
                "glossary_triage",
                "Canonical glossary target is missing, but offline repair rules do not auto-patch from source-only evidence.",
            )
        return (
            "glossary_triage",
            "Canonical glossary target is missing; review the observed wording and update glossary candidates/block variants or edit the translation.",
        )
    if item.check_id == "system_panel_count":
        return (
            "live_candidate_selection",
            "Panel preservation needs bounded candidate selection or a manual panel restoration.",
        )
    if item.check_id in {"residual_chinese", "chinese_punctuation"}:
        return (
            "live_candidate_selection",
            "Residual source-language artifact remains after rule repair; retry with live/replay repair or edit manually.",
        )
    return (
        "manual_review",
        "Finding is not safely patchable by offline rules; inspect context and resolve manually or with a targeted live repair.",
    )


def _work_order_summary(items: list[AgentWorkOrderItem]) -> AgentWorkOrderSummary:
    by_action: dict[str, int] = {}
    by_chapter: dict[str, int] = {}
    live_retry: list[str] = []
    glossary: list[str] = []
    manual: list[str] = []
    for item in items:
        by_action[item.action] = by_action.get(item.action, 0) + 1
        by_chapter[item.chapter] = by_chapter.get(item.chapter, 0) + 1
        if item.action in {"live_candidate_selection", "failed_chapter_retry"}:
            live_retry.append(item.chapter)
        if item.action == "glossary_triage":
            glossary.append(item.chapter)
        if item.action == "manual_review":
            manual.append(item.chapter)
    chapters = _sorted_unique_chapters(list(by_chapter))
    live_retry_chapters = _sorted_unique_chapters(live_retry)
    glossary_chapters = _sorted_unique_chapters(glossary)
    manual_review_chapters = _sorted_unique_chapters(manual)
    return AgentWorkOrderSummary(
        total_items=len(items),
        by_action=by_action,
        by_chapter=by_chapter,
        chapters=chapters,
        chapter_selection=",".join(chapters),
        live_retry_chapters=live_retry_chapters,
        live_retry_selection=",".join(live_retry_chapters),
        glossary_chapters=glossary_chapters,
        glossary_selection=",".join(glossary_chapters),
        manual_review_chapters=manual_review_chapters,
        manual_review_selection=",".join(manual_review_chapters),
    )


def _work_order_commands(
    run_dir: Path,
    summary: AgentWorkOrderSummary,
    *,
    tool_agent_enabled: bool = False,
) -> dict[str, str]:
    commands: dict[str, str] = {
        "review_queue": f"agentic-translation batch review {run_dir} --write --write-markdown",
        "glossary_triage": f"agentic-translation batch glossary-report {run_dir} --write",
        "glossary_update_plan": f"agentic-translation batch glossary-update-plan {run_dir} --write",
    }
    if summary.live_retry_selection:
        commands["live_retry_dry_run"] = (
            f"agentic-translation batch execute-work-order {run_dir} --action live-retry "
            "--provider-mode live --translation-provider offline --judge-provider openai --repair-provider openai "
            "--record-cache --cache-dir .agentic_cache --model \"$AGENTIC_TRANSLATION_MODEL\" --dry-run --write-preview --json"
        )
        commands["live_retry"] = (
            f"agentic-translation batch execute-work-order {run_dir} --action live-retry "
            "--provider-mode live --translation-provider offline --judge-provider openai --repair-provider openai "
            "--record-cache --cache-dir .agentic_cache --model \"$AGENTIC_TRANSLATION_MODEL\" --write-proof"
        )
        if tool_agent_enabled:
            commands["live_retry_dry_run"] += " --tool-agent"
            commands["live_retry"] += " --tool-agent"
    if summary.manual_review_selection:
        commands["manual_accept"] = (
            f'agentic-translation batch accept {run_dir} --chapters "{summary.manual_review_selection}" '
            '--reviewer "$USER" --note "Accepted manual review edits." --write-proof'
        )
        commands["manual_refresh"] = (
            f'agentic-translation batch refresh {run_dir} --chapters "{summary.manual_review_selection}" --write-proof'
        )
    return commands


def build_agent_work_order(run_dir: Path) -> AgentWorkOrder:
    queue = collect_review_queue(run_dir)
    manifest = load_batch_manifest(run_dir / "batch_manifest.json")
    items: list[AgentWorkOrderItem] = []
    for queue_item in queue.items:
        action, reason = _work_order_action_and_reason(queue_item)
        items.append(
            AgentWorkOrderItem(
                chapter=queue_item.chapter,
                action=action,
                check_id=queue_item.check_id,
                severity=queue_item.severity,
                reason=reason,
                found=queue_item.found,
                expected=queue_item.expected,
                final_path=queue_item.final_path,
                report_path=queue_item.report_path,
                source_context=queue_item.source_context,
                final_context=queue_item.final_context,
            )
        )
    summary = _work_order_summary(items)
    return AgentWorkOrder(
        run_id=queue.run_id,
        story_slug=queue.story_slug,
        run_dir=queue.run_dir,
        items=items,
        summary=summary,
        commands=_work_order_commands(
            run_dir,
            summary,
            tool_agent_enabled=bool(manifest.run_config and manifest.run_config.tool_agent_enabled),
        ),
    )


def render_agent_work_order_markdown(work_order: AgentWorkOrder) -> str:
    lines = [
        f"# Agent Work Order: {work_order.run_id}",
        "",
        f"- Story: {_inline_code(work_order.story_slug)}",
        f"- Run directory: {_inline_code(work_order.run_dir)}",
        f"- Items: {work_order.summary.total_items}",
        f"- Chapter selector: {_inline_code(work_order.summary.chapter_selection)}",
        f"- Live retry chapters: {_inline_code(work_order.summary.live_retry_selection)}",
        f"- Glossary triage chapters: {_inline_code(work_order.summary.glossary_selection)}",
        f"- Manual review chapters: {_inline_code(work_order.summary.manual_review_selection)}",
        "",
    ]
    if work_order.summary.by_action:
        lines.extend(["## Summary By Action", ""])
        for action, count in work_order.summary.by_action.items():
            lines.append(f"- {_inline_code(action)}: {count}")
        lines.append("")
    if work_order.commands:
        lines.extend(["## Commands", ""])
        for name, command in work_order.commands.items():
            lines.extend([f"### {name}", "", "```bash", command, "```", ""])
    if not work_order.items:
        lines.extend(["No unresolved work items were found.", ""])
        return "\n".join(lines).rstrip() + "\n"
    lines.extend(["## Items", ""])
    for item in work_order.items:
        lines.extend(
            [
                f"## {item.chapter} - {item.action}",
                "",
                f"- Check: {_inline_code(item.check_id)}",
                f"- Severity: {_inline_code(item.severity)}",
                f"- Reason: {item.reason}",
                f"- Found: {_inline_code(item.found)}",
                f"- Expected: {_inline_code(item.expected)}",
                f"- Report: {_inline_code(item.report_path)}",
                f"- Final: {_inline_code(item.final_path)}",
                "",
                "### Source Context",
                "",
                _fenced_text(item.source_context),
                "",
                "### Final Context",
                "",
                _fenced_text(item.final_context),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _manual_edit_instruction(item: AgentWorkOrderItem) -> str:
    if item.check_id == "glossary_required" and item.expected:
        if item.found and CHINESE_RE.search(item.found):
            return (
                f"Use canonical term `{item.expected}` where faithful; "
                "source-only evidence cannot be patched automatically."
            )
        return f"Replace observed wording `{item.found}` with canonical term `{item.expected}` if the context matches."
    if item.check_id == "system_panel_count":
        return "Restore the missing or mismatched system panel in the final text."
    if item.check_id in {"residual_chinese", "chinese_punctuation"}:
        return "Edit the affected paragraph to remove source-language residue and punctuation."
    return "Inspect the source/final context and edit the final file, then run batch refresh or batch accept."


def _manual_edit_plan_summary(items: list[ManualEditPlanItem]) -> ManualEditPlanSummary:
    by_file: dict[str, int] = {}
    by_action: dict[str, int] = {}
    by_chapter: dict[str, int] = {}
    for item in items:
        by_file[item.final_path] = by_file.get(item.final_path, 0) + 1
        by_action[item.action] = by_action.get(item.action, 0) + 1
        by_chapter[item.chapter] = by_chapter.get(item.chapter, 0) + 1
    chapters = _sorted_unique_chapters(list(by_chapter))
    return ManualEditPlanSummary(
        total_items=len(items),
        by_file=by_file,
        by_action=by_action,
        chapters=chapters,
        chapter_selection=",".join(chapters),
    )


def build_manual_edit_plan(run_dir: Path) -> ManualEditPlan:
    work_order = build_agent_work_order(run_dir)
    items: list[ManualEditPlanItem] = []
    for work_item in work_order.items:
        if not work_item.final_path:
            continue
        items.append(
            ManualEditPlanItem(
                chapter=work_item.chapter,
                action=work_item.action,
                check_id=work_item.check_id,
                final_path=work_item.final_path,
                report_path=work_item.report_path,
                found=work_item.found,
                expected=work_item.expected,
                instruction=_manual_edit_instruction(work_item),
                source_context=work_item.source_context,
                final_context=work_item.final_context,
            )
        )
    return ManualEditPlan(
        run_id=work_order.run_id,
        story_slug=work_order.story_slug,
        run_dir=work_order.run_dir,
        items=items,
        summary=_manual_edit_plan_summary(items),
    )


def render_manual_edit_plan_markdown(plan: ManualEditPlan) -> str:
    lines = [
        f"# Manual Edit Plan: {plan.run_id}",
        "",
        f"- Story: {_inline_code(plan.story_slug)}",
        f"- Run directory: {_inline_code(plan.run_dir)}",
        f"- Items: {plan.summary.total_items}",
        f"- Chapter selector: {_inline_code(plan.summary.chapter_selection)}",
        "",
    ]
    if plan.summary.by_action:
        lines.extend(["## Summary By Action", ""])
        for action, count in plan.summary.by_action.items():
            lines.append(f"- {_inline_code(action)}: {count}")
        lines.append("")
    if not plan.items:
        lines.extend(["No manual edits were planned.", ""])
        return "\n".join(lines).rstrip() + "\n"

    current_file: str | None = None
    for item in sorted(plan.items, key=lambda plan_item: (plan_item.final_path, _chapter_sort_key(plan_item.chapter), plan_item.check_id)):
        if item.final_path != current_file:
            current_file = item.final_path
            lines.extend([f"## {_inline_code(current_file)}", ""])
        lines.extend(
            [
                f"### {item.chapter} - {item.check_id}",
                "",
                f"- Action: {_inline_code(item.action)}",
                f"- Instruction: {item.instruction}",
                f"- Found: {_inline_code(item.found)}",
                f"- Expected: {_inline_code(item.expected)}",
                f"- Report: {_inline_code(item.report_path)}",
                "",
                "#### Source Context",
                "",
                _fenced_text(item.source_context),
                "",
                "#### Current Final Context",
                "",
                _fenced_text(item.final_context),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_batch_triage_artifacts(run_dir: Path) -> dict[str, str]:
    queue = collect_review_queue(run_dir)
    glossary_report = build_glossary_gap_report(run_dir)
    work_order = build_agent_work_order(run_dir)
    manual_edit_plan = build_manual_edit_plan(run_dir)
    glossary_update_plan = build_glossary_update_plan(run_dir)
    artifacts = {
        "review_queue": "review_queue.json",
        "review_chapters": "review_chapters.txt",
        "review_queue_markdown": "review_queue.md",
        "glossary_gap_report": "glossary_gap_report.json",
        "glossary_gap_report_markdown": "glossary_gap_report.md",
        "agentic_work_order": "agentic_work_order.json",
        "agentic_work_order_markdown": "agentic_work_order.md",
        "manual_edit_plan": "manual_edit_plan.json",
        "manual_edit_plan_markdown": "manual_edit_plan.md",
        "glossary_update_plan": "glossary_update_plan.json",
        "glossary_update_plan_markdown": "glossary_update_plan.md",
    }
    (run_dir / artifacts["review_queue"]).write_text(queue.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / artifacts["review_chapters"]).write_text(queue.summary.chapter_selection + "\n", encoding="utf-8")
    (run_dir / artifacts["review_queue_markdown"]).write_text(render_review_queue_markdown(queue), encoding="utf-8")
    (run_dir / artifacts["glossary_gap_report"]).write_text(glossary_report.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / artifacts["glossary_gap_report_markdown"]).write_text(
        render_glossary_gap_report_markdown(glossary_report),
        encoding="utf-8",
    )
    (run_dir / artifacts["agentic_work_order"]).write_text(work_order.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / artifacts["agentic_work_order_markdown"]).write_text(
        render_agent_work_order_markdown(work_order),
        encoding="utf-8",
    )
    (run_dir / artifacts["manual_edit_plan"]).write_text(manual_edit_plan.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / artifacts["manual_edit_plan_markdown"]).write_text(
        render_manual_edit_plan_markdown(manual_edit_plan),
        encoding="utf-8",
    )
    (run_dir / artifacts["glossary_update_plan"]).write_text(
        glossary_update_plan.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (run_dir / artifacts["glossary_update_plan_markdown"]).write_text(
        render_glossary_update_plan_markdown(glossary_update_plan),
        encoding="utf-8",
    )
    manifest_path = run_dir / "batch_manifest.json"
    manifest = load_batch_manifest(manifest_path)
    manifest.artifacts.update(artifacts)
    write_batch_manifest(manifest_path, manifest)
    _write_batch_status(run_dir / "batch_status.json", manifest)
    return artifacts


def _execute_work_order_command(
    run_dir: Path,
    *,
    action: str,
    provider_mode: str,
    translation_provider_name: str,
    judge_provider_name: str,
    repair_provider_name: str,
    record_cache: bool,
    cache_dir: Path | None,
    model_name: str | None,
    tool_agent_enabled: bool = False,
    dry_run: bool = False,
    write_preview: bool = False,
    json_output: bool = False,
) -> str:
    parts = [
        "agentic-translation",
        "batch",
        "execute-work-order",
        str(run_dir),
        "--action",
        action,
        "--provider-mode",
        provider_mode,
        "--translation-provider",
        translation_provider_name,
        "--judge-provider",
        judge_provider_name,
        "--repair-provider",
        repair_provider_name,
    ]
    if record_cache:
        parts.append("--record-cache")
    else:
        parts.append("--no-record-cache")
    if cache_dir is not None:
        parts.extend(["--cache-dir", str(cache_dir)])
    if model_name:
        parts.extend(["--model", model_name])
    parts.append("--tool-agent" if tool_agent_enabled else "--no-tool-agent")
    if dry_run:
        parts.append("--dry-run")
    if write_preview:
        parts.append("--write-preview")
    if json_output:
        parts.append("--json")
    return " ".join(f'"{part}"' if " " in part else part for part in parts)


def preview_agent_work_order_execution(
    run_dir: Path,
    *,
    action: str = "live-retry",
    provider_mode: str = "live",
    translation_provider_name: str = "offline",
    judge_provider_name: str = "openai",
    repair_provider_name: str = "openai",
    record_cache: bool = True,
    cache_dir: Path | None = Path(".agentic_cache"),
    model_name: str | None = None,
    tool_agent_enabled: bool | None = None,
) -> AgentWorkOrderExecutionPreview:
    if action != "live-retry":
        raise ValueError(f"Unsupported work-order action: {action}. Only live-retry is executable.")
    work_order = build_agent_work_order(run_dir)
    chapters = work_order.summary.live_retry_chapters
    if not chapters:
        raise ValueError("No live-retry chapters were found in the current work order.")
    manifest = load_batch_manifest(run_dir / "batch_manifest.json")
    if tool_agent_enabled is None:
        tool_agent_enabled = bool(manifest.run_config and manifest.run_config.tool_agent_enabled)
    from .preflight import run_preflight

    preflight = run_preflight(
        Path(manifest.story_yaml),
        chapters=chapters,
        provider_mode=provider_mode,
        translation_provider_name=translation_provider_name,
        judge_provider_name=judge_provider_name,
        repair_provider_name=repair_provider_name,
        record_cache=record_cache,
        cache_dir=cache_dir,
        model_name=model_name,
        tool_agent_enabled=tool_agent_enabled,
        terminology_consensus=(
            manifest.run_config.terminology_consensus
            if manifest.run_config is not None
            else None
        ),
    )
    execution_command = _execute_work_order_command(
        run_dir,
        action=action,
        provider_mode=provider_mode,
        translation_provider_name=translation_provider_name,
        judge_provider_name=judge_provider_name,
        repair_provider_name=repair_provider_name,
        record_cache=record_cache,
        cache_dir=cache_dir,
        model_name=model_name,
        tool_agent_enabled=tool_agent_enabled,
    )
    dry_run_command = _execute_work_order_command(
        run_dir,
        action=action,
        provider_mode=provider_mode,
        translation_provider_name=translation_provider_name,
        judge_provider_name=judge_provider_name,
        repair_provider_name=repair_provider_name,
        record_cache=record_cache,
        cache_dir=cache_dir,
        model_name=model_name,
        tool_agent_enabled=tool_agent_enabled,
        dry_run=True,
        write_preview=True,
        json_output=True,
    )
    preflight_blockers = [
        f"{check.name}: {check.message}"
        for check in preflight.checks
        if check.status == "fail"
    ]
    recommended_next_action = "execute_live_retry" if preflight.passed else "fix_preflight"
    return AgentWorkOrderExecutionPreview(
        run_id=work_order.run_id,
        story_slug=work_order.story_slug,
        run_dir=work_order.run_dir,
        action=action,
        provider_mode=provider_mode,
        translation_provider=translation_provider_name,
        judge_provider=judge_provider_name,
        repair_provider=repair_provider_name,
        record_cache=record_cache,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        model_name=model_name,
        tool_agent_enabled=tool_agent_enabled,
        chapters=chapters,
        dry_run=True,
        would_mutate=False,
        command=execution_command,
        dry_run_command=dry_run_command,
        execution_command=execution_command,
        recommended_next_action=recommended_next_action,
        recommended_command=execution_command if preflight.passed else dry_run_command,
        preflight_blockers=preflight_blockers,
        preflight_passed=preflight.passed,
        preflight_status_counts=preflight.status_counts,
        preflight_checks=[check.model_dump() for check in preflight.checks],
    )


def render_agent_work_order_execution_preview_markdown(preview: AgentWorkOrderExecutionPreview) -> str:
    preflight_label = "passed" if preview.preflight_passed else "failed"
    lines = [
        f"# Agent Work-Order Execution Preview: {preview.run_id}",
        "",
        f"- Story: {_inline_code(preview.story_slug)}",
        f"- Run directory: {_inline_code(preview.run_dir)}",
        f"- Action: {_inline_code(preview.action)}",
        f"- Provider mode: {_inline_code(preview.provider_mode)}",
        f"- Chapters: {_inline_code(','.join(preview.chapters))}",
        f"- Dry run: {_inline_code(str(preview.dry_run).lower())}",
        f"- Would mutate: {_inline_code(str(preview.would_mutate).lower())}",
        f"- Preflight: {_inline_code(preflight_label)}",
        f"- Recommended next action: {_inline_code(preview.recommended_next_action)}",
        "",
        "## Recommended Command",
        "",
        "```bash",
        preview.recommended_command,
        "```",
        "",
        "## Execution Command",
        "",
        "```bash",
        preview.execution_command,
        "```",
        "",
        "## Dry-Run Command",
        "",
        "```bash",
        preview.dry_run_command,
        "```",
        "",
        "## Preflight Blockers",
        "",
    ]
    if not preview.preflight_blockers:
        lines.append("- none")
    for blocker in preview.preflight_blockers:
        lines.append(f"- {blocker}")
    lines.extend(
        [
            "",
            "## Preflight Checks",
            "",
        ]
    )
    if not preview.preflight_checks:
        lines.append("- none")
    for check in preview.preflight_checks:
        lines.append(f"- {_inline_code(str(check.get('status')))} {check.get('name')}: {check.get('message')}")
    return "\n".join(lines).rstrip() + "\n"


def write_agent_work_order_execution_preview(
    run_dir: Path,
    preview: AgentWorkOrderExecutionPreview,
) -> tuple[Path, Path]:
    json_path = run_dir / "agentic_execution_preview.json"
    markdown_path = run_dir / "agentic_execution_preview.md"
    json_path.write_text(preview.model_dump_json(indent=2), encoding="utf-8")
    markdown_path.write_text(render_agent_work_order_execution_preview_markdown(preview), encoding="utf-8")
    return json_path, markdown_path


def execute_agent_work_order(
    run_dir: Path,
    *,
    action: str = "live-retry",
    provider_mode: str = "live",
    translation_provider_name: str = "offline",
    judge_provider_name: str = "openai",
    repair_provider_name: str = "openai",
    record_cache: bool = True,
    cache_dir: Path | None = Path(".agentic_cache"),
    model_name: str | None = None,
    tool_agent_enabled: bool | None = None,
    seed: int = 7,
    skip_epub: bool = False,
    allow_source_qa_fail: bool = False,
    allow_live_provider_fallback: bool = False,
    report_mode: str | None = "excerpt",
    retry_review_required: bool = True,
    write_proof: bool = True,
) -> BatchPipelineResult:
    preview = preview_agent_work_order_execution(
        run_dir,
        action=action,
        provider_mode=provider_mode,
        translation_provider_name=translation_provider_name,
        judge_provider_name=judge_provider_name,
        repair_provider_name=repair_provider_name,
        record_cache=record_cache,
        cache_dir=cache_dir,
        model_name=model_name,
        tool_agent_enabled=tool_agent_enabled,
    )
    if not preview.preflight_passed:
        blockers = "; ".join(preview.preflight_blockers) or "see preflight checks"
        raise ValueError(f"Preflight failed for work-order execution: {blockers}")
    return resume_batch_pipeline(
        run_dir,
        chapters=preview.chapters,
        provider_mode=provider_mode,
        translation_provider_name=translation_provider_name,
        judge_provider_name=judge_provider_name,
        repair_provider_name=repair_provider_name,
        record_cache=record_cache,
        cache_dir=cache_dir,
        model_name=model_name,
        tool_agent_enabled=preview.tool_agent_enabled,
        seed=seed,
        force=False,
        retry_review_required=retry_review_required,
        skip_epub=skip_epub,
        allow_source_qa_fail=allow_source_qa_fail,
        allow_live_provider_fallback=allow_live_provider_fallback,
        report_mode=report_mode,
        write_proof=write_proof,
    )


def _validate_live_batch_mode(
    *,
    provider_mode: str,
    translation_provider_name: str,
    judge_provider_name: str,
    repair_provider_name: str,
    record_cache: bool,
    cache_dir: Path | None,
    model_name: str | None,
    allow_live_provider_fallback: bool = False,
    tool_agent_enabled: bool = False,
) -> None:
    _validate_tool_agent_batch_config(
        tool_agent_enabled=tool_agent_enabled,
        provider_mode=provider_mode,
        repair_provider_name=repair_provider_name,
        model_name=model_name,
    )
    if provider_mode == "live" and (not record_cache or cache_dir is None):
        raise ValueError("live batch runs require --record-cache and --cache-dir so runs can be replayed.")
    if provider_mode == "replay" and cache_dir is None:
        raise ValueError("replay batch runs require --cache-dir.")
    if provider_mode == "offline":
        return
    provider_names = {translation_provider_name, judge_provider_name, repair_provider_name}
    if not any(name != "offline" for name in provider_names):
        raise ValueError(f"{provider_mode} batch mode requires at least one non-offline provider.")
    live_provider_names = openai_compatible_provider_names(provider_names)
    if provider_mode == "live" and not live_provider_names:
        raise ValueError("live batch mode requires at least one live provider such as openai or deepseek.")
    if provider_mode == "live" and live_provider_names:
        missing_env: list[str] = []
        for provider_name in live_provider_names:
            missing_env.extend(required_live_provider_config(provider_name, model_name=model_name))
        missing_env = sorted(set(missing_env))
        if missing_env and not allow_live_provider_fallback:
            raise ValueError(
                "Missing required live provider environment/config value(s): "
                + ", ".join(missing_env)
                + "."
            )


def _validate_tool_agent_batch_config(
    *,
    tool_agent_enabled: bool,
    provider_mode: str,
    repair_provider_name: str,
    model_name: str | None,
) -> None:
    if not tool_agent_enabled:
        return
    errors: list[str] = []
    if provider_mode not in {"live", "replay"}:
        errors.append("Tool-agent mode requires provider mode live or replay.")
    if repair_provider_name == "offline":
        errors.append("Tool-agent mode requires a non-offline repair provider.")
    if not model_name:
        errors.append("Tool-agent mode requires an explicit repair model.")
    if errors:
        raise ValueError(" ".join(errors))


def _validate_terminology_batch_config(
    *,
    config: TerminologyConsensusConfig,
    tool_agent_enabled: bool,
    provider_mode: str,
    repair_provider_name: str,
    record_cache: bool,
    cache_dir: Path | None,
) -> None:
    """Validate terminology settings before a batch run directory is created."""

    if not config.enabled:
        return
    errors: list[str] = []
    if not tool_agent_enabled:
        errors.append("Terminology consensus requires tool-agent mode.")
    if provider_mode not in {"live", "replay"}:
        errors.append("Terminology consensus requires provider mode live or replay.")
    if repair_provider_name == "offline":
        errors.append("Terminology consensus requires a non-offline repair provider.")
    if not config.openai_model:
        errors.append("Terminology consensus requires an explicit OpenAI model.")
    if not config.deepseek_model:
        errors.append("Terminology consensus requires an explicit DeepSeek model.")
    if cache_dir is None:
        errors.append("Terminology consensus requires --cache-dir.")
    elif provider_mode == "replay":
        if not cache_dir.exists():
            errors.append("Terminology consensus replay cache directory does not exist.")
        elif not cache_dir.is_dir():
            errors.append("Terminology consensus replay cache path is not a directory.")
    if provider_mode == "live":
        if not record_cache:
            errors.append("Live terminology consensus requires --record-cache.")
        for provider_name, model_name in (
            ("openai", config.openai_model),
            ("deepseek", config.deepseek_model),
        ):
            if model_name:
                errors.extend(required_live_provider_config(provider_name, model_name=model_name))
    if errors:
        raise ValueError(" ".join(dict.fromkeys(errors)))


def _provider_labels(
    *,
    provider_mode: str,
    translation_provider_name: str,
    judge_provider_name: str,
    repair_provider_name: str,
    cache_dir: Path | None,
    record_cache: bool,
    model_name: str | None,
) -> dict[str, ProviderLabel]:
    if provider_mode == "offline":
        translation_provider_name = judge_provider_name = repair_provider_name = "offline"
    translation_provider = get_translation_provider(
        translation_provider_name,
        provider_mode=provider_mode,
        cache_dir=cache_dir,
        record_cache=record_cache,
        model_name=model_name,
    )
    judge_provider = get_judge_provider(
        judge_provider_name,
        provider_mode=provider_mode,
        cache_dir=cache_dir,
        record_cache=record_cache,
        model_name=model_name,
    )
    repair_provider = get_repair_provider(
        repair_provider_name,
        provider_mode=provider_mode,
        cache_dir=cache_dir,
        record_cache=record_cache,
        model_name=model_name,
    )
    return {
        "translation": ProviderLabel(provider=translation_provider.provider_name, model=translation_provider.model_name),
        "judge": ProviderLabel(provider=judge_provider.provider_name, model=judge_provider.model_name),
        "repair": ProviderLabel(provider=repair_provider.provider_name, model=repair_provider.model_name),
    }


def _stamp_run_config(
    manifest: BatchManifest,
    *,
    provider_mode: str,
    translation_provider_name: str,
    judge_provider_name: str,
    repair_provider_name: str,
    providers: dict[str, ProviderLabel],
    record_cache: bool,
    cache_dir: Path | None,
    model_name: str | None,
    allow_live_provider_fallback: bool = False,
    tool_agent_enabled: bool = False,
    terminology_consensus: TerminologyConsensusConfig | None = None,
) -> None:
    manifest.mode = provider_mode
    manifest.providers = providers
    manifest.run_config = build_batch_run_config(
        provider_mode=provider_mode,
        translation_provider_name=translation_provider_name,
        judge_provider_name=judge_provider_name,
        repair_provider_name=repair_provider_name,
        record_cache=record_cache,
        cache_dir=cache_dir,
        model_name=model_name,
        allow_live_provider_fallback=allow_live_provider_fallback,
        tool_agent_enabled=tool_agent_enabled,
        terminology_consensus=terminology_consensus,
    )


def _model_from_manifest(manifest: BatchManifest) -> str | None:
    for label in manifest.providers.values():
        if label.provider != "offline" and label.model:
            return label.model
    return None


def _validate_replay_cache_ready_for_config(
    run_config: BatchRunConfig,
    *,
    required_extra_namespaces: list[str] | None = None,
    observed_namespaces: set[str] | None = None,
) -> Path:
    if not run_config.cache_dir:
        raise ValueError("Source batch run_config has no cache_dir; cannot replay recorded provider calls.")
    cache_dir = Path(run_config.cache_dir)
    if not cache_dir.exists():
        raise ValueError(f"Replay cache directory does not exist: {cache_dir}")
    if not cache_dir.is_dir():
        raise ValueError(f"Replay cache path is not a directory: {cache_dir}")

    cache_report = inspect_response_cache(cache_dir)
    if cache_report.total_entries == 0:
        raise ValueError(f"Replay cache has no indexed entries: {cache_dir}")
    if not cache_report.integrity_passed:
        raise ValueError(
            f"Replay cache integrity failed with {cache_report.invalid_entries} invalid indexed entrie(s)."
        )

    provider_by_role = {
        "translation": run_config.translation_provider,
        "judge": run_config.judge_provider,
        "repair": run_config.repair_provider,
    }
    tool_agent_only = (
        run_config.tool_agent_enabled
        and observed_namespaces is not None
        and "agent_action" in observed_namespaces
        and not (observed_namespaces & set(PROVIDER_ROLE_ORDER))
    )
    required_namespaces = [] if tool_agent_only else [
        role for role in PROVIDER_ROLE_ORDER if provider_by_role.get(role) != "offline"
    ]
    for namespace in required_extra_namespaces or []:
        if namespace not in required_namespaces:
            required_namespaces.append(namespace)
    missing_namespaces = [
        namespace for namespace in required_namespaces if cache_report.by_namespace.get(namespace, 0) == 0
    ]
    if missing_namespaces:
        raise ValueError(
            "Replay cache is missing indexed namespace(s): " + ", ".join(missing_namespaces)
        )
    return cache_dir


def _validate_source_manifest_replayable(source_manifest: BatchManifest, selected_chapters: list[str]) -> None:
    evidence_manifest = source_manifest.model_copy(deep=True)
    evidence_manifest.chapters = {
        chapter: evidence_manifest.chapters[chapter]
        for chapter in selected_chapters
    }
    evidence_manifest.refresh_summary()
    evidence = build_agentic_evidence(evidence_manifest)
    tool_agent_evidence = build_tool_agent_evidence(evidence_manifest)
    if evidence.replay_cache_ready or (tool_agent_evidence.applicable and tool_agent_evidence.proof_ready):
        return

    details: list[str] = []
    if not evidence.provider_call_records:
        details.append("no recorded provider calls for selected source chapter(s)")
    if evidence.cache_missing_namespaces:
        details.append("missing namespace(s): " + ", ".join(evidence.cache_missing_namespaces))
    if evidence.cache_missing_call_records:
        details.append("missing call record(s): " + ", ".join(evidence.cache_missing_call_records))
    if evidence.cache_metadata_mismatches:
        details.append("metadata mismatch: " + ", ".join(evidence.cache_metadata_mismatches))
    if evidence.cache_integrity_issues:
        details.append("cache integrity issue(s): " + "; ".join(evidence.cache_integrity_issues))
    if tool_agent_evidence.applicable and tool_agent_evidence.mismatches:
        details.append("tool-agent mismatch(es): " + "; ".join(tool_agent_evidence.mismatches))
    if not evidence.cache_available:
        details.append("cache directory is unavailable")
    elif not evidence.cache_integrity_passed:
        details.append("cache integrity did not pass")

    raise ValueError(
        "Source batch run is not replayable for selected chapter(s): "
        + "; ".join(details or [evidence.reason])
    )


def _attempt_provider_label(manifest: BatchManifest) -> str:
    return ";".join(
        f"{role}={manifest.providers.get(role, ProviderLabel(provider='offline', model='')).provider}"
        for role in ("translation", "judge", "repair")
    )


def _attempt_model_label(manifest: BatchManifest) -> str:
    for role in ("translation", "judge", "repair"):
        label = manifest.providers.get(role)
        if label and label.model:
            return label.model
    return "unknown"


def _start_attempt(manifest: BatchManifest, chapter_run, *, action: str) -> AgentAttempt:  # noqa: ANN001
    attempt = AgentAttempt(
        attempt_id=f"{chapter_run.chapter}-attempt-{len(chapter_run.attempts) + 1:03d}",
        chapter=chapter_run.chapter,
        provider=_attempt_provider_label(manifest),
        model=_attempt_model_label(manifest),
        action=action,
        status="pending",
        message="Chapter run started.",
    )
    chapter_run.attempts.append(attempt)
    return attempt


def _clear_tool_agent_metadata(chapter_run) -> None:  # noqa: ANN001 - compact manifest model helper.
    chapter_run.tool_agent_episode_path = None
    chapter_run.tool_agent_report_path = None
    chapter_run.tool_agent_html_report_path = None
    chapter_run.tool_agent_final_status = None
    chapter_run.tool_agent_steps = 0
    chapter_run.tool_agent_initial_findings = 0
    chapter_run.tool_agent_final_findings = 0
    chapter_run.tool_agent_accepted_patches = 0
    chapter_run.tool_agent_rejected_patches = 0
    chapter_run.tool_agent_final_text_sha256 = None


def _copy_tool_agent_metadata(chapter_run, tool_agent) -> None:  # noqa: ANN001 - PipelineResult metadata.
    if tool_agent is None:
        _clear_tool_agent_metadata(chapter_run)
        return
    chapter_run.tool_agent_episode_path = str(tool_agent.episode_path)
    chapter_run.tool_agent_report_path = str(tool_agent.markdown_report_path)
    chapter_run.tool_agent_html_report_path = str(tool_agent.html_report_path)
    chapter_run.tool_agent_final_status = tool_agent.final_status
    chapter_run.tool_agent_steps = tool_agent.step_count
    chapter_run.tool_agent_initial_findings = tool_agent.initial_findings
    chapter_run.tool_agent_final_findings = tool_agent.final_findings
    chapter_run.tool_agent_accepted_patches = tool_agent.accepted_patch_count
    chapter_run.tool_agent_rejected_patches = tool_agent.rejected_patch_count
    chapter_run.tool_agent_final_text_sha256 = getattr(
        tool_agent,
        "final_text_sha256",
        getattr(getattr(tool_agent, "run_record", None), "final_text_sha256", None),
    )


def _preserve_tool_agent_artifacts_from_disk(
    chapter_run,
    *,
    run_dir: Path,
    chapter: str,
) -> None:  # noqa: ANN001 - failure-path manifest model helper.
    try:
        batch_root_input = Path(run_dir)
        if batch_root_input.is_symlink():
            return
        batch_root = batch_root_input.resolve(strict=False)
        configured_root = (
            Path(chapter_run.chapter_run_dir)
            if chapter_run.chapter_run_dir
            else batch_root / "chapters" / chapter
        )
        if configured_root.is_symlink():
            return
        chapter_root = configured_root.resolve(strict=False)
        expected_root = (batch_root / "chapters" / chapter).resolve(strict=False)
        if chapter_root != expected_root or not chapter_root.is_relative_to(batch_root):
            return
        agent_dir = chapter_root / "agent_repair"
        if agent_dir.exists() and agent_dir.is_symlink():
            return
        if not agent_dir.resolve(strict=False).is_relative_to(chapter_root):
            return
        safe_paths: dict[str, Path] = {}
        for field_name, filename in (
            ("tool_agent_episode_path", "agent_episode.json"),
            ("tool_agent_report_path", "report.md"),
            ("tool_agent_html_report_path", "report.html"),
        ):
            candidate = agent_dir / filename
            if candidate.is_symlink():
                continue
            resolved = candidate.resolve(strict=False)
            if resolved.is_relative_to(chapter_root) and candidate.exists():
                safe_paths[field_name] = resolved
        for field_name, path in safe_paths.items():
            setattr(chapter_run, field_name, str(path))

        episode_path = safe_paths.get("tool_agent_episode_path")
        if episode_path is None:
            return
        try:
            episode = AgentEpisode.model_validate_json(episode_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - preserve the original provider failure.
            return
        chapter_run.tool_agent_final_status = episode.final_status
        chapter_run.tool_agent_steps = len(episode.steps)
        chapter_run.tool_agent_initial_findings = episode.initial_qa.summary.total_findings
        chapter_run.tool_agent_final_findings = (
            episode.final_qa.summary.total_findings if episode.final_qa is not None else 0
        )
        chapter_run.tool_agent_accepted_patches = sum(
            step.observation.kind == "patch_accepted" for step in episode.steps
        )
        chapter_run.tool_agent_rejected_patches = sum(
            step.observation.kind == "patch_rejected" for step in episode.steps
        )
        final_path = chapter_root / "translated_final" / f"{chapter}.txt"
        if final_path.exists() and not final_path.is_symlink() and final_path.resolve(strict=False).is_relative_to(chapter_root):
            chapter_run.tool_agent_final_text_sha256 = _text_sha256(final_path.read_text(encoding="utf-8"))
        episode_calls = [step.provider_call for step in episode.steps if step.provider_call is not None]
        if episode_calls:
            chapter_run.provider_calls = episode_calls
    except Exception:  # noqa: BLE001 - never mask the original batch exception.
        return


def _artifact_qa(*, txt_path: Path, epub_path: Path | None, expected_chapters: int) -> ArtifactQAReport:
    txt_audit = verify_txt_artifact(txt_path)
    epub_audit = verify_epub_artifact(epub_path) if epub_path else None
    failures: list[str] = []
    if txt_audit["contains_chinese"]:
        failures.append("TXT contains Chinese residue.")
    if txt_audit["contains_prompt_leakage"]:
        failures.append("TXT contains prompt leakage.")
    if txt_audit["chapter_markers"] != expected_chapters:
        failures.append("TXT chapter marker count does not match expected chapter count.")
    if epub_audit:
        if epub_audit["contains_chinese"]:
            failures.append("EPUB contains Chinese residue.")
        if epub_audit["contains_prompt_leakage"]:
            failures.append("EPUB contains prompt leakage.")
        if epub_audit["xhtml_chapters"] != expected_chapters:
            failures.append("EPUB XHTML chapter count does not match expected chapter count.")
    return ArtifactQAReport(
        expected_chapters=expected_chapters,
        txt=txt_audit,
        epub=epub_audit,
        passed=not failures,
        failures=failures,
    )


def _baseline_comparison(*, baseline_dir: Path | None, chapter: str, final_path: Path) -> BaselineComparison | None:
    if baseline_dir is None:
        return None
    baseline_path = baseline_dir / f"{chapter}.txt"
    if not baseline_path.exists():
        return None
    baseline_sha = sha256_file(baseline_path)
    final_sha = sha256_file(final_path)
    return BaselineComparison(
        baseline_path=str(baseline_path),
        baseline_sha256=baseline_sha,
        final_sha256=final_sha,
        changed=baseline_sha != final_sha,
    )


def last_attempt_label(chapter_run) -> str:  # noqa: ANN001 - shared display helper for CLI/report code.
    if chapter_run.status == "packaged" and chapter_run.final_findings == 0:
        return "final: packaged with 0 final QA findings"
    if not chapter_run.attempts:
        return ""
    attempt = chapter_run.attempts[-1]
    if attempt.message:
        return f"{attempt.status}: {attempt.message}"
    return attempt.status


def _markdown_cell(value: str | None) -> str:
    return (value or "").replace("|", "\\|").replace("\n", " ")


def _write_batch_report(path: Path, manifest: BatchManifest) -> None:
    inspection = build_batch_inspection_report(manifest)
    evidence = inspection.agentic_evidence
    evidence_label = "supported" if evidence.agentic_claim_supported else "not supported"
    rows = [
        "| Chapter | Status | Score | Findings | Attempts | Last Attempt | Repairs | Accepted | Report |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for chapter, chapter_run in manifest.chapters.items():
        accepted_repairs = sum(1 for attempt in chapter_run.patch_attempts if attempt.accepted)
        rows.append(
            "| "
            + " | ".join(
                [
                    chapter,
                    chapter_run.status,
                    "" if chapter_run.final_score is None else str(chapter_run.final_score),
                    "" if chapter_run.final_findings is None else str(chapter_run.final_findings),
                    str(len(chapter_run.attempts)),
                    last_attempt_label(chapter_run),
                    str(len(chapter_run.patch_attempts)),
                    str(accepted_repairs),
                    chapter_run.report_path or "",
                ]
            )
            + " |"
        )
    provider_failure_lines: list[str] = []
    if inspection.provider_failures:
        provider_failure_lines = [
            "",
            "## Provider Failures",
            "",
            "| Chapter | Role | Provider | Fallback Used | Reason |",
            "| --- | --- | --- | --- | --- |",
        ]
        for failure in inspection.provider_failures:
            provider_label = "/".join(piece for piece in [failure.provider, failure.model] if piece)
            provider_failure_lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_cell(failure.chapter),
                        _markdown_cell(failure.role),
                        _markdown_cell(provider_label),
                        "yes" if failure.fallback_used else "no",
                        _markdown_cell(failure.reason),
                    ]
                )
                + " |"
            )
    path.write_text(
        "\n".join(
            [
                f"# Batch Report: {manifest.story_slug}",
                "",
                f"- Run: `{manifest.run_id}`",
                f"- Mode: `{manifest.mode}`",
                f"- Chapters: {manifest.summary.total_chapters}",
                f"- Packaged: {manifest.summary.packaged}",
                f"- Review required: {manifest.summary.review_required}",
                f"- Failed: {manifest.summary.failed}",
                f"- Agentic evidence: {evidence_label} - {evidence.reason}",
                "",
                *rows,
                *provider_failure_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_batch_status(path: Path, manifest: BatchManifest) -> None:
    manifest.artifacts["status_json"] = str(path.name)
    report = build_batch_inspection_report(manifest)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def _collect_final_texts(manifest: BatchManifest) -> dict[str, str]:
    final_texts: dict[str, str] = {}
    for chapter, chapter_run in manifest.chapters.items():
        if chapter_run.status not in {"packaged", "review_required"} or not chapter_run.final_path:
            continue
        final_path = Path(chapter_run.final_path)
        if final_path.exists():
            final_texts[chapter] = final_path.read_text(encoding="utf-8")
    return final_texts


def _clean_stale_aggregate_artifacts(review_dir: Path, story_slug: str) -> None:
    if not review_dir.exists():
        return
    for suffix in ("txt", "epub"):
        for path in review_dir.glob(f"{story_slug}_*.{suffix}"):
            path.unlink()


def _write_aggregate_artifacts(
    *,
    run_dir: Path,
    story_slug: str,
    story_title: str,
    final_texts: dict[str, str],
    skip_epub: bool,
    manifest: BatchManifest,
) -> ArtifactQAReport | None:
    if not final_texts:
        return None
    range_label = chapter_range_label(list(final_texts))
    review_dir = run_dir / "review"
    _clean_stale_aggregate_artifacts(review_dir, story_slug)
    txt_path = build_txt_collection(
        output_path=review_dir / f"{story_slug}_{range_label}.txt",
        chapters=final_texts,
    )
    epub_path = None
    if not skip_epub:
        epub_path = build_epub_collection(
            output_path=review_dir / f"{story_slug}_{range_label}.epub",
            story_title=story_title,
            chapters=final_texts,
        )
    artifact_qa = _artifact_qa(
        txt_path=txt_path,
        epub_path=epub_path,
        expected_chapters=len(final_texts),
    )
    manifest.artifacts["txt"] = str(txt_path.relative_to(run_dir))
    if epub_path:
        manifest.artifacts["epub"] = str(epub_path.relative_to(run_dir))
    else:
        manifest.artifacts.pop("epub", None)
    manifest.artifact_qa = artifact_qa
    return artifact_qa


def _run_one_chapter(
    *,
    story_yaml: Path,
    story_source_dir: Path,
    story_baseline_dir: Path | None,
    run_dir: Path,
    manifest: BatchManifest,
    manifest_path: Path,
    chapter: str,
    provider_mode: str,
    translation_provider_name: str,
    judge_provider_name: str,
    repair_provider_name: str,
    record_cache: bool,
    cache_dir: Path | None,
    model_name: str | None,
    tool_agent_enabled: bool,
    terminology_consensus: TerminologyConsensusConfig,
    allow_live_provider_fallback: bool,
    live_provider_fallback_state: LiveProviderFallbackState | None,
    seed: int,
    overwrite: bool,
    force: bool,
    retry_review_required: bool,
    skip_epub: bool,
    allow_source_qa_fail: bool,
    report_mode: str | None,
) -> None:
    chapter_run = manifest.chapters[chapter]
    original_status = chapter_run.status
    if chapter_run.status in COMPLETE_STATUSES and not force:
        if chapter_run.status != "review_required" or not retry_review_required:
            write_batch_manifest(manifest_path, manifest)
            return
    replace_existing_chapter_run = force or overwrite
    if original_status not in COMPLETE_STATUSES or (original_status == "review_required" and retry_review_required):
        replace_existing_chapter_run = True
    chapter_run.status = "running"
    chapter_run.source_path = str(story_source_dir / f"{chapter}.txt")
    chapter_run.error = None
    chapter_run.final_score = None
    chapter_run.final_findings = None
    _clear_tool_agent_metadata(chapter_run)
    attempt = _start_attempt(manifest, chapter_run, action="run_chapter")
    write_batch_manifest(manifest_path, manifest)
    try:
        result = run_demo_pipeline(
            story_yaml,
            chapter_override=chapter,
            provider_mode=provider_mode,
            translation_provider_name=translation_provider_name,
            judge_provider_name=judge_provider_name,
            repair_provider_name=repair_provider_name,
            record_cache=record_cache,
            cache_dir=cache_dir,
            model_name=model_name,
            tool_agent_enabled=tool_agent_enabled,
            terminology_consensus=terminology_consensus,
            allow_live_provider_fallback=allow_live_provider_fallback,
            live_provider_fallback_state=live_provider_fallback_state,
            run_id=chapter,
            seed=seed,
            overwrite=replace_existing_chapter_run,
            skip_epub=skip_epub,
            allow_source_qa_fail=allow_source_qa_fail,
            runs_dir=run_dir / "chapters",
            report_mode=report_mode,
        )
        final_path = result.run_dir / "translated_final" / f"{chapter}.txt"
        chapter_run.chapter_run_dir = str(result.run_dir)
        chapter_run.final_path = str(final_path)
        chapter_run.report_path = str(result.report_path)
        chapter_run.final_score = result.qa_final.score
        chapter_run.final_findings = result.qa_final.summary.total_findings
        chapter_run.repair_decisions = result.repair_decisions
        chapter_run.patch_attempts = result.patch_attempts
        chapter_run.provider_calls = result.provider_calls
        _copy_tool_agent_metadata(chapter_run, getattr(result, "tool_agent", None))
        chapter_run.baseline_comparison = _baseline_comparison(
            baseline_dir=story_baseline_dir,
            chapter=chapter,
            final_path=final_path,
        )
        if result.tool_agent is None:
            chapter_run.status = "packaged" if result.qa_final.summary.total_findings == 0 else "review_required"
        elif result.tool_agent.final_status == "failed":
            chapter_run.status = "failed"
        elif result.tool_agent.final_status == "verified" and result.qa_final.summary.total_findings == 0:
            chapter_run.status = "packaged"
        else:
            chapter_run.status = "review_required"
        attempt.status = "fail" if chapter_run.status == "failed" else "ok" if chapter_run.status == "packaged" else "warn"
        attempt_message = f"Chapter {chapter_run.status} with {result.qa_final.summary.total_findings} final QA findings."
        if result.provider_failure_messages:
            attempt_message = "; ".join(result.provider_failure_messages + [attempt_message])
        attempt.message = attempt_message
    except Exception as exc:  # noqa: BLE001 - batch must persist chapter failure details.
        chapter_run.status = "failed"
        chapter_run.error = str(exc)
        if tool_agent_enabled:
            _preserve_tool_agent_artifacts_from_disk(run_dir=run_dir, chapter_run=chapter_run, chapter=chapter)
        attempt.status = "fail"
        attempt.message = str(exc)
    write_batch_manifest(manifest_path, manifest)


def run_batch_pipeline(
    story_yaml: Path,
    *,
    chapters: list[str],
    provider_mode: str = "offline",
    translation_provider_name: str = "offline",
    judge_provider_name: str = "offline",
    repair_provider_name: str = "offline",
    record_cache: bool = False,
    cache_dir: Path | None = None,
    model_name: str | None = None,
    allow_live_provider_fallback: bool = False,
    tool_agent_enabled: bool = False,
    terminology_consensus: TerminologyConsensusConfig | None = None,
    run_id: str | None = None,
    seed: int = 7,
    overwrite: bool = False,
    force: bool = False,
    skip_epub: bool = False,
    allow_source_qa_fail: bool = False,
    report_mode: str | None = None,
    write_proof: bool = False,
    replay_source_run_dir: Path | None = None,
) -> BatchPipelineResult:
    if not chapters:
        raise ValueError("batch run requires at least one chapter.")
    _validate_live_batch_mode(
        provider_mode=provider_mode,
        translation_provider_name=translation_provider_name,
        judge_provider_name=judge_provider_name,
        repair_provider_name=repair_provider_name,
        record_cache=record_cache,
        cache_dir=cache_dir,
        model_name=model_name,
        allow_live_provider_fallback=allow_live_provider_fallback,
        tool_agent_enabled=tool_agent_enabled,
    )
    story = load_story_config(story_yaml)
    effective_terminology_consensus = (
        story.agent.terminology_consensus
        if terminology_consensus is None
        else terminology_consensus
    )
    _validate_terminology_batch_config(
        config=effective_terminology_consensus,
        tool_agent_enabled=tool_agent_enabled,
        provider_mode=provider_mode,
        repair_provider_name=repair_provider_name,
        record_cache=record_cache,
        cache_dir=cache_dir,
    )
    missing = [chapter for chapter in chapters if not (story.paths.source_dir / f"{chapter}.txt").exists()]
    if missing:
        raise ValueError(f"Missing source chapters: {', '.join(missing)}")

    run_id = run_id or make_run_id(f"{story.slug}_batch")
    run_dir = prepare_run_dir(story.paths.runs_dir, run_id, overwrite=overwrite)
    manifest_path = run_dir / "batch_manifest.json"
    providers = _provider_labels(
        provider_mode=provider_mode,
        translation_provider_name=translation_provider_name,
        judge_provider_name=judge_provider_name,
        repair_provider_name=repair_provider_name,
        cache_dir=cache_dir,
        record_cache=record_cache,
        model_name=model_name,
    )
    manifest = BatchManifest.create(
        run_id=run_id,
        story_slug=story.slug,
        title=story.title,
        story_yaml=story_yaml,
        chapters=chapters,
        mode=provider_mode,
        providers=providers,
        run_dir=run_dir,
        replay_source_run_dir=str(replay_source_run_dir.resolve()) if replay_source_run_dir else None,
        run_config=build_batch_run_config(
            provider_mode=provider_mode,
            translation_provider_name=translation_provider_name,
            judge_provider_name=judge_provider_name,
            repair_provider_name=repair_provider_name,
            record_cache=record_cache,
            cache_dir=cache_dir,
            model_name=model_name,
            allow_live_provider_fallback=allow_live_provider_fallback,
            tool_agent_enabled=tool_agent_enabled,
            terminology_consensus=effective_terminology_consensus,
        ),
    )
    write_batch_manifest(manifest_path, manifest)
    live_provider_fallback_state = LiveProviderFallbackState()

    for index, chapter in enumerate(chapters):
        _run_one_chapter(
            story_yaml=story_yaml,
            story_source_dir=story.paths.source_dir,
            story_baseline_dir=story.paths.baseline_dir,
            run_dir=run_dir,
            manifest=manifest,
            manifest_path=manifest_path,
            chapter=chapter,
            provider_mode=provider_mode,
            translation_provider_name=translation_provider_name,
            judge_provider_name=judge_provider_name,
            repair_provider_name=repair_provider_name,
            record_cache=record_cache,
            cache_dir=cache_dir,
            model_name=model_name,
            tool_agent_enabled=tool_agent_enabled,
            terminology_consensus=effective_terminology_consensus,
            allow_live_provider_fallback=allow_live_provider_fallback,
            live_provider_fallback_state=live_provider_fallback_state,
            seed=seed + index,
            overwrite=overwrite,
            force=force,
            retry_review_required=False,
            skip_epub=skip_epub,
            allow_source_qa_fail=allow_source_qa_fail,
            report_mode=report_mode,
        )
    final_texts = _collect_final_texts(manifest)
    artifact_qa = _write_aggregate_artifacts(
        run_dir=run_dir,
        story_slug=story.slug,
        story_title=story.title,
        final_texts=final_texts,
        skip_epub=skip_epub,
        manifest=manifest,
    )
    _write_batch_report(run_dir / "batch_report.md", manifest)
    manifest.artifacts["batch_report"] = "batch_report.md"
    _write_batch_status(run_dir / "batch_status.json", manifest)
    if write_proof:
        write_batch_proof_artifacts(run_dir, manifest)
    else:
        write_batch_manifest(manifest_path, manifest)
    return BatchPipelineResult(
        run_dir=run_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        artifact_qa=artifact_qa,
    )


def replay_batch_pipeline(
    source_run_dir: Path,
    *,
    chapters: list[str] | None = None,
    run_id: str | None = None,
    seed: int = 7,
    overwrite: bool = False,
    skip_epub: bool = False,
    allow_source_qa_fail: bool = False,
    report_mode: str | None = None,
    write_proof: bool = True,
) -> BatchPipelineResult:
    source_manifest = load_batch_manifest(source_run_dir / "batch_manifest.json")
    if source_manifest.run_config is None:
        raise ValueError("Source batch manifest has no run_config; cannot derive replay configuration.")
    _validate_tool_agent_batch_config(
        tool_agent_enabled=source_manifest.run_config.tool_agent_enabled,
        provider_mode="replay",
        repair_provider_name=source_manifest.run_config.repair_provider,
        model_name=source_manifest.run_config.model_name,
    )
    if all(
        provider == "offline"
        for provider in (
            source_manifest.run_config.translation_provider,
            source_manifest.run_config.judge_provider,
            source_manifest.run_config.repair_provider,
        )
    ):
        raise ValueError("Source batch run_config has no non-offline provider to replay.")
    selected_chapters = chapters or list(source_manifest.chapters)
    missing_chapters = [chapter for chapter in selected_chapters if chapter not in source_manifest.chapters]
    if missing_chapters:
        raise ValueError(f"Selected chapter(s) are not in the source batch manifest: {', '.join(missing_chapters)}")
    source_agent_action_calls = any(
        call.namespace == "agent_action"
        for chapter in selected_chapters
        for call in source_manifest.chapters[chapter].provider_calls
    )
    cache_dir = _validate_replay_cache_ready_for_config(
        source_manifest.run_config,
        required_extra_namespaces=["agent_action"] if source_agent_action_calls else None,
        observed_namespaces={
            call.namespace
            for chapter in selected_chapters
            for call in source_manifest.chapters[chapter].provider_calls
            if call.provider != "offline"
        },
    )

    _validate_source_manifest_replayable(source_manifest, selected_chapters)

    return run_batch_pipeline(
        Path(source_manifest.story_yaml),
        chapters=selected_chapters,
        provider_mode="replay",
        translation_provider_name=source_manifest.run_config.translation_provider,
        judge_provider_name=source_manifest.run_config.judge_provider,
        repair_provider_name=source_manifest.run_config.repair_provider,
        record_cache=False,
        cache_dir=cache_dir,
        model_name=source_manifest.run_config.model_name or _model_from_manifest(source_manifest),
        tool_agent_enabled=source_manifest.run_config.tool_agent_enabled,
        terminology_consensus=source_manifest.run_config.terminology_consensus,
        run_id=run_id or f"{source_manifest.run_id}_replay",
        seed=seed,
        overwrite=overwrite,
        skip_epub=skip_epub,
        allow_source_qa_fail=allow_source_qa_fail,
        report_mode=report_mode,
        write_proof=write_proof,
        replay_source_run_dir=source_run_dir,
    )


def _proof_failure_message(label: str, report: BatchProofReport) -> str:
    blockers = "; ".join(report.blockers) if report.blockers else "proof gates did not pass"
    return f"{label} proof failed for {report.run_id}: {blockers}"


def render_live_proof_summary_markdown(result: BatchLiveProofResult) -> str:
    live_blockers = "\n".join(f"- {blocker}" for blocker in result.live_proof.blockers) or "- none"
    replay_blockers = (
        "\n".join(f"- {blocker}" for blocker in result.replay_proof.blockers)
        if result.replay_proof
        else "- not attempted"
    ) or "- none"
    replay_run_label = (
        f"`{result.replay_result.manifest.run_id}` at `{result.replay_result.run_dir}`"
        if result.replay_result
        else "`not attempted`"
    )
    replay_passed = str(result.replay_proof.proof_passed).lower() if result.replay_proof else "not attempted"
    replay_agentic = str(result.replay_proof.gates.get("agentic", False)).lower() if result.replay_proof else "not attempted"
    replay_replayable = str(result.replay_proof.gates.get("replayable", False)).lower() if result.replay_proof else "not attempted"
    return (
        "\n".join(
            [
                "# Live Proof Summary",
                "",
                f"- Story YAML: `{result.story_yaml}`",
                f"- Chapters: `{', '.join(result.chapters)}`",
                f"- Proof passed: `{str(result.proof_passed).lower()}`",
                "",
                "## Runs",
                f"- Live run: `{result.live_result.manifest.run_id}` at `{result.live_result.run_dir}`",
                f"- Replay run: {replay_run_label}",
                "",
                "## Live Proof",
                f"- Passed: `{str(result.live_proof.proof_passed).lower()}`",
                f"- Agentic: `{str(result.live_proof.gates.get('agentic', False)).lower()}`",
                f"- Replayable: `{str(result.live_proof.gates.get('replayable', False)).lower()}`",
                "- Blockers:",
                live_blockers,
                "",
                "## Replay Proof",
                f"- Passed: `{replay_passed}`",
                f"- Agentic: `{replay_agentic}`",
                f"- Replayable: `{replay_replayable}`",
                "- Blockers:",
                replay_blockers,
            ]
        )
        + "\n"
    )


def write_live_proof_summary_artifacts(result: BatchLiveProofResult) -> BatchLiveProofResult:
    live_run_dir = result.live_result.run_dir
    live_run_dir.mkdir(parents=True, exist_ok=True)
    result.artifacts["live_proof_summary_json"] = "live_proof_summary.json"
    result.artifacts["live_proof_summary_markdown"] = "live_proof_summary.md"
    (live_run_dir / "live_proof_summary.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    (live_run_dir / "live_proof_summary.md").write_text(render_live_proof_summary_markdown(result), encoding="utf-8")
    return result


def run_live_proof_pipeline(
    story_yaml: Path,
    *,
    chapters: list[str],
    translation_provider_name: str = "offline",
    judge_provider_name: str = "openai",
    repair_provider_name: str = "offline",
    cache_dir: Path | None = Path(".agentic_cache"),
    model_name: str | None = None,
    run_id: str | None = None,
    replay_run_id: str | None = None,
    seed: int = 7,
    overwrite: bool = False,
    skip_epub: bool = False,
    allow_source_qa_fail: bool = False,
    report_mode: str | None = "excerpt",
    tool_agent_enabled: bool = False,
    terminology_consensus: TerminologyConsensusConfig | None = None,
) -> BatchLiveProofResult:
    from .preflight import run_preflight

    preflight = run_preflight(
        story_yaml,
        chapters=chapters,
        provider_mode="live",
        translation_provider_name=translation_provider_name,
        judge_provider_name=judge_provider_name,
        repair_provider_name=repair_provider_name,
        record_cache=True,
        cache_dir=cache_dir,
        model_name=model_name,
        tool_agent_enabled=tool_agent_enabled,
        terminology_consensus=terminology_consensus,
    )
    if not preflight.passed:
        failures = [check.message for check in preflight.checks if check.status == "fail"]
        raise ValueError("Live proof preflight failed: " + "; ".join(failures))

    live_result = run_batch_pipeline(
        story_yaml,
        chapters=chapters,
        provider_mode="live",
        translation_provider_name=translation_provider_name,
        judge_provider_name=judge_provider_name,
        repair_provider_name=repair_provider_name,
        record_cache=True,
        cache_dir=cache_dir,
        model_name=model_name,
        run_id=run_id,
        seed=seed,
        overwrite=overwrite,
        skip_epub=skip_epub,
        allow_source_qa_fail=allow_source_qa_fail,
        report_mode=report_mode,
        write_proof=True,
        tool_agent_enabled=tool_agent_enabled,
        terminology_consensus=terminology_consensus,
    )
    live_proof = build_batch_proof_report(live_result.manifest)
    if tool_agent_enabled:
        tool_calls = [
            call
            for chapter_run in live_result.manifest.chapters.values()
            for call in chapter_run.provider_calls
            if call.namespace == "agent_action"
        ]
        if live_proof.tool_agent_evidence.episodes_observed == 0:
            raise RuntimeError("Live proof requested tool-agent evidence but observed no episode.")
        if any(call.cache_hit for call in tool_calls):
            raise RuntimeError("Fresh live tool-agent proof cannot reuse cached action calls.")
    if not live_proof.proof_passed:
        write_live_proof_summary_artifacts(
            BatchLiveProofResult(
                story_yaml=str(story_yaml),
                chapters=chapters,
                proof_passed=False,
                live_result=live_result,
                live_proof=live_proof,
            )
        )
        raise RuntimeError(_proof_failure_message("Live", live_proof))

    replay_result = replay_batch_pipeline(
        live_result.run_dir,
        chapters=chapters,
        run_id=replay_run_id or f"{live_result.manifest.run_id}_replay",
        seed=seed,
        overwrite=overwrite,
        skip_epub=skip_epub,
        allow_source_qa_fail=allow_source_qa_fail,
        report_mode=report_mode,
        write_proof=True,
    )
    replay_proof = build_batch_proof_report(replay_result.manifest)
    result = BatchLiveProofResult(
        story_yaml=str(story_yaml),
        chapters=chapters,
        proof_passed=live_proof.proof_passed and replay_proof.proof_passed,
        live_result=live_result,
        live_proof=live_proof,
        replay_result=replay_result,
        replay_proof=replay_proof,
    )
    write_live_proof_summary_artifacts(result)
    if not replay_proof.proof_passed:
        raise RuntimeError(_proof_failure_message("Replay", replay_proof))
    return result


def resume_batch_pipeline(
    run_dir: Path,
    *,
    chapters: list[str] | None = None,
    provider_mode: str | None = None,
    translation_provider_name: str | None = None,
    judge_provider_name: str | None = None,
    repair_provider_name: str | None = None,
    record_cache: bool = False,
    cache_dir: Path | None = None,
    model_name: str | None = None,
    allow_live_provider_fallback: bool = False,
    tool_agent_enabled: bool | None = None,
    seed: int = 7,
    force: bool = False,
    retry_review_required: bool = False,
    skip_epub: bool = False,
    allow_source_qa_fail: bool = False,
    report_mode: str | None = None,
    write_proof: bool = False,
) -> BatchPipelineResult:
    manifest_path = run_dir / "batch_manifest.json"
    manifest = load_batch_manifest(manifest_path)
    story_yaml = Path(manifest.story_yaml)
    story = load_story_config(story_yaml)
    provider_mode = provider_mode or manifest.mode
    translation_provider_name = translation_provider_name or manifest.providers.get("translation", ProviderLabel(provider="offline", model="")).provider
    judge_provider_name = judge_provider_name or manifest.providers.get("judge", ProviderLabel(provider="offline", model="")).provider
    repair_provider_name = repair_provider_name or manifest.providers.get("repair", ProviderLabel(provider="offline", model="")).provider
    model_name = model_name or _model_from_manifest(manifest)
    terminology_consensus = (
        manifest.run_config.terminology_consensus
        if manifest.run_config is not None
        else story.agent.terminology_consensus
    )
    if tool_agent_enabled is None:
        tool_agent_enabled = manifest.run_config.tool_agent_enabled if manifest.run_config else False
    _validate_live_batch_mode(
        provider_mode=provider_mode,
        translation_provider_name=translation_provider_name,
        judge_provider_name=judge_provider_name,
        repair_provider_name=repair_provider_name,
        record_cache=record_cache,
        cache_dir=cache_dir,
        model_name=model_name,
        allow_live_provider_fallback=allow_live_provider_fallback,
        tool_agent_enabled=tool_agent_enabled,
    )
    _validate_terminology_batch_config(
        config=terminology_consensus,
        tool_agent_enabled=tool_agent_enabled,
        provider_mode=provider_mode,
        repair_provider_name=repair_provider_name,
        record_cache=record_cache,
        cache_dir=cache_dir,
    )
    providers = _provider_labels(
        provider_mode=provider_mode,
        translation_provider_name=translation_provider_name,
        judge_provider_name=judge_provider_name,
        repair_provider_name=repair_provider_name,
        cache_dir=cache_dir,
        record_cache=record_cache,
        model_name=model_name,
    )
    _stamp_run_config(
        manifest,
        provider_mode=provider_mode,
        translation_provider_name=translation_provider_name,
        judge_provider_name=judge_provider_name,
        repair_provider_name=repair_provider_name,
        providers=providers,
        record_cache=record_cache,
        cache_dir=cache_dir,
        model_name=model_name,
        allow_live_provider_fallback=allow_live_provider_fallback,
        tool_agent_enabled=tool_agent_enabled,
        terminology_consensus=terminology_consensus,
    )
    write_batch_manifest(manifest_path, manifest)
    live_provider_fallback_state = LiveProviderFallbackState()

    selected_chapters = chapters or list(manifest.chapters)
    missing_chapters = [chapter for chapter in selected_chapters if chapter not in manifest.chapters]
    if missing_chapters:
        raise ValueError(f"Selected chapter(s) are not in the batch manifest: {', '.join(missing_chapters)}")

    for index, chapter in enumerate(selected_chapters):
        _run_one_chapter(
            story_yaml=story_yaml,
            story_source_dir=story.paths.source_dir,
            story_baseline_dir=story.paths.baseline_dir,
            run_dir=run_dir,
            manifest=manifest,
            manifest_path=manifest_path,
            chapter=chapter,
            provider_mode=provider_mode,
            translation_provider_name=translation_provider_name,
            judge_provider_name=judge_provider_name,
            repair_provider_name=repair_provider_name,
            record_cache=record_cache,
            cache_dir=cache_dir,
            model_name=model_name,
            tool_agent_enabled=tool_agent_enabled,
            terminology_consensus=terminology_consensus,
            allow_live_provider_fallback=allow_live_provider_fallback,
            live_provider_fallback_state=live_provider_fallback_state,
            seed=seed + index,
            overwrite=False,
            force=force,
            retry_review_required=retry_review_required,
            skip_epub=skip_epub,
            allow_source_qa_fail=allow_source_qa_fail,
            report_mode=report_mode,
        )
    final_texts = _collect_final_texts(manifest)
    artifact_qa = _write_aggregate_artifacts(
        run_dir=run_dir,
        story_slug=manifest.story_slug,
        story_title=manifest.title,
        final_texts=final_texts,
        skip_epub=skip_epub,
        manifest=manifest,
    )
    _write_batch_report(run_dir / "batch_report.md", manifest)
    manifest.artifacts["batch_report"] = "batch_report.md"
    _write_batch_status(run_dir / "batch_status.json", manifest)
    if write_proof:
        write_batch_proof_artifacts(run_dir, manifest)
    else:
        write_batch_manifest(manifest_path, manifest)
    return BatchPipelineResult(
        run_dir=run_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        artifact_qa=artifact_qa,
    )


def refresh_batch_pipeline(
    run_dir: Path,
    *,
    chapters: list[str] | None = None,
    skip_epub: bool = False,
    write_proof: bool = False,
) -> BatchPipelineResult:
    manifest_path = run_dir / "batch_manifest.json"
    manifest = load_batch_manifest(manifest_path)
    story = load_story_config(Path(manifest.story_yaml))
    glossary = load_glossary(story.paths.glossary_path)

    selected_chapters = chapters or list(manifest.chapters)
    missing_chapters = [chapter for chapter in selected_chapters if chapter not in manifest.chapters]
    if missing_chapters:
        raise ValueError(f"Selected chapter(s) are not in the batch manifest: {', '.join(missing_chapters)}")

    for chapter in selected_chapters:
        chapter_run = manifest.chapters[chapter]
        chapter_run_dir = _chapter_run_dir(run_dir, chapter_run, chapter)
        final_path = Path(chapter_run.final_path) if chapter_run.final_path else chapter_run_dir / "translated_final" / f"{chapter}.txt"
        source_path = _chapter_source_path(run_dir, chapter_run, chapter, story.paths.source_dir)
        chapter_run.chapter_run_dir = str(chapter_run_dir)
        chapter_run.final_path = str(final_path)
        chapter_run.source_path = str(source_path)

        if not final_path.exists():
            chapter_run.status = "failed"
            chapter_run.error = f"Missing final translation file: {final_path}"
            chapter_run.final_score = None
            chapter_run.final_findings = None
            continue
        if not source_path.exists():
            chapter_run.status = "failed"
            chapter_run.error = f"Missing source chapter file: {source_path}"
            chapter_run.final_score = None
            chapter_run.final_findings = None
            continue

        qa_report = run_translation_qa(
            run_id=chapter,
            story_slug=manifest.story_slug,
            chapter=chapter,
            source_text=source_path.read_text(encoding="utf-8"),
            translated_text=final_path.read_text(encoding="utf-8"),
            glossary=glossary,
        )
        chapter_run_dir.mkdir(parents=True, exist_ok=True)
        (chapter_run_dir / "qa_final.json").write_text(qa_report.model_dump_json(indent=2), encoding="utf-8")
        chapter_run.final_score = qa_report.score
        chapter_run.final_findings = qa_report.summary.total_findings
        chapter_run.error = None
        chapter_run.status = "packaged" if qa_report.summary.total_findings == 0 else "review_required"
        chapter_run.baseline_comparison = _baseline_comparison(
            baseline_dir=story.paths.baseline_dir,
            chapter=chapter,
            final_path=final_path,
        )

    final_texts = _collect_final_texts(manifest)
    artifact_qa = _write_aggregate_artifacts(
        run_dir=run_dir,
        story_slug=manifest.story_slug,
        story_title=manifest.title,
        final_texts=final_texts,
        skip_epub=skip_epub,
        manifest=manifest,
    )
    _write_batch_report(run_dir / "batch_report.md", manifest)
    manifest.artifacts["batch_report"] = "batch_report.md"
    _write_batch_status(run_dir / "batch_status.json", manifest)
    if write_proof:
        write_batch_proof_artifacts(run_dir, manifest)
    else:
        write_batch_manifest(manifest_path, manifest)
    return BatchPipelineResult(
        run_dir=run_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        artifact_qa=artifact_qa,
    )


def _default_replacement_note(chapter: str, old_text: str, new_text: str) -> str:
    old_preview = old_text.replace("\n", "\\n")
    new_preview = new_text.replace("\n", "\\n")
    if len(old_preview) > 80:
        old_preview = old_preview[:77] + "..."
    if len(new_preview) > 80:
        new_preview = new_preview[:77] + "..."
    return f"Manual text replacement in {chapter}: {old_preview} -> {new_preview}"


def apply_manual_text_replacement(
    run_dir: Path,
    *,
    chapter: str,
    old_text: str,
    new_text: str,
    reviewer: str = "human",
    note: str | None = None,
    refresh_only: bool = False,
    skip_epub: bool = False,
    write_proof: bool = False,
) -> ManualTextReplacementResult:
    old_text = old_text.strip()
    if not old_text:
        raise ValueError("Manual replacement old text must be non-empty.")
    manifest_path = run_dir / "batch_manifest.json"
    manifest = load_batch_manifest(manifest_path)
    if chapter not in manifest.chapters:
        raise ValueError(f"Selected chapter is not in the batch manifest: {chapter}")
    chapter_run = manifest.chapters[chapter]
    chapter_run_dir = Path(chapter_run.chapter_run_dir) if chapter_run.chapter_run_dir else run_dir / "chapters" / chapter
    final_path = Path(chapter_run.final_path) if chapter_run.final_path else chapter_run_dir / "translated_final" / f"{chapter}.txt"
    if not final_path.exists():
        raise ValueError(f"Final translation file is missing for chapter {chapter}: {final_path}")
    before = final_path.read_text(encoding="utf-8")
    occurrence_count = before.count(old_text)
    if occurrence_count == 0:
        raise ValueError(f"Manual replacement old text was not found in chapter {chapter}.")
    final_path.write_text(before.replace(old_text, new_text), encoding="utf-8")

    effective_note = None
    if refresh_only:
        batch_result = refresh_batch_pipeline(run_dir, chapters=[chapter], skip_epub=skip_epub, write_proof=write_proof)
    else:
        effective_note = note.strip() if note and note.strip() else _default_replacement_note(chapter, old_text, new_text)
        batch_result = accept_reviewed_chapters(
            run_dir,
            chapters=[chapter],
            reviewer=reviewer,
            note=effective_note,
            skip_epub=skip_epub,
            write_proof=write_proof,
        )

    refreshed_chapter = batch_result.manifest.chapters[chapter]
    return ManualTextReplacementResult(
        run_id=batch_result.manifest.run_id,
        run_dir=str(run_dir),
        chapter=chapter,
        final_path=str(final_path),
        old_text=old_text,
        new_text=new_text,
        occurrence_count=occurrence_count,
        refresh_only=refresh_only,
        reviewer=None if refresh_only else reviewer,
        note=effective_note,
        status_after=refreshed_chapter.status,
        final_score_after=refreshed_chapter.final_score,
        final_findings_after=refreshed_chapter.final_findings,
        summary_after=batch_result.manifest.summary,
    )


NOTE_PANEL_FIRST_RE = re.compile(
    r"^\[\s*(?:note|notes|註|注)\s*[:：]?\s*1\s*[\.\):：、，,．]\s*(?P<body>.*?)\s*\]$",
    re.IGNORECASE,
)
NUMBERED_PANEL_RE = re.compile(r"^\[\s*(?P<number>\d+)\s*[\.\):：、，,．]\s*(?P<body>.*?)\s*\]$")


def _merge_numbered_note_panel_splits(text: str) -> tuple[str, int]:
    lines = text.splitlines()
    output: list[str] = []
    replacements = 0
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        first_match = NOTE_PANEL_FIRST_RE.match(stripped)
        if not first_match:
            output.append(lines[index])
            index += 1
            continue

        parts = [stripped[1:-1].strip()]
        scan = index + 1
        expected_number = 2
        matched_end = index + 1
        while scan < len(lines):
            if not lines[scan].strip():
                scan += 1
                continue
            number_match = NUMBERED_PANEL_RE.match(lines[scan].strip())
            if not number_match or int(number_match.group("number")) != expected_number:
                break
            parts.append(lines[scan].strip()[1:-1].strip())
            expected_number += 1
            matched_end = scan + 1
            scan += 1

        if len(parts) == 1:
            output.append(lines[index])
            index += 1
            continue

        output.append("[" + " ".join(part for part in parts if part) + "]")
        replacements += 1
        index = matched_end

    trailing_newline = "\n" if text.endswith("\n") else ""
    return "\n".join(output) + trailing_newline, replacements


def _panel_inner_text(panel_text: str) -> str | None:
    stripped = panel_text.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        return None
    return stripped[1:-1].strip()


def _replace_panel_line_pair(text: str, first_line_number: int, second_line_number: int, merged_line: str) -> str:
    lines = text.splitlines()
    first_index = first_line_number - 1
    second_index = second_line_number - 1
    if first_index < 0 or second_index <= first_index or second_index >= len(lines):
        return text
    lines[first_index] = merged_line
    del lines[second_index]
    trailing_newline = "\n" if text.endswith("\n") else ""
    return "\n".join(lines) + trailing_newline


def _panel_length_score(source_panel_texts: list[str], final_panel_texts: list[str]) -> float:
    if len(source_panel_texts) != len(final_panel_texts):
        return float("inf")
    ratios = [
        (len(final_text) + 8) / max(len(source_text) + 4, 1)
        for source_text, final_text in zip(source_panel_texts, final_panel_texts)
    ]
    sorted_ratios = sorted(ratios)
    median = sorted_ratios[len(sorted_ratios) // 2]
    if median <= 0:
        return float("inf")
    return sum(abs(math.log(max(ratio, 0.001) / median)) for ratio in ratios)


def _merge_single_extra_panel_by_length(source_text: str, final_text: str) -> tuple[str, int]:
    source_panels = extract_panel_segments(source_text)
    final_panels = extract_panel_segments(final_text)
    if len(final_panels) != len(source_panels) + 1:
        return final_text, 0

    source_panel_texts = [panel.text for panel in source_panels]
    candidates: list[tuple[float, int, str]] = []
    for index in range(len(final_panels) - 1):
        first = final_panels[index]
        second = final_panels[index + 1]
        first_inner = _panel_inner_text(first.text)
        second_inner = _panel_inner_text(second.text)
        if first_inner is None or second_inner is None:
            continue
        merged_panel_text = f"[{first_inner} {second_inner}]"
        candidate_final_panels = (
            [panel.text for panel in final_panels[:index]]
            + [merged_panel_text]
            + [panel.text for panel in final_panels[index + 2 :]]
        )
        score = _panel_length_score(source_panel_texts, candidate_final_panels)
        candidate_text = _replace_panel_line_pair(
            final_text,
            first.line_number,
            second.line_number,
            merged_panel_text,
        )
        candidates.append((score, index, candidate_text))

    if not candidates:
        return final_text, 0
    _, _, best_text = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    if best_text == final_text:
        return final_text, 0
    return best_text, 1


def normalize_panel_splits(
    run_dir: Path,
    *,
    chapters: list[str] | None = None,
    reviewer: str = "human",
    note_prefix: str = "Merged split numbered note panels.",
    skip_epub: bool = False,
    write_proof: bool = False,
) -> PanelNormalizationResult:
    queue = collect_review_queue(run_dir)
    selected = set(chapters or queue.summary.chapters)
    panel_chapters = {item.chapter for item in queue.items if item.check_id == "system_panel_count"}
    manifest = load_batch_manifest(run_dir / "batch_manifest.json")
    story = load_story_config(Path(manifest.story_yaml))
    glossary = load_glossary(story.paths.glossary_path)
    items: list[PanelNormalizationItem] = []
    normalized_count = 0
    skipped_count = 0

    for chapter in sorted(selected):
        if chapter not in manifest.chapters:
            raise ValueError(f"Selected chapter is not in the batch manifest: {chapter}")
        if chapter not in panel_chapters:
            items.append(
                PanelNormalizationItem(
                    chapter=chapter,
                    status="skipped",
                    reason="Chapter does not currently have an unresolved panel-count finding.",
                )
            )
            skipped_count += 1
            continue
        chapter_run = manifest.chapters[chapter]
        chapter_run_dir = _chapter_run_dir(run_dir, chapter_run, chapter)
        final_path = Path(chapter_run.final_path) if chapter_run.final_path else chapter_run_dir / "translated_final" / f"{chapter}.txt"
        if not final_path.exists():
            items.append(
                PanelNormalizationItem(
                    chapter=chapter,
                    status="skipped",
                    reason=f"Final translation file is missing: {final_path}",
                )
            )
            skipped_count += 1
            continue
        before = final_path.read_text(encoding="utf-8")
        after, replacement_count = _merge_numbered_note_panel_splits(before)
        source_path = _chapter_source_path(run_dir, chapter_run, chapter, story.paths.source_dir)
        source_text = source_path.read_text(encoding="utf-8") if source_path.exists() else ""
        if replacement_count == 0 and source_text:
            after, replacement_count = _merge_single_extra_panel_by_length(source_text, before)
            if replacement_count:
                candidate_qa = run_translation_qa(
                    run_id=chapter,
                    story_slug=manifest.story_slug,
                    chapter=chapter,
                    source_text=source_text,
                    translated_text=after,
                    glossary=glossary,
                )
                if "system_panel_count" in {finding.check_id for finding in candidate_qa.findings}:
                    after = before
                    replacement_count = 0
        if replacement_count == 0 or after == before:
            items.append(
                PanelNormalizationItem(
                    chapter=chapter,
                    status="skipped",
                    reason="No adjacent numbered note panel split was found.",
                )
            )
            skipped_count += 1
            continue
        final_path.write_text(after, encoding="utf-8")
        note = f"{note_prefix} {replacement_count} merge(s) in {chapter}."
        accept_result = accept_reviewed_chapters(
            run_dir,
            chapters=[chapter],
            reviewer=reviewer,
            note=note,
            skip_epub=skip_epub,
            write_proof=write_proof,
        )
        refreshed = accept_result.manifest.chapters[chapter]
        items.append(
            PanelNormalizationItem(
                chapter=chapter,
                status="normalized",
                reason="Merged adjacent split panels into one bracketed panel.",
                replacement_count=replacement_count,
                status_after=refreshed.status,
                final_score_after=refreshed.final_score,
                final_findings_after=refreshed.final_findings,
            )
        )
        normalized_count += 1
        manifest = accept_result.manifest

    manifest = load_batch_manifest(run_dir / "batch_manifest.json")
    return PanelNormalizationResult(
        run_id=manifest.run_id,
        run_dir=str(run_dir),
        items=items,
        normalized_count=normalized_count,
        skipped_count=skipped_count,
        summary_after=manifest.summary,
    )


def accept_reviewed_chapters(
    run_dir: Path,
    *,
    chapters: list[str] | None = None,
    reviewer: str = "human",
    note: str,
    skip_epub: bool = False,
    write_proof: bool = False,
) -> BatchPipelineResult:
    note = note.strip()
    if not note:
        raise ValueError("A non-empty review note is required.")
    reviewer = reviewer.strip() or "human"

    manifest_path = run_dir / "batch_manifest.json"
    before_manifest = load_batch_manifest(manifest_path)
    selected_chapters = chapters or list(before_manifest.chapters)
    missing_chapters = [chapter for chapter in selected_chapters if chapter not in before_manifest.chapters]
    if missing_chapters:
        raise ValueError(f"Selected chapter(s) are not in the batch manifest: {', '.join(missing_chapters)}")
    status_before = {chapter: before_manifest.chapters[chapter].status for chapter in selected_chapters}

    result = refresh_batch_pipeline(run_dir, chapters=selected_chapters, skip_epub=skip_epub)
    records: list[ManualReviewRecord] = []
    for chapter in selected_chapters:
        chapter_run = result.manifest.chapters[chapter]
        record = ManualReviewRecord(
            chapter=chapter,
            reviewer=reviewer,
            note=note,
            status_before=status_before[chapter],
            status_after=chapter_run.status,
            qa_score_after=chapter_run.final_score,
            qa_findings_after=chapter_run.final_findings,
            artifact_qa_passed=result.artifact_qa.passed if result.artifact_qa else None,
        )
        chapter_run.manual_reviews.append(record)
        records.append(record)

    ledger_path = run_dir / "manual_review.jsonl"
    with ledger_path.open("a", encoding="utf-8") as ledger:
        for record in records:
            ledger.write(record.model_dump_json() + "\n")
    if write_proof:
        write_batch_proof_artifacts(run_dir, result.manifest)
    else:
        write_batch_manifest(manifest_path, result.manifest)
    return result
