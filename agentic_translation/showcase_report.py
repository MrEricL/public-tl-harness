"""Render the persisted source-to-book showcase as a standalone HTML receipt."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader


_TEMPLATE = "showcase_report.html.j2"
_MAX_TEXT = 8000
_MAX_EVIDENCE = 1800


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any, limit: int = _MAX_EVIDENCE) -> str:
    if value is None:
        return ""
    result = str(value).replace("\r\n", "\n").replace("\r", "\n")
    if len(result) <= limit:
        return result
    marker = "…[truncated]"
    return result[: max(limit - len(marker), 0)].rstrip() + marker


def _local_path(root: Path, value: str | Path | None) -> str | None:
    """Resolve a manifest path only when it stays inside the run directory."""

    if not value:
        return None
    candidate = Path(str(value))
    try:
        if candidate.is_absolute():
            candidate = candidate.resolve().relative_to(root.resolve())
        resolved = (root / candidate).resolve()
        resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return candidate.as_posix()


def _href(root: Path, value: str | Path | None) -> str | None:
    relative = _local_path(root, value)
    if relative is None or not (root / relative).is_file():
        return None
    return relative


def _read(root: Path, value: str | Path | None, limit: int = _MAX_TEXT) -> str:
    relative = _local_path(root, value)
    if relative is None:
        return ""
    try:
        return _text((root / relative).read_text(encoding="utf-8"), limit)
    except (OSError, UnicodeError):
        return ""


def _json(root: Path, value: str | Path | None) -> dict[str, Any]:
    relative = _local_path(root, value)
    if relative is None:
        return {}
    try:
        parsed = json.loads((root / relative).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _chapter_ids(manifest: Mapping[str, Any]) -> list[str]:
    values = manifest.get("chapter_ids", [])
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return [str(value) for value in values if str(value)]
    chapters = manifest.get("chapters")
    return [str(value) for value in chapters] if isinstance(chapters, Mapping) else []


def _entry(manifest: Mapping[str, Any], chapter: str) -> dict[str, Any]:
    chapters = _mapping(manifest.get("chapters"))
    value = chapters.get(chapter, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _count(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def _safe_action(value: Any) -> dict[str, Any]:
    """Keep only the typed action fields useful in an editorial trace."""

    source = _mapping(value)
    action: dict[str, Any] = {}
    for key in (
        "tool",
        "document",
        "start",
        "count",
        "query",
        "limit",
        "specialists",
        "objective",
        "term",
        "target",
        "rationale",
        "summary",
    ):
        if key not in source:
            continue
        item = source[key]
        action[key] = (
            [_text(value, 600) for value in item[:8]]
            if isinstance(item, list)
            else _text(item, 1200)
        )
    edits = source.get("edits")
    if isinstance(edits, list):
        action["edits"] = [
            {
                "old_text": _text(_mapping(edit).get("old_text"), 900),
                "new_text": _text(_mapping(edit).get("new_text"), 900),
            }
            for edit in edits[:8]
            if isinstance(edit, Mapping)
        ]
    return action


def _step(value: Any, sequence: int) -> dict[str, Any]:
    row = _mapping(value)
    action = _safe_action(row.get("action", row))
    observation = _mapping(row.get("observation"))
    kind = str(observation.get("kind", "")).lower()
    status = str(row.get("status") or "")
    if not status:
        if "reject" in kind:
            status = "REJECTED"
        elif "accept" in kind:
            status = "ACCEPTED"
        elif "pending" in kind:
            status = "PENDING"
        elif row.get("ok") is True or observation.get("ok") is True:
            status = "OK"
        else:
            status = "RECORDED"
    data = row.get("data") if isinstance(row.get("data"), Mapping) else observation.get("data")
    edits = action.get("edits") if isinstance(action.get("edits"), list) else []
    return {
        "sequence": row.get("sequence", sequence),
        "tool": str(action.get("tool") or "unknown"),
        "status": status.upper(),
        "ok": bool(row.get("ok", observation.get("ok", False))),
        "message": _text(row.get("message", observation.get("message", ""))),
        "action": action,
        "edits": edits,
        "data": dict(data) if isinstance(data, Mapping) else {},
    }


def _reviews(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = snapshot.get("specialist_reviews", [])
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    rows: list[dict[str, Any]] = []
    for value in values[:24]:
        raw = _mapping(value)
        findings: list[dict[str, Any]] = []
        raw_findings = raw.get("findings", [])
        if isinstance(raw_findings, Sequence) and not isinstance(raw_findings, (str, bytes)):
            for finding in raw_findings[:8]:
                item = _mapping(finding)
                findings.append(
                    {
                        "category": _text(item.get("category") or raw.get("role"), 80),
                        "message": _text(item.get("message")),
                        "source_excerpt": _text(item.get("source_excerpt")),
                        "translation_excerpt": _text(item.get("translation_excerpt")),
                        "blocking": bool(item.get("blocking", True)),
                    }
                )
        edits: list[dict[str, str]] = []
        raw_edits = raw.get("proposed_edits", [])
        if isinstance(raw_edits, Sequence) and not isinstance(raw_edits, (str, bytes)):
            for edit in raw_edits[:4]:
                item = _mapping(edit)
                edits.append(
                    {
                        "old_text": _text(item.get("old_text"), 900),
                        "new_text": _text(item.get("new_text"), 900),
                    }
                )
        terms: list[dict[str, str]] = []
        raw_terms = raw.get("term_suggestions", [])
        if isinstance(raw_terms, Sequence) and not isinstance(raw_terms, (str, bytes)):
            for term in raw_terms[:4]:
                item = _mapping(term)
                terms.append(
                    {
                        "term": _text(item.get("term"), 220),
                        "target": _text(item.get("target"), 220),
                        "rationale": _text(item.get("rationale"), 900),
                    }
                )
        overlaps = raw.get("overlaps", [])
        if isinstance(overlaps, str):
            overlaps = [overlaps]
        raw_steps = raw.get("steps", [])
        rows.append(
            {
                "child_id": _text(raw.get("child_id"), 180),
                "role": _text(raw.get("role") or "specialist", 80),
                "status": str(raw.get("status") or "completed").upper(),
                "summary": _text(raw.get("summary")),
                "findings": findings,
                "proposed_edits": edits,
                "term_suggestions": terms,
                "overlaps": [_text(item, 900) for item in overlaps[:8]] if isinstance(overlaps, Sequence) else [],
                "steps": [_step(item, index) for index, item in enumerate(raw_steps or [], 1)] if isinstance(raw_steps, list) else [],
                "draft_sha256": _digest(raw.get("draft_sha256")),
            }
        )
    return rows


def _review_evidence_key(review: Mapping[str, Any]) -> str:
    """Return the identity-independent evidence used to join projections.

    Session snapshots currently serialize ``SpecialistReview`` without its
    child id, while the typed receipt written by ``SpecialistRunner`` keeps
    that durable id.  The evidence fields let us join those two projections
    when the id is absent from the snapshot.  The merge below still matches
    each snapshot row to at most one receipt, so two children with identical
    evidence remain two rows when their durable ids differ.
    """

    return json.dumps(
        {
            key: review.get(key)
            for key in (
                "role",
                "status",
                "summary",
                "findings",
                "proposed_edits",
                "term_suggestions",
                "overlaps",
                "draft_sha256",
            )
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _merge_reviews(
    snapshot_reviews: Sequence[Mapping[str, Any]],
    child_reviews: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge the snapshot projection with durable child receipts once.

    A session snapshot is the convenient summary, but child JSON files are
    the durable source of identity.  Prefer a receipt when the two rows join;
    retain unmatched rows from either source.  Matching is one-to-one by
    evidence rather than a set membership check, which preserves distinct
    children that happen to return the same review.
    """

    merged: list[dict[str, Any]] = []
    consumed_receipts: set[int] = set()
    emitted_child_ids: set[str] = set()
    receipt_by_child_id = {
        str(review.get("child_id")): index
        for index, review in enumerate(child_reviews)
        if review.get("child_id")
    }

    for snapshot_review in snapshot_reviews:
        review = dict(snapshot_review)
        child_id = str(review.get("child_id") or "")
        receipt_index: int | None = None
        if child_id:
            receipt_index = receipt_by_child_id.get(child_id)
        else:
            snapshot_key = _review_evidence_key(review)
            receipt_index = next(
                (
                    index
                    for index, receipt in enumerate(child_reviews)
                    if index not in consumed_receipts
                    and _review_evidence_key(receipt) == snapshot_key
                ),
                None,
            )

        if receipt_index is not None and receipt_index not in consumed_receipts:
            receipt = dict(child_reviews[receipt_index])
            consumed_receipts.add(receipt_index)
            child_id = str(receipt.get("child_id") or child_id)
            if child_id and child_id in emitted_child_ids:
                continue
            if child_id:
                emitted_child_ids.add(child_id)
            merged.append(receipt)
            continue

        if child_id and child_id in emitted_child_ids:
            continue
        if child_id:
            emitted_child_ids.add(child_id)
        merged.append(review)

    for index, child_review in enumerate(child_reviews):
        if index in consumed_receipts:
            continue
        review = dict(child_review)
        child_id = str(review.get("child_id") or "")
        if child_id and child_id in emitted_child_ids:
            continue
        consumed_receipts.add(index)
        if child_id:
            emitted_child_ids.add(child_id)
        merged.append(review)
    return merged


def _child_reviews(root: Path, chapter: str) -> list[dict[str, Any]]:
    """Read typed child receipts when a snapshot has not inlined them."""

    child_dir = root / "chapters" / chapter / "children"
    values: list[dict[str, Any]] = []
    if not child_dir.is_dir():
        return values
    for path in sorted(child_dir.glob("*.json")):
        raw = _json(root, path.relative_to(root))
        durable_child_id = raw.get("child_id")
        if isinstance(raw.get("review"), Mapping):
            raw = dict(raw["review"])
        if durable_child_id:
            raw.setdefault("child_id", durable_child_id)
        if raw.get("role"):
            raw.setdefault("child_id", path.stem)
            values.append(raw)
    return _reviews({"specialist_reviews": values})


def _digest(value: Any) -> str:
    raw = str(value or "")
    return raw if len(raw) <= 18 else f"{raw[:10]}…{raw[-8:]}"


def _usage(record: Mapping[str, Any]) -> dict[str, Any]:
    token_values: dict[str, int | None] = {}
    for key in ("input_tokens", "output_tokens", "cached_input_tokens", "total_tokens"):
        value = record.get(key)
        try:
            token_values[key] = int(value) if value is not None else None
        except (TypeError, ValueError):
            token_values[key] = None
    origin = str(record.get("usage_origin") or "")
    elapsed = record.get("elapsed_ms")
    recorded = record.get("recorded_elapsed_ms")
    current = record.get("current_elapsed_ms")
    if recorded is None and origin not in {"replay", "current"}:
        recorded = elapsed
    if current is None and origin in {"replay", "current"}:
        current = elapsed

    def number(value: Any) -> int | float | None:
        try:
            result = round(float(value), 2)
            return int(result) if result.is_integer() else result
        except (TypeError, ValueError):
            return None

    return {
        "provider": _text(record.get("provider"), 160),
        "model": _text(record.get("model"), 240),
        "namespace": _text(record.get("namespace"), 160),
        "cache_hit": bool(record.get("cache_hit")),
        **token_values,
        "recorded_elapsed_ms": number(recorded),
        "current_elapsed_ms": number(current),
        "has_tokens": any(value is not None for value in token_values.values()),
    }


def _usage_summaries(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (str(record.get("provider") or "unavailable"), str(record.get("model") or "unavailable"))
        row = grouped.setdefault(
            key,
            {
                "provider": key[0],
                "model": key[1],
                "calls": 0,
                "cache_hits": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "has_tokens": False,
                "recorded_elapsed_ms": 0.0,
                "current_elapsed_ms": 0.0,
                "has_recorded_elapsed": False,
                "has_current_elapsed": False,
            },
        )
        row["calls"] += 1
        row["cache_hits"] += int(record.get("cache_hit", False))
        for key_name in ("input_tokens", "output_tokens", "total_tokens"):
            if isinstance(record.get(key_name), int):
                row[key_name] += record[key_name]
                row["has_tokens"] = True
        if record.get("recorded_elapsed_ms") is not None:
            row["recorded_elapsed_ms"] += float(record["recorded_elapsed_ms"])
            row["has_recorded_elapsed"] = True
        if record.get("current_elapsed_ms") is not None:
            row["current_elapsed_ms"] += float(record["current_elapsed_ms"])
            row["has_current_elapsed"] = True
    for row in grouped.values():
        row["recorded_elapsed_ms"] = round(row["recorded_elapsed_ms"], 2)
        row["current_elapsed_ms"] = round(row["current_elapsed_ms"], 2)
    return list(grouped.values())


def _pending(manifest: Mapping[str, Any], chapters: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    approvals = manifest.get("approvals", [])
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(approvals, Sequence) and not isinstance(approvals, (str, bytes)):
        candidates.extend(
            (str(_mapping(item).get("chapter") or ""), _mapping(item))
            for item in approvals
            if isinstance(item, Mapping)
        )
    for chapter in chapters:
        snapshot = _mapping(chapter.get("snapshot"))
        proposal = snapshot.get("pending_proposal")
        if proposal is None and isinstance(snapshot.get("data"), Mapping):
            proposal = snapshot["data"].get("pending_proposal")
        if isinstance(proposal, Mapping):
            candidates.append((str(chapter.get("chapter") or ""), proposal))
    for chapter_id, item in candidates:
        decision = str(item.get("decision") or item.get("status") or "").lower()
        if decision in {"approved", "rejected", "applied", "completed", "complete"}:
            continue
        term = item.get("term") or item.get("source_term")
        target = item.get("target") or item.get("proposed_target") or item.get("selected_translation")
        if not (term or target or decision in {"", "pending", "awaiting_approval"}):
            continue
        return {
            "chapter": chapter_id,
            "term": _text(term, 220),
            "target": _text(target, 220),
            "rationale": _text(item.get("rationale") or item.get("note") or item.get("reason")),
            "proposal_id": _text(item.get("proposal_id"), 180),
            "decision": decision or "pending",
        }
    return None


def _glossary_rows(root: Path, chapters: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    paths = [("run", "inputs/master_glossary.txt"), ("run", "glossary.txt")]
    paths.extend(
        (str(chapter.get("chapter") or ""), f"chapters/{chapter.get('chapter')}/chapter_glossary.txt")
        for chapter in chapters
    )
    rows: list[dict[str, Any]] = []
    previous = ""
    seen: set[str] = set()
    for chapter, relative in paths:
        if relative in seen or _href(root, relative) is None:
            continue
        seen.add(relative)
        content = _read(root, relative, 3000)
        terms: list[dict[str, str]] = []
        for line in content.splitlines()[:30]:
            if "->" in line and not line.lstrip().startswith("#"):
                source, target = line.split("->", 1)
                terms.append({"source": source.strip(), "target": target.strip()})
        if chapter == "run":
            label, reuse = "Run glossary", "Shared glossary used by the chapter workflow."
        elif previous:
            label, reuse = f"Chapter {chapter} start", f"Reused the shared glossary after Chapter {previous}."
        else:
            label, reuse = f"Chapter {chapter} start", "Captured from the master glossary at chapter start."
        rows.append({"label": label, "path": relative, "href": _href(root, relative), "reuse": reuse, "terms": terms, "line_count": len([line for line in content.splitlines() if line.strip()])})
        if chapter != "run":
            previous = chapter
    return rows


def _artifacts(root: Path, manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    values = manifest.get("artifacts", {})
    if not isinstance(values, Mapping):
        return []
    rows: list[dict[str, str]] = []
    for name, value in values.items():
        if isinstance(value, Mapping):
            relative = str(value.get("path") or value.get("relative_path") or "")
            status = str(value.get("status") or "")
        else:
            relative, status = str(value or ""), ""
        rows.append({"name": _text(name, 160), "path": _text(relative, 500), "href": _href(root, relative) or "", "status": _text(status, 80)})
    return rows


def _status_class(status: Any) -> str:
    value = str(status or "unknown").lower()
    if value in {"completed", "verified", "ok", "accepted", "passed", "pass"}:
        return "verified"
    if value in {"running", "pending", "awaiting_approval", "interrupted", "warning", "warn"}:
        return "pending"
    if value in {"failed", "review_required", "rejected", "error", "fail"}:
        return "rejected"
    return "neutral"


def build_showcase_report_context(run_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Build the bounded context used by the standalone showcase template."""

    root = Path(run_dir)
    chapters: list[dict[str, Any]] = []
    chapter_ids = _chapter_ids(manifest)
    for index, chapter_id in enumerate(chapter_ids, 1):
        entry = _entry(manifest, chapter_id)
        base = f"chapters/{chapter_id}"
        source_path, draft_path = f"{base}/source.txt", f"{base}/draft.txt"
        final_path = f"{base}/translated_final.txt"
        snapshot_path, episode_path = f"{base}/session_snapshot.json", f"{base}/agent_episode.json"
        events_path = f"{base}/session_events.jsonl"
        episode, snapshot = _json(root, episode_path), _json(root, snapshot_path)
        raw_steps = episode.get("steps", [])
        steps = [_step(item, sequence) for sequence, item in enumerate(raw_steps, 1)] if isinstance(raw_steps, list) else []
        reviews = _merge_reviews(_reviews(snapshot), _child_reviews(root, chapter_id))
        edits = [edit.get("old_text", "") for review in reviews for edit in review["proposed_edits"] if edit.get("old_text")]
        overlap_rows = [f"{span} proposed by more than one specialist" for span in sorted({value for value in edits if edits.count(value) > 1})]
        overlap_rows.extend(overlap for review in reviews for overlap in review["overlaps"])
        initial_findings = episode.get("initial_qa", {}).get("findings", []) if isinstance(episode.get("initial_qa"), Mapping) else []
        final_findings = episode.get("final_qa", {}).get("findings", []) if isinstance(episode.get("final_qa"), Mapping) else []
        initial_count = _count(entry.get("initial_findings")) or len(initial_findings)
        final_count = _count(entry.get("final_findings")) if "final_findings" in entry else len(final_findings)
        provider_calls = [_usage(value) for value in entry.get("provider_calls", []) if isinstance(value, Mapping)]
        source_text, draft_text, final_text = _read(root, source_path), _read(root, draft_path), _read(root, final_path)
        chapters.append(
            {
                "chapter": chapter_id,
                "index": index,
                "status": str(entry.get("status") or snapshot.get("status") or "pending"),
                "status_class": _status_class(entry.get("status") or snapshot.get("status")),
                "initial_finding_count": initial_count,
                "final_finding_count": final_count,
                "initial_findings": list(initial_findings)[:24] if isinstance(initial_findings, list) else [],
                "final_findings": list(final_findings)[:24] if isinstance(final_findings, list) else [],
                "steps": steps[:40],
                "reviews": reviews,
                "overlaps": overlap_rows[:16],
                "provider_calls": provider_calls,
                "elapsed_ms": entry.get("elapsed_ms"),
                "accepted_patches": _count(entry.get("accepted_patches")),
                "source": {"path": source_path, "href": _href(root, source_path), "text": source_text or _text(entry.get("source_text"))},
                "draft": {"path": draft_path, "href": _href(root, draft_path), "text": draft_text or _text(entry.get("draft_text") or entry.get("translated_text"))},
                "final": {"path": final_path, "href": _href(root, final_path), "text": final_text or _text(entry.get("final_text") or entry.get("translated_final"))},
                "snapshot": {"path": snapshot_path, "href": _href(root, snapshot_path), "data": snapshot},
                "episode": {"path": episode_path, "href": _href(root, episode_path)},
                "events": {"path": events_path, "href": _href(root, events_path)},
                "glossary_path": f"{base}/chapter_glossary.txt",
                "glossary_href": _href(root, f"{base}/chapter_glossary.txt"),
            }
        )

    records = [record for chapter in chapters for record in chapter["provider_calls"]]
    usage_records = [record for record in records if record["has_tokens"]]
    totals = {
        key: sum(record[key] for record in usage_records if isinstance(record.get(key), int))
        for key in ("input_tokens", "output_tokens", "cached_input_tokens", "total_tokens")
    }
    totals = {key: value for key, value in totals.items() if value}
    provider_mode = str(manifest.get("provider_mode") or "unknown")
    execution_mode = str(manifest.get("execution_mode") or provider_mode)
    if execution_mode == "replay":
        provenance = "Replay run; responses came from cache. Recorded provider usage and current replay latency are shown separately."
    elif execution_mode == "offline" or provider_mode == "offline":
        provenance = "Offline fixture; output and control flow are deterministic. Provider usage is not measured."
    elif execution_mode == "live" or provider_mode == "live":
        provenance = "Live provider mode; inspect persisted receipts for provider usage and model evidence."
    else:
        provenance = "Provider provenance is recorded by the run manifest."
    pending = _pending(manifest, chapters)
    complete = sum(str(chapter["status"]).lower() in {"completed", "verified", "ok"} for chapter in chapters)
    current = next((chapter for chapter in chapters if str(chapter["status"]).lower() not in {"completed", "verified", "ok"}), chapters[-1] if chapters else None)
    artifact_rows = _artifacts(root, manifest)
    stages = [
        {"label": "Source", "status": "complete" if any(chapter["source"]["text"] for chapter in chapters) else "pending"},
        {"label": "Translation", "status": "complete" if any(chapter["draft"]["text"] for chapter in chapters) else "pending"},
        {"label": "Specialist review", "status": "complete" if any(chapter["reviews"] for chapter in chapters) else "pending"},
        {"label": "Repair & approval", "status": "pending" if pending else "complete" if not sum(chapter["final_finding_count"] for chapter in chapters) else "running"},
        {"label": "Next chapter", "status": "complete" if complete > 1 else "running" if len(chapters) > 1 else "pending"},
        {"label": "Export", "status": "complete" if artifact_rows else "pending"},
    ]
    profile = _mapping(manifest.get("profile"))
    return {
        "run_id": _text(manifest.get("run_id") or root.name, 180),
        "title": _text(manifest.get("title") or manifest.get("slug") or "Translation showcase", 240),
        "slug": _text(manifest.get("slug"), 180),
        "schema": _text(manifest.get("schema") or "translation-showcase.v1", 120),
        "provider_mode": provider_mode,
        "execution_mode": execution_mode,
        "profile": {key: _text(profile.get(key), 260) for key in ("provider", "model", "tool_protocol")},
        "strategy": _text(manifest.get("strategy") or "specialists", 120),
        "status": str(manifest.get("status") or "running"),
        "status_class": _status_class(manifest.get("status")),
        "provenance": provenance,
        "chapter_ids": chapter_ids,
        "chapters": chapters,
        "chapter_links": [{"chapter": chapter["chapter"], "href": f"#chapter-{html.escape(chapter['chapter'], quote=True)}", "status": chapter["status"], "status_class": chapter["status_class"]} for chapter in chapters],
        "current_chapter": current,
        "completed_count": complete,
        "total_initial": sum(chapter["initial_finding_count"] for chapter in chapters),
        "total_final": sum(chapter["final_finding_count"] for chapter in chapters),
        "total_patches": sum(chapter["accepted_patches"] for chapter in chapters),
        "total_reviews": sum(len(chapter["reviews"]) for chapter in chapters),
        "stages": stages,
        "pending_approval": pending,
        "glossary_rows": _glossary_rows(root, chapters),
        "artifacts": artifact_rows,
        "provider_records": records,
        "usage_summaries": _usage_summaries(records),
        "token_totals": totals,
        "usage_available": bool(usage_records),
        "replay_current_latency_available": any(record.get("current_elapsed_ms") is not None for record in records),
        "approvals": [dict(item) for item in manifest.get("approvals", []) if isinstance(item, Mapping)][:48],
    }


def render_showcase_report(run_dir: Path, manifest: dict[str, Any]) -> Path:
    """Write a local, escaped ``report.html`` for one showcase run."""

    root = Path(run_dir)
    environment = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        autoescape=True,
    )
    rendered = environment.get_template(_TEMPLATE).render(**build_showcase_report_context(root, manifest))
    rendered = rendered.replace("{{", "&#123;&#123;").replace("}}", "&#125;&#125;")
    output = root / "report.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return output


__all__ = ["build_showcase_report_context", "render_showcase_report"]
