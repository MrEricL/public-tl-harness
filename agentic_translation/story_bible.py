"""Render a work's source-only story memory tape into a human-readable bible.

Reads the immutable per-chapter ``MemorySnapshot`` JSON files that the
unattended web-novel runner writes under
``<run_dir>/story_memory/<work_id>/<chapter_id>.json`` (each snapshot's
``entries``/``terminology`` lists are cumulative -- the last chapter in
narrative order carries the full history) and produces:

- ``glossary.md``   -- Chinese term -> chosen English, status, first-seen
                       chapter, chapter coverage, and any conflicts/changes.
- ``entities.md``   -- memory entries grouped by kind (characters/entities,
                       organizations, locations, techniques, ranks,
                       artifacts, other), each labeled with its scope
                       (narrator / character_belief / dialogue_claim /
                       editorial), status, and first-known chapter.
- ``changes.md``    -- a chapter-by-chapter log of added/changed/kept terms
                       and added memory entries.
- ``bible.json``    -- the same data, structured.

This module never calls a live model and only reads local run artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


MAX_EVIDENCE_EXCERPT_CHARS = 30

CHARACTER_KINDS = frozenset({"entity", "identity", "alias", "relationship"})
ORGANIZATION_KINDS = frozenset({"organization"})
LOCATION_KINDS = frozenset({"location"})
TECHNIQUE_KINDS = frozenset({"technique"})
RANK_KINDS = frozenset({"rank"})
ARTIFACT_KINDS = frozenset({"artifact"})

ENTITY_GROUPS: tuple[tuple[str, str, frozenset[str]], ...] = (
    ("characters_entities", "Characters / Entities", CHARACTER_KINDS),
    ("organizations", "Organizations", ORGANIZATION_KINDS),
    ("locations", "Locations", LOCATION_KINDS),
    ("techniques", "Techniques", TECHNIQUE_KINDS),
    ("ranks", "Ranks", RANK_KINDS),
    ("artifacts", "Artifacts", ARTIFACT_KINDS),
)
_GROUPED_KINDS = frozenset().union(*(kinds for _, _, kinds in ENTITY_GROUPS))

SCOPE_LABELS = {
    "narrator": "narrator (established fact)",
    "character_belief": "character_belief (a character's private belief -- may be wrong)",
    "dialogue_claim": "dialogue_claim (something a character asserted aloud -- may be wrong)",
    "editorial": "editorial (translation/editorial convention, not a story fact)",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _short_excerpt(text: str) -> str | None:
    return text if isinstance(text, str) and 0 < len(text) <= MAX_EVIDENCE_EXCERPT_CHARS else None


def _load_snapshot_files(run_dir: Path, work_id: str) -> list[dict[str, Any]]:
    directory = run_dir / "story_memory" / work_id
    if not directory.exists():
        raise FileNotFoundError(f"No story memory snapshots found for work_id={work_id!r} under {directory}")
    snapshots = []
    for path in sorted(directory.glob("*.json")):
        payload = _read_json(path)
        if isinstance(payload, dict) and "snapshot_sha256" in payload and "entries" in payload:
            snapshots.append(payload)
    if not snapshots:
        raise FileNotFoundError(f"No chapter snapshot files found under {directory}")
    return snapshots


def _order_chapters(snapshots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Order snapshots narratively by following the preceding_snapshot_sha256 chain."""
    by_sha = {snap["snapshot_sha256"]: snap for snap in snapshots}
    referenced = {snap["preceding_snapshot_sha256"] for snap in snapshots if snap.get("preceding_snapshot_sha256")}
    roots = [snap for snap in snapshots if not snap.get("preceding_snapshot_sha256")]
    if not roots:
        # Fall back to chapter_id ordering if no clean root is found (e.g. a
        # partial/legacy export missing its first chapter's snapshot file).
        return sorted(snapshots, key=lambda snap: str(snap["chapter_id"]))
    ordered: list[dict[str, Any]] = []
    seen_sha: set[str] = set()
    current = min(roots, key=lambda snap: str(snap["chapter_id"]))
    while current is not None and current["snapshot_sha256"] not in seen_sha:
        ordered.append(current)
        seen_sha.add(current["snapshot_sha256"])
        next_snap = next((snap for snap in snapshots
                          if snap.get("preceding_snapshot_sha256") == current["snapshot_sha256"]), None)
        current = next_snap
    # Any snapshot not reached by the chain (shouldn't happen for a clean
    # append-only tape) is appended at the end in chapter_id order so no
    # data is silently dropped.
    missing = [snap for snap in snapshots if snap["snapshot_sha256"] not in seen_sha]
    ordered.extend(sorted(missing, key=lambda snap: str(snap["chapter_id"])))
    return ordered


def _leaf_snapshot(ordered: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The narratively last snapshot; its entries/terminology are cumulative."""
    return ordered[-1]


def _entity_group_for_kind(kind: str) -> tuple[str, str]:
    for key, label, kinds in ENTITY_GROUPS:
        if kind in kinds:
            return key, label
    return "other", "Other"


def _build_bible_data(work_id: str, snapshots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = _order_chapters(snapshots)
    chapter_order = [str(snap["chapter_id"]) for snap in ordered]
    leaf = _leaf_snapshot(ordered)
    entries = list(leaf.get("entries", []))
    terminology = list(leaf.get("terminology", []))

    # --- Glossary -----------------------------------------------------
    terms_by_source: dict[str, list[dict[str, Any]]] = {}
    for decision in terminology:
        terms_by_source.setdefault(decision["source"], []).append(decision)

    glossary: list[dict[str, Any]] = []
    for source in sorted(terms_by_source):
        decisions = terms_by_source[source]
        chapters_seen = list(dict.fromkeys(decision["chapter_id"] for decision in decisions))
        last = decisions[-1]
        changes = [
            {
                "chapter_id": decision["chapter_id"],
                "from_target": None,
                "to_target": decision["target"],
                "reason": decision.get("adjudication_reason"),
                "detail": decision.get("adjudication_detail"),
            }
            for decision in decisions if decision.get("outcome") == "changed"
        ]
        # Fill in from_target using the previously chosen target, when known.
        previous_target = decisions[0]["target"]
        change_index = 0
        for decision in decisions:
            if decision.get("outcome") == "changed":
                changes[change_index]["from_target"] = previous_target
                change_index += 1
            if decision.get("outcome") in {"accepted", "changed"}:
                previous_target = decision["target"]
        glossary.append({
            "source": source,
            "chosen_target": last["target"],
            "status": last.get("status"),
            "first_seen_chapter": decisions[0]["chapter_id"],
            "chapters_seen": len(chapters_seen),
            "chapter_ids": chapters_seen,
            "changes": changes,
            "evidence_excerpts": [
                excerpt for decision in decisions
                for item in decision.get("source_evidence", [])
                for excerpt in [_short_excerpt(item.get("exact_excerpt", ""))] if excerpt
            ][:3],
        })

    # --- Entities -------------------------------------------------------
    entity_groups: dict[str, list[dict[str, Any]]] = {key: [] for key, _, _ in ENTITY_GROUPS}
    entity_groups["other"] = []
    for entry in entries:
        key, _ = _entity_group_for_kind(entry["kind"])
        entity_groups[key].append({
            "entry_id": entry["entry_id"],
            "kind": entry["kind"],
            "subject": entry["subject"],
            "value": entry["value"],
            "scope": entry["scope"],
            "status": entry["status"],
            "first_known_chapter": entry["known_from_chapter"],
            "parent_entry_id": entry.get("parent_entry_id"),
            "evidence_excerpts": [
                excerpt for item in entry.get("source_evidence", [])
                for excerpt in [_short_excerpt(item.get("exact_excerpt", ""))] if excerpt
            ][:3],
        })
    for group in entity_groups.values():
        group.sort(key=lambda row: (row["subject"], row["first_known_chapter"]))

    # --- Chapter-by-chapter change log -----------------------------------
    entries_by_chapter: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        entries_by_chapter.setdefault(entry["known_from_chapter"], []).append(entry)
    terms_by_chapter: dict[str, list[dict[str, Any]]] = {}
    for decision in terminology:
        terms_by_chapter.setdefault(decision["chapter_id"], []).append(decision)

    changes_log = []
    for chapter_id in chapter_order:
        added_entries = sorted(entries_by_chapter.get(chapter_id, []), key=lambda e: e["subject"])
        term_events = sorted(terms_by_chapter.get(chapter_id, []), key=lambda d: d["source"])
        if not added_entries and not term_events:
            continue
        changes_log.append({
            "chapter_id": chapter_id,
            "added_entries": [
                {"kind": e["kind"], "subject": e["subject"], "value": e["value"],
                 "scope": e["scope"], "status": e["status"]}
                for e in added_entries
            ],
            "terminology_events": [
                {"source": d["source"], "target": d["target"], "outcome": d["outcome"],
                 "status": d["status"], "reason": d.get("adjudication_reason"),
                 "detail": d.get("adjudication_detail")}
                for d in term_events
            ],
        })

    return {
        "work_id": work_id,
        "chapter_order": chapter_order,
        "glossary": glossary,
        "entities": entity_groups,
        "changes": changes_log,
    }


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------


def _render_glossary_md(data: Mapping[str, Any]) -> str:
    lines = [f"# Glossary -- {data['work_id']}", ""]
    if not data["glossary"]:
        lines.append("_No terminology decisions recorded._")
        return "\n".join(lines) + "\n"
    for term in data["glossary"]:
        lines.append(f"## {term['source']} -> {term['chosen_target']}")
        status_note = " (provisional)" if term["status"] == "provisional" else ""
        lines.append(f"- Status: {term['status']}{status_note}")
        lines.append(f"- First seen: chapter `{term['first_seen_chapter']}`")
        lines.append(f"- Seen in {term['chapters_seen']} chapter(s): "
                     + ", ".join(f"`{cid}`" for cid in term["chapter_ids"]))
        if term["evidence_excerpts"]:
            lines.append("- Evidence: " + "; ".join(f"“{excerpt}”" for excerpt in term["evidence_excerpts"]))
        if term["changes"]:
            lines.append("- Changes:")
            for change in term["changes"]:
                reason = change.get("reason") or "unspecified"
                detail = f" -- {change['detail']}" if change.get("detail") else ""
                lines.append(f"  - chapter `{change['chapter_id']}`: "
                             f"`{change['from_target']}` -> `{change['to_target']}` ({reason}){detail}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _render_entities_md(data: Mapping[str, Any]) -> str:
    lines = [f"# Entities -- {data['work_id']}", ""]
    groups = data["entities"]
    for key, label, _ in ENTITY_GROUPS:
        rows = groups.get(key, [])
        lines.append(f"## {label}")
        if not rows:
            lines.append("_None recorded._")
            lines.append("")
            continue
        for row in rows:
            scope_label = SCOPE_LABELS.get(row["scope"], row["scope"])
            lines.append(f"### {row['subject']} ({row['kind']})")
            lines.append(f"- Value: {row['value']}")
            lines.append(f"- Scope: {scope_label}")
            lines.append(f"- Status: {row['status']}")
            lines.append(f"- First known: chapter `{row['first_known_chapter']}`")
            if row.get("parent_entry_id"):
                lines.append(f"- Supersedes/derives from: `{row['parent_entry_id']}`")
            if row["evidence_excerpts"]:
                lines.append("- Evidence: " + "; ".join(f"“{excerpt}”" for excerpt in row["evidence_excerpts"]))
            lines.append("")
    other_rows = groups.get("other", [])
    if other_rows:
        lines.append("## Other")
        for row in other_rows:
            scope_label = SCOPE_LABELS.get(row["scope"], row["scope"])
            lines.append(f"### {row['subject']} ({row['kind']})")
            lines.append(f"- Value: {row['value']}")
            lines.append(f"- Scope: {scope_label}")
            lines.append(f"- Status: {row['status']}")
            lines.append(f"- First known: chapter `{row['first_known_chapter']}`")
            lines.append("")
    return "\n".join(lines) + "\n"


def _render_changes_md(data: Mapping[str, Any]) -> str:
    lines = [f"# Changes -- {data['work_id']}", ""]
    if not data["changes"]:
        lines.append("_No chapter activity recorded._")
        return "\n".join(lines) + "\n"
    for chapter in data["changes"]:
        lines.append(f"## Chapter `{chapter['chapter_id']}`")
        if chapter["added_entries"]:
            lines.append("### Added entries")
            for entry in chapter["added_entries"]:
                lines.append(f"- [{entry['kind']}] **{entry['subject']}**: {entry['value']} "
                             f"({entry['scope']}, {entry['status']})")
        if chapter["terminology_events"]:
            lines.append("### Terminology")
            for event in chapter["terminology_events"]:
                verb = {"accepted": "added", "changed": "changed", "kept_existing": "kept"}.get(
                    event["outcome"], event["outcome"])
                reason = f" ({event['reason']})" if event.get("reason") else ""
                lines.append(f"- {verb}: {event['source']} -> {event['target']} [{event['status']}]{reason}")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_bible(run_dir: str | Path, work_id: str, *, out_dir: str | Path | None = None) -> dict[str, Any]:
    run = Path(run_dir)
    snapshots = _load_snapshot_files(run, work_id)
    data = _build_bible_data(work_id, snapshots)
    out = Path(out_dir) if out_dir is not None else run / "bible" / work_id
    out.mkdir(parents=True, exist_ok=True)
    _write_text(out / "glossary.md", _render_glossary_md(data))
    _write_text(out / "entities.md", _render_entities_md(data))
    _write_text(out / "changes.md", _render_changes_md(data))
    _write_text(out / "bible.json", json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return {"work_id": work_id, "out_dir": str(out),
            "chapters": len(data["chapter_order"]), "terms": len(data["glossary"]),
            "entities": sum(len(rows) for rows in data["entities"].values())}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--work", required=True, dest="work_id")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    result = render_bible(args.run, args.work_id, out_dir=args.out)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
