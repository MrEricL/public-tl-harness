"""Folder → translation → the real bounded harness → inspectable local report."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import getpass
from importlib.util import find_spec
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import webbrowser

from .adaptive import run_adaptive_chapter
from .autonomous_provider import write_json
from .demo_metrics import resources, summarize, term_checks, write_report
from .demo_provider import DemoGenerator, PROVIDERS, digest, provider_config
from .glossary import parse_glossary_text
from .models import GlossaryEntry, GlossaryParseResult
from .package import build_epub_collection, verify_epub_artifact
from .profiles import (PORTFOLIO_REVIEW_PROFILE as PROFILE,
                       portfolio_review_budget, portfolio_review_policy as demo_policy)
from .qa import run_translation_qa
from .segmentation import parse_draft_envelope, render_draft, segment_source
from .story_memory import StoryMemoryTape, saved_chapter_ids
from .unattended import (GenerativeReviewAdapter, SourceMemoryExtractor, TermAlignmentAdapter,
                         TERM_ADHERENCE_INSTRUCTION, TERM_MEMORY_V2, _context_payload, _glossary,
                         _translate, _with_translator_terms)

# Story-memory settings for new runs. Runs recorded before these were
# written into config.json replay with the settings they were recorded under.
# Names-only pack (fact_policy terms_only): the registered follow-up found it no worse on
# unseen chapters at 14% of the size. A run without a recorded fact_policy used the legacy pack.
# Term alignment enforces only the translator's names: patching memory's names after the draft was
# judged worse in four development pilots. Older runs aligned to both.
DEMO_MEMORY = {"instruction_version": TERM_MEMORY_V2, "memory_overlap": "drop_conflicting", "fact_policy": "terms_only",
               "alignment_names": "translator_only"}
LEGACY_DEMO_MEMORY = {"instruction_version": "term-quality-v1", "memory_overlap": "drop_any"}


def memory_settings(replay_from: Path | None = None) -> dict:
    """New runs use DEMO_MEMORY; a replay uses what its recorded run used (legacy if unrecorded)."""
    if replay_from is None:
        return dict(DEMO_MEMORY)
    saved = json.loads((replay_from / "config.json").read_text(encoding="utf-8"))
    return dict(saved.get("memory") or LEGACY_DEMO_MEMORY)


def repair_segments(segments, drafts, glossary, generator, out, *, context=None, memory_hash="",
                    alignment_glossary=None):
    return run_adaptive_chapter(
        segments, drafts, glossary=glossary, decision_provider=None,
        review_provider=GenerativeReviewAdapter(generator),
        alignment_provider=TermAlignmentAdapter(generator, bounded_edits=True), alignment_glossary=alignment_glossary,
        verification_policy=demo_policy(),
        budget_policy=portfolio_review_budget(len(segments)),
        context_by_segment=context, memory_snapshot_sha256=memory_hash or None,
        session_dir=out, story_slug="portfolio", run_id="portfolio",
    )


def load_contract(path: Path | None) -> tuple[GlossaryParseResult, dict[str, str]]:
    if path is None:
        return GlossaryParseResult(entries=[]), {}
    raw = path.read_text(encoding="utf-8-sig")
    categories = {}
    if path.suffix.lower() == ".json":
        rows = json.loads(raw)
        if not isinstance(rows, list):
            raise ValueError("JSON glossary must be an array of source/target/category entries")
        entries = []
        for row in rows:
            if not isinstance(row, dict) or set(row)-{"source", "target", "category", "candidates", "blocked_variants"}:
                raise ValueError("Unexpected JSON glossary fields")
            entry = GlossaryEntry.model_validate({k: v for k, v in row.items() if k != "category"})
            if not isinstance(row.get("category", "other terms"), str) or not row.get("category", "other terms").strip():
                raise ValueError("Term categories must be nonempty strings")
            entries.append(entry)
            categories[entry.source] = row.get("category", "other terms")
        glossary = GlossaryParseResult(entries=entries)
    else:
        glossary = parse_glossary_text(raw)
    if glossary.warnings or any(not e.source.strip() or not e.target.strip() for e in glossary.entries):
        raise ValueError("Glossary contains incomplete or invalid entries")
    if len({e.source for e in glossary.entries}) != len(glossary.entries):
        raise ValueError("Glossary contains duplicate source terms")
    return glossary, categories


def natural_key(path: Path):
    return tuple((0, int(part)) if part.isdigit() else (1, part.casefold())
                 for part in re.split(r"(\d+)", path.name)), path.name


def build_context_receipts(segments, tape, supplied, context, *, max_context_chars=3000, fact_policy="legacy"):
    """Explain the existing pack selection without another model invocation."""
    receipts = []
    for segment in segments:
        pack = tape.context_for_segment(segment.chapter_id, segment.segment_id,
                                        max_context_chars=max_context_chars, fact_policy=fact_policy)
        full = tape.context_for_segment(segment.chapter_id, segment.segment_id,
                                        max_context_chars=10**9, fact_policy=fact_policy)
        snapshot = tape.snapshot_for_chapter(segment.chapter_id)
        entries = {entry.entry_id: entry for entry in snapshot.entries}
        text = context[segment.segment_id]
        supplied_lines = set(text.splitlines())

        def fact(entry_id):
            entry = entries[entry_id]
            matches = [alias for alias in (entry.subject, *entry.source_aliases)
                       if alias and alias in segment.text]
            return {"entry_id": entry_id, "kind": entry.kind, "subject": entry.subject,
                    "value": entry.value, "scope": entry.scope, "status": entry.status,
                    "source_evidence": [asdict(evidence) for evidence in entry.source_evidence],
                    "retrieval_reason": "source_match" if matches else "recent_visible_fact",
                    "matched_strings": matches,
                    "known_from_chapter": entry.known_from_chapter,
                    "known_from_segment": entry.known_from_segment,
                    "earlier_source": entry.known_from_chapter != segment.chapter_id}

        terms = tape.glossary_for_segment(segment.chapter_id, segment.segment_id)
        selected_terms = []
        for source in full.terminology_sources:
            decisions = [item for item in snapshot.terminology if item.source == source
                         and item.target == terms[source] and item.outcome in {"accepted", "changed"}
                         and tape._known_position(item.chapter_id, item.segment_id)
                         <= tape._known_position(segment.chapter_id, segment.segment_id)]
            delivered = f"TERM {source} -> {terms[source]}" in supplied_lines
            replaced = (not delivered and source in pack.terminology_sources
                        and any(source in term or term in source for term in supplied))
            selected_terms.append({"source": source, "target": terms[source],
                                   "source_evidence": [asdict(e) for e in decisions[-1].source_evidence],
                                   "retrieval_reason": "source_match", "matched_strings": [source],
                                   "supplied": delivered,
                                   "omission_reason": "translator_glossary_precedence" if replaced else
                                   "size_limit" if not delivered else None})
        translator_terms = {term: target for term, target in supplied.items()
                            if term in segment.text}
        sent_translator_terms = {
            term: target for term, target in translator_terms.items()
            if f"TERM {term} -> {target} [translator glossary]" in supplied_lines
        }
        receipts.append({"segment_id": segment.segment_id,
                         "snapshot_sha256": pack.snapshot_sha256,
                         "selected_facts": [fact(entry_id) for entry_id in pack.entry_ids],
                         "omitted_facts": [fact(entry_id) for entry_id in full.entry_ids
                                           if entry_id not in pack.entry_ids],
                         "memory_terms": selected_terms,
                         "translator_terms": sent_translator_terms,
                         "omitted_translator_terms": {
                             term: target for term, target in translator_terms.items()
                             if term not in sent_translator_terms},
                         "memory_pack_truncated": pack.truncated,
                         "memory_pack_characters": len(pack.text),
                         "supplied_context_characters": len(text),
                         "context_tokens_estimate": len(text.encode("utf-8")) / 2.5,
                         "token_estimate_basis": "UTF-8 bytes / 2.5; estimate, not billed usage"})
    return receipts


def discover_sources(source: Path, out: Path, limit: int, max_source_chars: int) -> tuple[list, list]:
    source, out = source.resolve(), out.resolve()
    if not source.is_dir():
        raise ValueError("Source must be a folder containing UTF-8 .txt chapter files")
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Keep the output folder separate from the input folder")
    if limit < 0 or max_source_chars < 1:
        raise ValueError("limit must be nonnegative; max_source_chars must be positive")
    paths = sorted((p for p in source.iterdir() if p.suffix.lower() == ".txt"), key=natural_key)
    selected = paths[:limit] if limit else paths
    if not selected:
        raise ValueError("No .txt chapters found (one file per chapter; subfolders are not scanned)")
    result = []
    for i, path in enumerate(selected, 1):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Chapter paths must be regular files, not symlinks or folders")
        text = path.read_text(encoding="utf-8-sig")
        if not text.strip() or len(text) > max_source_chars:
            raise ValueError(f"{path.name}: empty or exceeds --max-source-chars; split long chapters first")
        result.append((f"{i:04d}", path.name, text))
    return result, [p.name for p in paths[len(selected):]]


def run_folder(source: Path, out: Path, config: dict, *, glossary_path: Path | None = None,
               compare: bool = False, limit: int = 3, max_source_chars: int = 12000,
               api_key: str | None = None, generator=None, replay_from: Path | None = None,
               lead_in: int = 0, memory_from: Path | None = None) -> dict:
    """No source is overwritten. Partial outputs and every assigned denominator survive failure.

    lead_in: the first N chapter files are read into story memory only, then translation
    starts. memory_from: continue story memory from an earlier run of this demo (its saved,
    verified snapshots), numbering new chapters after the carried ones. A replay rebuilds
    both from the recorded run's own folder.
    """
    if lead_in < 0:
        raise ValueError("lead_in must be nonnegative")
    saved = json.loads((replay_from / "config.json").read_text(encoding="utf-8")) if replay_from else None
    if saved is not None and (lead_in or memory_from is not None):
        raise ValueError("A replay uses the recorded lead-in and carried memory")
    discovered, excluded = discover_sources(source, out, (lead_in + limit) if (lead_in and limit) else limit,
                                            max_source_chars)
    if saved is not None:
        recorded = saved["inputs"]
        if len(recorded) != len(discovered):
            raise ValueError("Recorded inputs differ; replay refused")
        sources = [(item["id"], item["name"], text) for item, (_, _, text) in zip(recorded, discovered)]
        memory_dir = replay_from / "memory_sources"
        lead = [(item["id"], item["name"], (memory_dir / f"{item['id']}.txt").read_text(encoding="utf-8"))
                for item in saved.get("lead_in", [])]
        carried_record = saved.get("carried_memory")
        carried_ids = [item["id"] for item in carried_record["chapters"]] if carried_record else []
        carried_texts = {cid: (memory_dir / f"{cid}.txt").read_text(encoding="utf-8") for cid in carried_ids}
        carried_from, snapshots_from = (carried_record or {}).get("source_run"), replay_from
    else:
        lead_raw, translate_raw = discovered[:lead_in], discovered[lead_in:]
        if not translate_raw:
            raise ValueError("No chapters left to translate after the lead-in chapters")
        carried_ids, carried_texts, carried_from, snapshots_from = [], {}, None, memory_from
        if memory_from is not None:
            prior = json.loads((memory_from / "config.json").read_text(encoding="utf-8"))
            if prior.get("schema_version") != "portfolio-run.v1":
                raise ValueError("--memory-from needs an earlier run folder of this demo")
            carried_ids = saved_chapter_ids(memory_from / "memory")
            for cid in carried_ids:
                path = next((p for p in (memory_from / "inputs" / f"{cid}.txt", memory_from / "memory_sources" / f"{cid}.txt")
                             if p.is_file()), None)
                if path is None:
                    raise ValueError(f"The earlier run has no saved source text for chapter {cid}")
                carried_texts[cid] = path.read_text(encoding="utf-8")
            carried_from = str(memory_from.resolve())
        offset = max((int(cid) for cid in carried_ids if cid.isdigit()), default=0)
        lead = [(f"{offset + i:04d}", name, text) for i, (_, name, text) in enumerate(lead_raw, 1)]
        sources = [(f"{offset + len(lead) + i:04d}", name, text) for i, (_, name, text) in enumerate(translate_raw, 1)]
    glossary, categories = load_contract(glossary_path)
    if glossary_path and glossary_path.resolve().parent == source.resolve():
        raise ValueError("Keep the glossary outside the chapter input folder")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Use a new or empty output folder (recorded runs are never overwritten)")
    contract = [{**entry.model_dump(), "category": categories.get(entry.source, "other terms")}
                for entry in glossary.entries]
    arms = ["naive", "glossary", "contextual", "harness"] if compare else ["contextual", "harness"]
    memory = memory_settings(replay_from)
    run_config = dict(schema_version="portfolio-run.v1", provider=config, compare=compare, memory=memory,
                      profile=asdict(demo_policy()), limit=limit, max_source_chars=max_source_chars,
                      inputs=[dict(id=cid, name=name, sha256=digest(text)) for cid, name, text in sources],
                      excluded_files=excluded, glossary=contract)
    if lead:
        run_config["lead_in"] = [dict(id=cid, name=name, sha256=digest(text)) for cid, name, text in lead]
    if carried_ids:
        run_config["carried_memory"] = dict(source_run=carried_from,
                                            chapters=[dict(id=cid, sha256=digest(carried_texts[cid])) for cid in carried_ids])
    owned = generator is None
    generator = generator or DemoGenerator(out, config, api_key=api_key, replay=replay_from is not None,
                                           cache_read_dir=None if replay_from is None else replay_from / "cache")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", run_config)
    write_json(out / "glossary.json", contract)
    (out / "inputs").mkdir(exist_ok=True)
    if lead or carried_ids:
        (out / "memory_sources").mkdir(exist_ok=True)
        for cid, text in [*carried_texts.items(), *((cid, text) for cid, _, text in lead)]:
            (out / "memory_sources" / f"{cid}.txt").write_text(text, encoding="utf-8")
    if carried_ids:
        snapshots = out / "memory" / "story_memory" / "snapshots"
        snapshots.mkdir(parents=True, exist_ok=True)
        for cid in carried_ids:
            shutil.copy2(snapshots_from / "memory" / "story_memory" / "snapshots" / f"{cid}.json", snapshots / f"{cid}.json")
        tape = StoryMemoryTape.load(out / "memory", {cid: segment_source("portfolio", cid, carried_texts[cid])
                                                     for cid in carried_ids},
                                    policy_version="source-memory-v2", chapter_ids=carried_ids)
    else:
        tape = StoryMemoryTape(run_dir=out / "memory", policy_version="source-memory-v2")
    extractor = SourceMemoryExtractor(generator, tape, defer_facts_to_chapter_end=True,
                                      instruction_version=memory["instruction_version"])
    chapters = []
    mode = "cache replay" if replay_from else getattr(generator, "evidence_mode", "live model")
    report = dict(mode=mode, profile=PROFILE, config_sha256=digest(run_config), arms=arms,
                  description="Same writer, ordered chapters. Glossary prompting → source-only memory → bounded repair. "
                              "Each repair starts from the exact contextual draft shown here.",
                  status="running", chapters=chapters, excluded_files=excluded, delivery=False)
    fatal = False
    if carried_ids:
        report["carried_memory"] = dict(source_run=carried_from, chapters=len(carried_ids))
    try:
        if lead:
            report["lead_in"] = []
        for cid, name, source_text in lead:
            print(f"[lead-in {cid}] {name}", flush=True)
            record = dict(id=cid, name=name, status="not_started")
            report["lead_in"].append(record)
            if fatal:
                continue
            try:
                segments = segment_source("portfolio", cid, source_text)
                previous = tape.snapshot_for_chapter(tape.chapter_ids[-1]) if tape.chapter_ids else None
                entries, terms = extractor(cid, segments, previous)
                snapshot = tape.add_chapter(cid, segments, entries, terms)
                record.update(status="read",
                              names=sum(d.chapter_id == cid and d.outcome in {"accepted", "changed"} for d in snapshot.terminology),
                              facts=sum(e.known_from_chapter == cid for e in snapshot.entries))
            except Exception as exc:
                record.update(status="failed", error_type=type(exc).__name__)
                fatal = bool(getattr(exc, "fatal_provider_error", False))
                print(f"  {type(exc).__name__}: lead-in chapter not read", flush=True)
        for position, (cid, name, source_text) in enumerate(sources, 1):
            print(f"[{position}/{len(sources)}] {name} (chapter {cid})", flush=True)
            (out / "inputs" / f"{cid}.txt").write_text(source_text, encoding="utf-8")
            segments = segment_source("portfolio", cid, source_text)
            checks = term_checks(segments, (), glossary, categories)
            chapter = dict(id=cid, name=name, source=source_text, status="not_started", texts={},
                           arms={arm: {"checks": [dict(c) for c in checks], "status": "not_started"} for arm in arms})
            chapters.append(chapter)
            chapter_dir = out / "chapters" / cid
            chapter_dir.mkdir(parents=True, exist_ok=True)
            if fatal:
                # Not attempted after a fatal provider error. It stays "not_started" (the
                # workbench shows "the run stopped before this chapter"), and its
                # result.json is written like every other assigned chapter's.
                write_json(chapter_dir / "result.json", chapter)
                continue
            supplied = {e.source: e.target for e in glossary.entries if e.source in source_text}
            plain_context = _with_translator_terms({}, segments, supplied)
            def save_arm(arm, drafts, receipts, **extra):
                text = render_draft(drafts)
                chapter["texts"][arm] = text
                qa = run_translation_qa(run_id="portfolio", story_slug="portfolio", chapter=cid,
                                        source_text=source_text, translated_text=text, glossary=glossary)
                chapter["arms"][arm] = dict(checks=term_checks(segments, drafts, glossary, categories),
                                             resources=resources(receipts), status="delivered",
                                             deterministic_findings=qa.summary.model_dump(), **extra)
                (chapter_dir / f"{arm}.txt").write_text(text, encoding="utf-8")
                write_json(chapter_dir / f"{arm}.segments.json", [asdict(d) for d in drafts])
            def translate(operation, context, dependencies=()):
                start = len(generator.receipts)
                _, raw = _translate(generator, operation, segments, context, config["max_output_tokens"],
                                    envelope_mode="segment-text-v1", context_instruction=TERM_ADHERENCE_INSTRUCTION)
                drafts = parse_draft_envelope({"segments": raw}, segments)
                receipts = [*dependencies, *generator.receipts[start:]]
                save_arm(operation, drafts, receipts)
                return drafts, receipts
            try:
                if compare:
                    translate("naive", None)
                    translate("glossary", plain_context)
                start = len(generator.receipts)
                previous = tape.snapshot_for_chapter(tape.chapter_ids[-1]) if tape.chapter_ids else None
                entries, terms = extractor(cid, segments, previous)
                chapter["warnings"] = [item for item in extractor.warnings if item["chapter_id"] == cid]
                snapshot = tape.add_chapter(cid, segments, entries, terms)
                memory_receipts = generator.receipts[start:]
                fact_policy = memory.get("fact_policy", "legacy")
                context = _with_translator_terms(_context_payload(segments, tape, fact_policy=fact_policy), segments, supplied,
                                                drop_overlapping_memory_terms=True,
                                                memory_overlap=memory["memory_overlap"])
                chapter["context"] = context
                chapter["context_receipts"] = build_context_receipts(segments, tape, supplied, context,
                                                                     fact_policy=fact_policy)
                memory_glossary = _glossary(tape, segments)
                effective = GlossaryParseResult(entries=[
                    *[entry for entry in memory_glossary.entries
                      if not any(entry.source in term or term in entry.source for term in supplied)],
                    *[entry for entry in glossary.entries if entry.source in supplied]])
                initial, context_receipts = translate("contextual", context, memory_receipts)
                start = len(generator.receipts)
                translator_only = memory.get("alignment_names") == "translator_only"
                result = repair_segments(segments, initial, effective, generator, out / "sessions" / cid,
                                         context=context, memory_hash=snapshot.snapshot_sha256,
                                         alignment_glossary=GlossaryParseResult(entries=[
                                             entry for entry in glossary.entries if entry.source in supplied])
                                         if translator_only else None)
                write_json(chapter_dir / "repair.json", result.to_dict())
                if result.final_text is not None:
                    save_arm("harness", result.draft_segments,
                             [*context_receipts, *generator.receipts[start:]],
                             warnings=[asdict(w) for w in result.warnings], events=list(result.events))
                    chapter["arms"]["harness"]["status"] = result.outcome
                chapter["status"] = result.outcome
            except Exception as exc:
                chapter["status"] = "failed"
                chapter["error_type"] = type(exc).__name__
                # Arms this chapter never reached failed with it; none stays "not_started".
                for arm_state in chapter["arms"].values():
                    if arm_state["status"] == "not_started":
                        arm_state["status"] = "failed"
                # Transport/auth/budget errors stop spending, while the manifest
                # still includes every assigned chapter in final coverage.
                fatal = bool(getattr(exc, "fatal_provider_error", False))
                print(f"  {type(exc).__name__}: saved partial evidence", flush=True)
            write_json(chapter_dir / "result.json", chapter)
        delivered = [c for c in chapters if c["texts"].get("harness")]
        report["status"] = ("failed_partial" if len(delivered) != len(chapters) else
                            "delivered_with_warnings" if any(c["status"] != "delivered" for c in chapters) else "delivered")
        if len(delivered) == len(chapters):
            delivery = out / "delivery"
            delivery.mkdir(exist_ok=True)
            texts = {c["id"]: c["texts"]["harness"] for c in chapters}
            (delivery / "book.txt").write_text("\n\n".join(texts.values()), encoding="utf-8")
            try:
                build_epub_collection(output_path=delivery / "book.epub", story_title="Translation workbench", chapters=texts)
                report["epub_check"] = verify_epub_artifact(delivery / "book.epub")
                report["delivery"] = True
            except Exception as exc:
                report["status"] = "failed_packaging"
                report["packaging_error_type"] = type(exc).__name__
        report["summary"] = summarize(chapters, arms, metric="Specified term-window adherence")
        report["resources"] = resources(generator.receipts)
        report["memory_rejections"] = extractor.rejections
        write_report(out, report)
        return report
    finally:
        if owned:
            generator.close()


def load_replay(replay: Path) -> dict:
    """The recorded configuration, refused if its sources, glossary or policy changed since recording."""
    saved = json.loads((replay / "config.json").read_text(encoding="utf-8"))
    for item in saved["inputs"]:
        text = (replay / "inputs" / f'{item["id"]}.txt').read_text(encoding="utf-8")
        if digest(text) != item["sha256"]:
            raise ValueError("Recorded source changed; replay refused")
    if json.loads((replay / "glossary.json").read_text(encoding="utf-8")) != saved["glossary"]:
        raise ValueError("Recorded glossary changed; replay refused")
    if saved["profile"] != asdict(demo_policy()):
        raise ValueError("Recorded policy differs; replay refused")
    return saved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Drop in a chapter folder; inspect translation, repairs and gains.")
    parser.add_argument("--source", type=Path, help="Folder of UTF-8 .txt chapters, naturally sorted")
    parser.add_argument("--out", type=Path, default=Path("runs/try-it"))
    parser.add_argument("--glossary", type=Path, help="Optional arrow-format TXT or categorized JSON glossary")
    parser.add_argument("--provider", choices=PROVIDERS, default="openai")
    parser.add_argument("--model", help="An actual chat-completion model ID available to your key")
    parser.add_argument("--base-url", help="Optional OpenAI-compatible endpoint; custom URLs use TL_API_KEY")
    parser.add_argument("--compare", action="store_true", help="Test 1: naive → glossary → memory → repair")
    parser.add_argument("--repair-bench", action="store_true", help="Test 2: small seeded semantic-repair challenge")
    # The scripted showcase isn't part of every installation; one found outside this package must not enable it.
    showcase = find_spec("agentic_translation.showcase")
    offline_available = bool(showcase and showcase.origin
                             and Path(showcase.origin).resolve().is_relative_to(Path(__file__).resolve().parent))
    parser.set_defaults(offline=False)
    if offline_available:
        parser.add_argument("--offline", action="store_true", help="Run the existing scripted showcase, not a live quality evaluation")
    parser.add_argument("--replay", type=Path, help="Replay a saved folder demo's successful provider calls into a new output directory")
    parser.add_argument("--limit", type=int, default=3, help="Chapter preview limit, 0 for all (default: 3)")
    parser.add_argument("--lead-in", type=int, default=0,
                        help="Read the first N chapter files into story memory only, then translate the rest")
    parser.add_argument("--memory-from", type=Path,
                        help="Continue story memory from an earlier run folder of this demo")
    parser.add_argument("--max-source-chars", type=int, default=12000)
    parser.add_argument("--max-calls", type=int, default=120)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--call-timeout", type=float, default=300,
                        help="Total seconds allowed per model call, including provider queueing (default: 300)")
    parser.add_argument("--temperature", type=float, default=None, help="Omitted by default for reasoning-model compatibility")
    parser.add_argument("--thinking-mode", choices=("enabled", "disabled"),
                        help="DeepSeek thinking mode (default: disabled for the bounded demo)")
    parser.add_argument("--input-price", type=float, help="USD/million input tokens; optional, not fetched from the internet")
    parser.add_argument("--output-price", type=float, help="USD/million output tokens")
    parser.add_argument("--max-usd", type=float, help="Conservative estimated ceiling; requires both prices")
    parser.add_argument("--open", action="store_true", help="Open the local report in your browser")
    args = parser.parse_args(argv)
    try:
        if sum(bool(v) for v in (args.offline, args.replay, args.repair_bench)) > 1:
            choices = "--offline, --replay, --repair-bench" if offline_available else "--replay, --repair-bench"
            raise ValueError(f"Choose only one of {choices}")
        if args.offline:
            if args.source or args.model or args.glossary or args.compare or args.thinking_mode:
                raise ValueError("--offline runs the bundled fixture only; remove live input options")
            import subprocess
            story = Path(__file__).resolve().parents[1] / "samples/showcase/story.yaml"
            if args.out.exists() and any(args.out.iterdir()):
                raise ValueError("Use a new or empty output directory")
            subprocess.run([sys.executable, "-m", "agentic_translation", "harness", "run", "--story", str(story),
                            "--out", str(args.out), "--auto-approve"], check=True)
            print("Scripted offline showcase only; no live translation-quality measurement.")
            report = {"status": "delivered"}
        elif args.replay:
            if args.source or args.glossary or args.model or args.compare or args.thinking_mode or args.lead_in or args.memory_from:
                raise ValueError("Replay uses the recorded sources, glossary, model, comparison design, lead-in and carried memory")
            saved = load_replay(args.replay)
            report = run_folder(args.replay / "inputs", args.out, saved["provider"],
                                glossary_path=args.replay / "glossary.json", compare=saved["compare"],
                                limit=0, max_source_chars=saved["max_source_chars"], replay_from=args.replay)
        else:
            if args.repair_bench and (args.source or args.glossary or args.compare):
                raise ValueError("--repair-bench uses its own authored inputs, not --source/--glossary/--compare")
            interactive = sys.stdin.isatty()
            if args.source is None and not args.repair_bench:
                if not interactive:
                    hint = ", or --offline for the credential-free showcase" if offline_available else ""
                    raise ValueError(f"Pass --source FOLDER{hint}")
                raw = input("Drag a chapter folder here (Enter for samples/showcase/source): ").strip()
                if raw:
                    parts = shlex.split(raw)
                    raw = parts[0] if len(parts) == 1 else raw
                args.source = Path(raw or "samples/showcase/source").expanduser()
                args.provider = input("Provider [openai / anthropic / deepseek / openrouter / custom]: ").strip() or "openai"
                default_glossary = ("samples/showcase/terms/demo_glossary.json" if args.source.resolve() == Path("samples/showcase/source").resolve() else "")
                path = input(f"Glossary path (Enter for {default_glossary or 'none'}): ").strip().strip("\"'") or default_glossary
                if path:
                    args.glossary = Path(path).expanduser()
                args.compare = input("Compare plain → glossary → memory → repair? [Y/n]: ").strip().lower() != "n"
            if not args.model:
                if not interactive:
                    raise ValueError("Pass --model with the exact model ID available to your key")
                args.model = input("Model ID: ").strip()
            if args.provider == "custom" and not args.base_url and interactive:
                args.base_url = input("OpenAI-compatible base URL (including /v1 when required): ").strip()
            if args.provider == "deepseek":
                args.thinking_mode = args.thinking_mode or "disabled"
                print(f"DeepSeek thinking mode: {args.thinking_mode}")
            config = provider_config(args.provider, args.model, base_url=args.base_url,
                                     temperature=args.temperature, max_calls=args.max_calls,
                                     max_output_tokens=args.max_output_tokens,
                                     input_price=args.input_price, output_price=args.output_price, max_usd=args.max_usd,
                                     call_timeout_seconds=args.call_timeout, thinking_mode=args.thinking_mode)
            # All local input validation happens before asking for secrets or creating output.
            if not args.repair_bench:
                selected, excluded = discover_sources(args.source, args.out, args.limit, args.max_source_chars)
                load_contract(args.glossary)
                print(f"Selected {len(selected)} chapter(s); {len(excluded)} excluded by --limit. Output: {args.out}")
            if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
                raise ValueError("Use a new or empty output folder")
            default_url, key_env = PROVIDERS[args.provider]
            if config["base_url"] != default_url.rstrip("/"):
                key_env = "TL_API_KEY"
            api_key = None
            if not os.getenv(key_env) and interactive:
                api_key = getpass.getpass(f"API key for {config['base_url']} (hidden; never saved): ")
            if args.repair_bench:
                from experiments.portfolio_demo.repair_bench import run_benchmark
                report = run_benchmark(args.out, config, api_key=api_key)
            else:
                report = run_folder(args.source, args.out, config, glossary_path=args.glossary,
                                    compare=args.compare, limit=args.limit, max_source_chars=args.max_source_chars, api_key=api_key,
                                    lead_in=args.lead_in, memory_from=args.memory_from)
        print(f"Report: {(args.out / 'report.html').resolve()}")
        if args.open:
            webbrowser.open((args.out / "report.html").resolve().as_uri())
        return 2 if report["status"].startswith("failed") else 0
    except (ValueError, OSError, TypeError) as exc:
        print(f"Demo input error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
