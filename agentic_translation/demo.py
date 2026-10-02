"""Folder → translation → the real bounded harness → inspectable local report."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import sys
import webbrowser

from .adaptive import BudgetPolicy, VerificationPolicy, run_adaptive_chapter
from .autonomous_provider import write_json
from .demo_metrics import resources, summarize, term_checks, write_report
from .demo_provider import DemoGenerator, PROVIDERS, digest, provider_config
from .glossary import parse_glossary_text
from .models import GlossaryEntry, GlossaryParseResult
from .package import build_epub_collection, verify_epub_artifact
from .qa import run_translation_qa
from .segmentation import parse_draft_envelope, render_draft, segment_source
from .story_memory import StoryMemoryTape
from .unattended import (GenerativeReviewAdapter, SourceMemoryExtractor, TermAlignmentAdapter,
                         TERM_ADHERENCE_INSTRUCTION, _context_payload, _glossary,
                         _translate, _with_translator_terms)

PROFILE = "portfolio-review.v1"


def demo_policy() -> VerificationPolicy:
    """Both new live tests execute this exact named policy, with no router SDK."""
    return VerificationPolicy(version=PROFILE, mode="full_segment_review",
                              instruction_version="natural-review-v6",
                              qa_warning_policy_version="qa-warnings.v2",
                              route_deterministic_findings=True, term_alignment_pass=True,
                              gloss_guard=True, duplication_guard=True)


def repair_segments(segments, drafts, glossary, generator, out, *, context=None, memory_hash=""):
    return run_adaptive_chapter(
        segments, drafts, glossary=glossary, decision_provider=None,
        review_provider=GenerativeReviewAdapter(generator),
        alignment_provider=TermAlignmentAdapter(generator), verification_policy=demo_policy(),
        budget_policy=BudgetPolicy(max_coordinator_steps=max(24, len(segments)*8),
                                   max_patch_actions=max(12, len(segments)*2)),
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
               api_key: str | None = None, generator=None, replay_from: Path | None = None) -> dict:
    """No source is overwritten. Partial outputs and every assigned denominator survive failure."""
    sources, excluded = discover_sources(source, out, limit, max_source_chars)
    glossary, categories = load_contract(glossary_path)
    if glossary_path and glossary_path.resolve().parent == source.resolve():
        raise ValueError("Keep the glossary outside the chapter input folder")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Use a new or empty output folder (recorded runs are never overwritten)")
    contract = [{**entry.model_dump(), "category": categories.get(entry.source, "other terms")}
                for entry in glossary.entries]
    arms = ["naive", "glossary", "contextual", "harness"] if compare else ["contextual", "harness"]
    run_config = dict(schema_version="portfolio-run.v1", provider=config, compare=compare,
                      profile=asdict(demo_policy()), limit=limit, max_source_chars=max_source_chars,
                      inputs=[dict(id=cid, name=name, sha256=digest(text)) for cid, name, text in sources],
                      excluded_files=excluded, glossary=contract)
    owned = generator is None
    generator = generator or DemoGenerator(out, config, api_key=api_key, replay=replay_from is not None,
                                           cache_read_dir=None if replay_from is None else replay_from / "cache")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", run_config)
    write_json(out / "glossary.json", contract)
    (out / "inputs").mkdir(exist_ok=True)
    tape = StoryMemoryTape(run_dir=out / "memory", policy_version="source-memory-v2")
    extractor = SourceMemoryExtractor(generator, tape, defer_facts_to_chapter_end=True,
                                      instruction_version="term-quality-v1")
    chapters = []
    mode = "cache replay" if replay_from else getattr(generator, "evidence_mode", "live model")
    report = dict(mode=mode, profile=PROFILE, config_sha256=digest(run_config), arms=arms,
                  description="Same writer, ordered chapters. Glossary prompting → source-only memory → bounded repair. "
                              "Each repair starts from the exact contextual draft shown here.",
                  status="running", chapters=chapters, excluded_files=excluded, delivery=False)
    fatal = False
    try:
        for cid, name, source_text in sources:
            print(f"[{cid}/{len(sources):04d}] {name}", flush=True)
            (out / "inputs" / f"{cid}.txt").write_text(source_text, encoding="utf-8")
            segments = segment_source("portfolio", cid, source_text)
            checks = term_checks(segments, (), glossary, categories)
            chapter = dict(id=cid, name=name, source=source_text, status="not_started", texts={},
                           arms={arm: {"checks": [dict(c) for c in checks], "status": "not_started"} for arm in arms})
            chapters.append(chapter)
            if fatal:
                continue
            chapter_dir = out / "chapters" / cid
            chapter_dir.mkdir(parents=True, exist_ok=True)
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
                snapshot = tape.add_chapter(cid, segments, entries, terms)
                memory_receipts = generator.receipts[start:]
                context = _with_translator_terms(_context_payload(segments, tape), segments, supplied,
                                                drop_overlapping_memory_terms=True)
                chapter["context"] = context
                memory_glossary = _glossary(tape, segments)
                effective = GlossaryParseResult(entries=[
                    *[entry for entry in memory_glossary.entries
                      if not any(entry.source in term or term in entry.source for term in supplied)],
                    *[entry for entry in glossary.entries if entry.source in supplied]])
                initial, context_receipts = translate("contextual", context, memory_receipts)
                start = len(generator.receipts)
                result = repair_segments(segments, initial, effective, generator, out / "sessions" / cid,
                                         context=context, memory_hash=snapshot.snapshot_sha256)
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
    parser.add_argument("--offline", action="store_true", help="Run the existing scripted showcase, not a live quality evaluation")
    parser.add_argument("--replay", type=Path, help="Replay a saved folder demo's successful provider calls into a new output directory")
    parser.add_argument("--limit", type=int, default=3, help="Chapter preview limit, 0 for all (default: 3)")
    parser.add_argument("--max-source-chars", type=int, default=12000)
    parser.add_argument("--max-calls", type=int, default=120)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--call-timeout", type=float, default=300,
                        help="Total seconds allowed per model call, including provider queueing (default: 300)")
    parser.add_argument("--temperature", type=float, default=None, help="Omitted by default for reasoning-model compatibility")
    parser.add_argument("--input-price", type=float, help="USD/million input tokens; optional, not fetched from the internet")
    parser.add_argument("--output-price", type=float, help="USD/million output tokens")
    parser.add_argument("--max-usd", type=float, help="Conservative estimated ceiling; requires both prices")
    parser.add_argument("--open", action="store_true", help="Open the local report in your browser")
    args = parser.parse_args(argv)
    try:
        if sum(bool(v) for v in (args.offline, args.replay, args.repair_bench)) > 1:
            raise ValueError("Choose only one of --offline, --replay, --repair-bench")
        if args.offline:
            if args.source or args.model or args.glossary or args.compare:
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
            if args.source or args.glossary or args.model or args.compare:
                raise ValueError("Replay uses the recorded sources, glossary, model and comparison design")
            saved = json.loads((args.replay / "config.json").read_text(encoding="utf-8"))
            for item in saved["inputs"]:
                text = (args.replay / "inputs" / f'{item["id"]}.txt').read_text(encoding="utf-8")
                if digest(text) != item["sha256"]:
                    raise ValueError("Recorded source changed; replay refused")
            if json.loads((args.replay / "glossary.json").read_text(encoding="utf-8")) != saved["glossary"]:
                raise ValueError("Recorded glossary changed; replay refused")
            if saved["profile"] != asdict(demo_policy()):
                raise ValueError("Recorded policy differs; replay refused")
            report = run_folder(args.replay / "inputs", args.out, saved["provider"],
                                glossary_path=args.replay / "glossary.json", compare=saved["compare"],
                                limit=0, max_source_chars=saved["max_source_chars"], replay_from=args.replay)
        else:
            if args.repair_bench and (args.source or args.glossary or args.compare):
                raise ValueError("--repair-bench uses its own authored inputs, not --source/--glossary/--compare")
            interactive = sys.stdin.isatty()
            if args.source is None and not args.repair_bench:
                if not interactive:
                    raise ValueError("Pass --source FOLDER, or --offline for the credential-free showcase")
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
            config = provider_config(args.provider, args.model, base_url=args.base_url,
                                     temperature=args.temperature, max_calls=args.max_calls,
                                     max_output_tokens=args.max_output_tokens,
                                     input_price=args.input_price, output_price=args.output_price, max_usd=args.max_usd,
                                     call_timeout_seconds=args.call_timeout)
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
                                    compare=args.compare, limit=args.limit, max_source_chars=args.max_source_chars, api_key=api_key)
        print(f"Report: {(args.out / 'report.html').resolve()}")
        if args.open:
            webbrowser.open((args.out / "report.html").resolve().as_uri())
        return 2 if report["status"].startswith("failed") else 0
    except (ValueError, OSError) as exc:
        print(f"Demo input error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
