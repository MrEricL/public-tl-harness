"""Source-grounded book translation, with explicit contextual or review policies."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import hashlib

from .adaptive import failed_adaptive_chapter, run_adaptive_chapter
from .autonomous_provider import BudgetedDecisions, BudgetedGenerator, load_credentials, write_json
from .decisions import TypeSafeDecisionProvider
from .glossary import parse_glossary_text
from .models import GlossaryParseResult
from .profiles import (BOOK_PROFILES, CONTEXTUAL_PROFILE, LEGACY_ADAPTIVE_PROFILE,
                       PORTFOLIO_REVIEW_PROFILE, book_verification_policy,
                       portfolio_review_budget, resolve_book_profile)
from .qa import run_translation_qa
from .segmentation import parse_draft_envelope, segment_source, sha256_text
from .story import load_story_config
from .story_memory import StoryMemoryTape
from .unattended import (
    SourceMemoryExtractor, GenerativeReviewAdapter, TermAlignmentAdapter, TERM_ADHERENCE_INSTRUCTION,
    _context_payload, _glossary, _translate, _with_translator_terms,
)


def _contextual_result(source_text: str, memory_hash: str, *, text: str | None = None,
                       drafts=(), warnings=()) -> dict:
    """A translation receipt, without invented model review or coverage evidence."""
    return {
        "outcome": "failed_no_output" if text is None else "delivered_with_warnings" if warnings else "delivered",
        "final_text": text,
        "draft_segments": [asdict(draft) for draft in drafts],
        "source_sha256": sha256_text(source_text),
        "initial_draft_sha256": sha256_text(text) if text is not None else None,
        "memory_snapshot_sha256": memory_hash,
        "verification_policy": None,
        "evidence": [],
        "warnings": list(warnings),
        "events": [{"kind": "generation_or_coverage_failed" if text is None else "contextual_translation",
                    "model_review_performed": False}],
        "patch_actions": 0,
        "review_calls": 0,
        "screen_calls": 0,
    }


def run_unattended_book(story_path: Path, out: Path, *, model: str = "deepseek-flash",
                        decision_model: str = "jev-1.13.0", max_usd: float = 10,
                        strategy: str = "contextual") -> dict:
    profile = resolve_book_profile(strategy)
    verification = book_verification_policy(profile, decision_model=decision_model)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty output directory; book-level resume is not implemented")
    story = load_story_config(story_path)
    if not story.chapter_ids or len(set(story.chapter_ids)) != len(story.chapter_ids):
        raise ValueError("Provide a nonempty ordered list of unique chapter IDs")
    if any(not cid or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in cid)
           for cid in story.chapter_ids):
        raise ValueError("Chapter IDs must be safe filename components")
    sources = [(cid, (story.paths.source_dir / f"{cid}.txt").read_text(encoding="utf-8"))
               for cid in story.chapter_ids]
    # Validate the caller's terminology contract before credentials or spend.
    glossary_bytes = story.paths.glossary_path.read_bytes()
    translator_glossary = parse_glossary_text(glossary_bytes.decode("utf-8"))
    if translator_glossary.warnings:
        raise ValueError("Invalid translator glossary: " + "; ".join(translator_glossary.warnings))
    if len({entry.source for entry in translator_glossary.entries}) != len(translator_glossary.entries):
        raise ValueError("Translator glossary contains duplicate source terms")
    glossary_identity = {"sha256": hashlib.sha256(glossary_bytes).hexdigest(),
                         "entries": len(translator_glossary.entries),
                         "precedence": "translator_over_source_memory"}
    out.mkdir(parents=True, exist_ok=True)
    extraction_instruction = "term-quality-v1" if profile in {CONTEXTUAL_PROFILE, PORTFOLIO_REVIEW_PROFILE} else ""
    config = {
        "writer": {"model": model, "call_timeout_seconds": 300},
        "translation_envelope": "segment-text-v1",
        "memory": {"policy_version": "source-memory-v2", "fact_availability": "chapter_end",
                   "instruction_version": extraction_instruction},
        "budgets": {"ledger": str(out / "budget.json"), "max_experiment_usd": max_usd,
                    "evaluation_reserve_usd": 0},
    }
    config["glossary"] = glossary_identity
    config["profile"] = profile
    config["review_scope"] = BOOK_PROFILES[profile]
    config["verification_policy"] = asdict(verification) if verification is not None else None
    write_json(out/'config.json', config)
    load_credentials()
    tape = StoryMemoryTape(run_dir=out/'memory', policy_version='source-memory-v2')
    generator = BudgetedGenerator(out, config)
    extractor = SourceMemoryExtractor(generator, tape, defer_facts_to_chapter_end=True,
                                      instruction_version=extraction_instruction)
    decisions = (BudgetedDecisions(TypeSafeDecisionProvider(record_dir=out/'decisions'), config, out)
                 if profile == LEGACY_ADAPTIVE_PROFILE else None)
    reviewer = GenerativeReviewAdapter(generator) if verification is not None else None
    aligner = TermAlignmentAdapter(generator, bounded_edits=True) if profile == PORTFOLIO_REVIEW_PROFILE else None
    outputs: list[dict[str, object]] = []

    def manifest(*, status: str, error: BaseException | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "title": story.title,
            "interaction": "unattended",
            "glossary": glossary_identity,
            "profile": profile,
            "review_scope": BOOK_PROFILES[profile],
            "verification_policy": config["verification_policy"],
            "decision_provider": "jev" if decisions is not None else None,
            "chapters": outputs,
            "decision_actor": "automation",
            "status": status,
            "delivery": "delivery/book.txt" if status in {"delivered", "delivered_with_warnings"} else None,
            "budget": generator.ledger.snapshot(),
        }
        if error is not None:
            payload["error"] = {
                "type": type(error).__name__,
                "fatal_provider_error": bool(getattr(error, "fatal_provider_error", False)),
            }
        return payload

    write_json(out / "manifest.json", manifest(status="running"))
    for cid, source in sources:
        snapshot_sha256 = ""
        try:
            segments = segment_source(story.slug, cid, source)
            previous = tape.snapshot_for_chapter(tape.chapter_ids[-1]) if tape.chapter_ids else None
            entries, terms = extractor(cid, segments, previous)
            snapshot = tape.add_chapter(cid, segments, entries, terms)
            snapshot_sha256 = snapshot.snapshot_sha256
            supplied = {entry.source: entry for entry in translator_glossary.entries if entry.source in source}
            terms = {term: entry.target for term, entry in supplied.items()}
            context = _with_translator_terms(_context_payload(segments, tape), segments, terms,
                                             drop_overlapping_memory_terms=True)
            # Preserve explicit aliases and blocked variants, not just targets.
            # Remove overlapping memory choices just as the prompt does.
            memory_glossary = _glossary(tape, segments)
            effective_glossary = GlossaryParseResult(entries=[
                *[entry for entry in memory_glossary.entries
                  if not any(entry.source in term or term in entry.source for term in supplied)],
                *supplied.values(),
            ])
            text, records = _translate(
                generator,
                "contextual",
                segments,
                context,
                16384,
                envelope_mode="segment-text-v1",
                context_instruction=TERM_ADHERENCE_INSTRUCTION,
            )
            drafts = parse_draft_envelope({"segments": records}, segments)
            if verification is None:
                qa = run_translation_qa(run_id=out.name, story_slug=story.slug, chapter=cid,
                                        source_text=source, translated_text=text, glossary=effective_glossary)
                warnings = [{"code": finding.check_id, "segment_id": None, "message": finding.message}
                            for finding in qa.findings if finding.severity != "info"]
                chapter_payload = _contextual_result(source, snapshot_sha256, text=text,
                                                     drafts=drafts, warnings=warnings)
                chapter_payload["deterministic_qa"] = {
                    "summary": qa.summary.model_dump(),
                    "findings": [finding.model_dump() for finding in qa.findings],
                }
            else:
                result = run_adaptive_chapter(
                    segments, drafts, glossary=effective_glossary,
                    decision_provider=decisions, review_provider=reviewer,
                    alignment_provider=aligner, verification_policy=verification,
                    budget_policy=portfolio_review_budget(len(segments))
                    if profile == PORTFOLIO_REVIEW_PROFILE else None,
                    context_by_segment=context, memory_snapshot_sha256=snapshot_sha256,
                    session_dir=out / "sessions" / cid, story_slug=story.slug, run_id=out.name,
                )
                chapter_payload = result.to_dict()
        except Exception as exc:
            # Persist a truthful terminal chapter result before returning.  A
            # provider access/balance error retains its fatal marker and stops
            # scheduling at this chapter; it is never converted into a clean
            # fallback or an invented empty delivery.
            if verification is None:
                chapter_payload = _contextual_result(
                    source, snapshot_sha256,
                    warnings=[{"code": "failed_no_output", "segment_id": None, "message": type(exc).__name__}],
                )
            else:
                chapter_payload = failed_adaptive_chapter(
                    source, type(exc).__name__, verification_policy=verification,
                    memory_snapshot_sha256=snapshot_sha256, session_dir=out / "sessions" / cid,
                ).to_dict()
            chapter_payload["profile"] = profile
            chapter_payload["error"] = {
                "type": type(exc).__name__,
                "fatal_provider_error": bool(getattr(exc, "fatal_provider_error", False)),
            }
            write_json(out / "chapters" / f"{cid}.json", chapter_payload)
            outputs.append(
                {
                    "chapter_id": cid,
                    "outcome": "failed_no_output",
                    "warnings": list(chapter_payload["warnings"]),
                    "error_type": type(exc).__name__,
                    "fatal_provider_error": bool(getattr(exc, "fatal_provider_error", False)),
                }
            )
            terminal = manifest(status="failed_no_output", error=exc)
            write_json(out / "manifest.json", terminal)
            return terminal

        chapter_payload["profile"] = profile
        write_json(out / "chapters" / f"{cid}.json", chapter_payload)
        if chapter_payload["final_text"] is not None:
            (out / "chapters" / f"{cid}.txt").parent.mkdir(parents=True, exist_ok=True)
            (out / "chapters" / f"{cid}.txt").write_text(chapter_payload["final_text"], encoding="utf-8")
        outputs.append(
            {
                "chapter_id": cid,
                "outcome": chapter_payload["outcome"],
                "warnings": list(chapter_payload["warnings"]),
            }
        )
        if chapter_payload["outcome"] == "failed_no_output":
            terminal = manifest(status="failed_no_output")
            write_json(out / "manifest.json", terminal)
            return terminal
        write_json(out / "manifest.json", manifest(status="running"))

    delivery = out / "delivery"
    delivery.mkdir(exist_ok=True)
    (delivery / "book.txt").write_text(
        "\n\n".join((out / "chapters" / f"{cid}.txt").read_text() for cid, _ in sources),
        encoding="utf-8",
    )
    final_status = (
        "delivered_with_warnings"
        if any(row["outcome"] != "delivered" for row in outputs)
        else "delivered"
    )
    terminal = manifest(status=final_status)
    write_json(out / "manifest.json", terminal)
    return terminal
