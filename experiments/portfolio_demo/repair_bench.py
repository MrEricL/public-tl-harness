"""One small live repair test. Hidden checks never enter the repair prompt."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re

from agentic_translation.autonomous_provider import write_json
from agentic_translation.demo import demo_policy, repair_segments
from agentic_translation.demo_metrics import resources, summarize, write_report
from agentic_translation.demo_provider import DemoGenerator, digest
from agentic_translation.models import GlossaryEntry, GlossaryParseResult
from agentic_translation.segmentation import parse_draft_envelope, segment_source

CASES = Path(__file__).with_name("repair_cases.json")


def checks(case: dict, text: str) -> list[dict]:
    """Narrow anchored fact checks, with both required and forbidden patterns."""
    return [dict(category=case["category"], check_id=str(i),
                 passed=bool(text) and all(re.search(p, text, re.I) for p in rule["require"])
                 and not any(re.search(p, text, re.I) for p in rule.get("forbid", [])),
                 require=rule["require"], forbid=rule.get("forbid", []))
            for i, rule in enumerate(case["checks"])]


def run_benchmark(out: Path, config: dict, *, api_key=None, generator=None, cases_path=CASES) -> dict:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not cases or len({row["id"] for row in cases}) != len(cases):
        raise ValueError("Benchmark requires nonempty unique case IDs")
    for case in cases:
        for rule in case["checks"]:
            for pattern in rule["require"] + rule.get("forbid", []):
                re.compile(pattern)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Use a new or empty benchmark output directory")
    owned = generator is None
    generator = generator or DemoGenerator(out, config, api_key=api_key)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", {"provider": config, "profile": asdict(demo_policy()),
                                     "cases_sha256": digest(cases), "test": "repair-challenge.v1"})
    chapters, stop = [], False
    try:
        for case in cases:
            cid = case["id"]
            print(f"[repair challenge] {cid}: {case['category']}", flush=True)
            chapter = dict(id=cid, name=cid + " / " + case["category"], source=case["source"],
                           status="not_started", texts={"supplied_draft": case["draft"]},
                           arms={"supplied_draft": {"checks": checks(case, case["draft"])},
                                 "harness": {"checks": checks(case, "")}})
            chapters.append(chapter)
            if stop:
                continue
            # Small authored passages are one complete source window.
            segments = segment_source("portfolio", cid, case["source"], target_chars=2000, max_chars=2400)
            if len(segments) != 1:
                raise ValueError("Repair fixture must fit in one window")
            drafts = parse_draft_envelope({"segments": [{"segment_id": segments[0].segment_id,
                                                         "translated_text": case["draft"]}]}, segments)
            glossary = GlossaryParseResult(entries=[GlossaryEntry(**row) for row in case.get("glossary", [])])
            context = {segments[0].segment_id: "\n".join(f"TERM {e.source} -> {e.target}" for e in glossary.entries)}
            start = len(generator.receipts)
            try:
                result = repair_segments(segments, drafts, glossary, generator, out / "sessions" / cid, context=context)
                text = result.final_text or ""
                chapter["texts"]["harness"] = text
                chapter["status"] = result.outcome
                chapter["arms"]["harness"] = dict(checks=checks(case, text),
                                                    resources=resources(generator.receipts[start:]),
                                                    warnings=[asdict(w) for w in result.warnings], events=list(result.events))
                write_json(out / "cases" / f"{cid}.json", result.to_dict())
            except Exception as exc:
                chapter.update(status="failed", error_type=type(exc).__name__)
                stop = bool(getattr(exc, "fatal_provider_error", False))
        summary = summarize(chapters, ["supplied_draft", "harness"], metric="Authored semantic fact-check pass rate")
        summary["note"] = ("Seeded Chinese/English repair challenge, not natural error prevalence. Narrow regex fact checks allow "
                           "some paraphrases but are not a semantic judge: inspect the paired text. The reference and checks are never sent to the model.")
        controls = [(case, row) for case, row in zip(cases, chapters) if case.get("clean_control")]
        summary["clean_controls"] = dict(unchanged=sum(row["texts"].get("harness") == case["draft"] for case, row in controls),
                                         total=len(controls))
        report = dict(mode=getattr(generator, "evidence_mode", "live model"),
                      description="Test 2: a dozen short seeded errors and clean controls; same bounded executor and policy as the folder demo.",
                      status="failed_partial" if any(not c["texts"].get("harness") for c in chapters) else "completed",
                      arms=["supplied_draft", "harness"], chapters=chapters, summary=summary,
                      resources=resources(generator.receipts), delivery=False,
                      cases_sha256=digest(cases), profile=asdict(demo_policy()))
        write_report(out, report)
        return report
    finally:
        if owned:
            generator.close()
