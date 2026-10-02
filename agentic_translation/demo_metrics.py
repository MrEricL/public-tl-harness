"""Count inspectable terminology contracts, not a made-up overall quality score."""
from __future__ import annotations

from collections import defaultdict
import difflib
import html
import json
from pathlib import Path

from .models import GlossaryParseResult
from .qa import _glossary_source_spans
from .text import literal_term_pattern
from .autonomous_provider import write_json


def term_checks(sources, drafts, glossary: GlossaryParseResult, categories: dict[str, str]) -> list[dict]:
    by_id = {item.segment_id: item.translated_text for item in drafts}
    rows = []
    for source in sources:
        _, independent = _glossary_source_spans(source.text, glossary)
        for entry in glossary.entries:
            if not independent.get(entry.source):
                continue
            text = by_id.get(source.segment_id, "")
            present = bool(literal_term_pattern(entry.target).search(text))
            blocked = any(literal_term_pattern(alias).search(text) for alias in entry.blocked_variants)
            rows.append(dict(segment=source.segment_id, source=entry.source, target=entry.target,
                             category=categories.get(entry.source, "other terms"),
                             passed=present and not blocked, target_present=present,
                             blocked_variant_present=blocked, output_present=bool(text)))
    return rows


def resources(receipts: list[dict]) -> dict:
    def total(key):
        values = [row.get(key) for row in receipts]
        return sum(values) if all(value is not None for value in values) else None
    return dict(calls=len(receipts), physical_calls=sum(row.get("physical_calls", 0) for row in receipts),
                input_tokens=total("input_tokens"), output_tokens=total("output_tokens"),
                estimated_usd=total("estimated_usd"), latency_seconds=total("latency_seconds"))


def summarize(chapters: list[dict], arms: list[str], *, metric: str) -> dict:
    groups = defaultdict(lambda: {arm: {"passed": 0, "total": 0} for arm in arms})
    transitions = []
    for chapter in chapters:
        for arm in arms:
            for check in chapter["arms"][arm]["checks"]:
                for category in {"all", check["category"]}:
                    groups[category][arm]["total"] += 1
                    groups[category][arm]["passed"] += int(check["passed"])
        for before, after in zip(arms, arms[1:]):
            old, new = chapter["arms"][before]["checks"], chapter["arms"][after]["checks"]
            if len(old) != len(new):
                raise ValueError("Paired checks must have identical denominators")
            transitions.extend(dict(chapter=chapter["id"], before=before, after=after,
                                    category=a["category"], fixed=not a["passed"] and b["passed"],
                                    regressed=a["passed"] and not b["passed"])
                               for a, b in zip(old, new))
    for group in groups.values():
        for value in group.values():
            value["rate"] = value["passed"] / value["total"] if value["total"] else None
    gains = []
    for before, after in zip(arms, arms[1:]):
        for category in groups:
            rows = [row for row in transitions if row["before"] == before and row["after"] == after
                    and (category == "all" or row["category"] == category)]
            previous, current = groups[category][before], groups[category][after]
            a, b = previous["rate"], current["rate"]
            gains.append(dict(before=before, after=after, category=category,
                              fixed=sum(row["fixed"] for row in rows),
                              regressed=sum(row["regressed"] for row in rows),
                              percentage_points=None if a is None else 100 * (b-a),
                              relative_gain_percent=None if not a else 100 * (b/a-1)))
    return dict(metric=metric, groups=dict(groups), gains=gains,
                note="A terminology check requires the specified target and no blocked variant in its source window. "
                     "This is not a translation-quality or semantic-completeness score. Missing outputs stay in the denominator.")


def write_report(out: Path, report: dict) -> None:
    """Static, escaped report: no server, CDN, credential, or third-party script."""
    write_json(out / "results.json", report)
    arms = report["arms"]
    summary = report["summary"]
    def esc(value):
        return html.escape(str(value))
    def score(value):
        return f'{value["passed"]}/{value["total"]} ({100*value["rate"]:.1f}%)' if value["rate"] is not None else "Not measured"
    header = "<tr><th>Area</th>" + "".join(f"<th>{esc(arm)}</th>" for arm in arms) + "</tr>"
    table = "".join("<tr><th>" + esc(cat) + "</th>" + "".join(f"<td>{score(row[arm])}</td>" for arm in arms) + "</tr>"
                    for cat, row in summary["groups"].items())
    gains = "".join(f'<tr><td>{esc(row["before"])} → {esc(row["after"])}</td><td>{esc(row["category"])}</td>'
                    f'<td>{row["fixed"]}</td><td>{row["regressed"]}</td><td>{row["percentage_points"]:+.1f} pp</td></tr>'
                    for row in summary["gains"] if row["percentage_points"] is not None)
    details = []
    for chapter in report["chapters"]:
        texts = chapter["texts"]
        details.append(f'<details><summary>{esc(chapter["name"])} · {esc(chapter["status"])}</summary>'
                       f'<h3>Chinese source</h3><pre>{esc(chapter["source"])}</pre>')
        for arm in arms:
            data = chapter["arms"][arm]
            details.append(f'<h3>{esc(arm)}</h3><pre>{esc(texts.get(arm) or "No output")}</pre>'
                           f'<p>{esc(json.dumps(data.get("resources", {})))}</p>')
        if texts.get("harness"):
            before = texts.get("contextual", texts.get(arms[0], ""))
            diff = "\n".join(difflib.unified_diff(before.splitlines(), texts["harness"].splitlines(),
                                                  fromfile="before repair", tofile="after repair", lineterm=""))
            details.append(f'<h3>Exact changes</h3><pre>{esc(diff or "No text changes")}</pre>')
        details.append(f'<h3>Checks, warnings, and repair events</h3><pre>{esc(json.dumps(chapter["arms"], ensure_ascii=False, indent=2))}</pre>')
        if chapter.get("context"):
            details.append(f'<h3>Context actually sent</h3><pre>{esc(json.dumps(chapter["context"], ensure_ascii=False, indent=2))}</pre>')
        details.append("</details>")
    all_scores = summary["groups"].get("all", {})
    cards = "".join(f'<div class="card"><span>{esc(arm)}</span><strong>{score(all_scores[arm])}</strong></div>'
                    for arm in arms if arm in all_scores)
    headline = "No supplied terminology checks; inspect the translation and repair evidence below."
    # Missing text counts as failed checks below; a delta over it is not a measurement.
    missing = sum(any(not chapter["texts"].get(arm) for arm in arms) for chapter in report["chapters"])
    if missing:
        headline = (f"No score: {missing} of {len(report['chapters'])} item(s) lack output from at least one arm "
                    f"(status: {report['status']}). The tables below count missing text as failed checks.")
    elif all_scores:
        before, after = all_scores[arms[0]], all_scores[arms[-1]]
        a, b = before["rate"], after["rate"]
        if a is not None and b is not None:
            delta = 100 * (b-a)
            relative = f" ({100*(b/a-1):+.1f}% relative)" if a else " (relative change undefined from a zero baseline)"
            headline = f"{delta:+.1f} percentage points{relative}: {arms[0]} → {arms[-1]} on {summary['metric']}."
    controls = summary.get("clean_controls")
    control_text = (f"Clean controls left unchanged: {controls['unchanged']}/{controls['total']}. "
                    "Check their fact scores too; an unchanged clean passage is not a missed repair."
                    if controls and not missing else "")
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Translation workbench · run evidence</title><style>
body{{font:16px/1.55 system-ui;background:#111927;color:#e7edf4;margin:0}}main{{max-width:1120px;margin:auto;padding:32px 20px}}
h1{{font-size:clamp(26px,5vw,42px);line-height:1.1}}h2{{margin-top:36px}}.headline{{font-size:21px;border-left:3px solid #8ee0cd;padding-left:16px}}a{{color:#8ee0cd}}.tag{{color:#8ee0cd;text-transform:uppercase;letter-spacing:.12em}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}.card,details{{background:#1c283b;padding:18px;border-radius:10px;margin:12px 0}}
.card span,.card strong{{display:block}}.card strong{{font-size:25px;margin-top:8px}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}
th,td{{text-align:left;padding:10px;border-bottom:1px solid #35445c}}.scroll{{overflow-x:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace;background:#101827;padding:16px;border-radius:6px}}
summary{{cursor:pointer;font-weight:650}}.muted{{color:#b4c1d2}}p{{overflow-wrap:anywhere}}</style><main>
<p class="tag">Translation workbench / {esc(report['mode'])}</p><h1>What changed. What stayed intact.</h1>
<p>{esc(report['description'])}</p><p class="muted">Status: {esc(report['status'])} · Metric: {esc(summary['metric'])}</p>
<p class="headline">{esc(headline)}</p><div class="cards">{cards}</div><p>{esc(control_text)}</p><p>{esc(summary['note'])}</p>
<p><a href="results.json">Full JSON evidence</a> · <a href="report.md">Markdown summary</a>{' · <a href="delivery/book.txt">Read the translation</a> · <a href="delivery/book.epub">EPUB</a>' if report.get('delivery') else ''}</p>
<h2>Where the gains are</h2><div class="scroll"><table>{header}{table}</table></div>
<h2>Incremental contribution</h2><p>Positive and negative changes are both retained. Percentage points are not relative percentages.</p>
<div class="scroll"><table><tr><th>Intervention</th><th>Area</th><th>Fixed</th><th>Regressed</th><th>Net change</th></tr>{gains}</table></div>
<h2>API work</h2><pre>{esc(json.dumps(report.get('resources', {}), indent=2))}</pre>
<p class="muted">Per-arm resources include their dependencies as standalone estimates; their sum is not the physical bill. Dollar estimates require your supplied rate card and ignore cache discounts. Unknown usage is not zero.</p>
<h2>Inspect the translation and harness</h2>{''.join(details)}<p class="muted">Local report contains your text. Do not publish private novel runs.</p></main></html>'''
    (out / "report.html").write_text(text, encoding="utf-8")
    lines = ["# Translation workbench", "", report["description"], "", f"Mode: {report['mode']}. Status: {report['status']}.", "",
             summary["metric"], "", headline, "", control_text, "", "| Area | " + " | ".join(arms) + " |", "| --- | " + " | ".join("---" for _ in arms) + " |"]
    for cat, row in summary["groups"].items():
        lines.append("| " + cat + " | " + " | ".join(score(row[arm]) for arm in arms) + " |")
    lines += ["", summary["note"], "", "## Incremental effects", ""]
    for row in summary["gains"]:
        if row["percentage_points"] is not None:
            lines.append(f"- {row['before']} → {row['after']}, {row['category']}: {row['percentage_points']:+.1f} pp; "
                         f"{row['fixed']} fixed, {row['regressed']} regressed.")
    (out / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
