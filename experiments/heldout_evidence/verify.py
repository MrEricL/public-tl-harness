"""Recompute the public study summary with Python's standard library only.

No model, private corpus, API key, network access, or package installation.
Checks aggregate arithmetic and coverage, not translation/extractor accuracy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import product
import json
import math
import random
import statistics
from pathlib import Path


DEFAULT_EVIDENCE = Path(__file__).with_name("RESULTS.json")
ARMS = ("naive", "contextual", "full_harness", "review_everything")
PAIRS = ("full_harness/naive", "contextual/naive", "full_harness/contextual",
         "full_harness/review_everything")


def _count(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Expected a nonnegative integer count")
    return value


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected a finite number")
    return value


def _require(condition, label):
    if not condition:
        raise ValueError(label)


def _close(left, right, label):
    _require(math.isclose(left, right, abs_tol=1e-10, rel_tol=1e-10), label)


def _quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = int(position)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (position - lo)


def _work_interval(rows):
    """Enumerate the four-work bootstrap exactly, without simulation noise."""
    draws = [math.fsum(row["sum_modal_share_differences"] for row in draw) /
             sum(row["paired_terms"] for row in draw)
             for draw in product(rows, repeat=len(rows))]
    return {"method": "exact ordered work-cluster bootstrap enumeration",
            "work_clusters": len(rows), "resamples": len(draws),
            "interval_95": [_quantile(draws, .025), _quantile(draws, .975)],
            "interpretation": "Descriptive sensitivity with four works; not precise population uncertainty"}


def verify(data):
    if data.get("schema_version") == "heldout-ablation-evidence.v1":
        return verify_ablation(data)
    _require(data.get("schema_version") == "heldout-evidence.v1", "Unknown evidence schema")
    works = data["work_summaries"]
    _require(len(works) == 4 and len({w["work"] for w in works}) == 4,
             "This archived study requires four distinct work summaries")
    _require({w["work"] for w in works} == {"A", "B", "C", "D"}, "Expected anonymous work labels")
    arm_totals = {arm: defaultdict(float) for arm in ARMS}
    comparison_rows = {pair: [] for pair in PAIRS}
    work_results = []
    total_chapters = 0
    for work in works:
        chapters = _count(work["chapters"])
        _require(chapters > 0, "A work has no assigned chapters")
        total_chapters += chapters
        work_scores = {}
        for arm in ARMS:
            row = work["arms"][arm]
            terms, modal, valid, drifted, delivered, warnings = (
                _count(row[key]) for key in ("included_terms", "modal_count", "valid_renderings",
                "drifted_terms", "delivered_chapters", "warning_chapters"))
            share_sum = _number(row["sum_term_modal_shares"])
            cost = _number(row["standalone_cost_usd"])
            _require(0 < terms <= modal <= valid <= terms * chapters and valid >= 2 * terms,
                     "Impossible modal/valid/included counts")
            _require(0 <= drifted <= terms and drifted <= valid - modal,
                     "Impossible drift counts")
            _require(terms - drifted <= share_sum <= terms, "Impossible per-term modal-share sum")
            _require(0 <= warnings <= delivered <= chapters and cost >= 0, "Invalid delivery/cost totals")
            for field in ("included_terms", "modal_count", "valid_renderings", "drifted_terms",
                          "delivered_chapters", "warning_chapters", "standalone_cost_usd"):
                arm_totals[arm][field] += row[field]
            work_scores[arm] = modal / valid
        for pair in PAIRS:
            left, right = pair.split("/")
            row = work["paired_comparisons"][pair]
            paired = _count(row["paired_terms"])
            _require(paired == work["arms"][left]["included_terms"] == work["arms"][right]["included_terms"],
                     "Pairing coverage differs from the archived full shared term inventory")
            _close(_number(row["sum_modal_share_differences"]),
                   work["arms"][left]["sum_term_modal_shares"] - work["arms"][right]["sum_term_modal_shares"],
                   "Paired modal-share sum disagrees with arm sums")
            _require(sum(_count(row[key]) for key in ("terms_improved", "terms_unchanged", "terms_regressed")) == paired,
                     "Term outcomes do not cover all paired terms")
            matrix = row["distinct_rendering_transitions"]
            _require(len(matrix) == 3 and all(len(line) == 3 for line in matrix), "Expected a 3 by 3 drift transition table")
            _require(sum(_count(cell) for line in matrix for cell in line) == paired,
                     "Drift transition table does not cover all paired terms")
            _require(sum(sum(line) for line in matrix[1:]) == work["arms"][right]["drifted_terms"] and
                     sum(sum(line[1:]) for line in matrix) == work["arms"][left]["drifted_terms"],
                     "Drift transition margins disagree with arm counts")
            quality = row["quality"]
            _require(sum(_count(quality[key]) for key in ("wins", "ties", "losses", "unrated")) == chapters,
                     "Quality outcomes do not cover assigned chapters")
            _require(sum(_count(quality[key]) for key in ("two_judges", "one_judge", "identical_outputs", "unrated")) == chapters,
                     "Judge coverage does not cover assigned chapters")
            _require(sum(_count(row["quality_with_two_judges"][key]) for key in ("wins", "ties", "losses")) == quality["two_judges"],
                     "Two-judge sensitivity subset has the wrong denominator")
            comparison_rows[pair].append(row)
        work_results.append({"work": work["work"], "consistency": work_scores,
                             "full_vs_naive_gain": work_scores["full_harness"] - work_scores["naive"]})
    arms = {}
    for arm, counts in arm_totals.items():
        arms[arm] = {key: int(value) if key != "standalone_cost_usd" else value for key, value in counts.items()}
        arms[arm]["consistency"] = counts["modal_count"] / counts["valid_renderings"]
        arms[arm]["drift_rate"] = counts["drifted_terms"] / counts["included_terms"]
        arms[arm]["usd_per_chapter"] = counts["standalone_cost_usd"] / total_chapters
    comparisons = {}
    for pair, rows in comparison_rows.items():
        counts = {key: sum(row[key] for row in rows) for key in
                  ("paired_terms", "terms_improved", "terms_unchanged", "terms_regressed")}
        counts["paired_term_mean_gain"] = math.fsum(row["sum_modal_share_differences"] for row in rows) / counts["paired_terms"]
        counts["work_bootstrap"] = _work_interval(rows)
        counts["baseline_three_plus_renderings"] = sum(sum(row["distinct_rendering_transitions"][2]) for row in rows)
        counts["three_plus_to_one_rendering"] = sum(row["distinct_rendering_transitions"][2][0] for row in rows)
        counts["quality"] = {key: sum(row["quality"][key] for row in rows) for key in rows[0]["quality"]}
        counts["quality_with_two_judges"] = {key: sum(row["quality_with_two_judges"][key] for row in rows) for key in ("wins", "ties", "losses")}
        comparisons[pair] = counts
    naive_miss, full_miss = (1 - arms[arm]["consistency"] for arm in ("naive", "full_harness"))
    return {"assigned_chapters": total_chapters, "work_clusters": len(works), "arms": arms,
            "inconsistency_reduction": 1 - full_miss / naive_miss,
            "inconsistency_ratio": naive_miss / full_miss,
            "work_results": work_results, "comparisons": comparisons,
            "glossary_count_verification": "unavailable; rounded documentary rates are kept separate",
            "evidence_scope": "Aggregate arithmetic and coverage; no new models, human labels, or extraction validation"}


def verify_ablation(data):
    """Recompute the work-equal ablation estimator from opaque term counts."""
    assignments = data["work_assignments"]
    _require(bool(assignments), "No assigned works")
    _require(sum(_count(n) for n in assignments.values()) == _count(data["assigned_chapters"]),
             "Assigned chapter denominator mismatch")
    arms = data["arms"]
    _require(len(set(arms)) == len(arms) and bool(arms), "Invalid arm list")
    for arm in arms:
        row = data["delivery_by_arm"][arm]
        _require(_count(row["assigned"]) == data["assigned_chapters"] and
                 _count(row["complete"]) <= row["assigned"], "Delivery denominator mismatch")
    indexed = {}
    for row in data["consistency"].get("term_counts", []):
        key = (row["work_id"], row["arm"], row["term_id"])
        _require(key not in indexed and key[0] in assignments and key[1] in arms, "Invalid or duplicate term identity")
        valid, modal, distinct = (_count(row[f]) for f in ("valid_renderings", "modal_count", "distinct_renderings"))
        _require(0 <= modal <= valid and distinct <= valid and (valid == 0) == (modal == 0), "Invalid modal counts")
        _require(row["included"] is (valid >= 2), "Historical inclusion rule changed")
        if "source_eligible_opportunities" in row:
            eligible, missing, omitted, invalid = (_count(row[f]) for f in
                ("source_eligible_opportunities", "unavailable", "omitted", "invalid"))
            _require(eligible == valid + missing + omitted + invalid and 2 <= eligible <= assignments[key[0]],
                     "All-eligible denominator mismatch")
            _require(_count(row["chapter_occurrences"]) == valid + omitted + invalid,
                     "Extraction opportunity mismatch")
        indexed[key] = row
    for work in assignments:
        inventory = None
        for arm in arms:
            terms = {t for w, a, t in indexed if w == work and a == arm}
            expected_terms = data["inventory_terms_by_work"][work]
            _require((expected_terms is None and not terms) or
                     (expected_terms is not None and len(terms) == _count(expected_terms)), "Frozen inventory denominator mismatch")
            _require(inventory is None or terms == inventory, "Arm-independent inventory mismatch")
            inventory = terms
    resamples = _count(data["resamples"])
    _require(resamples > 0, "Resamples must be positive")
    result = {}
    for name in data["primary_contrasts"]:
        parts = name.split("/")
        _require(len(parts) == 2 and set(parts) <= set(arms) and parts[0] != parts[1], "Invalid primary contrast")
        left, right = parts
        per_work, means, sensitivity_means = {}, [], []
        assigned_terms = paired = 0
        for work in sorted(assignments):
            lt = {t: r for (w, a, t), r in indexed.items() if w == work and a == left}
            rt = {t: r for (w, a, t), r in indexed.items() if w == work and a == right}
            terms = set(lt) | set(rt)
            assigned_terms += len(terms)
            diffs = [lt[t]["modal_count"] / lt[t]["valid_renderings"] - rt[t]["modal_count"] / rt[t]["valid_renderings"]
                     for t in sorted(terms) if t in lt and t in rt and lt[t]["included"] and rt[t]["included"]]
            paired += len(diffs)
            mean = statistics.mean(diffs) if diffs else None
            per_work[work] = {"assigned_terms": len(terms), "paired_terms": len(diffs), "mean_difference": mean}
            if mean is not None:
                means.append(mean)
            coverage = {}
            for arm, records in ((left, lt), (right, rt)):
                known = bool(records) and all("source_eligible_opportunities" in r for r in records.values())
                eligible = sum(r["source_eligible_opportunities"] for r in records.values()) if known else None
                valid = sum(r["valid_renderings"] for r in records.values())
                modal = sum(r["modal_count"] for r in records.values())
                coverage[arm] = {"source_eligible_opportunities": eligible, "valid_renderings": valid,
                                 "modal_count": modal, "all_eligible_modal_success_rate": modal / eligible if eligible else None,
                                 "unavailable": sum(r["unavailable"] for r in records.values()) if known else None}
            per_work[work]["coverage"] = coverage
            values = [coverage[a]["all_eligible_modal_success_rate"] for a in parts]
            difference = values[0] - values[1] if all(v is not None for v in values) else None
            per_work[work]["all_eligible_success_difference"] = difference
            if difference is not None:
                sensitivity_means.append(difference)
        rng = random.Random(data["seed"])
        draws = [statistics.mean(rng.choices(means, k=len(means))) for _ in range(resamples)] if means else []
        interval = [_quantile(draws, .025), _quantile(draws, .975)] if draws else None
        mean = statistics.mean(means) if means else None
        reported = data["reported_contrasts"][name]
        _require(reported["assigned_term_pairs"] == assigned_terms and reported["paired_terms"] == paired,
                 "Paired term denominator mismatch")
        if mean is None:
            _require(reported["mean_difference"] is None and reported["exploratory_95_resampling_range"] is None,
                     "Missing consistency cannot be zero quality")
        else:
            _close(_number(reported["mean_difference"]), mean, "Work-equal estimate mismatch")
            for observed, expected in zip(reported["exploratory_95_resampling_range"], interval):
                _close(_number(observed), expected, "Work-equal resampling range mismatch")
        result[name] = {"assigned_term_pairs": assigned_terms, "paired_terms": paired,
                        "assigned_work_clusters": len(assignments), "work_clusters_with_pairs": len(means),
                        "mean_difference": mean, "exploratory_95_resampling_range": interval,
                        "all_eligible_success_difference": statistics.mean(sensitivity_means) if sensitivity_means else None,
                        "per_work": per_work}
    return {"schema_version": "verified-ablation.v1", "assigned_chapters": data["assigned_chapters"],
            "work_clusters": len(assignments), "comparisons": result,
            "evidence_scope": "Counts arithmetic only; does not establish human extraction accuracy or quality."}


def report(result):
    if result.get("schema_version") == "verified-ablation.v1":
        print(f"Ablation arithmetic verified: {result['assigned_chapters']} assigned chapters / {result['work_clusters']} works.")
        for name, row in result["comparisons"].items():
            print(f"  {name}: {row['paired_terms']}/{row['assigned_term_pairs']} paired terms; work-equal difference {row['mean_difference']}; range {row['exploratory_95_resampling_range']}")
        print(result["evidence_scope"])
        return
    print(f"Arithmetic verified: {result['assigned_chapters']} chapters / {result['work_clusters']} works.")
    print("\nModal / valid extracted term-chapter renderings:")
    for arm, row in result["arms"].items():
        print(f"  {arm:20} {row['modal_count']:3}/{row['valid_renderings']:3} = {row['consistency']:.1%}; ${row['usd_per_chapter']:.4f}/chapter")
    print(f"\nFull vs naive: {result['inconsistency_reduction']:.1%} fewer inconsistent renderings ({result['inconsistency_ratio']:.2f}x ratio).")
    primary = result["comparisons"]["full_harness/naive"]
    print(f"Severe drift resolved: {primary['three_plus_to_one_rendering']}/{primary['baseline_three_plus_renderings']} terms with 3+ baseline renderings went to one.")
    lo, hi = primary["work_bootstrap"]["interval_95"]
    print(f"Paired per-term gain: {100 * primary['paired_term_mean_gain']:+.1f} pp; four-work resampling range {100 * lo:+.1f} to {100 * hi:+.1f} pp.")
    print("This range describes four sampled works; it is not precise population uncertainty.")
    quality = primary["quality"]
    print(f"Quality: {quality['wins']}/{quality['ties']}/{quality['losses']} wins/ties/losses; {quality['two_judges']} pairs with both judges, {quality['one_judge']} with one.")
    print("Glossary percentages: documentary only; exact glossary numerator/denominator absent from the archived export.")
    print("Checks aggregate arithmetic, not extraction accuracy or overall translation quality.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--json", action="store_true", help="Print the recomputed summary as JSON")
    args = parser.parse_args()
    try:
        result = verify(json.loads(args.evidence.read_text(encoding="utf-8")))
    except (ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Evidence check failed: {exc}\n")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        report(result)


if __name__ == "__main__":
    main()
