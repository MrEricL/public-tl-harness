# Mid-Corpus Harness Benchmark Report

Data availability: the real five-chapter corpus and all derived full-text artifacts (source chapters, baseline translations, Harness outputs, blind packets, private mappings, scorecards, action files, and session evidence) are intentionally not distributed. This report retains aggregate metrics and methodology only.

## Bottom line

The bounded repair run cleared 68 deterministic QA rule hits to 0: 56
punctuation findings, 11 glossary findings, and one panel-structure mismatch
where the source and translation had different numbers of bracketed panels.
These are rule hits, not 68 independent semantic defects.

Across 20 chapter-judge decisions, overall pairwise preference was baseline 7, Harness 5, and tie 8. Formatting showed a weak directional formatting preference in this sample, not a general prose or terminology multiplier.

## Pairwise results

| Dimension | Baseline | Harness | Tie |
| --- | ---: | ---: | ---: |
| Overall | 7 | 5 | 8 |
| Formatting | 2 | 4 | 14 |
| Terminology | 5 | 1 | 14 |

Other anchored dimensions are retained in `results.json`; pairwise preferences and family splits are primary, while 1–5 means are secondary descriptive metadata.

## QA and repair totals

| Metric | Value |
| --- | ---: |
| Deterministic findings before → after | 68 → 0 |
| Punctuation / glossary / panel structure | 56 / 11 / 1 |
| Repair steps | 22 |
| Accepted mutations | 7 |
| Rejected mutations | 0 |
| Verified chapters | 5 |

## Provenance

The repair actions originated in Codex runs, were migrated to the Harness v3
schema, and re-executed through this codebase — not produced in a single
end-to-end run. The action JSON, episode traces, and all verbatim text artifacts
are intentionally omitted, so this public package does not claim a verbatim
replay.

## Sanitized comparison examples

See [`PUBLIC_EXAMPLES.md`](PUBLIC_EXAMPLES.md) for two brief, paraphrased examples. They show the shape of the larger divergences without reproducing a source sentence, a translation, or a chapter.

## Efficiency follow-up

The efficiency matrix was **not run**. No provider/token/price telemetry was available, so no cost, latency, token-efficiency, or QA-per-dollar claim is made. The planned three-arm matrix and required telemetry are recorded in [`FUTURE_WORK.md`](FUTURE_WORK.md) and `benchmark_config.json`.

## Limitations

- This is n=5 consecutive chapters from one novel.
- Pairwise preferences and anchored scores come from model judges, not human bilingual reviewers.
- Both candidates share the same one-shot baseline and the Harness receives a bounded repair pass; this is not an independent translation-system comparison.
- No statistical-significance test is justified or reported.
