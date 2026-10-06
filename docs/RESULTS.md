# Results

The held-out run reduced cross-chapter terminology inconsistency by 62.5%
(2.7×), with improvement in all four works. This page separates that measured
product benefit from the smaller repair effect and the mixed quality judgments.

The [public evidence bundle](../experiments/heldout_evidence/README.md)
recomputes consistency, drift transitions, costs, and judge coverage from
aggregate counts, without API calls:

```bash
python experiments/heldout_evidence/verify.py
```

The source chapters and translations are private. The bundle contains no
chapter text, glossary entries, per-term records, or identifying hashes;
see the [data notice](../DATA_NOTICE.md). Recomputing the arithmetic verifies
the saved evidence, while extraction accuracy still requires source inspection.

## Held-out web-novel study

Design: 40 chapters, ten consecutive chapters from each of four Chinese web
novels that were never used during development, each block after five
source-only warm-up chapters. Every arm uses the same writer model (DeepSeek
Flash, temperature 0, thinking disabled). The arms, consistency outcome,
permissive glossary-overlap outcome, and analysis were committed before any
held-out chapter was translated; the set was run once. Exact glossary matching
was already computed, but a later audit promoted it above the pre-specified
permissive-overlap outcome.

| Arm | What it receives |
| --- | --- |
| Naive | The chapter only |
| Context | Source-only story memory plus the translator's series glossary |
| Full harness | The context draft, batched glossary alignment, then Jev-routed review and bounded repair behind the guards |
| Review everything | The same as the full harness, but every passage is reviewed |

### Terminology consistency

The inventory selects up to 40 recurring terms per work by source frequency,
from the union of memory candidates and the supplied glossary. Selection does
not use which arm wins. This targets recurring names and terminology rather
than a random sample of all translation phenomena.

For each of the resulting 160 source terms, an arm-blind extraction pass records
the term's English rendering in each chapter where it appears. The archived
report lists 857 attempted term–chapter observations per arm; the count export
retains 851–854 valid renderings. The difference is 3–6 missing or invalid
extractions per arm, excluded from the score. Extraction uses DeepSeek Flash
and was not validated against human labels. Consistency is the share of valid
observations that use that term's most common rendering in the arm.

| Measure | Naive | Context | Full harness | Review everything |
| --- | ---: | ---: | ---: | ---: |
| Modal / valid extracted renderings | 561 / 851 | 734 / 854 | 744 / 853 | 744 / 854 |
| Consistency | 65.9% | 85.9% | 87.2% | 87.1% |
| Reported attempts (857) minus valid extractions | 6 | 3 | 4 | 3 |
| Terms rendered more than one way, out of 160 | 116 (72.5%) | 61 (38.1%) | 62 (38.8%) | 63 (39.4%) |

Inconsistency falls from 290/851 to 109/853 valid observations: a 62.5%
reduction, or 2.7× fewer. This is a reduction in the inconsistency rate, with
the arm-specific denominators shown above.

### The result in each held-out work

Each work contributes ten chapters and 40 recurring terms. The anonymous
labels below refer to separate works, not independent chapter samples.

| Work | Naive | Context | Full harness | Full minus naive |
| --- | ---: | ---: | ---: | ---: |
| A | 144 / 214 (67.3%) | 178 / 213 (83.6%) | 175 / 212 (82.5%) | +15.3 pp |
| B | 139 / 220 (63.2%) | 200 / 224 (89.3%) | 203 / 224 (90.6%) | +27.4 pp |
| C | 156 / 228 (68.4%) | 199 / 228 (87.3%) | 202 / 228 (88.6%) | +20.2 pp |
| D | 122 / 189 (64.6%) | 157 / 189 (83.1%) | 164 / 189 (86.8%) | +22.2 pp |

The paired per-term gain is +21.0 percentage points; the observation-weighted
gain is +21.3 points. The original 10,000-resample work-cluster bootstrap
interval is 16.7–25.1 points. Enumerating all 256 possible work-cluster
resamples gives the same rounded interval. Four works provide a useful
within-study sensitivity check, but are too few to treat that interval as
strong population-level inference.

Source memory and the supplied glossary produce about 94% of the
observation-weighted gain. Review and repair add +1.1 points when each term is
weighted equally (bootstrap interval −0.8 to +2.3). The table also preserves
the small repair regression in Work A. The historical study combines memory
and glossary in one arm; the current four-arm demo separates them.

Translated one chapter at a time, the character name 伊文 came out six ways
across ten chapters (Irwin, Evan, Ivan, Ewen, Evin, Erwin); the full harness
wrote "Evan" every time. Across the full sample, of 72 recurring terms that
naive translation rendered three or more ways, the harness rendered 39 exactly
one way. The aggregate drift-transition counts reproduce that 39/72 result.

Consistency is self-consistency: a consistently wrong rendering scores well.
The evaluator observes one rendering per term per chapter, so the metric also
does not establish that every mention within a chapter is consistent.

### Glossary adherence: stricter audit metric

The saved study reports the following rates for the 143 selected terms with a
series-glossary reference:

| Measure | Naive | Context | Full harness | Review everything |
| --- | ---: | ---: | ---: | ---: |
| Exact glossary match after normalization | 37.0% | 84.4% | 86.0% | 85.8% |
| Permissive overlap: either string contains the other | 45.9% | 93.9% | 95.5% | 95.4% |

The original plan specified permissive overlap. Exact matching was also
computed; a later audit selected it as the stricter headline. Its reported
63.0% → 14.0% miss rate is a 4.5× reduction. The committed count export does not
retain exact/overlap match numerators, so these rates remain documentary
results and are not certified by the public verifier. They are secondary to
the count-backed consistency headline. Adherence measures compliance with a
supplied vocabulary, not an
independent judgment that the vocabulary is correct.

### Judged quality

The protocol requested two blind judge families (Claude Sonnet 5 and GPT-6
Luna) in both presentation orders for the comparisons with naive translation.
Valid, complete votes from both families were available for 34/40 full-versus-
naive pairs and 33/40 context-versus-naive pairs. The remaining six and seven
pairs respectively use one complete family vote. Judges never saw arm names,
costs, or the glossary; all 40 assigned chapters remain in the reported counts.

| Comparison | Wins | Ties | Losses |
| --- | ---: | ---: | ---: |
| Context vs naive | 14 | 8 | 18 |
| Full harness vs naive | 14 | 9 | 17 |
| Full harness vs context (GPT-6 Luna only) | 16 | 16 | 8 |
| Full harness vs review everything (GPT-6 Luna only) | 16 | 11 | 13 |

Among the 34 full-versus-naive pairs with complete votes from both families,
the split is 14 wins, 6 ties, and 14 losses. For full versus context, 36 pairs
received the single-family comparison and four identical-output pairs tied
automatically; full versus review-everything has 37 judged pairs and three
identical-output ties.

Overall quality superiority was not established. Against naive translation,
the excess-loss rate (3 of 40, 7.5%) met the pre-registered 10% point rule, but
the work-cluster interval for net preference (−42.5% to +20.0%) is too wide to
establish no material quality loss. Judges agreed with their own verdict across
the two presentation orders 143 of 220 times. Because judges never see the
glossary, deliberately chosen series terms can count against the arms that
follow them.

### Cost, delivery, and routing

Estimates price token usage at DeepSeek's 2026-09-22 off-peak list rates (the
run executed off-peak; peak rates are double) plus Jev list rates, excluding
judging.

| Arm | Estimated cost per chapter | Median estimated standalone seconds per chapter |
| --- | ---: | ---: |
| Naive | $0.0021 | 15.3 |
| Context | $0.0066 | 30.6 |
| Review everything | $0.0152 | 51.4 |
| Full harness | $0.0168 | 146.5 |

Durations estimate sequential per-chapter work, including the allocated memory
work shared by the arms. They are not the wall-clock throughput of a parallel
book run.

At these chapter lengths, a 1,000-chapter novel costs about $17 at the full
harness's measured rates (roughly $29 at peak-hour rates).

Every arm delivered 40 of 40 chapters. The full harness delivered 31 of them
with recorded warnings (mostly Jev flags that nothing cleared) and
review-everything 37 (mostly unavailable review). A warning marks unresolved
checking, not a confirmed error.

Jev routing did not transfer: thresholds calibrated on development data routed
62% of held-out passages to review on the first screen (38% during
development), so the routed arm cost slightly more than reviewing every passage.

The private development repository retains the per-term counts. The public
bundle exports only anonymous work-level sufficient statistics and transition
counts, which reproduce the headline without exposing those records. The
offline demos remain separate, scripted checks of software behavior.

## Harness lab: same tools, different harnesses

One chapter-repair tool set (list, read, search, glossary lookup, exact edit,
guarded rewrite, done) was exposed through the CLI and the dependency-free MCP
server in this repository. Each condition started from the same
glossary-prompted draft of six development chapters. A model family other than
the one that made the edits judged each condition against the starting draft,
in both orders.

| Harness | Model | Tool calls | Edits accepted / attempted | Chapters edited | Judged vs starting draft (W / T / L) |
| --- | --- | ---: | ---: | ---: | --- |
| This project's loop | DeepSeek Flash | 74 | 0 / 0 | 0 / 6 | identical (no edits) |
| Claude Code | Claude Sonnet 5 | 76 | 9 / 10 | 5 / 6 | 5 / 1 / 0 |
| Codex | GPT-6 Luna | 105 | 16 / 20 | 6 / 6 | 1 / 3 / 2 |

The same guarded tools produced no edits, a few careful edits, or many edits
depending on the harness. Six chapters is small, each condition also differs
by model and judge, and the Claude judge's two presentation orders agreed on
only two of six pairs, so this demonstrates behavior, not a ranking.

## Earlier repair results

- **A strict verifier refused correct repairs.** In 252 short repair trials the
  first harness delivered 11 of 36 faithful outputs where direct prompting
  delivered 36 of 36. Its verifier accepted an edit only if a rule score
  improved, so it held all twelve semantic-error trials instead of repairing
  them.
- **The fix.** Accepting meaning repairs that leave QA unchanged, then
  requiring a fresh source review of the final text, corrected 10 of 10
  targeted meaning errors in a follow-up check on authored cases. The
  [automatic repair contract](AUTOMATIC_REPAIR.md) describes the resulting
  policy.
- **Five-chapter paired benchmark.** Bounded repair cleared all 68
  deterministic QA rule hits (56 punctuation, 11 glossary, 1 panel); blinded
  model preference was 7 for the baseline,
  5 for the harness, and 8 ties. See the
  [benchmark report](../experiments/mid_corpus_harness_benchmark/REPORT.md).

## What is not claimed

- No general improvement in literary quality.
- No ranking of harnesses or models from the six-chapter lab.
- No cost saving from Jev routing on the held-out novels.
- The offline demos and controlled-response tests check software behavior;
  they are not model-quality evidence.
