# Results

Every number here comes from saved runs. The web-novel studies used chapters
from a private corpus that is not distributed (see the
[data notice](../DATA_NOTICE.md)), so this page reports aggregate counts only,
plus one character-name example.

## Held-out web-novel study

**Design.** 40 chapters: ten consecutive chapters from each of four Chinese web
novels that were never used during development, each block after five
source-only warm-up chapters. Every arm uses the same writer model (DeepSeek
Flash, temperature 0, thinking disabled). The arms, primary outcomes, and
analysis were written down before any held-out chapter was translated, and the
held-out set was run once.

| Arm | What it receives |
| --- | --- |
| Naive | The chapter only |
| Context | Source-only story memory plus the translator's series glossary |
| Full harness | The context draft, batched glossary alignment, then Jev-routed review and bounded repair behind the guards |
| Review everything | The same as the full harness, but every passage is reviewed |

### Terminology consistency

For each of 160 recurring source terms, an arm-blind extraction pass records
the term's English rendering in each chapter where it appears (857 term–chapter
observations per arm). Extraction uses DeepSeek Flash and was not validated
against human labels. Consistency is the share of observations that use the
term's most common rendering in that arm.

| Measure | Naive | Context | Full harness | Review everything |
| --- | ---: | ---: | ---: | ---: |
| Term–chapter observations with the term's usual rendering | 65.9% | 85.9% | **87.2%** | 87.1% |
| Terms rendered more than one way | 72.5% | 38.1% | 38.8% | 39.4% |
| Series-glossary renderings matched exactly (143 terms) | 37.0% | 84.4% | **86.0%** | 85.8% |
| Permissive overlap with the glossary term (either string contains the other) | 45.9% | 93.9% | 95.5% | 95.4% |

- **Inconsistent renderings fell 2.7×** (34.1% → 12.8% of observations), and
  **glossary misses fell 4.5×** (63.0% → 14.0%).
- **The gain is statistically significant.** The paired per-term gain of the
  full harness over naive translation is +21.0 points, with a 95% work-cluster
  bootstrap interval of 16.7 to 25.1 (10,000 resamples). The
  observation-weighted table rises 21.3 points.
- **Drift example.** Translated one chapter at a time, the character name 伊文
  came out six ways across ten chapters (Irwin, Evan, Ivan, Ewen, Evin, Erwin);
  the full harness wrote "Evan" every time. Of the 72 recurring terms that
  naive translation rendered three or more ways, the full harness rendered 39
  exactly one way.
- **Most of the gain comes from context.** Story memory and the glossary
  produce +19.9 points. Review and repair add +1.1 points (interval −0.8 to
  +2.3, so possibly none) and 1.6 points of exact glossary use.

Consistency is self-consistency: a consistently wrong rendering would still
score well. Glossary adherence measures compliance with an input the harness
receives, not independent correctness.

### Judged quality

Two blind judge families (Claude Sonnet 5 and GPT-6 Luna) compared each pair of
chapter translations in both presentation orders. They never saw arm names,
costs, or the glossary.

| Comparison | Wins | Ties | Losses |
| --- | ---: | ---: | ---: |
| Context vs naive | 14 | 8 | 18 |
| Full harness vs naive | 14 | 9 | 17 |
| Full harness vs context (GPT-6 Luna only) | 16 | 16 | 8 |
| Full harness vs review everything (GPT-6 Luna only) | 16 | 11 | 13 |

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

| Arm | Estimated cost per chapter | Median seconds per chapter |
| --- | ---: | ---: |
| Naive | $0.0021 | 15.3 |
| Context | $0.0066 | 30.6 |
| Review everything | $0.0152 | 51.4 |
| Full harness | $0.0168 | 146.5 |

At the full harness's rate, a 1,000-chapter novel costs about $17 at those
rates (roughly $29 at peak-hour rates).

Every arm delivered 40 of 40 chapters. The full harness delivered 31 of them
with recorded warnings (mostly Jev flags that nothing cleared) and
review-everything 37 (mostly unavailable review). A warning marks unresolved
checking, not a confirmed error.

Jev routing
did not transfer: thresholds calibrated on development data routed 62% of
held-out passages to review on the first screen (38% during development), so
the routed arm cost slightly more than reviewing every passage.

The per-term counts behind the consistency percentages are kept in the private
development repository with a test that recomputes them exactly. They are
derived from the private corpus and are not published.

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
