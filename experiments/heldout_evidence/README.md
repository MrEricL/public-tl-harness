# Recompute the headline

Run this from the repository root with Python 3.11 or later:

```bash
python experiments/heldout_evidence/verify.py
python experiments/heldout_evidence/verify.py --json
```

No installation, API key, model calls, or private text is required. The script
recomputes the terminology counts, drift transitions, paired effects, cost
arithmetic, and judge coverage in [RESULTS.json](RESULTS.json). It checks the
internal consistency of these published aggregates; it cannot independently
validate the translations or the model that extracted the terminology.

## Count-backed result

| Arm | Modal / valid extracted renderings | Consistency | Terms with multiple renderings |
| --- | ---: | ---: | ---: |
| Chapter only | 561 / 851 | 65.9% | 116 / 160 |
| Source memory + glossary | 734 / 854 | 85.9% | 61 / 160 |
| Full harness | 744 / 853 | 87.2% | 62 / 160 |
| Review every passage | 744 / 854 | 87.1% | 63 / 160 |

An inconsistent observation is a valid extracted rendering other than that
term's most common rendering in the same arm. The reduction is
`1 - (109 / 853) / (290 / 851) = 62.5%`, or a 2.67-fold ratio of inconsistency
rates. Of 72 terms with at least three baseline renderings, 39 use exactly one
under the full harness. The transition tables retain all terms, including
regressions: 90 term scores improved, 13 worsened, and 57 were unchanged.

The effect is present in each of the four sampled works:

| Anonymous work | Chapter only | Source memory + glossary | Full harness | Full vs chapter only |
| --- | ---: | ---: | ---: | ---: |
| A | 67.3% | 83.6% | 82.5% | +15.3 pp |
| B | 63.2% | 89.3% | 90.6% | +27.4 pp |
| C | 68.4% | 87.3% | 88.6% | +20.2 pp |
| D | 64.6% | 83.1% | 86.8% | +22.2 pp |

The paired per-term gain is +21.0 pp; the observation-weighted table gains
+21.3 pp. The script enumerates all 256 ordered resamples of the four works,
reproducing a +16.7 to +25.1 pp interval. Treat this as a descriptive check on
four works, not precise uncertainty about all novels. Context supplies most
of the gain. Repair adds +1.1 pp paired per term, with a resampling interval
of -0.8 to +2.3 pp; 12 term scores improve, six regress, and 142 are unchanged.

## Provenance and measurement

These are new anonymous summaries of the same frozen September 2026 study,
derived from its October 1 count export. There was no new generation or
judging, and none of the historical result files was changed. Every arm
delivered all 40 assigned chapters. The file contains only numeric summaries
and study descriptions: no chapter text, rendering strings, corpus names,
term identifiers, source hashes, prompts, or private paths.

The inventory contains 40 recurring source terms per work, chosen by source
frequency from the union of source-memory candidates and supplied glossary
terms. Extraction is arm-blind and records one rendering per term per chapter,
not every mention. The primary score excludes null or invalid extractions;
the archived report records 857 extraction opportunities per arm. The valid
denominators above are smaller. The opportunity count is documentary because
the older export did not retain missingness per term.

The work totals and paired sums preserve the two original weightings.
Three-by-three transition tables group terms by one, two, or three-plus
renderings. The verifier checks their row and column totals against each arm's
drift counts. Costs are historical off-peak estimates, including memory and
repair, and exclude evaluation and subscription judges.

## Quality and the glossary evidence gap

The full harness vs baseline has 14 wins, nine ties, and 17 losses. Both judge
families supplied valid ratings in both presentation orders on 34 of 40
pairs; six use one complete judge after the other's calls failed. The
two-judge subset is 14 wins, six ties, and 14 losses. Context vs baseline has
33 two-judge and seven one-judge pairs. These counts establish neither overall
quality superiority nor statistical noninferiority.

Exact glossary match (37.0% to 86.0%) was already computed as a secondary
metric; a later audit promoted it to the headline. The preregistered glossary
endpoint was permissive string overlap. Both sets of rounded rates are retained under `documentary_metrics`,
separate from recomputable counts: the old export omitted their integer
numerators and denominators. The verifier explicitly reports that gap rather
than reconstructing counts from rounded percentages. Neither self-consistency
nor adherence to a supplied glossary proves that a translation is correct.
