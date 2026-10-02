# Two focused portfolio tests

Two small tests show where the gains come from, with a breakdown, rather than a large study.
Both tests use the same `portfolio-review.v1` runtime profile as `demo.py`:
source memory where applicable, term alignment, full passage review, bounded
edits, and existing loss/gloss/duplication guards. One model account; no Jev.

## Test 1 — where does terminology improve?

```bash
python demo.py --provider openai --model YOUR_MODEL \
  --source samples/showcase/source \
  --glossary samples/showcase/terms/demo_glossary.json \
  --compare --out runs/portfolio-terms --open
```

Start with the bundled story or three to five consecutive chapters of a story
you are permitted to use. Supply a small glossary before looking at outputs.
Use the same model and settings across all arms; the runner handles that.

| Arm | Added capability | Comparison |
| --- | --- | --- |
| `naive` | Chapter-only translation | Starting point |
| `glossary` | Translator's chosen terminology | Benefit of a prompt alone |
| `contextual` | Causal source-grounded memory | Additional context benefit |
| `harness` | Alignment and guarded review/repair | Incremental repair benefit |

The harness begins with the exact contextual draft, so this last contrast is
before/after repair, not two unrelated generations. Memory learns from Chinese
source only and translator terms win over conflicting automatic choices.

Primary output: requested-term/window adherence, overall and by glossary
category. The dashboard shows raw counts, percentage-point changes, relative
changes, fixed checks and regressions. It also preserves output failures, actual
context, model calls, latency, and optional user-priced cost. Read the paired
text in any category that moved materially. A 0/0 category is not a perfect score.
The score is intentionally a concrete product contract, not overall prose quality.

Summarize a run in this form: “Requested technique-name adherence rose
from A/B to C/B (+X percentage points); N violations fixed, M regressions.”
Use “X% higher” only for `(after / before - 1) * 100`, with a nonzero baseline.
The report also retains negative changes instead of selecting only winners.

## Test 2 — targeted semantic repair

```bash
python demo.py --repair-bench --provider openai --model YOUR_MODEL \
  --out runs/portfolio-repair --open
```

`repair_cases.json` contains ten seeded problems and two clean controls. The
areas are negation, quantities, chronology, conditions, actor attribution,
uncertainty, omissions, unsupported additions, terminology, and beliefs. Each
case is a short authored Chinese passage plus an English draft, not personal
novel content. The model receives source/draft/terminology, never references or
evaluator regexes. The exact same trusted executor used for folder translation
accepts or rejects its proposed edits.

The score is a narrow, inspectable fact check with allowed and forbidden patterns.
The starting drafts pass 2/12 checks by construction, so report “fixed X of ten
seeded errors, with Y of two clean controls preserved,” not a claim about natural
error prevalence. Some valid paraphrases may miss the matcher; inspect the
report before writing a headline, and keep the raw automated counts intact.

## Just run these, then decide

One run of each is enough to start. A second model or second small story is an
optional robustness check, not an open-ended data-collection commitment. Use a
fresh output directory each time. Model IDs, inputs, glossary, policy, exact
calls and results are recorded. Do not tune on a run and call its rerun held out.

No live model scores from these tests are claimed yet. A first live DeepSeek
attempt produced no output because the provider queued every request; calls now
fail at `--call-timeout`, and such runs report "No score". The
[focused tests](../../tests/test_portfolio_demo.py) exercise the real pipeline
with controlled responses and real SDK/local-HTTP transport.
The older 40-chapter study remains separate: 65.9% → 87.2% consistency, about
32% relative, with a different Jev-routed configuration. See the root README.
