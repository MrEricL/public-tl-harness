# Optional Jev semantic checks

This release includes an optional Jev integration for the translation-repair
showcase. Jev is a typed evaluation model: it returns bounded probabilities for
questions about possible meaning problems in a source/draft pair. It does not
translate text, propose edits, approve a patch, or replace the harness's final
fidelity review.

The integration uses the native Vercel AI Gateway evaluation route with
`typesafe-ai/jev`. Generation remains a separate provider concern. Jev is off
by default, so the ordinary workflow makes no Gateway request and requires no
Gateway credential.

## Policy modes

The JSON policy passed to `--jev-policy` is part of the saved session identity.
Changing it during resume is rejected. The public runtime supports these
controls:

| Setting | Values | Meaning |
| --- | --- | --- |
| `mode` | `off`, `shadow`, `advisory` | Disable sensing; save signals without exposing them to repair; or expose the bounded signal as advisory context. |
| `question_set` | `focused`, `dense` | Ask five focused questions or eighteen denser questions. |
| `schedule` | `initial`, `after_edit` | Sense the initial draft or opt in to sensing after an edit. |

Jev output is kept in an untrusted evidence layer. It cannot satisfy the
deterministic QA gate or the final current-draft fidelity-review requirement.
Missing, partial, or unavailable Jev evidence is not treated as a clean bill of
health.

## Try the public path

The public policy file is intentionally small and uses advisory focused sensing
on the initial draft. Set credentials through a secret manager or process
environment; the values below are placeholders and must not be committed:

```bash
export AI_GATEWAY_API_KEY="<your-vercel-ai-gateway-key>"
export DEEPSEEK_API_KEY="<your-deepseek-key>"

python -m agentic_translation harness run \
  --story samples/synthetic_repair_demo/story.yaml \
  --out runs/synthetic-jev \
  --provider-mode live --profile deepseek --model deepseek-v4-flash \
  --jev-policy samples/jev/policy.example.json
```

The sample story is newly authored technical test material. It is safe for a
public demonstration and does not stand in for a literary translation corpus.
Omit `--jev-policy` to run the same front door with Jev disabled. A completed
run can be replayed from its recorded decisions and Jev reports without a
network call:

```bash
python -m agentic_translation harness replay runs/synthetic-jev \
  --out runs/synthetic-jev-replay
```

Do not commit keys, generated `runs/` directories, source/draft text, Gateway
payloads, or provider receipts. The Gateway credential is used only for Jev;
the DeepSeek credential is used only for the generation profile in this
example.

## Aggregate evidence

Snapshot: 2026-09-21.

The following results are a public-safe summary of a private development study.
The study's corpus, orchestration, raw responses, and assignment packets are
not distributed, so these figures are historical evidence about the prototype,
not a public benchmark or a reproducible study.

### Six-arm pilot

Four development inputs were assigned to four generators under six direct or
harness configurations, once per combination: 96 assignments. There were 56
valid outputs, 38 offered, 18 held, and 40 execution failures. Offered and held
are workflow decisions, not independent fidelity scores.

| Configuration | Assigned | Offered | Held | Execution failure |
| --- | ---: | ---: | ---: | ---: |
| Direct, no Jev (D0) | 16 | 7 | 1 | 8 |
| Direct, focused Jev (DF) | 16 | 8 | 1 | 7 |
| Direct, dense Jev (DD) | 16 | 8 | 1 | 7 |
| Harness, no Jev (H0) | 16 | 7 | 4 | 5 |
| Harness, focused Jev (HF) | 16 | 3 | 7 | 6 |
| Harness, dense Jev (HD) | 16 | 5 | 4 | 7 |
| **Total** | **96** | **38** | **18** | **40** |

| Generator | Assigned | Valid output | Offered | Held | Execution failure |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek v4 flash | 24 | 22 | 17 | 5 | 2 |
| Luna Max | 24 | 7 | 2 | 5 | 17 |
| Terra High | 24 | 24 | 16 | 8 | 0 |
| Sonnet 5 High (shelved) | 24 | 3 | 3 | 0 | 21 |

The Sonnet arm hit a subscription limit. All twelve direct Luna generation
attempts timed out at 180 seconds. These availability and budget differences
prevent a clean model ranking. The study did not establish that Jev improves
translation quality.

The run also recorded 605 generator, coordinator, and specialist transport
invocations and produced eight raw/rules reference outputs. The invocation
count is not a count of vendor-internal retries.

### Jev cost, coverage, and assessment completeness

| Evidence | Recorded result | Interpretation |
| --- | --- | --- |
| Shared Jev reports | 8/8 returned every requested score; 8/8 replayed exactly | The live integration and cache-only replay worked on these reports. |
| Gateway attempts | 9 physical attempts, including one retry | The failed attempt's cost is unknown and is excluded from the successful-response total. |
| Successful-response cost | $0 promotional billed; $0.001505364 market total (about 0.15 cents) | The observed Jev response cost was negligible; this is not the total experiment cost or a price guarantee. |
| Evidence coverage | Complete source/draft; glossary context clipped in all 4 inputs and style context in 3/4 | Supporting context was partial; adapter limits are not vendor capacity claims. |
| Original assessment | 34 invocations: 14 individual ratings, 19 timeouts, 1 output-budget failure | 2/56 unique quality items resolved; 54 unresolved; no delivery item resolved. |
| Promotion | 0/8 eligible comparisons | No default promotion, significance claim, or general Jev quality-win claim is justified. |

The eight shared reports were reused across model and workflow consumers. Jev
cost did not constrain the study; generator/judge availability, budgets, and
context preparation did. A planned 48-case held-out study was not run.

### Smaller review of existing outputs

The follow-up reused existing outputs and made no new translation, Jev, or
shelved-model calls. It deduplicated 25 texts: 16 Terra outputs, 7 valid Luna
outputs, and 4 initial drafts. Four Luna High review calls completed in
33.7–165.7 seconds; no Medium fallback was needed. All 25 ratings replayed
identically with process and network access disabled. Reported CLI usage was
72,515 input tokens and 11,287 output tokens; that is usage data, not an API
invoice.

| Terra subset | Outputs reviewed | Offered | Held |
| --- | ---: | ---: | ---: |
| Direct, no Jev | 4 | 4 | 0 |
| Direct, dense Jev | 4 | 4 | 0 |
| Harness, no Jev | 4 | 3 | 1 |
| Harness, dense Jev | 4 | 1 | 3 |

All four holds were action-budget exhaustion. This was a small,
single-judge, meaning-focused review, not an exhaustive literary evaluation.
Luna judging earlier Luna outputs is an additional limitation. The raw labels
cannot be treated as a validated accuracy score.

## What the prototype taught us

The review exposed a false glossary requirement: a short glossary
key occurred only inside a longer glossary entry or name. The no-Jev path could
insert the short target into an already-correct entry, and QA accepted the
result. The fix now ignores fully contained short-key occurrences while still
checking independent mentions and the longer entry. Authored regression tests
cover this boundary; the original source text, names, and provider responses
are not part of the public release.

A separate breathing-label comparison was withdrawn as a confirmed translation
error. Literal order alone did not establish a material story consequence, so
those raw model labels are excluded from validated wins and losses. This is a
localization judgment boundary, not evidence for or against Jev.

## Limits

- Jev is advisory sensing, not a translator, editor, QA authority, or final
  reviewer.
- The study reused reports and had substantial execution failures. Its small
  samples, model availability, single-judge follow-up, and incomplete held-out
  plan do not support generalization.
- The adapter can mark larger or insufficiently aligned source/draft pairs
  unavailable rather than split them using an assumed paragraph correspondence.
  Its conservative 28,000-byte request budget is an implementation safeguard,
  not a claimed Jev model limit.
- The reported response cost excludes an unknown failed-attempt cost and is not a
  promise of future pricing. It is also not the total translation experiment
  cost.
- A Jev report cannot clear a stale or missing final fidelity review. Edits
  invalidate a review tied to an earlier draft hash.

Read [DATA_NOTICE.md](../DATA_NOTICE.md) before adding fixtures. The Jev release
additions consist of the runtime integration, authored policy/tests, and
aggregate findings. They do not distribute the private study's source passages,
private translations, private glossaries, raw prompts, raw model responses, or
private run artifacts.
