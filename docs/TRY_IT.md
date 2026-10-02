# Try the translation harness

One folder. One model account. A translation and an inspectable HTML report.
Run commands from the repository root, the folder containing `demo.py`.

## Fastest start

```bash
python -m pip install -e .
python demo.py
```

Drag a chapter folder into the terminal when asked. Use one UTF-8 `.txt` file per
chapter; filenames are naturally sorted (`chapter2.txt` before `chapter10.txt`).
Enter a provider and a real chat-completion model ID available to your account.
The API key prompt is hidden. It is never written to the run configuration.
Press Enter at the folder prompt for the included three-chapter story and glossary.

The first run uses at most three chapters and writes `runs/try-it`. It shows the
number selected and excluded before any requests. Use a new `--out` for another
run; existing results and source files are never overwritten. `--limit 0`
processes all chapters. Subfolders are not scanned. The glossary belongs outside
the chapter folder so it cannot accidentally become a chapter.

## Scripted walkthrough, no key

```bash
python demo.py --offline --out runs/offline-demo --open
```

This uses authored, scripted responses and produces the existing showcase,
including EPUB. It does not translate arbitrary input or measure a live model.

## Your model and chapters

```bash
# Set OPENAI_API_KEY in your shell, or omit it and answer the hidden prompt.
python demo.py --provider openai --model YOUR_MODEL \
  --source "/path/to/chapters" --out runs/my-book --open
```

This translates with causal source memory, then applies bounded terminology and
fidelity repair. The same model supplies extraction, translation and review. No
Jev SDK/key, judge account, database, server, or story YAML is needed.

| Preset | Key environment variable | Endpoint |
| --- | --- | --- |
| `openai` | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| `anthropic` | `ANTHROPIC_API_KEY` | `https://api.anthropic.com/v1` (compatibility API) |
| `deepseek` | `DEEPSEEK_API_KEY` | `https://api.deepseek.com` |
| `openrouter` | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |
| `custom` | `TL_API_KEY` | Supply `--base-url` |

The preset selects transport, not a guaranteed model entitlement. Choose a
model that supports chat completions and can return the requested JSON/text.
Temperature is omitted by default; pass `--temperature 0` only for models that
support it. An endpoint override uses `TL_API_KEY`, not another provider's key.
HTTPS is required except for a localhost endpoint. No endpoint secrets belong in
URLs or command-line flags. API access is separate from consumer chat subscriptions.

Provider references: [OpenAI chat completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create),
[Anthropic compatibility](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk),
[OpenRouter quickstart](https://openrouter.ai/docs/quickstart).
Real SDK transport is tested against a local HTTP endpoint, and preset request
shapes are covered. A first live DeepSeek attempt produced no output because the
provider queued every request; calls now fail at `--call-timeout`.

## Test 1: find where the gains come from

```bash
python demo.py --provider openai --model YOUR_MODEL \
  --source samples/showcase/source \
  --glossary samples/showcase/terms/demo_glossary.json \
  --compare --out runs/test-1 --open
```

The report compares four paths on the same ordered chapters: plain translation,
glossary-only prompting, glossary plus source memory, and the exact contextual
draft after repair. It reports people, places, sects, techniques and artifacts
separately. Both positive changes and regressions stay visible.

For your own story, a simple glossary can be `青云宗 -> Azure Cloud Sect` per line.
To get the area breakdown, use a JSON array:

```json
[
  {"source": "青云宗", "target": "Azure Cloud Sect", "category": "sects",
   "blocked_variants": ["Blue Cloud School"]},
  {"source": "月影步", "target": "Moon-Shadow Step", "category": "techniques"}
]
```

The metric is specified-term adherence per source window: the requested target
must appear, without a blocked variant. It is not per-mention correctness or a
universal translation grade. No glossary means no independent score, not 100%.
An automatic memory term is still visible in the context and repair trace.

## Test 2: does repair fix meaning without breaking clean text?

```bash
python demo.py --repair-bench --provider openai --model YOUR_MODEL \
  --out runs/test-2 --max-calls 120 --open
```

Twelve short authored passages cover negation, quantities, chronology,
conditions, actor attribution, uncertainty, omissions, unsupported additions,
terminology and beliefs, plus two clean controls. The evaluator's reference and
checks are never sent to the model. This isolates repair capability without
paying to translate another corpus. The checks are deliberately small; inspect
flagged text rather than treating a regex as a literary critic.

## Inspect and replay

Open `report.html` in the output folder. Expand a chapter for the source, each
translation, the exact before/after diff, actual context, and accepted/rejected
repair events. `results.json` preserves denominators, categories, resource usage
and failures. `calls/` contains each request, raw response and receipt;
`sessions/` contains the executor's snapshots. A completed folder run also has
`delivery/book.txt` and `delivery/book.epub`.

```bash
python demo.py --replay runs/test-1 --out runs/test-1-replay --open
```

Folder-demo replay uses copied sources and successful cached model responses,
never a network request. Changed source, glossary, policy, or a cache miss is
rejected. Replay is not mid-run recovery and the separate repair challenge has
no replay command. Replayed token/latency figures describe recorded work;
`physical_calls` is zero. A replay itself does not provide a new quality trial.

## Keep the first run small

The defaults are three chapters, at most 12,000 source characters per chapter,
120 physical calls, and 8,192 output tokens per call. Set `--limit`,
`--max-source-chars`, `--max-calls` and `--max-output-tokens` deliberately for
larger runs. Failed requests count against the call ceiling; there are no hidden
transport retries. Truncated output is retained as a failure, never completion.
A provider/auth/budget failure stops further spending and leaves assigned
chapters in the report's denominator. Already generated drafts remain inspectable.
Each call also has a total deadline, `--call-timeout` (300 seconds by default),
so a provider that queues requests behind keep-alive lines fails fast instead of
hanging. A run with any missing output reports "No score" rather than a delta.

Token and call counts are recorded whenever the provider returns usage. Dollar
figures are optional: supply `--input-price` and `--output-price` in USD per
million tokens. `--max-usd` additionally enables a conservative estimated-cost
ceiling using those rates. It is an estimate, not a provider-enforced spending
limit, and ignores discounts. Without prices, dollars are unknown, not free.
Per-arm estimates include shared dependencies; use the top-level physical totals
for the run, not the sum of all arms.

Runs contain private source and translated text. They are ignored under `runs/`;
do not commit or publish them without checking content rights and privacy.
