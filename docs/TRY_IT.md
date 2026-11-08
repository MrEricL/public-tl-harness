# Try the translation harness

Run from a source checkout with Python 3.11 or newer. The supported sample
workflow uses an editable install; bundled samples are not promised in a
standalone wheel installation.

## Start with the folder command or local page

```bash
python -m pip install -e .
python demo.py
```

Enter a folder containing one UTF-8 `.txt` file per chapter. Filenames are
naturally sorted, so `chapter2.txt` comes before `chapter10.txt`; subfolders
are not scanned. Press Enter at the folder prompt for the authored three-chapter
 story. Choose a provider, a real chat-completion model ID available
to your account, and a key at the hidden prompt. Keep the glossary outside
the chapter folder.

The first run selects at most three chapters and writes `runs/try-it`.
It prints the selected/excluded count before requesting a key. Use a fresh
output directory for each run; `--limit 0` selects all chapters.

For the local UI:

```bash
python -m agentic_translation.workbench --open
```

Choose the sample or add chapter files and an optional glossary. The workbench
starts a live folder run, shows activity and available usage, requests a stop
before the next model call, and opens terminology checks, chapter diffs,
accepted/rejected edits, memory receipts, and saved-run replay. It uses the
same folder engine and binds to `127.0.0.1`. Pasted keys stay in process
memory rather than run configuration. To inspect another output location:

```bash
python -m agentic_translation.workbench runs --open
```

Live translation requires API access. Replaying an existing completed folder
run requires its saved inputs and responses, with no new model calls.

## Supply a model and your chapters

```bash
# Set OPENAI_API_KEY in the shell, or answer the hidden prompt.
python demo.py --provider openai --model YOUR_MODEL \
  --source "/path/to/chapters" --out runs/my-book --open
```

The same model handles memory extraction, writing, and source review. The
normal workflow needs no separate judge/router account or database.

| Preset | Key environment variable | Base URL |
| --- | --- | --- |
| `openai` | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| `anthropic` | `ANTHROPIC_API_KEY` | `https://api.anthropic.com/v1` |
| `deepseek` | `DEEPSEEK_API_KEY` | `https://api.deepseek.com` |
| `openrouter` | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |
| `custom` | `TL_API_KEY` | Supply `--base-url` |

A preset selects transport; it does not grant access to a model. Use a model
that supports the requested chat-completion JSON/text contracts. HTTPS is
required except for a localhost endpoint. Custom endpoints use `TL_API_KEY`.
The workbench accepts an endpoint only with its Custom preset. Do not place
credentials in the URL. API access is separate from a consumer chat subscription.

Temperature is omitted by default for compatibility. Pass `--temperature 0`
only if the selected model supports it. For DeepSeek, the folder command
defaults `--thinking-mode` to `disabled`; an explicit `enabled` or `disabled`
value is recorded with the request/cache identity.

## Compare the four paths

```bash
python demo.py --provider openai --model YOUR_MODEL \
  --source samples/showcase/source \
  --glossary samples/showcase/terms/demo_glossary.json \
  --compare --out runs/comparison --open
```

The report compares each chapter alone, glossary-only prompting, glossary plus
source memory, and that exact memory-context draft after repair. Both positive
changes and regressions remain visible. The authored sample is a runnable
mechanism demonstration, not a held-out literary-quality benchmark.

`python demo.py --repair-bench` runs a small seeded repair challenge instead;
[the authored workflow checks](../experiments/portfolio_demo/README.md) explain
both.

The glossary accepts arrow-format text (`SOURCE_TERM -> Chosen Rendering`)
or a categorized JSON array:

```json
[
  {"source": "SOURCE_TERM", "target": "Chosen Rendering", "category": "terms",
   "blocked_variants": ["Disallowed Rendering"]}
]
```

The displayed adherence check looks for requested renderings in corresponding
source windows. It is not a per-mention correctness score. Without a supplied
glossary there is no independent reference score. Memory names and the exact
context remain inspectable even when no reference score is available.

## Continue an installment or read a lead-in

Use `--memory-from` with an earlier completed folder run of the same series:

```bash
python demo.py --provider openai --model YOUR_MODEL \
  --source "/path/to/new-chapters" --memory-from runs/my-book \
  --limit 0 --out runs/next-installment --open
```

The runner validates the carried memory against its saved source and snapshot
chain. It assigns the new chapters positions after the carried prefix and
copies the needed material into the new run so replay is self-contained.

To read earlier source files without translating them:

```bash
python demo.py --provider openai --model YOUR_MODEL \
  --source "/path/to/ordered-chapters" --lead-in 5 --limit 0 \
  --out runs/with-lead-in --open
```

The first five selected files populate memory; subsequent files are translated.
A longer lead-in increased name delivery in the registered study but did not
establish a consistency improvement. It is optional. These two inputs belong
to the folder CLI; the workbench form and book CLI do not expose them.

## Inspect and replay

Open `report.html` in the run folder. It shows chapter source, drafts and final
text, exact diffs, context, edits, warnings, and available resource counts.
`results.json` retains denominators, failures, and resources. `calls/` holds
requests and receipts; `sessions/` holds repair state. Completed folder delivery
is in `delivery/book.txt` and `delivery/book.epub`.

```bash
python demo.py --replay runs/comparison --out runs/comparison-replay --open
```

Replay consumes saved successful responses without a network request. It
rejects changed source/glossary, incompatible recorded settings, and missing
calls. It restores the recorded memory, alignment, and thinking settings;
do not add new live-input or policy overrides. A replay can itself be moved
and replayed. Replay reproduces a completed run rather than resuming midway
through a chapter.

## Run a configured book

The thin book CLI supports the same named contextual and full-review policies:

```bash
# Set DEEPSEEK_API_KEY before this live book command.
python -m agentic_translation harness run \
  --story samples/showcase/story.yaml --out runs/book \
  --provider-mode live --profile deepseek --model YOUR_DEEPSEEK_MODEL \
  --strategy contextual --interaction unattended \
  --memory source-grounded --glossary-policy run-local-auto --max-usd 1
```

`--profile deepseek` selects the provider; `--strategy contextual` selects the
workflow. `contextual` (`contextual.v1`) generates with source memory and the
translator's glossary, then performs deterministic checks. Use
`--strategy portfolio-review` (`portfolio-review.v1`) to add term alignment
and full source review with bounded edits. The folder demo/workbench support
more provider presets; this configured book command uses DeepSeek.

The story YAML names the ordered chapter IDs, source folder, glossary, and
style inputs. The bundled file is an authored example. Book output records
its chosen profile, warnings, and delivery. Book-level resume is not
implemented; use an empty output directory. For folder replay and carried
memory, use `demo.py`.

The book CLI also keeps the older `adaptive` and `always-review` strategies,
with optional Jev routing for `adaptive`. These still make live API calls;
they do not replay saved book runs. Use the named profiles above for new work.

## Limits and spending

The folder default is three chapters, at most 12,000 source characters per
chapter, 120 attempted calls, 8,192 output tokens, and 300 seconds per call.
Raise limits deliberately for larger runs. Failed requests consume the call
budget; missing or truncated output remains a failure. Fatal provider or
budget errors stop later spending and preserve assigned chapters in the report.
A stop request takes effect before another call, not during a provider request.

Dollar reporting is optional for the folder runner. Supply `--input-price`
and `--output-price` in USD per million tokens; `--max-usd` then provides a
conservative estimate ceiling using those rates. Unknown usage is not free,
and these are not provider-enforced billing limits. Per-arm estimates can
share dependencies: use top-level physical totals for total spending.

Run folders contain source text, translations, and model responses. Keep them
private unless you have checked rights and content. [Data notice](../DATA_NOTICE.md)

[How it works](HOW_IT_WORKS.md) · [Memory](MEMORY.md) ·
[Code map](CODE_MAP.md) · [Results](RESULTS.md)
