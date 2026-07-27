# User Guide

This guide explains how to install, run, inspect, and safely share the Agentic Long-Form Translation Prototype.

The short version:

```text
Run a story config.
Inspect the report.
Use QA artifacts to understand what failed and what got repaired.
Use TXT/EPUB outputs only after artifact QA passes.
```

The project is a bench for agentic translation: cheap translation output is useful, but not trustworthy by default. The offline score is compliance evidence, not semantic quality.

For a concise public walkthrough, use `DEMO_SCRIPT.md` as the 90-second demo narration guide.

## 1. Core Concepts

### Story Config

A story config is a YAML file that tells the pipeline where to find source chapters, glossary terms, expected demo baseline output, and where to write runs.

The public demo config is:

```text
samples/public_demo/story.yaml
```

Minimal shape:

```yaml
slug: public_demo
title: Public Demo
language: zh
public_safe: true
chapter_ids:
  - "0001"
paths:
  source_dir: samples/public_demo/source
  glossary_path: samples/public_demo/terms/master_glossary.txt
  prompt_path: samples/public_demo/prompt.txt
  expected_dir: samples/public_demo/expected
  runs_dir: runs
qa:
  max_repairs: 3
report:
  mode: full
  max_source_chars: 1200
  max_translation_chars: 1200
package:
  formats:
    - txt
    - epub
```

### Chapter Files

Source chapter files use four-digit IDs:

```text
source/0001.txt
source/0002.txt
source/0003.txt
```

Output files preserve that contract:

```text
translated_final/0001.txt
review/<story_slug>_0001.txt
review/<story_slug>_0001.epub
```

### Glossary Format

Preferred glossary format:

```text
天道 -> Heavenly Dao
# block: Way of Heaven; Heavenly Way

推演 -> simulation
# block: deduction; deduction begins
```

The parser also supports older colon-style entries:

```text
天道: Heavenly Dao, Way of Heaven
```

For colon entries, the first candidate is the canonical target.

### Provider Modes

The prototype has three provider modes:

```text
offline  deterministic, no model calls
live     OpenAI-compatible model calls
replay   cached live responses, no new model calls
```

Offline mode is the stable demo and test path. Live/replay mode is the path where the model-backed agentic loop becomes real, but the claim gate still requires cached model-response evidence that matches the manifest: judge responses must match the selected candidate, and repair responses must match the accepted patch.

### Compliance vs Quality

The deterministic QA score is a compliance score. It checks things such as residue, glossary drift, panels, prompt leakage, and artifact safety.

It is not proof that the translation is beautiful, faithful, or literary.

Quality scoring is separate and only appears when live/replay candidate selection supplies faithfulness and fluency scores.

## 2. Installation

### Standard Install

```bash
cd "agentic_translation_prototype"
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

## 3. Run The Public Demo

Run:

```bash
agentic-translation demo run samples/public_demo/story.yaml \
  --provider-mode offline \
  --run-id demo \
  --overwrite \
  --seed 7
```

Expected terminal story:

```text
OK load_story
OK source_qa, findings 0, score 100
OK translate_baseline
OK translate_glossary
WARN qa_baseline, findings 10, score 19
WARN qa_glossary, findings 2, score 84
OK repair, patches 2, accepted 2
OK qa_final, findings 0, score 100
OK package_txt
OK package_epub
OK artifact_qa, failures 0
```

Expected final metrics:

```text
Residual Chinese:      baseline 1, glossary 0, final 0
Glossary Violations:   baseline 6, glossary 0, final 0
Panel Mismatches:      baseline 1, glossary 1, final 0
Score:                 baseline 19, glossary 84, final 100
```

Open:

```bash
open runs/demo/report.html
```

Final TXT:

```bash
sed -n '1,120p' runs/demo/review/public_demo_0001.txt
```

Expected final text:

```text
Chapter: 0001

Chapter 1: The Simulator Starts

[Simulator Started]

The Heavenly Dao split open above the city.

Lin Che looked at the panel and whispered, "Begin simulation."

[Remaining uses: 3]
```

## 4. Read The Run Artifacts

Each run writes to:

```text
runs/<run_id>/
```

Important files:

```text
source/0001.txt                  copied source chapter
translated_baseline/0001.txt     cheap baseline translation
translated_glossary/0001.txt     code-produced glossary transform
translated_final/0001.txt        accepted final translation
review/*.txt                     TXT deliverable
review/*.epub                    EPUB deliverable
qa_source.json                   source QA report
qa_baseline.json                 baseline translation QA
qa_glossary.json                 glossary translation QA
qa_final.json                    final translation QA
artifact_qa.json                 TXT/EPUB artifact QA
bench_ablation.json              bench/cost/compliance ablation receipt
manifest.json                    run summary and provenance
trace.jsonl                      stage-by-stage timeline
run_notes.md                     human-readable run notes
report.html                      main review/demo report
```

### `report.html`

Use this as the primary review artifact.

It includes:

- Summary cards.
- Run Cockpit cards that frame the demo as productive distrust over an offline replayable harness.
- Harness trace, evaluator-optimizer loop, router decision, Editor's Room candidate-selection, patch-acceptance, and artifact-gate cards.
- Decision Timeline cards that narrate translate cheap, QA gauntlet, route findings, Editor's Room, patch acceptance, re-QA, artifact gate, and cost receipt.
- Bench Ablation Strip comparing cheap baseline, glossary canon, router/patch loop, artifact gate, and a frontier-everywhere cost estimate.
- Pipeline timeline.
- Compliance vs quality explanation.
- Glossary canon.
- Source, baseline, glossary, and final text.
- QA findings.
- Router and Editor's Room candidate-selection decision.
- Candidate score table with literal, fluent, and canon-strict worker outputs plus the blind judge winner.
- Patch acceptance diffs.
- Artifact QA.
- Output paths.

### `bench_ablation.json`

This is the compact bench receipt behind the report strip. It records the compliance score and estimated cost for each demo step:

```text
cheap baseline -> glossary canon -> router + patch loop -> artifact gate
```

It also includes a frontier-everywhere counterfactual. The numbers are cost estimates for demo comparison, not billing receipts, and the score is still compliance-only.

### `trace.jsonl`

Each line is a stage record:

```json
{"stage":"qa_glossary","status":"warn","findings":2,"score":84}
```

Useful statuses:

```text
ok    stage completed cleanly
warn  stage completed with QA findings
fail  stage failed or blocked the run
```

### `artifact_qa.json`

This verifies packaged outputs:

```json
{
  "expected_chapters": 1,
  "txt": {
    "chapter_markers": 1,
    "contains_chinese": false,
    "contains_prompt_leakage": false
  },
  "epub": {
    "xhtml_chapters": 1,
    "contains_chinese": false,
    "contains_prompt_leakage": false
  },
  "passed": true,
  "failures": []
}
```

If artifact QA fails, treat the deliverable as unsafe.

## 5. Understand Repair Routing

The repair router chooses one of:

```text
rule                 deterministic repair logic
candidate_selection  generate candidates, score/select one, patch it in
human_review         do not auto-repair
none                 no repair needed
```

In the public demo:

```text
system_panel_count -> candidate_selection
heading_format     -> rule
```

For local corpus runs, glossary-required findings can also route to `rule` when the translated paragraph contains an explicit known alias from the glossary entry's `candidates` list. Example:

```text
仙盟: Immortal Alliance, Fairy Alliance
```

If the source contains `仙盟`, the final text contains `Fairy Alliance`, and `Immortal Alliance` is missing, the rule patch can replace the observed alias with the canonical term.

The same rule can use a bounded cross-glossary alias when another glossary entry's canonical English target is observed, shares at least two meaningful word stems with the missing target, and that other source term is not present in the chapter. For example, if `法則之主 -> Lord of Laws` is missing and the final text says `Law Lord`, but the source does not contain `法主`, the patch may replace the observed `Law Lord` span. If the canonical term is missing and no known alias is observed, the finding remains human review.

English `replace_span` patches are token-bounded, so replacing `Law Lord` does not corrupt a larger token such as `Law Lords`.

The candidate-selection example is intentionally small and visible:

```text
raw candidate:       remaining uses: 3
panel candidate:     [Remaining uses: 3]
alternate candidate: [Remaining attempts: 3]
selected candidate:  [Remaining uses: 3]
```

The rule repair normalizes the title:

```text
Chapter One: simulation
-> Chapter 1: The Simulator Starts
```

The title polish comes from an explicit source-title mapping in the offline rule provider. It is not a hidden full polished translation fixture.

## 6. Run A Local Corpus Smoke Test

Use this to test the pipeline shape against private scraped material without putting that material in public bundles.

Fastest useful path:

```bash
agentic-translation produce ../simulator_alliance \
  --count 2 \
  --dry-run
```

`produce` is the preferred operator command for the old corpus layout. It reads `scraped/`, `terms/master_glossary.txt`, prior matching runs, and available `translated*` baseline directories. In dry-run mode it should not create fixtures, run batches, or call providers; it tells you source count, processed count, latest chapter, next chapter, selected chapters, run ids, source dir, glossary path, translated baseline dir, and the exact follow-up command. It also prints copyable `Source dir(s):`, `Glossary:`, `Run ID(s):`, and `Translated dir(s):` lines outside the table. With `--json`, the first planned chunk is available as `next_chunk` and `next_command`; all commands are available in `follow_up_commands`; each chunk also carries its own `translated_dir` and `follow_up_command`. The JSON payload also includes `project_status`, so scripts can read the same progress snapshot without running a second command. Actual execution uses the same planned run id and ends by printing `Produced run(s): ...`, `Review run(s): runs/...`, `Status: <packaged>/<total> packaged, ...`, direct `TXT artifact(s): ...` / `EPUB artifact(s): ...` paths when the manifest points at packaged artifacts, `Project progress: <processed>/<source> processed, latest <chapter>, next <chapter>`, and `Next action(s): ...`; actual execution with `--json` suppresses progress output and emits one parseable receipt with pre-run `project_status`, post-run `project_status_after`, `run_ids`, `review_runs`, per-run `run_summaries` with artifact paths, top-level aggregate `artifacts`, aggregate `status_summary`, and `recommended_next_actions`.

Run the planned chunk:

```bash
agentic-translation produce ../simulator_alliance \
  --count 2 \
  --overwrite
```

Use an explicit chapter selection when you need to skip around:

```bash
agentic-translation produce ../simulator_alliance \
  --chapters 0029,0030 \
  --overwrite
```

Tiny DeepSeek excerpt probe:

```bash
agentic-translation --env-file .env.local produce ../simulator_alliance \
  --provider deepseek \
  --cheap 500 \
  --count 1 \
  --overwrite
```

`--provider deepseek` probes DeepSeek first with `deepseek-chat`, records through `.agentic_cache/produce_deepseek`, and then delegates to the practical smoke path with fallback enabled. If the key is missing or the account is out of balance, the run records the provider failure and falls back instead of retrying expensive calls.

Lower-level path with more knobs:

```bash
agentic-translation smoke-project ../simulator_alliance \
  --first 2 \
  --provider-mode offline \
  --run-id simulator_project_first2_0001_0002 \
  --overwrite \
  --practical \
  --no-write-proof
```

`smoke-project` expects a corpus project directory with `scraped/`, `terms/master_glossary.txt`, and optionally `translated*` baseline directories. It imports the selected chapters into `local_fixtures/<project>_smoke_<range>/`, picks the latest translated directory that covers those chapters, runs the batch pipeline, writes aggregate TXT/EPUB review artifacts, and writes the triage packet: review queue, glossary gap report, manual edit plan, and work order. Use `--first N` for a quick small batch from sorted chapter files, `--start 0042 --first 3` for a window, or `--chapters 0001-0005,0010` for explicit picks. When `--run-id` is omitted, `smoke-project` prints and uses an id based on project name, mode, and selected chapters, such as `simulator_alliance_practical_0015_0016`. `--practical` is the default useful corpus loop: it applies safe glossary-update candidates, reruns affected chapters, bridges remaining glossary items through the auditable manual-review path, and attempts narrow split-panel merges. English aliases become `alias (expected)`, common plurals are handled, and source-only misses become visible `Term audit: <expected>.` paragraphs unless `--skip-source-terms` is passed. Practical bridge can run up to three bounded passes because accepted bridge edits may expose new glossary checks after QA refresh.

Continue from the previous chunk:

```bash
agentic-translation smoke-project ../simulator_alliance \
  --continue-latest \
  --first 2 \
  --chunks 2 \
  --provider-mode offline \
  --overwrite \
  --practical \
  --no-write-proof
```

`--continue-latest` scans `runs/*/batch_manifest.json`, finds the matching project smoke run with the highest chapter id, and starts the next `--first N` selection after that chapter. `--chunks N` repeats the same continuation cycle N times. Loop mode deliberately rejects custom `--run-id` and `--out`, because each chunk needs its own inferred run and fixture directory. Use `--after-run <run-dir>` when you want to continue from a specific run instead.

Check progress:

```bash
agentic-translation project-status ../simulator_alliance
agentic-translation project-status ../simulator_alliance --json
```

`project-status` reads matching `runs/*/batch_manifest.json` files, reports unique processed chapter count, latest and next chapter, current status counts, and recent matching runs.

Use `smoke-local` when you need to choose each path manually:

```bash
agentic-translation smoke-local \
  --source-dir ../simulator_alliance/scraped \
  --glossary ../simulator_alliance/terms/master_glossary.txt \
  --chapters 0001 \
  --translated-dir ../simulator_alliance/translated_001_421 \
  --provider-mode offline \
  --run-id simulator_one_command_bridge_0001 \
  --overwrite \
  --practical \
  --no-write-proof
```

Tiny DeepSeek translation smoke:

```bash
agentic-translation --env-file .env.local provider-probe deepseek \
  --cache-dir .agentic_cache/deepseek_probe
```

Use `provider-probe` first when you only want to verify that a live provider is reachable. It sends one tiny JSON request, records the response when possible, and fails before any chapter import or batch run if credentials are missing. In this shell, the probe currently reports `DEEPSEEK_API_KEY is required for live deepseek providers`.

```bash
agentic-translation --env-file .env.local smoke-project ../simulator_alliance \
  --chapters 0001 \
  --deepseek \
  --source-char-limit 500 \
  --cache-dir .agentic_cache/deepseek_smoke \
  --run-id simulator_deepseek_smoke \
  --overwrite
```

`--deepseek` is intentionally narrow: live DeepSeek translation, offline judge, offline repair, cache recording, `deepseek-chat`, excerpt report mode, and offline fallback for live-provider failures. Add `--source-char-limit 500` when you want to spend pennies on a real-corpus excerpt instead of translating a whole chapter. If `DEEPSEEK_API_KEY` is missing or the account returns a hard error such as `402 Insufficient Balance`, the smoke command still writes the offline fallback package and records the reason in `provider_failures`; hard account/request errors are not retried.

Create the local fixture:

```bash
agentic-translation import-local \
  --source-dir ../simulator_alliance/scraped \
  --glossary ../simulator_alliance/terms/master_glossary.txt \
  --chapters 0001-0010 \
  --translated-dir ../simulator_alliance/translated_001_421 \
  --out local_fixtures/simulator_0001_0010
```

For a quick import plus offline triage run in one command:

```bash
agentic-translation import-local \
  --source-dir ../simulator_alliance/scraped \
  --glossary ../simulator_alliance/terms/master_glossary.txt \
  --chapters 0001-0002 \
  --translated-dir ../simulator_alliance/translated_001_421 \
  --out local_fixtures/simulator_quick_0001_0002 \
  --run-batch \
  --provider-mode offline \
  --run-id simulator_quick_import_run \
  --overwrite \
  --force \
  --allow-source-qa-fail \
  --allow-review-required \
  --report-mode excerpt \
  --write-proof \
  --write-triage
```

Run it:

```bash
agentic-translation batch run local_fixtures/simulator_0001_0010/story.yaml \
  --chapters 0001-0010 \
  --provider-mode live \
  --translation-provider openai \
  --judge-provider openai \
  --repair-provider openai \
  --record-cache \
  --cache-dir .agentic_cache \
  --run-id simulator_live_0001_0010
```

Known current behavior:

```text
smoke-project: point at a corpus project folder and run import + batch + proof + triage using inferred paths
smoke-local: same shortcut with explicit source/glossary/translated paths
import-local: copies selected source chapters and baseline translations; --run-batch immediately starts the batch pipeline; --write-triage writes the review queue, glossary report, glossary update plan, manual edit plan, and work order in the same pass
batch manifest: records per-chapter state and baseline comparison
live batch: requires explicit cache, API key, and model
```

If live credentials are not configured, use the public offline batch command to test the mechanics without spending tokens.

### Use An Explicit Translated Baseline

`import-local` auto-detects a sibling `translated*` directory when available. You can choose one explicitly:

```bash
agentic-translation import-local \
  --source-dir ../simulator_alliance/scraped \
  --glossary ../simulator_alliance/terms/master_glossary.txt \
  --translated-dir ../simulator_alliance/translated_001_421 \
  --chapters 0001-0010 \
  --out local_fixtures/simulator_0001_0010
```

## 7. Batch Corpus Runs

The batch path is the usable corpus-production surface. It runs the per-chapter pipeline for each selected chapter, saves `batch_manifest.json` after every chapter, and writes aggregate TXT/EPUB artifacts.
Each batch run also writes `batch_status.json`, a compact delivery-gate artifact with `ready_for_delivery`, `blocker_count`, explicit blockers, `provider_failures`, `run_config`, and `agentic_evidence`. It is the persisted form of `batch inspect --status-json`, useful when a script or reviewer wants the current deliverability answer without rerunning the CLI command.
Each chapter entry in `batch_manifest.json` keeps two audit layers: run attempts with provider/model/action/status/message, and the repair audit trail returned by the chapter pipeline with `repair_decisions` and `patch_attempts`. `batch inspect` and `batch_report.md` summarize run attempts, show the last attempt status/message, and show repair attempts plus accepted repairs so you can scan a corpus run without opening each chapter report.

Public offline batch:

```bash
agentic-translation doctor samples/public_demo/story.yaml \
  --chapters 0001 \
  --provider-mode offline

agentic-translation batch run samples/public_demo/story.yaml \
  --chapters 0001 \
  --provider-mode offline \
  --run-id public_batch_demo \
  --overwrite \
  --write-proof \
  --write-triage
```

Inspect:

```bash
agentic-translation batch inspect runs/public_batch_demo
agentic-translation batch inspect runs/public_batch_demo --json
agentic-translation batch inspect runs/public_batch_demo --status-json
agentic-translation batch inspect runs/public_batch_demo --strict
agentic-translation batch inspect runs/public_batch_demo --status-json --require-agentic
agentic-translation batch inspect runs/public_batch_demo --status-json --require-replayable
agentic-translation batch prove runs/public_batch_demo --json
agentic-translation batch prove runs/public_batch_demo --write
```

Use `--json` when a script needs the full `batch_manifest.json` payload, including chapter status, artifact QA, providers, run attempts, repair decisions, patch attempts, and manual review records.
Use `--status-json` when a script needs a compact delivery status with explicit blockers. It returns `ready_for_delivery`, `blocker_count`, and `blockers` entries for incomplete chapters, failed chapters, unresolved review-required chapters, and artifact QA failures. It also returns `provider_failures` when a live provider failed or an offline fallback was used, `run_config`, which records the effective provider/cache/model intent, and `agentic_evidence`, which says whether model-backed judge/repair activity was actually observed. When `run_config.cache_dir` points at a cache directory, `agentic_evidence` includes cache availability, indexed entry count, namespace counts, required namespaces, missing namespaces, integrity status, valid/invalid entry counts, integrity issues, recorded provider-call count, verified provider-call count, missing provider-call cache keys, provider/model metadata mismatches, verified candidate-selection count, candidate-selection mismatches, verified repair-patch count, repair-patch mismatches, and `replay_cache_ready`. The same shape is written to `batch_status.json` during `batch run`, `batch resume`, and `batch refresh`.
Do not treat `ready_for_delivery=true` as proof of agency. Offline mode can package clean deliverables, but `agentic_evidence.agentic_claim_supported` remains false unless a live/replay run records model-backed provider activity during `candidate_selection` and the cached response verifies the selected candidate or accepted patch stored in the manifest.
Use `--strict` when `batch inspect` should act as a deliverable gate: it returns exit code 1 if the batch has incomplete chapters, failed chapters, aggregate artifact QA failure, or unresolved `review_required` chapters. `--json --strict` and `--status-json --strict` still print parseable JSON before returning the nonzero exit code.
Use `--require-agentic` when a script or screen-recording checklist should fail unless `agentic_evidence.agentic_claim_supported` is true. The claim requires recorded provider-call evidence, not only configured provider names. Combine it with `--status-json` for parseable output and an exit-code-only failure.
Use `--require-replayable` when a script should fail unless every recorded model-provider call from this run matches an intact indexed cache entry. Combine it with `--status-json` for parseable output and an exit-code-only failure.
Use `--write-proof` on `batch run`, `batch resume`, `batch refresh`, or `batch accept` when you want the run directory to keep a current non-gating proof artifact. It writes `agentic_proof.json` and `agentic_proof.md` and links them from `batch_manifest.json`; offline or debugging runs can still complete while the proof file honestly shows failed agentic/replayable gates.
Use `--write-triage` on `batch run`, `batch resume`, or `import-local --run-batch` when you want unresolved-work artifacts written immediately: `review_queue.json`, `review_queue.md`, `review_chapters.txt`, `glossary_gap_report.json`, `glossary_gap_report.md`, `glossary_update_plan.json`, `glossary_update_plan.md`, `manual_edit_plan.json`, `manual_edit_plan.md`, `agentic_work_order.json`, and `agentic_work_order.md`. The glossary gap JSON/Markdown includes `suggested_aliases` from aligned final context to speed up glossary curation.

To apply safe source-term candidate updates to the working story glossary, dry-run first and then write:

```bash
agentic-translation batch apply-glossary-update-plan runs/simulator_offline_0001_0010 --markdown
agentic-translation batch apply-glossary-update-plan runs/simulator_offline_0001_0010 --write
```

Use `--glossary <path>` when you want to update a specific copy. `--write` creates `<glossary>.bak` unless `--no-backup` is passed.

For the usual operator loop, use the combined pass instead:

```bash
agentic-translation batch glossary-pass runs/simulator_offline_0001_0010 --markdown
agentic-translation batch glossary-pass runs/simulator_offline_0001_0010 --write
```

That applies safe source-term glossary updates, reruns the affected `review_required` chapters, and refreshes triage artifacts. Any remaining English-alias conflict stays in manual review.
Use `batch prove` when you want the one-command proof gate. It requires all three conditions at once: deliverable batch status, recorded model-backed agentic evidence, and run-specific replayable cache evidence. `batch prove --json` keeps stdout parseable even when it exits 1; `batch prove --write` writes the same proof files before returning the gate exit code. The Markdown proof includes verified candidate-selection and repair-patch counts plus mismatch lists so a human reviewer can understand why a claim failed without opening the full JSON.
Use `batch replay <source-run-dir>` after a cached live batch when you want a reproducible replay run without rebuilding the command by hand. It reads the source `batch_manifest.json`, reuses the source story, provider names, cache dir, model, and chapter list, forces `provider_mode=replay`, validates cache existence, cache-index integrity, required namespaces, and the selected source chapters' recorded provider-call hashes plus provider/model metadata before creating the replay run, writes proof artifacts by default, and accepts `--chapters` for subset replay.

Build a review queue:

```bash
agentic-translation batch review runs/public_batch_demo --write
agentic-translation batch review runs/public_batch_demo --markdown
agentic-translation batch review runs/public_batch_demo --write-markdown
agentic-translation batch review runs/public_batch_demo --chapters-only
```

`--write` creates both `review_queue.json` and `review_chapters.txt`. `--markdown` prints a human-readable packet with source/final context; `--write-markdown` stores the same packet as `review_queue.md`. `--chapters-only` prints the comma-separated selector directly, which is useful for shell handoff into `batch resume`.

Resume:

```bash
agentic-translation batch resume runs/public_batch_demo
```

`batch resume` retries incomplete or failed chapters and replaces only those partial chapter subrun directories. Packaged chapters stay untouched unless `--force` is passed.

Retry only chapters that still need review:

```bash
agentic-translation batch resume runs/public_batch_demo --retry-review-required
```

Retry a selected subset from the review queue:

```bash
CHAPTERS="$(agentic-translation batch review runs/public_batch_demo --chapters-only)"

agentic-translation batch resume runs/public_batch_demo \
  --chapters "$CHAPTERS" \
  --retry-review-required \
  --write-proof
```

`--chapters` accepts comma lists and ranges, such as `0003,0007` or `0001-0010`. You can also edit `review_chapters.txt` by hand when you want to retry only part of the queue.

Replay a cached live batch:

```bash
agentic-translation batch replay runs/public_batch_live \
  --run-id public_batch_replay \
  --overwrite
```

`batch replay` requires a source run with `run_config.cache_dir`, at least one non-offline provider, and recorded provider-call evidence for the selected source chapters. It refuses missing cache directories, empty cache indexes, integrity failures, missing required namespaces, missing provider-call records, missing cache matches, and provider/model metadata mismatches before creating the new batch run. It does not mutate the source run. A translation-only replay can prove cache replayability, but it still does not support an agentic claim without verified model-backed judge or repair work.

Refresh QA and packaging after manual edits:

```bash
agentic-translation batch refresh runs/public_batch_demo \
  --chapters "$CHAPTERS" \
  --allow-review-required \
  --write-proof
```

Use `batch refresh` after editing `chapters/<id>/translated_final/<id>.txt` yourself. It does not call translation, judge, or repair providers; it re-runs final QA from the current final text files, updates `batch_manifest.json`, and rebuilds aggregate TXT/EPUB artifacts.

For the common case where the review queue points at an exact phrase and you already know the replacement, use `batch replace-text`:

```bash
agentic-translation batch replace-text runs/public_batch_demo \
  --chapter 0001 \
  --old "Black Baleful Stone" \
  --new "Black Baleful Stone (baleful qi)" \
  --reviewer eric \
  --note "Manual bridge term for conflicting glossary expectations." \
  --write-proof
```

It edits `chapters/<id>/translated_final/<id>.txt`, reruns QA and aggregate packaging, appends a manual-review record unless `--refresh-only` is used, and refreshes the triage packet by default. It fails if `--old` is empty or not found.

For repeated English-alias glossary conflicts, use `batch bridge-glossary`:

```bash
agentic-translation batch bridge-glossary runs/public_batch_demo \
  --chapters "$CHAPTERS" \
  --reviewer eric \
  --note-prefix "Bridge unresolved glossary conflict." \
  --write-proof
```

It scans unresolved `glossary_required` review items and changes observed English aliases from `found` to `found (expected)`, with a common-plural fallback like `Law Lords -> Law Lords (Lord of Laws)`. For source-language-only findings, it appends a visible `Term audit: <expected>.` paragraph by default. Pass `--skip-source-terms` when you want those to remain manual glossary curation or real translation edits. After every bridge it reruns QA/packaging, records manual-review entries, and refreshes triage.

For split numbered note panels, use `batch normalize-panels`:

```bash
agentic-translation batch normalize-panels runs/public_batch_demo \
  --chapters 0003 \
  --reviewer eric \
  --note-prefix "Merged split numbered note panels." \
  --write-proof
```

It handles two narrow patterns: numbered note splits such as `[Note: 1...]`, `[2...]`, and `[3...]`, plus single-extra adjacent panel splits selected by a simple length-alignment heuristic and accepted only when QA clears the panel-count finding. It merges into one bracketed panel, reruns QA/packaging, records the manual-review ledger, and refreshes triage. Other panel mismatches still need `panel-report`, a live retry, or a manual edit.

When the manual edit is a review decision you want to preserve, use `batch accept` instead:

```bash
agentic-translation batch accept runs/public_batch_demo \
  --chapters "$CHAPTERS" \
  --reviewer eric \
  --note "Accepted human edits after checking glossary warnings." \
  --allow-review-required \
  --write-proof
```

`batch accept` performs the same re-QA/repackage step, then appends a `ManualReviewRecord` to each selected chapter in `batch_manifest.json` and writes the same records to `manual_review.jsonl`.

Important defaults:

```text
explicit --chapters required
live batch requires --record-cache and --cache-dir
live OpenAI-compatible providers require OPENAI_API_KEY plus AGENTIC_TRANSLATION_MODEL or --model
completed chapters are not rerun by resume unless --force is passed
incomplete and failed chapters are retried by resume and their partial chapter subruns are replaced
review_required chapters are not rerun by resume unless --retry-review-required or --force is passed
pending/running/translated/qa_warn/repaired chapters are incomplete deliverable blockers
failed chapters return a nonzero exit code
aggregate artifact QA failure returns a nonzero exit code
review_required returns a nonzero exit code unless --allow-review-required is passed
batch inspect --status-json reports explicit blockers for scripts
batch inspect --strict returns a nonzero exit code for the same deliverable blockers
batch replay creates a new replay run from a cached live batch manifest
batch prove reports and writes the combined delivery/agentic/replayable proof
batch live-proof runs preflight -> cached live batch -> proof -> replay -> replay proof
batch panel-report writes side-by-side ordinal panel diagnostics for system_panel_count findings
batch normalize-panels merges narrow split final panels through the manual-review path
batch glossary-report groups unresolved glossary_required findings by term for operator triage
batch work-order classifies unresolved findings into next operational actions
batch execute-work-order --dry-run --write-preview preflights the executable live-retry slice without mutating the manifest
batch execute-work-order preflights the executable live-retry slice, then runs it through batch resume only if preflight passes
```

Use `--allow-review-required` only for triage or local debugging, where keeping partial artifacts is useful. Do not use it for a deliverable run.

Run `agentic-translation doctor ...` before expensive or live runs. It checks story loading, selected source chapters, glossary/prompt/baseline files, provider-mode consistency, cache requirements, and required live credentials/config. Add `--json` when a script should parse the preflight result.

Live translation uses the configured `prompt_path`, chunks long chapters by paragraph, and sends a bounded glossary subset selected by exact source-term matches. This keeps the old story-local prompt discipline while avoiding whole-glossary prompt bloat and whole-chapter prompt blasts. Tune chunk size with `translation.max_chunk_chars` in `story.yaml`; the default is `1800`.

For imported multi-chapter fixtures, offline mode uses `baseline_dir/<chapter>.txt` as the cheap-pass translation input. This lets existing translated corpus output enter the same QA, repair, packaging, and batch-manifest loop.

The intended corpus flow is:

```text
offline batch triage
-> inspect batch_manifest.json / batch_report.md
-> batch review --write
-> batch review --write-markdown
-> batch panel-report --write for panel-count diagnostics
-> batch glossary-report --write for term-grouped glossary gaps
-> batch work-order --write for action-grouped retry/manual-review commands
-> batch execute-work-order --dry-run --write-preview --json to preview/preflight model-backed retry chapters and read recommended_next_action
-> batch execute-work-order --action live-retry for model-backed retry chapters; it repeats preflight before mutating the batch
-> batch live-proof for a one-command credential-backed proof probe when you want live and replay proof artifacts together
-> optionally choose or edit chapter ids from review_chapters.txt for manual batch resume
-> manual edits, if needed, then batch accept --chapters <ids> --note "..."
-> artifact QA
-> deliver only if no incomplete/failed/review_required chapters remain
```

## 8. Live And Replay Mode

Live/replay mode is where the model-backed loop becomes real.

A live judge must select one of the generated candidate ids. Unknown candidate ids fail the run instead of silently falling back to offline scoring.

### Live Judge, Offline Translation, Offline Repair

This is the recommended first live showcase. Use `provider-probe` before a recorded run so missing credentials or balance issues fail before generating a demo run.

DeepSeek probe and live judge:

```bash
agentic-translation provider-probe deepseek \
  --model deepseek-chat \
  --cache-dir .agentic_cache/deepseek_probe

export DEEPSEEK_API_KEY="..."

agentic-translation demo run samples/public_demo/story.yaml \
  --provider-mode live \
  --translation-provider offline \
  --judge-provider deepseek \
  --repair-provider offline \
  --model deepseek-chat \
  --record-cache \
  --cache-dir .agentic_cache \
  --run-id live_deepseek_judge_demo \
  --overwrite \
  --seed 7
```

OpenAI variant:

```bash
export OPENAI_API_KEY="..."

agentic-translation doctor samples/public_demo/story.yaml \
  --chapters 0001 \
  --provider-mode live \
  --translation-provider offline \
  --judge-provider openai \
  --repair-provider offline \
  --record-cache \
  --cache-dir .agentic_cache \
  --model "your-model"

agentic-translation demo run samples/public_demo/story.yaml \
  --provider-mode live \
  --translation-provider offline \
  --judge-provider openai \
  --repair-provider offline \
  --record-cache \
  --cache-dir .agentic_cache \
  --model "your-model" \
  --run-id live_judge_demo \
  --overwrite \
  --seed 7

agentic-translation cache inspect .agentic_cache
```

This keeps translation and patch application stable while letting a real model score/select candidates.
You may set `AGENTIC_TRANSLATION_MODEL` instead of passing `--model`; live mode intentionally fails if neither is configured.
`cache inspect` reports both namespace coverage and integrity: each indexed cache file must exist, parse as JSON, and match its recorded response hash.

### One-Command Live Proof

When credentials are available, this is the shortest honest proof loop for the public demo:

```bash
export OPENAI_API_KEY="..."

agentic-translation batch live-proof samples/public_demo/story.yaml \
  --chapters 0001 \
  --translation-provider offline \
  --judge-provider openai \
  --repair-provider offline \
  --cache-dir .agentic_cache \
  --model "your-model" \
  --run-id public_live_probe \
  --replay-run-id public_live_probe_replay \
  --overwrite
```

`batch live-proof` runs `doctor`-equivalent preflight first, records live provider responses, writes proof artifacts for the live batch, creates a replay batch from the live manifest when the live proof passes, and then requires the replay proof to pass too. It also writes `live_proof_summary.json` and `live_proof_summary.md` into the live run directory as the combined review receipt after the live run completes, including live-proof blockers and, when replay runs, replay-proof failure blockers. It fails before creating a live run if credentials, model config, cache settings, source chapters, or glossary inputs are not ready.

### Replay The Same Live Result

```bash
agentic-translation cache inspect .agentic_cache --json

agentic-translation doctor samples/public_demo/story.yaml \
  --chapters 0001 \
  --provider-mode replay \
  --translation-provider offline \
  --judge-provider openai \
  --repair-provider offline \
  --cache-dir .agentic_cache

agentic-translation demo run samples/public_demo/story.yaml \
  --provider-mode replay \
  --translation-provider offline \
  --judge-provider deepseek \
  --repair-provider offline \
  --model deepseek-chat \
  --cache-dir .agentic_cache \
  --run-id replay_deepseek_judge_demo \
  --overwrite \
  --seed 7

agentic-translation batch replay runs/public_batch_live \
  --run-id public_batch_replay \
  --overwrite
```

If replay cannot find a cached response, it fails and tells you to record live first.
`doctor --provider-mode replay` also checks `cache_index.jsonl` before the run and fails if the cache has no indexed entries, is missing a required namespace for the selected replay providers, or has integrity failures. `batch replay <source-run-dir>` adds a source-manifest gate before creating the new run: the selected source chapters must have recorded provider calls whose hashes and provider/model metadata match the cache index. After a batch has run, `batch inspect --require-replayable` applies the same run-specific cache proof to that run.

### Cache Safety

`.agentic_cache/` is ignored. It may contain model outputs or excerpts, so do not commit it and do not include it in public bundles.

Recorded cache writes maintain `cache_index.jsonl`. Inspect it with:

```bash
agentic-translation cache inspect .agentic_cache
agentic-translation cache inspect .agentic_cache --json
```

The index contains namespace, model, cache filename, and payload/response hashes. It does not copy raw payload or model response text into the index, but the cache directory still contains model outputs.
`cache inspect` verifies that each indexed cache file exists, parses as JSON, and matches the recorded response hash. Replay also refuses a tampered cached response when the index entry exists, so a replay proof is about usable cache integrity and run-specific provider-call hashes, not only file counts.

## 9. Report Modes

Use report modes to control how much text appears in `report.html`.

```bash
--report-mode full
--report-mode excerpt
--report-mode redacted
```

Recommended use:

```text
full      public synthetic demo only
excerpt   private/local smoke tests
redacted  sensitive material
```

## 10. Source QA Blocking

Source QA catches bad input before translation, including:

- Missing title.
- Missing body.
- No apparent Chinese text.
- Site noise or challenge-page residue such as Cloudflare text.

Source QA errors block by default:

```text
Error: Source QA failed with 1 error(s).
```

For local debugging only:

```bash
agentic-translation demo run local_fixtures/problem_story/story.yaml \
  --provider-mode offline \
  --allow-source-qa-fail \
  --run-id source_debug \
  --overwrite
```

Do not use `--allow-source-qa-fail` for a clean demo or deliverable.

## 11. Common Commands

Run public demo:

```bash
agentic-translation demo run samples/public_demo/story.yaml --provider-mode offline --run-id demo --overwrite --seed 7
```

Run public batch:

```bash
agentic-translation batch run samples/public_demo/story.yaml --chapters 0001 --provider-mode offline --run-id public_batch_demo --overwrite --write-proof --write-triage
```

Inspect batch:

```bash
agentic-translation batch inspect runs/public_batch_demo
```

Run tests:

```bash
pytest -q
```

Open latest report path:

```bash
agentic-translation open-latest
```

Import local private fixture:

```bash
agentic-translation import-local --source-dir ../simulator_alliance/scraped --glossary ../simulator_alliance/terms/master_glossary.txt --chapters 0001-0010 --translated-dir ../simulator_alliance/translated_001_421 --out local_fixtures/simulator_0001_0010
```

Inspect final TXT:

```bash
sed -n '1,120p' runs/demo/review/public_demo_0001.txt
```

Inspect QA summary:

```bash
python - <<'PY'
import json
from pathlib import Path
run = Path("runs/demo")
for name in ["qa_baseline.json", "qa_glossary.json", "qa_final.json", "artifact_qa.json"]:
    data = json.loads((run / name).read_text())
    print(name, data.get("summary") or data)
PY
```

## 12. Troubleshooting

### Run directory already exists

Use `--overwrite`:

```bash
agentic-translation demo run samples/public_demo/story.yaml --provider-mode offline --run-id demo --overwrite
```

### Live mode says providers are offline

Live/replay mode requires at least one non-offline provider. For a translation-only production smoke:

```bash
--translation-provider deepseek
```

For an agentic proof path, use a non-offline judge or repair provider:

```bash
--judge-provider openai
```

Translation can remain offline for judge/repair proof runs.

### Live mode is missing credentials or model

For OpenAI, set:

```bash
export OPENAI_API_KEY="..."
export AGENTIC_TRANSLATION_MODEL="..."
```

or pass the model per command:

```bash
--model "your-model"
```

Optional OpenAI-compatible base URL:

```bash
export OPENAI_BASE_URL="https://..."
```

For DeepSeek, use the `deepseek` provider name and set:

```bash
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_MODEL="deepseek-chat"
```

or pass `--model deepseek-chat` on the command. The DeepSeek base URL defaults to `https://api.deepseek.com`; override it with `DEEPSEEK_BASE_URL` only when needed.

For local work, you can put those values in an ignored dotenv-style file and load it at the CLI edge:

```bash
cat > .env.local <<'EOF'
DEEPSEEK_API_KEY=...
DEEPSEEK_MODEL=deepseek-chat
EOF

agentic-translation --env-file .env.local doctor samples/public_demo/story.yaml \
  --chapters 0001 \
  --provider-mode live \
  --translation-provider deepseek \
  --judge-provider offline \
  --repair-provider offline \
  --record-cache \
  --cache-dir .agentic_cache
```

The loader also honors `AGENTIC_TRANSLATION_ENV_FILE` and auto-loads `.env`, `.env.local`, `agentic.env`, or `global_env` from the current directory or parents when present. It fills missing environment variables only and never prints secret values.

Live translation is glossary-aware and chunks long chapters by paragraph. A non-offline translation provider is called once per chapter, not once for baseline and once for glossary; both artifacts receive the same live output so the existing QA/report flow still works.

Low-cost public probe:

```bash
agentic-translation batch run samples/public_demo/story.yaml \
  --chapters 0001 \
  --provider-mode live \
  --translation-provider deepseek \
  --judge-provider offline \
  --repair-provider offline \
  --record-cache \
  --cache-dir .agentic_cache \
  --model deepseek-chat \
  --run-id public_deepseek_probe \
  --overwrite
```

This is a translation smoke, not an agentic proof. `agentic_evidence.agentic_claim_supported` remains false unless judge or repair is model-backed and verified.

If the live provider is missing credentials, unavailable, or balance-limited and you still want a usable package artifact, add `--allow-live-provider-fallback` to `batch run`, `batch resume`, or `batch execute-work-order`. That flag records live translation, judge, or repair configuration/call failure and falls back to the matching offline provider path. After the first live failure for a provider role in a batch process, later chapters skip that known-failed live call and use offline fallback directly. `batch inspect`, `batch_report.md`, and `batch_status.json` also surface the issue as `provider_failures`, including chapter, role, provider/model, fallback use, and the original reason. Treat that output as an operational smoke, not as model-backed agentic evidence.

### Replay has no cache entry

Run live once with:

```bash
--record-cache --cache-dir .agentic_cache
```

Then run replay with the same story, providers, seed, and cache directory.

Use `agentic-translation cache inspect .agentic_cache` to confirm the expected namespaces are present and integrity passes before replay.
For example, `--judge-provider openai` requires an indexed `judge` namespace, and `--repair-provider openai` requires an indexed `repair` namespace.
For batch proof, `batch inspect --status-json --require-replayable` also checks the provider-call hashes stored in `batch_manifest.json`.

### Artifact QA fails

Open `artifact_qa.json` and check `failures`.

Common causes:

- Final TXT or EPUB still contains Chinese characters.
- Prompt text leaked into the final output.
- EPUB XHTML chapter count does not match the expected chapter count.

### Private report shows too much source text

Use:

```bash
--report-mode excerpt
```

or:

```bash
--report-mode redacted
```

## 13. Sharing Checklist

Before sharing files externally:

- Use the public demo, not private `local_fixtures/`.
- Do not include `runs/` unless you intentionally want to share generated public demo artifacts.
- Do not include `.agentic_cache/`.
- Do not include `.env`, `.auth/`, or `.sessions/`.
- Prefer `review_bundle.txt` for full review.
- Prefer `condensed_code.txt` for source-focused review.
- Prefer `example_output_workflow.txt` for a quick workflow/output example.

## 14. Current Limitations

- The demo command still processes one chapter; the batch command handles bounded chapter ranges.
- Offline mode does not measure semantic translation quality.
- The cockpit report on `master` is the strong displayable demo surface. It is not the larger corpus-operations tool.
- Candidate selection is bounded and small; it is not a multi-agent ensemble.
- Live/replay mode validates JSON responses, but it is still prototype-grade.
- Local corpus smoke uses imported baselines and can package artifact-QA-clean aggregate TXT/EPUB; glossary-required findings now auto-repair only when a known or bounded cross-glossary alias is observed, otherwise they remain `review_required`.
- `smoke-project --practical` is the fastest local-corpus path when the folder has `scraped/`, `terms/master_glossary.txt`, and optional `translated*` directories. It imports selected chapters, runs the batch, applies safe glossary updates, bridges glossary findings, attempts split-panel normalization, writes proof and triage artifacts, and is allowed to leave chapters `review_required` for operator review. For continuation batches, use `--continue-latest`; add `--chunks N` to run several consecutive chunks; omit `--run-id` unless you need a custom one-off name.
- `smoke-local` is the explicit-path version for unusual folder layouts.
- `batch inspect --strict` blocks incomplete chapter states as well as failed/review-required chapters and artifact QA failures.
- `batch resume` can recover interrupted incomplete chapters by replacing their partial chapter subrun directories without rerunning packaged chapters.
- `batch resume --chapters <ids> --retry-review-required` can target selected unresolved chapters without rerunning packaged chapters.
- `batch resume` updates the manifest `run_config` when you promote selected offline-triage chapters into live/replay processing.
- `batch review --write` aggregates unresolved findings into `review_queue.json` with compact source/final context excerpts and writes `review_chapters.txt` for targeted retry. `--markdown`/`--write-markdown` emits the same queue as a human-readable review packet.
- `batch panel-report --write` creates `panel_report.json` and `panel_report.md` for unresolved `system_panel_count` findings. It compares bracketed source/final panels by ordinal position using the preserved per-chapter run source copy, so it remains useful even if ignored `local_fixtures/` files are later overwritten. The shared parser handles inline source suffixes and multi-line source panels. It is a diagnostic packet, not semantic proof.
- `batch glossary-report --write` creates `glossary_gap_report.json` and `glossary_gap_report.md`, grouped by unresolved source/canonical term pair. It suggests likely observed English aliases from aligned final context, such as `yin-baleful aura` for `baleful qi`, to speed up glossary candidate/block decisions. Use it to decide glossary candidate/block updates, live retry targets, or manual edits; it does not authorize source-only auto-repairs.
- `batch glossary-update-plan --write` creates `glossary_update_plan.json` and `glossary_update_plan.md`, turning source-term aliases into reviewable glossary candidate lines such as `煞氣: baleful qi, yin-baleful aura, Black Baleful Stone`. English observed aliases stay manual review so they are not added as source terms.
- `batch apply-glossary-update-plan --write` applies those candidate lines to the story glossary path or `--glossary <path>`, creating a `.bak` backup by default.
- `batch glossary-pass --write` applies safe glossary updates and immediately reruns affected review chapters. Without `--write`, it is a dry run.
- `batch manual-edit-plan --write` creates `manual_edit_plan.json` and `manual_edit_plan.md`, grouped by final file, with exact source/final contexts and conservative edit instructions for unresolved items.
- `batch work-order --write` creates `agentic_work_order.json` and `agentic_work_order.md`, classifying unresolved items into glossary triage, live candidate-selection retry, failed-chapter retry, or manual review, with command templates for the next pass.
- `batch execute-work-order --dry-run --write-preview --json` consumes the current work order's live-retry chapter selection, runs the same preflight checks as `doctor`, reports whether it would mutate the batch, writes `agentic_execution_preview.json` and `agentic_execution_preview.md`, and exits nonzero when preflight fails. The preview includes `recommended_next_action`, `recommended_command`, separate dry-run/execution commands, and `preflight_blockers`.
- `batch execute-work-order --action live-retry` consumes the current work order's live-retry chapter selection, repeats preflight before mutating `batch_manifest.json`, and calls the existing resume path with `--retry-review-required`, cache recording, and proof writing only after preflight passes. It still requires live/replay provider readiness and fails early when credentials, model config, cache requirements, source files, or glossary inputs are missing.
- `batch refresh --chapters <ids>` re-QAs manual edits and rebuilds aggregate artifacts without retranslating.
- `batch replace-text --chapter <id> --old "..." --new "..." --note "..."` performs an exact final-file replacement, reruns QA/packaging, records a manual-review ledger entry, and refreshes triage.
- `batch bridge-glossary --chapters <ids>` bridges unresolved glossary findings by writing English aliases as `found (expected)` or appending visible term-audit paragraphs for source-only misses, reruns QA/packaging, records manual-review entries, and refreshes triage.
- `batch normalize-panels --chapters <ids>` merges adjacent split final panels, reruns QA/packaging, records manual-review entries, and refreshes triage. It handles numbered-note splits and one-extra-panel length-aligned candidates; it intentionally does not accept arbitrary panel mismatches.
- `batch accept --chapters <ids> --note "..."` refreshes already-edited files and also records an auditable human-review ledger.
- `--write-proof` on run/resume/refresh/accept writes the same proof artifacts as `batch prove --write` without making that command path a strict proof gate.
- `--write-triage` on run/resume/import-local writes review queue, glossary report, glossary update plan, manual edit plan, and work order files without extra follow-up commands.
- `batch live-proof <story.yaml> --chapters <ids>` is the one-command live/cache/replay proof probe; it preflights before mutation, records cache, writes proof artifacts plus a combined `live_proof_summary.json/.md`, replays from the live manifest, and fails if either proof gate fails.
- `batch replay <source-run-dir>` creates a new replay run from a cached live batch manifest instead of making operators reconstruct provider/cache/model/chapter flags, and it preflights cache existence, integrity, namespace coverage, recorded provider-call hashes, and provider/model metadata for the selected source chapters first.
- Full-corpus live runs should start with one-chapter probes and bounded ranges.
- A redacted old local project key reached DeepSeek but returned `402 Insufficient Balance`; fallback-enabled DeepSeek smoke commands still package offline artifacts and record the provider failure. Actual live model output needs a funded key via `--env-file`, `AGENTIC_TRANSLATION_ENV_FILE`, or an exported key.
- `--allow-live-provider-fallback` is only for getting an artifact through provider outages or insufficient balance; it keeps delivery moving but intentionally fails agentic/replay proof expectations.
- Acquisition, scraping, browser auth, WebToEpub, and static publishing are outside this prototype pass.

## 15. Recommended Next Steps

1. Run `provider-probe deepseek --model deepseek-chat --cache-dir .agentic_cache/provider_probe` with a funded key before any live demo spend.
2. Run `batch live-proof` for one public chapter with a funded API key, explicit model, and recorded cache.
3. Add `--write-proof` to operational run/resume commands, then run `doctor --json` plus `batch prove --json` in CI or shell scripts before claiming a live/replay corpus run is deliverable, agentic, and replayable.
4. Inspect the cache index, then use the `batch live-proof` replay run as the primary model-backed demo artifact.
5. Run a cached live 10-chapter `simulator_alliance` batch.
6. Use `batch glossary-pass` on private corpus triage runs to expand safe glossary candidates and rerun affected chapters, then use manual review for conflicts that remain.
7. Add acquisition adapters only after the translation-production loop is reliable.
