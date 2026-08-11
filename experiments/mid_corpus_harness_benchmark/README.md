# Mid-Corpus Harness Benchmark

This folder contains the code and aggregate results for a five-chapter
comparison. It compares a one-shot translation with the same draft after
bounded Harness repair.

The five source chapters (`0247`–`0251`) and text-bearing artifacts derived
from them are omitted for copyright reasons. The public version keeps the
benchmark code, aggregate results, synthetic tests, and two short paraphrased
examples.

See the [data notice](../../DATA_NOTICE.md) for the full code and data boundary.

## Results

The result was mixed.

- Deterministic QA: 68 → 0 rule hits. The 68 hits included 56 punctuation
  findings, 11 glossary findings, and one mismatch in bracketed-panel counts.
- Overall blind preference: baseline 7 / Harness 5 / tie 8 across 20 decisions.
- Formatting: Harness 4 / baseline 2 / tie 14.
- Terminology: baseline 5 / Harness 1 / tie 14.
- Repair run: 22 steps, 7 accepted mutations, 0 rejected mutations, and 5
  verified chapters.

The deterministic rules all cleared. The blind evaluation did not show an
overall translation-quality improvement.

## Reproduce the public checks

From the repository root:

```bash
python -m pytest -q experiments/mid_corpus_harness_benchmark/test_benchmark.py
python experiments/mid_corpus_harness_benchmark/benchmark.py aggregate --validate-only
```

- The test command builds synthetic chapters, candidates, scorecards, packets,
  and session traces in a temporary directory.
- The validation command checks the retained aggregate and confirms that the
  private text files are absent.

Both commands run without the withheld chapters or a provider key.

## Run with private text

Full reruns use `select`, `run-harness`, `blind`, and `aggregate`. Point these
commands at a separate fixture directory. Their inputs and outputs can contain
copyrighted text.

## Files

- [`benchmark.py`](benchmark.py): selection, repair replay, blind packet
  generation, aggregation, and report rendering.
- [`test_benchmark.py`](test_benchmark.py): synthetic tests for the benchmark
  method.
- [`benchmark_config.json`](benchmark_config.json): run limits, selection
  details, model provenance, and the unexecuted efficiency matrix.
- [`results.json`](results.json): aggregate results without chapter text or
  per-chapter judge evaluations.
- [`REPORT.md`](REPORT.md): results, provenance, and limitations.
- [`PUBLIC_EXAMPLES.md`](PUBLIC_EXAMPLES.md): two paraphrased comparison
  examples.
- [`FUTURE_WORK.md`](FUTURE_WORK.md): deferred human evaluation, semantic QA,
  and efficiency work.

## Provenance

- Codex runs produced the original repair actions.
- The actions were migrated to the Harness v3 schema and run again through
  this repository.
- No single end-to-end run produced the published result.

Action JSON and session records are omitted with the text-bearing evidence, so
the public files cannot replay the original action sequence.
