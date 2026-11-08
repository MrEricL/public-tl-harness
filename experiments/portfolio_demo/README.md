# Authored workflow checks

These small examples exercise the same folder runtime as `demo.py`. They are
workflow checks, not a substitute for the historical studies in
[the results page](../../docs/RESULTS.md).

## Compare the four folder setups

```bash
python demo.py --provider openai --model YOUR_MODEL \
  --source samples/showcase/source \
  --glossary samples/showcase/terms/demo_glossary.json \
  --compare --out runs/portfolio-terms --open
```

The setups are each chapter alone, glossary only, memory plus glossary, and
memory plus glossary with review and guarded repair. The final setup starts
from the exact contextual draft. The report records specified-term adherence,
failed outputs, selected context, calls and usage. These checks measure the
supplied terminology contract, not literary quality.

## Try the seeded repair challenge

```bash
python demo.py --repair-bench --provider openai --model YOUR_MODEL \
  --out runs/portfolio-repair --open
```

`repair_cases.json` holds ten seeded errors and two clean controls in short,
independently authored Chinese passages. The model receives source, draft and
terminology, but never the reference answers or evaluator patterns. The same
trusted edit executor used by folder translation accepts or rejects changes.

The versioned checks cover narrow facts such as negation, quantities, sequence,
conditions and actor attribution. Report errors fixed and clean controls
preserved; inspect the text when a valid paraphrase misses a pattern. These
constructed starting errors do not measure natural error prevalence.

Both commands make provider calls and require a key. For local development,
[the focused runtime tests](../../tests/test_portfolio_demo.py) run the folder,
repair and record/replay paths with scripted responses and no external model.
See [Try it](../../docs/TRY_IT.md) for provider setup and replay.
