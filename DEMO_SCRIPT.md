# Automatic repair walkthrough

This short walkthrough uses synthetic technical statements. It makes no live
provider call and contains no story or private-corpus text.

## 1. Run to the approval boundary

```bash
python -m agentic_translation harness run \
  --story samples/synthetic_repair_demo/story.yaml \
  --out runs/synthetic-repair
```

Open `runs/synthetic-repair/report.html`. The first controller statement
deliberately says that readings are recorded before the valve opens, reversing the source order.
It also leaves a technical term untranslated and uses Chinese punctuation.

Point out these controls in the report:

- bounded terminology and fidelity specialists can read but cannot edit;
- the coordinator selects `采样周期 → sampling cycle`;
- `submit_patch` restores the source operation order with exact replacements;
- deterministic QA accepts the candidate without allowing new findings;
- punctuation normalization runs through the same verifier-owned boundary;
- a fresh fidelity review covers the exact corrected draft hash; and
- the glossary promotion pauses as `awaiting_approval`.

## 2. Approve and resume

```bash
python -m agentic_translation harness resume runs/synthetic-repair \
  --approve --reviewer demo
```

The second synthetic statement reuses `sampling cycle`. The completed report
shows both statements, the approval receipt, the current-text review, provider
call receipts, and the generated TXT/EPUB paths.

## 3. Replay without a network call

```bash
python -m agentic_translation harness replay runs/synthetic-repair \
  --out runs/synthetic-repair-replay
```

The replay consumes the recorded scripted requests and responses. Compare the
two `translated_final.txt` files or the consolidated TXT output; they should be
identical.

## 4. Show the original Harness v3 gate

The existing golden demonstration remains useful for dynamic tool discovery,
strict QA-improvement gating, and a separate approval flow:

```bash
python -m agentic_translation harness golden \
  --runs-dir runs --pause-for-approval --overwrite
```

The existing contract bench remains available too:

```bash
python -m agentic_translation harness bench \
  --suite samples/harness_eval/v3_cases.json \
  --out runs/harness-eval --overwrite
```

## Scope statement

This demonstration proves the runtime contract on authored synthetic input. It
does not prove literary translation quality, production scale, or universal
provider compatibility. A clean reviewer receipt is evidence about one exact
draft, not proof that the draft is correct.
