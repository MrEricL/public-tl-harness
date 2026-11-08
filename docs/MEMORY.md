# How the harness remembers

The harness keeps a full run record, a source-grounded memory tape, the
translator's glossary, and the small context supplied to one passage. These
correspond to the transcript, memory-note, representation, and injected-context
layers described in Anthropic's article “The components of agent memory.”

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/readme/memory-flow-dark.svg">
  <img alt="Chapter n on the left, chapter n+1 on the right, and four layers between them, bottom to top. Run record: every call, cost and edit, replayable; Python adds to it and nothing edits it. Story memory: names and facts the model proposes from chapter n, kept only if their quote is in the Chinese; one snapshot per chapter, never rewritten; by chapter 25 it held 57,000–81,000 characters, 378–523 names and about 290 facts per novel. Translator's glossary: the translator's own names, listed first; they win over memory's. Passage context, highlighted: the few lines one passage is given, with memory's part at most 3,000 characters: 392 characters of memory per passage with names only, the default, or 2,862 when facts were included. Arrows: extract from chapter n into memory, select from memory and the glossary into the passage context, write from the context into chapter n+1, record from chapter n into the run record." src="../assets/readme/memory-flow-light.svg" width="100%">
</picture>

## Four layers and their owners

| Layer | Stored material | Who changes it |
| --- | --- | --- |
| Run record | Requests, responses, usage, edits, and snapshots | Runtime code appends records |
| Story memory | Names and facts with source quotations and availability positions | A model proposes entries; code validates and records them |
| Translator's glossary | Approved English names | The translator |
| Passage context | A selected glossary and memory pack | Code builds it under the configured budget |

A model never directly overwrites an accepted chapter snapshot. Each saved
snapshot records its predecessor; later chapters build on earlier states.
The translator's glossary wins when a memory name conflicts with an approved
name. The source remains the evidence for facts.

## Selecting context

Each passage gets up to 40 translator glossary lines, then a memory pack of at
most 3,000 characters built by `context_for_segment`:

1. Relevant translator glossary lines come first.
2. Matching memory names follow, for example `TERM 青云宗 -> Azure Cloud Sect`.
3. Facts follow only when the recorded policy includes them.

The `drop_conflicting` overlap setting removes memory names that disagree with overlapping glossary
choices while preserving compatible compounds. Entries are eligible only at
positions where their source evidence is available. Memory does not look at
later chapters. Context receipts record what reached the passage and why
other candidates were omitted.

New demo and named book-profile runs use a names-only pack. Facts are still
extracted and stored, so this changes injected context rather than erasing
memory. Historical replay restores the run's recorded policy.

## Accepting a memory entry

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/readme/memory-rules-dark.svg">
  <img alt="Four steps. The extractor, a model, proposes names and facts, each with the sentence it came from. Python keeps an entry only if that quote is in the source, and what a character merely claims stays provisional. The translator's glossary wins over memory whenever the two disagree. Each chapter's snapshot is saved and never rewritten; the next one builds on it. Example from the demo: the sentence 她自称来自镜湖镇 (she says she is from Mirror Lake Town) gives 镜湖镇, Mirror Lake Town, a location kept as provisional because only she says so." src="../assets/readme/memory-rules-light.svg" width="100%">
</picture>

The extractor attaches an exact source quote to each proposal. Code rejects
an entry whose quote cannot be found. Statements framed as a character's claim
remain provisional. A matching quote proves where the proposal came from;
it does not independently establish a correct interpretation.

A future offline adjudication pass could merge variants or correct stale
facts with an explicit reason and citation. That “dreaming” pass is **planned,
not built**. The saved-run audit estimated it could fix about 15 of the 51
inconsistent renderings left in Stage 1; it did not measure an implemented
improvement.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/readme/memory-flow-planned-dark.svg">
  <img alt="Chapter n on the left, chapter n+1 on the right, and four layers and one planned pass between them, bottom to top. Run record: every call, cost and edit, replayable; Python adds to it and nothing edits it. Story memory: names and facts the model proposes from chapter n, kept only if their quote is in the Chinese; one snapshot per chapter, never rewritten; by chapter 25 it held 57,000–81,000 characters, 378–523 names and about 290 facts per novel. Translator's glossary: the translator's own names, listed first; they win over memory's. Passage context, highlighted: the few lines one passage is given, with memory's part at most 3,000 characters: 392 characters of memory per passage with names only, the default, or 2,862 when facts were included. Arrows: extract from chapter n into memory, select from memory and the glossary into the passage context, write from the context into chapter n+1, record from chapter n into the run record. Dreaming is planned but not built: an offline pass that would merge variant names and settle them, sized at about 15 of the 51 inconsistent renderings left after Stage 1." src="../assets/readme/memory-flow-planned-light.svg" width="100%">
</picture>

## Carrying memory into another installment

These options belong to the folder CLI:

- `--lead-in N` reads the first N selected source files into memory without
  translating them, then translates the remainder.
- `--memory-from RUN` loads an earlier folder run's saved memory before the
  new installment. The chain and source references are checked before use.

The source-only preparation and carried prefix are saved with the new run.
Replay reconstructs them from that run's own files and refuses new overrides.
Keep run folders intact; partial or modified memory chains are rejected.
Neither option is currently exposed in the workbench form or book CLI.

## What was measured

| Check | Result |
| --- | --- |
| Stage 1, 60 unseen chapters without a glossary | 22.2% → 9.0% inconsistency, 59% fewer |
| Stage 1, hidden half of glossary | 22.7% → 8.9%, 61% fewer on hidden names |
| Saved-run pack audit | Facts used 81–88% of memory characters |
| Names-only development pilot | About 2,449 → 217 memory characters per passage, with similar limited checks |
| Stage 2, 60 fresh chapters from the same novels | Names-and-facts memory: 24.7% → 11.4%, 54% fewer |
| Stage 2, names-only vs names and facts, both with 20-chapter lead-in | Passed the registered no-worse rule; about 2,862 → 392 memory characters, 86% smaller |
| Stage 2, 20-chapter vs five-chapter lead-in | Delivery rose 87.7% → 93.2%; consistency improvement was not established |
| Post-draft naming patches | Development pilots failed their adoption rules; alignment now targets supplied translator names |

[Results](RESULTS.md) explains the fixed chapter blocks, corrected delivery
measure, missing-output denominators, judged errors, and pending human
validation. These checks support naming consistency. Stage 2 tested the writer;
the names-only context now supplied to the reviewer has not been separately
evaluated.

[Try it](TRY_IT.md) · [How it works](HOW_IT_WORKS.md) · [Code map](CODE_MAP.md)
