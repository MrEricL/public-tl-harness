# How the translation harness works

The harness carries source-checked story memory and a translator's glossary
between chapters. Models write drafts and propose changes. Python chooses
context, validates quotations and edits, records outcomes, and packages the
translation. An exact quote establishes provenance; it does not prove that
the extracted interpretation is correct.

## One chapter

```mermaid
flowchart LR
    source[Ordered Chinese source] --> memory[Quoted story memory]
    glossary[Translator glossary] --> context[Bounded context]
    memory --> context
    context --> writer[Model draft]
    writer --> checks[Deterministic checks]
    checks --> contextual[Contextual result]
    checks --> alignment[Translator-name alignment]
    alignment --> review[Source review and guarded repair]
    review --> report[Saved report and book]
```

The folder demo follows the review-and-repair path. The book runner's
`contextual.v1` profile stops after generation and deterministic QA;
`portfolio-review.v1` also performs term alignment and passage review.
The same configured model handles extraction, writing, and review. Optional
legacy routing is outside the normal one-provider workflow.

## What enters the prompt

The translator's relevant glossary lines come first. Memory then supplies
matching names within a 3,000-character memory budget per passage; up to 40
glossary lines go ahead of it and don't count against it.
A conflicting memory name yields to the glossary; compatible compounds can
remain. New demo and named book-profile runs use `fact_policy: terms_only`:
facts stay in the tape but do not enter this pack. Older runs replay with
the settings recorded at the time.

Entries have source positions. Retrieval respects when the quoted evidence
became available, and memory does not read later chapters. Receipts show
selected terms and facts, exact citations, retrieval reasons, glossary
precedence, and size omissions. The delivered prompt and receipts are part
of the run record. [Memory](MEMORY.md) explains ownership and persistence.

## Who changes the translation

The model proposes an exact edit or bounded rewrite. Trusted code checks the
target against the current text, tries changes on a working copy, and applies
them only if the configured QA and content guards permit them. These include
bounds, content-loss checks, and selected gloss/duplication guards. Accepted
changes are checkpointed; review coverage is tied to the current draft.
The report preserves accepted and rejected attempts and unresolved warnings.

Term alignment now receives only the translator's supplied naming targets.
Memory names can still appear in generation context and QA. Narrowing the
alignment targets does not establish that every remaining edit improves
fluency or meaning; development pilots exposed that limitation.

“Delivered” describes the workflow's recorded outcome. It is not a certificate
that all meaning, naming, and style are correct.

## Saved runs and the local workbench

`demo.py` copies selected inputs, saves configuration and model responses,
records memory and repair sessions, then writes `results.json`, HTML, and
TXT/EPUB delivery. Failed assigned chapters remain visible. A fatal provider
or budget error stops additional spending; skipped chapters are recorded.

The workbench starts and watches this same engine, requests a stop before the
next call, and presents chapter diffs, naming checks, memory receipts, and
activity. It binds to localhost and keeps entered keys in process memory.
Saved reports are served with a sandbox policy.

Folder replay reads copied source and successful cached responses instead of
making network requests. It preserves recorded settings and rejects changed
inputs or missing calls. This reproduces a completed workflow; it is not
mid-chapter crash recovery. The CLI's `--memory-from` and `--lead-in` options
support installments and source-only preparation; the workbench form and book
runner do not expose those options.

## What the evidence covers

Stage 1 isolates memory from a supplied glossary. Stage 2 replicates its
no-glossary naming result on fresh blocks and supports a smaller memory pack
within registered naming and character-error limits. Stage 2 evaluated the
writer; the new pack's effect on the reviewer remains untested. The historical
full-harness quality comparison was mixed. [Results](RESULTS.md) gives the
methods, denominators, corrections, and remaining validation.

[Try it](TRY_IT.md) · [Code map](CODE_MAP.md) · [Data notice](../DATA_NOTICE.md)
