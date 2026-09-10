# Automatic repair

Automatic mode addresses a gap in the original Harness v3 policy: a
source-supported meaning correction could be rejected when deterministic QA
was already at 100. The new policy admits that bounded edit only when the
candidate text changes, QA does not decrease, and the edit introduces no new
finding identities.

## Completion contract

Automatic completion requires both clear deterministic QA and a completed,
nonblocking fidelity review of the current draft. The runtime hashes the exact
draft bytes with SHA-256. A review of an earlier draft cannot authorize a later
one; every accepted edit invalidates the previous review.

Specialist agents are deliberately read-only. They can retrieve bounded source
and draft paragraphs, inspect the episode glossary, and return structured
findings, proposed edits, and term suggestions. The coordinator alone may
submit patches, select a term, or request persistent glossary promotion.

Persistent promotion remains a human approval boundary. Instructions, review
requirements, nonregressing-patch policy, and delegation limit are saved with
the session and checked again on resume.

## Provider profile

The named DeepSeek V4 Flash profile uses strict JSON actions, thinking disabled,
temperature zero, and a 2,048-token output limit. The validator accepts either
the supported flat or nested action shape and rejects mixed or ambiguous
payloads. These are tested settings for that profile, not a universal provider
contract.

## Evidence and limits

Historical abbreviated checks corrected ten clear-source meaning cases. There
were 15 attempts across 14 cases, including one initial punctuation failure
and one disclosed successful retry. Fourteen completed attempts carried
current-text reviews, 12 QA-neutral patches were accepted, and one live
semantic repair replayed with identical text and 8/8 cache hits.

Those aggregates describe earlier private runs; they are not measurements of
the public synthetic fixture and do not establish an overall quality or cost
advantage. An intermediate specialist mode delivered 11/36 faithful outputs in
a separate 252-trial study, while direct DeepSeek and Terra each delivered
36/36. The study did not prove harness superiority.

Known limitations remain:

- one correct draft gained an unnecessary gendered pronoun;
- one explicitly illegible source gap completed without the expected hold;
- a clean model review is not proof of correctness; and
- bounded paragraph access is not exhaustive long-document review.

See [DATA_NOTICE.md](../DATA_NOTICE.md) for the public-data boundary and
[SESSION_IDENTITY.md](SESSION_IDENTITY.md) for resume guarantees.
