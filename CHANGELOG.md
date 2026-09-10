# Changelog

## Published baseline — Harness v3

Starting point: [`8bf860e`](https://github.com/MrEricL/public-tl-harness/tree/8bf860e1236fe654d102fd68a3d369d3a47dfb9d).

- Typed tools, dynamic discovery, bounded repair, deterministic QA, durable
  sessions, approval-gated glossary writes, provider recording, cache-only
  replay, reports, batch workflows, and TXT/EPUB packaging.
- The original repair policy required strict mechanical QA improvement.

## Curated automatic-repair release

- Makes source-grounded automatic repair the default for new showcase runs.
- Allows changed text with nondecreasing QA and no new finding identities,
  while retaining exact-target, atomic-bundle, whole-document, and budget
  checks.
- Requires a completed nonblocking fidelity review matching the exact final
  draft hash; edits invalidate older stale reviews.
- Adds bounded read-only terminology and fidelity specialists while keeping
  mutations and glossary selection with the coordinator.
- Persists instructions and automatic policy across resume and rejects
  incompatible continuation arguments.
- Adds tested DeepSeek V4 Flash settings, flat/nested action validation, usage
  receipts, and a readable consolidated report.
- Adds a small offline demonstration made only from synthetic technical test
  statements, with approval/resume and replay workflows.

This release does not include private corpora, comparison fixtures, raw
experiment artifacts, provider payloads, or private workspace notes. Targeted
historical checks are summarized in [docs/AUTOMATIC_REPAIR.md](docs/AUTOMATIC_REPAIR.md);
they are not a new public benchmark run.
