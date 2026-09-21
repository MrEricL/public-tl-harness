# Changelog

## Published baseline — Harness v3

Starting point: [`8bf860e`](https://github.com/MrEricL/public-tl-harness/tree/8bf860e1236fe654d102fd68a3d369d3a47dfb9d).

- Typed tools, dynamic discovery, bounded repair, deterministic QA, durable
  sessions, approval-gated glossary writes, provider recording, cache-only
  replay, reports, batch workflows, and TXT/EPUB packaging.
- The original repair policy required strict mechanical QA improvement.

## Optional Jev sensing and public evidence

- Adds an optional Jev semantic-sensing path through the native Vercel AI
  Gateway evaluation route for `typesafe-ai/jev`.
- Keeps Jev off by default and supports focused or dense questions, shadow or
  advisory delivery, initial or after-edit schedules, and cache-only replay.
- Adds a public synthetic live-command example using separate
  `AI_GATEWAY_API_KEY` and `DEEPSEEK_API_KEY` environment variables. No key
  values are included.
- Documents aggregate pilot and follow-up results, including costs, coverage,
  execution failures, and review limits. These are historical summaries of a
  private development study, not a public benchmark or a claim of general Jev
  quality improvement.
- Explains the verifier fix for a nested glossary occurrence and the withdrawn
  breathing-label comparison without distributing source text, private
  translations, raw provider payloads, or private run artifacts.

See [JEV_EXTENSION.md](docs/JEV_EXTENSION.md) for the public contract and
evidence notes.

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
