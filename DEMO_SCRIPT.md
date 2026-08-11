# Two-minute Harness v3 demo

This is the fastest end-to-end walkthrough of the package. It supersedes the
older **90-second agentic replay demo** while leaving that replay path intact.
Run every command from the directory containing `pyproject.toml`.

## 0:00 — Start a no-network golden run

Say:

> This is a bounded, QA-gated translation-repair harness. The provider in this
> demo is a deterministic synthetic native-contract fixture: no API key, live
> provider, replay cache, or network call is involved.

After installation (see [README.md](README.md) or [USER_GUIDE.md](USER_GUIDE.md)), run:

```bash
python -m agentic_translation harness golden --runs-dir runs --pause-for-approval --overwrite
```

The provider-native code path is covered by injected-client tests. No checked-in artifact claims a real network provider-native call. The command ends at
`Status: awaiting_approval`. Open the report in a browser:

```text
runs/agentic_harness_v3_demo/report.html
```

## 0:15 — Show search and dynamic exposure

In the report, expand **`02 · Persisted session receipt`**. Point at the first
`model_requested` event to show the initial exposure. The exact bootstrap
control/discovery set is `escalate`, `finish`, `get_qa_findings`, and
`tools.search`. Then move to the primary numbered action rail and point at
`tools.search` as the first action. The synthetic fixture emits a
provider-native-shaped `tools_search` call; the harness normalizes it to the
logical `tools.search` action, validates it through the typed registry, and
then exposes the rest of the trusted tools.

Say:

> The model can search for a capability, but it cannot invent one. Exposure is
> dynamic and recorded before every request.

The next action is `resolve_terminology`. The deterministic two-voter fixture
records bounded consensus for `道心 → Dao Heart`; the master glossary is still
unchanged.

## 0:35 — Show the verifier-controlled patch

Scroll to the chronology and call out both `submit_patch` entries:

1. **Rejected:** the proposed old text is absent, so the executor fails closed
   before changing the working copy.
2. **Accepted:** `Heart of Dao guarded 道心.` becomes `Dao Heart guarded the
   mountain gate.` and deterministic QA moves from findings to zero.

Say:

> The model proposes a patch; the RepairToolExecutor applies it only when the
> old text is unique and QA strictly improves. This is the verifier-controlled
> patch boundary.

## 0:55 — Pause at policy PENDING

Point at `promote_glossary_term`. Its policy decision is
`require_approval`, and the report marks the action **PENDING**. The snapshot
is `awaiting_approval`; no canonical glossary write has happened yet. The
reviewed proposal is exactly `道心 → Dao Heart`.

## 1:05 — Approve the exact proposal

Run:

```bash
python -m agentic_translation harness resume runs/agentic_harness_v3_demo --approve --reviewer demo-reviewer --note "Approve the reviewed run-local glossary promotion."
```

Refresh `report.html`. Point out the approval receipt, the applied delta, and
the fact that the run-local glossary was written once. The resumed session
then calls `finish`.

## 1:25 — Show verified evidence

Say:

> The final status is `verified`, with zero deterministic QA findings. The
> session events, snapshot, typed episode, glossary, final text, and report are
> durable evidence. The golden artifacts are inspectable and the paused session is resumable. This
> local native-contract fixture is not replay-cache evidence.

Show `session_events.jsonl` or `session_snapshot.json` briefly. The evidence
also records the native-function protocol, exposed tools, policy decisions,
rejected and accepted patches, reviewer identity, and the approval note.

## 1:35 — Compare the contract bench

Run:

```bash
python -m agentic_translation harness bench --suite samples/harness_eval/v3_cases.json --out runs/harness_eval --overwrite
```

Say:

> This bench is three deterministic cases across three variants: prompt JSON
> with all tools, native function calls with all tools, and native calls with
> dynamic exposure. The current fixture expectation is 9/9 passed. It compares
> first-request exposure and schema size; it is not a claim about exact bytes,
> latency, or model quality.

## 1:55 — Summarize the evidence

> This proves a bounded QA-gated agent harness for translation repair and makes
> its decisions reviewable. It does not prove literary translation quality,
> production scale, or that a live provider will behave this way.

## Optional one-command path

For a quick screen recording with no pause/resume handoff:

```bash
python -m agentic_translation harness golden --runs-dir runs --auto-approve --overwrite
```

The resulting report has the same native contract, policy, patch, approval,
and verified evidence; it simply applies the scripted approval in one process.

## Keep the legacy replay story available

If a reviewer asks about the earlier cache-only flow, run the terminology replay
from its fixture story:

```bash
python -m agentic_translation demo-repair \
  --story samples/agentic_terminology_demo/story.yaml \
  --chapter 0001 \
  --provider-mode replay \
  --term-consensus \
  --openai-term-model fixture-openai-term \
  --deepseek-term-model fixture-deepseek-term \
  --term-evaluator openai \
  --runs-dir runs \
  --overwrite
```

That replay remains useful for showing two-model terminology arbitration and
cache-only determinism. Keep the scope line intact: a green mechanical gate
does not prove literary translation quality.
