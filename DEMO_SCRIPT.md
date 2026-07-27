# 90-second agentic replay demo

Use this script for a screen recording or a short project walkthrough.

## Setup

Run the cache-only terminology replay:

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

Open the generated report:

```bash
open runs/agentic_terminology_demo_replay/report.html
```

## Narration

### 0–15 seconds: thesis

> This is a bounded translation repair agent. The model chooses tools, but
> ordinary Python code decides whether a proposed patch is accepted.

The harness limits the action surface, validates arguments, records each
observation, and keeps final promotion behind deterministic QA.

### 15–35 seconds: model-directed action

Show the first action: `resolve_terminology`.

The agent encounters the disputed term `道心` and chooses the terminology tool.
One cached OpenAI voter proposes `Dao Heart`; one cached DeepSeek voter proposes
`Heart of Dao`. Neither voter sees the other response.

### 35–50 seconds: bounded consensus

Show the blinded evaluator result.

The harness normalizes the two proposals, detects disagreement, and asks the
configured evaluator to select from blinded candidate ids. It selects
`Dao Heart`. The decision is an episode-local override; it does not silently
rewrite the master glossary.

### 50–70 seconds: verifier-controlled patch

Show `submit_patch`.

The agent proposes a minimal replacement. The harness applies it to a working
copy, requires the old text to occur exactly once, reruns QA, and checks that
the change strictly improves the result without adding a new finding identity.
Only then does the harness commit the patch.

### 70–82 seconds: durable evidence

Show `agent_episode.json` or the report timeline.

The episode records typed actions, validated arguments, observations, provider
and model labels, cache hits, QA before and after, and the final status. Replay
is cache-only: a missing indexed response is an error, never a silent live call.

### 82–90 seconds: honest close

> The replay finishes with zero deterministic QA findings. That proves the
> bounded tool loop and mechanical gate worked on this fixture. It does not
> prove literary translation quality or a funded live-provider run.

## Accurate claims

- Bounded tool-using repair agent.
- Two-model terminology arbitration with blinded evaluator tie-breaking.
- Schema-constrained actions and fixed execution budgets.
- Deterministic QA-gated patch promotion.
- Persistent action trajectories and cache-only replay.
- Human escalation for unresolved work.

## Claims to avoid

- Fully autonomous translation.
- Semantic or literary quality guarantees.
- Production-scale performance.
- A live provider run from the bundled synthetic replay.
- Multi-model majority voting: this demo uses two voters and evaluator-based
  arbitration when they disagree.

## Optional advanced coda

If the reviewer asks about the larger system, show the secondary batch commands:

```bash
python -m agentic_translation batch run samples/public_demo/story.yaml \
  --chapters 0001 --provider-mode offline --overwrite
python -m agentic_translation batch inspect runs/public_demo_0001
python -m agentic_translation batch review runs/public_demo_0001 --write
python -m agentic_translation batch prove runs/public_demo_0001 --json
```

Keep this short. The main story is the model-directed terminology action,
deterministic patch gate, and replayable evidence.
