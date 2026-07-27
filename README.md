# Agentic Translation Reliability Harness

A bounded repair agent for Chinese–English translation pipelines.

The model selects typed tools. Python validates every action and accepts a
patch only when deterministic QA improves without creating a new failure.

```text
find issue → inspect evidence → resolve terminology → propose patch → re-run QA → finish or escalate
```

> **Governing rule:** the model proposes; the verifier disposes.

## Why this project matters

LLM output can sound convincing while breaking terminology, leaving source
language in the text, or changing document structure. This project puts a
controlled execution harness around the model:

- The model chooses the next repair action.
- Tool arguments are schema-validated.
- Step and patch budgets keep each episode bounded.
- Proposed changes are tested on a working copy.
- Deterministic QA controls whether a patch is promoted.
- Every action, observation, decision, and provider call is recorded.
- Unresolved work is escalated for human review.

The result is a translation workflow that is inspectable, reproducible, and
safe to demo without depending on a live API call.

## Run the main demo

### 1. Install

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

### 2. Run the replay

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

The replay is local, deterministic, and requires no API key.

Expected trajectory:

```text
resolve_terminology → submit_patch → finish
```

Expected result:

```text
OpenAI proposal:   Dao Heart
DeepSeek proposal: Heart of Dao
Evaluator choice:  Dao Heart
Initial findings:  1
Final findings:    0
Episode status:    verified
```

The agent asks for terminology arbitration, receives two independent proposals,
and uses a blinded evaluator to select a candidate. It then proposes a patch.
The harness applies that patch to a working copy, reruns QA, and promotes the
change only after the finding count reaches zero.

### 3. Inspect the evidence

| Artifact | What it shows |
|---|---|
| `runs/agentic_terminology_demo_replay/agent_episode.json` | Complete tool-call trajectory, budgets, observations, and provider-call records |
| `runs/agentic_terminology_demo_replay/repair_report.md` | Human-readable repair and terminology decision |
| `runs/agentic_terminology_demo_replay/report.html` | Visual timeline of the episode |
| `runs/agentic_terminology_demo_replay/qa_initial.json` | Findings before repair |
| `runs/agentic_terminology_demo_replay/qa_final.json` | Findings after repair |
| `runs/agentic_terminology_demo_replay/translated_final/0001.txt` | Verified final translation |

For a 90-second walkthrough, see [DEMO_SCRIPT.md](DEMO_SCRIPT.md).

## What “agentic” means here

The agent owns action selection. On each step, the model receives the current
findings, the observable episode history, the remaining budgets, and the
available tool schemas. It chooses one tool and supplies structured arguments.

Available tools include:

```text
get_qa_findings
read_source_context
read_translation_context
lookup_glossary
resolve_terminology
submit_patch
escalate
finish
```

Python does not prescribe a fixed repair sequence. It validates and executes
the selected action, returns the observation, and lets the model choose again.
That creates a bounded plan–act–observe–adjust loop.

## What the harness does

The harness is the software around the model. It supplies the rules and
infrastructure that make model-directed work controlled and repeatable.

```text
QA finding
    ↓
model selects a typed tool
    ↓
harness validates arguments and budgets
    ↓
tool returns an observable result
    ↓
model proposes a patch or another action
    ↓
deterministic verifier accepts, rejects, or escalates
```

The harness provides:

- **Typed tools:** Pydantic schemas reject malformed actions.
- **Bounded execution:** Step and patch-attempt limits stop open-ended loops.
- **Patch isolation:** The agent cannot write the delivered file directly.
- **Deterministic verification:** QA decides whether a proposed patch advances.
- **Durable state:** Each episode persists its tools, observations, and outcome.
- **Record and replay:** Cached provider responses reproduce the same trajectory.
- **Human escalation:** Ambiguous or unresolved findings become review work.

## Two-model terminology arbitration

Terminology repair is exposed as one bounded tool:
`resolve_terminology`.

The tool:

1. Sends the same term and context independently to OpenAI and DeepSeek roles.
2. Requires structured recommendations, alternatives, confidence, and rationale.
3. Normalizes superficial differences before comparing the recommendations.
4. Uses a blinded evaluator when the two models disagree.
5. Returns the selected term to the repair episode as a local override.
6. Records all votes, evaluation results, provider metadata, and cache hashes.

The selected term does not silently rewrite the permanent glossary. The agent
uses it within the current repair episode, and the verifier still controls the
patch.

## Deterministic patch promotion

`submit_patch` is the main control boundary. A model proposal is not a direct
file edit.

The harness:

1. Confirms that the expected text exists exactly once.
2. Applies the replacement to a working copy.
3. Runs the existing QA suite on that copy.
4. Compares the complete before-and-after finding identities.
5. Rejects a patch that does not improve QA.
6. Rejects a patch that introduces a new finding.
7. Promotes only an improving, non-regressing result.

This verifier checks observable compliance such as residual Chinese,
terminology drift, punctuation, panel structure, and prompt leakage. Semantic
and literary review remains a human or evaluator responsibility.

## Durable evidence and replay

Every model-backed action produces a `ProviderCallRecord` containing:

- provider and model identity;
- cache namespace and cache file;
- canonical payload hash;
- response hash;
- cache-hit status.

Replay mode reads the recorded cache only. Missing or mismatched entries fail
instead of falling through to a live provider. This makes the public demo
stable and gives technical reviewers an inspectable execution receipt.

Live mode uses the same interfaces and can record a new cache for later replay.
The committed replay fixture is the deterministic public example; the runtime
also supports OpenAI-compatible live endpoints for OpenAI and DeepSeek.

## Translation pipeline

The repair agent sits inside a larger corpus workflow:

```text
story configuration
    → source and glossary loading
    → translation
    → deterministic QA
    → bounded repair
    → human review when needed
    → TXT and EPUB packaging
    → artifact QA
```

The pipeline includes:

- chapter-scoped glossary context;
- residual-source and terminology checks;
- candidate evaluation and repair routing;
- resumable per-chapter manifests;
- review queues and work orders;
- TXT and EPUB artifact validation;
- batch proof and replay commands.

These advanced workflows remain secondary to the replay demo. Their full
commands and operator procedures are documented in
[USER_GUIDE.md](USER_GUIDE.md).

## Execution modes

| Mode | Purpose | Network |
|---|---|---:|
| `offline` | Run the deterministic translation and QA baseline | No |
| `replay` | Reproduce recorded model decisions and repair trajectories | No |
| `live` | Call configured providers and optionally record responses | Yes |

## Advanced corpus workflows

The CLI retains the production-oriented batch layer for reviewers who want to
inspect more than the golden path.

```bash
# Run one offline batch chapter
agentic-translation batch run samples/public_demo/story.yaml \
  --chapters 0001 \
  --provider-mode offline \
  --run-id public_batch_demo \
  --overwrite

# Inspect delivery state
agentic-translation batch inspect runs/public_batch_demo --status-json

# Generate a human review packet
agentic-translation batch review runs/public_batch_demo --write-markdown

# Check delivery, model evidence, and replayability
agentic-translation batch prove runs/public_batch_demo --json

# Replay a recorded live batch
agentic-translation batch replay runs/<recorded-live-run>
```

See the [User Guide](USER_GUIDE.md) for resume, refresh, acceptance, glossary,
panel, work-order, live-proof, and artifact-QA operations.

## Repository layout

```text
agentic_translation_prototype/
├── agentic_translation/             Python package and CLI
│   └── templates/                   Packaged HTML report templates
├── samples/
│   ├── agentic_terminology_demo/    Main two-model replay fixture
│   ├── agentic_repair_demo/         Rejected-then-accepted patch fixture
│   └── public_demo/                 Offline pipeline fixture
├── tests/                            Unit and end-to-end tests
├── DEMO_SCRIPT.md                   90-second presentation
├── USER_GUIDE.md                    Full operator documentation
└── README.md                        Project overview
```

Generated runs are written below `runs/` and are ignored by Git.

## Test

Run the full test suite:

```bash
pytest -q
```

Run only the main replay path:

```bash
pytest -q tests/test_terminology_golden.py
```

The focused replay test verifies:

- the expected three-tool action sequence;
- cache-only OpenAI and DeepSeek terminology calls;
- blinded evaluator use;
- the selected terminology;
- final QA with zero findings;
- the final text;
- the persisted report and episode.

## Technical highlights

- Python 3.11+
- Pydantic v2 typed action schemas
- Typer CLI
- OpenAI-compatible provider adapters
- OpenAI and DeepSeek terminology arbitration
- SHA-256 content-addressed provider cache
- Persistent JSON episode and batch manifests
- Deterministic QA-gated patch promotion
- Jinja2 HTML evidence reports
- TXT and EPUB artifact production

## Documentation

- [90-second demo script](DEMO_SCRIPT.md)
- [Complete user guide](USER_GUIDE.md)
