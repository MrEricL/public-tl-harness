# Code map

Start at the root `demo.py`, which calls
`agentic_translation.demo.main` → `demo.run_folder`. Follow the chapter through
source preparation, memory, generation, and repair before reading the UI.
The workbench calls the same folder engine.

| Pipeline step | Module | Responsibility |
| --- | --- | --- |
| Select inputs and run chapters | `demo.py` | Natural file order, limits, carried memory, lead-in, recorded settings, replay, outcomes, and delivery |
| Segment source | `segmentation.py` | Bounded passages and source positions |
| Extract and supply memory | `unattended.py`, `story_memory.py` | Exact-quote validation, causal availability, glossary precedence, snapshots, and bounded context |
| Call the model | `demo_provider.py`, `structured_transport.py` | Provider settings, deadlines, schema correction, receipts, and cached replay. The Codex and Claude CLI transports in `structured_transport.py` are older study tools, not a supported provider |
| Generate and align | `unattended.py` | Translation envelopes, known-name extraction, and translator-name alignment adapter |
| Review and repair | `adaptive.py`, `agent_session.py`, `agent_repair.py` | Review coverage, exact-target edits, guards, checkpointed decisions, and budgets |
| Check naming and structure | `qa.py`, `glossary.py`, `terminology.py` | Deterministic findings and vocabulary contracts |
| Report and package | `demo_metrics.py`, `package.py` | Diffs, terminology checks, context receipts, resources, HTML, TXT, and EPUB |
| Start and inspect in a browser | `workbench.py`, `workbench_runs.py`, `static/workbench.*` | Local server, input validation, run activity, stop requests, saved views, and replay |
| Run a configured book | `cli.py`, `autonomous_book.py`, `profiles.py` | Thin book CLI, contextual/full-review policies, glossary, and ordered chapter delivery; it doesn't replay saved runs (use `demo.py --replay`) |

Names in the table are inside `agentic_translation/`. The support modules
below share the same guards and model contracts; the local UI does not
introduce a second translation implementation.

| Support module | Role |
| --- | --- |
| `models.py`, `agent_models.py`, `terminology_models.py` | Typed contracts: story paths and settings, repair actions, terminology decisions |
| `agent_tools.py`, `agent_policy.py`, `agent_provider.py` | The repair agent's typed tools, deterministic policy checks and replayable action provider |
| `autonomous_provider.py`, `costing.py` | Budgeted live transports over the response cache; cost estimates and a spend ceiling |
| `providers_llm.py`, `showcase_providers.py`, `providers_offline.py` | OpenAI-compatible provider settings, provider profiles, and deterministic text normalization used by the guards |
| `env_config.py`, `report.py`, `__main__.py` | `.env` key loading, HTML report rendering, and `python -m agentic_translation` |
| `semantic_models.py`, `semantic_provider.py`, `semantic_signals.py`, `decisions/` | Optional Jev checks, off by default, used by the older live `adaptive` strategy |

The public tree is the folder demo, the workbench and the book runner. Earlier
command-line, MCP and batch tools are not in this release. Corpus preparation,
scoring runners, private inputs and generated run folders are not distributed.

Useful reading order:

1. Run `python demo.py --help` and read `demo.run_folder`.
2. Follow `SourceMemoryExtractor` and `StoryMemoryTape.context_for_segment`.
3. Read the review/repair loop and executor guards in `adaptive.py` and
   `agent_session.py`.
4. Inspect `demo_metrics.py`, then the workbench's HTTP and run adapters.
5. Read `autonomous_book.py` for the configured book workflow.

[Try it](TRY_IT.md) · [How it works](HOW_IT_WORKS.md) ·
[Memory](MEMORY.md) · [Results](RESULTS.md)
