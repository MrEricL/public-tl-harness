# Recompute the historical consistency result

From the repository root, run the standard-library verifier:

    python experiments/heldout_evidence/verify.py

The script recomputes reported terminology counts, drift transitions, paired effects, cost arithmetic, and judge coverage from the anonymous aggregate file, RESULTS.json. It does not need an API key or source text.

The verifier checks arithmetic and internal consistency of the published summary. It cannot validate translation correctness, confirm the source labels, or independently verify the private corpus extraction.

The bundle contains anonymous numeric summaries and study descriptions. It contains no chapter text, translations, term names, renderings, source hashes, prompts, or provider responses. See the [data notice](../../DATA_NOTICE.md) for what this repository distributes and the [results page](../../docs/RESULTS.md) for denominators and limitations.
