# Direct TypeSafe decisions

`TypeSafeDecisionProvider.evaluate(DecisionRequest)` asks eight focused questions
about one Chinese source passage, its English draft, and supplied context in one
System One request. The request contains six Noul material-error checks, one
Choice for source sufficiency, and one Score for English readability. A
`candidate_text` adds a Noul check for newly introduced unsupported claims.
See [`questions.py`](questions.py) for the versioned wording. The service
receives the actual Chinese text; this adapter does not translate it first.

Install the optional integration with `pip install -e '.[jev]'` (using TypeSafe's
package index if required by the environment) and set `TYPESAFE_API_KEY` in the
environment. The key is never stored in a decision receipt. The live provider
uses the TypeSafe Python SDK with its internal retries disabled, then performs
at most two explicit retries of transient HTTP or connection errors. Authentication,
schema, and validation failures do not retry. SDK/model availability and the
request's exact served model are recorded during live use.

The default request pins `jev-1.13.0`. The receipt preserves requested and
served model IDs, text hashes, a question schema hash, policy version and
thresholds, raw typed answers, derived flags, usage, estimated cost (only when
the served model is the priced Jev 1.13 version), latency, retry count, HTTP
status on service errors, and transport status. Error response bodies are not
retained. A mismatched served model invalidates a pinned-version request. The
receipt contains no source or draft prose. The 1.13 list price
used for estimates was $0.042 per million input tokens on 2026-09-22; refresh
the pricing snapshot before an experiment. Output tokens are recorded but not
charged at that listed rate.

`DecisionResult.route_level` is `clean`, `diagnostic`, `material`, or `fallback`.
Any unavailable or malformed response returns `fallback`; it never certifies
a segment as clean. The caller must use a bounded generative review fallback
and record its cost. Choice/Score confidence and Noul probabilities are kept
as raw model judgments, not treated as correctness guarantees. Thresholds are
development defaults and must be frozen after testing on the actual workload.

With `record_dir`, successful live decisions are written atomically as
source-free JSON receipts. `RecordedDecisionProvider` reads those receipts
without creating a network client. It returns a labeled miss when source,
draft, context (including neighbors/glossary), candidate, model, question
version, or policy changes. A receipt checksum detects corruption. Do not
place raw source text in `segment_id` or policy metadata; those fields are
included in the saved receipt.

The adapter follows TypeSafe's [System One quick start](https://docs.typesafe.ai/introduction/quickstart),
[Python SDK](https://docs.typesafe.ai/sdk/python), [Noul](https://docs.typesafe.ai/primitives/noul),
and [citation-check cookbook](https://docs.typesafe.ai/cookbooks/citation_check).
The cookbook's source-first verification pattern informed the explicit source
and patch checks here; the project still needs empirical CJK calibration.
