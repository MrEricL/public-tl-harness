# Resumable session identity

Harness v3 may pause immediately before an approval-gated persistent effect.
Resuming that session is safe only when the caller supplies the same task and
runtime contract that created the paused snapshot.

## Bound identity

Every newly created session snapshot carries a strict `AgentSessionIdentity`
record with:

- the run ID, story slug, and chapter;
- SHA-256 digests of the exact UTF-8 source text and canonical JSON form of the
  parsed master glossary;
- provider mode, provider name, and model name;
- tool protocol and tool-schema version; and
- a SHA-256 digest of the complete Harness v3 registry contract.

The registry digest covers every logical and provider-facing tool name,
description, argument schema, tag, approval requirement, side-effect class,
risk class, and declared timeout. Registry entries are serialized in
deterministic logical-name order before hashing.

## Resume boundary

`resume_repair_session` rebuilds the identity candidate from its arguments and
the current runtime, then compares it with the persisted identity immediately
after loading the snapshot. Any missing or mismatched field raises a dedicated
identity error that names only the mismatched fields.

The check occurs before the harness:

- records an approval decision;
- calls a provider or terminology resolver;
- changes the canonical glossary; or
- rewrites the event log, snapshot, or episode projection.

Callers may omit `run_id`, `story_slug`, `chapter`, and `provider_mode`; omitted
values resolve to the persisted values. Explicit values must match. Snapshots
created before this identity contract remain readable, but cannot be resumed:
the caller must start a new session so the missing source and glossary
fingerprints are captured honestly.

The canonical glossary write retains its separate path and before/after digest
checks. Those checks bind the approved filesystem effect; session identity
binds the task and runtime that are allowed to reach that effect.

Showcase snapshots also persist the instruction context, fidelity-review
requirement, nonregressing-patch policy, and review-round limit. Resume derives
omitted policy arguments from the snapshot and rejects explicit mismatches.
Automatic repair requires fidelity review; a resumed session cannot silently
turn that requirement off. Completion reviews must match the current draft's
SHA-256, so editing text invalidates its earlier review.

Enabled Jev sensing also binds the policy and semantic configuration digest to
the session. Resume rejects a different policy, including enabling Jev on an
existing no-Jev session. Signal replay checks source/draft and configuration
identity and fails on missing or mismatched evidence without a network fallback.
Jev reports cannot satisfy the final fidelity-review requirement.

## Deliberate scope

This contract does not add event hash chains, crash recovery, session locking,
provider retry policy, resolver fingerprints, or generalized artifact
attestation. Those remain separate hardening concerns. The guarantee is
narrow: a paused session cannot continue under different caller inputs or a
different Harness v3 tool contract.

## Verification contract

Focused tests cover a matching resume and mismatches in source, master
glossary, provider/model, provider mode, protocol, schema version, registry,
and run/story/chapter metadata. Every rejection asserts that provider calls,
canonical glossary bytes, event bytes, snapshot bytes, and episode bytes are
unchanged. Tests also cover identity serialization, legacy-snapshot rejection,
and the existing successful write-once/idempotent-resume behavior.
