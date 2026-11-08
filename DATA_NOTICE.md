# Data notice

The MIT License in [LICENSE](LICENSE) covers the original source code in this
repository. It does not grant rights to third-party dependencies, material
with a separate notice, external source texts, or translations derived from
them.

The September and November chapter studies used a private corpus. Full source
chapters, translations, personal glossaries, model requests/responses, judging
packets, source-to-anonymous-work mappings, and generated run folders are not
distributed. Private study runners and preparation/scoring files remain outside
the public tree. [Results](docs/RESULTS.md) reports aggregate measurements,
methods, failures, corrections, and limits.

Short illustrative names, phrases, and sentences may appear in documentation
and graphics, including the character-name example 伊文. This is distinct
from distributing books, bulk corpus text, or translated chapters. No novel
titles or identifying work slugs are included in this release.

The November evidence is reported as text rather than publishing its study
files. The existing `experiments/heldout_evidence/` bundle contains anonymous
numeric summaries and sufficient statistics for the September result. Its
standard-library verifier checks consistency arithmetic, costs, and judge
coverage without source access. It contains no term identifiers, rendering
strings, or hashes identifying private source material. Checking those
aggregates does not independently validate model extraction or correctness.

The files in `samples/showcase/` and `samples/public_demo/` are independently
authored for invented sample stories. Any retained small repair fixtures are authored software checks,
not text drawn from the private corpus. Tests use authored or abstract fixtures
to exercise contracts and failure handling. Scripted or replayed behavior is
not presented as new live-model quality evidence.

The local workbench reads the user's own inputs and saved runs. Those inputs,
translations, memory snapshots, requests, and caches are local data; they are
not bundled with this repository or uploaded by the workbench to a publication
service. A live run sends the selected source and context to the configured
model provider. Keep generated run folders private unless their content and
publication rights have been reviewed.

Release preparation selects exact files, scans for private references and
credential patterns, and reviews included assets. These are bounded checks,
not a general certification of authorship or of historical Git objects.
