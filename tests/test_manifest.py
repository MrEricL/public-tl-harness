from __future__ import annotations

import json

from agentic_translation.manifest import build_manifest
from agentic_translation.models import EvalMetrics, ProviderLabel


def test_manifest_serializes_schema_version_and_artifacts() -> None:
    manifest = build_manifest(
        run_id="demo",
        story_slug="public_demo",
        title="Public Demo",
        chapter_ids=["0001"],
        mode="offline",
        public_safe=True,
        inputs={"story_yaml": "samples/public_demo/story.yaml"},
        providers={
            "translation": ProviderLabel(provider="offline", model="offline-fixture-v1"),
            "judge": ProviderLabel(provider="offline", model="offline-rubric-v1"),
            "repair": ProviderLabel(provider="offline", model="offline-patch-v1"),
        },
        qa={"source_findings": 0, "baseline_findings": 6, "glossary_findings": 1, "final_findings": 0},
        artifacts={"report_html": "report.html", "epub": "review/public_demo_0001.epub"},
        eval_metrics=[EvalMetrics(mode="final", score=96)],
    )

    payload = json.loads(manifest.model_dump_json())
    assert payload["schema_version"] == "0.1"
    assert payload["providers"]["translation"]["provider"] == "offline"
    assert payload["artifacts"]["epub"] == "review/public_demo_0001.epub"
