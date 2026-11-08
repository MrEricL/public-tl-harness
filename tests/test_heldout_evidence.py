"""Verify the public numeric evidence without models, private text, or SDKs."""
from copy import deepcopy
import json
import subprocess
import sys

import pytest

from experiments.heldout_evidence.verify import DEFAULT_EVIDENCE, verify


def test_public_evidence_recomputes_the_reported_result_and_coverage():
    result = verify(json.loads(DEFAULT_EVIDENCE.read_text()))
    assert result["assigned_chapters"] == 40
    assert result["work_clusters"] == 4
    assert [(result["arms"][arm]["modal_count"], result["arms"][arm]["valid_renderings"])
            for arm in ("naive", "contextual", "full_harness", "review_everything")] == [
                (561, 851), (734, 854), (744, 853), (744, 854)]
    assert result["inconsistency_reduction"] == pytest.approx(.6250192020050935)
    assert result["inconsistency_ratio"] == pytest.approx(2.6668032212507677)
    primary = result["comparisons"]["full_harness/naive"]
    assert (primary["three_plus_to_one_rendering"], primary["baseline_three_plus_renderings"]) == (39, 72)
    assert (primary["terms_improved"], primary["terms_unchanged"], primary["terms_regressed"]) == (90, 57, 13)
    assert primary["paired_term_mean_gain"] == pytest.approx(.20951636904761903)
    assert primary["work_bootstrap"]["resamples"] == 256
    assert primary["work_bootstrap"]["interval_95"] == pytest.approx([.16749751984126984, .2507142857142857])
    assert all(row["full_vs_naive_gain"] > 0 for row in result["work_results"])
    assert primary["quality"] == {"wins": 14, "ties": 9, "losses": 17, "unrated": 0,
                                    "two_judges": 34, "one_judge": 6, "identical_outputs": 0}
    assert primary["quality_with_two_judges"] == {"wins": 14, "ties": 6, "losses": 14}
    assert "unavailable" in result["glossary_count_verification"]


def test_public_evidence_rejects_dropped_regressions_or_changed_denominators():
    original = json.loads(DEFAULT_EVIDENCE.read_text())
    data = deepcopy(original)
    data["work_summaries"][0]["paired_comparisons"]["full_harness/naive"]["terms_regressed"] = 0
    with pytest.raises(ValueError, match="Term outcomes"):
        verify(data)
    data = deepcopy(original)
    data["work_summaries"][0]["paired_comparisons"]["full_harness/naive"]["quality"]["two_judges"] += 1
    with pytest.raises(ValueError, match="Judge coverage"):
        verify(data)


def test_public_evidence_command_runs_without_the_repository_on_pythonpath(tmp_path):
    script = DEFAULT_EVIDENCE.with_name("verify.py")
    # -I isolates import paths and ignores user Python configuration. The public
    # checker must remain usable without this project's optional dependencies.
    completed = subprocess.run([sys.executable, "-I", str(script), "--json"], cwd=tmp_path,
                               text=True, capture_output=True, check=True)
    assert json.loads(completed.stdout)["assigned_chapters"] == 40
