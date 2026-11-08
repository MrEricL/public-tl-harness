"""Costing checks for disjoint token categories and persistent reservations."""

from decimal import Decimal
import json

import pytest

from agentic_translation.costing import (
    BudgetExceeded, RateCard, SpendLedger, allocate_logical_costs, cold_input_estimate, price_usage,
)


def _rate() -> RateCard:
    return RateCard("test", "model", "2026-09", "peak", Decimal("1"), Decimal("2"),
                    Decimal("0.1"), Decimal("1.5"), Decimal("3"))


def test_cache_and_reasoning_counters_are_disjoint() -> None:
    receipt = price_usage({"input_tokens": 1000, "cached_input_tokens": 600,
                           "input_tokens_cache_write": 100, "output_tokens": 200,
                           "reasoning_tokens": 50}, _rate(), operation="translate")
    assert receipt["input_tokens_uncached"] == 300
    assert receipt["input_tokens_cache_read"] == 600
    assert receipt["output_tokens_billable_ordinary"] == 150
    assert Decimal(receipt["estimated_charge"]) == Decimal("0.00096")
    assert cold_input_estimate(receipt, _rate()) == "0.00145"
    assert receipt["usage_complete"] is True


def test_missing_usage_is_unknown_and_missing_cache_rate_prevents_false_charge() -> None:
    rate = RateCard("test", "model", "2026-09", "peak", Decimal("1"), Decimal("2"))
    assert price_usage({}, rate, operation="review")["estimated_charge"] is None
    assert price_usage({"input_tokens": 100, "cached_input_tokens": 50, "output_tokens": 10},
                       rate, operation="review")["estimated_charge"] is None


def test_nested_provider_details_do_not_add_cache_or_reasoning_twice() -> None:
    rate = RateCard("test", "model", "snapshot", "all_day", Decimal("1"), Decimal("2"), Decimal("0.5"))
    receipt = price_usage({"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 40},
                           "completion_tokens": 20, "completion_tokens_details": {"reasoning_tokens": 5}},
                          rate, operation="judge")
    assert receipt["input_tokens_uncached"] == 60
    assert receipt["output_tokens_billable_ordinary"] == 15
    assert Decimal(receipt["estimated_charge"]) == Decimal("0.00012")
    zero_reasoning = RateCard("test", "model", "snapshot", "all_day", Decimal("1"), Decimal("2"),
                              Decimal("0.5"), reasoning_per_million=Decimal("0"))
    free_reasoning = price_usage({"input_tokens": 0, "output_tokens": 10, "reasoning_tokens": 10},
                                 zero_reasoning, operation="jev")
    assert free_reasoning["estimated_charge"] == "0"


def test_shared_reservation_persists_and_protects_evaluation(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    first = SpendLedger(path, "5", "2")
    first.reserve("gen", "2.5", phase="system")
    resumed = SpendLedger(path, "5", "2")
    with pytest.raises(BudgetExceeded):
        resumed.reserve("gen2", "0.6", phase="system")
    resumed.settle("gen", None, usage_complete=False)
    assert first.snapshot()["unknown_usage_calls"] == 1
    resumed.reserve("judge", "2.4", phase="evaluation")
    with pytest.raises(BudgetExceeded):
        first.reserve("judge2", "0.2", phase="evaluation")
    assert json.loads(path.read_text())["calls"]["gen"]["budget_charge_usd"] == "2.5"
    with pytest.raises(ValueError, match="frozen budget"):
        SpendLedger(path, "10", "0")


def test_shared_physical_draft_is_charged_once_per_arm_and_once_observed() -> None:
    receipts = {"memory": {"estimated_charge": "0.2"},
                "draft": {"estimated_charge": "0.3"},
                "repair": {"estimated_charge": "0.1"}}
    allocation = allocate_logical_costs(
        receipts, {"contextual": ["draft"], "adaptive": ["draft", "draft", "repair"]},
        {"book": ["memory"]}, {"book": 2})
    assert allocation["standalone_arm_usd_excluding_allocated_memory"] == {
        "contextual": "0.3", "adaptive": "0.4"}
    assert allocation["observed_physical_usd"] == "0.6"
    assert allocation["memory_preparation"]["book"]["per_scored_chapter_usd"] == "0.1"
