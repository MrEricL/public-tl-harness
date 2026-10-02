"""Auditable API cost estimates and a persistent experiment spend ceiling.

All money here is an estimate from an explicitly supplied rate snapshot.  A
missing usage category is never interpreted as a free request.  The ledger is
shared by generation and evaluation processes and keeps reservations until a
request is settled, including when provider usage is unavailable.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping


def _money(value: Decimal | str | float | int) -> Decimal:
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise ValueError("USD amount must be finite and nonnegative")
    return amount


def _tokens(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class RateCard:
    """Per-million-token USD rates at a named, dated pricing snapshot."""

    provider: str
    model: str
    rate_card_id: str
    pricing_period: str
    input_per_million: Decimal
    output_per_million: Decimal
    cache_read_per_million: Decimal | None = None
    cache_write_per_million: Decimal | None = None
    reasoning_per_million: Decimal | None = None
    currency: str = "USD"

    def __post_init__(self) -> None:
        if self.currency != "USD":
            raise ValueError("The spend ledger currently requires USD rate cards")
        for name in ("input_per_million", "output_per_million", "cache_read_per_million",
                     "cache_write_per_million", "reasoning_per_million"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _money(value))


def normalize_usage(usage: Mapping[str, Any], *, input_total_includes_cache: bool = True,
                    output_total_includes_reasoning: bool = True) -> dict[str, Any]:
    """Turn provider usage into disjoint billing categories.

    The OpenAI-compatible default includes cache reads and reasoning in the
    reported input/output totals.  Callers can override those semantics for a
    provider with disjoint counters.  A missing total leaves pricing incomplete.
    """

    def first(*names: str) -> int | None:
        for name in names:
            if name in usage and usage[name] is not None:
                return _tokens(usage[name], name)
        return None

    prompt_details = usage.get("prompt_tokens_details") or usage.get("input_token_details") or {}
    completion_details = usage.get("completion_tokens_details") or usage.get("output_token_details") or {}
    if not isinstance(prompt_details, Mapping):
        prompt_details = {}
    if not isinstance(completion_details, Mapping):
        completion_details = {}
    total_input = first("input_tokens_total", "input_tokens", "prompt_tokens")
    explicit_uncached = first("input_tokens_uncached")
    cache_read = first("input_tokens_cache_read", "cached_input_tokens", "prompt_cache_hit_tokens",
                       "prompt_cache_read_tokens")
    if cache_read is None:
        cache_read = _tokens(prompt_details.get("cached_tokens"), "prompt_tokens_details.cached_tokens") or 0
    cache_write = first("input_tokens_cache_write", "cache_creation_input_tokens") or 0
    output = first("output_tokens", "completion_tokens")
    reasoning = first("reasoning_tokens_if_separately_reported", "reasoning_tokens")
    if reasoning is None:
        reasoning = _tokens(completion_details.get("reasoning_tokens"),
                            "completion_tokens_details.reasoning_tokens") or 0

    if total_input is None and explicit_uncached is not None:
        total_input = explicit_uncached + cache_read + cache_write
    if total_input is not None:
        uncached = total_input - cache_read - cache_write if input_total_includes_cache else total_input
        if explicit_uncached is not None and explicit_uncached != uncached:
            raise ValueError("Conflicting uncached and total input counters")
        if uncached < 0:
            raise ValueError("Cached input exceeds total input")
    else:
        uncached = None
    if output is not None and output_total_includes_reasoning:
        ordinary_output = output - reasoning
        if ordinary_output < 0:
            raise ValueError("Reasoning output exceeds total output")
    else:
        ordinary_output = output
    return {
        "input_tokens_total": total_input if input_total_includes_cache or total_input is None else total_input + cache_read + cache_write,
        "input_tokens_uncached": uncached,
        "input_tokens_cache_read": cache_read,
        "input_tokens_cache_write": cache_write,
        "output_tokens": output,
        "output_tokens_billable_ordinary": ordinary_output,
        "reasoning_tokens_if_separately_reported": reasoning,
        "usage_complete": uncached is not None and output is not None,
    }


def price_usage(usage: Mapping[str, Any], rate: RateCard, *, operation: str,
                timestamp_utc: str | None = None, input_total_includes_cache: bool = True,
                output_total_includes_reasoning: bool = True) -> dict[str, Any]:
    """Return a receipt with normalized usage and estimated charge or ``None``."""
    normalized = normalize_usage(
        usage, input_total_includes_cache=input_total_includes_cache,
        output_total_includes_reasoning=output_total_includes_reasoning,
    )
    charge: Decimal | None = None
    if normalized["usage_complete"]:
        parts = (
            (normalized["input_tokens_uncached"], rate.input_per_million),
            (normalized["input_tokens_cache_read"], rate.cache_read_per_million),
            (normalized["input_tokens_cache_write"], rate.cache_write_per_million),
            (normalized["output_tokens_billable_ordinary"], rate.output_per_million),
            (normalized["reasoning_tokens_if_separately_reported"],
             rate.reasoning_per_million if rate.reasoning_per_million is not None else rate.output_per_million),
        )
        charge = Decimal(0)
        for count, category_rate in parts:
            if count:
                if category_rate is None:
                    charge = None
                    break
                charge += Decimal(count) * category_rate / Decimal(1_000_000)
    return {
        "provider": rate.provider,
        "model": rate.model,
        "operation": operation,
        "timestamp_utc": timestamp_utc or datetime.now(timezone.utc).isoformat(),
        **normalized,
        "rate_card_id": rate.rate_card_id,
        "pricing_period": rate.pricing_period,
        "currency": rate.currency,
        "estimated_charge": str(charge) if charge is not None else None,
        "charge_basis": "rate_card_estimate" if charge is not None else "usage_or_rate_incomplete",
    }


def cold_input_estimate(receipt: Mapping[str, Any], rate: RateCard) -> str | None:
    """Reprice observed usage as if every input token missed the provider cache.

    This is a sensitivity estimate, never an observed charge.  It retains the
    same output/reasoning categories and rate period as the physical receipt.
    """
    total_input = _tokens(receipt.get("input_tokens_total"), "input_tokens_total")
    ordinary_output = _tokens(receipt.get("output_tokens_billable_ordinary"),
                              "output_tokens_billable_ordinary")
    reasoning = _tokens(receipt.get("reasoning_tokens_if_separately_reported"),
                        "reasoning_tokens_if_separately_reported") or 0
    if total_input is None or ordinary_output is None:
        return None
    amount = Decimal(total_input) * rate.input_per_million
    amount += Decimal(ordinary_output) * rate.output_per_million
    amount += Decimal(reasoning) * (rate.reasoning_per_million
                                    if rate.reasoning_per_million is not None else rate.output_per_million)
    return str(amount / Decimal(1_000_000))


class BudgetExceeded(RuntimeError):
    """A new request cannot fit within the frozen experiment ceiling."""
    fatal_provider_error = True


class SpendLedger:
    """File-backed atomic reservation ledger shared across CLI processes.

    `evaluation_reserve_usd` is protected from system calls; evaluation calls
    can use it.  Unknown actual usage is settled at its reservation amount and
    marked incomplete, so subsequent calls cannot treat it as zero cost.
    """

    def __init__(self, path: str | Path, ceiling_usd: Decimal | str | float | int,
                 evaluation_reserve_usd: Decimal | str | float | int = 0):
        self.path = Path(path)
        self.ceiling = _money(ceiling_usd)
        self.evaluation_reserve = _money(evaluation_reserve_usd)
        if self.evaluation_reserve > self.ceiling:
            raise ValueError("Evaluation reserve exceeds ceiling")
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            if self.path.exists():
                state = self._read()
                if _money(state["ceiling_usd"]) != self.ceiling or _money(state["evaluation_reserve_usd"]) != self.evaluation_reserve:
                    raise ValueError("Existing ledger budget differs from requested frozen budget")
            else:
                self._write({"schema_version": 1, "ceiling_usd": str(self.ceiling),
                             "evaluation_reserve_usd": str(self.evaluation_reserve), "calls": {}})

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, state: dict[str, Any]) -> None:
        fd, name = tempfile.mkstemp(prefix=".spend-", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(state, stream, indent=2, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @staticmethod
    def _committed(state: Mapping[str, Any]) -> Decimal:
        return sum((_money(row["budget_charge_usd"]) for row in state["calls"].values()), Decimal(0))

    def reserve(self, call_id: str, estimate_usd: Decimal | str | float | int,
                *, phase: str = "system") -> dict[str, Any]:
        if not call_id:
            raise ValueError("call_id is required")
        if phase not in {"system", "evaluation"}:
            raise ValueError("phase must be system or evaluation")
        estimate = _money(estimate_usd)
        with self._locked():
            state = self._read()
            existing = state["calls"].get(call_id)
            if existing is not None:
                if existing["phase"] != phase or _money(existing["reserved_usd"]) != estimate:
                    raise ValueError("call_id already exists with different reservation")
                return existing
            limit = self.ceiling - (self.evaluation_reserve if phase == "system" else Decimal(0))
            if self._committed(state) + estimate > limit:
                raise BudgetExceeded(f"Cannot reserve {estimate} USD for {phase}; ceiling {limit} USD")
            entry = {"phase": phase, "reserved_usd": str(estimate),
                     "budget_charge_usd": str(estimate), "actual_estimate_usd": None,
                     "usage_complete": False, "status": "reserved",
                     "timestamp_utc": datetime.now(timezone.utc).isoformat()}
            state["calls"][call_id] = entry
            self._write(state)
            return entry

    def settle(self, call_id: str, actual_usd: Decimal | str | float | int | None,
               *, usage_complete: bool) -> dict[str, Any]:
        with self._locked():
            state = self._read()
            if call_id not in state["calls"]:
                raise KeyError(call_id)
            entry = state["calls"][call_id]
            if entry["status"] == "settled":
                return entry
            actual = _money(actual_usd) if actual_usd is not None else None
            # Do not release the reservation if usage is unknown.  An actual
            # charge above the reservation remains visible even after the call.
            budget_charge = (actual if actual is not None and usage_complete else
                             max(actual or Decimal(0), _money(entry["reserved_usd"])))
            entry.update(status="settled", budget_charge_usd=str(budget_charge),
                         actual_estimate_usd=str(actual) if actual is not None else None,
                         usage_complete=bool(usage_complete and actual is not None))
            self._write(state)
            return entry

    def snapshot(self) -> dict[str, Any]:
        with self._locked():
            state = self._read()
        state["budget_committed_usd"] = str(self._committed(state))
        state["estimated_actual_usd"] = str(sum(
            (_money(row["actual_estimate_usd"]) for row in state["calls"].values()
             if row["actual_estimate_usd"] is not None), Decimal(0)))
        state["unknown_usage_calls"] = sum(row["status"] == "settled" and not row["usage_complete"]
                                           for row in state["calls"].values())
        return state


def allocate_logical_costs(receipts: Mapping[str, Mapping[str, Any]],
                           arm_receipt_ids: Mapping[str, list[str]],
                           memory_receipt_ids: Mapping[str, list[str]] | None = None,
                           scored_chapters_per_work: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Estimate each arm's standalone cost while deduplicating physical calls.

    `memory_receipt_ids` maps work ID to preparation receipts.  The returned
    per-work allocation is divided by scored chapters and is applied to every
    memory-using arm by its caller.  Unknown receipt charges propagate as None.
    """
    def sum_ids(ids: list[str]) -> str | None:
        amounts = []
        for receipt_id in dict.fromkeys(ids):
            if receipt_id not in receipts:
                raise KeyError(receipt_id)
            charge = receipts[receipt_id].get("estimated_charge")
            if charge is None:
                return None
            amounts.append(_money(charge))
        return str(sum(amounts, Decimal(0)))

    logical = {arm: sum_ids(ids) for arm, ids in arm_receipt_ids.items()}
    all_physical = [receipt_id for ids in arm_receipt_ids.values() for receipt_id in ids]
    memory = {}
    for work, ids in (memory_receipt_ids or {}).items():
        total = sum_ids(ids)
        count = (scored_chapters_per_work or {}).get(work)
        if count is None or count <= 0:
            raise ValueError(f"Positive scored chapter count required for work {work}")
        memory[work] = {"total_usd": total,
                        "per_scored_chapter_usd": str(_money(total) / count) if total is not None else None}
        all_physical.extend(ids)
    return {"standalone_arm_usd_excluding_allocated_memory": logical,
            "observed_physical_usd": sum_ids(all_physical),
            "memory_preparation": memory}
