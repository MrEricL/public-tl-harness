from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agentic_translation import autonomous_book
from agentic_translation.cli import app
from agentic_translation.decisions.models import (
    QUESTION_SCHEMA_VERSION,
    DecisionReceipt,
    DecisionResult,
)
from agentic_translation.decisions.questions import question_schema_digest, request_id


def _story_yaml(root: Path) -> Path:
    source_dir = root / "source"
    source_dir.mkdir()
    (source_dir / "0001.txt").write_text("甲说。", encoding="utf-8")
    story = root / "story.yaml"
    story.write_text(
        "\n".join(
            [
                "slug: autonomous_test",
                "title: Autonomous Test",
                'chapter_ids: ["0001"]',
                "paths:",
                "  source_dir: source",
                "  glossary_path: terms.txt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "terms.txt").write_text("", encoding="utf-8")
    return story


class _FakeLedger:
    def snapshot(self) -> dict[str, object]:
        return {"budget_committed_usd": 0.0, "unknown_usage_calls": 0}


class _FakeGenerator:
    operations: list[str] = []

    def __init__(self, run_dir: Path, config: dict, *, model: str | None = None, phase: str = "system") -> None:
        self.run_dir = run_dir
        self.config = config
        self.model = model or config["writer"]["model"]
        self.phase = phase
        self.ledger = _FakeLedger()
        self.receipts: list[dict[str, object]] = []

    def call(
        self,
        operation: str,
        prompt: str,
        schema: dict,
        *,
        max_output_tokens: int = 4096,
    ) -> dict[str, object]:
        del prompt, schema
        self.operations.append(operation)
        self.receipts.append(
            {
                "operation": operation,
                "max_output_tokens": max_output_tokens,
                "estimated_charge": 0.0,
            }
        )
        if operation == "memory_extract":
            return {
                "entries": [
                    {
                        "kind": "identity",
                        "subject": "甲",
                        "value": "speaker",
                        "segment_id": "s0001",
                        "exact_excerpt": "甲",
                        "scope": "narrator",
                        "status": "active",
                    }
                ],
                "terms": [
                    {
                        "source": "甲",
                        "target": "A",
                        "segment_id": "s0001",
                        "exact_excerpt": "甲",
                        "provisional": False,
                    }
                ],
            }
        if operation == "contextual":
            return {
                "segments": [
                    {
                        "segment_id": "s0001",
                        "translated_text": "Chapter 1\n\nA speaks.",
                    }
                ]
            }
        raise AssertionError(f"Unexpected mocked generation operation: {operation}")


class _FailingGenerator(_FakeGenerator):
    def call(
        self,
        operation: str,
        prompt: str,
        schema: dict,
        *,
        max_output_tokens: int = 4096,
    ) -> dict[str, object]:
        del prompt, schema, max_output_tokens
        self.operations.append(operation)
        raise RuntimeError("simulated first generation failure")


class _CleanDecisionProvider:
    def evaluate(self, request):  # noqa: ANN001 - test protocol double
        return DecisionResult(
            status="ok",
            receipt=DecisionReceipt(
                request_id=request_id(request),
                segment_id=request.segment_id,
                requested_model=request.requested_model,
                served_model=request.requested_model,
                source_sha256=request.source_sha256,
                draft_sha256=request.draft_sha256,
                context_sha256=request.context_sha256,
                question_schema_version=QUESTION_SCHEMA_VERSION,
                question_schema_sha256=question_schema_digest(request),
                routing_policy_version=request.routing_policy_version,
                routing_thresholds=request.routing_thresholds,
                derived_flags={},
                latency_ms=0,
                attempt_count=1,
                transport_status="ok",
                cache_status="live",
            ),
        )


class _FakeBudgetedDecisions:
    def __init__(self, provider, config: dict, run_dir: Path) -> None:  # noqa: ANN001 - test protocol double
        del config, run_dir
        self.provider = _CleanDecisionProvider()
        self.receipts: list[dict[str, object]] = []

    def evaluate(self, request):  # noqa: ANN001 - test protocol double
        self.receipts.append({"estimated_charge": 0.0})
        return self.provider.evaluate(request)


def test_unattended_book_runs_memory_translation_and_adaptive_output(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001 - pytest fixture typing is not needed here
    story = _story_yaml(tmp_path)
    out = tmp_path / "run"
    _FakeGenerator.operations = []
    monkeypatch.setattr(autonomous_book, "BudgetedGenerator", _FakeGenerator)
    monkeypatch.setattr(autonomous_book, "BudgetedDecisions", _FakeBudgetedDecisions)
    monkeypatch.setattr(autonomous_book, "load_credentials", lambda: None)

    manifest = autonomous_book.run_unattended_book(story, out, strategy="adaptive")

    assert _FakeGenerator.operations == ["memory_extract", "contextual"]
    assert manifest["status"] == "delivered"
    assert manifest["chapters"] == [{"chapter_id": "0001", "outcome": "delivered", "warnings": []}]
    assert (out / "memory" / "story_memory" / "snapshots" / "0001.json").exists()
    chapter = json.loads((out / "chapters" / "0001.json").read_text(encoding="utf-8"))
    assert chapter["verification_policy"]["mode"] == "adaptive_segment_screen"
    assert chapter["final_text"] == "Chapter 1\n\nA speaks."
    assert (out / "delivery" / "book.txt").read_text(encoding="utf-8") == "Chapter 1\n\nA speaks."


def test_unattended_book_persists_failed_no_output_after_first_generation_error(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001 - pytest fixture typing is not needed here
    story = _story_yaml(tmp_path)
    out = tmp_path / "failed-run"
    _FailingGenerator.operations = []
    monkeypatch.setattr(autonomous_book, "BudgetedGenerator", _FailingGenerator)
    monkeypatch.setattr(autonomous_book, "BudgetedDecisions", _FakeBudgetedDecisions)
    monkeypatch.setattr(autonomous_book, "load_credentials", lambda: None)

    manifest = autonomous_book.run_unattended_book(story, out, strategy="adaptive")

    assert _FailingGenerator.operations == ["memory_extract"]
    assert manifest["status"] == "failed_no_output"
    assert manifest["delivery"] is None
    assert manifest["budget"] == {"budget_committed_usd": 0.0, "unknown_usage_calls": 0}
    assert manifest["chapters"] == [
        {
            "chapter_id": "0001",
            "outcome": "failed_no_output",
            "warnings": [
                {
                    "code": "failed_no_output",
                    "segment_id": None,
                    "message": "RuntimeError",
                }
            ],
            "error_type": "RuntimeError",
            "fatal_provider_error": False,
        }
    ]
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["status"] == "failed_no_output"
    chapter = json.loads((out / "chapters" / "0001.json").read_text(encoding="utf-8"))
    assert chapter["outcome"] == "failed_no_output"
    assert chapter["error"] == {"type": "RuntimeError", "fatal_provider_error": False}
    assert not (out / "delivery" / "book.txt").exists()


def test_adaptive_cli_requires_explicit_unattended_policy_flags(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "harness",
            "run",
            "--story",
            str(tmp_path / "story.yaml"),
            "--out",
            str(tmp_path / "run"),
            "--strategy",
            "adaptive",
        ],
    )

    assert result.exit_code == 1
    assert "--interaction unattended" in result.output
    assert "--memory source-grounded" in result.output
    assert "--glossary-policy run-local-auto" in result.output


def test_adaptive_cli_routes_to_unattended_book_runner(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    story = _story_yaml(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_runner(story_path: Path, out: Path, **kwargs: object) -> dict[str, object]:
        calls.append({"story": story_path, "out": out, **kwargs})
        return {"status": "delivered", "delivery": "delivery/book.txt"}

    monkeypatch.setattr("agentic_translation.autonomous_book.run_unattended_book", fake_runner)
    result = CliRunner().invoke(
        app,
        [
            "harness",
            "run",
            "--story",
            str(story),
            "--out",
            str(tmp_path / "run"),
            "--provider-mode",
            "live",
            "--profile",
            "deepseek",
            "--model",
            "deepseek-flash",
            "--strategy",
            "adaptive",
            "--interaction",
            "unattended",
            "--memory",
            "source-grounded",
            "--glossary-policy",
            "run-local-auto",
            "--decision-provider",
            "jev",
            "--decision-model",
            "jev-1.13.0",
            "--max-usd",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "story": story,
            "out": tmp_path / "run",
            "model": "deepseek-flash",
            "decision_model": "jev-1.13.0",
            "max_usd": 1.0,
            "strategy": "adaptive",
        }
    ]
    assert "delivered" in result.output


def test_adaptive_cli_reports_failed_manifest_without_delivery(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    story = _story_yaml(tmp_path)

    def failed_runner(story_path: Path, out: Path, **kwargs: object) -> dict[str, object]:
        del story_path, out, kwargs
        return {"status": "failed_no_output", "delivery": None}

    monkeypatch.setattr("agentic_translation.autonomous_book.run_unattended_book", failed_runner)
    result = CliRunner().invoke(
        app,
        [
            "harness",
            "run",
            "--story",
            str(story),
            "--out",
            str(tmp_path / "failed-run"),
            "--provider-mode",
            "live",
            "--profile",
            "deepseek",
            "--strategy",
            "adaptive",
            "--interaction",
            "unattended",
            "--memory",
            "source-grounded",
            "--glossary-policy",
            "run-local-auto",
            "--decision-provider",
            "jev",
        ],
    )

    assert result.exit_code == 1
    assert "Status: failed_no_output" in result.output
    assert "Manifest:" in result.output
    assert "failed-run/manifest.json" in result.output.replace("\n", "")
    assert "Delivery:" not in result.output


def test_unattended_book_honors_translator_glossary_over_memory(tmp_path: Path, monkeypatch) -> None:
    story = _story_yaml(tmp_path)
    (tmp_path / "terms.txt").write_text("甲 -> Aaron\n# block: A\n乙 -> Unrelated", encoding="utf-8")
    captured = []

    class ContractGenerator(_FakeGenerator):
        def call(self, operation, prompt, schema, **kwargs):
            captured.append((operation, prompt))
            response = super().call(operation, prompt, schema, **kwargs)
            if operation == "contextual" and "TERM 甲 -> Aaron [translator glossary]" in prompt:
                response["segments"][0]["translated_text"] = "Chapter 1\n\nAaron speaks."
            return response

    monkeypatch.setattr(autonomous_book, "load_credentials", lambda: None)
    monkeypatch.setattr(autonomous_book, "BudgetedGenerator", ContractGenerator)
    monkeypatch.setattr(autonomous_book, "TypeSafeDecisionProvider", lambda **kwargs: object())
    monkeypatch.setattr(autonomous_book, "BudgetedDecisions", _FakeBudgetedDecisions)
    result = autonomous_book.run_unattended_book(story, tmp_path / "out")
    assert result["status"] == "delivered"
    assert (tmp_path / "out/delivery/book.txt").read_text() == "Chapter 1\n\nAaron speaks."
    prompt = next(p for op, p in captured if op == "contextual")
    assert "TERM 甲 -> A [" not in prompt  # source-memory choice must not compete
    assert "Unrelated" not in prompt
    config = json.loads((tmp_path / "out/config.json").read_text())
    assert config["glossary"]["sha256"] == result["glossary"]["sha256"]
    assert len(result["glossary"]["sha256"]) == 64


def test_missing_glossary_fails_before_credentials_or_run_creation(tmp_path: Path, monkeypatch) -> None:
    import pytest
    story = _story_yaml(tmp_path)
    (tmp_path / "terms.txt").unlink()
    def no_credentials():
        raise AssertionError("Must validate local inputs before accessing providers")
    monkeypatch.setattr(autonomous_book, "load_credentials", no_credentials)
    with pytest.raises(FileNotFoundError):
        autonomous_book.run_unattended_book(story, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_malformed_glossary_fails_preflight(tmp_path: Path, monkeypatch) -> None:
    import pytest
    story = _story_yaml(tmp_path)
    (tmp_path / "terms.txt").write_text("not a glossary line", encoding="utf-8")
    monkeypatch.setattr(autonomous_book, "load_credentials", lambda: None)
    with pytest.raises(ValueError, match="glossary"):
        autonomous_book.run_unattended_book(story, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_duplicate_glossary_terms_fail_before_run_creation(tmp_path: Path) -> None:
    import pytest
    story = _story_yaml(tmp_path)
    (tmp_path / "terms.txt").write_text("甲 -> A\n甲 -> Aaron", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate source"):
        autonomous_book.run_unattended_book(story, tmp_path / "out")
    assert not (tmp_path / "out").exists()
