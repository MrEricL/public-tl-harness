"""Small deterministic Harness v3 contract evaluations.

The benchmark intentionally exercises the public session APIs with synthetic
providers.  It compares transport and exposure contracts under fixed fixtures;
it is not a model-quality or live-provider latency benchmark.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .agent_models import AgentAction
from .agent_provider import AgentActionRequest
from .agent_session import AgentSessionResult, resume_repair_session, run_repair_session
from .agent_tools import AGENT_TOOL_REGISTRY, ToolCall
from .glossary import parse_glossary_text, load_glossary
from .harness_demo import synthetic_terminology_resolver
from .story import prepare_run_dir


DISCLAIMER = (
    "This compares harness contracts and exposure under synthetic fixtures; "
    "it does not measure model quality or live-provider latency."
)
CASE_ORDER = ("repair_verified", "promotion_approved", "promotion_rejected")
VARIANT_CONFIG: dict[str, tuple[Literal["json_prompt", "native_function"], bool]] = {
    "prompt_json_all_tools": ("json_prompt", False),
    "native_all_tools": ("native_function", False),
    "native_dynamic_tools": ("native_function", True),
}


class HarnessEvalAction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    tool: str = Field(min_length=1, max_length=64)
    arguments: dict[str, Any] = Field(default_factory=dict)


class HarnessEvalExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    session_status: str = Field(min_length=1)
    final_status: str = Field(min_length=1)
    final_finding_count: int = Field(ge=0)
    glossary_write_count: int = Field(default=0, ge=0)
    approval_outcome: Literal["approved", "rejected"] | None = None


class HarnessEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    case: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=1000)
    source_text: str = Field(min_length=1)
    translated_text: str = Field(min_length=1)
    glossary: str = ""
    canonical_glossary: str = ""
    actions: list[HarnessEvalAction] = Field(min_length=1, max_length=16)
    expect: HarnessEvalExpectation


class HarnessEvalSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    suite: str = Field(min_length=1, max_length=120)
    disclaimer: str = DISCLAIMER
    cases: list[HarnessEvalCase] = Field(min_length=1, max_length=16)


class HarnessEvalResult(BaseModel):
    """Stable per-case/variant contract measurements."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        strict=True,
    )

    case: str
    variant: str
    passed: bool = Field(alias="pass")
    tool_protocol: Literal["json_prompt", "native_function"]
    dynamic_tools: bool
    session_status: str
    final_status: str | None = None
    step_count: int = Field(ge=0)
    first_exposed_tool_count: int = Field(ge=0)
    max_exposed_tool_count: int = Field(ge=0)
    first_request_schema_bytes: int = Field(ge=0)
    observation_kinds: dict[str, int] = Field(default_factory=dict)
    rejection_count: int = Field(ge=0)
    approval_outcome: Literal["approved", "rejected"] | None = None
    final_finding_count: int = Field(ge=0)
    glossary_write_count: int = Field(ge=0)
    artifact_path: str = Field(min_length=1)


class HarnessEvalVariantSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    variant: str
    passed: int = Field(ge=0)
    total: int = Field(ge=0)
    verified: int = Field(ge=0)
    rejected: int = Field(ge=0)
    mean_first_schema_bytes: float = Field(ge=0)
    mean_steps: float = Field(ge=0)


class HarnessEvalReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    suite: str
    disclaimer: str = DISCLAIMER
    results: list[HarnessEvalResult]
    summaries: list[HarnessEvalVariantSummary]


class ScriptedFixtureProvider:
    """Script one deterministic logical action through either transport."""

    provider_name = "synthetic_harness_fixture"

    def __init__(
        self,
        *,
        actions: list[HarnessEvalAction],
        tool_protocol: Literal["json_prompt", "native_function"],
        variant: str,
    ) -> None:
        self.actions = list(actions)
        self.tool_protocol = tool_protocol
        self.model_name = f"harness-v3-{variant}"
        self.requests: list[AgentActionRequest] = []
        # The public session runtime reads this optional field for provider
        # evidence.  Synthetic fixtures deliberately never append network
        # records.
        self.call_records: list[Any] = []

    def next_action(self, request: AgentActionRequest) -> AgentAction:
        self.requests.append(request.model_copy(deep=True))
        index = request.step_number - 1
        if index < 0 or index >= len(self.actions):
            raise RuntimeError(f"fixture has no action for step {request.step_number}")
        scripted = self.actions[index]
        spec = AGENT_TOOL_REGISTRY.spec(scripted.tool)
        if self.tool_protocol == "json_prompt":
            payload = {"tool": scripted.tool, **scripted.arguments}
            call = ToolCall.from_json_action(payload)
        else:
            call = ToolCall.from_native(
                name=spec.provider_name,
                arguments=json.dumps(scripted.arguments, ensure_ascii=False, sort_keys=True),
            )
        visible_names = request.exposed_tool_names or tuple(
            item.name for item in AGENT_TOOL_REGISTRY.visible_specs()
        )
        return AGENT_TOOL_REGISTRY.action_from_call(call, visible_names=visible_names)


def load_suite(path: str | Path) -> HarnessEvalSuite:
    """Load and validate a local deterministic fixture suite."""

    suite_path = Path(path).expanduser()
    data = json.loads(suite_path.read_text(encoding="utf-8"))
    suite = HarnessEvalSuite.model_validate(data)
    case_names = [item.case for item in suite.cases]
    if len(case_names) != len(set(case_names)):
        raise ValueError("harness eval suite contains duplicate case names")
    unknown = set(case_names) - set(CASE_ORDER)
    missing = set(CASE_ORDER) - set(case_names)
    if unknown or missing:
        raise ValueError(
            "harness v3 suite must declare exactly repair_verified, "
            "promotion_approved, and promotion_rejected"
        )
    return suite.model_copy(update={"cases": sorted(suite.cases, key=lambda item: CASE_ORDER.index(item.case))})


def _prepare_output_dir(path: str | Path, *, overwrite: bool) -> Path:
    selected = Path(path).expanduser()
    # Keep the destructive boundary to one named run directory.  In
    # particular, ``--out .`` and filesystem roots are never valid benchmark
    # targets, and a populated directory may only be overwritten when it is a
    # prior benchmark output (identified by its report marker).
    if not selected.name or selected.name in {".", ".."}:
        raise ValueError("--out must name a dedicated output directory")
    if selected.is_symlink():
        raise ValueError(f"Refusing to overwrite output symlink: {selected}")
    selected_resolved = selected.resolve()
    if selected_resolved == Path.cwd().resolve() or selected_resolved.parent == selected_resolved:
        raise ValueError("--out must be a dedicated child directory")
    if selected.exists() and not selected.is_dir():
        raise ValueError(f"Refusing to use non-directory output path: {selected}")
    if selected.exists() and any(selected.iterdir()) and overwrite:
        if not (selected / "harness_eval.json").is_file():
            raise ValueError(
                "--overwrite is limited to an existing harness benchmark output directory"
            )
    return prepare_run_dir(selected.parent, selected.name, overwrite=overwrite)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def report_json(report: HarnessEvalReport) -> str:
    """Return the exact stable JSON representation written to disk."""

    return json.dumps(
        report.model_dump(mode="json", by_alias=True),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"


def _schema_bytes(request: AgentActionRequest) -> int:
    payload = request.canonical_payload()
    return len(
        json.dumps(
            payload.get("tool_schema", []),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _approval_outcome(result: AgentSessionResult) -> Literal["approved", "rejected"] | None:
    for event in result.events:
        if event.event_type != "approval_decided":
            continue
        decision = event.payload.get("decision")
        if decision in {"approved", "rejected"}:
            return decision  # type: ignore[return-value]
    return None


def _rejection_count(result: AgentSessionResult, observations: Counter[str]) -> int:
    observed = sum(count for kind, count in observations.items() if "rejected" in kind)
    approval_rejections = sum(
        1 for event in result.events if event.event_type == "glossary_promotion_rejected"
    )
    # Tool rejection observations already represent their corresponding event;
    # approval rejection has no AgentStep observation and is counted here.
    return observed + approval_rejections


def _run_case_variant(
    *,
    suite_case: HarnessEvalCase,
    variant: str,
    output_dir: Path,
) -> HarnessEvalResult:
    tool_protocol, dynamic_tools = VARIANT_CONFIG[variant]
    artifact_dir = output_dir / "artifacts" / variant / suite_case.case
    artifact_dir.mkdir(parents=True, exist_ok=False)
    canonical_path = artifact_dir / "glossary.txt"
    canonical_path.write_text(suite_case.canonical_glossary or suite_case.glossary, encoding="utf-8")
    glossary = parse_glossary_text(suite_case.glossary)
    provider = ScriptedFixtureProvider(
        actions=suite_case.actions,
        tool_protocol=tool_protocol,
        variant=variant,
    )
    resolver = (
        synthetic_terminology_resolver()
        if any(action.tool == "resolve_terminology" for action in suite_case.actions)
        else None
    )
    result = run_repair_session(
        provider=provider,
        session_dir=artifact_dir,
        source_text=suite_case.source_text,
        translated_text=suite_case.translated_text,
        glossary=glossary,
        canonical_glossary_path=canonical_path,
        run_id=f"harness_eval_{variant}_{suite_case.case}",
        story_slug="harness_eval_v3",
        chapter="0001",
        provider_mode="synthetic_fixture",
        max_steps=max(len(suite_case.actions) + 2, 6),
        max_patch_attempts=2,
        dynamic_tools=dynamic_tools,
        terminology_resolver=resolver,
    )
    if result.snapshot.status == "awaiting_approval":
        decision = suite_case.expect.approval_outcome
        if decision is None:
            raise ValueError(f"{suite_case.case} paused without an expected approval decision")
        result = resume_repair_session(
            session_dir=artifact_dir,
            provider=provider,
            source_text=suite_case.source_text,
            glossary=glossary,
            canonical_glossary_path=canonical_path,
            terminology_resolver=resolver,
            run_id=result.episode.run_id,
            story_slug=result.episode.story_slug,
            chapter=result.episode.chapter,
            provider_mode=result.episode.provider_mode,
            decision=decision,
            reviewer="harness-eval",
            note=f"Synthetic {decision} fixture decision.",
        )

    (artifact_dir / "translated_final.txt").write_text(result.final_text, encoding="utf-8")
    _write_json(
        artifact_dir / "requests.json",
        [request.canonical_payload() for request in provider.requests],
    )
    observations = Counter(step.observation.kind for step in result.episode.steps)
    first_request = provider.requests[0] if provider.requests else None
    exposed_counts = [
        len(request.exposed_tool_names or ()) for request in provider.requests
    ]
    if result.snapshot.exposed_tool_names:
        exposed_counts.append(len(result.snapshot.exposed_tool_names))
    first_exposed = len(first_request.exposed_tool_names or ()) if first_request else 0
    first_schema = _schema_bytes(first_request) if first_request else 0
    glossary_writes = sum(
        1
        for event in result.events
        if event.event_type == "glossary_promotion_applied"
        and bool(event.payload.get("wrote"))
    )
    approval = _approval_outcome(result)
    final_findings = result.final_qa.summary.total_findings
    expected = suite_case.expect
    passed = (
        result.snapshot.status == expected.session_status
        and result.episode.final_status == expected.final_status
        and final_findings == expected.final_finding_count
        and glossary_writes == expected.glossary_write_count
        and approval == expected.approval_outcome
    )
    return HarnessEvalResult(
        case=suite_case.case,
        variant=variant,
        **{"pass": passed},
        tool_protocol=provider.requests[0].tool_protocol if provider.requests else tool_protocol,
        dynamic_tools=dynamic_tools,
        session_status=result.snapshot.status,
        final_status=result.episode.final_status,
        step_count=len(result.episode.steps),
        first_exposed_tool_count=first_exposed,
        max_exposed_tool_count=max(exposed_counts, default=0),
        first_request_schema_bytes=first_schema,
        observation_kinds=dict(sorted(observations.items())),
        rejection_count=_rejection_count(result, observations),
        approval_outcome=approval,
        final_finding_count=final_findings,
        glossary_write_count=glossary_writes,
        artifact_path=artifact_dir.relative_to(output_dir).as_posix(),
    )


def run_harness_eval(
    suite: HarnessEvalSuite,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> HarnessEvalReport:
    """Run every fixed case under every fixed transport/exposure variant."""

    selected_suite = suite if isinstance(suite, HarnessEvalSuite) else HarnessEvalSuite.model_validate(suite)
    output = _prepare_output_dir(output_dir, overwrite=overwrite)
    results: list[HarnessEvalResult] = []
    for variant in VARIANT_CONFIG:
        for suite_case in selected_suite.cases:
            results.append(
                _run_case_variant(
                    suite_case=suite_case,
                    variant=variant,
                    output_dir=output,
                )
            )
    summaries: list[HarnessEvalVariantSummary] = []
    for variant in VARIANT_CONFIG:
        variant_results = [item for item in results if item.variant == variant]
        summaries.append(
            HarnessEvalVariantSummary(
                variant=variant,
                passed=sum(item.passed for item in variant_results),
                total=len(variant_results),
                verified=sum(item.final_status == "verified" for item in variant_results),
                rejected=sum(item.approval_outcome == "rejected" for item in variant_results),
                mean_first_schema_bytes=round(
                    sum(item.first_request_schema_bytes for item in variant_results)
                    / max(len(variant_results), 1),
                    2,
                ),
                mean_steps=round(
                    sum(item.step_count for item in variant_results) / max(len(variant_results), 1),
                    2,
                ),
            )
        )
    report = HarnessEvalReport(
        suite=selected_suite.suite,
        disclaimer=selected_suite.disclaimer or DISCLAIMER,
        results=results,
        summaries=summaries,
    )
    (output / "harness_eval.json").write_text(report_json(report), encoding="utf-8")
    return report


__all__ = [
    "CASE_ORDER",
    "DISCLAIMER",
    "HarnessEvalAction",
    "HarnessEvalCase",
    "HarnessEvalExpectation",
    "HarnessEvalReport",
    "HarnessEvalResult",
    "HarnessEvalSuite",
    "HarnessEvalVariantSummary",
    "ScriptedFixtureProvider",
    "VARIANT_CONFIG",
    "load_suite",
    "report_json",
    "run_harness_eval",
]
