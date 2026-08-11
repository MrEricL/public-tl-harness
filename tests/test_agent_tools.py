from __future__ import annotations

from dataclasses import replace
import json

import pytest
from pydantic import Field, ValidationError

from agentic_translation.agent_models import (
    FinishAction,
    NormalizePunctuationAction,
    PromoteGlossaryTermAction,
    SubmitPatchAction,
)
from agentic_translation.agent_tools import (
    AGENT_TOOL_REGISTRY,
    AGENT_TOOL_SPECS,
    ToolCall,
    ToolCallValidationError,
    ToolRegistry,
    ToolSpec,
)


class _FinishWithReceiptAction(FinishAction):
    """Test-only action model used to prove argument schema affects identity."""

    receipt: str = Field(min_length=1)


def test_default_registry_has_one_spec_per_action_and_typed_metadata() -> None:
    specs = AGENT_TOOL_REGISTRY.visible_specs()
    assert len(specs) == 11
    assert {spec.name for spec in specs} == {
        "get_qa_findings",
        "read_source_context",
        "read_translation_context",
        "lookup_glossary",
        "resolve_terminology",
        "submit_patch",
        "normalize_punctuation",
        "escalate",
        "finish",
        "tools.search",
        "promote_glossary_term",
    }
    promotion = AGENT_TOOL_REGISTRY.spec("promote_glossary_term")
    assert promotion.side_effect == "persistent"
    assert promotion.risk == "high"
    assert promotion.requires_approval is True
    assert AGENT_TOOL_REGISTRY.spec("tools.search").native_name == "tools_search"


def test_tool_spec_docstring_classifies_operational_and_descriptive_metadata() -> None:
    docstring = ToolSpec.__doc__ or ""

    assert "schemas" in docstring
    assert "aliases" in docstring
    assert "tags" in docstring
    assert "requires_approval" in docstring
    assert "operational" in docstring
    assert "side_effect" in docstring
    assert "risk" in docstring
    assert "timeout_seconds" in docstring
    assert "descriptive metadata only" in docstring


def test_spec_schema_and_provider_contract_omit_action_discriminator() -> None:
    spec = AGENT_TOOL_REGISTRY.spec("submit_patch")
    schema = spec.argument_schema()

    assert "tool" not in schema.get("properties", {})
    assert "tool" not in schema.get("required", [])
    assert schema["additionalProperties"] is False
    assert spec.prompt_contract() == {
        "tool": "submit_patch",
        "description": spec.description,
        "arguments": schema,
    }
    assert spec.chat_tool() == {
        "type": "function",
        "function": {
            "name": "submit_patch",
            "description": spec.description,
            "parameters": schema,
        },
    }
    assert "old_text" not in schema["properties"]
    assert "new_text" not in schema["properties"]
    assert "edits" in schema["properties"]


def test_normalize_punctuation_registry_spec_is_working_copy() -> None:
    spec = AGENT_TOOL_REGISTRY.spec("normalize_punctuation")
    assert spec.action_model is NormalizePunctuationAction
    assert spec.side_effect == "working_copy"
    assert spec.risk == "medium"


def test_json_and_native_calls_normalize_and_validate_through_registry() -> None:
    json_call = ToolCall.from_json_action(
        {"tool": "submit_patch", "old_text": "x", "new_text": "y", "rationale": "fix"}
    )
    native_call = ToolCall.from_native(
        name="submit_patch",
        arguments='{"old_text":"x","new_text":"y","rationale":"fix"}',
    )

    assert json_call == native_call
    action = AGENT_TOOL_REGISTRY.action_from_call(json_call, visible_names={"submit_patch"})
    assert isinstance(action, SubmitPatchAction)
    assert action.new_text == "y"
    assert json_call.as_action_payload() == {
        "tool": "submit_patch",
        "old_text": "x",
        "new_text": "y",
        "rationale": "fix",
    }


def test_native_alias_resolves_and_hidden_or_invalid_calls_are_rejected() -> None:
    alias_call = ToolCall.from_native(name="tools_search", arguments='{"query":"glossary"}')
    action = AGENT_TOOL_REGISTRY.action_from_call(alias_call, visible_names={"tools.search"})
    assert action.tool == "tools.search"

    with pytest.raises(ToolCallValidationError, match="not exposed"):
        AGENT_TOOL_REGISTRY.action_from_call(
            ToolCall(name="finish", arguments={"summary": "done"}),
            visible_names={"get_qa_findings"},
        )
    with pytest.raises(ToolCallValidationError, match="validation"):
        AGENT_TOOL_REGISTRY.action_from_call(
            ToolCall(name="finish", arguments={"summary": "done", "extra": True}),
            visible_names={"finish"},
        )


def test_tool_call_helpers_reject_malformed_inputs() -> None:
    with pytest.raises(ToolCallValidationError):
        ToolCall.from_json_action({"summary": "missing discriminator"})
    with pytest.raises(ToolCallValidationError):
        ToolCall.from_native(name="finish", arguments="not json")
    with pytest.raises(ToolCallValidationError):
        ToolCall.from_native(name="finish", arguments=json.dumps(["not", "object"]))


def test_search_is_deterministic_lexical_and_bounded() -> None:
    names = [spec.name for spec in AGENT_TOOL_REGISTRY.search("terminology glossary", limit=3)]
    assert names == sorted(names)
    assert "resolve_terminology" in names
    assert len(names) <= 3
    assert AGENT_TOOL_REGISTRY.search("zzzz-unmatched-token") == ()


def test_registry_contract_payload_is_ordered_and_contains_full_tool_metadata() -> None:
    registry = ToolRegistry(tuple(reversed(AGENT_TOOL_SPECS)))

    payload = registry.contract_payload()

    assert [entry["name"] for entry in payload] == sorted(
        spec.name for spec in AGENT_TOOL_SPECS
    )
    assert all(
        set(entry) == {
            "name",
            "provider_name",
            "description",
            "argument_schema",
            "tags",
            "requires_approval",
            "side_effect",
            "risk",
            "timeout_seconds",
        }
        for entry in payload
    )
    promotion = next(entry for entry in payload if entry["name"] == "promote_glossary_term")
    promotion_spec = registry.spec("promote_glossary_term")
    assert promotion == {
        "name": promotion_spec.name,
        "provider_name": promotion_spec.provider_name,
        "description": promotion_spec.description,
        "argument_schema": promotion_spec.argument_schema(),
        "tags": list(promotion_spec.tags),
        "requires_approval": promotion_spec.requires_approval,
        "side_effect": promotion_spec.side_effect,
        "risk": promotion_spec.risk,
        "timeout_seconds": promotion_spec.timeout_seconds,
    }


def test_registry_contract_hash_is_order_independent_and_metadata_sensitive() -> None:
    ordered = ToolRegistry(AGENT_TOOL_SPECS)
    reversed_registry = ToolRegistry(tuple(reversed(AGENT_TOOL_SPECS)))

    assert ordered.contract_payload() == reversed_registry.contract_payload()
    assert ordered.contract_sha256() == reversed_registry.contract_sha256()

    finish = ordered.spec("finish")
    variants = (
        replace(finish, native_name="finish_native"),
        replace(finish, tags=("terminal", "changed")),
        replace(finish, requires_approval=True),
        replace(finish, description="Finish with changed description."),
        replace(finish, action_model=_FinishWithReceiptAction),
    )
    baseline_hash = ToolRegistry((finish,)).contract_sha256()

    assert all(
        ToolRegistry((variant,)).contract_sha256() != baseline_hash
        for variant in variants
    )


def test_registry_rejects_duplicate_logical_and_native_names() -> None:
    finish = ToolSpec(
        action_model=FinishAction,
        description="finish duplicate",
        side_effect="none",
        risk="low",
        timeout_seconds=1.0,
        tags=("duplicate",),
    )
    duplicate_logical = ToolSpec(
        action_model=FinishAction,
        description="finish duplicate 2",
        side_effect="none",
        risk="low",
        timeout_seconds=1.0,
        tags=("duplicate",),
    )
    with pytest.raises(ValueError, match="duplicate"):
        ToolRegistry((finish, duplicate_logical))

    alias = ToolSpec(
        action_model=PromoteGlossaryTermAction,
        description="alias duplicate",
        side_effect="persistent",
        risk="high",
        timeout_seconds=1.0,
        tags=("duplicate",),
        native_name="finish",
    )
    with pytest.raises(ValueError, match="duplicate"):
        ToolRegistry((finish, alias))
