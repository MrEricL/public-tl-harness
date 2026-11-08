"""Typed, deterministic tool contracts for Harness v3.

The registry is deliberately small and boring: action models remain the source
of truth for validation, while this module supplies transport normalization,
provider representations, metadata, exposure checks, and deterministic search.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Collection, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .agent_models import (
    AgentAction,
    AgentActionBase,
    CompleteReviewAction,
    DelegateReviewAction,
    EscalateAction,
    FinishAction,
    GetQAFindingsAction,
    LookupGlossaryAction,
    NormalizePunctuationAction,
    PromoteGlossaryTermAction,
    ReadParagraphsAction,
    ReadSourceContextAction,
    ReadTranslationContextAction,
    ResolveTerminologyAction,
    SearchToolsAction,
    SelectTermAction,
    SubmitPatchAction,
)


SHOWCASE_TOOL_SCHEMA_VERSION = "agent-tools.v4"


class ToolCallValidationError(ValueError):
    """Raised when a tool call cannot be normalized or validated."""


class ToolCall(BaseModel):
    """One normalized logical/native tool invocation.

    ``name`` is the transport name as received.  A registry resolves a native
    alias to the logical action name before validating the arguments.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1, max_length=64)
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = Field(default=None, min_length=1, max_length=200)

    @classmethod
    def from_json_action(
        cls,
        action: Mapping[str, Any] | BaseModel | str,
        *,
        call_id: str | None = None,
    ) -> "ToolCall":
        """Normalize an action-shaped object (or JSON object) to ``ToolCall``."""

        try:
            if isinstance(action, str):
                action = json.loads(action)
            if isinstance(action, BaseModel):
                payload = action.model_dump(mode="python")
            elif isinstance(action, Mapping):
                payload = dict(action)
            else:
                raise ToolCallValidationError("JSON action must be an object")

            if "tool" not in payload:
                raise ToolCallValidationError("JSON action is missing tool")
            name = payload.pop("tool")
            if "arguments" in payload:
                arguments = payload.pop("arguments")
                if payload:
                    raise ToolCallValidationError(
                        "JSON action must not mix nested arguments with flattened fields"
                    )
                if not isinstance(arguments, Mapping):
                    raise ToolCallValidationError("JSON action arguments must be an object")
                payload = dict(arguments)
            return cls.model_validate(
                {"name": name, "arguments": payload, "call_id": call_id}
            )
        except ToolCallValidationError:
            raise
        except (JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
            raise ToolCallValidationError("Invalid JSON tool action") from exc

    @classmethod
    def from_native(
        cls,
        *,
        name: str,
        arguments: str | Mapping[str, Any],
        call_id: str | None = None,
    ) -> "ToolCall":
        """Normalize a native function call with JSON arguments."""

        try:
            if isinstance(arguments, str):
                parsed = json.loads(arguments)
            elif isinstance(arguments, Mapping):
                parsed = dict(arguments)
            else:
                raise ToolCallValidationError("Native tool arguments must be a JSON object")
            if not isinstance(parsed, dict):
                raise ToolCallValidationError("Native tool arguments must be a JSON object")
            return cls.model_validate({"name": name, "arguments": parsed, "call_id": call_id})
        except ToolCallValidationError:
            raise
        except (JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
            raise ToolCallValidationError("Invalid native tool arguments") from exc

    def as_action_payload(self, *, logical_name: str | None = None) -> dict[str, Any]:
        """Return the action-shaped payload consumed by the action models."""

        if "tool" in self.arguments:
            raise ToolCallValidationError("Tool call arguments may not include tool")
        payload = dict(self.arguments)
        payload["tool"] = logical_name if logical_name is not None else self.name
        return payload


_SideEffect = Literal["none", "working_copy", "persistent"]
_Risk = Literal["low", "medium", "high"]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Immutable registry metadata and provider representations for an action.

    Derived schemas, provider aliases, discovery tags, and
    ``requires_approval`` are operational.  ``side_effect``, ``risk``, and
    ``timeout_seconds`` are descriptive metadata only.
    """

    action_model: type[AgentActionBase]
    description: str
    side_effect: _SideEffect
    risk: _Risk
    timeout_seconds: float
    tags: tuple[str, ...]
    requires_approval: bool = False
    native_name: str | None = None

    @property
    def name(self) -> str:
        """Logical action name derived from the model's literal default."""

        field = self.action_model.model_fields.get("tool")
        default = None if field is None else field.default
        if not isinstance(default, str) or not default:
            raise ValueError("Tool action models must define a non-empty tool default")
        return default

    @property
    def logical_name(self) -> str:
        return self.name

    @property
    def provider_name(self) -> str:
        return self.native_name or self.name

    def argument_schema(self) -> dict[str, Any]:
        """Return the action model's strict argument schema without ``tool``."""

        schema = copy.deepcopy(self.action_model.model_json_schema())
        properties = schema.get("properties")
        if isinstance(properties, dict):
            properties.pop("tool", None)
        required = schema.get("required")
        if isinstance(required, list):
            schema["required"] = [name for name in required if name != "tool"]
            if not schema["required"]:
                schema.pop("required", None)
        # Action models currently all forbid extras.  Keep this explicit here so
        # registry-generated provider schemas remain strict if a model changes.
        schema["additionalProperties"] = False
        return schema

    @property
    def arguments_schema(self) -> dict[str, Any]:
        """Property alias useful to callers that prefer noun-style access."""

        return self.argument_schema()

    def prompt_contract(self) -> dict[str, Any]:
        """Return the legacy-aligned JSON-in-prompt tool contract."""

        return {
            "tool": self.name,
            "description": self.description,
            "arguments": self.argument_schema(),
        }

    def chat_tool(self) -> dict[str, Any]:
        """Return an OpenAI-compatible native function tool definition."""

        return {
            "type": "function",
            "function": {
                "name": self.provider_name,
                "description": self.description,
                "parameters": self.argument_schema(),
            },
        }

    @property
    def search_tokens(self) -> frozenset[str]:
        return frozenset(
            token
            for value in (self.name, self.description, *self.tags)
            for token in _tokenize(value)
        )


def _tokenize(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", value.casefold()))


class ToolRegistry:
    """Validated collection of logical tools and provider-native aliases."""

    def __init__(self, specs: Collection[ToolSpec] = ()) -> None:
        self._specs = tuple(specs)
        self._by_name: dict[str, ToolSpec] = {}
        self._by_native_name: dict[str, ToolSpec] = {}
        for spec in self._specs:
            if not isinstance(spec, ToolSpec):
                raise TypeError("ToolRegistry specs must be ToolSpec instances")
            if spec.name in self._by_name:
                raise ValueError(f"duplicate logical tool name: {spec.name}")
            self._by_name[spec.name] = spec

        for spec in self._specs:
            native_name = spec.provider_name
            if native_name in self._by_native_name:
                raise ValueError(f"duplicate native tool name: {native_name}")
            self._by_native_name[native_name] = spec
            # A native alias may not shadow another logical name.  A tool's own
            # logical/native identity is valid and already represented above.
            owner = self._by_name.get(native_name)
            if owner is not None and owner is not spec:
                raise ValueError(f"duplicate logical/native tool name: {native_name}")

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self):
        return iter(self.visible_specs())

    def spec(self, name: str) -> ToolSpec:
        """Look up a tool by logical or native name."""

        spec = self._by_name.get(name)
        if spec is None:
            spec = self._by_native_name.get(name)
        if spec is None:
            raise ToolCallValidationError(f"Unknown tool: {name}")
        return spec

    def _logical_name(self, name: str) -> str:
        return self.spec(name).name

    def visible_specs(self, names: Collection[str] | None = None) -> tuple[ToolSpec, ...]:
        """Return exposed specs in deterministic logical-name order."""

        if names is None:
            selected = set(self._by_name)
        else:
            selected = {self._logical_name(name) for name in names}
        return tuple(self._by_name[name] for name in sorted(selected))

    def contract_payload(self) -> list[dict[str, Any]]:
        """Return the complete deterministic registry contract payload.

        Every exposed tool is represented in logical-name order.  The payload
        intentionally includes both provider-facing identity and operational
        metadata so changes to any contract-relevant field invalidate the
        registry digest used by durable session identity.
        """

        return [
            {
                "name": spec.name,
                "provider_name": spec.provider_name,
                "description": spec.description,
                "argument_schema": spec.argument_schema(),
                "tags": list(spec.tags),
                "requires_approval": spec.requires_approval,
                "side_effect": spec.side_effect,
                "risk": spec.risk,
                "timeout_seconds": spec.timeout_seconds,
            }
            for spec in self.visible_specs()
        ]

    def contract_sha256(self) -> str:
        """Return the SHA-256 digest of the canonical registry contract JSON."""

        encoded = json.dumps(
            self.contract_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def action_from_call(
        self,
        call: ToolCall,
        *,
        visible_names: Collection[str],
    ) -> AgentAction:
        """Resolve exposure and validate one call against its specific model."""

        try:
            normalized_call = call if isinstance(call, ToolCall) else ToolCall.model_validate(call)
        except ValidationError as exc:
            raise ToolCallValidationError("Invalid tool call") from exc

        logical_name = self._logical_name(normalized_call.name)
        visible_logical_names = {self._logical_name(name) for name in visible_names}
        if logical_name not in visible_logical_names:
            raise ToolCallValidationError(f"Tool was not exposed: {normalized_call.name}")

        spec = self._by_name[logical_name]
        try:
            return spec.action_model.model_validate(
                normalized_call.as_action_payload(logical_name=logical_name)
            )
        except ValidationError as exc:
            raise ToolCallValidationError(
                f"Tool call validation failed for {logical_name}"
            ) from exc

    def search(self, query: str, *, limit: int = 8) -> tuple[ToolSpec, ...]:
        """Search trusted metadata using token overlap and lexical ordering."""

        if not isinstance(query, str):
            raise ToolCallValidationError("Tool search query must be a string")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ToolCallValidationError("Tool search limit must be an integer")
        if limit <= 0:
            return ()
        tokens = frozenset(_tokenize(query))
        if not tokens:
            return ()
        matches = [spec for spec in self._specs if tokens & spec.search_tokens]
        return tuple(sorted(matches, key=lambda spec: spec.name)[: min(limit, 16)])


def _spec(
    action_model: type[AgentActionBase],
    description: str,
    *,
    side_effect: _SideEffect,
    risk: _Risk,
    timeout_seconds: float,
    tags: tuple[str, ...],
    requires_approval: bool = False,
    native_name: str | None = None,
) -> ToolSpec:
    return ToolSpec(
        action_model=action_model,
        description=description,
        side_effect=side_effect,
        risk=risk,
        timeout_seconds=timeout_seconds,
        tags=tags,
        requires_approval=requires_approval,
        native_name=native_name,
    )


AGENT_TOOL_SPECS: tuple[ToolSpec, ...] = (
    _spec(
        GetQAFindingsAction,
        "Read the current deterministic QA findings.",
        side_effect="none",
        risk="low",
        timeout_seconds=5.0,
        tags=("qa", "read", "findings"),
    ),
    _spec(
        ReadSourceContextAction,
        "Read source context around one QA finding.",
        side_effect="none",
        risk="low",
        timeout_seconds=5.0,
        tags=("context", "source", "read"),
    ),
    _spec(
        ReadTranslationContextAction,
        "Read translated context around one QA finding.",
        side_effect="none",
        risk="low",
        timeout_seconds=5.0,
        tags=("context", "translation", "read"),
    ),
    _spec(
        LookupGlossaryAction,
        "Look up a trusted canonical glossary term.",
        side_effect="none",
        risk="low",
        timeout_seconds=5.0,
        tags=("glossary", "terminology", "read"),
    ),
    _spec(
        ResolveTerminologyAction,
        "Ask the terminology resolver to arbitrate one source term.",
        side_effect="working_copy",
        risk="medium",
        timeout_seconds=30.0,
        tags=("terminology", "glossary", "consensus"),
    ),
    _spec(
        SubmitPatchAction,
        "Propose a bounded working-copy translation patch.",
        side_effect="working_copy",
        risk="medium",
        timeout_seconds=15.0,
        tags=("translation", "patch", "working-copy"),
    ),
    _spec(
        NormalizePunctuationAction,
        "Normalize Chinese punctuation to English punctuation in the working copy.",
        side_effect="working_copy",
        risk="medium",
        timeout_seconds=15.0,
        tags=("translation", "punctuation", "normalize", "working-copy"),
    ),
    _spec(
        EscalateAction,
        "Escalate the episode for human review.",
        side_effect="none",
        risk="low",
        timeout_seconds=5.0,
        tags=("review", "terminal"),
    ),
    _spec(
        FinishAction,
        "Finish the episode after deterministic QA is satisfied.",
        side_effect="none",
        risk="low",
        timeout_seconds=5.0,
        tags=("terminal", "qa"),
    ),
    _spec(
        SearchToolsAction,
        "Search trusted tool names, descriptions, and tags.",
        side_effect="none",
        risk="low",
        timeout_seconds=5.0,
        tags=("discovery", "search", "tools"),
        native_name="tools_search",
    ),
    _spec(
        PromoteGlossaryTermAction,
        "Prepare a proposed canonical glossary promotion for approval.",
        side_effect="persistent",
        risk="high",
        timeout_seconds=20.0,
        tags=("glossary", "terminology", "promotion", "persistent"),
        requires_approval=True,
    ),
)

AGENT_TOOL_REGISTRY = ToolRegistry(AGENT_TOOL_SPECS)

# Showcase v4 keeps the canonical v3 registry object above unchanged.  The
# additional coordinator actions are opt-in so v1-v3 payloads and exposure
# behavior retain their historical identities.
SHOWCASE_TOOL_SPECS: tuple[ToolSpec, ...] = AGENT_TOOL_SPECS + (
    _spec(
        ReadParagraphsAction,
        "Read a bounded indexed window of source or translated paragraphs.",
        side_effect="none",
        risk="low",
        timeout_seconds=5.0,
        tags=("context", "paragraph", "source", "translation", "read"),
    ),
    _spec(
        DelegateReviewAction,
        "Delegate one bounded batch of terminology or source-fidelity specialist reviews.",
        side_effect="none",
        risk="medium",
        timeout_seconds=60.0,
        tags=("review", "delegation", "specialist", "terminology", "fidelity"),
    ),
    _spec(
        SelectTermAction,
        "Select an exact terminology suggestion into the episode-local glossary.",
        side_effect="working_copy",
        risk="medium",
        timeout_seconds=15.0,
        tags=("terminology", "glossary", "selection", "working-copy"),
    ),
)

SHOWCASE_TOOL_REGISTRY = ToolRegistry(SHOWCASE_TOOL_SPECS)

# Child agents receive a deliberately smaller surface.  They can inspect
# bounded context and report evidence, but cannot mutate the coordinator's
# working translation or persist glossary changes.
SHOWCASE_CHILD_TOOL_SPECS: tuple[ToolSpec, ...] = (
    next(spec for spec in SHOWCASE_TOOL_SPECS if spec.name == "read_paragraphs"),
    next(spec for spec in AGENT_TOOL_SPECS if spec.name == "lookup_glossary"),
    _spec(
        CompleteReviewAction,
        "Return one bounded specialist review with evidence and proposed changes.",
        side_effect="none",
        risk="low",
        timeout_seconds=15.0,
        tags=("review", "specialist", "result", "evidence"),
    ),
)
SHOWCASE_CHILD_TOOL_REGISTRY = ToolRegistry(SHOWCASE_CHILD_TOOL_SPECS)
# Friendly alias for callers that refer to delegated workers as specialists.
SPECIALIST_TOOL_REGISTRY = SHOWCASE_CHILD_TOOL_REGISTRY


__all__ = [
    "AGENT_TOOL_REGISTRY",
    "AGENT_TOOL_SPECS",
    "SHOWCASE_CHILD_TOOL_REGISTRY",
    "SHOWCASE_CHILD_TOOL_SPECS",
    "SHOWCASE_TOOL_REGISTRY",
    "SHOWCASE_TOOL_SCHEMA_VERSION",
    "SHOWCASE_TOOL_SPECS",
    "SPECIALIST_TOOL_REGISTRY",
    "ToolCall",
    "ToolCallValidationError",
    "ToolRegistry",
    "ToolSpec",
]
