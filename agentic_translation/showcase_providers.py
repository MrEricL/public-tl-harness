"""Provider profiles and deterministic adapters for the translation showcase.

The production providers in :mod:`providers_llm` already implement the
OpenAI-compatible transport, response cache, and replay checks used by the
rest of the project.  This module is a small composition layer around those
providers:

* ``resolve_profile`` turns a named provider into a serialisable, redacted
  runtime profile;
* fixture providers exercise the exact same action/translation cache boundary
  without making a network request; and
* live and replay providers use the ordinary implementation unchanged.

The fixture implementation intentionally overrides only the final transport
call.  Prompt construction, canonical request payloads, native function
normalisation, response cache names, and indexed replay therefore stay on the
same path as a live run.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Literal

from .agent_provider import LLMAgentActionProvider
from .agent_tools import SHOWCASE_TOOL_REGISTRY, ToolCall, ToolRegistry
from .providers_llm import (
    LLMProviderUnavailable,
    LLMTranslationProvider,
)


ProviderMode = Literal["offline", "live", "replay"]


def _elapsed_ms(started_at: float) -> float:
    return max(0.0, (time.perf_counter() - started_at) * 1000.0)


_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "model_env": "AGENTIC_TRANSLATION_MODEL",
        "base_url_env": "OPENAI_BASE_URL",
        "default_base_url": None,
        "default_model": "gpt-4.1-mini",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
    },
}


# These settings are intentionally separate from ``resolve_profile``.  The
# latter is part of the historical showcase/replay contract and must continue
# returning its 2,048-token, 60-second native-function profile.  New
# experiment code opts into one of these phase profiles explicitly and records
# the resulting values in its run identity.
_PHASE_DEFAULTS: dict[str, dict[str, int | float]] = {
    "translation": {
        "max_output_tokens": 16_384,
        "request_timeout_seconds": 180.0,
    },
    "simple_revision": {
        "max_output_tokens": 16_384,
        "request_timeout_seconds": 180.0,
    },
    "memory": {
        "max_output_tokens": 4_096,
        "request_timeout_seconds": 180.0,
    },
    "review": {
        "max_output_tokens": 4_096,
        "request_timeout_seconds": 180.0,
    },
    "action": {
        "max_output_tokens": 2_048,
        "request_timeout_seconds": 180.0,
    },
    "repair": {
        "max_output_tokens": 2_048,
        "request_timeout_seconds": 180.0,
    },
}
_THINKING_MODES = frozenset({"disabled", "enabled"})
_TOOL_PROTOCOLS = frozenset({"json_prompt", "native_function"})


def _json_copy(value: Any, *, label: str) -> Any:
    """Copy JSON-native profile/context data and fail clearly on other values."""

    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only JSON-compatible values") from exc


def resolve_profile(name: str, model: str | None = None) -> dict[str, Any]:
    """Resolve a named OpenAI-compatible provider into an effective profile.

    API keys are represented only by their environment variable names.  The
    returned mapping is safe to persist in a run manifest.  Model precedence
    is explicit argument, provider-specific environment variable, shared
    ``AGENTIC_TRANSLATION_MODEL`` fallback, then the documented default.
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError("provider profile name must be a non-empty string")
    provider = name.strip().casefold()
    if provider not in _PROFILE_DEFAULTS:
        supported = ", ".join(sorted(_PROFILE_DEFAULTS))
        raise ValueError(f"Unknown provider profile {name!r}; expected one of {supported}.")

    config = _PROFILE_DEFAULTS[provider]
    requested_model = model.strip() if isinstance(model, str) else None
    if requested_model == "":
        raise ValueError("model must be a non-empty string when supplied")
    env_model = os.environ.get(str(config["model_env"])) or os.environ.get(
        "AGENTIC_TRANSLATION_MODEL"
    )
    if requested_model:
        effective_model = requested_model
        model_source = "explicit"
    elif env_model and env_model.strip():
        effective_model = env_model.strip()
        model_source = "environment"
    else:
        effective_model = str(config["default_model"])
        model_source = "default"

    profile = {
        "profile": provider,
        "provider": provider,
        "model": effective_model,
        "model_source": model_source,
        "api_key_env": config["api_key_env"],
        "model_env": config["model_env"],
        "base_url_env": config["base_url_env"],
        "default_base_url": config["default_base_url"],
        "tool_protocol": "native_function",
        "max_output_tokens": 2048,
        "temperature": 0.0,
        "request_timeout_seconds": 60.0,
    }
    if (
        provider == "deepseek"
        and model_source == "explicit"
        and effective_model.casefold().startswith("deepseek-v4")
    ):
        # DeepSeek v4 enables thinking by default.  The showcase uses the
        # non-thinking path so every comparison arm retains the same 2K output
        # budget.  Its native required-tool responses are not consistently
        # shaped across turns, so the showcase uses the strict JSON action
        # adapter while retaining native transport as an explicitly testable
        # provider capability.
        profile["extra_body"] = {"thinking": {"type": "disabled"}}
        profile["tool_protocol"] = "json_prompt"
    return profile


def resolve_phase_profile(
    name: str,
    model: str | None = None,
    *,
    phase: str = "translation",
    thinking: str = "disabled",
    temperature: int | float = 0,
    max_output_tokens: int | None = None,
    request_timeout_seconds: int | float | None = None,
    tool_protocol: str = "json_prompt",
) -> dict[str, Any]:
    """Resolve an explicit, phase-specific profile for a new experiment.

    Provider/model selection follows :func:`resolve_profile` exactly, including
    provider-specific environment variables, the shared model fallback, and
    the redacted ``model_source`` field.  Every generation option that differs
    by phase is then written into the returned mapping so callers can persist
    the effective profile with their run identity.

    ``thinking`` is an explicit option rather than a model-name heuristic.  A
    DeepSeek phase profile carries it through the provider's ``extra_body``
    request extension, including for the current ``deepseek-flash`` alias and
    legacy names.  Other providers retain the recorded option without being
    sent a DeepSeek-specific request extension.
    """

    if not isinstance(phase, str) or not phase.strip():
        raise ValueError("phase must be a non-empty string")
    phase_name = phase.strip().casefold()
    if phase_name not in _PHASE_DEFAULTS:
        supported = ", ".join(_PHASE_DEFAULTS)
        raise ValueError(f"Unknown provider phase {phase!r}; expected one of {supported}.")

    if not isinstance(thinking, str) or not thinking.strip():
        raise ValueError("thinking must be a non-empty string")
    thinking_mode = thinking.strip().casefold()
    if thinking_mode not in _THINKING_MODES:
        supported = ", ".join(sorted(_THINKING_MODES))
        raise ValueError(f"thinking must be one of: {supported}")

    if not isinstance(tool_protocol, str) or not tool_protocol.strip():
        raise ValueError("tool_protocol must be a non-empty string")
    protocol = tool_protocol.strip().casefold()
    if protocol not in _TOOL_PROTOCOLS:
        supported = ", ".join(sorted(_TOOL_PROTOCOLS))
        raise ValueError(f"tool_protocol must be one of: {supported}")

    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a finite number")
    if not math.isfinite(float(temperature)) or float(temperature) < 0:
        raise ValueError("temperature must be a finite non-negative number")

    phase_defaults = _PHASE_DEFAULTS[phase_name]
    output_limit = (
        phase_defaults["max_output_tokens"]
        if max_output_tokens is None
        else max_output_tokens
    )
    if isinstance(output_limit, bool) or not isinstance(output_limit, int) or output_limit <= 0:
        raise ValueError("max_output_tokens must be a positive integer")

    timeout = (
        phase_defaults["request_timeout_seconds"]
        if request_timeout_seconds is None
        else request_timeout_seconds
    )
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("request_timeout_seconds must be a finite positive number")
    if not math.isfinite(float(timeout)) or float(timeout) <= 0:
        raise ValueError("request_timeout_seconds must be a finite positive number")

    # Resolve the provider and model through the legacy function to keep
    # explicit-vs-environment precedence identical.  Remove its historical
    # DeepSeek V4 special case before applying this function's explicit option;
    # the new path must not infer capabilities from a model-name prefix.
    profile = resolve_profile(name, model)
    profile.pop("extra_body", None)
    profile.update(
        {
            "phase": phase_name,
            "thinking": thinking_mode,
            "tool_protocol": protocol,
            "temperature": float(temperature),
            "max_output_tokens": output_limit,
            "request_timeout_seconds": float(timeout),
        }
    )
    if profile["provider"] == "deepseek":
        profile["extra_body"] = {"thinking": {"type": thinking_mode}}
    return profile


def fixture_profile() -> dict[str, Any]:
    """Return the redacted profile used by the checked-in offline fixtures."""

    return {
        "profile": "fixture",
        "provider": "fixture",
        "model": "showcase-scripted-v1",
        "model_source": "fixture",
        "tool_protocol": "native_function",
        "max_output_tokens": 2048,
        "temperature": 0.0,
        "request_timeout_seconds": 60.0,
    }


def _profile_value(profile: dict[str, Any], key: str, default: Any = None) -> Any:
    value = profile.get(key, default)
    return default if value is None else value


def _provider_kwargs(profile: dict[str, Any]) -> dict[str, Any]:
    """Build constructor kwargs without persisting or reading secret values."""

    kwargs: dict[str, Any] = {
        "provider_name": str(_profile_value(profile, "provider", "openai")),
        "model_name": str(_profile_value(profile, "model", "")),
        "max_output_tokens": _profile_value(profile, "max_output_tokens", 2048),
        "temperature": _profile_value(profile, "temperature", 0.0),
        "request_timeout_seconds": _profile_value(
            profile, "request_timeout_seconds", 60.0
        ),
    }
    # Explicit profile fields make tests and saved manifests portable even if
    # the environment's provider defaults are changed later.
    for field in ("api_key_env", "model_env", "base_url_env", "default_base_url"):
        if field in profile:
            kwargs[field] = profile[field]
    if "extra_body" in profile:
        kwargs["extra_body"] = _json_copy(profile["extra_body"], label="extra_body")
    return kwargs


def _validate_mode(mode: str) -> ProviderMode:
    if mode not in {"offline", "live", "replay"}:
        raise ValueError("provider mode must be one of: offline, live, replay")
    return mode  # type: ignore[return-value]


class _FixtureActionProvider(LLMAgentActionProvider):
    """Scripted transport that uses the normal action provider pipeline."""

    def __init__(
        self,
        *,
        actions: list[dict[str, Any]] | None,
        registry: ToolRegistry,
        tool_protocol: Literal["json_prompt", "native_function"],
        cache_dir: Path,
    ) -> None:
        self._actions = _json_copy(actions or [], label="actions")
        if not isinstance(self._actions, list):  # pragma: no cover - guarded by factory type.
            raise ValueError("actions must be a list")
        super().__init__(
            provider_mode="live",
            cache_dir=cache_dir,
            record_cache=True,
            provider_name="fixture",
            model_name="showcase-scripted-v1",
            tool_protocol=tool_protocol,
            registry=registry,
            max_output_tokens=2048,
            temperature=0.0,
            request_timeout_seconds=60.0,
        )

    def _scripted_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        step_number = payload.get("step_number")
        if isinstance(step_number, bool) or not isinstance(step_number, int) or step_number < 1:
            raise LLMProviderUnavailable("Offline fixture request has no valid step_number.")
        index = step_number - 1
        if index >= len(self._actions):
            raise LLMProviderUnavailable(
                f"Offline fixture has no scripted action for step {step_number}."
            )
        action = self._actions[index]
        if not isinstance(action, dict):
            raise LLMProviderUnavailable(
                f"Offline fixture action {step_number} must be a JSON object."
            )
        return copy.deepcopy(action)

    def _call_json(
        self,
        *,
        namespace: str,
        payload: dict[str, Any],
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        del messages
        started_at = time.perf_counter()
        cached = self.cache.load(namespace, payload)
        if cached is not None:
            receipt = self.cache.load_usage_receipt(namespace, payload)
            self._record_call(
                namespace=namespace,
                payload=payload,
                response=cached,
                cache_hit=True,
                usage=receipt,
                elapsed_ms=_elapsed_ms(started_at),
                recorded_elapsed_ms=receipt.get("elapsed_ms") if receipt else None,
            )
            return cached
        if self.provider_mode == "replay":  # defensive; factory uses base replay for this mode.
            raise LLMProviderUnavailable(
                f"No replay cache entry for {namespace}. Run offline with --record-cache first."
            )
        response = self._scripted_action(payload)
        self.cache.save(
            namespace,
            payload,
            response,
            metadata={"provider": self.provider_name, "model": self.model_name},
        )
        self._record_call(
            namespace=namespace,
            payload=payload,
            response=response,
            cache_hit=False,
            elapsed_ms=_elapsed_ms(started_at),
        )
        return response

    def _call_chat_tool(
        self,
        *,
        namespace: str,
        payload: dict[str, Any],
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        normalize_call: Callable[[Any], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        del messages, tools
        started_at = time.perf_counter()
        cached = self.cache.load(namespace, payload)
        if cached is not None:
            receipt = self.cache.load_usage_receipt(namespace, payload)
            self._record_call(
                namespace=namespace,
                payload=payload,
                response=cached,
                cache_hit=True,
                usage=receipt,
                elapsed_ms=_elapsed_ms(started_at),
                recorded_elapsed_ms=receipt.get("elapsed_ms") if receipt else None,
            )
            return cached
        if self.provider_mode == "replay":  # defensive; factory uses base replay for this mode.
            raise LLMProviderUnavailable(
                f"No replay cache entry for {namespace}. Run offline with --record-cache first."
            )
        action = self._scripted_action(payload)
        action_tool = action.get("tool")
        if not isinstance(action_tool, str):
            raise LLMProviderUnavailable("Offline fixture action is missing a string tool name.")
        try:
            spec = self.registry.spec(action_tool)
        except Exception as exc:  # registry exposes a typed validation error; keep fixture error stable.
            raise LLMProviderUnavailable(
                f"Offline fixture action names an unknown tool: {action_tool!r}."
            ) from exc
        arguments = {key: value for key, value in action.items() if key != "tool"}
        try:
            call = ToolCall.from_native(
                name=spec.provider_name,
                arguments=arguments,
            )
            response = normalize_call(call) if normalize_call else call.as_action_payload()
        except Exception as exc:  # normalize errors should match the live boundary, without network.
            raise LLMProviderUnavailable(
                f"Offline fixture action failed native tool validation for {action_tool!r}."
            ) from exc
        self.cache.save(
            namespace,
            payload,
            response,
            metadata={"provider": self.provider_name, "model": self.model_name},
        )
        self._record_call(
            namespace=namespace,
            payload=payload,
            response=response,
            cache_hit=False,
            elapsed_ms=_elapsed_ms(started_at),
        )
        return response


def make_action_provider(
    *,
    mode: str,
    profile: dict[str, Any],
    cache_dir: Path,
    registry: ToolRegistry | None = None,
    actions: list[dict[str, Any]] | None = None,
) -> LLMAgentActionProvider:
    """Create a showcase action provider for offline, live, or replay mode."""

    selected_mode = _validate_mode(mode)
    if not isinstance(profile, dict):
        raise ValueError("profile must be a mapping")
    cache_path = Path(cache_dir)
    active_registry = registry or SHOWCASE_TOOL_REGISTRY
    protocol = _profile_value(profile, "tool_protocol", "native_function")
    if protocol not in {"json_prompt", "native_function"}:
        raise ValueError("profile tool_protocol must be json_prompt or native_function")

    if selected_mode == "offline":
        return _FixtureActionProvider(
            actions=actions,
            registry=active_registry,
            tool_protocol=protocol,
            cache_dir=cache_path,
        )

    provider_name = str(_profile_value(profile, "provider", "")).casefold()
    if provider_name not in {"openai", "deepseek", "fixture"}:
        raise ValueError("live/replay action profiles must use openai, deepseek, or fixture")
    if selected_mode == "live" and provider_name == "fixture":
        raise ValueError("fixture profile is available only in offline or replay mode")
    kwargs = _provider_kwargs(profile)
    return LLMAgentActionProvider(
        provider_mode="live" if selected_mode == "live" else "replay",
        cache_dir=cache_path,
        record_cache=selected_mode == "live",
        tool_protocol=protocol,
        registry=active_registry,
        **kwargs,
    )


class _ShowcaseTranslationProvider(LLMTranslationProvider):
    """Translation provider that binds trusted showcase context to cache keys."""

    def __init__(self, *, instruction_context: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self.instruction_context = _json_copy(
            instruction_context, label="instruction_context"
        ) if instruction_context is not None else None
        super().__init__(**kwargs)

    def _translation_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.instruction_context is None:
            return payload
        decorated = dict(payload)
        decorated["instruction_context"] = copy.deepcopy(self.instruction_context)
        return decorated

    def _call_json(
        self,
        *,
        namespace: str,
        payload: dict[str, Any],
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        return super()._call_json(
            namespace=namespace,
            payload=self._translation_payload(payload) if namespace == "translation" else payload,
            messages=messages,
        )


class _FixtureTranslationProvider(_ShowcaseTranslationProvider):
    def __init__(self, *, translation: str | None, **kwargs: Any) -> None:
        self.fixture_translation = translation
        super().__init__(**kwargs)

    def _call_json(
        self,
        *,
        namespace: str,
        payload: dict[str, Any],
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        del messages
        payload = self._translation_payload(payload) if namespace == "translation" else payload
        started_at = time.perf_counter()
        cached = self.cache.load(namespace, payload)
        if cached is not None:
            receipt = self.cache.load_usage_receipt(namespace, payload)
            self._record_call(
                namespace=namespace,
                payload=payload,
                response=cached,
                cache_hit=True,
                usage=receipt,
                elapsed_ms=_elapsed_ms(started_at),
                recorded_elapsed_ms=receipt.get("elapsed_ms") if receipt else None,
            )
            return cached
        if self.provider_mode == "replay":
            raise LLMProviderUnavailable(
                f"No replay cache entry for {namespace}. Run offline with --record-cache first."
            )
        if not isinstance(self.fixture_translation, str) or not self.fixture_translation.strip():
            raise LLMProviderUnavailable(
                "Offline translation fixture requires a non-empty translation string."
            )
        response = {"translation": self.fixture_translation.strip()}
        self.cache.save(
            namespace,
            payload,
            response,
            metadata={"provider": self.provider_name, "model": self.model_name},
        )
        self._record_call(
            namespace=namespace,
            payload=payload,
            response=response,
            cache_hit=False,
            elapsed_ms=_elapsed_ms(started_at),
        )
        return response


def make_translation_provider(
    *,
    mode: str,
    profile: dict[str, Any],
    cache_dir: Path,
    translation: str | None = None,
    instruction_context: dict[str, Any] | None = None,
) -> LLMTranslationProvider:
    """Create a showcase translation provider with standard cache semantics."""

    selected_mode = _validate_mode(mode)
    if not isinstance(profile, dict):
        raise ValueError("profile must be a mapping")
    cache_path = Path(cache_dir)
    provider_name = str(_profile_value(profile, "provider", "")).casefold()
    if selected_mode == "offline":
        return _FixtureTranslationProvider(
            translation=translation,
            instruction_context=instruction_context,
            provider_mode="live",
            cache_dir=cache_path,
            record_cache=True,
            provider_name="fixture",
            model_name="showcase-scripted-v1",
            **{
                key: value
                for key, value in {
                    "max_output_tokens": _profile_value(profile, "max_output_tokens", 2048),
                    "temperature": _profile_value(profile, "temperature", 0.0),
                    "request_timeout_seconds": _profile_value(
                        profile, "request_timeout_seconds", 60.0
                    ),
                }.items()
                if value is not None
            },
        )

    if provider_name not in {"openai", "deepseek", "fixture"}:
        raise ValueError("live/replay translation profiles must use openai, deepseek, or fixture")
    if selected_mode == "live" and provider_name == "fixture":
        raise ValueError("fixture profile is available only in offline or replay mode")
    kwargs = _provider_kwargs(profile)
    return _ShowcaseTranslationProvider(
        instruction_context=instruction_context,
        provider_mode="live" if selected_mode == "live" else "replay",
        cache_dir=cache_path,
        record_cache=selected_mode == "live",
        **kwargs,
    )


__all__ = [
    "fixture_profile",
    "make_action_provider",
    "make_translation_provider",
    "resolve_phase_profile",
    "resolve_profile",
]
