from __future__ import annotations

import pytest

from agentic_translation.showcase_providers import resolve_phase_profile, resolve_profile


@pytest.mark.parametrize(
    ("phase", "max_output_tokens"),
    [
        ("translation", 16_384),
        ("simple_revision", 16_384),
        ("memory", 4_096),
        ("review", 4_096),
        ("action", 2_048),
        ("repair", 2_048),
    ],
)
def test_phase_defaults_are_explicit(phase: str, max_output_tokens: int) -> None:
    profile = resolve_phase_profile("deepseek", "deepseek-flash", phase=phase)

    assert profile["phase"] == phase
    assert profile["model"] == "deepseek-flash"
    assert profile["model_source"] == "explicit"
    assert profile["thinking"] == "disabled"
    assert profile["temperature"] == 0.0
    assert profile["max_output_tokens"] == max_output_tokens
    assert profile["request_timeout_seconds"] == 180.0
    assert profile["tool_protocol"] == "json_prompt"
    assert profile["extra_body"] == {"thinking": {"type": "disabled"}}


def test_environment_model_selection_matches_explicit_effective_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-flash")
    from_environment = resolve_phase_profile("deepseek", phase="review")
    explicit = resolve_phase_profile("deepseek", "deepseek-flash", phase="review")

    assert from_environment["model"] == explicit["model"]
    assert from_environment["model_source"] == "environment"
    assert explicit["model_source"] == "explicit"
    for key in (
        "provider",
        "phase",
        "thinking",
        "temperature",
        "max_output_tokens",
        "request_timeout_seconds",
        "tool_protocol",
        "extra_body",
    ):
        assert from_environment[key] == explicit[key]


@pytest.mark.parametrize("model", ["deepseek-flash", "deepseek-v4-flash", "deepseek-chat"])
def test_thinking_is_explicit_for_current_and_legacy_deepseek_names(model: str) -> None:
    enabled = resolve_phase_profile("deepseek", model, thinking="enabled")
    disabled = resolve_phase_profile("deepseek", model, thinking="disabled")

    assert enabled["extra_body"] == {"thinking": {"type": "enabled"}}
    assert disabled["extra_body"] == {"thinking": {"type": "disabled"}}
    assert enabled["tool_protocol"] == "json_prompt"
    assert disabled["tool_protocol"] == "json_prompt"


def test_phase_profile_does_not_change_legacy_resolution() -> None:
    legacy = resolve_profile("deepseek", "deepseek-v4-flash")
    phase = resolve_phase_profile("deepseek", "deepseek-v4-flash")

    assert legacy["tool_protocol"] == "json_prompt"
    assert legacy["max_output_tokens"] == 2_048
    assert legacy["request_timeout_seconds"] == 60.0
    assert legacy["extra_body"] == {"thinking": {"type": "disabled"}}
    assert phase["phase"] == "translation"
    assert phase["max_output_tokens"] == 16_384
    assert phase["request_timeout_seconds"] == 180.0


def test_non_deepseek_profiles_record_thinking_without_deepseek_extension() -> None:
    profile = resolve_phase_profile("openai", "gpt-4.1-mini", thinking="disabled")

    assert profile["thinking"] == "disabled"
    assert "extra_body" not in profile


def test_phase_overrides_are_recorded() -> None:
    profile = resolve_phase_profile(
        "deepseek",
        "deepseek-flash",
        phase="repair",
        thinking="enabled",
        temperature=0.2,
        max_output_tokens=333,
        request_timeout_seconds=12.5,
        tool_protocol="native_function",
    )

    assert profile["phase"] == "repair"
    assert profile["thinking"] == "enabled"
    assert profile["temperature"] == 0.2
    assert profile["max_output_tokens"] == 333
    assert profile["request_timeout_seconds"] == 12.5
    assert profile["tool_protocol"] == "native_function"
    assert profile["extra_body"] == {"thinking": {"type": "enabled"}}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"phase": "unknown"}, "Unknown provider phase"),
        ({"thinking": "auto"}, "thinking must be one of"),
        ({"tool_protocol": "text"}, "tool_protocol must be one of"),
        ({"max_output_tokens": 0}, "max_output_tokens must be a positive integer"),
        ({"request_timeout_seconds": 0}, "request_timeout_seconds must be a finite positive number"),
    ],
)
def test_phase_profile_rejects_invalid_options(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_phase_profile("deepseek", "deepseek-flash", **kwargs)
