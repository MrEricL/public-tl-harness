from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, model_validator

from .glossary import matched_entries
from .models import (
    CacheIndexEntry,
    CacheIndexReport,
    CacheIntegrityIssue,
    CandidateScore,
    EnsembleDecision,
    GlossaryParseResult,
    JudgeVote,
    ProviderCallRecord,
    QAFinding,
    QualityCandidateScore,
    RepairPatch,
    StoryConfig,
    TranslationCandidate,
)
from .providers_offline import OfflineJudgeProvider
from .text import join_paragraphs, split_paragraphs


class LLMProviderUnavailable(RuntimeError):
    pass


OPENAI_COMPATIBLE_PROVIDER_CONFIGS = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "model_env": "AGENTIC_TRANSLATION_MODEL",
        "base_url_env": "OPENAI_BASE_URL",
        "default_base_url": None,
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_base_url": "https://api.deepseek.com",
    },
}
OPENAI_COMPATIBLE_PROVIDER_NAMES = frozenset(OPENAI_COMPATIBLE_PROVIDER_CONFIGS)
NON_RETRYABLE_PROVIDER_STATUS_CODES = frozenset({400, 401, 402, 403, 404, 422})


def is_openai_compatible_provider(name: str) -> bool:
    return name in OPENAI_COMPATIBLE_PROVIDER_CONFIGS


def openai_compatible_provider_names(provider_names: set[str] | list[str] | tuple[str, ...]) -> list[str]:
    return sorted(name for name in provider_names if is_openai_compatible_provider(name))


def required_live_provider_config(provider_name: str, *, model_name: str | None = None) -> list[str]:
    config = OPENAI_COMPATIBLE_PROVIDER_CONFIGS[provider_name]
    missing: list[str] = []
    if not os.environ.get(str(config["api_key_env"])):
        missing.append(str(config["api_key_env"]))
    model_env = str(config["model_env"])
    if not (model_name or os.environ.get(model_env) or os.environ.get("AGENTIC_TRANSLATION_MODEL")):
        missing.append(model_env if model_env != "AGENTIC_TRANSLATION_MODEL" else "AGENTIC_TRANSLATION_MODEL")
    return missing


def _is_non_retryable_provider_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    return isinstance(status_code, int) and status_code in NON_RETRYABLE_PROVIDER_STATUS_CODES


class LiveJudgeResponse(BaseModel):
    selected_candidate_id: str
    quality_scores: dict[str, QualityCandidateScore] = Field(default_factory=dict)
    rationale: str = ""


class LiveRepairResponse(BaseModel):
    patch_type: Literal["replace_span", "replace_paragraph"]
    old_text: str
    new_text: str
    paragraph_index: int | None = None
    reason: str = ""

    @model_validator(mode="after")
    def validate_patch_payload(self) -> "LiveRepairResponse":
        if self.patch_type == "replace_span" and not self.old_text:
            raise ValueError("replace_span patch requires non-empty old_text")
        return self


class LiveProviderProbeResult(BaseModel):
    provider: str
    mode: str
    model: str
    cache_dir: str
    cache_hit: bool
    cache_file: str
    response: dict[str, Any] = Field(default_factory=dict)


class ResponseCache:
    index_filename = "cache_index.jsonl"

    def __init__(self, cache_dir: Path | None) -> None:
        self.cache_dir = cache_dir or Path(".agentic_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _payload_digest(self, payload: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def _response_digest(self, response: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(response, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def _path(self, namespace: str, payload: dict[str, Any]) -> Path:
        digest = self._payload_digest(payload)
        return self.cache_dir / f"{namespace}_{digest}.json"

    def _usage_path(self, namespace: str, payload: dict[str, Any]) -> Path:
        """Return the optional usage receipt path for one request identity.

        Usage is deliberately kept outside the response cache and its
        content-addressed index.  This lets telemetry evolve without changing
        response bytes or the hashes that make replay auditable.
        """

        digest = self._payload_digest(payload)
        return self.cache_dir / f"usage_{namespace}_{digest}.json"

    @property
    def index_path(self) -> Path:
        return self.cache_dir / self.index_filename

    def load(self, namespace: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        path = self._path(namespace, payload)
        if not path.exists():
            return None
        response = json.loads(path.read_text(encoding="utf-8"))
        payload_sha256 = self._payload_digest(payload)
        entry = next(
            (
                existing
                for existing in _read_cache_index(self.cache_dir)
                if existing.namespace == namespace and existing.payload_sha256 == payload_sha256
            ),
            None,
        )
        if entry is not None:
            expected_name = f"{namespace}_{payload_sha256}.json"
            if entry.cache_file != expected_name or entry.cache_file != path.name:
                raise LLMProviderUnavailable(
                    f"Replay cache integrity check failed for {path.name}: indexed cache filename does not match payload digest."
                )
            if self._response_digest(response) != entry.response_sha256:
                raise LLMProviderUnavailable(
                    f"Replay cache integrity check failed for {path.name}: response digest mismatch."
                )
        return response

    def save(
        self,
        namespace: str,
        payload: dict[str, Any],
        response: dict[str, Any],
        *,
        metadata: dict[str, str] | None = None,
    ) -> CacheIndexEntry:
        path = self._path(namespace, payload)
        path.write_text(json.dumps(response, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        entry = CacheIndexEntry(
            namespace=namespace,
            cache_file=path.name,
            payload_sha256=self._payload_digest(payload),
            response_sha256=self._response_digest(response),
            provider=(metadata or {}).get("provider"),
            model=(metadata or {}).get("model"),
        )
        self._record_index_entry(entry)
        return entry

    def save_usage_receipt(
        self,
        namespace: str,
        payload: dict[str, Any],
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cached_input_tokens: int | None = None,
        total_tokens: int | None = None,
        elapsed_ms: float | None = None,
    ) -> Path:
        """Persist optional provider usage without changing cached response data."""

        payload_sha256 = self._payload_digest(payload)
        receipt = {
            "namespace": namespace,
            "payload_sha256": payload_sha256,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "total_tokens": total_tokens,
            "elapsed_ms": elapsed_ms,
        }
        path = self._usage_path(namespace, payload)
        path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def load_usage_receipt(self, namespace: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Load one optional usage receipt, tolerating caches created before telemetry."""

        path = self._usage_path(namespace, payload)
        if not path.exists():
            return None
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(receipt, dict):
            return None
        if receipt.get("namespace") != namespace or receipt.get("payload_sha256") != self._payload_digest(payload):
            return None
        return receipt

    def _record_index_entry(self, entry: CacheIndexEntry) -> None:
        entries = [
            existing
            for existing in _read_cache_index(self.cache_dir)
            if not (existing.namespace == entry.namespace and existing.payload_sha256 == entry.payload_sha256)
        ]
        entries.append(entry)
        self.index_path.write_text(
            "".join(existing.model_dump_json() + "\n" for existing in entries),
            encoding="utf-8",
        )

    def inspect(self) -> CacheIndexReport:
        return inspect_response_cache(self.cache_dir)


def _read_cache_index(cache_dir: Path) -> list[CacheIndexEntry]:
    index_path = cache_dir / ResponseCache.index_filename
    if not index_path.exists():
        return []
    entries: list[CacheIndexEntry] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(CacheIndexEntry.model_validate_json(line))
    return entries


def inspect_response_cache(cache_dir: Path) -> CacheIndexReport:
    entries = _read_cache_index(cache_dir)
    by_namespace: dict[str, int] = {}
    integrity_issues: list[CacheIntegrityIssue] = []
    valid_entries = 0
    for entry in entries:
        by_namespace[entry.namespace] = by_namespace.get(entry.namespace, 0) + 1
        entry_has_issue = False
        expected_name = f"{entry.namespace}_{entry.payload_sha256}.json"
        if entry.cache_file != expected_name:
            entry_has_issue = True
            integrity_issues.append(
                CacheIntegrityIssue(
                    namespace=entry.namespace,
                    cache_file=entry.cache_file,
                    issue_type="cache_file_mismatch",
                    message=(
                        f"Indexed cache file {entry.cache_file} does not match namespace/payload digest "
                        f"expected name {expected_name}."
                    ),
                )
            )
        cache_path = cache_dir / entry.cache_file
        if not cache_path.exists():
            entry_has_issue = True
            integrity_issues.append(
                CacheIntegrityIssue(
                    namespace=entry.namespace,
                    cache_file=entry.cache_file,
                    issue_type="missing_file",
                    message=f"Indexed cache file is missing: {entry.cache_file}.",
                )
            )
        else:
            try:
                response = json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                entry_has_issue = True
                integrity_issues.append(
                    CacheIntegrityIssue(
                        namespace=entry.namespace,
                        cache_file=entry.cache_file,
                        issue_type="invalid_json",
                        message=f"Indexed cache file is not valid JSON: {entry.cache_file} ({exc}).",
                    )
                )
            else:
                response_digest = hashlib.sha256(
                    json.dumps(response, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest()
                if response_digest != entry.response_sha256:
                    entry_has_issue = True
                    integrity_issues.append(
                        CacheIntegrityIssue(
                            namespace=entry.namespace,
                            cache_file=entry.cache_file,
                            issue_type="response_digest_mismatch",
                            message=f"Indexed cache file response digest mismatch: {entry.cache_file}.",
                        )
                    )
        if not entry_has_issue:
            valid_entries += 1
    return CacheIndexReport(
        cache_dir=str(cache_dir),
        total_entries=len(entries),
        by_namespace=by_namespace,
        entries=entries,
        valid_entries=valid_entries,
        invalid_entries=len(entries) - valid_entries,
        integrity_passed=not integrity_issues,
        integrity_issues=integrity_issues,
    )


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _nonnegative_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def _extract_usage(response: Any) -> dict[str, int | None]:
    """Normalize OpenAI-compatible usage objects without touching response JSON."""

    usage = _value(response, "usage")
    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "cached_input_tokens": None,
            "total_tokens": None,
        }

    prompt_details = _value(usage, "prompt_tokens_details")
    input_details = _value(usage, "input_token_details")
    cached_candidates = (
        _value(usage, "cached_input_tokens"),
        _value(usage, "prompt_cache_hit_tokens"),
        _value(usage, "prompt_cache_read_tokens"),
        _value(prompt_details, "cached_tokens"),
        _value(input_details, "cached_tokens"),
    )
    cached_input_tokens = next(
        (candidate for candidate in cached_candidates if candidate is not None),
        None,
    )
    return {
        "input_tokens": _nonnegative_int(
            _value(usage, "prompt_tokens", _value(usage, "input_tokens"))
        ),
        "output_tokens": _nonnegative_int(
            _value(usage, "completion_tokens", _value(usage, "output_tokens"))
        ),
        "cached_input_tokens": _nonnegative_int(cached_input_tokens),
        "total_tokens": _nonnegative_int(_value(usage, "total_tokens")),
    }


def _elapsed_ms(started_at: float) -> float:
    return max(0.0, (time.perf_counter() - started_at) * 1000.0)


_SENSITIVE_EXTRA_BODY_KEY_SUFFIXES = frozenset(
    {
        "apikey",
        "authorization",
        "clientsecret",
        "password",
        "secret",
        "accesstoken",
        "refreshtoken",
    }
)


def _copy_request_extra_body(extra_body: dict[str, Any] | None) -> dict[str, Any] | None:
    """Validate and detach optional provider request extensions.

    Profiles are persisted in run manifests, so request extensions must be
    ordinary JSON data and must not be used as a back door for credentials.
    """

    if extra_body is None:
        return None
    if not isinstance(extra_body, dict):
        raise ValueError("extra_body must be a JSON object when supplied")

    def validate_keys(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError("extra_body object keys must be strings")
                normalized_key = "".join(
                    character for character in key.casefold() if character.isalnum()
                )
                if normalized_key == "token" or any(
                    normalized_key.endswith(suffix)
                    for suffix in _SENSITIVE_EXTRA_BODY_KEY_SUFFIXES
                ):
                    raise ValueError(f"extra_body must not contain credential field {key!r}")
                validate_keys(item)
        elif isinstance(value, list):
            for item in value:
                validate_keys(item)
        elif value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError("extra_body must contain only JSON-compatible values")

    validate_keys(extra_body)
    try:
        encoded = json.dumps(
            extra_body,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        copied = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError("extra_body must contain only finite JSON-compatible values") from exc
    if not isinstance(copied, dict):  # pragma: no cover - guarded above.
        raise ValueError("extra_body must be a JSON object when supplied")
    return copied


class _OpenAIJSONProvider:
    provider_name = "openai"
    model_name = ""

    def __init__(
        self,
        *,
        provider_mode: str = "live",
        cache_dir: Path | None = None,
        record_cache: bool = False,
        model_name: str | None = None,
        provider_name: str = "openai",
        api_key_env: str | None = None,
        model_env: str | None = None,
        base_url_env: str | None = None,
        default_base_url: str | None = None,
        client_factory: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 2,
        max_output_tokens: int | None = None,
        temperature: float | None = 0,
        request_timeout_seconds: float = 60,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.provider_mode = provider_mode
        self.cache = ResponseCache(cache_dir)
        self.record_cache = record_cache
        config = OPENAI_COMPATIBLE_PROVIDER_CONFIGS.get(provider_name, OPENAI_COMPATIBLE_PROVIDER_CONFIGS["openai"])
        self.provider_name = provider_name
        self.api_key_env = api_key_env or str(config["api_key_env"])
        self.model_env = model_env or str(config["model_env"])
        self.base_url_env = base_url_env or str(config["base_url_env"])
        self.default_base_url = default_base_url if default_base_url is not None else config["default_base_url"]
        self.model_name = model_name or os.environ.get(self.model_env, "") or os.environ.get("AGENTIC_TRANSLATION_MODEL", "")
        self.client_factory = client_factory
        self.sleep = sleep
        self.max_retries = max(0, max_retries)
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.request_timeout_seconds = request_timeout_seconds
        self.extra_body = _copy_request_extra_body(extra_body)
        self.call_records: list[ProviderCallRecord] = []

    def _call_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return the content-cache identity for the effective provider call.

        Existing providers keep their historical payload byte-for-byte.  When
        an extension changes the HTTP request, the extension is bound to the
        cached content so replay cannot silently use a response produced with
        different provider behavior.
        """

        if self.extra_body is None:
            return payload
        return {
            "payload": payload,
            "provider_request": {"extra_body": copy.deepcopy(self.extra_body)},
        }

    def _record_call(
        self,
        *,
        namespace: str,
        payload: dict[str, Any],
        response: dict[str, Any],
        cache_hit: bool,
        usage: dict[str, Any] | None = None,
        elapsed_ms: float | None = None,
        recorded_elapsed_ms: float | None = None,
    ) -> None:
        usage = usage or {}
        self.call_records.append(
            ProviderCallRecord(
                role=namespace,
                namespace=namespace,
                provider=self.provider_name,
                model=self.model_name,
                payload_sha256=self.cache._payload_digest(payload),
                response_sha256=self.cache._response_digest(response),
                cache_file=self.cache._path(namespace, payload).name,
                cache_hit=cache_hit,
                input_tokens=_nonnegative_int(usage.get("input_tokens")),
                output_tokens=_nonnegative_int(usage.get("output_tokens")),
                cached_input_tokens=_nonnegative_int(usage.get("cached_input_tokens")),
                total_tokens=_nonnegative_int(usage.get("total_tokens")),
                elapsed_ms=_nonnegative_float(elapsed_ms),
                recorded_elapsed_ms=_nonnegative_float(recorded_elapsed_ms),
            )
        )

    def _call_json(self, *, namespace: str, payload: dict[str, Any], messages: list[dict[str, str]]) -> dict[str, Any]:
        started_at = time.perf_counter()
        call_payload = self._call_payload(payload)
        cached = self.cache.load(namespace, call_payload)
        if cached is not None:
            receipt = self.cache.load_usage_receipt(namespace, call_payload)
            self._record_call(
                namespace=namespace,
                payload=call_payload,
                response=cached,
                cache_hit=True,
                usage=receipt,
                elapsed_ms=_elapsed_ms(started_at),
                recorded_elapsed_ms=receipt.get("elapsed_ms") if receipt else None,
            )
            return cached
        if self.provider_mode == "replay":
            raise LLMProviderUnavailable(f"No replay cache entry for {namespace}. Run live with --record-cache first.")

        api_key = os.environ.get(self.api_key_env)
        model = self.model_name
        if not api_key:
            raise LLMProviderUnavailable(f"{self.api_key_env} is required for live {self.provider_name} providers.")
        if not model:
            model_hint = f"{self.model_env} or --model"
            if self.model_env != "AGENTIC_TRANSLATION_MODEL":
                model_hint += " (AGENTIC_TRANSLATION_MODEL also works)"
            raise LLMProviderUnavailable(f"{model_hint} is required for live {self.provider_name} providers.")

        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - environment guard
            raise LLMProviderUnavailable("The openai package is required for live providers.") from exc

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        base_url = os.environ.get(self.base_url_env) or self.default_base_url
        if base_url:
            client_kwargs["base_url"] = str(base_url)
        client_factory = self.client_factory or OpenAI
        # Keep injected legacy test doubles' constructor shape stable at the
        # default while always configuring the real OpenAI client.  A custom
        # factory using a non-default timeout receives that explicit timeout.
        if self.client_factory is None or self.request_timeout_seconds != 60:
            client_kwargs["timeout"] = self.request_timeout_seconds
        last_error: Exception | None = None
        stopped_without_retry = False
        for attempt in range(self.max_retries + 1):
            try:
                client = client_factory(**client_kwargs)
                request_kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,  # type: ignore[arg-type]
                    "response_format": {"type": "json_object"},
                }
                if self.max_output_tokens is not None:
                    request_kwargs["max_tokens"] = self.max_output_tokens
                if self.temperature is not None:
                    request_kwargs["temperature"] = self.temperature
                if self.extra_body is not None:
                    request_kwargs["extra_body"] = copy.deepcopy(self.extra_body)
                response = client.chat.completions.create(**request_kwargs)
                content = response.choices[0].message.content or "{}"
                parsed = json.loads(content)
                usage = _extract_usage(response)
                elapsed_ms = _elapsed_ms(started_at)
                if self.record_cache:
                    self.cache.save(
                        namespace,
                        call_payload,
                        parsed,
                        metadata={"provider": self.provider_name, "model": model},
                    )
                    self.cache.save_usage_receipt(
                        namespace,
                        call_payload,
                        **usage,
                        elapsed_ms=elapsed_ms,
                    )
                self._record_call(
                    namespace=namespace,
                    payload=call_payload,
                    response=parsed,
                    cache_hit=False,
                    usage=usage,
                    elapsed_ms=elapsed_ms,
                )
                return parsed
            except Exception as exc:  # noqa: BLE001 - provider clients raise heterogeneous exceptions.
                last_error = exc
                if _is_non_retryable_provider_error(exc):
                    stopped_without_retry = True
                    break
                if attempt < self.max_retries:
                    self.sleep(min(2**attempt, 8))
        if stopped_without_retry:
            raise LLMProviderUnavailable(f"Live provider call failed without retry: {last_error}") from last_error
        raise LLMProviderUnavailable(f"Live provider call failed after retries: {last_error}") from last_error

    def _call_chat_tool(
        self,
        *,
        namespace: str,
        payload: dict[str, Any],
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        normalize_call: Callable[[Any], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Call one required native function and cache its action payload.

        This intentionally mirrors ``_call_json``'s cache, auth, retry, and
        recording behavior while keeping the native request free of
        ``response_format``.  A caller may provide a registry normalizer to
        map provider-native names to logical action names before caching.
        """

        started_at = time.perf_counter()
        call_payload = self._call_payload(payload)
        cached = self.cache.load(namespace, call_payload)
        if cached is not None:
            receipt = self.cache.load_usage_receipt(namespace, call_payload)
            self._record_call(
                namespace=namespace,
                payload=call_payload,
                response=cached,
                cache_hit=True,
                usage=receipt,
                elapsed_ms=_elapsed_ms(started_at),
                recorded_elapsed_ms=receipt.get("elapsed_ms") if receipt else None,
            )
            return cached
        if self.provider_mode == "replay":
            raise LLMProviderUnavailable(f"No replay cache entry for {namespace}. Run live with --record-cache first.")

        api_key = os.environ.get(self.api_key_env)
        model = self.model_name
        if not api_key:
            raise LLMProviderUnavailable(f"{self.api_key_env} is required for live {self.provider_name} providers.")
        if not model:
            model_hint = f"{self.model_env} or --model"
            if self.model_env != "AGENTIC_TRANSLATION_MODEL":
                model_hint += " (AGENTIC_TRANSLATION_MODEL also works)"
            raise LLMProviderUnavailable(f"{model_hint} is required for live {self.provider_name} providers.")

        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - environment guard
            raise LLMProviderUnavailable("The openai package is required for live providers.") from exc

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        base_url = os.environ.get(self.base_url_env) or self.default_base_url
        if base_url:
            client_kwargs["base_url"] = str(base_url)
        client_factory = self.client_factory or OpenAI
        if self.client_factory is None or self.request_timeout_seconds != 60:
            client_kwargs["timeout"] = self.request_timeout_seconds
        last_error: Exception | None = None
        stopped_without_retry = False

        def value(obj: Any, key: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        for attempt in range(self.max_retries + 1):
            try:
                client = client_factory(**client_kwargs)
                request_kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,  # type: ignore[arg-type]
                    "tools": tools,
                    "tool_choice": "required",
                    "parallel_tool_calls": False,
                }
                if self.max_output_tokens is not None:
                    request_kwargs["max_tokens"] = self.max_output_tokens
                if self.temperature is not None:
                    request_kwargs["temperature"] = self.temperature
                if self.extra_body is not None:
                    request_kwargs["extra_body"] = copy.deepcopy(self.extra_body)
                response = client.chat.completions.create(**request_kwargs)
                choices = value(response, "choices", [])
                message = value(choices[0], "message") if choices else None
                tool_calls = value(message, "tool_calls", [])
                if not isinstance(tool_calls, list) or len(tool_calls) != 1:
                    raise LLMProviderUnavailable("Native provider response must contain exactly one tool call.")
                function = value(tool_calls[0], "function")
                function_name = value(function, "name")
                function_arguments = value(function, "arguments")
                if not isinstance(function_name, str) or not isinstance(function_arguments, (str, dict)):
                    raise LLMProviderUnavailable("Native provider response contained an invalid tool call.")
                from .agent_tools import ToolCall

                call = ToolCall.from_native(
                    name=function_name,
                    arguments=function_arguments,
                    call_id=value(tool_calls[0], "id"),
                )
                parsed = normalize_call(call) if normalize_call else call.as_action_payload()
                usage = _extract_usage(response)
                elapsed_ms = _elapsed_ms(started_at)
                if self.record_cache:
                    self.cache.save(
                        namespace,
                        call_payload,
                        parsed,
                        metadata={"provider": self.provider_name, "model": model},
                    )
                    self.cache.save_usage_receipt(
                        namespace,
                        call_payload,
                        **usage,
                        elapsed_ms=elapsed_ms,
                    )
                self._record_call(
                    namespace=namespace,
                    payload=call_payload,
                    response=parsed,
                    cache_hit=False,
                    usage=usage,
                    elapsed_ms=elapsed_ms,
                )
                return parsed
            except Exception as exc:  # noqa: BLE001 - provider clients raise heterogeneous exceptions.
                last_error = exc
                if _is_non_retryable_provider_error(exc):
                    stopped_without_retry = True
                    break
                if attempt < self.max_retries:
                    self.sleep(min(2**attempt, 8))
        if stopped_without_retry:
            raise LLMProviderUnavailable(f"Live provider call failed without retry: {last_error}") from last_error
        raise LLMProviderUnavailable(f"Live provider call failed after retries: {last_error}") from last_error


def probe_live_provider(
    *,
    provider_name: str,
    provider_mode: Literal["live", "replay"] = "live",
    cache_dir: Path | None = None,
    record_cache: bool = True,
    model_name: str | None = None,
    client_factory: Callable[..., Any] | None = None,
) -> LiveProviderProbeResult:
    if not is_openai_compatible_provider(provider_name):
        supported = ", ".join(sorted(OPENAI_COMPATIBLE_PROVIDER_NAMES))
        raise ValueError(f"Unsupported live provider {provider_name!r}; expected one of {supported}.")
    provider = _OpenAIJSONProvider(
        provider_mode=provider_mode,
        cache_dir=cache_dir,
        record_cache=record_cache,
        model_name=model_name,
        provider_name=provider_name,
        client_factory=client_factory,
        max_retries=0,
    )
    payload = {
        "task": "provider_probe",
        "provider": provider_name,
        "model": provider.model_name,
        "instruction": "Return a tiny JSON health response.",
    }
    data = provider._call_json(
        namespace="probe",
        payload=payload,
        messages=[
            {
                "role": "system",
                "content": 'Return JSON only: {"ok": true, "message": "pong"}.',
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    if not provider.call_records:
        raise LLMProviderUnavailable("Provider probe did not record a provider call.")
    record = provider.call_records[-1]
    return LiveProviderProbeResult(
        provider=provider_name,
        mode=provider_mode,
        model=provider.model_name,
        cache_dir=str(provider.cache.cache_dir),
        cache_hit=record.cache_hit,
        cache_file=record.cache_file,
        response=data,
    )


class LLMTranslationProvider(_OpenAIJSONProvider):
    def _source_chunks(self, source_text: str, *, max_chars: int) -> list[str]:
        stripped = source_text.strip()
        if not stripped:
            return [source_text]
        if max_chars <= 0 or len(stripped) <= max_chars:
            return [stripped]
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for paragraph in split_paragraphs(stripped):
            projected_len = current_len + len(paragraph) + (2 if current else 0)
            if current and projected_len > max_chars:
                chunks.append(join_paragraphs(current))
                current = [paragraph]
                current_len = len(paragraph)
            else:
                current.append(paragraph)
                current_len = projected_len
        if current:
            chunks.append(join_paragraphs(current))
        return chunks or [stripped]

    def _translate_chunk(
        self,
        *,
        chunk_text: str,
        chunk_index: int,
        chunk_count: int,
        story: StoryConfig,
        glossary: GlossaryParseResult,
        mode: str,
        prompt_text: str,
    ) -> str:
        relevant_glossary = matched_entries(chunk_text, glossary)
        if not relevant_glossary:
            relevant_glossary = glossary.entries[: story.translation.max_glossary_entries]
        else:
            relevant_glossary = relevant_glossary[: story.translation.max_glossary_entries]
        system_prompt = (
            "Return JSON only: {\"translation\": \"...\"}. Produce clear, faithful English. "
            "Translate only the supplied chunk and preserve paragraph breaks."
        )
        if chunk_count > 1:
            system_prompt += f" This is chunk {chunk_index} of {chunk_count}; do not repeat other chunks."
        if prompt_text:
            system_prompt += "\n\nStory translation prompt:\n" + prompt_text
        payload = {
            "task": "translate",
            "story": story.slug,
            "mode": mode,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "source_text": chunk_text,
            "glossary": [entry.model_dump() for entry in relevant_glossary],
        }
        data = self._call_json(
            namespace="translation",
            payload=payload,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        translation = data.get("translation")
        if not isinstance(translation, str) or not translation.strip():
            raise LLMProviderUnavailable("Live translation provider returned no translation string.")
        return translation.strip()

    def translate(
        self,
        source_text: str,
        *,
        story: StoryConfig,
        glossary: GlossaryParseResult,
        mode: str,
    ) -> str:
        prompt_text = ""
        if story.paths.prompt_path and story.paths.prompt_path.exists():
            prompt_text = story.paths.prompt_path.read_text(encoding="utf-8").strip()
        chunks = self._source_chunks(source_text, max_chars=story.translation.max_chunk_chars)
        translations = [
            self._translate_chunk(
                chunk_text=chunk,
                chunk_index=index,
                chunk_count=len(chunks),
                story=story,
                glossary=glossary,
                mode=mode,
                prompt_text=prompt_text,
            )
            for index, chunk in enumerate(chunks, start=1)
        ]
        return join_paragraphs(translations)


class LLMJudgeProvider(_OpenAIJSONProvider):
    def judge(
        self,
        *,
        source_text: str,
        candidates: list[TranslationCandidate],
        glossary: GlossaryParseResult,
        seed: int,
    ) -> EnsembleDecision:
        offline = OfflineJudgeProvider().judge(source_text=source_text, candidates=candidates, glossary=glossary, seed=seed)
        payload = {
            "task": "judge_repair_candidates",
            "source_text": source_text,
            "candidates": [candidate.model_dump() for candidate in candidates],
            "glossary": [entry.model_dump() for entry in glossary.entries],
        }
        data = self._call_json(
            namespace="judge",
            payload=payload,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return JSON only. Pick the best repair candidate for fidelity and fluency. "
                        "Schema: {\"selected_candidate_id\":\"candidate_a\","
                        "\"quality_scores\":{\"candidate_a\":{\"faithfulness\":8,\"fluency\":8,\"rationale\":\"...\"}},"
                        "\"rationale\":\"...\"}."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        response = LiveJudgeResponse.model_validate(data)
        candidate_ids = {candidate.candidate_id for candidate in candidates}
        selected = response.selected_candidate_id
        if selected not in candidate_ids:
            raise LLMProviderUnavailable(
                "Live judge selected unknown candidate "
                f"{selected!r}; expected one of {', '.join(sorted(candidate_ids))}."
            )

        base_scores = offline.votes[0].scores if offline.votes else {}
        scores: dict[str, CandidateScore] = {}
        aggregate_scores: dict[str, float] = {}
        for candidate_id, base_score in base_scores.items():
            quality = response.quality_scores.get(
                candidate_id,
                QualityCandidateScore(faithfulness=7, fluency=7, rationale="No live score returned for this candidate."),
            )
            quality_average = (quality.faithfulness + quality.fluency) / 2
            aggregate = 0.65 * base_score.aggregate + 0.35 * quality_average
            scores[candidate_id] = base_score.model_copy(update={"quality": quality, "aggregate": aggregate})
            aggregate_scores[candidate_id] = aggregate

        vote = JudgeVote(
            judge_id=f"{self.provider_name}_judge_1",
            winner=str(selected),
            scores=scores,
            rationale=response.rationale or "Selected by live judge.",
        )
        return EnsembleDecision(
            selected_candidate_id=str(selected),
            votes=[vote],
            aggregate_scores=aggregate_scores,
            disagreement=0.0,
            requires_human_review=False,
        )


class LLMRepairProvider(_OpenAIJSONProvider):
    def propose_patch(
        self,
        *,
        chapter: str,
        source_text: str,
        translation_text: str,
        finding: QAFinding,
        glossary: GlossaryParseResult,
        ensemble_decision: EnsembleDecision | None = None,
        candidates: list[TranslationCandidate] | None = None,
    ) -> RepairPatch | None:
        payload = {
            "task": "repair_patch",
            "chapter": chapter,
            "source_text": source_text,
            "translation_text": translation_text,
            "finding": finding.model_dump(),
            "selected_candidate_id": ensemble_decision.selected_candidate_id if ensemble_decision else None,
            "candidates": [candidate.model_dump() for candidate in candidates or []],
            "glossary": [entry.model_dump() for entry in glossary.entries],
        }
        data = self._call_json(
            namespace="repair",
            payload=payload,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return JSON only for one minimal patch. "
                        "Schema: {\"patch_type\":\"replace_span|replace_paragraph\","
                        "\"old_text\":\"...\",\"new_text\":\"...\",\"paragraph_index\":1,\"reason\":\"...\"}."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        response = LiveRepairResponse.model_validate(data)
        paragraph_index = response.paragraph_index
        if response.patch_type == "replace_paragraph" and paragraph_index is None:
            paragraph_index = next((i for i, paragraph in enumerate(split_paragraphs(translation_text)) if paragraph == response.old_text), None)
        if response.patch_type == "replace_paragraph" and paragraph_index is None:
            return None
        return RepairPatch(
            patch_id=f"patch_live_{finding.check_id}",
            patch_type=response.patch_type,
            chapter=chapter,
            paragraph_index=paragraph_index if isinstance(paragraph_index, int) else None,
            old_text=response.old_text,
            new_text=response.new_text,
            reason=response.reason or "Live provider proposed minimal patch.",
            source_finding_check_id=finding.check_id,
        )
