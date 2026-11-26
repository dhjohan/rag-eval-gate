"""Ordered, quota-aware LLM router over OpenAI-compatible free-tier endpoints.

Every provider in the chain speaks the OpenAI Chat Completions shape, so one
HTTP client serves NVIDIA NIM, Gemini's compatibility layer, Cloudflare Workers
AI, and OpenRouter. The router owns retries, throttling, fallback, structured
output, and the prompt cache.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from rag_eval.cache.sqlite_cache import SqliteCache, canonical_json, sha256_text
from rag_eval.config import LLMConfig, LLMProviderConfig, Settings, resolve_api_key
from rag_eval.providers.rate_limit import ProviderLimiter, compute_backoff, parse_retry_after

Message = Mapping[str, Any]
Messages = Sequence[Message]

_STRUCTURED_TIER = {"json_schema": 0, "json_object": 1, "none": 2}


class LLMError(RuntimeError):
    """Base class for LLM routing failures."""


class AllProvidersFailed(LLMError):
    """Raised when every provider in the chain was skipped or errored."""

    def __init__(self, failures: Sequence[ProviderFailure]) -> None:
        self.failures = tuple(failures)
        detail = "; ".join(
            f"{f.provider}: {f.reason}" + (f" ({f.status_code})" if f.status_code else "")
            for f in self.failures
        )
        super().__init__(f"all LLM providers failed: {detail or 'no providers configured'}")


@dataclass(frozen=True)
class ProviderFailure:
    provider: str
    reason: str
    status_code: int | None = None


@dataclass(frozen=True)
class LLMResult:
    """Normalised result shape shared by every provider."""

    content: str
    usage_tokens: int
    latency_ms: int
    provider_used: str
    model_used: str
    cache_hit: bool
    parsed: BaseModel | None = None


class LLMRouter:
    """Try providers in order, skipping the ones that cannot or should not run."""

    def __init__(
        self,
        config: LLMConfig,
        cache: SqliteCache,
        *,
        api_key_resolver: Callable[[str], str | None] | None = None,
        transport: httpx.BaseTransport | None = None,
        client: httpx.Client | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._cache = cache
        self._resolve_key = api_key_resolver or (lambda name: resolve_api_key(name))
        self._clock = clock
        self._sleep = sleep
        self._rng = rng or random.Random(0)
        self._limiters = {
            name: ProviderLimiter(
                name,
                cache,
                rpm=cfg.rpm,
                rpd=cfg.rpd,
                timezone=cfg.rpd_timezone,
                clock=clock,
                sleep=sleep,
                now=now,
            )
            for name, cfg in config.providers.items()
        }
        self._client = client or httpx.Client(transport=transport)
        self._owns_client = client is None

    @classmethod
    def from_settings(
        cls,
        config: LLMConfig,
        cache: SqliteCache,
        settings: Settings,
        **kwargs: Any,
    ) -> LLMRouter:
        return cls(
            config,
            cache,
            api_key_resolver=lambda name: resolve_api_key(name, settings),
            **kwargs,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> LLMRouter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- public API --------------------------------------------------------

    def generate(
        self,
        messages: Messages,
        *,
        response_model: type[BaseModel] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        use_cache: bool | None = None,
    ) -> LLMResult:
        """Generate a completion, falling back through the configured chain."""
        cache_enabled = (
            self._config.prompt_cache_enabled if use_cache is None else use_cache
        )
        target = response_model.__name__ if response_model is not None else "freeform"
        candidates = self._candidate_order(response_model)
        prompt_hashes = {
            name: self._prompt_hash(
                messages,
                response_model,
                temperature,
                max_tokens,
                reasoning_effort,
                self._config.providers[name],
            )
            for name in candidates
        }

        if cache_enabled:
            # A cached answer is only valid for the provider/model that produced
            # it, so probe each candidate provider's cache entry before going
            # to the network. A corrupt entry is skipped, not fatal: another
            # provider may still have a valid answer.
            for name in candidates:
                cfg = self._config.providers[name]
                candidate = self._cache.get_llm_response(
                    prompt_hashes[name],
                    target,
                    provider_used=name,
                    requested_model=cfg.model,
                )
                if candidate is None:
                    continue
                parsed = None
                if response_model is not None:
                    try:
                        parsed = _parse_structured(candidate.content, response_model)
                    except LLMError:
                        continue
                return LLMResult(
                    content=candidate.content,
                    usage_tokens=candidate.usage_tokens,
                    latency_ms=candidate.latency_ms,
                    provider_used=candidate.provider_used,
                    model_used=candidate.model_used,
                    cache_hit=True,
                    parsed=parsed,
                )

        failures: list[ProviderFailure] = []
        for name in candidates:
            cfg = self._config.providers[name]
            if not cfg.is_configured:
                missing = ", ".join(cfg.unresolved_env)
                failures.append(ProviderFailure(name, f"unconfigured_env:{missing}"))
                continue
            api_key = self._resolve_key(cfg.api_key_env)
            if not api_key:
                failures.append(ProviderFailure(name, f"missing_api_key:{cfg.api_key_env}"))
                continue

            result = self._try_provider(
                name,
                cfg,
                api_key,
                messages,
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                failures=failures,
            )
            if result is None:
                continue
            if cache_enabled:
                self._cache.put_llm_response(
                    prompt_hashes[name],
                    target,
                    content=result.content,
                    provider_used=result.provider_used,
                    requested_model=cfg.model,
                    model_used=result.model_used,
                    usage_tokens=result.usage_tokens,
                    latency_ms=result.latency_ms,
                )
            return result

        raise AllProvidersFailed(failures)

    # -- internals ---------------------------------------------------------

    def _candidate_order(self, response_model: type[BaseModel] | None) -> list[str]:
        order = self._config.ordered_providers
        if response_model is None:
            return order
        ranked = sorted(
            enumerate(order),
            key=lambda item: (
                _STRUCTURED_TIER[self._config.providers[item[1]].structured_output],
                item[0],
            ),
        )
        return [name for _, name in ranked]

    def _prompt_hash(
        self,
        messages: Messages,
        response_model: type[BaseModel] | None,
        temperature: float | None,
        max_tokens: int | None,
        reasoning_effort: str | None,
        cfg: LLMProviderConfig,
    ) -> str:
        """Hash the prompt plus the *effective* per-provider request parameters.

        Hashing the raw arguments would replay a stale answer after a provider's
        configured ``max_tokens`` or ``reasoning_effort`` changes.
        """
        effective_max_tokens = max_tokens if max_tokens is not None else cfg.max_tokens
        effective_effort = (
            reasoning_effort if reasoning_effort is not None else cfg.reasoning_effort
        )
        return sha256_text(
            canonical_json(
                {
                    "messages": [dict(message) for message in messages],
                    "schema": (
                        response_model.model_json_schema() if response_model is not None else None
                    ),
                    "structured_output": (
                        cfg.structured_output if response_model is not None else None
                    ),
                    "temperature": temperature,
                    "max_tokens": effective_max_tokens,
                    "reasoning_effort": effective_effort,
                }
            )
        )

    def _try_provider(
        self,
        name: str,
        cfg: LLMProviderConfig,
        api_key: str,
        messages: Messages,
        *,
        response_model: type[BaseModel] | None,
        temperature: float | None,
        max_tokens: int | None,
        reasoning_effort: str | None,
        failures: list[ProviderFailure],
    ) -> LLMResult | None:
        limiter = self._limiters[name]
        url = f"{cfg.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            **cfg.extra_headers,
        }
        payload = self._build_payload(
            cfg,
            messages,
            response_model=response_model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

        for attempt in range(self._config.max_retries + 1):
            outcome = limiter.acquire()
            if not outcome.allowed:
                failures.append(ProviderFailure(name, outcome.reason or "throttled"))
                return None

            started = self._clock()
            try:
                response = self._client.post(
                    url, json=payload, headers=headers, timeout=cfg.timeout_s
                )
            except httpx.HTTPError as exc:
                # A transport error may mean the request never reached the
                # provider, so give the quota slot back. A retry re-acquires.
                limiter.release(outcome.window)
                if attempt < self._config.max_retries:
                    self._sleep(self._backoff(attempt))
                    continue
                failures.append(
                    ProviderFailure(name, f"transport_error:{type(exc).__name__}", None)
                )
                return None
            latency_ms = int((self._clock() - started) * 1000)
            status = response.status_code

            if status == 429:
                # The provider rejected before processing, so return the slot
                # before retrying; otherwise one throttled call would burn the
                # daily budget once per attempt.
                limiter.release(outcome.window)
                retry_after = parse_retry_after(response.headers.get("retry-after"))
                if retry_after is not None and retry_after > self._config.backoff_cap_s:
                    # Waiting out a long provider-imposed cooldown is worse than
                    # failing over to the next provider now.
                    failures.append(
                        ProviderFailure(
                            name, f"http_429:retry_after_{retry_after:.0f}s", status
                        )
                    )
                    return None
                if attempt < self._config.max_retries:
                    delay = retry_after if retry_after is not None else self._backoff(attempt)
                    self._sleep(max(0.0, min(delay, self._config.backoff_cap_s)))
                    continue
                failures.append(ProviderFailure(name, "http_429", status))
                return None

            if status >= 500:
                if attempt < self._config.max_retries:
                    self._sleep(self._backoff(attempt))
                    continue
                failures.append(ProviderFailure(name, "http_5xx", status))
                return None

            if status in (401, 403):
                # Provider rejected before processing; return the quota slot.
                limiter.release(outcome.window)
                failures.append(ProviderFailure(name, "auth_rejected", status))
                return None

            if 300 <= status < 400:
                # The client does not follow redirects, and a redirect body is
                # not a completion; treat it as a provider-level failure.
                failures.append(ProviderFailure(name, f"http_{status}:redirect", status))
                return None

            if status >= 400:
                # A 4xx that is not auth/rate-limit is almost always a request the
                # provider dislikes (unsupported parameter, model retired, ...).
                failures.append(
                    ProviderFailure(name, f"http_{status}:{_error_snippet(response)}", status)
                )
                return None

            try:
                body = response.json()
                content = _extract_content(body)
            except (ValueError, LLMError) as exc:
                failures.append(ProviderFailure(name, f"bad_response:{exc}", status))
                return None

            parsed = None
            if response_model is not None:
                try:
                    parsed = _parse_structured(content, response_model)
                except LLMError as exc:
                    failures.append(
                        ProviderFailure(name, f"invalid_structured_output:{exc}", status)
                    )
                    return None

            usage = body.get("usage")
            if not isinstance(usage, dict):
                usage = {}
            tokens = usage.get("total_tokens")
            if not isinstance(tokens, int):
                tokens = int(usage.get("prompt_tokens") or 0) + int(
                    usage.get("completion_tokens") or 0
                )
            model_used = body.get("model")
            if not isinstance(model_used, str) or not model_used:
                model_used = cfg.model

            return LLMResult(
                content=content,
                usage_tokens=tokens,
                latency_ms=latency_ms,
                provider_used=name,
                model_used=model_used,
                cache_hit=False,
                parsed=parsed,
            )

        return None

    def _backoff(self, attempt: int) -> float:
        return compute_backoff(
            attempt,
            base_s=self._config.backoff_base_s,
            cap_s=self._config.backoff_cap_s,
            rng=self._rng,
        )

    @staticmethod
    def _build_payload(
        cfg: LLMProviderConfig,
        messages: Messages,
        *,
        response_model: type[BaseModel] | None,
        temperature: float | None,
        max_tokens: int | None,
        reasoning_effort: str | None,
    ) -> dict[str, Any]:
        outgoing = [dict(message) for message in messages]
        if response_model is not None and cfg.structured_output != "json_schema":
            # Providers without native schema enforcement still get the
            # contract. Merge into an existing leading system message so chat
            # templates do not see two consecutive system turns.
            instruction = _schema_instruction(response_model)
            merged = False
            if outgoing and outgoing[0].get("role") == "system":
                existing = outgoing[0].get("content")
                if isinstance(existing, str):
                    outgoing[0] = {
                        **outgoing[0],
                        "content": f"{instruction}\n\n{existing}",
                    }
                    merged = True
            if not merged:
                outgoing.insert(0, {"role": "system", "content": instruction})

        payload: dict[str, Any] = {
            "model": cfg.model,
            "messages": outgoing,
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        effective_max_tokens = max_tokens if max_tokens is not None else cfg.max_tokens
        if effective_max_tokens is not None:
            payload["max_tokens"] = effective_max_tokens
        effort = reasoning_effort if reasoning_effort is not None else cfg.reasoning_effort
        if effort is not None:
            payload["reasoning_effort"] = effort

        if response_model is not None:
            if cfg.structured_output == "json_schema":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": _strict_json_schema(response_model.model_json_schema()),
                        "strict": True,
                    },
                }
            elif cfg.structured_output == "json_object":
                payload["response_format"] = {"type": "json_object"}

        return payload


def _schema_instruction(response_model: type[BaseModel]) -> str:
    schema = json.dumps(
        _strict_json_schema(response_model.model_json_schema()), sort_keys=True
    )
    return (
        "Respond with a single JSON object and nothing else. "
        f"It must validate against this JSON Schema:\n{schema}"
    )


# Keys whose values are data (defaults, examples, enums), not subschemas.
_SCHEMA_METADATA_KEYS = frozenset({"default", "examples", "const", "enum"})


def _strict_json_schema(schema: Any) -> Any:
    """Close every object and require every property for strict schema mode.

    OpenAI-compatible strict validators reject a schema where an object lacks
    ``additionalProperties: false`` or where a property is absent from
    ``required``. Pydantic only emits the former for ``extra="forbid"`` models
    and leaves optional fields out of ``required``.

    Values under ``default``/``examples``/``const``/``enum`` are data, not
    schemas, so they are copied verbatim. A schema-valued
    ``additionalProperties`` (a free-form map field) is left intact because
    strict mode does not support it.
    """
    if isinstance(schema, dict):
        normalised = {
            key: (
                value if key in _SCHEMA_METADATA_KEYS else _strict_json_schema(value)
            )
            for key, value in schema.items()
        }
        if normalised.get("type") == "object":
            if not isinstance(normalised.get("additionalProperties"), dict):
                normalised["additionalProperties"] = False
            properties = normalised.get("properties")
            if isinstance(properties, dict):
                normalised["required"] = list(properties)
        return normalised
    if isinstance(schema, list):
        return [_strict_json_schema(item) for item in schema]
    return schema


def _extract_content(body: Any) -> str:
    if not isinstance(body, dict):
        raise LLMError("response body was not a JSON object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMError("response contained no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise LLMError("response choice had no message")
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    if not isinstance(content, str) or not content.strip():
        raise LLMError("response contained empty content")
    return content


def _parse_structured(text: str, response_model: type[BaseModel]) -> BaseModel:
    cleaned = text.strip()
    # Reasoning models (e.g. Nemotron Ultra) wrap answers in
    # <think>...</think> blocks; strip them before JSON parsing.
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Some models omit the opening tag and only emit a closing one; drop the
    # reasoning prefix in that case.
    cleaned = re.sub(
        r"^.*?</think>",
        "",
        cleaned,
        count=1,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    fence = re.search(
        r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```",
        cleaned,
        flags=re.DOTALL,
    )
    if fence is not None:
        cleaned = fence.group(1).strip()
    elif cleaned.startswith("```"):
        # Unclosed fence: drop the opening marker and an optional language tag.
        cleaned = cleaned[3:]
        if cleaned[:4].lower() == "json":
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Some models wrap JSON in prose; decode the first complete object and
        # ignore anything that follows it (including stray closing braces).
        start = cleaned.find("{")
        if start == -1:
            raise LLMError("model did not return valid JSON") from None
        try:
            data, _end = json.JSONDecoder().raw_decode(cleaned, start)
        except json.JSONDecodeError as exc:
            raise LLMError(f"model did not return valid JSON: {exc}") from exc

    try:
        return response_model.model_validate(data)
    except ValidationError as exc:
        raise LLMError(f"structured output failed validation: {exc}") from exc


def _error_snippet(response: httpx.Response) -> str:
    try:
        text = response.text[:1000]
    except Exception:  # pragma: no cover - defensive
        return "unreadable"
    return " ".join(text.split())[:200]
