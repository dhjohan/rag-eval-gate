"""Ordered, quota-aware LLM router over OpenAI-compatible free-tier endpoints.

Every provider in the chain speaks the OpenAI Chat Completions shape, so one
HTTP client serves NVIDIA NIM, Gemini's compatibility layer, Cloudflare Workers
AI, and OpenRouter. The router owns retries, throttling, fallback, structured
output, and the prompt cache.
"""

from __future__ import annotations

import json
import random
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
        prompt_hash = sha256_text(
            canonical_json(
                {
                    "messages": [dict(m) for m in messages],
                    "schema": (
                        response_model.model_json_schema() if response_model is not None else None
                    ),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "reasoning_effort": reasoning_effort,
                }
            )
        )

        if cache_enabled:
            cached = self._cache.get_llm_response(prompt_hash, target)
            if cached is not None:
                parsed = None
                if response_model is not None:
                    try:
                        parsed = _parse_structured(cached.content, response_model)
                    except LLMError:
                        cached = None
                if cached is not None:
                    return LLMResult(
                        content=cached.content,
                        usage_tokens=cached.usage_tokens,
                        latency_ms=cached.latency_ms,
                        provider_used=cached.provider_used,
                        model_used=cached.model_used,
                        cache_hit=True,
                        parsed=parsed,
                    )

        failures: list[ProviderFailure] = []
        for name in self._candidate_order(response_model):
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
                    prompt_hash,
                    target,
                    content=result.content,
                    provider_used=result.provider_used,
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
        url = f"{cfg.base_url}/chat/completions"
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

            started = time.perf_counter()
            try:
                response = self._client.post(
                    url, json=payload, headers=headers, timeout=cfg.timeout_s
                )
            except httpx.HTTPError as exc:
                # A transport error may mean the request never reached the
                # provider, so give the quota slot back. A retry re-acquires.
                limiter.release()
                if attempt < self._config.max_retries:
                    self._sleep(self._backoff(attempt))
                    continue
                failures.append(
                    ProviderFailure(name, f"transport_error:{type(exc).__name__}", None)
                )
                return None
            latency_ms = int((time.perf_counter() - started) * 1000)
            status = response.status_code

            if status == 429:
                if attempt < self._config.max_retries:
                    retry_after = parse_retry_after(response.headers.get("retry-after"))
                    delay = retry_after if retry_after is not None else self._backoff(attempt)
                    self._sleep(min(delay, self._config.backoff_cap_s))
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
                limiter.release()
                failures.append(ProviderFailure(name, "auth_rejected", status))
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

            usage = body.get("usage") or {}
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
            # Providers without native schema enforcement still get the contract.
            outgoing = [
                {"role": "system", "content": _schema_instruction(response_model)},
                *outgoing,
            ]

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
        effort = reasoning_effort or cfg.reasoning_effort
        if effort is not None:
            payload["reasoning_effort"] = effort

        if response_model is not None:
            if cfg.structured_output == "json_schema":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": response_model.model_json_schema(),
                        "strict": True,
                    },
                }
            elif cfg.structured_output == "json_object":
                payload["response_format"] = {"type": "json_object"}

        return payload


def _schema_instruction(response_model: type[BaseModel]) -> str:
    schema = json.dumps(response_model.model_json_schema(), sort_keys=True)
    return (
        "Respond with a single JSON object and nothing else. "
        f"It must validate against this JSON Schema:\n{schema}"
    )


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
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str) or not content.strip():
        raise LLMError("response contained empty content")
    return content


def _parse_structured(text: str, response_model: type[BaseModel]) -> BaseModel:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
        if cleaned[:4].lower() == "json":
            cleaned = cleaned[4:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Some models wrap JSON in prose; salvage the outermost object.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMError("model did not return valid JSON") from None
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"model did not return valid JSON: {exc}") from exc

    try:
        return response_model.model_validate(data)
    except ValidationError as exc:
        raise LLMError(f"structured output failed validation: {exc}") from exc


def _error_snippet(response: httpx.Response) -> str:
    try:
        text = response.text
    except Exception:  # pragma: no cover - defensive
        return "unreadable"
    return " ".join(text.split())[:200]
