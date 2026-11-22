from __future__ import annotations

import json
import random
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from rag_eval.cache.sqlite_cache import SqliteCache
from rag_eval.config import LLMConfig
from rag_eval.providers.llm import AllProvidersFailed, LLMRouter


class Verdict(BaseModel):
    score: int
    reason: str


def ok_body(content: str = "hello", model: str = "test/model", tokens: int = 12) -> dict[str, Any]:
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"total_tokens": tokens},
    }


Handler = Callable[[httpx.Request, int], httpx.Response]


def make_router(
    config: LLMConfig,
    cache: SqliteCache,
    handler: Handler,
    *,
    keys: tuple[str, ...] = ("TEST_API_KEY",),
    sleep: Callable[[float], None] | None = None,
) -> tuple[LLMRouter, list[httpx.Request]]:
    calls: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request, len(calls))

    router = LLMRouter(
        config,
        cache,
        api_key_resolver=lambda name: "secret" if name in keys else None,
        transport=httpx.MockTransport(recording),
        sleep=sleep or (lambda _: None),
        rng=random.Random(0),
    )
    return router, calls


def test_generate_returns_normalised_fields(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    config = llm_config({"a": provider()})
    router, calls = make_router(config, cache, lambda req, n: httpx.Response(200, json=ok_body()))

    result = router.generate([{"role": "user", "content": "hi"}])

    assert result.content == "hello"
    assert result.provider_used == "a"
    assert result.model_used == "test/model"
    assert result.usage_tokens == 12
    assert result.cache_hit is False
    assert result.parsed is None
    assert calls[0].headers["authorization"] == "Bearer secret"
    assert json.loads(calls[0].content)["model"] == "test/model"


def test_usage_falls_back_to_prompt_plus_completion(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    body = ok_body()
    body["usage"] = {"prompt_tokens": 7, "completion_tokens": 5}
    config = llm_config({"a": provider()})
    router, _ = make_router(config, cache, lambda req, n: httpx.Response(200, json=body))

    assert router.generate([{"role": "user", "content": "hi"}]).usage_tokens == 12


def test_429_is_retried_and_retry_after_is_honoured(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    sleeps: list[float] = []
    config = llm_config({"a": provider()}).model_copy(update={"backoff_cap_s": 30.0})

    def handler(request: httpx.Request, n: int) -> httpx.Response:
        if n == 1:
            return httpx.Response(429, headers={"retry-after": "7"}, json={"error": "slow down"})
        return httpx.Response(200, json=ok_body())

    router, calls = make_router(config, cache, handler, sleep=sleeps.append)
    result = router.generate([{"role": "user", "content": "hi"}])

    assert result.provider_used == "a"
    assert len(calls) == 2
    assert sleeps == [7.0]


def test_5xx_exhausts_retries_then_falls_back(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    providers = {
        "a": provider(base_url="https://a.test/v1"),
        "b": provider(base_url="https://b.test/v1"),
    }
    config = llm_config(providers, order=["a", "b"])

    def handler(request: httpx.Request, n: int) -> httpx.Response:
        if "a.test" in str(request.url):
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json=ok_body(model="b-model"))

    router, calls = make_router(config, cache, handler)
    result = router.generate([{"role": "user", "content": "hi"}])

    assert result.provider_used == "b"
    assert result.model_used == "b-model"
    assert sum(1 for call in calls if "a.test" in str(call.url)) == 2


def test_missing_api_key_skips_provider(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    providers = {
        "a": provider(base_url="https://a.test/v1", api_key_env="A_KEY"),
        "b": provider(base_url="https://b.test/v1", api_key_env="B_KEY"),
    }
    config = llm_config(providers, order=["a", "b"])
    router, calls = make_router(
        config,
        cache,
        lambda req, n: httpx.Response(200, json=ok_body()),
        keys=("B_KEY",),
    )

    result = router.generate([{"role": "user", "content": "hi"}])

    assert result.provider_used == "b"
    assert len(calls) == 1


def test_unconfigured_provider_is_skipped(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    providers = {
        "a": provider(base_url="https://a.test/${MISSING_VAR}/v1", unresolved_env=["MISSING_VAR"]),
        "b": provider(base_url="https://b.test/v1"),
    }
    config = llm_config(providers, order=["a", "b"])
    router, calls = make_router(config, cache, lambda req, n: httpx.Response(200, json=ok_body()))

    result = router.generate([{"role": "user", "content": "hi"}])

    assert result.provider_used == "b"
    assert len(calls) == 1


def test_exhausted_daily_quota_is_skipped(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    providers = {
        "a": provider(base_url="https://a.test/v1", rpd=1),
        "b": provider(base_url="https://b.test/v1"),
    }
    config = llm_config(providers, order=["a", "b"])
    today = datetime.now(UTC).date().isoformat()
    assert cache.try_consume_quota("a", today, 1) is True

    router, calls = make_router(config, cache, lambda req, n: httpx.Response(200, json=ok_body()))
    result = router.generate([{"role": "user", "content": "hi"}])

    assert result.provider_used == "b"
    assert len(calls) == 1


def test_auth_rejection_does_not_retry(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    providers = {
        "a": provider(base_url="https://a.test/v1"),
        "b": provider(base_url="https://b.test/v1"),
    }
    config = llm_config(providers, order=["a", "b"])

    def handler(request: httpx.Request, n: int) -> httpx.Response:
        if "a.test" in str(request.url):
            return httpx.Response(401, text="bad key")
        return httpx.Response(200, json=ok_body())

    router, calls = make_router(config, cache, handler)
    assert router.generate([{"role": "user", "content": "hi"}]).provider_used == "b"
    assert sum(1 for call in calls if "a.test" in str(call.url)) == 1


def test_transport_error_releases_quota_slot_for_retry(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    providers = {
        "a": provider(base_url="https://a.test/v1", rpd=1),
        "b": provider(base_url="https://b.test/v1"),
    }
    config = llm_config(providers, order=["a", "b"])
    today = datetime.now(UTC).date().isoformat()

    attempts = 0

    def handler(request: httpx.Request, n: int) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("connection reset")
        return httpx.Response(200, json=ok_body())

    router, calls = make_router(config, cache, handler)
    result = router.generate([{"role": "user", "content": "hi"}])

    # The retry must not be suppressed by an unreleased quota slot.
    assert result.provider_used == "a"
    assert attempts == 2
    assert len(calls) == 2
    assert cache.get_quota_used("a", today) == 1


def test_auth_rejection_releases_quota_slot(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    providers = {
        "a": provider(base_url="https://a.test/v1", rpd=1),
        "b": provider(base_url="https://b.test/v1"),
    }
    config = llm_config(providers, order=["a", "b"])
    today = datetime.now(UTC).date().isoformat()

    def handler(request: httpx.Request, n: int) -> httpx.Response:
        if "a.test" in str(request.url):
            return httpx.Response(401, text="bad key")
        return httpx.Response(200, json=ok_body())

    router, _calls = make_router(config, cache, handler)
    result = router.generate([{"role": "user", "content": "hi"}])

    assert result.provider_used == "b"
    # The provider rejected before processing, so the slot is returned.
    assert cache.get_quota_used("a", today) == 0


def test_5xx_keeps_quota_slot(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    providers = {
        "a": provider(base_url="https://a.test/v1", rpd=1),
        "b": provider(base_url="https://b.test/v1"),
    }
    config = llm_config(providers, order=["a", "b"])
    today = datetime.now(UTC).date().isoformat()

    def handler(request: httpx.Request, n: int) -> httpx.Response:
        if "a.test" in str(request.url):
            return httpx.Response(500, text="down")
        return httpx.Response(200, json=ok_body())

    router, _calls = make_router(config, cache, handler)
    result = router.generate([{"role": "user", "content": "hi"}])

    assert result.provider_used == "b"
    # The provider saw and counted the request; the slot stays consumed.
    assert cache.get_quota_used("a", today) == 1


def test_all_providers_failing_raises_with_reasons(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    config = llm_config({"a": provider(base_url="https://a.test/v1")})
    router, _ = make_router(config, cache, lambda req, n: httpx.Response(503, text="down"))

    with pytest.raises(AllProvidersFailed) as excinfo:
        router.generate([{"role": "user", "content": "hi"}])

    assert excinfo.value.failures[0].provider == "a"
    assert excinfo.value.failures[0].status_code == 503


def test_structured_output_uses_json_schema(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    config = llm_config({"a": provider()})
    payload = ok_body(content='{"score": 5, "reason": "relevant"}')
    router, calls = make_router(config, cache, lambda req, n: httpx.Response(200, json=payload))

    result = router.generate([{"role": "user", "content": "judge"}], response_model=Verdict)

    body = json.loads(calls[0].content)
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["name"] == "Verdict"
    assert isinstance(result.parsed, Verdict)
    assert result.parsed.score == 5


def test_structured_output_prefers_schema_capable_provider(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    providers = {
        "plain": provider(base_url="https://plain.test/v1", structured_output="none"),
        "schema": provider(base_url="https://schema.test/v1"),
    }
    config = llm_config(providers, order=["plain", "schema"])
    payload = ok_body(content='{"score": 1, "reason": "ok"}')
    router, calls = make_router(config, cache, lambda req, n: httpx.Response(200, json=payload))

    router.generate([{"role": "user", "content": "judge"}], response_model=Verdict)

    assert len(calls) == 1
    assert "schema.test" in str(calls[0].url)


def test_non_schema_provider_gets_schema_in_system_prompt(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    providers = {"plain": provider(structured_output="none")}
    config = llm_config(providers)
    payload = ok_body(content='{"score": 2, "reason": "ok"}')
    router, calls = make_router(config, cache, lambda req, n: httpx.Response(200, json=payload))

    router.generate([{"role": "user", "content": "judge"}], response_model=Verdict)

    body = json.loads(calls[0].content)
    assert "response_format" not in body
    assert body["messages"][0]["role"] == "system"
    assert "JSON Schema" in body["messages"][0]["content"]
    assert body["messages"][1]["content"] == "judge"


def test_invalid_json_falls_back_to_next_provider(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    providers = {
        "a": provider(base_url="https://a.test/v1"),
        "b": provider(base_url="https://b.test/v1"),
    }
    config = llm_config(providers, order=["a", "b"])

    def handler(request: httpx.Request, n: int) -> httpx.Response:
        if "a.test" in str(request.url):
            return httpx.Response(200, json=ok_body(content="I think the score is five."))
        return httpx.Response(200, json=ok_body(content='{"score": 4, "reason": "ok"}'))

    router, _ = make_router(config, cache, handler)
    result = router.generate([{"role": "user", "content": "judge"}], response_model=Verdict)

    assert result.provider_used == "b"
    assert result.parsed is not None
    assert result.parsed.score == 4


def test_json_inside_code_fence_is_parsed(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    config = llm_config({"a": provider()})
    fenced = '```json\n{"score": 3, "reason": "fenced"}\n```'
    router, _ = make_router(
        config, cache, lambda req, n: httpx.Response(200, json=ok_body(content=fenced))
    )

    result = router.generate([{"role": "user", "content": "judge"}], response_model=Verdict)

    assert result.parsed is not None
    assert result.parsed.score == 3


def test_cache_hit_returns_stored_provider_and_model(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    config = llm_config({"a": provider()})
    router, calls = make_router(
        config, cache, lambda req, n: httpx.Response(200, json=ok_body(model="actual/model"))
    )

    first = router.generate([{"role": "user", "content": "hi"}])
    second = router.generate([{"role": "user", "content": "hi"}])

    assert len(calls) == 1
    assert second.cache_hit is True
    assert second.model_used == "actual/model"
    assert second.provider_used == first.provider_used


def test_cache_can_be_bypassed(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    config = llm_config({"a": provider()})
    router, calls = make_router(config, cache, lambda req, n: httpx.Response(200, json=ok_body()))

    router.generate([{"role": "user", "content": "hi"}], use_cache=False)
    router.generate([{"role": "user", "content": "hi"}], use_cache=False)

    assert len(calls) == 2


def test_openrouter_reports_the_actual_routed_model(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    config = llm_config({"openrouter_free": provider(model="openrouter/free")})
    router, _ = make_router(
        config,
        cache,
        lambda req, n: httpx.Response(
            200, json=ok_body(model="nvidia/nemotron-3.5-super-vl:free")
        ),
    )

    result = router.generate([{"role": "user", "content": "hi"}])

    assert result.model_used == "nvidia/nemotron-3.5-super-vl:free"


def test_extra_headers_are_sent(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    config = llm_config({"a": provider(extra_headers={"X-Title": "rag-eval-gate"})})
    router, calls = make_router(config, cache, lambda req, n: httpx.Response(200, json=ok_body()))

    router.generate([{"role": "user", "content": "hi"}])

    assert calls[0].headers["x-title"] == "rag-eval-gate"
