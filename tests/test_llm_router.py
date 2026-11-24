from __future__ import annotations

import json
import random
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from rag_eval.cache.sqlite_cache import SqliteCache
from rag_eval.config import LLMConfig
from rag_eval.providers.llm import AllProvidersFailed, LLMRouter, _strict_json_schema


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


def test_llm_cache_key_includes_provider_and_model(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    # Same prompt, same temperature, different model/provider: must NOT be a
    # stale cache hit.
    providers = {
        "a": provider(base_url="https://a.test/v1", model="model-A"),
        "b": provider(base_url="https://b.test/v1", model="model-B"),
    }
    config = llm_config(providers, order=["a", "b"], max_retries=0)

    calls = 0

    def handler(request: httpx.Request, n: int) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = request.read().decode()
        model = "model-A" if "model-A" in body else "model-B"
        return httpx.Response(
            200,
            json=ok_body(content=f"reply from {model}", model=model),
        )

    router, _calls = make_router(config, cache, handler, keys=("TEST_API_KEY",))

    first = router.generate([{"role": "user", "content": "hi"}])
    assert first.provider_used == "a"
    assert first.model_used == "model-A"
    assert first.cache_hit is False

    second = router.generate([{"role": "user", "content": "hi"}])
    # Same provider + same model: must be a cache hit.
    assert second.provider_used == "a"
    assert second.model_used == "model-A"
    assert second.cache_hit is True
    assert calls == 1


def test_think_tags_stripped_before_structured_parse(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    providers = {
        "a": provider(base_url="https://a.test/v1"),
    }
    config = llm_config(providers, order=["a"], max_retries=0)

    def handler(request: httpx.Request, n: int) -> httpx.Response:
        return httpx.Response(
            200,
            json=ok_body(
                content=(
                    "<think>checking the rubric</think>\n"
                    '```json\n{"score": 7, "reason": "ok"}\n```'
                ),
                model="nvidia/nemotron-3-ultra-550b-a55b",
            ),
        )

    router, _calls = make_router(config, cache, handler, keys=("TEST_API_KEY",))

    result = router.generate(
        [{"role": "user", "content": "judge this"}],
        response_model=Verdict,
    )

    assert result.provider_used == "a"
    assert isinstance(result.parsed, Verdict)
    assert result.parsed.score == 7
    assert result.parsed.reason == "ok"


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
    assert isinstance(result.parsed, Verdict)
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

    assert isinstance(result.parsed, Verdict)
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
    router, calls = make_router(
        config,
        cache,
        lambda req, n: httpx.Response(
            200, json=ok_body(model="nvidia/nemotron-3.5-super-vl:free")
        ),
    )

    first = router.generate([{"role": "user", "content": "hi"}])
    second = router.generate([{"role": "user", "content": "hi"}])

    assert first.model_used == "nvidia/nemotron-3.5-super-vl:free"
    # The cache is keyed by the requested model, while the routed model is
    # replayed as provenance.
    assert second.cache_hit is True
    assert second.model_used == "nvidia/nemotron-3.5-super-vl:free"
    assert len(calls) == 1


def test_extra_headers_are_sent(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    config = llm_config({"a": provider(extra_headers={"X-Title": "rag-eval-gate"})})
    router, calls = make_router(config, cache, lambda req, n: httpx.Response(200, json=ok_body()))

    router.generate([{"role": "user", "content": "hi"}])

    assert calls[0].headers["x-title"] == "rag-eval-gate"


def test_429_releases_quota_slot_for_retry(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    config = llm_config({"a": provider(rpd=1)})
    today = datetime.now(UTC).date().isoformat()

    def handler(request: httpx.Request, n: int) -> httpx.Response:
        if n == 1:
            return httpx.Response(429, json={"error": "slow down"})
        return httpx.Response(200, json=ok_body())

    router, calls = make_router(config, cache, handler)
    result = router.generate([{"role": "user", "content": "hi"}])

    assert result.provider_used == "a"
    assert len(calls) == 2
    # The throttled attempt returned its reservation, so the retry fits in a
    # budget of one request per day.
    assert cache.get_quota_used("a", today) == 1


def test_retry_after_http_date_is_honoured(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    sleeps: list[float] = []
    config = llm_config({"a": provider()}, backoff_cap_s=30.0)
    header = format_datetime(datetime.now(UTC) + timedelta(seconds=5), usegmt=True)

    def handler(request: httpx.Request, n: int) -> httpx.Response:
        if n == 1:
            return httpx.Response(429, headers={"retry-after": header})
        return httpx.Response(200, json=ok_body())

    router, calls = make_router(config, cache, handler, sleep=sleeps.append)
    router.generate([{"role": "user", "content": "hi"}])

    assert len(calls) == 2
    assert 3.0 <= sleeps[0] <= 6.0


def test_retry_after_above_cap_fails_over_immediately(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    sleeps: list[float] = []
    providers = {
        "a": provider(base_url="https://a.test/v1"),
        "b": provider(base_url="https://b.test/v1"),
    }
    config = llm_config(providers, order=["a", "b"], backoff_cap_s=5.0)

    def handler(request: httpx.Request, n: int) -> httpx.Response:
        if "a.test" in str(request.url):
            return httpx.Response(429, headers={"retry-after": "100"})
        return httpx.Response(200, json=ok_body(model="b-model"))

    router, calls = make_router(config, cache, handler, sleep=sleeps.append)
    result = router.generate([{"role": "user", "content": "hi"}])

    # Waiting out a 100s cooldown is pointless; fail over to b right away.
    assert result.provider_used == "b"
    assert sleeps == []
    assert sum(1 for call in calls if "a.test" in str(call.url)) == 1


def test_prompt_hash_includes_effective_provider_defaults(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    router_a, _calls_a = make_router(
        llm_config({"a": provider(max_tokens=16)}),
        cache,
        lambda req, n: httpx.Response(200, json=ok_body()),
    )
    router_b, calls_b = make_router(
        llm_config({"a": provider(max_tokens=32)}),
        cache,
        lambda req, n: httpx.Response(200, json=ok_body()),
    )

    router_a.generate([{"role": "user", "content": "hi"}])
    second = router_b.generate([{"role": "user", "content": "hi"}])

    # Changing max_tokens in YAML must not replay the old cached answer.
    assert second.cache_hit is False
    assert len(calls_b) == 1


def test_json_with_trailing_prose_brace_is_parsed(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    config = llm_config({"a": provider()})
    content = 'Verdict: {"score": 4, "reason": "ok"} (note: {not json})'
    router, _calls = make_router(
        config, cache, lambda req, n: httpx.Response(200, json=ok_body(content=content))
    )

    result = router.generate([{"role": "user", "content": "judge"}], response_model=Verdict)

    assert isinstance(result.parsed, Verdict)
    assert result.parsed.score == 4


class _Detail(BaseModel):
    note: str


class _StrictVerdict(BaseModel):
    score: int
    reason: str | None = None
    detail: _Detail | None = None


def test_strict_schema_closes_every_object(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    config = llm_config({"a": provider()})
    payload = ok_body(content='{"score": 1, "reason": "ok", "detail": {"note": "n"}}')
    router, calls = make_router(config, cache, lambda req, n: httpx.Response(200, json=payload))

    router.generate([{"role": "user", "content": "judge"}], response_model=_StrictVerdict)

    schema = json.loads(calls[0].content)["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"score", "reason", "detail"}
    nested = schema["$defs"]["_Detail"]
    assert nested["additionalProperties"] is False
    assert nested["required"] == ["note"]


def test_think_block_json_is_not_mistaken_for_the_answer(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    config = llm_config({"a": provider()})
    content = (
        '<think>I first considered {"score": 0, "reason": "draft"} but that is wrong</think>'
        '{"score": 9, "reason": "final"}'
    )
    router, _calls = make_router(
        config, cache, lambda req, n: httpx.Response(200, json=ok_body(content=content))
    )

    result = router.generate([{"role": "user", "content": "judge"}], response_model=Verdict)

    assert isinstance(result.parsed, Verdict)
    assert result.parsed.score == 9


def test_stray_closing_think_tag_is_stripped(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    config = llm_config({"a": provider()})
    content = 'reasoning without an opening tag</think>{"score": 5, "reason": "ok"}'
    router, _calls = make_router(
        config, cache, lambda req, n: httpx.Response(200, json=ok_body(content=content))
    )

    result = router.generate([{"role": "user", "content": "judge"}], response_model=Verdict)

    assert isinstance(result.parsed, Verdict)
    assert result.parsed.score == 5


def test_corrupt_cache_entry_does_not_hide_a_valid_one(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    providers = {
        "a": provider(base_url="https://a.test/v1", model="model-A"),
        "b": provider(base_url="https://b.test/v1", model="model-B"),
    }
    config = llm_config(providers, order=["a", "b"])
    messages = [{"role": "user", "content": "judge"}]

    def handler(request: httpx.Request, n: int) -> httpx.Response:
        raise AssertionError("a valid cache entry must be served without network")

    router, _calls = make_router(config, cache, handler)
    hash_a = router._prompt_hash(
        messages, Verdict, None, None, None, config.providers["a"]
    )
    cache.put_llm_response(
        hash_a,
        "Verdict",
        content="not json at all",
        provider_used="a",
        requested_model="model-A",
        model_used="model-A",
        usage_tokens=1,
        latency_ms=1,
    )
    hash_b = router._prompt_hash(
        messages, Verdict, None, None, None, config.providers["b"]
    )
    cache.put_llm_response(
        hash_b,
        "Verdict",
        content='{"score": 6, "reason": "cached"}',
        provider_used="b",
        requested_model="model-B",
        model_used="model-B",
        usage_tokens=2,
        latency_ms=2,
    )

    result = router.generate(messages, response_model=Verdict)

    assert result.cache_hit is True
    assert result.provider_used == "b"
    assert isinstance(result.parsed, Verdict)
    assert result.parsed.score == 6


def test_multipart_content_skips_non_string_parts(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    config = llm_config({"a": provider()})
    payload = ok_body()
    payload["choices"][0]["message"]["content"] = [
        {"type": "text", "text": None},
        {"type": "text", "text": "hello "},
        {"type": "text", "text": "world"},
    ]
    router, _calls = make_router(
        config, cache, lambda req, n: httpx.Response(200, json=payload)
    )

    assert router.generate([{"role": "user", "content": "hi"}]).content == "hello world"


@pytest.mark.parametrize("usage", [[], "n/a", 5])
def test_non_object_usage_does_not_crash(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
    usage: Any,
) -> None:
    config = llm_config({"a": provider()})
    body = ok_body()
    body["usage"] = usage
    router, _calls = make_router(
        config, cache, lambda req, n: httpx.Response(200, json=body)
    )

    assert router.generate([{"role": "user", "content": "hi"}]).usage_tokens == 0


def test_schema_instruction_merges_into_existing_system_message(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    config = llm_config({"a": provider(structured_output="none")})
    payload = ok_body(content='{"score": 2, "reason": "ok"}')
    router, calls = make_router(
        config, cache, lambda req, n: httpx.Response(200, json=payload)
    )

    router.generate(
        [
            {"role": "system", "content": "You are a strict judge."},
            {"role": "user", "content": "judge"},
        ],
        response_model=Verdict,
    )

    messages = json.loads(calls[0].content)["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "You are a strict judge." in messages[0]["content"]
    assert "JSON Schema" in messages[0]["content"]


def test_redirect_fails_over_to_next_provider(
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
            return httpx.Response(302, headers={"location": "https://elsewhere.test/"})
        return httpx.Response(200, json=ok_body())

    router, calls = make_router(config, cache, handler)
    result = router.generate([{"role": "user", "content": "hi"}])

    assert result.provider_used == "b"
    assert sum(1 for call in calls if "a.test" in str(call.url)) == 1


def test_prompt_hash_includes_structured_output_mode(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    messages = [{"role": "user", "content": "judge"}]
    content = '{"score": 1, "reason": "ok"}'
    router_none, _calls_none = make_router(
        llm_config({"a": provider(structured_output="none")}),
        cache,
        lambda req, n: httpx.Response(200, json=ok_body(content=content)),
    )
    router_schema, calls_schema = make_router(
        llm_config({"a": provider(structured_output="json_schema")}),
        cache,
        lambda req, n: httpx.Response(200, json=ok_body(content=content)),
    )

    router_none.generate(messages, response_model=Verdict)
    second = router_schema.generate(messages, response_model=Verdict)

    # Same model, different structured-output mode: must not replay the old
    # cached answer.
    assert second.cache_hit is False
    assert len(calls_schema) == 1


def test_code_fence_with_nonstandard_language_tag_is_parsed(
    provider: Callable[..., dict[str, Any]],
    llm_config: Callable[..., LLMConfig],
    cache: SqliteCache,
) -> None:
    config = llm_config({"a": provider()})
    fenced = '```json5\n{"score": 8, "reason": "fenced"}\n```'
    router, _calls = make_router(
        config, cache, lambda req, n: httpx.Response(200, json=ok_body(content=fenced))
    )

    result = router.generate([{"role": "user", "content": "judge"}], response_model=Verdict)

    assert isinstance(result.parsed, Verdict)
    assert result.parsed.score == 8


def test_strict_schema_leaves_defaults_and_map_values_untouched() -> None:
    schema = {
        "type": "object",
        "properties": {
            "nested": {"type": "object", "default": {"type": "object"}},
            "mapping": {"type": "object", "additionalProperties": {"type": "string"}},
        },
    }

    normalised = _strict_json_schema(schema)

    # `default` is data, not a schema, so it must not be rewritten.
    assert "additionalProperties" not in normalised["properties"]["nested"]["default"]
    # A schema-valued additionalProperties (free-form map) is preserved.
    assert normalised["properties"]["mapping"]["additionalProperties"] == {
        "type": "string"
    }
    assert normalised["additionalProperties"] is False
