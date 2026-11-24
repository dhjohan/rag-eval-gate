from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from rag_eval.config import RerankConfig
from rag_eval.providers.rerank import Reranker, RerankError

Handler = Callable[[httpx.Request], httpx.Response]


def make_reranker(
    handler: Handler,
    *,
    key: str | None = "secret",
    sleep: Callable[[float], None] | None = None,
    max_retries: int = 2,
    **overrides: Any,
) -> Reranker:
    payload: dict[str, Any] = {
        "enabled": True,
        "base_url": "https://api.jina.ai/v1",
        "model": "jina-reranker-v3.5",
        "api_key_env": "JINA_API_KEY",
        "top_n": 10,
    }
    payload.update(overrides)
    config = RerankConfig.model_validate(payload)
    return Reranker(
        config,
        api_key_resolver=lambda name: key,
        transport=httpx.MockTransport(handler),
        sleep=sleep or (lambda _: None),
        max_retries=max_retries,
    )


def test_request_shape_and_ordering() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.2},
                    {"index": 0, "relevance_score": 0.9},
                ]
            },
        )

    reranker = make_reranker(handler)
    hits = reranker.rerank("what is RAG", ["doc a", "doc b"])

    assert [hit.index for hit in hits] == [0, 1]
    assert hits[0].score == pytest.approx(0.9)
    assert seen[0]["model"] == "jina-reranker-v3.5"
    assert seen[0]["query"] == "what is RAG"
    assert seen[0]["documents"] == ["doc a", "doc b"]
    assert seen[0]["return_documents"] is False


def test_top_n_is_clamped_to_document_count() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"results": [{"index": i, "relevance_score": 1.0 - i / 10} for i in range(5)]},
        )

    reranker = make_reranker(handler)
    hits = reranker.rerank("q", ["a", "b", "c", "d", "e"], top_n=99)

    assert len(hits) == 5


def test_top_n_limits_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"results": [{"index": i, "relevance_score": 1.0 - i / 10} for i in range(5)]},
        )

    reranker = make_reranker(handler)
    hits = reranker.rerank("q", ["a", "b", "c", "d", "e"], top_n=2)

    assert len(hits) == 2


def test_empty_documents_short_circuit() -> None:
    reranker = make_reranker(lambda request: httpx.Response(500))
    assert reranker.rerank("q", []) == []


def test_http_error_is_raised() -> None:
    reranker = make_reranker(lambda request: httpx.Response(400, text="bad request"))
    with pytest.raises(RerankError, match="400"):
        reranker.rerank("q", ["a"])


def test_out_of_range_index_is_rejected() -> None:
    reranker = make_reranker(
        lambda request: httpx.Response(200, json={"results": [{"index": 7, "relevance_score": 1}]})
    )
    with pytest.raises(RerankError, match="out-of-range"):
        reranker.rerank("q", ["a"])


def test_missing_key_is_reported() -> None:
    reranker = make_reranker(lambda request: httpx.Response(200, json={"results": []}), key=None)
    assert reranker.is_configured is False
    with pytest.raises(RerankError, match="missing API key"):
        reranker.rerank("q", ["a"])


def test_malformed_payload_is_rejected() -> None:
    reranker = make_reranker(lambda request: httpx.Response(200, json={"oops": []}))
    with pytest.raises(RerankError, match="results"):
        reranker.rerank("q", ["a"])


def test_non_json_success_is_wrapped() -> None:
    reranker = make_reranker(lambda request: httpx.Response(200, text="<html>blocked</html>"))
    with pytest.raises(RerankError, match="invalid JSON"):
        reranker.rerank("q", ["a"])


def test_non_object_payload_is_rejected() -> None:
    reranker = make_reranker(lambda request: httpx.Response(200, json=[1, 2, 3]))
    with pytest.raises(RerankError, match="not a JSON object"):
        reranker.rerank("q", ["a"])


def test_429_is_retried_then_succeeds() -> None:
    sleeps: list[float] = []
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "2"})
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 1.0}]})

    reranker = make_reranker(handler, sleep=sleeps.append)
    hits = reranker.rerank("q", ["a"])

    assert attempts == 2
    assert sleeps == [2.0]
    assert hits[0].index == 0


def test_5xx_exhausts_retries() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, text="down")

    reranker = make_reranker(handler, max_retries=2)
    with pytest.raises(RerankError, match="503"):
        reranker.rerank("q", ["a"])

    assert attempts == 3
