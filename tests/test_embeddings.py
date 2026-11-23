from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import numpy as np
import pytest

from rag_eval.cache.sqlite_cache import SqliteCache
from rag_eval.config import EmbeddingsConfig
from rag_eval.providers.embeddings import (
    AllEmbeddingProvidersFailed,
    EmbeddingClient,
    EmbeddingError,
    GoogleEmbeddingBackend,
    JinaEmbeddingBackend,
)

DIMS = 64


def unit_vector(first: float = 1.0) -> list[float]:
    vector = [0.0] * DIMS
    vector[0] = first
    return vector


class RecordingBackend:
    def __init__(self, *, fail: bool = False, dims: int = DIMS) -> None:
        self.calls: list[tuple[list[str], str | None]] = []
        self._fail = fail
        self._dims = dims

    def embed(self, texts: Sequence[str], *, task_type: str | None) -> list[list[float]]:
        self.calls.append((list(texts), task_type))
        if self._fail:
            raise EmbeddingError("provider exploded")
        return [unit_vector()[: self._dims] for _ in texts]

    @property
    def call_count(self) -> int:
        return len(self.calls)


def make_client(
    config: EmbeddingsConfig,
    cache: SqliteCache,
    backends: dict[str, RecordingBackend],
    *,
    keys: tuple[str, ...] = ("TEST_API_KEY",),
) -> EmbeddingClient:
    return EmbeddingClient(
        config,
        cache,
        api_key_resolver=lambda name: "secret" if name in keys else None,
        backend_factory=lambda name, cfg, key: backends[name],
        sleep=lambda _: None,
    )


def single_client(
    provider_cfg: dict[str, Any],
    config_factory: Callable[..., EmbeddingsConfig],
    cache: SqliteCache,
    backend: RecordingBackend,
    *,
    keys: tuple[str, ...] = ("TEST_API_KEY",),
) -> EmbeddingClient:
    config = config_factory({"google": provider_cfg})
    return make_client(config, cache, {"google": backend}, keys=keys)


def test_second_call_is_served_from_cache(
    embedding_provider: Callable[..., dict[str, Any]],
    embeddings_config: Callable[..., EmbeddingsConfig],
    cache: SqliteCache,
) -> None:
    backend = RecordingBackend()
    client = single_client(embedding_provider(), embeddings_config, cache, backend)

    first = client.embed_texts(["alpha", "beta"])
    second = client.embed_texts(["alpha", "beta"])

    assert first.cache_misses == 2
    assert first.cache_hits == 0
    assert second.cache_hits == 2
    assert second.cache_misses == 0
    assert backend.call_count == 1
    np.testing.assert_allclose(first.vectors, second.vectors)


def test_partial_cache_hit_only_embeds_the_missing_text(
    embedding_provider: Callable[..., dict[str, Any]],
    embeddings_config: Callable[..., EmbeddingsConfig],
    cache: SqliteCache,
) -> None:
    backend = RecordingBackend()
    client = single_client(embedding_provider(), embeddings_config, cache, backend)

    client.embed_texts(["alpha"])
    result = client.embed_texts(["alpha", "beta"])

    assert result.cache_hits == 1
    assert result.cache_misses == 1
    assert backend.calls[-1][0] == ["beta"]


def test_vectors_are_l2_normalised(
    embedding_provider: Callable[..., dict[str, Any]],
    embeddings_config: Callable[..., EmbeddingsConfig],
    cache: SqliteCache,
) -> None:
    class SkewedBackend(RecordingBackend):
        def embed(self, texts: Sequence[str], *, task_type: str | None) -> list[list[float]]:
            vector = [0.0] * DIMS
            vector[0], vector[1] = 3.0, 4.0
            return [list(vector) for _ in texts]

    client = make_client(
        embeddings_config({"google": embedding_provider()}), cache, {"google": SkewedBackend()}
    )

    result = client.embed_texts(["alpha"])

    assert result.vectors[0][0] == pytest.approx(0.6)
    assert result.vectors[0][1] == pytest.approx(0.8)
    assert float(np.linalg.norm(result.vectors[0])) == pytest.approx(1.0)


def test_wrong_dimensions_are_rejected(
    embedding_provider: Callable[..., dict[str, Any]],
    embeddings_config: Callable[..., EmbeddingsConfig],
    cache: SqliteCache,
) -> None:
    backend = RecordingBackend(dims=3)
    client = single_client(embedding_provider(), embeddings_config, cache, backend)

    with pytest.raises(AllEmbeddingProvidersFailed, match="dims"):
        client.embed_texts(["alpha"])


def test_fallback_is_off_by_default(
    embedding_provider: Callable[..., dict[str, Any]],
    embeddings_config: Callable[..., EmbeddingsConfig],
    cache: SqliteCache,
) -> None:
    providers = {
        "google": embedding_provider(base_url="https://g.test/v1"),
        "jina": embedding_provider(
            base_url="https://j.test/v1", model="jina-embeddings-v5-text-nano"
        ),
    }
    config = embeddings_config(providers, order=["google", "jina"], fallback=False)
    client = make_client(
        config, cache, {"google": RecordingBackend(fail=True), "jina": RecordingBackend()}
    )

    with pytest.raises(AllEmbeddingProvidersFailed, match="google"):
        client.embed_texts(["alpha"])


def test_fallback_reembeds_the_whole_batch_with_one_model(
    embedding_provider: Callable[..., dict[str, Any]],
    embeddings_config: Callable[..., EmbeddingsConfig],
    cache: SqliteCache,
) -> None:
    providers = {
        "google": embedding_provider(base_url="https://g.test/v1"),
        "jina": embedding_provider(
            base_url="https://j.test/v1", model="jina-embeddings-v5-text-nano"
        ),
    }
    config = embeddings_config(providers, order=["google", "jina"], fallback=True)
    failing, healthy = RecordingBackend(fail=True), RecordingBackend()
    client = make_client(config, cache, {"google": failing, "jina": healthy})

    result = client.embed_texts(["alpha", "beta"])

    assert result.provider_used == "jina"
    assert result.model_used == "jina-embeddings-v5-text-nano"
    assert healthy.calls[-1][0] == ["alpha", "beta"]


def test_missing_key_is_reported(
    embedding_provider: Callable[..., dict[str, Any]],
    embeddings_config: Callable[..., EmbeddingsConfig],
    cache: SqliteCache,
) -> None:
    client = single_client(
        embedding_provider(), embeddings_config, cache, RecordingBackend(), keys=()
    )

    with pytest.raises(AllEmbeddingProvidersFailed, match="missing_api_key"):
        client.embed_texts(["alpha"])


def test_unsupported_embedding_model_is_reported(
    embedding_provider: Callable[..., dict[str, Any]],
    embeddings_config: Callable[..., EmbeddingsConfig],
    cache: SqliteCache,
) -> None:
    config = embeddings_config({"google": embedding_provider(model="text-embedding-3-small")})
    client = EmbeddingClient(
        config,
        cache,
        api_key_resolver=lambda name: "secret",
        sleep=lambda _: None,
    )

    with pytest.raises(AllEmbeddingProvidersFailed, match="unsupported embedding model"):
        client.embed_texts(["alpha"])


def test_empty_input_returns_empty_matrix(
    embedding_provider: Callable[..., dict[str, Any]],
    embeddings_config: Callable[..., EmbeddingsConfig],
    cache: SqliteCache,
) -> None:
    client = make_client(
        embeddings_config({"google": embedding_provider()}), cache, {"google": RecordingBackend()}
    )

    result = client.embed_texts([])

    assert result.vectors.shape == (0, DIMS)
    assert result.model_used == "none"


class _StubEmbedding:
    def __init__(self, values: list[float]) -> None:
        self.values = values


class _StubResponse:
    def __init__(self, values: list[float]) -> None:
        self.embeddings = [_StubEmbedding(values)]


class _StubModels:
    def __init__(self, values: list[float]) -> None:
        self._values = values
        self.calls: list[dict[str, Any]] = []

    def embed_content(self, *, model: str, contents: str, config: Any) -> _StubResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return _StubResponse(self._values)


class _StubClient:
    def __init__(self, values: list[float]) -> None:
        self.models = _StubModels(values)


def test_google_backend_passes_task_type_and_dimensions() -> None:
    stub = _StubClient(unit_vector())
    backend = GoogleEmbeddingBackend(
        api_key="secret", model="gemini-embedding-2", dims=DIMS, client=stub
    )

    vectors = backend.embed(["alpha", "beta"], task_type="RETRIEVAL_DOCUMENT")

    assert len(vectors) == 2
    assert len(stub.models.calls) == 2  # embedding-2 aggregates, so one call per text
    config = stub.models.calls[0]["config"]
    assert config.task_type == "RETRIEVAL_DOCUMENT"
    assert config.output_dimensionality == DIMS


def test_jina_backend_request_shape() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": index, "embedding": unit_vector()}
                    for index in range(len(body["input"]))
                ]
            },
        )

    backend = JinaEmbeddingBackend(
        api_key="secret",
        model="jina-embeddings-v5-text-nano",
        dims=DIMS,
        transport=httpx.MockTransport(handler),
    )

    vectors = backend.embed(["alpha"], task_type="RETRIEVAL_DOCUMENT")

    assert len(vectors) == 1
    assert seen[0]["model"] == "jina-embeddings-v5-text-nano"
    assert seen[0]["dimensions"] == DIMS
    assert seen[0]["task"] == "retrieval.passage"


def test_jina_backend_surfaces_http_errors() -> None:
    backend = JinaEmbeddingBackend(
        api_key="secret",
        model="jina-embeddings-v5-text-nano",
        dims=DIMS,
        transport=httpx.MockTransport(lambda request: httpx.Response(429, text="slow down")),
    )

    with pytest.raises(EmbeddingError, match="429"):
        backend.embed(["alpha"], task_type=None)
