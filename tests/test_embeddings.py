from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import numpy as np
import pytest

from rag_eval.cache.sqlite_cache import SqliteCache
from rag_eval.config import EmbeddingProviderConfig, EmbeddingsConfig
from rag_eval.providers.embeddings import (
    AllEmbeddingProvidersFailed,
    EmbeddingClient,
    EmbeddingError,
    GoogleEmbeddingBackend,
    JinaEmbeddingBackend,
    _default_backend,
    _normalise,
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

    def embed(
        self,
        texts: Sequence[str],
        *,
        task_type: str | None,
        on_vector: Callable[[int, Sequence[float]], None] | None = None,
    ) -> list[list[float]]:
        self.calls.append((list(texts), task_type))
        if self._fail:
            raise EmbeddingError("provider exploded")
        vectors = [unit_vector()[: self._dims] for _ in texts]
        if on_vector is not None:
            for position, vector in enumerate(vectors):
                on_vector(position, vector)
        return vectors

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
        def embed(
            self,
            texts: Sequence[str],
            *,
            task_type: str | None,
            on_vector: Callable[[int, Sequence[float]], None] | None = None,
        ) -> list[list[float]]:
            vector = [0.0] * DIMS
            vector[0], vector[1] = 3.0, 4.0
            vectors = [list(vector) for _ in texts]
            if on_vector is not None:
                for position, item in enumerate(vectors):
                    on_vector(position, item)
            return vectors

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


def test_jina_backend_wraps_non_json_success() -> None:
    backend = JinaEmbeddingBackend(
        api_key="secret",
        model="jina-embeddings-v5-text-nano",
        dims=DIMS,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text="<html>blocked</html>")
        ),
    )

    with pytest.raises(EmbeddingError, match="invalid JSON"):
        backend.embed(["alpha"], task_type=None)


def test_duplicate_texts_are_embedded_once(
    embedding_provider: Callable[..., dict[str, Any]],
    embeddings_config: Callable[..., EmbeddingsConfig],
    cache: SqliteCache,
) -> None:
    backend = RecordingBackend()
    client = single_client(embedding_provider(), embeddings_config, cache, backend)

    result = client.embed_texts(["duplicate", "duplicate"])

    assert backend.calls[-1][0] == ["duplicate"]
    assert result.cache_misses == 2
    np.testing.assert_allclose(result.vectors[0], result.vectors[1])


class PerItemFlakyBackend:
    """Fails on the second item of its first call, like a per-item SDK."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed(
        self,
        texts: Sequence[str],
        *,
        task_type: str | None,
        on_vector: Callable[[int, Sequence[float]], None] | None = None,
    ) -> list[list[float]]:
        self.batches.append(list(texts))
        vectors: list[list[float]] = []
        for position, _text in enumerate(texts):
            if len(self.batches) == 1 and position == 1:
                raise EmbeddingError("second item failed")
            vector = unit_vector()
            vectors.append(vector)
            if on_vector is not None:
                on_vector(position, vector)
        return vectors


def test_retry_reuses_backend_and_keeps_partial_progress(
    embedding_provider: Callable[..., dict[str, Any]],
    embeddings_config: Callable[..., EmbeddingsConfig],
    cache: SqliteCache,
) -> None:
    backend = PerItemFlakyBackend()
    constructed: list[str] = []

    def factory(name: str, cfg: Any, key: str) -> PerItemFlakyBackend:
        constructed.append(name)
        return backend

    client = EmbeddingClient(
        embeddings_config({"google": embedding_provider()}),
        cache,
        api_key_resolver=lambda name: "secret",
        backend_factory=factory,
        sleep=lambda _: None,
    )

    result = client.embed_texts(["alpha", "beta", "gamma"])

    assert constructed == ["google"]  # one backend reused across the retry
    assert backend.batches == [["alpha", "beta", "gamma"], ["beta", "gamma"]]
    assert result.cache_misses == 3
    assert result.vectors.shape == (3, DIMS)


def test_google_backend_receives_configured_timeout(
    embedding_provider: Callable[..., dict[str, Any]],
) -> None:
    cfg = EmbeddingProviderConfig.model_validate(embedding_provider(timeout_s=17))
    backend = _default_backend("google", cfg, "key", None)

    assert isinstance(backend, GoogleEmbeddingBackend)
    assert backend.timeout_s == 17


def test_normalise_rejects_non_numeric_vector() -> None:
    with pytest.raises(EmbeddingError, match="not numeric"):
        _normalise(["not", "a", "number"], 3)  # type: ignore


def test_normalise_rejects_nan_and_inf() -> None:
    with pytest.raises(EmbeddingError, match="non-finite"):
        _normalise([float("nan"), 0.0, 0.0], 3)
    with pytest.raises(EmbeddingError, match="non-finite"):
        _normalise([float("inf"), 0.0, 0.0], 3)


def test_normalise_rejects_wrong_shape() -> None:
    with pytest.raises(EmbeddingError, match="1-D"):
        _normalise([[0.1, 0.2]], 2)  # type: ignore[arg-type, list-item]
    with pytest.raises(EmbeddingError, match="dims"):
        _normalise([0.1, 0.2, 0.3], 2)


def test_normalise_rejects_zero_norm() -> None:
    with pytest.raises(EmbeddingError, match="zero norm"):
        _normalise([0.0, 0.0, 0.0], 3)


class PermanentFailureBackend:
    """Always fails with a non-retryable error (e.g. bad API key)."""

    call_count = 0

    def embed(
        self,
        texts: Sequence[str],
        *,
        task_type: str | None,
        on_vector: Callable[[int, Sequence[float]], None] | None = None,
    ) -> list[list[float]]:
        PermanentFailureBackend.call_count += 1
        raise EmbeddingError("invalid API key", retryable=False)


def test_non_retryable_error_fails_over_without_retry(
    embedding_provider: Callable[..., dict[str, Any]],
    embeddings_config: Callable[..., EmbeddingsConfig],
    cache: SqliteCache,
) -> None:
    PermanentFailureBackend.call_count = 0
    providers = {
        "google": embedding_provider(base_url="https://g.test/v1"),
        "jina": embedding_provider(
            base_url="https://j.test/v1", model="jina-embeddings-v5-text-nano"
        ),
    }
    config = embeddings_config(providers, order=["google", "jina"], fallback=True)
    client = EmbeddingClient(
        config,
        cache,
        api_key_resolver=lambda name: "secret",
        backend_factory=lambda name, cfg, key: PermanentFailureBackend(),
        sleep=lambda _: None,
    )

    with pytest.raises(AllEmbeddingProvidersFailed, match="google"):
        client.embed_texts(["alpha"])

    # Each provider tried once; no retries on a non-retryable error.
    assert PermanentFailureBackend.call_count == 2


def test_retryable_429_is_retried_before_failing_over(
    embedding_provider: Callable[..., dict[str, Any]],
    embeddings_config: Callable[..., EmbeddingsConfig],
    cache: SqliteCache,
) -> None:
    attempts = 0

    def flaky_embed(
        texts: Sequence[str], *, task_type: str | None, on_vector=None
    ) -> list[list[float]]:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise EmbeddingError("rate limited", retryable=True)
        return [unit_vector() for _ in texts]

    providers = {
        "google": embedding_provider(base_url="https://g.test/v1"),
    }
    config = embeddings_config(providers, order=["google"], fallback=False)
    client = EmbeddingClient(
        config,
        cache,
        api_key_resolver=lambda name: "secret",
        backend_factory=lambda name, cfg, key: type(
            "F", (), {"embed": staticmethod(flaky_embed), "close": lambda self: None}
        )(),
        sleep=lambda _: None,
    )

    result = client.embed_texts(["alpha"])

    assert attempts == 2
    assert result.provider_used == "google"
