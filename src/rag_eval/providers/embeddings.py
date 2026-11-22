"""Embedding clients for Google Gemini and Jina, with a mandatory SQLite cache.

A single provider is chosen per call so a batch never mixes vector spaces. If
that provider fails and fallback is enabled, the *whole* batch is re-embedded by
the next provider rather than stitched together from two models.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import numpy as np

from rag_eval.cache.sqlite_cache import SqliteCache, sha256_text
from rag_eval.config import EmbeddingProviderConfig, EmbeddingsConfig, resolve_api_key

_JINA_TASKS = {
    "RETRIEVAL_DOCUMENT": "retrieval.passage",
    "RETRIEVAL_QUERY": "retrieval.query",
    "SEMANTIC_SIMILARITY": "text-matching",
    "CLASSIFICATION": "classification",
    "CLUSTERING": "clustering",
}


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider fails or returns unusable data."""


class AllEmbeddingProvidersFailed(EmbeddingError):
    """Raised when no configured embedding provider could serve the request."""


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: np.ndarray
    provider_used: str
    model_used: str
    cache_hits: int
    cache_misses: int


class EmbeddingBackend(Protocol):
    """Minimal seam so tests never need the network."""

    def embed(self, texts: Sequence[str], *, task_type: str | None) -> list[list[float]]: ...


class GoogleEmbeddingBackend:
    """Gemini embeddings through the official ``google-genai`` SDK."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dims: int,
        client: Any | None = None,
    ) -> None:
        self._api_key = api_key
        self.model = model
        self.dims = dims
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def embed(self, texts: Sequence[str], *, task_type: str | None) -> list[list[float]]:
        from google.genai import types

        config_kwargs: dict[str, Any] = {"output_dimensionality": self.dims}
        if task_type:
            config_kwargs["task_type"] = task_type
        config = types.EmbedContentConfig(**config_kwargs)

        client = self._get_client()
        vectors: list[list[float]] = []
        for text in texts:
            # gemini-embedding-2 aggregates multiple inputs into one vector, so
            # each chunk is embedded on its own to keep cache keys 1:1.
            try:
                response = client.models.embed_content(
                    model=self.model, contents=text, config=config
                )
            except Exception as exc:  # SDK raises provider-specific errors
                raise EmbeddingError(f"google embedding failed: {exc}") from exc
            embeddings = getattr(response, "embeddings", None)
            if not embeddings:
                raise EmbeddingError("google embedding returned no vectors")
            values = getattr(embeddings[0], "values", None)
            if not values:
                raise EmbeddingError("google embedding returned an empty vector")
            vectors.append(list(values))
        return vectors


class JinaEmbeddingBackend:
    """Jina embeddings over plain HTTP, so ``httpx.MockTransport`` can drive it."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dims: int,
        base_url: str = "https://api.jina.ai/v1",
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self.model = model
        self.dims = dims
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._client = client or httpx.Client(transport=transport)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def embed(self, texts: Sequence[str], *, task_type: str | None) -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": list(texts),
            "dimensions": self.dims,
        }
        if task_type:
            payload["task"] = _JINA_TASKS.get(task_type.upper(), task_type)
        try:
            response = self._client.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=self.timeout_s,
            )
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"jina embedding transport error: {exc}") from exc
        if response.status_code >= 400:
            raise EmbeddingError(
                f"jina embedding HTTP {response.status_code}: {response.text[:200]}"
            )
        body = response.json()
        items = body.get("data")
        if not isinstance(items, list) or len(items) != len(texts):
            raise EmbeddingError("jina embedding returned an unexpected payload")
        ordered = sorted(items, key=lambda item: item.get("index", 0))
        return [list(item["embedding"]) for item in ordered]


class EmbeddingClient:
    """Cache-first embedding facade shared by both providers."""

    def __init__(
        self,
        config: EmbeddingsConfig,
        cache: SqliteCache,
        *,
        api_key_resolver: Callable[[str], str | None] | None = None,
        backend_factory: (
            Callable[[str, EmbeddingProviderConfig, str], EmbeddingBackend] | None
        ) = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._config = config
        self._cache = cache
        self._resolve_key = api_key_resolver or (lambda name: resolve_api_key(name))
        self._backend_factory = backend_factory or (
            lambda name, cfg, key: _default_backend(name, cfg, key, transport)
        )

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        if not texts:
            dims = self._config.providers[self._config.ordered_providers[0]].dims
            return EmbeddingResult(
                vectors=np.zeros((0, dims), dtype=np.float32),
                provider_used="none",
                model_used="none",
                cache_hits=0,
                cache_misses=0,
            )

        failures: list[str] = []
        for name in self._config.ordered_providers:
            cfg = self._config.providers[name]
            if not cfg.is_configured:
                failures.append(f"{name}: unconfigured_env:{','.join(cfg.unresolved_env)}")
                continue
            api_key = self._resolve_key(cfg.api_key_env)
            if not api_key:
                failures.append(f"{name}: missing_api_key:{cfg.api_key_env}")
                continue

            effective_task = task_type if task_type is not None else cfg.task_type
            model_id = f"{cfg.model}@{cfg.dims}:{effective_task or 'default'}"
            hashes = [sha256_text(text) for text in texts]
            cached = [
                self._cache.get_embedding(digest, model_id, cfg.dims) for digest in hashes
            ]
            hits = sum(1 for vector in cached if vector is not None)
            if hits == len(texts):
                return EmbeddingResult(
                    vectors=np.vstack([vector for vector in cached if vector is not None]),
                    provider_used=name,
                    model_used=cfg.model,
                    cache_hits=hits,
                    cache_misses=0,
                )

            missing = [index for index, vector in enumerate(cached) if vector is None]
            backend: EmbeddingBackend | None = None
            try:
                backend = self._backend_factory(name, cfg, api_key)
                fresh = backend.embed(
                    [texts[index] for index in missing], task_type=effective_task
                )
                if len(fresh) != len(missing):
                    raise EmbeddingError(
                        f"returned {len(fresh)} vectors for {len(missing)} inputs"
                    )
                normalised = [_normalise(vector, cfg.dims) for vector in fresh]
            except EmbeddingError as exc:
                failures.append(f"{name}: {exc}")
                continue
            finally:
                if backend is not None:
                    close = getattr(backend, "close", None)
                    if callable(close):
                        close()

            for index, vector in zip(missing, normalised, strict=True):
                self._cache.put_embedding(hashes[index], model_id, cfg.dims, vector)
                cached[index] = vector

            return EmbeddingResult(
                vectors=np.vstack([vector for vector in cached if vector is not None]),
                provider_used=name,
                model_used=cfg.model,
                cache_hits=hits,
                cache_misses=len(missing),
            )

        raise AllEmbeddingProvidersFailed(
            "no embedding provider available: " + "; ".join(failures)
        )


def _default_backend(
    name: str,
    cfg: EmbeddingProviderConfig,
    api_key: str,
    transport: httpx.BaseTransport | None,
) -> EmbeddingBackend:
    model = cfg.model.lower()
    if model.startswith("gemini-embedding"):
        return GoogleEmbeddingBackend(api_key=api_key, model=cfg.model, dims=cfg.dims)
    if model.startswith("jina-"):
        return JinaEmbeddingBackend(
            api_key=api_key,
            model=cfg.model,
            dims=cfg.dims,
            base_url=cfg.base_url,
            timeout_s=cfg.timeout_s,
            transport=transport,
        )
    raise EmbeddingError(
        f"provider {name!r} uses unsupported embedding model {cfg.model!r}; "
        "expected a 'gemini-embedding*' or 'jina-*' model"
    )


def _normalise(vector: Sequence[float], dims: int) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1:
        raise EmbeddingError(f"embedding must be 1-D, got shape {array.shape}")
    if array.size != dims:
        raise EmbeddingError(f"embedding has {array.size} dims, expected {dims}")
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise EmbeddingError("embedding vector has zero norm")
    return (array / norm).astype(np.float32)
