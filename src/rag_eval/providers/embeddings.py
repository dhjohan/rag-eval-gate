"""Embedding clients for Google Gemini and Jina, with a mandatory SQLite cache.

A single provider is chosen per call so a batch never mixes vector spaces. If
that provider fails and fallback is enabled, the *whole* batch is re-embedded by
the next provider rather than stitched together from two models.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import numpy as np

from rag_eval.cache.sqlite_cache import SqliteCache, sha256_text
from rag_eval.config import EmbeddingProviderConfig, EmbeddingsConfig, resolve_api_key
from rag_eval.providers.rate_limit import compute_backoff

_JINA_TASKS = {
    "RETRIEVAL_DOCUMENT": "retrieval.passage",
    "RETRIEVAL_QUERY": "retrieval.query",
    "SEMANTIC_SIMILARITY": "text-matching",
    "CLASSIFICATION": "classification",
    "CLUSTERING": "clustering",
}


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider fails or returns unusable data."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


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

    @property
    def max_batch_size(self) -> int:
        """Maximum number of texts per request for this backend."""
        ...

    def embed(
        self,
        texts: Sequence[str],
        *,
        task_type: str | None,
        on_vector: Callable[[int, Sequence[float]], None] | None = None,
    ) -> list[list[float]]: ...


class GoogleEmbeddingBackend:
    """Gemini embeddings through the official ``google-genai`` SDK."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dims: int,
        timeout_s: float = 60.0,
        client: Any | None = None,
    ) -> None:
        self._api_key = api_key
        self.model = model
        self.dims = dims
        self.timeout_s = timeout_s
        self._client = client

    @property
    def max_batch_size(self) -> int:
        # Gemini embeds one text at a time to keep cache keys 1:1.
        return 1

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)),
            )
        return self._client

    def embed(
        self,
        texts: Sequence[str],
        *,
        task_type: str | None,
        on_vector: Callable[[int, Sequence[float]], None] | None = None,
    ) -> list[list[float]]:
        from google.genai import types

        config_kwargs: dict[str, Any] = {"output_dimensionality": self.dims}
        if task_type:
            config_kwargs["task_type"] = task_type
        config = types.EmbedContentConfig(**config_kwargs)

        client = self._get_client()
        vectors: list[list[float]] = []
        for position, text in enumerate(texts):
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
            vector = list(values)
            vectors.append(vector)
            # Report progress so a mid-batch failure does not throw away the
            # vectors that were already paid for.
            if on_vector is not None:
                on_vector(position, vector)
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

    @property
    def max_batch_size(self) -> int:
        # 64 chunks * ~450 tokens stays ~22k-28k tokens per request, safely
        # under Jina Free Tier's 100k TPM and well below the 1024-item cap.
        return 64

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def embed(
        self,
        texts: Sequence[str],
        *,
        task_type: str | None,
        on_vector: Callable[[int, Sequence[float]], None] | None = None,
    ) -> list[list[float]]:
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
            raise EmbeddingError(f"jina embedding transport error: {exc}", retryable=True) from exc
        if response.status_code >= 400:
            retryable = response.status_code == 429 or response.status_code >= 500
            raise EmbeddingError(
                f"jina embedding HTTP {response.status_code}: {response.text[:200]}",
                retryable=retryable,
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise EmbeddingError(f"jina embedding returned invalid JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise EmbeddingError("jina embedding returned a non-object payload")
        items = body.get("data")
        if not isinstance(items, list) or len(items) != len(texts):
            raise EmbeddingError("jina embedding returned an unexpected payload")
        try:
            ordered = sorted(items, key=lambda item: item.get("index", 0))
            vectors = [list(item["embedding"]) for item in ordered]
        except (AttributeError, KeyError, TypeError) as exc:
            raise EmbeddingError(
                f"jina embedding returned an unexpected payload: {exc}"
            ) from exc
        if on_vector is not None:
            for position, vector in enumerate(vectors):
                on_vector(position, vector)
        return vectors


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
        max_retries: int = 2,
        backoff_base_s: float = 1.0,
        backoff_cap_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._config = config
        self._cache = cache
        self._resolve_key = api_key_resolver or (lambda name: resolve_api_key(name))
        self._backend_factory = backend_factory or (
            lambda name, cfg, key: _default_backend(name, cfg, key, transport)
        )
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        self._backoff_cap_s = backoff_cap_s
        self._sleep = sleep
        self._rng = rng or random.Random(0)

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        if not texts:
            dims = (
                self._config.providers[self._config.ordered_providers[0]].dims
                if self._config.ordered_providers
                else 0
            )
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
            cache_misses = len(missing)
            indices_by_hash: dict[str, list[int]] = {}
            for index in missing:
                indices_by_hash.setdefault(hashes[index], []).append(index)

            backend: EmbeddingBackend | None = None
            try:
                # Build the backend once and reuse it across retries so a
                # transient failure does not pay for a new TLS handshake.
                backend = self._backend_factory(name, cfg, api_key)
                for attempt in range(self._max_retries + 1):
                    # Recompute from the cache each attempt: vectors stored by an
                    # earlier sub-batch are excluded, so a retry never re-sends them.
                    pending = [index for index in missing if cached[index] is None]
                    if not pending:
                        break
                    batch: list[int] = []
                    seen: set[str] = set()
                    for index in pending:
                        digest = hashes[index]
                        if digest not in seen:
                            seen.add(digest)
                            batch.append(index)
                    try:
                        self._embed_batch(
                            backend,
                            cfg,
                            model_id,
                            texts,
                            hashes,
                            batch,
                            indices_by_hash,
                            effective_task,
                            cached,
                        )
                    except EmbeddingError as exc:
                        if exc.retryable and attempt < self._max_retries:
                            self._sleep(
                                compute_backoff(
                                    attempt,
                                    base_s=self._backoff_base_s,
                                    cap_s=self._backoff_cap_s,
                                    rng=self._rng,
                                )
                            )
                            continue
                        failures.append(f"{name}: {exc}")
                        break
            except EmbeddingError as exc:
                failures.append(f"{name}: {exc}")
                continue
            finally:
                if backend is not None:
                    close = getattr(backend, "close", None)
                    if callable(close):
                        close()

            if any(cached[index] is None for index in missing):
                continue

            return EmbeddingResult(
                vectors=np.vstack([vector for vector in cached if vector is not None]),
                provider_used=name,
                model_used=cfg.model,
                cache_hits=hits,
                cache_misses=cache_misses,
            )

        raise AllEmbeddingProvidersFailed(
            "no embedding provider available: " + "; ".join(failures)
        )

    def _embed_batch(
        self,
        backend: EmbeddingBackend,
        cfg: EmbeddingProviderConfig,
        model_id: str,
        texts: Sequence[str],
        hashes: Sequence[str],
        batch: Sequence[int],
        indices_by_hash: dict[str, list[int]],
        task_type: str | None,
        cached: list[np.ndarray | None],
    ) -> None:
        """Embed one deduplicated batch, caching each vector as it arrives.

        The batch is a list of indices into ``texts`` (it may be sparse when
        only some cache entries were missing). It is split into sub-batches no
        larger than ``backend.max_batch_size`` so providers with per-request
        item caps (e.g. Jina) never reject the whole batch.
        """

        max_batch = backend.max_batch_size
        if max_batch <= 0:
            raise EmbeddingError(
                f"backend {type(backend).__name__} has invalid max_batch_size={max_batch}"
            )

        def store(absolute_index: int, vector: Sequence[float]) -> None:
            digest = hashes[absolute_index]
            normalised = _normalise(vector, cfg.dims)
            self._cache.put_embedding(digest, model_id, cfg.dims, normalised)
            for duplicate in indices_by_hash[digest]:
                cached[duplicate] = normalised

        embedded = 0
        for offset in range(0, len(batch), max_batch):
            sub_batch = batch[offset : offset + max_batch]
            sub_texts = [texts[index] for index in sub_batch]

            def on_vector(
                position: int, vector: Sequence[float], *, _sub=sub_batch
            ) -> None:
                store(_sub[position], vector)

            fresh = backend.embed(
                sub_texts,
                task_type=task_type,
                on_vector=on_vector,
            )
            if len(fresh) != len(sub_batch):
                raise EmbeddingError(
                    f"backend returned {len(fresh)} vectors for {len(sub_batch)} inputs"
                )
            for position, vector in enumerate(fresh):
                absolute_index = sub_batch[position]
                if cached[absolute_index] is None:
                    store(absolute_index, vector)
            embedded += len(fresh)

        if embedded != len(batch):
            raise EmbeddingError(
                f"accumulated {embedded} vectors for {len(batch)} inputs"
            )


def _default_backend(
    name: str,
    cfg: EmbeddingProviderConfig,
    api_key: str,
    transport: httpx.BaseTransport | None,
) -> EmbeddingBackend:
    model = cfg.model.lower()
    if model.startswith("gemini-embedding"):
        return GoogleEmbeddingBackend(
            api_key=api_key,
            model=cfg.model,
            dims=cfg.dims,
            timeout_s=cfg.timeout_s,
        )
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
    try:
        array = np.asarray(vector, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise EmbeddingError(f"embedding vector is not numeric: {exc}") from exc
    if array.ndim != 1:
        raise EmbeddingError(f"embedding must be 1-D, got shape {array.shape}")
    if array.size != dims:
        raise EmbeddingError(f"embedding has {array.size} dims, expected {dims}")
    if not np.all(np.isfinite(array)):
        raise EmbeddingError("embedding vector contains non-finite values")
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise EmbeddingError("embedding vector has zero norm")
    return (array / norm).astype(np.float32)


