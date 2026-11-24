"""Jina reranker client.

Kept deliberately small: reranking is a single HTTP call whose only job is to
return ``(index, score)`` pairs that preserve the caller's document order.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from rag_eval.config import RerankConfig, resolve_api_key
from rag_eval.providers.rate_limit import compute_backoff, parse_retry_after


class RerankError(RuntimeError):
    """Raised when the reranker is unavailable or returns an unusable payload."""


@dataclass(frozen=True)
class RerankHit:
    index: int
    score: float


class Reranker:
    def __init__(
        self,
        config: RerankConfig,
        *,
        api_key_resolver: Callable[[str], str | None] | None = None,
        transport: httpx.BaseTransport | None = None,
        client: httpx.Client | None = None,
        max_retries: int = 2,
        backoff_base_s: float = 1.0,
        backoff_cap_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._config = config
        self._resolve_key = api_key_resolver or (lambda name: resolve_api_key(name))
        self._client = client or httpx.Client(transport=transport)
        self._owns_client = client is None
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        self._backoff_cap_s = backoff_cap_s
        self._sleep = sleep
        self._rng = rng or random.Random(0)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Reranker:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def is_configured(self) -> bool:
        return (
            self._config.enabled
            and self._config.is_configured
            and bool(self._resolve_key(self._config.api_key_env))
        )

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int | None = None,
    ) -> list[RerankHit]:
        if not documents:
            return []
        api_key = self._resolve_key(self._config.api_key_env)
        if not api_key:
            raise RerankError(f"missing API key: {self._config.api_key_env}")
        if not self._config.enabled:
            raise RerankError("reranking is disabled")
        if not self._config.is_configured:
            missing = ", ".join(self._config.unresolved_env)
            raise RerankError(f"unresolved environment variables: {missing}")

        effective_top_n = top_n if top_n is not None else self._config.top_n
        if effective_top_n < 0:
            raise RerankError(f"top_n must be non-negative, got {effective_top_n}")
        if effective_top_n == 0:
            return []
        limit = min(effective_top_n, len(documents))
        payload: dict[str, Any] = {
            "model": self._config.model,
            "query": query,
            "documents": list(documents),
            "top_n": limit,
            "return_documents": False,
        }
        response = self._post_with_retries(payload, api_key)
        try:
            body = response.json()
        except ValueError as exc:
            raise RerankError(f"rerank returned invalid JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise RerankError("rerank response was not a JSON object")
        results = body.get("results")
        if not isinstance(results, list):
            raise RerankError("rerank response had no 'results' list")

        hits: list[RerankHit] = []
        for item in results:
            if not isinstance(item, dict):
                raise RerankError("rerank result entry was not an object")
            index = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or isinstance(score, bool)
                or not isinstance(score, int | float)
                or not math.isfinite(score)
            ):
                raise RerankError("rerank result entry was missing valid index/score")
            if not 0 <= index < len(documents):
                raise RerankError(f"rerank returned out-of-range index {index}")
            hits.append(RerankHit(index=index, score=float(score)))

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    def _post_with_retries(self, payload: dict[str, Any], api_key: str) -> httpx.Response:
        """POST once, retrying transient statuses and transport errors with backoff."""
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(
                    f"{self._config.base_url.rstrip('/')}/rerank",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    timeout=self._config.timeout_s,
                )
            except httpx.TransportError as exc:
                if attempt < self._max_retries:
                    self._sleep(self._backoff(attempt))
                    continue
                raise RerankError(f"rerank transport error: {exc}") from exc

            status = response.status_code
            if status == 429 or status >= 500:
                if attempt < self._max_retries:
                    retry_after = parse_retry_after(response.headers.get("retry-after"))
                    delay = retry_after if retry_after is not None else self._backoff(attempt)
                    self._sleep(min(delay, self._backoff_cap_s))
                    continue
                raise RerankError(f"rerank HTTP {status}: {response.text[:200]}")
            if status >= 400:
                raise RerankError(f"rerank HTTP {status}: {response.text[:200]}")
            return response
        raise RerankError("rerank failed without a response")  # pragma: no cover

    def _backoff(self, attempt: int) -> float:
        return compute_backoff(
            attempt,
            base_s=self._backoff_base_s,
            cap_s=self._backoff_cap_s,
            rng=self._rng,
        )
