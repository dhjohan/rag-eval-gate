"""Jina reranker client.

Kept deliberately small: reranking is a single HTTP call whose only job is to
return ``(index, score)`` pairs that preserve the caller's document order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from rag_eval.config import RerankConfig, resolve_api_key


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
    ) -> None:
        self._config = config
        self._resolve_key = api_key_resolver or (lambda name: resolve_api_key(name))
        self._client = client or httpx.Client(transport=transport)
        self._owns_client = client is None

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
        if not self._config.is_configured:
            missing = ", ".join(self._config.unresolved_env)
            raise RerankError(f"unresolved environment variables: {missing}")

        limit = min(top_n if top_n is not None else self._config.top_n, len(documents))
        payload: dict[str, Any] = {
            "model": self._config.model,
            "query": query,
            "documents": list(documents),
            "top_n": limit,
            "return_documents": False,
        }
        try:
            response = self._client.post(
                f"{self._config.base_url}/rerank",
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=self._config.timeout_s,
            )
        except httpx.HTTPError as exc:
            raise RerankError(f"rerank transport error: {exc}") from exc

        if response.status_code >= 400:
            raise RerankError(f"rerank HTTP {response.status_code}: {response.text[:200]}")

        body = response.json()
        results = body.get("results")
        if not isinstance(results, list):
            raise RerankError("rerank response had no 'results' list")

        hits: list[RerankHit] = []
        for item in results:
            if not isinstance(item, dict):
                raise RerankError("rerank result entry was not an object")
            index = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if not isinstance(index, int) or not isinstance(score, int | float):
                raise RerankError("rerank result entry was missing index/score")
            if not 0 <= index < len(documents):
                raise RerankError(f"rerank returned out-of-range index {index}")
            hits.append(RerankHit(index=index, score=float(score)))

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]
