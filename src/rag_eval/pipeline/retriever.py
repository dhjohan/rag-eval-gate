"""Query-side hybrid retriever over one built index.

Modes are deliberately explicit: ``bm25`` never touches the network, while
``dense``/``hybrid_rrf``/``rerank`` reuse the embedding identity recorded in the
index manifest so a query can never be compared against a different vector
space.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from rag_eval.cache.sqlite_cache import SqliteCache
from rag_eval.config import (
    EmbeddingsConfig,
    Settings,
    build_env,
    load_providers_config,
    resolve_api_key,
)
from rag_eval.pipeline.index import IndexBuildError, read_manifest
from rag_eval.pipeline.models import ChunkConfig, IndexManifest
from rag_eval.providers.embeddings import EmbeddingClient
from rag_eval.providers.rerank import Reranker, RerankHit

RetrievalMode = Literal["bm25", "dense", "hybrid_rrf", "rerank"]
RETRIEVAL_MODES: tuple[str, ...] = ("bm25", "dense", "hybrid_rrf", "rerank")

RRF_K = 60
# Each arm must contribute at least 50 candidates, otherwise RRF fuses a
# narrow lexical/dense window and the reranker has little to reorder.
CANDIDATE_N = 50

Embedder = Callable[[Sequence[str]], np.ndarray]


class RetrievalError(RuntimeError):
    """Raised when an index/query cannot be served as requested."""


class RerankClient(Protocol):
    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> list[RerankHit]: ...


@dataclass(frozen=True)
class RetrievedChunk:
    """One ranked chunk with the score produced by the active mode."""

    chunk_id: str
    doc_id: str
    chunk_strategy: str
    chunk_index: int
    start_char: int
    end_char: int
    page_start: int
    page_end: int
    section_path: str
    raw_text: str
    score: float


class HybridRetriever:
    """Search one built index using bm25, dense, hybrid RRF, or reranking."""

    def __init__(
        self,
        index_dir: str | Path,
        *,
        mode: RetrievalMode = "hybrid_rrf",
        top_k: int = 5,
        candidates: int = CANDIDATE_N,
        rrf_k: int = RRF_K,
        providers_config_path: str | Path = "configs/providers.yaml",
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        reranker: RerankClient | None = None,
    ) -> None:
        if mode not in RETRIEVAL_MODES:
            raise RetrievalError(f"unknown retrieval mode {mode!r}; choose from {RETRIEVAL_MODES}")
        if top_k < 1:
            raise RetrievalError("top_k must be >= 1")
        if candidates < 1:
            raise RetrievalError("candidates must be >= 1")
        if rrf_k < 1:
            raise RetrievalError("rrf_k must be >= 1")

        self.mode = mode
        self.top_k = top_k
        self._candidates = candidates
        self._rrf_k = rrf_k
        self._index_dir = Path(index_dir)
        try:
            self._manifest = read_manifest(self._index_dir)
        except IndexBuildError as exc:
            raise RetrievalError(str(exc)) from exc
        self._providers_config_path = providers_config_path
        self._settings = settings
        self._embedder = embedder
        self._reranker = reranker
        self._owns_reranker = reranker is None
        self._chunk_ids: list[str] | None = None

        for name in ("index.sqlite", "vectors.npy", "manifest.json"):
            if not (self._index_dir / name).is_file():
                raise RetrievalError(f"index is missing {name}: {self._index_dir}")

    @property
    def manifest(self) -> IndexManifest:
        return self._manifest

    @property
    def chunk_config(self) -> ChunkConfig:
        return self._manifest.chunk_config

    def __enter__(self) -> HybridRetriever:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_reranker and self._reranker is not None:
            close = getattr(self._reranker, "close", None)
            if callable(close):
                close()

    def search(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the best chunks for ``query`` in descending mode-specific score."""
        if not query.strip():
            raise RetrievalError("query must not be empty")
        limit = self.top_k if top_k is None else top_k
        if limit < 1:
            raise RetrievalError("top_k must be >= 1")

        if self.mode == "bm25":
            return self._hydrate(self._bm25(query, limit))
        if self.mode == "dense":
            return self._hydrate(self._dense(query, limit))
        if self.mode == "hybrid_rrf":
            return self._hydrate(self._hybrid(query, limit))
        return self._rerank_search(query, limit)

    # -- arms --------------------------------------------------------------

    def _bm25(self, query: str, limit: int) -> list[tuple[str, float]]:
        match = _fts_query(query)
        if not match:
            return []
        with sqlite3.connect(self._index_dir / "index.sqlite") as connection:
            rows = connection.execute(
                "SELECT c.chunk_id, bm25(chunks_fts) FROM chunks_fts "
                "JOIN chunks c ON c.rowid = chunks_fts.rowid "
                "WHERE chunks_fts MATCH ? "
                "ORDER BY bm25(chunks_fts), c.chunk_id LIMIT ?",
                (match, limit),
            ).fetchall()
        # FTS5 bm25() returns a negative score where more negative is better.
        return [(str(row[0]), -float(row[1])) for row in rows]

    def _dense(self, query: str, limit: int) -> list[tuple[str, float]]:
        vector = self._embed_query(query)
        try:
            matrix = np.load(self._index_dir / "vectors.npy", mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise RetrievalError(f"cannot read index vectors: {exc}") from exc
        if matrix.ndim != 2 or matrix.shape[1] != self._manifest.dims:
            raise RetrievalError(
                f"vectors have shape {matrix.shape}, expected (N, {self._manifest.dims})"
            )
        ids = self._index_chunk_ids()
        if matrix.shape[0] != len(ids):
            raise RetrievalError(
                f"vectors have {matrix.shape[0]} rows but the index has {len(ids)} chunks"
            )
        scores = np.asarray(matrix @ vector, dtype=np.float32)
        order = sorted(range(len(ids)), key=lambda index: (-float(scores[index]), ids[index]))
        return [(ids[index], float(scores[index])) for index in order[:limit]]

    def _hybrid(self, query: str, limit: int) -> list[tuple[str, float]]:
        window = max(limit, self._candidates)
        fused: dict[str, float] = {}
        for arm in (self._bm25(query, window), self._dense(query, window)):
            for rank, (chunk_id, _) in enumerate(arm, start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (self._rrf_k + rank)
        return sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:limit]

    def _rerank_search(self, query: str, limit: int) -> list[RetrievedChunk]:
        candidates = self._hydrate(self._hybrid(query, max(limit, self._candidates)))
        if not candidates:
            return []
        client = self._reranker if self._reranker is not None else self._make_reranker()
        hits = client.rerank(query, [chunk.raw_text for chunk in candidates], top_n=limit)
        for hit in hits:
            if not 0 <= hit.index < len(candidates):
                raise RetrievalError(f"reranker returned out-of-range index {hit.index}")
        ordered = sorted(hits, key=lambda hit: (-hit.score, candidates[hit.index].chunk_id))
        return [
            _with_score(candidates[hit.index], hit.score) for hit in ordered[:limit]
        ]

    # -- plumbing ----------------------------------------------------------

    def _index_chunk_ids(self) -> list[str]:
        if self._chunk_ids is None:
            with sqlite3.connect(self._index_dir / "index.sqlite") as connection:
                self._chunk_ids = [
                    str(row[0])
                    for row in connection.execute(
                        "SELECT chunk_id FROM chunks ORDER BY rowid"
                    ).fetchall()
                ]
        return self._chunk_ids

    def _embed_query(self, query: str) -> np.ndarray:
        if self._embedder is None:
            self._embedder = self._make_embedder()
        matrix = np.asarray(self._embedder([query]), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape != (1, self._manifest.dims):
            raise RetrievalError(
                f"query embedding has shape {matrix.shape}, "
                f"expected (1, {self._manifest.dims})"
            )
        vector = matrix[0]
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm == 0.0:
            raise RetrievalError("query embedding has zero or non-finite norm")
        return (vector / norm).astype(np.float32)

    def _make_embedder(self) -> Embedder:
        settings = self._settings or Settings()
        env = build_env(settings)
        providers = load_providers_config(self._providers_config_path, env=env)
        name = self._manifest.embedding_provider
        config = providers.embeddings.providers.get(name)
        if config is None:
            available = ", ".join(sorted(providers.embeddings.providers))
            raise RetrievalError(
                f"index embedding provider {name!r} is no longer configured; available: {available}"
            )
        if config.model != self._manifest.embedding_model or config.dims != self._manifest.dims:
            raise RetrievalError(
                f"index was built with {name}/{self._manifest.embedding_model}/"
                f"{self._manifest.dims}d but providers.yaml now says "
                f"{config.model}/{config.dims}d"
            )
        single = EmbeddingsConfig.model_validate(
            {
                "order": [name],
                "providers": {name: config.model_dump()},
                "fallback_enabled": False,
            }
        )
        cache_path = settings.rag_eval_cache_path

        def embed(texts: Sequence[str]) -> np.ndarray:
            with SqliteCache(cache_path) as cache:
                result = EmbeddingClient(
                    single,
                    cache,
                    api_key_resolver=lambda env_name: resolve_api_key(
                        env_name, settings, environ=env
                    ),
                ).embed_texts(list(texts), task_type="RETRIEVAL_QUERY")
            if (
                result.provider_used != self._manifest.embedding_provider
                or result.model_used != self._manifest.embedding_model
            ):
                raise RetrievalError(
                    f"query embedded with {result.provider_used}/{result.model_used}, "
                    f"index expects {self._manifest.embedding_provider}/"
                    f"{self._manifest.embedding_model}"
                )
            return result.vectors

        return embed

    def _make_reranker(self) -> RerankClient:
        settings = self._settings or Settings()
        env = build_env(settings)
        providers = load_providers_config(self._providers_config_path, env=env)
        self._reranker = Reranker(
            providers.rerank,
            api_key_resolver=lambda env_name: resolve_api_key(
                env_name, settings, environ=env
            ),
        )
        self._owns_reranker = True
        return self._reranker

    def _hydrate(self, ranked: Sequence[tuple[str, float]]) -> list[RetrievedChunk]:
        if not ranked:
            return []
        placeholders = ",".join("?" for _ in ranked)
        with sqlite3.connect(self._index_dir / "index.sqlite") as connection:
            rows = connection.execute(
                "SELECT chunk_id, doc_id, chunk_strategy, chunk_index, start_char, end_char, "
                "page_start, page_end, section_path, raw_text "
                f"FROM chunks WHERE chunk_id IN ({placeholders})",
                [chunk_id for chunk_id, _ in ranked],
            ).fetchall()
        scores = dict(ranked)
        by_id: dict[str, RetrievedChunk] = {}
        for row in rows:
            chunk_id = str(row[0])
            by_id[chunk_id] = RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=str(row[1]),
                chunk_strategy=str(row[2]),
                chunk_index=int(row[3]),
                start_char=int(row[4]),
                end_char=int(row[5]),
                page_start=int(row[6]),
                page_end=int(row[7]),
                section_path=str(row[8]),
                raw_text=str(row[9]),
                score=scores[chunk_id],
            )
        missing = [chunk_id for chunk_id, _ in ranked if chunk_id not in by_id]
        if missing:
            raise RetrievalError(f"index is missing ranked chunks: {missing[:3]}")
        return [by_id[chunk_id] for chunk_id, _ in ranked]


def _with_score(chunk: RetrievedChunk, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        chunk_strategy=chunk.chunk_strategy,
        chunk_index=chunk.chunk_index,
        start_char=chunk.start_char,
        end_char=chunk.end_char,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section_path=chunk.section_path,
        raw_text=chunk.raw_text,
        score=score,
    )


def _fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 OR-query over alphanumeric tokens."""
    tokens = re.findall(r"[A-Za-z0-9]+", query)
    return " OR ".join(f'"{token}"' for token in tokens)


__all__ = [
    "CANDIDATE_N",
    "RETRIEVAL_MODES",
    "RRF_K",
    "Embedder",
    "HybridRetriever",
    "RerankClient",
    "RetrievalError",
    "RetrievalMode",
    "RetrievedChunk",
]
