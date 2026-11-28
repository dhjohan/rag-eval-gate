"""Offline tests for the query-side hybrid retriever and ``rag-eval query``.

Indexes are built with the deterministic ``OfflineTokenizer`` and a topic
embedder, so bm25/dense/hybrid/rerank are all exercised without the network.
"""

from __future__ import annotations

import gc
import json
import sqlite3
import warnings
from contextlib import closing
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from rag_eval.cli import app
from rag_eval.config import Settings
from rag_eval.ingest.models import CanonicalDocument, CanonicalPage
from rag_eval.pipeline.index import build_index
from rag_eval.pipeline.models import ChunkConfig, Strategy
from rag_eval.pipeline.retriever import (
    CANDIDATE_N,
    HybridRetriever,
    RetrievalError,
    _fts_query,
)
from rag_eval.pipeline.tokenizer import OfflineTokenizer
from rag_eval.providers.embeddings import EmbeddingResult
from rag_eval.providers.rerank import Reranker, RerankHit

SHA = "0" * 64
DIMS = 64
DOC_TEXT = (
    "# Network Security\n"
    "The device must include a cybersecurity plan before submission.\n"
    "# Radio Security\n"
    "Wireless encryption protects radio communications."
)


class _TopicEmbedder:
    """cyber -> axis 0, wireless -> axis 1, anything else -> axis 2."""

    def __call__(self, texts: Any) -> np.ndarray:
        items = list(texts)
        matrix = np.zeros((len(items), DIMS), dtype=np.float32)
        for index, text in enumerate(items):
            lowered = text.lower()
            if "cyber" in lowered:
                matrix[index, 0] = 1.0
            elif "wireless" in lowered:
                matrix[index, 1] = 1.0
            else:
                matrix[index, 2] = 1.0
        return matrix


class _ExplodingEmbedder:
    def __call__(self, texts: Any) -> np.ndarray:
        raise AssertionError("bm25 must not embed the query")


class _WrongDimsEmbedder:
    def __call__(self, texts: Any) -> np.ndarray:
        return np.ones((len(list(texts)), DIMS - 1), dtype=np.float32)


class _ReverseReranker:
    """Scores later candidates higher so rerank order is observable."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...], int | None]] = []

    def rerank(
        self, query: str, documents: Any, *, top_n: int | None = None
    ) -> list[RerankHit]:
        items = list(documents)
        self.calls.append((query, tuple(items), top_n))
        limit = len(items) if top_n is None else min(top_n, len(items))
        hits = [RerankHit(index=index, score=float(index)) for index in range(len(items))]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]


def _document(text: str) -> CanonicalDocument:
    page = CanonicalPage(
        page_num=1,
        start_char=0,
        end_char=len(text),
        text=text,
        table_count=0,
    )
    return CanonicalDocument(
        doc_id="synthetic-doc",
        title="Synthetic Document",
        effective_date=None,
        sha256=SHA,
        page_count=1,
        table_count=0,
        token_count=len(text.split()),
        full_text=text,
        pages=[page],
    )


def _providers_config(tmp_path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "llm": {
            "order": ["a"],
            "providers": {
                "a": {
                    "enabled": True,
                    "base_url": "https://a.test/v1",
                    "model": "test-model",
                    "api_key_env": "TEST_API_KEY",
                }
            },
        },
        "embeddings": {
            "fallback_enabled": False,
            "order": ["jina"],
            "providers": {
                "jina": {
                    "enabled": True,
                    "base_url": "https://j.test/v1",
                    "model": "jina-test",
                    "api_key_env": "JINA_API_KEY",
                    "dims": DIMS,
                    "task_type": "RETRIEVAL_DOCUMENT",
                }
            },
        },
        "rerank": {
            "enabled": True,
            "base_url": "https://r.test/v1",
            "model": "jina-reranker-v3.5",
            "api_key_env": "JINA_API_KEY",
        },
    }
    path = tmp_path / "providers.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _build(tmp_path: Path, *, strategy: Strategy = "structure_aware") -> tuple[Path, Path]:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "synthetic-doc.json").write_text(
        json.dumps(_document(DOC_TEXT).model_dump(mode="json")), encoding="utf-8"
    )
    config = _providers_config(tmp_path)
    result = build_index(
        ChunkConfig(strategy=strategy, max_tokens=20, overlap_tokens=4),
        docs_dir=docs_dir,
        out_dir=tmp_path / "indexes",
        embedding_provider="jina",
        providers_config_path=config,
        tokenizer=OfflineTokenizer(),
        embedder=_TopicEmbedder(),
    )
    return result.index_dir, config


def test_bm25_is_offline_and_matches_section(tmp_path: Path) -> None:
    index_dir, _ = _build(tmp_path)

    with HybridRetriever(
        index_dir,
        mode="bm25",
        providers_config_path=tmp_path / "missing.yaml",
        embedder=_ExplodingEmbedder(),
    ) as retriever:
        hits = retriever.search("cybersecurity", top_k=1)

    assert hits
    assert "cybersecurity" in hits[0].raw_text.lower()
    assert hits[0].doc_id == "synthetic-doc"


def test_dense_uses_topic_aligned_query_embedding(tmp_path: Path) -> None:
    index_dir, config = _build(tmp_path)

    with HybridRetriever(
        index_dir, mode="dense", embedder=_TopicEmbedder(), providers_config_path=config
    ) as retriever:
        hits = retriever.search("wireless", top_k=1)

    assert hits
    assert "wireless" in hits[0].raw_text.lower()


def test_hybrid_rrf_is_deterministic(tmp_path: Path) -> None:
    index_dir, config = _build(tmp_path)
    with HybridRetriever(
        index_dir,
        mode="hybrid_rrf",
        embedder=_TopicEmbedder(),
        providers_config_path=config,
    ) as retriever:
        first = retriever.search("cybersecurity wireless", top_k=5)
        second = retriever.search("cybersecurity wireless", top_k=5)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.score for chunk in first] == sorted(
        (chunk.score for chunk in first), reverse=True
    )


def test_rerank_reorders_hybrid_candidates(tmp_path: Path) -> None:
    index_dir, config = _build(tmp_path)
    reranker = _ReverseReranker()

    with HybridRetriever(
        index_dir,
        mode="rerank",
        top_k=1,
        embedder=_TopicEmbedder(),
        reranker=reranker,
        providers_config_path=config,
    ) as retriever:
        hits = retriever.search("cybersecurity")

    assert reranker.calls
    assert reranker.calls[0][2] == 1
    assert hits
    assert "wireless" in hits[0].raw_text.lower()


def test_reranker_uses_settings_only_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_dir, config = _build(tmp_path)
    monkeypatch.delenv("JINA_API_KEY", raising=False)

    with HybridRetriever(
        index_dir,
        mode="rerank",
        providers_config_path=config,
        settings=Settings(jina_api_key="settings-only"),
    ) as retriever:
        reranker = retriever._make_reranker()

        assert isinstance(reranker, Reranker)
        assert reranker.is_configured


def test_query_embedding_dims_must_match_manifest(tmp_path: Path) -> None:
    index_dir, config = _build(tmp_path)

    with (
        HybridRetriever(
            index_dir,
            mode="dense",
            embedder=_WrongDimsEmbedder(),
            providers_config_path=config,
        ) as retriever,
        pytest.raises(RetrievalError, match="expected"),
    ):
        retriever.search("cybersecurity")


def test_missing_index_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(RetrievalError, match="manifest not found"):
        HybridRetriever(tmp_path / "missing", mode="bm25")


def test_cli_query_bm25(tmp_path: Path) -> None:
    index_dir, config = _build(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "query",
            "cybersecurity",
            "--mode",
            "bm25",
            "--strategy",
            "structure_aware",
            "--out-dir",
            str(index_dir.parent),
            "--config",
            str(config),
        ],
        env={"COLUMNS": "250"},
    )

    assert result.exit_code == 0, result.output
    assert "synthetic-doc" in result.output
    assert "OK" in result.output


def test_cli_rejects_unknown_mode(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["query", "anything", "--mode", "nope"])

    assert result.exit_code == 2
    assert "mode must be one of" in result.output


# -- review fixes -------------------------------------------------------------


def test_bm25_and_dense_searches_close_sqlite_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: ``with sqlite3.connect(...)`` only manages the transaction.

    Every connection opened while serving bm25/dense queries must really be
    closed, otherwise sqlite3 emits ResourceWarning("unclosed database") when
    the garbage collector reclaims the connection.
    """
    index_dir, config = _build(tmp_path)
    created: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def spy_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = real_connect(*args, **kwargs)
        created.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", spy_connect)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        with HybridRetriever(
            index_dir,
            mode="bm25",
            providers_config_path=tmp_path / "missing.yaml",
            embedder=_ExplodingEmbedder(),
        ) as retriever:
            bm25_hits = retriever.search("cybersecurity", top_k=5)
        with HybridRetriever(
            index_dir, mode="dense", embedder=_TopicEmbedder(), providers_config_path=config
        ) as dense:
            dense.search("cybersecurity", top_k=3)
        gc.collect()

    assert created, "expected the searches to open the index database"
    for connection in created:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
    assert bm25_hits


class _FakeCache:
    instances: ClassVar[list[_FakeCache]] = []

    def __init__(self, path: Any) -> None:
        self.path = path
        self.closed = False
        _FakeCache.instances.append(self)

    def close(self) -> None:
        self.closed = True


class _FakeEmbeddingClient:
    instances: ClassVar[list[_FakeEmbeddingClient]] = []

    def __init__(self, config: Any, cache: Any, *, api_key_resolver: Any = None) -> None:
        self.cache = cache
        self.task_types: list[str | None] = []
        _FakeEmbeddingClient.instances.append(self)

    def embed_texts(self, texts: Any, *, task_type: str | None = None) -> EmbeddingResult:
        items = list(texts)
        self.task_types.append(task_type)
        matrix = np.zeros((len(items), DIMS), dtype=np.float32)
        matrix[:, 0] = 1.0
        return EmbeddingResult(
            vectors=matrix,
            provider_used="jina",
            model_used="jina-test",
            cache_hits=0,
            cache_misses=len(items),
        )


def test_embedder_cache_and_client_persist_across_searches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One SqliteCache + one EmbeddingClient for the retriever lifetime.

    Recreating them per search() would reset the client's RPM/TPM token
    buckets on every query; close() must release the cache.
    """
    index_dir, config = _build(tmp_path)
    _FakeCache.instances.clear()
    _FakeEmbeddingClient.instances.clear()
    monkeypatch.setattr("rag_eval.pipeline.retriever.SqliteCache", _FakeCache)
    monkeypatch.setattr("rag_eval.pipeline.retriever.EmbeddingClient", _FakeEmbeddingClient)

    with HybridRetriever(index_dir, mode="dense", providers_config_path=config) as retriever:
        first = retriever.search("cybersecurity", top_k=2)
        second = retriever.search("wireless", top_k=2)
        assert callable(retriever._make_embedder())

        assert first and second
        assert len(_FakeCache.instances) == 1
        assert len(_FakeEmbeddingClient.instances) == 1

    client = _FakeEmbeddingClient.instances[0]
    assert client.task_types == ["RETRIEVAL_QUERY", "RETRIEVAL_QUERY"]
    assert _FakeCache.instances[0].closed


@pytest.mark.parametrize(
    ("field", "value"),
    [("model", "jina-other"), ("dims", DIMS + 1)],
)
def test_make_embedder_enforces_manifest_pinning(
    tmp_path: Path, field: str, value: Any
) -> None:
    index_dir, config = _build(tmp_path)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["embeddings"]["providers"]["jina"][field] = value
    mismatched = tmp_path / "providers-mismatch.yaml"
    mismatched.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with (
        HybridRetriever(index_dir, mode="dense", providers_config_path=mismatched) as retriever,
        pytest.raises(RetrievalError, match="index was built with"),
    ):
        retriever._make_embedder()


def test_rerank_sends_index_text_but_results_keep_raw_text(tmp_path: Path) -> None:
    index_dir, config = _build(tmp_path, strategy="contextual_heading")
    reranker = _ReverseReranker()

    with HybridRetriever(
        index_dir,
        mode="rerank",
        top_k=2,
        candidates=CANDIDATE_N,
        embedder=_TopicEmbedder(),
        reranker=reranker,
        providers_config_path=config,
    ) as retriever:
        hits = retriever.search("cybersecurity")

    with HybridRetriever(
        index_dir,
        mode="hybrid_rrf",
        candidates=CANDIDATE_N,
        embedder=_TopicEmbedder(),
        providers_config_path=config,
    ) as hybrid:
        candidates = hybrid.search("cybersecurity", top_k=CANDIDATE_N)

    sent = reranker.calls[0][1]
    assert sent == tuple(chunk.index_text for chunk in candidates)
    # contextual_heading prefixes every index_text; raw_text never has it.
    assert all(document.startswith("[Document:") for document in sent)

    assert hits
    by_id = {chunk.chunk_id: chunk for chunk in candidates}
    for hit in hits:
        original = by_id[hit.chunk_id]
        assert hit.raw_text == original.raw_text
        assert not hit.raw_text.startswith("[Document:")
        assert hit.index_text == original.index_text


def test_fts_query_is_unicode_aware_against_real_fts5(tmp_path: Path) -> None:
    db = tmp_path / "fts.sqlite"
    with closing(sqlite3.connect(db)) as connection:
        connection.execute(
            "CREATE VIRTUAL TABLE docs USING fts5("
            "body, tokenize='unicode61 remove_diacritics 2')"
        )
        connection.execute(
            "INSERT INTO docs(body) VALUES (?)",
            ("Le café cybersécurité et la 日本語 are indexed; plain ascii too.",),
        )
        connection.commit()

        for query in ("café", "cybersécurité 日本語", "日本語 café ascii"):
            match = _fts_query(query)
            rows = connection.execute(
                "SELECT body FROM docs WHERE docs MATCH ?", (match,)
            ).fetchall()
            assert rows, f"query {query!r} (match {match!r}) matched nothing"

        # Punctuation-only input yields no tokens; callers bail out on "".
        assert _fts_query("!!! …") == ""
