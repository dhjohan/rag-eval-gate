"""Offline tests for the query-side hybrid retriever and ``rag-eval query``.

Indexes are built with the deterministic ``OfflineTokenizer`` and a topic
embedder, so bm25/dense/hybrid/rerank are all exercised without the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from rag_eval.cli import app
from rag_eval.config import Settings
from rag_eval.ingest.models import CanonicalDocument, CanonicalPage
from rag_eval.pipeline.index import build_index
from rag_eval.pipeline.models import ChunkConfig
from rag_eval.pipeline.retriever import HybridRetriever, RetrievalError
from rag_eval.pipeline.tokenizer import OfflineTokenizer
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


def _build(tmp_path: Path) -> tuple[Path, Path]:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "synthetic-doc.json").write_text(
        json.dumps(_document(DOC_TEXT).model_dump(mode="json")), encoding="utf-8"
    )
    config = _providers_config(tmp_path)
    result = build_index(
        ChunkConfig(strategy="structure_aware", max_tokens=20, overlap_tokens=4),
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
