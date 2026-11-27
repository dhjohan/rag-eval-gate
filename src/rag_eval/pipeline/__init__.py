"""Chunking, hybrid indexing, and query-side retrieval (Sprint 3B)."""

from __future__ import annotations

from rag_eval.pipeline.chunkers import (
    chunk_contextual_heading,
    chunk_document,
    chunk_documents,
    chunk_fixed_token,
    chunk_structure_aware,
)
from rag_eval.pipeline.index import (
    IndexBuildError,
    IndexBuildResult,
    build_index,
    compute_index_id,
    corpus_digest,
    list_indexes,
    load_documents,
    read_manifest,
)
from rag_eval.pipeline.models import (
    STRATEGY_VALUES,
    Chunk,
    ChunkConfig,
    IndexManifest,
)
from rag_eval.pipeline.retriever import (
    RETRIEVAL_MODES,
    HybridRetriever,
    RetrievalError,
    RetrievedChunk,
)
from rag_eval.pipeline.tokenizer import (
    OfflineTokenizer,
    TiktokenTokenizer,
    Tokenizer,
    default_tokenizer,
)

__all__ = [
    "RETRIEVAL_MODES",
    "STRATEGY_VALUES",
    "Chunk",
    "ChunkConfig",
    "HybridRetriever",
    "IndexBuildError",
    "IndexBuildResult",
    "IndexManifest",
    "OfflineTokenizer",
    "RetrievalError",
    "RetrievedChunk",
    "TiktokenTokenizer",
    "Tokenizer",
    "build_index",
    "chunk_contextual_heading",
    "chunk_document",
    "chunk_documents",
    "chunk_fixed_token",
    "chunk_structure_aware",
    "compute_index_id",
    "corpus_digest",
    "default_tokenizer",
    "list_indexes",
    "load_documents",
    "read_manifest",
]
