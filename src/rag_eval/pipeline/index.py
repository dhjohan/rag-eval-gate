"""Hybrid index builder: SQLite FTS5 for lexical + NumPy for dense vectors.

Artifacts for one index live in ``data/indexes/{index_id}/``:

* ``index.sqlite`` — ``chunks`` table + external-content ``chunks_fts`` (FTS5).
* ``vectors.npy`` — L2-normalized ``float32`` matrix aligned row-for-row.
* ``manifest.json`` — :class:`IndexManifest`.

The whole directory is staged in a temporary sibling and atomically renamed, so
readers never observe a half-built index.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rag_eval.cache.sqlite_cache import SqliteCache
from rag_eval.config import (
    EmbeddingsConfig,
    Settings,
    build_env,
    load_providers_config,
    resolve_api_key,
)
from rag_eval.ingest.models import CanonicalDocument
from rag_eval.pipeline.chunkers import chunk_documents
from rag_eval.pipeline.models import Chunk, ChunkConfig, IndexManifest
from rag_eval.pipeline.tokenizer import Tokenizer, default_tokenizer
from rag_eval.providers.embeddings import EmbeddingClient

DEFAULT_DOCS_DIR = Path("data/processed/docs")
DEFAULT_INDEXES_DIR = Path("data/indexes")

Embedder = Callable[[Sequence[str]], np.ndarray]

_CHUNK_COLUMNS = (
    "chunk_id",
    "doc_id",
    "chunk_strategy",
    "chunk_index",
    "start_char",
    "end_char",
    "page_start",
    "page_end",
    "section_path",
    "raw_text",
    "index_text",
    "token_count",
)


class IndexBuildError(RuntimeError):
    """Raised when an index cannot be built from the current corpus/config."""


@dataclass(frozen=True)
class IndexBuildResult:
    index_id: str
    index_dir: Path
    manifest: IndexManifest
    chunk_count: int
    skipped: bool


def compute_index_id(
    config: ChunkConfig,
    embedding_provider: str,
    embedding_model: str,
    dims: int,
) -> str:
    """Deterministic 16-hex index id from the chunking + embedding identity."""
    payload = {
        "strategy": config.strategy,
        "max_tokens": config.max_tokens,
        "overlap_tokens": config.overlap_tokens,
        "hard_max_table_tokens": config.hard_max_table_tokens,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "dims": dims,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def corpus_digest(documents: Iterable[CanonicalDocument]) -> str:
    """Hash the corpus identity, including metadata baked into chunk text.

    ``title`` and ``effective_date`` feed the C3 contextual prefix, while page
    geometry controls chunk page spans, so all must invalidate a built index.
    """
    entries = sorted(
        [
            document.doc_id,
            hashlib.sha256(document.full_text.encode("utf-8")).hexdigest(),
            document.title,
            document.effective_date or "",
            json.dumps(
                [
                    [page.page_num, page.start_char, page.end_char]
                    for page in document.pages
                ],
                separators=(",", ":"),
            ),
        ]
        for document in documents
    )
    # ``sorted`` over same-length lists is deterministic.
    encoded = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_documents(docs_dir: str | Path) -> list[CanonicalDocument]:
    directory = Path(docs_dir)
    if not directory.is_dir():
        raise IndexBuildError(
            f"canonical docs directory not found: {directory}; run `rag-eval corpus parse`"
        )
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise IndexBuildError(f"no canonical documents in {directory}")
    documents: list[CanonicalDocument] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            documents.append(CanonicalDocument.model_validate(payload))
        except (ValueError, TypeError) as exc:
            raise IndexBuildError(f"invalid canonical document {path.name}: {exc}") from exc
    return documents


def _resolve_embedding(
    provider: str, providers_config_path: str | Path
) -> tuple[str, int]:
    providers = load_providers_config(providers_config_path)
    config = providers.embeddings.providers.get(provider)
    if config is None:
        available = ", ".join(sorted(providers.embeddings.providers))
        raise IndexBuildError(
            f"unknown embedding provider '{provider}'; configured: {available}"
        )
    if not config.enabled:
        raise IndexBuildError(f"embedding provider '{provider}' is disabled")
    return config.model, config.dims


def _derive_embeddings_config(
    provider: str,
    providers_config_path: str | Path,
    *,
    env: Mapping[str, str] | None = None,
) -> EmbeddingsConfig:
    providers = load_providers_config(providers_config_path, env=env)
    if provider not in providers.embeddings.providers:
        raise IndexBuildError(f"unknown embedding provider '{provider}'")
    config = providers.embeddings.providers[provider]
    return EmbeddingsConfig.model_validate(
        {
            "order": [provider],
            "providers": {provider: config.model_dump()},
            "fallback_enabled": False,
        }
    )


def default_embedder(
    provider: str, providers_config_path: str | Path, texts: Sequence[str]
) -> np.ndarray:
    """Embed ``texts`` through the real ``EmbeddingClient`` (network)."""
    settings = Settings()
    env = build_env(settings)
    embeddings_config = _derive_embeddings_config(
        provider, providers_config_path, env=env
    )
    with SqliteCache(settings.rag_eval_cache_path) as cache:
        client = EmbeddingClient(
            embeddings_config,
            cache,
            api_key_resolver=lambda env_name: resolve_api_key(
                env_name, settings, environ=env
            ),
        )
        # ``task_type=None`` lets the provider config supply the correct format
        # (Jina maps RETRIEVAL_DOCUMENT internally); do not hardcode one.
        result = client.embed_texts(list(texts), task_type=None)
    return result.vectors


def _normalise_vectors(vectors: np.ndarray, dims: int, expected: int) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1 and array.size == 0:
        array = array.reshape(0, dims)
    if array.ndim != 2 or array.shape[1] != dims:
        raise IndexBuildError(
            f"embedding matrix has shape {array.shape}, expected (N, {dims})"
        )
    if array.shape[0] != expected:
        raise IndexBuildError(
            f"embedding matrix has {array.shape[0]} rows, expected {expected}"
        )
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return (array / np.maximum(norms, 1e-12)).astype(np.float32)


def _write_sqlite(path: Path, chunks: Sequence[Chunk]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE chunks ("
            "chunk_id TEXT PRIMARY KEY,"
            "doc_id TEXT NOT NULL,"
            "chunk_strategy TEXT NOT NULL,"
            "chunk_index INTEGER NOT NULL,"
            "start_char INTEGER NOT NULL,"
            "end_char INTEGER NOT NULL,"
            "page_start INTEGER NOT NULL,"
            "page_end INTEGER NOT NULL,"
            "section_path TEXT NOT NULL,"
            "raw_text TEXT NOT NULL,"
            "index_text TEXT NOT NULL,"
            "token_count INTEGER NOT NULL"
            ")"
        )
        connection.execute(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5("
            "index_text, section_path, doc_id,"
            "tokenize='unicode61 remove_diacritics 2',"
            "content='chunks', content_rowid='rowid'"
            ")"
        )
        placeholders = ",".join("?" for _ in _CHUNK_COLUMNS)
        connection.executemany(
            f"INSERT INTO chunks ({','.join(_CHUNK_COLUMNS)}) VALUES ({placeholders})",
            [
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.chunk_strategy,
                    chunk.chunk_index,
                    chunk.start_char,
                    chunk.end_char,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.section_path,
                    chunk.raw_text,
                    chunk.index_text,
                    chunk.token_count,
                )
                for chunk in chunks
            ],
        )
        # External-content FTS5 tables must be rebuilt after bulk inserts.
        connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        connection.commit()
    finally:
        connection.close()


def _save_vectors(path: Path, vectors: np.ndarray) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    # Open explicitly: ``np.save`` would append ``.npy`` to a string path.
    with open(tmp, "wb") as handle:
        np.save(handle, vectors)
    os.replace(tmp, path)


def _is_fresh(index_dir: Path, index_id: str, corpus_sha256: str) -> bool:
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = IndexManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (ValueError, OSError):
        return False
    if manifest.index_id != index_id or manifest.corpus_sha256 != corpus_sha256:
        return False
    return all(
        (index_dir / name).is_file()
        for name in ("index.sqlite", "vectors.npy", "manifest.json")
    )


def build_index(
    config: ChunkConfig,
    *,
    docs_dir: str | Path = DEFAULT_DOCS_DIR,
    out_dir: str | Path = DEFAULT_INDEXES_DIR,
    embedding_provider: str = "jina",
    providers_config_path: str | Path = "configs/providers.yaml",
    force: bool = False,
    tokenizer: Tokenizer | None = None,
    embedder: Embedder | None = None,
) -> IndexBuildResult:
    """Chunk, embed, and publish one hybrid index atomically."""
    resolved_tokenizer = tokenizer or default_tokenizer()
    documents = load_documents(docs_dir)
    corpus_sha256 = corpus_digest(documents)
    embedding_model, dims = _resolve_embedding(embedding_provider, providers_config_path)
    index_id = compute_index_id(config, embedding_provider, embedding_model, dims)

    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / index_id

    if not force and _is_fresh(final_dir, index_id, corpus_sha256):
        manifest = IndexManifest.model_validate_json(
            (final_dir / "manifest.json").read_text(encoding="utf-8")
        )
        return IndexBuildResult(
            index_id=index_id,
            index_dir=final_dir,
            manifest=manifest,
            chunk_count=manifest.chunk_count,
            skipped=True,
        )

    chunks = chunk_documents(documents, config, resolved_tokenizer)
    if not chunks:
        raise IndexBuildError("chunking produced no chunks")

    texts = [chunk.index_text for chunk in chunks]
    vectors = (
        embedder(texts)
        if embedder is not None
        else default_embedder(embedding_provider, providers_config_path, texts)
    )
    matrix = _normalise_vectors(np.asarray(vectors), dims, len(chunks))

    manifest = IndexManifest(
        index_id=index_id,
        chunk_config=config,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        dims=dims,
        corpus_sha256=corpus_sha256,
        chunk_count=len(chunks),
        created_at=datetime.now(UTC).isoformat(),
    )

    temp_dir = Path(tempfile.mkdtemp(dir=output_root, prefix=f".{index_id}."))
    try:
        _write_sqlite(temp_dir / "index.sqlite", chunks)
        _save_vectors(temp_dir / "vectors.npy", matrix)
        (temp_dir / "manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if final_dir.exists():
            shutil.rmtree(final_dir)
        temp_dir.replace(final_dir)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    return IndexBuildResult(
        index_id=index_id,
        index_dir=final_dir,
        manifest=manifest,
        chunk_count=len(chunks),
        skipped=False,
    )


def read_manifest(index_dir: str | Path) -> IndexManifest:
    path = Path(index_dir) / "manifest.json"
    if not path.is_file():
        raise IndexBuildError(f"index manifest not found: {path}")
    try:
        return IndexManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise IndexBuildError(f"invalid index manifest {path}: {exc}") from exc


def list_indexes(out_dir: str | Path = DEFAULT_INDEXES_DIR) -> list[IndexManifest]:
    root = Path(out_dir)
    if not root.is_dir():
        return []
    manifests: list[IndexManifest] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        try:
            manifests.append(read_manifest(child))
        except IndexBuildError:
            continue
    return manifests


__all__ = [
    "DEFAULT_DOCS_DIR",
    "DEFAULT_INDEXES_DIR",
    "Embedder",
    "IndexBuildError",
    "IndexBuildResult",
    "build_index",
    "compute_index_id",
    "corpus_digest",
    "default_embedder",
    "list_indexes",
    "load_documents",
    "read_manifest",
]
