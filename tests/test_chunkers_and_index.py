"""Offline tests for the offset-invariant chunkers and hybrid index builder.

Everything uses the deterministic ``OfflineTokenizer`` and an injected fake
embedder, so no test imports tiktoken or touches the network.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from rag_eval.cli import app
from rag_eval.ingest.models import CanonicalDocument, CanonicalPage
from rag_eval.pipeline.chunkers import (
    _context_prefix,
    chunk_contextual_heading,
    chunk_fixed_token,
    chunk_structure_aware,
)
from rag_eval.pipeline.index import (
    IndexBuildError,
    build_index,
    compute_index_id,
    corpus_digest,
)
from rag_eval.pipeline.models import ChunkConfig
from rag_eval.pipeline.tokenizer import OfflineTokenizer, TiktokenTokenizer

SHA = "0" * 64


# -- fixtures / helpers ------------------------------------------------------


def _build_document(
    page_texts: list[str],
    *,
    doc_id: str = "synthetic-doc",
    title: str = "Synthetic Document",
    effective_date: str | None = None,
) -> CanonicalDocument:
    pages: list[CanonicalPage] = []
    cursor = 0
    for index, text in enumerate(page_texts):
        if index:
            cursor += 2
        start = cursor
        end = start + len(text)
        pages.append(
            CanonicalPage(
                page_num=index + 1,
                start_char=start,
                end_char=end,
                text=text,
                table_count=0,
            )
        )
        cursor = end
    full_text = "\n\n".join(page_texts)
    return CanonicalDocument(
        doc_id=doc_id,
        title=title,
        effective_date=effective_date,
        sha256=SHA,
        page_count=len(pages),
        table_count=0,
        token_count=len(full_text.split()),
        full_text=full_text,
        pages=pages,
    )


def _write_document(docs_dir: Path, document: CanonicalDocument) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    target = docs_dir / f"{document.doc_id}.json"
    target.write_text(json.dumps(document.model_dump(mode="json")), encoding="utf-8")
    return target


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
                    "dims": 64,
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


class _FakeEmbedder:
    def __init__(self, dims: int = 64) -> None:
        self.dims = dims
        self.calls: list[list[str]] = []

    def __call__(self, texts: Any) -> np.ndarray:
        texts = list(texts)
        self.calls.append(texts)
        matrix = np.zeros((len(texts), self.dims), dtype=np.float32)
        for index in range(len(texts)):
            matrix[index, index % self.dims] = 1.0
        return matrix


class _ZeroEmbedder:
    def __call__(self, texts: Any) -> np.ndarray:
        return np.zeros((len(list(texts)), 64), dtype=np.float32)


class _FakeEncoding:
    """Splits a two-byte codepoint across two tokens."""

    def encode_ordinary(self, text: str) -> list[int]:
        return [0, 1, 2]

    def decode_tokens_bytes(self, tokens: list[int]) -> list[bytes]:
        return [b"\xc3", b"\xa9", b" x"]


def _assert_invariant(document: CanonicalDocument, chunk: Any) -> None:
    assert document.full_text[chunk.start_char : chunk.end_char] == chunk.raw_text
    assert chunk.chunk_id == f"{document.doc_id}:{chunk.chunk_index:04d}"
    assert chunk.page_start <= chunk.page_end


# -- C1 fixed token ----------------------------------------------------------


def test_c1_offsets_are_exact() -> None:
    document = _build_document(["Alpha beta gamma delta epsilon zeta eta theta."])
    config = ChunkConfig(strategy="fixed_token", max_tokens=4, overlap_tokens=1)

    chunks = chunk_fixed_token(document, config, OfflineTokenizer())

    assert chunks
    for chunk in chunks:
        _assert_invariant(document, chunk)
        assert chunk.page_start == 1 and chunk.page_end == 1
        assert chunk.raw_text.strip() in document.full_text


def test_c1_short_document_does_not_overrun() -> None:
    document = _build_document(["one two three"])
    config = ChunkConfig(strategy="fixed_token", max_tokens=50, overlap_tokens=8)

    chunks = chunk_fixed_token(document, config, OfflineTokenizer())

    assert len(chunks) == 1
    assert chunks[0].raw_text == "one two three"


def test_c1_stride_and_overlap() -> None:
    document = _build_document([" ".join(f"w{index}" for index in range(10))])
    config = ChunkConfig(strategy="fixed_token", max_tokens=4, overlap_tokens=1)

    chunks = chunk_fixed_token(document, config, OfflineTokenizer())

    assert len(chunks) == 3
    # Overlap of one token means the next chunk starts before the previous ends.
    assert chunks[0].end_char > chunks[1].start_char


def test_c1_rejects_overlap_not_smaller_than_window() -> None:
    with pytest.raises(ValidationError, match="overlap_tokens"):
        ChunkConfig(strategy="fixed_token", max_tokens=4, overlap_tokens=4)


def test_c1_multibyte_words_are_never_split() -> None:
    document = _build_document(["café 😀 naïve façade résumé"])
    config = ChunkConfig(strategy="fixed_token", max_tokens=1, overlap_tokens=0)

    chunks = chunk_fixed_token(document, config, OfflineTokenizer())

    assert chunks
    for chunk in chunks:
        _assert_invariant(document, chunk)
        assert chunk.raw_text
        assert "\ufffd" not in chunk.raw_text


def test_tiktoken_byte_mapping_snaps_to_codepoints() -> None:
    tokenizer = TiktokenTokenizer(encoding=_FakeEncoding())

    spans = tokenizer.token_char_spans("é x")

    # Both tokens that cover the split 'é' bytes map to the whole character.
    assert spans == [(0, 1), (0, 1), (1, 3)]
    for start, end in spans:
        assert "é x"[start:end]


def test_c1_multibyte_split_tokens_stay_whole() -> None:
    tokenizer = TiktokenTokenizer(encoding=_FakeEncoding())
    document = _build_document(["é x"])
    config = ChunkConfig(strategy="fixed_token", max_tokens=1, overlap_tokens=0)

    chunks = chunk_fixed_token(document, config, tokenizer)

    assert chunks
    for chunk in chunks:
        _assert_invariant(document, chunk)
        assert "\ufffd" not in chunk.raw_text


# -- C2 structure aware ------------------------------------------------------


def test_c2_offsets_and_section_paths() -> None:
    text = (
        "# Introduction\n"
        "The scope covers devices.\n"
        "## Subsection\n"
        "More detail here.\n"
        "IV.A.2 Numbered heading\n"
        "Numbered body text."
    )
    document = _build_document([text])
    config = ChunkConfig(strategy="structure_aware", max_tokens=40, overlap_tokens=8)

    chunks = chunk_structure_aware(document, config, OfflineTokenizer())

    assert chunks
    for chunk in chunks:
        _assert_invariant(document, chunk)
        assert chunk.section_path

    paths = [chunk.section_path for chunk in chunks]
    assert any(path == "Introduction" for path in paths)
    assert any(path == "Introduction > Subsection" for path in paths)
    assert any(path.endswith("IV.A.2 Numbered heading") for path in paths)


def test_c2_preamble_before_first_heading() -> None:
    text = "Preamble paragraph before any heading.\n# Section\nBody text."
    document = _build_document([text])
    config = ChunkConfig(strategy="structure_aware", max_tokens=40, overlap_tokens=8)

    chunks = chunk_structure_aware(document, config, OfflineTokenizer())

    assert any(chunk.section_path == "Preamble" for chunk in chunks)


def test_c2_oversized_table_splits_on_rows_with_header_only_in_index_text() -> None:
    text = (
        "# Data\n"
        "| Header | Value |\n"
        "| --- | --- |\n"
        "| row1 | 1 |\n"
        "| row2 | 2 |\n"
        "| row3 | 3 |"
    )
    document = _build_document([text])
    config = ChunkConfig(
        strategy="structure_aware",
        max_tokens=50,
        overlap_tokens=5,
        hard_max_table_tokens=12,
    )

    chunks = chunk_structure_aware(document, config, OfflineTokenizer())

    for chunk in chunks:
        _assert_invariant(document, chunk)

    table_chunks = [chunk for chunk in chunks if chunk.raw_text.startswith("|")]
    assert len(table_chunks) == 3
    assert "Header" in table_chunks[0].raw_text
    assert "Header" not in table_chunks[1].raw_text
    assert table_chunks[1].index_text.startswith("| Header | Value |")
    assert table_chunks[1].raw_text == "| row2 | 2 |"
    assert table_chunks[2].raw_text == "| row3 | 3 |"
    assert all(chunk.section_path == "Data" for chunk in table_chunks)


def test_c2_page_span_spans_a_page_break() -> None:
    document = _build_document(
        ["First page text here.", "Second page text continues."]
    )
    config = ChunkConfig(strategy="fixed_token", max_tokens=100, overlap_tokens=8)

    chunks = chunk_fixed_token(document, config, OfflineTokenizer())

    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2


# -- C3 contextual heading ---------------------------------------------------


def test_c3_prefix_with_effective_date() -> None:
    document = _build_document(
        ["# Scope\nThe device must include a plan."],
        title="My Guidance",
        effective_date="2024-01-15",
    )
    config = ChunkConfig(strategy="contextual_heading", max_tokens=40, overlap_tokens=8)

    chunks = chunk_contextual_heading(document, config, OfflineTokenizer())

    assert chunks
    for chunk in chunks:
        assert chunk.index_text.startswith("[Document: My Guidance | Section: ")
        assert " | Effective: 2024-01-15]" in chunk.index_text.split("\n", 1)[0]
        _assert_invariant(document, chunk)


def test_c3_prefix_without_effective_date() -> None:
    document = _build_document(["# Scope\nBody text."], title="No Date Doc")
    config = ChunkConfig(strategy="contextual_heading", max_tokens=40, overlap_tokens=8)

    chunks = chunk_contextual_heading(document, config, OfflineTokenizer())

    assert chunks
    for chunk in chunks:
        header = chunk.index_text.split("\n", 1)[0]
        assert header.startswith("[Document: No Date Doc | Section: ")
        assert "Effective" not in header


def test_c3_title_falls_back_to_doc_id() -> None:
    document = _build_document(["# Scope\nBody."])
    without_title = document.model_copy(update={"title": ""})

    prefix = _context_prefix(without_title, "Scope")

    assert prefix == "[Document: synthetic-doc | Section: Scope]"


def test_c3_spans_and_raw_text_match_c2() -> None:
    document = _build_document(
        ["# Scope\nParagraph one is here.\n| a | b |\n| --- | --- |\n| 1 | 2 |"]
    )
    c2_config = ChunkConfig(strategy="structure_aware", max_tokens=40, overlap_tokens=8)
    c3_config = ChunkConfig(
        strategy="contextual_heading", max_tokens=40, overlap_tokens=8
    )

    c2 = chunk_structure_aware(document, c2_config, OfflineTokenizer())
    c3 = chunk_contextual_heading(document, c3_config, OfflineTokenizer())

    assert [(c.start_char, c.end_char, c.raw_text) for c in c2] == [
        (c.start_char, c.end_char, c.raw_text) for c in c3
    ]
    assert [c.page_start for c in c2] == [c.page_start for c in c3]


# -- index id / corpus digest ------------------------------------------------


def test_index_id_is_deterministic_and_sensitive() -> None:
    config = ChunkConfig(strategy="fixed_token", max_tokens=512, overlap_tokens=64)

    first = compute_index_id(config, "jina", "jina-test", 64)
    second = compute_index_id(config, "jina", "jina-test", 64)
    larger = compute_index_id(
        ChunkConfig(strategy="fixed_token", max_tokens=256, overlap_tokens=64),
        "jina",
        "jina-test",
        64,
    )

    assert first == second
    assert len(first) == 16
    assert first != larger


def test_corpus_digest_tracks_text_changes() -> None:
    first = _build_document(["alpha beta"])
    same = _build_document(["alpha beta"])
    changed = _build_document(["alpha gamma"])

    assert corpus_digest([first]) == corpus_digest([same])
    assert corpus_digest([first]) != corpus_digest([changed])


def test_corpus_digest_tracks_title_and_date_changes() -> None:
    base = _build_document(["alpha beta"])
    retitled = _build_document(["alpha beta"], title="Other Title")
    redated = _build_document(["alpha beta"], effective_date="2026-01-01")

    assert corpus_digest([base]) != corpus_digest([retitled])
    assert corpus_digest([base]) != corpus_digest([redated])


# -- index build -------------------------------------------------------------


def _build(
    tmp_path: Path,
    *,
    force: bool = False,
    embedder: Any | None = None,
    text: str | None = None,
) -> Any:
    document = _build_document(
        [
            text
            or (
                "# Intro\nThe device must include a cybersecurity plan.\n"
                "Another sentence here."
            )
        ]
    )
    docs_dir = tmp_path / "docs"
    _write_document(docs_dir, document)
    config = ChunkConfig(strategy="structure_aware", max_tokens=20, overlap_tokens=4)
    return build_index(
        config,
        docs_dir=docs_dir,
        out_dir=tmp_path / "indexes",
        embedding_provider="jina",
        providers_config_path=_providers_config(tmp_path),
        tokenizer=OfflineTokenizer(),
        embedder=embedder or _FakeEmbedder(),
        force=force,
    )


def test_build_index_writes_all_artifacts(tmp_path: Path) -> None:
    result = _build(tmp_path)

    assert not result.skipped
    assert result.index_id == compute_index_id(
        ChunkConfig(strategy="structure_aware", max_tokens=20, overlap_tokens=4),
        "jina",
        "jina-test",
        64,
    )
    assert (result.index_dir / "index.sqlite").is_file()
    assert (result.index_dir / "vectors.npy").is_file()
    assert (result.index_dir / "manifest.json").is_file()
    assert not any(
        path.name.startswith(".") for path in (tmp_path / "indexes").iterdir()
    )
    assert not list((tmp_path / "indexes").rglob("*.tmp"))

    vectors = np.load(result.index_dir / "vectors.npy")
    assert vectors.shape == (result.chunk_count, 64)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    manifest = json.loads((result.index_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["index_id"] == result.index_id
    assert manifest["chunk_count"] == result.chunk_count
    assert manifest["embedding_model"] == "jina-test"
    assert manifest["dims"] == 64
    assert manifest["chunk_config"]["strategy"] == "structure_aware"


def test_sqlite_chunks_and_fts_are_populated(tmp_path: Path) -> None:
    result = _build(tmp_path)

    with sqlite3.connect(result.index_dir / "index.sqlite") as connection:
        count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count == result.chunk_count

        hits = connection.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?", ("cybersecurity",)
        ).fetchone()[0]
        assert hits > 0

        section_hits = connection.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?", ("intro",)
        ).fetchone()[0]
        assert section_hits > 0

        row = connection.execute(
            "SELECT raw_text, index_text, section_path FROM chunks LIMIT 1"
        ).fetchone()
        assert row is not None
        assert row[0] in row[1] or row[1].endswith(row[0])


def test_build_index_skips_when_fresh_and_force_rebuilds(tmp_path: Path) -> None:
    embedder = _FakeEmbedder()
    first = _build(tmp_path, embedder=embedder)
    second = _build(tmp_path, embedder=embedder)

    assert not first.skipped
    assert second.skipped
    assert len(embedder.calls) == 1

    third = _build(tmp_path, force=True, embedder=embedder)
    assert not third.skipped
    assert len(embedder.calls) == 2


def test_build_index_rebuilds_when_corpus_changes(tmp_path: Path) -> None:
    embedder = _FakeEmbedder()
    _build(tmp_path, embedder=embedder)

    rebuilt = _build(
        tmp_path, embedder=embedder, text="# Intro\nDifferent device text now."
    )
    assert not rebuilt.skipped
    assert len(embedder.calls) == 2


def test_build_index_zero_vectors_stay_finite(tmp_path: Path) -> None:
    result = _build(tmp_path, embedder=_ZeroEmbedder())

    vectors = np.load(result.index_dir / "vectors.npy")
    assert np.all(np.isfinite(vectors))
    assert np.allclose(vectors, 0.0)


def test_build_index_requires_documents(tmp_path: Path) -> None:
    with pytest.raises(IndexBuildError, match="not found"):
        build_index(
            ChunkConfig(strategy="fixed_token"),
            docs_dir=tmp_path / "missing",
            out_dir=tmp_path / "indexes",
            providers_config_path=_providers_config(tmp_path),
            tokenizer=OfflineTokenizer(),
            embedder=_FakeEmbedder(),
        )


# -- CLI ---------------------------------------------------------------------


def _prepare_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    document = _build_document(
        ["# Intro\nThe device must include a cybersecurity plan.\nAnother sentence."],
        title="CLI Doc",
    )
    docs_dir = tmp_path / "docs"
    _write_document(docs_dir, document)
    config = _providers_config(tmp_path)
    out_dir = tmp_path / "indexes"
    monkeypatch.setattr(
        "rag_eval.pipeline.index.default_tokenizer", lambda: OfflineTokenizer()
    )
    monkeypatch.setattr(
        "rag_eval.pipeline.index.default_embedder",
        lambda provider, config_path, texts: np.ones(
            (len(list(texts)), 64), dtype=np.float32
        ),
    )
    return docs_dir, config, out_dir


def test_cli_index_build_and_info(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docs_dir, config, out_dir = _prepare_cli(tmp_path, monkeypatch)

    result = CliRunner().invoke(
        app,
        [
            "index",
            "build",
            "--strategy",
            "structure_aware",
            "--config",
            str(config),
            "--docs-dir",
            str(docs_dir),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    index_dirs = [path for path in out_dir.iterdir() if path.is_dir()]
    assert len(index_dirs) == 1
    manifest = json.loads((index_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["chunk_config"]["strategy"] == "structure_aware"

    listing = CliRunner().invoke(
        app, ["index", "info", "--out-dir", str(out_dir)], env={"COLUMNS": "250"}
    )
    assert listing.exit_code == 0, listing.output
    assert index_dirs[0].name in listing.output

    detail = CliRunner().invoke(
        app, ["index", "info", index_dirs[0].name, "--out-dir", str(out_dir)]
    )
    assert detail.exit_code == 0, detail.output
    assert "chunks" in detail.output


def test_cli_fixed_token_defaults_to_512(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs_dir, config, out_dir = _prepare_cli(tmp_path, monkeypatch)

    result = CliRunner().invoke(
        app,
        [
            "index",
            "build",
            "--strategy",
            "fixed_token",
            "--config",
            str(config),
            "--docs-dir",
            str(docs_dir),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    index_dir = next(path for path in out_dir.iterdir() if path.is_dir())
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["chunk_config"]["max_tokens"] == 512
    assert manifest["chunk_config"]["overlap_tokens"] == 64


def test_cli_rejects_unknown_strategy(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["index", "build", "--strategy", "nope"])

    assert result.exit_code == 2
    assert "strategy must be one of" in result.output


def test_cli_info_with_no_indexes(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["index", "info", "--out-dir", str(tmp_path / "none")]
    )

    assert result.exit_code == 0
    assert "no indexes" in result.output
