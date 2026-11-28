from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from rag_eval.cli import app
from rag_eval.ingest.models import CanonicalDocument, CanonicalPage
from rag_eval.pipeline.index import build_index
from rag_eval.pipeline.models import ChunkConfig
from rag_eval.pipeline.tokenizer import OfflineTokenizer

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "providers.yaml"
runner = CliRunner()

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


def _document() -> CanonicalDocument:
    page = CanonicalPage(
        page_num=1,
        start_char=0,
        end_char=len(DOC_TEXT),
        text=DOC_TEXT,
        table_count=0,
    )
    return CanonicalDocument(
        doc_id="synthetic-doc",
        title="Synthetic Document",
        sha256="0" * 64,
        page_count=1,
        table_count=0,
        token_count=len(DOC_TEXT.split()),
        full_text=DOC_TEXT,
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


def _build_one_index(tmp_path: Path) -> tuple[Path, Path]:
    """Build one real offline index; returns ``(index_dir, providers_config)``."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "synthetic-doc.json").write_text(
        json.dumps(_document().model_dump(mode="json")), encoding="utf-8"
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


def _add_fake_index(indexes_dir: Path, index_id: str, strategy: str) -> None:
    """Drop a manifest-only index directory (enough for ``list_indexes``)."""
    manifest = {
        "schema_version": 1,
        "index_id": index_id,
        "chunk_config": {
            "strategy": strategy,
            "max_tokens": 32,
            "overlap_tokens": 4,
            "hard_max_table_tokens": 900,
        },
        "embedding_provider": "jina",
        "embedding_model": "jina-test",
        "dims": DIMS,
        "corpus_sha256": "0" * 64,
        "chunk_count": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "pipeline_version": "2",
        "embedding_task_type": "",
    }
    index_dir = indexes_dir / index_id
    index_dir.mkdir(parents=True)
    (index_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Run from a temp dir so a developer's real .env is never picked up.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAG_EVAL_PROVIDERS_CONFIG", str(CONFIG_PATH))


def test_doctor_fails_when_no_llm_provider_is_usable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "missing credentials" in result.output


def test_doctor_passes_with_a_single_configured_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output
    assert "nvidia_super" in result.output
    assert "ready" in result.output


def test_doctor_strict_fails_on_partial_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    result = runner.invoke(app, ["doctor", "--strict"])

    assert result.exit_code == 1
    assert "strict" in result.output


def test_doctor_reports_unresolved_cloudflare_account_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output
    assert "CLOUDFLARE_ACCOUNT_ID" in result.output


def test_doctor_never_prints_key_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-super-secret-value")
    result = runner.invoke(app, ["doctor"])

    assert "nvapi-super-secret-value" not in result.output


def test_doctor_rejects_unknown_live_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    result = runner.invoke(app, ["doctor", "--live", "--target", "banana"])

    assert result.exit_code == 2


def test_cache_stats_requires_an_existing_cache() -> None:
    result = runner.invoke(app, ["cache-stats"])
    assert result.exit_code == 1
    assert "cache not found" in result.output


# -- query index auto-detection ------------------------------------------------


def test_query_without_strategy_uses_the_single_index(tmp_path: Path) -> None:
    index_dir, config = _build_one_index(tmp_path)

    result = runner.invoke(
        app,
        [
            "query",
            "cybersecurity",
            "--mode",
            "bm25",
            "--out-dir",
            str(index_dir.parent),
            "--config",
            str(config),
        ],
        env={"COLUMNS": "250"},
    )

    assert result.exit_code == 0, result.output
    assert index_dir.name in result.output
    assert "synthetic-doc" in result.output


def test_query_without_strategy_fails_when_no_index_exists(tmp_path: Path) -> None:
    empty = tmp_path / "indexes"
    empty.mkdir()

    result = runner.invoke(
        app, ["query", "anything", "--mode", "bm25", "--out-dir", str(empty)]
    )

    assert result.exit_code == 1
    assert "no indexes found" in result.output
    assert "--index" in result.output


def test_query_without_strategy_fails_listing_multiple_indexes(tmp_path: Path) -> None:
    index_dir, _ = _build_one_index(tmp_path)
    _add_fake_index(index_dir.parent, "aaaa1111bbbb2222", strategy="fixed_token")

    result = runner.invoke(
        app,
        ["query", "cybersecurity", "--mode", "bm25", "--out-dir", str(index_dir.parent)],
    )

    assert result.exit_code == 1
    assert "multiple indexes" in result.output
    assert index_dir.name in result.output
    assert "aaaa1111bbbb2222" in result.output
    assert "--strategy" in result.output
    assert "--index" in result.output


def test_query_index_flag_overrides_multiple_indexes(tmp_path: Path) -> None:
    index_dir, config = _build_one_index(tmp_path)
    _add_fake_index(index_dir.parent, "aaaa1111bbbb2222", strategy="fixed_token")

    result = runner.invoke(
        app,
        [
            "query",
            "cybersecurity",
            "--mode",
            "bm25",
            "--index",
            index_dir.name,
            "--out-dir",
            str(index_dir.parent),
            "--config",
            str(config),
        ],
        env={"COLUMNS": "250"},
    )

    assert result.exit_code == 0, result.output
    assert index_dir.name in result.output
    assert "synthetic-doc" in result.output


def test_query_strategy_still_selects_one_of_several_indexes(tmp_path: Path) -> None:
    index_dir, config = _build_one_index(tmp_path)
    _add_fake_index(index_dir.parent, "aaaa1111bbbb2222", strategy="fixed_token")

    result = runner.invoke(
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
    assert index_dir.name in result.output
    assert "synthetic-doc" in result.output


def test_query_rejects_unknown_strategy_value(tmp_path: Path) -> None:
    result = runner.invoke(app, ["query", "anything", "--strategy", "nope"])

    assert result.exit_code == 2
    assert "strategy must be one of" in result.output
